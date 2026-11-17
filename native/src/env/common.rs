//! Shared environment toolkit (the Rust twin of the helpers at the top of raingen/synth/wind.py
//! plus the bits of raingen.physics.{wind, world, atmosphere} the environment voices need):
//! filter designs, white-noise gain of filter cascades, the shaped-noise OLA bank, event rings,
//! sparse modal strikes, ISO 9613-1 absorption (exact on FFT grids, a fitted shelf cascade for
//! time-domain paths), surface-layer wind laws and solar day/night.
//!
//! Units: sound pressure in pascals at the listener (1.0 = 94 dB SPL), like the Python reference.

use crate::dsp::{Biquad, BiquadCoefs};
use crate::rng::Rng;
use crate::types::{season, Scene, N_AMBI};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::{Arc, OnceLock};

pub const SQRT_2: f32 = std::f32::consts::SQRT_2;
pub const TAU64: f64 = 2.0 * PI;

// ---------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------
#[inline]
pub fn db_spl(db: f32) -> f32 {
    crate::types::db_spl_to_pa(db)
}

#[inline]
pub fn smoothstep(x: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return if x >= hi { 1.0 } else { 0.0 };
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn smoothstep64(x: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        return if x >= hi { 1.0 } else { 0.0 };
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn clip_f(f: f32, sr: f32) -> f32 {
    f.max(5.0).min(0.45 * sr)
}

/// Projection of a listener-frame position on the direction the wind comes from.
#[inline]
pub fn upwind_distance(pos: [f32; 3], wind_azimuth: f32) -> f32 {
    pos[0] * wind_azimuth.cos() + pos[1] * wind_azimuth.sin()
}

/// Linear per-block ramp of a gain: `g0 + (g1 - g0) (i + 1) / n` (Python's Ramp).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ramp {
    pub v: f32,
}

impl Ramp {
    pub fn new(v: f32) -> Ramp {
        Ramp { v }
    }
    /// Move to `target`; returns (start, end) of this block's ramp.
    #[inline]
    pub fn to(&mut self, target: f32) -> (f32, f32) {
        let g0 = self.v;
        self.v = target;
        (g0, target)
    }
    #[inline]
    pub fn idle(&self) -> bool {
        self.v.abs() < 1e-9
    }
}

#[inline]
pub fn ramp_at(g0: f32, g1: f32, i: usize, n: usize) -> f32 {
    g0 + (g1 - g0) * (i + 1) as f32 / n as f32
}

/// buf *= ramp(g0 -> g1)
pub fn apply_ramp(buf: &mut [f32], g0: f32, g1: f32) {
    let n = buf.len();
    if g0 == g1 {
        if g0 != 1.0 {
            buf.iter_mut().for_each(|v| *v *= g0);
        }
        return;
    }
    let d = (g1 - g0) / n as f32;
    for (i, v) in buf.iter_mut().enumerate() {
        *v *= g0 + d * (i + 1) as f32;
    }
}

/// Fill with unit-variance white noise (uniform, cheap; filtered noise is Gaussian by the CLT).
pub fn fill_white(rng: &mut Rng, out: &mut [f32]) {
    rng.fill_bipolar(out);
    let k = 3f32.sqrt();
    out.iter_mut().for_each(|v| *v *= k);
}

/// Fill with standard normal samples.
pub fn fill_gauss(rng: &mut Rng, out: &mut [f32]) {
    rng.fill_normal(out);
}

/// Poisson event offsets in [0, n) at `rate_per_s`, sorted (clears `out`; never grows it past capacity).
pub fn poisson_offsets(rng: &mut Rng, rate_per_s: f32, n: usize, sr: f32, out: &mut Vec<usize>) {
    out.clear();
    let lam = rate_per_s.max(0.0) as f64 * n as f64 / sr as f64;
    if lam <= 0.0 {
        return;
    }
    let k = rng.poisson(lam) as usize;
    let cap = out.capacity();
    for _ in 0..k.min(cap) {
        out.push(rng.below(n as u64) as usize);
    }
    out.sort_unstable();
}

/// Peak-normalise in place.
pub fn normalize_peak(y: &mut [f32]) {
    let pk = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if pk > 0.0 {
        let g = 1.0 / pk;
        y.iter_mut().for_each(|v| *v *= g);
    }
}

// ---------------------------------------------------------------------------
// filter designs with the Python reference's conventions (raingen/synth/wind.py)
// ---------------------------------------------------------------------------
/// RBJ band-pass, 0 dB peak (fc clipped to [5, 0.45 sr], Q >= 0.05).
pub fn bp(fc: f32, q: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::bandpass(sr, clip_f(fc, sr), q.max(0.05))
}

/// RBJ peaking equaliser.
pub fn peak_eq(fc: f32, gain_db: f32, q: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::peaking(sr, clip_f(fc, sr), q.max(0.05), gain_db)
}

/// 2nd-order Butterworth low-pass (bilinear, prewarped; what scipy.signal.butter(2) returns).
pub fn lp2(fc: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::lowpass(sr, clip_f(fc, sr), std::f32::consts::FRAC_1_SQRT_2)
}

pub fn hp2(fc: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::highpass(sr, clip_f(fc, sr), std::f32::consts::FRAC_1_SQRT_2)
}

pub fn lp1(fc: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::lowpass1(sr, clip_f(fc, sr))
}

pub fn hp1(fc: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::highpass1(sr, clip_f(fc, sr))
}

/// Constant spectral slope between f_lo and f_hi (flat outside), 0 dB at DC: an interleaved
/// real pole/zero ladder as first-order sections (raingen.synth.wind.slope_sos).
pub fn slope_sections(slope_db_per_decade: f32, f_lo: f32, f_hi: f32, sr: f32) -> Vec<BiquadCoefs> {
    let f_lo = f_lo.max(1.0) as f64;
    let f_hi = (f_hi.min(0.45 * sr)) as f64;
    let s = slope_db_per_decade as f64;
    let srd = sr as f64;
    if f_hi <= f_lo * 1.01 || s.abs() < 1e-6 {
        return vec![BiquadCoefs::identity()];
    }
    let n_dec = (f_hi / f_lo).log10();
    let k = ((n_dec * 2.0).ceil() as usize).max(1);
    let step = (f_hi / f_lo).powf(1.0 / k as f64);
    let r = (10f64.powf(s.abs() * n_dec / (20.0 * k as f64))).min(step * 0.999);
    let warp = |f: f64| srd / PI * (PI * f / srd).tan();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let c = f_lo * step.powf(i as f64 + 0.5);
        let (lo, hi) = (c / r.sqrt(), c * r.sqrt());
        let (pole_f, zero_f) = if s < 0.0 { (lo, hi) } else { (hi, lo) };
        // analog section (s + wz) / (s + wp) * wp / wz, bilinear: unity gain at DC
        let wz = 2.0 * PI * warp(zero_f);
        let wp = 2.0 * PI * warp(pole_f);
        let kk = 2.0 * srd;
        let zd = (kk - wz) / (kk + wz);
        let pd = (kk - wp) / (kk + wp);
        let g = (1.0 - pd) / (1.0 - zd);
        out.push(BiquadCoefs { b0: g, b1: -g * zd, b2: 0.0, a1: -pd, a2: 0.0 });
    }
    out
}

/// Cascade of peaking filters matching gains_db at freqs (iteratively corrected), as
/// raingen.synth.wind.graphic_eq_sos. Returns the sections (frequencies >= 0.45 sr dropped).
pub fn graphic_eq(freqs: &[f32], gains_db: &[f32], sr: f32, q: f32, iterations: usize) -> Vec<BiquadCoefs> {
    let mut f: Vec<f32> = Vec::new();
    let mut t: Vec<f32> = Vec::new();
    for (fi, gi) in freqs.iter().zip(gains_db) {
        if *fi < 0.45 * sr {
            f.push(*fi);
            t.push(*gi);
        }
    }
    if f.is_empty() {
        return vec![BiquadCoefs::identity()];
    }
    let offset = t.iter().sum::<f32>() / t.len() as f32;
    let mut g: Vec<f32> = t.iter().map(|v| v - offset).collect();
    let mut sos = Vec::new();
    for _ in 0..iterations.max(1) {
        sos = f.iter().zip(&g).map(|(fi, gi)| peak_eq(*fi, *gi, q, sr)).collect::<Vec<_>>();
        let k = 10f64.powf(offset as f64 / 20.0);
        sos[0].b0 *= k;
        sos[0].b1 *= k;
        sos[0].b2 *= k;
        for (j, fj) in f.iter().enumerate() {
            let h: f32 = sos.iter().map(|c| c.magnitude(*fj, sr)).product();
            let err = t[j] - 20.0 * h.max(1e-12).log10();
            g[j] += 0.85 * err;
        }
    }
    sos
}

// ---------------------------------------------------------------------------
// white-noise RMS gain of filters
// ---------------------------------------------------------------------------
struct NrgGrid {
    w: Vec<f64>,
    dw: Vec<f64>,
}

fn nrg_grid() -> &'static NrgGrid {
    static G: OnceLock<NrgGrid> = OnceLock::new();
    G.get_or_init(|| {
        let mut w: Vec<f64> = Vec::with_capacity(2048);
        let (a, b) = ((1e-5f64).ln(), PI.ln());
        for i in 0..1536 {
            w.push((a + (b - a) * i as f64 / 1535.0).exp());
        }
        for i in 0..512 {
            w.push(PI * i as f64 / 511.0);
        }
        w.sort_by(|x, y| x.partial_cmp(y).unwrap());
        w.dedup_by(|x, y| (*x - *y).abs() < 1e-15);
        let dw = w.windows(2).map(|p| p[1] - p[0]).collect();
        NrgGrid { w, dw }
    })
}

#[inline]
fn section_mag2(c: &BiquadCoefs, w: f64) -> f64 {
    let (s1, c1) = w.sin_cos();
    let (s2, c2) = (2.0 * w).sin_cos();
    let nr = c.b0 + c.b1 * c1 + c.b2 * c2;
    let ni = -(c.b1 * s1 + c.b2 * s2);
    let dr = 1.0 + c.a1 * c1 + c.a2 * c2;
    let di = -(c.a1 * s1 + c.a2 * s2);
    (nr * nr + ni * ni) / (dr * dr + di * di).max(1e-300)
}

/// |H|^2 of a cascade on the noise-gain grid (to combine with variable sections later).
pub fn grid_mag2(secs: &[BiquadCoefs]) -> Vec<f64> {
    let g = nrg_grid();
    g.w.iter().map(|&w| secs.iter().map(|c| section_mag2(c, w)).product()).collect()
}

fn integrate(p: impl Iterator<Item = f64>) -> f32 {
    let g = nrg_grid();
    let mut prev = None;
    let mut acc = 0.0;
    for (i, v) in p.enumerate() {
        if let Some(pv) = prev {
            acc += 0.5 * (pv + v) * g.dw[i - 1];
        }
        prev = Some(v);
    }
    ((acc / PI).max(1e-24)).sqrt() as f32
}

/// RMS gain of a cascade for white noise input: sqrt((1/pi) int_0^pi |H|^2 dw).
pub fn noise_rms_gain(secs: &[BiquadCoefs]) -> f32 {
    let g = nrg_grid();
    integrate(g.w.iter().map(|&w| secs.iter().map(|c| section_mag2(c, w)).product()))
}

/// Noise gain of (a fixed cascade with precomputed `static_mag2`) followed by `var` sections.
pub fn noise_rms_gain_with(static_mag2: &[f64], var: &[BiquadCoefs]) -> f32 {
    let g = nrg_grid();
    integrate(g.w.iter().zip(static_mag2).map(|(&w, &m)| m * var.iter().map(|c| section_mag2(c, w)).product::<f64>()))
}

/// Exact white-noise power gain (sum h^2) of one biquad (stationary Lyapunov solution).
pub fn biquad_power_gain(c: &BiquadCoefs) -> f64 {
    let (b0, b1, b2, a1, a2) = (c.b0, c.b1, c.b2, c.a1, c.a2);
    let g1 = b1 - a1 * b0;
    let g2 = b2 - a2 * b0;
    let den = 1.0 - a1 * a1 - a2 * a2 + 2.0 * a1 * a1 * a2 / (1.0 + a2);
    if den.abs() < 1e-300 || (1.0 + a2).abs() < 1e-300 {
        return 1.0;
    }
    let p11 = (g1 * g1 + g2 * g2 - 2.0 * a1 * g1 * g2 / (1.0 + a2)) / den;
    (p11 + b0 * b0).max(1e-30)
}

/// RMS gain (sqrt of the power gain) of one biquad.
#[inline]
pub fn biquad_rms_gain(c: &BiquadCoefs) -> f32 {
    biquad_power_gain(c).sqrt() as f32
}

/// White-noise RMS gain of `cascade` identical 0 dB-peak band-passes (narrow-band closed form).
pub fn bandpass_rms_gain(fc: f32, q: f32, sr: f32, cascade: usize) -> f32 {
    let f0 = clip_f(fc, sr);
    let bw = std::f32::consts::PI * f0 / (2.0 * q.max(0.05)) / if cascade <= 1 { 1.0 } else { 2.0 };
    (2.0 * bw / sr).max(1e-24).sqrt()
}

/// A design table of one variable section on a log-frequency grid (gust-driven retunes):
/// entry k holds the section at grid ** k and the noise gain of (static cascade + section).
pub struct DesignTable {
    pub grid: f32,
    pub k0: i32,
    pub coefs: Vec<BiquadCoefs>,
    pub norm: Vec<f32>,
}

impl DesignTable {
    pub fn new(static_secs: &[BiquadCoefs], f_lo: f32, f_hi: f32, grid: f32, build: impl Fn(f32) -> BiquadCoefs) -> DesignTable {
        let static_mag2 = grid_mag2(static_secs);
        let k0 = (f_lo.max(1.0).ln() / grid.ln()).floor() as i32;
        let k1 = (f_hi.max(f_lo + 1.0).ln() / grid.ln()).ceil() as i32;
        let mut coefs = Vec::new();
        let mut norm = Vec::new();
        for k in k0..=k1 {
            let f = grid.powi(k);
            let c = build(f);
            norm.push(noise_rms_gain_with(&static_mag2, &[c]));
            coefs.push(c);
        }
        DesignTable { grid, k0, coefs, norm }
    }

    /// (section, noise gain) for `fc` snapped to the grid (clamped to the table).
    #[inline]
    pub fn get(&self, fc: f32) -> (BiquadCoefs, f32) {
        let k = (fc.max(1.0).ln() / self.grid.ln()).round() as i32;
        let i = (k - self.k0).clamp(0, self.coefs.len() as i32 - 1) as usize;
        (self.coefs[i], self.norm[i])
    }
}

// ---------------------------------------------------------------------------
// ISO 9613-1 absorption
// ---------------------------------------------------------------------------
/// Pure-tone atmospheric absorption coefficient alpha (dB/m), ISO 9613-1, at 1013.25 hPa.
pub struct Iso9613 {
    fr_o: f64,
    fr_n: f64,
    k_cl: f64,
    k_o: f64,
    k_n: f64,
}

impl Iso9613 {
    pub fn new(temp_c: f32, rh_pct: f32) -> Iso9613 {
        let t = temp_c as f64 + 273.15;
        let t0 = 293.15;
        let t01 = 273.16;
        let p_rel = 1.0;
        let c = -6.8346 * (t01 / t).powf(1.261) + 4.6151;
        let psat = 10f64.powf(c);
        let h = (rh_pct as f64).clamp(0.0, 100.0) * psat / p_rel;
        let fr_o = p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
        let fr_n = p_rel * (t / t0).powf(-0.5) * (9.0 + 280.0 * h * (-4.170 * ((t / t0).powf(-1.0 / 3.0) - 1.0)).exp());
        Iso9613 {
            fr_o,
            fr_n,
            k_cl: 1.84e-11 / p_rel * (t / t0).sqrt(),
            k_o: (t / t0).powf(-2.5) * 0.01275 * (-2239.1 / t).exp(),
            k_n: (t / t0).powf(-2.5) * 0.1068 * (-3352.0 / t).exp(),
        }
    }

    #[inline]
    pub fn alpha(&self, f: f64) -> f64 {
        let f2 = f * f;
        8.686 * f2 * (self.k_cl + self.k_o / (self.fr_o + f2 / self.fr_o) + self.k_n / (self.fr_n + f2 / self.fr_n))
    }
}

pub fn iso9613_alpha(f: f64, temp_c: f32, rh_pct: f32) -> f64 {
    Iso9613::new(temp_c, rh_pct).alpha(f)
}

const AIR_CORNERS: [f32; 6] = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];

/// High-shelf cascade (corners an octave apart, 500 Hz .. 16 kHz) fitted to the ISO 9613-1 loss
/// over `distance_m` (least squares on the shelves' own shapes, 3 Newton passes). Within ~1-2 dB
/// up to 0.45 sr for losses up to 60 dB. Returns (sections, count); count 0 = passthrough.
pub fn design_air(distance_m: f32, temp_c: f32, rh_pct: f32, sr: f32) -> ([BiquadCoefs; 6], usize) {
    let mut out = [BiquadCoefs::identity(); 6];
    if distance_m < 1.0 {
        return (out, 0);
    }
    let iso = Iso9613::new(temp_c, rh_pct);
    let fmax = 0.45 * sr;
    let mut nc = 0;
    let mut corners = [0f32; 6];
    for c in AIR_CORNERS {
        if c < 0.8 * fmax {
            corners[nc] = c;
            nc += 1;
        }
    }
    let mut pts = [0f32; 6];
    for k in 0..nc {
        pts[k] = if k + 1 == nc { fmax } else { corners[k] * SQRT_2 };
    }
    let mut target = [0f64; 6];
    let mut top = 0.0f64;
    for k in 0..nc {
        let l = (iso.alpha(pts[k] as f64) * distance_m as f64).min(60.0);
        target[k] = -l;
        top = top.max(l);
    }
    if top < 0.05 {
        return (out, 0);
    }
    let mut g = [0f64; 6];
    let mut prev = 0.0;
    for k in 0..nc {
        g[k] = target[k] - prev;
        prev = target[k];
    }
    let shelf = |c: f32, gdb: f64| BiquadCoefs::high_shelf(sr, c, std::f32::consts::FRAC_1_SQRT_2, gdb as f32);
    for _ in 0..3 {
        // shape matrix at the current gains (response in dB per dB of gain)
        let mut m = [[0f64; 6]; 6];
        for k in 0..nc {
            let gk = if g[k].abs() < 0.5 { -0.5 } else { g[k] };
            let s = shelf(corners[k], gk);
            for j in 0..nc {
                let db = 20.0 * (s.magnitude(pts[j], sr) as f64).max(1e-12).log10();
                m[j][k] = db / gk;
            }
        }
        let mut rhs = target;
        if !solve_small(&mut m, &mut rhs, nc) {
            break;
        }
        for k in 0..nc {
            g[k] = rhs[k].clamp(-80.0, 6.0);
        }
    }
    for k in 0..nc {
        out[k] = shelf(corners[k], g[k]);
    }
    (out, nc)
}

/// Gaussian elimination with partial pivoting on an n x n (n <= 6) system; solution in `b`.
fn solve_small(a: &mut [[f64; 6]; 6], b: &mut [f64; 6], n: usize) -> bool {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return false;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in col + 1..n {
            s -= a[col][c] * b[c];
        }
        b[col] = s / a[col][col];
    }
    true
}

/// ISO 9613-1 absorption over a fixed path for a static source, redesigned only when the
/// temperature moves by 1 C or the humidity by 5 % (raingen.synth.wind.AirPath).
#[derive(Clone, Debug)]
pub struct AirPath {
    pub d: f32,
    min_m: f32,
    sr: f32,
    key: (i32, i32),
    st: [Biquad; 6],
    n: usize,
}

impl AirPath {
    pub fn new(distance_m: f32, sr: f32, min_m: f32) -> AirPath {
        AirPath { d: distance_m, min_m, sr, key: (i32::MIN, i32::MIN), st: [Biquad::default(); 6], n: 0 }
    }

    pub fn process(&mut self, buf: &mut [f32], temp_c: f32, rh_pct: f32) {
        if self.d <= self.min_m {
            return;
        }
        let key = (temp_c.round() as i32, (rh_pct / 5.0).round() as i32);
        if key != self.key {
            let (c, n) = design_air(self.d, temp_c, rh_pct, self.sr);
            for i in 0..6 {
                self.st[i].set(c[i]);
            }
            self.n = n;
            self.key = key;
        }
        for s in self.st[..self.n].iter_mut() {
            s.process_block(buf);
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.st {
            s.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// FFT helpers
// ---------------------------------------------------------------------------
pub struct Fft {
    pub n: usize,
    pub fwd: Arc<dyn RealToComplex<f32>>,
    pub inv: Arc<dyn ComplexToReal<f32>>,
    pub sf: Vec<Complex<f32>>,
    pub si: Vec<Complex<f32>>,
}

impl Fft {
    pub fn new(n: usize) -> Fft {
        let mut p = RealFftPlanner::<f32>::new();
        let fwd = p.plan_fft_forward(n);
        let inv = p.plan_fft_inverse(n);
        let sf = fwd.make_scratch_vec();
        let si = inv.make_scratch_vec();
        Fft { n, fwd, inv, sf, si }
    }
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }
    /// time (n, destroyed) -> spec (n/2+1)
    pub fn forward(&mut self, time: &mut [f32], spec: &mut [Complex<f32>]) {
        let _ = self.fwd.process_with_scratch(time, spec, &mut self.sf);
    }
    /// spec (destroyed) -> time (unnormalised: x n)
    pub fn inverse(&mut self, spec: &mut [Complex<f32>], time: &mut [f32]) {
        spec[0].im = 0.0;
        let l = spec.len() - 1;
        spec[l].im = 0.0;
        let _ = self.inv.process_with_scratch(spec, time, &mut self.si);
    }
}

/// rfft grid frequencies of an n-point transform.
pub fn rfft_freqs(n: usize, sr: f32) -> Vec<f32> {
    (0..n / 2 + 1).map(|k| k as f32 * sr / n as f32).collect()
}

/// Parseval weights: variance = sum_k w_k |H_k|^2 for unit-variance white bins.
pub fn rfft_weights(n: usize) -> Vec<f32> {
    let f = n / 2 + 1;
    let mut w = vec![2.0 / n as f32; f];
    w[0] = 1.0 / n as f32;
    if n % 2 == 0 {
        w[f - 1] = 1.0 / n as f32;
    }
    w
}

/// Fill a spectrum with zero-mean unit-variance complex parts (uniform draws: a time sample sums
/// ~bins independent parts, so it is Gaussian by the central limit theorem).
#[inline]
pub fn white_spectrum(rng: &mut Rng, spec: &mut [Complex<f32>]) {
    const K: f32 = 3.464_101_6 / 16_777_216.0; // sqrt(12) / 2^24
    for c in spec.iter_mut() {
        let x = rng.next_u64();
        let a = ((x >> 40) as f32) * K - 1.732_050_8;
        let b = (((x >> 8) & 0xFF_FFFF) as f32) * K - 1.732_050_8;
        *c = Complex::new(a, b);
    }
}

/// |H|^2 of a 0 dB-peak analog band-pass on a grid: 1 / (1 + Q^2 (f/fc - fc/f)^2).
#[inline]
pub fn bandpass_mag2(f: f32, fc: f32, q: f32) -> f32 {
    let fc = fc.max(1e-3);
    let ff = f.max(1e-3);
    let x = q * (ff / fc - fc / ff);
    1.0 / (1.0 + x * x)
}

/// Gaussian noise of one voice with a per-block magnitude spectrum: frames of 2n samples drawn in
/// the frequency domain, shaped by |H(f)|, sine-windowed and overlap-added at 50 %
/// (raingen.synth.wind.ShapedNoiseOLA for one voice). Output variance = `power(|H|^2)`.
pub struct ShapedNoise {
    pub n: usize,
    pub nfft: usize,
    pub freqs: Vec<f32>,
    pub wgt: Vec<f32>,
    window: Vec<f32>,
    scale: Vec<f32>,
    fft: Fft,
    spec: Vec<Complex<f32>>,
    frame: Vec<f32>,
    tail: Vec<f32>,
    pub tail_active: bool,
}

impl ShapedNoise {
    pub fn new(n: usize, sr: f32) -> ShapedNoise {
        let nfft = 2 * n;
        let fft = Fft::new(nfft);
        let bins = fft.bins();
        let mut scale = vec![(nfft as f32 / 2.0).sqrt(); bins];
        scale[0] = (nfft as f32).sqrt();
        scale[bins - 1] = (nfft as f32).sqrt();
        ShapedNoise {
            n,
            nfft,
            freqs: rfft_freqs(nfft, sr),
            wgt: rfft_weights(nfft),
            window: (0..nfft).map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / nfft as f32).sin()).collect(),
            scale,
            spec: vec![Complex::new(0.0, 0.0); bins],
            frame: vec![0.0; nfft],
            tail: vec![0.0; n],
            tail_active: false,
            fft,
        }
    }

    pub fn bins(&self) -> usize {
        self.freqs.len()
    }

    /// Output variance for squared magnitudes `mag2` on this grid.
    pub fn power(&self, mag2: &[f32]) -> f32 {
        mag2.iter().zip(&self.wgt).map(|(a, b)| a * b).sum()
    }

    /// Next n samples into `out` (written). `mag` None = silence (the previous frame still decays).
    pub fn process(&mut self, rng: &mut Rng, mag: Option<&[f32]>, out: &mut [f32]) {
        let n = self.n;
        match mag {
            None => {
                if self.tail_active {
                    out[..n].copy_from_slice(&self.tail);
                    self.tail.iter_mut().for_each(|v| *v = 0.0);
                    self.tail_active = false;
                } else {
                    out[..n].iter_mut().for_each(|v| *v = 0.0);
                }
            }
            Some(mag) => {
                white_spectrum(rng, &mut self.spec);
                for ((c, m), s) in self.spec.iter_mut().zip(mag).zip(&self.scale) {
                    *c *= *m * *s;
                }
                self.fft.inverse(&mut self.spec, &mut self.frame);
                let k = 1.0 / self.nfft as f32;
                for (v, w) in self.frame.iter_mut().zip(&self.window) {
                    *v *= *w * k;
                }
                for i in 0..n {
                    out[i] = self.frame[i] + self.tail[i];
                }
                self.tail.copy_from_slice(&self.frame[n..]);
                self.tail_active = true;
            }
        }
    }

    pub fn reset(&mut self) {
        self.tail.iter_mut().for_each(|v| *v = 0.0);
        self.tail_active = false;
    }
}

// ---------------------------------------------------------------------------
// event ring (audio that spans blocks, addressed in absolute samples)
// ---------------------------------------------------------------------------
pub struct EventRing {
    ch: usize,
    cap: usize,
    mask: usize,
    buf: Vec<f32>,
    base: u64,
    end: u64,
}

impl EventRing {
    pub fn new(channels: usize, capacity: usize) -> EventRing {
        let cap = capacity.max(64).next_power_of_two();
        EventRing { ch: channels.max(1), cap, mask: cap - 1, buf: vec![0.0; channels.max(1) * cap], base: 0, end: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn active(&self) -> bool {
        self.end > self.base
    }

    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.base = 0;
        self.end = 0;
    }

    /// Add `sig * gain` to channel `c` from absolute sample `start` (clipped to the ring).
    pub fn add(&mut self, start: u64, c: usize, sig: &[f32], gain: f32) {
        let cap = self.cap;
        let c = c.min(self.ch - 1);
        self.add_inner(start, sig, |ring, idx, v| ring[c * cap + idx] += gain * v);
    }

    fn add_inner<F: FnMut(&mut [f32], usize, f32)>(&mut self, start: u64, sig: &[f32], mut f: F) {
        let mut s = start;
        let mut off = 0usize;
        if s < self.base {
            off = (self.base - s) as usize;
            s = self.base;
        }
        if off >= sig.len() {
            return;
        }
        let room = (self.base + self.cap as u64).saturating_sub(s) as usize;
        let len = (sig.len() - off).min(room);
        let mask = self.mask;
        for i in 0..len {
            let idx = ((s as usize) + i) & mask;
            f(&mut self.buf, idx, sig[off + i]);
        }
        self.end = self.end.max(s + len as u64);
    }

    /// Add `sig * gain * gains[c]` to every channel (gains.len() == channels).
    pub fn add_gains(&mut self, start: u64, gains: &[f32], sig: &[f32], gain: f32) {
        let cap = self.cap;
        let ch = self.ch;
        for (c, g) in gains.iter().enumerate().take(ch) {
            let gg = *g * gain;
            if gg == 0.0 {
                continue;
            }
            self.add_inner(start, sig, |ring, idx, v| ring[c * cap + idx] += gg * v);
        }
    }

    /// Add channel `c` of the next n samples starting at absolute `start` into out (ch x n) and
    /// clear them; samples before `start` are discarded.
    pub fn take_add(&mut self, start: u64, n: usize, out: &mut [f32]) {
        if start > self.base {
            let skip = (start - self.base).min(self.cap as u64) as usize;
            for c in 0..self.ch {
                for i in 0..skip {
                    let idx = ((self.base as usize) + i) & self.mask;
                    self.buf[c * self.cap + idx] = 0.0;
                }
            }
            self.base = start;
        }
        if self.end > self.base {
            let m = n.min(self.cap);
            for c in 0..self.ch {
                let o = &mut out[c * n..(c + 1) * n];
                for i in 0..m {
                    let idx = ((start as usize) + i) & self.mask;
                    let v = &mut self.buf[c * self.cap + idx];
                    o[i] += *v;
                    *v = 0.0;
                }
            }
        }
        self.base = start + n as u64;
        if self.end < self.base {
            self.end = self.base;
        }
    }
}

// ---------------------------------------------------------------------------
// sparse modal strikes (raingen.synth.wind.SparseModalBank)
// ---------------------------------------------------------------------------
/// Parallel damped modes s[k] = p s[k-1] + x[k], output sum Im(s); strikes carry one amplitude
/// per mode. Modes outside (5 Hz, 0.45 sr) are dropped (`keep` maps kept -> original index).
pub struct StrikeBank {
    pub m: usize,
    pr: Vec<f64>,
    pi: Vec<f64>,
    re: Vec<f64>,
    im: Vec<f64>,
    pub keep: Vec<usize>,
}

impl StrikeBank {
    pub fn new(freqs: &[f64], t60s: &[f64], sr: f32) -> StrikeBank {
        let sr = sr as f64;
        let mut b = StrikeBank { m: 0, pr: vec![], pi: vec![], re: vec![], im: vec![], keep: vec![] };
        for (i, (&f, &t)) in freqs.iter().zip(t60s).enumerate() {
            if f > 5.0 && f < 0.45 * sr {
                let t60 = t.max(4.0 / sr);
                let r = (-(1000f64).ln() / (t60 * sr)).exp();
                let w = 2.0 * PI * f / sr;
                b.pr.push(r * w.cos());
                b.pi.push(r * w.sin());
                b.re.push(0.0);
                b.im.push(0.0);
                b.keep.push(i);
            }
        }
        b.m = b.keep.len();
        b
    }

    pub fn silent(&self) -> bool {
        self.m == 0 || self.re.iter().zip(&self.im).all(|(a, b)| (a * a + b * b) < 1e-18)
    }

    /// out[k] += gain * sum_m Im(s_m[k]); `strikes` (offset, row) sorted by offset, `amps` rows of m.
    pub fn process(&mut self, strikes: &[(usize, usize)], amps: &[f32], out: &mut [f32], gain: f32) {
        let n = out.len();
        if self.m == 0 || (strikes.is_empty() && self.silent()) {
            for i in 0..self.m {
                self.re[i] = 0.0;
                self.im[i] = 0.0;
            }
            return;
        }
        let g = gain as f64;
        for md in 0..self.m {
            let (pr, pi) = (self.pr[md], self.pi[md]);
            let (mut re, mut im) = (self.re[md], self.im[md]);
            let mut sp = 0usize;
            for (k, o) in out.iter_mut().enumerate().take(n) {
                let nre = pr * re - pi * im;
                im = pi * re + pr * im;
                re = nre;
                while sp < strikes.len() && strikes[sp].0.min(n - 1) == k {
                    re += amps[strikes[sp].1 * self.m + md] as f64;
                    sp += 1;
                }
                *o += (g * im) as f32;
            }
            self.re[md] = if re.abs() < 1e-30 { 0.0 } else { re };
            self.im[md] = if im.abs() < 1e-30 { 0.0 } else { im };
        }
    }

    pub fn reset(&mut self) {
        self.re.iter_mut().for_each(|v| *v = 0.0);
        self.im.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ---------------------------------------------------------------------------
// SH helpers
// ---------------------------------------------------------------------------
const ACN_ORDER: [i32; N_AMBI] = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];

/// ACN/SN3D encoding gains; spread tapers higher orders toward W: (1 - spread)^order
/// (raingen.synth.wind.sh_gains).
pub fn sh_gains(az: f32, el: f32, spread: f32) -> [f32; N_AMBI] {
    let mut y = crate::spatial::sh_real_f32(az, el);
    if spread > 0.0 {
        let b = 1.0 - spread.clamp(0.0, 1.0);
        for (k, v) in y.iter_mut().enumerate() {
            *v *= b.powi(ACN_ORDER[k]);
        }
    }
    y
}

#[inline]
pub fn acn_order(k: usize) -> usize {
    ACN_ORDER[k.min(N_AMBI - 1)] as usize
}

/// out (N_AMBI x n) += ramp(g_prev -> g_cur) * x per channel.
pub fn encode_ramped(out: &mut [f32], x: &[f32], g0: &[f32; N_AMBI], g1: &[f32; N_AMBI]) {
    let n = x.len();
    for k in 0..N_AMBI {
        let (a, b) = (g0[k], g1[k]);
        if a == 0.0 && b == 0.0 {
            continue;
        }
        let o = &mut out[k * n..(k + 1) * n];
        if a == b {
            for (ov, xv) in o.iter_mut().zip(x) {
                *ov += a * *xv;
            }
        } else {
            let d = (b - a) / n as f32;
            for (i, (ov, xv)) in o.iter_mut().zip(x).enumerate() {
                *ov += (a + d * (i + 1) as f32) * *xv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// surface-layer wind (raingen.physics.wind)
// ---------------------------------------------------------------------------
const Z0_BY_PLACE: &[(&str, f32)] = &[
    ("open_sea", 0.0002), ("sea", 0.0002), ("ocean", 0.0002), ("offshore", 0.0002),
    ("coast", 0.005), ("beach", 0.003), ("lake", 0.005), ("lakeside", 0.005), ("river", 0.01), ("pond", 0.01),
    ("snowfield", 0.001), ("tundra", 0.005), ("desert", 0.005), ("mudflat", 0.003), ("salt_flat", 0.001),
    ("meadow", 0.03), ("field", 0.03), ("grassland", 0.03), ("open", 0.03), ("moor", 0.03), ("heath", 0.04),
    ("mountain_ridge", 0.05), ("ridge", 0.05), ("mountain", 0.08), ("hill", 0.05),
    ("farm", 0.1), ("farmland", 0.1), ("countryside", 0.1), ("vineyard", 0.15),
    ("garden", 0.2), ("backyard", 0.2), ("campsite", 0.2), ("park", 0.3), ("orchard", 0.4),
    ("suburb", 0.25), ("village", 0.3), ("residential", 0.3),
    ("deciduous_forest", 0.7), ("conifer_forest", 0.8), ("forest", 0.7), ("woodland", 0.6), ("wood", 0.6),
    ("jungle", 1.0), ("rainforest", 1.0),
    ("city_street", 0.7), ("street", 0.7), ("town", 0.7), ("alley", 1.0), ("city", 1.0),
    ("city_centre", 1.5), ("city_center", 1.5), ("downtown", 1.5),
];

pub fn norm_name(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// Aerodynamic roughness length z0 (m) for a place preset name.
pub fn terrain_z0(preset: &str) -> f32 {
    let key = norm_name(preset);
    if let Some((_, z)) = Z0_BY_PLACE.iter().find(|(k, _)| *k == key) {
        return *z;
    }
    let mut names: Vec<&(&str, f32)> = Z0_BY_PLACE.iter().collect();
    names.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (k, z) in names {
        if key.contains(k) {
            return *z;
        }
    }
    0.1
}

pub fn scene_z0(scene: &Scene) -> f32 {
    scene.place.z0_m.unwrap_or_else(|| terrain_z0(&scene.place.preset))
}

#[inline]
fn clamp_z0(z0: f32) -> f64 {
    (z0 as f64).clamp(1e-5, 2.0)
}

/// Mean wind (m/s) at height z from the 10 m report, neutral log law (clamped).
pub fn wind_at_height(u10: f32, z_m: f32, z0: f32) -> f32 {
    let z0 = clamp_z0(z0);
    let z = (z_m as f64).max(3.0 * z0).max(0.05);
    let rf = (10.0f64).max(3.0 * z0 * 1.0001);
    let ratio = (z / z0).ln() / (rf / z0).ln();
    (u10.max(0.0) as f64 * ratio.clamp(0.0, 2.0)) as f32
}

/// Longitudinal turbulence intensity I_u = 1/ln(z/z0) (neutral surface layer).
pub fn turbulence_intensity_for(z0: f32, z_m: f32) -> f32 {
    let z0 = clamp_z0(z0);
    let z = (z_m as f64).max(3.0 * z0).max(0.05);
    ((1.0 / (z / z0).ln()).clamp(0.04, 0.6)) as f32
}

/// Gust correlation time from the Kaimal peak n = f z / U ~ 0.045: T = 1/(2 pi f_peak), 0.5..60 s.
pub fn kaimal_integral_time_s(u_mps: f64, z_m: f64) -> f64 {
    let f_peak = 0.045 * u_mps.max(0.3) / z_m.max(1.0);
    (1.0 / (2.0 * PI * f_peak)).clamp(0.5, 60.0)
}

pub fn mps_to_beaufort(v: f32) -> f32 {
    (v.max(0.0) / 0.836).powf(2.0 / 3.0)
}

/// Log-std of a mean-one log-normal multiplier with relative std `ti`.
#[inline]
pub fn lognormal_sigma(ti: f32) -> f32 {
    (ti.max(0.0) * ti.max(0.0)).ln_1p().sqrt()
}

/// Rescale a mean-one log-normal gust multiplier from turbulence intensity ti_from to ti_to.
pub fn remap_gust(g: f32, ti_from: f32, ti_to: f32) -> f32 {
    let s0 = lognormal_sigma(ti_from);
    let s1 = lognormal_sigma(ti_to);
    if s0 < 1e-6 || (s1 - s0).abs() < 1e-9 {
        return g;
    }
    let z = (g.max(1e-6).ln() + 0.5 * s0 * s0) / s0;
    (s1 * z - 0.5 * s1 * s1).exp()
}

/// E[g^k] of a mean-one log-normal multiplier.
pub fn gust_power_mean(exponent: f32, ti: f32) -> f32 {
    let s2 = (ti.max(0.0) * ti.max(0.0)).ln_1p();
    (0.5 * exponent * (exponent - 1.0) * s2).exp()
}

// ---------------------------------------------------------------------------
// solar day / stability (raingen.physics.world)
// ---------------------------------------------------------------------------
pub fn season_doy(s: u8) -> f64 {
    match s {
        season::SPRING => 80.0,
        season::SUMMER => 172.0,
        season::AUTUMN => 266.0,
        season::WINTER => 355.0,
        _ => 172.0,
    }
}

/// (sunrise, sunset) in local solar hours for a representative day of the season.
pub fn sun_times(latitude_deg: f32, season_code: u8) -> (f32, f32) {
    let mut doy = season_doy(season_code);
    if latitude_deg < 0.0 {
        doy = (doy + 182.0) % 365.0;
    }
    let decl = 23.44f64.to_radians() * (2.0 * PI * (284.0 + doy) / 365.0).sin();
    let phi = (latitude_deg as f64).clamp(-89.9, 89.9).to_radians();
    let x = (-phi.tan() * decl.tan()).clamp(-1.0, 1.0);
    let w0 = x.acos().to_degrees();
    ((12.0 - w0 / 15.0) as f32, (12.0 + w0 / 15.0) as f32)
}

pub fn is_daytime(hour: f32, latitude_deg: f32, season_code: u8) -> bool {
    let (rise, sset) = sun_times(latitude_deg, season_code);
    let hr = hour.rem_euclid(24.0);
    rise <= hr && hr < sset
}

/// Pasquill-Gifford-style stability: 0 unstable, 1 neutral, 2 stable (types::stability codes).
pub fn stability_class(is_day: bool, cloud_okta: f32, wind_mps: f32, rain_mm_h: f32) -> u8 {
    use crate::types::stability::*;
    if wind_mps >= 6.0 || rain_mm_h >= 2.0 || cloud_okta >= 7.0 {
        return NEUTRAL;
    }
    if is_day {
        if cloud_okta <= 4.0 && wind_mps < 5.0 {
            UNSTABLE
        } else {
            NEUTRAL
        }
    } else if cloud_okta <= 6.0 && wind_mps < 5.0 {
        STABLE
    } else {
        NEUTRAL
    }
}

// ---------------------------------------------------------------------------
// scene helpers
// ---------------------------------------------------------------------------
/// Numeric value of `place.biophony[key]` (0 when absent).
pub fn biophony_num(scene: &Scene, key: &str) -> f32 {
    match scene.place.biophony.get(key) {
        Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(0.0) as f32,
        Some(serde_json::Value::String(s)) => s.trim().parse().unwrap_or(0.0),
        Some(serde_json::Value::Bool(b)) => *b as u8 as f32,
        _ => 0.0,
    }
}

pub fn biophony_str(scene: &Scene, key: &str) -> Option<String> {
    match scene.place.biophony.get(key) {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Bool(b)) => Some(if *b { "on".into() } else { "off".into() }),
        Some(serde_json::Value::Null) | None => None,
        Some(v) => Some(v.to_string()),
    }
}

/// Small JSON number formatting (finite, rounded to `dp` decimals).
pub fn jnum(x: f64, dp: usize) -> String {
    if !x.is_finite() {
        return "null".into();
    }
    let p = 10f64.powi(dp as i32);
    let r = (x * p).round() / p;
    let s = format!("{:.*}", dp, r);
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

/// JSON string literal.
pub fn jstr(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

// ---------------------------------------------------------------------------
// live ambience levels
// ---------------------------------------------------------------------------
/// One category's live level (`ctx.ambience[cat]`): auto (-1) follows the ecology value given to
/// `step`, a fixed level (0..1) replaces it. Glides with a 0.3 s time constant.
#[derive(Clone, Debug)]
pub struct AmbLevel {
    pub cat: usize,
    /// smoothed effective value
    pub v: f64,
    /// smoothed playback gate: 0 while the category is set to 0 (Off), else 1
    pub gate: f32,
    init: bool,
    /// true while a fixed level is set
    pub manual: bool,
    prev: f32,
    /// seconds since the category was switched on by hand (large when not)
    pub on_s: f64,
}

pub const AMB_TAU_S: f64 = 0.3;

impl AmbLevel {
    pub fn new(cat: usize) -> AmbLevel {
        AmbLevel { cat, v: 0.0, gate: 1.0, init: false, manual: false, prev: -1.0, on_s: 1e9 }
    }

    /// Raw level now: None = auto.
    #[inline]
    pub fn raw(&self, ctx: &crate::engine::BlockCtx) -> Option<f32> {
        let l = ctx.ambience.get(self.cat).copied().unwrap_or(-1.0);
        if l >= 0.0 {
            Some(l)
        } else {
            None
        }
    }

    /// Effective (smoothed) value for this block; call once per block.
    pub fn step(&mut self, ctx: &crate::engine::BlockCtx, auto: f64) -> f64 {
        let dt = ctx.n as f64 / ctx.sr as f64;
        let l = ctx.ambience.get(self.cat).copied().unwrap_or(-1.0);
        let manual = l >= 0.0;
        let tgt = if manual { l as f64 } else { auto.max(0.0) };
        let gate_t = if manual && l <= 0.0 { 0.0 } else { 1.0 };
        if !self.init {
            self.init = true;
            self.v = tgt;
            self.gate = gate_t;
        } else {
            let a = 1.0 - (-dt / AMB_TAU_S).exp();
            self.v += (tgt - self.v) * a;
            self.gate += (gate_t - self.gate) * a as f32;
            if (self.v - tgt).abs() < 1e-5 {
                self.v = tgt;
            }
            if (self.gate - gate_t).abs() < 1e-4 {
                self.gate = gate_t;
            }
        }
        if manual && l > 0.0 && !(self.manual && self.prev > 0.0) {
            self.on_s = 0.0;
        } else {
            self.on_s += dt;
        }
        self.manual = manual;
        self.prev = l;
        self.v
    }

    /// Event-rate boost right after a category is switched on by hand, so the first call comes
    /// within a second or two instead of after a long natural pause.
    #[inline]
    pub fn kick(&self) -> f64 {
        if self.manual {
            1.0 + 8.0 * (-self.on_s / 6.0).exp()
        } else {
            1.0
        }
    }

    pub fn reset(&mut self) {
        self.init = false;
        self.manual = false;
        self.prev = -1.0;
        self.on_s = 1e9;
    }
}
