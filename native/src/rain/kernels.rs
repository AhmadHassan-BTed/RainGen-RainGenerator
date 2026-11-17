//! Sound kernels for drop, bubble, splash and hail impacts, and the resonators they drive (port of
//! raingen/synth/kernels.py), laid out for real-time use.
//!
//! Physics
//! - Liquid-drop force on a rigid surface (Gordillo, Sun & Cheng 2018): rises like sqrt(t), peaks at
//!   F_max ~ 0.83 rho D^2 V^2 at t ~ 0.18 D/V, log-normal tail (sigma 0.8).
//! - Hertz contact for hail / ice pellets with an ice-crushing limit (Hertz 1882; Johnson 1985).
//! - Air bubbles (Minnaert 1933), van den Doel (2005) voice: eps a sin(2 pi f0 (t + xi beta t^2 / 2))
//!   exp(-beta t), beta = 0.13/a + 0.0072 a^-1.5. Rendered per event as a chirped complex
//!   oscillator (exact, no kernel table), band-limited near 0.45 sr like the Python kernel.
//! - Impact click on water (Medwin et al. 1992): 50-100 us bipolar pulse, amplitude ~ U^3.
//! - Modal resonators: parallel damped modes g r^n sin(n theta), run as complex one-poles in
//!   struct-of-arrays lanes of 8 (SIMD-friendly).
//!
//! Real-time layout: every short kernel (force pulses, Hertz pulses, clicks, splashes) is built once
//! per (sample rate, block) into one flat table shared by all voices (`KernelBank`, behind an Arc),
//! together with the pooled power spectra the matched-noise rows need and a deterministic Gaussian
//! noise table. Voices only read it; per-voice buffers (`Ola`, `BubblePool`, `ModalLanes`,
//! `NoiseRow`) are preallocated, so nothing allocates on the audio thread.

use crate::dsp::BiquadCoefs;
use crate::rng::{hash_str, Rng};
use crate::types::{P0, RHO_WATER};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

pub const V_REF: f32 = 6.5;
pub const D_REF_MM: f32 = 2.0;
const LN1000: f64 = 6.907_755_278_982_137;
const PULSE_SIGMA: f64 = 0.8;
const PULSE_SPAN: f64 = 4.5;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
#[inline]
pub fn smoothstep(x: f32, lo: f32, hi: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
pub fn db_to_pa(db: f32) -> f32 {
    20e-6 * 10f32.powf(db / 20.0)
}

#[inline]
pub fn loguniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo * (hi / lo).powf(rng.uniform())
}

/// Radii distributed as r^-alpha on [r1, r2] (inverse CDF).
#[inline]
pub fn power_law(rng: &mut Rng, r1: f64, r2: f64, alpha: f64) -> f64 {
    let e = 1.0 - alpha;
    let u = rng.uniform();
    (r1.powf(e) + u * (r2.powf(e) - r1.powf(e))).powf(1.0 / e)
}

/// Peak drop force F_max ~ 0.83 rho D^2 V^2 (N), Gordillo et al. (2018).
#[inline]
pub fn force_peak(d_mm: f32, v_mps: f32) -> f32 {
    let d = d_mm * 1e-3;
    0.83 * RHO_WATER * d * d * v_mps * v_mps
}

/// Force of the 2 mm reference drop at 6.5 m/s (0.14 N).
#[inline]
pub fn f_ref() -> f32 {
    force_peak(D_REF_MM, V_REF)
}

/// Impulse of the reference drop pulse (0.511 tau F).
#[inline]
pub fn j_ref() -> f32 {
    0.511 * (D_REF_MM * 1e-3 / V_REF) * f_ref()
}

fn area_sample<F: Fn(f64) -> f64>(f: F, length: usize, sr: f64, os: usize) -> Vec<f64> {
    let mut out = vec![0.0; length];
    for (i, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for j in 0..os {
            let t = ((i * os + j) as f64 + 0.5) / (sr * os as f64);
            s += f(t);
        }
        *o = s / os as f64;
    }
    out
}

/// Raised-cosine fade over the last `frac` of a kernel.
pub fn fade_tail(k: &mut [f64], frac: f64) {
    let m = (k.len() as f64 * frac) as usize;
    if m >= 2 {
        let l = k.len();
        for j in 0..m {
            k[l - m + j] *= 0.5 + 0.5 * (PI * j as f64 / (m - 1) as f64).cos();
        }
    }
}

/// Force pulse of unit F_max with time scale tau = D/V (area-sampled, impulse-preserving).
pub fn pulse_shape(tau_s: f64, sr: f64) -> Vec<f64> {
    let tau = tau_s.max(2e-6);
    let tp = 0.18 * tau;
    let length = ((PULSE_SPAN * tau * sr).ceil() as usize + 1).max(2);
    let os = if tau * sr < 4.0 { 16 } else { 8 };
    let two_s2 = 2.0 * PULSE_SIGMA * PULSE_SIGMA;
    area_sample(
        |t| {
            let x = t / tp;
            if x <= 1.0 {
                x.sqrt()
            } else {
                let lx = x.max(1e-12).ln();
                (-lx * lx / two_s2).exp()
            }
        },
        length,
        sr,
        os,
    )
}

/// Force (N per sample) of a liquid drop of diameter d_mm striking a rigid surface at v_mps.
pub fn impact_pulse(d_mm: f64, v_mps: f64, sr: f64) -> Vec<f64> {
    let d = d_mm.max(0.01);
    let v = v_mps.max(0.05);
    let fp = force_peak(d as f32, v as f32) as f64;
    pulse_shape(d * 1e-3 / v, sr).into_iter().map(|x| x * fp).collect()
}

/// (F_max N, contact time s) of a sphere hitting a half-space; crushing-limited for ice.
#[allow(clippy::too_many_arguments)]
pub fn hertz_contact(d_mm: f64, v_mps: f64, rho: f64, e_proj: f64, e_target: f64, crush_mpa: f64, restitution: f64) -> (f64, f64) {
    let (nu_p, nu_t) = (0.33, 0.3);
    let d = d_mm.max(0.05) * 1e-3;
    let v = v_mps.max(0.05);
    let r = 0.5 * d;
    let m = rho * PI * d * d * d / 6.0;
    let e_star = 1.0 / ((1.0 - nu_p * nu_p) / e_proj + (1.0 - nu_t * nu_t) / e_target);
    let k = 4.0 / 3.0 * e_star * r.sqrt();
    let delta = (5.0 * m * v * v / (4.0 * k)).powf(0.4);
    let mut f_max = k * delta.powf(1.5);
    let mut tau = 2.94 * delta / v;
    if crush_mpa > 0.0 {
        let p_mean = f_max / (PI * r * delta);
        if p_mean > crush_mpa * 1e6 {
            let tau_c = tau.max(d / v);
            f_max = m * v * (1.0 + restitution) / (0.574 * tau_c);
            tau = tau_c;
        }
    }
    (f_max, tau)
}

/// Unit-peak Hertz force history sin(pi t / tau)^(3/2), area-sampled.
pub fn hertz_shape(tau_s: f64, sr: f64) -> Vec<f64> {
    let tau = tau_s.max(1e-6);
    let length = ((tau * sr).ceil() as usize + 1).max(1);
    let os = if tau * sr < 2.0 { 32 } else { 8 };
    area_sample(
        |t| {
            if t < tau {
                (PI * (t / tau).clamp(0.0, 1.0)).sin().powf(1.5)
            } else {
                0.0
            }
        },
        length,
        sr,
        os,
    )
}

/// Minnaert resonance f ~ 3.28 / a (Hz) at 1 atm.
#[inline]
pub fn minnaert_frequency(radius_m: f64) -> f64 {
    3.28 / radius_m.max(1e-6) * (P0 as f64 / P0 as f64).sqrt()
}

/// van den Doel (2005) bubble decay rate beta (1/s).
#[inline]
pub fn bubble_damping(radius_m: f64) -> f64 {
    let a = radius_m.max(1e-6);
    0.13 / a + 0.0072 * a.powf(-1.5)
}

/// Rising-pitch damped sinusoid of a freshly entrained bubble (van den Doel 2005), band-limited.
pub fn bubble_kernel(radius_m: f64, sr: f64, xi: f64, eps: f64, damping_extra: f64) -> Vec<f64> {
    let a = radius_m.max(1e-6);
    let f0 = minnaert_frequency(a);
    let f_lim = 0.45 * sr;
    if f0 >= f_lim {
        return vec![0.0];
    }
    let beta = bubble_damping(a) + PI * f0 * damping_extra.max(0.0);
    let mut dur = 9.21 / beta;
    let xi = xi.max(0.0);
    if xi > 0.0 {
        dur = dur.min((f_lim / f0 - 1.0) / (xi * beta));
    }
    let length = ((dur * sr).ceil() as usize + 1).max(4);
    let mut k = vec![0.0; length];
    for (i, v) in k.iter_mut().enumerate() {
        let t = i as f64 / sr;
        let inst = f0 * (1.0 + xi * beta * t);
        let phase = 2.0 * PI * f0 * (t + 0.5 * xi * beta * t * t);
        let w = ((f_lim - inst) / (0.05 * sr)).clamp(0.0, 1.0);
        let s = (0.5 * PI * w).sin();
        *v = eps * a * (-beta * t).exp() * phase.sin() * s * s;
    }
    fade_tail(&mut k, 0.08);
    k
}

/// Impact click on water: 50-100 us bipolar pulse, peak (U / 6.5 m/s)^3.
pub fn click_kernel(v_mps: f64, sr: f64) -> Vec<f64> {
    let u = v_mps.max(0.05);
    let amp = (u / V_REF as f64).powi(3);
    let width = (75e-6 * (V_REF as f64 / u).sqrt()).clamp(50e-6, 100e-6);
    let sigma = width / 4.0;
    let t0 = width;
    let length = (2.0 * width * sr).ceil() as usize + 2;
    let mut k = area_sample(
        |t| {
            let z = (t - t0) / sigma;
            -z * (0.5 - 0.5 * z * z).exp()
        },
        length,
        sr,
        16,
    );
    let peak = k.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if peak > 0.0 {
        k.iter_mut().for_each(|v| *v *= amp / peak);
    }
    k
}

// ---------------------------------------------------------------------------
// filter designs (scipy.signal.butter equivalents) and a biquad cascade with visible state
// ---------------------------------------------------------------------------
/// 2nd-order Butterworth low-pass (== scipy butter(2, f, fs=sr)).
pub fn butter2_lp(f: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::lowpass(sr, f, std::f32::consts::FRAC_1_SQRT_2)
}

/// 2nd-order Butterworth high-pass.
pub fn butter2_hp(f: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::highpass(sr, f, std::f32::consts::FRAC_1_SQRT_2)
}

/// 1st-order Butterworth low-pass.
pub fn butter1_lp(f: f32, sr: f32) -> BiquadCoefs {
    BiquadCoefs::lowpass1(sr, f)
}

/// scipy butter(2, [lo, hi], 'bandpass', fs=sr): a 4th-order band-pass as two sections.
pub fn butter2_bandpass(lo: f32, hi: f32, sr: f32) -> [BiquadCoefs; 2] {
    let fs = sr as f64;
    let fs2 = 2.0 * fs;
    let w1 = fs2 * (PI * lo as f64 / fs).tan();
    let w2 = fs2 * (PI * hi as f64 / fs).tan();
    let bw = w2 - w1;
    let w0 = (w1 * w2).sqrt();
    let p_lp = Complex::new(-std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2) * (bw / 2.0);
    let disc = (p_lp * p_lp - Complex::new(w0 * w0, 0.0)).sqrt();
    let q1 = p_lp + disc;
    let q2 = p_lp - disc;
    let poles_a = [q1, q1.conj(), q2, q2.conj()];
    let fsc = Complex::new(fs2, 0.0);
    let mut den = Complex::new(1.0, 0.0);
    for p in &poles_a {
        den *= fsc - p;
    }
    let k_z = (bw * bw * (fs2 * fs2) / den).re;
    let dz = |p: Complex<f64>| (fsc + p) / (fsc - p);
    let (d1, d2) = (dz(q1), dz(q2));
    let sec = |p: Complex<f64>, g: f64| BiquadCoefs { b0: g, b1: 0.0, b2: -g, a1: -2.0 * p.re, a2: p.norm_sqr() };
    [sec(d1, k_z), sec(d2, 1.0)]
}

/// Radiation band of a material: 2nd-order high-pass at radiation_hp_hz (if > 5 Hz) and 2nd-order
/// low-pass at radiation_lp_hz (if < 0.45 sr).
pub fn radiation_coefs(hp_hz: f32, lp_hz: f32, sr: f32) -> Vec<BiquadCoefs> {
    let mut v = Vec::new();
    if hp_hz > 5.0 {
        v.push(butter2_hp(hp_hz.min(0.4 * sr), sr));
    }
    if lp_hz < 0.45 * sr {
        v.push(butter2_lp(lp_hz, sr));
    }
    v
}

/// Cascade of transposed-direct-form-II biquads with f64 state (state visible for idle checks).
#[derive(Clone, Debug, Default)]
pub struct Cascade {
    c: Vec<BiquadCoefs>,
    z: Vec<[f64; 2]>,
}

impl Cascade {
    pub fn new(c: &[BiquadCoefs]) -> Cascade {
        Cascade { c: c.to_vec(), z: vec![[0.0; 2]; c.len()] }
    }
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }
    pub fn coefs(&self) -> &[BiquadCoefs] {
        &self.c
    }
    pub fn reset(&mut self) {
        self.z.iter_mut().for_each(|z| *z = [0.0; 2]);
    }
    /// True once every state has decayed to (near) zero.
    pub fn quiet(&self) -> bool {
        self.z.iter().all(|z| z[0].abs() < 1e-25 && z[1].abs() < 1e-25)
    }
    pub fn process(&mut self, buf: &mut [f32]) {
        for (c, z) in self.c.iter().zip(self.z.iter_mut()) {
            let (mut z1, mut z2) = (z[0], z[1]);
            for v in buf.iter_mut() {
                let x = *v as f64;
                let y = c.b0 * x + z1;
                z1 = c.b1 * x - c.a1 * y + z2;
                z2 = c.b2 * x - c.a2 * y;
                *v = y as f32;
            }
            z[0] = if z1.abs() < 1e-30 { 0.0 } else { z1 };
            z[1] = if z2.abs() < 1e-30 { 0.0 } else { z2 };
        }
    }
    /// f64 impulse-response helper (setup only).
    pub fn filter_f64(c: &[BiquadCoefs], x: &mut [f64]) {
        for c in c {
            let (mut z1, mut z2) = (0.0, 0.0);
            for v in x.iter_mut() {
                let xi = *v;
                let y = c.b0 * xi + z1;
                z1 = c.b1 * xi - c.a1 * y + z2;
                z2 = c.b2 * xi - c.a2 * y;
                *v = y;
            }
        }
    }
    /// |H(f)|^2 of a coefficient list.
    pub fn power_at(c: &[BiquadCoefs], f: f32, sr: f32) -> f32 {
        c.iter().map(|b| b.magnitude(f, sr)).map(|m| m * m).product()
    }
}

// ---------------------------------------------------------------------------
// deterministic Gaussian noise table (shared by every matched-noise row)
// ---------------------------------------------------------------------------
pub const NOISE_BITS: u32 = 20;
pub const NOISE_SIZE: usize = 1 << NOISE_BITS;
pub const NOISE_EXT: usize = 1 << 15;

pub fn noise_table() -> Arc<Vec<f32>> {
    static T: OnceLock<Arc<Vec<f32>>> = OnceLock::new();
    T.get_or_init(|| {
        let mut r = Rng::from_key(hash_str("kernels/noise_table"), 0);
        let mut base: Vec<f64> = (0..NOISE_SIZE).map(|_| r.normal()).collect();
        let mean = base.iter().sum::<f64>() / NOISE_SIZE as f64;
        base.iter_mut().for_each(|v| *v -= mean);
        let sd = (base.iter().map(|v| v * v).sum::<f64>() / NOISE_SIZE as f64).sqrt().max(1e-12);
        let mut out: Vec<f32> = base.iter().map(|v| (*v / sd) as f32).collect();
        out.extend_from_within(..NOISE_EXT);
        Arc::new(out)
    })
    .clone()
}

// ---------------------------------------------------------------------------
// kernel bank
// ---------------------------------------------------------------------------
pub const TAU_PER_OCT: f64 = 4.0;
pub const TAU_BIN_MIN: i32 = -13;
pub const TAU_BIN_MAX: i32 = 48;
pub const CLICK_STEP: f32 = 0.25;
pub const CLICK_BIN_MAX: i32 = 200;
pub const DUR_PER_OCT: f64 = 3.0;
pub const DUR_BIN_MIN: i32 = -3;
pub const DUR_BIN_MAX: i32 = 12;
/// Noise bursts per duration bin. Few variants repeat the same burst on every drip of a node (one
/// drop size -> one bin) and print its fine structure on the drips as a fixed colour.
pub const SPLASH_VARIANTS: usize = 32;
pub const RAD_PER_OCT: f64 = 36.0;
pub const RBIN_MIN: i32 = 120;
pub const RBIN_MAX: i32 = 300;
pub const DAMP_BINS: usize = 3;

#[inline]
pub fn tau_bin(tau_s: f64) -> i32 {
    ((tau_s.max(2e-6) / 1e-5).log2() * TAU_PER_OCT).round().clamp(TAU_BIN_MIN as f64, TAU_BIN_MAX as f64) as i32
}
#[inline]
pub fn tau_center(b: i32) -> f64 {
    1e-5 * 2f64.powf(b as f64 / TAU_PER_OCT)
}
#[inline]
pub fn click_bin(v: f32) -> i32 {
    ((v / CLICK_STEP).round() as i32).clamp(1, CLICK_BIN_MAX)
}
#[inline]
pub fn dur_bin(dur_s: f32) -> i32 {
    ((dur_s.max(5e-4) as f64 / 1e-3).log2() * DUR_PER_OCT).round().clamp(DUR_BIN_MIN as f64, DUR_BIN_MAX as f64) as i32
}
#[inline]
pub fn dur_center(b: i32) -> f64 {
    1e-3 * 2f64.powf(b as f64 / DUR_PER_OCT)
}
#[inline]
pub fn radius_bin(r: f64) -> i32 {
    ((r.max(1e-5) / 1e-5).log2() * RAD_PER_OCT).round() as i32
}
#[inline]
pub fn radius_center(b: i32) -> f64 {
    1e-5 * 2f64.powf(b as f64 / RAD_PER_OCT)
}

/// Crown-splash / splat noise burst of unit peak: 0.3 ms attack, decay to -40 dB.
pub fn splash_kernel(duration_s: f64, sr: f64, variant: usize, hp_hz: f32, lp_hz: f32) -> Vec<f64> {
    let dur = duration_s.max(5e-4);
    let length = (dur * sr).ceil() as usize + 1;
    let mut r = Rng::from_key(hash_str(&format!("kernels/splash/{}", (dur * 1e5).round() as i64)), variant as u64);
    let mut k: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / sr;
            r.normal() * (1.0 - (-t / 3e-4).exp()) * (-t * (4.6 / dur)).exp()
        })
        .collect();
    let mut c = Vec::new();
    if hp_hz > 0.0 {
        c.push(butter2_hp(hp_hz, sr as f32));
    }
    if (lp_hz as f64) < 0.45 * sr {
        c.push(butter2_lp(lp_hz, sr as f32));
    }
    Cascade::filter_f64(&c, &mut k);
    fade_tail(&mut k, 0.15);
    let peak = k.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if peak > 0.0 {
        k.iter_mut().for_each(|v| *v /= peak);
    }
    k
}

/// Power spectrum of `k` on the rfft grid of size 2*half, pooled energy-preservingly from a finer
/// grid when the kernel is longer than the frame (KernelBank.pooled_power).
pub fn pooled_power(k: &[f64], half: usize, planner: &mut RealFftPlanner<f64>) -> Vec<f32> {
    let na = 2 * half;
    let mut mult = 1usize;
    while na * mult < k.len() {
        mult *= 2;
    }
    let size = na * mult;
    let fft = planner.plan_fft_forward(size);
    let mut inp = vec![0.0f64; size];
    inp[..k.len().min(size)].copy_from_slice(&k[..k.len().min(size)]);
    let mut spec = fft.make_output_vec();
    let _ = fft.process(&mut inp, &mut spec);
    let pw: Vec<f64> = spec.iter().map(|c| c.norm_sqr()).collect();
    let nb = half + 1;
    if mult == 1 {
        return pw[..nb].iter().map(|v| *v as f32).collect();
    }
    let hm = mult / 2;
    let mut ext: Vec<f64> = Vec::with_capacity(pw.len() + 2 * hm);
    for i in (1..=hm).rev() {
        ext.push(pw[i]);
    }
    ext.extend_from_slice(&pw);
    let l = pw.len();
    for i in (l - 1 - hm..l - 1).rev() {
        ext.push(pw[i]);
    }
    (0..nb)
        .map(|j| (ext[j * mult..(j + 1) * mult].iter().sum::<f64>() / mult as f64) as f32)
        .collect()
}

/// Every short kernel and pooled power spectrum of the precipitation family, built once per
/// (sample rate, block length) and shared read-only by all voices.
pub struct KernelBank {
    pub sr: u32,
    pub sr_f: f32,
    /// Frame length F = 2 m n of the matched-noise rows, overlap factor m, and F / 2.
    pub frame: usize,
    pub m: usize,
    pub half: usize,
    pub window: Vec<f32>,
    flat: Vec<f32>,
    pulse_idx: Vec<(u32, u32)>,
    hertz_idx: Vec<(u32, u32)>,
    click_idx: Vec<(u32, u32)>,
    splash_idx: Vec<(u32, u32)>,
    pulse_psd: Vec<f32>,
    click_psd: Vec<f32>,
    splash_psd: Vec<f32>,
    bubble_psd: Vec<f32>,
    pub noise: Arc<Vec<f32>>,
    pub ifft: Arc<dyn ComplexToReal<f32>>,
}

impl KernelBank {
    /// The shared bank for (sr, block) (built on first use, cached process-wide).
    pub fn shared(sr: u32, block: usize) -> Arc<KernelBank> {
        static CACHE: OnceLock<Mutex<Vec<(u32, usize, Arc<KernelBank>)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let mut g = match cache.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some((_, _, kb)) = g.iter().find(|(s, b, _)| *s == sr && *b == block) {
            return kb.clone();
        }
        let kb = Arc::new(KernelBank::build(sr, block));
        g.push((sr, block, kb.clone()));
        kb
    }

    fn build(sr: u32, block: usize) -> KernelBank {
        let srf = sr as f64;
        let n = block.max(1);
        let m = 512usize.div_ceil(n).max(1);
        let frame = 2 * m * n;
        let half = frame / 2;
        let window: Vec<f32> = (0..frame)
            .map(|i| ((0.5 - 0.5 * (2.0 * PI * i as f64 / frame as f64).cos()) / m as f64).sqrt() as f32)
            .collect();
        let mut planner = RealFftPlanner::<f64>::new();
        let mut flat: Vec<f32> = Vec::new();
        let push = |flat: &mut Vec<f32>, k: &[f64]| -> (u32, u32) {
            let off = flat.len() as u32;
            if k.is_empty() {
                flat.push(0.0);
                return (off, 1);
            }
            flat.extend(k.iter().map(|v| *v as f32));
            (off, k.len() as u32)
        };
        let nb = half + 1;
        let mut pulse_idx = Vec::new();
        let mut hertz_idx = Vec::new();
        let mut pulse_psd = Vec::with_capacity(nb * (TAU_BIN_MAX - TAU_BIN_MIN + 1) as usize);
        for b in TAU_BIN_MIN..=TAU_BIN_MAX {
            let k = pulse_shape(tau_center(b), srf);
            pulse_psd.extend(pooled_power(&k, half, &mut planner));
            pulse_idx.push(push(&mut flat, &k));
            let h = hertz_shape(tau_center(b), srf);
            hertz_idx.push(push(&mut flat, &h));
        }
        let mut click_idx = Vec::new();
        let mut click_psd = Vec::with_capacity(nb * CLICK_BIN_MAX as usize);
        for b in 1..=CLICK_BIN_MAX {
            let k = click_kernel(b as f64 * CLICK_STEP as f64, srf);
            click_psd.extend(pooled_power(&k, half, &mut planner));
            click_idx.push(push(&mut flat, &k));
        }
        let mut splash_idx = Vec::new();
        let mut splash_psd = Vec::new();
        for b in DUR_BIN_MIN..=DUR_BIN_MAX {
            for v in 0..SPLASH_VARIANTS {
                let k = splash_kernel(dur_center(b), srf, v, 1500.0, 14000.0);
                if v == 0 {
                    splash_psd.extend(pooled_power(&k, half, &mut planner));
                }
                splash_idx.push(push(&mut flat, &k));
            }
        }
        let mut bubble_psd = Vec::with_capacity(nb * DAMP_BINS * (RBIN_MAX - RBIN_MIN + 1) as usize);
        for db in 0..DAMP_BINS {
            for rb in RBIN_MIN..=RBIN_MAX {
                let k = bubble_kernel(radius_center(rb), srf, 4.0 * 0.025, 1.0, 0.01 * db as f64);
                bubble_psd.extend(pooled_power(&k, half, &mut planner));
            }
        }
        let mut p32 = RealFftPlanner::<f32>::new();
        let ifft = p32.plan_fft_inverse(frame);
        KernelBank {
            sr,
            sr_f: sr as f32,
            frame,
            m,
            half,
            window,
            flat,
            pulse_idx,
            hertz_idx,
            click_idx,
            splash_idx,
            pulse_psd,
            click_psd,
            splash_psd,
            bubble_psd,
            noise: noise_table(),
            ifft,
        }
    }

    #[inline]
    fn slice(&self, (o, l): (u32, u32)) -> &[f32] {
        &self.flat[o as usize..(o + l) as usize]
    }
    /// Unit-peak drop force pulse of tau bin b.
    #[inline]
    pub fn pulse(&self, b: i32) -> &[f32] {
        self.slice(self.pulse_idx[(b.clamp(TAU_BIN_MIN, TAU_BIN_MAX) - TAU_BIN_MIN) as usize])
    }
    /// Unit-peak Hertz force pulse of tau bin b.
    #[inline]
    pub fn hertz(&self, b: i32) -> &[f32] {
        self.slice(self.hertz_idx[(b.clamp(TAU_BIN_MIN, TAU_BIN_MAX) - TAU_BIN_MIN) as usize])
    }
    /// Water click of click bin b (impact speed b * 0.25 m/s), amplitude (U/6.5)^3 included.
    #[inline]
    pub fn click(&self, b: i32) -> &[f32] {
        self.slice(self.click_idx[(b.clamp(1, CLICK_BIN_MAX) - 1) as usize])
    }
    /// Bright splash burst of duration bin b and noise variant v.
    #[inline]
    pub fn splash(&self, b: i32, v: usize) -> &[f32] {
        let bi = (b.clamp(DUR_BIN_MIN, DUR_BIN_MAX) - DUR_BIN_MIN) as usize;
        self.slice(self.splash_idx[bi * SPLASH_VARIANTS + v % SPLASH_VARIANTS])
    }
    #[inline]
    fn psd_row<'a>(&self, tab: &'a [f32], i: usize) -> &'a [f32] {
        let nb = self.half + 1;
        &tab[i * nb..(i + 1) * nb]
    }
    #[inline]
    pub fn pulse_psd(&self, b: i32) -> &[f32] {
        self.psd_row(&self.pulse_psd, (b.clamp(TAU_BIN_MIN, TAU_BIN_MAX) - TAU_BIN_MIN) as usize)
    }
    #[inline]
    pub fn click_psd(&self, b: i32) -> &[f32] {
        self.psd_row(&self.click_psd, (b.clamp(1, CLICK_BIN_MAX) - 1) as usize)
    }
    #[inline]
    pub fn splash_psd(&self, b: i32) -> &[f32] {
        self.psd_row(&self.splash_psd, (b.clamp(DUR_BIN_MIN, DUR_BIN_MAX) - DUR_BIN_MIN) as usize)
    }
    /// Pooled power of the unit-eps bubble kernel (xi = 0.1) of radius bin rb, damping bin db.
    #[inline]
    pub fn bubble_psd(&self, rb: i32, db: usize) -> &[f32] {
        let nr = (RBIN_MAX - RBIN_MIN + 1) as usize;
        let i = db.min(DAMP_BINS - 1) * nr + (rb.clamp(RBIN_MIN, RBIN_MAX) - RBIN_MIN) as usize;
        self.psd_row(&self.bubble_psd, i)
    }
    /// Longest short kernel (for OLA sizing).
    pub fn max_kernel_len(&self) -> usize {
        self.pulse_idx
            .iter()
            .chain(self.hertz_idx.iter())
            .chain(self.splash_idx.iter())
            .map(|(_, l)| *l as usize)
            .max()
            .unwrap_or(1)
    }
}

// ---------------------------------------------------------------------------
// overlap-add accumulator
// ---------------------------------------------------------------------------
/// Linear overlap-add buffer: kernels added at any offset inside its capacity, popped a block at
/// a time (tails carried across blocks, never truncated within the capacity).
#[derive(Clone, Debug)]
pub struct Ola {
    buf: Vec<f32>,
    used: usize,
}

impl Ola {
    pub fn new(capacity: usize) -> Ola {
        Ola { buf: vec![0.0; capacity.max(16)], used: 0 }
    }
    #[inline]
    pub fn active(&self) -> bool {
        self.used > 0
    }
    pub fn clear(&mut self) {
        self.buf[..self.used].fill(0.0);
        self.used = 0;
    }
    /// buf[off..] += amp * k (clipped at the capacity).
    #[inline]
    pub fn add(&mut self, off: usize, k: &[f32], amp: f32) {
        let cap = self.buf.len();
        if off >= cap || amp == 0.0 || !amp.is_finite() {
            return;
        }
        let end = (off + k.len()).min(cap);
        for (b, v) in self.buf[off..end].iter_mut().zip(k) {
            *b += amp * *v;
        }
        if end > self.used {
            self.used = end;
        }
    }
    /// buf[off..] += x.
    #[inline]
    pub fn add_slice(&mut self, off: usize, x: &[f32]) {
        let cap = self.buf.len();
        if off >= cap {
            return;
        }
        let end = (off + x.len()).min(cap);
        for (b, v) in self.buf[off..end].iter_mut().zip(x) {
            *b += *v;
        }
        if end > self.used {
            self.used = end;
        }
    }
    /// out += the next out.len() samples; the buffer advances. Returns false when it was empty.
    pub fn pop_into(&mut self, out: &mut [f32]) -> bool {
        let n = out.len();
        if self.used == 0 {
            return false;
        }
        let m = n.min(self.used);
        for (o, b) in out[..m].iter_mut().zip(&self.buf[..m]) {
            *o += *b;
        }
        if self.used > n {
            self.buf.copy_within(n..self.used, 0);
            self.buf[self.used - n..self.used].fill(0.0);
            self.used -= n;
        } else {
            self.buf[..self.used].fill(0.0);
            self.used = 0;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// bubble oscillators
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, Default)]
struct Bubble {
    /// Offset in the current block where kernel sample `pos` lands (>= 0).
    start: usize,
    pos: u32,
    len: u32,
    zr: f64,
    zi: f64,
    wr: f64,
    wi: f64,
    cr: f64,
    ci: f64,
    /// band-limit taper argument at pos and its per-sample change
    tw: f64,
    dtw: f64,
    /// first sample of the taper (w < 1) and of the tail fade, and the fade length
    k_taper: u32,
    k_fade: u32,
    fade_m: u32,
}

/// Pool of active bubbles (van den Doel chirped damped sinusoids) rendered as complex oscillators.
#[derive(Clone, Debug)]
pub struct BubblePool {
    b: Vec<Bubble>,
    sr: f64,
}

impl BubblePool {
    pub fn new(capacity: usize, sr: f64) -> BubblePool {
        BubblePool { b: Vec::with_capacity(capacity), sr }
    }
    #[inline]
    pub fn active(&self) -> bool {
        !self.b.is_empty()
    }
    pub fn clear(&mut self) {
        self.b.clear();
    }

    /// Start a bubble `offset` samples after the current block start: kernel amplitude `amp`
    /// (eps * a in Pa at 1 m already folded in by the caller: amp = scale * eps * radius),
    /// radius (m), rising-pitch factor xi and extra damping constant delta.
    pub fn spawn(&mut self, offset: usize, amp: f64, radius: f64, xi: f64, damping_extra: f64) {
        if self.b.len() >= self.b.capacity() || !(amp > 0.0) || !amp.is_finite() {
            return;
        }
        let sr = self.sr;
        let a = radius.max(1e-6);
        let f0 = minnaert_frequency(a);
        let f_lim = 0.45 * sr;
        if f0 >= f_lim {
            return;
        }
        let beta = bubble_damping(a) + PI * f0 * damping_extra.max(0.0);
        let xi = xi.max(0.0);
        let mut dur = 9.21 / beta;
        if xi > 0.0 {
            dur = dur.min((f_lim / f0 - 1.0) / (xi * beta));
        }
        let len = ((dur * sr).ceil() as usize + 1).max(4);
        let r = (-beta / sr).exp();
        let dph = 2.0 * PI * f0 / sr * (1.0 + 0.5 * xi * beta / sr);
        let chirp = 2.0 * PI * f0 * xi * beta / (sr * sr);
        let tw = (f_lim - f0) / (0.05 * sr);
        let dtw = -f0 * xi * beta / sr / (0.05 * sr);
        let k_taper = if tw < 1.0 {
            0
        } else if dtw < 0.0 {
            (((tw - 1.0) / -dtw).ceil() as u64).min(len as u64) as u32
        } else {
            len as u32
        };
        let fade_m = (len as f64 * 0.08) as u32;
        let k_fade = if fade_m >= 2 { len as u32 - fade_m } else { len as u32 };
        self.b.push(Bubble {
            start: offset,
            pos: 0,
            len: len as u32,
            zr: amp * a,
            zi: 0.0,
            wr: r * dph.cos(),
            wi: r * dph.sin(),
            cr: chirp.cos(),
            ci: chirp.sin(),
            tw,
            dtw,
            k_taper,
            k_fade,
            fade_m,
        });
    }

    /// out += every active bubble's samples of this block; finished bubbles are dropped.
    pub fn render(&mut self, out: &mut [f32]) -> bool {
        let n = out.len();
        let mut any = false;
        let mut i = 0;
        while i < self.b.len() {
            let bb = &mut self.b[i];
            if bb.start >= n {
                bb.start -= n;
                i += 1;
                continue;
            }
            any = true;
            let mut k = bb.pos;
            let (mut zr, mut zi, mut wr, mut wi) = (bb.zr, bb.zi, bb.wr, bb.wi);
            let (cr, ci) = (bb.cr, bb.ci);
            let plain_end = bb.k_taper.min(bb.k_fade);
            let mut j = bb.start;
            // fast segment: no taper, no fade
            while j < n && k < plain_end {
                out[j] += zi as f32;
                let nzr = zr * wr - zi * wi;
                zi = zr * wi + zi * wr;
                zr = nzr;
                let nwr = wr * cr - wi * ci;
                wi = wr * ci + wi * cr;
                wr = nwr;
                j += 1;
                k += 1;
            }
            // general segment
            while j < n && k < bb.len {
                let mut g = 1.0;
                if k >= bb.k_taper {
                    let w = (bb.tw + bb.dtw * k as f64).clamp(0.0, 1.0);
                    g = 0.5 - 0.5 * (PI * w).cos();
                }
                if k >= bb.k_fade {
                    let jj = (k - bb.k_fade) as f64;
                    g *= 0.5 + 0.5 * (PI * jj / (bb.fade_m - 1) as f64).cos();
                }
                out[j] += (zi * g) as f32;
                let nzr = zr * wr - zi * wi;
                zi = zr * wi + zi * wr;
                zr = nzr;
                let nwr = wr * cr - wi * ci;
                wi = wr * ci + wi * cr;
                wr = nwr;
                j += 1;
                k += 1;
            }
            if k >= bb.len {
                self.b.swap_remove(i);
                continue;
            }
            bb.pos = k;
            bb.start = 0;
            bb.zr = zr;
            bb.zi = zi;
            bb.wr = wr;
            bb.wi = wi;
            i += 1;
        }
        any
    }
}

// ---------------------------------------------------------------------------
// modal bank in SIMD-friendly lanes
// ---------------------------------------------------------------------------
pub const LANES: usize = 8;

/// Parallel bank of damped modes: complex one-poles s <- p s + x, output sum g Im(s), in lanes
/// of 8 modes (struct of arrays). Exactly the block recursion of kernels.ModalBank.
#[derive(Clone, Debug)]
pub struct ModalLanes {
    pr: Vec<[f32; LANES]>,
    pi: Vec<[f32; LANES]>,
    g: Vec<[f32; LANES]>,
    re: Vec<[f32; LANES]>,
    im: Vec<[f32; LANES]>,
    acc: Vec<[f32; LANES]>,
    n_modes: usize,
    quiet: bool,
}

/// Kept modes of a mode list (f > 5 Hz, f < 0.45 sr, T60 > 0; T60 >= 2 / sr).
pub fn mode_arrays(modes: &[(f32, f32, f32)], sr: f32) -> Vec<(f32, f32, f32)> {
    modes
        .iter()
        .filter(|(f, t, _)| *f > 5.0 && *f < 0.45 * sr && *t > 0.0)
        .map(|(f, t, g)| (*f, t.max(2.0 / sr), *g))
        .collect()
}

impl ModalLanes {
    pub fn new(modes: &[(f32, f32, f32)], sr: f32, n_max: usize) -> ModalLanes {
        let kept = mode_arrays(modes, sr);
        let groups = kept.len().div_ceil(LANES);
        let mut b = ModalLanes {
            pr: vec![[0.0; LANES]; groups],
            pi: vec![[0.0; LANES]; groups],
            g: vec![[0.0; LANES]; groups],
            re: vec![[0.0; LANES]; groups],
            im: vec![[0.0; LANES]; groups],
            acc: vec![[0.0; LANES]; if groups > 0 { n_max } else { 0 }],
            n_modes: kept.len(),
            quiet: true,
        };
        for (i, (f, t60, g)) in kept.iter().enumerate() {
            let r = (-LN1000 / (*t60 as f64 * sr as f64)).exp();
            let th = 2.0 * PI * *f as f64 / sr as f64;
            b.pr[i / LANES][i % LANES] = (r * th.cos()) as f32;
            b.pi[i / LANES][i % LANES] = (r * th.sin()) as f32;
            b.g[i / LANES][i % LANES] = *g;
        }
        b
    }

    #[inline]
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    pub fn reset(&mut self) {
        self.re.iter_mut().for_each(|v| *v = [0.0; LANES]);
        self.im.iter_mut().for_each(|v| *v = [0.0; LANES]);
        self.quiet = true;
    }

    /// True once every mode's |state| * |g| is below eps (states are then zeroed).
    pub fn silent(&mut self, eps: f32) -> bool {
        if self.quiet {
            return true;
        }
        let mut mx = 0.0f32;
        for gi in 0..self.re.len() {
            for j in 0..LANES {
                let s = (self.re[gi][j].abs() + self.im[gi][j].abs()) * self.g[gi][j].abs();
                mx = mx.max(s);
            }
        }
        if mx < eps {
            self.reset();
            return true;
        }
        false
    }

    /// out += bank response to `input` (None = free ring) over out.len() samples.
    pub fn process(&mut self, input: Option<&[f32]>, out: &mut [f32]) {
        let n = out.len().min(self.acc.len());
        if self.n_modes == 0 || n == 0 {
            return;
        }
        if input.is_none() && self.quiet {
            return;
        }
        let acc = &mut self.acc[..n];
        acc.iter_mut().for_each(|a| *a = [0.0; LANES]);
        for gi in 0..self.re.len() {
            let pr = self.pr[gi];
            let pi = self.pi[gi];
            let g = self.g[gi];
            let mut re = self.re[gi];
            let mut im = self.im[gi];
            match input {
                Some(x) => {
                    for (a, &xk) in acc.iter_mut().zip(&x[..n]) {
                        for j in 0..LANES {
                            let nr = pr[j] * re[j] - pi[j] * im[j] + xk;
                            let ni = pi[j] * re[j] + pr[j] * im[j];
                            re[j] = nr;
                            im[j] = ni;
                            a[j] += g[j] * ni;
                        }
                    }
                }
                None => {
                    for a in acc.iter_mut() {
                        for j in 0..LANES {
                            let nr = pr[j] * re[j] - pi[j] * im[j];
                            let ni = pi[j] * re[j] + pr[j] * im[j];
                            re[j] = nr;
                            im[j] = ni;
                            a[j] += g[j] * ni;
                        }
                    }
                }
            }
            for j in 0..LANES {
                if re[j].abs() + im[j].abs() < 1e-20 {
                    re[j] = 0.0;
                    im[j] = 0.0;
                }
            }
            self.re[gi] = re;
            self.im[gi] = im;
        }
        for (o, a) in out[..n].iter_mut().zip(acc.iter()) {
            let s = (a[0] + a[4]) + (a[1] + a[5]) + (a[2] + a[6]) + (a[3] + a[7]);
            *o += s;
        }
        self.quiet = false;
    }

    /// f64 impulse response h[j] = sum g Im(p^j) of a mode list (setup only; h[0] = 0).
    pub fn impulse_response(modes: &[(f32, f32, f32)], sr: f32, length: usize) -> Vec<f64> {
        let kept = mode_arrays(modes, sr);
        let mut h = vec![0.0f64; length];
        for (f, t60, g) in kept {
            let r = (-LN1000 / (t60 as f64 * sr as f64)).exp();
            let th = 2.0 * PI * f as f64 / sr as f64;
            let (pr, pi) = (r * th.cos(), r * th.sin());
            let (mut zr, mut zi) = (1.0f64, 0.0f64);
            for v in h.iter_mut() {
                *v += g as f64 * zi;
                let nr = zr * pr - zi * pi;
                zi = zr * pi + zi * pr;
                zr = nr;
                if zr.abs() + zi.abs() < 1e-30 {
                    break;
                }
            }
        }
        h
    }
}

// ---------------------------------------------------------------------------
// matched-noise row (weighted overlap-add spectral shaper)
// ---------------------------------------------------------------------------
/// One matched-noise row: Gaussian noise whose power spectrum is a given PSD, synthesized frame by
/// frame in the frequency domain (random complex spectrum from the noise table times the PSD's
/// square root, one inverse FFT, a power-complementary sqrt-Hann window; frames of F = 2 m n
/// samples overlap-added at each block start). By Campbell's theorem a PSD of sum_i v_i |K_i|^2
/// matches a Poisson train of kernels K_i at rate x E[amp^2] = v_i per sample.
pub struct NoiseRow {
    spec: Vec<Complex<f32>>,
    frame: Vec<f32>,
    scratch: Vec<Complex<f32>>,
}

impl NoiseRow {
    pub fn new(kb: &KernelBank) -> NoiseRow {
        NoiseRow {
            spec: vec![Complex::new(0.0, 0.0); kb.half + 1],
            frame: vec![0.0; kb.frame],
            scratch: vec![Complex::new(0.0, 0.0); kb.ifft.get_scratch_len()],
        }
    }

    /// Add one frame with power spectrum `psd` (per sample, half+1 bins) at offset 0 of `ola`.
    pub fn synth(&mut self, kb: &KernelBank, psd: &[f32], noise_pos: u64, ola: &mut Ola) {
        let half = kb.half;
        let f = kb.frame as f32;
        let tab = &kb.noise;
        let p = (noise_pos as usize) & (NOISE_SIZE - 1);
        let zr = &tab[p..p + half + 1];
        let zi = &tab[p + half + 1..p + 2 * half + 2];
        let base = (0.5 * f).sqrt() / f;
        for k in 0..=half {
            let a = base * psd[k].max(0.0).sqrt();
            self.spec[k] = Complex::new(zr[k] * a, zi[k] * a);
        }
        self.spec[0] = Complex::new(0.0, 0.0);
        self.spec[half] = Complex::new(self.spec[half].re * std::f32::consts::SQRT_2, 0.0);
        if kb.ifft.process_with_scratch(&mut self.spec, &mut self.frame, &mut self.scratch).is_err() {
            return;
        }
        for (v, w) in self.frame.iter_mut().zip(&kb.window) {
            *v *= *w;
        }
        ola.add_slice(0, &self.frame);
    }
}

// ---------------------------------------------------------------------------
// solid voice: force OLA -> modal bank -> radiation -> Pa
// ---------------------------------------------------------------------------
/// Force pulses (impulse trains of kernel shapes) driving one modal bank with carried state, then
/// the material's radiation band and its calibration (Pa @ 1 m per N): `_SolidVoice` with n_var 1.
pub struct SolidVoice {
    bank: ModalLanes,
    ola: Ola,
    rad: Cascade,
    cal: f32,
    has_modes: bool,
    force: Vec<f32>,
    resp: Vec<f32>,
}

impl SolidVoice {
    pub fn new(modes: &[(f32, f32, f32)], rad: &[BiquadCoefs], cal: f32, sr: f32, n_max: usize, ola_cap: usize) -> SolidVoice {
        let bank = ModalLanes::new(modes, sr, n_max);
        let has_modes = bank.n_modes() > 0;
        SolidVoice {
            bank,
            ola: Ola::new(ola_cap),
            rad: Cascade::new(rad),
            cal,
            has_modes,
            force: vec![0.0; n_max],
            resp: vec![0.0; n_max],
        }
    }

    pub fn reset(&mut self) {
        self.bank.reset();
        self.ola.clear();
        self.rad.reset();
    }

    pub fn active(&mut self) -> bool {
        self.ola.active() || !self.bank.silent(1e-11) || !self.rad.quiet()
    }

    /// Force pulse of peak `force` (N) and time scale `tau` at `off` samples from the block start.
    #[inline]
    pub fn add_force(&mut self, kb: &KernelBank, off: usize, force: f32, tau: f64, hertz: bool) {
        let b = tau_bin(tau);
        let k = if hertz { kb.hertz(b) } else { kb.pulse(b) };
        self.ola.add(off, k, force);
    }

    /// out += this block's pressure (Pa @ 1 m). Returns false when nothing was produced.
    pub fn render(&mut self, out: &mut [f32]) -> bool {
        let n = out.len().min(self.force.len());
        let force = &mut self.force[..n];
        force.fill(0.0);
        let fed = self.ola.pop_into(force);
        let resp = &mut self.resp[..n];
        resp.fill(0.0);
        if !self.has_modes {
            if !fed && self.rad.quiet() {
                return false;
            }
            resp.copy_from_slice(force);
        } else {
            let ringing = !self.bank.silent(1e-11);
            if !fed && !ringing && self.rad.quiet() {
                return false;
            }
            if fed {
                self.bank.process(Some(force), resp);
            } else if ringing {
                self.bank.process(None, resp);
            }
        }
        self.rad.process(resp);
        let cal = self.cal;
        for (o, v) in out[..n].iter_mut().zip(resp.iter()) {
            *o += *v * cal;
        }
        true
    }
}
