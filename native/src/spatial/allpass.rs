//! Random-phase all-pass FIR design, the Rust twin of
//! raingen.spatial.encoder.design_allpass_firs.
//!
//! Start: exponentially decaying noise (time constant decay * L), magnitude-normalised on the
//! M = 2L point DFT grid and truncated to L taps. Then minimise with L-BFGS
//!   J(h) = sum_k (|H_k|^2 - 1)^2                       (M-point grid, full spectrum)
//!        + M (sum_{i != j} (h_i . h_j)^2 + 2 sum_i h_i[0]^2)   (orthogonal = true)
//! On M >= 2L - 1 points |H|^2 fixes the whole aperiodic autocorrelation, so a small residual
//! means |H(f)| ~ 1 at every frequency (true all-pass to ~0.05 dB). The orthogonality terms
//! make the channels mutually uncorrelated and uncorrelated with the dry W path, so white
//! noise through the bank has the diagonal SN3D diffuse-field covariance.
//! Designs are cached process-wide per (channels, length, seed, iterations).

use crate::rng::{combine, hash_str, Rng};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub const ALLPASS_ITERS: usize = 120;
pub const ALLPASS_DECAY: f64 = 0.25;

struct CostCtx {
    c: usize,
    l: usize,
    m: usize,
    orthogonal: bool,
    fwd: Arc<dyn RealToComplex<f64>>,
    inv: Arc<dyn ComplexToReal<f64>>,
    time: Vec<f64>,
    spec: Vec<Complex<f64>>,
    sf: Vec<Complex<f64>>,
    si: Vec<Complex<f64>>,
    x: Vec<f64>,
}

impl CostCtx {
    fn new(c: usize, l: usize, orthogonal: bool) -> CostCtx {
        let m = 2 * l;
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(m);
        let inv = planner.plan_fft_inverse(m);
        CostCtx {
            c,
            l,
            m,
            orthogonal,
            time: fwd.make_input_vec(),
            spec: fwd.make_output_vec(),
            sf: fwd.make_scratch_vec(),
            si: inv.make_scratch_vec(),
            fwd,
            inv,
            x: vec![0.0; c * c],
        }
    }

    /// J(h) and its gradient (h, grad: c x l row-major).
    fn eval(&mut self, h: &[f64], grad: &mut [f64]) -> f64 {
        let (c, l, m) = (self.c, self.l, self.m);
        let bins = m / 2 + 1;
        let mut val = 0.0;
        for ch in 0..c {
            let row = &h[ch * l..(ch + 1) * l];
            self.time[..l].copy_from_slice(row);
            self.time[l..].iter_mut().for_each(|v| *v = 0.0);
            let _ = self.fwd.process_with_scratch(&mut self.time, &mut self.spec, &mut self.sf);
            for (b, z) in self.spec.iter_mut().enumerate() {
                let e = z.re * z.re + z.im * z.im - 1.0;
                let w = if b == 0 || b == bins - 1 { 1.0 } else { 2.0 };
                val += w * e * e;
                *z *= e;
            }
            self.spec[0].im = 0.0;
            self.spec[bins - 1].im = 0.0;
            let _ = self.inv.process_with_scratch(&mut self.spec, &mut self.time, &mut self.si);
            let g = &mut grad[ch * l..(ch + 1) * l];
            for (gv, tv) in g.iter_mut().zip(&self.time[..l]) {
                *gv = 4.0 * *tv;
            }
        }
        if self.orthogonal {
            let mf = m as f64;
            for i in 0..c {
                self.x[i * c + i] = 0.0;
                for j in (i + 1)..c {
                    let d = dot(&h[i * l..(i + 1) * l], &h[j * l..(j + 1) * l]);
                    self.x[i * c + j] = d;
                    self.x[j * c + i] = d;
                }
            }
            let mut sx = 0.0;
            let mut s0 = 0.0;
            for i in 0..c {
                for j in 0..c {
                    sx += self.x[i * c + j] * self.x[i * c + j];
                }
                s0 += h[i * l] * h[i * l];
            }
            val += mf * (sx + 2.0 * s0);
            let k = 4.0 * mf;
            for i in 0..c {
                let g = &mut grad[i * l..(i + 1) * l];
                for j in 0..c {
                    let xij = self.x[i * c + j];
                    if xij == 0.0 {
                        continue;
                    }
                    let hj = &h[j * l..(j + 1) * l];
                    let s = k * xij;
                    for (gv, hv) in g.iter_mut().zip(hj) {
                        *gv += s * *hv;
                    }
                }
                g[0] += k * h[i * l];
            }
        }
        val
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Unconstrained L-BFGS (two-loop recursion, `mem` pairs) with a backtracking Armijo line
/// search (safeguarded quadratic interpolation). Minimises f in place.
fn lbfgs<F: FnMut(&[f64], &mut [f64]) -> f64>(x: &mut [f64], mut f: F, maxiter: usize, mem: usize) {
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut fx = f(x, &mut g);
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(mem);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(mem);
    let mut rho: Vec<f64> = Vec::with_capacity(mem);
    let mut alpha = vec![0.0; mem];
    let mut d = vec![0.0; n];
    let mut xn = vec![0.0; n];
    let mut gn = vec![0.0; n];
    for _ in 0..maxiter {
        // d = -H g
        d.copy_from_slice(&g);
        let k = s_hist.len();
        for i in (0..k).rev() {
            let a = rho[i] * dot(&s_hist[i], &d);
            alpha[i] = a;
            for (dv, yv) in d.iter_mut().zip(&y_hist[i]) {
                *dv -= a * yv;
            }
        }
        if k > 0 {
            let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
            let gamma = if yy > 0.0 { 1.0 / (rho[k - 1] * yy) } else { 1.0 };
            d.iter_mut().for_each(|v| *v *= gamma);
        }
        for i in 0..k {
            let b = rho[i] * dot(&y_hist[i], &d);
            let c = alpha[i] - b;
            for (dv, sv) in d.iter_mut().zip(&s_hist[i]) {
                *dv += c * sv;
            }
        }
        d.iter_mut().for_each(|v| *v = -*v);
        let mut gd = dot(&g, &d);
        if gd >= 0.0 || !gd.is_finite() {
            // not a descent direction: restart from steepest descent
            for (dv, gv) in d.iter_mut().zip(&g) {
                *dv = -gv;
            }
            gd = -dot(&g, &g);
            s_hist.clear();
            y_hist.clear();
            rho.clear();
        }
        if gd == 0.0 {
            break;
        }
        let mut step = if s_hist.is_empty() { 1.0 / dot(&d, &d).sqrt().max(1e-300) } else { 1.0 };
        let mut accepted = false;
        let mut fnew = fx;
        for _ in 0..40 {
            for i in 0..n {
                xn[i] = x[i] + step * d[i];
            }
            fnew = f(&xn, &mut gn);
            if fnew.is_finite() && fnew <= fx + 1e-4 * step * gd {
                accepted = true;
                break;
            }
            // minimiser of the quadratic through f(0), f'(0), f(step), kept in [0.1, 0.5] step
            let denom = 2.0 * (fnew - fx - gd * step);
            let t = if denom > 0.0 && fnew.is_finite() { -gd * step * step / denom } else { 0.5 * step };
            step = t.clamp(0.1 * step, 0.5 * step);
        }
        if !accepted {
            break;
        }
        // memory update
        let mut s = if s_hist.len() == mem { s_hist.remove(0) } else { vec![0.0; n] };
        let mut y = if y_hist.len() == mem { y_hist.remove(0) } else { vec![0.0; n] };
        if rho.len() == mem {
            rho.remove(0);
        }
        for i in 0..n {
            s[i] = xn[i] - x[i];
            y[i] = gn[i] - g[i];
        }
        let sy = dot(&s, &y);
        x.copy_from_slice(&xn);
        g.copy_from_slice(&gn);
        let improved = fx - fnew;
        fx = fnew;
        if sy > 1e-12 * dot(&y, &y).max(1e-300) {
            s_hist.push(s);
            y_hist.push(y);
            rho.push(1.0 / sy);
        } else {
            // keep the pair counts consistent
            s_hist.truncate(rho.len());
            y_hist.truncate(rho.len());
        }
        if improved <= 0.0 && fx == 0.0 {
            break;
        }
    }
}

/// Design `noise.len()` all-pass FIRs of `noise[0].len()` taps from the given noise rows.
/// Returns unit-energy rows.
pub fn design_allpass_firs(noise: &[Vec<f64>], iters: usize, decay: f64, orthogonal: bool) -> Vec<Vec<f64>> {
    let c = noise.len();
    if c == 0 {
        return Vec::new();
    }
    let l = noise[0].len().max(2);
    let m = 2 * l;
    let mut ctx = CostCtx::new(c, l, orthogonal);
    // initial guess: decaying noise, unit magnitude on the M grid, truncated to L
    let mut h = vec![0.0f64; c * l];
    for ch in 0..c {
        for t in 0..l {
            let v = noise[ch].get(t).copied().unwrap_or(0.0);
            ctx.time[t] = v * (-(t as f64) / (decay * l as f64)).exp();
        }
        ctx.time[l..].iter_mut().for_each(|v| *v = 0.0);
        let _ = ctx.fwd.process_with_scratch(&mut ctx.time, &mut ctx.spec, &mut ctx.sf);
        for z in ctx.spec.iter_mut() {
            let a = z.norm();
            *z = if a > 1e-12 { *z / a } else { Complex::new(1.0, 0.0) };
        }
        let last = ctx.spec.len() - 1;
        ctx.spec[0].im = 0.0;
        ctx.spec[last].im = 0.0;
        let _ = ctx.inv.process_with_scratch(&mut ctx.spec, &mut ctx.time, &mut ctx.si);
        for t in 0..l {
            h[ch * l + t] = ctx.time[t] / m as f64;
        }
    }
    if iters > 0 {
        lbfgs(&mut h, |x, g| ctx.eval(x, g), iters, 20);
    }
    (0..c)
        .map(|ch| {
            let row = &h[ch * l..(ch + 1) * l];
            let e = dot(row, row).sqrt().max(1e-300);
            row.iter().map(|v| v / e).collect()
        })
        .collect()
}

type BankKey = (usize, usize, u64, usize);

fn cache() -> &'static Mutex<HashMap<BankKey, Arc<Vec<Vec<f32>>>>> {
    static CACHE: OnceLock<Mutex<HashMap<BankKey, Arc<Vec<Vec<f32>>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The decorrelator bank: `channels` orthogonal all-pass FIRs of `length` taps from the seed's
/// counter RNG stream ("spatial.decorrelator", one stream per channel). Deterministic, cached.
pub fn allpass_bank(channels: usize, length: usize, seed: u64, iters: usize) -> Arc<Vec<Vec<f32>>> {
    let key = (channels, length, seed, iters);
    if let Ok(c) = cache().lock() {
        if let Some(b) = c.get(&key) {
            return b.clone();
        }
    }
    let l = length.max(16);
    let layer = combine(seed, hash_str("spatial.decorrelator"));
    let noise: Vec<Vec<f64>> = (0..channels)
        .map(|ch| {
            let mut r = Rng::from_key(layer, ch as u64);
            (0..l).map(|_| r.normal()).collect()
        })
        .collect();
    let firs = design_allpass_firs(&noise, iters, ALLPASS_DECAY, true);
    let bank: Arc<Vec<Vec<f32>>> = Arc::new(firs.into_iter().map(|r| r.into_iter().map(|v| v as f32).collect()).collect());
    if let Ok(mut c) = cache().lock() {
        c.insert(key, bank.clone());
    }
    bank
}
