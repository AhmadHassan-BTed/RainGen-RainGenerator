//! Shared DSP building blocks: biquads / SOS cascades, one-poles, smoothers, the modal
//! resonator bank, fractional delays, coloured noise and FFT block convolution.
//!
//! Conventions: audio buffers are `f32`; recursive filter states are `f64` where precision
//! matters (biquads) and `f32` where the recursion is a normalised rotation (modal bank).
//! Nothing here allocates after construction. Every recursive state is flushed to zero once
//! it decays below ~1e-20 (no denormals); `enable_ftz()` additionally sets the CPU's
//! flush-to-zero / denormals-are-zero mode on the calling thread (the engine calls it on the
//! render pool and producer threads).

use crate::types::Mode;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::Arc;

pub const DENORMAL_EPS: f32 = 1e-20;

#[inline]
pub fn flush(x: f32) -> f32 {
    if x.abs() < DENORMAL_EPS {
        0.0
    } else {
        x
    }
}

#[inline]
pub fn flush64(x: f64) -> f64 {
    if x.abs() < 1e-30 {
        0.0
    } else {
        x
    }
}

/// Set flush-to-zero and denormals-are-zero on the current thread.
pub fn enable_ftz() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut csr: u32 = 0;
        std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr as *mut u32, options(nostack));
        csr |= 0x8040; // FTZ (bit 15) | DAZ (bit 6)
        std::arch::asm!("ldmxcsr [{}]", in(reg) &csr as *const u32, options(nostack));
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut fpcr: u64;
        std::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack));
        fpcr |= 1 << 24; // FZ
        std::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nomem, nostack));
    }
}

#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[inline]
pub fn gain_to_db(g: f32) -> f32 {
    if g > 0.0 {
        20.0 * g.log10()
    } else {
        f32::NEG_INFINITY
    }
}

#[inline]
pub fn next_pow2(n: usize) -> usize {
    n.max(2).next_power_of_two()
}

/// One-pole coefficient for a time constant: y += c (x - y) reaches 1 - 1/e after `tau_s`.
#[inline]
pub fn tau_coef(tau_s: f32, sr: f32) -> f32 {
    if tau_s <= 0.0 {
        1.0
    } else {
        1.0 - (-1.0 / (tau_s * sr)).exp()
    }
}

/// Pole radius of a resonance that decays 60 dB in `t60_s`.
#[inline]
pub fn t60_radius(t60_s: f32, sr: f32) -> f32 {
    if t60_s <= 0.0 {
        0.0
    } else {
        (-6.907_755_3 / (t60_s * sr)).exp()
    }
}

// ---------------------------------------------------------------------------
// buffer helpers
// ---------------------------------------------------------------------------
#[inline]
pub fn mix_add(dst: &mut [f32], src: &[f32], g: f32) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += g * *s;
    }
}

/// dst += src * (g0 -> g1 linearly over the block; the last sample gets exactly g1).
#[inline]
pub fn mix_add_ramp(dst: &mut [f32], src: &[f32], g0: f32, g1: f32) {
    let n = dst.len().min(src.len());
    if n == 0 {
        return;
    }
    if g0 == g1 {
        mix_add(&mut dst[..n], &src[..n], g0);
        return;
    }
    let dg = (g1 - g0) / n as f32;
    for i in 0..n {
        dst[i] += (g0 + dg * (i + 1) as f32) * src[i];
    }
}

#[inline]
pub fn scale(buf: &mut [f32], g: f32) {
    for v in buf.iter_mut() {
        *v *= g;
    }
}

pub fn peak(buf: &[f32]) -> f32 {
    buf.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

pub fn rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    (buf.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / buf.len() as f64).sqrt() as f32
}

pub fn is_silent(buf: &[f32]) -> bool {
    buf.iter().all(|v| v.abs() < DENORMAL_EPS)
}

// ---------------------------------------------------------------------------
// biquad
// ---------------------------------------------------------------------------
/// Normalised biquad coefficients (a0 = 1): y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoefs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoefs {
    fn default() -> Self {
        BiquadCoefs::identity()
    }
}

#[inline]
fn clamp_f(sr: f64, f: f64) -> f64 {
    f.clamp(1e-3, 0.49 * sr)
}

impl BiquadCoefs {
    pub const fn identity() -> BiquadCoefs {
        BiquadCoefs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 }
    }

    pub fn gain(g: f64) -> BiquadCoefs {
        BiquadCoefs { b0: g, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 }
    }

    /// From a scipy SOS row [b0, b1, b2, a0, a1, a2].
    pub fn from_sos(row: &[f64; 6]) -> BiquadCoefs {
        let a0 = if row[3] != 0.0 { row[3] } else { 1.0 };
        BiquadCoefs { b0: row[0] / a0, b1: row[1] / a0, b2: row[2] / a0, a1: row[4] / a0, a2: row[5] / a0 }
    }

    fn norm(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> BiquadCoefs {
        BiquadCoefs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    /// RBJ low-pass (q = 1/sqrt2 is Butterworth).
    pub fn lowpass(sr: f32, f: f32, q: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        Self::norm((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + al, -2.0 * c, 1.0 - al)
    }

    pub fn highpass(sr: f32, f: f32, q: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        Self::norm((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + al, -2.0 * c, 1.0 - al)
    }

    /// Band-pass with 0 dB peak gain at `f`.
    pub fn bandpass(sr: f32, f: f32, q: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        Self::norm(al, 0.0, -al, 1.0 + al, -2.0 * c, 1.0 - al)
    }

    pub fn notch(sr: f32, f: f32, q: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        Self::norm(1.0, -2.0 * c, 1.0, 1.0 + al, -2.0 * c, 1.0 - al)
    }

    pub fn allpass(sr: f32, f: f32, q: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        Self::norm(1.0 - al, -2.0 * c, 1.0 + al, 1.0 + al, -2.0 * c, 1.0 - al)
    }

    pub fn peaking(sr: f32, f: f32, q: f32, gain_db: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let a = 10f64.powf(gain_db as f64 / 40.0);
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        Self::norm(1.0 + al * a, -2.0 * c, 1.0 - al * a, 1.0 + al / a, -2.0 * c, 1.0 - al / a)
    }

    /// RBJ low shelf, shelf slope S = 1 when `q` = 1/sqrt2.
    pub fn low_shelf(sr: f32, f: f32, q: f32, gain_db: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let a = 10f64.powf(gain_db as f64 / 40.0);
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        let sa = 2.0 * a.sqrt() * al;
        Self::norm(
            a * ((a + 1.0) - (a - 1.0) * c + sa),
            2.0 * a * ((a - 1.0) - (a + 1.0) * c),
            a * ((a + 1.0) - (a - 1.0) * c - sa),
            (a + 1.0) + (a - 1.0) * c + sa,
            -2.0 * ((a - 1.0) + (a + 1.0) * c),
            (a + 1.0) + (a - 1.0) * c - sa,
        )
    }

    pub fn high_shelf(sr: f32, f: f32, q: f32, gain_db: f32) -> BiquadCoefs {
        let (sr, f, q) = (sr as f64, clamp_f(sr as f64, f as f64), (q as f64).max(1e-4));
        let a = 10f64.powf(gain_db as f64 / 40.0);
        let w = 2.0 * PI * f / sr;
        let (s, c) = w.sin_cos();
        let al = s / (2.0 * q);
        let sa = 2.0 * a.sqrt() * al;
        Self::norm(
            a * ((a + 1.0) + (a - 1.0) * c + sa),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - sa),
            (a + 1.0) - (a - 1.0) * c + sa,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - sa,
        )
    }

    /// First-order (bilinear) low-pass.
    pub fn lowpass1(sr: f32, f: f32) -> BiquadCoefs {
        let k = (PI * clamp_f(sr as f64, f as f64) / sr as f64).tan();
        let a0 = 1.0 + k;
        BiquadCoefs { b0: k / a0, b1: k / a0, b2: 0.0, a1: (k - 1.0) / a0, a2: 0.0 }
    }

    /// First-order (bilinear) high-pass.
    pub fn highpass1(sr: f32, f: f32) -> BiquadCoefs {
        let k = (PI * clamp_f(sr as f64, f as f64) / sr as f64).tan();
        let a0 = 1.0 + k;
        BiquadCoefs { b0: 1.0 / a0, b1: -1.0 / a0, b2: 0.0, a1: (k - 1.0) / a0, a2: 0.0 }
    }

    /// Magnitude response at `f` Hz.
    pub fn magnitude(&self, f: f32, sr: f32) -> f32 {
        let w = 2.0 * PI * f as f64 / sr as f64;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        (((nr * nr + ni * ni) / (dr * dr + di * di).max(1e-300)).sqrt()) as f32
    }
}

/// Transposed direct form II biquad with f64 state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Biquad {
    pub c: BiquadCoefs,
    z1: f64,
    z2: f64,
}

impl Biquad {
    pub fn new(c: BiquadCoefs) -> Biquad {
        Biquad { c, z1: 0.0, z2: 0.0 }
    }

    /// Change coefficients, keeping the state (for slow modulation).
    #[inline]
    pub fn set(&mut self, c: BiquadCoefs) {
        self.c = c;
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let c = &self.c;
        let y = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * y + self.z2;
        self.z2 = c.b2 * x - c.a2 * y;
        y as f32
    }

    #[inline]
    pub fn flush_state(&mut self) {
        self.z1 = flush64(self.z1);
        self.z2 = flush64(self.z2);
    }

    pub fn process_block(&mut self, buf: &mut [f32]) {
        let c = self.c;
        let (mut z1, mut z2) = (self.z1, self.z2);
        for v in buf.iter_mut() {
            let x = *v as f64;
            let y = c.b0 * x + z1;
            z1 = c.b1 * x - c.a1 * y + z2;
            z2 = c.b2 * x - c.a2 * y;
            *v = y as f32;
        }
        self.z1 = flush64(z1);
        self.z2 = flush64(z2);
    }

    pub fn process_into(&mut self, input: &[f32], out: &mut [f32]) {
        let c = self.c;
        let (mut z1, mut z2) = (self.z1, self.z2);
        for (o, &xi) in out.iter_mut().zip(input) {
            let x = xi as f64;
            let y = c.b0 * x + z1;
            z1 = c.b1 * x - c.a1 * y + z2;
            z2 = c.b2 * x - c.a2 * y;
            *o = y as f32;
        }
        self.z1 = flush64(z1);
        self.z2 = flush64(z2);
    }

    /// out += g * filter(input).
    pub fn process_add(&mut self, input: &[f32], out: &mut [f32], g: f32) {
        let c = self.c;
        let g = g as f64;
        let (mut z1, mut z2) = (self.z1, self.z2);
        for (o, &xi) in out.iter_mut().zip(input) {
            let x = xi as f64;
            let y = c.b0 * x + z1;
            z1 = c.b1 * x - c.a1 * y + z2;
            z2 = c.b2 * x - c.a2 * y;
            *o += (g * y) as f32;
        }
        self.z1 = flush64(z1);
        self.z2 = flush64(z2);
    }
}

/// Cascade of biquads (second-order sections).
#[derive(Clone, Debug, Default)]
pub struct Sos {
    pub stages: Vec<Biquad>,
}

impl Sos {
    pub fn new(coefs: &[BiquadCoefs]) -> Sos {
        Sos { stages: coefs.iter().map(|c| Biquad::new(*c)).collect() }
    }

    /// From scipy-style rows [b0, b1, b2, a0, a1, a2].
    pub fn from_rows(rows: &[[f64; 6]]) -> Sos {
        Sos { stages: rows.iter().map(|r| Biquad::new(BiquadCoefs::from_sos(r))).collect() }
    }

    /// Butterworth sections of the given order (Q of each pole pair; first-order section if odd).
    fn butter_qs(order: usize) -> (Vec<f32>, bool) {
        let n = order.max(1);
        let mut qs = Vec::new();
        for k in 0..n / 2 {
            let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;
            qs.push((1.0 / (2.0 * theta.sin())) as f32);
        }
        (qs, n % 2 == 1)
    }

    pub fn butter_lowpass(order: usize, fc: f32, sr: f32) -> Sos {
        let (qs, odd) = Self::butter_qs(order);
        let mut c: Vec<BiquadCoefs> = qs.iter().map(|&q| BiquadCoefs::lowpass(sr, fc, q)).collect();
        if odd {
            c.push(BiquadCoefs::lowpass1(sr, fc));
        }
        Sos::new(&c)
    }

    pub fn butter_highpass(order: usize, fc: f32, sr: f32) -> Sos {
        let (qs, odd) = Self::butter_qs(order);
        let mut c: Vec<BiquadCoefs> = qs.iter().map(|&q| BiquadCoefs::highpass(sr, fc, q)).collect();
        if odd {
            c.push(BiquadCoefs::highpass1(sr, fc));
        }
        Sos::new(&c)
    }

    /// Band-pass as a high-pass of `order` at `lo` followed by a low-pass of `order` at `hi`.
    pub fn butter_bandpass(order: usize, lo: f32, hi: f32, sr: f32) -> Sos {
        let mut s = Self::butter_highpass(order, lo, sr);
        s.stages.extend(Self::butter_lowpass(order, hi, sr).stages);
        s
    }

    /// Linkwitz-Riley low-pass (2x Butterworth of order/2, e.g. LR4 = order 4).
    pub fn linkwitz_riley_lowpass(order: usize, fc: f32, sr: f32) -> Sos {
        let half = (order / 2).max(1);
        let mut s = Self::butter_lowpass(half, fc, sr);
        s.stages.extend(Self::butter_lowpass(half, fc, sr).stages);
        s
    }

    pub fn linkwitz_riley_highpass(order: usize, fc: f32, sr: f32) -> Sos {
        let half = (order / 2).max(1);
        let mut s = Self::butter_highpass(half, fc, sr);
        s.stages.extend(Self::butter_highpass(half, fc, sr).stages);
        s
    }

    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.reset();
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let mut y = x;
        for s in &mut self.stages {
            y = s.process(y);
        }
        y
    }

    pub fn process_block(&mut self, buf: &mut [f32]) {
        for s in &mut self.stages {
            s.process_block(buf);
        }
    }

    pub fn process_into(&mut self, input: &[f32], out: &mut [f32]) {
        let n = input.len().min(out.len());
        out[..n].copy_from_slice(&input[..n]);
        self.process_block(&mut out[..n]);
    }

    pub fn magnitude(&self, f: f32, sr: f32) -> f32 {
        self.stages.iter().map(|s| s.c.magnitude(f, sr)).product()
    }
}

// ---------------------------------------------------------------------------
// one-poles, DC blocker, smoothers
// ---------------------------------------------------------------------------
/// One-pole low-pass y += a (x - y), a = 1 - exp(-2 pi fc / sr).
#[derive(Clone, Copy, Debug, Default)]
pub struct OnePole {
    pub a: f32,
    pub z: f32,
}

impl OnePole {
    pub fn new(sr: f32, fc: f32) -> OnePole {
        let mut p = OnePole { a: 1.0, z: 0.0 };
        p.set_cutoff(sr, fc);
        p
    }
    pub fn with_coef(a: f32) -> OnePole {
        OnePole { a: a.clamp(0.0, 1.0), z: 0.0 }
    }
    #[inline]
    pub fn set_cutoff(&mut self, sr: f32, fc: f32) {
        self.a = 1.0 - (-2.0 * std::f32::consts::PI * fc.max(0.0) / sr).exp();
    }
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.z += self.a * (x - self.z);
        self.z
    }
    /// High-pass output (x - lowpass(x)).
    #[inline]
    pub fn process_hp(&mut self, x: f32) -> f32 {
        x - self.process(x)
    }
    pub fn process_block(&mut self, buf: &mut [f32]) {
        let (a, mut z) = (self.a, self.z);
        for v in buf.iter_mut() {
            z += a * (*v - z);
            *v = z;
        }
        self.z = flush(z);
    }
    pub fn process_block_hp(&mut self, buf: &mut [f32]) {
        let (a, mut z) = (self.a, self.z);
        for v in buf.iter_mut() {
            z += a * (*v - z);
            *v -= z;
        }
        self.z = flush(z);
    }
    pub fn reset(&mut self) {
        self.z = 0.0;
    }
}

/// DC blocker y = x - x1 + r y1 (r ~ 0.995 at 48 kHz = ~38 Hz corner).
#[derive(Clone, Copy, Debug)]
pub struct DcBlocker {
    pub r: f32,
    x1: f32,
    y1: f32,
}

impl DcBlocker {
    pub fn new(sr: f32, fc: f32) -> DcBlocker {
        DcBlocker { r: (-2.0 * std::f32::consts::PI * fc / sr).exp(), x1: 0.0, y1: 0.0 }
    }
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.process(*v);
        }
        self.y1 = flush(self.y1);
    }
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// Exponential parameter smoother (one-pole toward a target).
#[derive(Clone, Copy, Debug)]
pub struct Smoother {
    pub cur: f32,
    pub target: f32,
    pub coef: f32,
}

impl Smoother {
    pub fn new(value: f32, tau_s: f32, sr: f32) -> Smoother {
        Smoother { cur: value, target: value, coef: tau_coef(tau_s, sr) }
    }
    #[inline]
    pub fn set_target(&mut self, t: f32) {
        self.target = t;
    }
    #[inline]
    pub fn snap(&mut self, v: f32) {
        self.cur = v;
        self.target = v;
    }
    #[inline]
    pub fn next(&mut self) -> f32 {
        self.cur += self.coef * (self.target - self.cur);
        if (self.target - self.cur).abs() < 1e-9 {
            self.cur = self.target;
        }
        self.cur
    }
    /// Advance n samples at once (per-block smoothing): returns the value after n steps.
    #[inline]
    pub fn advance(&mut self, n: usize) -> f32 {
        let k = (1.0 - self.coef).powi(n as i32);
        self.cur = self.target + (self.cur - self.target) * k;
        self.cur
    }
    pub fn fill(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.next();
        }
    }
}

/// Linear glide: `set(target, n)` reaches `target` exactly after n samples.
#[derive(Clone, Copy, Debug, Default)]
pub struct Glide {
    pub cur: f32,
    pub target: f32,
    step: f32,
    remaining: usize,
}

impl Glide {
    pub fn new(v: f32) -> Glide {
        Glide { cur: v, target: v, step: 0.0, remaining: 0 }
    }
    pub fn set(&mut self, target: f32, n: usize) {
        self.target = target;
        if n == 0 {
            self.cur = target;
            self.remaining = 0;
            self.step = 0.0;
        } else {
            self.step = (target - self.cur) / n as f32;
            self.remaining = n;
        }
    }
    pub fn snap(&mut self, v: f32) {
        self.set(v, 0);
    }
    #[inline]
    pub fn next(&mut self) -> f32 {
        if self.remaining > 0 {
            self.remaining -= 1;
            self.cur = if self.remaining == 0 { self.target } else { self.cur + self.step };
        }
        self.cur
    }
    #[inline]
    pub fn is_gliding(&self) -> bool {
        self.remaining > 0
    }
    pub fn fill(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.next();
        }
    }
}

// ---------------------------------------------------------------------------
// modal resonator bank
// ---------------------------------------------------------------------------
/// Bank of damped modes as complex one-poles z <- p z + x, p = r e^{jw}; each mode outputs
/// g Im(z). An impulse of amplitude A gives A g r^n sin(n w): a sine-phase decaying mode that
/// starts at zero (no click). The rotation form is stable in f32 for any T60.
#[derive(Clone, Debug, Default)]
pub struct ModalBank {
    re: Vec<f32>,
    im: Vec<f32>,
    pr: Vec<f32>,
    pi: Vec<f32>,
    gain: Vec<f32>,
    n: usize,
}

impl ModalBank {
    pub fn with_capacity(cap: usize) -> ModalBank {
        ModalBank {
            re: vec![0.0; cap],
            im: vec![0.0; cap],
            pr: vec![0.0; cap],
            pi: vec![0.0; cap],
            gain: vec![0.0; cap],
            n: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.re.len()
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Set (or retune) mode i; keeps its state. Modes above 0.49 sr get zero gain.
    #[inline]
    pub fn set_mode(&mut self, i: usize, freq_hz: f32, t60_s: f32, gain: f32, sr: f32) {
        if i >= self.re.len() {
            return;
        }
        let r = t60_radius(t60_s, sr);
        let w = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let ok = freq_hz > 0.0 && freq_hz < 0.49 * sr;
        self.pr[i] = r * w.cos();
        self.pi[i] = r * w.sin();
        self.gain[i] = if ok { gain } else { 0.0 };
        if i >= self.n {
            self.n = i + 1;
        }
    }

    /// Load a material's modes with frequency / decay / gain scaling (keeps states of kept modes).
    pub fn set_modes(&mut self, modes: &[Mode], sr: f32, freq_scale: f32, t60_scale: f32, gain_scale: f32) {
        let m = modes.len().min(self.re.len());
        for (i, md) in modes.iter().take(m).enumerate() {
            self.set_mode(i, md.freq_hz * freq_scale, md.t60_s * t60_scale, md.gain * gain_scale, sr);
        }
        for i in m..self.n {
            self.re[i] = 0.0;
            self.im[i] = 0.0;
            self.gain[i] = 0.0;
        }
        self.n = m;
    }

    pub fn set_count(&mut self, n: usize) {
        let n = n.min(self.re.len());
        for i in n..self.n {
            self.re[i] = 0.0;
            self.im[i] = 0.0;
        }
        self.n = n;
    }

    /// Impulse of amplitude `amp` into every mode (weighted by each mode's gain at output).
    #[inline]
    pub fn strike(&mut self, amp: f32) {
        for i in 0..self.n {
            self.re[i] += amp;
        }
    }

    /// Impulse into one mode with its own amplitude.
    #[inline]
    pub fn strike_mode(&mut self, i: usize, amp: f32) {
        if i < self.n {
            self.re[i] += amp;
        }
    }

    /// out[k] += sum_i g_i Im(z_i[k]) with every mode driven by input[k].
    pub fn process(&mut self, input: &[f32], out: &mut [f32]) {
        let len = input.len().min(out.len());
        for i in 0..self.n {
            let g = self.gain[i];
            if g == 0.0 {
                continue;
            }
            let (pr, pi) = (self.pr[i], self.pi[i]);
            let (mut re, mut im) = (self.re[i], self.im[i]);
            for k in 0..len {
                let nre = pr * re - pi * im + input[k];
                im = pi * re + pr * im;
                re = nre;
                out[k] += g * im;
            }
            self.re[i] = flush(re);
            self.im[i] = flush(im);
        }
    }

    /// Free ring (no input): out += sum of the decaying modes. Silent modes are skipped.
    pub fn ring(&mut self, out: &mut [f32]) {
        for i in 0..self.n {
            let g = self.gain[i];
            let (mut re, mut im) = (self.re[i], self.im[i]);
            if g == 0.0 || (re.abs() + im.abs()) < DENORMAL_EPS {
                self.re[i] = 0.0;
                self.im[i] = 0.0;
                continue;
            }
            let (pr, pi) = (self.pr[i], self.pi[i]);
            for o in out.iter_mut() {
                let nre = pr * re - pi * im;
                im = pi * re + pr * im;
                re = nre;
                *o += g * im;
            }
            self.re[i] = flush(re);
            self.im[i] = flush(im);
        }
    }

    /// Sparse excitation: impulses (sample offset, amplitude) sorted by offset, then ring.
    /// Cheaper than `process` when the input is a few clicks per block.
    pub fn process_impulses(&mut self, impulses: &[(usize, f32)], out: &mut [f32]) {
        let len = out.len();
        for i in 0..self.n {
            let g = self.gain[i];
            if g == 0.0 {
                continue;
            }
            let (pr, pi) = (self.pr[i], self.pi[i]);
            let (mut re, mut im) = (self.re[i], self.im[i]);
            let mut next = 0usize;
            for k in 0..len {
                let nre = pr * re - pi * im;
                im = pi * re + pr * im;
                re = nre;
                while next < impulses.len() && impulses[next].0 == k {
                    re += impulses[next].1;
                    next += 1;
                }
                out[k] += g * im;
            }
            self.re[i] = flush(re);
            self.im[i] = flush(im);
        }
    }

    /// Sum of |state| (0 once every mode has decayed).
    pub fn energy(&self) -> f32 {
        (0..self.n).map(|i| self.re[i].abs() + self.im[i].abs()).sum()
    }

    pub fn reset(&mut self) {
        for i in 0..self.re.len() {
            self.re[i] = 0.0;
            self.im[i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// fractional delay line
// ---------------------------------------------------------------------------
/// Power-of-two ring buffer with linear and cubic (Hermite) fractional reads.
#[derive(Clone, Debug)]
pub struct DelayLine {
    buf: Vec<f32>,
    mask: usize,
    w: usize,
}

impl DelayLine {
    pub fn new(max_delay_samples: usize) -> DelayLine {
        let len = next_pow2(max_delay_samples + 4);
        DelayLine { buf: vec![0.0; len], mask: len - 1, w: 0 }
    }

    pub fn max_delay(&self) -> usize {
        self.buf.len() - 4
    }

    #[inline]
    pub fn push(&mut self, x: f32) {
        self.w = (self.w + 1) & self.mask;
        self.buf[self.w] = x;
    }

    /// Sample written `d` pushes ago (d = 0 is the latest).
    #[inline]
    pub fn tap(&self, d: usize) -> f32 {
        self.buf[(self.w.wrapping_sub(d)) & self.mask]
    }

    #[inline]
    pub fn read_linear(&self, d: f32) -> f32 {
        let d = d.clamp(0.0, self.max_delay() as f32);
        let i = d.floor() as usize;
        let f = d - i as f32;
        let a = self.tap(i);
        let b = self.tap(i + 1);
        a + f * (b - a)
    }

    /// 4-point Hermite interpolation (d >= 1 for a symmetric stencil).
    #[inline]
    pub fn read_cubic(&self, d: f32) -> f32 {
        let d = d.clamp(1.0, (self.max_delay() - 2) as f32);
        let i = d.floor() as usize;
        let f = d - i as f32;
        let xm1 = self.tap(i - 1);
        let x0 = self.tap(i);
        let x1 = self.tap(i + 1);
        let x2 = self.tap(i + 2);
        let c1 = 0.5 * (x1 - xm1);
        let c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
        let c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
        ((c3 * f + c2) * f + c1) * f + x0
    }

    /// Push x, return the sample `d` behind it (linear interpolation).
    #[inline]
    pub fn process(&mut self, x: f32, d: f32) -> f32 {
        self.push(x);
        self.read_linear(d)
    }

    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ---------------------------------------------------------------------------
// coloured noise
// ---------------------------------------------------------------------------
/// Pink filter for white input (Paul Kellet's refined method, +-0.05 dB above 9 Hz).
#[derive(Clone, Copy, Debug, Default)]
pub struct PinkFilter {
    b: [f32; 7],
}

impl PinkFilter {
    #[inline]
    pub fn process(&mut self, white: f32) -> f32 {
        let b = &mut self.b;
        b[0] = 0.99886 * b[0] + white * 0.055_517_9;
        b[1] = 0.99332 * b[1] + white * 0.075_075_9;
        b[2] = 0.96900 * b[2] + white * 0.153_852;
        b[3] = 0.86650 * b[3] + white * 0.310_485_6;
        b[4] = 0.55000 * b[4] + white * 0.532_952_2;
        b[5] = -0.7616 * b[5] - white * 0.016_898;
        let out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
        b[6] = white * 0.115_926;
        out * 0.11
    }
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.process(*v);
        }
        for x in self.b.iter_mut() {
            *x = flush(*x);
        }
    }
}

/// Brown (red) noise: leaky integrator of white input, normalised to roughly unit peak.
#[derive(Clone, Copy, Debug)]
pub struct BrownFilter {
    z: f32,
    leak: f32,
}

impl Default for BrownFilter {
    fn default() -> Self {
        BrownFilter { z: 0.0, leak: 0.998 }
    }
}

impl BrownFilter {
    #[inline]
    pub fn process(&mut self, white: f32) -> f32 {
        self.z = self.leak * self.z + 0.0625 * white;
        self.z * 3.5
    }
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.process(*v);
        }
        self.z = flush(self.z);
    }
}

// ---------------------------------------------------------------------------
// FFT block convolution (fixed block size, any FIR length)
// ---------------------------------------------------------------------------
/// Multichannel FIR matrix applied by FFT overlap-add: out_e = sum_k h_{e,k} * in_k.
///
/// Built for a fixed block size `n`; nfft = next_pow2(n + L - 1). Filters that are all zero are
/// skipped, silent inputs are not transformed. Output tails are carried across blocks, so the
/// result equals linear convolution of the whole stream. `diagonal` builds the k -> k case
/// (decorrelator banks), `matrix` the full E x K case (binaural decoding).
pub struct FirMatrix {
    n: usize,
    nfft: usize,
    taps: usize,
    inputs: usize,
    outputs: usize,
    /// spectra[e * inputs + k], pre-scaled by 1/nfft; None = zero filter.
    spectra: Vec<Option<Vec<Complex<f32>>>>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    time: Vec<f32>,
    xin: Vec<Vec<Complex<f32>>>,
    in_active: Vec<bool>,
    acc_spec: Vec<Complex<f32>>,
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
    /// per-output overlap accumulator (nfft samples)
    acc: Vec<Vec<f32>>,
    acc_live: Vec<bool>,
}

impl FirMatrix {
    /// `firs(e, k)` returns the FIR from input k to output e (empty slice = no path).
    pub fn new<'a, F>(n: usize, outputs: usize, inputs: usize, firs: F) -> FirMatrix
    where
        F: Fn(usize, usize) -> &'a [f32],
    {
        let mut taps = 1usize;
        for e in 0..outputs {
            for k in 0..inputs {
                taps = taps.max(firs(e, k).len());
            }
        }
        let nfft = next_pow2(n + taps - 1);
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(nfft);
        let inv = planner.plan_fft_inverse(nfft);
        let bins = nfft / 2 + 1;
        let mut time = fwd.make_input_vec();
        let mut scratch_f = fwd.make_scratch_vec();
        let mut spectra = Vec::with_capacity(outputs * inputs);
        let norm = 1.0 / nfft as f32;
        for e in 0..outputs {
            for k in 0..inputs {
                let h = firs(e, k);
                if h.is_empty() || h.iter().all(|v| *v == 0.0) {
                    spectra.push(None);
                    continue;
                }
                time.iter_mut().for_each(|v| *v = 0.0);
                for (t, v) in time.iter_mut().zip(h) {
                    *t = *v * norm;
                }
                let mut spec = vec![Complex::new(0.0f32, 0.0); bins];
                let _ = fwd.process_with_scratch(&mut time, &mut spec, &mut scratch_f);
                spectra.push(Some(spec));
            }
        }
        let scratch_i = inv.make_scratch_vec();
        FirMatrix {
            n,
            nfft,
            taps,
            inputs,
            outputs,
            spectra,
            fwd,
            inv,
            time,
            xin: vec![vec![Complex::new(0.0, 0.0); bins]; inputs],
            in_active: vec![false; inputs],
            acc_spec: vec![Complex::new(0.0, 0.0); bins],
            scratch_f,
            scratch_i,
            acc: vec![vec![0.0; nfft]; outputs],
            acc_live: vec![false; outputs],
        }
    }

    /// k -> k filters only.
    pub fn diagonal(n: usize, firs: &[Vec<f32>]) -> FirMatrix {
        let c = firs.len();
        FirMatrix::new(n, c, c, |e, k| if e == k { firs[k].as_slice() } else { &[] })
    }

    pub fn block(&self) -> usize {
        self.n
    }
    pub fn taps(&self) -> usize {
        self.taps
    }
    pub fn inputs(&self) -> usize {
        self.inputs
    }
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// True while any output still has a tail to flush.
    pub fn has_tail(&self) -> bool {
        self.acc_live.iter().any(|&b| b)
    }

    pub fn reset(&mut self) {
        for a in &mut self.acc {
            a.iter_mut().for_each(|v| *v = 0.0);
        }
        self.acc_live.iter_mut().for_each(|v| *v = false);
    }

    /// One block. `input` is `inputs * n` (channel-major), `out` is `outputs * n`; the result
    /// is ADDED into `out` when `add`, else written. `input_active[k] == false` marks input k as
    /// silent this block (skips its FFT); pass None to test every input.
    pub fn process(&mut self, input: &[f32], input_active: Option<&[bool]>, out: &mut [f32], add: bool) {
        let n = self.n;
        let bins = self.nfft / 2 + 1;
        debug_assert!(input.len() >= self.inputs * n && out.len() >= self.outputs * n);
        for k in 0..self.inputs {
            let x = &input[k * n..(k + 1) * n];
            let act = match input_active {
                Some(a) => a.get(k).copied().unwrap_or(true) && !is_silent(x),
                None => !is_silent(x),
            };
            self.in_active[k] = act;
            if !act {
                continue;
            }
            self.time[..n].copy_from_slice(x);
            self.time[n..].iter_mut().for_each(|v| *v = 0.0);
            let _ = self.fwd.process_with_scratch(&mut self.time, &mut self.xin[k], &mut self.scratch_f);
        }
        for e in 0..self.outputs {
            let mut any = false;
            self.acc_spec.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
            for k in 0..self.inputs {
                if !self.in_active[k] {
                    continue;
                }
                if let Some(h) = &self.spectra[e * self.inputs + k] {
                    any = true;
                    let x = &self.xin[k];
                    for b in 0..bins {
                        self.acc_spec[b] += h[b] * x[b];
                    }
                }
            }
            let acc = &mut self.acc[e];
            if any {
                self.acc_spec[0].im = 0.0;
                self.acc_spec[bins - 1].im = 0.0;
                let _ = self.inv.process_with_scratch(&mut self.acc_spec, &mut self.time, &mut self.scratch_i);
                for (a, t) in acc.iter_mut().zip(self.time.iter()) {
                    *a += *t;
                }
                self.acc_live[e] = true;
            }
            let o = &mut out[e * n..(e + 1) * n];
            if self.acc_live[e] {
                if add {
                    for (d, s) in o.iter_mut().zip(acc[..n].iter()) {
                        *d += *s;
                    }
                } else {
                    o.copy_from_slice(&acc[..n]);
                }
                acc.copy_within(n.., 0);
                let len = acc.len();
                acc[len - n..].iter_mut().for_each(|v| *v = 0.0);
                let live = acc.iter().any(|v| v.abs() > DENORMAL_EPS);
                if !live {
                    acc.iter_mut().for_each(|v| *v = 0.0);
                }
                self.acc_live[e] = live;
            } else if !add {
                o.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }
}

/// Direct-form FIR (short filters; also the reference for `FirMatrix`).
#[derive(Clone, Debug)]
pub struct Fir {
    h: Vec<f32>,
    hist: Vec<f32>,
    pos: usize,
}

impl Fir {
    pub fn new(h: &[f32]) -> Fir {
        let l = h.len().max(1);
        let mut hh = h.to_vec();
        if hh.is_empty() {
            hh.push(0.0);
        }
        Fir { h: hh, hist: vec![0.0; 2 * l], pos: 0 }
    }
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let l = self.h.len();
        self.pos = if self.pos == 0 { l - 1 } else { self.pos - 1 };
        self.hist[self.pos] = x;
        self.hist[self.pos + l] = x;
        let w = &self.hist[self.pos..self.pos + l];
        self.h.iter().zip(w).map(|(a, b)| a * b).sum()
    }
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.process(*v);
        }
    }
    pub fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
    }
}
