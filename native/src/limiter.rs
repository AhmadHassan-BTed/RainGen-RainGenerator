//! Streaming look-ahead true-peak safety limiter, the Rust twin of
//! raingen.master.limiter.TruePeakLimiter.
//!
//! Detection runs on the 8x polyphase-oversampled signal (48 taps per phase, Kaiser beta 10,
//! cutoff at the input Nyquist: the same interpolator as raingen's LoudnessMeter, flat to
//! ~20 kHz), linked across channels, plus the sample peak delayed by the interpolator's group
//! delay. A sample is only interpolated where the l1 bound of its stencil can reach the
//! ceiling, so a quiet bed costs a few compares per sample. Required attenuation
//! a = max(0, ln(peak / ceiling)) (nepers) is
//!   1. max-held over W = L + 3 samples (monotonic deque),
//!   2. released exponentially, r[n] = max(held[n], beta r[n-1]),
//!   3. smoothed by two cascaded moving averages of total span L (an S-shaped attack that is
//!      complete exactly when the peak arrives),
//! and applied as g = exp(-s) to the audio delayed by L + 24 samples (look-ahead + detector
//! delay). Every value inside the smoothing span already meets the peak's requirement, so the
//! average does too: no overshoot and no clipper on the signal path; an idle limiter passes the
//! delayed input bit-exactly. Non-finite input samples are zeroed. A final clamp at +-1.0
//! (0 dBFS) only guards against numerical accidents.
//! Real-time safe: buffers are allocated in `new` (for blocks up to 4096 frames; a larger
//! block grows them once).

const FACTOR: usize = 8;
const TPP: usize = 48; // taps per phase
const KAISER_BETA: f64 = 10.0;
const CHUNK: usize = 16; // window-max chunking
const PREALLOC_FRAMES: usize = 4096;
const NEPER_TO_DB: f32 = 8.685_889_6; // 20 / ln 10

use crate::spatial::simd;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..200 {
        term *= y / (k * k) as f64;
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// scipy.signal.firwin(FACTOR * TPP, 1 / FACTOR, window=("kaiser", beta)) * FACTOR, split into
/// FACTOR phases of TPP taps; returned per phase in REVERSED order (dot with the oldest-first
/// window of the last TPP input samples).
fn polyphase_kernel() -> Vec<[f32; TPP]> {
    let nt = FACTOR * TPP;
    let alpha = (nt - 1) as f64 / 2.0;
    let cutoff = 1.0 / FACTOR as f64;
    let i0b = bessel_i0(KAISER_BETA);
    let mut h = vec![0.0f64; nt];
    for (i, hv) in h.iter_mut().enumerate() {
        let m = i as f64 - alpha;
        let x = cutoff * m;
        let sinc = if x.abs() < 1e-15 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
        let r = 2.0 * i as f64 / (nt - 1) as f64 - 1.0;
        let w = bessel_i0(KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / i0b;
        *hv = cutoff * sinc * w;
    }
    let s: f64 = h.iter().sum();
    for hv in h.iter_mut() {
        *hv = *hv / s * FACTOR as f64;
    }
    (0..FACTOR)
        .map(|p| {
            let mut k = [0.0f32; TPP];
            for j in 0..TPP {
                // phase p tap j multiplies x[m - j]; window index i = TPP - 1 - j
                k[TPP - 1 - j] = h[j * FACTOR + p] as f32;
            }
            k
        })
        .collect()
}

/// The polyphase kernel transposed to taps x FACTOR (oldest-first window index, then phase), so
/// one vector multiply-add per tap evaluates all FACTOR interpolated values.
fn transpose_kernel(k: &[[f32; TPP]]) -> Vec<[f32; FACTOR]> {
    (0..TPP)
        .map(|i| {
            let mut row = [0.0f32; FACTOR];
            for (p, kp) in k.iter().enumerate().take(FACTOR) {
                row[p] = kp[i];
            }
            row
        })
        .collect()
}

pub struct Limiter {
    pub sr: u32,
    pub channels: usize,
    pub ceiling_dbtp: f32,
    pub enabled: bool,
    ceil: f32,
    l: usize,
    w: usize,
    l1: usize,
    l2: usize,
    beta: f64,
    kernel_t: Vec<[f32; FACTOR]>,
    l1_bound: f32,
    /// sample-peak delay matching the interpolator (ceil of its group delay)
    c: usize,
    // detector: per channel history (TPP - 1) + block
    ext: Vec<Vec<f32>>,
    cmax: Vec<f32>,
    pk: Vec<f32>,
    spk_hist: Vec<f32>,
    spk_pos: usize,
    gain: Vec<f32>,
    // sliding max deque over a (values + absolute indices)
    dq_val: Vec<f64>,
    dq_idx: Vec<u64>,
    dq_head: usize,
    dq_len: usize,
    t: u64,
    rel: f64,
    quiet: usize,
    b1: Vec<f64>,
    b1_pos: usize,
    b1_sum: f64,
    b2: Vec<f64>,
    b2_pos: usize,
    b2_sum: f64,
    resum_count: u32,
    // audio delay per channel: `latency` history + block
    delay: Vec<Vec<f32>>,
    latency: usize,
    planar: Vec<f32>,
    pub gain_reduction_db: f32,
    pub max_gain_reduction_db: f32,
}

impl Limiter {
    pub fn new(sr: u32, channels: usize, ceiling_dbtp: f32, lookahead_ms: f32, release_ms: f32) -> Limiter {
        let channels = channels.max(1);
        let l = ((lookahead_ms * 1e-3 * sr as f32).round() as usize).max(2);
        let l1 = l / 2;
        let l2 = l - l1;
        let d = (FACTOR * TPP - 1) as f64 / 2.0 / FACTOR as f64;
        let (f, c) = (d.floor() as usize, d.ceil() as usize);
        let w = l + 2 + (c - f);
        let latency = l + c;
        let kernel = polyphase_kernel();
        let l1_bound = kernel
            .iter()
            .map(|k| k.iter().map(|v| v.abs() as f64).sum::<f64>())
            .fold(0.0f64, f64::max) as f32
            * (1.0 + 1e-6);
        let nmax = PREALLOC_FRAMES;
        let mut lim = Limiter {
            sr,
            channels,
            ceiling_dbtp,
            enabled: true,
            ceil: 1.0,
            l,
            w,
            l1,
            l2,
            beta: (-1.0 / ((release_ms.max(1e-3) as f64) * 1e-3 * sr as f64)).exp(),
            kernel_t: transpose_kernel(&kernel),
            l1_bound,
            c,
            ext: vec![vec![0.0; TPP - 1 + nmax]; channels],
            cmax: vec![0.0; (TPP - 1 + nmax).div_ceil(CHUNK) + 1],
            pk: vec![0.0; nmax],
            spk_hist: vec![0.0; c.max(1)],
            spk_pos: 0,
            gain: vec![1.0; nmax],
            dq_val: vec![0.0; w + 2],
            dq_idx: vec![0; w + 2],
            dq_head: 0,
            dq_len: 0,
            t: 0,
            rel: 0.0,
            quiet: 0,
            b1: vec![0.0; l1.max(1)],
            b1_pos: 0,
            b1_sum: 0.0,
            b2: vec![0.0; l2.max(1)],
            b2_pos: 0,
            b2_sum: 0.0,
            resum_count: 0,
            delay: vec![vec![0.0; latency + nmax]; channels],
            latency,
            planar: Vec::new(),
            gain_reduction_db: 0.0,
            max_gain_reduction_db: 0.0,
        };
        lim.set_ceiling(ceiling_dbtp);
        lim
    }

    /// Defaults of the engine's safety stage: -1 dBTP, 5 ms look-ahead, 80 ms release.
    pub fn safety(sr: u32, channels: usize) -> Limiter {
        Limiter::new(sr, channels, -1.0, 5.0, 80.0)
    }

    pub fn set_ceiling(&mut self, ceiling_dbtp: f32) {
        self.ceiling_dbtp = ceiling_dbtp;
        self.ceil = 10f32.powf((ceiling_dbtp - 0.05) / 20.0);
    }

    /// Output delay in samples (constant, also when disabled).
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// The look-ahead in samples (without the detector delay).
    pub fn lookahead(&self) -> usize {
        self.l
    }

    #[doc(hidden)]
    pub fn box_spans(&self) -> (usize, usize) {
        (self.l1, self.l2)
    }

    pub fn reset(&mut self) {
        for e in &mut self.ext {
            e.iter_mut().for_each(|v| *v = 0.0);
        }
        self.spk_hist.iter_mut().for_each(|v| *v = 0.0);
        self.spk_pos = 0;
        self.dq_head = 0;
        self.dq_len = 0;
        self.t = 0;
        self.rel = 0.0;
        self.quiet = 0;
        self.b1.iter_mut().for_each(|v| *v = 0.0);
        self.b2.iter_mut().for_each(|v| *v = 0.0);
        self.b1_sum = 0.0;
        self.b2_sum = 0.0;
        self.b1_pos = 0;
        self.b2_pos = 0;
        for d in &mut self.delay {
            d.iter_mut().for_each(|v| *v = 0.0);
        }
        self.gain_reduction_db = 0.0;
        self.max_gain_reduction_db = 0.0;
    }

    fn ensure(&mut self, n: usize) {
        if self.pk.len() >= n {
            return;
        }
        for e in &mut self.ext {
            e.resize(TPP - 1 + n, 0.0);
        }
        self.cmax.resize((TPP - 1 + n).div_ceil(CHUNK) + 1, 0.0);
        self.pk.resize(n, 0.0);
        self.gain.resize(n, 1.0);
        let lat = self.latency;
        for d in &mut self.delay {
            d.resize(lat + n, 0.0);
        }
    }

    /// Limit an interleaved block (frames x channels) in place.
    pub fn process_interleaved(&mut self, buf: &mut [f32]) {
        let c = self.channels;
        let n = buf.len() / c;
        if n == 0 {
            return;
        }
        let mut planar = std::mem::take(&mut self.planar);
        if planar.len() < c * n {
            planar.resize(c * n, 0.0);
        }
        for i in 0..n {
            for ch in 0..c {
                planar[ch * n + i] = buf[i * c + ch];
            }
        }
        self.process_planar(&mut planar[..c * n], n);
        for i in 0..n {
            for ch in 0..c {
                buf[i * c + ch] = planar[ch * n + i];
            }
        }
        self.planar = planar;
    }

    /// Limit a channel-major block (channels x n) in place; output is delayed by `latency()`.
    pub fn process_planar(&mut self, buf: &mut [f32], n: usize) {
        let c = self.channels;
        if n == 0 || buf.len() < c * n {
            return;
        }
        self.ensure(n);
        let h = TPP - 1;

        // sanitise, detector history, sample peaks
        for ch in 0..c {
            let x = &mut buf[ch * n..(ch + 1) * n];
            for v in x.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            self.ext[ch][h..h + n].copy_from_slice(x);
        }
        self.pk[..n].iter_mut().for_each(|v| *v = 0.0);
        if self.enabled {
            self.detect(n);
        }

        // gain computer (scalar, per frame)
        let mut max_s2 = 0.0f64;
        let cap = self.dq_val.len();
        for m in 0..n {
            let pk = self.pk[m];
            let a = if pk > self.ceil { ((pk / self.ceil) as f64).ln() } else { 0.0 };
            // sliding max over the last w values of a
            while self.dq_len > 0 {
                let back = (self.dq_head + self.dq_len - 1) % cap;
                if self.dq_val[back] <= a {
                    self.dq_len -= 1;
                } else {
                    break;
                }
            }
            let pos = (self.dq_head + self.dq_len) % cap;
            self.dq_val[pos] = a;
            self.dq_idx[pos] = self.t;
            self.dq_len += 1;
            let w = self.w as u64;
            while self.dq_len > 0 && self.dq_idx[self.dq_head] + w <= self.t {
                self.dq_head = (self.dq_head + 1) % cap;
                self.dq_len -= 1;
            }
            let held = self.dq_val[self.dq_head];
            self.t += 1;
            let r = held.max(self.rel * self.beta);
            self.rel = if r < 1e-9 { 0.0 } else { r };
            if self.rel == 0.0 {
                self.quiet += 1;
            } else {
                self.quiet = 0;
            }
            let s2 = if self.quiet > self.l1 + self.l2 + 1 {
                // both boxes hold only zeros: drop the running sums' rounding residue
                if self.b1_sum != 0.0 || self.b2_sum != 0.0 {
                    self.b1.iter_mut().for_each(|v| *v = 0.0);
                    self.b2.iter_mut().for_each(|v| *v = 0.0);
                    self.b1_sum = 0.0;
                    self.b2_sum = 0.0;
                }
                self.b1_pos = (self.b1_pos + 1) % self.b1.len();
                self.b2_pos = (self.b2_pos + 1) % self.b2.len();
                0.0
            } else {
                self.b1_sum += self.rel - self.b1[self.b1_pos];
                self.b1[self.b1_pos] = self.rel;
                self.b1_pos = (self.b1_pos + 1) % self.b1.len();
                let s1 = (self.b1_sum / self.b1.len() as f64).max(0.0);
                self.b2_sum += s1 - self.b2[self.b2_pos];
                self.b2[self.b2_pos] = s1;
                self.b2_pos = (self.b2_pos + 1) % self.b2.len();
                self.resum_count += 1;
                if self.resum_count >= 1 << 16 {
                    self.resum_count = 0;
                    self.b1_sum = self.b1.iter().sum();
                    self.b2_sum = self.b2.iter().sum();
                }
                (self.b2_sum / self.b2.len() as f64).max(0.0)
            };
            if s2 > max_s2 {
                max_s2 = s2;
            }
            self.gain[m] = if s2 > 0.0 { (-s2).exp() as f32 } else { 1.0 };
        }
        let gr = max_s2 as f32 * NEPER_TO_DB;
        self.gain_reduction_db = gr;
        if gr > self.max_gain_reduction_db {
            self.max_gain_reduction_db = gr;
        }

        // delay the audio by `latency` and apply the gain
        let lat = self.latency;
        let gain = &self.gain[..n];
        let unity = gr == 0.0;
        for ch in 0..c {
            let line = &mut self.delay[ch];
            let x = &mut buf[ch * n..(ch + 1) * n];
            line[lat..lat + n].copy_from_slice(x);
            if unity {
                x.copy_from_slice(&line[..n]);
            } else {
                for ((o, d), g) in x.iter_mut().zip(&line[..n]).zip(gain) {
                    *o = (*d * *g).clamp(-1.0, 1.0);
                }
            }
            line.copy_within(n..n + lat, 0);
        }
        // detector history for the next block
        for e in &mut self.ext {
            e.copy_within(n..n + h, 0);
        }
    }

    /// pk[m] = max over channels of the sample peak (delayed by c) and, where the stencil's l1
    /// bound can reach the ceiling, the FACTOR interpolated values behind frame m.
    fn detect(&mut self, n: usize) {
        let h = TPP - 1;
        let thr = self.ceil / self.l1_bound;
        // sample peaks, delayed by c through a ring
        for m in 0..n {
            let mut s = 0.0f32;
            for e in &self.ext {
                s = s.max(e[h + m].abs());
            }
            let cl = self.spk_hist.len();
            let delayed = if self.c == 0 { s } else { self.spk_hist[self.spk_pos] };
            if self.c > 0 {
                self.spk_hist[self.spk_pos] = s;
                self.spk_pos = (self.spk_pos + 1) % cl;
            }
            self.pk[m] = delayed;
        }
        let len = h + n;
        let nchunks = len.div_ceil(CHUNK);
        for e in &self.ext {
            if simd::max_abs(&e[..len]) < thr {
                continue;
            }
            for (q, cm) in self.cmax[..nchunks].iter_mut().enumerate() {
                let lo = q * CHUNK;
                let hi = (lo + CHUNK).min(len);
                *cm = simd::max_abs(&e[lo..hi]);
            }
            for m in 0..n {
                // window ext[m .. m + TPP] (newest = block frame m)
                let (q0, q1) = (m / CHUNK, (m + TPP - 1) / CHUNK);
                let mut wmax = 0.0f32;
                for q in q0..=q1 {
                    wmax = wmax.max(self.cmax[q]);
                }
                if wmax < thr {
                    continue;
                }
                let y = simd::poly8(&self.kernel_t, &e[m..m + TPP]);
                let mut p = self.pk[m];
                for v in y {
                    p = p.max(v.abs());
                }
                self.pk[m] = p;
            }
        }
    }
}
