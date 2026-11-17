//! Event timing for precipitation: a doubly stochastic (Cox) process (port of
//! raingen/synth/events.py).
//!
//! Impacts are Poisson within a block at an intensity that is itself modulated by multiplicative
//! 1/f-type noise (`RateModulator`, a sum of octave-spaced Ornstein-Uhlenbeck processes,
//! mean one). Within a block the intensity ramps linearly from the previous to the current
//! modulator value and event times follow that ramp exactly (closed-form inverse CDF).

use crate::rng::{Rng, RngFactory};

/// Inverse CDF on [0, 1) of an intensity that ramps linearly from m0 to m1.
#[inline]
pub fn ramp_position(u: f64, m0: f64, m1: f64) -> f64 {
    let m0 = m0.max(0.0);
    let m1 = m1.max(0.0);
    let a = m1 - m0;
    let tot = 0.5 * (m0 + m1);
    let x = if a.abs() <= 1e-9 * tot.max(1e-300) {
        u
    } else {
        ((m0 * m0 + 2.0 * a * u * tot).max(0.0).sqrt() - m0) / a
    };
    x.clamp(0.0, 1.0 - f64::EPSILON)
}

/// Sample offsets in [0, n) of Poisson events at `rate_per_s` with a two-point intensity ramp
/// (m0 -> m1; pass 1, 1 for a flat rate). Writes at most `out.capacity()` events (no allocation).
pub fn cox_offsets(rng: &mut Rng, rate_per_s: f64, n: usize, sr: f64, m0: f64, m1: f64, out: &mut Vec<u32>) {
    out.clear();
    if n == 0 || !(rate_per_s > 0.0) {
        return;
    }
    let lam = rate_per_s * 0.5 * (m0.max(0.0) + m1.max(0.0)) * n as f64 / sr;
    if !(lam > 0.0) {
        return;
    }
    let k = (rng.poisson(lam) as usize).min(out.capacity());
    let flat = (m0 - m1).abs() <= 1e-12;
    for _ in 0..k {
        let u = rng.uniform();
        let x = if flat { u } else { ramp_position(u, m0, m1) };
        out.push(((x * n as f64) as usize).min(n - 1) as u32);
    }
}

/// Cox offsets with an event budget: above `cap_per_block` expected events the cap is drawn and
/// the energy-preserving amplitude factor sqrt(expected / cap) is returned (granular density).
pub fn thinned_offsets(
    rng: &mut Rng,
    rate_per_s: f64,
    n: usize,
    sr: f64,
    cap_per_block: f64,
    m0: f64,
    m1: f64,
    out: &mut Vec<u32>,
) -> f64 {
    let mean_mod = 0.5 * (m0 + m1);
    let expected = rate_per_s.max(0.0) * n as f64 / sr * mean_mod;
    if cap_per_block > 0.0 && expected > cap_per_block {
        let scale = (expected / cap_per_block).sqrt();
        let rate = rate_per_s * cap_per_block / expected;
        cox_offsets(rng, rate, n, sr, m0, m1, out);
        return scale;
    }
    cox_offsets(rng, rate_per_s, n, sr, m0, m1, out);
    1.0
}

const MAX_OU: usize = 16;

/// Mean-one multiplicative 1/f-type modulation: exp(sigma s - sigma^2 / 2), s the normalised sum
/// of equal-variance OU processes with time constants tau_min 2^k. Deterministic: block b draws
/// from stream (key, b); `step` must be called once per block, in order.
#[derive(Clone, Debug)]
pub struct RateModulator {
    key: u64,
    key_init: u64,
    sigma: f64,
    k: usize,
    taus: [f64; MAX_OU],
    x: [f64; MAX_OU],
    a: [f64; MAX_OU],
    b: [f64; MAX_OU],
    a_n: usize,
    sr: f64,
    pub value: f64,
    pub prev: f64,
    last_block: Option<u64>,
}

impl RateModulator {
    pub fn new(rngf: &RngFactory, layer: &str, sigma: f64, tau_min_s: f64, tau_max_s: f64, sr: f64) -> RateModulator {
        let k = (((tau_max_s.max(tau_min_s) / tau_min_s).log2().floor() as usize) + 1).clamp(1, MAX_OU);
        let mut taus = [0.0; MAX_OU];
        for (i, t) in taus.iter_mut().enumerate().take(k) {
            *t = tau_min_s * 2f64.powi(i as i32);
        }
        let mut m = RateModulator {
            key: rngf.key(layer),
            key_init: rngf.key(&format!("{layer}/init")),
            sigma,
            k,
            taus,
            x: [0.0; MAX_OU],
            a: [0.0; MAX_OU],
            b: [0.0; MAX_OU],
            a_n: 0,
            sr,
            value: 1.0,
            prev: 1.0,
            last_block: None,
        };
        m.reset();
        m
    }

    /// Back to the stationary start state drawn at construction.
    pub fn reset(&mut self) {
        let mut r = Rng::from_key(self.key_init, 0);
        for i in 0..self.k {
            self.x[i] = r.normal();
        }
        self.last_block = None;
        self.value = self.current();
        self.prev = self.value;
    }

    #[inline]
    fn current(&self) -> f64 {
        let s: f64 = self.x[..self.k].iter().sum::<f64>() / (self.k as f64).sqrt();
        (self.sigma * s - 0.5 * self.sigma * self.sigma).exp()
    }

    /// Advance to block `block` (n frames); returns the new value (prev holds the last one).
    pub fn step(&mut self, block: u64, n: usize) -> f64 {
        if self.last_block == Some(block) {
            return self.value;
        }
        if self.a_n != n {
            let dt = n as f64 / self.sr;
            for i in 0..self.k {
                self.a[i] = (-dt / self.taus[i]).exp();
                self.b[i] = (1.0 - self.a[i] * self.a[i]).max(0.0).sqrt();
            }
            self.a_n = n;
        }
        let mut r = Rng::from_key(self.key, block);
        for i in 0..self.k {
            self.x[i] = self.a[i] * self.x[i] + self.b[i] * r.normal();
        }
        self.prev = self.value;
        self.value = self.current();
        self.last_block = Some(block);
        self.value
    }
}
