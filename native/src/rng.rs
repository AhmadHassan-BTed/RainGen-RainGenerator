//! Deterministic, counter-based random streams (the Rust twin of raingen/core/rng.py).
//!
//! Every random draw comes from `RngFactory::get(layer, block)` (or `stream(key, block)` with a
//! precomputed key). A stream is a pure function of (root seed, layer key, block index): the
//! layer key is a stable 64-bit hash of the layer *name* (never creation order), so adding or
//! removing a voice never changes another voice, and rendering block k never depends on
//! whether blocks 0..k-1 were rendered. Streams are xoshiro256++ generators whose 256-bit
//! state is derived from (seed, key, block) through SplitMix64, so they are cheap to create
//! (a few ns) once per voice per block. `hash3` / `uniform_at` give stateless random access
//! (Philox-style keyed counters) for per-event or per-sample draws.
//!
//! Streams are not bit-identical to numpy's Philox: the native engine is deterministic with
//! itself (same seed + scene -> same output), not sample-identical to the Python reference.

/// SplitMix64 finalizer (Stafford variant 13): a bijective 64-bit mixer.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Stable 64-bit hash of a string (FNV-1a, then mixed). Same on every platform and run.
pub fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    mix64(h ^ (s.len() as u64).wrapping_mul(GOLDEN))
}

/// Combine two keys (order matters).
#[inline]
pub fn combine(a: u64, b: u64) -> u64 {
    mix64(a ^ mix64(b.wrapping_add(GOLDEN)).rotate_left(17))
}

/// Stateless keyed hash of three words: random access into a stream (key, a, b).
#[inline]
pub fn hash3(key: u64, a: u64, b: u64) -> u64 {
    let x = mix64(key ^ a.wrapping_mul(GOLDEN));
    mix64(x ^ b.wrapping_mul(0xD1B5_4A32_D192_ED03).rotate_left(29))
}

/// Uniform [0, 1) from 64 random bits (53-bit mantissa).
#[inline]
pub fn u64_to_unit(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform [0, 1) f32 from 32 random bits (24-bit mantissa).
#[inline]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

/// Stateless uniform [0, 1) at (key, a, b).
#[inline]
pub fn uniform_at(key: u64, a: u64, b: u64) -> f64 {
    u64_to_unit(hash3(key, a, b))
}

/// Stateless signed uniform [-1, 1) f32 at (key, counter): cheap per-sample white noise.
#[inline]
pub fn noise_at(key: u64, ctr: u64) -> f32 {
    let x = mix64(key ^ ctr.wrapping_mul(GOLDEN));
    ((x >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0
}

/// Factory of reproducible streams keyed by (root seed, layer, block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RngFactory {
    pub root_seed: u64,
}

impl RngFactory {
    pub fn new(root_seed: u64) -> RngFactory {
        RngFactory { root_seed }
    }

    /// The stream key of a layer name. Compute it once at voice construction and keep it.
    #[inline]
    pub fn key(&self, layer: &str) -> u64 {
        combine(self.root_seed, hash_str(layer))
    }

    /// Key of a layer name + numeric sub-index (e.g. voice id, event number).
    #[inline]
    pub fn key_idx(&self, layer: &str, idx: u64) -> u64 {
        combine(self.key(layer), idx)
    }

    /// Independent stream for `layer` at block `block`.
    #[inline]
    pub fn get(&self, layer: &str, block: u64) -> Rng {
        Rng::from_key(self.key(layer), block)
    }

    /// Independent stream for a precomputed `key` (from `key()`) at block `block`.
    #[inline]
    pub fn stream(&self, key: u64, block: u64) -> Rng {
        Rng::from_key(key, block)
    }

    /// A factory whose seed is derived from this one (variations).
    pub fn child(&self, layer: &str) -> RngFactory {
        RngFactory { root_seed: mix64(self.key(layer) ^ 0xC41D_C41D_C41D_C41D) }
    }
}

/// xoshiro256++ stream with distribution helpers. Plain data; Clone to fork.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    gauss_spare: Option<f64>,
}

impl Rng {
    /// Stream (key, counter): the 256-bit state is SplitMix64 of key ^ counter-mix.
    pub fn from_key(key: u64, counter: u64) -> Rng {
        let mut sm = key ^ mix64(counter.wrapping_add(0x6A09_E667_F3BC_C909));
        let mut next = || {
            sm = sm.wrapping_add(GOLDEN);
            mix64(sm)
        };
        let mut s = [next(), next(), next(), next()];
        if s.iter().all(|&x| x == 0) {
            s[0] = GOLDEN;
        }
        Rng { s, gauss_spare: None }
    }

    /// Stream from a plain seed (tests, design-time tables).
    pub fn seeded(seed: u64) -> Rng {
        Rng::from_key(mix64(seed), 0)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        u64_to_unit(self.next_u64())
    }

    /// Uniform [0, 1) as f32.
    #[inline]
    pub fn uniform_f32(&mut self) -> f32 {
        u32_to_unit_f32(self.next_u32())
    }

    /// Uniform (0, 1]: safe for ln().
    #[inline]
    pub fn uniform_open0(&mut self) -> f64 {
        1.0 - self.uniform()
    }

    /// Uniform [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    #[inline]
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform_f32()
    }

    /// Signed uniform [-1, 1) f32.
    #[inline]
    pub fn bipolar(&mut self) -> f32 {
        self.uniform_f32() * 2.0 - 1.0
    }

    /// Unbiased integer in [0, n) (Lemire). n == 0 returns 0.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let mut m = (self.next_u64() as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let t = n.wrapping_neg() % n;
            while l < t {
                m = (self.next_u64() as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Integer in [lo, hi] inclusive.
    #[inline]
    pub fn int_range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        lo + self.below((hi - lo) as u64 + 1) as i64
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Standard normal (Marsaglia polar; the spare is cached in the stream).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.gauss_spare.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.gauss_spare = Some(v * m);
                return u * m;
            }
        }
    }

    #[inline]
    pub fn normal_f32(&mut self) -> f32 {
        self.normal() as f32
    }

    #[inline]
    pub fn gauss(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.normal()
    }

    /// exp(N(mu, sigma)).
    #[inline]
    pub fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.normal()).exp()
    }

    /// Exponential with the given rate (mean 1/rate). rate <= 0 -> +inf.
    #[inline]
    pub fn exponential(&mut self, rate: f64) -> f64 {
        if rate <= 0.0 {
            return f64::INFINITY;
        }
        -self.uniform_open0().ln() / rate
    }

    /// Poisson(lam): inversion for small means, PTRS (Hormann 1993) for large ones.
    pub fn poisson(&mut self, lam: f64) -> u64 {
        if !(lam > 0.0) || !lam.is_finite() {
            return 0;
        }
        if lam < 12.0 {
            let l = (-lam).exp();
            let mut k = 0u64;
            let mut p = self.uniform();
            while p > l {
                k += 1;
                p *= self.uniform();
                if k > 1000 {
                    break;
                }
            }
            return k;
        }
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.024_83 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.uniform() - 0.5;
            let v = self.uniform();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
            if us >= 0.07 && v <= vr {
                return k as u64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if (v * invalpha / (a / (us * us) + b)).ln()
                <= -lam + k * loglam - crate::types::ln_gamma(k + 1.0)
            {
                return k as u64;
            }
        }
    }

    /// Gamma(shape k, scale theta) (Marsaglia & Tsang 2000).
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if !(shape > 0.0) {
            return 0.0;
        }
        if shape < 1.0 {
            let u = self.uniform_open0();
            return self.gamma(shape + 1.0, scale) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = self.normal();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.uniform_open0();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }

    /// Beta(a, b).
    pub fn beta(&mut self, a: f64, b: f64) -> f64 {
        let x = self.gamma(a, 1.0);
        let y = self.gamma(b, 1.0);
        if x + y > 0.0 {
            x / (x + y)
        } else {
            0.5
        }
    }

    /// Index drawn with probability proportional to `weights` (non-positive weights never win).
    pub fn choose_weighted(&mut self, weights: &[f32]) -> usize {
        let total: f64 = weights.iter().map(|&w| w.max(0.0) as f64).sum();
        if total <= 0.0 || weights.is_empty() {
            return 0;
        }
        let mut r = self.uniform() * total;
        for (i, &w) in weights.iter().enumerate() {
            let w = w.max(0.0) as f64;
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            xs.swap(i, j);
        }
    }

    /// Fill with uniform [-1, 1) (white noise, flat spectrum, variance 1/3).
    pub fn fill_bipolar(&mut self, out: &mut [f32]) {
        let mut i = 0;
        let n = out.len();
        while i + 2 <= n {
            let x = self.next_u64();
            out[i] = ((x >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
            out[i + 1] = (((x >> 8) & 0xFF_FFFF) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
            i += 2;
        }
        if i < n {
            out[i] = self.bipolar();
        }
    }

    /// Fill with standard normal samples.
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.normal() as f32;
        }
    }

    /// Fill with uniform [0, 1).
    pub fn fill_uniform(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.uniform_f32();
        }
    }
}
