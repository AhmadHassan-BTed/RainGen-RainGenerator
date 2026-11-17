//! NumPy-compatible random streams, used where the native engine must agree draw for draw with
//! the Python physics World: the shared gust process (layer "wind.gust") and the storm-cell flash
//! schedule (layers "storm.flash.<cell>").
//!
//! `raingen.core.rng.RngFactory.get(layer, block)` is
//! `numpy.random.Generator(Philox(key=blake2b_128(f"{seed:032x}|{layer}"), counter=[0, block, 0, 0]))`.
//! This module reproduces that exactly: BLAKE2b-128 layer keys, the Philox4x64-10 bit generator
//! with NumPy's buffering, and the Generator distributions the World uses (standard_normal by
//! NumPy's 256-layer ziggurat, uniform, lognormal, poisson). Everything is allocation-free.

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// BLAKE2b
// ---------------------------------------------------------------------------
const B2_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const B2_SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// Streaming BLAKE2b (unkeyed) with a configurable digest length (1..=64 bytes).
pub struct Blake2b {
    h: [u64; 8],
    buf: [u8; 128],
    buflen: usize,
    t: u128,
    outlen: usize,
}

#[inline(always)]
fn b2_g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

impl Blake2b {
    pub fn new(outlen: usize) -> Blake2b {
        let outlen = outlen.clamp(1, 64);
        let mut h = B2_IV;
        h[0] ^= 0x0101_0000 ^ outlen as u64;
        Blake2b { h, buf: [0; 128], buflen: 0, t: 0, outlen }
    }

    fn compress(&mut self, last: bool) {
        let mut m = [0u64; 16];
        for (i, w) in m.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&self.buf[8 * i..8 * i + 8]);
            *w = u64::from_le_bytes(b);
        }
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..].copy_from_slice(&B2_IV);
        v[12] ^= self.t as u64;
        v[13] ^= (self.t >> 64) as u64;
        if last {
            v[14] = !v[14];
        }
        for s in B2_SIGMA.iter() {
            b2_g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            b2_g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            b2_g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            b2_g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            b2_g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            b2_g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            b2_g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            b2_g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            if self.buflen == 128 {
                self.t += 128;
                self.compress(false);
                self.buflen = 0;
            }
            let k = (128 - self.buflen).min(data.len());
            self.buf[self.buflen..self.buflen + k].copy_from_slice(&data[..k]);
            self.buflen += k;
            data = &data[k..];
        }
    }

    /// Digest (first `outlen` bytes are meaningful).
    pub fn finalize(mut self) -> [u8; 64] {
        self.t += self.buflen as u128;
        for b in self.buf[self.buflen..].iter_mut() {
            *b = 0;
        }
        self.compress(true);
        let mut out = [0u8; 64];
        for i in 0..8 {
            out[8 * i..8 * i + 8].copy_from_slice(&self.h[i].to_le_bytes());
        }
        for b in out[self.outlen..].iter_mut() {
            *b = 0;
        }
        out
    }
}

/// `raingen.core.rng._layer_key(seed, layer)` for a layer name given in parts (concatenated):
/// BLAKE2b-128 of `f"{seed:032x}|{layer}"`, read as two little-endian u64 words.
pub fn layer_key_parts(root_seed: u64, parts: &[&[u8]]) -> [u64; 2] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hex = [b'0'; 33];
    for i in 0..16 {
        let nib = (root_seed >> (4 * (15 - i))) & 0xF;
        hex[16 + i] = HEX[nib as usize];
    }
    hex[32] = b'|';
    let mut b = Blake2b::new(16);
    b.update(&hex);
    for p in parts {
        b.update(p);
    }
    let d = b.finalize();
    let mut k0 = [0u8; 8];
    let mut k1 = [0u8; 8];
    k0.copy_from_slice(&d[0..8]);
    k1.copy_from_slice(&d[8..16]);
    [u64::from_le_bytes(k0), u64::from_le_bytes(k1)]
}

pub fn layer_key(root_seed: u64, layer: &str) -> [u64; 2] {
    layer_key_parts(root_seed, &[layer.as_bytes()])
}

// ---------------------------------------------------------------------------
// Philox4x64-10 with NumPy's buffering
// ---------------------------------------------------------------------------
const PHILOX_M0: u64 = 0xD2E7_470E_E14C_6C93;
const PHILOX_M1: u64 = 0xCA5A_8263_9512_1157;
const PHILOX_W0: u64 = 0x9E37_79B9_7F4A_7C15;
const PHILOX_W1: u64 = 0xBB67_AE85_84CA_A73B;

#[inline(always)]
fn mulhilo(a: u64, b: u64) -> (u64, u64) {
    let p = (a as u128) * (b as u128);
    ((p >> 64) as u64, p as u64)
}

#[inline]
fn philox4x64_10(mut ctr: [u64; 4], mut key: [u64; 2]) -> [u64; 4] {
    for r in 0..10 {
        if r > 0 {
            key[0] = key[0].wrapping_add(PHILOX_W0);
            key[1] = key[1].wrapping_add(PHILOX_W1);
        }
        let (hi0, lo0) = mulhilo(PHILOX_M0, ctr[0]);
        let (hi1, lo1) = mulhilo(PHILOX_M1, ctr[2]);
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
    }
    ctr
}

/// `numpy.random.Generator(Philox(key, counter))`.
#[derive(Clone, Debug)]
pub struct NpRng {
    ctr: [u64; 4],
    key: [u64; 2],
    buf: [u64; 4],
    pos: usize,
}

struct Zig {
    ki: [u64; 256],
    wi: [f64; 256],
    fi: [f64; 256],
}

const ZIG_R: f64 = 3.654_152_885_361_008_796_351_947_251_8;
const ZIG_INV_R: f64 = 0.273_661_237_329_758_272_033_382_475_96;

fn zig_tables() -> &'static Zig {
    static Z: OnceLock<Zig> = OnceLock::new();
    Z.get_or_init(|| {
        // Marsaglia & Tsang (2000) 256-layer normal ziggurat on 52-bit magnitudes, as NumPy's tables
        let m1 = 4_503_599_627_370_496.0_f64; // 2^52
        let v = 0.004_928_673_233_99_f64;
        let mut ki = [0u64; 256];
        let mut wi = [0f64; 256];
        let mut fi = [0f64; 256];
        let mut dn = ZIG_R;
        let mut tn = dn;
        let q = v / (-0.5 * dn * dn).exp();
        ki[0] = ((dn / q) * m1) as u64;
        ki[1] = 0;
        wi[0] = q / m1;
        wi[255] = dn / m1;
        fi[0] = 1.0;
        fi[255] = (-0.5 * dn * dn).exp();
        for i in (1..=254).rev() {
            dn = (-2.0 * (v / dn + (-0.5 * dn * dn).exp()).ln()).sqrt();
            ki[i + 1] = ((dn / tn) * m1) as u64;
            tn = dn;
            fi[i] = (-0.5 * dn * dn).exp();
            wi[i] = dn / m1;
        }
        Zig { ki, wi, fi }
    })
}

/// Build the ziggurat tables now (call from constructors so the audio thread never does).
pub fn warm_up() {
    let _ = zig_tables();
}

impl NpRng {
    pub fn new(key: [u64; 2], counter: [u64; 4]) -> NpRng {
        NpRng { ctr: counter, key, buf: [0; 4], pos: 4 }
    }

    /// `RngFactory(seed).get(layer, block)` with a precomputed layer key.
    pub fn for_block(key: [u64; 2], block: u64) -> NpRng {
        NpRng::new(key, [0, block, 0, 0])
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        if self.pos < 4 {
            let v = self.buf[self.pos];
            self.pos += 1;
            return v;
        }
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
            if self.ctr[1] == 0 {
                self.ctr[2] = self.ctr[2].wrapping_add(1);
                if self.ctr[2] == 0 {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
        self.buf = philox4x64_10(self.ctr, self.key);
        self.pos = 1;
        self.buf[0]
    }

    #[inline]
    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Generator.random()
    #[inline]
    pub fn random(&mut self) -> f64 {
        self.next_double()
    }

    /// Generator.standard_normal() (NumPy's ziggurat, bit-compatible draws).
    pub fn standard_normal(&mut self) -> f64 {
        let z = zig_tables();
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * z.wi[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < z.ki[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIG_INV_R * (-self.next_double()).ln_1p();
                    let yy = -(-self.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 { -(ZIG_R + xx) } else { ZIG_R + xx };
                    }
                }
            } else if (z.fi[idx - 1] - z.fi[idx]) * self.next_double() + z.fi[idx] < (-0.5 * x * x).exp() {
                return x;
            }
        }
    }

    /// Generator.uniform(lo, hi)
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_double()
    }

    /// Generator.normal(loc, scale)
    #[inline]
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        loc + scale * self.standard_normal()
    }

    /// Generator.lognormal(mean, sigma)
    #[inline]
    pub fn lognormal(&mut self, mean: f64, sigma: f64) -> f64 {
        self.normal(mean, sigma).exp()
    }

    /// Generator.poisson(lam) (multiplication method below 10, PTRS above, as NumPy).
    pub fn poisson(&mut self, lam: f64) -> u64 {
        if !(lam > 0.0) || !lam.is_finite() {
            return 0;
        }
        if lam >= 10.0 {
            let slam = lam.sqrt();
            let loglam = lam.ln();
            let b = 0.931 + 2.53 * slam;
            let a = -0.059 + 0.02483 * b;
            let invalpha = 1.1239 + 1.1328 / (b - 3.4);
            let vr = 0.9277 - 3.6224 / (b - 2.0);
            loop {
                let u = self.next_double() - 0.5;
                let v = self.next_double();
                let us = 0.5 - u.abs();
                let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
                if us >= 0.07 && v <= vr {
                    return k.max(0.0) as u64;
                }
                if k < 0.0 || (us < 0.013 && v > us) {
                    continue;
                }
                if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
                    <= -lam + k * loglam - crate::types::ln_gamma(k + 1.0)
                {
                    return k as u64;
                }
            }
        }
        let enlam = (-lam).exp();
        let mut x = 0u64;
        let mut prod = 1.0;
        loop {
            prod *= self.next_double();
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }
}
