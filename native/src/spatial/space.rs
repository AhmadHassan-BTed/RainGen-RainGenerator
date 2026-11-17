//! Place acoustics of the rain field: early reflections + a diffuse outdoor tail.
//!
//! Rain outdoors is not heard in free field: walls give slapback and flutter, tree trunks scatter
//! a dense train of weak reflections into a ~1-2 s tail, a lake or a meadow is nearly dry with a
//! distant tree-line echo. This bus processor runs on the exterior rain stems (rain_near,
//! rain_far, aftermath) BEFORE the shelter transfer, so indoors the place is heard through the
//! window / roof like the direct rain. Everything it adds goes to the rain_far stem (the
//! diffuse field), which keeps rain_near the dry, localised layer.
//!
//! * Early reflections: a 16-channel SH delay line. A tap is either an image source of a plane
//!   through the listener (wall / facade: the SH field mirrored by a per-channel sign mask, so
//!   every source's reflection comes from its own mirrored direction) or a scatterer (W
//!   re-encoded at a fixed direction, e.g. a tree trunk or a far tree line). Each tap has a
//!   one-pole low-pass (surface absorption + air absorption over the extra path) and an
//!   optional one-pole high-pass (scatterers only act for ka > 1).
//!   A listener-centred image has one delay for the whole field, but the rain sources spread over
//!   metres, so their reflections arrive over a few ms around it: a single coherent delay printed
//!   a comb (a "pipe" colour) on the field. Each tap is therefore a small cluster: `SUB` sub-taps
//!   per SH channel (image taps) or on W (scatterers) at randomised offsets within +-`jitter_ms`,
//!   different per channel, so the reflected field is diffuse and decorrelated between the ears.
//!   Taps sit at least `MIN_ER_MS` after the direct sound and after each other (no flutter faster
//!   than ~200 Hz); only the alley keeps a gentle wall-to-wall flutter.
//! * Diffuse tail: a 16-line Hadamard FDN fed omni (W), one line per ACN channel with the SN3D
//!   diffuse-field weights sqrt(16 / (2n + 1)), per-line one-pole damping solved for the mid
//!   RT60 and the 8 kHz HF ratio. Its level is calibrated at construction from its own impulse
//!   response, so `tail_db` is the steady-state reverberant / direct energy ratio.
//!
//! Real-time safe: all buffers are allocated in `new`; idle stems cost one energy check.

use super::sh::{maxre_weights, per_acn, sh_real};
use super::stems::StemBuses;
use crate::engine::{BlockCtx, BuildCfg, BusProcessor};
use crate::types::{Scene, Stem, N_AMBI};

const L: usize = 16;
const F_MID_HZ: f64 = 700.0;
const F_HI_HZ: f64 = 8000.0;
/// Share of rain_far fed to the early reflections (far sources: the reflection path is barely
/// longer than the direct one, so their reflections add little that is not already diffuse).
const FAR_FEED_ER: f32 = 0.35;
/// Minimum delay of a reflection and minimum spacing between successive taps (ms).
const MIN_ER_MS: f32 = 5.0;
/// Sub-taps per tap (per SH channel for images, on W for scatterers).
const SUB: usize = 3;
/// Spread of a tap's sub-taps: +-(fraction of its delay), clamped (ms).
const JITTER_FRAC: f32 = 0.2;
const JITTER_MIN_MS: f32 = 1.5;
const JITTER_MAX_MS: f32 = 5.0;

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// image of a plane through the listener: bit 0 mirrors x (front/back wall), bit 1 y
    /// (left/right facade), bit 2 z; 0 = same direction (second-order image between two walls)
    Mirror(u8),
    /// scatterer at (azimuth deg, elevation deg), W re-encoded there
    Point(f32, f32),
}

#[derive(Clone, Copy, Debug)]
struct TapSpec {
    ms: f32,
    gain: f32,
    lp_hz: f32,
    hp_hz: f32,
    kind: Kind,
}

/// High-pass of the image taps. The images are listener-centred, one delay for every source;
/// real sources spread over metres, so below ~200 Hz their reflections add with random phase
/// and no comb survives. A single fixed delay would print one on the whole field instead
/// (measured: +14 dB at 160 Hz in the alley before this filter).
const IMAGE_HP_HZ: f32 = 180.0;

const fn m(ms: f32, gain: f32, lp_hz: f32, mask: u8) -> TapSpec {
    TapSpec { ms, gain, lp_hz, hp_hz: IMAGE_HP_HZ, kind: Kind::Mirror(mask) }
}

const fn p(ms: f32, gain: f32, lp_hz: f32, hp_hz: f32, az: f32, el: f32) -> TapSpec {
    TapSpec { ms, gain, lp_hz, hp_hz, kind: Kind::Point(az, el) }
}

#[derive(Clone, Debug)]
struct Profile {
    taps: &'static [TapSpec],
    rt60: f32,
    /// steady-state reverberant / input energy (dB)
    tail_db: f32,
    hf_ratio: f32,
    size_m: f32,
}

// Forest. Trunks (radius ~0.1-0.2 m) scatter for ka > 1, i.e. above ~300 Hz, so the taps are
// high-passed at 250 Hz; reflections arrive from all around within ~50 ms (trunks 1-8 m away).
// Measured forest reverberation: RT60 ~1-2 s at 1-4 kHz (Padgham 2004; Swearingen & White
// 2007), low in level (open canopy losses): -13 dB tail, HF ratio 0.55 (foliage + air).
const FOREST: &[TapSpec] = &[
    p(7.0, 0.11, 5500.0, 250.0, 35.0, 0.0),
    p(11.0, 0.10, 5500.0, 250.0, -80.0, 0.0),
    p(16.0, 0.09, 5000.0, 250.0, 150.0, 4.0),
    p(21.0, 0.08, 5000.0, 250.0, -140.0, 0.0),
    p(27.0, 0.07, 4500.0, 250.0, 95.0, -3.0),
    p(34.0, 0.06, 4500.0, 250.0, -30.0, 2.0),
    p(42.0, 0.05, 4000.0, 250.0, -110.0, 0.0),
    p(51.0, 0.045, 4000.0, 250.0, 60.0, 5.0),
];
// City street, facades ~16 m apart: lateral images every W/c ~ 47 ms, the near-side facade
// (same direction) and the far-side facade (mirrored in y) at each order, |R| ~ 0.9 for brick /
// glass with 1/r spreading from ~3 m (0.2, 0.1, 0.06); the street end / crossing facade as an
// x image. Street-canyon reverberation RT60 ~1.5 s (Kang 2000; Picaut et al. 2005).
const CITY: &[TapSpec] = &[
    m(44.6, 0.20, 6500.0, 0b010),
    m(51.3, 0.16, 6500.0, 0b000),
    m(73.0, 0.09, 5000.0, 0b001),
    m(91.7, 0.10, 5500.0, 0b000),
    m(101.9, 0.09, 5500.0, 0b010),
    m(139.4, 0.06, 4500.0, 0b010),
    m(148.3, 0.05, 4500.0, 0b000),
];
// Narrow alley, walls ~4.5 m apart: a gentle flutter between the parallel walls every 2W/c ~ 26 ms
// (images every W/c ~ 13 ms), hard surfaces -> little HF loss; dead end as an x image. Gains
// below the image-source values: rough brick and doorways scatter part of each order.
const ALLEY: &[TapSpec] = &[
    m(11.3, 0.20, 8000.0, 0b010),
    m(17.6, 0.15, 8000.0, 0b000),
    m(23.7, 0.13, 7000.0, 0b000),
    m(29.2, 0.11, 7000.0, 0b010),
    m(37.9, 0.08, 6000.0, 0b010),
    m(43.4, 0.07, 6000.0, 0b000),
    m(49.1, 0.06, 5000.0, 0b001),
    m(55.8, 0.05, 5000.0, 0b000),
];
// Garden / suburb: the house facade ~4 m behind (x image, 22 ms), a neighbour ~10 m to the side,
// hedge / trees scattering; short, quiet tail.
const GARDEN: &[TapSpec] = &[
    m(22.0, 0.18, 6000.0, 0b001),
    p(35.0, 0.07, 4500.0, 250.0, 120.0, 0.0),
    m(62.0, 0.10, 5000.0, 0b010),
    p(80.0, 0.05, 3500.0, 250.0, -60.0, 0.0),
];
// Lake / shore: water is acoustically hard, so the tree line across the water (~65 m) returns a
// clear late echo, low-passed by ~130 m of air and leaf scattering; shore trees behind.
const LAKE: &[TapSpec] = &[
    p(40.0, 0.06, 4000.0, 250.0, 180.0, 0.0),
    p(380.0, 0.08, 2500.0, 150.0, 0.0, 1.0),
    p(520.0, 0.04, 2000.0, 150.0, 70.0, 1.0),
];
// Open field / meadow / beach: essentially dry; one distant tree line or hillside echo.
const OPEN: &[TapSpec] = &[
    p(230.0, 0.05, 2500.0, 150.0, 120.0, 0.0),
    p(410.0, 0.03, 1800.0, 150.0, -60.0, 0.0),
];
// Mountain valley / canyon: discrete rock-face echoes.
const VALLEY: &[TapSpec] = &[
    p(180.0, 0.10, 3500.0, 100.0, 60.0, 5.0),
    p(330.0, 0.08, 3000.0, 100.0, -120.0, 5.0),
    p(560.0, 0.05, 2200.0, 100.0, 10.0, 8.0),
];

fn has(s: &str, keys: &[&str]) -> bool {
    keys.iter().any(|k| s.contains(k))
}

fn profile_for(scene: &Scene) -> Profile {
    let pr = scene.place.preset.to_ascii_lowercase();
    let ground = scene.place.ground.to_ascii_lowercase();
    if has(&pr, &["alley", "lane", "courtyard", "passage"]) {
        Profile { taps: ALLEY, rt60: 1.1, tail_db: -10.0, hf_ratio: 0.6, size_m: 7.0 }
    } else if has(&pr, &["city", "street", "urban", "town", "plaza", "square", "downtown", "market", "station"]) {
        Profile { taps: CITY, rt60: 1.5, tail_db: -12.0, hf_ratio: 0.6, size_m: 16.0 }
    } else if has(&pr, &["forest", "wood", "jungle", "grove", "taiga", "cabin", "camp", "trail"]) {
        Profile { taps: FOREST, rt60: 1.4, tail_db: -13.0, hf_ratio: 0.55, size_m: 14.0 }
    } else if has(&pr, &["lake", "pond", "river", "sea", "ocean", "marsh", "shore", "harbo", "bay", "swamp", "dock", "pier"])
        || ground.contains("water")
    {
        Profile { taps: LAKE, rt60: 1.0, tail_db: -21.0, hf_ratio: 0.45, size_m: 30.0 }
    } else if has(&pr, &["mountain", "valley", "canyon", "cliff", "gorge", "fjord"]) {
        Profile { taps: VALLEY, rt60: 1.6, tail_db: -18.0, hf_ratio: 0.45, size_m: 30.0 }
    } else if has(&pr, &["field", "meadow", "prairie", "desert", "beach", "coast", "plain", "tundra", "savanna", "steppe", "heath", "moor", "dune"]) {
        Profile { taps: OPEN, rt60: 0.6, tail_db: -24.0, hf_ratio: 0.4, size_m: 30.0 }
    } else {
        Profile { taps: GARDEN, rt60: 0.9, tail_db: -18.0, hf_ratio: 0.5, size_m: 12.0 }
    }
}

struct Tap {
    /// sub-tap delays (samples): per SH channel for images; row 0 for scatterers
    d: [[usize; SUB]; N_AMBI],
    lp_a: f32,
    hp_a: f32,
    /// per-channel output gain (sign mask x gain, or SH x maxRE x gain)
    out: [f32; N_AMBI],
    mirror: bool,
    lp: [f32; N_AMBI],
    hp: [f32; N_AMBI],
}

/// One-pole smoothing coefficient for cutoff `hz` (0 = bypass).
fn one_pole(hz: f32, sr: f32) -> f32 {
    if hz <= 0.0 {
        return 1.0;
    }
    1.0 - (-2.0 * std::f32::consts::PI * hz.min(0.45 * sr) / sr).exp()
}

/// Per-ACN sign of the SH field mirrored across the planes in `mask` (measured on sh_real, so it
/// follows whatever convention the encoder uses).
fn mirror_signs(mask: u8) -> [f32; N_AMBI] {
    let dirs = [(0.31f64, 0.17f64), (1.9, -0.4), (-2.4, 0.6), (-0.8, -0.9), (2.9, 0.3)];
    let mut acc = [0f64; N_AMBI];
    for &(az, el) in &dirs {
        let (x, y, z) = (el.cos() * az.cos(), el.cos() * az.sin(), el.sin());
        let x2 = if mask & 1 != 0 { -x } else { x };
        let y2 = if mask & 2 != 0 { -y } else { y };
        let z2 = if mask & 4 != 0 { -z } else { z };
        let a = sh_real(az, el);
        let b = sh_real(y2.atan2(x2), z2.clamp(-1.0, 1.0).asin());
        for k in 0..N_AMBI {
            acc[k] += a[k] * b[k];
        }
    }
    let mut s = [1f32; N_AMBI];
    for k in 0..N_AMBI {
        s[k] = if acc[k] < 0.0 { -1.0 } else { 1.0 };
    }
    s
}

#[inline]
fn fwht16(v: &mut [f32; L]) {
    let mut h = 1;
    while h < L {
        let mut i = 0;
        while i < L {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    for x in v.iter_mut() {
        *x *= 0.25;
    }
}

type CalKey = (u32, u32, u32, u32);
static TAIL_CAL: std::sync::Mutex<Vec<(CalKey, f32)>> = std::sync::Mutex::new(Vec::new());

/// 16-line Hadamard FDN, omni in, SH (diffuse) out.
struct Tail {
    lines: Vec<Vec<f32>>,
    delays: [usize; L],
    ptr: [usize; L],
    b: [f32; L],
    g: [f32; L],
    z: [f32; L],
    signs: [f32; L],
    out_g: [f32; L],
    wet: f32,
}

impl Tail {
    fn new(rt60: f32, hf_ratio: f32, size_m: f32, sr: u32) -> Tail {
        let srf = sr as f64;
        let d_lo = size_m as f64 / 343.0 * srf;
        let d_hi = 2.7 * d_lo;
        let mut delays = [0usize; L];
        let mut used: Vec<usize> = Vec::new();
        for (i, d) in delays.iter_mut().enumerate() {
            let mut k = (d_lo * (d_hi / d_lo).powf(i as f64 / (L - 1) as f64)).round().max(17.0) as usize;
            loop {
                let prime = (2..).take_while(|q: &usize| q * q <= k).all(|q| k % q != 0);
                if prime && !used.contains(&k) {
                    break;
                }
                k += 1;
            }
            used.push(k);
            *d = k;
        }
        // reorder so the order-0..3 channels each get lines from across the length range
        let perm = [0usize, 5, 10, 15, 3, 8, 13, 2, 7, 12, 1, 6, 11, 4, 9, 14];
        let mut dd = [0usize; L];
        for i in 0..L {
            dd[i] = delays[perm[i]];
        }
        let delays = dd;
        let signs = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0];
        let mut out_g = [0f32; L];
        for (c, o) in out_g.iter_mut().enumerate() {
            let n = (c as f32).sqrt().floor();
            *o = (L as f32 / (2.0 * n + 1.0)).sqrt();
        }
        let rt = rt60.max(0.05) as f64;
        let hf = (hf_ratio as f64).clamp(0.05, 1.0);
        let w_m = 2.0 * std::f64::consts::PI * F_MID_HZ / srf;
        let w_h = 2.0 * std::f64::consts::PI * F_HI_HZ.min(0.45 * srf) / srf;
        let d_db = |b: f64, w: f64| 20.0 * ((1.0 - b) / (1.0 - 2.0 * b * w.cos() + b * b).sqrt()).log10();
        let mut bb = [0f32; L];
        let mut gg = [0f32; L];
        for i in 0..L {
            let mm = delays[i] as f64;
            let loss_mid = -60.0 * mm / (srf * rt);
            let want = loss_mid * (1.0 / hf - 1.0);
            let (mut lo, mut hi) = (0.0f64, 0.995f64);
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                if d_db(mid, w_h) - d_db(mid, w_m) < want {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let b = 0.5 * (lo + hi);
            bb[i] = b as f32;
            gg[i] = 10f64.powf((loss_mid - d_db(b, w_m)) / 20.0).min(0.999) as f32;
        }
        let mut t = Tail {
            lines: delays.iter().map(|&d| vec![0.0; d]).collect(),
            delays,
            ptr: [0; L],
            b: bb,
            g: gg,
            z: [0.0; L],
            signs,
            out_g,
            wet: 1.0,
        };
        // calibrate: W energy of the impulse response (unit wet), cached per design
        let key = (rt60.to_bits(), hf_ratio.to_bits(), size_m.to_bits(), sr);
        if let Ok(c) = TAIL_CAL.lock() {
            if let Some(&(_, w)) = c.iter().find(|(k, _)| *k == key) {
                t.wet = w;
                return t;
            }
        }
        let len = ((2.0 * rt60.max(0.05) + 0.2) * sr as f32) as usize;
        let mut ew = 0f64;
        let mut w = [0f32; 1];
        let mut sink = [0f32; N_AMBI];
        for i in 0..len {
            w[0] = if i == 0 { 1.0 } else { 0.0 };
            sink.fill(0.0);
            t.tick(w[0], &mut sink);
            ew += (sink[0] as f64) * (sink[0] as f64);
        }
        t.reset();
        t.wet = (1.0 / ew.max(1e-12)).sqrt() as f32;
        if let Ok(mut c) = TAIL_CAL.lock() {
            c.push((key, t.wet));
        }
        t
    }

    fn reset(&mut self) {
        for l in &mut self.lines {
            l.iter_mut().for_each(|v| *v = 0.0);
        }
        self.ptr = [0; L];
        self.z = [0.0; L];
    }

    /// One sample: omni input x, adds the (unit-wet) SH output to `out`.
    #[inline]
    fn tick(&mut self, x: f32, out: &mut [f32; N_AMBI]) {
        let mut y = [0f32; L];
        for i in 0..L {
            let v = self.lines[i][self.ptr[i]];
            self.z[i] = (1.0 - self.b[i]) * v + self.b[i] * self.z[i];
            y[i] = self.z[i] * self.g[i];
        }
        for c in 0..N_AMBI {
            out[c] += y[c] * self.out_g[c];
        }
        let mut fb = [0f32; L];
        for i in 0..L {
            fb[i] = self.signs[i] * y[i];
        }
        fwht16(&mut fb);
        let xi = 0.25 * x;
        for i in 0..L {
            let p = self.ptr[i];
            self.lines[i][p] = xi + fb[i];
            self.ptr[i] = if p + 1 == self.delays[i] { 0 } else { p + 1 };
        }
    }
}

pub struct SpaceProcessor {
    sr: f32,
    n: usize,
    taps: Vec<Tap>,
    tail: Tail,
    tail_gain: f32,
    /// 16-channel ring of the early-reflection input, channel-major (N_AMBI x len)
    ring: Vec<f32>,
    len: usize,
    wpos: usize,
    max_d: usize,
    idle: usize,
    tail_idle: usize,
    tail_len: usize,
    er_in: Vec<f32>,
    tail_in: Vec<f32>,
    wet: Vec<f32>,
}

impl SpaceProcessor {
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> SpaceProcessor {
        let sr = cfg.sr as f32;
        let n = cfg.block;
        let prof = profile_for(scene);
        let w = per_acn(&maxre_weights());
        let mut taps = Vec::new();
        let mut max_d = 1;
        let mut rng = cfg.rng.get("place.space.er", 0);
        let mut specs: Vec<TapSpec> = prof.taps.to_vec();
        specs.sort_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap_or(std::cmp::Ordering::Equal));
        let mut prev = 0.0f32;
        for t in specs.iter_mut() {
            t.ms = t.ms.max(prev + MIN_ER_MS);
            prev = t.ms;
        }
        let sub_g = 1.0 / (SUB as f32).sqrt();
        for t in &specs {
            let jit = (JITTER_FRAC * t.ms).clamp(JITTER_MIN_MS, JITTER_MAX_MS).min(t.ms - 1.0);
            let mut d = [[1usize; SUB]; N_AMBI];
            for row in d.iter_mut() {
                for (k, v) in row.iter_mut().enumerate() {
                    // stratified within +-jit so the sub-taps never pile up on one delay
                    let lo = -jit + 2.0 * jit * k as f32 / SUB as f32;
                    let ms = t.ms + lo + 2.0 * jit / SUB as f32 * rng.uniform_f32();
                    *v = ((ms * 1e-3 * sr).round() as usize).max(1);
                    max_d = max_d.max(*v);
                }
            }
            let (out, mirror) = match t.kind {
                Kind::Mirror(mask) => {
                    let s = mirror_signs(mask);
                    let mut o = [0f32; N_AMBI];
                    for k in 0..N_AMBI {
                        o[k] = s[k] * t.gain * sub_g;
                    }
                    (o, true)
                }
                Kind::Point(az, el) => {
                    let y = sh_real((az as f64).to_radians(), (el as f64).to_radians());
                    let mut o = [0f32; N_AMBI];
                    for k in 0..N_AMBI {
                        o[k] = (y[k] * w[k]) as f32 * t.gain * sub_g;
                    }
                    (o, false)
                }
            };
            taps.push(Tap {
                d,
                lp_a: one_pole(t.lp_hz, sr),
                hp_a: if t.hp_hz > 0.0 { one_pole(t.hp_hz, sr) } else { 0.0 },
                out,
                mirror,
                lp: [0.0; N_AMBI],
                hp: [0.0; N_AMBI],
            });
        }
        let len = max_d + n + 1;
        let tail = Tail::new(prof.rt60, prof.hf_ratio, prof.size_m, cfg.sr);
        SpaceProcessor {
            sr,
            n,
            taps,
            tail_gain: 10f32.powf(prof.tail_db / 20.0) * tail.wet,
            tail_len: ((prof.rt60 * 1.3 + 0.1) * sr) as usize,
            tail,
            ring: vec![0.0; N_AMBI * len],
            len,
            wpos: 0,
            max_d,
            idle: usize::MAX / 2,
            tail_idle: usize::MAX / 2,
            er_in: vec![0.0; N_AMBI * n],
            tail_in: vec![0.0; n],
            wet: vec![0.0; N_AMBI * n],
        }
    }

    /// False when the place has nothing to add, or RAINGEN_SPACE=0 (A/B against the dry field).
    pub fn needed(&self) -> bool {
        let off = std::env::var("RAINGEN_SPACE")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"))
            .unwrap_or(false);
        !off && (!self.taps.is_empty() || self.tail_gain > 0.0)
    }
}

impl BusProcessor for SpaceProcessor {
    fn name(&self) -> &str {
        "space"
    }

    fn process(&mut self, ctx: &BlockCtx, buses: &mut StemBuses) {
        let n = ctx.n.min(self.n).min(buses.n);
        if n == 0 {
            return;
        }
        let _ = self.sr;
        // inputs
        let mut e_in = 0f32;
        for c in 0..N_AMBI {
            let near = buses.ch(Stem::RainNear, c);
            let aft = buses.ch(Stem::Aftermath, c);
            let far = buses.ch(Stem::RainFar, c);
            let dst = &mut self.er_in[c * n..(c + 1) * n];
            for i in 0..n {
                let v = near[i] + aft[i] + FAR_FEED_ER * far[i];
                dst[i] = v;
                e_in += v * v;
            }
            if c == 0 {
                for i in 0..n {
                    self.tail_in[i] = near[i] + aft[i] + far[i];
                }
            }
        }
        let live = e_in > 1e-18 && e_in.is_finite();
        if live {
            self.idle = 0;
            self.tail_idle = 0;
        } else {
            self.idle = self.idle.saturating_add(n);
            self.tail_idle = self.tail_idle.saturating_add(n);
        }
        let er_on = self.idle <= self.max_d + n;
        let tail_on = self.tail_idle <= self.tail_len + n;
        if !er_on && !tail_on {
            return;
        }
        self.wet.iter_mut().for_each(|v| *v = 0.0);
        let len = self.len;
        // write the block into the ring (zeros while idle, so the line drains)
        if er_on {
            for c in 0..N_AMBI {
                let src = &self.er_in[c * n..(c + 1) * n];
                let ring = &mut self.ring[c * len..(c + 1) * len];
                let mut w = self.wpos;
                for &v in src {
                    ring[w] = if live { v } else { 0.0 };
                    w += 1;
                    if w == len {
                        w = 0;
                    }
                }
            }
            for t in self.taps.iter_mut() {
                if t.mirror {
                    for c in 0..N_AMBI {
                        let g = t.out[c];
                        if g == 0.0 {
                            continue;
                        }
                        let ring = &self.ring[c * len..(c + 1) * len];
                        let out = &mut self.wet[c * n..(c + 1) * n];
                        let mut r = [0usize; SUB];
                        for k in 0..SUB {
                            r[k] = (self.wpos + len - t.d[c][k]) % len;
                        }
                        let (mut s, mut h) = (t.lp[c], t.hp[c]);
                        let (a, ha) = (t.lp_a, t.hp_a);
                        for o in out.iter_mut() {
                            let mut x = 0.0;
                            for rk in r.iter_mut() {
                                x += ring[*rk];
                                *rk += 1;
                                if *rk == len {
                                    *rk = 0;
                                }
                            }
                            s += a * (x - s);
                            h += ha * (s - h);
                            *o += g * (s - h);
                        }
                        t.lp[c] = if s.abs() < 1e-25 { 0.0 } else { s };
                        t.hp[c] = if h.abs() < 1e-25 { 0.0 } else { h };
                    }
                } else {
                    // scatterer: W -> band-limit -> encode at its direction
                    let ring = &self.ring[..len];
                    let mut r = [0usize; SUB];
                    for k in 0..SUB {
                        r[k] = (self.wpos + len - t.d[0][k]) % len;
                    }
                    let (mut s, mut h) = (t.lp[0], t.hp[0]);
                    let (a, ha) = (t.lp_a, t.hp_a);
                    for i in 0..n {
                        let mut x = 0.0;
                        for rk in r.iter_mut() {
                            x += ring[*rk];
                            *rk += 1;
                            if *rk == len {
                                *rk = 0;
                            }
                        }
                        s += a * (x - s);
                        h += ha * (s - h);
                        let y = s - h;
                        for c in 0..N_AMBI {
                            self.wet[c * n + i] += t.out[c] * y;
                        }
                    }
                    t.lp[0] = if s.abs() < 1e-25 { 0.0 } else { s };
                    t.hp[0] = if h.abs() < 1e-25 { 0.0 } else { h };
                }
            }
            self.wpos = (self.wpos + n) % len;
        }
        if tail_on && self.tail_gain > 0.0 {
            let g = self.tail_gain;
            let mut acc = [0f32; N_AMBI];
            for i in 0..n {
                acc.fill(0.0);
                let x = if live { self.tail_in[i] } else { 0.0 };
                self.tail.tick(x, &mut acc);
                for c in 0..N_AMBI {
                    self.wet[c * n + i] += g * acc[c];
                }
            }
            for z in self.tail.z.iter_mut() {
                if z.abs() < 1e-25 {
                    *z = 0.0;
                }
            }
        }
        let far = buses.bus_mut(Stem::RainFar);
        for (o, v) in far[..N_AMBI * n].iter_mut().zip(self.wet.iter()) {
            *o += *v;
        }
    }

    fn reset(&mut self) {
        self.ring.iter_mut().for_each(|v| *v = 0.0);
        self.wpos = 0;
        self.idle = usize::MAX / 2;
        self.tail_idle = usize::MAX / 2;
        for t in self.taps.iter_mut() {
            t.lp = [0.0; N_AMBI];
            t.hp = [0.0; N_AMBI];
        }
        self.tail.reset();
    }
}
