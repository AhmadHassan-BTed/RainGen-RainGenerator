//! Places and shelters (port of raingen/synth/places.py): envelope transmission loss, the
//! SH-domain FDN room, and car wipers.
//!
//! * Mass law with leaks (Fahy; Bies & Hansen): TL_0 = 10 log10(1 + (pi f m / rho c)^2), field
//!   incidence ~TL_0 - 5 dB, tau = (1 - s) 10^(-TL/10) + s for an open-area fraction s; double
//!   glazing dip; an opened window adds open area and re-radiates as an aperture ahead. Open
//!   structures (porch, cave, underpass) only shield. Transmitted sound is mostly diffuse: higher
//!   SH orders are tapered. Applied to the exterior stem buses as a graphic-EQ cascade.
//! * Room: feedback delay network (Jot & Chaigne 1991) with an orthogonal (signed Hadamard) matrix,
//!   per-line one-pole absorption solved so the mid-band decay is the shelter's RT60 exactly; line
//!   k drives ACN channel k with gain sqrt(L / (2n + 1)). The reverb of the sum of all stems is
//!   added to the interior stem.
//! * Car wipers (FMVSS 104): slow 40, fast 62 cycles/min, intermittent one wipe per 7.5 s; rubber
//!   on glass chatters (stick-slip) when barely wet and swishes when wet; linkage thunks and blade
//!   flips at each reversal; the motor hums with its load.

use super::common::*;
use crate::dsp::{flush64, Biquad, BiquadCoefs};
use rayon::prelude::*;
use crate::engine::{BlockCtx, BuildCfg, BusProcessor, Voice};
use crate::rng::Rng;
use crate::spatial::StemBuses;
use crate::types::{precip, Emission, Patch, Scene, Shelter, Stem, N_AMBI, N_STEMS, RHO_AIR};

pub const BANDS_HZ: [f32; 9] = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];
const WINDOW_AREA_SHARE: f32 = 0.1;
const WINDOW_AZIMUTH: f32 = 0.0;

struct Envelope {
    mass: f32,
    leak: f32,
    dip_hz: f32,
    dip_db: f32,
    shield_db: f32,
    taper: [f32; 4],
}

fn envelope(s: Shelter) -> Option<Envelope> {
    let e = |mass, leak, dip_hz, dip_db, shield_db, taper| Some(Envelope { mass, leak, dip_hz, dip_db, shield_db, taper });
    match s {
        Shelter::Car => e(8.0, 2.0e-4, 0.0, 0.0, 0.0, [1.0, 0.55, 0.3, 0.15]),
        Shelter::Tent => e(0.15, 0.12, 0.0, 0.0, 0.0, [1.0, 0.9, 0.75, 0.6]),
        Shelter::RoomClosed | Shelter::RoomOpen => e(14.0, 3.0e-4, 250.0, 6.0, 0.0, [1.0, 0.35, 0.15, 0.08]),
        Shelter::Attic => e(6.0, 3.0e-3, 0.0, 0.0, 0.0, [1.0, 0.45, 0.2, 0.1]),
        Shelter::Porch => e(0.0, 0.0, 0.0, 0.0, 1.5, [1.0, 0.95, 0.9, 0.85]),
        Shelter::Cave => e(0.0, 0.0, 0.0, 0.0, 3.0, [1.0, 0.8, 0.6, 0.45]),
        Shelter::Underpass => e(0.0, 0.0, 0.0, 0.0, 3.0, [1.0, 0.85, 0.7, 0.55]),
        _ => None,
    }
}

/// Field-incidence mass-law transmission loss with leakage and an optional mass-air-mass dip.
pub fn mass_law_tl_db(f: f32, mass: f32, leak: f32, dip_hz: f32, dip_db: f32) -> f32 {
    let rho_c = RHO_AIR * 343.0;
    let x = std::f32::consts::PI * f * mass.max(0.0) / rho_c;
    let tl0 = 10.0 * (1.0 + x * x).log10();
    let mut tl = (tl0 - 5.0 * (tl0 / 15.0).clamp(0.0, 1.0)).max(0.0);
    if dip_hz > 0.0 && dip_db > 0.0 {
        let z = (f / dip_hz).log2() / 0.6;
        tl = (tl - dip_db * (-z * z).exp()).max(0.0);
    }
    let s = leak.clamp(0.0, 1.0);
    let tau = (1.0 - s) * 10f32.powf(-tl / 10.0) + s;
    -10.0 * tau.max(1e-12).log10()
}

/// Transmission loss (dB, positive = attenuation) of the shelter envelope at f.
pub fn shelter_tl_db(s: Shelter, window_open: f32, f: f32) -> f32 {
    match envelope(s) {
        None => 0.0,
        Some(env) if env.shield_db > 0.0 => env.shield_db * (f / 1000.0).powf(0.35).clamp(0.3, 2.5),
        Some(env) => {
            let mut leak = env.leak;
            if matches!(s, Shelter::RoomOpen | Shelter::RoomClosed) {
                leak += WINDOW_AREA_SHARE * window_open.clamp(0.0, 1.0);
            }
            mass_law_tl_db(f, env.mass, leak, env.dip_hz, env.dip_db)
        }
    }
}

fn effective_window_open(s: Shelter, wo: f32) -> f32 {
    let wo = wo.clamp(0.0, 1.0);
    if s == Shelter::RoomOpen && wo <= 0.0 {
        0.5
    } else {
        wo
    }
}

/// (RT60 s, wet gain) of the listener's enclosure; (0, 0) when outdoors.
pub fn room_for_shelter(s: Shelter, window_open: f32) -> (f32, f32) {
    let wo = effective_window_open(s, window_open);
    let (mut rt, mut wet) = match s {
        Shelter::Tent => (0.12, 0.06),
        Shelter::Car => (0.08, 0.25),
        Shelter::Porch => (0.35, 0.08),
        // Sabine T = 0.161 V / A: furnished bedroom V ~ 38 m3, A ~ 15 m2 -> 0.4 s;
        // wooden cabin / attic V ~ 70 m3, A ~ 19 m2 -> 0.6 s
        Shelter::RoomClosed => (0.4, 0.22),
        Shelter::RoomOpen => (0.4, 0.20),
        Shelter::Attic => (0.6, 0.3),
        Shelter::Cave => (1.8, 0.35),
        Shelter::Underpass => (1.4, 0.3),
        _ => (0.0, 0.0),
    };
    if matches!(s, Shelter::RoomOpen | Shelter::RoomClosed) && wo > 0.0 {
        rt *= 1.0 - 0.15 * wo;
        wet *= 1.0 - 0.2 * wo;
    }
    (rt, wet)
}

// --------------------------------------------------------------------------
// SH-domain FDN reverb
// --------------------------------------------------------------------------
const L: usize = 16;
const F_MID_HZ: f64 = 700.0;
const F_HI_HZ: f64 = 8000.0;

fn primes_near(values: &[f64]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &v in values {
        let mut k = (v.round() as usize).max(2);
        loop {
            let is_prime = (2..).take_while(|p: &usize| p * p <= k).all(|p| k % p != 0);
            if is_prime && !out.contains(&k) {
                break;
            }
            k += 1;
        }
        out.push(k);
    }
    out
}

#[inline]
fn fwht16(v: &mut [f64; L]) {
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
        *x *= 0.25; // 1 / sqrt(16)
    }
}

/// Spherical-harmonic-domain feedback delay network: `process` returns the reverberant part only.
pub struct Room {
    pub rt60: f32,
    sr: f64,
    hf_ratio: f64,
    delays: [usize; L],
    lines: Vec<Vec<f64>>,
    ptr: [usize; L],
    signs: [f64; L],
    b: [f64; L],
    g: [f64; L],
    z: [f64; L],
    w_in: [f64; N_AMBI],
    out_g: [f64; L],
    max_delay: usize,
    written: [f64; 64],
    wpos: usize,
    keep: usize,
    energy: f64,
    frames: Vec<[f64; L]>, // block transposed to 16-lane frames (input in, reverb out)
}

impl Room {
    pub fn new(rt60_s: f32, sr: u32, hf_ratio: f32, size_m: Option<f32>, directional_input: f32) -> Room {
        let rt60 = rt60_s.max(0.02);
        let size = size_m.unwrap_or_else(|| (0.6 + 7.0 * rt60).clamp(1.0, 30.0)) as f64;
        let srf = sr as f64;
        let d_lo = size / 343.0 * srf;
        let d_hi = 3.0 * d_lo;
        let targets: Vec<f64> = (0..L).map(|i| d_lo * (d_hi / d_lo).powf(i as f64 / (L - 1) as f64)).collect();
        let dl = primes_near(&targets);
        let mut delays = [0usize; L];
        delays.copy_from_slice(&dl);
        let max_delay = *delays.iter().max().unwrap();
        // fixed sign pattern (no runtime randomness)
        let mut signs = [1.0f64; L];
        let mut r = Rng::seeded(0x6F64_6E66);
        for s in signs.iter_mut() {
            if r.uniform() < 0.5 {
                *s = -1.0;
            }
        }
        let mut w_in = [0f64; N_AMBI];
        for (c, w) in w_in.iter_mut().enumerate() {
            *w = if c == 0 { 1.0 } else { directional_input as f64 };
        }
        let mut out_g = [0f64; L];
        for (c, o) in out_g.iter_mut().enumerate().take(N_AMBI) {
            *o = (L as f64 / (2.0 * acn_order(c) as f64 + 1.0)).sqrt();
        }
        let mut room = Room {
            rt60,
            sr: srf,
            hf_ratio: (hf_ratio as f64).clamp(0.05, 1.0),
            delays,
            lines: delays.iter().map(|&d| vec![0.0; d]).collect(),
            ptr: [0; L],
            signs,
            b: [0.0; L],
            g: [0.0; L],
            z: [0.0; L],
            w_in,
            out_g,
            max_delay,
            written: [0.0; 64],
            wpos: 0,
            keep: 1,
            energy: 0.0,
            frames: vec![[0.0; L]; 2048],
        };
        room.set_rt60(rt60);
        room
    }

    /// Re-solve the per-line absorption for a new mid-band RT60 (delays kept).
    pub fn set_rt60(&mut self, rt60: f32) {
        self.rt60 = rt60.max(0.02);
        let rt = self.rt60 as f64;
        let w_m = 2.0 * std::f64::consts::PI * F_MID_HZ / self.sr;
        let w_h = 2.0 * std::f64::consts::PI * F_HI_HZ.min(0.45 * self.sr) / self.sr;
        let d_db = |b: f64, w: f64| 20.0 * ((1.0 - b) / (1.0 - 2.0 * b * w.cos() + b * b).sqrt()).log10();
        for i in 0..L {
            let m = self.delays[i] as f64;
            let loss_mid = -60.0 * m / (self.sr * rt);
            let want = loss_mid * (1.0 / self.hf_ratio - 1.0);
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
            self.b[i] = b;
            self.g[i] = 10f64.powf((loss_mid - d_db(b, w_m)) / 20.0).min(0.999);
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.lines {
            l.iter_mut().for_each(|v| *v = 0.0);
        }
        self.ptr = [0; L];
        self.z = [0.0; L];
        self.written = [0.0; 64];
        self.energy = 0.0;
    }

    /// `out` (N_AMBI x n) += wet * reverb(bus) where bus is N_AMBI x n (channel-major).
    /// The block is transposed to 16-lane frames first so the per-sample network runs on fixed-size
    /// arrays (vectorised), then transposed back.
    pub fn process_add(&mut self, bus: &[f32], n: usize, out: &mut [f32], wet: f32) {
        let in_energy: f64 = bus[..N_AMBI * n].iter().map(|v| (*v as f64) * (*v as f64)).sum();
        if in_energy == 0.0 && self.energy < 1e-14 {
            return;
        }
        if self.frames.len() < n {
            self.frames.resize(n, [0.0; L]); // only if the block outgrows the preallocation
        }
        let wet = wet as f64;
        let mut written = 0.0;
        let fr = &mut self.frames[..n];
        for c in 0..N_AMBI {
            let w = self.w_in[c];
            for (f, v) in fr.iter_mut().zip(&bus[c * n..(c + 1) * n]) {
                f[c] = w * *v as f64;
            }
        }
        let mut og = [0f64; L];
        for c in 0..L {
            og[c] = wet * self.out_g[c];
        }
        let (b, g, signs) = (self.b, self.g, self.signs);
        let mut z = self.z;
        for f in fr.iter_mut() {
            let mut xin = *f;
            fwht16(&mut xin);
            let mut y = [0f64; L];
            for i in 0..L {
                let v = self.lines[i][self.ptr[i]];
                z[i] = (1.0 - b[i]) * v + b[i] * z[i];
                y[i] = z[i] * g[i];
            }
            let mut fb = [0f64; L];
            for i in 0..L {
                f[i] = y[i] * og[i];
                fb[i] = signs[i] * y[i];
            }
            fwht16(&mut fb);
            for i in 0..L {
                let v = xin[i] + fb[i];
                written += v * v;
                let p = self.ptr[i];
                self.lines[i][p] = v;
                self.ptr[i] = if p + 1 == self.delays[i] { 0 } else { p + 1 };
            }
        }
        for zz in z.iter_mut() {
            if zz.abs() < 1e-30 {
                *zz = 0.0;
            }
        }
        self.z = z;
        for c in 0..N_AMBI {
            for (o, f) in out[c * n..(c + 1) * n].iter_mut().zip(fr.iter()) {
                *o += f[c] as f32;
            }
        }
        // what is still circulating: everything written within the longest delay
        self.keep = (self.max_delay.div_ceil(n.max(1)) + 1).min(64);
        self.written[self.wpos % 64] = written;
        self.wpos += 1;
        let k = self.keep.min(self.wpos);
        self.energy = (0..k).map(|j| self.written[(self.wpos - 1 - j) % 64]).sum();
        if self.energy < 1e-14 && in_energy == 0.0 {
            for l in &mut self.lines {
                l.iter_mut().for_each(|v| *v = 0.0);
            }
            self.z = [0.0; L];
        }
    }
}

// --------------------------------------------------------------------------
// shelter transfer + room (bus processor)
// --------------------------------------------------------------------------
const EXTERIOR: [Stem; 5] = [Stem::RainFar, Stem::Wind, Stem::Thunder, Stem::Ambience, Stem::Aftermath];

/// The shelter's graphic-EQ cascade on all 16 SH channels of one stem at once: the samples are
/// transposed to 16-lane frames so every section runs 16 independent recursions in SIMD lanes
/// (same coefficients, per-channel state; TDF-II in f64, denormal-flushed per block).
struct StemEq {
    z1: Vec<[f64; N_AMBI]>,
    z2: Vec<[f64; N_AMBI]>,
    frames: Vec<[f64; N_AMBI]>,
}

impl StemEq {
    fn new(sections: usize, n: usize) -> StemEq {
        StemEq {
            z1: vec![[0.0; N_AMBI]; sections],
            z2: vec![[0.0; N_AMBI]; sections],
            frames: vec![[0.0; N_AMBI]; n],
        }
    }

    fn ensure(&mut self, sections: usize) {
        if self.z1.len() < sections {
            self.z1.resize(sections, [0.0; N_AMBI]);
            self.z2.resize(sections, [0.0; N_AMBI]);
        }
    }

    fn reset(&mut self) {
        self.z1.iter_mut().for_each(|z| *z = [0.0; N_AMBI]);
        self.z2.iter_mut().for_each(|z| *z = [0.0; N_AMBI]);
    }

    /// `bus` is N_AMBI x n channel-major, filtered in place by the `sos` cascade.
    fn process(&mut self, sos: &[BiquadCoefs], bus: &mut [f32], n: usize) {
        let ns = sos.len().min(self.z1.len());
        if ns == 0 || n > self.frames.len() {
            return;
        }
        let fr = &mut self.frames[..n];
        for c in 0..N_AMBI {
            let ch = &bus[c * n..(c + 1) * n];
            for (f, v) in fr.iter_mut().zip(ch) {
                f[c] = *v as f64;
            }
        }
        for (k, co) in sos[..ns].iter().enumerate() {
            let (b0, b1, b2, a1, a2) = (co.b0, co.b1, co.b2, co.a1, co.a2);
            let mut z1 = self.z1[k];
            let mut z2 = self.z2[k];
            for f in fr.iter_mut() {
                for c in 0..N_AMBI {
                    let x = f[c];
                    let y = b0 * x + z1[c];
                    z1[c] = b1 * x - a1 * y + z2[c];
                    z2[c] = b2 * x - a2 * y;
                    f[c] = y;
                }
            }
            for c in 0..N_AMBI {
                z1[c] = flush64(z1[c]);
                z2[c] = flush64(z2[c]);
            }
            self.z1[k] = z1;
            self.z2[k] = z2;
        }
        for c in 0..N_AMBI {
            let ch = &mut bus[c * n..(c + 1) * n];
            for (v, f) in ch.iter_mut().zip(fr.iter()) {
                *v = f[c] as f32;
            }
        }
    }
}

pub struct ShelterProcessor {
    shelter: Shelter,
    sr: f32,
    n: usize,
    active: bool,
    stems: [bool; N_STEMS],
    window_raw: f32,
    window_open: f32,
    sos: Vec<BiquadCoefs>,
    eq: Vec<StemEq>, // one 16-lane cascade state per stem
    live: [bool; N_STEMS],
    channel_gain: [f32; N_AMBI],
    aperture: f32,
    aperture_y: [f32; N_AMBI],
    room: Option<Room>,
    room_wet: f32,
    send: Vec<f32>,
    modes: Vec<(Biquad, f32)>,
    mode_x: Vec<f32>,
}

/// Low room modes of the enclosure (Hz) and their decay time scale: f = c/2 sqrt((p/Lx)^2 +
/// (q/Ly)^2 + (r/Lz)^2) for the first axial / tangential modes of a typical size (Kuttruff, Room
/// Acoustics, ch. 3). Below the Schroeder frequency 2000 sqrt(T/V) (~200 Hz in a bedroom, ~400 Hz
/// in a car) the interior response is these discrete resonances, which is the room's body.
fn room_modes(s: Shelter) -> (&'static [(f32, f32)], f32) {
    match s {
        // bedroom 4.2 x 3.6 x 2.5 m
        Shelter::RoomClosed | Shelter::RoomOpen => {
            (&[(40.8, 0.9), (47.6, 1.0), (62.7, 0.7), (68.6, 0.9), (81.7, 0.6), (95.3, 0.6), (137.2, 0.45)], 1.0)
        }
        // attic / cabin / barn 6 x 4.5 x 2.6 m (sloped roof)
        Shelter::Attic => (&[(38.1, 1.0), (57.2, 0.8), (66.0, 0.9), (76.2, 0.6), (85.8, 0.6), (114.3, 0.45)], 1.0),
        // car cabin 2.0 x 1.4 x 1.1 m
        Shelter::Car => (&[(85.8, 0.7), (122.5, 0.6), (149.6, 0.45), (155.9, 0.5)], 0.6),
        // tent 2.2 x 2.0 x 1.3 m, fabric walls: heavily damped
        Shelter::Tent => (&[(78.0, 0.5), (85.8, 0.5), (131.9, 0.35)], 0.5),
        _ => (&[], 0.0),
    }
}

fn build_modes(s: Shelter, rt60: f32, sr: f32) -> Vec<(Biquad, f32)> {
    let (list, level) = room_modes(s);
    // modal half-power bandwidth B = 2.2 / T60 (T60 of the low band ~1.3 x mid): Q = f T60 / 2.2
    let t_lo = 1.3 * rt60.max(0.05);
    list.iter()
        .map(|&(f, g)| {
            let q = (f * t_lo / 2.2).clamp(2.0, 12.0);
            (Biquad::new(BiquadCoefs::bandpass(sr, f, q)), g * level)
        })
        .collect()
}

impl ShelterProcessor {
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> ShelterProcessor {
        let shelter = scene.place.shelter;
        let active = envelope(shelter).is_some();
        let mut stems = [false; N_STEMS];
        if active {
            for s in EXTERIOR {
                stems[s.index()] = true;
            }
            if shelter.is_enclosed() {
                stems[Stem::RainNear.index()] = true;
            }
        }
        let mut p = ShelterProcessor {
            shelter,
            sr: cfg.sr as f32,
            n: cfg.block,
            active,
            stems,
            window_raw: f32::NAN,
            window_open: -1.0,
            sos: Vec::new(),
            eq: (0..N_STEMS).map(|_| StemEq::new(BANDS_HZ.len(), cfg.block)).collect(),
            live: [false; N_STEMS],
            channel_gain: [1.0; N_AMBI],
            aperture: 0.0,
            aperture_y: crate::spatial::sh_real_f32(WINDOW_AZIMUTH, 0.0),
            room: None,
            room_wet: 0.0,
            send: vec![0.0; N_AMBI * cfg.block],
            modes: Vec::new(),
            mode_x: vec![0.0; cfg.block],
        };
        p.set_window_open(scene.place.window_open, cfg.sr);
        p
    }

    pub fn needed(&self) -> bool {
        self.active || self.room.is_some()
    }

    fn set_window_open(&mut self, raw: f32, sr: u32) {
        self.window_raw = raw;
        let wo = effective_window_open(self.shelter, raw);
        if (wo - self.window_open).abs() < 1e-3 {
            return;
        }
        self.window_open = wo;
        let tl: Vec<f32> = BANDS_HZ.iter().map(|f| shelter_tl_db(self.shelter, wo, *f)).collect();
        let max_tl = tl.iter().cloned().fold(0.0f32, f32::max);
        self.sos = if self.active && max_tl > 0.05 {
            let gains: Vec<f32> = tl.iter().map(|v| -v).collect();
            graphic_eq(&BANDS_HZ, &gains, self.sr, 1.2, 5)
        } else {
            Vec::new()
        };
        for e in &mut self.eq {
            e.ensure(self.sos.len());
        }
        let env = envelope(self.shelter);
        let mut taper = env.as_ref().map(|e| e.taper).unwrap_or([1.0; 4]);
        self.aperture = 0.0;
        if matches!(self.shelter, Shelter::RoomOpen | Shelter::RoomClosed) && wo > 0.0 {
            self.aperture = 0.7 * wo.sqrt();
            for (o, t) in taper.iter_mut().enumerate() {
                let w = if o == 0 { 1.0 } else { 0.0 };
                *t = *t * (1.0 - 0.5 * wo.sqrt()) + 0.5 * wo.sqrt() * w;
            }
        }
        for c in 0..N_AMBI {
            self.channel_gain[c] = taper[acn_order(c)];
        }
        let (rt, wet) = room_for_shelter(self.shelter, wo);
        self.room_wet = wet;
        if rt > 0.0 && wet > 0.0 {
            let hf = if self.shelter == Shelter::Cave { 0.6 } else { 0.45 };
            match &mut self.room {
                Some(r) if (r.rt60 - rt).abs() <= 0.02 => {}
                Some(r) => r.set_rt60(rt),
                None => self.room = Some(Room::new(rt, sr, hf, None, 0.25)),
            }
            if self.modes.is_empty() {
                self.modes = build_modes(self.shelter, rt, sr as f32);
            }
        } else {
            self.room = None;
            self.modes.clear();
        }
    }
}

impl BusProcessor for ShelterProcessor {
    fn name(&self) -> &str {
        "shelter_transfer"
    }

    fn process(&mut self, ctx: &BlockCtx, buses: &mut StemBuses) {
        let n = buses.n;
        if n != self.n || self.send.len() < N_AMBI * n {
            return;
        }
        if ctx.window_open != self.window_raw {
            self.set_window_open(ctx.window_open, ctx.sr);
        }
        if self.active {
            let sos = &self.sos[..];
            let stems = self.stems;
            let gains = self.channel_gain;
            let aperture = self.aperture;
            let ap_y = self.aperture_y;
            buses.data[..N_STEMS * N_AMBI * n]
                .par_chunks_mut(N_AMBI * n)
                .zip(self.eq.par_iter_mut())
                .zip(self.live.par_iter_mut())
                .enumerate()
                .for_each(|(st, ((bus, eq), live))| {
                    if !stems[st] {
                        return;
                    }
                    let silent = bus.iter().all(|v| v.abs() < 1e-20);
                    if silent && !*live {
                        return;
                    }
                    eq.process(sos, bus, n);
                    // aperture uses the filtered W before the taper
                    if aperture > 0.0 {
                        let (w, rest) = bus.split_at_mut(n);
                        for c in 1..N_AMBI {
                            let g = gains[c];
                            let a = aperture * ap_y[c];
                            let ch = &mut rest[(c - 1) * n..c * n];
                            for (v, wv) in ch.iter_mut().zip(w.iter()) {
                                *v = *v * g + a * *wv;
                            }
                        }
                        let g0 = gains[0];
                        w.iter_mut().for_each(|v| *v *= g0);
                    } else {
                        for c in 0..N_AMBI {
                            let g = gains[c];
                            if g != 1.0 {
                                bus[c * n..(c + 1) * n].iter_mut().for_each(|v| *v *= g);
                            }
                        }
                    }
                    let peak = bus.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    *live = !silent || peak > 1e-12;
                });
        }
        if let Some(room) = &mut self.room {
            if self.room_wet > 0.0 {
                let s = N_AMBI * n;
                self.send[..s].iter_mut().for_each(|v| *v = 0.0);
                for st in 0..N_STEMS {
                    let b = buses.bus(Stem::from_index(st));
                    for (o, v) in self.send[..s].iter_mut().zip(b) {
                        *o += *v;
                    }
                }
                let interior = buses.bus_mut(Stem::Interior);
                room.process_add(&self.send[..s], n, interior, self.room_wet);
                // low room modes driven by the pressure (W) of everything in the room
                if !self.modes.is_empty() && self.mode_x.len() >= n {
                    let w = &self.send[..n];
                    for (bq, g) in &mut self.modes {
                        let x = &mut self.mode_x[..n];
                        x.copy_from_slice(w);
                        bq.process_block(x);
                        let g = *g;
                        for (o, v) in interior[..n].iter_mut().zip(x.iter()) {
                            *o += g * *v;
                        }
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        for e in &mut self.eq {
            e.reset();
        }
        self.live = [false; N_STEMS];
        if let Some(r) = &mut self.room {
            r.reset();
        }
        for (b, _) in &mut self.modes {
            b.reset();
        }
    }
}

// --------------------------------------------------------------------------
// car wipers
// --------------------------------------------------------------------------
const WIPER_SLOW: f32 = 40.0;
const WIPER_FAST: f32 = 62.0;
const INTERMITTENT_PERIOD_S: f32 = 7.5;

#[derive(Clone, Copy, PartialEq, Eq)]
enum WiperMode {
    Off,
    Intermittent,
    Slow,
    Fast,
    Auto,
}

fn wiper_mode(s: &str) -> WiperMode {
    match s.trim().to_ascii_lowercase().as_str() {
        "int" | "interval" | "intermittent" | "delay" => WiperMode::Intermittent,
        "slow" | "low" => WiperMode::Slow,
        "fast" | "high" => WiperMode::Fast,
        "auto" => WiperMode::Auto,
        _ => WiperMode::Off,
    }
}

pub struct CarWipersVoice {
    sr: f32,
    mode: WiperMode,
    az: f32,
    el: f32,
    dist: f32,
    f_motor: f32,
    theta: f64,
    running: bool,
    dwell: f32,
    motor_phase: f64,
    swish: Biquad,
    swish_fc: f32,
    swish_g: f32,
    slosh: Biquad,
    slosh_g: f32,
    squeal: Biquad,
    squeal_g: f32,
    slip_phase: f64,
    thunk: StrikeBank,
    clicks: EventRing,
    motor_amp: Ramp,
    blade_amp: Ramp,
    hp: Biquad,
    key: u64,
    theta_buf: Vec<f64>,
    speed: Vec<f32>,
    y: Vec<f32>,
    nz: Vec<f32>,
    imp: Vec<f32>,
    click: Vec<f32>,
    strikes: Vec<(usize, usize)>,
    amps: Vec<f32>,
}

impl CarWipersVoice {
    pub fn new(scene: &Scene, patches: &[Patch], cfg: &BuildCfg) -> CarWipersVoice {
        let sr = cfg.sr as f32;
        let n = cfg.block;
        let pos = patches
            .iter()
            .find(|p| p.id.starts_with("shelter.car.windshield"))
            .map(|p| p.position)
            .unwrap_or([0.75, -0.3, 0.05]);
        let [x, y, z] = pos;
        let mut c = Rng::from_key(cfg.rng.key("wipers.cfg"), 0);
        let f_motor = c.range_f32(85.0, 115.0);
        let f_squeal = c.range_f32(1300.0, 2300.0);
        let thunk_f0 = 110.0 * c.range_f32(0.9, 1.1) as f64;
        let swish_c = bp(2200.0, 0.7, sr);
        let slosh_c = bp(550.0, 1.0, sr);
        let squeal_c = bp(f_squeal, 5.0, sr);
        CarWipersVoice {
            sr,
            mode: wiper_mode(&scene.place.wipers),
            az: y.atan2(x),
            el: z.atan2(x.hypot(y)),
            dist: (x * x + y * y + z * z).sqrt().max(0.3),
            f_motor,
            theta: 0.0,
            running: false,
            dwell: 0.0,
            motor_phase: 0.0,
            swish: Biquad::new(swish_c),
            swish_fc: 2200.0,
            swish_g: biquad_rms_gain(&swish_c),
            slosh: Biquad::new(slosh_c),
            slosh_g: biquad_rms_gain(&slosh_c),
            squeal: Biquad::new(squeal_c),
            squeal_g: biquad_rms_gain(&squeal_c),
            slip_phase: 0.0,
            thunk: StrikeBank::new(&[thunk_f0, 240.0, 430.0], &[0.09, 0.06, 0.04], sr),
            clicks: EventRing::new(1, 4 * n + (0.02 * sr) as usize),
            motor_amp: Ramp::default(),
            blade_amp: Ramp::default(),
            hp: Biquad::new(hp2(40.0, sr)),
            key: cfg.rng.key("wipers"),
            theta_buf: vec![0.0; n],
            speed: vec![0.0; n],
            y: vec![0.0; n],
            nz: vec![0.0; n],
            imp: vec![0.0; n],
            click: vec![0.0; (0.004 * sr) as usize + 1],
            strikes: Vec::with_capacity(16),
            amps: Vec::with_capacity(16 * 3),
        }
    }

    /// (cycles per minute while sweeping, intermittent?)
    fn cpm(&self, rain: f32) -> (f32, bool) {
        match self.mode {
            WiperMode::Auto => {
                if rain < 0.05 {
                    (0.0, false)
                } else if rain < 1.0 {
                    (WIPER_SLOW, true)
                } else if rain < 6.0 {
                    (WIPER_SLOW, false)
                } else {
                    (WIPER_FAST, false)
                }
            }
            WiperMode::Intermittent => (WIPER_SLOW, true),
            WiperMode::Slow => (WIPER_SLOW, false),
            WiperMode::Fast => (WIPER_FAST, false),
            WiperMode::Off => (0.0, false),
        }
    }

    fn make_click(&mut self, rng: &mut Rng) -> usize {
        let sr = self.sr;
        let l = self.click.len().min((0.004 * sr) as usize);
        let mut bq = Biquad::new(bp(rng.range_f32(2500.0, 4000.0), 1.5, sr));
        for i in 0..l {
            let t = i as f32 / sr;
            self.click[i] = bq.process(rng.normal_f32() * (-t / 0.0008).exp());
        }
        normalize_peak(&mut self.click[..l]);
        l
    }
}

impl Voice for CarWipersVoice {
    fn name(&self) -> &str {
        "car_wipers"
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n;
        let sr = self.sr;
        let dt = n as f32 / sr;
        if self.y.len() < n {
            return Emission::silent(Stem::Interior);
        }
        let rain = if w.precip_type != precip::SNOW && w.precip_type != precip::NONE {
            w.rain_rate_mm_h
        } else {
            0.3 * w.rain_rate_mm_h
        };
        let (cpm, inter) = self.cpm(rain);
        let eff_cpm = if cpm > 0.0 { cpm } else { WIPER_SLOW };
        let park_after = inter || cpm <= 0.0;
        if cpm > 0.0 && !self.running {
            self.dwell -= dt;
            if !inter || self.dwell <= 0.0 {
                self.running = true;
            }
        }
        let idle = !self.running
            && self.theta == 0.0
            && self.motor_amp.idle()
            && self.blade_amp.idle()
            && !self.clicks.active()
            && self.thunk.silent();
        if idle {
            return Emission::silent(Stem::Interior);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let start = (ctx.t0 * ctx.sr as f64).round() as u64;
        let rate = eff_cpm as f64 / 60.0 / sr as f64;
        let was_running = self.running;
        let old_theta = self.theta;
        for v in self.theta_buf[..n].iter_mut() {
            *v = self.theta;
        }
        if self.running {
            let mut cut = None;
            for i in 0..n {
                let th = self.theta + rate * (i + 1) as f64;
                self.theta_buf[i] = th;
                if park_after && cut.is_none() && th >= 1.0 {
                    cut = Some(i);
                }
            }
            let last = self.theta_buf[n - 1];
            if park_after && last >= 1.0 {
                let c = cut.unwrap_or(n - 1);
                for v in self.theta_buf[c..n].iter_mut() {
                    *v = 0.0;
                }
                self.running = false;
                self.dwell = if inter { INTERMITTENT_PERIOD_S - 60.0 / eff_cpm } else { 0.0 };
                self.theta = 0.0;
            } else {
                self.theta = last % 1.0;
            }
        }
        let mut moving_count = 0usize;
        for i in 0..n {
            let th = self.theta_buf[i];
            self.speed[i] = if th > 0.0 { (2.0 * std::f64::consts::PI * th).sin().abs() as f32 } else { 0.0 };
            if th > 0.0 {
                moving_count += 1;
            }
        }
        let moving = moving_count as f32 / n as f32;

        // glass wetness: water collected during one wipe period
        let period = if !inter { 60.0 / cpm.max(1.0) } else { INTERMITTENT_PERIOD_S };
        let film_mm = rain / 3600.0 * period;
        let glass = w.wetness_named("glass");
        let wet = smoothstep(film_mm, 0.0002, 0.003).max(0.6 * glass * if inter { 1.0 } else { 0.4 }).min(1.0);

        // reversals: top (theta crosses 0.5) and park (wraps back to 0)
        self.strikes.clear();
        self.amps.clear();
        let mut offs = [0usize; 8];
        let mut no = 0;
        for i in 0..n {
            let th = self.theta_buf[i];
            let prev = if i == 0 { if was_running { old_theta } else { self.theta_buf[0] } } else { self.theta_buf[i - 1] };
            let half = (prev * 2.0).floor() != (th * 2.0).floor() && (th - prev).abs() < 0.5;
            let park = prev > 0.9 && th < 0.1;
            if (half || park) && no < offs.len() {
                offs[no] = i;
                no += 1;
            }
        }
        for &off in &offs[..no] {
            let a = rng.lognormal(0.0, 0.2) as f32;
            if self.strikes.len() < self.strikes.capacity() {
                let row = self.strikes.len();
                self.strikes.push((off, row));
                for g in [1.0f32, 0.5, 0.25].iter().take(self.thunk.m) {
                    self.amps.push(a * g);
                }
                while self.amps.len() < (row + 1) * self.thunk.m {
                    self.amps.push(0.0);
                }
            }
            let l = self.make_click(&mut rng);
            let click = std::mem::take(&mut self.click);
            self.clicks.add(start + off as u64 + (0.004 * sr) as u64, 0, &click[..l], db_spl(52.0) * a);
            self.click = click;
        }
        self.strikes.sort_unstable();
        let y = &mut self.y[..n];
        y.iter_mut().for_each(|v| *v = 0.0);
        self.thunk.process(&self.strikes, &self.amps, y, db_spl(60.0));
        self.clicks.take_add(start, n, y);

        // blade on glass
        let fc = 1600.0 + 1400.0 * wet;
        if (fc - self.swish_fc).abs() / self.swish_fc > 0.05 {
            self.swish_fc = fc;
            let c = bp(fc, 0.7, sr);
            self.swish.set(c);
            self.swish_g = biquad_rms_gain(&c);
        }
        let on = self.running || moving > 0.0;
        let (b0, b1) = self.blade_amp.to(if on { db_spl(50.0) } else { 0.0 });
        let dry = (1.0 - wet).powf(1.5);
        let p_slip = 0.5 + 0.5 * dry;
        let slosh_amt = (rain / 20.0).min(1.0);
        let mut f_slip_sum = 0.0f32;
        self.imp[..n].iter_mut().for_each(|v| *v = 0.0);
        let mut ph = self.slip_phase;
        for i in 0..n {
            let f_slip = 280.0 + 500.0 * self.speed[i];
            f_slip_sum += f_slip;
            let prev = ph;
            ph += f_slip as f64 / sr as f64;
            if ph.floor() > prev.floor() {
                let a = rng.lognormal(0.0, 0.4) as f32;
                if rng.uniform() < p_slip as f64 {
                    self.imp[i] = a;
                }
            }
        }
        self.slip_phase = ph % 1.0;
        let slip_rate = (f_slip_sum / n as f32 * p_slip).max(1.0);
        let chatter_norm = 1.0 / (self.squeal_g * (slip_rate / sr * 1.377).sqrt());
        let inv_sw = 1.0 / self.swish_g;
        let inv_sl = 1.0 / self.slosh_g;
        fill_white(&mut rng, &mut self.nz[..n]);
        for i in 0..n {
            let env = ramp_at(b0, b1, i, n) * self.speed[i];
            let sw = self.swish.process(self.nz[i]) * inv_sw * (0.35 + 0.65 * wet);
            let chat = self.squeal.process(self.imp[i]) * chatter_norm;
            y[i] += env * (sw + 1.6 * dry * chat);
        }
        fill_white(&mut rng, &mut self.nz[..n]);
        for i in 0..n {
            let env = ramp_at(b0, b1, i, n) * self.speed[i];
            y[i] += env * 0.8 * self.slosh.process(self.nz[i]) * inv_sl * slosh_amt;
        }

        // motor and gearbox: pitch sags slightly with the load at the reversals
        let (m0, m1) = self.motor_amp.to(if on { db_spl(42.0) } else { 0.0 });
        let fm0 = self.f_motor * (eff_cpm / 40.0).powf(0.3);
        let mut mph = self.motor_phase;
        for i in 0..n {
            let sp = self.speed[i];
            let f_m = fm0 * (1.0 - 0.04 * (1.0 - sp));
            mph += 2.0 * std::f64::consts::PI * f_m as f64 / sr as f64;
            let p = mph as f32;
            let hum = p.sin() + 0.5 * (2.0 * p + 0.4).sin() + 0.3 * (3.0 * p + 1.2).sin() + 0.12 * (7.0 * p + 2.1).sin();
            y[i] += ramp_at(m0, m1, i, n) * (0.75 + 0.25 * sp) * hum / 1.2;
        }
        self.motor_phase = mph % (2.0 * std::f64::consts::PI);
        self.swish.flush_state();
        self.slosh.flush_state();
        self.squeal.flush_state();
        self.hp.process_block(y);
        out[..n].copy_from_slice(y);
        Emission::point(self.az, self.el, self.dist, 0.45, Stem::Interior)
    }

    fn reset(&mut self) {
        self.theta = 0.0;
        self.running = false;
        self.dwell = 0.0;
        self.motor_phase = 0.0;
        self.slip_phase = 0.0;
        self.swish.reset();
        self.slosh.reset();
        self.squeal.reset();
        self.thunk.reset();
        self.clicks.reset();
        self.motor_amp = Ramp::default();
        self.blade_amp = Ramp::default();
        self.hp.reset();
    }
}
