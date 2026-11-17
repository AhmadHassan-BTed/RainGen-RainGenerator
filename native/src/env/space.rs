//! Place acoustics of the ambience stem (bus processor): early reflections from the surfaces of
//! the place and a diffuse scattering tail, so birds, insects, frogs and traffic sit in the same
//! space as the rain instead of in free field. (The rain stems get theirs from
//! crate::spatial::space; thunder has its own terrain echoes; the wind bed is near-field flow
//! noise.)
//!
//! * Early reflections are image sources of the exterior field's pressure (W) in the nearby
//!   surfaces: arrival 2 d / c, reflection factor sqrt(1 - alpha) and a one-pole for absorption
//!   (facades alpha ~ 0.05-0.1, Kuttruff "Room Acoustics" 2009 tab. 2.1 / Kang 2000 street
//!   canyons; trunks scatter, foliage absorbs above ~2 kHz), encoded at the surface's direction.
//! * The tail is the SH-domain FDN of places.rs (Jot & Chaigne 1991): outdoor reverberation is
//!   scattering, measured at RT60 ~ 1-2 s in woods (Embleton 1963; Huisman & Attenborough 1991,
//!   JASA 90) and ~ 1.5-3 s in street canyons (Kang 2000, JASA 107), short and weak over open
//!   ground (ground and terrain scattering only).
//! * Open structures (porch, cave, underpass) add their own ceiling / wall reflections.
//!
//! Runs before the shelter transfer, so enclosures filter the reflected field too.
//! Real-time safe: everything is allocated in `new`; fixed cost per block.

use super::common::*;
use super::places::Room;
use crate::engine::{BlockCtx, BuildCfg, BusProcessor};
use crate::spatial::StemBuses;
use crate::types::{Scene, Shelter, Stem, N_AMBI};
use std::f64::consts::PI;

const C_AIR: f64 = 343.0;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PlaceKind {
    Urban,
    Suburb,
    Forest,
    Jungle,
    Valley,
    Open,
}

pub fn place_kind(preset: &str) -> PlaceKind {
    let p = preset.to_ascii_lowercase();
    let has = |keys: &[&str]| keys.iter().any(|k| p.contains(k));
    if has(&["city", "street", "alley", "downtown", "urban", "plaza", "square", "market", "station", "industrial", "harbor", "harbour"]) {
        PlaceKind::Urban
    } else if has(&["jungle", "rainforest", "tropical"]) {
        PlaceKind::Jungle
    } else if has(&["forest", "wood", "grove", "taiga", "orchard", "bamboo", "cabin", "camp"]) {
        PlaceKind::Forest
    } else if has(&["suburb", "garden", "yard", "village", "town", "park", "farm", "porch", "house"]) {
        PlaceKind::Suburb
    } else if has(&["mountain", "ridge", "valley", "canyon", "fjord", "cliff", "hill", "gorge"]) {
        PlaceKind::Valley
    } else {
        PlaceKind::Open
    }
}

/// (RT60 s, wet, FDN size m, HF ratio) of the outdoor tail. Wet values put the steady tail
/// ~ 10-14 dB under the dry exterior field in woods and streets, ~ 25 dB over open ground.
fn tail_of(kind: PlaceKind) -> (f32, f32, f32, f32) {
    match kind {
        PlaceKind::Urban => (1.6, 0.12, 22.0, 0.45),
        PlaceKind::Suburb => (0.9, 0.09, 16.0, 0.5),
        PlaceKind::Forest => (1.3, 0.16, 14.0, 0.6),
        PlaceKind::Jungle => (1.6, 0.2, 12.0, 0.55),
        PlaceKind::Valley => (1.2, 0.05, 30.0, 0.4),
        PlaceKind::Open => (0.5, 0.035, 10.0, 0.45),
    }
}

struct Tap {
    delay: usize,
    g: f32,
    a: f32, // one-pole low-pass coefficient (absorption)
    z: f32,
    y: [f32; 4], // first-order encoding (W, Y, Z, X): reflections need direction, not 3rd-order focus
}

/// (distance m, reflection gain, low-pass Hz, azimuth rad, elevation rad) of the place's surfaces.
fn surfaces(kind: PlaceKind, shelter: Shelter, rng: &mut crate::rng::Rng, h: f64) -> Vec<(f64, f64, f64, f64, f64)> {
    let mut s: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    let axis = rng.range(-PI, PI);
    match kind {
        PlaceKind::Urban => {
            // street canyon 12-20 m wide: first-order facades, then flutter between them;
            // brick / glass alpha ~ 0.05-0.1 -> |R| ~ 0.95, minus spreading of the image
            let d1 = rng.range(4.0, 7.0);
            let d2 = rng.range(8.0, 13.0);
            s.push((d1, 0.48, 9000.0, axis + PI / 2.0, 0.12));
            s.push((d2, 0.36, 8000.0, axis - PI / 2.0, 0.1));
            s.push((d1 + d2, 0.24, 6500.0, axis + PI / 2.0, 0.08));
            s.push((d1 + d2 + 0.3, 0.22, 6500.0, axis - PI / 2.0, 0.08));
            s.push((2.0 * d1 + d2, 0.14, 5000.0, axis + PI / 2.0, 0.05));
            s.push((d1 + 2.0 * d2, 0.12, 5000.0, axis - PI / 2.0, 0.05));
            // a cross-street building far down the axis
            s.push((rng.range(25.0, 45.0), 0.2, 4500.0, axis + rng.range(-0.3, 0.3), 0.05));
        }
        PlaceKind::Suburb => {
            for _ in 0..3 {
                let d = rng.range(10.0, 30.0);
                s.push((d, 0.28 * 10.0 / d.max(10.0), 6000.0, rng.range(-PI, PI), 0.08));
            }
        }
        PlaceKind::Forest | PlaceKind::Jungle => {
            // trunk scattering: many weak arrivals 6-70 ms from every side, foliage absorbs the top
            let n = if kind == PlaceKind::Jungle { 10 } else { 8 };
            for k in 0..n {
                let t = 0.006 * (12.0f64).powf((k as f64 + rng.uniform()) / n as f64);
                let d = 0.5 * C_AIR * t;
                let g = 0.22 * (-t / 0.05).exp();
                s.push((d, g, 4500.0, rng.range(-PI, PI), rng.range(-0.1, 0.5)));
            }
        }
        PlaceKind::Valley => {
            for _ in 0..2 {
                let d = rng.range(150.0, 500.0);
                s.push((d, 0.12 * 150.0 / d, 2500.0, rng.range(-PI, PI), 0.05));
            }
        }
        PlaceKind::Open => {
            // isolated objects (a hedge, a dune, a rock) only
            s.push((rng.range(15.0, 40.0), 0.06, 4000.0, rng.range(-PI, PI), 0.03));
        }
    }
    match shelter {
        Shelter::Porch => {
            // porch ceiling ~1 m overhead (wood), the house wall behind
            s.push(((2.6 - h).max(0.6), 0.3, 5000.0, 0.0, 1.3));
            s.push((1.5, 0.35, 7000.0, PI, 0.1));
        }
        Shelter::Underpass => {
            // concrete deck ~4.5 m and abutment walls: hard (alpha ~ 0.02)
            s.push(((4.5 - h).max(1.0), 0.55, 8000.0, 0.0, 1.4));
            s.push((rng.range(5.0, 8.0), 0.5, 8000.0, axis + PI / 2.0, 0.2));
            s.push((rng.range(5.0, 8.0) + 6.0, 0.35, 7000.0, axis - PI / 2.0, 0.2));
        }
        Shelter::Cave => {
            for _ in 0..4 {
                s.push((rng.range(2.5, 8.0), 0.45, 6000.0, rng.range(-PI, PI), rng.range(-0.2, 0.9)));
            }
        }
        _ => {}
    }
    s
}

struct Group {
    ring: Vec<f32>,
    mask: usize,
    w: usize,
    taps: Vec<Tap>,
    room: Room,
    wet: f32,
    send: Vec<f32>,
    x: Vec<f32>,
    quiet: usize,
    hold: usize,
}

impl Group {
    fn new(taps: &[(usize, f32, f32, [f32; N_AMBI])], tail: (f32, f32, f32, f32), sr: u32, n: usize) -> Group {
        let max_d = taps.iter().map(|t| t.0).max().unwrap_or(0);
        let size = (max_d + 2 * n + 1).next_power_of_two();
        let (rt, wet, room_m, hf) = tail;
        Group {
            ring: vec![0.0; size],
            mask: size - 1,
            w: 0,
            taps: taps.iter().map(|&(delay, g, a, y)| Tap { delay, g, a, z: 0.0, y: [y[0], y[1], y[2], y[3]] }).collect(),
            room: Room::new(rt, sr, hf, Some(room_m), 0.3),
            wet,
            send: vec![0.0; N_AMBI * n],
            x: vec![0.0; n],
            quiet: usize::MAX / 2,
            hold: max_d + ((1.5 * rt + 0.2) * sr as f32) as usize,
        }
    }

    /// Reflections and tail of (out + extra[..]) added to `out` (each N_AMBI x n).
    fn process(&mut self, extra: &[&[f32]], out: &mut [f32], n: usize) {
        let s = N_AMBI * n;
        if self.send.len() < s || self.x.len() < n || out.len() < s {
            return;
        }
        self.send[..s].copy_from_slice(&out[..s]);
        for b in extra {
            for (o, v) in self.send[..s].iter_mut().zip(&b[..s]) {
                *o += *v;
            }
        }
        let loud = self.send[..n].iter().any(|v| *v != 0.0);
        self.quiet = if loud { 0 } else { self.quiet.saturating_add(n) };
        if self.quiet > self.hold {
            return;
        }
        for i in 0..n {
            self.ring[(self.w + i) & self.mask] = self.send[i];
        }
        for tap in &mut self.taps {
            let x = &mut self.x[..n];
            let (a, mut z) = (tap.a, tap.z);
            for (i, v) in x.iter_mut().enumerate() {
                let r = self.ring[(self.w + i).wrapping_sub(tap.delay) & self.mask];
                z += a * (r - z);
                *v = z;
            }
            tap.z = if z.abs() < 1e-25 { 0.0 } else { z };
            for c in 0..4 {
                let g = tap.y[c] * tap.g;
                if g == 0.0 {
                    continue;
                }
                let o = &mut out[c * n..(c + 1) * n];
                for (ov, xv) in o.iter_mut().zip(x.iter()) {
                    *ov += g * *xv;
                }
            }
        }
        self.w = self.w.wrapping_add(n);
        self.room.process_add(&self.send[..s], n, out, self.wet);
    }

    fn reset(&mut self) {
        self.ring.iter_mut().for_each(|v| *v = 0.0);
        self.w = 0;
        self.quiet = usize::MAX / 2;
        for t in &mut self.taps {
            t.z = 0.0;
        }
        self.room.reset();
    }
}

pub struct PlaceSpace {
    n: usize,
    amb: Group,
}

impl PlaceSpace {
    /// None when switched off (RAINGEN_NATIVE_SPACE=0, read once at scene build).
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> Option<PlaceSpace> {
        if matches!(std::env::var("RAINGEN_NATIVE_SPACE").ok().as_deref().map(str::trim), Some("0" | "off" | "false")) {
            return None;
        }
        let sr = cfg.sr as f32;
        let n = cfg.block.max(1);
        let kind = place_kind(&scene.place.preset);
        let shelter = scene.place.shelter;
        let mut rng = cfg.rng.get("place.space", 0);
        let surf = surfaces(kind, shelter, &mut rng, scene.listener.height_m as f64);
        let taps: Vec<(usize, f32, f32, [f32; N_AMBI])> = surf
            .iter()
            .filter_map(|&(d, g, lp, az, el)| {
                let delay = (2.0 * d / C_AIR * sr as f64).round() as usize;
                if delay < 1 || g <= 0.0 {
                    return None;
                }
                // ISO 9613-1 air absorption over the extra 2 d is folded into the low-pass
                let lp = (lp * (1.0 / (1.0 + 2.0 * d / 400.0))).max(800.0);
                let a = 1.0 - (-2.0 * PI * lp / sr as f64).exp();
                Some((delay, g as f32, a as f32, sh_gains(az as f32, el as f32, 0.25)))
            })
            .collect();
        let mut tail = tail_of(kind);
        match shelter {
            // the shelter's own room (places.rs) carries the long enclosed tail
            Shelter::Cave | Shelter::Underpass => tail.1 *= 0.5,
            _ => {}
        }
        Some(PlaceSpace { n, amb: Group::new(&taps, tail, cfg.sr, n) })
    }
}

impl BusProcessor for PlaceSpace {
    fn name(&self) -> &str {
        "place_space"
    }

    fn process(&mut self, _ctx: &BlockCtx, buses: &mut StemBuses) {
        let n = buses.n;
        if n != self.n {
            return;
        }
        self.amb.process(&[], buses.bus_mut(Stem::Ambience), n);
    }

    fn reset(&mut self) {
        self.amb.reset();
    }
}
