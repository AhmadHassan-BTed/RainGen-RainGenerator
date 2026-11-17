//! Snow (SnowGenerator / _SnowPatch of raingen/synth/rain.py): near-silent snowfall as soft flake
//! ticks on the nearest surfaces, and blowing-snow saltation hiss in cold wind.
//!
//! Flakes: dry cold crystals tick, wet aggregates near 0 C thud softly (size, mass, fall speed and
//! crumple time from the temperature). Number flux = liquid-equivalent mass flux / flake mass;
//! each landing is an impulse m V spread over the crumple time through the target's modes and
//! radiation band. Saltation: band-limited noise above 6 m/s below -1 C (snowing or snow cover).

use super::air::AirFilter;
use super::events::thinned_offsets;
use super::gutters::{filter_norm, noise_block};
use super::kernels::{butter2_bandpass, db_to_pa, smoothstep, Cascade, KernelBank, SolidVoice};
use super::materials::{interior_transfer, patch_geometry, spreading_gain, stem_for, unit, wind_vector};
use super::rain::AIR_MIN_DIST_M;
use super::BuildCtx;
use crate::engine::{BlockCtx, Voice};
use crate::rng::Rng;
use crate::types::{precip, Emission, Patch, Stem, Weather, RHO_WATER};
use std::sync::Arc;

const MAX_EVENTS: f64 = 300.0;

/// Flake parameters at a temperature: (diameter mm, mass kg, fall speed m/s, crumple time s).
#[derive(Clone, Copy, Debug)]
pub struct Flake {
    pub d_mm: f32,
    pub mass: f32,
    pub v: f32,
    pub tau: f32,
}

pub fn flake_params(temp_c: f32) -> Flake {
    let s = smoothstep(temp_c, -6.0, 0.5);
    Flake { d_mm: 2.0 + 8.0 * s, mass: (0.4 + 9.6 * s.powf(1.5)) * 1e-6, v: 0.7 + 0.8 * s, tau: (0.8 + 4.0 * s) * 1e-3 }
}

#[inline]
fn snowing(w: &Weather) -> bool {
    w.precip_type == precip::SNOW && w.rain_rate_mm_h > 0.0
}

/// Flake ticks on one near surface.
pub struct SnowPatchVoice {
    name: String,
    kb: Arc<KernelBank>,
    n: usize,
    sr: f32,
    em: Emission,
    stem: Stem,
    voice: SolidVoice,
    air: Option<AirFilter>,
    r_eq: f32,
    normal: [f32; 3],
    area: f32,
    exposure: f32,
    xfer: Option<Cascade>,
    xfer_gain: f32,
    key: u64,
    offs: Vec<u32>,
}

impl SnowPatchVoice {
    pub fn new(bc: &mut BuildCtx, patch: &Patch) -> SnowPatchVoice {
        let sr = bc.sr as f32;
        let n = bc.block;
        let kb = bc.kb.clone();
        let mat = bc.mats.get(&patch.material);
        let (r_eq, spread) = patch_geometry(patch);
        let stem = stem_for(patch);
        let (xfer_gain, xc) = interior_transfer(patch, sr);
        let ola_cap = n + (0.02 * sr) as usize + kb.max_kernel_len() + 64;
        SnowPatchVoice {
            name: format!("snow/{}", patch.id),
            em: Emission::point(patch.azimuth(), patch.elevation(), patch.distance(), spread, stem),
            stem,
            voice: SolidVoice::new(&mat.modes, &mat.rad, mat.cal, sr, n, ola_cap),
            air: if r_eq >= AIR_MIN_DIST_M { Some(AirFilter::new(r_eq, sr, n)) } else { None },
            r_eq,
            normal: unit(patch.normal),
            area: patch.area_m2,
            exposure: patch.exposure,
            xfer: if xc.is_empty() { None } else { Some(Cascade::new(&xc)) },
            xfer_gain,
            key: bc.rngf.key(&format!("snow/{}", patch.id)),
            offs: Vec::with_capacity(512),
            kb,
            n,
            sr,
        }
    }
}

impl Voice for SnowPatchVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n.min(self.n).min(out.len());
        if n == 0 {
            return Emission::silent(self.stem);
        }
        let w = ctx.weather;
        let active = snowing(w);
        if !active && !self.voice.active() {
            return Emission::silent(self.stem);
        }
        if active && self.exposure > 0.0 {
            let fl = flake_params(w.temperature_c);
            // liquid-equivalent mass flux / flake mass
            let flux = w.rain_rate_mm_h * 1e-3 / 3600.0 * RHO_WATER / fl.mass;
            let [nx, ny, nz] = self.normal;
            let (ux, uy) = wind_vector(w);
            let wn = nx * ux + ny * uy;
            let fac = (nz - wn / fl.v.max(0.3)).max(0.0);
            let rate = flux * self.area * self.exposure * fac;
            if rate > 0.0 {
                let mut rng = Rng::from_key(self.key, ctx.index);
                let mut offs = std::mem::take(&mut self.offs);
                let scale = thinned_offsets(&mut rng, rate as f64, n, self.sr as f64, MAX_EVENTS, 1.0, 1.0, &mut offs) as f32;
                let kb = self.kb.clone();
                for &off in offs.iter() {
                    let mass = fl.mass * (0.4 * rng.normal()).exp() as f32;
                    let vt = fl.v * (0.15 * rng.normal()).exp() as f32;
                    let un = (vt * nz - wn).max(0.3 * vt);
                    let tau = fl.tau * (0.2 * rng.normal()).exp() as f32;
                    // impulse m V spread over the crumple
                    let force = mass * un / (0.511 * tau) * scale;
                    self.voice.add_force(&kb, off as usize, force, tau as f64, false);
                }
                self.offs = offs;
            }
        }
        if !self.voice.active() {
            return Emission::silent(self.stem);
        }
        let y = &mut out[..n];
        self.voice.render(y);
        let mut g = spreading_gain(self.r_eq);
        if let Some(x) = self.xfer.as_mut() {
            x.process(y);
            g *= self.xfer_gain;
        }
        for v in y.iter_mut() {
            *v *= g;
        }
        if let Some(a) = self.air.as_mut() {
            a.process(y, w.temperature_c, w.humidity_pct);
        }
        self.em
    }

    fn reset(&mut self) {
        self.voice.reset();
        if let Some(x) = self.xfer.as_mut() {
            x.reset();
        }
        if let Some(a) = self.air.as_mut() {
            a.reset();
        }
    }
}

/// Blowing-snow saltation hiss (`SnowGenerator._saltation`).
pub struct SaltationVoice {
    kb: Arc<KernelBank>,
    n: usize,
    snow_ground: bool,
    filt: Cascade,
    norm: f32,
    prev: f32,
    key: u64,
    tmp: Vec<f32>,
}

impl SaltationVoice {
    pub fn new(bc: &mut BuildCtx, snow_ground: bool) -> SaltationVoice {
        let sr = bc.sr as f32;
        let c = butter2_bandpass(2000.0, 10000.0f32.min(0.4 * sr), sr);
        SaltationVoice {
            kb: bc.kb.clone(),
            n: bc.block,
            snow_ground,
            filt: Cascade::new(&c),
            norm: filter_norm(&c),
            prev: 0.0,
            key: bc.rngf.key("snow/saltation"),
            tmp: vec![0.0; bc.block],
        }
    }
}

impl Voice for SaltationVoice {
    fn name(&self) -> &str {
        "snow/saltation"
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n.min(self.n).min(out.len());
        let w = ctx.weather;
        let u = w.wind_listener_mps;
        let mut level = 0.0f32;
        if (snowing(w) || self.snow_ground) && w.temperature_c < -1.0 && u > 6.0 {
            level = db_to_pa(26.0 + 30.0 * (u / 6.0).log10()) * w.gust.clamp(0.3, 2.5).powf(1.5);
        }
        if n == 0 || (level == 0.0 && self.prev == 0.0) {
            if self.prev == 0.0 && !self.filt.quiet() {
                self.filt.reset();
            }
            return Emission::silent(Stem::RainFar);
        }
        let tmp = &mut self.tmp[..n];
        let pos = Rng::from_key(self.key, ctx.index).below(1 << 30);
        noise_block(&self.kb, pos, tmp);
        self.filt.process(tmp);
        let (g0, g1) = (self.prev, level);
        let k = self.norm;
        for (i, (o, t)) in out[..n].iter_mut().zip(tmp.iter()).enumerate() {
            let g = g0 + (g1 - g0) * i as f32 / n as f32;
            *o += *t * k * g;
        }
        self.prev = level;
        Emission::point(w.wind_azimuth, -0.3, 3.0, 0.85, Stem::RainFar)
    }

    fn reset(&mut self) {
        self.filt.reset();
        self.prev = 0.0;
    }
}
