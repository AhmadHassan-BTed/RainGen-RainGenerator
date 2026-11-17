//! Hail, ice pellets (sleet) and graupel on the nearest surfaces (HailGenerator / _HailPatch of
//! raingen/synth/rain.py).
//!
//! NOAA size and fall-speed bands (V = 8.5 D_cm^0.57 for hail; ice pellets 3.5 D^0.4; graupel
//! 1.3 D^0.6), a truncated exponential size spectrum, Hertz contact with an ice-crushing limit
//! through the target's modes and radiation, ice cracks on hard targets, bounces with restitution
//! (v_k = e^k v, flight 2 v_k / g) and water impacts on water / puddles. One voice per patch.

use super::air::AirFilter;
use super::kernels::{db_to_pa, dur_bin, hertz_contact, j_ref, smoothstep, Cascade, KernelBank, SolidVoice, SPLASH_VARIANTS};
use super::materials::{interior_transfer, patch_geometry, puddle_mm, spreading_gain, stem_for, unit, wind_vector, MatData};
use super::rain::{water_event, DirectBus, WaterParams, AIR_MIN_DIST_M};
use super::BuildCtx;
use crate::engine::{BlockCtx, Voice};
use crate::rng::Rng;
use crate::types::{precip, Emission, MaterialKind, Patch, PatchKind, Stem, Weather, G, RHO_WATER};
use std::sync::Arc;

pub const KIND_HAIL: u8 = 0;
pub const KIND_SLEET: u8 = 1;
pub const KIND_GRAUPEL: u8 = 2;

/// Ice species of a precipitation type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IceSpecies {
    pub kind: u8,
    pub size: f32,
    pub rho: f32,
    pub e: f32,
    pub crush: f32,
    pub frac: f32,
    pub rest: f32,
    /// Upper size cut as a multiple of `size` (hail: the reported size is the largest stone).
    pub hi: f32,
}

pub fn ice_species(w: &Weather) -> Option<IceSpecies> {
    match w.precip_type {
        precip::HAIL => Some(IceSpecies {
            kind: KIND_HAIL,
            size: w.hail_size_mm.max(5.0),
            rho: 900.0,
            e: 5.0e9,
            crush: 10.0,
            frac: 0.5,
            rest: 0.5,
            hi: 1.0,
        }),
        precip::SLEET => Some(IceSpecies { kind: KIND_SLEET, size: 3.5, rho: 900.0, e: 9.0e9, crush: 15.0, frac: 0.5, rest: 0.45, hi: 1.6 }),
        precip::GRAUPEL => Some(IceSpecies { kind: KIND_GRAUPEL, size: 3.0, rho: 350.0, e: 3.0e8, crush: 1.5, frac: 1.0, rest: 0.15, hi: 1.6 }),
        _ => None,
    }
}

pub fn ice_kind_name(k: u8) -> &'static str {
    match k {
        KIND_HAIL => "hail",
        KIND_SLEET => "sleet",
        _ => "graupel",
    }
}

/// Truncated exponential on [0.3, hi] x size with slope 2/size.
pub fn ice_size(rng: &mut Rng, size_mm: f32, hi_frac: f32) -> f32 {
    let (lo, hi) = (0.3 * size_mm as f64, hi_frac as f64 * size_mm as f64);
    let lam = 2.0 / size_mm as f64;
    let u = rng.uniform();
    (lo - (1.0 - u * (1.0 - (-lam * (hi - lo)).exp())).ln() / lam) as f32
}

pub fn ice_fall_speed(d_mm: f32, kind: u8) -> f32 {
    let d = d_mm.max(0.1);
    match kind {
        KIND_HAIL => 8.5 * (d / 10.0).powf(0.57),
        KIND_SLEET => 3.5 * d.powf(0.4),
        _ => 1.3 * d.powf(0.6),
    }
}

pub fn ice_mean_volume(size_mm: f32, hi_frac: f32) -> f32 {
    let (lo, hi) = (0.3 * size_mm as f64, hi_frac as f64 * size_mm as f64);
    let mut sp = 0.0;
    let mut sv = 0.0;
    for i in 0..256 {
        let d = lo + (hi - lo) * i as f64 / 255.0;
        let p = (-2.0 / size_mm as f64 * (d - lo)).exp();
        sp += p;
        sv += p * std::f64::consts::PI * (d * 1e-3).powi(3) / 6.0;
    }
    (sv / sp) as f32
}

/// Mean stone diameter / size of the truncated exponential (scale-free).
pub fn ice_mean_diameter_rel(hi_frac: f32) -> f32 {
    let mut sp = 0.0;
    let mut sd = 0.0;
    for i in 0..256 {
        let d = 0.3 + (hi_frac as f64 - 0.3) * i as f64 / 255.0;
        let p = (-2.0 * (d - 0.3)).exp();
        sp += p;
        sd += p * d;
    }
    (sd / sp) as f32
}

/// Stones per m^2 per s from the liquid-equivalent rate (share `frac` falls as ice).
pub fn ice_number_flux(rate_mm_h: f32, sp: &IceSpecies) -> f32 {
    let q = rate_mm_h.max(0.0) * 1e-3 / 3600.0 * sp.frac * RHO_WATER / sp.rho;
    q / ice_mean_volume(sp.size, sp.hi).max(1e-12)
}

const MAX_EVENTS: f64 = 400.0;
const MAX_PENDING: usize = 4096;

#[derive(Clone, Copy, Debug)]
struct Bounce {
    t: u64,
    v: f32,
    d: f32,
    s: f32,
}

pub struct HailPatchVoice {
    name: String,
    kb: Arc<KernelBank>,
    n: usize,
    sr: f32,
    em: Emission,
    stem: Stem,
    mat: Arc<MatData>,
    is_water: bool,
    kind: MaterialKind,
    voice: Option<SolidVoice>,
    direct: DirectBus,
    air: Option<AirFilter>,
    r_eq: f32,
    normal: [f32; 3],
    area: f32,
    exposure: f32,
    e_target: f32,
    click_pa: f32,
    water_quality: f32,
    xfer: Option<Cascade>,
    xfer_gain: f32,
    pend: Vec<Bounce>,
    key: u64,
    /// The first hail voice emits the 'hail' onset marker.
    marker: bool,
    was_active: bool,
    // per-block event scratch
    ev: Vec<(u32, f32, f32, f32)>,
    offs: Vec<u32>,
    buf: Vec<f32>,
}

impl HailPatchVoice {
    pub fn new(bc: &mut BuildCtx, patch: &Patch, marker: bool) -> HailPatchVoice {
        let sr = bc.sr as f32;
        let n = bc.block;
        let kb = bc.kb.clone();
        let mat = bc.mats.get(&patch.material);
        let is_water = patch.kind == PatchKind::Water || mat.is_water();
        let kind = if is_water { MaterialKind::Water } else { mat.kind };
        let (r_eq, spread) = patch_geometry(patch);
        let stem = stem_for(patch);
        let (xfer_gain, xc) = interior_transfer(patch, sr);
        let ola_cap = n + (0.08 * sr) as usize + kb.max_kernel_len() + 64;
        let voice = if is_water { None } else { Some(SolidVoice::new(&mat.modes, &mat.rad, mat.cal, sr, n, ola_cap)) };
        let air = if r_eq >= AIR_MIN_DIST_M { Some(AirFilter::new(r_eq, sr, n)) } else { None };
        HailPatchVoice {
            name: format!("hail/{}", patch.id),
            em: Emission::point(patch.azimuth(), patch.elevation(), patch.distance(), spread, stem),
            stem,
            is_water,
            kind,
            voice,
            direct: DirectBus::new(ola_cap, 1024, sr as f64),
            air,
            r_eq,
            normal: unit(patch.normal),
            area: patch.area_m2,
            exposure: patch.exposure,
            e_target: mat.youngs,
            click_pa: db_to_pa(bc.mats.level_of("puddle")),
            water_quality: bc.water_quality,
            xfer: if xc.is_empty() { None } else { Some(Cascade::new(&xc)) },
            xfer_gain,
            pend: Vec::with_capacity(MAX_PENDING),
            key: bc.rngf.key(&format!("hail/{}", patch.id)),
            marker,
            was_active: false,
            ev: Vec::with_capacity(2 * 512 + MAX_PENDING),
            offs: Vec::with_capacity(512),
            buf: vec![0.0; n],
            mat,
            kb,
            n,
            sr,
        }
    }

    fn depth(&self, w: &Weather) -> f32 {
        if self.mat.name == "water" {
            return 1000.0;
        }
        let d = super::materials::puddle_named(w, &self.mat.name).unwrap_or(12.0);
        (if d == 0.0 { 12.0 } else { d }).max(0.5)
    }

    fn busy(&mut self) -> bool {
        !self.pend.is_empty()
            || self.direct.active()
            || self.voice.as_mut().map(|v| v.active()).unwrap_or(false)
            || self.xfer.as_ref().map(|x| !x.quiet()).unwrap_or(false)
            || self.air.as_ref().map(|a| !a.quiet()).unwrap_or(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn synth(&mut self, ctx: &BlockCtx, rng: &mut Rng, sp: &IceSpecies, primary: usize, t_abs0: u64) {
        let w = ctx.weather;
        let sr = self.sr as f64;
        let kb = self.kb.clone();
        let n_ev = self.ev.len();
        if self.is_water {
            let wp = WaterParams { click_pa: self.click_pa, depth_mm: self.depth(w), wind_mps: 0.0, water_quality: self.water_quality, p_scale: 1.0 };
            for i in 0..n_ev {
                let (off, d, v, sc) = self.ev[i];
                water_event(&kb, rng, &mut self.direct, off as usize, d, v, 4, sc, &wp);
            }
            return;
        }
        let depth = puddle_mm(w, &self.mat);
        let mut primary = primary;
        if depth > 0.5 && matches!(self.kind, MaterialKind::Solid | MaterialKind::Porous) {
            let cover = (depth / 8.0).min(1.0).powf(0.8);
            let wp = WaterParams { click_pa: self.click_pa, depth_mm: depth.max(2.0), wind_mps: 0.0, water_quality: self.water_quality, p_scale: 1.0 };
            let mut keep = 0usize;
            let mut kept_primary = 0usize;
            for i in 0..n_ev {
                let e = self.ev[i];
                if rng.uniform() < cover as f64 {
                    water_event(&kb, rng, &mut self.direct, e.0 as usize, e.1, e.2, 4, e.3, &wp);
                } else {
                    self.ev[keep] = e;
                    keep += 1;
                    if i < primary {
                        kept_primary += 1;
                    }
                }
            }
            self.ev.truncate(keep);
            primary = kept_primary;
        }
        let n_ev = self.ev.len();
        if n_ev == 0 {
            return;
        }
        let voice = match self.voice.as_mut() {
            Some(v) => v,
            None => return,
        };
        for i in 0..n_ev {
            let (off, d, v, sc) = self.ev[i];
            let (f_max, tau) = hertz_contact(d as f64, v as f64, sp.rho as f64, sp.e as f64, self.e_target as f64, sp.crush as f64, sp.rest as f64);
            let f = f_max as f32 * sc * (0.2 * rng.normal()).exp() as f32;
            voice.add_force(&kb, off as usize, f, tau, true);
        }
        // ice crack on hard targets
        if sp.kind == KIND_HAIL && self.mat.hardness >= 0.6 {
            let lvl = db_to_pa(self.mat.level_db);
            for i in 0..n_ev {
                let (off, d, v, sc) = self.ev[i];
                if d > 8.0 && rng.uniform() < 0.35 {
                    let j = sp.rho * std::f32::consts::PI * (d * 1e-3).powi(3) / 6.0 * v * (1.0 + sp.rest);
                    let amp = 0.25 * lvl * (j / j_ref()) * sc;
                    let dur = rng.range(0.0015, 0.003) as f32;
                    let var = rng.below(SPLASH_VARIANTS as u64) as usize;
                    self.direct.ola.add(off as usize, kb.splash(dur_bin(dur), var), amp);
                }
            }
        }
        // bounces of primary impacts on firm targets
        if primary > 0 && self.mat.hardness >= 0.4 {
            let hard = ((self.mat.hardness - 0.3) / 0.6).clamp(0.0, 1.0);
            for i in 0..primary.min(n_ev) {
                let (off, d, v, sc) = self.ev[i];
                let e = sp.rest * hard * (1.0 - 0.3 * smoothstep(d, 10.0, 40.0));
                let mut vk = v;
                let mut t_acc = 0.0f64;
                for _ in 0..3 {
                    vk *= e;
                    t_acc += 2.0 * vk as f64 / G as f64;
                    if vk <= 0.8 {
                        break;
                    }
                    if self.pend.len() < MAX_PENDING {
                        self.pend.push(Bounce { t: t_abs0 + off as u64 + (t_acc * sr) as u64, v: vk, d, s: sc });
                    }
                }
            }
        }
    }
}

impl Voice for HailPatchVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n.min(self.n).min(out.len());
        let w = ctx.weather;
        let sp = ice_species(w);
        let active = sp.is_some() && w.rain_rate_mm_h > 0.0;
        if self.marker && active && !self.was_active {
            if let Some(s) = &sp {
                ctx.marker(ctx.t0, "hail", format!("{{\"type\": \"{}\", \"size_mm\": {}}}", ice_kind_name(s.kind), s.size));
            }
        }
        self.was_active = active;
        if !active && !self.busy() {
            return Emission::silent(self.stem);
        }
        let sr = self.sr as f64;
        let t_abs0 = (ctx.t0 * sr).round() as u64;
        let mut rng = Rng::from_key(self.key, ctx.index);
        self.ev.clear();
        let mut primary = 0usize;
        if let (true, Some(s)) = (active, sp.as_ref()) {
            let flux = ice_number_flux(w.rain_rate_mm_h, s);
            if flux > 0.0 && self.exposure > 0.0 {
                let (uhx, uhy) = wind_vector(w);
                let [nx, ny, nz] = self.normal;
                let vmean = ice_fall_speed(0.6 * s.size, s.kind);
                let fac = (nz - (nx * uhx + ny * uhy) / vmean.max(0.5)).max(0.0);
                let rate = flux * self.area * self.exposure * fac;
                let mut offs = std::mem::take(&mut self.offs);
                let scale = super::events::thinned_offsets(&mut rng, rate as f64, n, sr, MAX_EVENTS, 1.0, 1.0, &mut offs) as f32;
                for &off in offs.iter() {
                    let d = ice_size(&mut rng, s.size, s.hi);
                    let vt = ice_fall_speed(d, s.kind);
                    let un = (vt * nz - (nx * uhx + ny * uhy)).max(0.25 * vt);
                    self.ev.push((off, d, un, scale));
                }
                primary = offs.len();
                self.offs = offs;
            }
        }
        // due bounces
        if !self.pend.is_empty() {
            let end = t_abs0 + n as u64;
            let mut i = 0;
            while i < self.pend.len() {
                let b = self.pend[i];
                if b.t < end {
                    let off = b.t.saturating_sub(t_abs0).min(n as u64 - 1) as u32;
                    if self.ev.len() < self.ev.capacity() {
                        self.ev.push((off, b.d, b.v, b.s));
                    }
                    self.pend.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }
        if !self.ev.is_empty() {
            let s = sp.unwrap_or(IceSpecies { kind: KIND_HAIL, size: 10.0, rho: 900.0, e: 5.0e9, crush: 10.0, frac: 0.5, rest: 0.5, hi: 1.0 });
            self.synth(ctx, &mut rng, &s, primary, t_abs0);
        }
        let y = &mut out[..n];
        let mut live = false;
        if let Some(v) = self.voice.as_mut() {
            if v.active() {
                live |= v.render(y);
            }
        }
        live |= self.direct.render(y);
        if !live && self.xfer.as_ref().map(|x| x.quiet()).unwrap_or(true) && self.air.as_ref().map(|a| a.quiet()).unwrap_or(true) {
            return Emission::silent(self.stem);
        }
        if let Some(x) = self.xfer.as_mut() {
            x.process(y);
        }
        // (as in Python, the transfer gain comes with the transfer filter)
        let g = if self.xfer.is_some() { self.xfer_gain } else { 1.0 } * spreading_gain(self.r_eq);
        for v in y.iter_mut() {
            *v *= g;
        }
        if let Some(a) = self.air.as_mut() {
            a.process(y, w.temperature_c, w.humidity_pct);
        }
        let _ = &self.buf;
        self.em
    }

    fn reset(&mut self) {
        if let Some(v) = self.voice.as_mut() {
            v.reset();
        }
        self.direct.clear();
        self.pend.clear();
        self.was_active = false;
        if let Some(x) = self.xfer.as_mut() {
            x.reset();
        }
        if let Some(a) = self.air.as_mut() {
            a.reset();
        }
    }
}
