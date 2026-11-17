//! Rain on surface patches (port of RainPatchGenerator / _RainField of raingen/synth/rain.py).
//!
//! One voice per rain-receiving patch; all voices render in parallel. Per patch and acoustic drop
//! class, impacts arrive as a Cox process (rate = class flux x area x exposure x wind-angle factor,
//! modulated by the patch's own 1/f rate modulator). Level of detail as in Python: a class whose
//! expected overlap (impacts/s x effective response duration) reaches the quality's `dense`
//! threshold, or whose single impacts are > 35 dB below the patch's loudest class, is rendered as
//! matched Gaussian noise with the same power spectrum (Campbell's theorem); the other classes are
//! synthesized per drop, capped per 1024 frames, the excess joining the matched-noise layer.
//!
//! Per drop:
//! - solids: Gordillo force pulses into the material's modal-bank variants (detuned / re-damped
//!   copies), the radiation band and the calibration; prompt-splash micro-impacts on dry hard
//!   surfaces; a wet film cushions the pulse and adds crown-splash hiss; puddle cover turns the
//!   impacts into water impacts; rain onto a snow cover is soft and swallowed;
//! - water: impact click ~ D U^3, probabilistic Minnaert bubble with the van den Doel chirp (small
//!   class rings 13-25 kHz, large drops 2-10 kHz, r^-2.5), extra bubbles for very large drops,
//!   broadband splat; shallow puddles pinch bubbles off and damp them; wind 7.5-10 m/s kills the
//!   small-drop ring; surfactants (water_quality) kill all bubbles.
//! Then the interior transfer (through-ceiling mass law, slabs, glazing), 1/r spreading at the
//! patch's energy-equivalent distance and ISO 9613-1 air absorption (beyond 3 m).
//!
//! Units: Pa at the listener (digital 1.0 = 1 Pa).

use super::air::AirFilter;
use super::dsd::{fall_speed, ClassTable, QUAD};
use super::events::{ramp_position, RateModulator};
use super::kernels::{
    click_bin, db_to_pa, dur_bin, f_ref, force_peak, loguniform, power_law, radius_bin, smoothstep, tau_bin, BubblePool,
    Cascade, KernelBank, ModalLanes, NoiseRow, Ola, D_REF_MM, SPLASH_VARIANTS,
};
use super::materials::{
    interior_transfer, patch_geometry, spreading_gain, stem_for, unit, variant_modes, wetness, wind_vector, MatData,
};
use super::BuildCtx;
use crate::engine::{BlockCtx, Voice};
use crate::rng::Rng;
use crate::types::{precip, Emission, MaterialKind, Patch, PatchKind, Stem, Weather, G, N_CLASSES, RHO_WATER, SIGMA_WATER};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------
pub const BUBBLE_REF_DB: f32 = 52.0;
pub const SPLAT_REF_DB: f32 = 54.0;
pub const FILM_HISS_REL_DB: f32 = -6.0;

/// Pa per (eps * radius[m]) of an in-air bubble (52 dB SPL peak @ 1 m for a = 1 mm, eps = 0.05).
#[inline]
pub fn bubble_pa_per_m() -> f32 {
    db_to_pa(BUBBLE_REF_DB) / (0.05 * 1e-3)
}

#[inline]
pub fn splat_pa() -> f32 {
    db_to_pa(SPLAT_REF_DB)
}

const EPS_LO: [f64; 5] = [0.01, 0.02, 0.005, 0.03, 0.04];
const EPS_HI: [f64; 5] = [0.02, 0.07, 0.02, 0.10, 0.12];
const DELAY_LO_MS: [f64; 5] = [1.0, 0.5, 2.0, 6.0, 10.0];
const DELAY_HI_MS: [f64; 5] = [2.0, 2.5, 8.0, 20.0, 30.0];
pub const VL_EXTRA_BUBBLES: f64 = 0.8;

/// E[eps^2] of log-uniform eps per class.
#[inline]
fn eps2(c: usize) -> f32 {
    let (lo, hi) = (EPS_LO[c], EPS_HI[c]);
    ((hi * hi - lo * lo) / (2.0 * (hi / lo).ln())) as f32
}

const T_EFF_WATER: [f32; 5] = [1e-4, 5e-4, 1e-4, 3e-3, 3e-3];
const SALIENCE: f32 = 3.162_277_7e-4; // -35 dB
const SIG_FORCE: f32 = 0.25;
const SIG_CLICK: f32 = 0.2;
const SPLASH_REPS: f32 = 6.0;
const BLOCK_REF: f32 = 1024.0;
const LIVE_EPS: f32 = 1e-12;
pub const AIR_MIN_DIST_M: f32 = 3.0;

#[inline]
fn splash_a2() -> f32 {
    ((0.03f64 * 0.03 - 0.005 * 0.005) / (2.0 * (0.03f64 / 0.005).ln())) as f32
}

#[inline]
fn splash_tau() -> f64 {
    (5e-5f64 * 1.5e-4).sqrt()
}

/// Share of the drop flux that arrives as liquid drops (the rest is hail / snow).
#[inline]
pub fn rain_gate(p: u8) -> f32 {
    match p {
        precip::NONE | precip::SNOW | precip::GRAUPEL => 0.0,
        precip::SLEET | precip::HAIL => 0.5,
        _ => 1.0,
    }
}

/// Liquid share feeding drips / runoff (mirrors raingen.physics.world).
#[inline]
pub fn liquid_share(p: u8) -> f32 {
    match p {
        precip::NONE | precip::SNOW => 0.0,
        precip::GRAUPEL => 0.3,
        precip::SLEET | precip::HAIL => 0.5,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// level-of-detail settings
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct Lod {
    pub k: usize,
    pub voice_cap: [f32; 5],
    pub gran_cap: [f32; 5],
    pub dense: f32,
    pub var: (usize, usize),
}

pub fn lod_settings(quality: &str) -> Lod {
    match quality.trim().to_ascii_lowercase().as_str() {
        "mobile" => Lod { k: 6, voice_cap: [30., 80., 80., 40., 20.], gran_cap: [6., 16., 16., 10., 6.], dense: 3.0, var: (1, 2) },
        "game" => Lod { k: 12, voice_cap: [60., 160., 160., 80., 40.], gran_cap: [10., 24., 24., 16., 10.], dense: 5.0, var: (1, 2) },
        "cinematic" => Lod {
            k: 48,
            voice_cap: [600., 1500., 1500., 800., 400.],
            gran_cap: [40., 100., 100., 60., 30.],
            dense: 40.0,
            var: (3, 4),
        },
        "reference" => Lod { k: 1 << 20, voice_cap: [1e9; 5], gran_cap: [1e9; 5], dense: f32::INFINITY, var: (2, 2) },
        _ => Lod { k: 24, voice_cap: [150., 400., 400., 200., 100.], gran_cap: [20., 50., 50., 30., 16.], dense: 10.0, var: (2, 2) },
    }
}

// ---------------------------------------------------------------------------
// water impact physics
// ---------------------------------------------------------------------------
pub fn entrainment_probability(d: f32, ci: usize, wind_mps: f32) -> f32 {
    match ci {
        1 => 0.95 * (1.0 - smoothstep(wind_mps, 7.5, 10.0)),
        2 => 0.03,
        c if c >= 3 => 0.05 + 0.60 * smoothstep(d, 2.2, 4.0),
        _ => 0.0,
    }
}

/// Regular-entrainment ring of the small class: ~22 kHz at 0.8 mm falling to ~13 kHz at 1.1 mm.
#[inline]
pub fn small_bubble_freq(d: f32) -> f32 {
    (22e3 - (d - 0.8) / 0.3 * 9e3).clamp(12e3, 24e3)
}

/// Smallest radius of the large-drop bubble band (2-10 kHz, lower for bigger drops).
#[inline]
pub fn large_bubble_r1(d: f32) -> f32 {
    3.28 / (10e3 * (2.2 / d.max(2.2)).powf(0.8))
}

pub fn bubble_radius(rng: &mut Rng, d: f32, ci: usize) -> f64 {
    match ci {
        1 => 3.28 / (small_bubble_freq(d) as f64 * (0.05 * rng.normal()).exp()),
        0 | 2 => loguniform(rng, 3e-4, 1.2e-3),
        _ => {
            let r1 = large_bubble_r1(d) as f64;
            power_law(rng, r1, 4.0 * r1, 2.5)
        }
    }
}

/// Extra bubble damping (0.01 steps) from a shallow puddle bottom.
#[inline]
pub fn damp_bin(depth_mm: f32) -> usize {
    ((0.02 * (1.0 - smoothstep(depth_mm, 3.0, 15.0)) / 0.01).round() as i64).clamp(0, 7) as usize
}

#[inline]
pub fn splat_amp(d: f32, u: f32, ci: usize) -> f32 {
    splat_pa() * (d / 4.0).powi(2) * (u / 8.7).powi(3) * if ci == 4 { 1.0 } else { 0.4 }
}

#[inline]
pub fn splat_dur(d: f32) -> f32 {
    (0.003 * d / 3.0).clamp(0.002, 0.012)
}

/// True where (D, U) lies in the regular-entrainment band 41.3 Fr^0.179 <= We <= 48.3 Fr^0.247
/// (Oguz & Prosperetti 1990), with a 10 % tolerance.
pub fn regular_entrainment(d_mm: f32, u_mps: f32) -> bool {
    let d = d_mm.max(0.05) as f64 * 1e-3;
    let u = u_mps.max(0.01) as f64;
    let we = RHO_WATER as f64 * d * u * u / SIGMA_WATER as f64;
    let fr = u * u / (G as f64 * d);
    let lo = 41.3 * fr.powf(0.179) * 0.9;
    let hi = 48.3 * fr.powf(0.247) * 1.1;
    we >= lo && we <= hi
}

/// Pressure-domain sound of impacts on water and splashes: short kernels plus bubble oscillators.
pub struct DirectBus {
    pub ola: Ola,
    pub bubbles: BubblePool,
}

impl DirectBus {
    pub fn new(ola_cap: usize, bubble_cap: usize, sr: f64) -> DirectBus {
        DirectBus { ola: Ola::new(ola_cap), bubbles: BubblePool::new(bubble_cap, sr) }
    }
    #[inline]
    pub fn active(&self) -> bool {
        self.ola.active() || self.bubbles.active()
    }
    pub fn clear(&mut self) {
        self.ola.clear();
        self.bubbles.clear();
    }
    /// out += this block of the bus.
    pub fn render(&mut self, out: &mut [f32]) -> bool {
        let a = self.ola.pop_into(out);
        let b = self.bubbles.render(out);
        a || b
    }
}

/// Parameters of `water_event` that are fixed per call site.
#[derive(Clone, Copy, Debug)]
pub struct WaterParams {
    pub click_pa: f32,
    pub depth_mm: f32,
    pub wind_mps: f32,
    pub water_quality: f32,
    pub p_scale: f32,
}

/// Impact click + probabilistic bubble (+ splat for big drops) of one impact on a water surface
/// (`_water_events`): amplitudes in Pa @ 1 m, `sc` the event's amplitude scale.
#[allow(clippy::too_many_arguments)]
pub fn water_event(kb: &KernelBank, rng: &mut Rng, bus: &mut DirectBus, off: usize, d: f32, u: f32, ci: usize, sc: f32, wp: &WaterParams) {
    let sr = kb.sr as f64;
    let ci = ci.min(4);
    // impact click ~ D U^3 (U^3 is inside the click kernel)
    let mut amp = wp.click_pa * (d / D_REF_MM) * sc * (0.2 * rng.normal()).exp() as f32;
    if ci == 0 {
        amp *= 0.3;
    }
    bus.ola.add(off, kb.click(click_bin(u)), amp);
    // entrained bubbles
    let p = entrainment_probability(d, ci, wp.wind_mps) * wp.water_quality.clamp(0.0, 1.0) * wp.p_scale;
    if p > 0.0 && rng.uniform() < p as f64 {
        let n_b = 1 + if ci == 4 { rng.poisson(VL_EXTRA_BUBBLES) as usize } else { 0 };
        let fd = smoothstep(wp.depth_mm, 1.5, 8.0);
        if fd > 0.0 {
            let damp = damp_bin(wp.depth_mm) as f64 * 0.01;
            for _ in 0..n_b {
                let radius = bubble_radius(rng, d, ci);
                let xi = (0.1 * (0.25 * rng.normal()).exp()).clamp(0.03, 0.3);
                let eps = loguniform(rng, EPS_LO[ci], EPS_HI[ci]);
                let delay = ((DELAY_LO_MS[ci] + rng.uniform() * (DELAY_HI_MS[ci] - DELAY_LO_MS[ci])) * 1e-3 * sr) as usize;
                let a = bubble_pa_per_m() as f64 * eps * sc as f64 * fd as f64;
                bus.bubbles.spawn(off + delay, a, radius, xi, damp);
            }
        }
    }
    // broadband splat of very large drops
    if ci == 4 || (ci == 3 && d >= 3.0) {
        let amp_s = splat_amp(d, u, ci) * sc * (0.25 * rng.normal()).exp() as f32;
        let v = rng.below(SPLASH_VARIANTS as u64) as usize;
        bus.ola.add(off, kb.splash(dur_bin(splat_dur(d)), v), amp_s);
    }
}

/// Expected unit-eps bubble power spectrum of a class (`_bubble_color`), into `out` (half+1 bins).
fn bubble_color(kb: &KernelBank, cls: usize, dq: &[f32; QUAD], qw: &[f32; QUAD], damp: usize, out: &mut [f32]) {
    out.fill(0.0);
    let qs: f32 = qw.iter().sum::<f32>().max(1e-30);
    let add = |r: f64, w: f32, out: &mut [f32]| {
        if w <= 0.0 {
            return;
        }
        let p = kb.bubble_psd(radius_bin(r), damp);
        for (o, v) in out.iter_mut().zip(p) {
            *o += w * *v;
        }
    };
    if cls == 1 {
        for q in 0..QUAD {
            let r0 = 3.28 / small_bubble_freq(dq[q]) as f64;
            for j in [-1.5f64, -0.5, 0.5, 1.5] {
                add(r0 * (0.05 * j).exp(), qw[q] / qs / 4.0, out);
            }
        }
    } else if cls >= 3 {
        let mut p_d = [0.0f32; QUAD];
        for q in 0..QUAD {
            p_d[q] = (0.05 + 0.60 * smoothstep(dq[q], 2.2, 4.0)) * qw[q] / qs;
        }
        let pds: f32 = p_d.iter().sum::<f32>().max(1e-30);
        for q in 0..QUAD {
            let r1 = large_bubble_r1(dq[q]) as f64;
            let mut rw = [0.0f64; 24];
            let mut tot = 0.0;
            for (i, w) in rw.iter_mut().enumerate() {
                let r = r1 * 4f64.powf((i as f64 + 0.5) / 24.0);
                *w = r.powf(-1.5);
                tot += *w;
            }
            for (i, w) in rw.iter().enumerate() {
                let r = r1 * 4f64.powf((i as f64 + 0.5) / 24.0);
                add(r, (*w / tot) as f32 * p_d[q] / pds, out);
            }
        }
    } else {
        for i in 0..24 {
            add(3e-4 * 4f64.powf((i as f64 + 0.5) / 24.0), 1.0 / 24.0, out);
        }
    }
}

// ---------------------------------------------------------------------------
// the rain patch voice
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq)]
struct DenseKey {
    table_ver: u64,
    glazed: bool,
    dense: [bool; 5],
    spill: [bool; 5],
    wet: f32,
    cover: f32,
    depth: f32,
}

pub struct RainPatchVoice {
    name: String,
    kb: Arc<KernelBank>,
    n: usize,
    sr: f32,
    em: Emission,
    stem: Stem,
    // static description
    mat: Arc<MatData>,
    is_water: bool,
    full: bool,
    nx: f32,
    ny: f32,
    nz: f32,
    base: f32,
    spreading: f32,
    caps: [f32; 5],
    dense_thr: f32,
    lod_dense: f32,
    hiss_pa: f32,
    click_pa: f32,
    snow: bool,
    hiss_ok: bool,
    hiss_scale: f32,
    splash_p: f32,
    puddle_ok: bool,
    porous: bool,
    teff: [f32; 5],
    nystuen: bool,
    water_quality: f32,
    fixed_depth: Option<f32>,
    depth_param: f32,
    // modal path (non-water)
    n_var: usize,
    banks: Vec<ModalLanes>,
    force: Vec<Ola>,
    rad: Cascade,
    cal: f32,
    has_modes: bool,
    // pressure path
    direct: DirectBus,
    // matched noise
    force_rows: Vec<NoiseRow>,
    direct_row: NoiseRow,
    u_force: Vec<f32>,
    u_dir: Vec<f32>,
    psd: Vec<f32>,
    color: Vec<f32>,
    color_key: [Option<(i32, usize)>; 5],
    dense_key: Option<(DenseKey, f32, f32)>,
    lod_prev: Option<([bool; 5], [bool; 5])>,
    // state
    cox: RateModulator,
    table: ClassTable,
    table_ver: u64,
    key_ev: u64,
    xfer: Option<Cascade>,
    xfer_gain: f32,
    air: Option<AirFilter>,
    // scratch
    sum: Vec<f32>,
    fbuf: Vec<f32>,
}

impl RainPatchVoice {
    pub fn new(bc: &mut BuildCtx, patch: &Patch, full: bool) -> RainPatchVoice {
        let sr = bc.sr as f32;
        let n = bc.block;
        let kb = bc.kb.clone();
        let lod = bc.lod;
        let mat = bc.mats.get(&patch.material);
        let kind = if patch.kind == PatchKind::Water || mat.is_water() { MaterialKind::Water } else { mat.kind };
        let is_water = kind == MaterialKind::Water;
        let normal = unit(patch.normal);
        let (r_eq, spread) = patch_geometry(patch);
        let stem = stem_for(patch);
        let single_panel = matches!(patch.kind, PatchKind::Membrane | PatchKind::VehicleBody | PatchKind::Gutter)
            || patch.interior
            || (patch.area_m2 <= 2.5 && !patch.param_bool("ground", false));
        let click_db = if is_water { mat.level_db } else { bc.mats.level_of("puddle") };
        let (xfer_gain, xfer_c) = interior_transfer(patch, sr);
        let (n_var, spread_v) = if is_water {
            (0, 0.0)
        } else if single_panel {
            (1, 0.03)
        } else {
            let mut nv = if kind == MaterialKind::Vegetation { lod.var.1 } else { lod.var.0 };
            if !full {
                nv = nv.max(2);
            }
            (nv, if kind == MaterialKind::Vegetation { 0.25 } else { 0.08 })
        };
        let banks: Vec<ModalLanes> = (0..n_var)
            .map(|v| ModalLanes::new(&variant_modes(&bc.rngf, &mat, if v > 0 { spread_v } else { 0.0 }, v), sr, n))
            .collect();
        let has_modes = banks.first().map(|b| b.n_modes() > 0).unwrap_or(false);
        let ola_cap = n + kb.frame + (0.06 * sr) as usize + kb.max_kernel_len() + 64;
        let bubble_cap = if is_water {
            2048
        } else if !matches!(kind, MaterialKind::Vegetation | MaterialKind::Membrane | MaterialKind::Snow) {
            1024
        } else {
            0
        };
        let step = (2f32).ln() / 6.0;
        let dq = (r_eq.max(0.1).ln() / step).round();
        let d_air = (dq * step).exp();
        let air = if d_air >= AIR_MIN_DIST_M { Some(AirFilter::new(d_air, sr, n)) } else { None };
        let fixed_depth = if mat.name == "water" || patch.param_bool("deep", false) { Some(1000.0) } else { None };
        let nb = kb.half + 1;
        let name = format!("rain/{}", patch.id);
        RainPatchVoice {
            em: Emission::point(patch.azimuth(), patch.elevation(), patch.distance(), spread, stem),
            stem,
            is_water,
            full,
            nx: normal[0],
            ny: normal[1],
            nz: normal[2],
            base: patch.area_m2 * patch.exposure.max(0.0),
            spreading: spreading_gain(r_eq),
            caps: if full { lod.voice_cap } else { lod.gran_cap },
            dense_thr: if full { lod.dense } else if is_water { 0.25 * lod.dense } else { 0.02 * lod.dense },
            lod_dense: lod.dense,
            hiss_pa: db_to_pa(mat.level_db + FILM_HISS_REL_DB),
            click_pa: db_to_pa(click_db),
            snow: kind == MaterialKind::Snow,
            hiss_ok: matches!(kind, MaterialKind::Solid | MaterialKind::Porous | MaterialKind::Membrane),
            hiss_scale: if kind == MaterialKind::Membrane { 0.5 } else { 1.0 },
            splash_p: if mat.splash > 0.0 && mat.hardness >= 0.45 && !matches!(kind, MaterialKind::Snow | MaterialKind::Water) {
                mat.splash
            } else {
                0.0
            },
            puddle_ok: !matches!(kind, MaterialKind::Vegetation | MaterialKind::Membrane | MaterialKind::Snow | MaterialKind::Water),
            porous: kind == MaterialKind::Porous,
            teff: if is_water { T_EFF_WATER } else { [mat.teff; 5] },
            nystuen: bc.nystuen,
            water_quality: bc.water_quality,
            fixed_depth,
            depth_param: patch.param_f32("depth_mm", 12.0),
            n_var,
            force: (0..n_var).map(|_| Ola::new(ola_cap)).collect(),
            banks,
            rad: Cascade::new(&mat.rad),
            cal: mat.cal,
            has_modes,
            direct: DirectBus::new(ola_cap, bubble_cap, sr as f64),
            force_rows: (0..n_var).map(|_| NoiseRow::new(&kb)).collect(),
            direct_row: NoiseRow::new(&kb),
            u_force: vec![0.0; N_CLASSES * nb],
            u_dir: vec![0.0; N_CLASSES * nb],
            psd: vec![0.0; nb],
            color: vec![0.0; N_CLASSES * nb],
            color_key: [None; 5],
            dense_key: None,
            lod_prev: None,
            cox: RateModulator::new(&bc.rngf, &format!("{name}/cox"), 0.22, 0.1, 12.0, sr as f64),
            table: ClassTable::default(),
            table_ver: 0,
            key_ev: bc.rngf.key(&format!("{name}/events")),
            xfer: if xfer_c.is_empty() { None } else { Some(Cascade::new(&xfer_c)) },
            xfer_gain,
            air,
            sum: vec![0.0; n],
            fbuf: vec![0.0; n],
            mat,
            kb,
            n,
            sr,
            name,
        }
    }

    fn water_depth(&self, w: &Weather) -> f32 {
        if let Some(d) = self.fixed_depth {
            return d;
        }
        let d = super::materials::puddle_named(w, &self.mat.name).unwrap_or(self.depth_param);
        d.max(0.5)
    }

    /// (wetness, puddle cover, water depth, glaze) of the patch (`_surface_state`).
    fn surface_state(&self, w: &Weather) -> (f32, f32, f32, bool) {
        let glazed = w.precip_type == precip::FREEZING_RAIN && w.temperature_c <= 0.5;
        let wet_m = (wetness(w, &self.mat).clamp(0.0, 1.0) * 64.0).round() / 64.0;
        let pud_m = (super::materials::puddle_mm(w, &self.mat) * 4.0).round() / 4.0;
        let mut wet = if glazed { 0.0 } else { wet_m };
        let has = pud_m > 0.2 && self.puddle_ok && !glazed;
        let cover = if has { (pud_m / 8.0).min(1.0).powf(0.8) * if self.porous { 0.5 } else { 1.0 } } else { 0.0 };
        let mut depth = if has { pud_m.max(2.0) } else { 1000.0 };
        if self.is_water {
            wet = 0.0;
            depth = (self.water_depth(w) * 4.0).round() / 4.0;
        }
        (wet, cover, depth, glazed)
    }

    fn busy(&mut self) -> bool {
        if self.direct.active() || self.force.iter().any(|o| o.active()) {
            return true;
        }
        for b in self.banks.iter_mut() {
            if !b.silent(1e-11) {
                return true;
            }
        }
        !self.rad.quiet()
            || self.xfer.as_ref().map(|x| !x.quiet()).unwrap_or(false)
            || self.air.as_ref().map(|a| !a.quiet()).unwrap_or(false)
    }

    /// One drop on the solid face (`_solid_events`).
    #[allow(clippy::too_many_arguments)]
    fn solid_event(&mut self, rng: &mut Rng, off: usize, d: f32, u: f32, ss: f32, wp: f32, glazed: bool) {
        let sr = self.sr as f64;
        let f_phys = force_peak(d, u);
        let mut tau = d as f64 * 1e-3 / (u as f64).max(0.05) * (1.0 + 0.6 * wp as f64) * if glazed { 0.7 } else { 1.0 };
        let mut force = f_phys * ss * (SIG_FORCE as f64 * rng.normal()).exp() as f32 * (1.0 - 0.35 * wp);
        if self.snow {
            force *= 0.2;
            tau *= 4.0;
        }
        let nv = self.n_var.max(1);
        let var = ((rng.uniform() * nv as f64) as usize).min(nv - 1);
        // prompt-splash micro-impacts on dry hard surfaces
        let ps = self.splash_p * (1.0 - wp) * (1.0 - wp) * smoothstep(d, 0.6, 1.6) * if wp < 0.95 { 1.0 } else { 0.0 };
        if ps > 0.0 && rng.uniform() < ps as f64 {
            let reps = rng.int_range(2, 10) as usize;
            for _ in 0..reps {
                let delay = (rng.range(0.005f64.ln(), 0.05f64.ln()).exp() * sr) as usize;
                let amp = force * rng.range(0.005f64.ln(), 0.03f64.ln()).exp() as f32;
                let ts = rng.range(5e-5f64.ln(), 1.5e-4f64.ln()).exp();
                if self.n_var > 0 {
                    self.force[var].add(off + delay, self.kb.pulse(tau_bin(ts)), amp);
                }
            }
        }
        if self.n_var > 0 {
            self.force[var].add(off, self.kb.pulse(tau_bin(tau)), force);
        }
        // crown / dome splash hiss on a wet film (no bubbles)
        if wp > 0.05 && self.hiss_ok {
            let amp = self.hiss_pa * wp * (f_phys / f_ref()).powf(0.75) * ss * self.hiss_scale;
            let dur = (0.0025 * (d / 2.0).sqrt()).clamp(0.0015, 0.008);
            let v = rng.below(SPLASH_VARIANTS as u64) as usize;
            self.direct.ola.add(off, self.kb.splash(dur_bin(dur), v), amp);
        }
    }

    /// Matched-noise class spectra per unit impact rate (`_dense`), into u_force / u_dir.
    #[allow(clippy::too_many_arguments)]
    fn rebuild_dense(
        &mut self,
        noisy: [bool; 5],
        dq: &[[f32; QUAD]; 5],
        unq: &[[f32; QUAD]; 5],
        fq: &[[f32; QUAD]; 5],
        qw: &[f32; QUAD],
        wet: f32,
        cover: f32,
        depth: f32,
        glazed: bool,
        wind_e: f32,
    ) {
        let kb = self.kb.clone();
        let nb = kb.half + 1;
        self.u_force.fill(0.0);
        self.u_dir.fill(0.0);
        let fr = f_ref();
        for c in 0..N_CLASSES {
            if !noisy[c] {
                continue;
            }
            let water_share = if self.is_water { 1.0 } else { cover };
            let solid_share = 1.0 - water_share;
            let uf = c * nb;
            if solid_share > 0.0 && self.n_var > 0 {
                let rs = solid_share;
                let nv = self.n_var as f32;
                let k_f = (2.0 * SIG_FORCE * SIG_FORCE).exp() * (1.0 - 0.35 * wet).powi(2) * if self.snow { 0.04 } else { 1.0 };
                let tau_mult = (1.0 + 0.6 * wet as f64) * if glazed { 0.7 } else { 1.0 } * if self.snow { 4.0 } else { 1.0 };
                let ps_base = self.splash_p * (1.0 - wet).powi(2) * if wet < 0.95 { 1.0 } else { 0.0 };
                let mut vs = 0.0f32;
                for q in 0..QUAD {
                    let f2 = fq[c][q] * fq[c][q] * k_f;
                    let v = rs * f2 * qw[q] / nv;
                    let tau = dq[c][q] as f64 * 1e-3 / (unq[c][q] as f64).max(0.05) * tau_mult;
                    let p = kb.pulse_psd(tau_bin(tau));
                    for (o, pv) in self.u_force[uf..uf + nb].iter_mut().zip(p) {
                        *o += v * *pv;
                    }
                    vs += ps_base * smoothstep(dq[c][q], 0.6, 1.6) * f2 * qw[q];
                }
                let v_s = rs * vs * SPLASH_REPS * splash_a2() / nv;
                if v_s > 0.0 {
                    let p = kb.pulse_psd(tau_bin(splash_tau()));
                    for (o, pv) in self.u_force[uf..uf + nb].iter_mut().zip(p) {
                        *o += v_s * *pv;
                    }
                }
                if wet > 0.05 && self.hiss_ok {
                    let mut s15 = 0.0f32;
                    let mut dmean = 0.0f32;
                    for q in 0..QUAD {
                        s15 += (fq[c][q] / fr).powf(1.5) * qw[q];
                        dmean += dq[c][q] * qw[q];
                    }
                    let a2 = (self.hiss_pa * wet * self.hiss_scale).powi(2) * s15;
                    let dur = (0.0025 * (dmean / 2.0).sqrt()).clamp(0.0015, 0.008);
                    let p = kb.splash_psd(dur_bin(dur));
                    for (o, pv) in self.u_dir[uf..uf + nb].iter_mut().zip(p) {
                        *o += rs * a2 * *pv;
                    }
                }
            }
            if water_share > 0.0 {
                let rw = water_share;
                let (dd, uu) = (&dq[c], &unq[c]);
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                let mut dmean = 0.0f32;
                for q in 0..QUAD {
                    let (d2, u2) = ((dd[q] as f64).powi(2), (uu[q] as f64).powi(2));
                    num += d2 * u2.powi(4) * qw[q] as f64;
                    den += d2 * u2.powi(3) * qw[q] as f64;
                    dmean += dd[q] * qw[q];
                }
                let u_e = (num / den.max(1e-30)).sqrt() as f32;
                let cb = click_bin(u_e);
                let u_bin = cb as f32 * super::kernels::CLICK_STEP;
                let mut s = 0.0f32;
                for q in 0..QUAD {
                    s += (dd[q] / D_REF_MM).powi(2) * (uu[q] / u_bin).powi(6) * qw[q];
                }
                let a2 = self.click_pa.powi(2) * (2.0 * SIG_CLICK * SIG_CLICK).exp() * if c == 0 { 0.09 } else { 1.0 } * s;
                let p = kb.click_psd(cb);
                for (o, pv) in self.u_dir[uf..uf + nb].iter_mut().zip(p) {
                    *o += rw * a2 * *pv;
                }
                let mut pent = 0.0f32;
                for q in 0..QUAD {
                    pent += entrainment_probability(dd[q], c, wind_e) * qw[q];
                }
                pent *= self.water_quality.clamp(0.0, 1.0) * if c == 4 { 1.0 + VL_EXTRA_BUBBLES as f32 } else { 1.0 };
                let fd = smoothstep(depth, 1.5, 8.0);
                let vb = rw * pent * (bubble_pa_per_m() * fd).powi(2) * eps2(c);
                if vb > 0.0 {
                    let db = damp_bin(depth).min(2);
                    let key = ((dmean / 0.05).round() as i32, db);
                    let col = &mut self.color[uf..uf + nb];
                    if self.color_key[c] != Some(key) {
                        bubble_color(&kb, c, dd, qw, db, col);
                        self.color_key[c] = Some(key);
                    }
                    for (o, pv) in self.u_dir[uf..uf + nb].iter_mut().zip(col.iter()) {
                        *o += vb * *pv;
                    }
                }
                if c >= 3 {
                    let mut a2s = 0.0f32;
                    for q in 0..QUAD {
                        let ind = if c == 4 || dd[q] >= 3.0 { 1.0 } else { 0.0 };
                        a2s += ind * splat_amp(dd[q], uu[q], c).powi(2) * qw[q];
                    }
                    a2s *= (2.0f32 * 0.25 * 0.25).exp();
                    if a2s > 0.0 {
                        let p = kb.splash_psd(dur_bin(splat_dur(dmean)));
                        for (o, pv) in self.u_dir[uf..uf + nb].iter_mut().zip(p) {
                            *o += rw * a2s * *pv;
                        }
                    }
                }
            }
        }
    }

    /// Impacts of one block: Cox draws, per-drop synthesis, matched noise.
    fn events(&mut self, ctx: &BlockCtx, gate: f32, m0: f64, m1: f64, n: usize) {
        let w = ctx.weather;
        let sr = self.sr;
        if !self.table.matches(&w.dsd, self.nystuen) {
            self.table.build(&w.dsd, self.nystuen);
            self.table_ver += 1;
        }
        let valid = self.table.valid;
        let vt_c = self.table.vt_c;
        let dq = self.table.dq;
        let vtq = self.table.vtq;
        let qw = self.table.wq;
        let (uhx, uhy) = wind_vector(w);
        let wn = self.nx * uhx + self.ny * uhy;
        let mut rate = [0.0f32; 5];
        let mut any = false;
        for c in 0..N_CLASSES {
            let flux = if valid[c] { w.class_flux[c].max(0.0) } else { 0.0 };
            let ff = (self.nz - wn / vt_c[c].max(0.3)).max(0.0);
            rate[c] = flux * self.base * gate * ff;
            any |= rate[c] > 0.0;
        }
        if !any {
            return;
        }
        let mmean = 0.5 * (m0 + m1) as f32;
        let (wet, cover, depth, glazed) = self.surface_state(w);
        let wind10 = w.wind_u10_mps * w.gust.max(0.0);
        let mut unq = [[0.0f32; QUAD]; 5];
        let mut fq = [[0.0f32; QUAD]; 5];
        for c in 0..N_CLASSES {
            for q in 0..QUAD {
                unq[c][q] = (vtq[c][q] * self.nz - wn).max(0.2 * vtq[c][q]);
                fq[c][q] = force_peak(dq[c][q], unq[c][q]);
            }
        }
        let mut dense = [false; 5];
        let mut spill = [false; 5];
        if self.lod_dense.is_finite() {
            let mut e_rel = [0.0f32; 5];
            let mut emax = 0.0f32;
            for c in 0..N_CLASSES {
                e_rel[c] = (0..QUAD).map(|q| fq[c][q] * fq[c][q] * qw[q]).sum();
                emax = emax.max(e_rel[c]);
            }
            let (pd, ps) = self.lod_prev.unwrap_or(([false; 5], [false; 5]));
            for c in 0..N_CLASSES {
                let er = e_rel[c] / emax.max(1e-30);
                let overlap = rate[c] * self.teff[c];
                let hy_d = if pd[c] { 0.85 } else { 1.15 };
                let faint = er < SALIENCE && overlap >= 0.3 * hy_d && !self.is_water;
                dense[c] = rate[c] > 0.0 && (overlap >= self.dense_thr * hy_d || faint);
                spill[c] = rate[c] > 0.0
                    && !dense[c]
                    && rate[c] * (BLOCK_REF / sr) > 0.5 * self.caps[c] * if ps[c] { 0.85 } else { 1.15 };
            }
            self.lod_prev = Some((dense, spill));
        }

        // ---- sparse classes: Poisson counts; above the cap the excess is matched noise -------
        let mut rng = Rng::from_key(self.key_ev, ctx.index);
        let noise_base = rng.below(1 << 30);
        let kb = self.kb.clone();
        let wp = WaterParams { click_pa: self.click_pa, depth_mm: depth, wind_mps: wind10, water_quality: self.water_quality, p_scale: 1.0 };
        let mut share = [0.0f32; 5];
        for c in 0..N_CLASSES {
            let lam = if dense[c] { 0.0 } else { rate[c] * (n as f32 / sr) * mmean };
            let cap = self.caps[c] * (n as f32 / BLOCK_REF);
            let capped = lam > cap;
            let lam_draw = if capped { cap } else { lam };
            let excess = if capped { 1.0 - cap / lam.max(1e-12) } else { 0.0 };
            let scale = if capped && !spill[c] { (lam / cap.max(1e-12)).sqrt() } else { 1.0 };
            share[c] = if dense[c] {
                1.0
            } else if spill[c] {
                excess
            } else {
                0.0
            };
            if lam_draw <= 0.0 {
                continue;
            }
            let count = rng.poisson(lam_draw as f64);
            for _ in 0..count {
                let off = ((ramp_position(rng.uniform(), m0, m1) * n as f64) as usize).min(n - 1);
                let d = self.table.sample(rng.uniform(), c);
                let vt = fall_speed(d);
                let un = (vt * self.nz - wn).max(0.2 * vt);
                let to_water = self.is_water || (cover > 0.0 && rng.uniform() < cover as f64);
                if to_water {
                    water_event(&kb, &mut rng, &mut self.direct, off, d, un, c, scale, &wp);
                } else {
                    self.solid_event(&mut rng, off, d, un, scale, wet, glazed);
                }
            }
        }

        // ---- dense classes and capped excess: matched noise --------------------------------
        let mut noisy = [false; 5];
        let mut any_noisy = false;
        for c in 0..N_CLASSES {
            noisy[c] = dense[c] || spill[c];
            any_noisy |= noisy[c] && share[c] > 0.0;
        }
        if !any_noisy {
            return;
        }
        let wind_e = wind10.clamp(7.5, 10.0);
        let key = DenseKey { table_ver: self.table_ver, glazed, dense, spill, wet, cover, depth };
        let rebuild = match &self.dense_key {
            None => true,
            Some((k, wn0, we0)) => *k != key || (wn - *wn0).abs() > 0.15 || (wind_e - *we0).abs() > 0.2,
        };
        if rebuild {
            self.rebuild_dense(noisy, &dq, &unq, &fq, &qw, wet, cover, depth, glazed, wind_e);
            self.dense_key = Some((key, wn, wind_e));
        }
        let nb = kb.half + 1;
        let mut rf = [0.0f32; 5];
        for c in 0..N_CLASSES {
            if noisy[c] {
                rf[c] = rate[c] * share[c] / sr * m1.max(0.0) as f32;
            }
        }
        let stride = (kb.frame + 2) as u64;
        // force rows (one per bank variant, independent noise)
        if self.n_var > 0 {
            let mut tot = 0.0f32;
            self.psd.fill(0.0);
            for c in 0..N_CLASSES {
                if rf[c] <= 0.0 {
                    continue;
                }
                let u = &self.u_force[c * nb..(c + 1) * nb];
                for (o, v) in self.psd.iter_mut().zip(u) {
                    *o += rf[c] * *v;
                }
            }
            for v in self.psd.iter() {
                tot += *v;
            }
            if tot > 0.0 {
                for v in 0..self.n_var {
                    self.force_rows[v].synth(&kb, &self.psd, noise_base + v as u64 * stride, &mut self.force[v]);
                }
            }
        }
        // direct row
        self.psd.fill(0.0);
        let mut tot = 0.0f32;
        for c in 0..N_CLASSES {
            if rf[c] <= 0.0 {
                continue;
            }
            let u = &self.u_dir[c * nb..(c + 1) * nb];
            for (o, v) in self.psd.iter_mut().zip(u) {
                *o += rf[c] * *v;
            }
        }
        for v in self.psd.iter() {
            tot += *v;
        }
        if tot > 0.0 {
            self.direct_row.synth(&kb, &self.psd, noise_base + self.n_var as u64 * stride, &mut self.direct.ola);
        }
    }
}

impl Voice for RainPatchVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n.min(self.n).min(out.len());
        if n == 0 {
            return Emission::silent(self.stem);
        }
        let w = ctx.weather;
        let m1 = self.cox.step(ctx.index, n);
        let m0 = self.cox.prev;
        let gate = rain_gate(w.precip_type);
        if gate > 0.0 && self.base > 0.0 && w.class_flux.iter().any(|v| *v > 0.0) {
            self.events(ctx, gate, m0, m1, n);
        }
        if !self.busy() {
            return Emission::silent(self.stem);
        }
        let y = &mut out[..n];
        if self.n_var > 0 {
            let sum = &mut self.sum[..n];
            sum.fill(0.0);
            let mut fed_any = false;
            for v in 0..self.n_var {
                let f = &mut self.fbuf[..n];
                f.fill(0.0);
                let fed = self.force[v].pop_into(f);
                if self.has_modes {
                    if fed {
                        self.banks[v].process(Some(f), sum);
                        fed_any = true;
                    } else if !self.banks[v].silent(1e-11) {
                        self.banks[v].process(None, sum);
                        fed_any = true;
                    }
                } else if fed {
                    for (s, x) in sum.iter_mut().zip(f.iter()) {
                        *s += *x;
                    }
                    fed_any = true;
                }
            }
            if fed_any || !self.rad.quiet() {
                self.rad.process(sum);
                let cal = self.cal;
                for (o, s) in y.iter_mut().zip(sum.iter()) {
                    *o += *s * cal;
                }
            }
        }
        self.direct.render(y);
        if let Some(x) = self.xfer.as_mut() {
            x.process(y);
        }
        let g = self.xfer_gain * self.spreading;
        for v in y.iter_mut() {
            *v *= g;
        }
        if let Some(a) = self.air.as_mut() {
            a.process(y, w.temperature_c, w.humidity_pct);
        }
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak > LIVE_EPS && peak.is_finite() {
            self.em
        } else {
            Emission::silent(self.stem)
        }
    }

    fn reset(&mut self) {
        for o in self.force.iter_mut() {
            o.clear();
        }
        for b in self.banks.iter_mut() {
            b.reset();
        }
        self.rad.reset();
        self.direct.clear();
        self.cox.reset();
        self.lod_prev = None;
        self.dense_key = None;
        self.color_key = [None; 5];
        if let Some(x) = self.xfer.as_mut() {
            x.reset();
        }
        if let Some(a) = self.air.as_mut() {
            a.reset();
        }
        let _ = self.full;
    }
}
