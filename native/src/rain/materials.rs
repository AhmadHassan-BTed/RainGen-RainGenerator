//! Surface materials for rain, hail, drip and snow impacts (raingen/synth/materials.py and the
//! material-derived parts of raingen/synth/rain.py).
//!
//! The MATERIALS table itself comes from Python (`Engine.set_materials`): modes (f, T60, gain),
//! hardness, splash propensity, radiation band and the calibration anchor `level_db` (peak dB SPL
//! at 1 m of a 2 mm drop at 6.5 m/s on the dry surface). Here every referenced material is turned
//! into its synthesis data once per scene build: radiation filter, calibration (Pa per N through
//! modes + radiation), the effective response duration used by the level-of-detail decisions, and
//! the detuned / re-damped modal-bank variants. Also the patch geometry (energy-equivalent distance
//! and spread), the interior transfer (through-ceiling mass law, slabs, glazing, window sills) and
//! the stem routing.

use super::kernels::{butter1_lp, db_to_pa, impact_pulse, radiation_coefs, Cascade, ModalLanes, D_REF_MM, V_REF};
use crate::dsp::BiquadCoefs;
use crate::rng::RngFactory;
use crate::types::{Material, MaterialKind, MaterialTable, Patch, Shelter, Stem};
use std::collections::HashMap;
use std::sync::Arc;

/// Synthesis data of one material.
#[derive(Clone, Debug)]
pub struct MatData {
    /// Index in the engine's MaterialTable (wetness / puddle lookups), if any.
    pub idx: Option<usize>,
    pub name: String,
    pub kind: MaterialKind,
    pub modes: Vec<(f32, f32, f32)>,
    pub hardness: f32,
    pub splash: f32,
    pub level_db: f32,
    pub rad: Vec<BiquadCoefs>,
    /// Pa @ 1 m per N through modes + radiation, so the reference drop peaks at level_db.
    pub cal: f32,
    /// Effective duration (s) of the impact response: (sum y^2)^2 / sum y^4, onset-capped.
    pub teff: f32,
    pub youngs: f32,
}

impl MatData {
    #[inline]
    pub fn is_water(&self) -> bool {
        self.kind == MaterialKind::Water
    }
}

/// Widest gap between neighbouring modes before it is filled (ratio, half an octave).
const MODE_GAP_RATIO: f32 = 1.414;
/// Gaps are filled only above this frequency.
const MODE_FILL_MIN_HZ: f32 = 250.0;
/// Spacing of the filling modes (third octave).
const MODE_FILL_RATIO: f32 = 1.26;

/// The material's modes, with gaps wider than half an octave filled. A plate's modal density is
/// constant in frequency (n = S/(4 pi) sqrt(rho h / D), Cremer & Heckl), so a table that lists the
/// lowest plate modes and then jumps to a log-spaced high series (tin_roof: 408 Hz -> 900 Hz)
/// leaves a hole that real panels do not have: measured as a 15-17 dB notch at 500-630 Hz in
/// the attic presets. The filling modes take log-interpolated T60 and gain from their neighbours.
fn modes_of(m: &Material) -> Vec<(f32, f32, f32)> {
    let mut v: Vec<(f32, f32, f32)> = m.modes.iter().map(|md| (md.freq_hz, md.t60_s, md.gain)).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() < 2 {
        return v;
    }
    let mut out = Vec::with_capacity(v.len() + 8);
    for i in 0..v.len() {
        out.push(v[i]);
        if i + 1 == v.len() {
            break;
        }
        let (a, b) = (v[i], v[i + 1]);
        // only above the fundamental region: the lowest plate / membrane modes are genuinely
        // sparse (constant density per Hz means wide ratios at the bottom)
        if a.0 < MODE_FILL_MIN_HZ || b.0 / a.0 <= MODE_GAP_RATIO {
            continue;
        }
        let k = ((b.0 / a.0).ln() / MODE_FILL_RATIO.ln()).ceil() as usize;
        for j in 1..k {
            let x = j as f32 / k as f32;
            let lerp = |p: f32, q: f32| (p.max(1e-9).ln() * (1.0 - x) + q.max(1e-9).ln() * x).exp();
            out.push((lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2)));
        }
    }
    out
}

fn calibration(modes: &[(f32, f32, f32)], rad: &[BiquadCoefs], level_db: f32, sr: f32) -> f32 {
    let pulse = impact_pulse(D_REF_MM as f64, V_REF as f64, sr as f64);
    let length = (0.08 * sr) as usize;
    let ir = ModalLanes::impulse_response(modes, sr, length);
    let mut y: Vec<f64> = if ir.iter().any(|v| *v != 0.0) {
        let mut y = vec![0.0; length];
        for (i, p) in pulse.iter().enumerate() {
            if *p == 0.0 {
                continue;
            }
            for j in 0..length - i.min(length) {
                y[i + j] += p * ir[j];
            }
        }
        y
    } else {
        pulse.clone()
    };
    Cascade::filter_f64(rad, &mut y);
    let peak = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if peak > 0.0 {
        (db_to_pa(level_db) as f64 / peak) as f32
    } else {
        0.0
    }
}

fn response_teff(modes: &[(f32, f32, f32)], sr: f32) -> f32 {
    let pulse = impact_pulse(D_REF_MM as f64, V_REF as f64, sr as f64);
    let length = (0.25 * sr) as usize;
    let ir = ModalLanes::impulse_response(modes, sr, length);
    let y: Vec<f64> = if ir.iter().any(|v| *v != 0.0) {
        let mut y = vec![0.0; length];
        for (i, p) in pulse.iter().enumerate() {
            if i >= length {
                break;
            }
            for j in 0..length - i {
                y[i + j] += p * ir[j];
            }
        }
        y
    } else {
        pulse
    };
    let e2: f64 = y.iter().map(|v| v * v).sum();
    let e4: f64 = y.iter().map(|v| v * v * v * v).sum();
    let t = if e4 > 0.0 { e2 * e2 / e4 / sr as f64 } else { 1e-4 };
    t.clamp(2e-4, 5e-3) as f32
}

/// Per-build cache of material synthesis data (resolution by name or alias like get_material).
pub struct Materials<'a> {
    table: &'a MaterialTable,
    sr: f32,
    cache: HashMap<usize, Arc<MatData>>,
    fallback: Option<Arc<MatData>>,
}

impl<'a> Materials<'a> {
    pub fn new(table: &'a MaterialTable, sr: f32) -> Materials<'a> {
        Materials { table, sr, cache: HashMap::new(), fallback: None }
    }

    fn make(&self, m: &Material, idx: Option<usize>) -> MatData {
        let modes = modes_of(m);
        let rad = radiation_coefs(m.radiation_hp_hz, m.radiation_lp_hz, self.sr);
        let cal = calibration(&modes, &rad, m.level_db, self.sr);
        let teff = response_teff(&modes, self.sr);
        MatData {
            idx,
            name: m.name.clone(),
            kind: m.kind,
            modes,
            hardness: m.hardness,
            splash: m.splash,
            level_db: m.level_db,
            rad,
            cal,
            teff,
            youngs: m.youngs_modulus_pa(),
        }
    }

    /// Material by name or alias ('concrete' when unknown).
    pub fn get(&mut self, name: &str) -> Arc<MatData> {
        match self.table.resolve(name) {
            Some(i) => {
                if let Some(d) = self.cache.get(&i) {
                    return d.clone();
                }
                let d = Arc::new(self.make(&self.table.materials[i], Some(i)));
                self.cache.insert(i, d.clone());
                d
            }
            None => {
                if let Some(f) = &self.fallback {
                    return f.clone();
                }
                let d = Arc::new(self.make(&Material::default(), None));
                self.fallback = Some(d.clone());
                d
            }
        }
    }

    /// Kind of a material by name without building its data.
    pub fn kind_of(&self, name: &str) -> MaterialKind {
        self.table.get(name).kind
    }

    pub fn level_of(&self, name: &str) -> f32 {
        self.table.get(name).level_db
    }

    pub fn canonical(&self, name: &str) -> String {
        self.table.get(name).name.clone()
    }

    pub fn table(&self) -> &MaterialTable {
        self.table
    }
}

/// Modes of bank variant v of a material (variant 0: the material itself): detuned and re-damped
/// by log-normal factors (stream "rain/bank/{name}/{spread}/{v}"), each gain scaled by
/// 1/sqrt(damping factor) so every mode keeps its impulse energy.
pub fn variant_modes(rngf: &RngFactory, mat: &MatData, spread: f32, v: usize) -> Vec<(f32, f32, f32)> {
    if v == 0 || mat.modes.is_empty() {
        return mat.modes.clone();
    }
    let mut r = rngf.get(&format!("rain/bank/{}/{}/{}", mat.name, spread, v), 0);
    let det: Vec<f64> = (0..mat.modes.len()).map(|_| (spread as f64 * r.normal()).exp()).collect();
    let dmp: Vec<f64> = (0..mat.modes.len()).map(|_| (0.15 * r.normal()).exp()).collect();
    mat.modes
        .iter()
        .zip(det.iter().zip(dmp.iter()))
        .map(|((f, t, g), (a, b))| (f * *a as f32, t * *b as f32, g / (*b as f32).sqrt()))
        .collect()
}

// ---------------------------------------------------------------------------
// patch geometry, stems, interior transfer
// ---------------------------------------------------------------------------
pub const CEILING_MASS_KG_M2: f32 = 10.0;
pub const CEILING_TL_MAX_DB: f32 = 26.0;
pub const SILL_GLAZING_SHARE: f32 = 0.25;
pub const WINDOW_AREA_M2: f32 = 1.5;
pub const ATTIC_DEPTH_M: f32 = 1.2;
pub const ATTIC_ABSORPTION: f32 = 0.4;
pub const SLAB_RHO_KG_M3: f32 = 2300.0;

/// Energy-equivalent distance of a patch and its angular spread (0..1).
pub fn patch_geometry(p: &Patch) -> (f32, f32) {
    let pi = std::f32::consts::PI;
    let ground = p.param_bool("ground", false);
    if ground && p.params.get("r_inner").is_some() && p.params.get("r_outer").is_some() {
        let r1 = p.param_f32("r_inner", 0.0).max(0.0);
        let r2 = p.param_f32("r_outer", r1 + 1.0).max(r1 + 1e-3);
        let h = (-p.position[2]).max(0.05);
        let inv = ((r2 * r2 + h * h) / (r1 * r1 + h * h)).ln() / (r2 * r2 - r1 * r1);
        let r_eq = 1.0 / inv.max(1e-12).sqrt();
        let (th1, th2) = (r1.atan2(h), r2.atan2(h));
        let dphi = p.param_f32("az_width", pi / 4.0) * (0.5 * (th1 + th2)).sin();
        let spread = (dphi.max(th2 - th1) / pi).clamp(0.0, 1.0);
        return (r_eq.max(0.1), spread);
    }
    let d = p.distance().max(0.1);
    let rad = (p.area_m2.max(1e-4) / pi).sqrt();
    let q = (rad / d) * (rad / d);
    let r_eq = if q > 1e-6 { d * (q / q.ln_1p()).sqrt() } else { d };
    let spread = (2.0 * (rad / d).atan() / pi).clamp(0.0, 1.0);
    (r_eq.max(0.1), spread)
}

/// Patches closer than this are the near layer (rain_near, coherent, localised). Within ~5 m a
/// listener resolves single drop impacts on a patch: the per-patch event rate per critical band is
/// below the ear's ~2-5 ms temporal resolution and the direct sound dominates the outdoor
/// reverberation (direct-to-reverberant > 0 dB for the place tails of spatial::space). The
/// 1.5-4 m ground ring (r_eq ~3.3 m) belongs here, not to the diffuse far field.
pub const NEAR_STEM_M: f32 = 5.0;

pub fn stem_for(p: &Patch) -> Stem {
    if p.interior {
        Stem::Interior
    } else if p.distance() < NEAR_STEM_M {
        Stem::RainNear
    } else {
        Stem::RainFar
    }
}

/// First-order shelf of the field-incidence mass law TL = 20 log10(m f) - 47 dB from TL = 0 up to a
/// coincidence-limited plateau (bilinear transform of (s/w_p + 1) / (s/w_z + 1), as scipy).
pub fn mass_law_coefs(mass_kg_m2: f32, tl_max_db: f32, sr: f32) -> BiquadCoefs {
    let f_z = 10f64.powf(47.0 / 20.0) / (mass_kg_m2 as f64).max(1e-3);
    let mut f_p = f_z * 10f64.powf((tl_max_db as f64).max(0.0) / 20.0);
    f_p = f_p.min(0.45 * sr as f64);
    let (w_z, w_p) = (2.0 * std::f64::consts::PI * f_z, 2.0 * std::f64::consts::PI * f_p);
    let k = 2.0 * sr as f64;
    let (n0, n1) = (1.0 + k / w_p, 1.0 - k / w_p);
    let (d0, d1) = (1.0 + k / w_z, 1.0 - k / w_z);
    BiquadCoefs { b0: n0 / d0, b1: n1 / d0, b2: 0.0, a1: d1 / d0, a2: 0.0 }
}

/// (gain, filter) of what reaches the room from a patch whose struck face is not the face
/// radiating into the listener's space (`_interior_transfer`).
pub fn interior_transfer(p: &Patch, sr: f32) -> (f32, Vec<BiquadCoefs>) {
    if !p.interior {
        let sh = p.param_str("shelter").map(Shelter::from_name);
        if p.param_bool("sill", false) && matches!(sh, Some(Shelter::RoomClosed) | Some(Shelter::RoomOpen)) {
            let (r_eq, _) = patch_geometry(p);
            let s_win = p.param_f32("window_area_m2", WINDOW_AREA_M2).max(0.1);
            let g = (SILL_GLAZING_SHARE * 2.0 * std::f32::consts::PI * r_eq * r_eq / (0.5 * s_win)).sqrt();
            return (g, Vec::new());
        }
        return (1.0, Vec::new());
    }
    if p.param_bool("through_ceiling", false) {
        let (r_eq, _) = patch_geometry(p);
        let a_attic = (ATTIC_ABSORPTION * p.area_m2).max(1.0);
        let h = p.param_f32("attic_depth_m", ATTIC_DEPTH_M);
        let build_up = r_eq * (2.0 * std::f32::consts::PI * 4.0 / a_attic + 1.0 / (h * h)).sqrt();
        let mass = p.param_f32("ceiling_mass_kg_m2", CEILING_MASS_KG_M2);
        let tl = p.param_f32("ceiling_tl_max_db", CEILING_TL_MAX_DB);
        return (build_up, vec![mass_law_coefs(mass, tl, sr)]);
    }
    let th = p.param_f32("thickness_m", 0.0);
    if th != 0.0 {
        let mass = SLAB_RHO_KG_M3 * th;
        let tl0 = (20.0 * (mass * 100.0).log10() - 47.0).max(0.0);
        return (10f32.powf(-tl0 / 20.0), vec![butter1_lp(100.0, sr)]);
    }
    match p.param_str("glazing").map(|s| s.to_ascii_lowercase()) {
        Some(g) if g == "double" => (0.4, vec![butter1_lp(3500.0, sr)]),
        Some(g) if g == "triple" => (0.25, vec![butter1_lp(2500.0, sr)]),
        _ => (1.0, Vec::new()),
    }
}

/// Spherical spreading 1/r re 1 m (r >= 0.05 m).
#[inline]
pub fn spreading_gain(distance_m: f32) -> f32 {
    1.0 / distance_m.max(0.05)
}

/// Unit vector (default +z).
pub fn unit(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Horizontal wind velocity (listener frame, blowing TOWARD); wind_azimuth is where it comes FROM.
#[inline]
pub fn wind_vector(w: &crate::types::Weather) -> (f32, f32) {
    let u = w.wind_listener_mps.max(0.0);
    let a = w.wind_azimuth;
    (-u * a.cos(), -u * a.sin())
}

/// Puddle depth by material name when the weather carries it (None when absent).
pub fn puddle_named(w: &crate::types::Weather, name: &str) -> Option<f32> {
    w.puddle_depth_mm.iter().find(|(k, _)| k == name).map(|(_, v)| *v)
}

/// Wetness of a material (index lookup, name fallback).
#[inline]
pub fn wetness(w: &crate::types::Weather, m: &MatData) -> f32 {
    match m.idx {
        Some(i) if !w.wet_by_mat.is_empty() => w.wetness_of(i),
        _ => w.wetness_named(&m.name),
    }
}

/// Puddle depth (mm) of a material (0 when absent).
#[inline]
pub fn puddle_mm(w: &crate::types::Weather, m: &MatData) -> f32 {
    match m.idx {
        Some(i) if !w.puddle_by_mat.is_empty() => w.puddle_mm_of(i),
        _ => puddle_named(w, &m.name).unwrap_or(0.0),
    }
}
