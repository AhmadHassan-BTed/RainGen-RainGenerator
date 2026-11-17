//! Precipitation family (port of raingen/synth/rain.py + kernels.py + events.py + materials.py):
//! rain patches with level of detail, the diffuse rain bed, drips, gutters, hail and snow.
//!
//! Entry point used by the engine: `build_voices`. Voices are created here from the scene and
//! the layout patches, in a fixed deterministic order, each with its own RNG stream key
//! (`make_rain_generators` of raingen/synth/rain.py).

pub mod air;
pub mod bed;
pub mod drips;
pub mod dsd;
pub mod events;
pub mod gutters;
pub mod hail;
pub mod kernels;
pub mod materials;
pub mod rain;
pub mod snow;

use crate::engine::{BoxedVoice, BuildCfg};
use crate::rng::RngFactory;
use crate::types::{precip, MaterialKind, MaterialTable, Patch, PatchKind, Scene};
use kernels::KernelBank;
use materials::Materials;
use rain::{lod_settings, Lod};
use serde_json::Value;
use std::sync::Arc;

/// Patch kinds that receive rain (RAIN_KINDS).
pub fn is_rain_kind(k: &PatchKind) -> bool {
    matches!(
        k,
        PatchKind::Surface | PatchKind::Water | PatchKind::Membrane | PatchKind::Vegetation | PatchKind::VehicleBody | PatchKind::Gutter
    )
}

/// Beyond this distance a patch is never a per-drop voice.
pub const FULL_MAX_DIST_M: f32 = 12.0;
/// Larger surfaces are voices only within VOICE_NEAR_M.
pub const VOICE_MAX_AREA_M2: f32 = 25.0;
pub const VOICE_NEAR_M: f32 = 4.0;

/// Everything a precipitation voice needs at construction.
pub struct BuildCtx<'a> {
    pub scene: &'a Scene,
    pub patches: &'a [Patch],
    pub sr: u32,
    pub block: usize,
    pub kb: Arc<KernelBank>,
    pub mats: Materials<'a>,
    pub rngf: RngFactory,
    pub lod: Lod,
    /// Nystuen (2001) class boundaries instead of Medwin et al. (1992).
    pub nystuen: bool,
    pub water_quality: f32,
}

impl<'a> BuildCtx<'a> {
    pub fn new(scene: &'a Scene, patches: &'a [Patch], materials: &'a MaterialTable, cfg: &BuildCfg) -> BuildCtx<'a> {
        let pr = &scene.weather.precipitation;
        let quality = scene.render.quality.to_ascii_lowercase();
        BuildCtx {
            scene,
            patches,
            sr: cfg.sr,
            block: cfg.block,
            kb: KernelBank::shared(cfg.sr, cfg.block),
            mats: Materials::new(materials, cfg.sr as f32),
            rngf: cfg.rng,
            lod: lod_settings(&quality),
            nystuen: pr.class_boundaries.trim().eq_ignore_ascii_case("nystuen"),
            water_quality: pr.water_quality.clamp(0.0, 1.0),
        }
    }
}

/// `_loudness_score`: rough dB of a patch at the listener, for the voice budget.
fn loudness_score(p: &Patch, table: &MaterialTable) -> f32 {
    let m = table.get(&p.material);
    let lvl = m.level_db + if m.kind == MaterialKind::Water { 6.0 } else { 0.0 };
    lvl + 10.0 * (p.area_m2 * p.exposure.max(0.0)).max(1e-3).log10() - 20.0 * p.distance().max(0.3).log10()
        + if p.interior { 6.0 } else { 0.0 }
}

/// `_is_voice`: per-drop voice candidates (shelter panels, membranes, vehicle bodies, gutters, and
/// surfaces near enough for single drops to stand out).
fn is_voice(p: &Patch) -> bool {
    if p.interior || matches!(p.kind, PatchKind::Membrane | PatchKind::VehicleBody | PatchKind::Gutter) {
        return true;
    }
    if p.distance() > FULL_MAX_DIST_M {
        return false;
    }
    p.area_m2 <= VOICE_MAX_AREA_M2 || p.distance() <= VOICE_NEAR_M
}

/// `_is_downspout`.
pub fn is_downspout(p: &Patch) -> bool {
    p.param_str("type").map(|s| s.eq_ignore_ascii_case("downspout")).unwrap_or(false) || p.params.get("pipe_length_m").is_some()
}

/// `_roof_area_from_layout`: roof catchment (m^2) feeding the gutters.
pub fn roof_area_from_layout(layout: &[Patch]) -> f32 {
    let mut best: Option<f32> = None;
    for p in layout {
        if p.kind == PatchKind::Gutter {
            let c = p.param_f32("catchment_m2", 0.0);
            if c != 0.0 {
                best = Some(best.map_or(c, |b: f32| b.max(c)));
            }
        }
    }
    let roofs: f32 = layout
        .iter()
        .filter(|p| (p.material == "tin_roof" || p.material == "tile") && !p.param_bool("ground", false))
        .map(|p| p.area_m2)
        .sum();
    let area = if best.is_some() || roofs != 0.0 { best.unwrap_or(f32::MIN).max(roofs.min(200.0)) } else { 30.0 };
    area.max(5.0)
}

fn add_type(set: &mut [bool; 9], v: &Value) {
    match v {
        Value::String(s) => {
            let s = s.trim().to_ascii_lowercase();
            if s == "all" {
                set.iter_mut().for_each(|x| *x = true);
            } else {
                set[(precip::from_name(&s) as usize).min(8)] = true;
            }
        }
        Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                set[(i as usize).min(8)] = true;
            }
        }
        Value::Array(a) => a.iter().for_each(|x| add_type(set, x)),
        _ => {}
    }
}

/// `_precip_types`: the scene's own type plus timeline keyframes on the type; an explicit
/// `precip_types` entry (top level or under `native`) of "all" / a list overrides it.
fn precip_types(scene: &Scene) -> [bool; 9] {
    let mut set = [false; 9];
    for path in ["native.precip_types", "precip_types"] {
        if let Some(v) = scene.get(path) {
            add_type(&mut set, v);
            if set.iter().any(|x| *x) {
                return set;
            }
        }
    }
    set[(scene.weather.precipitation.precip_type as usize).min(8)] = true;
    if let Value::Array(kfs) = &scene.timeline {
        for kf in kfs {
            let path = kf.get("path").and_then(|p| p.as_str()).unwrap_or("");
            if path.ends_with("precipitation.type") {
                if let Some(v) = kf.get("value") {
                    add_type(&mut set, v);
                }
            }
        }
    }
    set
}

/// Every precipitation voice the scene needs around `patches` (make_rain_generators).
pub fn build_voices(scene: &Scene, patches: &[Patch], materials: &MaterialTable, cfg: &BuildCfg) -> Vec<BoxedVoice> {
    let mut bc = BuildCtx::new(scene, patches, materials, cfg);
    let types = precip_types(scene);
    let has = |t: u8| types[t as usize];
    let want_rain = (0..9u8).any(|t| types[t as usize] && !matches!(t, precip::NONE | precip::SNOW | precip::GRAUPEL));
    let want_hail = has(precip::HAIL) || has(precip::SLEET) || has(precip::GRAUPEL);
    let snow_place = scene.place.ground.trim().to_ascii_lowercase().starts_with("snow")
        || scene
            .place
            .surfaces
            .as_object()
            .map(|m| m.iter().any(|(k, v)| v.as_f64().unwrap_or(0.0) > 0.0 && k.to_ascii_lowercase().contains("snow")))
            .unwrap_or(false);
    let want_snow = has(precip::SNOW) || snow_place;

    let k_full = bc.lod.k;
    let rain_patches: Vec<&Patch> = patches.iter().filter(|p| is_rain_kind(&p.kind) && p.exposure > 0.0).collect();
    let mut ranked: Vec<(&Patch, f32)> = rain_patches.iter().map(|p| (*p, loudness_score(p, materials))).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.id.cmp(&b.0.id)));
    let full_ids: Vec<&str> = ranked.iter().filter(|(p, _)| is_voice(p)).take(k_full).map(|(p, _)| p.id.as_str()).collect();

    let mut voices: Vec<BoxedVoice> = Vec::new();
    if want_rain {
        for p in &rain_patches {
            let full = full_ids.contains(&p.id.as_str());
            voices.push(Box::new(rain::RainPatchVoice::new(&mut bc, p, full)));
        }
    }
    if want_rain || want_hail {
        voices.push(Box::new(bed::RainBedVoice::new(&mut bc)));
    }

    // drips: every layout drip edge, plus crown drips under up to three nearby trees
    let roof_area = roof_area_from_layout(patches);
    for p in patches.iter().filter(|p| p.kind == PatchKind::DripEdge) {
        for v in drips::build(&mut bc, p, false, roof_area) {
            voices.push(Box::new(v));
        }
    }
    let with_edges: Vec<String> = patches
        .iter()
        .filter(|p| p.kind == PatchKind::DripEdge)
        .map(|p| match p.params.get("source") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => "None".to_string(),
            Some(v) => v.to_string(),
        })
        .collect();
    let mut crowns: Vec<(u8, f32, &str, &Patch)> = Vec::new();
    for p in patches.iter().filter(|p| p.kind == PatchKind::Vegetation) {
        let horiz = p.position[0].hypot(p.position[1]);
        let species = p.param_str("species").unwrap_or("").trim().to_ascii_lowercase();
        if p.param_bool("canopy", false) {
            crowns.push((0, horiz, p.id.as_str(), p));
        } else if !species.is_empty() && species != "grass" && !with_edges.iter().any(|s| *s == p.id) && horiz <= 15.0 {
            crowns.push((1, horiz, p.id.as_str(), p));
        }
    }
    crowns.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).then(a.2.cmp(b.2)));
    for (_, _, _, p) in crowns.into_iter().take(3) {
        for v in drips::build(&mut bc, p, true, roof_area) {
            voices.push(Box::new(v));
        }
    }

    // runoff: gutter channels share the roof flow, downspouts carry it again
    let gutters: Vec<&Patch> = patches.iter().filter(|p| p.kind == PatchKind::Gutter).collect();
    let channels: Vec<&Patch> = gutters.iter().filter(|p| !is_downspout(p)).cloned().collect();
    let pipes: Vec<&Patch> = gutters.iter().filter(|p| is_downspout(p)).cloned().collect();
    for group in [&channels, &pipes] {
        for p in group.iter() {
            voices.push(Box::new(gutters::GutterVoice::new(&mut bc, p, group.len())));
        }
    }

    let near: Vec<&Patch> = ranked.iter().filter(|(p, _)| p.distance() <= 15.0).take(k_full).map(|(p, _)| *p).collect();
    if want_hail {
        for (i, p) in near.iter().enumerate() {
            voices.push(Box::new(hail::HailPatchVoice::new(&mut bc, p, i == 0)));
        }
    }
    if want_snow {
        for p in near.iter().filter(|p| p.distance() <= 8.0) {
            let k = bc.mats.kind_of(&p.material);
            if matches!(k, MaterialKind::Water | MaterialKind::Snow) {
                continue;
            }
            voices.push(Box::new(snow::SnowPatchVoice::new(&mut bc, p)));
        }
        voices.push(Box::new(snow::SaltationVoice::new(&mut bc, snow_place)));
    }
    voices
}
