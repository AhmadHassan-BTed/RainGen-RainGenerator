//! Drips (DripGenerator of raingen/synth/rain.py, DripNode of raingen/physics/wetness.py).
//!
//! Integrate-and-fire drip nodes: each node fills its pendant drop from its catchment (feed:
//! rain on the catchment plus the edge film, gutter-routed roof flow, canopy release plus slow
//! drainage of the canopy store, or seepage from stored water) and releases a drop when the volume
//! crosses its jittered detachment threshold (Tate's law size; Shaw 1984: periodic at low flow,
//! period doubling ~3 Hz, chaotic above ~10 Hz through a logistic map). Each landing spot is one
//! voice: on water a drop in the regular-entrainment band of Oguz & Prosperetti (1990) traps one
//! bubble of radius ~0.3-0.4 D and rings (the 'plink'), otherwise it is an irregular splash with
//! the large-drop bubble statistics (the 'splat'); solid targets get a force pulse through their
//! modes plus prompt-splash micro-impacts, a wet film adds crown-splash hiss and a puddled share
//! becomes water impacts. A tree crown drips from up to four spots around it.

use super::air::AirFilter;
use super::dsd::{fall_speed_from_height, tate_drop_diameter_mm};
use super::kernels::{click_bin, db_to_pa, dur_bin, f_ref, force_peak, smoothstep, KernelBank, SolidVoice, D_REF_MM, SPLASH_VARIANTS};
use super::materials::{puddle_mm, puddle_named, spreading_gain, wetness, MatData};
use super::rain::{bubble_pa_per_m, liquid_share, regular_entrainment, water_event, DirectBus, WaterParams, AIR_MIN_DIST_M, FILM_HISS_REL_DB};
use super::BuildCtx;
use crate::engine::{BlockCtx, Voice};
use crate::rng::Rng;
use crate::types::{Emission, MaterialKind, Patch, Stem, Weather};
use std::sync::Arc;

const TAU_STORE_S: f64 = 180.0;
const MAX_DRIP_HZ: f64 = 40.0;
/// Fraction of the pendant volume left on the edge after detachment.
const RESIDUAL: f64 = 0.04;
const MAX_EVENTS_PER_NODE: usize = 512;
const MAX_GROUPS: usize = 4;
const MARKER_GAP_S: f64 = 15.0;

/// Ground rings and sectors of raingen.physics.layout (for `ground_material_at`).
const RINGS: [(f32, f32); 4] = [(0.5, 1.5), (1.5, 4.0), (4.0, 10.0), (10.0, 30.0)];
const N_SECTORS: i64 = 8;

/// Material of the ground ring patch under listener-frame point (x, y).
pub fn ground_material_at(patches: &[Patch], x: f32, y: f32, default: &str) -> String {
    let r = x.hypot(y);
    let mut i = RINGS.len() - 1;
    for (k, (_, r2)) in RINGS.iter().enumerate() {
        if r < *r2 {
            i = k;
            break;
        }
    }
    let sw = 2.0 * std::f32::consts::PI / N_SECTORS as f32;
    let j = ((y.atan2(x) / sw).round() as i64).rem_euclid(N_SECTORS);
    let pid = format!("ground.r{i}.s{j}");
    patches.iter().find(|p| p.id == pid).map(|p| p.material.clone()).unwrap_or_else(|| default.to_string())
}

/// One drip point (leaf tip, eave lip, umbrella rim, cave ceiling).
#[derive(Clone, Debug)]
pub struct DripNode {
    key: u64,
    catchment_m2: f64,
    fall_height_m: f32,
    d_mm: f64,
    volume_mm3: f64,
    acc: Option<f64>,
    threshold: f64,
    x: f64,
    pub rate_hz: f64,
    pub chaos: f64,
    /// Excess flow beyond the dripping regime (a trickle), l/s.
    pub stream_l_s: f64,
    pub drops_total: u64,
}

impl DripNode {
    pub fn new(key: u64, catchment_m2: f32, edge_radius_mm: f32, fall_height_m: f32) -> DripNode {
        let d = tate_drop_diameter_mm(edge_radius_mm) as f64;
        DripNode {
            key,
            catchment_m2: (catchment_m2 as f64).max(1e-6),
            fall_height_m: fall_height_m.max(0.0),
            d_mm: d,
            volume_mm3: std::f64::consts::PI / 6.0 * d * d * d,
            acc: None,
            threshold: 0.0,
            x: 0.5,
            rate_hz: 0.0,
            chaos: 0.0,
            stream_l_s: 0.0,
            drops_total: 0,
        }
    }

    pub fn reset(&mut self) {
        self.acc = None;
        self.threshold = 0.0;
        self.x = 0.5;
        self.rate_hz = 0.0;
        self.chaos = 0.0;
        self.stream_l_s = 0.0;
        self.drops_total = 0;
    }

    /// Integrate one block; appends (offset, diameter mm, impact speed m/s) of released drops.
    pub fn step(&mut self, block: u64, n: usize, sr: f64, inflow_mm_h: f32, storage_mm: f32, out: &mut Vec<(u32, f32, f32)>) {
        let v = self.volume_mm3;
        let mut rng: Option<Rng> = None;
        let mut acc = match self.acc {
            Some(a) => a,
            None => {
                let mut r = Rng::from_key(self.key, block);
                let a = r.range(0.0, 0.9) * v;
                self.threshold = v;
                self.x = r.range(0.2, 0.8);
                rng = Some(r);
                a
            }
        };
        let dur = n as f64 / sr;
        let mut q = ((inflow_mm_h.max(0.0) as f64) / 3600.0 + (storage_mm.max(0.0) as f64) / TAU_STORE_S) * self.catchment_m2 * 1e6;
        let mut rate = q / v;
        if rate > MAX_DRIP_HZ {
            self.stream_l_s = (q - MAX_DRIP_HZ * v) * 1e-6;
            q = MAX_DRIP_HZ * v;
            rate = MAX_DRIP_HZ;
        } else {
            self.stream_l_s = 0.0;
        }
        self.rate_hz = rate;
        self.chaos = smoothstep(rate as f32, 2.0, 12.0) as f64;
        let r_map = 3.2 + 0.79 * self.chaos;
        if q <= 0.0 || n == 0 {
            self.acc = Some(acc);
            return;
        }
        let mut t = 0.0f64;
        let mut k = 0usize;
        while k < MAX_EVENTS_PER_NODE {
            let need = self.threshold - acc;
            let dt_fire = need.max(0.0) / q;
            if t + dt_fire >= dur {
                acc += q * (dur - t);
                break;
            }
            t += dt_fire;
            let r = rng.get_or_insert_with(|| Rng::from_key(self.key, block));
            let released = self.threshold * (1.0 - RESIDUAL);
            acc = self.threshold * RESIDUAL;
            let jitter = 1.0 + r.normal() * (0.015 + 0.08 * self.chaos);
            let diam = self.d_mm * (released / v).max(0.05).cbrt() * jitter.max(0.5);
            let v_imp = fall_speed_from_height(diam as f32, self.fall_height_m);
            let off = ((t * sr) as usize).min(n - 1) as u32;
            if out.len() < out.capacity() {
                out.push((off, diam as f32, v_imp));
            }
            self.drops_total += 1;
            self.x = (r_map * self.x * (1.0 - self.x)).clamp(1e-6, 1.0 - 1e-6);
            let thr = v * (1.0 + 0.03 * r.normal()) * (1.0 + 0.6 * self.chaos * (self.x - 0.5));
            self.threshold = thr.max(0.3 * v);
            k += 1;
        }
        self.acc = Some(acc);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Feed {
    Rain,
    Roof,
    Canopy,
    Storage,
}

/// One landing spot of a drip generator, with the nodes that feed it.
pub struct DripVoice {
    name: String,
    gen_name: String,
    kb: Arc<KernelBank>,
    n: usize,
    sr: f32,
    em: Emission,
    stem: Stem,
    nodes: Vec<DripNode>,
    feed: Feed,
    edge: Arc<MatData>,
    roof: Arc<MatData>,
    water_quality: f32,
    exposure: f32,
    depth_param: Option<f32>,
    roof_area: f32,
    target: Arc<MatData>,
    target_kind: MaterialKind,
    distance: f32,
    direct: DirectBus,
    voice: Option<SolidVoice>,
    air: Option<AirFilter>,
    click_pa: f32,
    hiss_pa: f32,
    key_sound: u64,
    marker: bool,
    last_marker_t: f64,
    events: Vec<(u32, f32, f32)>,
}

/// The drip voices of one drip edge / tree crown patch (one per landing spot).
pub fn build(bc: &mut BuildCtx, patch: &Patch, canopy: bool, roof_area_m2: f32) -> Vec<DripVoice> {
    let scene = bc.scene;
    let lh = scene.listener.height_m;
    let name = format!("drip/{}", patch.id);
    let feed = match patch.param_str("feed").map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s == "roof" => Feed::Roof,
        Some(s) if s == "canopy" => Feed::Canopy,
        Some(s) if s == "storage" => Feed::Storage,
        Some(s) if s == "rain" => Feed::Rain,
        _ => {
            if canopy {
                Feed::Canopy
            } else {
                Feed::Rain
            }
        }
    };
    let nonempty = |k: &str| patch.param_str(k).filter(|s| !s.is_empty()).map(|s| s.to_string());
    let edge_name = nonempty("edge_material")
        .or_else(|| nonempty("roof_material"))
        .unwrap_or_else(|| if canopy { "leaves".into() } else { "tile".into() });
    let edge = bc.mats.get(&edge_name);
    let roof_name = nonempty("roof_material").unwrap_or_else(|| edge.name.clone());
    let roof = bc.mats.get(&roof_name);
    let depth_param = patch.params.get("target_depth_mm").and_then(|v| v.as_f64()).map(|v| v as f32);

    let default_target_name = if canopy {
        let mut best: Option<(String, f32)> = None;
        if let Some(m) = scene.place.surfaces.as_object() {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            for k in keys {
                let v = m[k].as_f64().unwrap_or(0.0) as f32;
                if v > 0.0 && bc.mats.kind_of(k) != MaterialKind::Water && best.as_ref().map(|b| v > b.1).unwrap_or(true) {
                    best = Some((k.clone(), v));
                }
            }
        }
        best.map(|b| b.0).unwrap_or_else(|| "leaves".into())
    } else {
        "puddle".into()
    };
    let default_target = nonempty("target_material").or_else(|| nonempty("target")).unwrap_or(default_target_name);
    let default_target = bc.mats.canonical(&default_target);

    let mut rng = bc.rngf.get(&format!("drip/{}/layout", patch.id), 0);
    let [x, y, z] = patch.position;
    let (n_nodes, catch, edge_r, fall, spots): (usize, f32, f32, f32, Vec<([f32; 3], f32)>) = if canopy {
        let n_nodes = patch.param_f32("n_points", 8.0).max(0.0) as usize;
        let catch = patch.param_f32("catchment_m2", 0.01);
        let edge_r = patch.param_f32("edge_radius_mm", 2.2);
        let fall = patch.param_f32("fall_height_m", (z + lh).max(1.0));
        let radius = patch.param_f32("canopy_radius_m", 2.5).max(0.3);
        let n_groups = n_nodes.clamp(1, MAX_GROUPS);
        let phi0 = rng.range(0.0, 2.0 * std::f64::consts::PI) as f32;
        let mut spots = Vec::with_capacity(n_groups);
        for k in 0..n_groups {
            let a = phi0 + 2.0 * std::f32::consts::PI * k as f32 / n_groups as f32;
            let rr = if n_groups > 1 { 0.6 * radius } else { 0.0 };
            spots.push(([x + rr * a.cos(), y + rr * a.sin(), -lh], radius));
        }
        (n_nodes, catch, edge_r, fall, spots)
    } else {
        let n_nodes = patch.param_f32("n_points", 1.0).max(0.0) as usize;
        let catch = patch.param_f32("catchment_m2", 0.08);
        let edge_r = patch.param_f32("edge_radius_mm", 2.0);
        let fall = if patch.params.get("fall_height_m").is_some() {
            patch.param_f32("fall_height_m", 1.0)
        } else {
            let ez = match patch.params.get("edge_position") {
                Some(serde_json::Value::Array(a)) if a.len() >= 3 => a[2].as_f64().unwrap_or((z + 2.5) as f64) as f32,
                _ => z + 2.5,
            };
            (ez - z).max(0.3)
        };
        (n_nodes, catch, edge_r, fall, vec![([x, y, z], patch.param_f32("length_m", 0.2).max(0.05))])
    };
    let mut nodes = Vec::new();
    for i in 0..n_nodes.max(1) {
        let c = catch * rng.range(0.6, 1.4) as f32;
        let e = edge_r * rng.range(0.85, 1.15) as f32;
        let f = (fall * rng.range(0.9, 1.1) as f32).max(0.05);
        nodes.push(DripNode::new(bc.rngf.key(&format!("drip/{}/{}", patch.id, i)), c, e, f));
    }
    let n_groups = spots.len().max(1);
    let mut groups: Vec<Vec<DripNode>> = (0..n_groups).map(|_| Vec::new()).collect();
    for (i, nd) in nodes.into_iter().enumerate() {
        groups[i % n_groups].push(nd);
    }

    let sr = bc.sr as f32;
    let n = bc.block;
    let kb = bc.kb.clone();
    let click_pa = db_to_pa(bc.mats.level_of("puddle"));
    let ola_cap = n + (0.08 * sr) as usize + kb.max_kernel_len() + 64;
    let mut out = Vec::with_capacity(n_groups);
    for (k, ((pos, ext), nodes)) in spots.into_iter().zip(groups).enumerate() {
        let tgt_name = if canopy { ground_material_at(bc.patches, pos[0], pos[1], &default_target) } else { default_target.clone() };
        let target = bc.mats.get(&tgt_name);
        let target_kind = if target.kind == MaterialKind::Water { MaterialKind::Water } else { target.kind };
        let [px, py, pz] = pos;
        let distance = (px * px + py * py + pz * pz).sqrt().max(0.2);
        let azimuth = py.atan2(px);
        let elevation = pz.atan2(px.hypot(py));
        let spread = (2.0 * (0.5 * ext / distance).atan() / std::f32::consts::PI).clamp(0.0, 1.0);
        let stem = if patch.interior { Stem::Interior } else { Stem::Aftermath };
        let voice = if target_kind == MaterialKind::Water {
            None
        } else {
            Some(SolidVoice::new(&target.modes, &target.rad, target.cal, sr, n, ola_cap))
        };
        let node_cap = nodes.len().max(1) * MAX_EVENTS_PER_NODE;
        out.push(DripVoice {
            name: if n_groups == 1 { name.clone() } else { format!("{name}/{k}") },
            gen_name: name.clone(),
            kb: kb.clone(),
            n,
            sr,
            em: Emission::point(azimuth, elevation, distance, spread, stem),
            stem,
            nodes,
            feed,
            edge: edge.clone(),
            roof: roof.clone(),
            water_quality: bc.water_quality,
            exposure: patch.exposure.max(0.0),
            depth_param,
            roof_area: roof_area_m2,
            hiss_pa: db_to_pa(target.level_db + FILM_HISS_REL_DB),
            target,
            target_kind,
            distance,
            direct: DirectBus::new(ola_cap, 512, sr as f64),
            voice,
            air: if distance >= AIR_MIN_DIST_M { Some(AirFilter::new(distance, sr, n)) } else { None },
            click_pa,
            key_sound: bc.rngf.key(&format!("drip/{}/sound/{}", patch.id, k)),
            marker: k == 0,
            last_marker_t: -1e9,
            events: Vec::with_capacity(node_cap),
        });
    }
    out
}

impl DripVoice {
    /// (inflow mm/h on the catchment, upstream storage mm) for this block (`_feed`).
    fn feed(&self, w: &Weather) -> (f32, f32) {
        let liquid = liquid_share(w.precip_type);
        let thaw = smoothstep(w.temperature_c, -2.0, 0.0);
        let rain = w.rain_rate_mm_h.max(0.0) * liquid;
        match self.feed {
            Feed::Canopy => (0.25 * rain * self.exposure * thaw, 0.1 * w.canopy_storage_mm.max(0.0) * thaw),
            Feed::Roof => {
                let q = w.roof_flow_l_s.max(0.0);
                let inflow = if self.roof_area > 0.0 && q > 0.0 { q * 3600.0 / self.roof_area } else { rain * self.exposure };
                (inflow * thaw, 0.4 * wetness(w, &self.roof).clamp(0.0, 1.0) * thaw)
            }
            Feed::Storage => {
                let store = w.wetness.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
                (0.02 * rain * thaw, 0.3 * store.clamp(0.0, 1.0) * thaw)
            }
            Feed::Rain => (rain * self.exposure * thaw, 0.4 * wetness(w, &self.edge).clamp(0.0, 1.0) * thaw),
        }
    }

    fn target_depth(&self, w: &Weather) -> f32 {
        if self.target.name == "water" {
            return 1000.0;
        }
        if let Some(d) = self.depth_param {
            return d.max(0.5);
        }
        let d = puddle_named(w, &self.target.name).unwrap_or(15.0);
        (if d == 0.0 { 15.0 } else { d }).max(3.0)
    }

    /// Landing sounds of this block's drops; returns whether any drop 'plinked'.
    fn synth(&mut self, rng: &mut Rng, w: &Weather) -> bool {
        let kb = self.kb.clone();
        let sr = self.sr as f64;
        let mut plinked = false;
        if self.target_kind == MaterialKind::Water {
            let depth = self.target_depth(w);
            let wp = WaterParams { click_pa: self.click_pa, depth_mm: depth, wind_mps: 0.0, water_quality: self.water_quality, p_scale: 1.0 };
            let fd = smoothstep(depth, 1.5, 8.0) * self.water_quality;
            for i in 0..self.events.len() {
                let (off, d, u) = self.events[i];
                let off = off as usize;
                if regular_entrainment(d, u) {
                    plinked = true;
                    let radius = d as f64 * 1e-3 * rng.range(0.3, 0.42);
                    let eps = rng.range(0.08, 0.15);
                    let xi = rng.range(0.12, 0.3);
                    let delay = (rng.range(0.008, 0.02) * sr) as usize;
                    if fd > 0.0 {
                        self.direct.bubbles.spawn(off + delay, bubble_pa_per_m() as f64 * eps * fd as f64, radius, xi, 0.0);
                    }
                    self.direct.ola.add(off, kb.click(click_bin(u)), self.click_pa * d / D_REF_MM);
                } else {
                    let ci = if d >= 3.5 { 4 } else { 3 };
                    water_event(&kb, rng, &mut self.direct, off, d, u, ci, 1.0, &wp);
                }
            }
            return plinked;
        }

        // solid / porous / vegetation / membrane target; the puddled share becomes water impacts
        let depth = puddle_mm(w, &self.target);
        if depth > 0.2 && matches!(self.target_kind, MaterialKind::Solid | MaterialKind::Porous) {
            let cover = (depth / 8.0).min(1.0).powf(0.8);
            let wp =
                WaterParams { click_pa: self.click_pa, depth_mm: depth.max(2.0), wind_mps: 0.0, water_quality: self.water_quality, p_scale: 1.0 };
            let mut keep = 0usize;
            for i in 0..self.events.len() {
                let e = self.events[i];
                if rng.uniform() < cover as f64 {
                    let ci = if e.1 >= 3.5 { 4 } else { 3 };
                    water_event(&kb, rng, &mut self.direct, e.0 as usize, e.1, e.2, ci, 1.0, &wp);
                } else {
                    self.events[keep] = e;
                    keep += 1;
                }
            }
            self.events.truncate(keep);
        }
        if self.events.is_empty() {
            return false;
        }
        let wet = wetness(w, &self.target).clamp(0.0, 1.0);
        let splashy = self.target.splash > 0.0 && self.target.hardness >= 0.45 && wet < 0.9;
        let hiss = wet > 0.05 && matches!(self.target_kind, MaterialKind::Solid | MaterialKind::Porous);
        let voice = match self.voice.as_mut() {
            Some(v) => v,
            None => return false,
        };
        let fr = f_ref();
        for i in 0..self.events.len() {
            let (off, d, u) = self.events[i];
            let off = off as usize;
            let fp = force_peak(d, u);
            let force = fp * (0.15 * rng.normal()).exp() as f32 * (1.0 - 0.35 * wet);
            let tau = d as f64 * 1e-3 / (u as f64).max(0.05) * (1.0 + 0.6 * wet as f64);
            voice.add_force(&kb, off, force, tau, false);
            if splashy {
                let reps = rng.int_range(3, 8);
                for _ in 0..reps {
                    let delay = (rng.range(0.005f64.ln(), 0.05f64.ln()).exp() * sr) as usize;
                    let amp = force * rng.range(0.005f64.ln(), 0.03f64.ln()).exp() as f32 * (1.0 - wet);
                    let ts = rng.range(5e-5f64.ln(), 1.5e-4f64.ln()).exp();
                    voice.add_force(&kb, off + delay, amp, ts, false);
                }
            }
            if hiss {
                let amp = self.hiss_pa * wet * (fp / fr).powf(0.75);
                let dur = (0.0025 * (d / 2.0).sqrt()).clamp(0.0015, 0.008);
                let v = rng.below(SPLASH_VARIANTS as u64) as usize;
                self.direct.ola.add(off, kb.splash(dur_bin(dur), v), amp);
            }
        }
        false
    }

    fn busy(&mut self) -> bool {
        self.direct.active()
            || self.voice.as_mut().map(|v| v.active()).unwrap_or(false)
            || self.air.as_ref().map(|a| !a.quiet()).unwrap_or(false)
    }
}

impl Voice for DripVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n.min(self.n).min(out.len());
        if n == 0 {
            return Emission::silent(self.stem);
        }
        let w = ctx.weather;
        let sr = self.sr as f64;
        let (inflow, storage) = self.feed(w);
        self.events.clear();
        for nd in self.nodes.iter_mut() {
            nd.step(ctx.index, n, sr, inflow, storage, &mut self.events);
        }
        if !self.events.is_empty() {
            for e in self.events.iter_mut() {
                e.1 = e.1.clamp(0.5, 8.0);
                e.2 = e.2.max(0.05);
            }
            let first_off = self.events.iter().map(|e| e.0).min().unwrap_or(0);
            let first_d = self.events[0].1;
            let mut rng = Rng::from_key(self.key_sound, ctx.index);
            let plinked = self.synth(&mut rng, w);
            if self.marker && ctx.t0 - self.last_marker_t >= MARKER_GAP_S {
                ctx.marker(
                    ctx.t0 + first_off as f64 / sr,
                    "drip",
                    format!(
                        "{{\"source\": \"{}\", \"sound\": \"{}\", \"diameter_mm\": {}, \"target\": \"{}\"}}",
                        self.gen_name,
                        if plinked { "plink" } else { "splat" },
                        first_d,
                        self.target.name
                    ),
                );
                self.last_marker_t = ctx.t0;
            }
        }
        if !self.busy() {
            return Emission::silent(self.stem);
        }
        let y = &mut out[..n];
        if let Some(v) = self.voice.as_mut() {
            if v.active() {
                v.render(y);
            }
        }
        self.direct.render(y);
        let g = spreading_gain(self.distance);
        for v in y.iter_mut() {
            *v *= g;
        }
        if let Some(a) = self.air.as_mut() {
            a.process(y, w.temperature_c, w.humidity_pct);
        }
        self.em
    }

    fn reset(&mut self) {
        for nd in self.nodes.iter_mut() {
            nd.reset();
        }
        if let Some(v) = self.voice.as_mut() {
            v.reset();
        }
        self.direct.clear();
        if let Some(a) = self.air.as_mut() {
            a.reset();
        }
        self.last_marker_t = -1e9;
        self.events.clear();
    }
}
