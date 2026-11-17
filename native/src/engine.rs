//! The voice graph and the block renderer.
//!
//! Per block (n frames):
//!   1. drain control commands (weather, new graph, output, gain, ...) - lock-free channel;
//!   2. render every voice IN PARALLEL (rayon) into its own preallocated buffer;
//!   3. encode serially, in graph order, into the per-stem SH buses (deterministic sum);
//!   4. run the per-stem decorrelators (parallel over stems), stem gains, bus processors
//!      (shelter transfer / room);
//!   5. sum stems, decode, apply the output gain (ramped), run the safety limiter.
//! Voices come from `rain::build_voices` and `env::build_voices`; bus processors from
//! `env::build_bus_processors`. A voice that panics is disabled (like a Python generator that
//! raises) and the rest of the graph keeps playing.
//!
//! Live listener: every voice tied to a world patch keeps that patch's WORLD position; the
//! pose set by `ListenerPose::set` (lock-free, any thread) is smoothed per block (~50 ms) and
//! moves those voices (direction, 1/r gain, air-absorption shelf, all ramped per sample),
//! while turning the head rotates the whole SH field once after the bus processors. Ground
//! rings, the listener's own shelter and listener-relative beds move with the listener.
//! A new graph (`Command::Graph`) is diffed against the running one by a stable per-voice key:
//! unchanged voices keep running with their state, only added / removed ones crossfade.

use crate::dsp::enable_ftz;
use crate::limiter::Limiter;
use crate::rng::RngFactory;
use crate::rng::{combine, hash_str};
use crate::spatial::{listener_rotation, Decoder, Encoder, SourceState, StemBuses};
use crate::types::{
    Emission, Marker, MaterialTable, Patch, Scene, Shelter, Weather, N_AMBI, N_STEMS,
};
use crossbeam_channel::Receiver;
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Crossfade of voices added / removed by an incremental scene update (s).
pub const VOICE_FADE_S: f32 = 0.15;
/// Smoothing time constant of the live listener pose (s).
pub const POSE_TAU_S: f64 = 0.05;

// ---------------------------------------------------------------------------
// contracts for the porting agents
// ---------------------------------------------------------------------------
/// Everything a voice factory needs besides the scene and the patches.
#[derive(Clone, Copy, Debug)]
pub struct BuildCfg {
    pub sr: u32,
    pub block: usize,
    pub seed: u64,
    /// RngFactory(seed): derive per-voice stream keys with `rng.key("layer/name")`.
    pub rng: RngFactory,
}

/// Lock-free, bounded marker queue shared by all voices of a block.
pub struct MarkerSink {
    q: ArrayQueue<Marker>,
}

impl MarkerSink {
    pub fn new(capacity: usize) -> MarkerSink {
        MarkerSink { q: ArrayQueue::new(capacity.max(16)) }
    }

    /// Record an event at absolute time `t` (s). Drops the event if the queue is full.
    pub fn push(&self, t: f64, kind: &'static str, data: String) {
        let _ = self.q.push(Marker { t, kind, data });
    }

    pub fn drain_into(&self, out: &mut Vec<Marker>) {
        while let Some(m) = self.q.pop() {
            out.push(m);
        }
    }
}

/// A manual lightning strike requested with `Engine.strike()` (delivered to the thunder voice
/// through `BlockCtx::strikes` on the next block).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strike {
    pub id: u64,
    pub distance_km: f32,
    /// Compass bearing (deg true) from the listener.
    pub bearing_deg: f32,
    /// Stroke intensity 0..1.
    pub intensity: f32,
}

/// The live listener pose: feet position (world metres, as Scene.listener.position), heading
/// (deg true) and pitch (deg, up positive). Written by control threads without locks; the
/// renderer reads it once per block (a torn read is harmless: the pose is smoothed).
pub struct ListenerPose {
    vals: [AtomicU32; 5],
    seq: AtomicU64,
}

impl Default for ListenerPose {
    fn default() -> Self {
        ListenerPose { vals: Default::default(), seq: AtomicU64::new(0) }
    }
}

impl ListenerPose {
    pub fn set(&self, v: [f32; 5]) {
        for (a, x) in self.vals.iter().zip(v) {
            a.store(x.to_bits(), Ordering::Relaxed);
        }
        self.seq.fetch_add(1, Ordering::Release);
    }

    /// The last pose set, or None while nobody has set one.
    pub fn get(&self) -> Option<[f32; 5]> {
        if self.seq.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut v = [0.0f32; 5];
        for (x, a) in v.iter_mut().zip(&self.vals) {
            *x = f32::from_bits(a.load(Ordering::Relaxed));
        }
        Some(v)
    }

    /// Forget the pose (the scene's listener is used again).
    pub fn clear(&self) {
        self.seq.store(0, Ordering::Release);
    }
}

/// The listener pose a graph was built for (Scene.listener): the frame of its voices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefPose {
    pub pos: [f32; 3],
    pub heading_deg: f32,
    pub height_m: f32,
}

impl Default for RefPose {
    fn default() -> Self {
        RefPose { pos: [0.0; 3], heading_deg: 0.0, height_m: 1.7 }
    }
}

impl RefPose {
    pub fn of(scene: &Scene) -> RefPose {
        let l = &scene.listener;
        RefPose { pos: l.position, heading_deg: l.heading_deg, height_m: l.height_m }
    }

    pub fn ears(&self) -> [f32; 3] {
        [self.pos[0], self.pos[1], self.pos[2] + self.height_m]
    }

    /// World (x east, y north, z up) offset -> this frame (+x forward, +y left, +z up).
    pub fn to_frame(&self, d: [f32; 3]) -> [f32; 3] {
        let (s, c) = self.heading_deg.to_radians().sin_cos();
        [d[0] * s + d[1] * c, -d[0] * c + d[1] * s, d[2]]
    }

    /// Inverse of `to_frame`.
    pub fn to_world(&self, p: [f32; 3]) -> [f32; 3] {
        let (s, c) = self.heading_deg.to_radians().sin_cos();
        [p[0] * s - p[1] * c, p[0] * c + p[1] * s, p[2]]
    }

    fn hash(&self) -> u64 {
        let mut h = hash_str("refpose");
        for v in [self.pos[0], self.pos[1], self.pos[2], self.heading_deg, self.height_m] {
            h = combine(h, v.to_bits() as u64);
        }
        h
    }
}

/// Per-block listener motion shared by every voice: the smoothed pose relative to the graph's
/// reference pose.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoseCtx {
    /// Ears displacement in the reference frame (m).
    pub delta: [f32; 3],
    /// Air absorption at 8 kHz (dB/m) for the distance changes.
    pub air_db_per_m: f32,
    /// Head turn (rad, clockwise positive) and pitch (rad) relative to the reference pose.
    pub dheading: f32,
    pub pitch: f32,
}

/// Per-block context handed to every voice (the Rust BlockContext).
pub struct BlockCtx<'a> {
    pub sr: u32,
    pub sr_f: f32,
    /// Frames in this block.
    pub n: usize,
    /// Block index from render start.
    pub index: u64,
    /// Block start time (s).
    pub t0: f64,
    pub weather: &'a Weather,
    /// True on the first block after `Engine.update()` delivered a new WeatherState.
    pub weather_changed: bool,
    /// RngFactory(scene seed): `ctx.rng.stream(key, ctx.index)` is the voice's stream for this block.
    pub rng: &'a RngFactory,
    pub listener_height_m: f32,
    pub shelter: Shelter,
    pub window_open: f32,
    pub scene: &'a Scene,
    pub materials: &'a MaterialTable,
    pub markers: &'a MarkerSink,
    /// Manual strikes to start in this block (thunder voice).
    pub strikes: &'a [Strike],
}

impl<'a> BlockCtx<'a> {
    /// Absolute time (s) of frame i of this block.
    #[inline]
    pub fn t_at(&self, i: usize) -> f64 {
        self.t0 + i as f64 / self.sr as f64
    }

    /// Emit a marker (lightning, thunder, gust, drip, ...). `data` is a JSON object text.
    #[inline]
    pub fn marker(&self, t: f64, kind: &'static str, data: String) {
        self.markers.push(t, kind, data);
    }

    /// Block duration (s).
    #[inline]
    pub fn dt(&self) -> f32 {
        self.n as f32 / self.sr_f
    }
}

/// A sound source rendered once per block.
///
/// `out` holds `channels() * ctx.n` samples (channel-major), ZEROED by the engine before the
/// call; write or add the block's pressure signal (Pa at the listener, level and propagation
/// already applied) and return its spatial metadata. Mono voices (`channels() == 1`) are
/// encoded at the returned azimuth / elevation / spread; ambisonic voices
/// (`channels() == N_AMBI`) write ACN/SN3D order-3 signals and only `stem` / `active` of the
/// returned Emission are read. Return `Emission { active: false, .. }` (e.g.
/// `Emission::silent(stem)`) for a silent block. Must not allocate, lock or block: every
/// buffer is created in the factory. Draw randomness only from `ctx.rng.stream(key, ctx.index)`
/// with a key fixed at construction, so output is deterministic and independent of other voices.
pub trait Voice: Send {
    fn name(&self) -> &str;
    /// 1 (mono, encoded by the engine) or N_AMBI (already ambisonic). Fixed for the voice's life.
    fn channels(&self) -> usize {
        1
    }
    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission;
    /// Return to the initial state (offline re-render from t = 0).
    fn reset(&mut self) {}
}

pub type BoxedVoice = Box<dyn Voice + Send>;

/// Processes the encoded per-stem SH buses after encoding and before decoding (shelter
/// transfer, room reverberation into the interior stem). Same real-time rules as `Voice`.
pub trait BusProcessor: Send {
    fn name(&self) -> &str;
    fn process(&mut self, ctx: &BlockCtx, buses: &mut StemBuses);
    fn reset(&mut self) {}
}

pub type BoxedProcessor = Box<dyn BusProcessor + Send>;

// ---------------------------------------------------------------------------
// background jobs (off the render pool)
// ---------------------------------------------------------------------------
type Job = Box<dyn FnOnce() + Send + 'static>;

/// Run `job` on raingen's background workers (two threads, started on first use), never on
/// the render pool: a long job picked up by a pool thread waiting in a join would stall the
/// block it belongs to. Voices use it for whole-call synthesis they need a few blocks later.
pub fn background<F: FnOnce() + Send + 'static>(job: F) {
    static TX: std::sync::OnceLock<crossbeam_channel::Sender<Job>> = std::sync::OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = crossbeam_channel::unbounded::<Job>();
        for i in 0..2 {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("raingen-bg-{i}")).spawn(move || {
                enable_ftz();
                while let Ok(job) = rx.recv() {
                    let _ = catch_unwind(AssertUnwindSafe(job));
                }
            });
        }
        tx
    });
    if let Err(e) = tx.send(Box::new(job)) {
        (e.0)(); // no worker could be started: run it here rather than lose it
    }
}

// ---------------------------------------------------------------------------
// voice graph
// ---------------------------------------------------------------------------
/// Where a world-fixed voice sits: its patch's world position in the graph's reference frame
/// and the geometry its level and air absorption were built for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    /// Patch position now, reference frame (moved objects update it in place).
    pub w_ref: [f32; 3],
    /// Patch position the voice was built for, reference frame.
    pub p_build: [f32; 3],
    pub d_ref: f32,
    /// Patch radius sqrt(area / pi) (near-field limit of the 1/r law).
    pub rad: f32,
    /// sqrt(area * exposure) when built, and the gain toward the current patch's value.
    pub amp_build: f32,
    pub gain_mul: f32,
}

impl Anchor {
    fn r_eff(&self, d: f32) -> f32 {
        (d * d + 0.5 * self.rad * self.rad).sqrt().max(0.3)
    }
}

#[inline]
fn fade_gain(u: f32) -> f32 {
    if u >= 1.0 {
        1.0
    } else if u <= 0.0 {
        0.0
    } else {
        (std::f32::consts::FRAC_PI_2 * u).sin()
    }
}

pub struct Slot {
    pub voice: BoxedVoice,
    channels: usize,
    buf: Vec<f32>,
    meta: Emission,
    enc: SourceState,
    dead: bool,
    /// Crossfade position 0..1 (gain sin(pi/2 u)) and direction (+1 in, -1 out, 0 steady).
    fade_u: f32,
    fade_dir: i8,
    /// Stable identity for incremental scene updates.
    pub key: u64,
    pub anchor: Option<Anchor>,
    /// Listener-motion gain and air-absorption shelf gain at the end of the last block, and
    /// the shelf's low-pass state.
    pose_g: f32,
    hf_g: f32,
    lp: f32,
    /// Smoothed render time of this voice (ms per block).
    pub cpu_ms: f64,
    /// Slowest block of this voice (ms).
    pub max_ms: f64,
}

impl Slot {
    fn new(voice: BoxedVoice, n: usize) -> Slot {
        let ch = if voice.channels() >= N_AMBI { N_AMBI } else { 1 };
        let key = hash_str(voice.name());
        Slot {
            voice,
            channels: ch,
            buf: vec![0.0; ch * n],
            meta: Emission::SILENT,
            enc: SourceState::default(),
            dead: false,
            fade_u: 1.0,
            fade_dir: 0,
            key,
            anchor: None,
            pose_g: 1.0,
            hf_g: 1.0,
            lp: 0.0,
            cpu_ms: 0.0,
            max_ms: 0.0,
        }
    }

    fn render(&mut self, ctx: &BlockCtx, pose: &PoseCtx) {
        let t = Instant::now();
        self.render_inner(ctx);
        self.apply_pose(pose, ctx.sr_f);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.cpu_ms = if self.cpu_ms == 0.0 { ms } else { 0.95 * self.cpu_ms + 0.05 * ms };
        self.max_ms = self.max_ms.max(ms);
    }

    fn render_inner(&mut self, ctx: &BlockCtx) {
        if self.dead {
            self.meta = Emission::SILENT;
            return;
        }
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        let voice = &mut self.voice;
        let buf = &mut self.buf;
        match catch_unwind(AssertUnwindSafe(|| voice.render(ctx, buf))) {
            Ok(m) => {
                self.meta = m;
                if m.active && !self.buf.iter().all(|v| v.is_finite()) {
                    self.buf.iter_mut().for_each(|v| *v = 0.0);
                    self.meta.active = false;
                }
            }
            Err(_) => {
                self.dead = true;
                self.meta = Emission::SILENT;
                self.buf.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }

    /// Move a world-fixed mono voice to where it is heard from the live listener position:
    /// direction from its world position, 1/r gain and the air-absorption difference against
    /// the distance it was built for (gains ramped per sample, so walking never zippers).
    fn apply_pose(&mut self, pose: &PoseCtx, sr: f32) {
        let a = match self.anchor {
            Some(a) if self.channels == 1 && self.meta.active => a,
            _ => return,
        };
        let o = [
            a.w_ref[0] - pose.delta[0] - a.p_build[0],
            a.w_ref[1] - pose.delta[1] - a.p_build[1],
            a.w_ref[2] - pose.delta[2] - a.p_build[2],
        ];
        let still = o[0].abs() < 1e-4 && o[1].abs() < 1e-4 && o[2].abs() < 1e-4 && a.gain_mul == 1.0;
        if still && self.pose_g == 1.0 && self.hf_g == 1.0 {
            return; // bit-exact at the reference pose
        }
        let m = &mut self.meta;
        let (se, ce) = m.elevation.sin_cos();
        let (sa, ca) = m.azimuth.sin_cos();
        let p = [a.d_ref * ce * ca + o[0], a.d_ref * ce * sa + o[1], a.d_ref * se + o[2]];
        let pn = [a.p_build[0] + o[0], a.p_build[1] + o[1], a.p_build[2] + o[2]];
        let d_new = (pn[0] * pn[0] + pn[1] * pn[1] + pn[2] * pn[2]).sqrt().max(0.05);
        if p[0].abs() + p[1].abs() + p[2].abs() > 1e-6 {
            m.azimuth = p[1].atan2(p[0]);
            m.elevation = p[2].atan2(p[0].hypot(p[1]));
        }
        let ratio = a.r_eff(a.d_ref) / a.r_eff(d_new);
        m.spread = (m.spread * ratio.min(4.0)).clamp(0.0, 1.0);
        m.distance = d_new;
        let g1 = if still { 1.0 } else { (ratio * a.gain_mul).clamp(0.0, 4.0) };
        let hf_db = (-pose.air_db_per_m * (d_new - a.d_ref)).clamp(-30.0, 4.0);
        let h1 = if still { 1.0 } else { 10f32.powf(hf_db / 20.0) };
        let (g0, h0) = (self.pose_g, self.hf_g);
        let k = 1.0 - (-std::f32::consts::TAU * 4000.0 / sr).exp();
        let n = self.buf.len();
        let inv = 1.0 / n.max(1) as f32;
        let mut lp = self.lp;
        for (i, v) in self.buf.iter_mut().enumerate() {
            let t = (i + 1) as f32 * inv;
            let x = *v;
            lp += k * (x - lp);
            let h = h0 + (h1 - h0) * t;
            let g = g0 + (g1 - g0) * t;
            *v = g * (lp + h * (x - lp));
        }
        self.lp = if lp.is_finite() && lp.abs() > 1e-30 { lp } else { 0.0 };
        self.pose_g = g1;
        self.hf_g = h1;
    }

    fn reset_live(&mut self) {
        self.enc = SourceState::default();
        self.dead = false;
        self.fade_u = 1.0;
        self.fade_dir = 0;
        self.pose_g = 1.0;
        self.hf_g = 1.0;
        self.lp = 0.0;
    }
}

pub struct Graph {
    pub rng: RngFactory,
    pub scene: Arc<Scene>,
    pub patches: Arc<Vec<Patch>>,
    pub slots: Vec<Slot>,
    pub procs: Vec<BoxedProcessor>,
    /// Identity of the bus processors (place / shelter / room): kept when unchanged.
    pub procs_key: u64,
    pub ref_pose: RefPose,
}

impl Graph {
    pub fn empty() -> Graph {
        Graph {
            rng: RngFactory::new(0),
            scene: Arc::new(Scene::default()),
            patches: Arc::new(Vec::new()),
            slots: Vec::new(),
            procs: Vec::new(),
            procs_key: 0,
            ref_pose: RefPose::default(),
        }
    }
}

fn discard_procs(p: Vec<BoxedProcessor>) {
    if !p.is_empty() {
        drop_later(p);
    }
}

/// Drop `x` on a background worker (voices can own large buffers; never free them on the
/// render thread).
fn drop_later<T: Send + 'static>(x: T) {
    background(move || drop(x));
}

/// Numbers of the scene's weather are live parameters (they glide through `Weather`), so they
/// are not part of a voice's identity; strings and flags are.
fn strip_numbers(v: &mut Value) {
    match v {
        Value::Number(_) => *v = Value::Null,
        Value::Array(a) => a.iter_mut().for_each(strip_numbers),
        Value::Object(m) => m.values_mut().for_each(strip_numbers),
        _ => {}
    }
}

/// Hash of everything in the scene that shapes voices besides the patches: precipitation
/// kind and texture settings, place (without its objects and live fields), listener height,
/// mode, seed, quality. Weather numbers and the objects are not in it.
pub fn scene_struct_hash(scene: &Scene) -> u64 {
    let mut v = scene.raw.clone();
    if let Value::Object(m) = &mut v {
        m.remove("name");
        m.remove("overrides");
        if let Some(w) = m.get_mut("weather") {
            strip_numbers(w);
        }
        if let Some(Value::Object(p)) = m.get_mut("place") {
            for k in ["objects", "window_open", "aftermath_tau_min"] {
                p.remove(k);
            }
        }
        if let Some(l) = m.get_mut("listener") {
            let h = l.get("height_m").cloned().unwrap_or(Value::Null);
            *l = Value::Object([("height_m".to_string(), h)].into_iter().collect());
        }
        if let Some(r) = m.get_mut("render") {
            let q = r.get("quality").cloned().unwrap_or(Value::Null);
            *r = Value::Object([("quality".to_string(), q)].into_iter().collect());
        }
        if let Some(Value::Object(vb)) = m.get_mut("vibe") {
            for k in ["intensity", "closeness", "wildness", "volume_db"] {
                vb.remove(k);
            }
        }
    }
    hash_str(&serde_json::to_string(&v).unwrap_or_default())
}

/// Identity of a patch for its voices: everything but where it is (moving keeps the voice).
fn patch_hash(p: &Patch) -> u64 {
    let mut params = p.params.clone();
    if let Value::Object(m) = &mut params {
        m.remove("world_position");
        m.remove("upwind_m");
    }
    let mut h = combine(hash_str(&p.id), hash_str(p.kind.name()));
    h = combine(h, hash_str(&p.material));
    h = combine(h, p.interior as u64);
    combine(h, hash_str(&serde_json::to_string(&params).unwrap_or_default()))
}

/// The patch a voice renders, from the voice naming convention (rain/<id>, drip/<id>/<k>,
/// gutter/<id>, hail/<id>, snow/<id>, vegetation.<id>, aeolian.<id>).
fn patch_of<'p>(name: &str, by_id: &HashMap<&str, &'p Patch>) -> Option<&'p Patch> {
    for pre in ["rain/", "hail/", "snow/", "gutter/", "vegetation.", "aeolian."] {
        if let Some(id) = name.strip_prefix(pre) {
            return by_id.get(id).copied();
        }
    }
    if let Some(rest) = name.strip_prefix("drip/") {
        if let Some(p) = by_id.get(rest) {
            return Some(p);
        }
        if let Some(i) = rest.rfind('/') {
            return by_id.get(&rest[..i]).copied();
        }
    }
    None
}

/// Ground rings, the listener's own shelter and interior surfaces travel with the listener.
pub fn listener_attached(p: &Patch) -> bool {
    p.interior
        || p.param_bool("ground", false)
        || p.param_bool("listener_attached", false)
        || p.id.starts_with("shelter.")
}

fn anchor_for(p: &Patch, rp: &RefPose) -> Option<Anchor> {
    if listener_attached(p) {
        return None;
    }
    let world = match p.params.get("world_position") {
        Some(Value::Array(a)) if a.len() >= 3 && a.iter().all(|x| x.as_f64().map(|v| v.is_finite()).unwrap_or(false)) => {
            [a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32, a[2].as_f64().unwrap() as f32]
        }
        _ => {
            let w = rp.to_world(p.position);
            let e = rp.ears();
            [w[0] + e[0], w[1] + e[1], w[2] + e[2]]
        }
    };
    let e = rp.ears();
    let w_ref = rp.to_frame([world[0] - e[0], world[1] - e[1], world[2] - e[2]]);
    Some(Anchor {
        w_ref,
        p_build: p.position,
        d_ref: p.distance(),
        rad: (p.area_m2.max(0.0) / std::f32::consts::PI).sqrt().min(20.0),
        amp_build: (p.area_m2.max(0.0) * p.exposure.max(0.0)).sqrt(),
        gain_mul: 1.0,
    })
}

/// Build the voice graph of a scene: rain voices first, then environment voices (fixed order).
pub fn build_graph(scene: Scene, patches: Vec<Patch>, materials: &MaterialTable, cfg: &BuildCfg) -> Graph {
    let mut voices: Vec<BoxedVoice> = crate::rain::build_voices(&scene, &patches, materials, cfg);
    voices.extend(crate::env::build_voices(&scene, &patches, materials, cfg));
    let mut procs: Vec<BoxedProcessor> = Vec::new();
    // place acoustics (early reflections + outdoor diffuse tail) of the exterior rain field,
    // first, so the shelter transfer that follows also filters the reflections
    let space = crate::spatial::space::SpaceProcessor::new(&scene, cfg);
    if space.needed() {
        procs.push(Box::new(space));
    }
    procs.extend(crate::env::build_bus_processors(&scene, &patches, materials, cfg));
    let rp = RefPose::of(&scene);
    let base = combine(scene_struct_hash(&scene), rp.hash());
    let base = combine(base, cfg.seed);
    let by_id: HashMap<&str, &Patch> = patches.iter().map(|p| (p.id.as_str(), p)).collect();
    let mut seen: HashMap<String, u64> = HashMap::new();
    let slots = voices
        .into_iter()
        .map(|v| {
            let mut s = Slot::new(v, cfg.block);
            let name = s.voice.name().to_string();
            let occ = seen.entry(name.clone()).or_insert(0);
            *occ += 1;
            let mut key = combine(combine(hash_str(&name), *occ), base);
            if let Some(p) = patch_of(&name, &by_id) {
                key = combine(key, patch_hash(p));
                s.anchor = anchor_for(p, &rp);
            }
            s.key = key;
            s
        })
        .collect();
    Graph { rng: cfg.rng, scene: Arc::new(scene), patches: Arc::new(patches), slots, procs, procs_key: base, ref_pose: rp }
}

// ---------------------------------------------------------------------------
// calibration / speaker-check source
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestKind {
    Pink,
    White,
    Sine,
}

/// Steady test signal at a fixed direction: pink / white noise or a sine, `level_db` dB SPL rms.
pub struct TestSource {
    kind: TestKind,
    em: Emission,
    amp: f32,
    phase: f64,
    dphase: f64,
    pink: crate::dsp::PinkFilter,
    key: u64,
    name: String,
}

impl TestSource {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: TestKind,
        azimuth: f32,
        elevation: f32,
        spread: f32,
        level_db: f32,
        freq_hz: f32,
        stem: crate::types::Stem,
        sr: u32,
        seed: u64,
    ) -> TestSource {
        let rms = crate::types::db_spl_to_pa(level_db);
        // uniform white [-1, 1) has rms 1/sqrt3; the pink filter output has ~0.343x the rms of its input
        let amp = match kind {
            TestKind::Sine => rms * std::f32::consts::SQRT_2,
            TestKind::White => rms * 3f32.sqrt(),
            TestKind::Pink => rms * 3f32.sqrt() / 0.343,
        };
        let name = format!("test.{:?}.{:.2}.{:.2}", kind, azimuth, elevation).to_lowercase();
        TestSource {
            kind,
            em: Emission::point(azimuth, elevation, 1.0, spread.clamp(0.0, 1.0), stem),
            amp,
            phase: 0.0,
            dphase: std::f64::consts::TAU * freq_hz as f64 / sr as f64,
            pink: crate::dsp::PinkFilter::default(),
            key: crate::rng::combine(seed, crate::rng::hash_str(&name)),
            name,
        }
    }
}

impl Voice for TestSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        match self.kind {
            TestKind::Sine => {
                for v in out.iter_mut() {
                    *v = self.amp * self.phase.sin() as f32;
                    self.phase += self.dphase;
                    if self.phase > std::f64::consts::TAU {
                        self.phase -= std::f64::consts::TAU;
                    }
                }
            }
            TestKind::White | TestKind::Pink => {
                let mut rng = ctx.rng.stream(self.key, ctx.index);
                rng.fill_bipolar(out);
                if self.kind == TestKind::Pink {
                    self.pink.process_block(out);
                }
                crate::dsp::scale(out, self.amp);
            }
        }
        self.em
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.pink = crate::dsp::PinkFilter::default();
    }
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------
pub enum Command {
    Weather(Box<Weather>),
    Graph(Box<Graph>),
    Output { decoder: Box<Decoder> },
    GainDb(f32),
    Limiter { ceiling_dbtp: f32, enabled: bool },
    StemGains([f32; N_STEMS]),
    Materials(Arc<MaterialTable>),
    Decorrelators(Vec<Vec<f32>>),
    AddVoice(BoxedVoice),
    /// Manual lightning strike (flash now, thunder at distance / c).
    Strike(Strike),
    /// Move the scene clock to t (s) without touching the voices.
    Seek(f64),
    Reset,
}

// ---------------------------------------------------------------------------
// sample FIFO (offline render of arbitrary frame counts)
// ---------------------------------------------------------------------------
pub struct Fifo {
    pub chans: Vec<Vec<f32>>,
    /// samples to discard from the next pushes (latency trim / alignment)
    pub drop: usize,
}

impl Fifo {
    pub fn new(channels: usize) -> Fifo {
        Fifo { chans: vec![Vec::new(); channels], drop: 0 }
    }
    pub fn len(&self) -> usize {
        self.chans.first().map(|c| c.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn clear(&mut self) {
        for c in &mut self.chans {
            c.clear();
        }
        self.drop = 0;
    }
    /// Push a channel-major block (channels x n).
    pub fn push(&mut self, block: &[f32], n: usize) {
        let skip = self.drop.min(n);
        self.drop -= skip;
        for (c, ch) in self.chans.iter_mut().enumerate() {
            ch.extend_from_slice(&block[c * n + skip..(c + 1) * n]);
        }
    }
    pub fn push_zeros(&mut self, k: usize) {
        for ch in &mut self.chans {
            ch.resize(ch.len() + k, 0.0);
        }
    }
    /// Take `frames` (must be available) into `out` (channels x frames, channel-major).
    pub fn take_into(&mut self, frames: usize, out: &mut [f32]) {
        for (c, ch) in self.chans.iter_mut().enumerate() {
            out[c * frames..(c + 1) * frames].copy_from_slice(&ch[..frames]);
            ch.drain(..frames);
        }
    }
    /// Discard `frames` (what is not there yet is dropped from future pushes).
    pub fn discard(&mut self, frames: usize) {
        let have = self.len().min(frames);
        for ch in &mut self.chans {
            ch.drain(..have);
        }
        self.drop += frames - have;
    }
}

// ---------------------------------------------------------------------------
// the core
// ---------------------------------------------------------------------------
pub struct Stats {
    pub cpu_ms_per_block: f64,
    pub realtime_factor: f64,
    pub blocks: u64,
    pub voices: usize,
    pub voices_active: usize,
    pub dead_voices: usize,
    pub gain_reduction_db: f32,
    pub max_gain_reduction_db: f32,
    /// Voices built and faded in by scene updates after the first graph (0 = nothing rebuilt).
    pub voice_rebuilds: u64,
    /// Voices kept running across scene updates.
    pub voices_kept: u64,
    /// Scene updates applied after the first graph.
    pub graph_updates: u64,
    /// Bus processor (place / shelter) crossfades.
    pub processor_swaps: u64,
}

pub struct Core {
    pub sr: u32,
    pub n: usize,
    pub seed: u64,
    pub rngf: RngFactory,
    pool: Arc<rayon::ThreadPool>,
    rx: Receiver<Command>,
    materials: Arc<MaterialTable>,
    scene: Arc<Scene>,
    patches: Arc<Vec<Patch>>,
    weather: Box<Weather>,
    weather_changed: bool,
    slots: Vec<Slot>,
    fading: Vec<Slot>,
    procs: Vec<BoxedProcessor>,
    encoder: Encoder,
    decoder: Decoder,
    limiter: Limiter,
    decoder_latency: usize,
    gain: f32,
    gain_target: f32,
    stem_gain: [f32; N_STEMS],
    buses: StemBuses,
    mix_bus: Vec<f32>,
    out_block: Vec<f32>,
    pub block_index: u64,
    sink: MarkerSink,
    markers: Vec<Marker>,
    block_markers: Vec<Marker>,
    fifo_mix: Fifo,
    fifo_stems: Fifo,
    stems_enabled: bool,
    frames_out: u64,
    cpu_ms_ema: f64,
    last_cpu_ms: f64,
    pub underruns: Arc<AtomicU64>,
    /// Blocks rendered so far (lock-free mirror of `block_index` for control threads).
    pub progress: Arc<AtomicU64>,
    /// Smoothed ms per block of each stage: voices, encode, decorrelate, processors, decode+limit.
    pub stage_ms: [f64; 5],
    /// Slowest block since the last reset (ms) and blocks slower than 75 % of real time.
    pub max_cpu_ms: f64,
    pub slow_blocks: u64,
    /// Live listener pose (set from any thread) and its smoothed value (x, y, z, heading, pitch).
    pub pose: Arc<ListenerPose>,
    smooth: Option<[f64; 5]>,
    ref_pose: RefPose,
    pose_ctx: PoseCtx,
    rot_prev: [[f32; N_AMBI]; N_AMBI],
    rot_identity: bool,
    rot_scratch: Vec<f32>,
    ramp: Vec<f32>,
    procs_key: u64,
    procs_old: Vec<BoxedProcessor>,
    procs_u: f32,
    buses_alt: StemBuses,
    strikes: Vec<Strike>,
    /// Scene time minus render time (s), moved by `Seek`.
    pub time_offset: f64,
    voice_rebuilds: u64,
    voices_kept: u64,
    graph_updates: u64,
    processor_swaps: u64,
}

fn identity16() -> [[f32; N_AMBI]; N_AMBI] {
    let mut r = [[0.0f32; N_AMBI]; N_AMBI];
    for (k, row) in r.iter_mut().enumerate() {
        row[k] = 1.0;
    }
    r
}

pub const STAGES: [&str; 5] = ["voices", "encode", "finish", "processors", "decode_limit"];

const MAX_MARKERS: usize = 16_384;

pub fn make_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    let t = if threads == 0 {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4)
    } else {
        threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .thread_name(|i| format!("raingen-render-{i}"))
        .start_handler(|_| enable_ftz())
        .build()
        .map(Arc::new)
        .map_err(|e| format!("thread pool: {e}"))
}

impl Core {
    pub fn new(sr: u32, n: usize, seed: u64, pool: Arc<rayon::ThreadPool>, rx: Receiver<Command>) -> Core {
        let decoder = Decoder::default_stereo(n);
        let ch = decoder.channels();
        let limiter = Limiter::safety(sr, ch);
        let mut fifo_mix = Fifo::new(ch);
        fifo_mix.drop = limiter.latency();
        Core {
            sr,
            n,
            seed,
            rngf: RngFactory::new(seed),
            pool,
            rx,
            materials: Arc::new(MaterialTable::default()),
            scene: Arc::new(Scene::default()),
            patches: Arc::new(Vec::new()),
            weather: Box::new(Weather::default()),
            weather_changed: true,
            slots: Vec::new(),
            fading: Vec::new(),
            procs: Vec::new(),
            encoder: Encoder::new(n, sr),
            decoder,
            limiter,
            decoder_latency: 0,
            gain: 1.0,
            gain_target: 1.0,
            stem_gain: [1.0; N_STEMS],
            buses: StemBuses::new(n),
            mix_bus: vec![0.0; N_AMBI * n],
            out_block: vec![0.0; ch * n],
            block_index: 0,
            sink: MarkerSink::new(4096),
            markers: Vec::new(),
            block_markers: Vec::with_capacity(256),
            fifo_mix,
            fifo_stems: Fifo::new(N_STEMS * N_AMBI),
            stems_enabled: false,
            frames_out: 0,
            cpu_ms_ema: 0.0,
            last_cpu_ms: 0.0,
            underruns: Arc::new(AtomicU64::new(0)),
            progress: Arc::new(AtomicU64::new(0)),
            stage_ms: [0.0; 5],
            max_cpu_ms: 0.0,
            slow_blocks: 0,
            pose: Arc::new(ListenerPose::default()),
            smooth: None,
            ref_pose: RefPose::default(),
            pose_ctx: PoseCtx::default(),
            rot_prev: identity16(),
            rot_identity: true,
            rot_scratch: vec![0.0; N_AMBI * n],
            ramp: (1..=n).map(|i| i as f32 / n as f32).collect(),
            procs_key: 0,
            procs_old: Vec::new(),
            procs_u: 1.0,
            buses_alt: StemBuses::new(n),
            strikes: Vec::with_capacity(8),
            time_offset: 0.0,
            voice_rebuilds: 0,
            voices_kept: 0,
            graph_updates: 0,
            processor_swaps: 0,
        }
    }

    pub fn channels(&self) -> usize {
        self.decoder.channels()
    }

    /// Output latency of the decoded mix in samples (decoder + limiter).
    pub fn latency(&self) -> usize {
        self.decoder_latency + self.limiter.latency()
    }

    pub fn materials(&self) -> &Arc<MaterialTable> {
        &self.materials
    }

    /// Apply every queued command (start of each block; also callable directly).
    pub fn drain_commands(&mut self) {
        while let Ok(cmd) = self.rx.try_recv() {
            self.apply(cmd);
        }
    }

    fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::Weather(w) => {
                self.weather = w;
                self.weather_changed = true;
            }
            Command::Graph(g) => self.apply_graph(*g),
            Command::Output { decoder } => {
                let ch_changed = decoder.channels() != self.decoder.channels();
                let new_lat = decoder.latency();
                self.decoder = *decoder;
                if ch_changed {
                    let (ceil, en) = (self.limiter.ceiling_dbtp, self.limiter.enabled);
                    self.limiter = Limiter::new(self.sr, self.decoder.channels(), ceil, 5.0, 80.0);
                    self.limiter.enabled = en;
                    self.out_block = vec![0.0; self.decoder.channels() * self.n];
                    self.fifo_mix = Fifo::new(self.decoder.channels());
                }
                if self.block_index == 0 {
                    self.fifo_mix.clear();
                    self.fifo_mix.drop = new_lat + self.limiter.latency();
                }
                self.decoder_latency = new_lat;
            }
            Command::GainDb(db) => {
                self.gain_target = 10f32.powf(db / 20.0);
                if self.block_index == 0 {
                    self.gain = self.gain_target;
                }
            }
            Command::Limiter { ceiling_dbtp, enabled } => {
                self.limiter.set_ceiling(ceiling_dbtp);
                self.limiter.enabled = enabled;
            }
            Command::StemGains(g) => self.stem_gain = g,
            Command::Materials(m) => {
                self.materials = m;
                self.weather.resolve(&self.materials);
            }
            Command::Decorrelators(firs) => self.encoder.set_decorrelators(firs),
            Command::AddVoice(v) => {
                let mut s = Slot::new(v, self.n);
                s.key = combine(s.key, self.block_index ^ 0xADD5);
                self.slots.push(s)
            }
            Command::Strike(st) => self.strikes.push(st),
            Command::Seek(t) => {
                if t.is_finite() {
                    self.time_offset = t - self.block_index as f64 * self.n as f64 / self.sr as f64;
                }
            }
            Command::Reset => self.reset(),
        }
    }

    /// Install a new graph. Before the first block it simply replaces the old one; while
    /// playing it is diffed against the running graph by voice key: matching voices keep
    /// running (state, phase, tails) and only take the new anchor (moved objects) and gain,
    /// added voices fade in and removed ones fade out over VOICE_FADE_S. Bus processors are
    /// kept when their key is unchanged, else crossfaded.
    fn apply_graph(&mut self, g: Graph) {
        let Graph { rng, scene, patches, slots: new_slots, procs, procs_key, ref_pose } = g;
        self.rngf = rng;
        self.seed = rng.root_seed;
        self.scene = scene;
        self.patches = patches;
        self.ref_pose = ref_pose;
        let live = self.block_index > 0 && !(self.slots.is_empty() && self.fading.is_empty() && self.procs.is_empty());
        if !live {
            let old = std::mem::replace(&mut self.slots, new_slots);
            let old_fading = std::mem::take(&mut self.fading);
            let old_procs = std::mem::replace(&mut self.procs, procs);
            let old_po = std::mem::take(&mut self.procs_old);
            drop_later((old, old_fading, old_procs, old_po));
            self.procs_key = procs_key;
            self.procs_u = 1.0;
            for s in &mut self.slots {
                s.fade_u = 1.0;
                s.fade_dir = 0;
            }
            return;
        }
        self.graph_updates += 1;
        let mut old: HashMap<u64, Slot> = HashMap::with_capacity(self.slots.len() + self.fading.len());
        let mut extra: Vec<Slot> = Vec::new();
        for s in std::mem::take(&mut self.slots).into_iter().chain(std::mem::take(&mut self.fading)) {
            if let Some(prev) = old.insert(s.key, s) {
                extra.push(prev);
            }
        }
        let mut slots = Vec::with_capacity(new_slots.len());
        let mut discard = Vec::new();
        let mut kept = 0u64;
        for ns in new_slots {
            match old.remove(&ns.key) {
                Some(mut os) => {
                    if let (Some(oa), Some(na)) = (os.anchor.as_mut(), ns.anchor.as_ref()) {
                        oa.w_ref = na.w_ref;
                        oa.gain_mul = if oa.amp_build > 1e-9 { (na.amp_build / oa.amp_build).clamp(0.0, 4.0) } else { 1.0 };
                    }
                    if os.fade_dir < 0 || os.fade_u < 1.0 {
                        os.fade_dir = 1;
                    }
                    kept += 1;
                    slots.push(os);
                    discard.push(ns);
                }
                None => {
                    let mut ns = ns;
                    ns.fade_u = 0.0;
                    ns.fade_dir = 1;
                    self.voice_rebuilds += 1;
                    slots.push(ns);
                }
            }
        }
        self.voices_kept = kept;
        self.slots = slots;
        let mut fading: Vec<Slot> = old.into_values().collect();
        fading.sort_by_key(|s| s.key);
        for s in &mut fading {
            s.fade_dir = -1;
        }
        fading.extend(extra.into_iter().map(|mut s| {
            s.fade_dir = -1;
            s
        }));
        self.fading = fading;
        if procs_key == self.procs_key {
            discard_procs(procs);
        } else {
            self.processor_swaps += 1;
            let prev = std::mem::replace(&mut self.procs, procs);
            let older = std::mem::replace(&mut self.procs_old, prev);
            discard_procs(older);
            self.procs_key = procs_key;
            self.procs_u = 0.0;
        }
        drop_later(discard);
    }

    /// Back to t = 0: block index, every voice, the encoder, decoder, limiter and FIFOs.
    pub fn reset(&mut self) {
        self.block_index = 0;
        self.progress.store(0, Ordering::Release);
        self.max_cpu_ms = 0.0;
        self.slow_blocks = 0;
        self.frames_out = 0;
        let f = std::mem::take(&mut self.fading);
        drop_later(f);
        for s in &mut self.slots {
            s.voice.reset();
            s.reset_live();
        }
        for p in &mut self.procs {
            p.reset();
        }
        discard_procs(std::mem::take(&mut self.procs_old));
        self.procs_u = 1.0;
        self.smooth = None;
        self.pose_ctx = PoseCtx::default();
        self.rot_prev = identity16();
        self.rot_identity = true;
        self.strikes.clear();
        self.time_offset = 0.0;
        self.encoder.reset();
        self.decoder.reset();
        self.limiter.reset();
        self.gain = self.gain_target;
        self.fifo_mix.clear();
        self.fifo_mix.drop = self.latency();
        self.fifo_stems.clear();
        self.markers.clear();
        self.weather_changed = true;
    }

    /// Smooth the live pose toward its target (~50 ms one-pole, shortest arc for the heading)
    /// and express it relative to the graph's reference pose.
    fn update_pose(&mut self) {
        let rp = self.ref_pose;
        let target = match self.pose.get() {
            Some(v) if v.iter().all(|x| x.is_finite()) => v,
            _ => {
                // no live pose: the scene's listener, exactly (offline renders stay bit-exact)
                self.smooth = None;
                self.pose_ctx = PoseCtx::default();
                return;
            }
        };
        let t = [target[0] as f64, target[1] as f64, target[2] as f64, target[3] as f64, target[4].clamp(-90.0, 90.0) as f64];
        let sm = match self.smooth.as_mut() {
            None => {
                self.smooth = Some(t);
                self.smooth.as_mut().unwrap()
            }
            Some(sm) => {
                let a = 1.0 - (-(self.n as f64 / self.sr as f64) / POSE_TAU_S).exp();
                for k in [0, 1, 2, 4] {
                    sm[k] += a * (t[k] - sm[k]);
                }
                let dh = (t[3] - sm[3] + 540.0).rem_euclid(360.0) - 180.0;
                sm[3] = (sm[3] + a * dh).rem_euclid(360.0);
                sm
            }
        };
        let e = rp.ears();
        let d = [sm[0] as f32 - rp.pos[0], sm[1] as f32 - rp.pos[1], (sm[2] as f32 + rp.height_m) - e[2]];
        let delta = rp.to_frame(d);
        let dheading = (((sm[3] - rp.heading_deg as f64 + 540.0).rem_euclid(360.0)) - 180.0).to_radians() as f32;
        let pitch = sm[4].to_radians() as f32;
        let moved = delta.iter().any(|v| v.abs() > 1e-4);
        let air = if moved {
            let w = &self.weather;
            crate::rain::air::iso9613_alpha_db_per_m(8000.0, w.temperature_c as f64, w.humidity_pct as f64, w.pressure_hpa as f64)
                as f32
        } else {
            0.0
        };
        self.pose_ctx = PoseCtx { delta, air_db_per_m: air, dheading, pitch };
    }

    /// Rotate the summed SH field for the listener's head turn / pitch (matrix interpolated
    /// across the block; yaw-only matrices are sparse, so only their non-zero terms run).
    fn rotate_mix(&mut self) {
        let (yaw, pitch) = (self.pose_ctx.dheading, self.pose_ctx.pitch);
        let still = yaw.abs() < 1e-6 && pitch.abs() < 1e-6;
        if still && self.rot_identity {
            return;
        }
        let r1 = if still { identity16() } else { listener_rotation(yaw, pitch) };
        let n = self.n;
        self.rot_scratch.copy_from_slice(&self.mix_bus[..N_AMBI * n]);
        let r0 = self.rot_prev;
        for i in 0..N_AMBI {
            let out = &mut self.mix_bus[i * n..(i + 1) * n];
            out.iter_mut().for_each(|v| *v = 0.0);
            for j in 0..N_AMBI {
                let a = r0[i][j];
                let d = r1[i][j] - a;
                if a == 0.0 && d == 0.0 {
                    continue;
                }
                let x = &self.rot_scratch[j * n..(j + 1) * n];
                if d == 0.0 {
                    for (o, v) in out.iter_mut().zip(x) {
                        *o += a * *v;
                    }
                } else {
                    for ((o, v), r) in out.iter_mut().zip(x).zip(&self.ramp) {
                        *o += (a + d * *r) * *v;
                    }
                }
            }
        }
        self.rot_prev = r1;
        self.rot_identity = still;
    }

    /// Render one block into `out_block` (channels x n) and, if `stems`, leave the per-stem SH
    /// buses in `self.buses`.
    pub fn render_block(&mut self) {
        let started = Instant::now();
        self.drain_commands();
        self.update_pose();
        let n = self.n;
        let sr = self.sr;
        let t0 = self.block_index as f64 * n as f64 / sr as f64;
        let fade_step = (n as f32 / sr as f32) / VOICE_FADE_S;

        // split borrows: the context reads these fields while the slots are rendered
        let Core {
            pool,
            weather,
            weather_changed,
            rngf,
            scene,
            materials,
            sink,
            slots,
            fading,
            procs,
            procs_old,
            procs_u,
            buses_alt,
            encoder,
            buses,
            stem_gain,
            block_index,
            strikes,
            pose_ctx,
            ramp,
            ..
        } = self;
        let ctx = BlockCtx {
            sr,
            sr_f: sr as f32,
            n,
            index: *block_index,
            t0,
            weather: &**weather,
            weather_changed: *weather_changed,
            rng: &*rngf,
            listener_height_m: scene.listener.height_m,
            shelter: scene.place.shelter,
            window_open: scene.place.window_open,
            scene: &**scene,
            materials: &**materials,
            markers: &*sink,
            strikes: &strikes[..],
        };
        let pose = *pose_ctx;

        let mut marks = [Instant::now(); 6];
        pool.install(|| {
            slots.par_iter_mut().chain(fading.par_iter_mut()).for_each(|s| s.render(&ctx, &pose));
        });
        marks[1] = Instant::now();

        // encode serially, in graph order
        buses.clear();
        encoder.begin_block();
        for s in slots.iter_mut().chain(fading.iter_mut()) {
            let u0 = s.fade_u;
            let u1 = (u0 + s.fade_dir as f32 * fade_step).clamp(0.0, 1.0);
            s.fade_u = u1;
            if !s.meta.active {
                continue;
            }
            let (g0, g1) = (fade_gain(u0), fade_gain(u1));
            let bus = buses.bus_mut(s.meta.stem);
            if s.channels == N_AMBI {
                encoder.add_ambisonic(bus, &s.buf, g0, g1);
            } else {
                encoder.encode_mono(bus, &s.buf, &s.meta, &mut s.enc, g0, g1);
            }
        }
        for s in slots.iter_mut() {
            if s.fade_dir > 0 && s.fade_u >= 1.0 {
                s.fade_dir = 0;
            }
        }
        if fading.iter().any(|s| s.fade_u <= 0.0) {
            let (done, keep): (Vec<Slot>, Vec<Slot>) = std::mem::take(fading).into_iter().partition(|s| s.fade_u <= 0.0);
            *fading = keep;
            drop_later(done);
        }
        marks[2] = Instant::now();
        pool.install(|| encoder.finish_block(buses));
        marks[3] = Instant::now();

        for (i, g) in stem_gain.iter().enumerate() {
            if *g != 1.0 {
                buses.scale_stem(crate::types::Stem::from_index(i), *g);
            }
        }
        let xfade = *procs_u < 1.0;
        if xfade {
            buses_alt.data.copy_from_slice(&buses.data);
        }
        pool.install(|| {
            if xfade {
                for p in procs_old.iter_mut() {
                    let _ = catch_unwind(AssertUnwindSafe(|| p.process(&ctx, buses_alt)));
                }
            }
            for p in procs.iter_mut() {
                let r = catch_unwind(AssertUnwindSafe(|| p.process(&ctx, buses)));
                if r.is_err() {
                    // a failing processor is bypassed for this block; buses stay as encoded
                }
            }
        });
        if xfade {
            // equal-power crossfade from the old place / shelter processors to the new ones
            let u0 = *procs_u;
            let u1 = (u0 + fade_step).min(1.0);
            let (n0, n1) = (fade_gain(u0), fade_gain(u1));
            let (o0, o1) = (fade_gain(1.0 - u0), fade_gain(1.0 - u1));
            for (nb, ob) in buses.data.chunks_mut(n).zip(buses_alt.data.chunks(n)) {
                for ((v, w), r) in nb.iter_mut().zip(ob).zip(ramp.iter()) {
                    *v = *v * (n0 + (n1 - n0) * *r) + *w * (o0 + (o1 - o0) * *r);
                }
            }
            *procs_u = u1;
            if u1 >= 1.0 {
                discard_procs(std::mem::take(procs_old));
            }
        }
        drop(ctx);
        strikes.clear();
        marks[4] = Instant::now();

        // mix, rotate for the head, decode, gain, limiter
        self.buses.sum_into(&mut self.mix_bus);
        self.rotate_mix();
        self.decoder.process(&self.mix_bus, &mut self.out_block);
        let ch = self.decoder.channels();
        let (g0, g1) = (self.gain, self.gain_target);
        if g0 == g1 {
            if g0 != 1.0 {
                self.out_block.iter_mut().for_each(|v| *v *= g0);
            }
        } else {
            for c in 0..ch {
                let o = &mut self.out_block[c * n..(c + 1) * n];
                for (i, v) in o.iter_mut().enumerate() {
                    *v *= g0 + (g1 - g0) * (i + 1) as f32 / n as f32;
                }
            }
            self.gain = g1;
        }
        self.limiter.process_planar(&mut self.out_block, n);
        marks[5] = Instant::now();
        for (i, v) in self.stage_ms.iter_mut().enumerate() {
            let ms = (marks[i + 1] - marks[i]).as_secs_f64() * 1e3;
            *v = if *v == 0.0 { ms } else { 0.95 * *v + 0.05 * ms };
        }

        // markers (sorted for determinism), on the scene clock
        self.block_markers.clear();
        self.sink.drain_into(&mut self.block_markers);
        if !self.block_markers.is_empty() {
            self.block_markers.sort_by(|a, b| {
                a.t.partial_cmp(&b.t)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.kind.cmp(b.kind))
                    .then(a.data.cmp(&b.data))
            });
            if self.time_offset != 0.0 {
                for m in &mut self.block_markers {
                    m.t += self.time_offset;
                }
            }
            let room = MAX_MARKERS.saturating_sub(self.markers.len());
            if room < self.block_markers.len() {
                let excess = self.block_markers.len() - room;
                self.markers.drain(..excess.min(self.markers.len()));
            }
            self.markers.append(&mut self.block_markers);
        }

        self.block_index += 1;
        self.progress.store(self.block_index, Ordering::Release);
        self.weather_changed = false;

        let ms = started.elapsed().as_secs_f64() * 1e3;
        self.last_cpu_ms = ms;
        self.max_cpu_ms = self.max_cpu_ms.max(ms);
        if ms > 0.75e3 * n as f64 / sr as f64 {
            self.slow_blocks += 1;
        }
        self.cpu_ms_ema = if self.cpu_ms_ema == 0.0 { ms } else { 0.95 * self.cpu_ms_ema + 0.05 * ms };
    }

    /// The last rendered block (channels x n, channel-major).
    pub fn out_block(&self) -> &[f32] {
        &self.out_block
    }

    /// The last rendered per-stem SH buses.
    pub fn buses(&self) -> &StemBuses {
        &self.buses
    }

    /// Offline: the next `frames` of the decoded, limited mix (channels x frames). Sample 0 is
    /// scene time 0 (decoder + limiter latency trimmed).
    pub fn render_mix(&mut self, frames: usize) -> Vec<f32> {
        self.drain_commands();
        let n = self.n;
        while self.fifo_mix.len() < frames {
            self.render_block();
            self.fifo_mix.push(&self.out_block, n);
            if self.stems_enabled {
                self.fifo_stems.push(&self.buses.data, n);
            }
        }
        let ch = self.decoder.channels();
        let mut out = vec![0.0f32; ch * frames];
        self.fifo_mix.take_into(frames, &mut out);
        if self.stems_enabled {
            self.fifo_stems.discard(frames);
        }
        self.frames_out += frames as u64;
        out
    }

    /// Offline: the next `frames` of every stem's SH bus (N_STEMS x N_AMBI x frames).
    pub fn render_stems(&mut self, frames: usize) -> Vec<f32> {
        self.drain_commands();
        let n = self.n;
        if !self.stems_enabled {
            self.stems_enabled = true;
            self.fifo_stems.clear();
            let pad = (self.block_index * n as u64).saturating_sub(self.frames_out) as usize;
            self.fifo_stems.push_zeros(pad);
        }
        while self.fifo_stems.len() < frames {
            self.render_block();
            self.fifo_mix.push(&self.out_block, n);
            self.fifo_stems.push(&self.buses.data, n);
        }
        let mut out = vec![0.0f32; N_STEMS * N_AMBI * frames];
        self.fifo_stems.take_into(frames, &mut out);
        self.fifo_mix.discard(frames);
        self.frames_out += frames as u64;
        out
    }

    /// Live: render one block and write it interleaved (n x channels) into `dst`.
    pub fn render_interleaved(&mut self, dst: &mut [f32]) {
        self.render_block();
        let ch = self.decoder.channels();
        let n = self.n;
        for i in 0..n {
            for c in 0..ch {
                dst[i * ch + c] = self.out_block[c * n + i];
            }
        }
    }

    pub fn take_markers(&mut self) -> Vec<Marker> {
        std::mem::take(&mut self.markers)
    }

    pub fn stats(&self) -> Stats {
        let block_ms = self.n as f64 / self.sr as f64 * 1e3;
        let dead = self.slots.iter().filter(|s| s.dead).count();
        Stats {
            cpu_ms_per_block: self.cpu_ms_ema,
            realtime_factor: if block_ms > 0.0 { self.cpu_ms_ema / block_ms } else { 0.0 },
            blocks: self.block_index,
            voices: self.slots.len(),
            voices_active: self.slots.iter().filter(|s| s.meta.active).count(),
            dead_voices: dead,
            gain_reduction_db: self.limiter.gain_reduction_db,
            max_gain_reduction_db: self.limiter.max_gain_reduction_db,
            voice_rebuilds: self.voice_rebuilds,
            voices_kept: self.voices_kept,
            graph_updates: self.graph_updates,
            processor_swaps: self.processor_swaps,
        }
    }

    /// Scene time (s) at the start of the next block.
    pub fn scene_time(&self) -> f64 {
        self.block_index as f64 * self.n as f64 / self.sr as f64 + self.time_offset
    }

    /// The smoothed live pose (x, y, z, heading, pitch), or the scene listener's.
    pub fn listener_now(&self) -> [f64; 5] {
        match self.smooth {
            Some(v) => v,
            None => {
                let r = self.ref_pose;
                [r.pos[0] as f64, r.pos[1] as f64, r.pos[2] as f64, r.heading_deg as f64, 0.0]
            }
        }
    }

    pub fn underrun_count(&self) -> u64 {
        self.underruns.load(Ordering::Relaxed)
    }

    pub fn last_cpu_ms(&self) -> f64 {
        self.last_cpu_ms
    }

    /// (voice name, smoothed ms per block) in graph order.
    pub fn voice_profile(&self) -> Vec<(String, f64, f64)> {
        self.slots.iter().map(|s| (s.voice.name().to_string(), s.cpu_ms, s.max_ms)).collect()
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.slots.iter().map(|s| s.voice.name().to_string()).collect()
    }

    pub fn weather(&self) -> &Weather {
        &self.weather
    }

    pub fn patches(&self) -> &[Patch] {
        &self.patches
    }
}
