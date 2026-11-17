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

use crate::dsp::enable_ftz;
use crate::limiter::Limiter;
use crate::rng::RngFactory;
use crate::spatial::{Decoder, Encoder, SourceState, StemBuses};
use crate::types::{
    Emission, Marker, MaterialTable, Patch, Scene, Shelter, Weather, N_AMBI, N_STEMS,
};
use crossbeam_channel::Receiver;
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
pub struct Slot {
    pub voice: BoxedVoice,
    channels: usize,
    buf: Vec<f32>,
    meta: Emission,
    enc: SourceState,
    dead: bool,
    fade: (f32, f32),
    /// Smoothed render time of this voice (ms per block).
    pub cpu_ms: f64,
    /// Slowest block of this voice (ms).
    pub max_ms: f64,
}

impl Slot {
    fn new(voice: BoxedVoice, n: usize) -> Slot {
        let ch = if voice.channels() >= N_AMBI { N_AMBI } else { 1 };
        Slot {
            voice,
            channels: ch,
            buf: vec![0.0; ch * n],
            meta: Emission::SILENT,
            enc: SourceState::default(),
            dead: false,
            fade: (1.0, 1.0),
            cpu_ms: 0.0,
            max_ms: 0.0,
        }
    }

    fn render(&mut self, ctx: &BlockCtx) {
        let t = Instant::now();
        self.render_inner(ctx);
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
}

pub struct Graph {
    pub rng: RngFactory,
    pub scene: Arc<Scene>,
    pub patches: Arc<Vec<Patch>>,
    pub slots: Vec<Slot>,
    pub procs: Vec<BoxedProcessor>,
}

impl Graph {
    pub fn empty() -> Graph {
        Graph { rng: RngFactory::new(0), scene: Arc::new(Scene::default()), patches: Arc::new(Vec::new()), slots: Vec::new(), procs: Vec::new() }
    }
}

/// Build the voice graph of a scene: rain voices first, then environment voices (fixed order).
pub fn build_graph(scene: Scene, patches: Vec<Patch>, materials: &MaterialTable, cfg: &BuildCfg) -> Graph {
    let mut voices: Vec<BoxedVoice> = crate::rain::build_voices(&scene, &patches, materials, cfg);
    voices.extend(crate::env::build_voices(&scene, &patches, materials, cfg));
    let procs = crate::env::build_bus_processors(&scene, &patches, materials, cfg);
    let slots = voices.into_iter().map(|v| Slot::new(v, cfg.block)).collect();
    Graph { rng: cfg.rng, scene: Arc::new(scene), patches: Arc::new(patches), slots, procs }
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
            Command::Graph(g) => {
                let g = *g;
                let crossfade = !self.slots.is_empty() && self.block_index > 0;
                let mut old = std::mem::take(&mut self.slots);
                self.fading.clear();
                if crossfade {
                    for s in &mut old {
                        s.fade = (1.0, 0.0);
                    }
                    self.fading = old;
                }
                self.slots = g.slots;
                for s in &mut self.slots {
                    s.fade = if crossfade { (0.0, 1.0) } else { (1.0, 1.0) };
                }
                self.procs = g.procs;
                self.rngf = g.rng;
                self.seed = g.rng.root_seed;
                self.scene = g.scene;
                self.patches = g.patches;
            }
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
            Command::AddVoice(v) => self.slots.push(Slot::new(v, self.n)),
            Command::Reset => self.reset(),
        }
    }

    /// Back to t = 0: block index, every voice, the encoder, decoder, limiter and FIFOs.
    pub fn reset(&mut self) {
        self.block_index = 0;
        self.progress.store(0, Ordering::Release);
        self.max_cpu_ms = 0.0;
        self.slow_blocks = 0;
        self.frames_out = 0;
        self.fading.clear();
        for s in &mut self.slots {
            s.voice.reset();
            s.enc = SourceState::default();
            s.dead = false;
            s.fade = (1.0, 1.0);
        }
        for p in &mut self.procs {
            p.reset();
        }
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

    /// Render one block into `out_block` (channels x n) and, if `stems`, leave the per-stem SH
    /// buses in `self.buses`.
    pub fn render_block(&mut self) {
        let started = Instant::now();
        self.drain_commands();
        let n = self.n;
        let sr = self.sr;
        let t0 = self.block_index as f64 * n as f64 / sr as f64;

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
            encoder,
            buses,
            stem_gain,
            block_index,
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
        };

        let mut marks = [Instant::now(); 6];
        pool.install(|| {
            slots.par_iter_mut().chain(fading.par_iter_mut()).for_each(|s| s.render(&ctx));
        });
        marks[1] = Instant::now();

        // encode serially, in graph order
        buses.clear();
        encoder.begin_block();
        for s in slots.iter_mut().chain(fading.iter_mut()) {
            if !s.meta.active {
                continue;
            }
            let (g0, g1) = s.fade;
            let bus = buses.bus_mut(s.meta.stem);
            if s.channels == N_AMBI {
                encoder.add_ambisonic(bus, &s.buf, g0, g1);
            } else {
                encoder.encode_mono(bus, &s.buf, &s.meta, &mut s.enc, g0, g1);
            }
        }
        for s in slots.iter_mut() {
            s.fade = (1.0, 1.0);
        }
        fading.clear();
        marks[2] = Instant::now();
        pool.install(|| encoder.finish_block(buses));
        marks[3] = Instant::now();

        for (i, g) in stem_gain.iter().enumerate() {
            if *g != 1.0 {
                buses.scale_stem(crate::types::Stem::from_index(i), *g);
            }
        }
        pool.install(|| {
            for p in procs.iter_mut() {
                let r = catch_unwind(AssertUnwindSafe(|| p.process(&ctx, buses)));
                if r.is_err() {
                    // a failing processor is bypassed for this block; buses stay as encoded
                }
            }
        });
        drop(ctx);
        marks[4] = Instant::now();

        // mix, decode, gain, limiter
        self.buses.sum_into(&mut self.mix_bus);
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

        // markers (sorted for determinism)
        self.block_markers.clear();
        self.sink.drain_into(&mut self.block_markers);
        if !self.block_markers.is_empty() {
            self.block_markers.sort_by(|a, b| {
                a.t.partial_cmp(&b.t)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.kind.cmp(b.kind))
                    .then(a.data.cmp(&b.data))
            });
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
