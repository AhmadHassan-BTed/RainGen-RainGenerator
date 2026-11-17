//! raingen_native: the native real-time synthesis core of raingen.
//!
//! Python keeps the scene, World stepping, layout, decoder design, mastering and the CLI; this
//! crate does all per-sample work: every generator voice, SH encoding, decoder application,
//! the safety limiter and the live cpal stream.
//!
//! ```python
//! import json, dataclasses, raingen_native as rn
//! e = rn.Engine(sample_rate=48000, block=1024, seed=0, threads=0)
//! e.set_materials(json.dumps({k: dataclasses.asdict(m) for k, m in MATERIALS.items()}))
//! e.set_scene(scene.to_json(), json.dumps([dataclasses.asdict(p) for p in build_layout(scene)]))
//! e.set_output("stereo", 2, stereo_matrix(3))            # decoders designed in Python
//! for k in range(blocks):
//!     e.update(json.dumps(dataclasses.asdict(world.step(k, k * 1024 / 48000, 1024, scene))))
//!     y = e.render(1024)                                  # (channels, frames) float32
//! ```

pub mod dsp;
pub mod engine;
pub mod env;
pub mod limiter;
pub mod rain;
pub mod rng;
pub mod spatial;
pub mod stream;
pub mod types;

use crate::engine::{build_graph, make_pool, BuildCfg, Command, Core, ListenerPose, Strike};
use crate::rng::RngFactory;
use crate::spatial::{Decoder, Lfe};
use crate::stream::LiveStream;
use crate::types::{MaterialTable, Patch, Scene, Weather, N_AMBI, N_STEMS};
use crossbeam_channel::{unbounded, Sender};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn verr(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn rerr(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn as_f64_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("ascontiguousarray", (obj, "float64"))
}

/// Any 2-D array-like -> (rows, cols, row-major f32).
fn extract_2d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Vec<f32>)> {
    let a = as_f64_array(py, obj)?;
    let arr: PyReadonlyArray2<f64> = a.extract().map_err(|_| verr("expected a 2-D numeric array"))?;
    let shape = arr.shape().to_vec();
    let v = arr.as_array().iter().map(|x| *x as f32).collect();
    Ok((shape[0], shape[1], v))
}

/// Any 3-D array-like -> (d0, d1, d2, row-major f32).
fn extract_3d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize, Vec<f32>)> {
    let a = as_f64_array(py, obj)?;
    let arr: PyReadonlyArray3<f64> = a.extract().map_err(|_| verr("expected a 3-D numeric array"))?;
    let shape = arr.shape().to_vec();
    let v = arr.as_array().iter().map(|x| *x as f32).collect();
    Ok((shape[0], shape[1], shape[2], v))
}

fn order_channels(kind: &str) -> Option<usize> {
    match kind {
        "foa" | "hoa1" => Some(4),
        "hoa2" => Some(9),
        "hoa3" | "hoa" | "ambix" => Some(16),
        "hoa0" => Some(1),
        _ => None,
    }
}

/// The native engine. One instance = one voice graph, one output, optionally one live stream.
#[pyclass(name = "Engine", module = "raingen_native")]
pub struct PyEngine {
    core: Arc<Mutex<Core>>,
    tx: Sender<Command>,
    materials: Arc<MaterialTable>,
    sr: u32,
    block: usize,
    seed: u64,
    threads: usize,
    channels: usize,
    layout: String,
    live: Option<LiveStream>,
    progress: Arc<AtomicU64>,
    pose: Arc<ListenerPose>,
    strikes: AtomicU64,
}

impl PyEngine {
    fn send(&self, cmd: Command) -> PyResult<()> {
        self.tx.send(cmd).map_err(|_| rerr("engine command channel closed"))
    }

    fn check_offline(&self) -> PyResult<()> {
        if self.live.as_ref().map(|l| l.is_running()).unwrap_or(false) {
            return Err(rerr("the engine is streaming live; call stop() before render()"));
        }
        Ok(())
    }
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (sample_rate = 48000, block = 1024, seed = 0, threads = 0))]
    fn new(sample_rate: u32, block: usize, seed: u64, threads: usize) -> PyResult<Self> {
        if !(8000..=384_000).contains(&sample_rate) {
            return Err(verr(format!("sample_rate {sample_rate} out of range")));
        }
        if !(16..=16384).contains(&block) {
            return Err(verr(format!("block {block} out of range (16..16384)")));
        }
        let pool = make_pool(threads).map_err(rerr)?;
        let (tx, rx) = unbounded();
        let core = Core::new(sample_rate, block, seed, pool, rx);
        let channels = core.channels();
        let progress = core.progress.clone();
        let pose = core.pose.clone();
        Ok(PyEngine {
            core: Arc::new(Mutex::new(core)),
            tx,
            materials: Arc::new(MaterialTable::default()),
            sr: sample_rate,
            block,
            seed,
            threads,
            channels,
            layout: "stereo".into(),
            live: None,
            progress,
            pose,
            strikes: AtomicU64::new(0),
        })
    }

    /// The whole MATERIALS table as JSON: {name: dataclasses.asdict(Material)} (or a list).
    fn set_materials(&mut self, json: &str) -> PyResult<()> {
        let t = MaterialTable::from_json(json).map_err(verr)?;
        self.materials = Arc::new(t);
        self.send(Command::Materials(self.materials.clone()))
    }

    /// Build the voice graph from Scene.to_json() and the SourcePatch list (JSON list of dicts).
    /// Takes effect at the next block boundary. While playing it is an incremental update:
    /// voices whose patch / settings did not change keep running (moved patches only move),
    /// added and removed voices crossfade over 150 ms, the place / shelter processors
    /// crossfade only when they change. Safe to call from any thread while rendering.
    fn set_scene(&self, py: Python<'_>, scene_json: &str, patches_json: &str) -> PyResult<()> {
        let scene = Scene::from_json(scene_json).map_err(verr)?;
        let patches = Patch::list_from_json(patches_json).map_err(verr)?;
        let seed = scene.seed ^ self.seed;
        let cfg = BuildCfg { sr: self.sr, block: self.block, seed, rng: RngFactory::new(seed) };
        let mats = self.materials.clone();
        let graph = py.detach(move || build_graph(scene, patches, &mats, &cfg));
        self.send(Command::Graph(Box::new(graph)))
    }

    /// The live listener pose: feet position (world metres, as Scene.listener.position),
    /// heading (deg true) and pitch (deg, up positive). Lock-free, any thread, any rate: it never
    /// rebuilds voices. The engine glides to it (~50 ms): world-fixed voices move, turning
    /// rotates the sound field. clear=True returns to the scene's listener.
    #[pyo3(signature = (x = 0.0, y = 0.0, z = 0.0, heading_deg = 0.0, pitch_deg = 0.0, clear = false))]
    fn set_listener(&self, x: f32, y: f32, z: f32, heading_deg: f32, pitch_deg: f32, clear: bool) -> PyResult<()> {
        if clear {
            self.pose.clear();
            return Ok(());
        }
        if ![x, y, z, heading_deg, pitch_deg].iter().all(|v| v.is_finite()) {
            return Err(verr("listener pose must be finite"));
        }
        self.pose.set([x, y, z, heading_deg.rem_euclid(360.0), pitch_deg.clamp(-90.0, 90.0)]);
        Ok(())
    }

    /// A manual lightning strike now: the flash marker at the next block and the thunder at
    /// distance / c(T) through the storm model's channel and propagation. Returns its id.
    #[pyo3(signature = (distance_km = 3.0, bearing_deg = 0.0, intensity = 0.7))]
    fn strike(&self, distance_km: f32, bearing_deg: f32, intensity: f32) -> PyResult<String> {
        let id = self.strikes.fetch_add(1, Ordering::Relaxed) + 1;
        self.send(Command::Strike(Strike { id, distance_km, bearing_deg, intensity }))?;
        Ok(format!("strike.{id}"))
    }

    /// Move the scene clock to `t` seconds (marker times and stats follow) without resetting
    /// any voice or restarting the audio.
    fn seek(&self, t: f64) -> PyResult<()> {
        if !t.is_finite() {
            return Err(verr("t must be finite"));
        }
        self.send(Command::Seek(t))
    }

    /// One WeatherState per control block: json.dumps(dataclasses.asdict(weather)).
    fn update(&self, weather_json: &str) -> PyResult<()> {
        let mut w = Weather::from_json(weather_json).map_err(verr)?;
        w.resolve(&self.materials);
        self.send(Command::Weather(Box::new(w)))
    }

    /// Select the output. kind: stereo | mono | matrix | 5.1 | 7.1 | 7.1.4 (with `matrix`,
    /// channels x 16), binaural (with `firs`, 2 x 16 x taps), foa | hoa1 | hoa2 | hoa3 (raw SH).
    #[pyo3(signature = (kind, channels, matrix = None, firs = None, latency = 0, lfe_index = None, lfe_gain = 0.0, lfe_cutoff_hz = 120.0))]
    #[allow(clippy::too_many_arguments)]
    fn set_output(
        &mut self,
        py: Python<'_>,
        kind: &str,
        channels: usize,
        matrix: Option<&Bound<'_, PyAny>>,
        firs: Option<&Bound<'_, PyAny>>,
        latency: usize,
        lfe_index: Option<usize>,
        lfe_gain: f32,
        lfe_cutoff_hz: f32,
    ) -> PyResult<()> {
        let k = kind.trim().to_ascii_lowercase();
        let n = self.block;
        let dec = if let Some(nch) = order_channels(&k) {
            if channels != 0 && channels != nch {
                return Err(verr(format!("{k} has {nch} channels, got channels={channels}")));
            }
            Decoder::passthrough(&k, n, nch)
        } else if k == "binaural" {
            let f = firs.ok_or_else(|| verr("binaural output needs firs (2 x 16 x taps)"))?;
            let (e, kk, taps, v) = extract_3d(py, f)?;
            if kk > N_AMBI || e == 0 || taps == 0 {
                return Err(verr(format!("firs must be (outputs, <=16, taps), got ({e}, {kk}, {taps})")));
            }
            if channels != 0 && channels != e {
                return Err(verr(format!("firs have {e} outputs, got channels={channels}")));
            }
            // pad to 16 inputs
            let mut full = vec![0.0f32; e * N_AMBI * taps];
            for oe in 0..e {
                for ik in 0..kk {
                    let src = &v[(oe * kk + ik) * taps..(oe * kk + ik + 1) * taps];
                    full[(oe * N_AMBI + ik) * taps..(oe * N_AMBI + ik + 1) * taps].copy_from_slice(src);
                }
            }
            py.detach(|| Decoder::fir(&k, n, e, taps, &full, latency))
        } else {
            let (rows, cols, v) = match matrix {
                Some(m) => {
                    let (r, c, v) = extract_2d(py, m)?;
                    if c > N_AMBI {
                        return Err(verr(format!("matrix must be (channels, <=16), got ({r}, {c})")));
                    }
                    (r, c, v)
                }
                None => match k.as_str() {
                    "stereo" => {
                        let d = Decoder::default_stereo(n);
                        if channels != 0 && channels != 2 {
                            return Err(verr("stereo has 2 channels"));
                        }
                        self.apply_output(d)?;
                        return Ok(());
                    }
                    "mono" => {
                        let mut v = vec![0.0f32; N_AMBI];
                        v[0] = 1.0;
                        (1, N_AMBI, v)
                    }
                    _ => return Err(verr(format!("output {k:?} needs a decoding matrix (channels x 16)"))),
                },
            };
            if channels != 0 && channels != rows {
                return Err(verr(format!("matrix has {rows} rows, got channels={channels}")));
            }
            let mut full = vec![0.0f32; rows * N_AMBI];
            for r in 0..rows {
                full[r * N_AMBI..r * N_AMBI + cols].copy_from_slice(&v[r * cols..(r + 1) * cols]);
            }
            let lfe = match lfe_index {
                Some(i) if lfe_gain > 0.0 && i < rows => Some(Lfe {
                    index: i,
                    gain: lfe_gain,
                    filter: crate::dsp::Sos::linkwitz_riley_lowpass(4, lfe_cutoff_hz, self.sr as f32),
                }),
                _ => None,
            };
            let mut d = Decoder::matrix(&k, n, rows, full, lfe);
            if latency > 0 {
                d = d.with_latency(latency);
            }
            d
        };
        self.apply_output(dec)
    }

    fn set_gain_db(&self, db: f32) -> PyResult<()> {
        if !db.is_finite() {
            return Err(verr("gain must be finite"));
        }
        self.send(Command::GainDb(db))
    }

    /// Safety limiter: ceiling in dBTP (default -1), or bypass with enabled=False.
    #[pyo3(signature = (ceiling_dbtp = -1.0, enabled = true))]
    fn set_limiter(&self, ceiling_dbtp: f32, enabled: bool) -> PyResult<()> {
        self.send(Command::Limiter { ceiling_dbtp, enabled })
    }

    /// Per-stem gains in dB, STEMS order (rain_near, rain_far, wind, thunder, aftermath, ambience, interior).
    fn set_stem_gains(&self, gains_db: Vec<f32>) -> PyResult<()> {
        if gains_db.len() != N_STEMS {
            return Err(verr(format!("expected {N_STEMS} stem gains")));
        }
        let mut g = [1.0f32; N_STEMS];
        for (i, db) in gains_db.iter().enumerate() {
            g[i] = 10f32.powf(db / 20.0);
        }
        self.send(Command::StemGains(g))
    }

    /// Replace the encoder's 15 all-pass decorrelator FIRs (15 x L), e.g. Python's designed bank.
    fn set_decorrelators(&self, py: Python<'_>, firs: &Bound<'_, PyAny>) -> PyResult<()> {
        let (r, c, v) = extract_2d(py, firs)?;
        if r != N_AMBI - 1 || c == 0 {
            return Err(verr(format!("decorrelators must be ({}, L), got ({r}, {c})", N_AMBI - 1)));
        }
        let f: Vec<Vec<f32>> = (0..r).map(|i| v[i * c..(i + 1) * c].to_vec()).collect();
        self.send(Command::Decorrelators(f))
    }

    /// Offline, deterministic: the next `frames` of the decoded, limited mix, (channels, frames).
    fn render<'py>(&self, py: Python<'py>, frames: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.check_offline()?;
        let core = self.core.clone();
        let (ch, data) = py.detach(move || {
            let mut c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
            let d = c.render_mix(frames);
            Ok::<_, String>((c.channels(), d))
        })
        .map_err(rerr)?;
        let arr = Array2::from_shape_vec((ch, frames), data).map_err(rerr)?;
        Ok(arr.into_pyarray(py))
    }

    /// Offline: the next `frames` of every stem's SH bus, (7, 16, frames).
    fn render_stems<'py>(&self, py: Python<'py>, frames: usize) -> PyResult<Bound<'py, PyArray3<f32>>> {
        self.check_offline()?;
        let core = self.core.clone();
        let data = py
            .detach(move || {
                let mut c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
                Ok::<_, String>(c.render_stems(frames))
            })
            .map_err(rerr)?;
        let arr = Array3::from_shape_vec((N_STEMS, N_AMBI, frames), data).map_err(rerr)?;
        Ok(arr.into_pyarray(py))
    }

    /// Live output through cpal: a producer thread renders ahead into a lock-free ring.
    #[pyo3(signature = (device = None, latency_ms = 30.0))]
    fn start(&mut self, py: Python<'_>, device: Option<String>, latency_ms: f32) -> PyResult<()> {
        if self.live.as_ref().map(|l| l.is_running()).unwrap_or(false) {
            return Ok(());
        }
        let core = self.core.clone();
        let live = py.detach(move || LiveStream::start(core, device, latency_ms)).map_err(rerr)?;
        self.live = Some(live);
        Ok(())
    }

    fn stop(&mut self, py: Python<'_>) {
        if let Some(mut l) = self.live.take() {
            py.detach(move || l.stop());
        }
    }

    /// Back to t = 0 (voices, encoder, decoder, limiter, FIFOs); the graph is kept.
    fn reset(&self) -> PyResult<()> {
        self.send(Command::Reset)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let core = self.core.clone();
        let (st, underruns, latency, block_index, last_ms, max_ms, slow, scene_t, pose) = py
            .detach(move || {
                let c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
                Ok::<_, String>((
                    c.stats(),
                    c.underrun_count(),
                    c.latency(),
                    c.block_index,
                    c.last_cpu_ms(),
                    c.max_cpu_ms,
                    c.slow_blocks,
                    c.scene_time(),
                    c.listener_now(),
                ))
            })
            .map_err(rerr)?;
        let d = PyDict::new(py);
        d.set_item("realtime_factor", st.realtime_factor)?;
        d.set_item("underruns", underruns)?;
        d.set_item("voices", st.voices)?;
        d.set_item("voices_active", st.voices_active)?;
        d.set_item("dead_voices", st.dead_voices)?;
        d.set_item("cpu_ms_per_block", st.cpu_ms_per_block)?;
        d.set_item("last_cpu_ms", last_ms)?;
        d.set_item("max_cpu_ms", max_ms)?;
        d.set_item("slow_blocks", slow)?;
        d.set_item("blocks", block_index)?;
        d.set_item("latency_samples", latency)?;
        d.set_item("gain_reduction_db", st.gain_reduction_db)?;
        d.set_item("max_gain_reduction_db", st.max_gain_reduction_db)?;
        d.set_item("voice_rebuilds", st.voice_rebuilds)?;
        d.set_item("voices_kept", st.voices_kept)?;
        d.set_item("graph_updates", st.graph_updates)?;
        d.set_item("processor_swaps", st.processor_swaps)?;
        d.set_item("scene_t", scene_t)?;
        d.set_item("listener", pose.to_vec())?;
        d.set_item("threads", if self.threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0) } else { self.threads })?;
        let live = self.live.as_ref().filter(|l| l.is_running());
        d.set_item("live", live.is_some())?;
        if let Some(l) = live {
            d.set_item("device", l.device.clone())?;
            d.set_item("device_channels", l.device_channels)?;
            d.set_item("ring_frames", l.ring_frames)?;
            d.set_item("sample_format", l.sample_format.clone())?;
            let ls = l.stats();
            d.set_item("xruns", ls.xruns)?;
            d.set_item("underrun_frames", ls.underrun_frames)?;
            d.set_item("queued_frames", ls.queued_frames)?;
            d.set_item("reconnects", ls.reconnects)?;
            d.set_item("device_rate", ls.device_rate)?;
            d.set_item("resampling", ls.resampling)?;
            d.set_item("stream_failed", ls.failed)?;
        }
        Ok(d)
    }

    /// Events since the last poll: [(t, kind, data_json), ...] (lightning, thunder, gust, drip).
    fn poll_markers(&self, py: Python<'_>) -> PyResult<Vec<(f64, String, String)>> {
        let core = self.core.clone();
        let ms = py
            .detach(move || {
                let mut c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
                Ok::<_, String>(c.take_markers())
            })
            .map_err(rerr)?;
        Ok(ms.into_iter().map(|m| (m.t, m.kind.to_string(), m.data)).collect())
    }

    /// Where the time goes: {"stages": {name: ms/block}, "voices": [(name, ms/block, max ms), ...]}
    /// (smoothed; voice times are CPU time on their worker, stages are wall time).
    fn profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let core = self.core.clone();
        let (stages, voices) = py
            .detach(move || {
                let c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
                Ok::<_, String>((c.stage_ms, c.voice_profile()))
            })
            .map_err(rerr)?;
        let d = PyDict::new(py);
        let st = PyDict::new(py);
        for (i, name) in engine::STAGES.iter().enumerate() {
            st.set_item(*name, stages[i])?;
        }
        d.set_item("stages", st)?;
        d.set_item("voices", voices)?;
        Ok(d)
    }

    /// Blocks rendered since t = 0 (lock-free; for control threads pacing update()).
    fn blocks_rendered(&self) -> u64 {
        self.progress.load(Ordering::Acquire)
    }

    /// Names of the voices in the current graph (graph order).
    fn voice_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let core = self.core.clone();
        py.detach(move || {
            let mut c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
            c.drain_commands();
            Ok::<_, String>(c.voice_names())
        })
        .map_err(rerr)
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sr
    }

    #[getter]
    fn block(&self) -> usize {
        self.block
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    #[getter]
    fn layout(&self) -> String {
        self.layout.clone()
    }

    /// Output latency of render() in samples before trimming (decoder + limiter look-ahead).
    #[getter]
    fn latency(&self) -> PyResult<usize> {
        let c = self.core.lock().map_err(|_| rerr("engine lock poisoned"))?;
        Ok(c.latency())
    }

    #[getter]
    fn is_live(&self) -> bool {
        self.live.as_ref().map(|l| l.is_running()).unwrap_or(false)
    }

    #[staticmethod]
    fn list_devices() -> Vec<(String, String)> {
        stream::list_devices()
    }

    /// Add a calibration / speaker-check source to the current graph: kind "pink" | "white" |
    /// "sine", at (azimuth, elevation) rad in the listener frame, `level_db` dB SPL rms at the
    /// listener. It lives until the next set_scene().
    #[pyo3(signature = (kind = "pink", azimuth = 0.0, elevation = 0.0, spread = 0.0, level_db = 74.0, freq_hz = 1000.0, stem = "ambience"))]
    #[allow(clippy::too_many_arguments)]
    fn add_test_source(
        &self,
        kind: &str,
        azimuth: f32,
        elevation: f32,
        spread: f32,
        level_db: f32,
        freq_hz: f32,
        stem: &str,
    ) -> PyResult<()> {
        let k = match kind.trim().to_ascii_lowercase().as_str() {
            "pink" => engine::TestKind::Pink,
            "white" => engine::TestKind::White,
            "sine" | "tone" => engine::TestKind::Sine,
            other => return Err(verr(format!("unknown test source kind {other:?}"))),
        };
        let v = engine::TestSource::new(
            k,
            azimuth,
            elevation,
            spread,
            level_db,
            freq_hz,
            types::Stem::from_name(stem),
            self.sr,
            self.seed,
        );
        self.send(Command::AddVoice(Box::new(v)))
    }

    fn __repr__(&self) -> String {
        format!(
            "<raingen_native.Engine sr={} block={} layout={} channels={} live={}>",
            self.sr,
            self.block,
            self.layout,
            self.channels,
            self.is_live()
        )
    }
}

impl PyEngine {
    fn apply_output(&mut self, dec: Decoder) -> PyResult<()> {
        let ch = dec.channels();
        if self.is_live() && ch != self.channels {
            return Err(rerr(format!(
                "cannot change the channel count ({} -> {ch}) while streaming; stop() first",
                self.channels
            )));
        }
        self.channels = ch;
        self.layout = dec.layout.clone();
        self.send(Command::Output { decoder: Box::new(dec) })
    }
}

/// (id, description) of every output device of the default audio host.
#[pyfunction]
fn list_devices() -> Vec<(String, String)> {
    stream::list_devices()
}

#[pymodule]
fn raingen_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(list_devices, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SAMPLE_RATE", types::SAMPLE_RATE)?;
    m.add("BLOCK", types::BLOCK)?;
    m.add("N_AMBI", N_AMBI)?;
    m.add("STEMS", types::STEM_NAMES.to_vec())?;
    Ok(())
}
