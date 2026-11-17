//! Live output: a producer thread renders blocks ahead into a lock-free SPSC ring (rtrb), and
//! the cpal audio callback only copies from the ring (no allocation, no locks, no panics).
//!
//!   Python thread --commands--> crossbeam channel --> Core (drained per block)
//!   audio thread:    picks the device and its configuration, then owns the cpal Stream (not
//!                    Send on every platform) until stop()
//!   producer thread: lock Core, render one block, (resample to the device rate if the device
//!                    cannot run at the engine rate), push interleaved samples into the ring
//!   cpal callback:   ring -> device buffer with channel mapping and sample-format conversion
//!                    (f32 / f64 / i32 / i16 / u16 / i8 / u8); on a short read it pads with
//!                    silence and counts an underrun.
//! The ring is primed to `latency_ms` (at least two blocks and two device periods) before
//! playback starts. If the device asks for larger periods than expected, the producer raises
//! its fill target to two periods (bounded by the ring capacity) instead of underrunning.

use crate::dsp::enable_ftz;
use crate::engine::Core;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, SizedSample, StreamConfig, SupportedBufferSize};
use crossbeam_channel::{bounded, RecvTimeoutError, Sender};
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Counters shared between the audio callback, the producer and the control side.
#[derive(Default)]
pub struct StreamShared {
    pub callbacks: AtomicU64,
    pub frames_played: AtomicU64,
    pub underrun_frames: AtomicU64,
    pub max_request: AtomicUsize,
    pub blocks_rendered: AtomicU64,
    pub errors: AtomicU64,
    /// device-side under/overruns reported by the backend (also counted as engine underruns)
    pub xruns: AtomicU64,
    /// streams rebuilt after a fatal device error
    pub reconnects: AtomicU64,
    pub failed: AtomicBool,
    pub last_error: Mutex<String>,
    /// the engine's underrun counter (Engine.stats()["underruns"])
    pub engine_underruns: Arc<AtomicU64>,
    /// current device name and channel count (they change on a reconnect)
    pub device: Mutex<String>,
    pub device_channels: AtomicUsize,
}

/// Snapshot of the live stream's counters.
#[derive(Clone, Debug, Default)]
pub struct LiveStats {
    pub device: String,
    pub device_rate: u32,
    pub engine_rate: u32,
    pub device_channels: usize,
    pub engine_channels: usize,
    pub sample_format: String,
    pub ring_frames: usize,
    pub ring_capacity_frames: usize,
    pub queued_frames: usize,
    pub callbacks: u64,
    pub frames_played: u64,
    pub underrun_frames: u64,
    pub max_request_frames: usize,
    pub blocks_rendered: u64,
    pub errors: u64,
    pub xruns: u64,
    pub reconnects: u64,
    pub failed: bool,
    pub last_error: String,
    pub resampling: bool,
}

pub struct LiveStream {
    stop: Arc<AtomicBool>,
    producer: Option<JoinHandle<()>>,
    audio: Option<JoinHandle<()>>,
    audio_stop: Option<Sender<()>>,
    shared: Arc<StreamShared>,
    queued: Arc<AtomicUsize>,
    pub device: String,
    pub device_channels: usize,
    pub engine_channels: usize,
    pub ring_frames: usize,
    pub ring_capacity_frames: usize,
    pub sample_format: String,
    pub device_rate: u32,
    pub engine_rate: u32,
    pub resampling: bool,
}

// ---------------------------------------------------------------------------
// devices and configurations
// ---------------------------------------------------------------------------
/// (id, description) of every output device of the default host. Either string can be
/// passed to `Engine.start(device=...)` (exact match first, then substring).
pub fn list_devices() -> Vec<(String, String)> {
    let host = cpal::default_host();
    match host.output_devices() {
        Ok(devs) => devs.map(|d| (device_id(&d), d.to_string())).collect(),
        Err(_) => Vec::new(),
    }
}

fn device_id(d: &cpal::Device) -> String {
    d.id().map(|i| i.id().to_string()).unwrap_or_default()
}

fn device_keys(d: &cpal::Device) -> Vec<String> {
    let mut k = vec![d.to_string().to_ascii_lowercase(), device_id(d).to_ascii_lowercase()];
    if let Ok(i) = d.id() {
        k.push(i.to_string().to_ascii_lowercase());
    }
    if let Ok(desc) = d.description() {
        k.push(desc.name().to_ascii_lowercase());
    }
    k.retain(|s| !s.is_empty());
    k
}

fn pick_device(host: &cpal::Host, name: Option<&str>) -> Result<cpal::Device, String> {
    match name.map(str::trim).filter(|s| !s.is_empty() && *s != "default") {
        None => host.default_output_device().ok_or_else(|| "no default output device".to_string()),
        Some(want) => {
            let w = want.to_ascii_lowercase();
            let devs: Vec<cpal::Device> =
                host.output_devices().map_err(|e| format!("output devices: {e}"))?.collect();
            if let Some(d) = devs.iter().find(|d| device_keys(d).iter().any(|k| *k == w)) {
                return Ok(d.clone());
            }
            if let Some(d) = devs.iter().find(|d| device_keys(d).iter().any(|k| k.contains(&w))) {
                return Ok(d.clone());
            }
            let names: Vec<String> = devs.iter().map(|d| d.to_string()).collect();
            Err(format!("no output device matching {want:?}; available: {}", names.join(", ")))
        }
    }
}

#[derive(Clone)]
struct Chosen {
    cfg: StreamConfig,
    fmt: SampleFormat,
    buffer: Option<(u32, u32)>,
}

/// Every usable configuration, best first: the engine rate if supported (else the nearest
/// supported rate, and the producer resamples), then the channel count (exact > more >
/// fewer), then the sample format (f32 > i32 > i16 > f64 > u16 > 8-bit). The device's default
/// configuration is appended as a last resort.
fn rank_configs(device: &cpal::Device, sr: u32, channels: usize) -> Result<Vec<Chosen>, String> {
    let ranges: Vec<cpal::SupportedStreamConfigRange> = match device.supported_output_configs() {
        Ok(r) => r.collect(),
        Err(e) => {
            // some backends cannot enumerate; the default configuration may still work
            return match device.default_output_config() {
                Ok(d) => Ok(vec![Chosen { cfg: d.config(), fmt: d.sample_format(), buffer: None }]),
                Err(e2) => Err(format!("output configs: {e}; default config: {e2}")),
            };
        }
    };
    let mut scored: Vec<(u64, Chosen)> = Vec::new();
    for r in ranges {
        let fmt = match r.sample_format() {
            SampleFormat::F32 => 0u64,
            SampleFormat::I32 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::F64 => 3,
            SampleFormat::U16 => 4,
            SampleFormat::I8 | SampleFormat::U8 => 5,
            _ => continue,
        };
        let (lo, hi) = (r.min_sample_rate(), r.max_sample_rate());
        let rate = sr.clamp(lo, hi.max(lo));
        let rate_cost = (rate as i64 - sr as i64).unsigned_abs();
        let ch = r.channels() as usize;
        let chs = if ch == channels {
            0u64
        } else if ch > channels {
            1 + (ch - channels) as u64
        } else {
            100 + (channels - ch) as u64
        };
        let score = rate_cost * 100_000 + chs * 10 + fmt;
        let buffer = match r.buffer_size() {
            SupportedBufferSize::Range { min, max } => Some((*min, (*max).max(*min))),
            SupportedBufferSize::Unknown => None,
        };
        let fmt = r.sample_format();
        let cfg = r.with_sample_rate(rate).config();
        scored.push((score, Chosen { cfg, fmt, buffer }));
    }
    scored.sort_by_key(|(s, _)| *s);
    let mut out: Vec<Chosen> = Vec::with_capacity(scored.len() + 1);
    for (_, c) in scored {
        let dup = out.iter().any(|o| {
            o.fmt == c.fmt && o.cfg.channels == c.cfg.channels && o.cfg.sample_rate == c.cfg.sample_rate
        });
        if !dup {
            out.push(c);
        }
    }
    if let Ok(d) = device.default_output_config() {
        let c = Chosen { cfg: d.config(), fmt: d.sample_format(), buffer: None };
        let dup = out.iter().any(|o| {
            o.fmt == c.fmt && o.cfg.channels == c.cfg.channels && o.cfg.sample_rate == c.cfg.sample_rate
        });
        if !dup {
            out.push(c);
        }
    }
    if out.is_empty() {
        return Err("output device offers no usable configuration".to_string());
    }
    Ok(out)
}

/// Try the ranked configurations until one opens: first with a fixed period of about a
/// quarter of the target latency (if the device takes fixed sizes), then with the backend's
/// default period. Returns the configuration and the fixed period, if any.
fn open_config(
    device: &cpal::Device,
    sr: u32,
    channels: usize,
    latency_ms: f32,
    need_rate: Option<u32>,
    shared: &Arc<StreamShared>,
) -> Result<(Chosen, Option<u32>), String> {
    let mut errors: Vec<String> = Vec::new();
    for c in rank_configs(device, sr, channels)? {
        let rate = c.cfg.sample_rate;
        if need_rate.is_some_and(|r| r != rate) {
            continue;
        }
        if let Some((lo, hi)) = c.buffer {
            let want = ((latency_ms * 1e-3 * rate as f32 / 4.0) as u32).clamp(64, 2048).clamp(lo, hi);
            let mut cfg = c.cfg.clone();
            cfg.buffer_size = cpal::BufferSize::Fixed(want);
            if let Ok(probe) = build_any(device, cfg.clone(), c.fmt, None, shared.clone()) {
                drop(probe);
                return Ok((Chosen { cfg, fmt: c.fmt, buffer: c.buffer }, Some(want)));
            }
        }
        let mut cfg = c.cfg.clone();
        cfg.buffer_size = cpal::BufferSize::Default;
        match build_any(device, cfg.clone(), c.fmt, None, shared.clone()) {
            Ok(probe) => {
                drop(probe);
                return Ok((Chosen { cfg, fmt: c.fmt, buffer: c.buffer }, None));
            }
            Err(e) => errors.push(format!("{} ch {} Hz {:?}: {e}", c.cfg.channels, rate, c.fmt)),
        }
    }
    Err(format!("no output configuration could be opened ({})", errors.join("; ")))
}

// ---------------------------------------------------------------------------
// channel fold-down (device or engine with a different channel count)
// ---------------------------------------------------------------------------
/// (L, R) rows of a stereo fold-down of `src` channels: SH (4 / 9 / 16 ACN channels) through
/// two virtual cardioids at +-60 deg; speaker layouts in WAV order (L R C LFE, then left /
/// right pairs) by ITU-style coefficients (C and surrounds at -3 dB, LFE dropped), scaled so a
/// diffuse field keeps its level.
fn stereo_rows(src: usize) -> (Vec<f32>, Vec<f32>) {
    let mut l = vec![0.0f32; src];
    let mut r = vec![0.0f32; src];
    match src {
        0 => {}
        1 => {
            l[0] = 1.0;
            r[0] = 1.0;
        }
        2 => {
            l[0] = 1.0;
            r[1] = 1.0;
        }
        4 | 9 | 16 => {
            let a = 60f32.to_radians();
            let g = (0.5f32 / (0.25 * (1.0 + 1.0 / 3.0))).sqrt();
            // ACN 0 = W, 1 = Y (left positive), 3 = X
            l[0] = 0.5 * g;
            r[0] = 0.5 * g;
            l[1] = 0.5 * g * a.sin();
            r[1] = -0.5 * g * a.sin();
            l[3] = 0.5 * g * a.cos();
            r[3] = 0.5 * g * a.cos();
        }
        _ => {
            let h = std::f32::consts::FRAC_1_SQRT_2;
            l[0] = 1.0;
            r[1] = 1.0;
            if src > 2 {
                l[2] = h;
                r[2] = h;
            }
            for k in 4..src {
                if k % 2 == 0 {
                    l[k] = h;
                } else {
                    r[k] = h;
                }
            }
            let e: f32 = l.iter().map(|v| v * v).sum::<f32>().max(1e-12);
            let s = 1.0 / e.sqrt();
            l.iter_mut().for_each(|v| *v *= s);
            r.iter_mut().for_each(|v| *v *= s);
        }
    }
    (l, r)
}

/// dst x src mixing matrix (row-major): identity on the shared channels when the destination
/// has room (mono is copied to both front channels), else a stereo / mono fold-down; a
/// destination with more than two but fewer channels than the source keeps its first channels.
fn fold_matrix(src: usize, dst: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; dst * src];
    if src == 0 || dst == 0 {
        return m;
    }
    if dst >= src {
        for k in 0..src {
            m[k * src + k] = 1.0;
        }
        if src == 1 && dst >= 2 {
            m[src] = 1.0;
        }
        return m;
    }
    let (l, r) = stereo_rows(src);
    if dst == 1 {
        if matches!(src, 4 | 9 | 16) {
            m[0] = 1.0;
        } else {
            for k in 0..src {
                m[k] = 0.5 * (l[k] + r[k]);
            }
        }
    } else if dst == 2 {
        m[..src].copy_from_slice(&l);
        m[src..2 * src].copy_from_slice(&r);
    } else {
        for k in 0..dst {
            m[k * src + k] = 1.0;
        }
    }
    m
}

/// True when the fold matrix only copies channel k to output k (plus mono duplication).
fn is_direct(src: usize, dst: usize) -> bool {
    dst >= src
}

// ---------------------------------------------------------------------------
// resampler (only when the device cannot run at the engine rate)
// ---------------------------------------------------------------------------
/// Streaming band-limited resampler: Kaiser-windowed sinc (16 zero crossings per side,
/// cutoff 0.95 x the lower Nyquist), 256-phase table with linear interpolation between phases.
struct Resampler {
    ch: usize,
    step: f64,
    z: usize,
    phases: usize,
    table: Vec<f32>,
    buf: Vec<Vec<f32>>,
    len: usize,
    t: f64,
    coef: Vec<f32>,
}

impl Resampler {
    fn new(ch: usize, sr_in: u32, sr_out: u32, max_in: usize) -> Resampler {
        let step = sr_in as f64 / sr_out as f64;
        let cut = 0.95 * (sr_out as f64 / sr_in as f64).min(1.0);
        let z = (16.0 / cut).ceil() as usize;
        let phases = 256usize;
        let taps = 2 * z;
        let beta = 8.6f64;
        let i0 = |x: f64| {
            let (mut s, mut t) = (1.0f64, 1.0f64);
            let y = x * x / 4.0;
            for k in 1..200 {
                t *= y / (k * k) as f64;
                s += t;
                if t < 1e-17 * s {
                    break;
                }
            }
            s
        };
        let i0b = i0(beta);
        let mut table = vec![0.0f32; (phases + 1) * taps];
        for p in 0..=phases {
            let frac = p as f64 / phases as f64;
            let mut row = vec![0.0f64; taps];
            for (j, rv) in row.iter_mut().enumerate() {
                let x = j as f64 - (z as f64 - 1.0) - frac;
                let r = x / z as f64;
                if r.abs() >= 1.0 {
                    continue;
                }
                let a = std::f64::consts::PI * cut * x;
                let sinc = if a.abs() < 1e-12 { 1.0 } else { a.sin() / a };
                *rv = cut * sinc * i0(beta * (1.0 - r * r).sqrt()) / i0b;
            }
            let s: f64 = row.iter().sum();
            for j in 0..taps {
                table[p * taps + j] = (row[j] / s) as f32;
            }
        }
        let cap = taps + max_in + 4;
        Resampler {
            ch,
            step,
            z,
            phases,
            table,
            buf: vec![vec![0.0; cap]; ch],
            len: z - 1,
            t: (z - 1) as f64,
            coef: vec![0.0; taps],
        }
    }

    fn max_out(&self, n_in: usize) -> usize {
        (n_in as f64 / self.step).ceil() as usize + 2
    }

    /// `input`: n_in interleaved frames; writes interleaved frames into `out`, returns the count.
    fn process(&mut self, input: &[f32], n_in: usize, out: &mut [f32]) -> usize {
        let ch = self.ch;
        let taps = 2 * self.z;
        for (c, b) in self.buf.iter_mut().enumerate() {
            for i in 0..n_in {
                b[self.len + i] = input[i * ch + c];
            }
        }
        self.len += n_in;
        let max_frames = out.len() / ch;
        let mut produced = 0;
        loop {
            let i0 = self.t.floor() as usize;
            if i0 + self.z >= self.len || produced >= max_frames {
                break;
            }
            let pf = (self.t - i0 as f64) * self.phases as f64;
            let p = (pf.floor() as usize).min(self.phases - 1);
            let w = (pf - p as f64) as f32;
            let r0 = &self.table[p * taps..(p + 1) * taps];
            let r1 = &self.table[(p + 1) * taps..(p + 2) * taps];
            for j in 0..taps {
                self.coef[j] = r0[j] + w * (r1[j] - r0[j]);
            }
            let base = i0 + 1 - self.z;
            for c in 0..ch {
                let x = &self.buf[c][base..base + taps];
                let mut acc = 0.0f32;
                for (a, b) in x.iter().zip(&self.coef) {
                    acc += a * b;
                }
                out[produced * ch + c] = acc;
            }
            produced += 1;
            self.t += self.step;
        }
        // drop what no future output needs
        let i0 = self.t.floor() as usize;
        let keep_from = (i0 + 1).saturating_sub(self.z).min(self.len);
        if keep_from > 0 {
            for b in &mut self.buf {
                b.copy_within(keep_from..self.len, 0);
            }
            self.len -= keep_from;
            self.t -= keep_from as f64;
        }
        produced
    }
}

// ---------------------------------------------------------------------------
// the audio callback
// ---------------------------------------------------------------------------
struct CallbackState {
    cons: Consumer<f32>,
    eng_ch: usize,
    dev_ch: usize,
    /// dev_ch x eng_ch fold-down, used when the device has fewer channels than the engine
    mix: Vec<f32>,
    direct: bool,
    underruns: Arc<AtomicU64>,
    shared: Arc<StreamShared>,
}

#[inline]
fn write_frames<T: SizedSample + FromSample<f32>>(
    src: &[f32],
    dst: &mut [T],
    eng_ch: usize,
    dev_ch: usize,
    mix: &[f32],
    direct: bool,
) {
    let conv = |v: f32| T::from_sample(if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 });
    if direct {
        for (frame, row) in src.chunks_exact(eng_ch).zip(dst.chunks_exact_mut(dev_ch)) {
            for (o, s) in row.iter_mut().zip(frame) {
                *o = conv(*s);
            }
            if eng_ch == 1 && dev_ch >= 2 {
                row[1] = conv(frame[0]);
                for o in row[2..].iter_mut() {
                    *o = T::EQUILIBRIUM;
                }
            } else {
                for o in row[eng_ch..].iter_mut() {
                    *o = T::EQUILIBRIUM;
                }
            }
        }
    } else {
        for (frame, row) in src.chunks_exact(eng_ch).zip(dst.chunks_exact_mut(dev_ch)) {
            for (o, m) in row.iter_mut().zip(mix.chunks_exact(eng_ch)) {
                let mut acc = 0.0f32;
                for (a, b) in frame.iter().zip(m) {
                    acc += a * b;
                }
                *o = conv(acc);
            }
        }
    }
}

impl CallbackState {
    #[inline]
    fn fill<T: SizedSample + FromSample<f32>>(&mut self, data: &mut [T]) {
        let dev_ch = self.dev_ch.max(1);
        let eng_ch = self.eng_ch.max(1);
        let frames = data.len() / dev_ch;
        self.shared.callbacks.fetch_add(1, Ordering::Relaxed);
        self.shared.max_request.fetch_max(frames, Ordering::Relaxed);
        let avail = self.cons.slots() / eng_ch;
        let take = frames.min(avail);
        if take < frames {
            self.underruns.fetch_add(1, Ordering::Relaxed);
            self.shared.underrun_frames.fetch_add((frames - take) as u64, Ordering::Relaxed);
        }
        if take > 0 {
            if let Ok(chunk) = self.cons.read_chunk(take * eng_ch) {
                {
                    let (a, b) = chunk.as_slices();
                    // the ring holds whole frames and its capacity is a multiple of eng_ch,
                    // so the wrap point is frame-aligned
                    let fa = a.len() / eng_ch;
                    let (mix, direct) = (&self.mix[..], self.direct);
                    write_frames(a, &mut data[..fa * dev_ch], eng_ch, dev_ch, mix, direct);
                    write_frames(b, &mut data[fa * dev_ch..take * dev_ch], eng_ch, dev_ch, mix, direct);
                }
                chunk.commit_all();
            }
            self.shared.frames_played.fetch_add(take as u64, Ordering::Relaxed);
        }
        for v in data[take * dev_ch..].iter_mut() {
            *v = T::EQUILIBRIUM;
        }
    }
}

/// Backend errors: an xrun is an audible glitch (counted as an underrun, the stream goes on);
/// a route change or a refused real-time promotion is only noted; anything else marks the
/// stream failed, and the audio thread rebuilds it.
fn error_callback(shared: Arc<StreamShared>) -> impl FnMut(cpal::Error) + Send + 'static {
    move |err| match err.kind() {
        ErrorKind::Xrun => {
            shared.xruns.fetch_add(1, Ordering::Relaxed);
            shared.engine_underruns.fetch_add(1, Ordering::Relaxed);
        }
        ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied => {
            if let Ok(mut s) = shared.last_error.try_lock() {
                *s = err.to_string();
            }
        }
        _ => {
            shared.errors.fetch_add(1, Ordering::Relaxed);
            shared.failed.store(true, Ordering::Release);
            if let Ok(mut s) = shared.last_error.lock() {
                *s = err.to_string();
            }
        }
    }
}

fn build_typed<T: SizedSample + FromSample<f32> + Send + 'static>(
    device: &cpal::Device,
    cfg: StreamConfig,
    st: Option<Arc<Mutex<CallbackState>>>,
    shared: Arc<StreamShared>,
) -> Result<cpal::Stream, String> {
    let r = match st {
        // the state is only ever locked by one stream's callback at a time (the audio thread
        // swaps streams after dropping the old one): try_lock never waits
        Some(st) => device.build_output_stream::<T, _, _>(
            cfg,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| match st.try_lock() {
                Ok(mut s) => s.fill(data),
                Err(_) => data.iter_mut().for_each(|v| *v = T::EQUILIBRIUM),
            },
            error_callback(shared),
            None,
        ),
        // probe: a silent stream with the same configuration (never played)
        None => device.build_output_stream::<T, _, _>(
            cfg,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| data.iter_mut().for_each(|v| *v = T::EQUILIBRIUM),
            error_callback(shared),
            None,
        ),
    };
    r.map_err(|e| format!("build output stream: {e}"))
}

fn build_any(
    device: &cpal::Device,
    cfg: StreamConfig,
    fmt: SampleFormat,
    st: Option<Arc<Mutex<CallbackState>>>,
    shared: Arc<StreamShared>,
) -> Result<cpal::Stream, String> {
    match fmt {
        SampleFormat::F32 => build_typed::<f32>(device, cfg, st, shared),
        SampleFormat::F64 => build_typed::<f64>(device, cfg, st, shared),
        SampleFormat::I32 => build_typed::<i32>(device, cfg, st, shared),
        SampleFormat::I16 => build_typed::<i16>(device, cfg, st, shared),
        SampleFormat::U16 => build_typed::<u16>(device, cfg, st, shared),
        SampleFormat::I8 => build_typed::<i8>(device, cfg, st, shared),
        SampleFormat::U8 => build_typed::<u8>(device, cfg, st, shared),
        other => Err(format!("unsupported sample format {other:?}")),
    }
}

/// What the audio thread negotiated with the device.
struct DevInfo {
    name: String,
    rate: u32,
    channels: usize,
    fmt: String,
    period: Option<u32>,
}

fn push_all(prod: &mut Producer<f32>, src: &[f32]) -> bool {
    if src.is_empty() {
        return true;
    }
    match prod.write_chunk(src.len()) {
        Ok(mut chunk) => {
            let (a, b) = chunk.as_mut_slices();
            let la = a.len();
            a.copy_from_slice(&src[..la]);
            b.copy_from_slice(&src[la..la + b.len()]);
            chunk.commit_all();
            true
        }
        Err(_) => false,
    }
}

/// Renders blocks at the engine rate and converts them to device frames.
struct BlockSource {
    core: Arc<Mutex<Core>>,
    eng_ch: usize,
    n: usize,
    block: Vec<f32>,
    out: Vec<f32>,
    rs: Option<Resampler>,
    /// (core channels, eng_ch x core channels) remap if the engine's layout changed under
    /// the stream (never panics; built once per change, outside the audio callback)
    remap: Option<(usize, Vec<f32>)>,
}

impl BlockSource {
    /// Largest number of device frames one block can produce.
    fn max_frames(&self) -> usize {
        self.rs.as_ref().map(|r| r.max_out(self.n)).unwrap_or(self.n)
    }

    /// Render one block; returns the interleaved device frames.
    fn next(&mut self, shared: &StreamShared) -> &[f32] {
        let n = self.n;
        let ec = self.eng_ch;
        let ok = match self.core.lock() {
            Ok(mut c) => {
                let r = catch_unwind(AssertUnwindSafe(|| c.render_block()));
                if r.is_err() {
                    false
                } else {
                    let ch = c.channels();
                    let src = c.out_block();
                    if ch == ec && src.len() >= ch * n {
                        for i in 0..n {
                            for k in 0..ch {
                                self.block[i * ch + k] = src[k * n + i];
                            }
                        }
                        true
                    } else if ch > 0 && src.len() >= ch * n {
                        if self.remap.as_ref().map(|(c0, _)| *c0 != ch).unwrap_or(true) {
                            self.remap = Some((ch, fold_matrix(ch, ec)));
                        }
                        let m = &self.remap.as_ref().map(|(_, m)| m.as_slice()).unwrap_or(&[]);
                        for i in 0..n {
                            for o in 0..ec {
                                let row = &m[o * ch..(o + 1) * ch];
                                let mut acc = 0.0f32;
                                for k in 0..ch {
                                    acc += row[k] * src[k * n + i];
                                }
                                self.block[i * ec + o] = acc;
                            }
                        }
                        true
                    } else {
                        false
                    }
                }
            }
            Err(_) => false,
        };
        if !ok {
            // the engine failed (or is poisoned): silence instead of a dead stream
            self.block.iter_mut().for_each(|v| *v = 0.0);
            shared.errors.fetch_add(1, Ordering::Relaxed);
        }
        shared.blocks_rendered.fetch_add(1, Ordering::Relaxed);
        match &mut self.rs {
            None => &self.block,
            Some(r) => {
                let k = r.process(&self.block, self.n, &mut self.out);
                &self.out[..k * self.eng_ch]
            }
        }
    }
}

/// Rebuild the output after a fatal device error: the requested device (or, if it is gone,
/// the default one) at the stream's device rate, any channel count.
#[allow(clippy::too_many_arguments)]
fn reopen(
    host: &cpal::Host,
    name: Option<&str>,
    sr: u32,
    eng_ch: usize,
    rate: u32,
    latency_ms: f32,
    sh: &Arc<StreamShared>,
    state: &Arc<Mutex<CallbackState>>,
) -> Result<cpal::Stream, String> {
    let dev = pick_device(host, name).or_else(|_| pick_device(host, None))?;
    let (chosen, _) = open_config(&dev, sr, eng_ch, latency_ms, Some(rate), sh)?;
    let dev_ch = (chosen.cfg.channels as usize).max(1);
    {
        let mut st = state.lock().map_err(|_| "callback state poisoned".to_string())?;
        st.dev_ch = dev_ch;
        st.mix = fold_matrix(eng_ch, dev_ch);
        st.direct = is_direct(eng_ch, dev_ch);
    }
    let stream = build_any(&dev, chosen.cfg.clone(), chosen.fmt, Some(state.clone()), sh.clone())?;
    stream.play().map_err(|e| format!("play: {e}"))?;
    if let Ok(mut d) = sh.device.lock() {
        *d = dev.to_string();
    }
    sh.device_channels.store(dev_ch, Ordering::Relaxed);
    Ok(stream)
}

impl LiveStream {
    pub fn start(core: Arc<Mutex<Core>>, device: Option<String>, latency_ms: f32) -> Result<LiveStream, String> {
        let (eng_ch, sr, n, underruns) = {
            let mut c = core.lock().map_err(|_| "engine lock poisoned".to_string())?;
            c.drain_commands();
            (c.channels(), c.sr, c.n, c.underruns.clone())
        };
        let latency_ms = if latency_ms.is_finite() { latency_ms.clamp(1.0, 2000.0) } else { 30.0 };
        let shared = Arc::new(StreamShared { engine_underruns: underruns.clone(), ..Default::default() });

        // 1. audio thread: device + configuration, then waits for the primed ring; afterwards
        //    it watches for fatal device errors and rebuilds the stream (same rate, any channel
        //    count: the callback folds or spreads the engine channels)
        let (info_tx, info_rx) = bounded::<Result<DevInfo, String>>(1);
        let (cons_tx, cons_rx) = bounded::<Consumer<f32>>(1);
        let (res_tx, res_rx) = bounded::<Result<(), String>>(1);
        let (stop_tx, stop_rx) = bounded::<()>(1);
        let dev_name = device.clone();
        let sh = shared.clone();
        let ur = underruns.clone();
        let audio = std::thread::Builder::new()
            .name("raingen-audio".into())
            .spawn(move || {
                let host = cpal::default_host();
                let dev = match pick_device(&host, dev_name.as_deref()) {
                    Ok(d) => d,
                    Err(e) => {
                        let _ = info_tx.send(Err(e));
                        return;
                    }
                };
                let (chosen, period) = match open_config(&dev, sr, eng_ch, latency_ms, None, &sh) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = info_tx.send(Err(e));
                        return;
                    }
                };
                // the probe streams never played; the real stream starts with a clean slate
                sh.failed.store(false, Ordering::Release);
                sh.errors.store(0, Ordering::Relaxed);
                let rate = chosen.cfg.sample_rate;
                let dev_ch = (chosen.cfg.channels as usize).max(1);
                let cfg = chosen.cfg.clone();
                let info = DevInfo { name: dev.to_string(), rate, channels: dev_ch, fmt: format!("{:?}", chosen.fmt), period };
                if let Ok(mut d) = sh.device.lock() {
                    *d = info.name.clone();
                }
                sh.device_channels.store(dev_ch, Ordering::Relaxed);
                if info_tx.send(Ok(info)).is_err() {
                    return;
                }
                let Ok(cons) = cons_rx.recv() else { return };
                let state = Arc::new(Mutex::new(CallbackState {
                    cons,
                    eng_ch,
                    dev_ch,
                    mix: fold_matrix(eng_ch, dev_ch),
                    direct: is_direct(eng_ch, dev_ch),
                    underruns: ur,
                    shared: sh.clone(),
                }));
                let stream = match build_any(&dev, cfg, chosen.fmt, Some(state.clone()), sh.clone()) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = res_tx.send(Err(e));
                        return;
                    }
                };
                if let Err(e) = stream.play() {
                    let _ = res_tx.send(Err(format!("play: {e}")));
                    return;
                }
                let _ = res_tx.send(Ok(()));
                let mut stream = Some(stream);
                let mut next_try = Instant::now();
                loop {
                    match stop_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                    if !sh.failed.load(Ordering::Acquire) || Instant::now() < next_try {
                        continue;
                    }
                    if let Some(s) = stream.take() {
                        let _ = s.pause();
                        drop(s);
                    }
                    next_try = Instant::now() + Duration::from_secs(1);
                    match reopen(&host, dev_name.as_deref(), sr, eng_ch, rate, latency_ms, &sh, &state) {
                        Ok(s) => {
                            stream = Some(s);
                            sh.reconnects.fetch_add(1, Ordering::Relaxed);
                            sh.failed.store(false, Ordering::Release);
                        }
                        Err(e) => {
                            if let Ok(mut l) = sh.last_error.lock() {
                                *l = e;
                            }
                        }
                    }
                }
                if let Some(s) = stream.take() {
                    let _ = s.pause();
                    drop(s);
                }
            })
            .map_err(|e| format!("spawn audio thread: {e}"))?;

        let info = match info_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(i)) => i,
            Ok(Err(e)) => {
                let _ = audio.join();
                return Err(e);
            }
            Err(_) => {
                drop(cons_tx);
                return Err("audio device did not answer within 10 s".into());
            }
        };

        // 2. ring sized in device frames, primed before playback
        let rate = info.rate;
        let rs = if rate != sr { Some(Resampler::new(eng_ch, sr, rate, n)) } else { None };
        let resampling = rs.is_some();
        let mut src = BlockSource {
            core: core.clone(),
            eng_ch,
            n,
            block: vec![0.0; n * eng_ch],
            out: Vec::new(),
            rs,
            remap: None,
        };
        let max_out = src.max_frames();
        src.out = vec![0.0; max_out * eng_ch];
        let period = info.period.map(|p| p as usize).unwrap_or(0);
        let want = (latency_ms * 1e-3 * rate as f32).ceil() as usize;
        let ring_frames = want.max(2 * max_out).max(2 * period);
        // room to raise the fill target to two device periods of up to 8192 frames
        let capacity_frames = (ring_frames.max(16_384) + max_out).max(ring_frames + max_out);
        let (mut prod, cons) = RingBuffer::<f32>::new(capacity_frames * eng_ch);
        {
            let queued = |p: &Producer<f32>| (p.buffer().capacity() - p.slots()) / eng_ch;
            while queued(&prod) < ring_frames && prod.slots() >= max_out * eng_ch {
                let blk = src.next(&shared);
                push_all(&mut prod, blk);
            }
        }
        if cons_tx.send(cons).is_err() {
            let _ = audio.join();
            return Err("audio thread exited before the stream was built".into());
        }
        match res_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = audio.join();
                return Err(e);
            }
            Err(_) => {
                let _ = stop_tx.send(());
                return Err("audio stream did not start within 10 s".into());
            }
        }

        // 3. producer: keeps the ring at its fill target
        let stop = Arc::new(AtomicBool::new(false));
        let queued_frames = Arc::new(AtomicUsize::new(0));
        let stop_p = stop.clone();
        let sh = shared.clone();
        let qf = queued_frames.clone();
        let block_dur = Duration::from_secs_f64(n as f64 / sr as f64);
        let nap = (block_dur / 4).clamp(Duration::from_micros(500), Duration::from_millis(5));
        let producer = std::thread::Builder::new()
            .name("raingen-producer".into())
            .spawn(move || {
                enable_ftz();
                let cap_samples = prod.buffer().capacity();
                while !stop_p.load(Ordering::Acquire) {
                    let queued = (cap_samples - prod.slots()) / eng_ch;
                    qf.store(queued, Ordering::Relaxed);
                    let req = sh.max_request.load(Ordering::Relaxed);
                    let target = ring_frames.max(2 * req).min(capacity_frames - max_out);
                    if queued < target && prod.slots() >= max_out * eng_ch {
                        let blk = src.next(&sh);
                        push_all(&mut prod, blk);
                    } else {
                        std::thread::sleep(nap);
                    }
                }
            })
            .map_err(|e| format!("spawn producer thread: {e}"))?;

        Ok(LiveStream {
            stop,
            producer: Some(producer),
            audio: Some(audio),
            audio_stop: Some(stop_tx),
            shared,
            queued: queued_frames,
            device: info.name,
            device_channels: info.channels,
            engine_channels: eng_ch,
            ring_frames,
            ring_capacity_frames: capacity_frames,
            sample_format: info.fmt,
            device_rate: rate,
            engine_rate: sr,
            resampling,
        })
    }

    pub fn is_running(&self) -> bool {
        self.producer.is_some()
    }

    /// True once the device reported an error (e.g. it was unplugged).
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> LiveStats {
        let device = self.shared.device.lock().map(|d| d.clone()).unwrap_or_else(|_| self.device.clone());
        let dch = self.shared.device_channels.load(Ordering::Relaxed);
        LiveStats {
            device,
            device_rate: self.device_rate,
            engine_rate: self.engine_rate,
            device_channels: if dch > 0 { dch } else { self.device_channels },
            engine_channels: self.engine_channels,
            sample_format: self.sample_format.clone(),
            ring_frames: self.ring_frames,
            ring_capacity_frames: self.ring_capacity_frames,
            queued_frames: self.queued.load(Ordering::Relaxed),
            callbacks: self.shared.callbacks.load(Ordering::Relaxed),
            frames_played: self.shared.frames_played.load(Ordering::Relaxed),
            underrun_frames: self.shared.underrun_frames.load(Ordering::Relaxed),
            max_request_frames: self.shared.max_request.load(Ordering::Relaxed),
            blocks_rendered: self.shared.blocks_rendered.load(Ordering::Relaxed),
            errors: self.shared.errors.load(Ordering::Relaxed),
            xruns: self.shared.xruns.load(Ordering::Relaxed),
            reconnects: self.shared.reconnects.load(Ordering::Relaxed),
            failed: self.failed(),
            last_error: self.shared.last_error.lock().map(|s| s.clone()).unwrap_or_default(),
            resampling: self.resampling,
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(tx) = self.audio_stop.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.audio.take() {
            let _ = h.join();
        }
        if let Some(h) = self.producer.take() {
            let _ = h.join();
        }
    }
}

impl Drop for LiveStream {
    fn drop(&mut self) {
        self.stop();
    }
}
