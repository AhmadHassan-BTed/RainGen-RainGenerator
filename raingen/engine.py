"""Real-time playback: the Python-era live tier (MASTER_PLAN 3.4, 9.5).

Architecture (Bencina, "Real-time audio programming 101"): the audio callback never
renders, allocates large buffers, waits on locks or touches Python objects other than a
FIFO of ready numpy blocks. A producer thread renders ahead with the same BlockRenderer
the offline renderer uses (reduced level of detail by default, quality 'game'), runs the
master chain (CalmGovernor for wellness modes, calibration gain, true-peak limiter) and
appends interleaved float32 blocks to a deque; the callback only copies them out.

Control: `set(path, value, ramp)` is safe from any thread. Values are coerced and clamped
with the manifest (strings may carry units: "30 km/h", "force 6", "8 mm/h"), then queued
(multi-producer). The producer applies them between blocks:

* weather parameters glide in the manifest's perceptual domain (log2 for rain rate, the
  Beaufort number for wind, shortest arc for directions) over max(ramp, manifest smoothing
  time), e.g. rain 3 s and wind 1 s, so physics never sees a step;
* structural changes (place, shelter, surfaces, listener, precipitation type, mode) build a
  new pipeline on a worker thread and crossfade to it with an equal-power fade, so the old
  scene keeps playing while the new one is prepared. With the native core they are an
  incremental update of the one running pipeline instead: the new voice graph is diffed in
  Rust (unchanged voices keep playing, added / removed ones crossfade over 150 ms) and the
  World keeps its weather state; only an output layout change builds a new pipeline;
* the listener pose (`set_listener`: position, heading, pitch) is live on the native core:
  a lock-free pose the voices follow per block, never a rebuild;
* `strike()` fires a lightning strike now, `seek(t)` moves the scene clock without a restart;
* Tier-0 macros (vibe.intensity / closeness / wildness) go through apply_vibe and then
  through the same diff, `mode` through apply_mode, `vibe.volume_db` ramps the output gain.

Loudness: the first `prime_s` seconds are rendered before playback starts and their
integrated loudness (BS.1770-5) sets one static calibration gain toward `target_lufs`
(None keeps the engine's physical calibration, digital 1.0 = 1 Pa = 94 dB SPL, see
raingen.core.constants.PA_PER_FULL_SCALE). Later changes are heard at their physical
level: heavier rain is louder, a closed window is quieter.

Markers (lightning, thunder, gusts, drips) are delivered to `on_marker` by `poll()` on the
caller's thread when the audio carrying them is played, never from the engine's threads.

Native core: when the compiled `raingen_native` module imports (and RAINGEN_NATIVE is not 0),
every pipeline is a raingen.native.NativeSession: the World still steps in Python, but all
voices, encoding, shelter transfer and decoding run in Rust on every core. `native=False`
(or a layout above third order) keeps the Python BlockRenderer.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .core.constants import BLOCK, SAMPLE_RATE
from .scene.schema import Scene

log = logging.getLogger("raingen.engine")

__all__ = ["Engine", "PATH_ALIASES"]

# Friendly or plan-style ids -> stored scene paths.
PATH_ALIASES = {
    "listener.shelter": "place.shelter",
    "shelter": "place.shelter",
    "weather.wind.speed_m_s": "weather.wind.speed_mps",
    "wind": "weather.wind.speed_mps",
    "rain": "weather.precipitation.rate_mm_h",
    "weather.precipitation.rate": "weather.precipitation.rate_mm_h",
    "thunder.distance_km": "weather.thunder.distance_km",
    "intensity": "vibe.intensity",
    "closeness": "vibe.closeness",
    "wildness": "vibe.wildness",
    "volume": "vibe.volume_db",
}
_LIVE_PLACE = ("window_open", "aftermath_tau_min")
# place.biophony.* levels are live (Engine.set_ambience on the native core); only the region
# changes which species are built.
_BIOPHONY_STRUCT = ("region",)
_MACROS = ("intensity", "closeness", "wildness")


def _sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except ImportError as exc:
        raise ImportError("real-time playback needs the optional 'sounddevice' package: "
                          "pip install 'rain-generator[audio]' (or: pip install sounddevice)") from exc
    except Exception as exc:  # PortAudio missing on a headless machine
        raise ImportError(f"sounddevice could not load PortAudio: {exc}") from exc
    return sd


def _live_paths() -> tuple[tuple[str, ...], tuple[str, ...]]:
    from .scene import manifest as mf

    numeric, discrete = [], []
    for s in mf.PARAMS:
        # precipitation.type is live within a family; _structure_key turns a cross-family change
        # (rain <-> snow / hail) into a rebuild.
        if s.id.startswith("weather."):
            (numeric if s.is_numeric else discrete).append(s.id)
        elif s.id in tuple(f"place.{p}" for p in _LIVE_PLACE):
            numeric.append(s.id)
        elif s.id.startswith("place.biophony.") and s.id.rsplit(".", 1)[1] not in _BIOPHONY_STRUCT:
            (discrete if getattr(s, "auto_level", False) or not s.is_numeric else numeric).append(s.id)
    return tuple(numeric), tuple(discrete)


def _safe_get(scene: Scene, path: str, default: Any = None) -> Any:
    try:
        return scene.get(path)
    except (AttributeError, KeyError, TypeError):
        return default


_LIVE_LISTENER = ("position", "heading_deg", "pitch_deg")


# Liquid rain types share the same generators (only the drop-size distribution and timing
# differ, and those glide), so switching between them must not rebuild voices.
_PRECIP_FAMILY = {"drizzle": "liquid", "rain": "liquid", "showers": "liquid"}
# Extra render-ahead around a structural update (native core): the new voices' first blocks and
# the old/new voice crossfade cost up to ~2-4x a normal block for ~150 ms; this much audio is
# rendered ahead first so that never reaches the output as an underrun.
_HEADROOM_S = 0.25
_HEADROOM_WAIT_S = 0.35              # at most this long waiting for the headroom before sending
_HEADROOM_HOLD_S = 0.35              # keep it through the voice fade (150 ms) and the first blocks


def _structure_key(scene: Scene, live_listener: bool = False, none_live: bool = False) -> str:
    """Everything that needs a new pipeline when it changes (geometry, generators, master).

    live_listener: the pipeline follows the listener pose live (native core), so position and
    heading are not structure. none_live: 'none' precipitation is in the liquid family (a live
    native pipeline keeps its rain voices built and silent then; see native.scene_json 'live').
    """
    d = scene.to_dict()
    vibe = d.get("vibe") or {}
    skip = ("render",) + (_LIVE_LISTENER if live_listener else ())
    ptype = d["weather"]["precipitation"]["type"]
    key = {
        "type": "liquid" if (none_live and ptype == "none") else _PRECIP_FAMILY.get(ptype, ptype),
        "place": {k: ({b: x for b, x in v.items() if b in _BIOPHONY_STRUCT} if k == "biophony" and isinstance(v, dict) else v)
                  for k, v in d["place"].items() if k not in _LIVE_PLACE},
        "listener": {k: v for k, v in d["listener"].items() if k not in skip},
        "mode": d.get("mode"),
        "seed": d.get("seed"),
        "quality": (d.get("render") or {}).get("quality"),
        "timer": vibe.get("timer_min"),
        "descent": vibe.get("descent_min"),
        "tingle": vibe.get("tingle"),
    }
    return json.dumps(key, sort_keys=True, default=str)


class _Delay:
    """Fixed integer delay (C, n) -> (C, n), state carried between blocks."""

    def __init__(self, samples: int, channels: int):
        self.samples = max(int(samples), 0)
        self._buf = np.zeros((int(channels), self.samples))

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.samples == 0:
            return x
        ext = np.concatenate([self._buf, x], axis=1)
        n = x.shape[1]
        self._buf = ext[:, n:].copy()
        return ext[:, :n]


class _GovernorSlot:
    """The CalmGovernor (or none) padded to a fixed latency, so changing mode never moves the timeline.

    Every path through the slot is delayed by the same `latency` (a governor's own look-ahead
    plus padding), and a mode change crossfades the old path into the new one with the
    structural crossfade's equal-power curve: no gap, no click, markers stay aligned.
    """

    def __init__(self, governor, channels: int, latency: int):
        self.channels = int(channels)
        self.latency = max(int(latency), int(getattr(governor, "latency", 0) or 0))
        self.governor = governor
        self._pad = _Delay(self.latency - int(getattr(governor, "latency", 0) or 0), channels)
        self._old = None                     # (governor, pad, pos, len) while crossfading

    def replace(self, governor, xfade: int) -> None:
        if xfade <= 0:
            self.governor = governor
            self._pad = _Delay(self.latency - int(getattr(governor, "latency", 0) or 0), self.channels)
            self._old = None
            return
        self._old = [self.governor, self._pad, 0, int(xfade)]
        self.governor = governor
        self._pad = _Delay(max(self.latency - int(getattr(governor, "latency", 0) or 0), 0), self.channels)

    @staticmethod
    def _run(governor, pad: _Delay, x: np.ndarray) -> np.ndarray:
        return pad.process(governor.process(x) if governor is not None else x)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self._run(self.governor, self._pad, x)
        if self._old is not None:
            gov, pad, pos, length = self._old
            y_old = self._run(gov, pad, x)
            n = x.shape[1]
            w = np.clip((pos + np.arange(1, n + 1)) / length, 0.0, 1.0)
            y = y_old * np.cos(0.5 * math.pi * w)[None, :] + y * np.sin(0.5 * math.pi * w)[None, :]
            self._old[2] = pos + n
            if self._old[2] >= length:
                self._old = None
        return y


@dataclass
class _Ramp:
    path: str
    spec: Any
    p0: float
    p1: float
    start: int
    blocks: int
    final: Any


class _GraphProxy:
    """The engine's ParameterGraph (Auto / Offset / Pinned) with thread-safe mutators."""

    _MUTATORS = frozenset({"set_manual", "set_offset", "nudge", "relink", "relink_all", "undo", "redo"})

    def __init__(self, engine: "Engine"):
        self._engine = engine

    def __getattr__(self, name: str):
        eng = self._engine
        attr = getattr(eng._graph, name)
        if name in self._MUTATORS and callable(attr):
            def call(*args, **kwargs):
                with eng._lock:
                    result = getattr(eng._graph, name)(*args, **kwargs)
                eng._cmd.put(("retarget", 0.5))
                return result
            return call
        return attr


class Engine:
    """Real-time weather playback. Use as a context manager or call start()/stop().

    scene      Scene, preset name, scene file path or plain-words text
    layout     stereo | binaural | 5.1 | 7.1 | 7.1.4 | foa | hoa3 | mono (device channels must match)
    device     sounddevice output device (index or name substring); None = default
    blocksize  device callback size in frames (the engine renders BLOCK frames internally)

    Keyword-only: buffer_s (render-ahead), sample_rate (device rate; resampled from 48 kHz),
    quality ('game' by default for real time), target_lufs (calibration at start, None = off),
    ceiling_dbtp, volume_db, on_marker(marker), xfade_s (structural crossfade), prime_s,
    latency (sounddevice latency hint), osc (port or (host, port) to serve OSC control),
    honor_timer (sleep timer and Descent fade, on by default live), native (None = the Rust core
    when installed, unless RAINGEN_NATIVE=0; False = the Python generators).
    """

    def __init__(self, scene, layout: Optional[str] = "stereo", device=None, blocksize: int = BLOCK, *,
                 buffer_s: float = 0.5, sample_rate: Optional[int] = None, quality: Optional[str] = "game",
                 target_lufs: Optional[float] = -20.0, ceiling_dbtp: float = -1.0, volume_db: float = 0.0,
                 on_marker: Optional[Callable] = None, xfade_s: float = 0.75, prime_s: float = 1.5,
                 latency: Any = "high", osc=None, honor_timer: bool = True, native: Optional[bool] = None):
        from .scene.graph import ParameterGraph

        if not isinstance(scene, Scene):
            from .api import load_scene
            scene = load_scene(scene)
        target = scene.copy()
        if quality:
            target.render.quality = str(quality).lower()
        from .wellness.modes import apply_mode, canonical_mode
        if canonical_mode(target.mode) != "none" and "guard" not in (target.vibe or {}):
            target = apply_mode(target, target.mode)
        self._target = target
        self._graph = ParameterGraph(self._target)
        self.graph = _GraphProxy(self)
        self._quality = str(quality).lower() if quality else None
        from .render import resolve_layout
        self.layout = resolve_layout(target, layout)
        from . import native as _native
        self.native = bool(_native.enabled(native) and _native.supports_layout(self.layout))
        self.device = device
        self.blocksize = int(blocksize)
        self.buffer_s = max(float(buffer_s), BLOCK / SAMPLE_RATE)
        self.sr_out = int(sample_rate or SAMPLE_RATE)
        self.target_lufs = target_lufs
        self.ceiling_dbtp = float(ceiling_dbtp)
        self.on_marker = on_marker
        self.xfade_s = max(float(xfade_s), 0.0)
        self.prime_s = max(float(prime_s), 0.5)
        self.latency = latency
        self.honor_timer = bool(honor_timer)
        self._osc_cfg = osc
        self._osc = None

        self._lock = threading.RLock()
        self._cmd: "queue.SimpleQueue" = queue.SimpleQueue()
        self._fifo: "collections.deque" = collections.deque()
        self._stop = threading.Event()
        self._primed = threading.Event()
        self._producer: Optional[threading.Thread] = None
        self._stream = None
        self._running = False
        self.error: Optional[BaseException] = None

        self._live_numeric, self._live_discrete = _live_paths()
        self._pipe = None
        self._eff: Optional[Scene] = None
        self._current: Optional[Scene] = None
        self._struct_key = ""
        self._ramps: dict[str, _Ramp] = {}
        self._index = 0
        self._xf: Optional[dict] = None
        self._build_gen = 0
        self._built: Optional[tuple] = None
        self._governor = None
        self._gov_slot: Optional[_GovernorSlot] = None
        self._gov_key: tuple = ()
        self._limiter = None
        self._resampler = None
        self._cal_db = 0.0
        self._vol_db = float(volume_db)
        self._vol_from = float(volume_db)
        self._vol_to = float(volume_db)
        self._vol_start = 0
        self._vol_blocks = 1
        self.channels = 0
        self._chain_latency = 0

        self._cur: Optional[np.ndarray] = None
        self._cur_pos = 0
        self._played = 0                     # device frames handed out (incl. underrun silence)
        self._consumed = 0                   # rendered frames handed out (the audio's own clock)
        self._pushed = 0                     # rendered frames queued by the producer
        self._g_prev: Optional[float] = None  # output gain at the end of the last block (per-sample ramps)
        self.underruns = 0
        self.xruns = 0
        self.cpu_load = 0.0
        self.slow_blocks = 0                 # render blocks that took longer than real time
        self._block_ms: "collections.deque" = collections.deque(maxlen=64)   # last ~1.4 s of block times
        self._markers: list = []             # (render-clock time it is heard, marker on the scene clock)
        self._marker_lock = threading.Lock()
        self.markers_delivered = 0
        self.last_weather = None
        self._pose: Optional[tuple] = None   # (position, heading_deg, pitch_deg) from set_listener
        self._pose_sent: Optional[tuple] = None
        self._pitch = 0.0
        self._t_offset = 0.0                 # scene time - render time (seek)
        self._full_pending = False
        self.pipeline_rebuilds = 0           # new pipelines built after start (layout changes, Python core)
        self.scene_updates = 0               # incremental structural updates (native core)
        self._target_blocks = max(1, int(math.ceil(self.buffer_s * SAMPLE_RATE / BLOCK)))
        self._extra_blocks = 0               # extra render-ahead while a structural update lands
        self._extra_until = -1               # block index after which it is released

    # ------------------------------------------------------------------ public state
    @property
    def scene(self) -> Scene:
        """A copy of the scene the engine is heading to (driver values, before smoothing)."""
        with self._lock:
            return self._target.copy()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def time(self) -> float:
        """Seconds of scene time heard so far (output latency removed).

        Counted from the rendered frames actually played, so an underrun (silence the device
        played while the producer was late) does not move the scene clock or the markers.
        """
        dev = 0.0
        if self._stream is not None:
            try:
                lat = self._stream.latency
                dev = float(lat if not isinstance(lat, (tuple, list)) else lat[-1])
            except Exception:
                dev = 0.0
        return self._render_time() + self._t_offset

    def _render_time(self) -> float:
        dev = 0.0
        if self._stream is not None:
            try:
                lat = self._stream.latency
                dev = float(lat if not isinstance(lat, (tuple, list)) else lat[-1])
            except Exception:
                dev = 0.0
        return max(0.0, self._consumed / self.sr_out - self._chain_latency / SAMPLE_RATE - dev)

    @property
    def block_ms_max(self) -> float:
        """The slowest render block (ms, pipeline + crossfade + governor) of the last ~1.4 s."""
        try:
            return max(self._block_ms) if self._block_ms else 0.0
        except ValueError:                   # appended to while iterating
            return 0.0

    @property
    def volume_db(self) -> float:
        return self._vol_db

    @volume_db.setter
    def volume_db(self, value: float) -> None:
        self.set("vibe.volume_db", float(value))

    @property
    def badge(self) -> Optional[str]:
        """The governor's visible note ('Volume eased for Sleep' ...) or None."""
        g = self._governor
        return g.badge() if g is not None else None

    @property
    def finished(self) -> bool:
        """True once a sleep timer has faded the output to silence."""
        g = self._governor
        return bool(g is not None and g.finished)

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"<raingen.Engine {self._target.name!r} {self.layout} {state} t={self.time:.1f}s>"

    # ------------------------------------------------------------------ control API
    def set(self, path: str, value: Any, ramp: Optional[float] = 0.5, immediate: bool = False) -> Any:
        """Change a parameter (thread-safe). Returns the coerced value.

        The glide lasts max(ramp, the parameter's manifest smoothing time); `immediate=True` jumps.
        Structural parameters (place, shelter, listener, precipitation type, mode) crossfade to a
        rebuilt scene instead of gliding.
        """
        from .scene import manifest as mf

        path = PATH_ALIASES.get(path, path)
        spec = mf.find_param(path)
        if spec is not None:
            with self._lock:
                temp = float(self._target.weather.temperature_c)
            value = mf.clamp(spec, mf.coerce(spec, value, temp_c=temp))
        else:
            with self._lock:
                try:
                    self._target.get(path)
                except (AttributeError, KeyError, TypeError):
                    raise KeyError(f"unknown parameter {path!r}") from None
        # None keeps the parameter's own (manifest) smoothing; an explicit ramp wins.
        self._cmd.put(("set", path, value, None if ramp is None else float(ramp), bool(immediate)))
        return value

    def set_scene(self, scene, xfade: Optional[float] = None) -> None:
        """Move to another scene: weather glides, structure crossfades (thread-safe)."""
        if not isinstance(scene, Scene):
            from .api import load_scene
            scene = load_scene(scene)
        self._cmd.put(("scene", scene.copy(), xfade))

    load_scene = set_scene

    def set_listener(self, position=None, heading_deg: Optional[float] = None, pitch_deg: Optional[float] = None,
                     height_m: Optional[float] = None) -> tuple:
        """Move / turn the listener (world metres, deg true, deg up); thread-safe, any rate.

        On the native core this is a live pose (the voices follow it within ~50 ms, nothing is
        rebuilt); a height change, or the Python core, rebuilds like any structural change.
        """
        struct = False
        with self._lock:
            L = self._target.listener
            if position is not None:
                p = [float(v) for v in tuple(position)[:3]]
                if len(p) < 2 or not all(math.isfinite(v) for v in p):
                    raise ValueError("position must be [x, y] or [x, y, z] in metres")
                if len(p) == 2:
                    p.append(float((tuple(L.position) + (0.0, 0.0, 0.0))[2]))
                L.position = tuple(p)
            if heading_deg is not None:
                h = float(heading_deg)
                if not math.isfinite(h):
                    raise ValueError("heading_deg must be finite")
                L.heading_deg = h % 360.0
            if pitch_deg is not None and math.isfinite(float(pitch_deg)):
                self._pitch = max(-90.0, min(90.0, float(pitch_deg)))
            if height_m is not None and math.isfinite(float(height_m)) and abs(float(height_m) - float(L.height_m)) > 0.005:
                L.height_m = float(height_m)
                struct = True
            pose = (tuple(L.position), float(L.heading_deg), float(self._pitch))
            self._pose = pose
        if not self._send_pose() or struct:
            self._cmd.put(("retarget", 0.5))
        return pose

    def _send_pose(self, eff: Optional[Scene] = None) -> bool:
        """Hand the live pose to the native pipeline(s); False when the pipeline cannot follow it."""
        pipe = self._pipe
        if not self.native or pipe is None or not hasattr(pipe, "set_listener"):
            return False
        pose = self._pose
        if pose is None:
            if eff is None:
                return True
            pose = (tuple(eff.listener.position), float(eff.listener.heading_deg), float(self._pitch))
        if pose != self._pose_sent:
            pos, h, p = pose
            pipe.set_listener(position=pos, heading_deg=h, pitch_deg=p)
            xf = self._xf
            if xf is not None and hasattr(xf.get("pipe"), "set_listener"):
                try:
                    xf["pipe"].set_listener(position=pos, heading_deg=h, pitch_deg=p)
                except Exception:
                    pass
            self._pose_sent = pose
        return True

    def strike(self, distance_km: Optional[float] = None, bearing_deg: Optional[float] = None,
               intensity: Optional[float] = None) -> dict:
        """A lightning strike now (native core): flash marker, then thunder at distance / c(T)."""
        pipe = self._pipe
        if pipe is None or not hasattr(pipe, "strike"):
            raise RuntimeError("manual strikes need the native core")
        return pipe.strike(distance_km, bearing_deg, intensity)

    def seek(self, t: float) -> None:
        """Jump the scene clock / timeline to `t` seconds without restarting the audio."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        self._cmd.put(("seek", max(0.0, t)))

    def poll(self) -> list:
        """Deliver due markers to on_marker on the calling thread; returns them."""
        if self.error is not None:
            err, self.error = self.error, None
            raise RuntimeError(f"raingen engine stopped: {err!r}") from err
        now = self._render_time()
        due = []
        with self._marker_lock:
            if self._markers:
                keep = []
                for item in self._markers:
                    (due if item[0] <= now else keep).append(item)
                self._markers = keep
        due = [m for _, m in due]
        due.sort(key=lambda m: float(m.t))
        if self.on_marker is not None:
            for m in due:
                self.on_marker(m)
        self.markers_delivered += len(due)
        return due

    # ------------------------------------------------------------------ lifecycle
    def start(self, audio: bool = True) -> "Engine":
        """Build the pipeline, pre-render `prime_s`, open the audio stream (audio=False: use read())."""
        if self._running:
            return self
        sd = _sounddevice() if audio else None
        self._drain(initial=True)
        self._build_initial()
        self._stop.clear()
        self._primed.clear()
        self._producer = threading.Thread(target=self._produce, name="raingen-producer", daemon=True)
        self._producer.start()
        if not self._primed.wait(timeout=120.0):
            self._stop.set()
            raise RuntimeError("raingen engine did not start in time (the scene is too heavy for real time?)")
        if self.error is not None:
            err, self.error = self.error, None
            self._stop.set()
            raise RuntimeError(f"raingen engine failed to start: {err!r}") from err
        if sd is not None:
            self._stream = sd.OutputStream(samplerate=self.sr_out, channels=self.channels, dtype="float32",
                                           blocksize=self.blocksize, device=self.device, latency=self.latency,
                                           callback=self._callback)
            self._stream.start()
        if self._osc_cfg is not None:
            from .control.osc import serve_osc
            host, port = ("127.0.0.1", self._osc_cfg) if isinstance(self._osc_cfg, int) else self._osc_cfg
            self._osc = serve_osc(self, host=host, port=int(port))
        self._running = True
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._producer is not None:
            self._producer.join(timeout=3.0)
            self._producer = None
        if self._osc is not None:
            try:
                self._osc.stop()
            except Exception:
                pass
            self._osc = None
        self._running = False

    def __enter__(self) -> "Engine":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def run(self, seconds: Optional[float] = None, poll_s: float = 0.05) -> None:
        """Play (blocking) for `seconds` or until Ctrl+C / the sleep timer ends; delivers markers."""
        started_here = not self._running
        if started_here:
            self.start()
        t_end = None if seconds is None else time.monotonic() + float(seconds)
        try:
            while t_end is None or time.monotonic() < t_end:
                self.poll()
                if self.finished:
                    break
                time.sleep(poll_s)
        except KeyboardInterrupt:
            pass
        finally:
            if started_here:
                self.stop()

    wait = run

    def read(self, frames: int, timeout: float = 10.0) -> np.ndarray:
        """Pull `frames` of output (frames, channels) float32 without a sound card (start(audio=False))."""
        frames = int(frames)
        out = np.zeros((frames, max(self.channels, 1)), dtype=np.float32)
        deadline = time.monotonic() + float(timeout)
        while True:
            avail = self._pushed - self._consumed
            if avail >= frames or self._stop.is_set():
                break
            if self.error is not None:
                self.poll()
            if time.monotonic() > deadline:
                raise TimeoutError("raingen engine could not render fast enough")
            time.sleep(0.002)
        self._callback(out, frames, None, None)
        return out

    # ------------------------------------------------------------------ audio callback
    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        pos = 0
        cur, cpos = self._cur, self._cur_pos
        fifo = self._fifo
        while pos < frames:
            if cur is None or cpos >= cur.shape[0]:
                if not fifo:
                    outdata[pos:] = 0.0
                    self.underruns += 1
                    cur, cpos = None, 0
                    break
                cur = fifo.popleft()
                cpos = 0
            take = min(frames - pos, cur.shape[0] - cpos)
            outdata[pos:pos + take] = cur[cpos:cpos + take]
            pos += take
            cpos += take
        self._cur, self._cur_pos = cur, cpos
        self._consumed += pos
        self._played += frames

    # ------------------------------------------------------------------ building
    def _prepare(self, scene: Scene) -> Scene:
        sc = scene.copy()
        if self._quality:
            sc.render.quality = self._quality
        return sc

    def _make_pipeline(self, scene: Scene):
        if self.native:
            from .native import NativeSession
            return NativeSession(scene, self.layout, stems=False, precip_types="live")
        from .render import BlockRenderer
        return BlockRenderer(scene, self.layout, stems=False)

    def _make_governor(self, scene: Scene):
        from .wellness.modes import CalmGovernor, canonical_mode

        if canonical_mode(scene.mode) == "none":
            return None
        if self.honor_timer:
            return CalmGovernor.from_scene(scene, SAMPLE_RATE)
        vibe = scene.vibe or {}
        return CalmGovernor(SAMPLE_RATE, scene.mode, timer_min=0.0, descent_min=0.0, tingle=vibe.get("tingle"))

    @staticmethod
    def _governor_key(scene: Scene) -> tuple:
        vibe = scene.vibe or {}
        return (scene.mode, vibe.get("timer_min"), vibe.get("descent_min"), vibe.get("tingle"))

    def _update_latency(self) -> None:
        gov_lat = self._gov_slot.latency if self._gov_slot is not None else 0
        self._chain_latency = int(self._pipe.latency) + int(gov_lat) + \
            int(self._limiter.latency if self._limiter is not None else 0)

    @staticmethod
    def _governor_latency() -> int:
        """The largest look-ahead any CalmGovernor uses (every mode is padded to it)."""
        from .wellness.modes import CalmGovernor
        return int(round(float(getattr(CalmGovernor, "LOOKAHEAD_S", 0.0)) * SAMPLE_RATE))

    def _build_initial(self) -> None:
        from .master.limiter import TruePeakLimiter
        from .render import StreamingResampler

        with self._lock:
            eff = self._graph.apply(self._target)
        eff = self._prepare(eff)
        self._pipe = self._make_pipeline(eff)
        self._eff = eff
        self._current = eff.copy()
        self._struct_key = self._skey(eff)
        self._pose_sent = None
        self._t_offset = 0.0
        self._full_pending = False
        self._send_pose(eff)
        self.channels = int(self._pipe.channels)
        self._governor = self._make_governor(eff)
        self._gov_slot = _GovernorSlot(self._governor, self.channels, self._governor_latency())
        self._limiter = TruePeakLimiter(SAMPLE_RATE, self.channels, self.ceiling_dbtp)
        self._resampler = StreamingResampler(SAMPLE_RATE, self.sr_out, self.channels) \
            if self.sr_out != SAMPLE_RATE else None
        self._gov_key = self._governor_key(eff)
        self._update_latency()
        self._index = 0
        self._ramps.clear()
        self._xf = None
        # a (re)start plays the scene from t = 0
        self._fifo.clear()
        self._cur, self._cur_pos = None, 0
        self._played = self._consumed = self._pushed = 0
        self._g_prev = None
        with self._marker_lock:
            self._markers = []

    def _skey(self, scene: Scene) -> str:
        return _structure_key(scene, live_listener=self.native, none_live=self.native)

    def _can_update(self) -> bool:
        """True when a structural change can update the running native pipeline in place."""
        pipe = self._pipe
        if not self.native or self._full_pending or pipe is None or not hasattr(pipe, "prepare_update"):
            return False
        from .spatial.decoders import normalize_layout
        try:
            return normalize_layout(self.layout) == pipe.layout
        except Exception:
            return False

    def _request_rebuild(self, eff: Scene) -> None:
        self._build_gen += 1
        gen = self._build_gen
        scene = self._prepare(eff)
        if self._can_update():
            pipe = self._pipe
            # render ahead before the new voice graph goes in (see _HEADROOM_S)
            self._extra_blocks = int(math.ceil(_HEADROOM_S * SAMPLE_RATE / BLOCK))
            self._extra_until = 1 << 62

            def ready() -> None:
                end = time.monotonic() + _HEADROOM_WAIT_S
                while (gen == self._build_gen and not self._stop.is_set() and time.monotonic() < end
                       and len(self._fifo) < self._target_blocks + self._extra_blocks):
                    time.sleep(0.005)

            def update():
                try:
                    prep = pipe.prepare_update(scene, wanted=lambda: gen == self._build_gen, ready=ready)
                    if prep is not None:
                        self._built = (gen, "update", (pipe, prep), scene)
                except Exception as exc:
                    log.exception("raingen engine: scene update failed; keeping the current scene")
                    self.error = exc
                    self._extra_until = self._index

            threading.Thread(target=update, name="raingen-update", daemon=True).start()
            return
        self._full_pending = True
        self._extra_until = self._index

        def build():
            try:
                pipe = self._make_pipeline(scene)
                self._built = (gen, "pipe", pipe, scene)
            except Exception as exc:
                log.exception("raingen engine: rebuild failed; keeping the current scene")
                self.error = exc

        threading.Thread(target=build, name="raingen-rebuild", daemon=True).start()

    def _carry_live(self, scene: Scene) -> Scene:
        old_current = self._current
        cur = scene.copy()
        for p in self._live_numeric + self._live_discrete:
            v = _safe_get(old_current, p, _safe_get(cur, p))
            try:
                cur.set(p, v)
            except (AttributeError, KeyError, TypeError):
                pass
        cur.timeline = list(scene.timeline)
        return cur

    def _new_governor_if_needed(self, scene: Scene) -> None:
        key = self._governor_key(scene)
        if key != self._gov_key:
            # a new mode / timer: a fresh governor, crossfaded in at the same total latency
            self._governor = self._make_governor(scene)
            self._gov_slot.replace(self._governor, max(1, int(self.xfade_s * SAMPLE_RATE)))
            self._gov_key = key

    def _swap_if_built(self) -> None:
        b = self._built
        if b is None:
            return
        gen, kind, obj, scene = b
        if kind == "update":
            self._built = None
            pipe, prep = obj
            if gen != self._build_gen or pipe is not self._pipe:
                if gen == self._build_gen:
                    self._extra_until = self._index
                return
            played = pipe.commit_update(prep)          # World state carried; voices diffed in Rust
            cur = self._carry_live(played)
            cur.timeline = list(scene.timeline)
            self._current = cur
            self._new_governor_if_needed(scene)
            self._update_latency()
            self.scene_updates += 1
            if gen == self._build_gen:
                self._extra_until = self._index + int(math.ceil(_HEADROOM_HOLD_S * SAMPLE_RATE / BLOCK))
            return
        if self._xf is not None:
            return                      # a crossfade in progress finishes first (dropping its old side clicks)
        self._built = None
        pipe = obj
        if gen != self._build_gen or pipe.channels != self.channels:
            return
        self._full_pending = False
        old_current = self._current
        cur = self._carry_live(scene)
        if hasattr(pipe, "start_at"):
            # continue the running timeline (a native pipeline would otherwise re-render every
            # block from t = 0 to catch up) with the old World's weather state
            pipe.start_at(self._index, self._index * BLOCK / SAMPLE_RATE + self._t_offset, carry_from=self._pipe)
        if self.xfade_s > 0.0:
            self._xf = {"pipe": self._pipe, "scene": old_current, "pos": 0,
                        "len": max(1, int(self.xfade_s * SAMPLE_RATE))}
        self._pipe = pipe
        self._current = cur
        self._pose_sent = None
        self._send_pose(scene)
        self.pipeline_rebuilds += 1
        self._new_governor_if_needed(scene)
        self._update_latency()

    # ------------------------------------------------------------------ commands
    def _drain(self, initial: bool = False) -> None:
        while True:
            try:
                cmd = self._cmd.get_nowait()
            except queue.Empty:
                return
            try:
                self._apply(cmd, initial)
            except Exception as exc:
                log.warning("raingen engine: command %r failed: %s", cmd[:2], exc)

    def _apply(self, cmd: tuple, initial: bool) -> None:
        from .scene.graph import ParameterGraph
        from .scene.vibe import apply_vibe
        from .wellness.modes import apply_mode

        kind = cmd[0]
        ramp, immediate = 0.5, initial
        with self._lock:
            if kind == "set":
                _, path, value, ramp, imm = cmd
                immediate = immediate or imm
                if path.startswith("vibe."):
                    name = path.split(".", 1)[1]
                    if name in _MACROS:
                        self._target = apply_vibe(self._target, **{name: float(value)})
                    else:
                        vibe = dict(self._target.vibe or {})
                        vibe[name] = value
                        self._target.vibe = vibe
                        if name == "volume_db":
                            self._set_volume(float(value), ramp, immediate)
                elif path == "mode":
                    self._target = apply_mode(self._target, str(value))
                else:
                    self._target.set(path, value)
                self._graph.scene = self._target
            elif kind == "scene":
                _, scene, xfade = cmd
                self._target = scene
                self._graph = ParameterGraph(self._target)
                if xfade is not None:
                    ramp = float(xfade)
            elif kind == "retarget":
                ramp = float(cmd[1])
            elif kind == "seek":
                self._seek_now(float(cmd[1]))
                return
            else:
                raise ValueError(f"unknown engine command {kind!r}")
        if not initial and self._pipe is not None:
            self._retarget(ramp, immediate)

    def _seek_now(self, t: float) -> None:
        """(producer thread) the scene clock jumps to t; voices, ramps and the stream keep going."""
        render_t = self._index * BLOCK / SAMPLE_RATE
        self._t_offset = t - render_t
        for pipe in (self._pipe, (self._xf or {}).get("pipe")):
            if pipe is not None and hasattr(pipe, "seek"):
                try:
                    pipe.seek(t)
                except Exception as exc:
                    log.warning("raingen engine: seek failed: %s", exc)

    def _set_volume(self, db: float, ramp: float, immediate: bool) -> None:
        self._vol_from = self._vol_db
        self._vol_to = float(db)
        self._vol_start = self._index
        self._vol_blocks = 1 if immediate else max(1, int(round(max(ramp or 0.0, 0.05) * SAMPLE_RATE / BLOCK)))

    def _retarget(self, ramp: float, immediate: bool) -> None:
        with self._lock:
            eff = self._graph.apply(self._target)
        eff = self._prepare(eff)
        key = self._skey(eff)
        if key != self._struct_key:
            self._struct_key = key
            self._request_rebuild(eff)
        self._send_pose(eff)
        old = self._eff
        for p in self._live_numeric:
            a, b = _safe_get(old, p), _safe_get(eff, p)
            if a != b:
                self._schedule(p, b, ramp, immediate)
        for p in self._live_discrete:
            a, b = _safe_get(old, p), _safe_get(eff, p)
            if a != b:
                self._ramps.pop(p, None)
                try:
                    self._current.set(p, b)
                except (AttributeError, KeyError, TypeError):
                    pass
        self._current.timeline = list(eff.timeline)
        self._eff = eff

    def _schedule(self, path: str, value: Any, ramp: float, immediate: bool) -> None:
        from .scene import manifest as mf

        spec = mf.find_param(path)
        cur = _safe_get(self._current, path)
        numeric = (spec is not None and spec.is_numeric and isinstance(cur, (int, float))
                   and isinstance(value, (int, float)) and not isinstance(cur, bool) and not isinstance(value, bool))
        dur = max(float(ramp), 0.05) if ramp is not None else (spec.smoothing_s if spec is not None else 0.0)
        if immediate or not numeric or dur <= 0.0:
            self._ramps.pop(path, None)
            self._current.set(path, value)
            return
        p0 = mf.to_perceptual(spec, cur)
        p1 = mf.to_perceptual(spec, value)
        if spec.wrap and spec.min is not None and spec.max is not None and spec.scale == "lin":
            span = float(spec.max - spec.min)
            p1 = p0 + ((p1 - p0 + 0.5 * span) % span) - 0.5 * span
        blocks = max(1, int(round(dur * SAMPLE_RATE / BLOCK)))
        self._ramps[path] = _Ramp(path, spec, float(p0), float(p1), self._index, blocks, value)

    def _advance_ramps(self) -> None:
        from .scene import manifest as mf

        done = []
        for path, r in self._ramps.items():
            u = (self._index - r.start + 1) / r.blocks
            if u >= 1.0:
                self._current.set(path, r.final)
                done.append(path)
            else:
                v = mf.clamp(r.spec, mf.from_perceptual(r.spec, r.p0 + (r.p1 - r.p0) * u))
                self._current.set(path, v)
        for p in done:
            del self._ramps[p]
        if self._vol_db != self._vol_to:
            u = min(1.0, (self._index - self._vol_start + 1) / max(self._vol_blocks, 1))
            self._vol_db = self._vol_from + (self._vol_to - self._vol_from) * u

    # ------------------------------------------------------------------ producer
    def _raw_block(self) -> np.ndarray:
        """Pipeline (+ crossfade) + governor at 48 kHz, before gain and limiter."""
        t_start = time.perf_counter()
        self._swap_if_built()
        self._advance_ramps()
        idx = self._index
        t0 = idx * BLOCK / SAMPLE_RATE
        off = self._t_offset
        sc = self._current
        if sc.timeline:
            from .scene.timeline import evaluate
            sc = evaluate(sc, t0 + off)
        out = self._pipe.render_block(idx, scene_at_t=sc)
        y = out.mix
        markers = self._clocked(self._pipe, out.markers, off)
        xf = self._xf
        if xf is not None:
            old_sc = xf["scene"]
            old = xf["pipe"].render_block(idx, scene_at_t=old_sc)
            n = y.shape[1]
            w = np.clip((xf["pos"] + np.arange(1, n + 1)) / xf["len"], 0.0, 1.0)
            a, b = np.cos(0.5 * math.pi * w), np.sin(0.5 * math.pi * w)
            k = self._xf_compensation(old.mix, y, a, b, xf, done=bool(w[-1] >= 1.0))
            y = (old.mix * a[None, :] + y * b[None, :]) * k[None, :]
            markers.extend(self._clocked(xf["pipe"], old.markers, off))
            xf["pos"] += n
            if xf["pos"] >= xf["len"]:
                self._xf = None
        self.last_weather = out.weather
        if markers:
            with self._marker_lock:
                self._markers.extend(markers)
        y = self._gov_slot.process(y)
        self._index += 1
        dt = time.perf_counter() - t_start
        self.cpu_load = 0.9 * self.cpu_load + 0.1 * dt / (BLOCK / SAMPLE_RATE)
        self._block_ms.append(dt * 1e3)
        if dt > BLOCK / SAMPLE_RATE:
            self.slow_blocks += 1
        return y

    @staticmethod
    def _clocked(pipe, markers, off: float) -> list:
        """(render-clock time heard, marker on the scene clock): the native core already stamps
        scene time (Rust seek), the Python pipeline render time."""
        if not markers:
            return []
        if not off:
            return [(float(m.t), m) for m in markers]
        if hasattr(pipe, "seek"):
            return [(float(m.t) - off, m) for m in markers]
        out = []
        for m in markers:
            try:
                m2 = dataclasses.replace(m, t=float(m.t) + off)
            except TypeError:
                m2 = m
            out.append((float(m.t), m2))
        return out

    @staticmethod
    def _xf_compensation(x_old: np.ndarray, x_new: np.ndarray, a: np.ndarray, b: np.ndarray, xf: dict,
                         done: bool = False) -> np.ndarray:
        """Per-sample gain that keeps an equal-power crossfade at constant power for correlated sides.

        cos/sin weights hold the power of *uncorrelated* signals; a rebuilt pipeline shares most
        layers with the old one (same seed, same wind, thunder, ambience), and a + b = sqrt 2 at
        the midpoint would swell such a fade by up to +3 dB. The block's cross-power C gives the
        actual power a^2 Po + b^2 Pn + 2ab C; scaling by sqrt(target / actual), target the
        uncorrelated value, removes the swell (identical sides fade into themselves unchanged).
        The factor is interpolated per sample from block to block and returns to 1 at the end.
        """
        n = x_new.shape[1]
        k0 = float(xf.get("k", 1.0))
        k1 = 1.0
        if not done and n:
            po = float(np.mean(x_old * x_old))
            pn = float(np.mean(x_new * x_new))
            c = float(np.mean(x_old * x_new))
            target = float(np.mean(a * a)) * po + float(np.mean(b * b)) * pn
            actual = target + 2.0 * float(np.mean(a * b)) * c
            if target > 1e-30 and actual > 1e-30:
                k1 = float(np.clip(math.sqrt(target / actual), 0.5, 2.0))
        xf["k"] = k1
        if k0 == k1:
            return np.full(n, k1)
        return k0 + (k1 - k0) * np.arange(1, n + 1) / max(n, 1)

    def _master(self, y: np.ndarray) -> np.ndarray:
        n = y.shape[1]
        g = 10.0 ** ((self._cal_db + self._vol_db) / 20.0)
        g0 = g if self._g_prev is None else self._g_prev
        if g0 != g and n:
            # volume glides move per block; ramp the gain per sample so they never zipper
            y = y * (g0 + (g - g0) * np.arange(1, n + 1) / n)[None, :]
        else:
            y = y * g
        self._g_prev = g
        y = self._limiter.process(y)
        if self._resampler is not None:
            y = self._resampler.process(y)
        return np.ascontiguousarray(y.T, dtype=np.float32) if n else np.zeros((0, self.channels), np.float32)

    def _prime(self) -> None:
        from .master.loudness import LoudnessMeter

        blocks = [self._raw_block() for _ in range(max(1, int(math.ceil(self.prime_s * SAMPLE_RATE / BLOCK))))]
        if self.target_lufs is not None:
            meter = LoudnessMeter(SAMPLE_RATE, self.channels, layout=self.layout, true_peak=False)
            for b in blocks:
                meter.process(b)
            from .render import MAX_NORMALISATION_GAIN_DB as G, normalisation_loudness

            lufs, _ = normalisation_loudness(meter)     # provisional under 400 ms; ungated below -70 LUFS
            self._cal_db = float(np.clip(float(self.target_lufs) - lufs, -G, G)) if math.isfinite(lufs) else 0.0
            log.info("raingen engine: start loudness %.1f LUFS, calibration %+.1f dB", lufs, self._cal_db)
        for b in blocks:
            self._push(self._master(b))

    def _push(self, block: np.ndarray) -> None:
        self._fifo.append(block)
        self._pushed += int(block.shape[0])

    def _produce(self) -> None:
        block_s = BLOCK / SAMPLE_RATE
        from .render import limit_threads

        try:
            with limit_threads(1):
                self._prime()
                self._primed.set()
                while not self._stop.is_set():
                    self._drain()
                    if self._extra_blocks and self._index >= self._extra_until:
                        self._extra_blocks = 0
                    if len(self._fifo) >= self._target_blocks + self._extra_blocks:
                        time.sleep(0.25 * block_s)
                        continue
                    self._push(self._master(self._raw_block()))
        except Exception as exc:
            log.exception("raingen engine producer failed")
            self.error = exc
            self._primed.set()
