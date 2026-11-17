"""The native (Rust) synthesis core, wired to the Python scene, physics and decoder design.

`raingen_native` (crate native/, built with maturin) does all per-sample work: every
generator voice, SH encoding (order 3, ACN/SN3D), shelter transfer, decoding, the safety
limiter and the live cpal stream. Python keeps what is cheap or design-time:

    Scene -> build_layout() patches -> raingen_native.Engine.set_scene()      (once)
    MATERIALS table                  -> set_materials()                        (once)
    raingen.spatial.decoders design  -> set_output(matrix | MagLS FIRs)        (once)
    World.step() per control block   -> update(WeatherState JSON)              (every block)

Live listener: the voices keep their patches' world positions, so moving and turning the
listener (`set_listener`, any rate) is a lock-free pose update in Rust, never a rebuild. Every
later structural change is laid out around the session's reference pose (the listener the
session was built for) and applied with `prepare_update` / `commit_update` (or `set_scene`
while streaming): an incremental voice diff in Rust, with the World's state carried over.

NativeSession is the one object that owns that wiring. It is used three ways:

* as a drop-in BlockRenderer (`render_block`, `channels`, `latency`, `decoder`, `stem_names`,
  `warnings`), so raingen.render.Renderer and raingen.engine.Engine keep their mastering,
  loudness, governor and writers while synthesis runs natively;
* offline on its own: `render(seconds)` -> (channels, frames) float32, deterministic per seed;
* live: `start()` streams through cpal; a Python control thread steps the World once per
  block (paced by the native producer's lock-free block counter) and sends `update()`.

Selection: the native core is used whenever `raingen_native` imports, unless the environment
variable RAINGEN_NATIVE is 0/off/false (or a caller passes native=False). Layouts beyond third
order fall back to Python.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from .core.constants import AMBI_ORDER, BLOCK, SAMPLE_RATE, STEMS
from .core.types import EventMarker
from .scene.schema import Scene

log = logging.getLogger("raingen.native")

try:                                                     # pragma: no cover - depends on the build
    import raingen_native as _rn
    IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:                                # pragma: no cover
    _rn = None
    IMPORT_ERROR = _exc

__all__ = ["NativeSession", "available", "enabled", "supports_layout", "materials_json", "weather_json",
           "patches_json", "scene_json", "IMPORT_ERROR"]

_OFF = ("0", "off", "false", "no", "python")
_NATIVE_LAYOUTS = ("stereo", "mono", "binaural", "5.1", "7.1", "7.1.4", "foa", "hoa1", "hoa2", "hoa3")


def available() -> bool:
    """True when the compiled raingen_native module is importable."""
    return _rn is not None


def enabled(prefer: Optional[bool] = None) -> bool:
    """Whether to synthesize natively. prefer=False forces Python, True requires the module.

    With prefer=None: native when importable, unless RAINGEN_NATIVE=0 (off / false / python).
    """
    if prefer is False:
        return False
    if prefer is True:
        if _rn is None:
            raise ImportError(f"raingen_native is not installed (pip install 'raingen[native]'): {IMPORT_ERROR!r}")
        return True
    if str(os.environ.get("RAINGEN_NATIVE", "")).strip().lower() in _OFF:
        return False
    return _rn is not None


def supports_layout(layout: str) -> bool:
    from .spatial.decoders import normalize_layout
    try:
        return normalize_layout(layout) in _NATIVE_LAYOUTS
    except Exception:
        return False


def _threads() -> int:
    try:
        return max(0, int(os.environ.get("RAINGEN_NATIVE_THREADS", "0")))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# JSON for the Rust side
# ---------------------------------------------------------------------------
def _plain(o: Any) -> Any:
    """Dataclasses / numpy -> JSON-safe Python (non-finite floats clamped: serde rejects NaN)."""
    if isinstance(o, float):
        if math.isfinite(o):
            return o
        return 0.0 if math.isnan(o) else math.copysign(1e30, o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _plain(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return _plain(o.item())
    return str(o)


def weather_json(w) -> str:
    return json.dumps(_plain(w), separators=(",", ":"))


def patches_json(patches) -> str:
    return json.dumps(_plain(list(patches)), separators=(",", ":"))


def scene_json(scene: Scene, precip_types=None) -> str:
    d = json.loads(scene.to_json())
    if precip_types is not None:
        pt = precip_types if isinstance(precip_types, str) else [str(p) for p in precip_types]
        d.setdefault("native", {})["precip_types"] = pt
    return json.dumps(d, separators=(",", ":"))


_MATERIALS_JSON: Optional[str] = None


def materials_json() -> str:
    global _MATERIALS_JSON
    if _MATERIALS_JSON is None:
        from .synth.materials import MATERIALS
        _MATERIALS_JSON = json.dumps({k: _plain(v) for k, v in MATERIALS.items()}, separators=(",", ":"))
    return _MATERIALS_JSON


_LISTENER_LIVE = ("listener.position", "listener.heading_deg")


def _carry_world(old, new, copy_state: bool = False) -> None:
    """Hand the running weather state (storm cells, gusts, wetness, puddles, canopy, roof flow)
    from `old` to a World rebuilt for a changed layout, so a structural edit never resets it.
    copy_state: deep-copy the stateful processes (both Worlds keep stepping, e.g. a crossfade)."""
    import copy as _copy

    for attr in ("storms", "gusts", "gust_integral_time_s", "gust_ti", "_stab_factor", "_curtain_gust",
                 "_primed", "state", "dsd"):
        if hasattr(old, attr):
            v = getattr(old, attr)
            setattr(new, attr, _copy.deepcopy(v) if copy_state and attr in ("storms", "gusts") else v)
    ow, nw = getattr(old, "wetness", None), getattr(new, "wetness", None)
    if ow is None or nw is None:
        return
    for name in ("_film", "_puddle", "runoff_mm_h"):
        src, dst = getattr(ow, name, None), getattr(nw, name, None)
        if isinstance(src, dict) and isinstance(dst, dict):
            for m in dst:
                if m in src:
                    dst[m] = src[m]
    for attr in ("canopy_mm", "canopy_drip_mm_h", "canopy_evap_mm_h", "canopy_catch_mm_h", "throughfall_fraction",
                 "_roof_film_l", "_gutter_l", "roof_flow_l_s"):
        if hasattr(ow, attr):
            setattr(nw, attr, getattr(ow, attr))


def _marker(t: float, kind: str, data: str) -> EventMarker:
    try:
        d = json.loads(data) if data else {}
    except ValueError:
        d = {"raw": data}
    return EventMarker(t=float(t), kind=str(kind), data=d if isinstance(d, dict) else {"value": d})


def apply_output(eng, decoder) -> None:
    """Hand a Python-designed decoder (matrix / MagLS FIRs / passthrough) to a native Engine."""
    from .spatial.decoders import BinauralDecoder, MatrixDecoder, PassthroughDecoder

    if isinstance(decoder, BinauralDecoder):
        eng.set_output("binaural", 2, firs=np.ascontiguousarray(decoder.filters, dtype=np.float64),
                       latency=int(decoder.latency))
    elif isinstance(decoder, PassthroughDecoder):
        order = int(decoder.out_order)
        if order > 3:
            raise ValueError(f"the native core decodes up to third order, not {decoder.layout}")
        eng.set_output("foa" if order == 1 else f"hoa{order}", int(decoder.channels))
    elif isinstance(decoder, MatrixDecoder):
        lfe = getattr(decoder, "_lfe_index", None)
        eng.set_output(str(decoder.layout), int(decoder.channels), matrix=np.asarray(decoder.matrix, np.float64),
                       lfe_index=lfe, lfe_gain=float(getattr(decoder, "_lfe_gain", 0.0) or 0.0))
    else:
        raise ValueError(f"no native equivalent for decoder {decoder!r}")


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------
class NativeSession:
    """Scene -> World -> raingen_native.Engine -> decoded audio (offline blocks or a live stream).

    scene      Scene, preset name, scene file or plain words (prepared: seed, quality and mode set)
    layout     stereo | binaural | 5.1 | 7.1 | 7.1.4 | foa | hoa1..3 | mono
    stems      decode every stem too (BlockRenderer-compatible stems mode; decoders run in Python)
    mix        per-stem gains ({'rain': -3, 'thunder': 'off'} ...), as Renderer's mix=
    threads    native worker threads (0 = all cores, or RAINGEN_NATIVE_THREADS)
    precip_types  extra precipitation types to prepare voices for ('all' for live type changes)

    The native limiter and gain are bypassed for block rendering (Python masters the output);
    start() turns the limiter on at `ceiling_dbtp` for the live stream.
    """

    def __init__(self, scene, layout: Optional[str] = "stereo", *, stems: bool = False, mix: Optional[dict] = None,
                 strict: bool = False, sr: int = SAMPLE_RATE, block: int = BLOCK, threads: Optional[int] = None,
                 precip_types=None, quality: Optional[str] = None):
        if _rn is None:
            raise ImportError(f"raingen_native is not available: {IMPORT_ERROR!r}")
        if not isinstance(scene, Scene):
            from .api import load_scene
            scene = load_scene(scene)
        if int(sr) != SAMPLE_RATE:
            raise ValueError(f"the synthesis core runs at {SAMPLE_RATE} Hz; resample the output instead")
        from .render import resolve_layout, stem_gains
        from .spatial.decoders import make_decoder, normalize_layout

        base = scene.copy()
        if quality:
            base.render.quality = str(quality).lower()
        self.sr = int(sr)
        self.block = int(block)
        self.strict = bool(strict)
        self.stems = bool(stems)
        self.base_scene = base
        self._graph = None
        if getattr(base, "overrides", None):
            from .scene.graph import ParameterGraph
            self._graph = ParameterGraph(base.copy())
        self.scene = self.scene_at(0.0)
        self.layout = normalize_layout(resolve_layout(self.scene, layout))
        if not supports_layout(self.layout):
            raise ValueError(f"layout {self.layout!r} is not supported by the native core")
        self.threads = _threads() if threads is None else int(threads)
        self.precip_types = precip_types
        self.stem_names = tuple(STEMS)
        self.stem_gain = stem_gains(mix)
        self.warnings: list[str] = []
        self.last_weather = None

        hrtf = getattr(self.scene.listener, "hrtf", None) or "builtin"
        width = float(getattr(self.scene.listener, "width", 1.0) or 1.0)
        self.decoder = make_decoder(self.layout, AMBI_ORDER, self.sr, hrtf=hrtf, width=width)
        self.stem_decoders = {s: make_decoder(self.layout, AMBI_ORDER, self.sr, hrtf=hrtf, width=width)
                              for s in self.stem_names} if self.stems else {}

        self.engine = _rn.Engine(sample_rate=self.sr, block=self.block, seed=0, threads=self.threads)
        self.engine.set_materials(materials_json())
        self._native_stems = list(getattr(_rn, "STEMS", self.stem_names))
        if self.stems:
            self.engine.set_output("mono", 1)               # the native mix is discarded in stems mode
        else:
            apply_output(self.engine, self.decoder)
        self.engine.set_limiter(-1.0, False)
        self.engine.set_gain_db(0.0)
        gains = []
        for s in self._native_stems:
            g = float(self.stem_gain.get(s, 1.0))
            gains.append(20.0 * math.log10(g) if g > 0.0 else float("-inf"))
        self.engine.set_stem_gains(gains)
        L = self.scene.listener
        self._ref_listener = (tuple(float(v) for v in (tuple(L.position) + (0.0, 0.0, 0.0))[:3]),
                              float(L.heading_deg))
        self._pose = self._ref_listener[0] + (self._ref_listener[1], 0.0)
        self._t_off = 0.0
        self._update_lock = threading.Lock()
        self.scene_updates = 0
        self._build(self.scene)

        # live state
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ctl: Optional[threading.Thread] = None
        self._live = False
        self._k = 0
        self._target: Scene = self.scene.copy()
        self._current: Scene = self.scene.copy()
        self._ramps: dict = {}
        self._pending: list = []
        self._pending_lock = threading.Lock()
        self.on_marker: Optional[Callable] = None
        self.markers_delivered = 0
        self.error: Optional[BaseException] = None
        self.cal_db = 0.0
        self.volume_db = 0.0
        self.latency_ms = 120.0
        self.control_ms = 0.0

    # -- structure ------------------------------------------------------------
    def _build(self, scene: Scene) -> None:
        from .core.rng import RngFactory
        from .physics.layout import build_layout
        from .physics.world import World

        self.rngf = RngFactory(int(scene.seed))
        self.patches = build_layout(scene)
        self.world = World(scene, self.rngf, sr=self.sr, block=self.block)
        self.engine.set_scene(scene_json(scene, self.precip_types), patches_json(self.patches))
        self._next = 0

    def scene_at(self, t: float) -> Scene:
        sc = self.base_scene
        if sc.timeline:
            from .scene.timeline import evaluate
            sc = evaluate(sc, float(t))
        if self._graph is not None:
            sc = self._graph.apply(sc)
        return sc

    @property
    def channels(self) -> int:
        return int(self.decoder.channels)

    @property
    def latency(self) -> int:
        """Samples of decoder latency left in render_block() output (native mix: trimmed, 0)."""
        return int(getattr(self.decoder, "latency", 0) or 0) if self.stems else 0

    @property
    def voices(self) -> list:
        return list(self.engine.voice_names())

    # -- live listener / strike / seek ---------------------------------------------
    def _pin(self, scene: Scene) -> Scene:
        """`scene` laid out around the session's reference pose (the live pose moves the voices)."""
        sc = scene.copy()
        pos, heading = self._ref_listener
        sc.listener.position = tuple(pos)
        sc.listener.heading_deg = float(heading)
        return sc

    def set_listener(self, position=None, heading_deg: Optional[float] = None,
                     pitch_deg: Optional[float] = None) -> tuple:
        """Move / turn the listener (world metres, deg true, deg up); lock-free, never rebuilds."""
        x, y, z, h, p = self._pose
        if position is not None:
            pos = [float(v) for v in tuple(position)[:3]] + [z] * max(0, 3 - len(tuple(position)[:3]))
            x, y, z = pos[0], pos[1], pos[2]
        if heading_deg is not None:
            h = float(heading_deg) % 360.0
        if pitch_deg is not None:
            p = max(-90.0, min(90.0, float(pitch_deg)))
        pose = (x, y, z, h, p)
        if not all(math.isfinite(v) for v in pose):
            raise ValueError("listener pose must be finite")
        self._pose = pose
        self.engine.set_listener(*pose)
        return pose

    @property
    def listener_pose(self) -> tuple:
        """(x, y, z, heading_deg, pitch_deg) last sent with set_listener (or the scene's listener)."""
        return self._pose

    def strike(self, distance_km: Optional[float] = None, bearing_deg: Optional[float] = None,
               intensity: Optional[float] = None) -> dict:
        """A lightning strike now: flash marker on the next block, thunder at distance / c(T)."""
        th = self.scene.weather.thunder
        d = float(distance_km) if distance_km is not None else float(getattr(th, "distance_km", 3.0) or 3.0)
        b = float(bearing_deg) if bearing_deg is not None else float(getattr(th, "bearing_deg", 0.0) or 0.0)
        k = float(intensity) if intensity is not None else 0.7
        d = max(0.25, min(40.0, d)) if math.isfinite(d) else 3.0
        b = b % 360.0 if math.isfinite(b) else 0.0
        k = max(0.02, min(1.0, k)) if math.isfinite(k) else 0.7
        sid = self.engine.strike(d, b, k)
        w = self.last_weather
        temp = float(getattr(w, "temperature_c", self.scene.weather.temperature_c))
        c = 331.3 * math.sqrt(max(1.0 + temp / 273.15, 0.1))
        return {"id": sid, "distance_km": d, "bearing_deg": b, "intensity": k, "delay_s": d * 1000.0 / c}

    def seek(self, t: float) -> None:
        """Move the scene clock (World, timeline, marker times) to `t` s; no voice is reset."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        k = self._k if self._live else self._next
        self._t_off = t - k * self.block / self.sr
        self.engine.seek(t)

    def start_at(self, index: int, scene_t: Optional[float] = None, carry_from=None) -> None:
        """Continue a running timeline from block `index` (a pipeline built mid-stream): the next
        render_block(index) renders at once instead of catching up from block 0. Marker times
        follow `scene_t` (default index * block / sr); `carry_from` (a session) hands over its
        World's weather state."""
        index = int(index)
        t = index * self.block / self.sr if scene_t is None else float(scene_t)
        self._next = index
        self._t_off = t - index * self.block / self.sr
        self.engine.seek(t)
        w = getattr(carry_from, "world", None)
        if w is not None:
            try:
                _carry_world(w, self.world, copy_state=True)
            except Exception as exc:                  # pragma: no cover - best effort
                log.warning("raingen native: weather state not carried over: %s", exc)

    # -- incremental structure ------------------------------------------------------
    def prepare_update(self, scene: Scene, wanted: Optional[Callable[[], bool]] = None):
        """Lay `scene` out around the reference pose, build its World and send the voice graph
        (Rust diffs it against the running one: unchanged voices keep playing). Any thread; the
        World is swapped in by `commit_update` on the rendering thread. None when no longer wanted."""
        from .core.rng import RngFactory
        from .physics.layout import build_layout
        from .physics.world import World

        sc = self._pin(scene)
        with self._update_lock:
            if wanted is not None and not wanted():
                return None
            rngf = RngFactory(int(sc.seed))
            patches = build_layout(sc)
            world = World(sc, rngf, sr=self.sr, block=self.block)
            if wanted is not None and not wanted():
                return None
            self.engine.set_scene(scene_json(sc, self.precip_types), patches_json(patches))
        return (sc, rngf, patches, world)

    def commit_update(self, prep) -> Scene:
        """Swap in a prepared update (call between blocks); returns the scene now playing."""
        sc, rngf, patches, world = prep
        _carry_world(self.world, world)
        self.world = world
        self.rngf = rngf
        self.patches = patches
        self.scene = sc.copy()
        self.base_scene = sc.copy()
        self.scene_updates += 1
        return sc

    # -- offline ----------------------------------------------------------------
    def reset(self) -> None:
        """Back to t = 0 (fresh World, every voice, encoder, decoder and limiter state)."""
        from .physics.world import World

        self.engine.reset()
        self.engine.poll_markers()
        self._t_off = 0.0
        self.world = World(self.scene, self.rngf, sr=self.sr, block=self.block)
        for d in self.stem_decoders.values():
            d.reset()
        if self.stems:
            self.decoder.reset()
        self._next = 0

    def _seek(self, index: int) -> None:
        if index < self._next:
            self.reset()
        while self._next < index:
            self.render_block(self._next)

    def render_block(self, index: int, scene_at_t: Optional[Scene] = None):
        """BlockRenderer.render_block: step the World, update the voices, render one block."""
        from .render import BlockOutput

        index = int(index)
        if index != self._next:
            self._seek(index)
        n = self.block
        t0 = index * n / self.sr
        off = self._t_off
        k_off = int(round(off * self.sr / n)) if off else 0
        sc = scene_at_t if scene_at_t is not None else self.scene_at(t0 + off)
        weather = self.world.step(max(0, index + k_off), t0 + off, n, sc)
        self.last_weather = weather
        self.engine.update(weather_json(weather))
        stems = None
        if self.stems:
            st = np.asarray(self.engine.render_stems(n), dtype=np.float64)        # (7, 16, n)
            by_name = {s: st[i] for i, s in enumerate(self._native_stems)}
            total = st.sum(axis=0)
            mix = self.decoder.process(total)
            stems = {s: self.stem_decoders[s].process(by_name.get(s, np.zeros_like(total)))
                     for s in self.stem_names}
        else:
            mix = np.asarray(self.engine.render(n), dtype=np.float64)
        markers = [_marker(*m) for m in self.engine.poll_markers()]
        self._next = index + 1
        return BlockOutput(index=index, t0=float(t0), mix=mix, stems=stems, markers=markers, weather=weather)

    def render(self, seconds: float, start: Optional[int] = None) -> tuple[np.ndarray, list]:
        """(audio (channels, frames) float32, markers) for `seconds` from block `start` (default: on)."""
        frames = int(round(float(seconds) * self.sr))
        k = self._next if start is None else int(start)
        parts, markers = [], []
        got = 0
        while got < frames:
            out = self.render_block(k)
            parts.append(out.mix.astype(np.float32))
            markers.extend(out.markers)
            got += out.mix.shape[1]
            k += 1
        y = np.concatenate(parts, axis=1)[:, :frames] if parts else np.zeros((self.channels, 0), np.float32)
        return y, markers

    # -- live -------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._live

    @property
    def time(self) -> float:
        """Approximate scene time being heard (rendered blocks minus the ring latency)."""
        k = int(self.engine.blocks_rendered())
        return max(0.0, k * self.block / self.sr - self.latency_ms / 1000.0 - self.block / self.sr) + self._t_off

    def calibrate(self, target_lufs: float, prime_s: float = 1.5) -> float:
        """Render `prime_s` offline, measure BS.1770-5 loudness, return the gain toward `target_lufs`."""
        from .master.loudness import LoudnessMeter
        from .render import MAX_NORMALISATION_GAIN_DB as G, normalisation_loudness

        y, _ = self.render(max(float(prime_s), 0.5), start=0)
        meter = LoudnessMeter(self.sr, self.channels, layout=self.layout, true_peak=False)
        meter.process(y.astype(np.float64))
        lufs, _ = normalisation_loudness(meter)
        self.reset()
        return float(np.clip(float(target_lufs) - lufs, -G, G)) if math.isfinite(lufs) else 0.0

    def start(self, device: Optional[str] = None, latency_ms: float = 120.0, *, target_lufs: Optional[float] = -20.0,
              prime_s: float = 1.5, volume_db: float = 0.0, ceiling_dbtp: float = -1.0,
              on_marker: Optional[Callable] = None) -> "NativeSession":
        """Stream live through cpal (Rust producer + lock-free ring); Python steps the World per block."""
        if self._live:
            return self
        self.on_marker = on_marker if on_marker is not None else self.on_marker
        self.cal_db = self.calibrate(target_lufs, prime_s) if target_lufs is not None else 0.0
        if target_lufs is None:
            self.reset()
        if self.cal_db:
            log.info("raingen native: calibration %+.1f dB toward %.1f LUFS", self.cal_db, target_lufs)
        self.volume_db = float(volume_db)
        self.latency_ms = float(latency_ms)
        self.engine.set_limiter(float(ceiling_dbtp), True)
        self.engine.set_gain_db(self.cal_db + self.volume_db)
        with self._lock:
            self._current = self.base_scene.copy()
            self._target = self.base_scene.copy()
            self._ramps.clear()
        self._k = 0
        self._step_send(0, send=True)                    # block 0's weather is queued before the stream
        self._k = 1
        self._stop.clear()
        self.engine.start(device, float(latency_ms))
        self._live = True
        self._ctl = threading.Thread(target=self._control, name="raingen-native-control", daemon=True)
        self._ctl.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._ctl is not None:
            self._ctl.join(timeout=2.0)
            self._ctl = None
        if self._live:
            try:
                self.engine.stop()
            finally:
                self._live = False
                self.engine.set_limiter(-1.0, False)
                self.engine.set_gain_db(0.0)

    def __enter__(self) -> "NativeSession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def run(self, seconds: Optional[float] = None, poll_s: float = 0.05, **start_kw) -> None:
        """Play (blocking) for `seconds` or until Ctrl+C; delivers markers to on_marker."""
        started_here = not self._live
        if started_here:
            self.start(**start_kw)
        t_end = None if seconds is None else time.monotonic() + float(seconds)
        try:
            while t_end is None or time.monotonic() < t_end:
                self.poll()
                time.sleep(poll_s)
        except KeyboardInterrupt:
            pass
        finally:
            if started_here:
                self.stop()

    wait = run

    def poll(self) -> list:
        """Deliver markers whose audio has been heard to on_marker (caller's thread); returns them."""
        if self.error is not None:
            err, self.error = self.error, None
            raise RuntimeError(f"raingen native session stopped: {err!r}") from err
        now = self.time
        with self._pending_lock:
            due = [m for m in self._pending if m.t <= now]
            if due:
                self._pending = [m for m in self._pending if m.t > now]
        due.sort(key=lambda m: m.t)
        if self.on_marker is not None:
            for m in due:
                self.on_marker(m)
        self.markers_delivered += len(due)
        return due

    def stats(self) -> dict:
        s = dict(self.engine.stats())
        s["control_ms_per_block"] = self.control_ms
        s["calibration_db"] = self.cal_db
        s["scene_updates"] = self.scene_updates
        return s

    # -- live control -------------------------------------------------------------
    def set(self, path: str, value: Any, ramp: Optional[float] = 0.5, immediate: bool = False) -> Any:
        """Change a parameter while playing (thread-safe); the same paths and units as Engine.set."""
        from .engine import PATH_ALIASES, _MACROS, _structure_key
        from .scene import manifest as mf

        path = PATH_ALIASES.get(path, path)
        if path in _LISTENER_LIVE:
            with self._lock:
                self._target.set(path, tuple(value) if path == "listener.position" else float(value))
            if path == "listener.position":
                self.set_listener(position=value)
            else:
                self.set_listener(heading_deg=value)
            return value
        spec = mf.find_param(path)
        with self._lock:
            if spec is not None:
                value = mf.clamp(spec, mf.coerce(spec, value, temp_c=float(self._target.weather.temperature_c)))
            else:
                try:
                    self._target.get(path)
                except (AttributeError, KeyError, TypeError):
                    raise KeyError(f"unknown parameter {path!r}") from None
            old_key = _structure_key(self._target, live_listener=True)
            if path.startswith("vibe."):
                name = path.split(".", 1)[1]
                if name in _MACROS:
                    from .scene.vibe import apply_vibe
                    self._target = apply_vibe(self._target, **{name: float(value)})
                else:
                    vibe = dict(self._target.vibe or {})
                    vibe[name] = value
                    self._target.vibe = vibe
                    if name == "volume_db":
                        self.volume_db = float(value)
                        self.engine.set_gain_db(self.cal_db + self.volume_db)
                        return value
            elif path == "mode":
                from .wellness.modes import apply_mode
                self._target = apply_mode(self._target, str(value))
            else:
                self._target.set(path, value)
            if _structure_key(self._target, live_listener=True) != old_key or path in ("place.window_open",):
                self._rebuild(self._target.copy())
                return value
            self._retarget(float(ramp or 0.0), bool(immediate))
        return value

    def set_scene(self, scene) -> None:
        if not isinstance(scene, Scene):
            from .api import load_scene
            scene = load_scene(scene)
        with self._lock:
            self._target = scene.copy()
            self._rebuild(self._target.copy())

    load_scene = set_scene

    def _rebuild(self, scene: Scene) -> None:
        """Structural change: new patches, World and voice graph (the native graph crossfades one block)."""
        from .core.rng import RngFactory
        from .physics.layout import build_layout
        from .physics.world import World

        if self._live:
            L = scene.listener
            self.set_listener(position=tuple(L.position), heading_deg=float(L.heading_deg))
            scene = self._pin(scene)
            self.rngf = RngFactory(int(scene.seed))
            self.patches = build_layout(scene)
            world = World(scene, self.rngf, sr=self.sr, block=self.block)
            _carry_world(self.world, world)
            self.world = world
            self.engine.set_scene(scene_json(scene, self.precip_types), patches_json(self.patches))
            self.scene_updates += 1
            self.scene = scene.copy()
            self.base_scene = scene.copy()
            self._current = scene.copy()
            self._ramps.clear()
        else:
            self.base_scene = scene.copy()
            self.scene = self.scene_at(0.0)
            L = self.scene.listener
            self._ref_listener = (tuple(float(v) for v in (tuple(L.position) + (0.0, 0.0, 0.0))[:3]),
                                  float(L.heading_deg))
            self._pose = self._ref_listener[0] + (self._ref_listener[1], 0.0)
            self.engine.set_listener(clear=True)
            self._build(self.scene)
            self._current = self.scene.copy()

    def _retarget(self, ramp: float, immediate: bool) -> None:
        from .engine import _live_paths, _safe_get
        from .scene import manifest as mf

        numeric, discrete = _live_paths()
        for p in numeric:
            a, b = _safe_get(self._current, p), _safe_get(self._target, p)
            if a == b:
                continue
            spec = mf.find_param(p)
            isnum = (spec is not None and spec.is_numeric and isinstance(a, (int, float)) and isinstance(b, (int, float))
                     and not isinstance(a, bool) and not isinstance(b, bool))
            dur = max(float(ramp), 0.05) if ramp is not None else (spec.smoothing_s if spec is not None else 0.0)
            if immediate or not isnum or dur <= 0.0 or not self._live:
                self._ramps.pop(p, None)
                self._current.set(p, b)
                continue
            p0, p1 = mf.to_perceptual(spec, a), mf.to_perceptual(spec, b)
            if spec.wrap and spec.min is not None and spec.max is not None and spec.scale == "lin":
                span = float(spec.max - spec.min)
                p1 = p0 + ((p1 - p0 + 0.5 * span) % span) - 0.5 * span
            blocks = max(1, int(round(dur * self.sr / self.block)))
            self._ramps[p] = (spec, float(p0), float(p1), self._k, blocks, b)
        for p in discrete:
            a, b = _safe_get(self._current, p), _safe_get(self._target, p)
            if a != b:
                self._ramps.pop(p, None)
                try:
                    self._current.set(p, b)
                except (AttributeError, KeyError, TypeError):
                    pass
        self._current.timeline = list(self._target.timeline)
        if not self._live:
            self.base_scene = self._current.copy()

    def _advance_ramps(self, k: int) -> None:
        from .scene import manifest as mf

        done = []
        for p, (spec, p0, p1, start, blocks, final) in self._ramps.items():
            u = (k - start + 1) / blocks
            if u >= 1.0:
                self._current.set(p, final)
                done.append(p)
            else:
                self._current.set(p, mf.clamp(spec, mf.from_perceptual(spec, p0 + (p1 - p0) * u)))
        for p in done:
            del self._ramps[p]

    def _step_send(self, k: int, send: bool) -> None:
        with self._lock:
            self._advance_ramps(k)
            sc = self._current
            off = self._t_off
            k_off = int(round(off * self.sr / self.block)) if off else 0
            t0 = k * self.block / self.sr + off
            if sc.timeline:
                from .scene.timeline import evaluate
                sc = evaluate(sc, t0)
            w = self.world.step(max(0, k + k_off), t0, self.block, sc)
        self.last_weather = w
        if send:
            self.engine.update(weather_json(w))

    def _control(self) -> None:
        block_s = self.block / self.sr
        try:
            while not self._stop.is_set():
                t = time.perf_counter()
                done = int(self.engine.blocks_rendered())
                target = done + 1                         # weather for the next block to render (+1 ahead)
                last = target
                stepped = 0
                while self._k <= last:
                    self._step_send(self._k, send=(self._k == last))
                    self._k += 1
                    stepped += 1
                ms = self.engine.poll_markers()
                if ms:
                    with self._pending_lock:
                        self._pending.extend(_marker(*m) for m in ms)
                        if len(self._pending) > 4096:
                            self._pending = self._pending[-4096:]
                if stepped:
                    dt = (time.perf_counter() - t) * 1e3 / stepped
                    self.control_ms = dt if self.control_ms == 0.0 else 0.9 * self.control_ms + 0.1 * dt
                self._stop.wait(0.25 * block_s)
        except Exception as exc:                          # pragma: no cover - surfaced by poll()
            log.exception("raingen native control thread failed")
            self.error = exc

    def __repr__(self) -> str:
        state = "live" if self._live else "offline"
        return f"<raingen.NativeSession {self.scene.name!r} {self.layout} {state}>"
