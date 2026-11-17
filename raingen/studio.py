"""raingen studio: the 4D Composer in an app window, its sound from the native (Rust) core.

    raingen studio                              # Composer + Rust engine, opens an app window
    raingen studio tent-in-a-storm --layout stereo
    raingen studio --no-window --port 8790      # just serve, print the URL

A stdlib-only local server (http.server.ThreadingHTTPServer, bound to 127.0.0.1) serves the
site/ folder and maps the Composer's engine calls (site/engine/native-bridge.js, the same
interface as the browser WeatherEngine) onto ONE raingen.engine.Engine whose pipelines are
raingen.native.NativeSession: voices, encoding, shelter transfer and decoding in Rust, the
World, mastering (calibration, governor, true-peak limiter) and the sound card in Python.

    GET  /native/hello      capabilities, version, engine, voices, the scene named on the CLI
    GET  /native/events     Server-Sent Events: flash, thunder, gust, caption, marker, meter,
                            stats, state, render (offline progress), engine-error
    GET  /native/stats      the same numbers as the 'stats' event
    POST /native/start      {scene?, renderMode?, volumeDb?}  build + prime + open the stream
    POST /native/stop | /native/suspend | /native/resume
    POST /native/scene      {scene}               structure crossfades, weather glides
    POST /native/param      {path, value, ramp} or {params: [{path, value, ramp}, ...]}
    POST /native/listener   {position?, headingDeg?, pitchDeg?}
    POST /native/object     {object}    POST /native/object/remove {id}
    POST /native/volume     {db, ramp?}
    POST /native/render-mode {mode}     binaural | stereo (ignored when --layout is given)
    POST /native/batch      {scene?, objects?, params?, listener?, volume?, renderMode?}
    POST /native/render     {scene?, seconds, sampleRate?, renderMode?}  -> audio/wav

POSTs must be JSON from this origin (Host / Origin checked), so other web pages cannot drive it.
"""

from __future__ import annotations

import collections
import copy
import errno
import http.server
import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import numpy as np

from . import __version__
from .engine import Engine
from .scene.schema import Scene

log = logging.getLogger("raingen.studio")

__all__ = ["Studio", "find_site", "serve", "open_window", "DEFAULT_PORT"]

DEFAULT_PORT = 8790
DEFAULT_SCENE = "steady-rain-leaves-night"
_GUI_LAYOUTS = ("binaural", "stereo")
_MACRO_KEYS = ("intensity", "closeness", "wildness")
_APPLY_EVERY_S = 0.35            # structural changes (objects, listener, scene) at most this often
_IDLE_STOP_S = 15.0              # stop the sound when every Composer window has been gone this long
_CAPTION_KINDS = ("hail", "rain_start", "rain_stop", "gutter_overflow")
_BUFFER_S = 0.2                  # render-ahead; the GUI window runs at lower CPU priority (open_window)
_RENDER_DESTINATION = "streaming_broadcaster"   # -18 LUFS / -1 dBTP for 'Render WAV' (scene destination wins)


# ---------------------------------------------------------------------------
# site folder
# ---------------------------------------------------------------------------
def find_site(explicit: Optional[str] = None) -> Path:
    """The site/ folder: --site, $RAINGEN_SITE, this checkout, or the installed share/raingen/site."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("RAINGEN_SITE"):
        cands.append(Path(os.environ["RAINGEN_SITE"]))
    cands.append(Path(__file__).resolve().parent.parent / "site")
    for base in {sys.prefix, sys.exec_prefix, sysconfig.get_path("data")}:
        cands.append(Path(base) / "share" / "raingen" / "site")
    try:
        import site as _site
        cands.append(Path(_site.getuserbase()) / "share" / "raingen" / "site")
    except Exception:
        pass
    for c in cands:
        if (c / "studio" / "index.html").is_file() and (c / "engine").is_dir():
            return c.resolve()
    raise FileNotFoundError("the Composer files (site/) were not found; run from the repository checkout, "
                            "reinstall raingen, or pass --site PATH (or set RAINGEN_SITE)")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _plain(o: Any) -> Any:
    from .native import _plain as p
    return p(o)


def _num(v: Any, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _dict_set(d: dict, path: str, value: Any) -> bool:
    """Set a dotted path in a plain scene dict (creating dicts on the way); False if blocked."""
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return False
    cur: Any = d
    for p in parts[:-1]:
        if isinstance(cur, dict):
            nxt = cur.get(p)
            if not isinstance(nxt, (dict, list)):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        elif isinstance(cur, list):
            hit = next((o for o in cur if isinstance(o, dict) and str(o.get("id")) == p), None)
            if hit is None:
                try:
                    hit = cur[int(p)]
                except (ValueError, IndexError):
                    return False
            cur = hit
        else:
            return False
    if isinstance(cur, dict):
        cur[parts[-1]] = copy.deepcopy(value)
        return True
    return False


def _canonical_layout(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    from .spatial.decoders import normalize_layout
    return normalize_layout(str(name))


class _MeteredEngine(Engine):
    """The live Engine plus a per-block (end frame, mean square, peak) log for the Composer's meter."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.meter_log: "collections.deque" = collections.deque(maxlen=1024)

    def _push(self, block: np.ndarray) -> None:
        super()._push(block)
        if block.size:
            b = block.astype(np.float64, copy=False)
            ms = float(np.mean(b * b, axis=0).sum())          # BS.1770 channel sum (G = 1 for L/R), unweighted
            pk = float(np.max(np.abs(b)))
            self.meter_log.append((self._pushed, int(block.shape[0]), ms, pk))

    def meter(self, window_s: float = 0.4) -> tuple[float, float]:
        """(momentary loudness ~LUFS (unweighted 400 ms), sample peak dBFS) of the audio being heard now."""
        now = self._consumed
        lo = now - int(window_s * self.sr_out)
        acc = n = 0
        pk = 0.0
        for end, frames, ms, p in list(self.meter_log):
            if end <= now and end > lo:
                acc += ms * frames
                n += frames
                pk = max(pk, p)
        if n == 0:
            return -120.0, -120.0
        lufs = -0.691 + 10.0 * math.log10(acc / n + 1e-20)
        return max(-120.0, lufs), max(-120.0, 20.0 * math.log10(pk + 1e-12))


# ---------------------------------------------------------------------------
# the studio: one live engine + the GUI's scene mirror + event fan-out
# ---------------------------------------------------------------------------
class Studio:
    """One live Engine driven by the Composer; thread-safe (HTTP handler threads + a pump thread)."""

    def __init__(self, site_dir: Path, scene=None, *, layout: Optional[str] = None, device=None,
                 quality: str = "game", verbose: bool = False):
        from .api import load_scene

        self.site_dir = Path(site_dir)
        self.fixed_layout = _canonical_layout(layout)
        self.device = device
        self.quality = quality
        self.verbose = bool(verbose)
        named = scene is not None
        sc = load_scene(scene if named else DEFAULT_SCENE)
        self.scene: dict = sc.to_dict()                  # mirror of the Composer's scene (rainscene dict)
        self.cli_scene: Optional[dict] = self.scene if named else None
        self.cli_scene_id = f"{sc.name}:{int(time.time())}" if named else None
        r = str(getattr(sc.listener, "render", "") or "")
        self.render_mode = r if r in _GUI_LAYOUTS else "binaural"
        self.volume_db = 0.0
        self.engine: Optional[_MeteredEngine] = None
        self.paused = False
        self.t = 0.0
        self.lock = threading.RLock()
        self.start_lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.dirty = False
        self.last_apply = 0.0
        self.stopping = threading.Event()
        self._clients: set = set()
        self._clients_lock = threading.Lock()
        self.had_client = False
        self.last_client = time.monotonic()
        self._flash_bearing: "collections.OrderedDict" = collections.OrderedDict()
        self._last_caption: dict = {}
        self.last_error: Optional[str] = None
        from . import native as nat
        self.native_available = nat.available()
        self.native_error = None if nat.available() else repr(nat.IMPORT_ERROR)
        self._pump = threading.Thread(target=self._pump_loop, name="raingen-studio-pump", daemon=True)
        self._pump.start()

    # -- layout / scene ----------------------------------------------------------
    @property
    def layout(self) -> str:
        return self.fixed_layout or self.render_mode

    def _scene_obj(self, d: dict, live: bool) -> Scene:
        """The Composer's scene dict -> a Scene for the engine (unknown keys ignored by from_dict).

        Live: the Composer plays its own automation (as /native/param), so the timeline is dropped;
        its Auto/Offset/Pinned overrides and 0..100 vibe macros are already resolved into values.
        """
        d = copy.deepcopy(d) if isinstance(d, dict) else {}
        vibe = d.get("vibe") if isinstance(d.get("vibe"), dict) else {}
        d["vibe"] = {k: v for k, v in vibe.items() if k not in _MACRO_KEYS}
        d["overrides"] = {}
        tl = d.get("timeline") if isinstance(d.get("timeline"), list) else []
        d["timeline"] = [] if live else [k for k in tl if isinstance(k, dict) and "at" in k and "path" in k
                                        and not str(k["path"]).startswith(("place.objects.", "studio."))]
        try:
            d["seed"] = int(d.get("seed", 0)) & 0xFFFFFFFF
        except (TypeError, ValueError):
            d["seed"] = abs(hash(str(d.get("seed")))) & 0xFFFFFFFF
        sc = Scene.from_dict(d)
        if not live:
            keep = []
            for k in sc.timeline:
                try:
                    sc.get(k.path)
                    keep.append(k)
                except (AttributeError, KeyError, TypeError):
                    pass
            sc.timeline = keep
        from .wellness.modes import apply_mode, canonical_mode
        if live and canonical_mode(sc.mode) != "none" and "guard" not in (sc.vibe or {}):
            sc = apply_mode(sc, sc.mode)                 # as Engine() does for its first scene
        return sc

    def _apply_scene_now(self) -> None:
        with self.lock:
            self.dirty = False
            self.last_apply = time.monotonic()
            eng = self.engine
            d = copy.deepcopy(self.scene)
        if eng is None:
            return
        try:
            eng.set_scene(self._scene_obj(d, live=True))
        except Exception as exc:
            self._fail("scene", exc)

    def _mark_dirty(self) -> None:
        with self.lock:
            self.dirty = True

    # -- lifecycle ---------------------------------------------------------------
    def start(self, scene: Optional[dict] = None, render_mode: Optional[str] = None,
              volume_db: Optional[float] = None) -> dict:
        with self.start_lock:
            with self.lock:
                if isinstance(scene, dict):
                    self.scene = copy.deepcopy(scene)
                if render_mode in _GUI_LAYOUTS:
                    self.render_mode = render_mode
                if volume_db is not None:
                    self.volume_db = _num(volume_db, self.volume_db)
                eng = self.engine
            if eng is not None and eng.running:
                if self.paused:
                    self.resume()
                self._apply_scene_now()
                self._set_volume_now(self.volume_db, 0.05)
                return self.info()
            if eng is not None:                          # a dead engine (producer error): replace it
                self.stop()
            sc = self._scene_obj(self.scene, live=True)
            eng = _MeteredEngine(sc, layout=self.layout, device=self.device, quality=self.quality,
                                 volume_db=self.volume_db, on_marker=self._on_marker, buffer_s=_BUFFER_S)
            t0 = time.perf_counter()
            eng.start(audio=True)
            with self.lock:
                self.engine = eng
                self.paused = False
                self.dirty = False
                self.last_error = None
            info = self.info()
            log.info("studio: started %s (%s, %s) in %.2f s", sc.name, info["engine"], self.layout,
                     time.perf_counter() - t0)
            if self.verbose:
                print(f"[studio] playing {sc.name!r}: {info['engine']} core, {self.layout}, "
                      f"{info['voices']} voices", file=sys.stderr, flush=True)
            self.emit("state", {"state": "running"})
            self.emit("stats", self.stats())
            return info

    def stop(self) -> dict:
        with self.lock:
            eng, self.engine = self.engine, None
            self.paused = False
        if eng is not None:
            try:
                eng.stop()
            except Exception as exc:
                log.warning("studio: stop failed: %s", exc)
            self.emit("state", {"state": "stopped"})
        return {"ok": True, "running": False}

    def suspend(self) -> dict:
        eng = self.engine
        if eng is not None and eng._stream is not None and not self.paused:
            try:
                eng._stream.stop()
            except Exception as exc:
                log.warning("studio: pause failed: %s", exc)
            self.paused = True
            self.emit("state", {"state": "suspended"})
        return self.info()

    def resume(self) -> dict:
        eng = self.engine
        if eng is not None and eng._stream is not None and self.paused:
            try:
                eng._stream.start()
            except Exception as exc:
                log.warning("studio: resume failed: %s", exc)
            self.paused = False
            self.emit("state", {"state": "running"})
        return self.info()

    def close(self) -> None:
        self.stopping.set()
        self.stop()
        with self._clients_lock:
            for q in list(self._clients):
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass

    # -- live control --------------------------------------------------------------
    def set_scene(self, scene: dict) -> dict:
        if not isinstance(scene, dict):
            raise ValueError("scene must be a JSON object")
        Scene.from_dict(scene)                           # reject garbage early (400)
        with self.lock:
            self.scene = copy.deepcopy(scene)
            r = str(((scene.get("listener") or {}).get("render")) or "")
            if r in _GUI_LAYOUTS and r != self.render_mode and not self.fixed_layout:
                self.set_render_mode(r)
            self.dirty = True
        return {"ok": True}

    def set_param(self, path: str, value: Any, ramp: Optional[float] = 0.5) -> dict:
        path = str(path or "")
        if not path:
            raise ValueError("path is required")
        if path.startswith("mix."):
            if path in ("mix.master", "mix.master_db"):
                return self.set_volume(value, ramp)
            return {"ok": True, "ignored": path}
        if path == "listener.render":
            return self.set_render_mode(value)
        if path.startswith("place.objects."):
            with self.lock:
                _dict_set(self.scene, path, value)
                self.dirty = True
            return {"ok": True, "applied": "scene"}
        with self.lock:
            _dict_set(self.scene, path, value)
            eng = self.engine
        if eng is None:
            return {"ok": True, "applied": "mirror"}
        try:
            v = eng.set(path, value, ramp=_num(ramp, 0.5))
            return {"ok": True, "applied": "engine", "value": _plain(v)}
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            self._mark_dirty()                           # not a live parameter: rebuild from the mirror
            return {"ok": True, "applied": "scene", "note": str(exc)}

    def set_listener(self, position=None, heading_deg=None, pitch_deg=None, height_m=None) -> dict:
        with self.lock:
            L = self.scene.setdefault("listener", {})
            changed = False
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                p = [_num(position[0], 0.0), _num(position[1], 0.0),
                     _num(position[2], 0.0) if len(position) > 2 else 0.0]
                old = L.get("position") or [0.0, 0.0, 0.0]
                if max(abs(a - b) for a, b in zip(p, list(old) + [0.0] * 3)) > 0.05:
                    changed = True
                L["position"] = p
            if heading_deg is not None:
                h = _num(heading_deg, 0.0) % 360.0
                if abs(((h - _num(L.get("heading_deg"), 0.0)) + 180.0) % 360.0 - 180.0) > 0.5:
                    changed = True
                L["heading_deg"] = h
            if height_m is not None:
                L["height_m"] = _num(height_m, 1.7)
                changed = True
            if changed:
                self.dirty = True
        return {"ok": True}

    def upsert_object(self, obj: dict) -> dict:
        if not isinstance(obj, dict):
            raise ValueError("object must be a JSON object")
        with self.lock:
            place = self.scene.setdefault("place", {})
            objs = place.setdefault("objects", [])
            oid = str(obj.get("id")) if obj.get("id") is not None else f"obj{len(objs)}"
            o = dict(copy.deepcopy(obj), id=oid)
            for i, cur in enumerate(objs):
                if isinstance(cur, dict) and str(cur.get("id")) == oid:
                    objs[i] = o
                    break
            else:
                objs.append(o)
            self.dirty = True
        return {"ok": True, "id": oid}

    def remove_object(self, oid) -> dict:
        with self.lock:
            place = self.scene.setdefault("place", {})
            objs = place.get("objects") or []
            place["objects"] = [o for o in objs if not (isinstance(o, dict) and str(o.get("id")) == str(oid))]
            self.dirty = len(place["objects"]) != len(objs) or self.dirty
        return {"ok": True}

    def _set_volume_now(self, db: float, ramp: float) -> None:
        eng = self.engine
        if eng is not None:
            # straight to the output gain ramp: no scene diff on the producer thread for the GUI's 20 Hz
            # fades, and no manifest clamp at -60 dB (the Composer's volume goes down to silence)
            eng._set_volume(float(db), max(0.0, float(ramp)), False)

    def set_volume(self, db, ramp=0.05) -> dict:
        db = max(-120.0, min(12.0, _num(db, 0.0)))
        self.volume_db = db
        self._set_volume_now(db, _num(ramp, 0.05))
        return {"ok": True, "db": db}

    def set_render_mode(self, mode) -> dict:
        mode = "stereo" if str(mode) == "stereo" else "binaural"
        with self.lock:
            self.scene.setdefault("listener", {})["render"] = mode
            if self.fixed_layout:
                return {"ok": True, "layout": self.layout, "fixed": True}
            changed = mode != self.render_mode
            self.render_mode = mode
            eng = self.engine
        if changed and eng is not None:
            eng.layout = mode                            # binaural and stereo are both 2 channels
            with eng._lock:
                eng._struct_key = ""                     # force a rebuild (+ crossfade) in the new layout
            eng._cmd.put(("retarget", 0.5))
        return {"ok": True, "layout": self.layout, "fixed": False}

    def batch(self, body: dict) -> dict:
        if isinstance(body.get("scene"), dict):
            self.set_scene(body["scene"])
        if body.get("renderMode"):
            self.set_render_mode(body["renderMode"])
        for op in body.get("objects") or []:
            if not isinstance(op, dict):
                continue
            if op.get("op") == "remove":
                self.remove_object(op.get("id"))
            elif isinstance(op.get("object"), dict):
                self.upsert_object(op["object"])
        for p in body.get("params") or []:
            if isinstance(p, dict) and p.get("path"):
                self.set_param(p["path"], p.get("value"), p.get("ramp", 0.5))
        L = body.get("listener")
        if isinstance(L, dict):
            self.set_listener(L.get("position"), L.get("headingDeg"), L.get("pitchDeg"), L.get("heightM"))
        V = body.get("volume")
        if isinstance(V, dict) and "db" in V:
            self.set_volume(V["db"], V.get("ramp", 0.05))
        if "t" in body:
            self.t = _num(body["t"], self.t)
        return {"ok": True}

    # -- offline render --------------------------------------------------------------
    def render_wav(self, scene: Optional[dict], seconds, sample_rate=None, render_mode=None) -> bytes:
        from .render import Renderer

        secs = max(1.0, min(3600.0, _num(seconds, 60.0)))
        sr = int(_num(sample_rate, 48000)) if sample_rate else 48000
        with self.lock:
            d = copy.deepcopy(scene if isinstance(scene, dict) else self.scene)
        layout = self.fixed_layout or (render_mode if render_mode in _GUI_LAYOUTS else None) or self.render_mode
        sc = self._scene_obj(d, live=False)
        if not self.render_lock.acquire(blocking=False):
            raise RuntimeError("a render is already running")
        last = [0.0]

        def progress(f, phase):
            now = time.monotonic()
            if now - last[0] >= 0.1 or phase == "done":
                last[0] = now
                self.emit("render", {"fraction": float(f), "phase": str(phase)})
            return None

        try:
            with tempfile.TemporaryDirectory(prefix="raingen-studio-") as tmp:
                out = os.path.join(tmp, "render.wav")
                t0 = time.perf_counter()
                dest = str(getattr(sc.render, "destination", "") or "raw")
                if dest == "raw":
                    dest = _RENDER_DESTINATION                # loudness-normalised, like the live calibration
                Renderer(sc, secs, layout=layout, destination=dest, stems=False, progress=progress,
                         sample_rate=sr).run(out)
                data = Path(out).read_bytes()
                log.info("studio: rendered %.1f s (%s) in %.1f s", secs, layout, time.perf_counter() - t0)
                return data
        finally:
            self.render_lock.release()

    # -- state ------------------------------------------------------------------------
    def _pipe_info(self) -> tuple[str, int, dict]:
        eng = self.engine
        if eng is None:
            return ("native" if self.native_available else "python"), 0, {}
        pipe = eng._pipe
        rn = getattr(pipe, "engine", None)
        if rn is not None and hasattr(rn, "stats"):
            try:
                st = dict(rn.stats())
            except Exception:
                st = {}
            return "native", int(st.get("voices") or len(getattr(pipe, "voices", []) or [])), st
        return "python", 0, {}

    def info(self) -> dict:
        kind, voices, _ = self._pipe_info()
        eng = self.engine
        return {"ok": True, "engine": kind, "voices": voices, "layout": self.layout,
                "fixedLayout": bool(self.fixed_layout), "running": bool(eng is not None and eng.running),
                "paused": bool(self.paused), "volumeDb": self.volume_db}

    def hello(self) -> dict:
        d = self.info()
        d.update({"app": "raingen studio", "version": __version__, "native": self.native_available,
                  "nativeError": self.native_error, "scene": self.cli_scene, "sceneId": self.cli_scene_id,
                  "lastError": self.last_error})
        return d

    def stats(self) -> dict:
        kind, voices, st = self._pipe_info()
        eng = self.engine
        out = {"engine": kind, "voices": voices, "layout": self.layout,
               "running": bool(eng is not None and eng.running), "paused": bool(self.paused)}
        if eng is not None:
            out.update({"t": round(float(eng.time), 3), "underruns": int(eng.underruns),
                        "cpuLoad": round(float(eng.cpu_load), 4), "badge": eng.badge,
                        "voicesActive": int(st.get("voices_active", 0) or 0),
                        "cpuMsPerBlock": round(float(st.get("cpu_ms_per_block", 0.0) or 0.0), 3),
                        "threads": int(st.get("threads", 0) or 0)})
        return out

    # -- events ------------------------------------------------------------------------
    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=512)
        with self._clients_lock:
            self._clients.add(q)
            self.had_client = True
            self.last_client = time.monotonic()
        return q

    def unsubscribe(self, q) -> None:
        with self._clients_lock:
            self._clients.discard(q)
            self.last_client = time.monotonic()

    def emit(self, event: str, data: dict) -> None:
        msg = (event, json.dumps(_plain(data), separators=(",", ":")))
        with self._clients_lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def _fail(self, where: str, exc: BaseException) -> None:
        self.last_error = f"{where}: {exc}"
        log.warning("studio: %s failed: %s", where, exc)
        self.emit("engine-error", {"where": where, "message": str(exc)})

    def _caption(self, kind: str, text: str, t: float, every_s: float = 0.0) -> None:
        now = time.monotonic()
        if every_s and now - self._last_caption.get(kind, -1e9) < every_s:
            return
        self._last_caption[kind] = now
        self.emit("caption", {"text": text, "t": t, "kind": kind})

    def _on_marker(self, m) -> None:
        from .io.captions import caption_for

        kind = str(m.kind).lower()
        d = m.data or {}
        t = float(m.t)
        if kind in ("lightning", "flash"):
            fid = str(d.get("id", f"n{t:.3f}"))
            bearing = d.get("bearing_deg")
            self._flash_bearing[fid] = bearing
            while len(self._flash_bearing) > 256:
                self._flash_bearing.popitem(last=False)
            si = _num(d.get("intensity"), 0.6)
            self.emit("flash", {"id": fid, "t": t, "distanceKm": d.get("distance_km"), "bearingDeg": bearing,
                                "delaySec": d.get("flash_to_bang_s"), "kind": d.get("type"),
                                "brightness": max(0.35, min(1.0, 0.5 + si)), "channel": None,
                                "source": "native"})
            self._caption(kind, caption_for(m), t)
        elif kind == "thunder":
            fid = str(d.get("id", ""))
            self.emit("thunder", {"id": fid or None, "t": t, "distanceKm": d.get("distance_km"),
                                  "bearingDeg": self._flash_bearing.get(fid), "audible": d.get("audible", True),
                                  "peakDbSpl": d.get("peak_db_spl")})
            if d.get("audible", True):
                self._caption(kind, caption_for(m), t)
        elif kind == "gust":
            with self.lock:
                mean = _num(((self.scene.get("weather") or {}).get("wind") or {}).get("speed_mps"), 0.0)
            self.emit("gust", {"t": t, "speedMps": d.get("wind_10m_mps", d.get("wind_mps")),
                               "fromDeg": d.get("from_deg"), "meanMps": mean, "gustFactor": d.get("gust"),
                               "beaufort": d.get("beaufort")})
        else:
            self.emit("marker", {"kind": kind, "t": t, "data": d})
            if kind in _CAPTION_KINDS:
                self._caption(kind, caption_for(m), t, every_s=10.0)

    def _pump_loop(self) -> None:
        last_meter = last_stats = 0.0
        while not self.stopping.is_set():
            now = time.monotonic()
            eng = self.engine
            if eng is not None and eng.running:
                try:
                    eng.poll()
                except Exception as exc:
                    self._fail("engine", exc)
                if now - last_meter >= 0.1:
                    last_meter = now
                    if self.paused:
                        self.emit("meter", {"lufsMomentary": -120.0, "peakDb": -120.0, "t": self.t})
                    else:
                        lufs, pk = eng.meter()
                        self.emit("meter", {"lufsMomentary": round(lufs, 2), "peakDb": round(pk, 2), "t": self.t})
                if now - last_stats >= 1.0:
                    last_stats = now
                    self.emit("stats", self.stats())
            if self.dirty and now - self.last_apply >= _APPLY_EVERY_S:
                self._apply_scene_now()
            with self._clients_lock:
                idle = self.had_client and not self._clients and now - self.last_client > _IDLE_STOP_S
            if idle and eng is not None:
                print("[studio] no Composer window is open: stopping the sound", file=sys.stderr, flush=True)
                self.stop()
                self.had_client = False
            self.stopping.wait(0.05)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(http.server.SimpleHTTPRequestHandler):
    server_version = "raingen-studio"
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
                      ".svg": "image/svg+xml", ".json": "application/json", ".wasm": "application/wasm",
                      ".webmanifest": "application/manifest+json", ".wav": "audio/wav"}
    studio: Studio = None  # type: ignore[assignment]
    port: int = 0

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(self.studio.site_dir), **kw)

    def log_message(self, fmt, *args) -> None:
        if self.studio.verbose:
            super().log_message(fmt, *args)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # -- guards -------------------------------------------------------------------
    def _allowed(self) -> set:
        p = self.port
        return {f"127.0.0.1:{p}", f"localhost:{p}", f"[::1]:{p}"}

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").lower() in self._allowed()

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlsplit(origin).netloc.lower() in self._allowed()

    # -- replies ------------------------------------------------------------------
    def _json(self, code: int, obj: Any) -> None:
        data = json.dumps(_plain(obj), separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, where: str) -> None:
        self.send_response(302)
        self.send_header("Location", where)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- GET ----------------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("", "/", "/studio"):
            return self._redirect("/studio/?engine=native")
        if path.startswith("/native/"):
            if not self._host_ok():
                return self._json(403, {"ok": False, "error": "forbidden host"})
            if path == "/native/hello":
                return self._json(200, self.studio.hello())
            if path == "/native/stats":
                return self._json(200, self.studio.stats())
            if path == "/native/events":
                return self._events()
            return self._json(404, {"ok": False, "error": f"unknown endpoint {path}"})
        return super().do_GET()

    def _events(self) -> None:
        st = self.studio
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = st.subscribe()
        try:
            hello = json.dumps(_plain(st.info()), separators=(",", ":"))
            self.wfile.write(f"retry: 1500\nevent: hello\ndata: {hello}\n\n".encode("utf-8"))
            self.wfile.flush()
            while not st.stopping.is_set():
                try:
                    msg = q.get(timeout=10.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if msg is None:
                    break
                ev, data = msg
                self.wfile.write(f"event: {ev}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            st.unsubscribe(q)
            self.close_connection = True

    # -- POST ---------------------------------------------------------------------
    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if not path.startswith("/native/"):
            return self._json(404, {"ok": False, "error": "not found"})
        if not self._host_ok() or not self._origin_ok():
            return self._json(403, {"ok": False, "error": "forbidden origin"})
        if "application/json" not in (self.headers.get("Content-Type") or "").lower():
            return self._json(415, {"ok": False, "error": "send JSON (Content-Type: application/json)"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > 32 * 1024 * 1024:
            return self._json(413, {"ok": False, "error": "request too large"})
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
        except ValueError as exc:
            return self._json(400, {"ok": False, "error": f"bad JSON: {exc}"})
        if not isinstance(body, dict):
            return self._json(400, {"ok": False, "error": "send a JSON object"})
        st = self.studio
        try:
            if path == "/native/render":
                return self._render(body)
            if path == "/native/start":
                res = st.start(body.get("scene"), body.get("renderMode"), body.get("volumeDb"))
            elif path == "/native/stop":
                res = st.stop()
            elif path == "/native/suspend":
                res = st.suspend()
            elif path == "/native/resume":
                res = st.resume()
            elif path == "/native/scene":
                res = st.set_scene(body.get("scene", body))
            elif path == "/native/param":
                items = body.get("params") if isinstance(body.get("params"), list) else [body]
                res = {"ok": True, "results": [st.set_param(p.get("path"), p.get("value"), p.get("ramp", 0.5))
                                               for p in items if isinstance(p, dict)]}
            elif path == "/native/listener":
                res = st.set_listener(body.get("position"), body.get("headingDeg"), body.get("pitchDeg"),
                                      body.get("heightM"))
            elif path == "/native/object":
                res = st.upsert_object(body.get("object", body))
            elif path == "/native/object/remove":
                res = st.remove_object(body.get("id"))
            elif path == "/native/volume":
                res = st.set_volume(body.get("db", 0.0), body.get("ramp", 0.05))
            elif path == "/native/render-mode":
                res = st.set_render_mode(body.get("mode"))
            elif path == "/native/batch":
                res = st.batch(body)
            else:
                return self._json(404, {"ok": False, "error": f"unknown endpoint {path}"})
        except (ValueError, KeyError, TypeError) as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            log.exception("studio: %s failed", path)
            st.last_error = f"{path}: {exc}"
            return self._json(500, {"ok": False, "error": str(exc)})
        return self._json(200, res)

    def _render(self, body: dict) -> None:
        data = self.studio.render_wav(body.get("scene"), body.get("seconds", 60), body.get("sampleRate"),
                                      body.get("renderMode"))
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", 'attachment; filename="raingen-render.wav"')
        self.end_headers()
        self.wfile.write(data)


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _bind(port: Optional[int], handler) -> _Server:
    tries = [int(port)] if port else [DEFAULT_PORT + i for i in range(20)]
    last: Optional[OSError] = None
    for p in tries:
        try:
            return _Server(("127.0.0.1", p), handler)
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1), errno.EACCES):
                raise
            last = exc
    raise OSError(f"no free port for the studio server ({tries[0]}..{tries[-1]}): {last}")


# ---------------------------------------------------------------------------
# app window
# ---------------------------------------------------------------------------
def _browsers() -> list[str]:
    names = ["microsoft-edge", "microsoft-edge-stable", "msedge", "google-chrome", "google-chrome-stable",
             "chromium", "chromium-browser", "chrome", "brave-browser", "brave"]
    paths = ["/opt/microsoft/msedge/msedge", "/opt/google/chrome/chrome",
             "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            paths += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    out = [p for p in paths if os.path.isfile(p) and os.access(p, os.X_OK)]
    for n in names:
        w = shutil.which(n)
        if w and w not in out:
            out.append(w)
    return out


def _profile_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    p = os.path.join(base, "raingen", "studio-window")
    os.makedirs(p, exist_ok=True)
    return p


def _low_priority() -> dict:
    """Run the GUI window below normal CPU priority so the audio engine always wins the CPU.

    Lowering priority needs no privileges (raising the audio thread's would), and the window
    uses its own browser profile, so this touches only the Composer window.
    """
    if os.name == "nt":
        return {"creationflags": 0x00004000}          # BELOW_NORMAL_PRIORITY_CLASS
    return {"preexec_fn": lambda: os.nice(10)}


def open_window(url: str) -> Optional[subprocess.Popen]:
    """An app window (Edge / Chrome / Chromium --app=URL, own profile) or the default browser."""
    for exe in _browsers():
        try:
            return subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={_profile_dir()}", "--no-first-run",
                                     "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
                                     "--window-size=1440,900"],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=(os.name != "nt"), **_low_priority())
        except OSError:
            continue
    webbrowser.open(url)
    return None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def serve(scene=None, *, port: Optional[int] = None, window: bool = True, layout: Optional[str] = None,
          device=None, quality: str = "game", site: Optional[str] = None, autoplay: bool = False,
          verbose: bool = False) -> int:
    """Run the studio until Ctrl+C (or SIGTERM, or the app window closes); stops the sound cleanly."""
    site_dir = find_site(site)
    studio = Studio(site_dir, scene, layout=layout, device=device, quality=quality, verbose=verbose)
    handler = type("StudioHandler", (_Handler,), {"studio": studio})
    httpd = _bind(port, handler)
    handler.port = int(httpd.server_address[1])
    url = f"http://127.0.0.1:{handler.port}/studio/?engine=native"
    th = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                          name="raingen-studio-http", daemon=True)
    th.start()
    done = threading.Event()

    def _term(*_):
        done.set()

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
        if sig is not None:
            try:
                signal.signal(sig, _term)
            except (ValueError, OSError):
                pass

    core = "native (Rust)" if studio.native_available else f"python (native core unavailable: {studio.native_error})"
    print(f"raingen studio {__version__}: {url}", flush=True)
    print(f"  engine: {core}, layout {studio.layout}{' (fixed)' if studio.fixed_layout else ''}; "
          f"site: {site_dir}. Ctrl+C to stop.", file=sys.stderr, flush=True)
    if autoplay:
        threading.Thread(target=lambda: _safe_start(studio), daemon=True).start()
    proc = open_window(url) if window else None
    if proc is not None:
        def _watch():
            t0 = time.monotonic()
            proc.wait()
            if time.monotonic() - t0 > 5.0:              # a quick exit = handed to a running browser
                print("[studio] window closed", file=sys.stderr, flush=True)
                done.set()
        threading.Thread(target=_watch, daemon=True).start()
    try:
        while not done.wait(0.25):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[studio] stopping…", file=sys.stderr, flush=True)
        studio.close()
        httpd.shutdown()
        httpd.server_close()
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    return 0


def _safe_start(studio: Studio) -> None:
    try:
        studio.start()
    except Exception as exc:
        studio._fail("start", exc)
