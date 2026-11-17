"""The public Python API (MASTER_PLAN 3.1).

    import raingen as rg

    rg.render("rain on a tin roof, wind 30 km/h from the west, distant thunder",
              minutes=60, out="cabin_storm.flac", destination="youtube")
    scene = rg.preset("tent-in-a-storm")
    scene.weather.wind.speed_mps = rg.beaufort(5)
    for v in scene.variations(20):
        v.render(seconds=90, out="lib/")
    with rg.Engine(scene) as eng:
        eng.set("weather.wind.speed_mps", "force 7", ramp=2.0)

Anything that names a scene accepts a Scene, a dict in scene-file form, a path to a
.json / .yaml / .wxscene file, a preset name, a METAR report or plain words. Scenes are
plain data; rendering takes a frozen snapshot, so editing a scene afterwards never
affects a render in progress. Variations derive their seeds with RngFactory.child, so the
same scene and count always give the same set.
"""

from __future__ import annotations

import math
import os
import sys
import weakref
from typing import Callable, Optional

from .core.rng import RngFactory
from .engine import Engine
from .render import RenderReport, Renderer, render_array as _render_array, thunder_one_shots
from .scene import manifest as _mf
from .scene.manifest import (PARAMS, fmod_manifest, get_param, markdown_reference, osc_namespace, to_json_schema,
                             wwise_manifest)
from .scene.presets import PRESETS, list_presets as _list_presets, preset as _preset
from .scene.schema import Keyframe, Listener, Place, Precipitation, RenderSettings, Scene, SceneObject, Thunder, \
    Weather, Wind
from .scene.textparse import TextParseError
from .scene.vibe import apply_vibe
from .wellness.modes import apply_mode

__all__ = [
    "Scene", "Weather", "Precipitation", "Wind", "Thunder", "Place", "SceneObject", "Listener", "Keyframe",
    "RenderSettings", "Engine", "Renderer", "RenderReport", "SceneParseError",
    "preset", "list_presets", "scene_from_text", "scene_from_metar", "scene_from_live", "load_scene",
    "render", "render_array", "play", "beaufort", "variations", "describe", "one_shots",
    "apply_vibe", "apply_mode", "manifest", "destinations", "PARAMS", "get_param", "to_json_schema",
    "osc_namespace", "wwise_manifest", "fmod_manifest", "markdown_reference", "render_file", "duration_seconds",
]

SCENE_EXTENSIONS = (".json", ".yaml", ".yml", ".wxscene", ".rainscene", ".scene")
DEFAULT_SECONDS = 60.0


class SceneParseError(TextParseError):
    """Raised by strict parsing when free text names nothing the parser knows.

    A TextParseError (and so a ValueError): `except TextParseError` catches it too.
    """


# ---------------------------------------------------------------------------
# scene resolution
# ---------------------------------------------------------------------------
def _read_scene_file(path: str) -> Scene:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return Scene.from_json(text)
    if ext not in (".yaml", ".yml"):
        try:
            return Scene.from_json(text)
        except ValueError:
            pass
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("YAML scene files need the optional 'pyyaml' package: "
                          "pip install 'rain-generator[yaml]'") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a scene mapping")
    return Scene.from_dict(data)


def _preset_key(text: str) -> Optional[str]:
    key = text.strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in key:
        key = key.replace("--", "-")
    if key in PRESETS:
        return key
    for name, spec in PRESETS.items():
        if spec["title"].strip().lower() == text.strip().lower():
            return name
    return None


def _resolve(what, base: Optional[Scene] = None, strict: bool = False) -> tuple[Scene, list[str]]:
    """(Scene, list of understood phrases) for anything that names a scene."""
    from .scene.metar import looks_like_metar, scene_from_metar as _from_metar
    from .scene.textparse import apply_parse, parse_text

    if isinstance(what, Scene):
        return what.copy(), []
    if isinstance(what, dict):
        return Scene.from_dict(what), []
    if isinstance(what, os.PathLike):
        what = os.fspath(what)
    if not isinstance(what, str):
        raise TypeError(f"cannot make a scene from {type(what).__name__}")
    text = what.strip()
    if not text:
        raise ValueError("empty scene description")
    path = os.path.expanduser(text)
    if os.path.isfile(path):
        return _read_scene_file(path), []
    if text.lower().endswith(SCENE_EXTENSIONS) and len(text.split()) == 1:
        raise FileNotFoundError(f"scene file not found: {path}")
    key = _preset_key(text)
    if key is not None:
        return _preset(key), [f"preset {key}"]
    if looks_like_metar(text):
        return _from_metar(text, base), ["METAR " + text.split()[0]]
    try:
        res = parse_text(text, base, strict=strict)
    except TextParseError as exc:
        if isinstance(exc, SceneParseError):
            raise
        hint = ", ".join(sorted(PRESETS)[:6])
        raise SceneParseError(f"{exc}; or name a preset ({hint}, ...)") from exc
    return apply_parse(res, base), [phrase for phrase, _ in res.matched]


def load_scene(what, base: Optional[Scene] = None) -> Scene:
    """A Scene from a Scene, dict, file path, preset name, METAR or plain words."""
    return _resolve(what, base)[0]


def preset(name: str) -> Scene:
    """A fresh Scene for a named preset (see list_presets())."""
    return _preset(name)


def list_presets(tag: Optional[str] = None) -> list[dict]:
    """Preset cards (name, title, tags, description, place, mode), optionally filtered by tag."""
    return _list_presets(tag)


def scene_from_text(text: str, base: Optional[Scene] = None) -> Scene:
    """'Describe your weather' -> Scene (a pasted METAR is parsed as a METAR)."""
    from .scene.textparse import scene_from_text as _from_text
    return _from_text(text, base)


def scene_from_metar(text: str, base: Optional[Scene] = None, utc_offset_h: Optional[float] = None) -> Scene:
    """Scene from a METAR report (wind, present weather, temperature, pressure, cloud)."""
    from .scene.metar import scene_from_metar as _from_metar
    return _from_metar(text, base, utc_offset_h=utc_offset_h)


def scene_from_live(location, base: Optional[Scene] = None, provider: str = "open-meteo", **kw) -> Scene:
    """Scene from current weather at a place name, 'lat,lon', (lat, lon) or ICAO station (network)."""
    from .live.providers import scene_from_live as _from_live
    return _from_live(location, base, provider=provider, **kw)


# ---------------------------------------------------------------------------
# durations and rendering
# ---------------------------------------------------------------------------
def duration_seconds(seconds: Optional[float] = None, minutes: Optional[float] = None,
                     hours: Optional[float] = None, duration=None, default: float = DEFAULT_SECONDS) -> float:
    """Sum of the given parts ('duration' may be '90 s', '5 min', '1.5 h', '01:30:00'); default if none."""
    total = 0.0
    given = False
    for v, k in ((seconds, 1.0), (minutes, 60.0), (hours, 3600.0)):
        if v is not None:
            total += float(v) * k
            given = True
    if duration is not None:
        from .scene.units import parse_duration_s
        total += float(parse_duration_s(duration))
        given = True
    if not given:
        return float(default)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("the render length must be positive")
    return total


def render_file(what, seconds: Optional[float] = None, minutes: Optional[float] = None,
                hours: Optional[float] = None, out=None, layout: Optional[str] = None,
                destination: Optional[str] = None, seed: Optional[int] = None, stems: bool = False, captions=False,
                mode: Optional[str] = None, progress: Optional[Callable] = None, duration=None, strict: bool = False,
                **kw) -> RenderReport:
    """Render anything that names a scene to a file; returns the report (a dict with a readable str()).

    This is `raingen.render(...)` (and `raingen.api.render(...)`), see the note after it.

    what         Scene | dict | path to .json/.yaml/.wxscene | preset name | METAR | plain words
    seconds / minutes / hours / duration   length (summed; default 60 s)
    out          file path (.wav or .flac) or a directory (UCS-named file); None = UCS name in cwd
    layout       stereo|binaural|5.1|7.1.4|foa|hoa3|mono (default: the scene's listener.render)
    destination  raw|youtube|spotify|apple_music|podcast|sleep|broadcast_eu|broadcast_us|
                 streaming_broadcaster|post_stems|game|vr (default: the scene's, else raw)
    seed         overrides the scene seed; stems: also one file per stem; captions: True or formats
    mode         apply a wellness mode (sleep|focus|calm|asmr|none) before rendering
    strict       raise SceneParseError when free text names nothing known
    Other keywords go to raingen.render.Renderer (sample_rate, bit_depth, mix, quality, ...).
    """
    scene, understood = _resolve(what, strict=strict)
    secs = duration_seconds(seconds, minutes, hours, duration)
    r = Renderer(scene, secs, layout=layout, destination=destination, stems=stems or None, progress=progress,
                 seed=seed, mode=mode, **kw)
    report = r.run(out, captions=captions)
    if understood:
        report["understood"] = understood
    return report


# `render` is the raingen.render submodule itself, which is callable (render_file) and carries
# render_file's signature: one object serves as the package attribute `raingen.render`, the
# function `rg.render(...)` and the module of `from raingen.render import Renderer`, so none of
# them can shadow another, in any import order (see raingen.render._CallableModule).
render = sys.modules[__package__ + ".render"]


def render_array(what, seconds: Optional[float] = None, minutes: Optional[float] = None,
                 hours: Optional[float] = None, layout: Optional[str] = None, destination: Optional[str] = None,
                 seed: Optional[int] = None, **kw):
    """Render to memory: (audio (C, n) float64, markers). For short clips, notebooks and tests."""
    scene = load_scene(what)
    return _render_array(scene, duration_seconds(seconds, minutes, hours, default=10.0), layout=layout,
                         destination=destination, seed=seed, **kw)


def play(what, seconds: Optional[float] = None, layout: Optional[str] = None, device=None, **kw) -> None:
    """Play a scene on the sound card (blocks; Ctrl+C stops).

    With the native core installed (pip install 'raingen[native]') the Rust engine streams through
    cpal at the scene's full quality; otherwise (or native=False, a wellness mode, or Engine-only
    options) the Python Engine plays through sounddevice (the 'audio' extra).
    """
    from . import native as _nat
    from .render import resolve_layout
    from .wellness.modes import canonical_mode

    scene = load_scene(what)
    native = kw.pop("native", None)
    native_kw = {"quality", "volume_db", "on_marker", "target_lufs", "ceiling_dbtp", "latency_ms"}
    lay = resolve_layout(scene, layout)
    if (_nat.enabled(native) and _nat.supports_layout(lay) and canonical_mode(scene.mode) == "none"
            and set(kw) <= native_kw):
        sess = _nat.NativeSession(scene, lay, quality=kw.pop("quality", None))
        sess.run(seconds, device=device, **kw)
        return
    eng = Engine(scene, layout=layout, device=device, native=native, **kw)
    eng.run(seconds)


def one_shots(kind: str = "thunder", count: int = 12, out_dir=None, distance_km: float = 3.0,
              seed: Optional[int] = None, sample_rate: int = 48000, bit_depth: int = 24, ucs: bool = True) -> list:
    """Synthesized one-shots (currently 'thunder'); written to out_dir when given, else returned as arrays."""
    if str(kind).lower() != "thunder":
        raise ValueError("one_shots currently supports kind='thunder'")
    clips = thunder_one_shots(count, distance_km=distance_km, sample_rate=sample_rate, seed=seed)
    if out_dir is None:
        return clips
    from .io.ucs import canonical_catid
    from .io.wav import StreamingWavWriter

    os.makedirs(out_dir, exist_ok=True)
    # UCS v8.2.1: CatID_FXName_CreatorID_SourceID_UserData; WEATHER/THUNDER is CatID THUN
    base = f"{canonical_catid('THUN')}_Thunder {distance_km:g} km one-shot_RGN_raingen" if ucs else "thunder"
    paths = []
    for i, clip in enumerate(clips, start=1):
        p = os.path.join(out_dir, f"{base}_{i:02d}.wav")
        with StreamingWavWriter(p, sample_rate, clip.shape[0], bit_depth) as w:
            w.write(clip)
        paths.append(os.path.abspath(p))
    return paths


def beaufort(force: float) -> float:
    """Beaufort force (continuous) -> mean wind at 10 m in m/s (v = 0.836 B^1.5, WMO)."""
    from .physics.wind import beaufort_to_mps
    return float(beaufort_to_mps(float(force)))


def destinations() -> list[dict]:
    """Delivery destinations with their loudness / true-peak targets."""
    from .master.destinations import list_destinations
    return list_destinations()


def manifest(format: str = "json-schema"):
    """The parameter manifest in one of: json-schema, osc, wwise, fmod, markdown, json."""
    fmt = str(format).strip().lower().replace("_", "-")
    if fmt in ("json-schema", "jsonschema", "schema"):
        return to_json_schema()
    if fmt == "osc":
        return osc_namespace()
    if fmt == "wwise":
        return wwise_manifest()
    if fmt == "fmod":
        return fmod_manifest()
    if fmt in ("markdown", "md"):
        return markdown_reference()
    if fmt == "json":
        return _mf.manifest_json()
    raise ValueError("format must be one of json-schema, osc, wwise, fmod, markdown, json")


# ---------------------------------------------------------------------------
# variations
# ---------------------------------------------------------------------------
_VAR_ORDER = ("rate", "shower", "hail", "wind", "from", "gust", "th_dist", "th_bearing", "th_rate", "th_power",
              "temp", "humid", "hour", "heading", "tau", "spare")


def variations(scene, count: int = 10, jitter: float = 1.0) -> list[Scene]:
    """`count` derived scenes: new seeds via RngFactory.child plus small, human-meaningful jitter.

    Each variation keeps the scene's story (precipitation type, place, shelter, mode) and moves
    the continuous weather inside realistic ranges: rain rate about +-20 % (log domain), wind
    +-0.4 Beaufort and +-15 deg, thunder distance about +-25 %, bearing +-25 deg, flash rate,
    temperature +-1.5 C (never across freezing), humidity, hour, listener heading +-20 deg.
    Parameters that are pinned or automated in the scene are left alone. `jitter` scales all
    of it (0 = new seeds only).
    """
    from .scene.units import beaufort_to_mps, mps_to_beaufort

    base = load_scene(scene)
    count = int(count)
    if count < 1:
        return []
    s = max(0.0, float(jitter))
    root = RngFactory(int(base.seed))
    automated = {getattr(kf, "path", "") for kf in base.timeline or []}
    overrides = base.overrides or {}

    def free(path: str) -> bool:
        return path not in automated and (overrides.get(path) or {}).get("state") != "pinned"

    out: list[Scene] = []
    for i in range(count):
        child = root.child(f"variation/{i}")
        z = dict(zip(_VAR_ORDER, child.get("variation.jitter", 0).standard_normal(len(_VAR_ORDER)) * s))
        v = base.copy()
        v.seed = int(child.root_seed) & 0xFFFFFFFFFFFFFFFF
        w = v.weather
        p = w.precipitation
        if p.type != "none" and p.rate_mm_h > 0.0 and free("weather.precipitation.rate_mm_h"):
            p.rate_mm_h = float(min(max(p.rate_mm_h * 2.0 ** (0.27 * z["rate"]), 0.02), 150.0))
        if free("weather.precipitation.shower_period_min"):
            p.shower_period_min = float(min(max(p.shower_period_min * 2.0 ** (0.25 * z["shower"]), 5.0), 30.0))
        if p.type == "hail" and free("weather.precipitation.hail_size_mm"):
            p.hail_size_mm = float(min(max(p.hail_size_mm * 2.0 ** (0.2 * z["hail"]), 6.0), 44.0))
        wind = w.wind
        if free("weather.wind.speed_mps"):
            b = mps_to_beaufort(wind.speed_mps) + 0.4 * z["wind"]
            wind.speed_mps = float(min(max(beaufort_to_mps(max(b, 0.0)), 0.0), 40.0))
        if free("weather.wind.from_deg"):
            wind.from_deg = float((wind.from_deg + 15.0 * z["from"]) % 360.0)
        if wind.gust_factor is not None and free("weather.wind.gust_factor"):
            wind.gust_factor = float(min(max(wind.gust_factor + 0.08 * z["gust"], 1.0), 2.5))
        th = w.thunder
        if th.enabled:
            if free("weather.thunder.distance_km"):
                th.distance_km = float(min(max(th.distance_km * 2.0 ** (0.33 * z["th_dist"]), 0.3), 30.0))
            if free("weather.thunder.bearing_deg"):
                th.bearing_deg = float((th.bearing_deg + 25.0 * z["th_bearing"]) % 360.0)
            if free("weather.thunder.flashes_per_min"):
                th.flashes_per_min = float(min(max(th.flashes_per_min * 2.0 ** (0.3 * z["th_rate"]), 0.05), 20.0))
            if free("weather.thunder.stroke_intensity"):
                th.stroke_intensity = float(min(max(th.stroke_intensity + 0.1 * z["th_power"], 0.0), 1.0))
        if free("weather.temperature_c"):
            t_new = w.temperature_c + 1.5 * z["temp"]
            if (t_new > 0.0) == (w.temperature_c > 0.0):
                w.temperature_c = float(t_new)
        if free("weather.humidity_pct"):
            w.humidity_pct = float(min(max(w.humidity_pct + 3.0 * z["humid"], 20.0), 100.0))
        if free("weather.hour"):
            w.hour = float((w.hour + 0.6 * z["hour"]) % 24.0)
        if free("listener.heading_deg"):
            v.listener.heading_deg = float((v.listener.heading_deg + 20.0 * z["heading"]) % 360.0)
        if free("place.aftermath_tau_min"):
            v.place.aftermath_tau_min = float(min(max(v.place.aftermath_tau_min * 2.0 ** (0.2 * z["tau"]), 1.0), 60.0))
        v.name = f"{base.name} (variation {i + 1})"
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# "What am I hearing?"
# ---------------------------------------------------------------------------
def describe(scene, system: str = "metric") -> str:
    """A plain-words summary of a scene (MASTER_PLAN 2.11)."""
    from .scene import units

    sc = load_scene(scene)
    w = sc.weather
    p = w.precipitation
    lines = [sc.name]
    temp = float(w.temperature_c)
    if p.type == "none" or p.rate_mm_h <= 0.0:
        lines.append("Precipitation: none")
    else:
        kind = "drizzle" if p.type == "drizzle" else "rain"
        word = units.rain_label(p.rate_mm_h, kind=kind)
        what = p.type.replace("_", " ")
        extra = f", hailstones {units.hail_label(p.hail_size_mm)}" if p.type == "hail" else ""
        lines.append(f"Precipitation: {word} {what}, "
                     f"{units.format_value('weather.precipitation.rate_mm_h', p.rate_mm_h, system)}, "
                     f"{p.drops} drops, {p.character}{extra}")
    wind = w.wind
    rel = units.relative_direction(wind.from_deg, sc.listener.heading_deg)
    lines.append(f"Wind: {units.format_value('weather.wind.speed_mps', wind.speed_mps, system)} from "
                 f"{units.compass_label(wind.from_deg)} (from your {rel})")
    th = w.thunder
    if th.enabled:
        lines.append(f"Thunder: {units.format_value('weather.thunder.distance_km', th.distance_km, system, temp)}, "
                     f"bearing {units.compass_label(th.bearing_deg)}, {th.flashes_per_min:g} flashes/min"
                     + (f", approaching {th.approach_kmh:g} km/h" if th.approach_kmh > 0 else
                        f", receding {-th.approach_kmh:g} km/h" if th.approach_kmh < 0 else ""))
    else:
        lines.append("Thunder: none")
    surf = sorted(((k, float(v)) for k, v in (sc.place.surfaces or {}).items() if float(v) > 0),
                  key=lambda kv: -kv[1])[:4]
    surf_txt = ", ".join(f"{k.replace('_', ' ')} {v:.0f}%" for k, v in surf)
    lines.append(f"Place: {sc.place.preset.replace('_', ' ')}, shelter {sc.place.shelter.replace('_', ' ')}"
                 + (f"; surfaces {surf_txt}" if surf_txt else ""))
    lines.append(f"Time: {units.time_of_day_label(w.hour)} ({w.hour:04.1f} h), {w.season}, "
                 f"{units.format_value('weather.temperature_c', temp, system)}, "
                 f"humidity {w.humidity_pct:.0f}%")
    lines.append(f"Story: {w.storm_arc} over {w.arc_minutes:g} min · listener {sc.listener.height_m:g} m, "
                 f"{sc.listener.render} · mode {sc.mode} · seed 0x{int(sc.seed):X}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scene conveniences (attached here so raingen.scene.schema stays plain data)
# ---------------------------------------------------------------------------
def _scene_load(cls, path: str) -> Scene:
    """Load a scene file; a preset name also works (Scene.load('tent-in-a-storm'))."""
    p = os.path.expanduser(os.fspath(path))
    if os.path.isfile(p):
        return _read_scene_file(p)
    key = _preset_key(str(path))
    if key is not None:
        return _preset(key)
    raise FileNotFoundError(f"no scene file or preset named {path!r}")


def _scene_render(self: Scene, seconds: Optional[float] = None, minutes: Optional[float] = None,
                  hours: Optional[float] = None, out=None, out_dir=None, **kw) -> RenderReport:
    return render_file(self, seconds=seconds, minutes=minutes, hours=hours, out=out if out is not None else out_dir,
                       **kw)


class _RenderSlot:
    """Data descriptor for the `Scene.render` field.

    `scene.render` stays the RenderSettings dataclass (scene.render.quality, asdict, JSON all
    unchanged) but remembers which scene it belongs to, and RenderSettings is made callable,
    so `scene.render(minutes=5, out="x.flac")` renders that scene.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            rs = obj.__dict__["render"]
        except KeyError:
            raise AttributeError("render") from None
        try:
            object.__setattr__(rs, "_raingen_scene", weakref.ref(obj))
        except (AttributeError, TypeError):
            pass
        return rs

    def __set__(self, obj, value) -> None:
        obj.__dict__["render"] = value


def _settings_call(self: RenderSettings, *args, **kw) -> RenderReport:
    ref = getattr(self, "_raingen_scene", None)
    scene = ref() if ref is not None else None
    if scene is None:
        raise TypeError("these RenderSettings are not attached to a Scene; use raingen.render(scene, ...)")
    return _scene_render(scene, *args, **kw)


def _scene_variations(self: Scene, count: int = 10, jitter: float = 1.0) -> list[Scene]:
    return variations(self, count, jitter)


def _scene_play(self: Scene, seconds: Optional[float] = None, **kw) -> None:
    play(self, seconds=seconds, **kw)


def _scene_engine(self: Scene, **kw) -> Engine:
    return Engine(self, **kw)


def _scene_describe(self: Scene, system: str = "metric") -> str:
    return describe(self, system)


def _scene_with_vibe(self: Scene, intensity=None, closeness=None, wildness=None) -> Scene:
    return apply_vibe(self, intensity=intensity, closeness=closeness, wildness=wildness)


def _scene_with_mode(self: Scene, mode: str) -> Scene:
    return apply_mode(self, mode)


def _attach_scene_methods() -> None:
    if getattr(Scene, "_raingen_api", False):
        return
    Scene.load = classmethod(_scene_load)
    Scene.preset = classmethod(lambda cls, name: _preset(name))
    Scene.from_text = classmethod(lambda cls, text, base=None: scene_from_text(text, base))
    Scene.from_metar = classmethod(lambda cls, text, base=None: scene_from_metar(text, base))
    RenderSettings.__call__ = _settings_call
    Scene.render = _RenderSlot()
    Scene.variations = _scene_variations
    Scene.play = _scene_play
    Scene.engine = _scene_engine
    Scene.describe = _scene_describe
    Scene.with_vibe = _scene_with_vibe
    Scene.with_mode = _scene_with_mode
    Scene._raingen_api = True


_attach_scene_methods()
