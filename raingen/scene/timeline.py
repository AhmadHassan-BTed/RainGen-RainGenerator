"""Timeline / automation evaluation (MASTER_PLAN 2.6 Integration, 9.4 time-addressable).

`evaluate(scene, t)` returns the scene as driven by its keyframes at time t.
Each keyframe's curve describes how the value *travels to* that keyframe from
the previous point (the scene's own value at t = 0 for the first keyframe):

* linear - straight line in plain units;
* log    - geometric interpolation (exponential ramp; constant ratio per second),
           the natural curve for rain rate and distance;
* smooth - smoothstep (C1 ease-in/out), for gentle macro moves;
* hold   - step: the previous value holds until the keyframe time, then jumps.

Non-numeric paths (choices, booleans) always hold. After the last keyframe the
value holds. Numeric values are clamped to the manifest range (circular ones
wrapped); `validate(scene)` lists keyframes with unknown paths, curves, times
or values that do not fit their parameter. Circular quantities (wind direction, hour of day) move the short
way round (350 deg -> 10 deg passes through north). Keyframe times may be seconds or strings ('05:00', '01:02:03',
'01:00:00:12' SMPTE at 24 fps). Values may carry units ('8 mm/h', 'force 5').
Evaluation is pure: the same (scene, t) always gives the same result, so any
segment can be rendered on its own.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from . import units
from .manifest import clamp, coerce, find_param
from .schema import Keyframe, Scene

CURVES = ("linear", "log", "hold", "smooth")
_CURVE_ALIASES = {"lin": "linear", "exp": "log", "exponential": "log", "geometric": "log", "step": "hold",
                  "bezier": "smooth", "ease": "smooth", "s": "smooth", "cosine": "smooth"}


def _curve(name: str) -> str:
    c = (name or "linear").strip().lower()
    c = _CURVE_ALIASES.get(c, c)
    return c if c in CURVES else "linear"


def keyframe_time(kf: Any) -> float:
    at = kf.at if hasattr(kf, "at") else kf["at"]
    return units.parse_duration_s(at) if isinstance(at, str) else float(at)


def _kf_fields(kf: Any) -> tuple[float, str, Any, str]:
    if isinstance(kf, dict):
        return keyframe_time(kf), kf["path"], kf.get("value"), _curve(kf.get("curve", "linear"))
    return keyframe_time(kf), kf.path, kf.value, _curve(kf.curve)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_path(path: str, sample: Any) -> bool:
    spec = find_param(path)
    if spec is not None:
        return spec.is_numeric
    return _is_number(sample)


def _coerce(path: str, value: Any, temp_c: float) -> Any:
    if isinstance(value, str):
        spec = find_param(path)
        if spec is not None and not spec.is_numeric:
            try:
                return coerce(spec, value, temp_c)         # 'on' -> True, 'tin roof' -> 'attic'
            except ValueError:
                return value
        try:
            return units.parse_value(path, value, temp_c)
        except ValueError:
            return value
    if isinstance(value, list) and len(value) == 3 and all(_is_number(v) for v in value):
        return tuple(float(v) for v in value)
    return value


def interpolate(a: Any, b: Any, u: float, curve: str) -> Any:
    """Value between a (u=0) and b (u=1) along `curve`."""
    curve = _curve(curve)
    u = min(1.0, max(0.0, float(u)))
    if curve == "hold":
        return b if u >= 1.0 else a
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(interpolate(x, y, u, "smooth" if curve == "smooth" else "linear") for x, y in zip(a, b, strict=True))
    if not (_is_number(a) and _is_number(b)):
        return b if u >= 1.0 else a
    a = float(a)
    b = float(b)
    if curve == "log":
        if a > 0.0 and b > 0.0:
            return math.exp(math.log(a) + (math.log(b) - math.log(a)) * u)
        return a + (b - a) * u
    if curve == "smooth":
        u = u * u * (3.0 - 2.0 * u)
    return a + (b - a) * u


def _get(scene: Scene, path: str) -> Any:
    try:
        return scene.get(path)
    except (KeyError, AttributeError, TypeError):
        spec = find_param(path)
        return spec.default if spec is not None else None


def _set(scene: Scene, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = scene
    for part in parts[:-1]:
        obj = obj.setdefault(part, {}) if isinstance(obj, dict) else getattr(obj, part)
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    else:
        setattr(obj, parts[-1], value)


def lanes(scene: Scene) -> dict[str, list[tuple[float, Any, str]]]:
    """Keyframes grouped by path and sorted by time: path -> [(t, value, curve), ...]."""
    temp = float(scene.weather.temperature_c)
    out: dict[str, list[tuple[float, Any, str]]] = {}
    for idx, kf in enumerate(scene.timeline or []):
        t, path, value, curve = _kf_fields(kf)
        out.setdefault(path, []).append((t, idx, _coerce(path, value, temp), curve))
    return {p: [(t, v, c) for t, _, v, c in sorted(kfs, key=lambda e: (e[0], e[1]))] for p, kfs in out.items()}


def value_at(base_value: Any, lane: list[tuple[float, Any, str]], t: float, numeric: bool) -> Any:
    """Evaluate one automation lane at time t."""
    if not lane:
        return base_value
    t = float(t)
    t0, v0, c0 = lane[0]
    if t < t0:
        if not numeric or c0 == "hold" or base_value is None or t0 <= 0.0 or t < 0.0:
            return base_value
        return interpolate(base_value, v0, t / t0, c0)
    prev_t, prev_v = t0, v0
    for tk, vk, ck in lane[1:]:
        if t < tk:
            if not numeric or ck == "hold" or tk <= prev_t:
                return prev_v
            return interpolate(prev_v, vk, (t - prev_t) / (tk - prev_t), ck)
        prev_t, prev_v = tk, vk
    return prev_v


def _unwrap_lane(base: Any, lane: list[tuple[float, Any, str]], period: float) -> list[tuple[float, Any, str]]:
    """Make each step of a circular lane (degrees, hour of day) take the short way round."""
    out = []
    prev = float(base) if _is_number(base) else None
    for tk, vk, ck in lane:
        if _is_number(vk):
            v = float(vk)
            if prev is not None:
                v = prev + ((v - prev + 0.5 * period) % period - 0.5 * period)
            prev = v
            vk = v
        out.append((tk, vk, ck))
    return out


def evaluate(scene: Scene, t: float) -> Scene:
    """The scene with every automation lane evaluated at time t (seconds). Never mutates `scene`."""
    out = scene.copy()
    if not scene.timeline:
        return out
    for path, lane in lanes(scene).items():
        base = _get(scene, path)
        numeric = _numeric_path(path, lane[0][1]) and all(_is_number(v) or isinstance(v, tuple) for _, v, _ in lane)
        spec = find_param(path)
        period = 0.0
        if numeric and spec is not None and spec.wrap and spec.min is not None and spec.max is not None:
            period = float(spec.max - spec.min)
            lane = _unwrap_lane(base, lane, period)
        value = value_at(base, lane, t, numeric)
        if spec is not None and spec.is_numeric and _is_number(value):
            value = clamp(spec, value)                     # range, wrap and int rounding from the manifest
        try:
            _set(out, path, value)
        except (AttributeError, KeyError, TypeError):
            continue
    return out


def validate(scene: Scene, strict: bool = False) -> list[str]:
    """Problems with the scene's keyframes, one message each: an unknown path, an unknown curve
    name, a time that is not a duration or is negative, or a value that does not fit the
    parameter (not a number, not one of its choices, outside its range). strict=True raises
    ValueError listing them. An empty list means every lane evaluates as written."""
    problems: list[str] = []
    temp = float(scene.weather.temperature_c)
    for i, kf in enumerate(scene.timeline or []):
        where = f"keyframe {i}"
        try:
            raw_path = kf["path"] if isinstance(kf, dict) else kf.path
            raw_curve = (kf.get("curve", "linear") if isinstance(kf, dict) else kf.curve) or "linear"
            raw_value = kf.get("value") if isinstance(kf, dict) else kf.value
        except (KeyError, AttributeError) as e:
            problems.append(f"{where}: missing field {e}")
            continue
        where = f"keyframe {i} ({raw_path})"
        try:
            t = keyframe_time(kf)
            if not math.isfinite(t) or t < 0.0:
                problems.append(f"{where}: time {t!r} must be >= 0 s")
        except (ValueError, TypeError) as e:
            problems.append(f"{where}: bad time: {e}")
        c = str(raw_curve).strip().lower()
        if c not in CURVES and c not in _CURVE_ALIASES:
            problems.append(f"{where}: unknown curve {raw_curve!r}; use one of {CURVES}")
        spec = find_param(raw_path)
        if spec is None:
            try:
                scene.get(raw_path)
            except (KeyError, AttributeError, TypeError):
                problems.append(f"{where}: unknown parameter path {raw_path!r}")
            continue
        value = _coerce(raw_path, raw_value, temp)
        if value is None:
            if not spec.nullable:
                problems.append(f"{where}: value is missing")
        elif spec.type == "vec3":
            if not (isinstance(value, tuple) and len(value) == 3):
                problems.append(f"{where}: needs three numbers (x, y, z), got {raw_value!r}")
        elif spec.is_numeric:
            if not _is_number(value):
                problems.append(f"{where}: {raw_value!r} is not a value in {spec.unit or 'plain units'}")
            elif not spec.wrap and spec.min is not None and spec.max is not None:
                v = float(value)
                if (v < spec.min and not (spec.zero_ok and v == 0.0)) or v > spec.max:
                    problems.append(f"{where}: {v:g} is outside {spec.min:g}..{spec.max:g} {spec.unit} "
                                    f"(it will be clamped)")
        elif spec.choices and value not in spec.choices:
            problems.append(f"{where}: {raw_value!r} is not one of {list(spec.choices)}")
    if strict and problems:
        raise ValueError("invalid timeline:\n  " + "\n  ".join(problems))
    return problems


def automated_paths(scene: Scene) -> set[str]:
    return {(_kf_fields(kf)[1]) for kf in (scene.timeline or [])}


def ramp(path: str, points: list[tuple[Any, Any]], curve: str = "linear") -> list[Keyframe]:
    """Keyframes for a ramp through (time, value) points, e.g.
    ramp('weather.precipitation.rate_mm_h', [(0, 0.5), ('05:00', 25), ('12:00', 2)], curve='log')."""
    out = []
    for at, value in points:
        tsec = units.parse_duration_s(at) if isinstance(at, str) else float(at)
        out.append(Keyframe(at=tsec, path=path, value=value, curve=_curve(curve)))
    return out


_ARC_SPLIT = (("arriving", 0.30), ("steady", 0.30), ("passing", 0.20), ("clearing", 0.20))


def storm_arc_keyframes(arc: str, minutes: float, start_s: float = 0.0,
                        thunder_km: Optional[tuple[float, float]] = None) -> list[Keyframe]:
    """Keyframes that walk the storm arc (MASTER_PLAN 2.5, 7.9).

    arc: 'arriving' (then steady), 'steady', 'passing' (then clearing), 'clearing',
    or 'full' / 'storm' / 'lifecycle' (arriving 30 %, steady 30 %, passing 20 %,
    clearing 20 % of `minutes`). Each phase sets weather.storm_arc and
    weather.arc_minutes (hold) at its start; the StormSystem shapes the envelope
    inside the phase. With `thunder_km=(far, near)` the thunder distance follows a
    log ramp in from `far` to `near` while arriving and back out while passing.
    """
    arc = arc.strip().lower()
    minutes = max(0.5, float(minutes))
    start = float(start_s)
    if arc in ("full", "storm", "lifecycle", "pass_through", "pass-through", "whole"):
        phases = [(name, frac * minutes) for name, frac in _ARC_SPLIT]
    elif arc == "arriving":
        phases = [("arriving", minutes), ("steady", minutes)]
    elif arc == "passing":
        phases = [("passing", minutes), ("clearing", minutes)]
    elif arc in ("steady", "clearing"):
        phases = [(arc, minutes)]
    else:
        raise ValueError(f"unknown storm arc {arc!r}")
    kfs: list[Keyframe] = []
    t = start
    bounds: dict[str, tuple[float, float]] = {}
    for name, dur in phases:
        kfs.append(Keyframe(at=t, path="weather.storm_arc", value=name, curve="hold"))
        kfs.append(Keyframe(at=t, path="weather.arc_minutes", value=float(dur), curve="hold"))
        bounds[name] = (t, t + dur * 60.0)
        t += dur * 60.0
    if thunder_km is not None:
        far, near = float(thunder_km[0]), float(thunder_km[1])
        kfs.append(Keyframe(at=start, path="weather.thunder.enabled", value=True, curve="hold"))
        # distance is keyframed explicitly, so the storm's own drift is disabled
        kfs.append(Keyframe(at=start, path="weather.thunder.approach_kmh", value=0.0, curve="hold"))
        if "arriving" in bounds:
            a0, a1 = bounds["arriving"]
            kfs.append(Keyframe(at=a0, path="weather.thunder.distance_km", value=far, curve="hold"))
            kfs.append(Keyframe(at=a1, path="weather.thunder.distance_km", value=near, curve="log"))
        else:
            kfs.append(Keyframe(at=start, path="weather.thunder.distance_km", value=near, curve="hold"))
        if "passing" in bounds:
            p0, p1 = bounds["passing"]
            kfs.append(Keyframe(at=p0, path="weather.thunder.distance_km", value=near, curve="hold"))
            kfs.append(Keyframe(at=p1, path="weather.thunder.distance_km", value=far, curve="log"))
        if "clearing" in bounds:
            c0, c1 = bounds["clearing"]
            kfs.append(Keyframe(at=c0 + 0.5 * (c1 - c0), path="weather.thunder.enabled", value=False, curve="hold"))
    kfs.sort(key=lambda k: k.at)
    return kfs


def with_storm_arc(scene: Scene, arc: str, minutes: float, thunder_km: Optional[tuple[float, float]] = None) -> Scene:
    """Copy of `scene` with storm-arc keyframes appended to its timeline."""
    out = scene.copy()
    out.timeline = list(out.timeline) + storm_arc_keyframes(arc, minutes, 0.0, thunder_km)
    return out
