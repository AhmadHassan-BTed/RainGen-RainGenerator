"""Tier-0 Vibe macros: Intensity, Closeness/Shelter, Calm<->Wild (MASTER_PLAN 2.3).

Intensity walks the storm path through fixed anchors, interpolated in
log(rain rate), Beaufort force and log(thunder distance). The rain anchors are
the "How much rain" detents of MASTER_PLAN 2.3 (WMO words: slight < 2.5,
moderate 2.5-10, heavy 10-50, intense >= 50, cloudburst >= 100 mm/h):

    0 dry, drips only | 20 drizzle 0.3 mm/h, Bft 2 | 40 moderate 5 mm/h, Bft 3
    60 heavy 20 mm/h, Bft 5 | 80 thunderstorm, very heavy 50 mm/h, Bft 7, thunder 5 km
    100 cloudburst 100 mm/h, Bft 9, thunder 1.5 km

Closeness: 0 out in it | 25 under umbrella or tree | 50 porch or tent (or the
scene's own mid shelter: car, attic, cave, underpass) | 75 indoors, window open
| 100 indoors, window shut.

Wildness: gust factor 1.15-1.9 (G = 1 + 3 I, so turbulence intensity follows),
gust rate, shower-cell contrast and thunder rate.

Macros never mutate their input; the scene's pre-macro state is remembered in
`scene.vibe["_base"]` so repeated application is idempotent.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .schema import Scene
from .units import beaufort_to_mps, mps_to_beaufort

VIBE_DEFAULTS = {"intensity": 40.0, "closeness": 50.0, "wildness": 30.0}

# Intensity anchors
_I_RATE_X = np.array([5.0, 20.0, 40.0, 60.0, 80.0, 100.0])
_I_RATE_Y = np.log(np.array([0.02, 0.3, 5.0, 20.0, 50.0, 100.0]))
_I_BFT_X = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
_I_BFT_Y = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
DRY_BELOW = 5.0

_RAINLIKE = ("none", "drizzle", "rain", "showers")
_FROZEN = ("snow", "sleet", "graupel", "hail", "freezing_rain")
_MID_SHELTERS = ("porch", "tent", "car", "attic", "cave", "underpass")
_OUTDOOR_PLACES = ("deciduous_forest", "conifer_forest", "meadow", "mountain_ridge", "desert", "jungle", "lake", "coast")
_TREE_PLACES = ("deciduous_forest", "conifer_forest", "jungle", "cabin", "garden", "lake")


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def turbulence_from_gust(g: float) -> float:
    """Turbulence intensity from the gust factor, G = 1 + 3 I (Appendix A), within the manifest's 0.05 .. 0.5."""
    return float(min(0.5, max(0.05, round((float(g) - 1.0) / 3.0, 9))))


def intensity_rate(intensity: float) -> float:
    """Rain rate (mm/h) on the storm path; 0 below the dry threshold."""
    i = float(min(100.0, max(0.0, intensity)))
    if i < DRY_BELOW:
        return 0.0
    return float(math.exp(np.interp(i, _I_RATE_X, _I_RATE_Y)))


def intensity_beaufort(intensity: float) -> float:
    i = float(min(100.0, max(0.0, intensity)))
    return float(np.interp(i, _I_BFT_X, _I_BFT_Y))


def intensity_thunder(intensity: float, start: float = 65.0) -> Optional[tuple[float, float, float]]:
    """(distance_km, flashes_per_min, stroke_intensity) or None when below the thunder threshold."""
    i = float(min(100.0, max(0.0, intensity)))
    if i < start:
        return None
    xs = [start, 80.0, 100.0] if start < 80.0 else [start, 100.0]
    d_ys = [math.log(20.0), math.log(5.0), math.log(1.5)] if start < 80.0 else [math.log(5.0), math.log(1.5)]
    f_ys = [0.3, 2.0, 6.0] if start < 80.0 else [2.0, 6.0]
    dist = math.exp(float(np.interp(i, xs, d_ys)))
    flashes = float(np.interp(i, xs, f_ys))
    stroke = _lerp(0.3, 0.9, _clip01((i - start) / max(1e-6, 100.0 - start)))
    return dist, flashes, stroke


def intensity_from_scene(scene: Scene) -> float:
    """Where a scene sits on the storm path (inverse of the rate anchors), for UI sync."""
    p = scene.weather.precipitation
    if p.type == "none" or p.rate_mm_h <= 0.0:
        return 0.0
    lr = math.log(max(0.02, min(100.0, float(p.rate_mm_h))))
    return float(np.interp(lr, _I_RATE_Y, _I_RATE_X))


def _snapshot(scene: Scene) -> dict:
    w = scene.weather
    return {
        "precip_type": w.precipitation.type,
        "rate_mm_h": float(w.precipitation.rate_mm_h),
        "character": w.precipitation.character,
        "shower_period_min": float(w.precipitation.shower_period_min),
        "speed_mps": float(w.wind.speed_mps),
        "gust_factor": w.wind.gust_factor,
        "gust_rate_per_min": w.wind.gust_rate_per_min,
        "turbulence_intensity": w.wind.turbulence_intensity,
        "thunder_enabled": bool(w.thunder.enabled),
        "distance_km": float(w.thunder.distance_km),
        "flashes_per_min": float(w.thunder.flashes_per_min),
        "stroke_intensity": float(w.thunder.stroke_intensity),
        "shelter": scene.place.shelter,
        "window_open": float(scene.place.window_open),
    }


def _shelter_for(closeness: float, scene: Scene, base: dict) -> tuple[str, float]:
    c = float(min(100.0, max(0.0, closeness)))
    place = scene.place
    if c < 12.5:
        return "open", 0.0
    if c < 37.5:
        trees = int(place.vegetation.get("deciduous", 0) or 0) + int(place.vegetation.get("conifer", 0) or 0)
        if base["shelter"] in ("tree", "umbrella"):
            return base["shelter"], 0.0
        if trees >= 3 and place.preset in _TREE_PLACES:
            return "tree", 0.0
        return "umbrella", 0.0
    if c < 62.5:
        if base["shelter"] in _MID_SHELTERS:
            return base["shelter"], 0.0
        if place.preset in _OUTDOOR_PLACES:
            return "tent", 0.0
        return "porch", 0.0
    window = _clip01((97.5 - c) / 22.5)
    return ("room_open" if window > 0.02 else "room_closed"), window


def apply_vibe(scene: Scene, intensity: Optional[float] = None, closeness: Optional[float] = None,
               wildness: Optional[float] = None) -> Scene:
    """Return a copy of `scene` driven by the Tier-0 macros (0..100 each).

    Arguments left as None fall back to `scene.vibe` values; macros absent from both
    leave their parameters untouched.
    """
    out = scene.copy()
    vibe = dict(out.vibe or {})
    base = vibe.get("_base")
    if not isinstance(base, dict):
        base = _snapshot(scene)
    vals = {
        "intensity": intensity if intensity is not None else vibe.get("intensity"),
        "closeness": closeness if closeness is not None else vibe.get("closeness"),
        "wildness": wildness if wildness is not None else vibe.get("wildness"),
    }
    w = out.weather
    pr = w.precipitation
    wind = w.wind
    th = w.thunder

    # ---- closeness -------------------------------------------------------------
    if vals["closeness"] is not None:
        shelter, window = _shelter_for(float(vals["closeness"]), out, base)
        out.place.shelter = shelter
        out.place.window_open = window if shelter.startswith("room") else float(base["window_open"])

    # ---- intensity -------------------------------------------------------------
    flashes_base = float(base["flashes_per_min"])
    if vals["intensity"] is not None:
        i = float(min(100.0, max(0.0, vals["intensity"])))
        rate = intensity_rate(i)
        btype = base["precip_type"]
        if rate <= 0.0:
            pr.type = "none"
            pr.rate_mm_h = 0.0
        elif btype in _FROZEN:
            pr.type = btype
            pr.rate_mm_h = rate if btype in ("hail", "freezing_rain") else rate * 0.5   # snow/sleet/graupel as water equivalent
        else:
            if rate < 0.5:
                pr.type = "drizzle"
            elif btype == "showers":
                pr.type = "showers"
            else:
                pr.type = "rain"
            pr.rate_mm_h = rate
        wind.speed_mps = beaufort_to_mps(intensity_beaufort(i))
        start = 45.0 if base["thunder_enabled"] else 65.0
        tv = intensity_thunder(i, start)
        if tv is None:
            th.enabled = False
            flashes_base = float(base["flashes_per_min"])
        else:
            th.enabled = True
            th.distance_km, flashes_base, th.stroke_intensity = tv
            th.flashes_per_min = flashes_base

    # ---- wildness --------------------------------------------------------------
    if vals["wildness"] is not None:
        u = _clip01(float(vals["wildness"]) / 100.0)
        g = 1.15 + 0.75 * u
        wind.gust_factor = g
        wind.turbulence_intensity = turbulence_from_gust(g)
        wind.gust_rate_per_min = 0.5 + 3.5 * u
        if pr.type in ("rain", "showers", "drizzle", "snow", "sleet", "hail", "graupel"):
            if u >= 0.6 and base["character"] == "continuous" and pr.type != "drizzle":
                pr.character = "showery"
            else:
                pr.character = base["character"]
            if pr.character == "showery":
                pr.shower_period_min = _lerp(25.0, 6.0, u)
            else:
                pr.shower_period_min = float(base["shower_period_min"])
        if th.enabled:
            th.flashes_per_min = float(min(20.0, flashes_base * (0.5 + 1.5 * u)))
    elif vals["intensity"] is not None and th.enabled:
        th.flashes_per_min = flashes_base

    vibe.update({k: float(v) for k, v in vals.items() if v is not None})
    vibe["_base"] = base
    out.vibe = vibe
    return out


def vibe_from_scene(scene: Scene) -> dict:
    """Best estimate of the macro positions that describe a scene (for initialising sliders)."""
    shelter = scene.place.shelter
    closeness = {"open": 0.0, "tree": 25.0, "umbrella": 25.0, "porch": 50.0, "tent": 50.0, "car": 50.0,
                 "attic": 50.0, "cave": 50.0, "underpass": 50.0, "room_open": 75.0, "room_closed": 100.0}.get(shelter, 50.0)
    if shelter == "room_open":
        closeness = 97.5 - 22.5 * _clip01(scene.place.window_open)
    g = scene.weather.wind.gust_factor
    wild = 30.0 if g is None else _clip01((float(g) - 1.15) / 0.75) * 100.0
    return {"intensity": intensity_from_scene(scene), "closeness": closeness, "wildness": wild,
            "beaufort": mps_to_beaufort(scene.weather.wind.speed_mps)}
