"""Storm lifecycle: arc envelopes, shower cells, Cox modulation, thunder cells, flashes.

Physics implemented here:

* Rain is a doubly stochastic (Cox) process: Poisson below ~0.1 s, with the rate
  modulated by multiplicative scale-invariant noise above it (Lovejoy & Schertzer
  2008). The modulation is a sum of 13 octaves of value noise from 0.1 s to ~7 min
  with equal variance per octave (a 1/f spectrum), applied as a mean-one log-normal
  multiplier. Every knot is drawn from RngFactory by (octave, knot index), so the
  modulation is a pure function of time (seekable, block-size independent).
* Storm arc (arriving / steady / passing / clearing over `arc_minutes`): sparse big
  drops -> downpour -> stratiform tail -> drips. Size sorting: large drops fall faster
  and reach the ground first at a cell's leading edge, so while a storm (or a shower
  cell) arrives Dm is raised and the count lowered at constant rate; in the
  stratiform tail Dm drops below the regime value.
* Showery rain: convective cells every `shower_period_min` with a fast rise, slower
  decay and log-normal peak strength; intermittent rain: smooth on/off gating.
* Thunder cells move: `approach_kmh` advances the cell along its bearing with a 0.3 km
  lateral miss, so a cell that passes overhead sweeps across the sky and recedes on
  the other side. Flashes are Poisson at `flashes_per_min` (scaled by the lifecycle),
  drawn per block slot so any query interval gives the same flashes.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import replace

import numpy as np

from ..core.constants import BLOCK, SAMPLE_RATE
from ..core.types import StormCellState

# u = progress through the arc (0..1); values beyond 1 hold the last knot
_ARC_CURVES = {
    "steady": {
        "u": [0.0, 1.0],
        "env": [1.0, 1.0], "dm": [1.0, 1.0], "thunder": [1.0, 1.0], "dist": [1.0, 1.0],
    },
    "arriving": {
        "u": [0.0, 0.1, 0.25, 0.45, 0.7, 0.9, 1.0],
        "env": [0.02, 0.03, 0.08, 0.30, 0.70, 0.95, 1.0],
        "dm": [1.45, 1.45, 1.35, 1.20, 1.08, 1.02, 1.0],
        "thunder": [0.30, 0.35, 0.45, 0.60, 0.80, 0.95, 1.0],
        "dist": [1.0, 0.95, 0.85, 0.65, 0.45, 0.30, 0.25],
    },
    "passing": {
        "u": [0.0, 0.08, 0.2, 0.3, 0.4, 0.55, 0.75, 0.9, 1.0],
        "env": [0.02, 0.06, 0.60, 1.50, 0.90, 0.50, 0.25, 0.08, 0.0],
        "dm": [1.45, 1.40, 1.15, 0.97, 0.92, 0.88, 0.85, 0.85, 0.85],
        "thunder": [0.30, 0.50, 0.90, 1.00, 0.70, 0.35, 0.15, 0.05, 0.0],
        "dist": [1.0, 0.90, 0.50, 0.12, 0.30, 0.55, 0.80, 1.00, 1.20],
    },
    "clearing": {
        "u": [0.0, 0.3, 0.6, 1.0],
        "env": [1.0, 0.55, 0.20, 0.0],
        "dm": [1.0, 0.92, 0.86, 0.85],
        "thunder": [0.60, 0.30, 0.10, 0.0],
        "dist": [1.0, 1.5, 2.2, 3.0],
    },
}
_ARC_KEYS = ("env", "dm", "thunder", "dist")
_ARC_BLEND_S = 20.0

_COX_OCTAVES = 13
_COX_BASE_S = 0.1
_COX_SIGMA = {"continuous": 0.22, "intermittent": 0.30, "showery": 0.35}

_LATERAL_MISS_KM = 0.3
_INTERMITTENT_KNOT_S = 40.0


def _ease_interp(u: float, xs: list, ys: list) -> float:
    if u <= xs[0]:
        return float(ys[0])
    if u >= xs[-1]:
        return float(ys[-1])
    i = bisect_right(xs, u) - 1
    t = (u - xs[i]) / (xs[i + 1] - xs[i])
    e = 0.5 - 0.5 * math.cos(math.pi * t)
    return float(ys[i] + (ys[i + 1] - ys[i]) * e)


def _smoothstep(x: float, lo: float, hi: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def arc_values(arc: str, u: float) -> dict:
    """Lifecycle values at arc progress `u`: env (rain), dm (size sorting), thunder, dist."""
    curve = _ARC_CURVES.get(str(arc).lower(), _ARC_CURVES["steady"])
    return {k: _ease_interp(u, curve["u"], curve[k]) for k in _ARC_KEYS}


def arc_phase_name(arc: str, u: float) -> str:
    arc = str(arc).lower()
    if arc == "arriving":
        return "arriving" if u < 1.0 else "steady"
    if arc == "passing":
        if u < 0.2:
            return "arriving"
        if u < 0.4:
            return "downpour"
        if u < 0.75:
            return "tail"
        return "clearing" if u < 1.0 else "after"
    if arc == "clearing":
        return "clearing" if u < 1.0 else "after"
    return "steady"


class StormSystem:
    """Slow weather dynamics shared by every generator (one per render)."""

    def __init__(self, scene, rngf, sr: int = SAMPLE_RATE, block: int = BLOCK):
        self.scene = scene
        self.rngf = rngf
        self.sr = int(sr)
        self.block = int(block)
        self.block_s = self.block / float(self.sr)
        self.cells: list[StormCellState] = []
        self.envelope = 1.0
        self.rain_multiplier = 1.0
        self.dm_factor = 1.0
        self.thunder_envelope = 1.0
        self.arc_phase = "steady"
        self.arc_progress = 0.0
        self.shower_activity = 1.0
        self.cox = 1.0
        self._arc_name = None
        self._arc_t0 = 0.0
        self._arc_offsets = None
        self._arc_vals = arc_values("steady", 0.0)
        self._knots: list[dict] = [dict() for _ in range(_COX_OCTAVES)]
        self._shower_cache: dict = {}
        self._inter_cache: dict = {}
        self._track_s = None
        self._track_s0 = 1.0
        self._obj_pos: dict = {}
        self._last_t = None

    # ---------------------------------------------------------------- arc
    def _arc(self, t: float, weather) -> dict:
        arc = str(getattr(weather, "storm_arc", "steady") or "steady").lower()
        minutes = max(float(getattr(weather, "arc_minutes", 20.0) or 20.0), 0.5)
        if arc != self._arc_name:
            if self._arc_name is not None:
                start = arc_values(arc, 0.0)
                self._arc_offsets = {k: self._arc_vals[k] - start[k] for k in _ARC_KEYS}
            self._arc_name = arc
            self._arc_t0 = t
        el = max(t - self._arc_t0, 0.0)
        u = el / (minutes * 60.0)
        vals = arc_values(arc, u)
        if self._arc_offsets is not None:
            w = math.exp(-el / _ARC_BLEND_S)
            if w < 1e-4:
                self._arc_offsets = None
            else:
                vals = {k: vals[k] + self._arc_offsets[k] * w for k in _ARC_KEYS}
        vals["flip"] = 180.0 * _smoothstep(u, 0.22, 0.40) if arc == "passing" else 0.0
        self.arc_phase = arc_phase_name(arc, u)
        self.arc_progress = u
        self._arc_vals = vals
        return vals

    # ---------------------------------------------------------------- showers
    def _shower_cell(self, k: int, period_s: float) -> tuple[float, float, float]:
        key = (k, round(period_s, 6))
        cell = self._shower_cache.get(key)
        if cell is None:
            rng = self.rngf.get("storm.shower", k)
            dur = period_s * float(rng.uniform(0.35, 0.65))
            start = k * period_s + float(rng.uniform(0.0, 0.45)) * period_s
            peak = math.exp(float(rng.normal(math.log(1.2), 0.35)))
            if k == 0:
                start = -0.2 * dur        # renders open inside a shower, not in a dry gap
            cell = (start, dur, peak)
            if len(self._shower_cache) > 16:
                self._shower_cache.clear()
            self._shower_cache[key] = cell
        return cell

    def _showers(self, t: float, period_s: float) -> tuple[float, float]:
        """(activity multiplier, Dm factor) from overlapping shower cells."""
        k = int(math.floor(t / period_s))
        total = 0.0
        dm_acc = 0.0
        for kk in (k - 1, k, k + 1):
            start, dur, peak = self._shower_cell(kk, period_s)
            x = (t - start) / dur
            if 0.0 <= x <= 1.0:
                shape = math.sin(math.pi * x ** 0.7) ** 2
                a = peak * shape
                if x < 0.37:
                    dmk = 1.0 + 0.35 * (1.0 - x / 0.37)      # fat drops at the leading edge
                else:
                    dmk = 1.0 - 0.08 * (x - 0.37) / 0.63
                total += a
                dm_acc += a * dmk
        dm = dm_acc / total if total > 1e-9 else 1.0
        return total, dm

    def _intermittent(self, t: float) -> float:
        x = t / _INTERMITTENT_KNOT_S
        k = int(math.floor(x))

        def knot(i: int) -> float:
            if i <= 1:
                return 1.0
            v = self._inter_cache.get(i)
            if v is None:
                v = float(self.rngf.get("storm.intermittent", i).random())
                if len(self._inter_cache) > 16:
                    self._inter_cache.clear()
                self._inter_cache[i] = v
            return v

        f = x - k
        e = 0.5 - 0.5 * math.cos(math.pi * f)
        v = knot(k) + (knot(k + 1) - knot(k)) * e
        on = _smoothstep(v, 0.30, 0.50)
        return 0.03 + 0.97 * on

    # ---------------------------------------------------------------- Cox modulation
    def _knot(self, octave: int, k: int) -> float:
        cache = self._knots[octave]
        v = cache.get(k)
        if v is None:
            v = float(self.rngf.get(f"storm.cox.{octave}", k).standard_normal())
            if len(cache) > 8:
                for old in [j for j in cache if j < k - 2]:
                    del cache[old]
            cache[k] = v
        return v

    def cox_modulation(self, t: float, sigma: float) -> float:
        """Mean-one multiplicative 1/f modulation at time t (0.1 s .. ~7 min)."""
        if sigma <= 0.0:
            return 1.0
        g = 0.0
        for o in range(_COX_OCTAVES):
            period = _COX_BASE_S * (2.0 ** o)
            x = t / period
            k = int(math.floor(x))
            f = x - k
            e = 0.5 - 0.5 * math.cos(math.pi * f)
            a = self._knot(o, k)
            b = self._knot(o, k + 1)
            g += a + (b - a) * e
        g *= sigma / math.sqrt(_COX_OCTAVES)
        # cosine-interpolated knots carry ~3/4 of the knot variance on average
        return math.exp(g - 0.375 * sigma * sigma)

    # ---------------------------------------------------------------- thunder cells
    def _main_cell(self, t: float, dt: float, weather, vals: dict, shower_act: float) -> StormCellState:
        th = weather.thunder
        b = _LATERAL_MISS_KM
        d_scene = max(float(th.distance_km), 0.1)
        approach = float(getattr(th, "approach_kmh", 0.0) or 0.0)
        if self._track_s is None:
            self._track_s = math.sqrt(max(d_scene * d_scene - b * b, 0.0))
            self._track_s0 = max(abs(self._track_s), 1e-3)
        if approach != 0.0:
            self._track_s -= approach * dt / 3600.0
        else:
            d_target = max(0.3, d_scene * vals["dist"])
            s_abs = math.sqrt(max(d_target * d_target - b * b, 0.0))
            self._track_s = math.copysign(s_abs, self._track_s if self._track_s != 0.0 else 1.0)
        s = self._track_s
        distance = max(math.hypot(s, b), 0.3)
        offset = math.degrees(math.atan2(b, s)) - math.degrees(math.atan2(b, self._track_s0))
        bearing = (float(th.bearing_deg) + offset + vals.get("flip", 0.0)) % 360.0
        rate = max(float(th.flashes_per_min), 0.0) * max(vals["thunder"], 0.0)
        if shower_act is not None:
            rate *= 0.3 + 0.7 * min(shower_act, 1.0)
        return StormCellState(
            id="cell0",
            distance_km=float(distance),
            bearing_deg=float(bearing),
            flash_rate_per_min=float(rate),
            stroke_intensity=float(min(max(th.stroke_intensity, 0.0), 1.0)),
            cg_fraction=float(min(max(th.cg_fraction, 0.0), 1.0)),
            rain_contribution=float(min(max(math.exp(-distance / 6.0) * max(vals["env"], 0.0), 0.0), 1.0)),
        )

    def _object_cell(self, obj, t: float, dt: float, scene) -> StormCellState:
        th = scene.weather.thunder
        lx, ly = float(scene.listener.position[0]), float(scene.listener.position[1])
        pos = self._obj_pos.get(obj.id)
        if pos is None:
            pos = np.array([float(obj.position[0]), float(obj.position[1])])
        params = obj.params or {}
        v_kmh = float(params.get("velocity_kmh", params.get("approach_kmh", 0.0)) or 0.0)
        if v_kmh != 0.0 and dt > 0.0:
            if "heading_deg" in params:
                hd = math.radians(float(params["heading_deg"]))
                direction = np.array([math.sin(hd), math.cos(hd)])
            else:
                rel = np.array([lx, ly]) - pos
                nrm = float(np.hypot(rel[0], rel[1]))
                direction = rel / nrm if nrm > 1e-6 else np.array([0.0, 0.0])
            pos = pos + direction * (v_kmh / 3.6) * dt
        self._obj_pos[obj.id] = pos
        dx, dy = pos[0] - lx, pos[1] - ly
        distance = max(math.hypot(dx, dy) / 1000.0, 0.3)
        bearing = math.degrees(math.atan2(dx, dy)) % 360.0
        rate = float(params.get("flashes_per_min", th.flashes_per_min))
        return StormCellState(
            id=str(obj.id),
            distance_km=float(distance),
            bearing_deg=float(bearing),
            flash_rate_per_min=max(rate, 0.0),
            stroke_intensity=float(min(max(params.get("stroke_intensity", th.stroke_intensity), 0.0), 1.0)),
            cg_fraction=float(min(max(params.get("cg_fraction", th.cg_fraction), 0.0), 1.0)),
            rain_contribution=float(min(max(math.exp(-distance / 6.0), 0.0), 1.0)),
        )

    # ---------------------------------------------------------------- public
    def step(self, block_index: int, t0: float, dt: float, scene_at_t=None):
        """Advance one block. Returns (rain_multiplier, storms, envelope)."""
        sc = scene_at_t if scene_at_t is not None else self.scene
        w = sc.weather
        p = w.precipitation
        t = float(t0)
        # cells start exactly where the scene puts them; afterwards they move by elapsed time
        dt_track = t - self._last_t if (self._last_t is not None and t > self._last_t) else 0.0
        self._last_t = t

        vals = self._arc(t, w)
        character = "showery" if str(p.type).lower() == "showers" else str(p.character or "continuous").lower()
        shower_act = None
        slow = max(vals["env"], 0.0)
        dm = max(vals["dm"], 0.3)
        if character == "showery":
            period = max(float(p.shower_period_min), 1.0) * 60.0
            act, dm_sh = self._showers(t, period)
            shower_act = act
            slow *= act
            dm *= dm_sh
        elif character == "intermittent":
            slow *= self._intermittent(t)
        self.shower_activity = shower_act if shower_act is not None else 1.0

        self.cox = self.cox_modulation(t, _COX_SIGMA.get(character, 0.22))
        self.envelope = slow
        self.rain_multiplier = slow * self.cox
        self.dm_factor = dm
        self.thunder_envelope = max(vals["thunder"], 0.0)

        cells: list[StormCellState] = []
        if bool(getattr(w.thunder, "enabled", False)):
            cells.append(self._main_cell(t, dt_track, w, vals, shower_act))
        for obj in getattr(sc.place, "objects", None) or []:
            if str(getattr(obj, "type", "")).lower() == "storm" and (obj.params or {}).get("enabled", True):
                cells.append(self._object_cell(obj, t, dt_track, sc))
        self.cells = cells
        return self.rain_multiplier, list(cells), self.envelope

    def flashes_between(self, t0: float, t1: float) -> list:
        """Flashes [(t_flash, StormCellState)] in [t0, t1), deterministic per block slot."""
        out = []
        if t1 <= t0 or not self.cells:
            return out
        bd = self.block_s
        k0 = int(math.floor(t0 / bd))
        k1 = int(math.floor((t1 - 1e-12) / bd))
        for cell in self.cells:
            lam = max(cell.flash_rate_per_min, 0.0) / 60.0 * bd
            if lam <= 0.0:
                continue
            for k in range(k0, k1 + 1):
                rng = self.rngf.get(f"storm.flash.{cell.id}", k)
                count = int(rng.poisson(lam))
                if count == 0:
                    continue
                times = k * bd + rng.uniform(0.0, bd, count)
                gains = rng.lognormal(0.0, 0.3, count)
                for tf, gk in zip(times, gains):
                    tf = float(tf)
                    if t0 <= tf < t1:
                        si = float(min(max(cell.stroke_intensity * gk, 0.02), 1.0))
                        out.append((tf, replace(cell, stroke_intensity=si)))
        out.sort(key=lambda e: e[0])
        return out
