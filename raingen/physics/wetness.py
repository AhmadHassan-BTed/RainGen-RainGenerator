"""Surface water stores (film, puddles, canopy, roof) and integrate-and-fire drips.

Physics implemented here:

* Wetness memory per material (Yu & Hopkins 2018: dry -> film -> puddle changes the
  impact mechanism): linear-reservoir film store filled by rain x exposure and drained
  with a time constant proportional to the scene's aftermath tau; overflow beyond the
  film capacity runs off and partly collects in hollows (puddle depth, mm).
  Reservoirs are integrated exactly (S_eq + (S - S_eq) e^{-dt/tau}) so any dt is stable.
* Canopy interception (Rutter-type store): 10-40 % of incident rain is intercepted,
  falling as the store fills, and released later as larger drip drops - including
  after the rain stops.
* Roof -> gutter routing: roof film store plus a fast gutter reservoir; flow in L/s
  (1 mm on 1 m^2 = 1 L).
* Drips (integrate-and-fire): a pendant drop grows on an edge until its weight beats
  surface tension. Tate's law with the Harkins-Brown correction (f ~ 0.6):
  m g = 2 pi r sigma f  ->  D = (12 r sigma f / (rho g))^(1/3)  (3-5.5 mm for 0.5-3 mm
  edges). Interval = drop volume / inflow (a 50 cm^2 leaf in 5 mm/h rain drips every
  ~7 s). At high flow the leaky-faucet goes period-doubling and chaotic (Shaw 1984),
  modelled with a logistic map on the detachment threshold. Impact speed follows the
  sub-terminal fall law U(h) = V_T sqrt(1 - e^{-2gh/V_T^2}).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..core.constants import G, RHO_WATER, SIGMA_WATER
from .dsd import fall_speed_from_height

_HARKINS_BROWN = 0.6


def tate_drop_diameter_mm(edge_radius_mm: float) -> float:
    """Detaching drop diameter (mm) from an edge of radius `edge_radius_mm` (Tate's law)."""
    r = max(float(edge_radius_mm), 0.05) * 1e-3
    vol = 2.0 * math.pi * r * SIGMA_WATER * _HARKINS_BROWN / (RHO_WATER * G)    # m^3
    d = (6.0 * vol / math.pi) ** (1.0 / 3.0) * 1e3
    return float(min(max(d, 1.5), 7.0))


# material -> (film_scale_mm, film_capacity_mm, drain_factor, pond_fraction, puddle_max_mm, puddle_tau_factor)
#   film_scale: water depth at which the surface reads ~63 % wet
#   capacity:   depth held before runoff starts
#   drain:      film drying time = drain * aftermath_tau
#   pond:       share of runoff that collects in hollows on this material
_WET_PARAMS = {
    "glass": (0.02, 0.10, 0.6, 0.0, 0.0, 6.0),
    "window": (0.02, 0.10, 0.6, 0.0, 0.0, 6.0),
    "metal_sheet": (0.02, 0.12, 0.5, 0.0, 0.0, 6.0),
    "tin_roof": (0.02, 0.12, 0.5, 0.0, 0.0, 6.0),
    "car_body": (0.02, 0.10, 0.5, 0.0, 0.0, 6.0),
    "metal_gutter": (0.03, 0.20, 0.6, 0.0, 0.0, 6.0),
    "bucket": (0.02, 0.10, 0.6, 1.0, 300.0, 1e9),
    "plastic": (0.03, 0.12, 0.6, 0.0, 0.0, 6.0),
    "wood": (0.08, 0.40, 1.5, 0.0, 0.0, 6.0),
    "concrete": (0.08, 0.60, 1.2, 0.15, 12.0, 6.0),
    "asphalt": (0.06, 0.50, 1.0, 0.25, 15.0, 6.0),
    "tile": (0.05, 0.30, 0.8, 0.10, 8.0, 6.0),
    "gravel": (0.15, 1.00, 0.8, 0.05, 6.0, 4.0),
    "soil": (0.30, 2.00, 2.0, 0.20, 25.0, 8.0),
    "grass": (0.20, 1.20, 1.5, 0.05, 8.0, 6.0),
    "leaves": (0.10, 0.50, 1.2, 0.05, 8.0, 6.0),
    "needles": (0.15, 0.80, 1.5, 0.02, 5.0, 6.0),
    "fabric": (0.05, 0.30, 0.6, 0.0, 0.0, 6.0),
    "umbrella": (0.03, 0.15, 0.4, 0.0, 0.0, 6.0),
    "tent": (0.04, 0.25, 0.6, 0.0, 0.0, 6.0),
    "canvas": (0.05, 0.35, 0.7, 0.0, 0.0, 6.0),
    "snow": (0.50, 3.00, 3.0, 0.0, 0.0, 6.0),
}
_DEFAULT_WET = (0.08, 0.50, 1.0, 0.10, 10.0, 6.0)

_PUDDLE_BASE_MM = 3.0         # a 'puddle' surface is a puddle even before it rains
_PUDDLE_MAX_MM = 60.0
_PUDDLE_CONCENTRATION = 4.0   # runoff from the surroundings collects in hollows
_WATER_DEPTH_MM = 1000.0
_ROOF_FILM_MM = 0.3
_GUTTER_TAU_S = 25.0


def _relax(s: float, inflow: float, tau: float, dt: float) -> float:
    """Exact linear-reservoir update: dS/dt = inflow - S/tau."""
    s_eq = inflow * tau
    return s_eq + (s - s_eq) * math.exp(-dt / tau)


class WetnessStores:
    """Per-material film/puddle stores plus canopy and roof stores.

    `update(dt, rain_rate_mm_h, exposure_by_material)`: exposure keys are material
    names (0..1, area-weighted exposure to the sky). Two special keys are honoured:
    'canopy' (fraction of the scene under tree canopy, default 0) and 'roof'
    (fraction of the roof catchment exposed, default 1).
    """

    def __init__(self, materials: Iterable[str], aftermath_tau_min: float = 8.0,
                 canopy_capacity_mm: float = 1.5, roof_area_m2: float = 30.0):
        self.materials = list(dict.fromkeys(str(m) for m in materials))
        self.aftermath_tau_min = max(float(aftermath_tau_min), 0.25)
        self.canopy_capacity_mm = max(float(canopy_capacity_mm), 0.05)
        self.roof_area_m2 = max(float(roof_area_m2), 0.0)
        self._film = {m: 0.0 for m in self.materials}          # mm of water in the film
        self._puddle = {m: 0.0 for m in self.materials}        # extra puddle depth, mm
        self.canopy_mm = 0.0
        self.canopy_drip_mm_h = 0.0        # release rate of the canopy (drives tree drips)
        self.throughfall_fraction = 1.0
        self._roof_film_l = 0.0
        self._gutter_l = 0.0
        self.roof_flow_l_s = 0.0
        self.runoff_mm_h = {m: 0.0 for m in self.materials}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def params_for(material: str) -> tuple:
        return _WET_PARAMS.get(str(material), _DEFAULT_WET)

    def add_material(self, material: str) -> None:
        if material not in self._film:
            self.materials.append(material)
            self._film[material] = 0.0
            self._puddle[material] = 0.0
            self.runoff_mm_h[material] = 0.0

    @property
    def _tau_s(self) -> float:
        return self.aftermath_tau_min * 60.0

    # -- dynamics --------------------------------------------------------
    def update(self, dt: float, rain_rate_mm_h: float, exposure_by_material: dict) -> None:
        dt = max(float(dt), 0.0)
        if dt <= 0.0:
            return
        rate = max(float(rain_rate_mm_h), 0.0)
        exp_map = exposure_by_material or {}
        tau = self._tau_s
        for m in self.materials:
            e = min(max(float(exp_map.get(m, 1.0)), 0.0), 1.0)
            inflow = e * rate / 3600.0                                   # mm/s
            if m == "water":
                self._film[m] = 1.0
                self._puddle[m] = 0.0
                self.runoff_mm_h[m] = 0.0
                continue
            scale, cap, drain, pond, pmax, ptau = self.params_for(m)
            tau_f = max(drain * tau, 5.0)
            w = _relax(self._film[m], inflow, tau_f, dt)
            overflow = max(0.0, w - cap)
            w = min(w, cap)
            self._film[m] = w
            self.runoff_mm_h[m] = overflow / dt * 3600.0
            tau_p = max(ptau * tau, 5.0)
            if m == "puddle":
                p_in = inflow * _PUDDLE_CONCENTRATION
                self._puddle[m] = min(_relax(self._puddle[m], p_in, tau_p, dt), _PUDDLE_MAX_MM - _PUDDLE_BASE_MM)
            elif m == "bucket":
                self._puddle[m] = min(self._puddle[m] + inflow * dt, pmax)
            elif pond > 0.0 and pmax > 0.0:
                p_in = pond * _PUDDLE_CONCENTRATION * overflow / dt
                self._puddle[m] = min(_relax(self._puddle[m], p_in, tau_p, dt), pmax)
            else:
                self._puddle[m] = 0.0

        # canopy interception store (Rutter-type)
        cover = min(max(float(exp_map.get("canopy", 0.0)), 0.0), 1.0)
        S = self.canopy_capacity_mm
        if cover > 0.0 or self.canopy_mm > 0.0:
            fill = min(self.canopy_mm / S, 1.5)
            f_int = min(max(0.10 + 0.30 * (1.0 - fill), 0.10), 0.40)
            intercepted = f_int * rate / 3600.0 if cover > 0.0 else 0.0
            tau_fast = 90.0
            tau_slow = max(1.5 * tau, 30.0)
            c = self.canopy_mm
            # fast drainage above ~25 % of capacity, slow drip/evaporation below
            excess = max(0.0, c - 0.25 * S)
            drain_rate = excess / tau_fast + c / tau_slow
            c_new = c + (intercepted - drain_rate) * dt
            if c_new > 1.5 * S:
                c_new = 1.5 * S
            c_new = max(c_new, 0.0)
            released = max(c + intercepted * dt - c_new, 0.0)
            self.canopy_mm = c_new
            self.canopy_drip_mm_h = released / dt * 3600.0
            self.throughfall_fraction = 1.0 - f_int
        else:
            self.canopy_drip_mm_h = 0.0
            self.throughfall_fraction = 1.0

        # roof -> gutter
        if self.roof_area_m2 > 0.0:
            e_roof = min(max(float(exp_map.get("roof", 1.0)), 0.0), 1.0)
            q_in = e_roof * rate / 3600.0 * self.roof_area_m2              # L/s
            film_cap = _ROOF_FILM_MM * self.roof_area_m2                   # L
            tau_rf = max(0.5 * tau, 10.0)
            f_new = min(_relax(self._roof_film_l, q_in, tau_rf, dt), film_cap)
            # what reached the gutter during dt: drainage plus spill over the film capacity
            to_gutter = max(self._roof_film_l + q_in * dt - f_new, 0.0)
            self._roof_film_l = f_new
            g_in = to_gutter / dt
            self._gutter_l = _relax(self._gutter_l, g_in, _GUTTER_TAU_S, dt)
            self.roof_flow_l_s = self._gutter_l / _GUTTER_TAU_S
        else:
            self.roof_flow_l_s = 0.0

    def equilibrate(self, rain_rate_mm_h: float, exposure_by_material: dict,
                    minutes: float = 120.0, step_s: float = 20.0) -> None:
        """Spin the stores up to (near) steady state for a scene that starts mid-rain."""
        n = max(1, int(minutes * 60.0 / step_s))
        for _ in range(n):
            self.update(step_s, rain_rate_mm_h, exposure_by_material)

    def decay(self, minutes: float, exposure_by_material: dict, step_s: float = 20.0) -> None:
        """Let the stores dry for `minutes` without rain (aftermath start states)."""
        n = max(0, int(minutes * 60.0 / step_s))
        for _ in range(n):
            self.update(step_s, 0.0, exposure_by_material)

    # -- readout ---------------------------------------------------------
    def film(self, material: str) -> float:
        """Surface wetness 0 (dry) .. 1 (continuous water film)."""
        if material in ("water", "puddle"):
            return 1.0
        if material not in self._film:
            return 0.0
        scale = self.params_for(material)[0]
        return float(1.0 - math.exp(-self._film[material] / max(scale, 1e-6)))

    def puddle_mm(self, material: str) -> float:
        """Standing water depth (mm) on/in this material."""
        if material == "water":
            return _WATER_DEPTH_MM
        if material == "puddle":
            return _PUDDLE_BASE_MM + float(self._puddle.get("puddle", 0.0))
        return float(self._puddle.get(material, 0.0))

    def film_mm(self, material: str) -> float:
        return float(self._film.get(material, 0.0))

    def snapshot(self) -> tuple[dict, dict]:
        wet = {m: self.film(m) for m in self.materials}
        depth = {m: self.puddle_mm(m) for m in self.materials}
        return wet, depth


# ------------------------------------------------------------------ drips
_TAU_STORE_S = 180.0
_MAX_DRIP_HZ = 40.0
_RESIDUAL = 0.04              # fraction of the pendant volume left on the edge after detachment


def _smoothstep(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class DripNode:
    """One drip point (leaf tip, eave lip, umbrella rim, cave ceiling).

    `step(block_index, n, sr, inflow_mm_h, storage_mm)` integrates the inflow over the
    catchment and fires drops when the pendant volume crosses the (jittered, possibly
    chaotic) detachment threshold. Returns [(sample_offset, diameter_mm, impact_speed)].
    `storage_mm` is water held upstream (film, canopy) that keeps feeding the node after
    the rain stops (drained with a 3-minute time constant).
    """

    def __init__(self, id: str, catchment_m2: float, edge_radius_mm: float, fall_height_m: float, rngf, layer: str):
        self.id = str(id)
        self.catchment_m2 = max(float(catchment_m2), 1e-6)
        self.edge_radius_mm = float(edge_radius_mm)
        self.fall_height_m = max(float(fall_height_m), 0.0)
        self.rngf = rngf
        self.layer = str(layer)
        self.d_mm = tate_drop_diameter_mm(edge_radius_mm)
        self.volume_mm3 = math.pi / 6.0 * self.d_mm ** 3
        self._acc = None
        self._threshold = None
        self._x = 0.5
        self.rate_hz = 0.0
        self.chaos = 0.0
        self.stream_l_s = 0.0        # excess flow beyond the dripping regime (a trickle)
        self.drops_total = 0

    def step(self, block_index: int, n: int, sr: int, inflow_mm_h: float, storage_mm: float = 0.0) -> list:
        rng = self.rngf.get(self.layer, int(block_index))
        V = self.volume_mm3
        if self._acc is None:
            self._acc = float(rng.uniform(0.0, 0.9)) * V      # random initial phase
            self._threshold = V
            self._x = float(rng.uniform(0.2, 0.8))
        n = int(n)
        dur = n / float(sr)
        q = (max(float(inflow_mm_h), 0.0) / 3600.0 + max(float(storage_mm), 0.0) / _TAU_STORE_S) \
            * self.catchment_m2 * 1e6                                   # mm^3/s
        rate = q / V
        if rate > _MAX_DRIP_HZ:
            self.stream_l_s = (q - _MAX_DRIP_HZ * V) * 1e-6
            q = _MAX_DRIP_HZ * V
            rate = _MAX_DRIP_HZ
        else:
            self.stream_l_s = 0.0
        self.rate_hz = rate
        # Shaw (1984): periodic at low flow, period doubling ~3 Hz, chaotic above ~10 Hz
        self.chaos = _smoothstep(rate, 2.0, 12.0)
        r_map = 3.2 + 0.79 * self.chaos
        events = []
        if q <= 0.0 or n <= 0:
            return events
        t = 0.0
        while len(events) < 512:
            need = self._threshold - self._acc
            dt_fire = max(need, 0.0) / q
            if t + dt_fire >= dur:
                self._acc += q * (dur - t)
                break
            t += dt_fire
            released = self._threshold * (1.0 - _RESIDUAL)
            self._acc = self._threshold * _RESIDUAL
            jitter = 1.0 + rng.normal(0.0, 0.015 + 0.08 * self.chaos)
            diam = self.d_mm * max(released / V, 0.05) ** (1.0 / 3.0) * max(jitter, 0.5)
            v_imp = float(fall_speed_from_height(diam, self.fall_height_m))
            offset = min(int(t * sr), n - 1)
            events.append((offset, float(diam), v_imp))
            self.drops_total += 1
            self._x = r_map * self._x * (1.0 - self._x)
            self._x = min(max(self._x, 1e-6), 1.0 - 1e-6)
            thr = V * (1.0 + rng.normal(0.0, 0.03)) * (1.0 + 0.6 * self.chaos * (self._x - 0.5))
            self._threshold = max(thr, 0.3 * V)
        return events
