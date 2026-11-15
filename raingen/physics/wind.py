"""Wind: Beaufort scale, surface-layer profile, turbulence and the gust process.

Physics implemented here:

* WMO Beaufort scale (WMO-No. 306 / Manual on Codes), 10 m mean wind, with the
  official land descriptions; continuous fit v = 0.836 B^1.5 m/s.
* Neutral log-law profile u(z) = u10 ln(z/z0) / ln(10/z0) (Davenport / WMO roughness
  classes for z0); turbulence intensity I_u = 1 / ln(z/z0); gust factor G = 1 + 3 I_u
  (open sea ~1.28, open land ~1.5, suburb ~1.8, forest/town ~2.15 at 10 m).
* Gusts: normalised Kaimal surface-layer spectrum. Kaimal integrates to
  sigma_u^2 = 4.77 u*^2 regardless of terrain (Simiu & Scanlan), so the spectral
  *shape* is normalised to unit variance and scaled by the terrain turbulence
  intensity instead of mixing Kaimal's variance with a terrain-dependent beta.
  The shape is synthesised as a sum of Ornstein-Uhlenbeck processes on octave-spaced
  corners whose variances scale as f_k^(1-5/3): flat below the integral-scale corner
  and ~f^(-5/3) above it. The multiplier is log-normal (mean 1, positively skewed:
  gusts overshoot more than lulls undershoot) and never below 0.05.
* Gusts are advected (Taylor's frozen-turbulence hypothesis): a point L metres
  upwind experiences the listener's gust L/U seconds earlier.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.constants import BLOCK, SAMPLE_RATE

# ------------------------------------------------------------------ Beaufort (WMO)
BEAUFORT: list[dict] = [
    {"force": 0, "min_mps": 0.0, "max_mps": 0.2, "name": "Calm",
     "land": "Calm; smoke rises vertically."},
    {"force": 1, "min_mps": 0.3, "max_mps": 1.5, "name": "Light air",
     "land": "Direction of wind shown by smoke drift but not by wind vanes."},
    {"force": 2, "min_mps": 1.6, "max_mps": 3.3, "name": "Light breeze",
     "land": "Wind felt on face; leaves rustle; ordinary vane moved by wind."},
    {"force": 3, "min_mps": 3.4, "max_mps": 5.4, "name": "Gentle breeze",
     "land": "Leaves and small twigs in constant motion; wind extends light flag."},
    {"force": 4, "min_mps": 5.5, "max_mps": 7.9, "name": "Moderate breeze",
     "land": "Raises dust and loose paper; small branches are moved."},
    {"force": 5, "min_mps": 8.0, "max_mps": 10.7, "name": "Fresh breeze",
     "land": "Small trees in leaf begin to sway; crested wavelets form on inland waters."},
    {"force": 6, "min_mps": 10.8, "max_mps": 13.8, "name": "Strong breeze",
     "land": "Large branches in motion; whistling heard in telegraph wires; umbrellas used with difficulty."},
    {"force": 7, "min_mps": 13.9, "max_mps": 17.1, "name": "Near gale",
     "land": "Whole trees in motion; inconvenience felt when walking against the wind."},
    {"force": 8, "min_mps": 17.2, "max_mps": 20.7, "name": "Gale",
     "land": "Breaks twigs off trees; generally impedes progress."},
    {"force": 9, "min_mps": 20.8, "max_mps": 24.4, "name": "Strong gale",
     "land": "Slight structural damage occurs (chimney pots and slates removed)."},
    {"force": 10, "min_mps": 24.5, "max_mps": 28.4, "name": "Storm",
     "land": "Seldom experienced inland; trees uprooted; considerable structural damage occurs."},
    {"force": 11, "min_mps": 28.5, "max_mps": 32.6, "name": "Violent storm",
     "land": "Very rarely experienced; accompanied by widespread damage."},
    {"force": 12, "min_mps": 32.7, "max_mps": math.inf, "name": "Hurricane",
     "land": "Devastation."},
]


def beaufort_to_mps(b: float) -> float:
    """Continuous Beaufort force -> 10 m mean wind (m/s), v = 0.836 B^1.5."""
    return 0.836 * max(float(b), 0.0) ** 1.5


def mps_to_beaufort(v: float) -> float:
    """10 m mean wind (m/s) -> continuous Beaufort force."""
    return (max(float(v), 0.0) / 0.836) ** (2.0 / 3.0)


def beaufort_description(b: float) -> tuple[str, str]:
    """(name, land description) for the nearest Beaufort force (0..12)."""
    force = int(min(max(round(float(b)), 0), 12))
    row = BEAUFORT[force]
    return row["name"], row["land"]


# ------------------------------------------------------------------ roughness / profile
# Davenport-WMO roughness classes (m); tuned so G = 1 + 3/ln(10/z0) matches the gust table
_Z0_BY_PLACE = {
    "open_sea": 0.0002, "sea": 0.0002, "ocean": 0.0002, "offshore": 0.0002,
    "coast": 0.005, "beach": 0.003, "lake": 0.005, "lakeside": 0.005, "river": 0.01, "pond": 0.01,
    "snowfield": 0.001, "tundra": 0.005, "desert": 0.005, "mudflat": 0.003, "salt_flat": 0.001,
    "meadow": 0.03, "field": 0.03, "grassland": 0.03, "open": 0.03, "moor": 0.03, "heath": 0.04,
    "mountain_ridge": 0.05, "ridge": 0.05, "mountain": 0.08, "hill": 0.05,
    "farm": 0.1, "farmland": 0.1, "countryside": 0.1, "vineyard": 0.15,
    "garden": 0.2, "backyard": 0.2, "campsite": 0.2, "park": 0.3, "orchard": 0.4,
    "suburb": 0.25, "village": 0.3, "residential": 0.3,
    "deciduous_forest": 0.7, "conifer_forest": 0.8, "forest": 0.7, "woodland": 0.6, "wood": 0.6,
    "jungle": 1.0, "rainforest": 1.0,
    "city_street": 0.7, "street": 0.7, "town": 0.7, "alley": 1.0, "city": 1.0,
    "city_centre": 1.5, "city_center": 1.5, "downtown": 1.5,
}


def _norm_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def terrain_z0(place_preset: str) -> float:
    """Aerodynamic roughness length z0 (m) for a place preset name."""
    key = _norm_name(place_preset)
    if key in _Z0_BY_PLACE:
        return _Z0_BY_PLACE[key]
    for name in sorted(_Z0_BY_PLACE, key=len, reverse=True):
        if name in key:
            return _Z0_BY_PLACE[name]
    return 0.1


def _clamp_z0(z0: float) -> float:
    return min(max(float(z0), 1e-5), 2.0)


def _effective_height(z_m: float, z0: float) -> float:
    # log law is invalid inside the roughness sublayer; hold the profile at ~3 z0
    return max(float(z_m), 3.0 * z0, 0.05)


def wind_at_height(u10: float, z_m: float, z0: float) -> float:
    """Mean wind (m/s) at height z from the 10 m report, neutral log law (clamped)."""
    z0 = _clamp_z0(z0)
    z = _effective_height(z_m, z0)
    ref = max(10.0, 3.0 * z0 * 1.0001)
    ratio = math.log(z / z0) / math.log(ref / z0)
    return max(float(u10), 0.0) * min(max(ratio, 0.0), 2.0)


def turbulence_intensity_for(z0: float, z_m: float) -> float:
    """Longitudinal turbulence intensity I_u = 1/ln(z/z0) (neutral surface layer)."""
    z0 = _clamp_z0(z0)
    z = _effective_height(z_m, z0)
    return float(min(max(1.0 / math.log(z / z0), 0.04), 0.6))


def gust_factor_for(z0: float, z_m: float = 10.0) -> float:
    """Peak-gust factor G = 1 + 3 I_u (3-s gust over the 10-min mean)."""
    return 1.0 + 3.0 * turbulence_intensity_for(z0, z_m)


def kaimal_integral_time_s(u_mps: float, z_m: float = 10.0) -> float:
    """Gust correlation time from the Kaimal peak n = f z / U ~ 0.045: T = 1/(2 pi f_peak)."""
    f_peak = 0.045 * max(float(u_mps), 0.3) / max(float(z_m), 1.0)
    return float(min(max(1.0 / (2.0 * math.pi * f_peak), 1.5), 60.0))


# ------------------------------------------------------------------ gust process
_KAIMAL_SLOPE = 5.0 / 3.0


class GustProcess:
    """Deterministic, advected, Kaimal-shaped gust multiplier at the control rate.

    `step(block_index)` returns the multiplier at the listener for that block.
    Internally the process runs `lead_s` ahead, so `at_delay` accepts negative delays
    (the future at the listener = the present at an upwind point). Use
    `at_position(upwind_m, wind_mps)` to get the gust for an emitter `upwind_m` metres
    upwind (positive) or downwind (negative) of the listener.

    Every internal block draws its innovations from `rngf.get(layer, k)`; a sequential
    run is bit-exact. Jumps of up to `max_replay_s` are replayed exactly; longer seeks
    restart from a stationary state (statistically identical, not sample-identical).
    """

    def __init__(self, rngf, layer: str, integral_time_s: float = 8.0, ti: float = 0.2,
                 control_rate: float = SAMPLE_RATE / BLOCK, history_s: float = 30.0,
                 lead_s: Optional[float] = None, max_replay_s: float = 120.0):
        self.rngf = rngf
        self.layer = str(layer)
        self.control_rate = float(control_rate)
        self.dt = 1.0 / self.control_rate
        self.history_s = float(history_s)
        self.lead_s = float(lead_s) if lead_s is not None else min(12.0, 0.4 * self.history_s)
        self._lead = int(math.ceil(self.lead_s * self.control_rate))
        self._n_hist = int(math.ceil((self.history_s + self.lead_s) * self.control_rate)) + 4
        self._buf = np.ones(self._n_hist)
        self._max_replay = int(max_replay_s * self.control_rate)
        self._z: Optional[np.ndarray] = None
        self._next_k = 0              # next internal block to synthesise
        self._first_k = 0             # oldest internal block ever generated since reset
        self._block = -1              # last listener block returned
        self.value = 1.0
        self.integral_time_s = 0.0
        self.ti = 0.0
        self._corners = np.empty(0)
        self.set_params(integral_time_s, ti)

    # -- parameters ------------------------------------------------------
    def set_params(self, integral_time_s: Optional[float] = None, ti: Optional[float] = None) -> None:
        """Change the integral time scale and/or turbulence intensity smoothly."""
        if integral_time_s is not None:
            T = min(max(float(integral_time_s), 0.3), 120.0)
            if abs(T - self.integral_time_s) > 1e-9:
                self.integral_time_s = T
                f_c = 1.0 / (2.0 * math.pi * T)
                f_max = min(self.control_rate / 4.0, 4.0)
                corners = [f_c]
                while corners[-1] * 2.0 <= f_max and len(corners) < 12:
                    corners.append(corners[-1] * 2.0)
                while len(corners) < 3:
                    corners.append(corners[-1] * 2.0)
                self._corners = np.array(corners)
                var = self._corners ** (1.0 - _KAIMAL_SLOPE)
                self._w = np.sqrt(var / var.sum())
                self._rho = np.exp(-2.0 * math.pi * self._corners * self.dt)
                self._innov = np.sqrt(1.0 - self._rho ** 2)
                if self._z is not None and self._z.size != self._corners.size:
                    z = np.zeros(self._corners.size)
                    m = min(z.size, self._z.size)
                    z[:m] = self._z[:m]
                    self._z = z
        if ti is not None:
            self.ti = min(max(float(ti), 0.0), 1.0)
        self._s = math.sqrt(math.log(1.0 + self.ti * self.ti))

    # -- synthesis -------------------------------------------------------
    def _synth(self, k: int) -> None:
        rng = self.rngf.get(self.layer, k)
        eps = rng.standard_normal(self._corners.size)
        if self._z is None:
            self._z = rng.standard_normal(self._corners.size)     # stationary start, no warm-up
        else:
            self._z = self._rho * self._z + self._innov * eps
        g = float(np.dot(self._w, self._z))
        s = self._s
        m = math.exp(s * g - 0.5 * s * s)
        self._buf[k % self._n_hist] = max(m, 0.05)

    def _generate_until(self, k_last: int) -> None:
        while self._next_k <= k_last:
            self._synth(self._next_k)
            self._next_k += 1

    def _reset_at(self, k_start: int) -> None:
        self._z = None
        self._next_k = max(0, k_start)
        self._first_k = self._next_k

    def step(self, block_index: int) -> float:
        """Advance to `block_index` and return the gust multiplier at the listener."""
        b = int(block_index)
        if self._block < 0 or b < self._block:
            # (re)start: exact replay from 0 when cheap, else stationary restart
            start = 0 if b <= self._max_replay else b - min(self._max_replay, int(10 * self.integral_time_s * self.control_rate))
            self._reset_at(start)
        elif b - self._block > self._max_replay:
            self._reset_at(b - min(self._max_replay, int(10 * self.integral_time_s * self.control_rate)))
        self._generate_until(b + self._lead)
        self._block = b
        self.value = float(self._buf[b % self._n_hist])
        return self.value

    def _available(self) -> tuple[int, int]:
        lo = max(self._first_k, self._next_k - self._n_hist + 1)
        return lo, self._next_k - 1

    def at_delay(self, delay_s: float) -> float:
        """Multiplier at the listener `delay_s` seconds ago (negative = ahead, <= lead_s)."""
        if self._block < 0:
            return 1.0
        pos = self._block - float(delay_s) * self.control_rate
        lo, hi = self._available()
        pos = min(max(pos, float(lo)), float(hi))
        i0 = int(math.floor(pos))
        frac = pos - i0
        i1 = min(i0 + 1, hi)
        v0 = self._buf[i0 % self._n_hist]
        v1 = self._buf[i1 % self._n_hist]
        return float(v0 + (v1 - v0) * frac)

    def at_position(self, upwind_m: float, wind_mps: float) -> float:
        """Gust at a point `upwind_m` metres upwind of the listener (Taylor advection)."""
        u = max(float(wind_mps), 0.5)
        return self.at_delay(-float(upwind_m) / u)
