"""Drop-size distributions, fall speed and the five acoustic drop classes.

Physics implemented here:

* Normalized gamma DSD (Testud et al. 2001; Bringi et al. 2003, J. Atmos. Sci. 60:354)::

      N(D) = Nw f(mu) (D/Dm)^mu exp(-(4+mu) D/Dm)          [m^-3 mm^-1, D in mm]
      f(mu) = 6/4^4 (4+mu)^(mu+4) / Gamma(mu+4)

  Climate regimes (Bringi 2003): maritime convective Dm ~1.5-1.75 mm with high Nw,
  continental convective Dm ~2-2.75 mm with low Nw, stratiform in between. In heavy
  rain extra rate comes mostly from *more* drops, so Dm saturates.
* Marshall & Palmer (1948): N0 = 8000 m^-3 mm^-1, Lambda = 4.1 R^-0.21 mm^-1
  (exponential: mu = 0, Dm = 4/Lambda, Nw = N0).
* Terminal fall speed: Atlas, Srivastava & Sekhon (1973) v = 9.65 - 10.3 exp(-0.6 D)
  for D >= 0.5 mm; Stokes drag below 0.1 mm; a linear bridge in between that matches
  Gunn & Kinzer (1949) to within 3 %. Sub-terminal speed after a fall h:
  U(h) = V_T sqrt(1 - exp(-2 g h / V_T^2)).
* Rain rate R = 6 pi 1e-4 Int D^3 v(D) N(D) dD [mm/h]; impact flux Int v(D) N(D) dD
  [m^-2 s^-1] (sanity: Marshall-Palmer, D >= 0.5 mm gives ~750 R^0.56).
* Acoustic classes on water: Medwin, Nystuen et al. (1992) JASA 92:1613 boundaries
  0.8 / 1.1 / 2.2 / 3.5 mm, or Nystuen (2001) 0.8 / 1.2 / 2.0 / 3.5 mm.
* Disdrometer import fits the normalized gamma by moments (Bringi & Chandrasekar 2001):
  Dm = M4/M3, Nw = 4^4/6 M3/Dm^4, mu = Dm^2/sigma_m^2 - 4.
"""

from __future__ import annotations

import csv
import math
from typing import Optional

import numpy as np
from scipy.special import gammaln

from ..core.constants import G, RHO_AIR, RHO_WATER
from ..core.types import DSDParams

CLASS_NAMES = ("minuscule", "small", "mid", "large", "very_large")

_CLASS_EDGES = {
    "medwin": (0.8, 1.1, 2.2, 3.5),
    "nystuen": (0.8, 1.2, 2.0, 3.5),
}

_MU_AIR = 1.81e-5          # dynamic viscosity of air, Pa s (~15 C)
_D_STOKES_MM = 0.1         # below: Stokes regime
_D_ATLAS_MM = 0.5          # above: Atlas et al. (1973)
_N_GRID = 384              # quadrature points per integral
_RATE_CONST = 6.0 * math.pi * 1e-4   # mm^3 m^-3 (m/s) -> mm/h


# ---------------------------------------------------------------- classes
def class_bounds(scheme: str = "medwin", d_max_mm: float = 6.0) -> dict[str, tuple[float, float]]:
    """Diameter bounds (lo_mm, hi_mm) of the five acoustic drop classes."""
    edges = _CLASS_EDGES.get(str(scheme).lower())
    if edges is None:
        raise ValueError(f"unknown class boundary scheme {scheme!r} (use 'medwin' or 'nystuen')")
    top = max(float(d_max_mm), edges[-1])
    pts = (0.0,) + edges + (top,)
    return {name: (pts[i], pts[i + 1]) for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------- fall speed
def _atlas(d: np.ndarray) -> np.ndarray:
    return 9.65 - 10.3 * np.exp(-0.6 * d)


def _stokes(d: np.ndarray) -> np.ndarray:
    r = np.asarray(d, dtype=float) * 1e-3
    return G * r * r * (RHO_WATER - RHO_AIR) / (18.0 * _MU_AIR)


_V_STOKES_EDGE = float(_stokes(np.array(_D_STOKES_MM)))
_V_ATLAS_EDGE = float(_atlas(np.array(_D_ATLAS_MM)))


def fall_speed(d_mm):
    """Terminal fall speed (m/s) of raindrops of diameter `d_mm` (vectorized).

    Atlas et al. (1973) above 0.5 mm, Stokes below 0.1 mm, linear bridge between;
    never negative.
    """
    d = np.maximum(np.asarray(d_mm, dtype=float), 0.0)
    v = _atlas(d)
    bridge = _V_STOKES_EDGE + (_V_ATLAS_EDGE - _V_STOKES_EDGE) * (d - _D_STOKES_MM) / (_D_ATLAS_MM - _D_STOKES_MM)
    v = np.where(d < _D_ATLAS_MM, bridge, v)
    v = np.where(d < _D_STOKES_MM, _stokes(d), v)
    v = np.maximum(v, 0.0)
    return v if v.ndim else float(v)


def fall_speed_from_height(d_mm, h_m):
    """Impact speed (m/s) of a drop released at rest that fell `h_m` metres (drips).

    U(h) = V_T sqrt(1 - exp(-2 g h / V_T^2)) (exponential approach to terminal speed).
    """
    vt = np.asarray(fall_speed(d_mm), dtype=float)
    h = np.maximum(np.asarray(h_m, dtype=float), 0.0)
    vt_safe = np.maximum(vt, 1e-6)
    u = vt * np.sqrt(1.0 - np.exp(-2.0 * G * h / (vt_safe * vt_safe)))
    u = np.where(vt > 0.0, u, 0.0)
    return u if u.ndim else float(u)


# ---------------------------------------------------------------- distribution
def _f_mu(mu: float) -> float:
    mu = float(mu)
    return 6.0 / 4.0 ** 4 * math.exp((mu + 4.0) * math.log(4.0 + mu) - float(gammaln(mu + 4.0)))


def number_density(dsd: DSDParams, d_mm) -> np.ndarray:
    """N(D) in m^-3 mm^-1 for diameters `d_mm` (zero outside [d_min, d_max])."""
    d = np.asarray(d_mm, dtype=float)
    if dsd.nw <= 0.0 or dsd.dm_mm <= 0.0:
        return np.zeros_like(d)
    x = np.maximum(d, 1e-9) / dsd.dm_mm
    mu = float(dsd.mu)
    with np.errstate(over="ignore", under="ignore"):
        n = dsd.nw * _f_mu(mu) * np.exp(mu * np.log(x) - (4.0 + mu) * x)
    inside = (d >= dsd.d_min_mm) & (d <= dsd.d_max_mm)
    return np.where(inside, n, 0.0)


def _trap(y: np.ndarray, x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x)))


def _moment_integral(dsd: DSDParams, lo: float, hi: float, power: float, with_speed: bool) -> float:
    lo = max(float(lo), float(dsd.d_min_mm))
    hi = min(float(hi), float(dsd.d_max_mm))
    if hi <= lo or dsd.nw <= 0.0:
        return 0.0
    d = np.linspace(lo, hi, _N_GRID)
    y = number_density(dsd, d)
    if power:
        y = y * d ** power
    if with_speed:
        y = y * fall_speed(d)
    return _trap(y, d)


def rain_rate(dsd: DSDParams) -> float:
    """Rain rate (mm/h) carried by the distribution: 6 pi 1e-4 Int D^3 v N dD."""
    return _RATE_CONST * _moment_integral(dsd, dsd.d_min_mm, dsd.d_max_mm, 3.0, True)


def total_flux(dsd: DSDParams, d_lo_mm: float = 0.0, d_hi_mm: float = math.inf) -> float:
    """Impacts per m^2 per s for drops between d_lo_mm and d_hi_mm."""
    return _moment_integral(dsd, d_lo_mm, d_hi_mm, 0.0, True)


def class_flux(dsd: DSDParams, scheme: str = "medwin") -> dict[str, float]:
    """Impacts / m^2 / s on a horizontal surface for each acoustic drop class."""
    bounds = class_bounds(scheme, dsd.d_max_mm)
    return {name: _moment_integral(dsd, lo, hi, 0.0, True) for name, (lo, hi) in bounds.items()}


def liquid_water_content(dsd: DSDParams) -> float:
    """Liquid water content in g/m^3: (pi/6) 1e-3 Int D^3 N dD."""
    return math.pi / 6.0 * 1e-3 * _moment_integral(dsd, dsd.d_min_mm, dsd.d_max_mm, 3.0, False)


def concentration(dsd: DSDParams) -> float:
    """Total number concentration (drops per m^3) inside the cutoffs."""
    return _moment_integral(dsd, dsd.d_min_mm, dsd.d_max_mm, 0.0, False)


def median_volume_diameter(dsd: DSDParams) -> float:
    """D0 (mm): half the liquid water is in drops smaller than D0."""
    lo, hi = float(dsd.d_min_mm), float(dsd.d_max_mm)
    if hi <= lo or dsd.nw <= 0.0:
        return 0.0
    d = np.linspace(lo, hi, _N_GRID)
    y = number_density(dsd, d) * d ** 3
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(d))))
    if cum[-1] <= 0.0:
        return 0.0
    return float(np.interp(0.5 * cum[-1], cum, d))


# ---------------------------------------------------------------- regimes
# drops -> (a, b, soft cap mm, mu) with Dm = a R^b, saturating at the cap (Bringi 2003 clusters)
_REGIMES = {
    "fine": (0.85, 0.17, 1.75, 4.0),     # maritime / tropical convective: many small drops
    "normal": (1.00, 0.20, 2.25, 3.0),   # stratiform / mid-latitude
    "fat": (1.35, 0.17, 2.75, 2.0),      # continental convective: few big drops
}
_DM_FLOOR_MM = 0.35


def _soft_cap(x: float, cap: float, p: float = 4.0) -> float:
    return x / (1.0 + (x / cap) ** p) ** (1.0 / p)


def _solve_nw(dm: float, mu: float, d_min: float, d_max: float, rate: float) -> float:
    unit = DSDParams(nw=1.0, dm_mm=dm, mu=mu, d_min_mm=d_min, d_max_mm=d_max)
    r1 = rain_rate(unit)
    return rate / r1 if r1 > 0.0 else 0.0


def _solve_dm(nw: float, mu: float, d_min: float, d_max: float, rate: float) -> float:
    """Dm such that rain_rate == rate at fixed Nw (rate grows monotonically with Dm)."""
    lo, hi = math.log(0.1), math.log(8.0)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        r = rain_rate(DSDParams(nw=nw, dm_mm=math.exp(mid), mu=mu, d_min_mm=d_min, d_max_mm=d_max))
        if r < rate:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def dsd_from_scene(precip, rate_mm_h: float) -> DSDParams:
    """Resolve a `Precipitation` block and an instantaneous rate into a DSD.

    `drops` fine/normal/fat moves Dm along the maritime / stratiform / continental
    clusters; Dm grows with rate but saturates (<= ~2.75 mm). Nw is then solved so
    that rain_rate(dsd) == rate_mm_h (closure). Engineer overrides `mu`, `dm_mm`,
    `nw` and the cutoffs are honoured: with `nw` pinned and `dm_mm` free, Dm is solved
    instead; with both pinned the rate follows from them.
    """
    rate = max(float(rate_mm_h), 0.0)
    ptype = str(getattr(precip, "type", "rain") or "rain").lower()
    d_min = max(float(getattr(precip, "d_min_mm", 0.3)), 0.01)
    d_max = max(float(getattr(precip, "d_max_mm", 6.0)), d_min + 0.05)
    model = str(getattr(precip, "dsd_model", "gamma") or "gamma").lower().replace("-", "_")
    drops = str(getattr(precip, "drops", "normal") or "normal").lower()

    if ptype == "drizzle":
        dm = min(max(0.35 * (max(rate, 1e-3) / 0.2) ** 0.08, 0.25), 0.5)
        mu = 3.0
        d_min = min(d_min, 0.1)
        d_max = max(min(d_max, 1.2), d_min + 0.05)
    elif model in ("marshall_palmer", "mp", "marshallpalmer"):
        lam = 4.1 * max(rate, 1e-3) ** -0.21
        dm = 4.0 / lam
        mu = 0.0
    else:
        a, b, cap, mu = _REGIMES.get(drops, _REGIMES["normal"])
        dm = max(_DM_FLOOR_MM, _soft_cap(a * max(rate, 1e-3) ** b, cap))

    mu_ovr = getattr(precip, "mu", None)
    dm_ovr = getattr(precip, "dm_mm", None)
    nw_ovr = getattr(precip, "nw", None)
    if mu_ovr is not None:
        mu = float(mu_ovr)
    mu = min(max(float(mu), -3.0), 20.0)
    if dm_ovr is not None:
        dm = max(float(dm_ovr), 0.05)

    if ptype == "none" or rate <= 0.0:
        return DSDParams(nw=0.0, dm_mm=dm, mu=mu, d_min_mm=d_min, d_max_mm=d_max)

    if nw_ovr is not None:
        nw = max(float(nw_ovr), 0.0)
        if dm_ovr is None and nw > 0.0:
            dm = _solve_dm(nw, mu, d_min, d_max, rate)
        return DSDParams(nw=nw, dm_mm=dm, mu=mu, d_min_mm=d_min, d_max_mm=d_max)

    nw = _solve_nw(dm, mu, d_min, d_max, rate)
    return DSDParams(nw=nw, dm_mm=dm, mu=mu, d_min_mm=d_min, d_max_mm=d_max)


def dsd_with_dm(dsd: DSDParams, dm_mm: float, rate_mm_h: float) -> DSDParams:
    """Same shape and cutoffs with a new Dm, Nw re-solved to keep the rain rate.

    Used for size sorting: at a cell's leading edge Dm rises while the drop count falls.
    """
    dm = max(float(dm_mm), 0.05)
    rate = max(float(rate_mm_h), 0.0)
    if rate <= 0.0 or dsd.nw <= 0.0:
        return DSDParams(nw=dsd.nw if rate > 0.0 else 0.0, dm_mm=dm, mu=dsd.mu,
                         d_min_mm=dsd.d_min_mm, d_max_mm=dsd.d_max_mm)
    nw = _solve_nw(dm, dsd.mu, dsd.d_min_mm, dsd.d_max_mm, rate)
    return DSDParams(nw=nw, dm_mm=dm, mu=dsd.mu, d_min_mm=dsd.d_min_mm, d_max_mm=dsd.d_max_mm)


def marshall_palmer(rate_mm_h: float, d_min_mm: float = 0.3, d_max_mm: float = 6.0) -> DSDParams:
    """The classic exponential DSD (N0 = 8000, Lambda = 4.1 R^-0.21), untruncated Nw."""
    lam = 4.1 * max(float(rate_mm_h), 1e-3) ** -0.21
    return DSDParams(nw=8000.0 if rate_mm_h > 0 else 0.0, dm_mm=4.0 / lam, mu=0.0,
                     d_min_mm=d_min_mm, d_max_mm=d_max_mm)


# ---------------------------------------------------------------- sampling
def sample_diameters(rng: np.random.Generator, dsd: DSDParams, lo_mm: float, hi_mm: float, n: int) -> np.ndarray:
    """Draw `n` impact diameters (mm) in [lo_mm, hi_mm].

    Impacts on a surface are flux-weighted, so the sampling density is N(D) v(D)
    (inverse-CDF on a fine grid).
    """
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=float)
    lo = max(float(lo_mm), float(dsd.d_min_mm))
    hi = min(float(hi_mm), float(dsd.d_max_mm))
    if not hi > lo:
        lo_c = float(lo_mm)
        hi_c = float(hi_mm) if math.isfinite(float(hi_mm)) else lo_c
        return np.full(n, 0.5 * (lo_c + hi_c) if hi_c > lo_c else lo_c, dtype=float)
    d = np.linspace(lo, hi, 256)
    pdf = number_density(dsd, d) * fall_speed(d)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(d))))
    u = rng.random(n)
    if cdf[-1] <= 0.0:
        return lo + (hi - lo) * u
    return np.interp(u * cdf[-1], cdf, d)


# ---------------------------------------------------------------- disdrometer import
def dsd_from_spectrum(d_mm, n_d, d_min_mm: Optional[float] = None, d_max_mm: Optional[float] = None) -> DSDParams:
    """Fit a normalized gamma to a measured spectrum N(D) [m^-3 mm^-1] by moments."""
    d = np.asarray(d_mm, dtype=float)
    nd = np.maximum(np.asarray(n_d, dtype=float), 0.0)
    order = np.argsort(d)
    d, nd = d[order], nd[order]
    if d.size < 2:
        raise ValueError("need at least two diameter bins")
    m3 = _trap(nd * d ** 3, d)
    m4 = _trap(nd * d ** 4, d)
    m5 = _trap(nd * d ** 5, d)
    lo = float(d_min_mm) if d_min_mm is not None else float(max(d[0], 0.05))
    hi = float(d_max_mm) if d_max_mm is not None else float(d[-1])
    if m3 <= 0.0:
        return DSDParams(nw=0.0, dm_mm=1.0, mu=3.0, d_min_mm=lo, d_max_mm=hi)
    dm = m4 / m3
    var_m = max(m5 / m3 - dm * dm, 1e-6)
    mu = min(max(dm * dm / var_m - 4.0, -3.0), 20.0)
    nw = 4.0 ** 4 / 6.0 * m3 / dm ** 4
    return DSDParams(nw=float(nw), dm_mm=float(dm), mu=float(mu), d_min_mm=lo, d_max_mm=hi)


def dsd_from_disdrometer_csv(path: str) -> DSDParams:
    """Read a two-column CSV (diameter_mm, N(D) in m^-3 mm^-1) and fit the DSD.

    Non-numeric rows (headers, comments) are skipped.
    """
    ds, ns = [], []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                dv, nv = float(row[0]), float(row[1])
            except ValueError:
                continue
            ds.append(dv)
            ns.append(nv)
    return dsd_from_spectrum(np.array(ds), np.array(ns))
