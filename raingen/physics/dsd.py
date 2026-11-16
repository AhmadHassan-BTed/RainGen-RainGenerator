"""Drop-size distributions, fall speed and the five acoustic drop classes.

Physics implemented here:

* Normalized gamma DSD (Testud et al. 2001; Bringi et al. 2003, J. Atmos. Sci. 60:354)::

      N(D) = Nw f(mu) (D/Dm)^mu exp(-(4+mu) D/Dm)          [m^-3 mm^-1, D in mm]
      f(mu) = 6/4^4 (4+mu)^(mu+4) / Gamma(mu+4)

  Climate regimes (Bringi 2003 convective clusters, which sit near ~30 mm/h): maritime
  convective Dm ~1.5-1.75 mm with log10 Nw ~4-4.5, continental convective Dm ~2-2.75 mm
  with log10 Nw ~3-3.5, stratiform in between. With mu = 3 the 'fine' / 'fat' settings
  reproduce, at 30 mm/h, ~3x more impacts per mm/h for maritime rain and a 3 % vs 19 %
  share of impacts from drops over 2.2 mm (Bringi cluster centres: 3.4 % / 18.6 %).
  In heavy rain extra rate comes mostly from *more* drops, so Dm saturates.
* Marshall & Palmer (1948): N0 = 8000 m^-3 mm^-1, Lambda = 4.1 R^-0.21 mm^-1
  (exponential: mu = 0, Dm = 4/Lambda, Nw = N0).
* Terminal fall speed: Atlas, Srivastava & Sekhon (1973) v = 9.65 - 10.3 exp(-0.6 D)
  for D >= 0.5 mm; Stokes drag below 0.1 mm; a linear bridge in between that matches
  Gunn & Kinzer (1949) to within 3 %. Both laws hold at sea level (1013 hPa, 20 C,
  rho_air = 1.204 kg/m^3); thinner air lets drops fall faster, v = v0 (rho0/rho)^0.4
  (Foote & du Toit 1969), available through `rho_air`. Sub-terminal speed after a
  fall h: U(h) = V_T sqrt(1 - exp(-2 g h / V_T^2)).
* Rain rate R = 6 pi 1e-4 Int D^3 v(D) N(D) dD [mm/h]; impact flux Int v(D) N(D) dD
  [m^-2 s^-1] (sanity: Marshall-Palmer, D >= 0.5 mm gives ~750 R^0.56).
* Acoustic classes on water: Medwin, Nystuen et al. (1992) JASA 92:1613 boundaries
  0.8 / 1.1 / 2.2 / 3.5 mm, or Nystuen (2001) 0.8 / 1.2 / 2.0 / 3.5 mm.
* Disdrometer import fits the normalized gamma by moments (Bringi & Chandrasekar 2001):
  Dm = M4/M3, Nw = 4^4/6 M3/Dm^4, mu = Dm^2/sigma_m^2 - 4.

Performance: every integral (rain rate, class and band fluxes, LWC, D0) and the
inverse-CDF sampler read one cumulative table per DSD *shape* (Dm, mu, cutoffs; Nw
only scales it). The table lives on a fixed grid per cutoff pair that contains the
class edges of both schemes and the fall-speed breakpoints, so class fluxes are exact
trapezoid sums. A block resolves one or two shapes, so the hundreds of per-patch
`sample_diameters` calls in a block share one table.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.special import gammaln

from ..core.constants import RHO_AIR, RHO_WATER, G
from ..core.types import DSDParams

CLASS_NAMES = ("minuscule", "small", "mid", "large", "very_large")

_CLASS_EDGES = {
    "medwin": (0.8, 1.1, 2.2, 3.5),
    "nystuen": (0.8, 1.2, 2.0, 3.5),
}

_MU_AIR = 1.81e-5          # dynamic viscosity of air, Pa s (~15 C)
_D_STOKES_MM = 0.1         # below: Stokes regime
_D_ATLAS_MM = 0.5          # above: Atlas et al. (1973)
_RATE_CONST = 6.0 * math.pi * 1e-4   # mm^3 m^-3 (m/s) -> mm/h
_R_DRY_AIR = 287.05        # J kg^-1 K^-1
_FOOTE_EXP = 0.4           # Foote & du Toit (1969) density exponent
_TABLE_INTERVALS = 768     # uniform intervals of the shape-table grid (plus breakpoints)


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
_BRIDGE_SLOPE = (_V_ATLAS_EDGE - _V_STOKES_EDGE) / (_D_ATLAS_MM - _D_STOKES_MM)
_STOKES_K = G * 1e-6 * (RHO_WATER - RHO_AIR) / (18.0 * _MU_AIR)      # v = K d_mm^2


def air_density(temp_c: float = 20.0, pressure_hpa: float = 1013.25) -> float:
    """Dry-air density (kg/m^3) from the ideal gas law (1.204 at 20 C, 1013.25 hPa)."""
    return max(float(pressure_hpa), 1.0) * 100.0 / (_R_DRY_AIR * (float(temp_c) + 273.15))


def _density_factor(rho_air) -> float:
    if rho_air is None:
        return 1.0
    return (RHO_AIR / max(float(rho_air), 1e-3)) ** _FOOTE_EXP


def _fall_speed_scalar(d: float) -> float:
    if d >= _D_ATLAS_MM:
        return max(9.65 - 10.3 * math.exp(-0.6 * d), 0.0)
    if d >= _D_STOKES_MM:
        return _V_STOKES_EDGE + _BRIDGE_SLOPE * (d - _D_STOKES_MM)
    if d > 0.0:
        return _STOKES_K * d * d
    return 0.0


# piecewise-linear lookup of the speed law on a 1 um grid (exact at the 0.1 / 0.5 mm
# breakpoints, < 1e-6 m/s interpolation error; beyond 16 mm Atlas is within 1e-3 of its asymptote)
_VT_GRID = np.arange(16001) / 1000.0
_VT_TABLE = np.where(_VT_GRID >= _D_ATLAS_MM, _atlas(_VT_GRID),
                     np.where(_VT_GRID >= _D_STOKES_MM, _V_STOKES_EDGE + _BRIDGE_SLOPE * (_VT_GRID - _D_STOKES_MM),
                              _STOKES_K * _VT_GRID * _VT_GRID))
_VT_TABLE = np.maximum(_VT_TABLE, 0.0)      # (kept writeable: np.interp copies read-only tables)


def fall_speed(d_mm, rho_air: Optional[float] = None):
    """Terminal fall speed (m/s) of raindrops of diameter `d_mm` (vectorized).

    Atlas et al. (1973) above 0.5 mm, Stokes below 0.1 mm, linear bridge between;
    never negative. `rho_air` (kg/m^3, default sea level) applies the Foote & du Toit
    density correction: drops fall faster in thinner air (higher, warmer).
    """
    k = _density_factor(rho_air)
    if isinstance(d_mm, (float, int)) and not isinstance(d_mm, bool):
        return _fall_speed_scalar(float(d_mm)) * k
    d = np.asarray(d_mm, dtype=float)
    if d.ndim == 0:
        return _fall_speed_scalar(max(float(d), 0.0)) * k
    v = np.interp(d, _VT_GRID, _VT_TABLE)
    if k != 1.0:
        v *= k
    return v


def fall_speed_from_height(d_mm, h_m, rho_air: Optional[float] = None):
    """Impact speed (m/s) of a drop released at rest that fell `h_m` metres (drips).

    U(h) = V_T sqrt(1 - exp(-2 g h / V_T^2)) (exponential approach to terminal speed).
    """
    vt = np.asarray(fall_speed(d_mm, rho_air), dtype=float)
    h = np.maximum(np.asarray(h_m, dtype=float), 0.0)
    vt_safe = np.maximum(vt, 1e-6)
    u = vt * np.sqrt(1.0 - np.exp(-2.0 * G * h / (vt_safe * vt_safe)))
    u = np.where(vt > 0.0, u, 0.0)
    return u if u.ndim else float(u)


# ---------------------------------------------------------------- distribution
@lru_cache(maxsize=256)
def _f_mu(mu: float) -> float:
    mu = float(mu)
    return 6.0 / 4.0 ** 4 * math.exp((mu + 4.0) * math.log(4.0 + mu) - float(gammaln(mu + 4.0)))


def number_density(dsd: DSDParams, d_mm) -> np.ndarray:
    """N(D) in m^-3 mm^-1 for diameters `d_mm` (zero outside [d_min, d_max])."""
    d = np.asarray(d_mm, dtype=float)
    if dsd.nw <= 0.0 or dsd.dm_mm <= 0.0:
        return np.zeros_like(d)
    mu = float(dsd.mu)
    inv_dm = 1.0 / float(dsd.dm_mm)
    # N = Nw f(mu) x^mu e^{-(4+mu) x}, x = D/Dm; the exponent cannot overflow for mu in
    # [-3, 20] and x >= 1e-9/Dm, and underflow is 0
    expo = d * (-(4.0 + mu) * inv_dm)
    if mu != 0.0:
        expo += mu * np.log(np.maximum(d, 1e-9) * inv_dm)
    n = np.exp(expo, out=expo) if expo.ndim else np.exp(expo)
    n *= float(dsd.nw) * _f_mu(mu)
    outside = (d < dsd.d_min_mm) | (d > dsd.d_max_mm)
    if n.ndim:
        n[outside] = 0.0
        return n
    return np.float64(0.0) if outside else n


def _trap(y: np.ndarray, x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x)))


def _cumtrap(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    out = np.empty(y.size)
    out[0] = 0.0
    np.cumsum(0.5 * (y[1:] + y[:-1]) * dx, out=out[1:])
    return out


@lru_cache(maxsize=32)
def _grid(d_min: float, d_max: float) -> tuple:
    """(d, dx, ln d, d^3, v(d)) on [d_min, d_max] including every class edge and speed breakpoint."""
    brk = {_D_STOKES_MM, _D_ATLAS_MM}
    for edges in _CLASS_EDGES.values():
        brk.update(edges)
    pts = [b for b in brk if d_min < b < d_max]
    d = np.unique(np.concatenate((np.linspace(d_min, d_max, _TABLE_INTERVALS + 1), np.array(pts, dtype=float))))
    return d, np.diff(d), np.log(d), d ** 3, fall_speed(d)


class _ShapeTable:
    """Cumulative integrals of the unit-Nw DSD shape on the grid, from d_min upwards.

    flux = Int v N dD, rate = Int D^3 v N dD (built at once: every block needs them);
    mass = Int D^3 N dD and count = Int N dD on first use (LWC, D0, concentration).
    """

    __slots__ = ("d", "flux", "rate", "_shape", "_dx", "_d3", "_mass", "_count", "_bands", "_classes")

    def __init__(self, dm: float, mu: float, d_min: float, d_max: float, full: bool = True):
        d, dx, lnd, d3, v = _grid(d_min, d_max)
        expo = lnd - math.log(dm)
        expo *= mu
        expo -= ((4.0 + mu) / dm) * d
        shape = np.exp(expo, out=expo)
        shape *= _f_mu(mu)
        nv = shape * v
        self.d = d
        self.flux = _cumtrap(nv, dx)
        nv *= d3
        self.rate = _cumtrap(nv, dx)
        self._shape, self._dx, self._d3 = shape, dx, d3
        self._mass = self._count = None
        self._bands = {}
        self._classes = {}

    @property
    def mass(self) -> np.ndarray:
        if self._mass is None:
            self._mass = _cumtrap(self._shape * self._d3, self._dx)
        return self._mass

    @property
    def count(self) -> np.ndarray:
        if self._count is None:
            self._count = _cumtrap(self._shape, self._dx)
        return self._count

    def band(self, lo: float, hi: float) -> tuple:
        """(flux in [lo, hi], normalised CDF knots 0..1, diameter knots lo..hi) for sampling."""
        key = (lo, hi)
        b = self._bands.get(key)
        if b is None:
            d, fl = self.d, self.flux
            c_lo = float(np.interp(lo, d, fl))
            c_hi = float(np.interp(hi, d, fl))
            i0 = int(np.searchsorted(d, lo, side="right"))
            i1 = int(np.searchsorted(d, hi, side="left"))
            span = c_hi - c_lo
            if span > 0.0:
                knots_d = np.concatenate(([lo], d[i0:i1], [hi]))
                knots_c = np.concatenate(([0.0], (fl[i0:i1] - c_lo) / span, [1.0]))
                np.clip(knots_c, 0.0, 1.0, out=knots_c)
            else:
                knots_d = np.array([lo, hi])
                knots_c = np.array([0.0, 1.0])
            b = (span, knots_c, knots_d)
            if len(self._bands) < 64:
                self._bands[key] = b
        return b

    def class_unit_flux(self, scheme: str, d_max_mm: float) -> np.ndarray:
        """Unit-Nw flux of the five classes (an exact partition of the grid's flux)."""
        key = (scheme, d_max_mm)
        cf = self._classes.get(key)
        if cf is None:
            bounds = class_bounds(scheme, d_max_mm)
            edges = [bounds[CLASS_NAMES[0]][0]] + [bounds[name][1] for name in CLASS_NAMES]
            cum = np.interp(np.clip(edges, self.d[0], self.d[-1]), self.d, self.flux)
            cf = np.maximum(np.diff(cum), 0.0)
            self._classes[key] = cf
        return cf

    def cum(self, column: np.ndarray, lo: float, hi: float) -> float:
        """Integral of `column` between lo and hi (clipped to the grid)."""
        d = self.d
        lo = min(max(lo, d[0]), d[-1])
        hi = min(max(hi, d[0]), d[-1])
        if hi <= lo:
            return 0.0
        return float(np.interp(hi, d, column) - np.interp(lo, d, column))


@lru_cache(maxsize=64)
def _table(dm: float, mu: float, d_min: float, d_max: float) -> _ShapeTable:
    return _ShapeTable(dm, mu, d_min, d_max)


def _shape_of(dsd: DSDParams) -> Optional[_ShapeTable]:
    if dsd.nw <= 0.0 or dsd.dm_mm <= 0.0 or not float(dsd.d_max_mm) > float(dsd.d_min_mm):
        return None
    return _table(float(dsd.dm_mm), float(dsd.mu), max(float(dsd.d_min_mm), 1e-6), float(dsd.d_max_mm))


def _moment_integral(dsd: DSDParams, lo: float, hi: float, power: float, with_speed: bool) -> float:
    tab = _shape_of(dsd)
    if tab is None:
        return 0.0
    name = {(0.0, True): "flux", (3.0, True): "rate",
            (3.0, False): "mass", (0.0, False): "count"}.get((float(power), bool(with_speed)))
    col = getattr(tab, name) if name is not None else None
    if col is not None:
        return dsd.nw * tab.cum(col, float(lo), float(hi))
    lo = max(float(lo), float(dsd.d_min_mm))
    hi = min(float(hi), float(dsd.d_max_mm))
    if hi <= lo:
        return 0.0
    d = np.linspace(lo, hi, _TABLE_INTERVALS + 1)
    y = number_density(dsd, d) * d ** power
    if with_speed:
        y = y * fall_speed(d)
    return _trap(y, d)


def rain_rate(dsd: DSDParams) -> float:
    """Rain rate (mm/h) carried by the distribution: 6 pi 1e-4 Int D^3 v N dD."""
    tab = _shape_of(dsd)
    return 0.0 if tab is None else _RATE_CONST * dsd.nw * float(tab.rate[-1])


def total_flux(dsd: DSDParams, d_lo_mm: float = 0.0, d_hi_mm: float = math.inf) -> float:
    """Impacts per m^2 per s for drops between d_lo_mm and d_hi_mm."""
    tab = _shape_of(dsd)
    if tab is None:
        return 0.0
    return dsd.nw * tab.cum(tab.flux, float(d_lo_mm), float(min(d_hi_mm, 1e9)))


def class_flux(dsd: DSDParams, scheme: str = "medwin") -> dict[str, float]:
    """Impacts / m^2 / s on a horizontal surface for each acoustic drop class (they sum to total_flux)."""
    tab = _shape_of(dsd)
    if tab is None:
        class_bounds(scheme, dsd.d_max_mm)          # still validate the scheme name
        return {name: 0.0 for name in CLASS_NAMES}
    cf = tab.class_unit_flux(str(scheme).lower(), float(dsd.d_max_mm)) * float(dsd.nw)
    return dict(zip(CLASS_NAMES, cf.tolist(), strict=True))


def liquid_water_content(dsd: DSDParams) -> float:
    """Liquid water content in g/m^3: (pi/6) 1e-3 Int D^3 N dD."""
    tab = _shape_of(dsd)
    return 0.0 if tab is None else math.pi / 6.0 * 1e-3 * dsd.nw * float(tab.mass[-1])


def concentration(dsd: DSDParams) -> float:
    """Total number concentration (drops per m^3) inside the cutoffs."""
    tab = _shape_of(dsd)
    return 0.0 if tab is None else dsd.nw * float(tab.count[-1])


def median_volume_diameter(dsd: DSDParams) -> float:
    """D0 (mm): half the liquid water is in drops smaller than D0."""
    tab = _shape_of(dsd)
    if tab is None or tab.mass[-1] <= 0.0:
        return 0.0
    return float(np.interp(0.5 * tab.mass[-1], tab.mass, tab.d))


# ---------------------------------------------------------------- regimes
# drops -> (a, b, cap mm, cap sharpness p, mu) with Dm = a R^b, saturating at the cap.
# 'fine' / 'fat' sit on the Bringi (2003) maritime / continental convective cluster
# centres at ~30 mm/h (Dm 1.62 / 2.40 mm, log10 Nw 4.25 / 3.48); 'normal' is stratiform /
# mid-latitude (D0 ~1.7-2.0 mm in heavy rain). Ordering fine < normal < fat holds at every rate.
_REGIMES = {
    "fine": (0.85, 0.206, 1.85, 8.0, 3.0),     # maritime / tropical convective: many small drops
    "normal": (1.00, 0.20, 2.25, 4.0, 3.0),    # stratiform / mid-latitude
    "fat": (1.39, 0.17, 2.90, 8.0, 3.0),       # continental convective: few big drops
}
_DM_FLOOR_MM = 0.35


def _soft_cap(x: float, cap: float, p: float = 4.0) -> float:
    return x / (1.0 + (x / cap) ** p) ** (1.0 / p)


def _unit_rate(dm: float, mu: float, d_min: float, d_max: float, cached: bool = True) -> float:
    """rain_rate of the shape with Nw = 1."""
    if cached:
        return _RATE_CONST * float(_table(dm, mu, d_min, d_max).rate[-1])
    return _RATE_CONST * float(_ShapeTable(dm, mu, d_min, d_max).rate[-1])


def _solve_nw(dm: float, mu: float, d_min: float, d_max: float, rate: float) -> float:
    r1 = _unit_rate(float(dm), float(mu), max(float(d_min), 1e-6), float(d_max))
    return rate / r1 if r1 > 0.0 else 0.0


def _solve_dm(nw: float, mu: float, d_min: float, d_max: float, rate: float) -> float:
    """Dm such that rain_rate == rate at fixed Nw (rate grows monotonically with Dm)."""
    lo, hi = math.log(0.1), math.log(8.0)
    d_min = max(float(d_min), 1e-6)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        r = nw * _unit_rate(math.exp(mid), float(mu), d_min, float(d_max), cached=False)
        if r < rate:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def dsd_from_scene(precip, rate_mm_h: float) -> DSDParams:
    """Resolve a `Precipitation` block and an instantaneous rate into a DSD.

    `drops` fine/normal/fat moves Dm along the maritime / stratiform / continental
    clusters; Dm grows with rate but saturates (<= ~2.8 mm). Nw is then solved so
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
        a, b, cap, p, mu = _REGIMES.get(drops, _REGIMES["normal"])
        dm = max(_DM_FLOOR_MM, _soft_cap(a * max(rate, 1e-3) ** b, cap, p))

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

    Impacts on a surface are flux-weighted, so the sampling density is N(D) v(D):
    inverse CDF on the cached cumulative-flux table (one `rng.random(n)` draw).
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
    u = rng.random(n)
    if dsd.dm_mm <= 0.0:
        return lo + (hi - lo) * u
    tab = _table(float(dsd.dm_mm), float(dsd.mu), max(float(dsd.d_min_mm), 1e-6), float(dsd.d_max_mm))
    span, knots_c, knots_d = tab.band(lo, hi)
    if not span > 0.0:
        return lo + (hi - lo) * u
    # the knots run exactly from (0, lo) to (1, hi): the result is inside [lo, hi] by construction
    return np.interp(u, knots_c, knots_d)


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
