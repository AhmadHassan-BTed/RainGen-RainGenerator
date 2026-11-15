"""Rain, drip, gutter, hail and snow sound generators.

Physics (MASTER_PLAN 7.1-7.5, 7.9, 9; Appendix A):
- Impacts arrive as Cox processes per acoustic drop class (Medwin et al. 1992 / Nystuen):
  rate = class_flux * area * exposure * wind-angle factor. The wind-angle factor is the rain
  flux through the patch normal for drops falling at V_T(D) (Atlas 1973) in the horizontal
  wind: n_z - (n_h . u) / V_T, so wind-driven rain wets windward walls, small drops most.
- Water surfaces: impact click (amplitude ~ D U^3, the dipole of dF/dt of the Gordillo force)
  plus a probabilistic Minnaert bubble with van den Doel chirp: P ~ 0.95 for the small class
  (0.8-1.1 mm, 13-25 kHz; Pumphrey & Elmore 1990), ~0 for mid, rising to ~0.65 near 4 mm for
  large drops (2-10 kHz, radii ~ r^-alpha, alpha 2.5). In-air bubble level is ~40 dB below
  hydrophone figures (Phillips, Agarwal & Jordan 2018). Small-drop bubbles vanish with wind
  8-10 m/s (Nystuen) and all bubbles with surfactant (water_quality).
- Solids: force pulses (Gordillo, Sun & Cheng 2018) through the material's modal bank and
  radiation band; prompt-splash micro-impacts (2-10 weak impacts over 5-50 ms) on dry hard
  surfaces; a water film cushions the pulse and adds crown-splash hiss; puddles turn the
  impacts into water impacts (Yu & Hopkins 2018: dry click -> film splash -> puddle plinks).
- Level of detail (Zhang, Raghuvanshi, Snyder & Marschner 2019): the nearest / loudest patches
  get per-event modal synthesis; far rings get pre-baked grains at capped, energy-preserving
  densities; beyond the rings a diffuse bed is synthesized directly in the SH domain.
- Drips (integrate-and-fire, raingen.physics.wetness): a plink inside the regular-entrainment
  band 41.3 Fr^0.179 <= We <= 48.3 Fr^0.247 (Oguz & Prosperetti 1990), a splat otherwise; fed
  by rain, routed roof flow, canopy release or stored water, so they outlast the rain.
- Gutters/downspouts: bubble cloud + turbulent rush exciting the open pipe's resonances
  f_k = k c / 2(L + 0.6 d), glugs in slug flow, an overflow sheet above the design capacity.
  No primary acoustic literature: a hypothesis model.
- Interior panels struck on their far side (a roof above a ceiling, a deck) reach the room through
  the field-incidence mass law TL = 20 log10(m'' f) - 47 dB (Cremer & Heckl).
- Hail / ice pellets / graupel: Hertz contact with an ice-crushing limit, NOAA size and
  fall-speed bands, bounces with restitution. Snow: flake impulses m V spread over a crumple
  time (near silent), plus blowing-snow saltation hiss in cold wind.

Units: every signal is sound pressure in pascals at the listener (digital 1.0 = 1 Pa = 94 dB SPL).
All randomness comes from RngFactory.get(layer, block) with layers named f'rain/{patch.id}/{cls}' etc.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from raingen.core.constants import G, N_AMBI, RHO_WATER, SAMPLE_RATE, SIGMA_WATER, speed_of_sound
from raingen.core.rng import RngFactory
from raingen.core.types import BlockContext, Emission, EventMarker, SourcePatch, WeatherState, bearing_to_azimuth
from raingen.physics import dsd as _dsd
from raingen.physics.atmosphere import AirAbsorptionFilter, air_absorption_sos, spreading_gain
from raingen.physics.wetness import DripNode

from .events import RateModulator, cox_offsets, thinned_offsets
from .kernels import (D_REF_MM, V_REF, KernelBank, ModalBank, OverlapAdd, StatefulSOS, bubble_damping,
                      fade_tail, force_peak, hertz_contact, impact_pulse, pulse_shape, radiation_sos,
                      render_binned)
from .materials import MATERIALS, Material, get_material

__all__ = [
    "RainPatchGenerator", "RainBedGenerator", "DripGenerator", "GutterGenerator", "HailGenerator",
    "SnowGenerator", "make_rain_generators", "regular_entrainment", "db_to_pa",
]

CLASSES = tuple(_dsd.CLASS_NAMES)
# representative diameters (mm) for class-level quantities (flux angle, grains, bed energies)
REP_D = {"minuscule": 0.6, "small": 0.95, "mid": 1.6, "large": 2.8, "very_large": 4.2}

P_REF = 20e-6
F_REF = float(force_peak(D_REF_MM, V_REF))                 # 0.14 N
TAU_REF = D_REF_MM * 1e-3 / V_REF
J_REF = 0.511 * TAU_REF * F_REF                            # impulse of the reference drop pulse

BUBBLE_REF_DB = 52.0      # peak dB SPL @1 m in air: a = 1 mm (3.3 kHz), eps = 0.05
SPLAT_REF_DB = 54.0       # peak dB SPL @1 m: broadband splat of a 4 mm drop at 8.7 m/s
FILM_HISS_REL_DB = -6.0   # crown-splash hiss on a wet film, relative to the dry click anchor


def db_to_pa(db):
    """dB SPL -> pascals (vectorized)."""
    return P_REF * 10.0 ** (np.asarray(db, dtype=float) / 20.0)


BUBBLE_PA_PER_M = float(db_to_pa(BUBBLE_REF_DB)) / (0.05 * 1e-3)   # Pa per (eps * radius[m])
SPLAT_PA = float(db_to_pa(SPLAT_REF_DB))

RAIN_KINDS = ("surface", "water", "membrane", "vegetation", "vehicle_body", "gutter")
HAIL_TYPES = ("hail", "sleet", "graupel")
SNOW_TYPES = ("snow",)
# share of the drop flux that arrives as liquid drops (the rest is handled by Hail/SnowGenerator)
_RAIN_GATE = {"none": 0.0, "snow": 0.0, "graupel": 0.0, "sleet": 0.5, "hail": 0.5}
# liquid share feeding drips / runoff (mirrors raingen.physics.world)
_LIQUID = {"none": 0.0, "snow": 0.0, "graupel": 0.3, "sleet": 0.5, "hail": 0.5}
LOD_K = {"mobile": 6, "game": 12, "studio": 24, "cinematic": 48}
FULL_MAX_DIST_M = 12.0
SLAB_RHO_KG_M3 = 2300.0          # concrete, for decks / slabs between the impact and the listener

_GRAIN_RNG = RngFactory(0x67726169)   # 'grai'
_CAL_CACHE: dict = {}
_GRAIN_CACHE: dict = {}
_ICE_VOL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# small helpers

def _smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    out = t * t * (3.0 - 2.0 * t)
    return float(out) if out.ndim == 0 else out


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])


def _wind_vector(w: WeatherState) -> tuple[float, float]:
    """Horizontal wind velocity (listener frame, blowing TOWARD); wind_azimuth is where it comes FROM."""
    u = max(float(w.wind_listener_mps), 0.0)
    a = float(w.wind_azimuth)
    return -u * math.cos(a), -u * math.sin(a)


def _patch_geometry(patch: SourcePatch) -> tuple[float, float]:
    """Energy-equivalent distance of a patch and its angular spread (0..1).

    Ground ring sectors (params r_inner/r_outer, ears at height h): the mean 1/r^2 over the
    annulus is ln((r2^2 + h^2) / (r1^2 + h^2)) / (r2^2 - r1^2); the spread is the larger of the
    sector's azimuthal extent (foreshortened by the polar angle) and its polar extent, over pi.
    Other patches are treated as a disk of radius R at distance d: mean 1/r^2 = ln(1 + R^2/d^2) / R^2.
    """
    p = patch.params or {}
    if p.get("ground") and "r_inner" in p and "r_outer" in p:
        r1 = max(float(p["r_inner"]), 0.0)
        r2 = max(float(p["r_outer"]), r1 + 1e-3)
        h = max(-float(patch.position[2]), 0.05)
        inv = math.log((r2 * r2 + h * h) / (r1 * r1 + h * h)) / (r2 * r2 - r1 * r1)
        r_eq = 1.0 / math.sqrt(max(inv, 1e-12))
        th1, th2 = math.atan2(r1, h), math.atan2(r2, h)
        dphi = float(p.get("az_width", math.pi / 4.0)) * math.sin(0.5 * (th1 + th2))
        spread = float(np.clip(max(dphi, th2 - th1) / math.pi, 0.0, 1.0))
        return max(r_eq, 0.1), spread
    d = max(patch.distance, 0.1)
    rad = math.sqrt(max(patch.area_m2, 1e-4) / math.pi)
    q = (rad / d) ** 2
    r_eq = d * math.sqrt(q / math.log1p(q)) if q > 1e-6 else d
    spread = float(np.clip(2.0 * math.atan(rad / d) / math.pi, 0.0, 1.0))
    return max(r_eq, 0.1), spread


def _interior_transfer(patch: SourcePatch, sr: int):
    """(gain, sos or None): what reaches the room from an interior patch whose struck face is not
    the face radiating into the listener's space.

    Ceilings / decks: field-incidence mass law TL = 20 log10(m'' f) - 47 dB (Cremer & Heckl),
    realised as its value at 100 Hz plus a one-pole low-pass at 100 Hz (the 6 dB/octave mass-law
    slope above it). Double / triple glazing: the struck outer pane reaches the room through the
    cavity and the inner pane - a broadband loss and a high-frequency roll-off (provisional).
    """
    p = patch.params or {}
    if not patch.interior:
        return 1.0, None
    mass = None
    if p.get("through_ceiling"):
        mass = float(p.get("ceiling_mass_kg_m2", 25.0))
    elif p.get("thickness_m"):
        mass = SLAB_RHO_KG_M3 * float(p["thickness_m"])
    if mass is not None and mass > 0.0:
        f0 = 100.0
        tl0 = max(20.0 * math.log10(mass * f0) - 47.0, 0.0)
        return 10.0 ** (-tl0 / 20.0), signal.butter(1, f0, "lowpass", fs=sr, output="sos")
    glazing = str(p.get("glazing", "")).lower()
    if glazing in ("double", "triple"):
        gain, fc = (0.4, 3500.0) if glazing == "double" else (0.25, 2500.0)
        return gain, signal.butter(1, fc, "lowpass", fs=sr, output="sos")
    return 1.0, None


def _stem_for(patch: SourcePatch) -> str:
    if patch.interior:
        return "interior"
    return "rain_near" if patch.distance < 3.0 else "rain_far"


def _fall_speed(d) -> np.ndarray:
    return np.asarray(_dsd.fall_speed(np.asarray(d, dtype=float)), dtype=float)


def _loguniform(rng, lo, hi, size):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return lo * (hi / lo) ** rng.random(size)


def _power_law(rng, r1, r2, alpha, size):
    """Radii distributed as r^-alpha on [r1, r2] (inverse CDF)."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    e = 1.0 - alpha
    u = rng.random(size)
    return (r1 ** e + u * (r2 ** e - r1 ** e)) ** (1.0 / e)


def regular_entrainment(d_mm, u_mps, tolerance: float = 0.1):
    """True where (D, U) lies in the regular-entrainment band 41.3 Fr^0.179 <= We <= 48.3 Fr^0.247.

    We = rho D U^2 / sigma, Fr = U^2 / (g D) (Oguz & Prosperetti 1990; Pumphrey & Elmore 1990).
    """
    d = np.maximum(np.asarray(d_mm, dtype=float), 0.05) * 1e-3
    u = np.maximum(np.asarray(u_mps, dtype=float), 0.01)
    we = RHO_WATER * d * u * u / SIGMA_WATER
    fr = u * u / (G * d)
    lo = 41.3 * fr ** 0.179 * (1.0 - tolerance)
    hi = 48.3 * fr ** 0.247 * (1.0 + tolerance)
    return (we >= lo) & (we <= hi)


# ---------------------------------------------------------------------------
# calibration and pre-baked grains

def _solid_calibration(mat: Material, sr: int) -> float:
    """Pa per (N through bank + radiation) so a 2 mm / 6.5 m/s drop peaks at mat.level_db @1 m."""
    key = (mat.name, int(sr))
    c = _CAL_CACHE.get(key)
    if c is not None:
        return c
    pulse = impact_pulse(D_REF_MM, V_REF, sr)
    bank = ModalBank(mat, sr)
    if bank.n_modes:
        length = int(0.08 * sr)
        y = np.convolve(pulse, bank.impulse_response(length))[:length]
    else:
        y = pulse.copy()
    rad = radiation_sos(mat, sr)
    if rad.shape[0]:
        y = signal.sosfilt(rad, y)
    peak = float(np.max(np.abs(y)))
    c = float(db_to_pa(mat.level_db)) / peak if peak > 0 else 0.0
    _CAL_CACHE[key] = c
    return c


def _grain_kernels(mat: Material, sr: int) -> list[np.ndarray]:
    """Pre-baked impact grains (unit F_max, already in Pa @1 m): 2 variants x 5 classes."""
    key = (mat.name, int(sr))
    ks = _GRAIN_CACHE.get(key)
    if ks is not None:
        return ks
    cal = _solid_calibration(mat, sr)
    rad = radiation_sos(mat, sr)
    length = int(np.clip(0.8 * mat.max_t60, 0.008, 0.12) * sr)
    rng = _GRAIN_RNG.get(f"grains/{mat.name}", 0)
    n_modes = len(mat.modes)
    irs = []
    for v in range(2):
        det = 1.0 if v == 0 or n_modes == 0 else np.exp(rng.normal(0.0, 0.1, n_modes))
        bank = ModalBank(mat, sr, detune=det)
        if bank.n_modes:
            irs.append(bank.impulse_response(length))
        else:
            ir = np.zeros(length)
            ir[0] = 1.0
            irs.append(ir)
    out = []
    for v in range(2):
        for cls in CLASSES:
            dd = REP_D[cls]
            vt = max(float(_fall_speed(dd)), 0.3)
            k = np.convolve(pulse_shape(dd * 1e-3 / vt, sr), irs[v])
            if rad.shape[0]:
                k = signal.sosfilt(rad, k)
            out.append(fade_tail(k, 0.3) * cal)
    _GRAIN_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# water impacts (rain on water/puddles, drip splats, hail into water)

_EPS_LO = np.array([0.01, 0.02, 0.005, 0.03, 0.04])
_EPS_HI = np.array([0.02, 0.07, 0.02, 0.10, 0.12])
_DELAY_LO_MS = np.array([1.0, 0.5, 2.0, 6.0, 10.0])
_DELAY_HI_MS = np.array([2.0, 2.5, 8.0, 20.0, 30.0])


def _entrainment_probability(d, ci, wind_mps: float) -> np.ndarray:
    p = np.zeros(d.shape)
    wsup = 1.0 - _smoothstep(wind_mps, 7.5, 10.0)          # small-drop ring dies at 8-10 m/s
    p[ci == 1] = 0.95 * wsup
    p[ci == 2] = 0.03
    big = ci >= 3
    if np.any(big):
        p[big] = 0.05 + 0.60 * _smoothstep(d[big], 2.2, 4.0)
    return p


def _bubble_radius(rng, d, ci) -> np.ndarray:
    r = np.full(d.shape, 3e-4)
    m = ci == 1
    if np.any(m):   # regular entrainment: ~22 kHz at 0.8 mm falling to ~13 kHz at 1.1 mm
        f = np.clip(22e3 - (d[m] - 0.8) / 0.3 * 9e3, 12e3, 24e3) * np.exp(rng.normal(0.0, 0.05, int(m.sum())))
        r[m] = 3.28 / f
    m = (ci == 0) | (ci == 2)
    if np.any(m):
        r[m] = _loguniform(rng, 3e-4, 1.2e-3, int(m.sum()))
    m = ci >= 3
    if np.any(m):   # large: 2-10 kHz band, lower for bigger drops; r^-2.5 within it
        f_hi = 10e3 * (2.2 / np.maximum(d[m], 2.2)) ** 0.8
        r1 = 3.28 / f_hi
        r[m] = _power_law(rng, r1, 4.0 * r1, 2.5, int(m.sum()))
    return r


def _bubble_kernel_fn(kb: KernelBank):
    return lambda k: kb.bubble_shape(k // 512, (k // 8) % 64, k % 8)


def _bubble_keys(kb: KernelBank, radius, xi, damp_bin: int):
    return (kb.radius_bin(radius) * 64 + kb.xi_bin(xi)) * 8 + int(np.clip(damp_bin, 0, 7))


def _water_impacts(kb: KernelBank, rng, off, d, u, ci, sc, ola: OverlapAdd, *, click_pa: float,
                   depth_mm: float = 1000.0, wind_mps: float = 0.0, water_quality: float = 1.0,
                   sr: int = SAMPLE_RATE, p_scale: float = 1.0):
    """Click + probabilistic bubble (+ splat for big drops) for impacts on a water surface."""
    n_ev = int(off.size)
    if n_ev == 0:
        return
    off = np.asarray(off, dtype=np.int64)
    sc = np.broadcast_to(np.asarray(sc, dtype=float), off.shape)
    ci = np.asarray(ci, dtype=np.int64)
    # impact click ~ D U^3 (U^3 is inside the click kernel)
    amp = click_pa * (d / D_REF_MM) * sc * np.exp(rng.normal(0.0, 0.2, n_ev))
    amp = np.where(ci == 0, 0.3 * amp, amp)
    render_binned(off, amp, kb.click_bin(u), kb.click_shape, ola)

    # entrained bubbles
    p = _entrainment_probability(d, ci, wind_mps) * float(np.clip(water_quality, 0.0, 1.0)) * p_scale
    hit = rng.random(n_ev) < p
    idx = np.nonzero(hit)[0]
    vl = np.nonzero(hit & (ci == 4))[0]
    if vl.size:
        extra = rng.poisson(0.8, vl.size)
        idx = np.concatenate([idx, np.repeat(vl, extra)])
    fd = _smoothstep(depth_mm, 1.5, 8.0)                    # shallow puddles pinch bubbles off
    if idx.size and fd > 0.0:
        m = idx.size
        cls = ci[idx]
        radius = _bubble_radius(rng, d[idx], cls)
        xi = np.clip(0.1 * np.exp(rng.normal(0.0, 0.25, m)), 0.03, 0.3)
        eps = _loguniform(rng, _EPS_LO[cls], _EPS_HI[cls], m)
        delay = ((_DELAY_LO_MS[cls] + rng.random(m) * (_DELAY_HI_MS[cls] - _DELAY_LO_MS[cls]))
                 * 1e-3 * sr).astype(np.int64)
        damp_bin = int(round(0.02 * (1.0 - _smoothstep(depth_mm, 3.0, 15.0)) / 0.01))
        render_binned(off[idx] + delay, BUBBLE_PA_PER_M * eps * sc[idx] * fd,
                      _bubble_keys(kb, radius, xi, damp_bin), _bubble_kernel_fn(kb), ola)

    # broadband splat of very large drops
    sp = np.nonzero((ci == 4) | ((ci == 3) & (d >= 3.0)))[0]
    if sp.size:
        dd = d[sp]
        uu = u[sp]
        amp_s = (SPLAT_PA * (dd / 4.0) ** 2 * (uu / 8.7) ** 3 * sc[sp]
                 * np.where(ci[sp] == 4, 1.0, 0.4) * np.exp(rng.normal(0.0, 0.25, sp.size)))
        dur = np.clip(0.003 * dd / 3.0, 0.002, 0.012)
        keys = kb.dur_bin(dur) * 16 + rng.integers(0, kb.SPLASH_VARIANTS, sp.size)
        render_binned(off[sp], amp_s, keys, lambda k: kb.splash(k // 16, k % 16, True), ola)


# ---------------------------------------------------------------------------
# solid voice: force OLA -> modal bank(s) -> radiation -> Pa

class _SolidVoice:
    """Force pulses (per-variant impulse trains) driving parallel modal banks with carried state."""

    def __init__(self, mat: Material, sr: int, kb: KernelBank, rngf: RngFactory | None = None,
                 layer: str = "", n_var: int = 1, detune_spread: float = 0.08, damp_spread: float = 0.15):
        self.mat = mat
        self.sr = int(sr)
        self.kb = kb
        self.cal = _solid_calibration(mat, sr)
        rng = rngf.get(layer + "/variants", 0) if (rngf is not None and n_var > 1) else None
        n_modes = len(mat.modes)
        self.banks = []
        for v in range(max(1, int(n_var))):
            if v == 0 or rng is None or n_modes == 0:
                det, dmp = 1.0, 1.0
            else:
                det = np.exp(rng.normal(0.0, detune_spread, n_modes))
                dmp = np.exp(rng.normal(0.0, damp_spread, n_modes))
            self.banks.append(ModalBank(mat, sr, detune=det, damping=dmp))
        self.olas = [OverlapAdd() for _ in self.banks]
        self.rad = StatefulSOS(radiation_sos(mat, sr))
        self.has_modes = self.banks[0].n_modes > 0
        self._rad_hot = False

    @property
    def n_var(self) -> int:
        return len(self.banks)

    @property
    def active(self) -> bool:
        if self._rad_hot or any(o.active for o in self.olas):
            return True
        return any(not bk.is_silent(1e-11) for bk in self.banks)

    def add_force(self, off, force, tau, family: str = "pulse", variant=None):
        off = np.asarray(off, dtype=np.int64)
        if off.size == 0:
            return
        keys = self.kb.tau_bin(tau)
        fn = self.kb.pulse if family == "pulse" else self.kb.hertz
        if self.n_var == 1 or variant is None:
            render_binned(off, force, keys, fn, self.olas[0])
            return
        force = np.broadcast_to(np.asarray(force, dtype=float), off.shape)
        variant = np.asarray(variant)
        for v in range(self.n_var):
            m = variant == v
            if np.any(m):
                render_binned(off[m], force[m], keys[m], fn, self.olas[v])

    def render(self, n: int) -> np.ndarray:
        y = np.zeros(n)
        fed = False
        for bank, ola in zip(self.banks, self.olas):
            f = ola.pop(n) if ola.active else None
            if not self.has_modes:
                if f is not None:
                    y += f
                    fed = True
                continue
            if f is not None and np.any(f):
                y += bank.process(f)
                fed = True
            elif not bank.is_silent(1e-11):
                y += bank.process(np.zeros(n))
                fed = True
        if fed:
            y = self.rad.process(y) * self.cal
            self._rad_hot = True
        elif self._rad_hot:
            y = self.rad.process(y) * self.cal
            self._rad_hot = False
        return y


# ---------------------------------------------------------------------------
# rain on one surface patch

class RainPatchGenerator:
    """Rain on one SourcePatch: per-class Cox event streams -> water / solid / granular synthesis."""

    CAPS_FULL = (600, 1500, 1500, 800, 400)        # max events per block per class
    CAPS_GRANULAR = (40, 100, 100, 60, 30)

    def __init__(self, scene, patch: SourcePatch, rngf: RngFactory, lod: str = "full",
                 kernels: KernelBank | None = None, sr: int = SAMPLE_RATE):
        self.name = f"rain/{patch.id}"
        self.patch = patch
        self.rngf = rngf
        self.lod = lod if lod in ("full", "granular") else "full"
        self.mat = get_material(patch.material)
        self.kind = "water" if (patch.kind == "water" or self.mat.kind == "water") else self.mat.kind
        pr = scene.weather.precipitation
        scheme = str(getattr(pr, "class_boundaries", "medwin") or "medwin").lower()
        self.scheme = scheme if scheme in ("medwin", "nystuen") else "medwin"
        self.water_quality = float(np.clip(pr.water_quality, 0.0, 1.0))
        self.normal = _unit(patch.normal)
        self.r_eq, self.spread = _patch_geometry(patch)
        self.stem = _stem_for(patch)
        # one panel rings with one set of modes; a ground sector is many independent pieces
        self.single_panel = (patch.kind in ("membrane", "vehicle_body", "gutter") or patch.interior
                             or (patch.area_m2 <= 2.5 and not (patch.params or {}).get("ground")))
        self.caps = self.CAPS_FULL if self.lod == "full" else self.CAPS_GRANULAR
        self._kb = kernels
        self._bounds_key = None
        self._bounds: dict = {}
        self.sr = None
        self._setup(int(sr))

    def _setup(self, sr: int):
        self.sr = int(sr)
        self.kb = self._kb if (self._kb is not None and self._kb.sr == self.sr) else KernelBank(self.sr)
        self.direct = OverlapAdd()
        self.voice = None
        self.grains = None
        pid = self.patch.id
        if self.kind != "water":
            if self.lod == "full":
                nv = 1 if self.single_panel else (4 if self.kind == "vegetation" else 3)
                spread = 0.25 if self.kind == "vegetation" else (0.03 if self.single_panel else 0.08)
                self.voice = _SolidVoice(self.mat, self.sr, self.kb, self.rngf, f"rain/{pid}", nv, spread)
            else:
                self.grains = _grain_kernels(self.mat, self.sr)
        self.air = AirAbsorptionFilter(self.sr)
        self.mod = RateModulator(self.rngf, f"rain/{pid}/cox", sigma=0.22, sr=self.sr)
        click_db = self.mat.level_db if self.kind == "water" else MATERIALS["puddle"].level_db
        self.click_pa = float(db_to_pa(click_db))
        self.xfer_gain, xs = _interior_transfer(self.patch, self.sr)
        self.xfer = StatefulSOS(xs) if xs is not None else None

    # -- geometry / weather helpers ------------------------------------------
    def _class_bounds(self, dsd) -> dict:
        key = round(float(dsd.d_max_mm), 3)
        if key != self._bounds_key:
            self._bounds = _dsd.class_bounds(self.scheme, d_max_mm=float(dsd.d_max_mm))
            self._bounds_key = key
        return self._bounds

    def _flux_factor(self, vt: float, uh) -> float:
        nx, ny, nz = self.normal
        return max(0.0, nz - (nx * uh[0] + ny * uh[1]) / max(vt, 0.3))

    def _puddle_cover(self, w: WeatherState) -> tuple[float, float]:
        if self.kind in ("vegetation", "membrane", "snow"):
            return 0.0, 0.0
        depth = float((w.puddle_depth_mm or {}).get(self.mat.name, 0.0) or 0.0)
        if depth <= 0.2:
            return 0.0, 0.0
        cover = min(1.0, depth / 8.0) ** 0.8
        if self.kind == "porous":
            cover *= 0.5
        return cover, max(depth, 2.0)

    def _water_depth(self, w: WeatherState) -> float:
        if self.mat.name == "water" or self.patch.params.get("deep"):
            return 1000.0
        d = (w.puddle_depth_mm or {}).get(self.mat.name)
        if d is None:
            d = self.patch.params.get("depth_mm", 12.0)
        return max(float(d), 0.5)

    # -- events --------------------------------------------------------------
    def _events(self, ctx: BlockContext):
        w = ctx.weather
        gate = _RAIN_GATE.get(w.precip_type, 1.0)
        if gate <= 0.0 or not w.class_flux or self.patch.exposure <= 0.0 or w.dsd is None:
            return None
        n, sr, b = ctx.n, ctx.sr, ctx.index
        uh = _wind_vector(w)
        curve = self.mod.curve(b, n)
        base = self.patch.area_m2 * max(self.patch.exposure, 0.0) * gate
        bounds = self._class_bounds(w.dsd)
        nx, ny, nz = self.normal
        pieces = []
        for i, cls in enumerate(CLASSES):
            flux = float(w.class_flux.get(cls, 0.0) or 0.0)
            if flux <= 0.0 or cls not in bounds:
                continue
            lo, hi = bounds[cls]
            lo = max(float(lo), float(w.dsd.d_min_mm))
            hi = min(float(hi), float(w.dsd.d_max_mm))
            if hi <= lo:
                continue
            vt_c = float(_fall_speed(math.sqrt(lo * hi) if lo > 0 else 0.5 * hi))
            rate = flux * base * self._flux_factor(vt_c, uh)
            if rate <= 0.0:
                continue
            rng = self.rngf.get(f"rain/{self.patch.id}/{cls}", b)
            off, scale = thinned_offsets(rng, rate, n, sr, self.caps[i], curve)
            if off.size == 0:
                continue
            d = np.asarray(_dsd.sample_diameters(rng, w.dsd, lo, hi, off.size), dtype=float).ravel()[: off.size]
            if d.size < off.size:
                d = np.resize(d, off.size) if d.size else np.full(off.size, math.sqrt(lo * hi))
            vt = _fall_speed(d)
            un = np.maximum(vt * nz - (nx * uh[0] + ny * uh[1]), 0.2 * vt)
            pieces.append((off, d, un, np.full(off.size, i, dtype=np.int64), np.full(off.size, scale)))
        if not pieces:
            return None
        return tuple(np.concatenate([p[k] for p in pieces]) for k in range(5))

    # -- synthesis -----------------------------------------------------------
    def _synthesize(self, ctx: BlockContext, ev):
        off, d, u, ci, sc = ev
        w = ctx.weather
        sr = ctx.sr
        rng = self.rngf.get(f"rain/{self.patch.id}/surface", ctx.index)
        wind10 = float(w.wind_u10_mps) * max(float(w.gust), 0.0)
        if self.kind == "water":
            _water_impacts(self.kb, rng, off, d, u, ci, sc, self.direct, click_pa=self.click_pa,
                           depth_mm=self._water_depth(w), wind_mps=wind10,
                           water_quality=self.water_quality, sr=sr)
            return

        glazed = w.precip_type == "freezing_rain" and w.temperature_c <= 0.5
        cover, depth = (0.0, 0.0) if glazed else self._puddle_cover(w)
        if cover > 0.0:
            wm = rng.random(off.size) < cover
            if np.any(wm):
                _water_impacts(self.kb, rng, off[wm], d[wm], u[wm], ci[wm], sc[wm], self.direct,
                               click_pa=self.click_pa, depth_mm=depth, wind_mps=wind10,
                               water_quality=self.water_quality, sr=sr)
                keep = ~wm
                off, d, u, ci, sc = off[keep], d[keep], u[keep], ci[keep], sc[keep]
        n_ev = off.size
        if n_ev == 0:
            return

        wet = 0.0 if glazed else float(np.clip((w.wetness or {}).get(self.mat.name, 0.0), 0.0, 1.0))
        f_phys = force_peak(d, u)
        tau = d * 1e-3 / np.maximum(u, 0.05) * (1.0 + 0.6 * wet) * (0.7 if glazed else 1.0)
        force = f_phys * sc * np.exp(rng.normal(0.0, 0.25, n_ev)) * (1.0 - 0.35 * wet)
        if self.kind == "snow":          # rain onto a snow cover: soft, swallowed
            force = force * 0.2
            tau = tau * 4.0

        if self.voice is not None:
            var = rng.integers(0, self.voice.n_var, n_ev) if self.voice.n_var > 1 else None
            self.voice.add_force(off, force, tau, "pulse", var)
            # prompt-splash micro-impacts on dry hard surfaces
            if self.mat.splash > 0.0 and self.mat.hardness >= 0.45 and wet < 0.95 and self.kind != "snow":
                p = self.mat.splash * (1.0 - wet) ** 2 * _smoothstep(d, 0.6, 1.6)
                sel = np.nonzero(rng.random(n_ev) < p)[0]
                if sel.size:
                    rep = np.repeat(sel, rng.integers(2, 11, sel.size))
                    m = rep.size
                    delay = (np.exp(rng.uniform(math.log(0.005), math.log(0.05), m)) * sr).astype(np.int64)
                    amp = force[rep] * np.exp(rng.uniform(math.log(0.005), math.log(0.03), m))
                    tau_s = np.exp(rng.uniform(math.log(5e-5), math.log(1.5e-4), m))
                    self.voice.add_force(off[rep] + delay, amp, tau_s, "pulse",
                                         None if var is None else var[rep])
        elif self.grains is not None:
            var = rng.integers(0, 2, n_ev)
            render_binned(off, force, var * len(CLASSES) + ci, self.grains.__getitem__, self.direct)

        # crown / dome splash hiss on a wet film (no bubbles)
        if wet > 0.05 and self.kind in ("solid", "porous", "membrane"):
            amp = (float(db_to_pa(self.mat.level_db + FILM_HISS_REL_DB)) * wet
                   * (f_phys / F_REF) ** 0.75 * sc * (0.5 if self.kind == "membrane" else 1.0))
            dur = np.clip(0.0025 * np.sqrt(d / 2.0), 0.0015, 0.008)
            keys = self.kb.dur_bin(dur) * 16 + rng.integers(0, self.kb.SPLASH_VARIANTS, n_ev)
            render_binned(off, amp, keys, lambda k: self.kb.splash(k // 16, k % 16, True), self.direct)

    def render(self, ctx: BlockContext) -> list[Emission]:
        if ctx.sr != self.sr:
            self._setup(int(ctx.sr))
        w = ctx.weather
        ev = self._events(ctx)
        if ev is not None:
            self._synthesize(ctx, ev)
        n = ctx.n
        y = np.zeros(n)
        live = False
        if self.voice is not None and self.voice.active:
            y += self.voice.render(n)
            live = True
        if self.direct.active:
            y += self.direct.pop(n)
            live = True
        if not live:
            return []
        if self.xfer is not None:
            y = self.xfer.process(y) * self.xfer_gain
        y = self.air.process(y * float(spreading_gain(self.r_eq)), self.r_eq, w.temperature_c, w.humidity_pct)
        return [Emission(signal=y, azimuth=self.patch.azimuth, elevation=self.patch.elevation,
                         distance=self.patch.distance, spread=self.spread, stem=self.stem,
                         is_ambisonic=False, source_id=self.name)]


# ---------------------------------------------------------------------------
# far-field diffuse bed, synthesized in the SH domain

class _XfadeSOS:
    """Multichannel air-absorption filter, redesigned on weather change with a one-block crossfade."""

    def __init__(self, distance_m: float):
        self.distance = float(distance_m)
        self.cur = None
        self.params = None

    def process(self, x: np.ndarray, temp_c: float, rh_pct: float, sr: int) -> np.ndarray:
        params = (float(temp_c), float(rh_pct))
        if self.cur is None:
            self.cur = StatefulSOS(air_absorption_sos(self.distance, temp_c, rh_pct, sr))
            self.params = params
            return self.cur.process(x)
        t0, h0 = self.params
        if abs(params[0] - t0) > 2.0 or abs(params[1] - h0) > 0.05 * max(h0, 1.0):
            new = StatefulSOS(air_absorption_sos(self.distance, temp_c, rh_pct, sr))
            y0 = self.cur.process(x)
            y1 = new.process(x)
            ramp = np.linspace(0.0, 1.0, x.shape[-1])
            self.cur = new
            self.params = params
            return y0 * (1.0 - ramp) + y1 * ramp
        return self.cur.process(x)


class RainBedGenerator:
    """Diffuse rain beyond the near rings: N_AMBI independent shaped noises, low-order anisotropy.

    Power follows the class fluxes (so the DSD sets level *and* timbre) integrated over the
    annulus r1..r2 (intensity ~ 2 pi ln(r2/r1) per unit source strength), the storm envelope
    and gusts; the spectrum is split into four bands by drop class and tilted by ISO 9613-1
    air absorption at the bed's effective distance.
    """

    BANDS = ((120.0, 700.0), (700.0, 2500.0), (2500.0, 8000.0), (8000.0, 18000.0))
    IMPACT_BANDS = np.array([[0.00, 0.10, 0.40, 0.50],
                             [0.02, 0.15, 0.48, 0.35],
                             [0.08, 0.32, 0.45, 0.15],
                             [0.25, 0.40, 0.28, 0.07],
                             [0.40, 0.35, 0.20, 0.05]])
    BUBBLE_BANDS = np.array([[0.00, 0.00, 0.00, 0.00],
                             [0.00, 0.00, 0.10, 0.90],
                             [0.00, 0.50, 0.50, 0.00],
                             [0.05, 0.45, 0.50, 0.00],
                             [0.15, 0.55, 0.30, 0.00]])
    HAIL_BANDS = np.array([0.15, 0.35, 0.35, 0.15])
    T_EFF = 5e-4      # effective energy duration of one ground impact (s)

    def __init__(self, scene, layout: list[SourcePatch], rngf: RngFactory, sr: int = SAMPLE_RATE):
        self.name = "rain/bed"
        self.rngf = rngf
        self.heading = float(scene.listener.heading_deg)
        self.water_quality = float(np.clip(scene.weather.precipitation.water_quality, 0.0, 1.0))
        rings = [float(p.params["r_outer"]) for p in layout
                 if (p.params or {}).get("ground") and "r_outer" in p.params]
        if rings:
            r1 = max(rings)
        else:
            ground = [p for p in layout if p.kind in ("surface", "water", "vegetation")
                      and p.elevation < 0.0 and not p.interior]
            r1 = 1.1 * max((p.distance for p in ground), default=30.0)
        r1 = float(np.clip(r1, 10.0, 80.0))
        r2 = max(300.0, 6.0 * r1)
        h = float(scene.listener.height_m)
        preset = str(scene.place.preset or "").lower()
        if any(k in preset for k in ("city", "street", "urban", "alley", "canyon", "town")):
            openness = 0.4
        elif any(k in preset for k in ("forest", "wood", "jungle")):
            openness = 0.5
        else:
            openness = 0.7
        self.area_factor = math.pi * math.log((r2 * r2 + h * h) / (r1 * r1 + h * h)) * openness
        self.r_air = 0.6 * math.sqrt(r1 * r2)

        surf = {k: float(v) for k, v in (scene.place.surfaces or {}).items() if float(v) > 0} or {"grass": 1.0}
        tot = sum(surf.values())
        lin = 0.0
        wf = 0.0
        for name, pct in surf.items():
            mat = get_material(name)
            lin += pct / tot * 10.0 ** (mat.level_db / 10.0)
            if mat.kind == "water":
                wf += pct / tot
        p_mix = float(db_to_pa(10.0 * math.log10(max(lin, 1e-12))))
        dd = np.array([REP_D[c] for c in CLASSES])
        vt = _fall_speed(dd)
        self.e_imp = (force_peak(dd, vt) / F_REF) ** 2 * p_mix ** 2 * self.T_EFF
        a = np.array([0.15e-3, 0.2e-3, 0.6e-3, 0.8e-3, 1.2e-3])
        eps = np.array([0.015, 0.045, 0.012, 0.06, 0.08])
        amp = BUBBLE_PA_PER_M * eps * a
        p_ent = np.array([0.0, 0.95, 0.03, 0.05 + 0.6 * _smoothstep(2.8, 2.2, 4.0),
                          0.05 + 0.6 * _smoothstep(4.2, 2.2, 4.0)])
        self.e_bub = p_ent * amp ** 2 / (4.0 * bubble_damping(a)) * wf * self.water_quality
        self.p_mix = p_mix
        self.sr = None
        self._setup(int(sr))

    def _setup(self, sr: int):
        self.sr = int(sr)
        self.bands = []
        self.band_e = np.zeros(len(self.BANDS))
        imp = np.zeros(16384)
        imp[0] = 1.0
        for k, (lo, hi) in enumerate(self.BANDS):
            hi = min(hi, 0.45 * self.sr)
            lo = min(lo, 0.5 * hi)
            sos = signal.butter(2, [lo, hi], "bandpass", fs=self.sr, output="sos")
            self.bands.append(StatefulSOS(sos))
            self.band_e[k] = float(np.sum(signal.sosfilt(sos, imp) ** 2))
        orders = np.floor(np.sqrt(np.arange(N_AMBI))).astype(int)
        self.ord_w = 1.0 / np.sqrt(2.0 * orders + 1.0)          # diffuse field in SN3D
        self.gains_prev = np.zeros(len(self.BANDS))
        self.aniso_prev = np.zeros(3)
        self.air = _XfadeSOS(self.r_air)
        # slowly wandering rain curtains: the louder side of the bed drifts (1/f, 1 s .. 30 s)
        self.wander = (RateModulator(self.rngf, "rain/bed/wander_x", sigma=0.6, tau_min_s=1.0,
                                     tau_max_s=30.0, sr=self.sr),
                       RateModulator(self.rngf, "rain/bed/wander_y", sigma=0.6, tau_min_s=1.0,
                                     tau_max_s=30.0, sr=self.sr))

    def _aniso(self, w: WeatherState, block: int, n: int) -> np.ndarray:
        v = np.array([0.0, 0.0, -0.2])                          # ground impacts: from below/horizon
        v[0] += 0.12 * math.log(self.wander[0].step(block, n))
        v[1] += 0.12 * math.log(self.wander[1].step(block, n))
        u10 = float(w.wind_u10_mps)
        if u10 > 0.0:                                           # wind-driven rain: louder windward
            k = 0.25 * min(u10 / 12.0, 1.0)
            v[0] += k * math.cos(w.wind_azimuth)
            v[1] += k * math.sin(w.wind_azimuth)
        for cell in (w.storms or []):
            az = bearing_to_azimuth(cell.bearing_deg, self.heading)
            k = 0.3 * float(np.clip(cell.rain_contribution, 0.0, 1.0)) / (1.0 + max(cell.distance_km, 0.0) / 8.0)
            v[0] += k * math.cos(az)
            v[1] += k * math.sin(az)
        nrm = float(np.linalg.norm(v))
        return v * (0.5 / nrm) if nrm > 0.5 else v

    def _band_power(self, w: WeatherState) -> np.ndarray:
        p = np.zeros(len(self.BANDS))
        gate = _RAIN_GATE.get(w.precip_type, 1.0)
        if gate > 0.0 and w.class_flux:
            flux = np.array([float(w.class_flux.get(c, 0.0) or 0.0) for c in CLASSES])
            e_b = self.e_bub.copy()
            e_b[1] *= 1.0 - _smoothstep(float(w.wind_u10_mps) * max(float(w.gust), 0.0), 7.5, 10.0)
            p += gate * ((flux * self.e_imp) @ self.IMPACT_BANDS + (flux * e_b) @ self.BUBBLE_BANDS)
        sp = _ice_species(w)
        if sp is not None and w.rain_rate_mm_h > 0.0:
            nflux = _ice_number_flux(w.rain_rate_mm_h, sp)
            dm = 0.6 * sp["size"]
            j = sp["rho"] * math.pi * (dm * 1e-3) ** 3 / 6.0 * float(_ice_fall_speed(dm, sp["kind"])) * (1.0 + sp["rest"])
            e_hail = (self.p_mix * j / J_REF) ** 2 * self.T_EFF * 0.3
            p += nflux * e_hail * self.HAIL_BANDS
        env = (0.6 + 0.4 * float(np.clip(w.intensity_envelope, 0.0, 2.0)))
        env *= float(np.clip(1.0 + 0.3 * (float(w.gust) - 1.0), 0.5, 1.6))
        return p * self.area_factor * env

    def render(self, ctx: BlockContext) -> list[Emission]:
        if ctx.sr != self.sr:
            self._setup(int(ctx.sr))
        w = ctx.weather
        n = ctx.n
        gains = np.sqrt(np.maximum(self._band_power(w), 0.0) / self.band_e)
        if not np.any(gains) and not np.any(self.gains_prev):
            return []
        noise = self.rngf.get("rain/bed", ctx.index).standard_normal((N_AMBI, n))
        ramp = np.arange(n, dtype=float) / n
        sig = np.zeros((N_AMBI, n))
        for k, band in enumerate(self.bands):
            g0, g1 = self.gains_prev[k], gains[k]
            if g0 == 0.0 and g1 == 0.0:
                band.reset()
                continue
            sig += band.process(noise) * (g0 + (g1 - g0) * ramp)
        self.gains_prev = gains
        sig *= self.ord_w[:, None]
        v = self._aniso(w, ctx.index, n)
        vr = self.aniso_prev[:, None] + (v - self.aniso_prev)[:, None] * ramp[None, :]
        self.aniso_prev = v
        if N_AMBI >= 4:
            wsig = sig[0].copy()
            sig[1] += wsig * vr[1]      # ACN 1 = Y (left)
            sig[2] += wsig * vr[2]      # ACN 2 = Z (up)
            sig[3] += wsig * vr[0]      # ACN 3 = X (front)
        sig = self.air.process(sig, w.temperature_c, w.humidity_pct, ctx.sr)
        return [Emission(signal=sig, azimuth=0.0, elevation=0.0, distance=self.r_air, spread=1.0,
                         stem="rain_far", is_ambisonic=True, source_id=self.name)]


# ---------------------------------------------------------------------------
# drips

class _DripLanding:
    """One landing spot of a drip generator: target surface, its voice and propagation state."""

    def __init__(self, target: Material, pos, extent_m: float, interior: bool, kb: KernelBank, sr: int):
        self.target = target
        self.target_kind = "water" if target.kind == "water" else target.kind
        x, y, z = (float(c) for c in pos)
        self.distance = max(math.sqrt(x * x + y * y + z * z), 0.2)
        self.azimuth = math.atan2(y, x)
        self.elevation = math.atan2(z, math.hypot(x, y))
        self.spread = float(np.clip(2.0 * math.atan(0.5 * extent_m / self.distance) / math.pi, 0.0, 1.0))
        self.stem = "interior" if interior else "aftermath"
        self.kb = kb
        self.direct = OverlapAdd()
        self.voice = _SolidVoice(target, sr, kb) if self.target_kind != "water" else None
        self.air = AirAbsorptionFilter(sr)
        self.events: list = []

    @property
    def active(self) -> bool:
        return self.direct.active or (self.voice is not None and self.voice.active)


class DripGenerator:
    """Integrate-and-fire drip nodes (raingen.physics.wetness.DripNode); plink or splat on landing.

    A layout 'drip_edge' patch is one drip point or a short run of `n_points` points (eave lip,
    umbrella rib, leaf tip, cave seep) landing at patch.position. With canopy=True the patch is a
    tree crown: `n_points` leaf-tip nodes drip onto the ground under it, grouped into up to four
    landing spots around the crown so the drips arrive from different directions.

    Feeds (patch.params['feed']): 'rain' (rain on the catchment, then the edge film after the rain),
    'roof' (gutter-routed roof flow, delayed and smoothed like the roof store), 'canopy' (canopy
    release share of the rain plus slow drainage of the canopy store) and 'storage' (seepage from
    the scene's stored water; keeps dripping long after the rain).

    Landing physics: a drop whose (D, U) lies in the regular-entrainment band of Oguz & Prosperetti
    (1990) traps one bubble of radius ~0.3-0.4 D and rings (the 'plink'); otherwise it is an
    irregular splash with the large-drop bubble statistics of `_water_impacts` (the 'splat').
    Solid targets get a force pulse through their modes plus prompt-splash micro-impacts.
    """

    MAX_GROUPS = 4

    def __init__(self, scene, patch: SourcePatch, rngf: RngFactory, canopy: bool = False,
                 kernels: KernelBank | None = None, sr: int = SAMPLE_RATE, layout: list | None = None,
                 roof_area_m2: float | None = None):
        self.name = f"drip/{patch.id}"
        self.patch = patch
        self.rngf = rngf
        self.canopy = bool(canopy)
        p = patch.params or {}
        lh = float(scene.listener.height_m)
        self.feed = str(p.get("feed") or ("canopy" if self.canopy else "rain")).lower()
        self.edge_material = get_material(p.get("edge_material") or p.get("roof_material")
                                          or ("leaves" if self.canopy else "tile")).name
        self.roof_material = get_material(p.get("roof_material") or self.edge_material).name
        self.water_quality = float(np.clip(scene.weather.precipitation.water_quality, 0.0, 1.0))
        self.exposure = float(max(patch.exposure, 0.0))
        self.depth_param = p.get("target_depth_mm")
        self.roof_area = float(roof_area_m2) if roof_area_m2 else 0.0

        if self.canopy:
            ground = {k: float(v) for k, v in (scene.place.surfaces or {}).items()
                      if float(v) > 0 and get_material(k).kind != "water"}
            default_target = max(ground, key=ground.get) if ground else "leaves"
        else:
            default_target = "puddle"
        self.default_target = get_material(p.get("target_material") or p.get("target") or default_target)

        rng = rngf.get(f"drip/{patch.id}/layout", 0)
        x, y, z = (float(c) for c in patch.position)
        if self.canopy:
            n_nodes = int(p.get("n_points", 8))
            catch = float(p.get("catchment_m2", 0.01))
            edge_r = float(p.get("edge_radius_mm", 2.2))
            fall = float(p.get("fall_height_m", max(z + lh, 1.0)))
            radius = max(float(p.get("canopy_radius_m", 2.5)), 0.3)
            n_groups = int(np.clip(n_nodes, 1, self.MAX_GROUPS))
            phi0 = float(rng.uniform(0.0, 2.0 * math.pi))
            spots = []
            for k in range(n_groups):
                a = phi0 + 2.0 * math.pi * k / n_groups
                rr = 0.6 * radius if n_groups > 1 else 0.0
                spots.append(((x + rr * math.cos(a), y + rr * math.sin(a), -lh), radius))
        else:
            n_nodes = int(p.get("n_points", 1))
            catch = float(p.get("catchment_m2", 0.08))
            edge_r = float(p.get("edge_radius_mm", 2.0))
            if "fall_height_m" in p:
                fall = float(p["fall_height_m"])
            else:
                ez = float((p.get("edge_position") or (x, y, z + 2.5))[2])
                fall = max(ez - z, 0.3)
            spots = [((x, y, z), max(float(p.get("length_m", 0.2)), 0.05))]

        self.nodes = []
        for i in range(max(1, n_nodes)):
            self.nodes.append(DripNode(f"{patch.id}/{i}", catch * rng.uniform(0.6, 1.4),
                                       edge_r * rng.uniform(0.85, 1.15), max(0.05, fall * rng.uniform(0.9, 1.1)),
                                       rngf, f"drip/{patch.id}/{i}"))
        self._spots = spots
        self._targets = []
        for (sx, sy, _sz), _ext in spots:
            tgt = self.default_target
            if self.canopy and layout is not None:
                try:
                    from raingen.physics.layout import ground_material_at
                    tgt = get_material(ground_material_at(layout, sx, sy, default=self.default_target.name))
                except ImportError:
                    tgt = self.default_target
            self._targets.append(tgt)
        self.last_marker_t = -1e9
        self._kb = kernels
        self.sr = None
        self._setup(int(sr))

    def _setup(self, sr: int):
        self.sr = int(sr)
        self.kb = self._kb if (self._kb is not None and self._kb.sr == self.sr) else KernelBank(self.sr)
        self.groups = [_DripLanding(tgt, pos, ext, self.patch.interior, self.kb, self.sr)
                       for tgt, (pos, ext) in zip(self._targets, self._spots)]
        self.click_pa = float(db_to_pa(MATERIALS["puddle"].level_db))

    # -- water supply --------------------------------------------------------
    def _feed(self, w: WeatherState) -> tuple[float, float]:
        """(inflow_mm_h on the catchment, upstream storage_mm) for this block."""
        liquid = _LIQUID.get(w.precip_type, 1.0)
        thaw = _smoothstep(float(w.temperature_c), -2.0, 0.0)       # edges freeze into icicles
        rain = max(float(w.rain_rate_mm_h), 0.0) * liquid
        wet = w.wetness or {}
        if self.feed == "canopy":
            # canopy release ~ the intercepted share (10-40 %) of the rain + slow drainage of the store
            return 0.25 * rain * self.exposure * thaw, 0.1 * max(float(w.canopy_storage_mm), 0.0) * thaw
        if self.feed == "roof":
            q = max(float(w.roof_flow_l_s), 0.0)
            inflow = q * 3600.0 / self.roof_area if (self.roof_area > 0.0 and q > 0.0) else rain * self.exposure
            return inflow * thaw, 0.4 * float(np.clip(wet.get(self.roof_material, 0.0), 0.0, 1.0)) * thaw
        if self.feed == "storage":
            store = max((float(v) for v in wet.values()), default=0.0)
            return 0.02 * rain * thaw, 0.3 * float(np.clip(store, 0.0, 1.0)) * thaw
        return rain * self.exposure * thaw, 0.4 * float(np.clip(wet.get(self.edge_material, 0.0), 0.0, 1.0)) * thaw

    # -- synthesis -----------------------------------------------------------
    def _target_depth(self, g: _DripLanding, w: WeatherState) -> float:
        if g.target.name == "water":
            return 1000.0
        if self.depth_param is not None:
            return max(float(self.depth_param), 0.5)
        return max(float((w.puddle_depth_mm or {}).get(g.target.name, 15.0) or 15.0), 3.0)

    def _synth(self, g: _DripLanding, rng, w: WeatherState, sr: int, off, d, u) -> bool:
        kb = self.kb
        n_ev = off.size
        plinked = False
        if g.target_kind == "water":
            depth = self._target_depth(g, w)
            plink = regular_entrainment(d, u)
            i = np.nonzero(plink)[0]
            if i.size:
                plinked = True
                m = i.size
                radius = d[i] * 1e-3 * rng.uniform(0.3, 0.42, m)
                eps = rng.uniform(0.08, 0.15, m)
                xi = rng.uniform(0.12, 0.3, m)
                delay = (rng.uniform(0.008, 0.02, m) * sr).astype(np.int64)
                fd = _smoothstep(depth, 1.5, 8.0) * self.water_quality
                if fd > 0.0:
                    render_binned(off[i] + delay, BUBBLE_PA_PER_M * eps * fd, _bubble_keys(kb, radius, xi, 0),
                                  _bubble_kernel_fn(kb), g.direct)
                render_binned(off[i], self.click_pa * d[i] / D_REF_MM, kb.click_bin(u[i]), kb.click_shape,
                              g.direct)
            s = np.nonzero(~plink)[0]
            if s.size:
                ci = np.where(d[s] >= 3.5, 4, 3).astype(np.int64)
                _water_impacts(kb, rng, off[s], d[s], u[s], ci, np.ones(s.size), g.direct,
                               click_pa=self.click_pa, depth_mm=depth, wind_mps=0.0,
                               water_quality=self.water_quality, sr=sr)
            return plinked

        # solid / porous / vegetation / membrane target; the puddled share becomes water impacts
        depth = float((w.puddle_depth_mm or {}).get(g.target.name, 0.0) or 0.0)
        if depth > 0.2 and g.target_kind in ("solid", "porous"):
            cover = min(1.0, depth / 8.0) ** 0.8
            wm = rng.random(n_ev) < cover
            if np.any(wm):
                ci = np.where(d[wm] >= 3.5, 4, 3).astype(np.int64)
                _water_impacts(kb, rng, off[wm], d[wm], u[wm], ci, np.ones(int(wm.sum())), g.direct,
                               click_pa=self.click_pa, depth_mm=max(depth, 2.0), sr=sr,
                               water_quality=self.water_quality)
                off, d, u = off[~wm], d[~wm], u[~wm]
                n_ev = off.size
        if n_ev == 0:
            return False
        wet = float(np.clip((w.wetness or {}).get(g.target.name, 0.0), 0.0, 1.0))
        force = force_peak(d, u) * np.exp(rng.normal(0.0, 0.15, n_ev)) * (1.0 - 0.35 * wet)
        tau = d * 1e-3 / np.maximum(u, 0.05) * (1.0 + 0.6 * wet)
        g.voice.add_force(off, force, tau, "pulse")
        if g.target.splash > 0.0 and g.target.hardness >= 0.45 and wet < 0.9:
            rep = np.repeat(np.arange(n_ev), rng.integers(3, 9, n_ev))
            m = rep.size
            delay = (np.exp(rng.uniform(math.log(0.005), math.log(0.05), m)) * sr).astype(np.int64)
            amp = force[rep] * np.exp(rng.uniform(math.log(0.005), math.log(0.03), m)) * (1.0 - wet)
            g.voice.add_force(off[rep] + delay, amp, np.exp(rng.uniform(math.log(5e-5), math.log(1.5e-4), m)),
                              "pulse")
        if wet > 0.05 and g.target_kind in ("solid", "porous"):
            amp = (float(db_to_pa(g.target.level_db + FILM_HISS_REL_DB)) * wet
                   * (force_peak(d, u) / F_REF) ** 0.75)
            dur = np.clip(0.0025 * np.sqrt(d / 2.0), 0.0015, 0.008)
            keys = kb.dur_bin(dur) * 16 + rng.integers(0, kb.SPLASH_VARIANTS, n_ev)
            render_binned(off, amp, keys, lambda k: kb.splash(k // 16, k % 16, True), g.direct)
        return False

    def render(self, ctx: BlockContext) -> list[Emission]:
        if ctx.sr != self.sr:
            self._setup(int(ctx.sr))
        w = ctx.weather
        n, sr, b = ctx.n, ctx.sr, ctx.index
        inflow, storage = self._feed(w)
        n_groups = len(self.groups)
        any_events = False
        for i, node in enumerate(self.nodes):
            ev = node.step(b, n, sr, inflow, storage)
            if ev:
                self.groups[i % n_groups].events.extend(ev)
                any_events = True
        if any_events:
            rng = self.rngf.get(f"drip/{self.patch.id}/sound", b)
            first = None
            for g in self.groups:
                if not g.events:
                    continue
                arr = np.asarray(g.events, dtype=float).reshape(-1, 3)
                g.events = []
                off = np.clip(arr[:, 0], 0, None).astype(np.int64)
                d = np.clip(arr[:, 1], 0.5, 8.0)
                u = np.maximum(arr[:, 2], 0.05)
                plinked = self._synth(g, rng, w, sr, off, d, u)
                if first is None:
                    first = (int(off.min()), plinked, float(d[0]), g.target.name)
            if first is not None and ctx.t0 - self.last_marker_t >= 15.0:
                ctx.markers.append(EventMarker(t=ctx.t0 + first[0] / sr, kind="drip",
                                               data={"source": self.name,
                                                     "sound": "plink" if first[1] else "splat",
                                                     "diameter_mm": first[2], "target": first[3]}))
                self.last_marker_t = ctx.t0
        out = []
        for k, g in enumerate(self.groups):
            if not g.active:
                continue
            y = np.zeros(n)
            if g.voice is not None and g.voice.active:
                y += g.voice.render(n)
            if g.direct.active:
                y += g.direct.pop(n)
            y = g.air.process(y * float(spreading_gain(g.distance)), g.distance, w.temperature_c, w.humidity_pct)
            sid = self.name if n_groups == 1 else f"{self.name}/{k}"
            out.append(Emission(signal=y, azimuth=g.azimuth, elevation=g.elevation, distance=g.distance,
                                spread=g.spread, stem=g.stem, is_ambisonic=False, source_id=sid))
        return out


# ---------------------------------------------------------------------------
# gutters and downspouts

class GutterGenerator:
    """Roof runoff in a gutter channel or a downspout pipe (hypothesis model, MASTER_PLAN 3.9).

    Driven by WeatherState.roof_flow_l_s (raingen.physics.wetness roof -> gutter routing):
    - trickle (q < ~5 ml/s): separate ~4 mm drops falling down the pipe / off the channel lip onto
      the elbow or the shell, each a force pulse through the metal's modes plus a water impact;
    - bubbly flow: a cloud of entrained bubbles, radii ~ r^-2.6 (larger bubbles as the flow grows);
    - slug flow (downspouts, q > ~0.15 l/s): big glugging bubbles;
    - turbulent rush noise. Everything excites the air column: an open pipe with end corrections,
      f_k = k c / 2 (L + 0.6 d); an open half-round channel couples only weakly to its length modes.
    - an open gutter overflows when the flow exceeds its design capacity (params['overflow_rate_mm_h']
      over the catchment): a falling sheet that splashes on the ground below.
    """

    BUBBLE_GAIN = 0.5            # bubbles inside a channel / pipe radiate less directly

    def __init__(self, scene, patch: SourcePatch, rngf: RngFactory, n_gutters: int = 1,
                 kernels: KernelBank | None = None, sr: int = SAMPLE_RATE):
        self.name = f"gutter/{patch.id}"
        self.patch = patch
        self.rngf = rngf
        p = patch.params or {}
        self.is_pipe = _is_downspout(patch)
        self.share = float(p.get("share", 1.0 / max(1, int(n_gutters))))
        if self.is_pipe:
            self.length = max(float(p.get("pipe_length_m", p.get("length_m", 2.5))), 0.3)
        else:
            self.length = max(float(p.get("length_m", 6.0)), 0.3)
        self.diam = max(float(p.get("diameter_m", p.get("width_m", 0.08))), 0.02)
        mat = get_material(p.get("material") or patch.material or "metal_gutter")
        self.pipe_mat = mat if mat.kind == "solid" else MATERIALS["metal_gutter"]
        self.temp = float(scene.weather.temperature_c)
        self.water_quality = float(np.clip(scene.weather.precipitation.water_quality, 0.0, 1.0))
        catch = float(p.get("catchment_m2", 25.0))
        over = p.get("overflow_rate_mm_h")
        self.q_cap = (float(over) * catch / 3600.0) if (over is not None and not self.is_pipe) else math.inf
        self.drop_fall_m = 0.8 * self.length if self.is_pipe else max(float(p.get("height_m", 2.5)), 0.3)
        self.pipe_mix = 0.8 if self.is_pipe else 0.3
        self.distance = max(patch.distance, 0.3)
        self.spread = float(np.clip(2.0 * math.atan(0.25 * self.length / self.distance) / math.pi, 0.0, 1.0))
        self._kb = kernels
        self.sr = None
        self._setup(int(sr))

    def _setup(self, sr: int):
        self.sr = int(sr)
        self.kb = self._kb if (self._kb is not None and self._kb.sr == self.sr) else KernelBank(self.sr)
        c = speed_of_sound(self.temp)
        f1 = c / (2.0 * (self.length + 0.6 * self.diam))      # open-open pipe with end corrections
        modes = [(k * f1, 0.35 * k ** -0.4, k ** -0.5) for k in range(1, 64)
                 if k * f1 < min(5000.0, 0.4 * self.sr)]
        self.pipe = ModalBank(modes, self.sr)
        e_ir = float(np.sum(self.pipe.impulse_response(int(1.5 * self.sr)) ** 2)) if self.pipe.n_modes else 1.0
        self.pipe_norm = 1.0 / math.sqrt(max(e_ir, 1e-12))
        self.shell = _SolidVoice(self.pipe_mat, self.sr, self.kb)
        imp = np.zeros(8192)
        imp[0] = 1.0
        sos = signal.butter(2, [250.0, min(5000.0, 0.4 * self.sr)], "bandpass", fs=self.sr, output="sos")
        self.rush = StatefulSOS(sos)
        self.rush_norm = 1.0 / math.sqrt(max(float(np.sum(signal.sosfilt(sos, imp) ** 2)), 1e-12))
        sos_o = signal.butter(2, [200.0, min(9000.0, 0.4 * self.sr)], "bandpass", fs=self.sr, output="sos")
        self.sheet = StatefulSOS(sos_o)
        self.sheet_norm = 1.0 / math.sqrt(max(float(np.sum(signal.sosfilt(sos_o, imp) ** 2)), 1e-12))
        self.ola = OverlapAdd()
        self.air = AirAbsorptionFilter(self.sr)
        self.mod = RateModulator(self.rngf, f"gutter/{self.patch.id}/gurgle", sigma=0.45,
                                 tau_min_s=0.05, tau_max_s=4.0, sr=self.sr)
        self.click_pa = float(db_to_pa(MATERIALS["puddle"].level_db))
        self.prev_rush = 0.0
        self.prev_sheet = 0.0
        self.overflowing = False

    def render(self, ctx: BlockContext) -> list[Emission]:
        if ctx.sr != self.sr:
            self._setup(int(ctx.sr))
        w = ctx.weather
        n, sr, b = ctx.n, ctx.sr, ctx.index
        kb = self.kb
        pid = self.patch.id
        q = max(float(w.roof_flow_l_s), 0.0) * self.share         # l/s through this channel / pipe
        tails = self.ola.active or self.shell.active or not self.pipe.is_silent(1e-12)
        if q < 1e-5 and not tails and self.prev_rush == 0.0 and self.prev_sheet == 0.0:
            return []
        curve = self.mod.curve(b, n)
        rush = 0.0
        sheet = 0.0
        if q > 0.0:
            # trickle: separate ~4 mm drops (3.4e-5 l each) onto the elbow / channel bottom
            drip_w = 1.0 - _smoothstep(q, 0.002, 0.006)
            rng = self.rngf.get(f"gutter/{pid}/drops", b)
            off = cox_offsets(rng, drip_w * q / 3.4e-5, n, sr, curve)
            if off.size:
                d = rng.uniform(3.5, 4.5, off.size)
                u = np.asarray(_dsd.fall_speed_from_height(d, self.drop_fall_m if self.is_pipe else 0.05),
                               dtype=float)
                u = np.maximum(u, 0.3)
                self.shell.add_force(off, 0.5 * force_peak(d, u), d * 1e-3 / u, "pulse")
                _water_impacts(kb, rng, off, d, u, np.full(off.size, 4, dtype=np.int64), np.full(off.size, 0.5),
                               self.ola, click_pa=self.click_pa, depth_mm=10.0,
                               water_quality=self.water_quality, sr=sr)
            # bubbly flow
            flow_w = _smoothstep(q, 0.001, 0.01)
            rng_b = self.rngf.get(f"gutter/{pid}/bubbles", b)
            lam = flow_w * min(900.0 * (q / 0.05) ** 0.8, 8000.0)
            offb, scb = thinned_offsets(rng_b, lam, n, sr, 500, curve)
            if offb.size:
                m = offb.size
                r_max = (1.5 + 6.0 * _smoothstep(q, 0.02, 0.5)) * 1e-3
                radius = _power_law(rng_b, 0.4e-3, r_max, 2.6, m)
                eps = _loguniform(rng_b, 0.01, 0.06, m)
                xi = rng_b.uniform(0.0, 0.15, m)
                amp = BUBBLE_PA_PER_M * eps * scb * self.BUBBLE_GAIN * self.water_quality
                render_binned(offb, amp, _bubble_keys(kb, radius, xi, 0), _bubble_kernel_fn(kb), self.ola)
            # slug-flow glugs (pipes only)
            if self.is_pipe:
                glug_rate = _smoothstep(q, 0.15, 0.6) * (0.7 + 2.5 * _smoothstep(q, 0.3, 1.2))
                offg = cox_offsets(rng_b, glug_rate, n, sr)
                if offg.size:
                    m = offg.size
                    radius = _loguniform(rng_b, 4e-3, 10e-3, m)
                    eps = rng_b.uniform(0.08, 0.18, m)
                    xi = rng_b.uniform(0.0, 0.05, m)
                    render_binned(offg, BUBBLE_PA_PER_M * eps * self.BUBBLE_GAIN, _bubble_keys(kb, radius, xi, 0),
                                  _bubble_kernel_fn(kb), self.ola)
            rush = flow_w * float(db_to_pa(50.0 + 10.0 * math.log10(max(q, 1e-6) / 0.1))) * float(curve.mean())
            if not self.is_pipe:
                rush *= 0.6
            # overflow sheet splashing on the ground below the gutter
            over = q - self.q_cap
            if over > 0.0:
                sheet = float(db_to_pa(54.0 + 10.0 * math.log10(max(over, 1e-6) / 0.05))) * float(curve.mean())
                if not self.overflowing:
                    ctx.markers.append(EventMarker(t=ctx.t0, kind="gutter_overflow",
                                                   data={"source": self.name, "flow_l_s": q}))
                self.overflowing = True
            else:
                self.overflowing = False

        exc = self.ola.pop(n)
        if rush > 0.0 or self.prev_rush > 0.0:
            noise = self.rngf.get(f"gutter/{pid}/rush", b).standard_normal(n)
            g = np.linspace(self.prev_rush, rush, n, endpoint=False)
            exc = exc + self.rush.process(noise) * self.rush_norm * g
        self.prev_rush = rush
        y = 0.6 * exc + self.pipe_mix * self.pipe_norm * self.pipe.process(exc)
        if self.shell.active:
            y += self.shell.render(n)
        if sheet > 0.0 or self.prev_sheet > 0.0:
            noise = self.rngf.get(f"gutter/{pid}/sheet", b).standard_normal(n)
            g = np.linspace(self.prev_sheet, sheet, n, endpoint=False)
            y = y + self.sheet.process(noise) * self.sheet_norm * g
        self.prev_sheet = sheet
        if not np.any(y) and not self.ola.active:
            return []
        y = self.air.process(y * float(spreading_gain(self.distance)), self.distance, w.temperature_c, w.humidity_pct)
        return [Emission(signal=y, azimuth=self.patch.azimuth, elevation=self.patch.elevation,
                         distance=self.patch.distance, spread=self.spread, stem="aftermath",
                         is_ambisonic=False, source_id=self.name)]


# ---------------------------------------------------------------------------
# hail, ice pellets, graupel

def _ice_species(w: WeatherState):
    t = w.precip_type
    if t == "hail":
        return {"kind": "hail", "size": max(float(w.hail_size_mm), 5.0), "rho": 900.0, "e": 5.0e9,
                "crush": 10.0, "frac": 0.5, "rest": 0.5}
    if t == "sleet":
        return {"kind": "sleet", "size": 3.5, "rho": 900.0, "e": 9.0e9, "crush": 15.0, "frac": 0.5, "rest": 0.45}
    if t == "graupel":
        return {"kind": "graupel", "size": 3.0, "rho": 350.0, "e": 3.0e8, "crush": 1.5, "frac": 1.0, "rest": 0.15}
    return None


def _ice_sizes(rng, size_mm: float, n: int) -> np.ndarray:
    """Truncated exponential on [0.3, 1.6] x size with slope 2/size (stones skew small)."""
    lo, hi = 0.3 * size_mm, 1.6 * size_mm
    lam = 2.0 / size_mm
    u = rng.random(n)
    return lo - np.log(1.0 - u * (1.0 - math.exp(-lam * (hi - lo)))) / lam


def _ice_fall_speed(d_mm, kind: str):
    """NOAA hail bands fit (V = 8.5 D_cm^0.57: pea 6.6, 1 in 14.5, golf ball 20, 4 in 32 m/s);
    ice pellets 3.5 D^0.4; graupel 1.3 D^0.6 (Locatelli & Hobbs 1974 style)."""
    d = np.maximum(np.asarray(d_mm, dtype=float), 0.1)
    if kind == "hail":
        return 8.5 * (d / 10.0) ** 0.57
    if kind == "sleet":
        return 3.5 * d ** 0.4
    return 1.3 * d ** 0.6


def _ice_mean_volume(size_mm: float) -> float:
    key = round(size_mm, 2)
    v = _ICE_VOL_CACHE.get(key)
    if v is None:
        lo, hi = 0.3 * size_mm, 1.6 * size_mm
        dgrid = np.linspace(lo, hi, 256)
        pdf = np.exp(-2.0 / size_mm * (dgrid - lo))
        pdf /= pdf.sum()
        v = float(np.sum(pdf * math.pi * (dgrid * 1e-3) ** 3 / 6.0))
        _ICE_VOL_CACHE[key] = v
    return v


def _ice_number_flux(rate_mm_h: float, sp: dict) -> float:
    """Stones per m^2 per s from the liquid-equivalent rate (share sp['frac'] falls as ice)."""
    q = max(rate_mm_h, 0.0) * 1e-3 / 3600.0 * sp["frac"] * RHO_WATER / sp["rho"]
    return q / max(_ice_mean_volume(sp["size"]), 1e-12)


class _HailPatch:
    MAX_EVENTS = 400

    def __init__(self, patch: SourcePatch, rngf: RngFactory, kb: KernelBank, sr: int, water_quality: float):
        self.patch = patch
        self.rngf = rngf
        self.kb = kb
        self.sr = int(sr)
        self.mat = get_material(patch.material)
        self.kind = "water" if (patch.kind == "water" or self.mat.kind == "water") else self.mat.kind
        self.voice = _SolidVoice(self.mat, self.sr, kb) if self.kind != "water" else None
        self.direct = OverlapAdd()
        self.air = AirAbsorptionFilter(self.sr)
        self.r_eq, self.spread = _patch_geometry(patch)
        self.stem = _stem_for(patch)
        self.normal = _unit(patch.normal)
        self.e_target = self.mat.youngs_modulus_pa
        self.click_pa = float(db_to_pa(MATERIALS["puddle"].level_db))
        self.water_quality = water_quality
        self.xfer_gain, xs = _interior_transfer(patch, self.sr)
        self.xfer = StatefulSOS(xs) if xs is not None else None
        self.pend_t = np.zeros(0, dtype=np.int64)    # absolute sample of pending bounce impacts
        self.pend_v = np.zeros(0)
        self.pend_d = np.zeros(0)
        self.pend_s = np.zeros(0)

    @property
    def active(self) -> bool:
        return (self.pend_t.size > 0 or self.direct.active
                or (self.voice is not None and self.voice.active))

    def render(self, ctx: BlockContext, sp, flux: float):
        w = ctx.weather
        n, sr, b = ctx.n, ctx.sr, ctx.index
        t_abs0 = int(round(ctx.t0 * sr))
        rng = self.rngf.get(f"hail/{self.patch.id}", b)
        nx, ny, nz = self.normal
        offs, ds, vs, scs = [], [], [], []
        primary = 0
        if sp is not None and flux > 0.0 and self.patch.exposure > 0.0:
            uh = _wind_vector(w)
            vmean = float(_ice_fall_speed(0.6 * sp["size"], sp["kind"]))
            fac = max(0.0, nz - (nx * uh[0] + ny * uh[1]) / max(vmean, 0.5))
            rate = flux * self.patch.area_m2 * self.patch.exposure * fac
            off, scale = thinned_offsets(rng, rate, n, sr, self.MAX_EVENTS)
            if off.size:
                d = _ice_sizes(rng, sp["size"], off.size)
                vt = _ice_fall_speed(d, sp["kind"])
                un = np.maximum(vt * nz - (nx * uh[0] + ny * uh[1]), 0.25 * vt)
                offs.append(off)
                ds.append(d)
                vs.append(un)
                scs.append(np.full(off.size, scale))
                primary = off.size
        if self.pend_t.size:
            due = self.pend_t < t_abs0 + n
            if np.any(due):
                offs.append(np.clip(self.pend_t[due] - t_abs0, 0, n - 1))
                ds.append(self.pend_d[due])
                vs.append(self.pend_v[due])
                scs.append(self.pend_s[due])
                keep = ~due
                self.pend_t, self.pend_v = self.pend_t[keep], self.pend_v[keep]
                self.pend_d, self.pend_s = self.pend_d[keep], self.pend_s[keep]
        if offs and sp is None:
            sp = {"kind": "hail", "size": 10.0, "rho": 900.0, "e": 5.0e9, "crush": 10.0, "frac": 0.5, "rest": 0.5}
        if offs:
            off = np.concatenate(offs).astype(np.int64)
            d = np.concatenate(ds)
            v = np.concatenate(vs)
            sc = np.concatenate(scs)
            self._synth(ctx, rng, sp, off, d, v, sc, primary, t_abs0)

        y = np.zeros(n)
        live = False
        if self.voice is not None and self.voice.active:
            y += self.voice.render(n)
            live = True
        if self.direct.active:
            y += self.direct.pop(n)
            live = True
        if not live:
            return None
        if self.xfer is not None:
            y = self.xfer.process(y) * self.xfer_gain
        return self.air.process(y * float(spreading_gain(self.r_eq)), self.r_eq, w.temperature_c, w.humidity_pct)

    def _synth(self, ctx, rng, sp, off, d, v, sc, primary: int, t_abs0: int):
        w = ctx.weather
        sr = ctx.sr
        n_ev = off.size
        if self.kind == "water":
            _water_impacts(self.kb, rng, off, d, v, np.full(n_ev, 4, dtype=np.int64), sc, self.direct,
                           click_pa=self.click_pa, depth_mm=self._depth(w), water_quality=self.water_quality,
                           sr=sr)
            return
        depth = float((w.puddle_depth_mm or {}).get(self.mat.name, 0.0) or 0.0)
        if depth > 0.5 and self.kind in ("solid", "porous"):
            wm = rng.random(n_ev) < min(1.0, depth / 8.0) ** 0.8
            if np.any(wm):
                _water_impacts(self.kb, rng, off[wm], d[wm], v[wm], np.full(int(wm.sum()), 4, dtype=np.int64),
                               sc[wm], self.direct, click_pa=self.click_pa, depth_mm=max(depth, 2.0),
                               water_quality=self.water_quality, sr=sr)
                keep = ~wm
                primary = int(np.sum(keep[:primary]))
                off, d, v, sc = off[keep], d[keep], v[keep], sc[keep]
                n_ev = off.size
        if n_ev == 0:
            return
        f_max, tau = hertz_contact(d, v, rho=sp["rho"], e_proj=sp["e"], e_target=self.e_target,
                                   crush_mpa=sp["crush"], restitution=sp["rest"])
        f_max = np.asarray(f_max, dtype=float) * sc * np.exp(rng.normal(0.0, 0.2, n_ev))
        self.voice.add_force(off, f_max, np.asarray(tau, dtype=float), "hertz")

        # ice crack on hard targets
        if sp["kind"] == "hail" and self.mat.hardness >= 0.6:
            c = np.nonzero((d > 8.0) & (rng.random(n_ev) < 0.35))[0]
            if c.size:
                j = sp["rho"] * math.pi * (d[c] * 1e-3) ** 3 / 6.0 * v[c] * (1.0 + sp["rest"])
                amp = 0.25 * float(db_to_pa(self.mat.level_db)) * (j / J_REF) * sc[c]
                dur = rng.uniform(0.0015, 0.003, c.size)
                keys = self.kb.dur_bin(dur) * 16 + rng.integers(0, self.kb.SPLASH_VARIANTS, c.size)
                render_binned(off[c], amp, keys, lambda k: self.kb.splash(k // 16, k % 16, True), self.direct)

        # bounces of primary impacts on firm targets: v_k = e^k v, flight 2 v_k / g
        if primary > 0 and self.mat.hardness >= 0.4:
            hard = float(np.clip((self.mat.hardness - 0.3) / 0.6, 0.0, 1.0))
            e = sp["rest"] * hard * (1.0 - 0.3 * _smoothstep(d[:primary], 10.0, 40.0))
            t_acc = np.zeros(primary)
            vk = v[:primary].copy()
            new_t, new_v, new_d, new_s = [], [], [], []
            for _ in range(3):
                vk = vk * e
                t_acc = t_acc + 2.0 * vk / G
                ok = vk > 0.8
                if not np.any(ok):
                    break
                new_t.append(t_abs0 + off[:primary][ok] + (t_acc[ok] * sr).astype(np.int64))
                new_v.append(vk[ok])
                new_d.append(d[:primary][ok])
                new_s.append(sc[:primary][ok])
            if new_t:
                self.pend_t = np.concatenate([self.pend_t] + new_t).astype(np.int64)
                self.pend_v = np.concatenate([self.pend_v] + new_v)
                self.pend_d = np.concatenate([self.pend_d] + new_d)
                self.pend_s = np.concatenate([self.pend_s] + new_s)

    def _depth(self, w: WeatherState) -> float:
        if self.mat.name == "water":
            return 1000.0
        return max(float((w.puddle_depth_mm or {}).get(self.mat.name, 12.0) or 12.0), 0.5)


class HailGenerator:
    """Hail (NOAA sizes), ice pellets (sleet) and graupel on the nearest surfaces."""

    def __init__(self, scene, patches: list[SourcePatch], rngf: RngFactory,
                 kernels: KernelBank | None = None, sr: int = SAMPLE_RATE):
        self.name = "hail"
        self.rngf = rngf
        self.patches = [p for p in patches if p.kind in RAIN_KINDS]
        self.water_quality = float(np.clip(scene.weather.precipitation.water_quality, 0.0, 1.0))
        self._kb = kernels
        self.sr = int(sr)
        self.kb = kernels if (kernels is not None and kernels.sr == self.sr) else KernelBank(self.sr)
        self.states: dict[str, _HailPatch] = {}
        self.was_active = False

    def render(self, ctx: BlockContext) -> list[Emission]:
        if ctx.sr != self.sr:
            self.sr = int(ctx.sr)
            self.kb = self._kb if (self._kb is not None and self._kb.sr == self.sr) else KernelBank(self.sr)
            self.states = {}
        w = ctx.weather
        sp = _ice_species(w)
        active = sp is not None and float(w.rain_rate_mm_h) > 0.0
        if active and not self.was_active:
            ctx.markers.append(EventMarker(t=ctx.t0, kind="hail",
                                           data={"type": sp["kind"], "size_mm": sp["size"]}))
        self.was_active = active
        if not active and not any(s.active for s in self.states.values()):
            return []
        flux = _ice_number_flux(float(w.rain_rate_mm_h), sp) if active else 0.0
        out = []
        for p in self.patches:
            st = self.states.get(p.id)
            if st is None:
                if not active:
                    continue
                st = _HailPatch(p, self.rngf, self.kb, self.sr, self.water_quality)
                self.states[p.id] = st
            elif not active and not st.active:
                continue
            y = st.render(ctx, sp if active else None, flux)
            if y is not None:
                out.append(Emission(signal=y, azimuth=p.azimuth, elevation=p.elevation, distance=p.distance,
                                    spread=st.spread, stem=st.stem, is_ambisonic=False,
                                    source_id=f"hail/{p.id}"))
        return out


# ---------------------------------------------------------------------------
# snow

class _SnowPatch:
    MAX_EVENTS = 300

    def __init__(self, patch: SourcePatch, rngf: RngFactory, kb: KernelBank, sr: int):
        self.patch = patch
        self.rngf = rngf
        self.mat = get_material(patch.material)
        self.voice = _SolidVoice(self.mat, sr, kb)
        self.air = AirAbsorptionFilter(sr)
        self.r_eq, self.spread = _patch_geometry(patch)
        self.stem = _stem_for(patch)
        self.normal = _unit(patch.normal)
        self.xfer_gain, xs = _interior_transfer(patch, sr)
        self.xfer = StatefulSOS(xs) if xs is not None else None

    @property
    def active(self) -> bool:
        return self.voice.active

    def render(self, ctx: BlockContext, flake, flux: float):
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        if flake is not None and flux > 0.0 and self.patch.exposure > 0.0:
            rng = self.rngf.get(f"snow/{self.patch.id}", ctx.index)
            nx, ny, nz = self.normal
            uh = _wind_vector(w)
            fac = max(0.0, nz - (nx * uh[0] + ny * uh[1]) / max(flake["v"], 0.3))
            rate = flux * self.patch.area_m2 * self.patch.exposure * fac
            off, scale = thinned_offsets(rng, rate, n, sr, self.MAX_EVENTS)
            if off.size:
                m = off.size
                mass = flake["mass"] * np.exp(rng.normal(0.0, 0.4, m))
                vt = flake["v"] * np.exp(rng.normal(0.0, 0.15, m))
                un = np.maximum(vt * nz - (nx * uh[0] + ny * uh[1]), 0.3 * vt)
                tau = flake["tau"] * np.exp(rng.normal(0.0, 0.2, m))
                force = mass * un / (0.511 * tau) * scale          # impulse m V spread over the crumple
                self.voice.add_force(off, force, tau, "pulse")
        if not self.voice.active:
            return None
        y = self.voice.render(n)
        if self.xfer is not None:
            y = self.xfer.process(y) * self.xfer_gain
        return self.air.process(y * float(spreading_gain(self.r_eq)), self.r_eq, w.temperature_c, w.humidity_pct)


class SnowGenerator:
    """Near-silent snowfall: soft flake ticks on near surfaces; saltation hiss in cold wind."""

    def __init__(self, scene, patches: list[SourcePatch], rngf: RngFactory,
                 kernels: KernelBank | None = None, sr: int = SAMPLE_RATE):
        self.name = "snow"
        self.rngf = rngf
        self.patches = [p for p in patches if p.kind in RAIN_KINDS
                        and get_material(p.material).kind not in ("water", "snow")]
        surf = {str(k).lower() for k, v in (scene.place.surfaces or {}).items() if float(v) > 0}
        self.snow_ground = str(scene.place.ground).startswith("snow") or any("snow" in s for s in surf)
        self._kb = kernels
        self.sr = int(sr)
        self.kb = kernels if (kernels is not None and kernels.sr == self.sr) else KernelBank(self.sr)
        self.states: dict[str, _SnowPatch] = {}
        self._salt_setup()

    def _salt_setup(self):
        sos = signal.butter(2, [2000.0, min(10000.0, 0.4 * self.sr)], "bandpass", fs=self.sr, output="sos")
        self.salt = StatefulSOS(sos)
        imp = np.zeros(8192)
        imp[0] = 1.0
        self.salt_norm = 1.0 / math.sqrt(max(float(np.sum(signal.sosfilt(sos, imp) ** 2)), 1e-12))
        self.salt_prev = 0.0

    @staticmethod
    def flake_params(temp_c: float) -> dict:
        """Dry cold crystals tick; wet aggregates near 0 C thud softly."""
        s = _smoothstep(temp_c, -6.0, 0.5)
        return {"d_mm": 2.0 + 8.0 * s, "mass": (0.4 + 9.6 * s ** 1.5) * 1e-6, "v": 0.7 + 0.8 * s,
                "tau": (0.8 + 4.0 * s) * 1e-3}

    def render(self, ctx: BlockContext) -> list[Emission]:
        if ctx.sr != self.sr:
            self.sr = int(ctx.sr)
            self.kb = self._kb if (self._kb is not None and self._kb.sr == self.sr) else KernelBank(self.sr)
            self.states = {}
            self._salt_setup()
        w = ctx.weather
        active = w.precip_type == "snow" and float(w.rain_rate_mm_h) > 0.0
        out = []
        if active or any(s.active for s in self.states.values()):
            flake = self.flake_params(float(w.temperature_c)) if active else None
            # liquid-equivalent mass flux / flake mass
            flux = (float(w.rain_rate_mm_h) * 1e-3 / 3600.0 * RHO_WATER / flake["mass"]) if active else 0.0
            for p in self.patches:
                st = self.states.get(p.id)
                if st is None:
                    if not active:
                        continue
                    st = _SnowPatch(p, self.rngf, self.kb, self.sr)
                    self.states[p.id] = st
                y = st.render(ctx, flake, flux)
                if y is not None:
                    out.append(Emission(signal=y, azimuth=p.azimuth, elevation=p.elevation, distance=p.distance,
                                        spread=st.spread, stem=st.stem, is_ambisonic=False,
                                        source_id=f"snow/{p.id}"))
        salt = self._saltation(ctx, active)
        if salt is not None:
            out.append(salt)
        return out

    def _saltation(self, ctx: BlockContext, snowing: bool):
        w = ctx.weather
        u = float(w.wind_listener_mps)
        level = 0.0
        if (snowing or self.snow_ground) and w.temperature_c < -1.0 and u > 6.0:
            level = float(db_to_pa(26.0 + 30.0 * math.log10(u / 6.0)))
            level *= float(np.clip(w.gust, 0.3, 2.5)) ** 1.5
        if level == 0.0 and self.salt_prev == 0.0:
            return None
        n = ctx.n
        noise = self.rngf.get("snow/saltation", ctx.index).standard_normal(n)
        g = np.linspace(self.salt_prev, level, n, endpoint=False)
        self.salt_prev = level
        y = self.salt.process(noise) * self.salt_norm * g
        return Emission(signal=y, azimuth=float(w.wind_azimuth), elevation=-0.3, distance=3.0, spread=0.85,
                        stem="rain_far", is_ambisonic=False, source_id="snow/saltation")


# ---------------------------------------------------------------------------
# factory

def _loudness_score(p: SourcePatch) -> float:
    mat = get_material(p.material)
    lvl = mat.level_db + (6.0 if mat.kind == "water" else 0.0)
    return (lvl + 10.0 * math.log10(max(p.area_m2 * max(p.exposure, 0.0), 1e-3))
            - 20.0 * math.log10(max(p.distance, 0.3)) + (6.0 if p.interior else 0.0))


def _is_downspout(patch: SourcePatch) -> bool:
    p = patch.params or {}
    return str(p.get("type", "")).lower() == "downspout" or "pipe_length_m" in p


def _roof_area_from_layout(layout: list[SourcePatch]) -> float:
    """Roof catchment (m^2) feeding the gutters, as raingen.physics.world routes it (for 'roof' drips)."""
    catch = [float(p.params.get("catchment_m2", 0.0)) for p in layout
             if p.kind == "gutter" and p.params.get("catchment_m2")]
    roofs = sum(p.area_m2 for p in layout
                if p.material in ("tin_roof", "tile") and not p.params.get("ground"))
    area = max(catch + [min(roofs, 200.0)]) if (catch or roofs) else 30.0
    return max(area, 5.0)


_ALL_TYPES = ("none", "drizzle", "rain", "showers", "freezing_rain", "sleet", "snow", "graupel", "hail")


def _precip_types(scene) -> set:
    """Precipitation types a scene can reach: its own plus any timeline keyframes on the type."""
    types = {str(scene.weather.precipitation.type or "rain").lower()}
    for kf in getattr(scene, "timeline", None) or []:
        if str(getattr(kf, "path", "")).endswith("precipitation.type"):
            types.add(str(getattr(kf, "value", "")).lower())
    return types


def make_rain_generators(scene, layout: list[SourcePatch], rngf: RngFactory, *, precip_types=None) -> list:
    """Rain, bed, drip, gutter, hail and snow generators for a scene layout, with LOD assignment.

    Generators are chosen by precipitation type: liquid-drop patches and the bed when rain can fall
    (anything but none / snow / graupel), HailGenerator for hail / sleet / graupel, SnowGenerator for
    snow or a snow-covered place (blowing-snow saltation). `precip_types` overrides the set of types
    to prepare for (default: the scene's type plus timeline keyframes on weather.precipitation.type;
    'all' for live control where the type can change at any time). Drips and gutters are always
    built: they are driven by the wetness stores, so they also play the aftermath of a dry scene.

    Level of detail (Zhang et al. 2019; MASTER_PLAN 3.5): the K loudest rain-receiving patches
    (K by render quality) within 12 m - and any interior, membrane or vehicle panel among them -
    get per-event modal synthesis; the rest are granular density layers; beyond the ground rings
    the RainBedGenerator renders a statistically matched diffuse bed directly in the SH domain.
    """
    sr = SAMPLE_RATE
    kb = KernelBank(sr)
    if precip_types is None:
        types = _precip_types(scene)
    elif isinstance(precip_types, str):
        types = set(_ALL_TYPES) if precip_types.lower() == "all" else {precip_types.lower()}
    else:
        types = {str(t).lower() for t in precip_types}
    want_rain = bool(types - {"none", "snow", "graupel"})
    want_hail = bool(types & set(HAIL_TYPES))
    surf = {str(k).lower() for k, v in (scene.place.surfaces or {}).items() if float(v) > 0}
    snow_place = str(scene.place.ground or "").lower().startswith("snow") or any("snow" in s for s in surf)
    want_snow = bool(types & set(SNOW_TYPES)) or snow_place

    k_full = LOD_K.get(str(getattr(scene.render, "quality", "studio")).lower(), 24)
    patches = [p for p in layout if p.kind in RAIN_KINDS and p.exposure > 0.0]
    ranked = sorted(patches, key=lambda p: (-_loudness_score(p), p.id))
    full_ids = {p.id for p in ranked[:k_full]
                if p.distance <= FULL_MAX_DIST_M or p.interior or p.kind in ("membrane", "vehicle_body")}

    gens: list = []
    if want_rain:
        for p in patches:
            gens.append(RainPatchGenerator(scene, p, rngf, lod="full" if p.id in full_ids else "granular",
                                           kernels=kb, sr=sr))
    if want_rain or want_hail:
        gens.append(RainBedGenerator(scene, layout, rngf, sr=sr))

    # drips: every layout drip edge, plus crown drips under up to three nearby trees
    roof_area = _roof_area_from_layout(layout)
    for p in layout:
        if p.kind == "drip_edge":
            gens.append(DripGenerator(scene, p, rngf, canopy=False, kernels=kb, sr=sr, layout=layout,
                                      roof_area_m2=roof_area))
    with_edges = {str(p.params.get("source")) for p in layout if p.kind == "drip_edge"}
    crowns = []
    for p in layout:
        if p.kind != "vegetation":
            continue
        horiz = math.hypot(p.position[0], p.position[1])
        species = str(p.params.get("species", "") or "").lower()
        if p.params.get("canopy"):
            crowns.append((0, horiz, p.id, p))
        elif species not in ("", "grass") and p.id not in with_edges and horiz <= 15.0:
            crowns.append((1, horiz, p.id, p))
    for _, _, _, p in sorted(crowns, key=lambda e: e[:3])[:3]:
        gens.append(DripGenerator(scene, p, rngf, canopy=True, kernels=kb, sr=sr, layout=layout,
                                  roof_area_m2=roof_area))

    # runoff: gutter channels share the roof flow, downspouts carry it again
    gutters = [p for p in layout if p.kind == "gutter"]
    pipes = [p for p in gutters if _is_downspout(p)]
    channels = [p for p in gutters if not _is_downspout(p)]
    for group in (channels, pipes):
        for p in group:
            gens.append(GutterGenerator(scene, p, rngf, n_gutters=len(group), kernels=kb, sr=sr))

    near = [p for p in ranked if p.distance <= 15.0][:k_full]
    if want_hail:
        gens.append(HailGenerator(scene, near, rngf, kernels=kb, sr=sr))
    if want_snow:
        gens.append(SnowGenerator(scene, [p for p in near if p.distance <= 8.0], rngf, kernels=kb, sr=sr))
    return gens
