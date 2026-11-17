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
- Level of detail (Zhang, Raghuvanshi, Snyder & Marschner 2019; MASTER_PLAN 7.5, 9.5): all rain
  patches of a scene are rendered by one vectorised field. Near "voice" panels get per-drop force
  pulses through shared modal banks; far patches get pre-baked grains for their rare loud drops.
  Per patch and drop class, a class whose expected overlap (impacts/s x response duration) is high,
  or whose single impacts are inaudible next to the patch's loudest class, is rendered as matched
  Gaussian noise with the same power spectrum (Campbell's theorem), so loudness and timbre match
  the per-drop path in expectation; render quality sets the thresholds, caps and bank variants.
  Beyond the rings a diffuse bed is synthesized directly in the SH domain.
- Drips (integrate-and-fire, raingen.physics.wetness): a plink inside the regular-entrainment
  band 41.3 Fr^0.179 <= We <= 48.3 Fr^0.247 (Oguz & Prosperetti 1990), a splat otherwise; fed
  by rain, routed roof flow, canopy release or stored water, so they outlast the rain.
- Gutters/downspouts: bubble cloud + turbulent rush exciting the open pipe's resonances
  f_k = k c / 2(L + 0.6 d), glugs in slug flow, an overflow sheet above the design capacity.
  No primary acoustic literature: a hypothesis model.
- Interior panels struck on their far side (a roof above an attic and a ceiling, a deck) reach the
  room through the attic's diffuse build-up and the field-incidence mass law TL = 20 log10(m'' f) - 47 dB
  of the ceiling, capped by its coincidence region (Cremer & Heckl).
- Hail / ice pellets / graupel: Hertz contact with an ice-crushing limit, NOAA size and
  fall-speed bands, bounces with restitution. Snow: flake impulses m V spread over a crumple
  time (near silent), plus blowing-snow saltation hiss in cold wind.

Units: every signal is sound pressure in pascals at the listener (digital 1.0 = 1 Pa = 94 dB SPL).
All randomness comes from RngFactory.get(layer, block), one generator per layer and block (Philox
construction is ~0.1 ms): the rain field draws every patch's impacts, and its matched-noise table position,
from 'rain/field' plus one draw for its Cox modulator bank ('.../cox'); hail and snow draw all their patches
from 'hail' / 'snow' in patch order; drips and gutters use layers named after their patch ('drip/{id}/sound',
'gutter/{id}'). Noise beds read the deterministic noise table of raingen.synth.kernels at drawn positions.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft
from scipy import signal

from raingen.core.constants import (
    N_AMBI,
    RHO_WATER,
    SAMPLE_RATE,
    SIGMA_WATER,
    G,
    speed_of_sound,
)
from raingen.core.rng import RngFactory
from raingen.core.types import (
    BlockContext,
    Emission,
    EventMarker,
    SourcePatch,
    WeatherState,
    bearing_to_azimuth,
)
from raingen.physics import dsd as _dsd
from raingen.physics.atmosphere import (
    AirAbsorptionFilter,
    air_absorption_sos,
    spreading_gain,
)
from raingen.physics.wetness import DripNode

from .events import (
    RateModulator,
    RateModulatorBank,
    cox_offsets,
    ramp_positions,
    thinned_offsets,
)
from .kernels import (
    D_REF_MM,
    FAM_HERTZ,
    FAM_PULSE,
    V_REF,
    KernelBank,
    ModalBank,
    ModalBatch,
    OverlapAdd,
    RowOLA,
    StatefulSOS,
    bubble_damping,
    fade_tail,
    fast_len,
    force_peak,
    hertz_contact,
    impact_pulse,
    noise_rows,
    pulse_shape,
    radiation_sos,
    render_binned,
    scatter_kernels,
)
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
_CAL_CACHE: dict = {}
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


CEILING_MASS_KG_M2 = 10.0        # 12.5 mm gypsum board ceiling (8.5-10.5 kg/m^2, Bies & Hansen table)
# The board's coincidence dip (fc ~2.5-3.2 kHz) caps the mass law: a single 12.5 mm board measures
# R ~26-30 dB from 2 to 4 kHz (Rw 28; NRC IR-761, Halliwell et al. 1998), and in situ flanking through
# the walls, downlights and the loft hatch takes the apparent R' ~2 dB lower (ISO 12354-1). Rain on
# tiles peaks at ~3.5 kHz, right in that dip.
CEILING_TL_MAX_DB = 26.0
# A sill under the glazing drives the pane with its near field: ~1/4 of its power falls on the glass,
# while the facade level difference of raingen.synth.places is stated for a distant source whose
# intensity at the listener, W / (2 pi r^2), meets the pane at an average obliquity of ~1/2.
SILL_GLAZING_SHARE = 0.25
WINDOW_AREA_M2 = 1.5             # bedroom window (raingen.physics.layout room window)
ATTIC_DEPTH_M = 1.2              # mean height of the attic space between roof and ceiling
ATTIC_ABSORPTION = 0.4           # Sabine absorption of the attic per m^2 of roof (insulated floor, alpha ~0.8)


def _mass_law_sos(mass_kg_m2: float, tl_max_db: float, sr: int) -> np.ndarray:
    """First-order shelf following the field-incidence mass law TL = 20 log10(m f) - 47 dB from
    TL = 0 (f_z = 10^(47/20) / m) up to a coincidence-limited plateau tl_max_db (Cremer & Heckl)."""
    f_z = 10.0 ** (47.0 / 20.0) / max(mass_kg_m2, 1e-3)
    f_p = f_z * 10.0 ** (max(tl_max_db, 0.0) / 20.0)
    f_p = min(f_p, 0.45 * sr)
    w_z, w_p = 2.0 * math.pi * f_z, 2.0 * math.pi * f_p
    b, a = signal.bilinear([1.0 / w_p, 1.0], [1.0 / w_z, 1.0], fs=sr)
    return signal.tf2sos(b, a)


def _interior_transfer(patch: SourcePatch, sr: int):
    """(gain, sos or None): what reaches the room from an interior patch whose struck face is not
    the face radiating into the listener's space.

    Roof above an attic and a ceiling ('through_ceiling'): the roof's underside fills the attic, whose
    level re the patch's free field at r_eq is r_eq^2 (2 pi * 4 / A_attic + 1 / h_attic^2) (diffuse
    field L_W + 10 log(4/A) plus the direct field at the attic depth; A_attic = 0.4 m^2 per m^2 of
    roof); the ceiling then transmits with the field-incidence mass law TL = 20 log10(m'' f) - 47 dB
    (Cremer & Heckl) of a gypsum board, capped by its coincidence region, into a room whose
    10 log(S/A) term is ~0 dB for a furnished bedroom. Decks / slabs ('thickness_m'): concrete mass law
    (the value at 100 Hz plus the 6 dB/octave slope). Double / triple glazing: the struck outer pane
    reaches the room through the cavity and the inner pane - a broadband loss and a high-frequency
    roll-off (~10 dB below single glazing for rain on skylights; provisional).
    An exterior window sill (params 'sill', in a room) is not interior, but its gain converts its free
    field at the listener into the near field its pane receives (SILL_GLAZING_SHARE).
    """
    p = patch.params or {}
    if not patch.interior:
        if p.get("sill") and str(p.get("shelter", "")) in ("room_closed", "room_open"):
            # facade near field: the envelope loss of ShelterTransfer applies to the sound the pane
            # receives, SILL_GLAZING_SHARE W, not to the free field W / (2 pi r_eq^2) at the listener
            r_eq, _ = _patch_geometry(patch)
            s_win = float(p.get("window_area_m2", WINDOW_AREA_M2))
            return math.sqrt(SILL_GLAZING_SHARE * 2.0 * math.pi * r_eq * r_eq / (0.5 * max(s_win, 0.1))), None
        return 1.0, None
    if p.get("through_ceiling"):
        r_eq, _ = _patch_geometry(patch)
        a_attic = max(ATTIC_ABSORPTION * float(patch.area_m2), 1.0)
        h = float(p.get("attic_depth_m", ATTIC_DEPTH_M))
        build_up = r_eq * math.sqrt(2.0 * math.pi * 4.0 / a_attic + 1.0 / (h * h))
        mass = float(p.get("ceiling_mass_kg_m2", CEILING_MASS_KG_M2))
        tl_max = float(p.get("ceiling_tl_max_db", CEILING_TL_MAX_DB))
        return build_up, _mass_law_sos(mass, tl_max, sr)
    if p.get("thickness_m"):
        mass = SLAB_RHO_KG_M3 * float(p["thickness_m"])
        f0 = 100.0
        tl0 = max(20.0 * math.log10(mass * f0) - 47.0, 0.0)
        return 10.0 ** (-tl0 / 20.0), signal.butter(1, f0, "lowpass", fs=sr, output="sos")
    glazing = str(p.get("glazing", "")).lower()
    if glazing in ("double", "triple"):
        gain, fc = (0.4, 3500.0) if glazing == "double" else (0.25, 2500.0)
        return gain, signal.butter(1, fc, "lowpass", fs=sr, output="sos")
    return 1.0, None


def _air(filt: AirAbsorptionFilter, y: np.ndarray, distance: float, w: WeatherState) -> np.ndarray:
    """ISO 9613-1 air absorption of one source's signal; skipped nearer than AIR_MIN_DIST_M (< 1 dB at 16 kHz)."""
    if distance < AIR_MIN_DIST_M:
        return y
    return filt.process(y, distance, w.temperature_c, w.humidity_pct)


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


# ---------------------------------------------------------------------------
# water impacts (rain on water/puddles, drip splats, hail into water)

_EPS_LO = np.array([0.01, 0.02, 0.005, 0.03, 0.04])
_EPS_HI = np.array([0.02, 0.07, 0.02, 0.10, 0.12])
_EPS2 = (_EPS_HI ** 2 - _EPS_LO ** 2) / (2.0 * np.log(_EPS_HI / _EPS_LO))     # E[eps^2], log-uniform eps
_DELAY_LO_MS = np.array([1.0, 0.5, 2.0, 6.0, 10.0])
_DELAY_HI_MS = np.array([2.0, 2.5, 8.0, 20.0, 30.0])
_VL_EXTRA_BUBBLES = 0.8          # Poisson mean of extra bubbles per entraining very large drop


def _entrainment_probability(d, ci, wind_mps: float) -> np.ndarray:
    p = np.zeros(d.shape)
    wsup = 1.0 - _smoothstep(wind_mps, 7.5, 10.0)          # small-drop ring dies at 8-10 m/s
    p[ci == 1] = 0.95 * wsup
    p[ci == 2] = 0.03
    big = ci >= 3
    if np.any(big):
        p[big] = 0.05 + 0.60 * _smoothstep(d[big], 2.2, 4.0)
    return p


def _small_bubble_freq(d):
    """Regular-entrainment ring of the small class: ~22 kHz at 0.8 mm falling to ~13 kHz at 1.1 mm."""
    return np.clip(22e3 - (np.asarray(d, dtype=float) - 0.8) / 0.3 * 9e3, 12e3, 24e3)


def _large_bubble_r1(d):
    """Smallest radius of the large-drop bubble band (2-10 kHz, lower for bigger drops)."""
    return 3.28 / (10e3 * (2.2 / np.maximum(np.asarray(d, dtype=float), 2.2)) ** 0.8)


def _bubble_radius(rng, d, ci) -> np.ndarray:
    r = np.full(d.shape, 3e-4)
    m = ci == 1
    if np.any(m):   # regular entrainment: ~22 kHz at 0.8 mm falling to ~13 kHz at 1.1 mm
        f = _small_bubble_freq(d[m]) * np.exp(rng.normal(0.0, 0.05, int(m.sum())))
        r[m] = 3.28 / f
    m = (ci == 0) | (ci == 2)
    if np.any(m):
        r[m] = _loguniform(rng, 3e-4, 1.2e-3, int(m.sum()))
    m = ci >= 3
    if np.any(m):   # large: 2-10 kHz band, lower for bigger drops; r^-2.5 within it
        r1 = _large_bubble_r1(d[m])
        r[m] = _power_law(rng, r1, 4.0 * r1, 2.5, int(m.sum()))
    return r


def _bubble_kernel_fn(kb: KernelBank):
    return lambda k: kb.bubble_shape(k // 512, (k // 8) % 64, k % 8)


def _bubble_keys(kb: KernelBank, radius, xi, damp_bin: int):
    return (kb.radius_bin(radius) * 64 + kb.xi_bin(xi)) * 8 + int(np.clip(damp_bin, 0, 7))


def _damp_bin(depth_mm):
    """Extra bubble damping (0.01 steps) from a shallow puddle bottom."""
    return np.round(0.02 * (1.0 - _smoothstep(depth_mm, 3.0, 15.0)) / 0.01).astype(np.int64)


def _splat_amp(d, u, ci):
    return SPLAT_PA * (d / 4.0) ** 2 * (u / 8.7) ** 3 * np.where(ci == 4, 1.0, 0.4)


def _splat_dur(d):
    return np.clip(0.003 * d / 3.0, 0.002, 0.012)


def _water_events(kb: KernelBank, rng, off, d, u, ci, sc, *, click_pa, depth_mm=1000.0, wind_mps: float = 0.0,
                  water_quality: float = 1.0, sr: int = SAMPLE_RATE, p_scale: float = 1.0):
    """Impact click + probabilistic bubble (+ splat for big drops) of impacts on a water surface.

    Returns (offsets, kernel codes, amplitudes in Pa @1 m, source-impact index) for the batched
    renderer; bubble offsets include their delay after the impact, so they may lie past the block.
    click_pa and depth_mm may be scalars or per-impact arrays.
    """
    off = np.asarray(off, dtype=np.int64)
    n_ev = int(off.size)
    if n_ev == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0), z
    d = np.asarray(d, dtype=float)
    u = np.asarray(u, dtype=float)
    ci = np.asarray(ci, dtype=np.int64)
    sc = np.broadcast_to(np.asarray(sc, dtype=float), off.shape)
    cpa = np.broadcast_to(np.asarray(click_pa, dtype=float), off.shape)
    depth = np.broadcast_to(np.asarray(depth_mm, dtype=float), off.shape)
    # impact click ~ D U^3 (U^3 is inside the click kernel)
    amp = cpa * (d / D_REF_MM) * sc * np.exp(rng.normal(0.0, 0.2, n_ev))
    amp = np.where(ci == 0, 0.3 * amp, amp)
    offs, codes, amps, src = [off], [kb.click_code(u)], [amp], [np.arange(n_ev)]

    # entrained bubbles
    p = _entrainment_probability(d, ci, wind_mps) * float(np.clip(water_quality, 0.0, 1.0)) * p_scale
    hit = rng.random(n_ev) < p
    idx = np.nonzero(hit)[0]
    vl = np.nonzero(hit & (ci == 4))[0]
    if vl.size:
        extra = rng.poisson(_VL_EXTRA_BUBBLES, vl.size)
        idx = np.concatenate([idx, np.repeat(vl, extra)])
    if idx.size:
        fd = _smoothstep(depth[idx], 1.5, 8.0)               # shallow puddles pinch bubbles off
        idx, fd = idx[fd > 0.0], fd[fd > 0.0]
    if idx.size:
        m = idx.size
        cls = ci[idx]
        radius = _bubble_radius(rng, d[idx], cls)
        xi = np.clip(0.1 * np.exp(rng.normal(0.0, 0.25, m)), 0.03, 0.3)
        eps = _loguniform(rng, _EPS_LO[cls], _EPS_HI[cls], m)
        delay = ((_DELAY_LO_MS[cls] + rng.random(m) * (_DELAY_HI_MS[cls] - _DELAY_LO_MS[cls]))
                 * 1e-3 * sr).astype(np.int64)
        offs.append(off[idx] + delay)
        codes.append(kb.bubble_code(radius, xi, _damp_bin(depth[idx])))
        amps.append(BUBBLE_PA_PER_M * eps * sc[idx] * fd)
        src.append(idx)

    # broadband splat of very large drops
    sp = np.nonzero((ci == 4) | ((ci == 3) & (d >= 3.0)))[0]
    if sp.size:
        amp_s = _splat_amp(d[sp], u[sp], ci[sp]) * sc[sp] * np.exp(rng.normal(0.0, 0.25, sp.size))
        offs.append(off[sp])
        codes.append(kb.splash_code(_splat_dur(d[sp]), rng.integers(0, kb.SPLASH_VARIANTS, sp.size), True))
        amps.append(amp_s)
        src.append(sp)
    return np.concatenate(offs), np.concatenate(codes), np.concatenate(amps), np.concatenate(src)


def _render_into(kb: KernelBank, ola: OverlapAdd, offs, codes, amps):
    """Render (offset, code, amp) events of one stream into an OverlapAdd (block start = offset 0)."""
    if np.size(offs) == 0:
        return
    offs = np.asarray(offs, dtype=np.int64)
    _, Y = scatter_kernels(kb, int(offs.max()) + 1, np.zeros(offs.size, dtype=np.int64), codes, offs, amps)
    ola.add(Y[0])


def _water_impacts(kb: KernelBank, rng, off, d, u, ci, sc, ola: OverlapAdd, *, click_pa: float,
                   depth_mm: float = 1000.0, wind_mps: float = 0.0, water_quality: float = 1.0,
                   sr: int = SAMPLE_RATE, p_scale: float = 1.0):
    """Click + probabilistic bubble (+ splat for big drops) for impacts on a water surface."""
    offs, codes, amps, _ = _water_events(kb, rng, off, d, u, ci, sc, click_pa=click_pa, depth_mm=depth_mm,
                                         wind_mps=wind_mps, water_quality=water_quality, sr=sr, p_scale=p_scale)
    _render_into(kb, ola, offs, codes, amps)


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
        off = np.asarray(off, dtype=np.int64).ravel()
        if off.size == 0:
            return
        codes = self.kb.pulse_code(np.broadcast_to(np.asarray(tau, dtype=float), off.shape),
                                   FAM_PULSE if family == "pulse" else FAM_HERTZ)
        if self.n_var == 1 or variant is None:
            rows = np.zeros(off.size, dtype=np.int64)
        else:
            rows = np.broadcast_to(np.asarray(variant, dtype=np.int64), off.shape)
        # sparse short pulses: time-domain scatter (no FFT whose size would change with every call)
        ur, Y = scatter_kernels(self.kb, int(off.max()) + 1, rows, codes, off, force)
        for k, row in enumerate(ur.tolist()):
            self.olas[row].add(Y[k])

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
# rain on surface patches: one vectorised field renders every patch of a scene

# Level of detail per render quality (MASTER_PLAN 7.5, 9.5). Per patch and drop class, a class whose
# expected overlap (impacts/s x effective response duration) reaches `dense` (a quarter of it for
# granular patches), or whose single impacts are > 35 dB below the patch's loudest class, is rendered
# as matched Gaussian noise: the same kernels and modal banks driven by white noise whose power equals
# the impact train's (rate x E[amplitude^2] per second), so spectrum and loudness match the per-drop
# path in expectation. The other classes keep per-drop synthesis, capped per 1024 frames; the excess
# above a cap joins the matched-noise layer (a class that cannot reach the noise layer keeps its energy
# by sqrt(expected / cap) amplitude scaling). `var` = modal-bank variants of a ground / vegetation voice
# patch (variants are shared across the patches of a material; leaves detune by 25 %, so one bank is not
# representative of a canopy and every quality keeps two); granular patches always use at least two, as
# their grains are pre-baked.
QUALITY_LOD = {
    "mobile": {"k": 6, "voice_cap": (30, 80, 80, 40, 20), "gran_cap": (6, 16, 16, 10, 6), "dense": 3.0,
               "var": (1, 2)},
    "game": {"k": 12, "voice_cap": (60, 160, 160, 80, 40), "gran_cap": (10, 24, 24, 16, 10), "dense": 5.0,
             "var": (1, 2)},
    "studio": {"k": 24, "voice_cap": (150, 400, 400, 200, 100), "gran_cap": (20, 50, 50, 30, 16), "dense": 10.0,
               "var": (2, 2)},
    "cinematic": {"k": 48, "voice_cap": (600, 1500, 1500, 800, 400), "gran_cap": (40, 100, 100, 60, 30),
                  "dense": 40.0, "var": (3, 4)},
    # reference: every class per drop, no caps, studio's bank variants (tests and offline comparisons)
    "reference": {"k": 1 << 20, "voice_cap": (1e9,) * 5, "gran_cap": (1e9,) * 5, "dense": math.inf,
                  "var": (2, 2)},
}
LOD_K = {q: v["k"] for q, v in QUALITY_LOD.items() if q != "reference"}
FULL_MAX_DIST_M = 12.0
VOICE_MAX_AREA_M2 = 25.0         # larger surfaces are voices only within VOICE_NEAR_M
VOICE_NEAR_M = 4.0
SLAB_RHO_KG_M3 = 2300.0          # concrete, for decks / slabs between the impact and the listener

_T_EFF_WATER = np.array([1e-4, 5e-4, 1e-4, 3e-3, 3e-3])   # s: click, small-drop ring, click, large rings
_T_EFF_CAP = 5e-3                # onsets matter, not the whole ring, for "is it a crackle or a hiss"
_SALIENCE = 10.0 ** (-35.0 / 10.0)
_QUAD = 8                        # Gauss-Legendre diameter nodes per class for expectations (< 0.05 dB)
_SIG_FORCE = 0.25                # log-normal force jitter (sigma)
_SIG_CLICK = 0.2
_SPLASH_REPS = 6.0               # mean of the 2..10 prompt-splash micro-impacts per eligible drop
_SPLASH_A2 = (0.03 ** 2 - 0.005 ** 2) / (2.0 * math.log(0.03 / 0.005))   # E[a^2], a log-uniform 0.005..0.03
_SPLASH_TAU = math.sqrt(5e-5 * 1.5e-4)
_BLOCK_REF = 1024.0              # caps are stated per 1024 frames
_LIVE_EPS = 1e-12                # Pa; quieter outputs are dropped and their filter states flushed
AIR_MIN_DIST_M = 3.0             # ISO 9613-1 loss < 1 dB up to 16 kHz below this: no air filter


def lod_settings(quality: str) -> dict:
    return QUALITY_LOD.get(str(quality or "studio").lower(), QUALITY_LOD["studio"])


class _ClassTable:
    """Per-DSD class geometry: bounds, representative fall speeds, flux-weighted inverse CDF, quadrature."""

    def __init__(self, dsd, scheme: str):
        bounds = _dsd.class_bounds(scheme, d_max_mm=float(dsd.d_max_mm))
        lo = np.array([bounds[c][0] if c in bounds else 0.0 for c in CLASSES], dtype=float)
        hi = np.array([bounds[c][1] if c in bounds else 0.0 for c in CLASSES], dtype=float)
        lo = np.maximum(lo, float(dsd.d_min_mm))
        hi = np.minimum(hi, float(dsd.d_max_mm))
        self.valid = hi > lo
        d0, d1 = float(dsd.d_min_mm), max(float(dsd.d_max_mm), float(dsd.d_min_mm) + 1e-3)
        grid = np.linspace(d0, d1, 768)
        pdf = np.asarray(_dsd.number_density(dsd, grid), dtype=float) * _fall_speed(grid)
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))))
        self.grid, self.cdf = grid, cdf
        self.c_lo = np.interp(lo, grid, cdf)
        self.c_hi = np.interp(hi, grid, cdf)
        self.valid &= self.c_hi > self.c_lo
        gm = np.where(lo > 0.0, np.sqrt(np.maximum(lo * hi, 1e-12)), 0.5 * hi)
        self.vt_c = _fall_speed(gm)
        x, wts = np.polynomial.legendre.leggauss(_QUAD)        # nodes / weights in the flux quantile
        uq = 0.5 * (x + 1.0)
        self.wq = 0.5 * wts
        self.dq = self.sample(np.broadcast_to(uq, (len(CLASSES), _QUAD)), np.arange(len(CLASSES))[:, None])
        self.dq = np.where(self.valid[:, None], self.dq, gm[:, None])
        self.vtq = _fall_speed(self.dq)

    def sample(self, u, ci) -> np.ndarray:
        """Flux-weighted diameters (mm) of class ci for uniforms u (inverse CDF of N(D) V(D))."""
        lo, hi = self.c_lo[ci], self.c_hi[ci]
        return np.interp(lo + np.asarray(u) * (hi - lo), self.cdf, self.grid)


def _dsd_key(dsd, scheme: str):
    """Class-table key on a 2 % Dm / 0.1 mu / 0.02 mm (bounds) grid. The table is the flux-weighted size
    distribution *within* each class, so Nw (a pure scale) does not enter, and the storm envelope's
    block-to-block wobble of Dm moves the drop sizes it samples far below audibility (the class fluxes
    themselves are always used exactly)."""
    return (scheme, int(round(math.log(max(float(dsd.dm_mm), 1e-6)) / 0.02)), int(round(float(dsd.mu) / 0.1)),
            int(round(float(dsd.d_min_mm) / 0.02)), int(round(float(dsd.d_max_mm) / 0.02)))


_TEFF_CACHE: dict = {}


def _response_teff(mat: Material, sr: int) -> float:
    """Effective duration (s) of a material's impact response: (sum y^2)^2 / sum y^4, onset-capped."""
    key = (mat.name, int(sr))
    t = _TEFF_CACHE.get(key)
    if t is None:
        t = _TEFF_CACHE[key] = _response_teff_calc(mat, sr)
    return t


def _response_teff_calc(mat: Material, sr: int) -> float:
    pulse = impact_pulse(D_REF_MM, V_REF, sr)
    bank = ModalBank(mat, sr)
    if bank.n_modes:
        length = int(0.25 * sr)
        y = np.convolve(pulse, bank.impulse_response(length))[:length]
    else:
        y = pulse
    e2 = float(np.sum(y * y))
    e4 = float(np.sum(y ** 4))
    t = (e2 * e2 / e4) / sr if e4 > 0 else 1e-4
    return float(np.clip(t, 2e-4, _T_EFF_CAP))


def _bubble_color(kb: KernelBank, cls: int, d_q: np.ndarray, damp_bin: int, qw=None) -> np.ndarray:
    """Zero-phase kernel whose power spectrum is the expected unit-eps bubble power spectrum of a class.

    Radii follow the per-drop draws of _bubble_radius on weighted grids (bubble energy grows ~a^3, so
    the large-radius tail must be integrated, not sampled at a few quantiles); d_q are the class's
    quadrature diameters with weights qw (equal weights when omitted).
    """
    d_q = np.asarray(d_q, dtype=float)
    qw = np.full(d_q.size, 1.0 / d_q.size) if qw is None else np.asarray(qw, dtype=float) / float(np.sum(qw))
    g = (np.arange(24) + 0.5) / 24
    if cls == 1:                          # ring frequency from the drop diameter, 5 % jitter
        radii = (3.28 / _small_bubble_freq(d_q)[:, None] * np.exp(0.05 * np.array([-1.5, -0.5, 0.5, 1.5])
                                                                  )[None, :]).ravel()
        wts = np.repeat(qw / 4.0, 4)
    elif cls >= 3:                        # r^-2.5 on [r1, 4 r1]: log grid, weight r^-2.5 dr ~ r^-1.5
        r1 = _large_bubble_r1(d_q)[:, None]
        radii = r1 * 4.0 ** g[None, :]
        wts = radii ** -1.5
        p_d = (0.05 + 0.60 * _smoothstep(d_q, 2.2, 4.0)) * qw    # bigger drops entrain more often (and lower)
        wts = (wts / wts.sum(axis=1, keepdims=True) * (p_d / p_d.sum())[:, None]).ravel()
        radii = radii.ravel()
    else:                                 # log-uniform 0.3 .. 1.2 mm
        radii = 3e-4 * 4.0 ** g
        wts = np.full(radii.size, 1.0 / radii.size)
    ks = [kb.bubble_shape(int(kb.radius_bin(r)), 4, int(damp_bin)) for r in radii]
    L = max(len(k) for k in ks)
    nfft = fast_len(2 * L)
    S = np.zeros(nfft // 2 + 1)
    energy = 0.0
    for k, wt in zip(ks, wts):
        S += wt * np.abs(sfft.rfft(k, nfft)) ** 2
        energy += wt * float(np.sum(k * k))
    k0 = sfft.irfft(np.sqrt(S), nfft)
    k = np.concatenate([k0[-L:], k0[:L]])
    taper = max(L // 8, 2)
    win = np.ones(k.size)
    win[:taper] = np.sin(0.5 * np.pi * (np.arange(taper) + 0.5) / taper) ** 2
    win[-taper:] = win[:taper][::-1]
    k *= win
    e = float(np.sum(k * k))
    return k * math.sqrt(energy / e) if e > 0 else k


class _AirBatch:
    """ISO 9613-1 air absorption for many rows sharing one distance; redesigns crossfade over a block."""

    def __init__(self, distance_m: float, n_rows: int, sr: int):
        self.distance = float(distance_m)
        self.n_rows = int(n_rows)
        self.sr = int(sr)
        self.sos = None
        self.zi = None
        self.params = None

    def reset(self):
        if self.zi is not None:
            self.zi[:] = 0.0

    def process(self, x: np.ndarray, temp_c: float, rh_pct: float) -> np.ndarray:
        params = (float(temp_c), float(rh_pct))
        if self.sos is None:
            self.sos = air_absorption_sos(self.distance, temp_c, rh_pct, self.sr)
            self.zi = np.zeros((self.sos.shape[0], self.n_rows, 2))
            self.params = params
        t0, h0 = self.params
        if abs(params[0] - t0) > 1.0 or abs(params[1] - h0) > 0.05 * max(h0, 1.0):
            new = air_absorption_sos(self.distance, temp_c, rh_pct, self.sr)
            y0, _ = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
            zi = np.zeros((new.shape[0], self.n_rows, 2))
            y1, zi = signal.sosfilt(new, x, axis=-1, zi=zi)
            ramp = np.linspace(0.0, 1.0, x.shape[-1])
            np.copyto(zi, 0.0, where=np.abs(zi) < 1e-30)
            self.sos, self.zi, self.params = new, zi, params
            return y0 * (1.0 - ramp) + y1 * ramp
        if self.sos.shape[0] == 1 and not np.any(self.sos[0, 1:3]) and not np.any(self.sos[0, 4:]):
            return x * self.sos[0, 0]
        y, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        np.copyto(self.zi, 0.0, where=np.abs(self.zi) < 1e-30)        # no denormal tails
        return y


def _bank_variation(rngf: RngFactory, name: str, spread: float, v: int):
    """(detune, damping) factors of modal-bank variant v of a material (variant 0 is the nominal bank)."""
    nm = len(get_material(name).modes)
    if v == 0 or nm == 0:
        return 1.0, 1.0
    rng = rngf.get(f"rain/bank/{name}/{spread:g}/{v}", 0)
    return np.exp(rng.normal(0.0, spread, nm)), np.exp(rng.normal(0.0, 0.15, nm))


def _variant_modes(rngf: RngFactory, name: str, spread: float, v: int):
    """Modes of bank variant v of a material (variant 0: the material itself): detuned and re-damped by
    `_bank_variation`, each gain scaled by 1 / sqrt(damping factor) so every mode keeps its impulse energy
    (~ g^2 T60): variants differ in pitch and decay, not in loudness, and match the nominal bank's level."""
    mat = get_material(name)
    det, dmp = _bank_variation(rngf, name, spread, v)
    if np.ndim(det) == 0:
        return mat
    arr = np.asarray(mat.modes, dtype=float).reshape(-1, 3)
    return [(f * a, t * b, g / math.sqrt(b)) for (f, t, g), a, b in zip(arr.tolist(), det.tolist(), dmp.tolist())]


def _modal_ir(modes, sr: int, detune=1.0, damping=1.0):
    """Impulse response of a modal bank (a material or a variant's mode list), truncated at its longest
    T60 (<= 0.25 s)."""
    bank = ModalBank(modes, sr, detune=detune, damping=damping)
    if not bank.n_modes:
        return None
    length = int(np.clip(float(np.max(bank.t60)), 0.01, 0.25) * sr)
    return bank.impulse_response(length)


def _grain(kb: KernelBank, mat: Material, tau_s: float, detune=1.0, damping=1.0, ir=None) -> np.ndarray:
    """Pre-baked impact grain in Pa @1 m per newton of peak force: pulse -> modes -> radiation -> calibration.

    `ir` is the modal bank's impulse response (`_modal_ir`) when the caller caches it per bank variant."""
    sr = kb.sr
    pulse = pulse_shape(tau_s, sr)
    if ir is None:
        ir = _modal_ir(mat, sr, detune, damping)
    if ir is not None:
        k = signal.oaconvolve(pulse, ir) if pulse.size > 1 else pulse[0] * ir
    else:
        k = pulse.copy()
    rad = radiation_sos(mat, sr)
    if rad.shape[0]:
        k = signal.sosfilt(rad, np.concatenate([k, np.zeros(int(0.004 * sr))]))
    k *= _solid_calibration(mat, sr)
    env = np.abs(k)
    tail = np.nonzero(env > 1e-3 * env.max())[0]
    k = k[: int(tail[-1]) + 1] if tail.size else k[:1]
    return fade_tail(k, 0.1)


class _Shaper:
    """Matched-noise rows: per row Gaussian noise whose power spectrum is the sum of its components' spectra.

    A row's components are (kernel code, variance per sample). The noise is synthesized frame by frame in
    the frequency domain (weighted overlap-add): random complex spectra (from the deterministic noise
    table) times the row's magnitude sqrt(sum_i v_i |K_i(f)|^2 / V) (energy-preserving pooled spectra),
    one inverse FFT per row and block, a power-complementary window, and a per-frame gain sqrt(V m) at the
    block's end, so successive frames crossfade the Cox modulation. Output power spectrum = sum_i v_i
    |K_i|^2 = that of a Poisson train of kernel i at rate x E[amp^2] (Campbell's theorem). A row's
    magnitude is rebuilt only when its mix changes. Frames span 2 m blocks (m = ceil(512 / n)), so the
    spectral resolution stays <= 47 Hz for any block length.
    """

    W_TOL = 0.05                 # rebuild a row's magnitude when a component's share moves by more than this
    MIN_HALF = 512               # frames are at least 2 x 512 samples long

    def __init__(self, kb: KernelBank, half_len: int = 0):
        self.kb = kb
        self.cache: dict = {}
        self._grp = None
        self._spec = None            # [frame length, shares the row magnitudes were built for, magnitudes]
        self._win = {}

    def frame(self, n: int) -> tuple[int, int]:
        """(frame length F, overlap factor m) for block length n: F = 2 m n."""
        m = max(1, -(-self.MIN_HALF // max(int(n), 1)))
        return 2 * m * int(n), m

    def _window(self, F: int, m: int) -> np.ndarray:
        w = self._win.get((F, m))
        if w is None:
            # periodic Hann at hop F / 2m sums to m: sqrt(hann / m) is power-complementary
            w = np.sqrt((0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(F) / F)) / m)
            self._win[(F, m)] = w
        return w

    def _magnitude(self, codes, w, half: int) -> np.ndarray:
        psd = np.zeros(half + 1)
        for c, wi in zip(codes, w):
            psd += wi * self.kb.pooled_power(int(c), half)[0]
        mag = np.sqrt(np.maximum(psd, 0.0))
        mag[0] = 0.0                 # no DC: every rain kernel ends in a radiation high-pass or is a dipole
        return mag

    def _row_magnitude(self, r: int, cs, w, half: int, built: dict):
        """Magnitude of row r for codes cs with shares w: the row's own (or one built in this pass for the
        same mix) while every share stays within W_TOL of the one it was built for."""
        key = tuple(cs.tolist())
        ent = self.cache.get(r)
        if ent is None or ent[0] != key or ent[2] != half or float(np.max(np.abs(ent[1] - w))) > self.W_TOL:
            ent = None
            for cand in built.get(key, ()):
                if float(np.max(np.abs(cand[1] - w))) <= self.W_TOL:
                    ent = cand
                    break
            if ent is None:
                ent = (key, w.copy(), half, self._magnitude(cs, w, half))
                built.setdefault(key, []).append(ent)
            self.cache[r] = ent
        return ent

    def render(self, n: int, rows, codes, var, m0, m1, noise_base: int, layout_key=None):
        """rows / codes / var (per sample, before the Cox modulation) entries; m1[row] the Cox value at the
        block's end (m0, its start, is reached by the previous frame's fade).

        `layout_key` (optional) names the (rows, codes) layout: while it repeats, the grouping is reused
        and only the variances are summed. Rows whose variance is zero are skipped. Returns (rows_u, Y):
        Y (len(rows_u), F) is this block's frame, to be overlap-added at the block start.
        """
        grp = self._grp
        if layout_key is None or grp is None or grp[0] != layout_key:
            pid = (np.asarray(rows, dtype=np.int64) << 44) | np.asarray(codes, dtype=np.int64)
            uq, inv = np.unique(pid, return_inverse=True)
            r_all, c_all = uq >> 44, uq & ((1 << 44) - 1)
            rows_u, start = np.unique(r_all, return_index=True)
            counts = np.diff(np.append(start, uq.size))
            grp = (layout_key, np.asarray(inv).ravel(), uq.size, start, counts, rows_u, c_all)
            self._grp = grp if layout_key is not None else None
            self._spec = None
        _, inv, nu, start, counts, rows_u, c_all = grp
        F, m = self.frame(n)
        half = F // 2
        v = np.bincount(inv, weights=var, minlength=nu)
        V = np.add.reduceat(v, start)
        act = V > 0.0
        if not np.any(act):
            return rows_u[:0], np.zeros((0, F))
        w_all = v / np.repeat(np.where(act, V, 1.0), counts)
        sp = self._spec
        if sp is None or sp[0] != F:
            sp = self._spec = [F, np.full(nu, np.inf), np.zeros((rows_u.size, half + 1))]
        dev = np.maximum.reduceat(np.abs(w_all - sp[1]), start)          # largest share drift per row
        redo = np.nonzero(act & ~(dev <= self.W_TOL))[0]
        if redo.size:
            stop = np.append(start[1:], nu)
            built: dict = {}
            for k in redo.tolist():
                a, b = start[k], stop[k]
                ent = self._row_magnitude(int(rows_u[k]), c_all[a:b], w_all[a:b], half, built)
                sp[1][a:b] = ent[1]
                sp[2][k] = ent[3]
        ia = np.nonzero(act)[0]
        r_a = rows_u[ia]
        m1 = np.asarray(m1, dtype=float)
        m1 = np.full(ia.size, float(m1)) if m1.ndim == 0 else m1[r_a]
        # random spectra with E|Z_k|^2 = F (irfft then gives unit variance per unit kernel energy), the
        # row magnitude and the frame gain sqrt(V m) folded into one real factor per bin
        z = noise_rows(noise_base + np.arange(ia.size, dtype=np.int64) * (2 * half + 2), 2 * half + 2)
        a = sp[2][ia] * (math.sqrt(F / 2.0) * np.sqrt(V[ia] * np.maximum(m1, 0.0)))[:, None]
        a[:, half] *= math.sqrt(2.0)                      # the Nyquist bin is real
        Z = np.empty((ia.size, half + 1), dtype=complex)
        np.multiply(z[:, : half + 1], a, out=Z.real)
        np.multiply(z[:, half + 1:], a, out=Z.imag)
        Z.imag[:, half] = 0.0
        Y = sfft.irfft(Z, F, axis=-1)
        Y *= self._window(F, m)[None, :]
        return r_a, Y


class _RainField:
    """Vectorised renderer behind every RainPatchGenerator of a scene (MASTER_PLAN 7.5, 9.5).

    The first patch generator asked for a block renders all member patches at once: one Cox
    modulator bank and one event draw per block for the whole field; sparse impacts scattered as
    short kernels; dense classes as one matched-noise row per patch; one batched modal-bank
    recursion per (material, variant) for near "voice" panels; one radiation filter per material and
    one air-absorption filter per distance group. Far ("granular") solid patches use pre-baked grains
    (pulse -> modes -> radiation) for their few sparse big drops. Every patch then picks up its own
    Emission. Results do not depend on which member asks first; the same seed gives the same output.
    """

    def __init__(self, rngf: RngFactory, quality: str = "studio", kernels: KernelBank | None = None,
                 sr: int = SAMPLE_RATE, layer: str = "rain/field"):
        self.rngf = rngf
        self.quality = str(quality or "studio").lower()
        self.lod = lod_settings(self.quality)
        self.layer = layer
        self.members: list = []
        self._kb = kernels
        self.sr = int(sr)
        self._built_sr = None
        self._key = None
        self._epoch = 0
        self._out: list = []

    def add(self, gen) -> int:
        self.members.append(gen)
        self._built_sr = None
        return len(self.members) - 1

    # -- setup ---------------------------------------------------------------
    def _build(self, sr: int):
        self.sr = int(sr)
        self.kb = self._kb if (self._kb is not None and self._kb.sr == self.sr) else KernelBank(self.sr)
        M = self.members
        P = len(M)
        self.P = P
        lod = self.lod
        self.is_water = np.array([g.kind == "water" for g in M], dtype=bool)
        self.voice = np.array([g.lod == "full" for g in M], dtype=bool)
        self.modal = self.voice & ~self.is_water          # near solid panels: per-drop force -> modal banks
        nrm = np.array([g.normal for g in M], dtype=float).reshape(P, 3)
        self.nx, self.ny, self.nz = nrm[:, 0], nrm[:, 1], nrm[:, 2]
        self.base = np.array([g.patch.area_m2 * max(g.patch.exposure, 0.0) for g in M], dtype=float)
        self.spreading = np.array([float(spreading_gain(g.r_eq)) for g in M])
        self.caps = np.array([lod["voice_cap"] if v else lod["gran_cap"] for v in self.voice],
                             dtype=float).reshape(P, len(CLASSES))
        # far solid patches: only the rare impacts that stand out of their own bed stay per-drop (as grains)
        self.dense_thr = np.where(self.voice, lod["dense"], np.where(self.is_water, 0.25, 0.02) * lod["dense"])
        mats = [g.mat for g in M]
        self.mats = mats
        self.hiss_pa = np.array([float(db_to_pa(m.level_db + FILM_HISS_REL_DB)) for m in mats])
        self.click_pa = np.array([g.click_pa for g in M])
        self.kind = [g.kind for g in M]
        self.snow = np.array([k == "snow" for k in self.kind])
        self.hiss_ok = np.array([k in ("solid", "porous", "membrane") for k in self.kind])
        self.hiss_scale = np.where(np.array([k == "membrane" for k in self.kind]), 0.5, 1.0)
        self.splash_p = np.array([m.splash if (m.splash > 0.0 and m.hardness >= 0.45 and g.kind not in ("snow", "water"))
                                  else 0.0 for g, m in zip(M, mats)])
        self.puddle_ok = np.array([k not in ("vegetation", "membrane", "snow", "water") for k in self.kind])
        self.porous = np.array([k == "porous" for k in self.kind])
        self.mat_names = [m.name for m in mats]
        self._uniq_names = sorted(set(self.mat_names))
        self._name_idx = np.array([self._uniq_names.index(nm) for nm in self.mat_names], dtype=np.int64)
        self._water_pats = np.nonzero(self.is_water)[0].tolist()
        teff_cache: dict = {}
        teff = np.zeros((P, len(CLASSES)))
        for i, g in enumerate(M):
            if g.kind == "water":
                teff[i] = _T_EFF_WATER
            else:
                t = teff_cache.get(g.mat.name)
                if t is None:
                    t = _response_teff(g.mat, self.sr)
                    teff_cache[g.mat.name] = t
                teff[i] = t
        self.teff = teff
        # force rows (voice solid panels): (patch, variant); modal banks shared per (material, spread, variant)
        row_patch, n_var, row0 = [], np.zeros(P, dtype=np.int64), np.full(P, -1, dtype=np.int64)
        groups: dict = {}
        g_rows: list = []
        self.spread_p = np.zeros(P)
        for i, g in enumerate(M):
            if self.is_water[i]:
                continue
            if g.single_panel:
                nv, spread = 1, 0.03
            else:
                nv = lod["var"][1] if g.kind == "vegetation" else lod["var"][0]
                if not self.modal[i]:
                    nv = max(nv, 2)          # grains are pre-baked: a second variant costs nothing per block
                spread = 0.25 if g.kind == "vegetation" else 0.08
            n_var[i] = nv
            self.spread_p[i] = spread
            if not self.modal[i]:
                continue
            row0[i] = len(row_patch)
            for v in range(nv):
                key = (g.mat.name, spread if v else 0.0, v)
                gi = groups.get(key)
                if gi is None:
                    gi = len(g_rows)
                    groups[key] = gi
                    g_rows.append([])
                g_rows[gi].append(len(row_patch))
                row_patch.append(i)
        self.n_var, self.row0 = n_var, row0
        self.row_patch = np.array(row_patch, dtype=np.int64)
        self.S = len(row_patch)
        self.row_starts = np.array([row0[i] for i in range(P) if self.modal[i]], dtype=np.int64)
        self.modal_pats = np.nonzero(self.modal)[0]
        self.banks = []
        for key, gi in sorted(groups.items(), key=lambda kv: kv[1]):
            name, spread, v = key
            bank = ModalBank(_variant_modes(self.rngf, name, spread, v), self.sr)
            self.banks.append((np.array(g_rows[gi], dtype=np.int64), ModalBatch(bank, len(g_rows[gi]))))
        # radiation + calibration per material (voice solid patch subsets)
        self.rad = []
        spos = {int(p): k for k, p in enumerate(self.modal_pats)}
        for name in sorted({self.mat_names[i] for i in self.modal_pats}):
            pats = np.array([i for i in self.modal_pats if self.mat_names[i] == name], dtype=np.int64)
            mat = get_material(name)
            sos = radiation_sos(mat, self.sr)
            zi = np.zeros((sos.shape[0], pats.size, 2)) if sos.shape[0] else None
            sel = np.array([spos[int(p)] for p in pats], dtype=np.int64)
            self.rad.append([pats, sos, zi, _solid_calibration(mat, self.sr), sel])
        # interior transfer and air absorption
        self.xfer = []
        for i, g in enumerate(M):
            if g.xfer is not None or g.xfer_gain != 1.0:
                self.xfer.append((i, g.xfer_gain, StatefulSOS(g.xfer.sos) if g.xfer is not None else None))
        # air absorption: 1/6-octave distance groups (< 0.5 dB error at 16 kHz and 20 m). The pressure-domain
        # rows get it folded into their kernels and noise spectra (no filter per block); the modal panels'
        # outputs are filtered per group
        step = math.log(2.0) / 6.0
        dq = np.array([int(round(math.log(max(g.r_eq, 0.1)) / step)) for g in M], dtype=np.int64)
        groups_q = sorted(q for q in set(dq.tolist()) if math.exp(q * step) >= AIR_MIN_DIST_M)
        self.air_d = [math.exp(q * step) for q in groups_q]
        self.air_g = np.array([groups_q.index(q) if q in groups_q else -1 for q in dq.tolist()], dtype=np.int64)
        self.air = []
        for gi, d in enumerate(self.air_d):
            pats = np.nonzero((self.air_g == gi) & self.modal)[0]
            if pats.size:
                self.air.append((pats, _AirBatch(d, pats.size, self.sr)))
        self._air_th = None                  # (temperature, RH) the folded air kernels were built for
        self._air_map: dict = {}
        self.force_ola = RowOLA(max(self.S, 1))
        self.direct_ola = RowOLA(P)
        self.force_shaper = _Shaper(self.kb)
        self.direct_shaper = _Shaper(self.kb)
        self.mod = RateModulatorBank(self.rngf, f"{self.layer}/cox", P, sigma=0.22, sr=self.sr)
        self.pend_f = (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0),)
        self.pend_d = (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0),)
        self._tables: dict = {}
        self._tab_cur = None
        self._mfir = None
        self._grain_codes: dict = {}
        self._grain_ir: dict = {}
        self.stats: dict = {}
        self._dn_key = None                  # quantized state the cached matched-noise entries were built for
        self._lod_prev = None                # (dense, spill) of the previous block (threshold hysteresis)
        self._dn_entries: tuple = ([], [])
        self._dn_epoch = 0
        self._noise_base = 0
        self._built_sr = self.sr

    # -- per-block helpers ---------------------------------------------------
    def _modal_fir(self, n: int):
        """(nfft, H (S, nfft // 2 + 1)): every force row's bank response spectrum for blocks of n samples."""
        if self._mfir is None or self._mfir[0] != n:
            nfft = fast_len(2 * n - 1)
            H = np.zeros((self.S, nfft // 2 + 1), dtype=complex)
            for rows, mb in self.banks:
                _, h = mb.fir(n)
                if h is not None:
                    H[rows] = h
            self._mfir = (n, nfft, H)
        return self._mfir[1], self._mfir[2]

    def _table(self, w: WeatherState, scheme: str) -> _ClassTable:
        """Class table of the current DSD; the table in use is kept while Dm stays within 2 % and mu within
        0.1 of the DSD it was built for (hysteresis: no flapping between two grid cells)."""
        cur = self._tab_cur
        dsd = w.dsd
        if (cur is not None and cur[0] == scheme and abs(math.log(max(float(dsd.dm_mm), 1e-6) / cur[1])) <= 0.02
                and abs(float(dsd.mu) - cur[2]) <= 0.1 and cur[3] == (round(float(dsd.d_min_mm), 2),
                                                                     round(float(dsd.d_max_mm), 2))):
            return cur[4]
        key = _dsd_key(dsd, scheme)
        t = self._tables.get(key)
        if t is None:
            if len(self._tables) > 64:
                self._tables.clear()
            t = _ClassTable(dsd, scheme)
            self._tables[key] = t
        self._tab_cur = (scheme, max(float(dsd.dm_mm), 1e-6), float(dsd.mu),
                         (round(float(dsd.d_min_mm), 2), round(float(dsd.d_max_mm), 2)), t)
        return t

    def _grain_code(self, pats, pulse_codes, variants) -> np.ndarray:
        """Codes of the pre-baked grains of (patch material + bank variant, pulse code) (built on first use)."""
        kb = self.kb
        pats = np.asarray(pats, dtype=np.int64).ravel()
        if pats.size == 0:
            return np.zeros(0, dtype=np.int64)
        pulse_codes = np.broadcast_to(np.asarray(pulse_codes, dtype=np.int64), pats.shape)
        variants = np.broadcast_to(np.asarray(variants, dtype=np.int64), pats.shape)
        # pulse codes are < 2^43 and variants < 8: one int64 key per (patch, pulse code, variant)
        uk, inv = np.unique((pats << 46) | (pulse_codes << 3) | variants, return_inverse=True)
        out = np.empty(uk.size, dtype=np.int64)
        for k, key in enumerate(uk.tolist()):
            c = self._grain_codes.get(key)
            if c is None:
                pi, pc, v = key >> 46, (key >> 3) & ((1 << 43) - 1), key & 7
                mat = self.mats[pi]
                spread = float(self.spread_p[pi]) if v else 0.0
                tau = kb.tau_center((pc & ((1 << 40) - 1)) - 256)

                def make(mat=mat, tau=tau, spread=spread, v=v):
                    ik = (mat.name, round(spread, 4), int(v))
                    ir = self._grain_ir.get(ik)
                    if ir is None:
                        ir = _modal_ir(_variant_modes(self.rngf, mat.name, spread, v), kb.sr)
                        self._grain_ir[ik] = ir
                    return _grain(kb, mat, tau, ir=ir)

                c = kb.register(("grain", mat.name, round(spread, 4), int(v), int(pc)), make)
                self._grain_codes[key] = c
            out[k] = c
        return out[np.asarray(inv).ravel()]

    def _surface_state(self, w: WeatherState):
        """Per patch: wetness, puddle cover and depth, water depth, glaze flag (from the weather).

        Wetness and puddle depth are read on a 1/64 and 0.25 mm grid: they drift slowly and key the
        cached matched-noise entries."""
        glazed = w.precip_type == "freezing_rain" and w.temperature_c <= 0.5
        wetd = w.wetness or {}
        pud = w.puddle_depth_mm or {}
        names = self._uniq_names
        wet_m = np.array([round(float(np.clip(wetd.get(nm, 0.0) or 0.0, 0.0, 1.0)) * 64.0) / 64.0 for nm in names])
        pud_m = np.array([round(float(pud.get(nm, 0.0) or 0.0) * 4.0) / 4.0 for nm in names])
        wet = np.zeros(self.P) if glazed else wet_m[self._name_idx]
        dd = pud_m[self._name_idx]
        has = (dd > 0.2) & self.puddle_ok & (not glazed)
        cover = np.where(has, np.minimum(1.0, dd / 8.0) ** 0.8 * np.where(self.porous, 0.5, 1.0), 0.0)
        depth = np.where(has, np.maximum(dd, 2.0), 1000.0)
        wet = np.where(self.is_water, 0.0, wet)
        for i in self._water_pats:
            depth[i] = round(self.members[i]._water_depth(w) * 4.0) / 4.0
        return wet, cover, depth, glazed

    @staticmethod
    def _take_due(pend, t_abs0: int, n: int):
        t, rows, codes, amps = pend
        if t.size == 0:
            return None, pend
        due = t < t_abs0 + n
        if not np.any(due):
            return None, pend
        keep = ~due
        cur = (t[due] - t_abs0, rows[due], codes[due], amps[due])
        return cur, (t[keep], rows[keep], codes[keep], amps[keep])

    def _push(self, ev, pend_name, t_abs0, n, offs, rows, codes, amps):
        now = offs < n
        if np.all(now):
            ev.append((offs, rows, codes, amps))
            return
        ev.append((offs[now], rows[now], codes[now], amps[now]))
        later = ~now
        t, r, c, a = getattr(self, pend_name)
        setattr(self, pend_name, (np.concatenate([t, offs[later] + t_abs0]), np.concatenate([r, rows[later]]),
                                  np.concatenate([c, codes[later]]), np.concatenate([a, amps[later]])))

    # -- one block -------------------------------------------------------------
    def render(self, gen, ctx: BlockContext):
        if self._built_sr != ctx.sr:
            self._build(int(ctx.sr))
            self._key = None
        key = (int(ctx.index), int(ctx.n), round(float(ctx.t0), 9))
        if key != self._key or gen._field_epoch == self._epoch:
            self._compute(ctx)
            self._key = key
            self._epoch += 1
        gen._field_epoch = self._epoch
        return self._out[gen._field_index]

    def _compute(self, ctx: BlockContext):
        w = ctx.weather
        n, sr, b = int(ctx.n), int(ctx.sr), int(ctx.index)
        kb = self.kb
        P = self.P
        t_abs0 = int(round(float(ctx.t0) * sr))
        m1 = self.mod.step(b, n)
        m0 = self.mod.prev
        f_ev, d_ev, f_dn, d_dn = [], [], [], []      # events (offs, rows, codes, amps); dense (rows, codes, var)
        cur, self.pend_f = self._take_due(self.pend_f, t_abs0, n)
        if cur is not None:
            f_ev.append(cur)
        cur, self.pend_d = self._take_due(self.pend_d, t_abs0, n)
        if cur is not None:
            d_ev.append(cur)
        gate = _RAIN_GATE.get(w.precip_type, 1.0)
        self.stats = {}
        self._noise_base = 0
        if self._air_update(w):
            self._dn_key = None                  # the cached noise entries carry folded air kernels
        if gate > 0.0 and w.class_flux and w.dsd is not None and P:
            self._events(ctx, gate, m0, m1, f_ev, d_ev, f_dn, d_dn, t_abs0)

        base = self._noise_base
        for dom, ev, dn, ola, shaper, to_pat in ((0, f_ev, f_dn, self.force_ola, self.force_shaper, self.row_patch),
                                                (1, d_ev, d_dn, self.direct_ola, self.direct_shaper, None)):
            if ev:
                offs = np.concatenate([e[0] for e in ev])
                if offs.size:
                    rows = np.concatenate([e[1] for e in ev])
                    codes = np.concatenate([e[2] for e in ev])
                    if dom == 1:
                        codes = self._air_codes(codes, rows)
                    rows_u, Y = scatter_kernels(kb, n, rows, codes, offs, np.concatenate([e[3] for e in ev]))
                    ola.add(rows_u, Y)
            if dn:
                rows = np.concatenate([d[0] for d in dn])
                codes = np.concatenate([d[1] for d in dn])
                var = np.concatenate([d[2] for d in dn])
                mr0, mr1 = (m0, m1) if to_pat is None else (m0[to_pat], m1[to_pat])
                rows_u, Y = shaper.render(n, rows, codes, var, mr0, mr1, base, layout_key=(self._dn_epoch, dom))
                if rows_u.size:
                    ola.add(rows_u, Y)
                    base += Y.shape[0] * (Y.shape[1] + 2)              # next rows read fresh table noise

        y = np.zeros((P, n))
        if self.S:
            force = self.force_ola.pop(n)
            has = np.any(force, axis=1)
            rows_y = np.zeros((self.S, n))
            fir = None
            if np.any(has):                         # every bank's in-block convolution in one batched FFT
                idx = np.nonzero(has)[0]
                nfft, H = self._modal_fir(n)
                fir = np.zeros((self.S, n))
                fir[idx] = sfft.irfft(sfft.rfft(force[idx], nfft, axis=-1) * H[idx], nfft, axis=-1)[:, :n]
            for rows, mb in self.banks:
                fin = bool(np.any(has[rows]))
                if not fin and mb.silent(1e-11):
                    continue
                if mb.bank.n_modes == 0:
                    rows_y[rows] = force[rows]
                else:
                    rows_y[rows] = mb.process(force[rows], has_input=fin, fir=fir[rows] if fin else None)
            solid = np.add.reduceat(rows_y, self.row_starts, axis=0)        # (voice solid patches, n)
            for item in self.rad:
                pats, sos, zi, cal, sel = item
                x = solid[sel]
                if not np.any(x) and (zi is None or not np.any(zi)):
                    continue
                if zi is not None:
                    x, zi = signal.sosfilt(sos, x, axis=-1, zi=zi)
                    np.copyto(zi, 0.0, where=np.abs(zi) < 1e-30)        # no denormal tails
                    item[2] = zi
                y[pats] = x * cal
            for pats, air in self.air:                  # air absorption of the modal panels' radiation
                x = y[pats]
                if not np.any(x) and (air.zi is None or not np.any(air.zi)):
                    continue
                y[pats] = air.process(x, w.temperature_c, w.humidity_pct)
        if self.direct_ola.active:
            y += self.direct_ola.pop(n)
        else:
            self.direct_ola.pop(n)
        for i, gain, xs in self.xfer:
            if xs is not None:
                y[i] = xs.process(y[i])
            y[i] *= gain
        y *= self.spreading[:, None]
        live = np.max(np.abs(y), axis=1) > _LIVE_EPS
        self._out = [y[i] if live[i] else None for i in range(P)]
        if not np.all(live):
            self._flush(~live)

    def _flush(self, quiet: np.ndarray):
        """Zero the filter states of silent patches whose inputs have died out (no denormal tails)."""
        f_act = self.force_ola.active_rows()
        for pats, air in self.air:
            if air.zi is None:
                continue
            q = quiet[pats]
            if self.S:
                q &= np.array([not np.any(f_act[self.row0[p]:self.row0[p] + max(self.n_var[p], 1)]) for p in pats])
            if np.any(q):
                air.zi[:, q, :] = 0.0

    # -- air absorption folded into the pressure-domain kernels ---------------------
    def _air_update(self, w: WeatherState) -> bool:
        """Track the air state; True when the folded kernels must be rebuilt (> 1 C or > 5 % RH away)."""
        t, h = float(w.temperature_c), float(w.humidity_pct)
        cur = self._air_th
        if cur is not None and abs(t - cur[0]) <= 1.0 and abs(h - cur[1]) <= 0.05 * max(cur[1], 1.0):
            return False
        self._air_th = (round(t * 2.0) / 2.0, float(round(h)))
        self._air_map = {}
        return True

    def _air_codes(self, codes, rows) -> np.ndarray:
        """Kernel codes with each row's ISO 9613-1 air absorption folded in (rows nearer than
        AIR_MIN_DIST_M keep the plain kernel); the filtered kernels are built once per air state."""
        codes = np.asarray(codes, dtype=np.int64)
        g = self.air_g[np.asarray(rows, dtype=np.int64)]
        far = g >= 0
        if not np.any(far) or self._air_th is None:
            return codes
        codes = codes.copy()
        uk, inv = np.unique((codes[far] << 8) | g[far], return_inverse=True)   # codes < 2^44
        out = np.empty(uk.size, dtype=np.int64)
        amap = self._air_map
        kb = self.kb
        t, h = self._air_th
        for i, k in enumerate(uk.tolist()):
            c = amap.get(k)
            if c is None:
                code, d = k >> 8, self.air_d[k & 255]

                def make(code=code, d=d):
                    ker = kb.kernel(code)
                    y = signal.sosfilt(air_absorption_sos(d, t, h, kb.sr), np.concatenate([ker, np.zeros(512)]))
                    env = np.abs(y)
                    nz = np.nonzero(env > 1e-5 * max(float(env.max()), 1e-300))[0]
                    return y[: max(int(nz[-1]) + 1 if nz.size else 1, len(ker))]

                c = amap[k] = kb.register(("air", int(code), round(d, 4), t, h), make)
            out[i] = c
        codes[far] = out[np.asarray(inv).ravel()]
        return codes

    # -- events -----------------------------------------------------------------
    def _events(self, ctx, gate, m0, m1, f_ev, d_ev, f_dn, d_dn, t_abs0):
        w = ctx.weather
        n, sr, b = int(ctx.n), int(ctx.sr), int(ctx.index)
        kb = self.kb
        scheme = self.members[0].scheme
        tab = self._table(w, scheme)
        uh = _wind_vector(w)
        wn = self.nx * uh[0] + self.ny * uh[1]                         # wind component along each normal
        flux = np.array([float(w.class_flux.get(c, 0.0) or 0.0) for c in CLASSES]) * tab.valid
        ff = np.maximum(0.0, self.nz[:, None] - wn[:, None] / np.maximum(tab.vt_c, 0.3)[None, :])
        rate = flux[None, :] * (self.base * gate)[:, None] * ff        # impacts/s per patch and class
        if not np.any(rate > 0.0):
            return
        mmean = 0.5 * (m0 + m1)
        wet, cover, depth, glazed = self._surface_state(w)
        wind10 = float(w.wind_u10_mps) * max(float(w.gust), 0.0)
        wq = float(self.members[0].water_quality)

        # quadrature over each class: normal impact speed and peak force per patch
        unq = np.maximum(tab.vtq[None, :, :] * self.nz[:, None, None] - wn[:, None, None],
                         0.2 * tab.vtq[None, :, :])                    # (P, 5, Q)
        dq = np.broadcast_to(tab.dq[None, :, :], unq.shape)
        fq = force_peak(dq, unq)
        # LOD decisions use the unmodulated rate, so they hold while the weather does (the Cox modulation
        # only scales the per-drop counts and the matched-noise variance)
        overlap = rate * self.teff
        cap = self.caps * (n / _BLOCK_REF)
        if math.isfinite(self.lod["dense"]):
            # salience by impact force holds on solids; on water the small class rings loudest (Medwin 1992)
            e_rel = fq * fq @ tab.wq
            e_rel = e_rel / np.maximum(e_rel.max(axis=1, keepdims=True), 1e-30)
            # +-15 % hysteresis on every threshold: a class near one does not flip from block to block
            pd, ps = self._lod_prev if self._lod_prev is not None else (np.zeros_like(rate, bool),) * 2
            hy_d = np.where(pd, 0.85, 1.15)
            faint = (e_rel < _SALIENCE) & (overlap >= 0.3 * hy_d) & ~self.is_water[:, None]
            dense = (rate > 0.0) & ((overlap >= self.dense_thr[:, None] * hy_d) | faint)
            # sparse classes that reach their per-drop cap send the excess to the matched-noise layer
            # (cap / rate ratio is block-length free: caps are stated per 1024 frames)
            spill = (rate > 0.0) & ~dense & (rate * (_BLOCK_REF / sr) > 0.5 * self.caps * np.where(ps, 0.85, 1.15))
            self._lod_prev = (dense, spill)
        else:
            dense = np.zeros(rate.shape, dtype=bool)
            spill = dense

        # ---- sparse classes: Poisson counts; above the cap the excess is matched noise ------------
        lam = np.where(dense, 0.0, rate * (n / sr) * mmean[:, None])
        capped = lam > cap
        lam_draw = np.where(capped, cap, lam)
        excess = np.where(capped, 1.0 - cap / np.maximum(lam, 1e-12), 0.0)
        # a capped class without a noise layer keeps its energy by sqrt(expected / cap) amplitude scaling
        scale = np.where(capped & ~spill, np.sqrt(lam / np.maximum(cap, 1e-12)), 1.0)
        rng = self.rngf.get(self.layer, b)
        self._noise_base = int(rng.integers(0, 1 << 30))              # matched-noise table position
        counts = rng.poisson(lam_draw)
        tot = int(counts.sum())
        # per block, per patch and class: expected impacts, per-drop draws, dense flags, matched-noise
        # share of the energy (tests, meters)
        share = np.where(dense, 1.0, np.where(spill, excess, 0.0))
        self.stats = {"expected": rate * (n / sr) * mmean[:, None], "drawn": counts, "dense": dense,
                      "scale": scale, "noise_share": share}
        if tot:
            flat = counts.ravel()
            pc = np.repeat(np.arange(flat.size), flat)
            pe, ce = pc // len(CLASSES), pc % len(CLASSES)
            se = scale.ravel()[pc]
            off = (ramp_positions(rng.random(tot), m0[pe], m1[pe]) * n).astype(np.int64)
            d = tab.sample(rng.random(tot), ce)
            vt = _fall_speed(d)
            un = np.maximum(vt * self.nz[pe] - wn[pe], 0.2 * vt)
            to_water = self.is_water[pe] | (rng.random(tot) < cover[pe])
            wi = np.nonzero(to_water)[0]
            if wi.size:
                p_w = pe[wi]
                offs, codes, amps, src = _water_events(
                    kb, rng, off[wi], d[wi], un[wi], ce[wi], se[wi], click_pa=self.click_pa[p_w],
                    depth_mm=depth[p_w], wind_mps=wind10, water_quality=wq, sr=sr)
                self._push(d_ev, "pend_d", t_abs0, n, offs, p_w[src], codes, amps)
            si = np.nonzero(~to_water)[0]
            if si.size:
                self._solid_events(rng, si, pe, off, d, un, se, wet, glazed, f_ev, d_ev, t_abs0, n, sr)

        # ---- dense classes and capped excess: matched noise --------------------------------
        # the entries (rows, kernels, variance per unit rate) depend only on the quantized surface / DSD /
        # wind state and the noise set, so they are rebuilt only when that state moves
        noisy = dense | spill
        if np.any(noisy):
            # rebuilt when the table, the noise set, the quantized surface state or the glaze change, the
            # wind along any patch normal moves by > 0.15 m/s, or the wind inside the small-drop ring's
            # 7.5-10 m/s cut-off band moves by > 0.2 m/s
            wind_e = min(max(wind10, 7.5), 10.0)
            key = (id(tab), bool(glazed), dense.tobytes(), spill.tobytes(), np.asarray(wet).tobytes(),
                   cover.tobytes(), depth.tobytes())
            st = self._dn_key
            if (st is None or st[0] != key or float(np.max(np.abs(wn - st[1]), initial=0.0)) > 0.15
                    or abs(wind_e - st[2]) > 0.2):
                self._dn_entries = self._dense(noisy, tab.wq, dq, unq, fq, wet, cover, depth, glazed, wind_e, wq)
                self._dn_key = (key, wn.copy(), wind_e, tab)
                self._dn_epoch += 1
            rf = (rate * share).ravel() / sr                           # noise impacts per sample per (patch, class)
            for out, entries in ((f_dn, self._dn_entries[0]), (d_dn, self._dn_entries[1])):
                for rows, codes, coef, src in entries:
                    out.append((rows, codes, coef * rf[src]))

    def _solid_events(self, rng, si, pe, off, d, un, se, wet, glazed, f_ev, d_ev, t_abs0, n, sr):
        kb = self.kb
        p = pe[si]
        dd, uu, oo, ss = d[si], un[si], off[si], se[si]
        m = si.size
        wp = wet[p]
        f_phys = force_peak(dd, uu)
        tau = dd * 1e-3 / np.maximum(uu, 0.05) * (1.0 + 0.6 * wp) * (0.7 if glazed else 1.0)
        force = f_phys * ss * np.exp(rng.normal(0.0, _SIG_FORCE, m)) * (1.0 - 0.35 * wp)
        sn = self.snow[p]
        if np.any(sn):                       # rain onto a snow cover: soft, swallowed
            force = np.where(sn, 0.2 * force, force)
            tau = np.where(sn, 4.0 * tau, tau)
        codes = kb.pulse_code(tau)
        mod = self.modal[p]
        var = (rng.random(m) * np.maximum(self.n_var[p], 1)).astype(np.int64)
        rows = np.where(mod, self.row0[p] + var, p)
        # prompt-splash micro-impacts on dry hard surfaces
        ps = self.splash_p[p] * (1.0 - wp) ** 2 * _smoothstep(dd, 0.6, 1.6) * (wp < 0.95)
        sel = np.nonzero(rng.random(m) < ps)[0]
        if sel.size:
            rep = np.repeat(sel, rng.integers(2, 11, sel.size))
            k = rep.size
            delay = (np.exp(rng.uniform(math.log(0.005), math.log(0.05), k)) * sr).astype(np.int64)
            amp = force[rep] * np.exp(rng.uniform(math.log(0.005), math.log(0.03), k))
            s_codes = kb.pulse_code(np.exp(rng.uniform(math.log(5e-5), math.log(1.5e-4), k)))
            rm = mod[rep]
            if np.any(rm):
                j = np.nonzero(rm)[0]
                self._push(f_ev, "pend_f", t_abs0, n, oo[rep][j] + delay[j], rows[rep][j], s_codes[j], amp[j])
            if not np.all(rm):
                j = np.nonzero(~rm)[0]
                self._push(d_ev, "pend_d", t_abs0, n, oo[rep][j] + delay[j], rows[rep][j],
                           self._grain_code(p[rep][j], s_codes[j], var[rep][j]), amp[j])
        if np.any(mod):
            j = np.nonzero(mod)[0]
            f_ev.append((oo[j], rows[j], codes[j], force[j]))
        if not np.all(mod):
            j = np.nonzero(~mod)[0]
            d_ev.append((oo[j], rows[j], self._grain_code(p[j], codes[j], var[j]), force[j]))
        # crown / dome splash hiss on a wet film (no bubbles)
        h = np.nonzero((wp > 0.05) & self.hiss_ok[p])[0]
        if h.size:
            ph = p[h]
            amp = self.hiss_pa[ph] * wp[h] * (f_phys[h] / F_REF) ** 0.75 * ss[h] * self.hiss_scale[ph]
            dur = np.clip(0.0025 * np.sqrt(dd[h] / 2.0), 0.0015, 0.008)
            d_ev.append((oo[h], ph, kb.splash_code(dur, rng.integers(0, kb.SPLASH_VARIANTS, h.size), True), amp))

    def _dense(self, noisy, qw, dq, unq, fq, wet, cover, depth, glazed, wind10, wq):
        """Matched-noise entries of the (patch, class) pairs in `noisy` for a unit impact rate (1 per sample).

        qw are the class quadrature weights (Gauss-Legendre in the flux-weighted size quantile). Solid
        force pulses keep one kernel per quadrature node, so the noise has the per-drop spectrum of the
        whole size range. Returns (force entries, direct entries), each a list of (rows, kernel codes,
        variance per unit rate, flat patch * 5 + class index); the caller scales by the current rates.
        """
        kb = self.kb
        f_out, d_out = [], []
        Q = qw.size
        pi, ci = np.nonzero(noisy)
        src = pi * len(CLASSES) + ci
        fqq, dqq, uqq = fq[pi, ci], dq[pi, ci], unq[pi, ci]     # (k, Q)
        water_share = np.where(self.is_water[pi], 1.0, cover[pi])
        solid_share = 1.0 - water_share
        # -- solid: force noise into the voice panel's banks, or grains for far patches -----
        s = np.nonzero(solid_share > 0.0)[0]
        if s.size:
            p = pi[s]
            ss_ = src[s]
            wp = wet[p]
            sn = self.snow[p]
            f2 = (fqq[s] ** 2 * math.exp(2.0 * _SIG_FORCE ** 2)
                  * ((1.0 - 0.35 * wp) ** 2 * np.where(sn, 0.04, 1.0))[:, None])
            tauq = (dqq[s] * 1e-3 / np.maximum(uqq[s], 0.05)
                    * ((1.0 + 0.6 * wp) * (0.7 if glazed else 1.0) * np.where(sn, 4.0, 1.0))[:, None])
            rs = solid_share[s]
            v_fq = rs[:, None] * f2 * qw[None, :]                      # (k, Q) per quadrature node
            ps = (self.splash_p[p][:, None] * ((1.0 - wp) ** 2 * (wp < 0.95))[:, None]
                  * _smoothstep(dqq[s], 0.6, 1.6))
            v_s = rs * ((ps * f2) @ qw) * _SPLASH_REPS * _SPLASH_A2
            codes_q = kb.pulse_code(tauq)                              # (k, Q)
            s_code = np.full(p.size, kb.pulse_code(_SPLASH_TAU), dtype=np.int64)
            mod = self.modal[p]
            nv = np.maximum(self.n_var[p], 1)
            jj = np.repeat(np.arange(p.size), nv)                       # one entry per bank variant
            vv = np.concatenate([np.arange(k) for k in nv])
            share = 1.0 / nv[jj]
            m_ = mod[jj]
            for sel, out in ((m_, f_out), (~m_, d_out)):
                if not np.any(sel):
                    continue
                j, v = jj[sel], vv[sel]
                if out is f_out:
                    rows = self.row0[p[j]] + v
                    cq, cs = codes_q[j].ravel(), s_code[j]
                else:
                    rows = p[j]
                    cq = self._grain_code(np.repeat(p[j], Q), codes_q[j].ravel(), np.repeat(v, Q))
                    cs = self._grain_code(p[j], s_code[j], v)
                out.append((np.concatenate([np.repeat(rows, Q), rows]), np.concatenate([cq, cs]),
                            np.concatenate([(v_fq[j] * share[sel][:, None]).ravel(), v_s[j] * share[sel]]),
                            np.concatenate([np.repeat(ss_[j], Q), ss_[j]])))
            h = np.nonzero((wp > 0.05) & self.hiss_ok[p])[0]
            if h.size:
                ph = p[h]
                a2 = (self.hiss_pa[ph] * wp[h] * self.hiss_scale[ph]) ** 2 * (((fqq[s][h] / F_REF) ** 1.5) @ qw)
                dur = np.clip(0.0025 * np.sqrt((dqq[s][h] @ qw) / 2.0), 0.0015, 0.008)
                d_out.append((ph, kb.splash_code(dur, 0, True), rs[h] * a2, ss_[h]))
        # -- water: clicks, bubbles and splats as coloured noise ---------------------
        wsel = np.nonzero(water_share > 0.0)[0]
        if wsel.size:
            p, c, sw = pi[wsel], ci[wsel], src[wsel]
            rw = water_share[wsel]
            dd, uu = dqq[wsel], uqq[wsel]
            u_e = np.sqrt(((dd ** 2 * uu ** 8) @ qw) / np.maximum((dd ** 2 * uu ** 6) @ qw, 1e-30))
            u_bin = kb.click_bin(u_e) * kb.CLICK_STEP
            a2 = (self.click_pa[p] ** 2 * math.exp(2.0 * _SIG_CLICK ** 2) * np.where(c == 0, 0.09, 1.0)
                  * (((dd / D_REF_MM) ** 2 * (uu / u_bin[:, None]) ** 6) @ qw))
            d_out.append((p, kb.click_code(u_e), rw * a2, sw))
            pent = _entrainment_probability(dd, np.repeat(c[:, None], dd.shape[1], axis=1), wind10) @ qw
            pent = pent * float(np.clip(wq, 0.0, 1.0)) * np.where(c == 4, 1.0 + _VL_EXTRA_BUBBLES, 1.0)
            fd = _smoothstep(depth[p], 1.5, 8.0)
            vb = rw * pent * (BUBBLE_PA_PER_M * fd) ** 2 * _EPS2[c]
            ok = np.nonzero(vb > 0.0)[0]
            if ok.size:
                db = _damp_bin(depth[p[ok]])
                codes = np.array([kb.register(("bubble_color", int(c[j]), int(round(float(dd[j] @ qw) / 0.05)),
                                               int(db[k])),
                                              lambda j=j, k=k: _bubble_color(kb, int(c[j]), dd[j], int(db[k]), qw))
                                  for k, j in enumerate(ok)], dtype=np.int64)
                d_out.append((p[ok], codes, vb[ok], sw[ok]))
            k = np.nonzero(c >= 3)[0]
            if k.size:
                ind = np.where(c[k, None] == 4, 1.0, (dd[k] >= 3.0).astype(float))
                a2s = ((ind * _splat_amp(dd[k], uu[k], c[k, None]) ** 2) @ qw) * math.exp(2.0 * 0.25 ** 2)
                if np.any(a2s > 0.0):
                    d_out.append((p[k], kb.splash_code(_splat_dur(dd[k] @ qw), 0, True), rw[k] * a2s, sw[k]))
        d_out = [(r, self._air_codes(c, r), v, i) for r, c, v, i in d_out]
        return f_out, d_out


class RainPatchGenerator:
    """Rain on one SourcePatch: per-class Cox event streams -> water / solid / matched-noise synthesis.

    Patches built together by make_rain_generators share one _RainField, which renders all of them in
    a single vectorised pass per block; a patch built on its own gets a private field.
    """

    CAPS_FULL = QUALITY_LOD["studio"]["voice_cap"]          # max per-drop events per 1024 frames per class
    CAPS_GRANULAR = QUALITY_LOD["studio"]["gran_cap"]

    def __init__(self, scene, patch: SourcePatch, rngf: RngFactory, lod: str = "full",
                 kernels: KernelBank | None = None, sr: int = SAMPLE_RATE, *, field: _RainField | None = None,
                 quality: str | None = None):
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
        click_db = self.mat.level_db if self.kind == "water" else MATERIALS["puddle"].level_db
        self.click_pa = float(db_to_pa(click_db))
        self.sr = int(sr)
        self.xfer_gain, xs = _interior_transfer(self.patch, self.sr)
        self.xfer = StatefulSOS(xs) if xs is not None else None
        if field is None:
            q = quality or str(getattr(getattr(scene, "render", None), "quality", "studio") or "studio")
            field = _RainField(rngf, q, kernels, self.sr, layer=f"rain/{patch.id}/field")
        self.field = field
        self._field_index = field.add(self)
        self._field_epoch = -1

    @property
    def caps(self):
        return tuple(self.field.lod["voice_cap" if self.lod == "full" else "gran_cap"])

    def _water_depth(self, w: WeatherState) -> float:
        if self.mat.name == "water" or self.patch.params.get("deep"):
            return 1000.0
        d = (w.puddle_depth_mm or {}).get(self.mat.name)
        if d is None:
            d = self.patch.params.get("depth_mm", 12.0)
        return max(float(d), 0.5)

    def render(self, ctx: BlockContext) -> list[Emission]:
        y = self.field.render(self, ctx)
        if y is None:
            return []
        return [Emission(signal=y, azimuth=self.patch.azimuth, elevation=self.patch.elevation,
                         distance=self.patch.distance, spread=self.spread, stem=self.stem,
                         is_ambisonic=False, source_id=self.name)]


# ---------------------------------------------------------------------------
# far-field diffuse bed, synthesized in the SH domain

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
        self.kb = KernelBank(self.sr)
        self.band_sos = []
        self.band_e = np.zeros(len(self.BANDS))
        imp = np.zeros(16384)
        imp[0] = 1.0
        for k, (lo, hi) in enumerate(self.BANDS):
            hi = min(hi, 0.45 * self.sr)
            lo = min(lo, 0.5 * hi)
            sos = signal.butter(2, [lo, hi], "bandpass", fs=self.sr, output="sos")
            self.band_sos.append(sos)
            self.band_e[k] = float(np.sum(signal.sosfilt(sos, imp) ** 2))
        orders = np.floor(np.sqrt(np.arange(N_AMBI))).astype(int)
        self.ord_w = 1.0 / np.sqrt(2.0 * orders + 1.0)          # diffuse field in SN3D
        self.v_prev = 0.0
        self._var_prev = np.zeros(len(self.BANDS))
        self.aniso_prev = np.zeros(3)
        self.shaper = _Shaper(self.kb)
        self.ola = RowOLA(N_AMBI)
        # slowly wandering rain curtains: the louder side of the bed drifts (1/f, 1 s .. 30 s)
        self.wander = (RateModulator(self.rngf, "rain/bed/wander_x", sigma=0.6, tau_min_s=1.0,
                                     tau_max_s=30.0, sr=self.sr),
                       RateModulator(self.rngf, "rain/bed/wander_y", sigma=0.6, tau_min_s=1.0,
                                     tau_max_s=30.0, sr=self.sr))

    def _band_codes(self, temp_c: float, rh_pct: float) -> np.ndarray:
        """Kernel codes of the four band filters, each followed by ISO 9613-1 absorption at r_air."""
        tq, hq = int(round(float(temp_c))), int(round(float(rh_pct) / 2.0)) * 2
        imp = np.zeros(8192)
        imp[0] = 1.0

        def make(k):
            y = signal.sosfilt(self.band_sos[k], imp)
            air = air_absorption_sos(self.r_air, float(tq), float(hq), self.sr)
            return fade_tail(signal.sosfilt(air, y), 0.1)

        return np.array([self.kb.register(("bed_band", k, round(self.r_air, 2), tq, hq), lambda k=k: make(k))
                         for k in range(len(self.BANDS))], dtype=np.int64)

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
            # representative stone: 0.6 x size for the 1.6 x tail; the hail spectrum, truncated at its reported
            # size, shifts it down with its mean diameter (the distribution scales with size)
            hi = sp.get("hi", 1.6)
            dm = 0.6 * sp["size"] * _ice_mean_diameter_rel(hi) / _ice_mean_diameter_rel(1.6)
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
        power = np.maximum(self._band_power(w), 0.0)
        v = float(power.sum())
        if v <= 0.0 and self.v_prev <= 0.0:
            return []
        var = power / self.band_e                               # white-noise variance feeding each band filter
        if v <= 0.0:                                            # fade out over one block on the last shape
            var = self._var_prev
            m0, m1 = 1.0, 0.0
        else:
            m0, m1 = self.v_prev / v, 1.0
            self._var_prev = var
        self.v_prev = v
        codes = self._band_codes(w.temperature_c, w.humidity_pct)
        rows = np.repeat(np.arange(N_AMBI), len(self.BANDS))
        base = int(self.rngf.get("rain/bed", ctx.index).integers(0, 1 << 30))
        rows_u, sig = self.shaper.render(n, rows, np.tile(codes, N_AMBI), np.tile(var, N_AMBI),
                                         np.full(N_AMBI, m0), np.full(N_AMBI, m1), base,
                                         layout_key=tuple(codes.tolist()))
        if rows_u.size:
            self.ola.add(rows_u, sig)
        out = self.ola.pop(n) * self.ord_w[:, None]
        ramp = np.arange(n, dtype=float) / n
        vdir = self._aniso(w, ctx.index, n)
        vr = self.aniso_prev[:, None] + (vdir - self.aniso_prev)[:, None] * ramp[None, :]
        self.aniso_prev = vdir
        if N_AMBI >= 4:
            wsig = out[0].copy()
            out[1] += wsig * vr[1]      # ACN 1 = Y (left)
            out[2] += wsig * vr[2]      # ACN 2 = Z (up)
            out[3] += wsig * vr[0]      # ACN 3 = X (front)
        return [Emission(signal=out, azimuth=0.0, elevation=0.0, distance=self.r_air, spread=1.0,
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
            y = _air(g.air, y * float(spreading_gain(g.distance)), g.distance, w)
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
            rng = self.rngf.get(f"gutter/{pid}", b)              # one generator per block for every stream
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
            rng_b = rng
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
            noise = noise_rows(np.array([int(self.rngf.get(f"gutter/{pid}/rush", b).integers(0, 1 << 30))]), n)[0]
            g = np.linspace(self.prev_rush, rush, n, endpoint=False)
            exc = exc + self.rush.process(noise) * self.rush_norm * g
        self.prev_rush = rush
        y = 0.6 * exc + self.pipe_mix * self.pipe_norm * self.pipe.process(exc)
        if self.shell.active:
            y += self.shell.render(n)
        if sheet > 0.0 or self.prev_sheet > 0.0:
            noise = noise_rows(np.array([int(self.rngf.get(f"gutter/{pid}/sheet", b).integers(0, 1 << 30))]), n)[0]
            g = np.linspace(self.prev_sheet, sheet, n, endpoint=False)
            y = y + self.sheet.process(noise) * self.sheet_norm * g
        self.prev_sheet = sheet
        if not np.any(y) and not self.ola.active:
            return []
        y = _air(self.air, y * float(spreading_gain(self.distance)), self.distance, w)
        return [Emission(signal=y, azimuth=self.patch.azimuth, elevation=self.patch.elevation,
                         distance=self.patch.distance, spread=self.spread, stem="aftermath",
                         is_ambisonic=False, source_id=self.name)]


# ---------------------------------------------------------------------------
# hail, ice pellets, graupel

def _ice_species(w: WeatherState):
    t = w.precip_type
    if t == "hail":
        # hail_size_mm is the reported size, i.e. the largest stones (NOAA/NWS reports compare the biggest
        # stone with a pea, marble, quarter ...): the spectrum is truncated there (hi = 1.0 x size)
        return {"kind": "hail", "size": max(float(w.hail_size_mm), 5.0), "rho": 900.0, "e": 5.0e9,
                "crush": 10.0, "frac": 0.5, "rest": 0.5, "hi": 1.0}
    if t == "sleet":
        return {"kind": "sleet", "size": 3.5, "rho": 900.0, "e": 9.0e9, "crush": 15.0, "frac": 0.5, "rest": 0.45}
    if t == "graupel":
        return {"kind": "graupel", "size": 3.0, "rho": 350.0, "e": 3.0e8, "crush": 1.5, "frac": 1.0, "rest": 0.15}
    return None


def _ice_sizes(rng, size_mm: float, n: int, hi_frac: float = 1.6) -> np.ndarray:
    """Truncated exponential on [0.3, hi_frac] x size with slope 2/size (stones skew small).

    Hail passes hi_frac = 1.0 (its size is the reported maximum); ice pellets and graupel, whose
    size is a typical one, keep the 1.6 x tail."""
    lo, hi = 0.3 * size_mm, hi_frac * size_mm
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


def _ice_mean_volume(size_mm: float, hi_frac: float = 1.6) -> float:
    key = (round(size_mm, 2), round(hi_frac, 3))
    v = _ICE_VOL_CACHE.get(key)
    if v is None:
        lo, hi = 0.3 * size_mm, hi_frac * size_mm
        dgrid = np.linspace(lo, hi, 256)
        pdf = np.exp(-2.0 / size_mm * (dgrid - lo))
        pdf /= pdf.sum()
        v = float(np.sum(pdf * math.pi * (dgrid * 1e-3) ** 3 / 6.0))
        _ICE_VOL_CACHE[key] = v
    return v


def _ice_mean_diameter_rel(hi_frac: float) -> float:
    """Mean stone diameter / size of the truncated exponential of _ice_sizes (scale-free)."""
    dgrid = np.linspace(0.3, hi_frac, 256)
    pdf = np.exp(-2.0 * (dgrid - 0.3))
    return float(np.sum(pdf * dgrid) / np.sum(pdf))


def _ice_number_flux(rate_mm_h: float, sp: dict) -> float:
    """Stones per m^2 per s from the liquid-equivalent rate (share sp['frac'] falls as ice)."""
    q = max(rate_mm_h, 0.0) * 1e-3 / 3600.0 * sp["frac"] * RHO_WATER / sp["rho"]
    return q / max(_ice_mean_volume(sp["size"], sp.get("hi", 1.6)), 1e-12)


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

    def render(self, ctx: BlockContext, sp, flux: float, rng=None):
        w = ctx.weather
        n, sr, b = ctx.n, ctx.sr, ctx.index
        t_abs0 = int(round(ctx.t0 * sr))
        if rng is None:
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
                d = _ice_sizes(rng, sp["size"], off.size, sp.get("hi", 1.6))
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
            sp = {"kind": "hail", "size": 10.0, "rho": 900.0, "e": 5.0e9, "crush": 10.0, "frac": 0.5, "rest": 0.5,
                  "hi": 1.0}
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
        return _air(self.air, y * float(spreading_gain(self.r_eq)), self.r_eq, w)

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
        rng = self.rngf.get("hail", ctx.index)          # one generator per block, drawn by the patches in order
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
            y = st.render(ctx, sp if active else None, flux, rng)
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

    def render(self, ctx: BlockContext, flake, flux: float, rng=None):
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        if flake is not None and flux > 0.0 and self.patch.exposure > 0.0:
            if rng is None:
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
        return _air(self.air, y * float(spreading_gain(self.r_eq)), self.r_eq, w)


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
            rng = self.rngf.get("snow", ctx.index) if active else None   # one generator per block
            for p in self.patches:
                st = self.states.get(p.id)
                if st is None:
                    if not active:
                        continue
                    st = _SnowPatch(p, self.rngf, self.kb, self.sr)
                    self.states[p.id] = st
                y = st.render(ctx, flake, flux, rng)
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


def _is_voice(p: SourcePatch) -> bool:
    """Per-drop 'voice' candidates (MASTER_PLAN 7.5): resonant objects and panels of the listener's own
    shelter, and surfaces near enough for single drops to stand out; a large surface in the mid-field
    is many independent pieces and renders as a granular / statistical layer."""
    if p.interior or p.kind in ("membrane", "vehicle_body", "gutter"):
        return True
    if p.distance > FULL_MAX_DIST_M:
        return False
    return p.area_m2 <= VOICE_MAX_AREA_M2 or p.distance <= VOICE_NEAR_M


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

    Level of detail (Zhang et al. 2019; MASTER_PLAN 7.5): the K loudest rain-receiving patches (K by
    render quality) that are voice candidates (shelter panels, membranes, vehicle bodies, and surfaces
    within 12 m that are small or within 4 m) get per-drop modal synthesis; the rest are granular /
    statistical layers; all share one vectorised _RainField; beyond the ground rings the
    RainBedGenerator renders a statistically matched diffuse bed directly in the SH domain.
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

    quality = str(getattr(scene.render, "quality", "studio") or "studio").lower()
    k_full = lod_settings(quality)["k"]
    patches = [p for p in layout if p.kind in RAIN_KINDS and p.exposure > 0.0]
    ranked = sorted(patches, key=lambda p: (-_loudness_score(p), p.id))
    # the K loudest voice candidates (large far surfaces must not crowd the near ring out of the budget)
    full_ids = {p.id for p in [p for p in ranked if _is_voice(p)][:k_full]}

    gens: list = []
    if want_rain:
        field = _RainField(rngf, quality, kb, sr)
        for p in patches:
            gens.append(RainPatchGenerator(scene, p, rngf, lod="full" if p.id in full_ids else "granular",
                                           kernels=kb, sr=sr, field=field))
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
