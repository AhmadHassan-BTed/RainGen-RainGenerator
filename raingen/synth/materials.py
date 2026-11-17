"""Surface materials for rain, hail, drip and snow impact synthesis.

A Material is a compact modal model of how a surface answers an impact
(MASTER_PLAN 3.3): a bank of damped modes (frequency, T60, gain) driven by the
force pulse, followed by a radiation band - the surface only radiates efficiently
between ``radiation_hp_hz`` and ``radiation_lp_hz``.

Physics behind the tables
- Thin plates (Kirchhoff, simply supported): f_mn = f_11 ((m/a)^2 + n^2) / ((1/a)^2 + 1).
  The coincidence (critical) frequency sets where radiation becomes efficient:
  0.5 mm steel ~24 kHz, 4 mm glass ~3 kHz, 12 mm plywood ~1.4 kHz (Cremer & Heckl).
- Mode decay from the loss factor eta: T60 = 2.2 / (f * eta), with eta ~1e-3 for
  bare steel up to ~0.05 for damped/fixed panels; higher modes decay faster.
- Membranes (umbrella, tent, canvas): ideal circular membrane, mode ratios equal
  to the Bessel zeros j_mn / j_01 - low, dense, drum-like (Morse & Ingard).
- Hard ground (concrete, asphalt, tile, gravel) has no audible slab modes; a few
  short "grit" resonances colour the splash click.
- Water kinds carry no modes: they are synthesized as impact click + Minnaert
  bubble (raingen.synth.kernels).
- Small radiators (leaves) are dipoles below ka ~ 1: their radiation band starts there.
- ``level_db`` is the calibration anchor: peak SPL (dB re 20 uPa) at 1 m of a single
  2 mm drop at terminal speed (6.5 m/s, Atlas et al. 1973) striking the dry surface.
  For water kinds it is the peak level of that drop's impact click. Everything else
  (rate, drop size, speed, wetness) scales physically from this anchor.
- ``hardness`` (0..1) maps to an effective Young's modulus for Hertz contact
  (hail, ice pellets): E = 10**(7 + 4.5 * hardness) Pa.
- ``splash`` (0..1) is the propensity for prompt-splash micro-impacts on the dry
  surface (splash parameter ~366 at terminal speed vs threshold ~58; Yu & Hopkins 2018).

``ucs_cat`` is the Universal Category System CatID used for file naming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from raingen.core.rng import RngFactory

MATERIAL_KINDS = ("solid", "water", "membrane", "porous", "vegetation", "snow")


@dataclass
class Material:
    name: str
    kind: str                                   # 'solid'|'water'|'membrane'|'porous'|'vegetation'|'snow'
    modes: list = field(default_factory=list)   # [(freq_hz, t60_s, gain)]
    hardness: float = 0.5                       # 0 soft .. 1 rigid (Hertz modulus mapping)
    splash: float = 0.5                         # dry-surface prompt-splash propensity 0..1
    radiation_hp_hz: float = 100.0
    radiation_lp_hz: float = 16000.0
    level_db: float = 45.0                      # peak dB SPL @1 m, 2 mm drop @6.5 m/s, dry
    ucs_cat: str = "RAINGen"

    @property
    def is_water(self) -> bool:
        return self.kind == "water"

    @property
    def youngs_modulus_pa(self) -> float:
        return 10.0 ** (7.0 + 4.5 * float(np.clip(self.hardness, 0.0, 1.0)))

    @property
    def max_t60(self) -> float:
        return max((m[1] for m in self.modes), default=0.0)


# ---------------------------------------------------------------------------
# Mode-set builders (static data; jitter drawn through RngFactory for determinism)

_MAT_RNG = RngFactory(0x6D617465)  # 'mate'

# Circular membrane: Bessel zeros j_mn / j_01
_MEMBRANE_RATIOS = (1.000, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.600, 3.652,
                    4.060, 4.154, 4.230, 4.601, 4.832, 4.903, 5.131, 5.412, 5.469, 5.650)


def _plate_series(f11: float, aspect: float, count: int) -> list[float]:
    """Lowest `count` modes of a simply supported a x 1 plate."""
    norm = (1.0 / aspect) ** 2 + 1.0
    freqs = sorted(f11 * ((m / aspect) ** 2 + k ** 2) / norm
                   for m in range(1, 24) for k in range(1, 24))
    out: list[float] = []
    for f in freqs:  # merge near-degenerate modes (they beat rather than add colour)
        if not out or f > out[-1] * 1.02:
            out.append(f)
        if len(out) >= count:
            break
    return out


def _log_spread(name: str, f_lo: float, f_hi: float, count: int, jitter: float = 0.12) -> list[float]:
    if count <= 0:
        return []
    rng = _MAT_RNG.get(f"materials/{name}", 0)
    base = np.geomspace(f_lo, f_hi, count)
    step = (math.log(f_hi / f_lo) / max(count - 1, 1)) if count > 1 else 0.0
    j = np.exp(rng.uniform(-jitter, jitter, count) * step) if step > 0 else np.ones(count)
    return list(np.clip(base * j, f_lo * 0.9, f_hi * 1.1))


def _build(name: str, freqs, t60_lo: float, t60_hi: float, *, tilt_db_oct: float = 0.0,
           peak_hz: float | None = None, peak_db: float = 0.0, peak_oct: float = 1.0,
           gain_jitter_db: float = 2.0) -> list[tuple[float, float, float]]:
    """Attach T60 (log-log interpolation) and an envelope gain to each frequency."""
    freqs = np.sort(np.asarray(list(freqs), dtype=float))
    if freqs.size == 0:
        return []
    f_lo, f_hi = float(freqs[0]), float(max(freqs[-1], freqs[0] * 1.0001))
    slope = math.log(t60_hi / t60_lo) / math.log(f_hi / f_lo)
    t60 = t60_lo * (freqs / f_lo) ** slope
    env_db = tilt_db_oct * np.log2(freqs / f_lo)
    if peak_hz is not None and peak_db != 0.0:
        env_db = env_db + peak_db * np.exp(-0.5 * (np.log2(freqs / peak_hz) / peak_oct) ** 2)
    rng = _MAT_RNG.get(f"materials/{name}/gain", 0)
    env_db = env_db + rng.uniform(-gain_jitter_db, gain_jitter_db, freqs.size)
    g = 10.0 ** (env_db / 20.0)
    g /= g.max()
    return [(float(f), float(t), float(a)) for f, t, a in zip(freqs, t60, g)]


def _membrane(name, f01, count, eta, extra=None, eta_extra=None, **kw):
    """Circular-membrane modes (Bessel ratios) with T60 = 2.2 / (eta f): a light fabric's
    radiation (air-loading) damping is so large that it pops instead of ringing."""
    freqs = [f01 * r for r in _MEMBRANE_RATIOS[:count]]
    if extra:
        freqs += _log_spread(name + "/weave", *extra)
    return _build_eta(name, freqs, lambda f: eta if eta_extra is None or f < 1000.0 else eta_extra, **kw)


def _grit(name, f_lo, f_hi, count, t60_lo, t60_hi, **kw):
    return _build(name, _log_spread(name, f_lo, f_hi, count, jitter=0.3), t60_lo, t60_hi, **kw)


# --- physically derived plates ----------------------------------------------------------------
# A thin plate of Young's modulus E, density rho, thickness h, Poisson nu over a x b (simply
# supported) has bending stiffness D = E h^3 / (12 (1 - nu^2)), mass m'' = rho h and modes
#   f_mn = (pi / 2) sqrt(D / m'') ((m / a)^2 + (n / b)^2)
# with the asymptotic modal density n(f) = (a b / 2) sqrt(m'' / D) per Hz (Cremer & Heckl), and
# the coincidence frequency f_c = c^2 / (2 pi) sqrt(m'' / D). Each mode decays with the in-situ
# total loss factor eta(f) = eta_int + eta_edge (1 kHz / f)^0.5 - the material's own loss plus
# the coupling / boundary losses of a fixed panel (Craik 1981: ~1/sqrt(f) for building elements)
# - and T60 = 2.2 / (eta f).
# An impulsive point force gives every mode the same velocity amplitude (mass-normalised), so a
# band holding N modes carries N times the energy of one: above the frequency where modes get
# denser than the table's log spacing, one representative mode stands for its whole band with
# gain sqrt(N). Radiated pressure adds sqrt(sigma), sigma ~ sqrt(f / f_c) below coincidence
# (capped at 1, a +3 dB bump at f_c). Mode shapes at the impact point scatter each gain
# (+-3 dB jitter); the bank variants at synthesis time re-draw them per impact position.
C_AIR = 343.0


def _plate_modes(name, *, E, rho, h, a, b, nu=0.3, D=None, m2=None, eta_int, eta_edge,
                 f_lo, f_hi, step_oct, max_low=12, gain_jitter_db=3.0):
    D = E * h ** 3 / (12.0 * (1.0 - nu * nu)) if D is None else D
    m2 = rho * h if m2 is None else m2
    k = 0.5 * math.pi * math.sqrt(D / m2)
    dens = 0.5 * a * b * math.sqrt(m2 / D)
    f_c = C_AIR ** 2 / (2.0 * math.pi) * math.sqrt(m2 / D)
    bw = 2.0 ** (step_oct / 2.0) - 2.0 ** (-step_oct / 2.0)
    f_split = max(f_lo, 1.0 / (dens * bw))                 # above: >= 1 mode per log bin
    low = sorted(k * ((m / a) ** 2 + (n / b) ** 2) for m in range(1, 40) for n in range(1, 40))
    freqs, counts = [], []
    for f in low:
        if f < f_lo:
            continue
        if f >= f_split or len(freqs) >= max_low:
            break
        if not freqs or f > freqs[-1] * 1.02:
            freqs.append(f)
            counts.append(1.0)
    start = max(f_split, freqs[-1] * 2.0 ** step_oct if freqs else f_split)
    nrep = max(int(math.floor(math.log2(f_hi / start) / step_oct)) + 1, 0)
    rep = _log_spread(name, start, start * 2.0 ** (step_oct * max(nrep - 1, 0)), nrep, jitter=0.35)
    for f in rep:
        freqs.append(float(f))
        counts.append(max(1.0, dens * f * bw))
    eta = lambda f: eta_int + eta_edge * math.sqrt(1000.0 / f)      # noqa: E731
    rng = _MAT_RNG.get(f"materials/{name}/gain", 0)
    out = []
    for f, n in zip(freqs, counts):
        sigma = min(1.0, math.sqrt(f / f_c)) * (1.41 if abs(math.log2(f / f_c)) < 0.25 else 1.0)
        g = math.sqrt(n * sigma) * 10.0 ** (rng.uniform(-gain_jitter_db, gain_jitter_db) / 20.0)
        out.append((float(f), float(np.clip(2.2 / (eta(f) * f), 0.002, 3.0)), g))
    gmax = max(m[2] for m in out)
    return [(f, t, g / gmax) for f, t, g in sorted(out)]


def _with_click(name, modes, *, E, rho, h, nu=0.3, D=None, m2=None, click=(1000.0, 12000.0, 10, 0.006, 0.002),
                y0=1e-3, **kw):
    """Panel modes mixed with the drop's own impact click by the panel's point mobility.

    A raindrop on a stiff, heavy panel barely moves it: the infinite-plate point mobility
    Y = 1 / (8 sqrt(D m'')) is 2e-3 s/kg for 4 mm glass, 5e-4 for a 25 mm board, 2e-2 for thin
    steel. The struck surface's own vibration then carries a share w = Y / (Y + y0) of the
    response amplitude and the impact click (contact + splash: short, dense "grit" decays) the
    rest; y0 ~ 1e-3 s/kg is where the two are comparable. Thin metals stay modal (w > 0.9), glass and
    wood are mostly tick / knock with a short ring behind it, masonry is all click.
    """
    D = E * h ** 3 / (12.0 * (1.0 - nu * nu)) if D is None else D
    m2 = rho * h if m2 is None else m2
    Y = 1.0 / (8.0 * math.sqrt(D * m2))
    w = Y / (Y + y0)
    grit = _grit(name + "/click", *click, **kw)
    return [(f, t, w * g) for f, t, g in modes] + [(f, t, (1.0 - w) * g) for f, t, g in grit]


def _build_eta(name, freqs, eta, *, tilt_db_oct=0.0, gain_jitter_db=2.0, **_):
    """Modes at `freqs` with T60 = 2.2 / (eta(f) f) and a spectral tilt (dB / octave)."""
    freqs = np.sort(np.asarray(list(freqs), dtype=float))
    rng = _MAT_RNG.get(f"materials/{name}/gain", 0)
    env_db = tilt_db_oct * np.log2(freqs / freqs[0]) + rng.uniform(-gain_jitter_db, gain_jitter_db, freqs.size)
    g = 10.0 ** (env_db / 20.0)
    g /= g.max()
    return [(float(f), float(np.clip(2.2 / (eta(f) * f), 0.002, 3.0)), float(a)) for f, a in zip(freqs, g)]


def _eta(eta_int, eta_edge):
    return lambda f: eta_int + eta_edge * math.sqrt(1000.0 / f)


# ---------------------------------------------------------------------------

def _make_materials() -> dict[str, Material]:
    M: dict[str, Material] = {}

    def add(m: Material):
        M[m.name] = m

    # ---- Metals: steel / aluminium eta_int ~ 1e-4..1e-3, so they ring; a fixed sheet adds
    # boundary losses at its fixings. Thin sheets are dense in modes (n ~ 0.1-0.5 per Hz), so the
    # ring is a bright shimmer rather than a few pitches. Coincidence ~ 17-24 kHz: radiation
    # rises through the whole band.
    # bare steel sheet (shed wall, sheet on a frame): 0.7 mm, 0.9 x 1.2 m between fixings
    add(Material("metal_sheet", "solid",
                 _plate_modes("metal_sheet", E=200e9, rho=7850.0, h=0.7e-3, a=0.9, b=1.2,
                              eta_int=1e-3, eta_edge=0.004, f_lo=120.0, f_hi=14000.0, step_oct=1 / 6),
                 hardness=0.95, splash=0.9, radiation_hp_hz=150.0, radiation_lp_hz=18000.0,
                 level_db=60.0, ucs_cat="RAINMetl"))
    # corrugated 0.5 mm galvanised steel on purlins 0.9 m apart, 0.76 m cover width: the 18 mm
    # corrugation stiffens it (D along the ribs ~ E h d^2 / 8; geometric mean with the flat D),
    # screws with EPDM washers every ~0.2 m add boundary loss
    add(Material("tin_roof", "solid",
                 _plate_modes("tin_roof", E=200e9, rho=7850.0, h=0.5e-3, a=0.9, b=0.76,
                              D=math.sqrt(2.29 * 200e9 * 0.5e-3 * 0.018 ** 2 / 8.0), m2=7850.0 * 0.5e-3 * 1.15,
                              eta_int=1e-3, eta_edge=0.006, f_lo=100.0, f_hi=13000.0, step_oct=1 / 6),
                 hardness=0.92, splash=0.85, radiation_hp_hz=100.0, radiation_lp_hz=16000.0,
                 level_db=58.0, ucs_cat="RAINMetl"))
    # car roof / bonnet: 1.0 mm aluminium (or 0.8 mm steel) between roof bows, with the
    # anti-flutter adhesive and bitumen pads every car has: a damped, drummy metal
    add(Material("car_body", "solid",
                 _plate_modes("car_body", E=70e9, rho=2700.0, h=1.0e-3, a=1.2, b=0.7,
                              eta_int=1e-3, eta_edge=0.010, f_lo=80.0, f_hi=10000.0, step_oct=1 / 6),
                 hardness=0.9, splash=0.8, radiation_hp_hz=60.0, radiation_lp_hz=12000.0,
                 level_db=56.0, ucs_cat="RAINMetl"))
    # half-round 0.6 mm galvanised gutter, 125 mm, brackets every 0.75 m: the curved section is
    # stiff (developed width 0.2 m, shell modes from a few hundred Hz); in rain it carries water,
    # whose loading and the brackets damp it well beyond a free sheet
    add(Material("metal_gutter", "solid",
                 _plate_modes("metal_gutter", E=200e9, rho=7850.0, h=0.6e-3, a=0.75, b=0.2,
                              D=200e9 * 0.6e-3 ** 3 / 10.92 * 40.0, eta_int=1e-3, eta_edge=0.012,
                              f_lo=250.0, f_hi=12000.0, step_oct=1 / 6),
                 hardness=0.9, splash=0.9, radiation_hp_hz=200.0, radiation_lp_hz=16000.0,
                 level_db=58.0, ucs_cat="RAINMetl"))
    # Galvanised bucket: cylindrical-shell (n, m) modes, measured-style ratios; free-standing
    add(Material("bucket", "solid",
                 _build_eta("bucket", [310, 455, 690, 960, 1240, 1610, 2050, 2580, 3150, 3890,
                                       4700, 5600, 6900, 8300], _eta(1e-3, 0.003), tilt_db_oct=-1.0),
                 hardness=0.92, splash=0.9, radiation_hp_hz=150.0, radiation_lp_hz=16000.0,
                 level_db=60.0, ucs_cat="RAINMetl"))

    # ---- Glass: eta_int ~ 1e-3 but a pane sits in putty / gaskets, whose edge losses dominate
    # (in-situ total loss factor of mounted glazing ~ 0.03-0.1 below 1 kHz). Coincidence ~ 2.4-3 kHz,
    # point mobility ~ 2e-3 s/kg: a raindrop gives a bright tick with little sustain.
    # pane lying flat (skylight, windscreen, table): 5 mm, 1.0 x 0.8 m on supports
    add(Material("glass", "solid",
                 _with_click("glass", _plate_modes("glass", E=70e9, rho=2500.0, h=5e-3, a=1.0, b=0.8, nu=0.22,
                                                   eta_int=1e-3, eta_edge=0.05, f_lo=250.0, f_hi=15000.0,
                                                   step_oct=1 / 6),
                             E=70e9, rho=2500.0, h=5e-3, nu=0.22, click=(1500.0, 14000.0, 10, 0.005, 0.002),
                             tilt_db_oct=1.5),
                 hardness=0.97, splash=0.95, radiation_hp_hz=300.0, radiation_lp_hz=18000.0,
                 level_db=54.0, ucs_cat="RAINGlas"))
    # window: 4 mm glazing 1.0 x 1.2 m in a frame (glazing beads / sealant)
    add(Material("window", "solid",
                 _with_click("window", _plate_modes("window", E=70e9, rho=2500.0, h=4e-3, a=1.0, b=1.2, nu=0.22,
                                                    eta_int=1e-3, eta_edge=0.05, f_lo=250.0, f_hi=14000.0,
                                                    step_oct=1 / 6),
                             E=70e9, rho=2500.0, h=4e-3, nu=0.22, click=(1500.0, 12000.0, 10, 0.005, 0.002),
                             tilt_db_oct=1.0),
                 hardness=0.95, splash=0.9, radiation_hp_hz=250.0, radiation_lp_hz=14000.0,
                 level_db=50.0, ucs_cat="RAINGlas"))

    # ---- Wood / plastic: lossy (eta 0.01-0.1) -> warm, short knocks
    # deck boards / door: 25 mm softwood (orthotropic, E_eff ~ 4 GPa), 0.6 x 0.6 m between joists,
    # nailed: eta_int 0.03 + joint friction (Craik: timber floors ~0.05-0.1); coincidence ~ 0.9 kHz
    add(Material("wood", "solid",
                 _with_click("wood", _plate_modes("wood", E=4e9, rho=500.0, h=0.025, a=0.6, b=0.6, nu=0.3,
                                                  eta_int=0.03, eta_edge=0.05, f_lo=150.0, f_hi=9000.0,
                                                  step_oct=1 / 4),
                             E=4e9, rho=500.0, h=0.025, click=(500.0, 7000.0, 10, 0.010, 0.003), tilt_db_oct=-2.0),
                 hardness=0.6, splash=0.5, radiation_hp_hz=100.0, radiation_lp_hz=9000.0,
                 level_db=47.0, ucs_cat="RAINWood"))
    # plastic lid / chair / PVC gutter / polycarbonate sheet: 4 mm, 0.5 x 0.6 m, polymer eta ~ 0.05-0.1
    add(Material("plastic", "solid",
                 _with_click("plastic", _plate_modes("plastic", E=2.4e9, rho=1200.0, h=4e-3, a=0.5, b=0.6, nu=0.37,
                                                     eta_int=0.07, eta_edge=0.03, f_lo=200.0, f_hi=10000.0,
                                                     step_oct=1 / 4),
                             E=2.4e9, rho=1200.0, h=4e-3, nu=0.37, click=(800.0, 10000.0, 8, 0.006, 0.002)),
                 hardness=0.7, splash=0.7, radiation_hp_hz=150.0, radiation_lp_hz=14000.0,
                 level_db=52.0, ucs_cat="RAINPlas"))

    # ---- Masonry and hard ground. A slab on grade or a 215 mm wall has a point mobility
    # 1 / (8 sqrt(D m'')) so small that a raindrop barely moves it: what is heard is the impact
    # click coloured by the surface texture ("grit": short, dense micro-resonances), plus for a
    # free-standing wall a faint low thud of its bending modes (Craik eta ~ 1/sqrt(f)).
    add(Material("concrete", "solid",
                 _grit("concrete", 1200.0, 11000.0, 10, 0.009, 0.004, peak_hz=4000.0, peak_db=4.0),
                 hardness=0.9, splash=0.95, radiation_hp_hz=300.0, radiation_lp_hz=16000.0,
                 level_db=48.0, ucs_cat="RAINConc"))
    # brick wall: porous fired clay (softer, absorbent surface -> duller click than concrete).
    # 215 mm (E 16 GPa, rho 1800): Y = 1 / (8 sqrt(D m'')) ~ 2e-6 s/kg, so its bending modes are
    # not heard at all; the dull thud is the local contact of the drop on the soft-faced brick
    # (heavily damped, eta_int 0.01-0.05 is irrelevant at this mobility), the click its texture.
    add(Material("brick_wall", "solid",
                 [(f, t, 0.5 * g) for f, t, g in _grit("brick_wall/thud", 160.0, 600.0, 6, 0.02, 0.010)]
                 + _grit("brick_wall", 700.0, 8000.0, 12, 0.007, 0.0025, tilt_db_oct=-1.5,
                         peak_hz=2000.0, peak_db=3.0),
                 hardness=0.85, splash=0.7, radiation_hp_hz=120.0, radiation_lp_hz=12000.0,
                 level_db=46.0, ucs_cat="RAINConc"))
    add(Material("asphalt", "solid",
                 _grit("asphalt", 800.0, 9000.0, 10, 0.007, 0.003, tilt_db_oct=-2.0,
                       peak_hz=2500.0, peak_db=3.0),
                 hardness=0.8, splash=0.85, radiation_hp_hz=250.0, radiation_lp_hz=12000.0,
                 level_db=46.0, ucs_cat="RAINConc"))
    # clay roof tile: 12 mm, 0.33 x 0.21 m resting on battens with overlaps (friction): a short
    # pitched "clink" (first modes ~0.5-2 kHz), eta_int 0.01 + loose support
    add(Material("tile", "solid",
                 _with_click("tile", _plate_modes("tile", E=20e9, rho=2000.0, h=0.012, a=0.33, b=0.21, nu=0.2,
                                                  eta_int=0.01, eta_edge=0.05, f_lo=300.0, f_hi=12000.0,
                                                  step_oct=1 / 4),
                             E=20e9, rho=2000.0, h=0.012, nu=0.2, click=(1500.0, 12000.0, 8, 0.006, 0.003)),
                 hardness=0.9, splash=0.9, radiation_hp_hz=300.0, radiation_lp_hz=15000.0,
                 level_db=48.0, ucs_cat="RAINConc"))
    add(Material("gravel", "porous",
                 _grit("gravel", 1500.0, 15000.0, 14, 0.006, 0.002, tilt_db_oct=1.0,
                       peak_hz=6000.0, peak_db=4.0),
                 hardness=0.85, splash=0.6, radiation_hp_hz=400.0, radiation_lp_hz=15000.0,
                 level_db=47.0, ucs_cat="RAINGen"))
    add(Material("soil", "porous",
                 _grit("soil", 120.0, 1600.0, 8, 0.008, 0.003, tilt_db_oct=-4.0,
                       peak_hz=300.0, peak_db=3.0),
                 hardness=0.15, splash=0.2, radiation_hp_hz=80.0, radiation_lp_hz=4000.0,
                 level_db=38.0, ucs_cat="RAINGen"))

    # ---- Vegetation: many small, very lossy leaf plates (eta ~ 0.1-0.3) -> crisp ticks, no
    # tonal ring. A leaf is an unbaffled dipole radiator: efficiency rises ~(ka)^2 (12 dB/octave)
    # up to ka ~ 1, i.e. ~1.2 kHz for a broadleaf of radius ~4.5 cm (2nd-order radiation HP).
    add(Material("grass", "vegetation",
                 _grit("grass", 400.0, 6000.0, 10, 0.008, 0.003, tilt_db_oct=-2.0,
                       peak_hz=1500.0, peak_db=4.0),
                 hardness=0.2, splash=0.1, radiation_hp_hz=200.0, radiation_lp_hz=9000.0,
                 level_db=38.0, ucs_cat="RAINVege"))
    add(Material("leaves", "vegetation",
                 _grit("leaves", 250.0, 6000.0, 16, 0.015, 0.004, tilt_db_oct=-1.0,
                       peak_hz=1200.0, peak_db=3.0),
                 hardness=0.35, splash=0.3, radiation_hp_hz=1200.0, radiation_lp_hz=12000.0,
                 level_db=50.0, ucs_cat="RAINVege"))
    add(Material("needles", "vegetation",
                 _grit("needles", 1000.0, 9000.0, 10, 0.006, 0.002, peak_hz=4000.0, peak_db=3.0),
                 hardness=0.25, splash=0.15, radiation_hp_hz=400.0, radiation_lp_hz=12000.0,
                 level_db=40.0, ucs_cat="RAINVege"))

    # ---- Membranes: tensioned fabric, drum-like modes (Bessel ratios), but a fabric weighs
    # 0.06-0.4 kg/m^2 against an air load of ~2 rho0 / k ~ 0.2 kg/m^2: radiation damping makes
    # eta ~ 0.35-0.5 -> a short "pop", heavier woven cotton duller still.
    add(Material("fabric", "membrane",
                 _membrane("fabric", 110.0, 14, 0.45, tilt_db_oct=-3.0),
                 hardness=0.3, splash=0.2, radiation_hp_hz=80.0, radiation_lp_hz=8000.0,
                 level_db=48.0, ucs_cat="RAINClth"))
    # nylon umbrella gores (60 g/m^2, taut between ribs): the brightest membrane
    add(Material("umbrella", "membrane",
                 _membrane("umbrella", 160.0, 16, 0.35, extra=(2000.0, 7000.0, 6), eta_extra=0.2,
                           tilt_db_oct=-2.0),
                 hardness=0.35, splash=0.3, radiation_hp_hz=90.0, radiation_lp_hz=10000.0,
                 level_db=56.0, ucs_cat="RAINClth"))
    # polyester / nylon tent fly (70 g/m^2), larger panels -> lower
    add(Material("tent", "membrane",
                 _membrane("tent", 85.0, 16, 0.4, extra=(1500.0, 5000.0, 5), eta_extra=0.2,
                           tilt_db_oct=-3.0),
                 hardness=0.3, splash=0.25, radiation_hp_hz=60.0, radiation_lp_hz=9000.0,
                 level_db=54.0, ucs_cat="RAINClth"))
    # cotton canvas (awning, bell tent; 350 g/m^2, woven + wet): dull thud
    add(Material("canvas", "membrane",
                 _membrane("canvas", 65.0, 14, 0.5, tilt_db_oct=-4.0),
                 hardness=0.25, splash=0.2, radiation_hp_hz=50.0, radiation_lp_hz=6000.0,
                 level_db=50.0, ucs_cat="RAINClth"))

    # Water: click + bubble, no modes
    add(Material("puddle", "water", [], hardness=0.0, splash=1.0, radiation_hp_hz=100.0,
                 radiation_lp_hz=20000.0, level_db=42.0, ucs_cat="RAINWatr"))
    add(Material("water", "water", [], hardness=0.0, splash=1.0, radiation_hp_hz=80.0,
                 radiation_lp_hz=20000.0, level_db=42.0, ucs_cat="RAINWatr"))

    # Snow cover: porous, almost silent
    add(Material("snow", "snow",
                 _grit("snow", 80.0, 600.0, 5, 0.004, 0.002, tilt_db_oct=-6.0),
                 hardness=0.05, splash=0.0, radiation_hp_hz=60.0, radiation_lp_hz=2500.0,
                 level_db=20.0, ucs_cat="RAINGen"))
    return M


MATERIALS: dict[str, Material] = _make_materials()

_ALIASES = {
    "metal": "metal_sheet", "steel": "metal_sheet", "aluminium": "metal_sheet", "aluminum": "metal_sheet",
    "sheet_metal": "metal_sheet", "shed": "metal_sheet", "tin": "tin_roof", "corrugated": "tin_roof",
    "roof_metal": "tin_roof", "car": "car_body", "vehicle": "car_body", "windshield": "glass",
    "windscreen": "glass", "pane": "glass", "glazing": "window", "timber": "wood", "deck": "wood",
    "plank": "wood", "fence": "wood", "polycarbonate": "plastic", "bin": "plastic",
    "stone": "concrete", "pavement": "concrete", "paving": "concrete", "sidewalk": "concrete",
    "brick": "brick_wall", "masonry": "brick_wall", "wall": "brick_wall", "door": "wood",
    "aluminium_body": "car_body", "road": "asphalt", "tarmac": "asphalt", "roof": "tile", "roof_tile": "tile",
    "slate": "tile", "ceramic": "tile", "pebbles": "gravel", "mud": "soil", "dirt": "soil",
    "earth": "soil", "forest_floor": "leaves", "lawn": "grass", "meadow": "grass",
    "deciduous": "leaves", "broadleaf": "leaves", "foliage": "leaves", "canopy": "leaves",
    "conifer": "needles", "pine": "needles", "spruce": "needles", "cloth": "fabric",
    "awning": "canvas", "nylon": "umbrella", "hood": "fabric", "pond": "water", "lake": "water",
    "river": "water", "sea": "water", "pool": "water", "gutter": "metal_gutter",
    "downspout": "metal_gutter", "snow_fresh": "snow", "snow_old": "snow",
}


def get_material(name: str) -> Material:
    """Material by name or alias (case-insensitive); unknown names fall back to 'concrete'."""
    if isinstance(name, Material):
        return name
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in MATERIALS:
        return MATERIALS[key]
    if key in _ALIASES:
        return MATERIALS[_ALIASES[key]]
    for mname in sorted(MATERIALS, key=len, reverse=True):
        if mname in key:
            return MATERIALS[mname]
    for alias, target in _ALIASES.items():
        if alias in key:
            return MATERIALS[target]
    return MATERIALS["concrete"]
