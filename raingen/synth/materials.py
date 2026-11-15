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


def _plate(name, f11, aspect, n_low, f_hi_lo, f_hi, n_high, t60_lo, t60_hi, **kw):
    freqs = _plate_series(f11, aspect, n_low) + _log_spread(name, f_hi_lo, f_hi, n_high)
    return _build(name, freqs, t60_lo, t60_hi, **kw)


def _membrane(name, f01, count, t60_lo, t60_hi, extra=None, **kw):
    freqs = [f01 * r for r in _MEMBRANE_RATIOS[:count]]
    if extra:
        freqs += _log_spread(name + "/weave", *extra)
    return _build(name, freqs, t60_lo, t60_hi, **kw)


def _grit(name, f_lo, f_hi, count, t60_lo, t60_hi, **kw):
    return _build(name, _log_spread(name, f_lo, f_hi, count, jitter=0.3), t60_lo, t60_hi, **kw)


# ---------------------------------------------------------------------------

def _make_materials() -> dict[str, Material]:
    M: dict[str, Material] = {}

    def add(m: Material):
        M[m.name] = m

    # Thin metals: bright, ringing, radiate up toward coincidence (~24 kHz for 0.5 mm steel)
    add(Material("metal_sheet", "solid",
                 _plate("metal_sheet", 110.0, 1.5, 10, 1200.0, 14000.0, 12, 0.9, 0.12,
                        tilt_db_oct=1.5, peak_hz=4000.0, peak_db=6.0, peak_oct=1.2),
                 hardness=0.95, splash=0.9, radiation_hp_hz=150.0, radiation_lp_hz=18000.0,
                 level_db=60.0, ucs_cat="RAINMetl"))
    add(Material("tin_roof", "solid",
                 _plate("tin_roof", 70.0, 2.5, 10, 900.0, 12000.0, 14, 0.5, 0.06,
                        tilt_db_oct=1.0, peak_hz=2500.0, peak_db=6.0, peak_oct=1.3),
                 hardness=0.92, splash=0.85, radiation_hp_hz=100.0, radiation_lp_hz=16000.0,
                 level_db=58.0, ucs_cat="RAINMetl"))
    add(Material("car_body", "solid",
                 _plate("car_body", 60.0, 1.3, 8, 500.0, 9000.0, 12, 0.25, 0.03,
                        tilt_db_oct=-1.0, peak_hz=1200.0, peak_db=4.0, peak_oct=1.2),
                 hardness=0.9, splash=0.8, radiation_hp_hz=60.0, radiation_lp_hz=12000.0,
                 level_db=56.0, ucs_cat="RAINMetl"))
    add(Material("metal_gutter", "solid",
                 _plate("metal_gutter", 260.0, 6.0, 10, 1500.0, 12000.0, 10, 0.5, 0.08,
                        tilt_db_oct=0.0, peak_hz=3000.0, peak_db=5.0),
                 hardness=0.9, splash=0.9, radiation_hp_hz=200.0, radiation_lp_hz=16000.0,
                 level_db=58.0, ucs_cat="RAINMetl"))
    # Galvanized bucket: cylindrical-shell (n, m) modes, measured-style ratios
    add(Material("bucket", "solid",
                 _build("bucket", [310, 455, 690, 960, 1240, 1610, 2050, 2580, 3150, 3890,
                                   4700, 5600, 6900, 8300], 0.8, 0.15,
                        tilt_db_oct=-1.0, peak_hz=1600.0, peak_db=4.0),
                 hardness=0.92, splash=0.9, radiation_hp_hz=150.0, radiation_lp_hz=16000.0,
                 level_db=60.0, ucs_cat="RAINMetl"))

    # Glass: 4-6 mm pane, coincidence ~3 kHz -> sharp bright tick, short ring
    add(Material("glass", "solid",
                 _plate("glass", 35.0, 1.3, 8, 1000.0, 15000.0, 14, 0.35, 0.05,
                        tilt_db_oct=2.0, peak_hz=3000.0, peak_db=8.0, peak_oct=1.0),
                 hardness=0.97, splash=0.95, radiation_hp_hz=300.0, radiation_lp_hz=18000.0,
                 level_db=54.0, ucs_cat="RAINGlas"))
    add(Material("window", "solid",
                 _plate("window", 45.0, 1.4, 8, 800.0, 12000.0, 12, 0.15, 0.03,
                        tilt_db_oct=1.0, peak_hz=2500.0, peak_db=6.0),
                 hardness=0.95, splash=0.9, radiation_hp_hz=250.0, radiation_lp_hz=14000.0,
                 level_db=50.0, ucs_cat="RAINGlas"))

    # Wood / plastic: warmer knocks, coincidence ~1.4 kHz for 12 mm ply
    add(Material("wood", "solid",
                 _plate("wood", 90.0, 1.8, 6, 400.0, 8000.0, 12, 0.08, 0.01,
                        tilt_db_oct=-2.0, peak_hz=1000.0, peak_db=5.0),
                 hardness=0.6, splash=0.5, radiation_hp_hz=100.0, radiation_lp_hz=9000.0,
                 level_db=47.0, ucs_cat="RAINWood"))
    add(Material("plastic", "solid",
                 _plate("plastic", 120.0, 1.4, 6, 600.0, 10000.0, 12, 0.06, 0.012,
                        tilt_db_oct=-1.0, peak_hz=1800.0, peak_db=5.0),
                 hardness=0.7, splash=0.7, radiation_hp_hz=150.0, radiation_lp_hz=14000.0,
                 level_db=52.0, ucs_cat="RAINPlas"))

    # Hard ground: grit resonances only
    add(Material("concrete", "solid",
                 _grit("concrete", 1200.0, 11000.0, 10, 0.009, 0.004, peak_hz=4000.0, peak_db=4.0),
                 hardness=0.9, splash=0.95, radiation_hp_hz=300.0, radiation_lp_hz=16000.0,
                 level_db=48.0, ucs_cat="RAINConc"))
    add(Material("asphalt", "solid",
                 _grit("asphalt", 800.0, 9000.0, 10, 0.007, 0.003, tilt_db_oct=-2.0,
                       peak_hz=2500.0, peak_db=3.0),
                 hardness=0.8, splash=0.85, radiation_hp_hz=250.0, radiation_lp_hz=12000.0,
                 level_db=46.0, ucs_cat="RAINConc"))
    add(Material("tile", "solid",
                 _grit("tile", 900.0, 12000.0, 12, 0.03, 0.008, peak_hz=3500.0, peak_db=5.0),
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

    # Vegetation: many small leaf plates (variants detune heavily at synthesis time)
    add(Material("grass", "vegetation",
                 _grit("grass", 400.0, 6000.0, 10, 0.008, 0.003, tilt_db_oct=-2.0,
                       peak_hz=1500.0, peak_db=4.0),
                 hardness=0.2, splash=0.1, radiation_hp_hz=200.0, radiation_lp_hz=9000.0,
                 level_db=38.0, ucs_cat="RAINVege"))
    add(Material("leaves", "vegetation",
                 _grit("leaves", 250.0, 6000.0, 16, 0.03, 0.006, tilt_db_oct=-1.0,
                       peak_hz=1200.0, peak_db=5.0),
                 hardness=0.35, splash=0.3, radiation_hp_hz=150.0, radiation_lp_hz=12000.0,
                 level_db=50.0, ucs_cat="RAINVege"))
    add(Material("needles", "vegetation",
                 _grit("needles", 1000.0, 9000.0, 10, 0.006, 0.002, peak_hz=4000.0, peak_db=3.0),
                 hardness=0.25, splash=0.15, radiation_hp_hz=400.0, radiation_lp_hz=12000.0,
                 level_db=40.0, ucs_cat="RAINVege"))

    # Membranes: tensioned fabrics act like drumheads
    add(Material("fabric", "membrane",
                 _membrane("fabric", 110.0, 14, 0.05, 0.015, tilt_db_oct=-3.0),
                 hardness=0.3, splash=0.2, radiation_hp_hz=80.0, radiation_lp_hz=8000.0,
                 level_db=48.0, ucs_cat="RAINClth"))
    add(Material("umbrella", "membrane",
                 _membrane("umbrella", 160.0, 16, 0.07, 0.02, extra=(2000.0, 7000.0, 6),
                           tilt_db_oct=-2.0, peak_hz=900.0, peak_db=3.0),
                 hardness=0.35, splash=0.3, radiation_hp_hz=90.0, radiation_lp_hz=10000.0,
                 level_db=56.0, ucs_cat="RAINClth"))
    add(Material("tent", "membrane",
                 _membrane("tent", 85.0, 16, 0.08, 0.02, extra=(1500.0, 5000.0, 5),
                           tilt_db_oct=-3.0),
                 hardness=0.3, splash=0.25, radiation_hp_hz=60.0, radiation_lp_hz=9000.0,
                 level_db=54.0, ucs_cat="RAINClth"))
    add(Material("canvas", "membrane",
                 _membrane("canvas", 65.0, 14, 0.05, 0.015, tilt_db_oct=-4.0),
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
    "brick": "concrete", "road": "asphalt", "tarmac": "asphalt", "roof": "tile", "roof_tile": "tile",
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
