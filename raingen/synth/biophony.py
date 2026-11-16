"""Life and towns that respond to the weather: insects, birds, frogs and traffic.

Physics / ecology:
* Crickets: Dolbear's law (Dolbear 1897) for the snowy tree cricket, T_F = 50 + (N60 - 40) / 4,
  i.e. N60 = 4 (T_F - 50) + 40 = 7.2 T_C - 32 chirps per minute; field-cricket chirp and syllable rates and
  tree-cricket carrier frequencies scale the same way with temperature (Walker 1962). Crickets
  are silent below ~12 C, in rain above ~2 mm/h, in strong wind and by day (they start up
  toward dusk).
* Cicadas sing by day in hot, dry weather (> ~22 C).
* Birds: a dawn chorus around sunrise (sunrise from latitude and season: Cooper 1969
  declination, cos w0 = -tan(phi) tan(delta)), a weaker dusk chorus, scattered daytime song.
  Song is suppressed in rain and wind (Lengagne & Slater 2002: a tawny owl's call covers a
  69x larger area in dry weather than in rain, and owls call less in rain) and rebounds
  after the rain stops. Bird song here is procedural and deliberately generic; recorded,
  region-true birds come from Nature packs.
* Frogs chorus on warm (> ~10 C), wet evenings and nights in spring and summer.
* Traffic: road-traffic level ~ 10 log10(flow) + 30 log10(speed) (CNOSSOS-EU style); line
  source (-3 dB per doubling of distance); a daily flow curve; wet roads add ~5-15 dB of
  tyre spray hiss above 1 kHz (Sandberg & Ejsmont, *Tyre/Road Noise Reference Book*, 2002).
  Pass-bys follow the closest-approach geometry with Doppler on the engine harmonics.

Stem 'ambience'. Levels are peak / RMS dB SPL at 1 m (birds, insects, frogs) or at 7.5 m
(vehicles), propagated with spherical spreading and ISO 9613-1 absorption.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy import signal

from raingen.core.constants import SAMPLE_RATE, speed_of_sound
from raingen.core.types import BlockContext, Emission
from raingen.physics.atmosphere import ground_effect_sos, iso9613_alpha_db_per_m

from .wind import (AirPath, PendingMixer, Ramp, StatefulSOS, StaticSHEncoder, bandpass_sos, block_start_sample, db_spl,
                   highpass_sos, lowpass_sos, noise_rms_gain, peaking_sos, poisson_offsets, ramp_frac, shelter_of,
                   slope_sos, smoothstep)

_SEASON_DOY = {"spring": 80, "summer": 172, "autumn": 266, "fall": 266, "winter": 355}
TROPICAL = ("tropical",)


def dolbear_chirps_per_min(temp_c: float) -> float:
    """Snowy tree cricket chirp rate (Dolbear 1897): T_F = 50 + (N60 - 40) / 4, so N60 = 4 (T_F - 50) + 40."""
    t_f = 1.8 * float(temp_c) + 32.0
    return max(4.0 * (t_f - 50.0) + 40.0, 0.0)


def sun_times(latitude_deg: float, season: str) -> tuple[float, float]:
    """(sunrise, sunset) in local solar hours for a representative day of the season."""
    doy = _SEASON_DOY.get(str(season).lower(), 172)
    if latitude_deg < 0.0:
        doy = (doy + 182) % 365
    decl = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + doy) / 365.0)
    phi = math.radians(min(max(float(latitude_deg), -89.9), 89.9))
    x = min(max(-math.tan(phi) * math.tan(decl), -1.0), 1.0)
    w0 = math.degrees(math.acos(x))
    return 12.0 - w0 / 15.0, 12.0 + w0 / 15.0


def _mode(v) -> str:
    s = str(v if v is not None else "auto").strip().lower()
    if s in ("on", "true", "yes", "1"):
        return "on"
    if s in ("off", "false", "no", "0", "none"):
        return "off"
    return "auto"


def _gauss(x: float, mu: float, sigma: float) -> float:
    """Gaussian bump, exactly 0 beyond 4 sigma (activity smoothing hides the 3e-4 step)."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) if abs(z) < 4.0 else 0.0


def _hour_dist(a: float, b: float) -> float:
    d = (a - b) % 24.0
    return d - 24.0 if d > 12.0 else d


def _norm(y: np.ndarray) -> np.ndarray:
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    return y / pk if pk > 0 else y


# --------------------------------------------------------------------------
# Elementary sound shapes
# --------------------------------------------------------------------------

def _tone(sr: int, dur: float, f0: float, f1: float, shape: str = "lin", vib_hz: float = 0.0,
          vib_depth: float = 0.0, harm=(), attack: float = 0.15, release: float = 0.35) -> np.ndarray:
    """FM tone with a smooth envelope (bird notes, frog peeps, owl hoots)."""
    n = max(int(dur * sr), 16)
    x = np.arange(n) / n
    if shape == "exp":
        f = f0 * (f1 / f0) ** x
    elif shape == "chevron":
        f = f0 + (f1 - f0) * np.sin(math.pi * x)
    elif shape == "hook":
        f = f0 + (f1 - f0) * x ** 3
    else:
        f = f0 + (f1 - f0) * x
    if vib_hz > 0.0 and vib_depth > 0.0:
        f = f * (1.0 + vib_depth * np.sin(2.0 * math.pi * vib_hz * np.arange(n) / sr))
    f = np.clip(f, 20.0, 0.45 * sr)
    ph = 2.0 * math.pi * np.cumsum(f) / sr
    y = np.sin(ph)
    for k, h in enumerate(harm):
        if h > 0.0:
            y = y + h * np.sin((k + 2) * ph) * ((k + 2) * f < 0.45 * sr)
    a = max(int(attack * n), 1)
    r = max(int(release * n), 1)
    env = np.ones(n)
    env[:a] = np.sin(0.5 * math.pi * np.arange(a) / a) ** 2
    env[n - r:] = np.cos(0.5 * math.pi * np.arange(r) / r) ** 2
    return y * env


def _pulse_train(rng: np.random.Generator, sr: int, dur: float, rate: float, pulse_frac: float, carrier: float,
                 noise: bool = False, rate_end: Optional[float] = None, glide: float = 0.0) -> np.ndarray:
    """Carrier gated by a train of smooth pulses (frog trills, croaks, katydid buzz)."""
    n = max(int(dur * sr), 16)
    r1 = rate if rate_end is None else rate_end
    rates = np.linspace(rate, r1, n)
    ph = np.cumsum(rates) / sr
    frac = ph % 1.0
    env = np.where(frac < pulse_frac, np.sin(math.pi * frac / max(pulse_frac, 1e-3)) ** 2, 0.0)
    if noise:
        c = signal.sosfilt(bandpass_sos(carrier, 3.0, sr), rng.standard_normal(n))
    else:
        f = carrier * (1.0 + glide * np.linspace(0.0, 1.0, n))
        c = np.sin(2.0 * math.pi * np.cumsum(f) / sr + rng.uniform(0.0, 2.0 * math.pi))
    a = max(int(0.05 * n), 1)
    shape = np.ones(n)
    shape[:a] = np.linspace(0.0, 1.0, a)
    shape[n - a:] = np.linspace(1.0, 0.0, a)
    return _norm(c * env * shape)


def _gap(sr: int, s: float) -> np.ndarray:
    return np.zeros(max(int(s * sr), 0))


# --------------------------------------------------------------------------
# Bird songs (procedural, generic)
# --------------------------------------------------------------------------

def _song_warble(rng, sr, p):
    parts = []
    f = rng.uniform(2800.0, 4800.0) * p
    for _ in range(int(rng.integers(6, 17))):
        f = float(np.clip(f * rng.lognormal(0.0, 0.18), 2200.0 * p, 7000.0 * p))
        f1 = f * rng.lognormal(0.0, 0.2)
        shape = ["lin", "exp", "chevron", "hook"][int(rng.integers(0, 4))]
        parts.append(_tone(sr, rng.uniform(0.025, 0.08), f, f1, shape, harm=(0.05,)))
        parts.append(_gap(sr, rng.uniform(0.01, 0.04)))
    return _norm(np.concatenate(parts))


def _song_flute(rng, sr, p):
    parts = []
    base = rng.uniform(1600.0, 2800.0) * p
    for _ in range(int(rng.integers(3, 7))):
        f0 = base * rng.choice([1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 6])
        parts.append(_tone(sr, rng.uniform(0.12, 0.3), f0, f0 * rng.uniform(0.92, 1.08), "chevron",
                           vib_hz=rng.uniform(22.0, 38.0), vib_depth=rng.uniform(0.006, 0.02), harm=(0.15, 0.04)))
        parts.append(_gap(sr, rng.uniform(0.03, 0.09)))
    if rng.random() < 0.6:                      # the twittery coda of a thrush / blackbird phrase
        for _ in range(int(rng.integers(3, 8))):
            f0 = rng.uniform(4500.0, 7000.0) * p
            parts.append(_tone(sr, rng.uniform(0.015, 0.04), f0, f0 * rng.uniform(0.7, 1.3), "lin"))
            parts.append(_gap(sr, rng.uniform(0.008, 0.02)))
    return _norm(np.concatenate(parts))


def _song_trill(rng, sr, p):
    parts = []
    for _ in range(int(rng.integers(1, 3))):
        rate = rng.uniform(14.0, 28.0)
        f0 = rng.uniform(4000.0, 7000.0) * p
        f1 = f0 * rng.uniform(0.75, 1.25)
        for _k in range(int(rate * rng.uniform(0.6, 1.4))):
            parts.append(_tone(sr, 0.6 / rate, f0, f1, "lin", attack=0.2, release=0.4))
            parts.append(_gap(sr, 0.4 / rate))
        parts.append(_gap(sr, rng.uniform(0.02, 0.08)))
    return _norm(np.concatenate(parts))


def _song_two_note(rng, sr, p):
    hi = rng.uniform(4800.0, 6200.0) * p
    lo = hi * rng.uniform(0.62, 0.8)
    parts = []
    for _ in range(int(rng.integers(3, 9))):
        parts.append(_tone(sr, rng.uniform(0.07, 0.12), hi, hi * 0.97, "lin"))
        parts.append(_gap(sr, 0.03))
        parts.append(_tone(sr, rng.uniform(0.07, 0.12), lo, lo * 0.98, "lin"))
        parts.append(_gap(sr, rng.uniform(0.06, 0.12)))
    return _norm(np.concatenate(parts))


def _song_chip(rng, sr, p):
    parts = []
    for _ in range(int(rng.integers(1, 4))):
        f0 = rng.uniform(3500.0, 7000.0) * p
        parts.append(_tone(sr, rng.uniform(0.015, 0.03), f0, f0 * rng.uniform(0.55, 0.8), "exp", attack=0.1, release=0.5))
        parts.append(_gap(sr, rng.uniform(0.08, 0.25)))
    return _norm(np.concatenate(parts))


def _song_coo(rng, sr, p):
    f0 = rng.uniform(450.0, 620.0) * p
    pattern = [(0.35, 1.0), (0.55, 1.08), (0.3, 0.98)] if rng.random() < 0.5 else \
        [(0.25, 1.0), (0.6, 1.1), (0.3, 1.0), (0.3, 0.97), (0.25, 0.95)]
    parts = []
    for dur, r in pattern:
        parts.append(_tone(sr, dur * rng.uniform(0.9, 1.1), f0 * r, f0 * r * 0.93, "chevron", harm=(0.3, 0.08),
                           attack=0.25, release=0.4))
        parts.append(_gap(sr, rng.uniform(0.08, 0.2)))
    return _norm(np.concatenate(parts))


def _song_caw(rng, sr, p):
    parts = []
    for _ in range(int(rng.integers(2, 6))):
        dur = rng.uniform(0.22, 0.4)
        n = int(dur * sr)
        f0 = rng.uniform(420.0, 620.0) * p
        f = f0 * (1.0 + 0.04 * rng.standard_normal(n).cumsum() / math.sqrt(n)) * np.linspace(1.05, 0.92, n)
        ph = 2.0 * math.pi * np.cumsum(np.clip(f, 50.0, 4000.0)) / sr
        y = sum(np.sin(k * ph) / k for k in range(1, 13) if k * f0 < 0.45 * sr)
        y = y * (1.0 + 0.5 * signal.sosfilt(lowpass_sos(80.0, sr), rng.standard_normal(n)) * 6.0)
        y = signal.sosfilt(np.vstack([bandpass_sos(rng.uniform(1100.0, 1700.0), 1.4, sr), highpass_sos(300.0, sr)]), y)
        env = np.sin(math.pi * np.arange(n) / n) ** 0.6
        parts.append(_norm(y * env))
        parts.append(_gap(sr, rng.uniform(0.25, 0.5)))
    return _norm(np.concatenate(parts))


def _song_owl(rng, sr, p):
    f0 = rng.uniform(400.0, 650.0) * p
    parts = [_tone(sr, rng.uniform(0.35, 0.55), f0, f0 * 0.97, "chevron", harm=(0.08,), attack=0.2, release=0.4),
             _gap(sr, rng.uniform(1.8, 3.2))]
    for _ in range(int(rng.integers(1, 4))):
        parts.append(_tone(sr, 0.12, f0 * 0.95, f0, "lin", harm=(0.06,)))
        parts.append(_gap(sr, 0.12))
    parts.append(_tone(sr, rng.uniform(0.6, 1.0), f0, f0 * 0.96, "lin", vib_hz=rng.uniform(5.0, 8.0),
                       vib_depth=0.03, harm=(0.08,), attack=0.1, release=0.45))
    return _norm(np.concatenate(parts))


BIRD_SONGS: dict[str, tuple[Callable, float, float]] = {
    # template: (synth, peak dB SPL at 1 m, songs per minute while active)
    "warbler": (_song_warble, 80.0, 6.0),
    "flute": (_song_flute, 82.0, 5.0),
    "trill": (_song_trill, 86.0, 5.0),
    "two_note": (_song_two_note, 82.0, 5.0),
    "chip": (_song_chip, 72.0, 9.0),
    "coo": (_song_coo, 74.0, 2.5),
    "caw": (_song_caw, 84.0, 1.5),
    "owl": (_song_owl, 78.0, 1.5),
}
REGION_BIRDS = {
    "temperate_europe": ("warbler", "flute", "trill", "two_note", "chip", "coo", "caw"),
    "north_america": ("warbler", "flute", "trill", "two_note", "chip", "coo", "caw"),
    "boreal": ("flute", "trill", "warbler", "chip", "caw"),
    "mediterranean": ("warbler", "two_note", "chip", "coo", "trill"),
    "east_asia": ("warbler", "two_note", "chip", "coo", "caw"),
    "tropical": ("flute", "two_note", "warbler", "chip", "coo", "trill"),
    "arid": ("chip", "coo", "caw", "two_note"),
    "oceania": ("flute", "warbler", "chip", "caw"),
}
OWL_REGIONS = ("temperate_europe", "boreal", "north_america", "east_asia", "mediterranean")


# --------------------------------------------------------------------------
# Frog calls
# --------------------------------------------------------------------------

def _frog_common(rng, sr, p):
    return _pulse_train(rng, sr, rng.uniform(0.5, 1.2), rng.uniform(22.0, 35.0), 0.5, rng.uniform(350.0, 550.0) * p, noise=True)


def _frog_tree(rng, sr, p):
    parts = []
    car = rng.uniform(1800.0, 2400.0) * p
    for _ in range(int(rng.integers(4, 13))):
        parts.append(_pulse_train(rng, sr, rng.uniform(0.05, 0.08), rng.uniform(150.0, 200.0), 0.5, car))
        parts.append(_gap(sr, rng.uniform(0.09, 0.14)))
    return _norm(np.concatenate(parts))


def _frog_toad_trill(rng, sr, p):
    return _pulse_train(rng, sr, rng.uniform(3.0, 10.0), rng.uniform(30.0, 40.0), 0.45, rng.uniform(1550.0, 1900.0) * p)


def _frog_peeper(rng, sr, p):
    f0 = rng.uniform(2700.0, 2950.0) * p
    return _norm(_tone(sr, rng.uniform(0.1, 0.14), f0, f0 * rng.uniform(1.06, 1.12), "exp", attack=0.15, release=0.3))


def _frog_chorus(rng, sr, p):
    r0 = rng.uniform(12.0, 16.0)
    return _pulse_train(rng, sr, rng.uniform(0.5, 0.9), r0, 0.45, rng.uniform(2500.0, 3200.0) * p, rate_end=r0 * 1.6,
                        glide=0.08)


def _frog_bull(rng, sr, p):
    parts = []
    for _ in range(int(rng.integers(2, 6))):
        dur = rng.uniform(0.45, 0.8)
        y = _pulse_train(rng, sr, dur, rng.uniform(90.0, 130.0), 0.6, rng.uniform(130.0, 180.0) * p)
        y = signal.sosfilt(np.vstack([peaking_sos(rng.uniform(300.0, 450.0), 9.0, 1.2, sr), lowpass_sos(900.0, sr)]),
                           np.sign(y) * np.abs(y) ** 0.6)
        parts.append(_norm(y))
        parts.append(_gap(sr, rng.uniform(0.3, 0.7)))
    return _norm(np.concatenate(parts))


def _frog_coqui(rng, sr, p):
    co = _tone(sr, 0.1, 1150.0 * p, 1200.0 * p, "lin", attack=0.2, release=0.3)
    qui = _tone(sr, 0.15, 1900.0 * p, 2250.0 * p, "exp", attack=0.15, release=0.35)
    return _norm(np.concatenate([co, _gap(sr, 0.05), qui]))


FROG_CALLS: dict[str, tuple[Callable, float, float]] = {
    # species: (synth, peak dB SPL at 1 m, calls per minute while active)
    "common_frog": (_frog_common, 66.0, 6.0),
    "tree_frog": (_frog_tree, 86.0, 5.0),
    "toad_trill": (_frog_toad_trill, 78.0, 2.0),
    "peeper": (_frog_peeper, 88.0, 50.0),
    "chorus_frog": (_frog_chorus, 82.0, 12.0),
    "bullfrog": (_frog_bull, 84.0, 2.0),
    "coqui": (_frog_coqui, 86.0, 25.0),
}
REGION_FROGS = {
    "temperate_europe": ("common_frog", "tree_frog", "toad_trill"),
    "north_america": ("peeper", "chorus_frog", "toad_trill", "bullfrog"),
    "boreal": ("common_frog", "chorus_frog"),
    "mediterranean": ("tree_frog", "common_frog"),
    "east_asia": ("tree_frog", "chorus_frog", "bullfrog"),
    "tropical": ("coqui", "peeper", "tree_frog", "chorus_frog"),
    "arid": ("toad_trill",),
    "oceania": ("tree_frog", "chorus_frog"),
}


# --------------------------------------------------------------------------
# Insects
# --------------------------------------------------------------------------

INSECT_SPECIES = {
    # chirp_rate(T) in chirps/s, carrier(T) Hz, duty, syllables per chirp, peak dB SPL at 1 m, noisy carrier
    "tree_cricket": {"rate": lambda T: dolbear_chirps_per_min(T) / 60.0, "carrier": lambda T: 1450.0 + 75.0 * T,
                     "duty": 0.45, "syll": 3, "level": 64.0, "noise": False},
    "field_cricket": {"rate": lambda T: 1.25 * dolbear_chirps_per_min(T) / 60.0, "carrier": lambda T: 4600.0,
                      "duty": 0.35, "syll": 4, "level": 70.0, "noise": False},
    "trill_cricket": {"rate": None, "carrier": lambda T: 2400.0 + 20.0 * T, "syll_rate": lambda T: 30.0 + 1.5 * (T - 20.0),
                      "level": 62.0, "noise": False},
    "bell_cricket": {"rate": lambda T: 1.6 + 0.08 * (T - 20.0), "carrier": lambda T: 4400.0, "duty": 0.3, "syll": 8,
                     "level": 62.0, "noise": False},
    "katydid": {"rate": lambda T: 0.8 + 0.05 * (T - 20.0), "carrier": lambda T: 10000.0, "duty": 0.3, "syll": 3,
                "level": 70.0, "noise": True},
}
REGION_INSECTS = {
    "temperate_europe": ("field_cricket", "trill_cricket"),
    "north_america": ("tree_cricket", "field_cricket", "katydid"),
    "boreal": ("field_cricket",),
    "mediterranean": ("field_cricket", "trill_cricket"),
    "east_asia": ("bell_cricket", "field_cricket"),
    "tropical": ("katydid", "field_cricket", "trill_cricket"),
    "arid": ("field_cricket",),
    "oceania": ("field_cricket", "katydid"),
}
CICADA_REGIONS = ("mediterranean", "east_asia", "tropical", "arid", "oceania", "north_america")


AIR_MIN_M = 8.0          # ISO 9613-1 absorption is applied to sources farther than this


def air_gain(f_hz, distance_m: float, temp_c: float, rh_pct: float):
    """ISO 9613-1 amplitude transmission at f_hz over distance_m (1 inside AIR_MIN_M)."""
    if distance_m <= AIR_MIN_M:
        return np.ones_like(np.asarray(f_hz, dtype=float))
    return 10.0 ** (-iso9613_alpha_db_per_m(f_hz, temp_c, rh_pct) * float(distance_m) / 20.0)


def absorb_clip(y: np.ndarray, distance_m: float, temp_c: float, rh_pct: float, sr: int) -> np.ndarray:
    """A finished call/song as heard over distance_m: ISO 9613-1 magnitude applied once by FFT."""
    if distance_m <= AIR_MIN_M or y.size == 0:
        return y
    N = 1 << int(y.size + 255).bit_length()
    Y = np.fft.rfft(y, N) * air_gain(np.fft.rfftfreq(N, 1.0 / sr), distance_m, temp_c, rh_pct)
    return np.fft.irfft(Y, N)[: y.size]


class _Point:
    """A fixed point source: geometry and spreading (absorption is applied by each voice type)."""

    def __init__(self, vid: str, position, sr: int):
        self.id = vid
        x, y, z = (float(v) for v in position)
        self.position = (x, y, z)
        self.dist = max(0.5, math.sqrt(x * x + y * y + z * z))
        self.az = math.atan2(y, x)
        self.el = math.atan2(z, math.hypot(x, y))
        self.sr = sr

    def emit(self, y: np.ndarray, w, spread: float = 0.05) -> Emission:
        return Emission(signal=y, azimuth=self.az, elevation=self.el, distance=self.dist, spread=spread,
                        stem="ambience", source_id=f"bio.{self.id}")


class _InsectVoice(_Point):
    """One singing insect: a temperature-driven chirp/trill pattern on a carrier, in bouts (rendered by _InsectBank)."""

    def __init__(self, vid, species: str, position, cfg: np.random.Generator, sr: int):
        super().__init__(vid, position, sr)
        self.spec = INSECT_SPECIES[species]
        self.species = species
        self.jit = float(cfg.uniform(0.95, 1.05))
        self.cjit = float(cfg.uniform(0.96, 1.04))
        self.chirp_phase = float(cfg.uniform(0.0, 1.0))
        self.car_phase = float(cfg.uniform(0.0, 2.0 * math.pi))
        self.singing = bool(cfg.random() < 0.7)
        self.trill_len = float(cfg.uniform(2.0, 6.0))
        self.trill_gap = float(cfg.uniform(0.6, 2.5))
        self.level = db_spl(self.spec["level"]) * float(cfg.lognormal(0.0, 0.2))

    def pattern(self, temp_c: float) -> tuple[float, float, int, float]:
        """(chirps per second, duty, syllables per chirp, carrier Hz) at temperature temp_c."""
        T = max(float(temp_c), 12.0)
        spec = self.spec
        if spec["rate"] is None:                    # long trills with gaps
            period = self.trill_len + self.trill_gap
            n_syll = max(1, int(round(spec["syll_rate"](T) * self.trill_len)))
            return 1.0 / period, self.trill_len / period, n_syll, spec["carrier"](T) * self.cjit
        return max(spec["rate"](T), 0.1) * self.jit, spec["duty"], spec["syll"], spec["carrier"](T) * self.cjit


class _InsectBank:
    """Every insect of the scene as one vectorised bank and one SH emission.

    Each insect sings bouts (~2 min) and pauses (~20 s, shorter when conditions are good);
    its chirps follow the species' temperature law (Dolbear for tree crickets) on a tonal
    carrier with a slight downward glide per syllable, or band noise (katydids). Carriers are
    narrow, so ISO 9613-1 absorption is the gain at the carrier frequency.
    """

    def __init__(self, voices: list, sr: int):
        self.voices = voices
        self.sr = sr
        V = len(voices)
        self.V = V
        self.chirp_phase = np.array([v.chirp_phase for v in voices])
        self.car_phase = np.array([v.car_phase for v in voices])
        self.singing = np.array([v.singing for v in voices], dtype=bool)
        self.level = np.array([v.level for v in voices])
        self.dist = np.array([v.dist for v in voices])
        self.noisy = np.array([bool(v.spec["noise"]) for v in voices], dtype=bool)
        self.amp = Ramp(np.zeros(V))
        self.noise_f = None
        if np.any(self.noisy):
            fc = INSECT_SPECIES[voices[int(np.argmax(self.noisy))].species]["carrier"](20.0)
            self.noise_f = StatefulSOS(bandpass_sos(fc, 3.0, sr))
            self.noise_g = noise_rms_gain(self.noise_f.sos)
        self.encoder = StaticSHEncoder([v.az for v in voices], [v.el for v in voices], 0.05, sr=sr)
        self._air_key = None
        self._air = np.ones(V)

    def render(self, ctx: BlockContext, act: float) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        dt = n / sr
        rng = ctx.rng.get("bio.insects", ctx.index)
        u = rng.random(self.V)
        stop = self.singing & (u < dt / 120.0)
        start = ~self.singing & (u < dt / 20.0 * max(act, 0.05))
        self.singing = (self.singing & ~stop) | start
        pats = [v.pattern(float(w.temperature_c)) for v in self.voices]
        rate = np.array([p[0] for p in pats])
        duty = np.array([p[1] for p in pats])
        n_syll = np.array([p[2] for p in pats], dtype=float)
        fc = np.array([p[3] for p in pats])
        key = (round(float(w.temperature_c)), round(float(w.humidity_pct) / 5.0))
        if key != self._air_key:
            self._air_key = key
            self._air = np.array([float(air_gain(f, d, float(w.temperature_c), float(w.humidity_pct)))
                                  for f, d in zip(fc, self.dist)])
        target = self.level * act * self.singing / self.dist * self._air
        if not np.any(target > 0.0) and float(np.max(self.amp.value)) < 1e-9:
            return None
        k = ramp_frac(n) * n
        ph = self.chirp_phase[:, None] + rate[:, None] * (k / sr)[None, :]
        self.chirp_phase = ph[:, -1] % 1.0
        frac = ph % 1.0
        sf = (frac / duty[:, None] * n_syll[:, None]) % 1.0
        env = np.where((frac < duty[:, None]) & (sf < 0.72), np.sin(np.pi * sf / 0.72) ** 2, 0.0)
        cph = self.car_phase[:, None] + 2.0 * np.pi / sr * np.cumsum(fc[:, None] * (1.0 - 0.015 * sf), axis=1)
        self.car_phase = cph[:, -1] % (2.0 * np.pi)
        car = np.sin(cph)
        if self.noise_f is not None:
            car[self.noisy] = self.noise_f.process(rng.standard_normal((int(np.sum(self.noisy)), n))) / self.noise_g
        y = self.amp(target, n) * env * car
        return Emission(signal=self.encoder.process(y), stem="ambience", is_ambisonic=True, spread=1.0,
                        source_id="bio.insects")


class _CicadaVoice(_Point):
    """A cicada in a tree: pulsed noise buzz with slow swells (day, hot)."""

    def __init__(self, vid, position, cfg: np.random.Generator, sr: int):
        super().__init__(vid, position, sr)
        self.fc = float(cfg.uniform(4500.0, 6500.0))
        self.pulse = float(cfg.uniform(110.0, 180.0))
        self.swell_f = 1.0 / float(cfg.uniform(20.0, 40.0))
        self.phase = float(cfg.uniform(0.0, 1.0))
        self.pphase = 0.0
        self.filt = StatefulSOS(np.vstack([bandpass_sos(self.fc, 2.5, sr), bandpass_sos(self.fc * 1.1, 2.0, sr)]))
        self.g = noise_rms_gain(self.filt.sos)
        self.amp = Ramp(0.0)
        self.level = db_spl(76.0)

    def render(self, ctx: BlockContext, act: float) -> Optional[Emission]:
        n, sr = ctx.n, ctx.sr
        target = self.level * act / self.dist
        if target <= 0.0 and float(self.amp.value) < 1e-9:
            return None
        rng = ctx.rng.get(f"bio.cicada.{self.id}", ctx.index)
        sw = self.phase + self.swell_f * np.arange(1, n + 1) / sr
        self.phase = float(sw[-1] % 1.0)
        swell = np.clip(1.4 * np.sin(2.0 * math.pi * sw), 0.0, 1.0) ** 1.5
        pp = self.pphase + self.pulse * np.arange(1, n + 1) / sr
        self.pphase = float(pp[-1] % 1.0)
        pulses = 0.4 + 0.6 * np.sin(math.pi * (pp % 1.0)) ** 2
        w = ctx.weather
        y = self.amp(target, n) * swell * pulses * self.filt.process(rng.standard_normal(n)) / self.g
        y *= float(air_gain(self.fc, self.dist, float(w.temperature_c), float(w.humidity_pct)))
        return self.emit(y, w, spread=0.15)


class _ChorusBed:
    """The incoherent sum of many distant singers: band noise with a weak shared rhythm (diffuse)."""

    def __init__(self, carriers, rate_fn: Callable, level_db: float, sr: int):
        self.filts = [StatefulSOS(bandpass_sos(f, 6.0, sr)) for f in carriers]
        self.g = [noise_rms_gain(f.sos) for f in self.filts]
        self.rate_fn = rate_fn
        self.level = db_spl(level_db)
        self.phase = 0.0
        self.amp = Ramp(0.0)

    def render(self, ctx: BlockContext, act: float) -> Optional[Emission]:
        n, sr = ctx.n, ctx.sr
        target = self.level * act
        if target <= 0.0 and float(self.amp.value) < 1e-9:
            return None
        rng = ctx.rng.get("bio.chorus", ctx.index)
        rate = max(self.rate_fn(max(float(ctx.weather.temperature_c), 12.0)), 0.2)
        ph = self.phase + rate * np.arange(1, n + 1) / sr
        self.phase = float(ph[-1] % 1.0)
        am = 1.0 + 0.3 * np.sin(2.0 * math.pi * ph)
        y = sum(f.process(rng.standard_normal(n)) / g for f, g in zip(self.filts, self.g)) / math.sqrt(len(self.filts))
        y = self.amp(target, n) * am * y
        return Emission(signal=y, azimuth=0.0, elevation=0.0, distance=30.0, spread=1.0, stem="ambience",
                        source_id="bio.chorus")


class _CallerVoice(_Point):
    """A bird, owl or frog: discrete songs/calls rendered whole into a pending buffer."""

    def __init__(self, vid, synth: Callable, level_db: float, per_min: float, position, cfg: np.random.Generator,
                 sr: int, layer: str):
        super().__init__(vid, position, sr)
        self.synth = synth
        self.level = db_spl(level_db) * float(cfg.lognormal(0.0, 0.15))
        self.per_min = per_min * float(cfg.uniform(0.7, 1.3))
        self.pitch = float(cfg.uniform(0.92, 1.08))
        self.pending = PendingMixer(1)
        self.busy_until = -1
        self.layer = layer

    def eligible(self, start: int, act: float) -> bool:
        return act > 1e-3 and start >= self.busy_until

    def begin(self, ctx: BlockContext, act: float, offset: int) -> None:
        """Start a song/call `offset` samples into this block (its content from the voice's own stream)."""
        sr = ctx.sr
        start = block_start_sample(ctx)
        rng = ctx.rng.get(f"{self.layer}.{self.id}", ctx.index)
        song = self.synth(rng, sr, self.pitch)
        amp = self.level * float(rng.lognormal(0.0, 0.15)) * (0.6 + 0.4 * min(act, 1.0)) / self.dist
        w = ctx.weather
        song = absorb_clip(song * amp, self.dist, float(w.temperature_c), float(w.humidity_pct), sr)
        t0 = start + int(offset)
        self.pending.add(t0, song)
        self.busy_until = t0 + song.size + int(rng.uniform(0.5, 3.0) * sr)

    def render(self, ctx: BlockContext, act: float = 0.0) -> Optional[Emission]:
        if not self.pending.active:
            return None
        y = self.pending.take(block_start_sample(ctx), ctx.n)
        return self.emit(y, ctx.weather)


# --------------------------------------------------------------------------
# Biophony generator
# --------------------------------------------------------------------------

class BiophonyGenerator:
    """Crickets, cicadas, birds (incl. owls) and frogs governed by time, season, temperature,
    rain, wind and region (place.biophony: birds / insects / frogs = auto | on | off)."""

    name = "biophony"

    def __init__(self, scene, layout, rngf, sr: int = SAMPLE_RATE):
        self.sr = sr
        bio = dict(scene.place.biophony or {})
        self.region = str(bio.get("region", "temperate_europe") or "temperate_europe").lower()
        if self.region not in REGION_BIRDS:
            self.region = "temperate_europe"
        self.m_birds = _mode(bio.get("birds"))
        self.m_insects = _mode(bio.get("insects"))
        self.m_frogs = _mode(bio.get("frogs"))
        self.lat = float(getattr(scene.weather, "latitude_deg", 50.0))
        h = float(scene.listener.height_m)
        self.has_water = False
        cfg = rngf.get("bio.cfg", 0)
        layout = list(layout or [])

        self.insects: list = []
        self.cicadas: list = []
        self.birds: list = []
        self.owls: list = []
        self.frogs: list = []
        self.chorus: Optional[_ChorusBed] = None

        if self.m_insects != "off":
            species = REGION_INSECTS.get(self.region, ("field_cricket",))
            n_ins = int(cfg.integers(6, 11)) if self.region not in TROPICAL else int(cfg.integers(10, 15))
            for k in range(n_ins):
                sp = species[k % len(species)]
                az = float(cfg.uniform(-math.pi, math.pi))
                d = float(cfg.uniform(2.5, 25.0))
                z = -h + (1.0 if sp in ("tree_cricket", "katydid") else 0.1)
                self.insects.append(_InsectVoice(f"insect{k}", sp, (d * math.cos(az), d * math.sin(az), z), cfg, sr))
            carriers = [INSECT_SPECIES[s]["carrier"](20.0) for s in species[:2]]
            rate_fn = INSECT_SPECIES[species[0]]["rate"] or (lambda T: 0.3)
            self.chorus = _ChorusBed(carriers, rate_fn, 46.0 if self.region in TROPICAL else 36.0, sr)
            if self.region in CICADA_REGIONS:
                for k in range(2):
                    az = float(cfg.uniform(-math.pi, math.pi))
                    d = float(cfg.uniform(10.0, 35.0))
                    self.cicadas.append(_CicadaVoice(f"cicada{k}", (d * math.cos(az), d * math.sin(az), 5.0 - h), cfg, sr))

        perches = [p for p in layout if p.kind == "vegetation" and str((p.params or {}).get("species", "")) != "grass"]
        if self.m_birds != "off":
            palette = REGION_BIRDS[self.region]
            n_b = {"arid": 3, "boreal": 5, "tropical": 9}.get(self.region, 7)
            for k in range(n_b):
                tpl = palette[k % len(palette)]
                synth, lvl, rate = BIRD_SONGS[tpl]
                pos = self._perch(perches, cfg, h)
                self.birds.append(_CallerVoice(f"bird{k}.{tpl}", synth, lvl, rate, pos, cfg, sr, "bio.bird"))
            if self.region in OWL_REGIONS:
                synth, lvl, rate = BIRD_SONGS["owl"]
                az = float(cfg.uniform(-math.pi, math.pi))
                d = float(cfg.uniform(40.0, 120.0))
                self.owls.append(_CallerVoice("owl0", synth, lvl, rate, (d * math.cos(az), d * math.sin(az), 8.0 - h),
                                              cfg, sr, "bio.owl"))

        if self.m_frogs != "off":
            ponds = [p for p in layout if p.kind == "water" or p.material == "puddle"]
            self.has_water = any(p.kind == "water" for p in ponds)
            species = REGION_FROGS.get(self.region, ("common_frog",))
            n_f = int(cfg.integers(6, 12)) if self.region not in TROPICAL else int(cfg.integers(12, 18))
            for k in range(n_f):
                sp = species[k % len(species)]
                synth, lvl, rate = FROG_CALLS[sp]
                if ponds:
                    src = ponds[int(cfg.integers(0, len(ponds)))]
                    jit = cfg.normal(0.0, 2.0, size=2)
                    pos = (src.position[0] + jit[0], src.position[1] + jit[1], -h + 0.05)
                else:
                    az = float(cfg.uniform(-math.pi, math.pi))
                    d = float(cfg.uniform(6.0, 40.0))
                    pos = (d * math.cos(az), d * math.sin(az), -h + 0.05)
                self.frogs.append(_CallerVoice(f"frog{k}.{sp}", synth, lvl, rate, pos, cfg, sr, "bio.frog"))

        self.insect_bank = _InsectBank(self.insects, sr) if self.insects else None

        # weather memory
        self.rain_ema = 0.0
        self.wet_memory = 0.0
        self.dry_s = 3600.0
        self.act: Optional[dict] = None

    @staticmethod
    def _perch(perches: list, cfg: np.random.Generator, h: float) -> tuple:
        if perches and cfg.random() < 0.8:
            p = perches[int(cfg.integers(0, len(perches)))]
            j = cfg.normal(0.0, 1.5, size=3)
            return (p.position[0] + j[0], p.position[1] + j[1], p.position[2] + 0.5 * j[2])
        az = float(cfg.uniform(-math.pi, math.pi))
        d = float(cfg.uniform(8.0, 60.0))
        return (d * math.cos(az), d * math.sin(az), float(cfg.uniform(2.0, 12.0)) - h)

    # -- activity rules -------------------------------------------------------
    def _targets(self, w, dt: float) -> dict:
        R = float(w.rain_rate_mm_h)
        u = float(w.wind_listener_mps)
        T = float(w.temperature_c)
        season = str(w.season).lower()
        hour = float(w.hour) % 24.0
        tropical = self.region in TROPICAL

        if self.act is None and self.rain_ema == 0.0:
            # a scene that starts in rain has been raining for a while (as the World primes its wetness)
            self.rain_ema = R
        a = 1.0 - math.exp(-dt / 600.0)
        self.rain_ema += (R - self.rain_ema) * a
        if R < 0.1:
            self.dry_s += dt
            self.wet_memory *= math.exp(-dt / 3600.0)
        else:
            self.dry_s = 0.0
            self.wet_memory = max(self.wet_memory, self.rain_ema)

        rise, sset = sun_times(self.lat, season)
        night = not bool(w.is_day)

        # insects (crickets): night, warm, not raining, not windy, summer
        rain_off = 1.0 - smoothstep(R, 0.6, 2.0)
        wind_off = 1.0 - smoothstep(u, 5.0, 9.0)
        seas_i = 1.0 if tropical else {"summer": 1.0, "autumn": 0.75, "fall": 0.75, "spring": 0.35}.get(season, 0.0)
        dusk = _gauss(_hour_dist(hour, sset), 0.5, 1.0)
        time_i = 1.0 if night else dusk            # silent by day, starting up toward dusk
        insects = smoothstep(T, 12.0, 16.0) * rain_off * wind_off * seas_i * time_i
        if self.m_insects == "on":
            insects = max(insects, (1.0 - smoothstep(R, 2.0, 6.0)) * wind_off)
        elif self.m_insects == "off":
            insects = 0.0
        cicadas = 0.0
        if self.m_insects != "off":
            seas_c = 1.0 if tropical else {"summer": 1.0, "autumn": 0.3, "fall": 0.3}.get(season, 0.0)
            cicadas = (0.0 if night else 1.0) * smoothstep(T, 22.0, 27.0) * rain_off * wind_off * seas_c

        # birds: dawn and dusk choruses, suppressed by rain and wind, rebound after rain
        dawn = _gauss(_hour_dist(hour, rise + 0.3), 0.0, 0.75)
        evening = 0.45 * _gauss(_hour_dist(hour, sset - 0.3), 0.0, 0.7)
        day = 0.0
        if rise + 0.5 < hour < sset - 0.5:
            day = 0.3 * (1.0 - 0.35 * _gauss(hour, 13.5, 1.5))
        seas_b = 0.9 if tropical else {"spring": 1.0, "summer": 0.7, "autumn": 0.35, "fall": 0.35}.get(season, 0.2)
        rain_b = 1.0 / (1.0 + (R / 0.8) ** 1.5)
        wind_b = 1.0 / (1.0 + (u / 8.0) ** 3)
        rebound = 1.0 + 1.5 * min(self.wet_memory / 2.0, 1.0) * math.exp(-self.dry_s / 1200.0) * smoothstep(self.dry_s, 30.0, 240.0)
        birds = max(dawn, evening, day) * seas_b * rain_b * wind_b * rebound
        if self.m_birds == "on":
            birds = max(birds, 0.4 * rain_b * wind_b)
        elif self.m_birds == "off":
            birds = 0.0
        owls = 0.0
        if self.m_birds != "off" and night:
            owls = (1.0 / (1.0 + (R / 0.5) ** 2)) * wind_b * (0.6 + 0.4 * (season in ("autumn", "fall", "winter")))

        # frogs: warm, wet evenings and nights
        wet = max(smoothstep(self.rain_ema, 0.1, 1.5),
                  smoothstep(self.wet_memory, 0.2, 2.0) * math.exp(-self.dry_s / 10800.0),
                  0.5 if self.has_water else 0.0)
        seas_f = 1.0 if tropical else {"spring": 1.0, "summer": 0.85, "autumn": 0.25, "fall": 0.25}.get(season, 0.0)
        time_f = 1.0 if night else max(0.12, _gauss(_hour_dist(hour, sset), 0.3, 0.8))
        frogs = smoothstep(T, 8.0, 14.0) * wet * seas_f * time_f * (1.0 - 0.4 * smoothstep(R, 15.0, 40.0)) * wind_off
        if self.m_frogs == "on":
            frogs = max(frogs, 0.7 * (1.0 - 0.4 * smoothstep(R, 15.0, 40.0)))
        elif self.m_frogs == "off":
            frogs = 0.0
        return {"insects": insects, "cicadas": cicadas, "birds": birds, "owls": owls, "frogs": frogs}

    def render(self, ctx: BlockContext) -> list[Emission]:
        dt = ctx.n / float(ctx.sr)
        tgt = self._targets(ctx.weather, dt)
        if self.act is None:
            self.act = dict(tgt)
        else:
            taus = {"insects": 90.0, "cicadas": 60.0, "birds": 60.0, "owls": 60.0, "frogs": 120.0}
            for k, v in tgt.items():
                self.act[k] += (v - self.act[k]) * (1.0 - math.exp(-dt / taus[k]))
        out = []
        if self.insect_bank is not None:
            em = self.insect_bank.render(ctx, self.act["insects"])
            if em is not None:
                out.append(em)
        for v in self.cicadas:
            em = v.render(ctx, self.act["cicadas"])
            if em is not None:
                out.append(em)
        # who starts a song/call in this block: one shared Poisson draw per eligible caller
        n, sr = ctx.n, ctx.sr
        start = block_start_sample(ctx)
        callers = [(v, self.act[key]) for voices, key in ((self.birds, "birds"), (self.owls, "owls"),
                                                           (self.frogs, "frogs"))
                   for v in voices if v.eligible(start, self.act[key])]
        if callers:
            crng = ctx.rng.get("bio.calls", ctx.index)
            lam = np.array([v.per_min / 60.0 * a * n / sr for v, a in callers])
            counts = crng.poisson(lam)
            offs = crng.integers(0, n, size=int(np.sum(counts))) if np.any(counts) else np.zeros(0, dtype=int)
            pos = 0
            for (v, a), c in zip(callers, counts):
                if c > 0:
                    v.begin(ctx, a, int(np.min(offs[pos:pos + c])))
                    pos += int(c)
        for voices in (self.birds, self.owls, self.frogs):
            for v in voices:
                em = v.render(ctx)
                if em is not None:
                    out.append(em)
        if self.chorus is not None:
            em = self.chorus.render(ctx, self.act["insects"])
            if em is not None:
                out.append(em)
        return out


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------

DAILY_FLOW = np.array([0.08, 0.05, 0.04, 0.04, 0.06, 0.15, 0.45, 0.85, 1.0, 0.8, 0.7, 0.72,
                       0.75, 0.72, 0.72, 0.8, 0.95, 1.0, 0.85, 0.6, 0.45, 0.35, 0.25, 0.15])


def daily_flow_factor(hour: float) -> float:
    h = float(hour) % 24.0
    i = int(math.floor(h))
    f = h - i
    return float(DAILY_FLOW[i] * (1.0 - f) + DAILY_FLOW[(i + 1) % 24] * f)


class _PassBy:
    """One vehicle passing along the road: tyre noise, wet spray hiss, Doppler-shifted engine."""

    KINDS = {   # tyre dB, engine dB at 7.5 m (50 km/h), firing order factor, rpm range
        "car": (70.0, 62.0, 2.0, (1700.0, 2800.0)),
        "truck": (76.0, 76.0, 3.0, (1100.0, 1700.0)),
        "moto": (68.0, 74.0, 1.0, (3500.0, 6000.0)),
    }

    def __init__(self, vid: str, rng: np.random.Generator, entry_sample: int, road_d: float, axis: float, side: float,
                 v_kmh: float, listener_h: float, sr: int):
        self.id = vid
        self.sr = sr
        self.kind = str(rng.choice(["car", "truck", "moto"], p=[0.86, 0.1, 0.04]))
        tyre_db, eng_db, order, rpm = self.KINDS[self.kind]
        self.v = max(v_kmh * float(rng.lognormal(0.0, 0.12)), 10.0) / 3.6
        self.d_min = max(road_d + float(rng.uniform(-1.5, 1.5)), 1.5)
        self.dir = float(rng.choice([-1.0, 1.0]))
        half = max(60.0, 10.0 * self.d_min)
        self.t_half = half / self.v
        self.t_c = entry_sample / sr + self.t_half
        self.end_sample = int((self.t_c + self.t_half) * sr)
        az_c = axis + side * 0.5 * math.pi
        self.pc = np.array([self.d_min * math.cos(az_c), self.d_min * math.sin(az_c)])
        self.e = np.array([math.cos(axis), math.sin(axis)])
        self.z = 0.5 - listener_h
        vk = self.v * 3.6
        self.tyre = db_spl(tyre_db + 30.0 * math.log10(vk / 50.0)) * 7.5
        self.engine = db_spl(eng_db + 10.0 * math.log10(vk / 50.0)) * 7.5
        self.f_e = float(rng.uniform(*rpm)) / 60.0 * order
        self.e_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        self.tyre_f = StatefulSOS(np.vstack([peaking_sos(1000.0, 8.0, 0.7, sr), slope_sos(-4.0, 100.0, 12000.0, sr),
                                             highpass_sos(60.0, sr)]))
        self.tyre_g = noise_rms_gain(self.tyre_f.sos)
        self.hiss_f = StatefulSOS(np.vstack([highpass_sos(1800.0, sr), lowpass_sos(9000.0, sr)]))
        self.hiss_g = noise_rms_gain(self.hiss_f.sos)

    def render(self, ctx: BlockContext, start: int, n: int, wet: float, c: float) -> Optional[Emission]:
        sr = self.sr
        if start + n <= int((self.t_c - self.t_half) * sr):
            return None
        rng = ctx.rng.get(f"traffic.car.{self.id}", ctx.index)
        t = (start + np.arange(n)) / sr - self.t_c
        x = self.dir * self.v * t
        r = np.sqrt(self.d_min ** 2 + x ** 2 + self.z ** 2)
        live = (np.abs(t) <= self.t_half).astype(float)
        edge = np.clip((self.t_half - np.abs(t)) / 1.0, 0.0, 1.0)
        g = live * edge / r
        r_dot = x * self.dir * self.v / r                      # >0 receding
        doppler = c / (c + r_dot)
        ph = self.e_phase + 2.0 * math.pi * np.cumsum(self.f_e * doppler) / sr
        self.e_phase = float(ph[-1] % (2.0 * math.pi))
        eng = (np.sin(ph) + 0.6 * np.sin(2.0 * ph + 0.3) + 0.35 * np.sin(3.0 * ph + 1.1) + 0.2 * np.sin(4.0 * ph + 2.0))
        tyre = self.tyre_f.process(rng.standard_normal(n)) / self.tyre_g
        hiss_db = -10.0 + 14.0 * wet
        hiss = self.hiss_f.process(rng.standard_normal(n)) / self.hiss_g * 10.0 ** (hiss_db / 20.0) * smoothstep(wet, 0.03, 0.3)
        y = g * (self.tyre * (tyre + hiss) + self.engine * eng / 1.3)
        xm = float(x[n // 2])
        p = self.pc + xm * self.e
        dist = float(r[n // 2])
        return Emission(signal=y, azimuth=math.atan2(p[1], p[0]), elevation=math.atan2(self.z, math.hypot(p[0], p[1])),
                        distance=dist, spread=0.05, stem="ambience", source_id=f"traffic.{self.id}")


class TrafficGenerator:
    """Road traffic: a distant bed with a daily flow curve and wet-tyre hiss, plus individual pass-bys
    on nearby roads (streets, suburbs, a parked car by the kerb)."""

    name = "traffic"

    def __init__(self, scene, layout, rngf, sr: int = SAMPLE_RATE):
        self.sr = sr
        bio = dict(scene.place.biophony or {})
        try:
            self.level = min(max(float(bio.get("traffic", 0.0) or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            self.level = 0.0
        preset = str(scene.place.preset or "").lower()
        shelter = shelter_of(scene)
        cfg = rngf.get("traffic.cfg", 0)
        self.listener_h = float(scene.listener.height_m)
        street = any(k in preset for k in ("city", "street", "downtown", "urban"))
        if street:
            self.road_d, self.v_kmh, self.q_max, self.pass_bys = float(cfg.uniform(3.5, 6.0)), 40.0, 900.0, True
            self.axis = 0.0
        elif "alley" in preset:
            self.road_d, self.v_kmh, self.q_max, self.pass_bys = float(cfg.uniform(15.0, 25.0)), 35.0, 600.0, True
            self.axis = 0.5 * math.pi
        elif shelter == "car":
            self.road_d, self.v_kmh, self.q_max, self.pass_bys = 4.5, 50.0, 400.0, True
            self.axis = 0.0
        elif any(k in preset for k in ("suburb", "village", "town", "residential", "garden", "backyard", "park")):
            self.road_d, self.v_kmh, self.q_max, self.pass_bys = float(cfg.uniform(20.0, 60.0)), 50.0, 300.0, True
            self.axis = float(cfg.uniform(-math.pi, math.pi))
        else:
            self.road_d, self.v_kmh, self.q_max, self.pass_bys = float(cfg.uniform(250.0, 900.0)), 100.0, 1800.0, False
            self.axis = float(cfg.uniform(-math.pi, math.pi))
        self.side = float(cfg.choice([-1.0, 1.0])) if not street else -1.0
        az_c = self.axis + self.side * 0.5 * math.pi
        self.az_c = az_c
        self.ground = str(getattr(scene.place, "ground", "grass") or "grass")
        shape = np.vstack([slope_sos(-3.0, 40.0, 16000.0, sr), peaking_sos(900.0, 6.0, 0.8, sr),
                           peaking_sos(90.0, 4.0, 1.0, sr), highpass_sos(25.0, sr)])
        ge = ground_effect_sos(self.road_d, 0.5, self.listener_h, self.ground, sr)
        self.bed_f = StatefulSOS(np.vstack([shape, ge]))
        self.bed_g = noise_rms_gain(shape)
        self.hiss_f = StatefulSOS(np.vstack([highpass_sos(1500.0, sr), lowpass_sos(8000.0, sr), ge]))
        self.hiss_g = noise_rms_gain(np.vstack([highpass_sos(1500.0, sr), lowpass_sos(8000.0, sr)]))
        self.air = AirPath(self.road_d, sr, min_m=8.0)
        self.bed_amp = Ramp(0.0)
        self.hiss_amp = Ramp(0.0)
        self.ou = 0.0
        self.road_film = -1.0
        self.cars: list[_PassBy] = []
        self.n_spawned = 0

    def _wetness(self, w, dt: float) -> float:
        liquid = float(w.rain_rate_mm_h) if w.precip_type not in ("snow", "none", "graupel") else 0.0
        tau = 900.0 * (1.0 + max(float(w.humidity_pct) - 60.0, 0.0) / 40.0)
        if self.road_film < 0.0:
            self.road_film = liquid / 3600.0 * tau
        self.road_film += (liquid / 3600.0 - self.road_film / tau) * dt
        wet = smoothstep(self.road_film, 0.002, 0.05)
        wd = w.wetness or {}
        return float(min(max(wet, float(wd.get("asphalt", 0.0)), 0.8 * float(wd.get("concrete", 0.0))), 1.0))

    def render(self, ctx: BlockContext) -> list[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        dt = n / sr
        wet = self._wetness(w, dt)
        q = self.q_max * self.level * daily_flow_factor(w.hour)       # vehicles per hour
        if q < 0.5 and not self.cars and float(self.bed_amp.value) < 1e-9:
            return []
        rng = ctx.rng.get("traffic.bed", ctx.index)
        start = block_start_sample(ctx)
        out = []

        # distant network bed (line source); nearby vehicles are rendered individually
        L_bed = (10.0 * math.log10(max(q, 1.0)) + 33.0 + 30.0 * math.log10(self.v_kmh / 50.0)
                 - 10.0 * math.log10(max(self.road_d, 1.0) / 10.0) - (8.0 if self.pass_bys else 0.0))
        a = math.exp(-dt / 15.0)
        self.ou = a * self.ou + math.sqrt(1.0 - a * a) * float(rng.standard_normal())
        depth = min(0.8, 2.0 / math.sqrt(q / 60.0 + 1.0))
        bed_t = db_spl(L_bed) * smoothstep(q, 0.5, 5.0) * max(1.0 + depth * 0.5 * self.ou, 0.1)
        hiss_t = db_spl(L_bed - 10.0 + 14.0 * wet) * smoothstep(wet, 0.03, 0.4) * smoothstep(q, 0.5, 5.0)
        y = self.bed_amp(bed_t, n) * self.bed_f.process(rng.standard_normal(n)) / self.bed_g
        y = y + self.hiss_amp(hiss_t, n) * self.hiss_f.process(rng.standard_normal(n)) / self.hiss_g
        y = self.air.process(y, w.temperature_c, w.humidity_pct)
        out.append(Emission(signal=y, azimuth=self.az_c, elevation=math.atan2(0.5 - self.listener_h, self.road_d),
                            distance=self.road_d, spread=0.75 if self.road_d < 100.0 else 0.5, stem="ambience",
                            source_id="traffic.bed"))

        # individual pass-bys
        if self.pass_bys:
            crng = ctx.rng.get("traffic.spawn", ctx.index)
            for off in poisson_offsets(crng, q / 3600.0, n, sr):
                vid = f"{ctx.index}.{int(off)}"
                self.cars.append(_PassBy(vid, crng, start + int(off), self.road_d, self.axis, self.side, self.v_kmh,
                                         self.listener_h, sr))
                self.n_spawned += 1
            c = speed_of_sound(float(w.temperature_c))
            alive = []
            for car in self.cars:
                em = car.render(ctx, start, n, wet, c)
                if em is not None:
                    out.append(em)
                if start + n < car.end_sample:
                    alive.append(car)
            self.cars = alive
        return out


__all__ = ["BiophonyGenerator", "TrafficGenerator", "dolbear_chirps_per_min", "sun_times", "daily_flow_factor",
           "BIRD_SONGS", "FROG_CALLS", "INSECT_SPECIES", "REGION_BIRDS", "REGION_FROGS", "REGION_INSECTS"]
