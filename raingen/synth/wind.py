"""Wind bed, microphone wind rumble, and the shared environment DSP toolkit.

Physics implemented here:

* Wind is heard through objects (MASTER_PLAN 3.6). The free-field "air movement"
  bed models turbulence scattering around the head and near obstacles: compact
  aerodynamic dipoles radiate intensity ~rho U^6 / c^3 (Curle 1955), so the bed's
  pressure follows U^3 (60 dB per decade of wind) with a brightness (spectral corner)
  rising with U; the felt low-frequency pseudo-sound follows dynamic pressure (U^2).
  Levels are calibrated on the *mean* wind (the gust-weighted average of U^6 is
  divided out), so a gustier day swings more around the same Leq.
* Gusts follow a normalized Kaimal process (raingen.physics.wind.GustProcess) and
  are *advected* (Taylor's frozen-turbulence hypothesis): a point s metres upwind
  of the listener experiences a gust s/U seconds before the listener does. The
  shared GustField re-runs the World's own gust stream (same RNG layer 'wind.gust',
  same constructor arguments, same per-block parameters and block length), so its
  value at the listener equals WeatherState.gust exactly and the gust heard in upwind
  trees is the one that reaches the listener moments later. Higher up, the same gust
  has a smaller relative excursion (I_u = 1/ln(z/z0)): `remap_gust` rescales the
  log-normal multiplier to another turbulence intensity with the same timing.
* Beaufort ladder (WMO land descriptions): loose debris is set in motion from
  force 8 (17.2 m/s).
* Microphone wind noise (optional "recorded-realism" artifact): turbulent
  pressure at the capsule with spectrum about -6.7 dB/decade below
  fc = V/(3D) and -26.7 dB/decade above (Strasberg 1988; Wuttke 1992 windscreen
  data), level ~ dynamic pressure (U^2).

Units: every generator in raingen.synth.{wind,vegetation,aeolian,thunder,
biophony,places} produces sound pressure in pascals at the listener
(1.0 = 94 dB SPL).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy import fft as sfft
from scipy import signal

try:                                    # the C kernel behind scipy.signal.sosfilt, without its
    from scipy.signal._sosfilt import _sosfilt as _sosfilt_inplace    # ~150 us per-call wrapper
except ImportError:                     # pragma: no cover - other scipy layouts
    _sosfilt_inplace = None

from raingen.core.constants import AMBI_ORDER, BLOCK, N_AMBI, SAMPLE_RATE
from raingen.core.types import BlockContext, Emission, EventMarker
from raingen.physics.layout import normalize_shelter
from raingen.physics.wind import (GustProcess, kaimal_integral_time_s, mps_to_beaufort, terrain_z0,
                                  turbulence_intensity_for, wind_at_height)
from raingen.spatial.sh import sh_real

P_REF = 20e-6
ACN_ORDER = np.floor(np.sqrt(np.arange(N_AMBI))).astype(int)


# --------------------------------------------------------------------------
# Shared helpers (used by vegetation, aeolian, thunder, biophony, places)
# --------------------------------------------------------------------------

def db_spl(db: float) -> float:
    """dB SPL -> RMS pressure in pascals."""
    return P_REF * 10.0 ** (db / 20.0)


def smoothstep(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def clip_f(f: float, sr: int) -> float:
    return float(min(max(f, 5.0), 0.45 * sr))


def bandpass_sos(fc: float, q: float, sr: int) -> np.ndarray:
    """RBJ band-pass, 0 dB peak gain."""
    w0 = 2.0 * math.pi * clip_f(fc, sr) / sr
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    a0 = 1.0 + alpha
    return np.array([[alpha / a0, 0.0, -alpha / a0, 1.0, -2.0 * math.cos(w0) / a0, (1.0 - alpha) / a0]])


def peaking_sos(fc: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """RBJ peaking equaliser."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * clip_f(fc, sr) / sr
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    c = math.cos(w0)
    a0 = 1.0 + alpha / A
    return np.array([[(1.0 + alpha * A) / a0, -2.0 * c / a0, (1.0 - alpha * A) / a0,
                      1.0, -2.0 * c / a0, (1.0 - alpha / A) / a0]])


def _butter_section(fc: float, sr: int, high: bool, order: int) -> np.ndarray:
    """Closed-form bilinear Butterworth section of order 1 or 2 (what scipy.signal.butter returns, ~100x faster)."""
    k = math.tan(math.pi * fc / sr)
    if order == 1:
        a1 = (k - 1.0) / (k + 1.0)
        b0 = 1.0 / (k + 1.0) if high else k / (k + 1.0)
        return np.array([[b0, -b0 if high else b0, 0.0, 1.0, a1, 0.0]])
    q = math.sqrt(2.0)
    norm = 1.0 / (1.0 + q * k + k * k)
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - q * k + k * k) * norm
    if high:
        return np.array([[norm, -2.0 * norm, norm, 1.0, a1, a2]])
    b0 = k * k * norm
    return np.array([[b0, 2.0 * b0, b0, 1.0, a1, a2]])


def lowpass_sos(fc: float, sr: int, order: int = 2) -> np.ndarray:
    if order in (1, 2):
        return _butter_section(clip_f(fc, sr), sr, False, order)
    return signal.butter(order, clip_f(fc, sr), "low", fs=sr, output="sos")


def highpass_sos(fc: float, sr: int, order: int = 2) -> np.ndarray:
    if order in (1, 2):
        return _butter_section(clip_f(fc, sr), sr, True, order)
    return signal.butter(order, clip_f(fc, sr), "high", fs=sr, output="sos")


def slope_sos(slope_db_per_decade: float, f_lo: float, f_hi: float, sr: int,
              per_decade: float = 2.0) -> np.ndarray:
    """Constant spectral slope between f_lo and f_hi (flat outside), 0 dB at DC.

    Interleaved real pole/zero ladder; each pair contributes 20*log10(z/p) dB, so
    k pairs over n decades give the requested slope (|slope| < 20 dB/decade).
    """
    f_lo = max(1.0, f_lo)
    f_hi = min(f_hi, 0.45 * sr)
    s = float(slope_db_per_decade)
    if f_hi <= f_lo * 1.01 or abs(s) < 1e-6:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    n_dec = math.log10(f_hi / f_lo)
    k = max(1, int(math.ceil(n_dec * per_decade)))
    step = (f_hi / f_lo) ** (1.0 / k)
    r = min(10.0 ** (abs(s) * n_dec / (20.0 * k)), step * 0.999)
    centers = f_lo * step ** (np.arange(k) + 0.5)
    lo, hi = centers / math.sqrt(r), centers * math.sqrt(r)
    poles, zeros = (lo, hi) if s < 0 else (hi, lo)

    def warp(f):
        return sr / math.pi * np.tan(math.pi * np.asarray(f) / sr)

    z_a = -2.0 * math.pi * warp(zeros)
    p_a = -2.0 * math.pi * warp(poles)
    k_a = float(np.prod(p_a / z_a))
    zd, pd, kd = signal.bilinear_zpk(z_a, p_a, k_a, sr)
    return signal.zpk2sos(zd, pd, kd)


def graphic_eq_sos(freqs, gains_db, sr: int, q: float = 1.41, iterations: int = 4) -> np.ndarray:
    """Cascade of peaking filters matching gains_db at freqs (iteratively corrected)."""
    freqs = np.asarray(freqs, float)
    target = np.asarray(gains_db, float)
    ok = freqs < 0.45 * sr
    freqs, target = freqs[ok], target[ok]
    offset = float(np.mean(target))
    g = target - offset
    sos = None
    for _ in range(max(1, iterations)):
        sos = np.vstack([peaking_sos(f, gi, q, sr) for f, gi in zip(freqs, g)])
        sos[0, :3] *= 10.0 ** (offset / 20.0)
        _, h = signal.sosfreqz(sos, worN=freqs, fs=sr)
        err = target - 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
        g = g + 0.85 * err
    return sos


_NRG_W = np.unique(np.concatenate([np.geomspace(1e-5, math.pi, 1536), np.linspace(0.0, math.pi, 512)]))
_NRG_Z1 = np.exp(-1j * _NRG_W)
_NRG_Z2 = _NRG_Z1 * _NRG_Z1
_NRG_DW = np.diff(_NRG_W)


def noise_rms_gain(sos: np.ndarray) -> float:
    """RMS gain of a filter for white noise input: sqrt((1/pi) int_0^pi |H(w)|^2 dw).

    Integrated on a log+linear frequency grid so very narrow / very low filters
    (a few Hz at 48 kHz) are measured accurately. The response is evaluated directly from the
    sections (as scipy.signal.sosfreqz does, without its per-section overhead).
    """
    sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
    num = sos[:, 0:1] + sos[:, 1:2] * _NRG_Z1 + sos[:, 2:3] * _NRG_Z2
    den = sos[:, 3:4] + sos[:, 4:5] * _NRG_Z1 + sos[:, 5:6] * _NRG_Z2
    p = np.prod((num.real ** 2 + num.imag ** 2) / np.maximum(den.real ** 2 + den.imag ** 2, 1e-300), axis=0)
    var = float(np.dot(0.5 * (p[1:] + p[:-1]), _NRG_DW)) / math.pi
    return float(math.sqrt(max(var, 1e-24)))


_DESIGNS: dict = {}


def cached_design(kind: str, fc: float, sr: int, build, grid: float = 1.02) -> tuple[np.ndarray, float, float]:
    """(sos, white-noise RMS gain, grid frequency) of `build(f)` with fc snapped to a log grid.

    Per-block retuned filters (gust-driven corners) revisit the same frequencies, so designs and
    their noise gains are shared across blocks and generator instances. `kind` must identify
    everything `build` depends on besides the frequency (and the sample rate).
    """
    k = int(round(math.log(max(float(fc), 1.0)) / math.log(grid)))
    key = (kind, int(sr), float(grid), k)
    d = _DESIGNS.get(key)
    if d is None:
        f = grid ** k
        sos = np.ascontiguousarray(np.atleast_2d(build(f)), dtype=np.float64)
        d = (sos, noise_rms_gain(sos), f)
        if len(_DESIGNS) < 8192:
            _DESIGNS[key] = d
    return d


def bandpass_rms_gain(fc: float, q: float, sr: int, cascade: int = 1) -> float:
    """White-noise RMS gain of `cascade` identical 0 dB-peak band-passes (narrow-band closed form).

    |H|^2 ~ 1 / (1 + (2 Q df / f0)^2) integrates to a noise bandwidth of pi f0 / (2Q) (pi f0 / (4Q)
    for two in cascade); white noise of unit variance has a one-sided PSD of 2/sr. Used for
    per-block retuning of resonators with Q >~ 2, where it is within a few per cent.
    """
    f0 = clip_f(fc, sr)
    bw = math.pi * f0 / (2.0 * max(q, 0.05)) / (1.0 if cascade <= 1 else 2.0)
    return float(math.sqrt(max(2.0 * bw / sr, 1e-24)))


def poisson_offsets(rng: np.random.Generator, rate_per_s: float, n: int, sr: int) -> np.ndarray:
    lam = max(0.0, rate_per_s) * n / sr
    if lam <= 0.0:
        return np.zeros(0, dtype=np.int64)
    k = int(rng.poisson(lam))
    return np.sort(rng.integers(0, n, size=k))


def sh_gains(az: float, el: float, spread: float = 0.0, order: int = AMBI_ORDER) -> np.ndarray:
    """ACN/SN3D encoding gains padded to N_AMBI; spread tapers higher orders toward W."""
    y = np.zeros(N_AMBI)
    g = np.asarray(sh_real(order, az, el), dtype=float)
    y[: g.size] = g
    if spread > 0.0:
        taper = (1.0 - min(max(spread, 0.0), 1.0)) ** ACN_ORDER
        y *= taper
    return y


def upwind_distance(position, wind_azimuth: float) -> float:
    """Projection of a listener-frame position on the direction the wind comes from."""
    x, y = float(position[0]), float(position[1])
    return x * math.cos(wind_azimuth) + y * math.sin(wind_azimuth)


def sosfilt_inplace(sos: np.ndarray, x: np.ndarray, zi: np.ndarray) -> None:
    """Filter the rows of C-contiguous float64 `x` (K, n) in place; `zi` (K, sections, 2) is updated.

    Bit-identical to scipy.signal.sosfilt (same C kernel) but without its per-call array-API
    validation, which costs several times the filtering itself on 1024-sample blocks.
    """
    if _sosfilt_inplace is not None:
        _sosfilt_inplace(sos, x, zi)
        return
    y, zf = signal.sosfilt(sos, x, axis=-1, zi=np.moveaxis(zi, 1, 0))      # pragma: no cover
    x[...] = y
    zi[...] = np.moveaxis(zf, 0, 1)


class StatefulSOS:
    """sosfilt with state carried across blocks, for (n,) or (C, n) signals.

    Coefficients may be replaced between blocks (gradual parameter changes); the
    state is kept when the section count is unchanged. The state is held as
    (channels, sections, 2) and the filter runs in the sosfilt C kernel directly.
    """

    def __init__(self, sos: np.ndarray):
        self.sos = np.ascontiguousarray(np.atleast_2d(np.asarray(sos, dtype=np.float64)))
        self.zi: Optional[np.ndarray] = None

    def set(self, sos: np.ndarray) -> None:
        sos = np.ascontiguousarray(np.atleast_2d(np.asarray(sos, dtype=np.float64)))
        if sos.shape != self.sos.shape:
            self.zi = None
        self.sos = sos

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        n = shape[-1]
        k = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
        y = np.array(x.reshape(k, n), dtype=np.float64, order="C", copy=True)
        if self.zi is None or self.zi.shape != (k, self.sos.shape[0], 2):
            self.zi = np.zeros((k, self.sos.shape[0], 2))
        if n:
            sosfilt_inplace(self.sos, y, self.zi)
        return y.reshape(shape)

    def reset(self) -> None:
        self.zi = None


_FRAC_CACHE: dict[int, np.ndarray] = {}


def ramp_frac(n: int) -> np.ndarray:
    """(1..n)/n, cached per block length (read-only)."""
    r = _FRAC_CACHE.get(n)
    if r is None:
        r = np.arange(1, n + 1, dtype=float) / max(n, 1)
        r.setflags(write=False)
        _FRAC_CACHE[n] = r
    return r


class AirPath:
    """ISO 9613-1 absorption over a fixed path for a static source (raingen.physics.atmosphere design).

    Runs as a StatefulSOS (the sosfilt C kernel, no per-call wrapper) and is redesigned only when
    the temperature moves by 1 C or the humidity by 5 %; paths up to `min_m` pass unchanged.
    """

    def __init__(self, distance_m: float, sr: int = SAMPLE_RATE, min_m: float = 3.0):
        self.d = float(distance_m)
        self.sr = int(sr)
        self.min_m = float(min_m)
        self.key = None
        self.filt: Optional[StatefulSOS] = None

    def process(self, x: np.ndarray, temp_c: float, rh_pct: float) -> np.ndarray:
        if self.d <= self.min_m:
            return x
        key = (round(float(temp_c)), round(float(rh_pct) / 5.0))
        if key != self.key:
            from raingen.physics.atmosphere import air_absorption_sos

            sos = air_absorption_sos(self.d, float(temp_c), float(rh_pct), self.sr)
            if self.filt is None:
                self.filt = StatefulSOS(sos)
            else:
                self.filt.set(sos)
            self.key = key
        return self.filt.process(x)


class Ramp:
    """Linear per-block parameter ramp (scalar or vector), avoids zipper noise."""

    def __init__(self, value=0.0):
        self.value = np.asarray(value, dtype=float)

    def __call__(self, target, n: int) -> np.ndarray:
        v1 = np.asarray(target, dtype=float)
        v0 = np.broadcast_to(self.value, v1.shape) if self.value.shape != v1.shape else self.value
        out = v0[..., None] + (v1 - v0)[..., None] * ramp_frac(n)
        self.value = v1.copy()
        return out


class PendingMixer:
    """Event audio that spans blocks, addressed in absolute samples."""

    def __init__(self, channels: int = 1):
        self.channels = int(channels)
        self.events: list[tuple[int, np.ndarray]] = []

    def add(self, start: int, sig: np.ndarray) -> None:
        sig = np.asarray(sig, dtype=float)
        if sig.ndim == 1:
            sig = sig[None, :]
        if sig.shape[0] != self.channels:
            full = np.zeros((self.channels, sig.shape[1]))
            full[: min(self.channels, sig.shape[0])] = sig[: self.channels]
            sig = full
        if sig.shape[1] > 0:
            self.events.append((int(start), sig))

    def take(self, start: int, n: int) -> np.ndarray:
        out = np.zeros((self.channels, n))
        end = start + n
        keep = []
        for s0, sig in self.events:
            e0 = s0 + sig.shape[1]
            if e0 <= start:
                continue
            if s0 < end:
                a, b = max(s0, start), min(e0, end)
                out[:, a - start:b - start] += sig[:, a - s0:b - s0]
            if e0 > end:
                keep.append((s0, sig))
        self.events = keep
        return out[0] if self.channels == 1 else out

    @property
    def active(self) -> bool:
        return bool(self.events)


def block_start_sample(ctx: BlockContext) -> int:
    return int(round(ctx.t0 * ctx.sr))


def scene_z0(scene) -> float:
    z0 = getattr(scene.place, "z0_m", None)
    return float(z0) if z0 is not None else float(terrain_z0(scene.place.preset))


def shelter_of(scene) -> str:
    """Canonical shelter name (open|tree|umbrella|tent|porch|car|room_open|room_closed|attic|cave|underpass)."""
    return normalize_shelter(getattr(scene.place, "shelter", "open") or "open")


def gust_integral_time_s(wind, u10: float, z0: float, listener_h: float) -> float:
    """Gust correlation time at the ears: user override, gust rate, or Kaimal at ear height."""
    if getattr(wind, "integral_time_s", None):
        return float(wind.integral_time_s)
    rate = getattr(wind, "gust_rate_per_min", None)
    if rate:
        return 60.0 / (2.0 * math.pi * max(float(rate), 0.05))
    z = max(float(listener_h), 0.3)
    return float(kaimal_integral_time_s(wind_at_height(u10, z, z0), max(z, 1.0)))


def ear_turbulence_intensity(wind, z0: float, listener_h: float, stability_factor: float = 1.0) -> float:
    """Turbulence intensity at the ears; a user TI / gust factor refers to 10 m (neutral profile ratio).

    With automatic turbulence the World also scales the intensity by its stability factor
    (sqrt 2 unstable, 1 neutral, sqrt 0.5 stable; Bolin); pass it to reproduce the World exactly.
    """
    ti10_auto = float(turbulence_intensity_for(z0, 10.0))
    auto = False
    if getattr(wind, "turbulence_intensity", None) is not None:
        ti10 = float(wind.turbulence_intensity)
    elif getattr(wind, "gust_factor", None) is not None:
        ti10 = max(float(wind.gust_factor) - 1.0, 0.0) / 3.0
    else:
        ti10 = ti10_auto
        auto = True
    ti = float(turbulence_intensity_for(z0, max(float(listener_h), 0.3))) * ti10 / max(ti10_auto, 1e-6)
    if auto:
        ti *= float(stability_factor)
    return float(min(max(ti, 0.0), 0.8))


def initial_stability_factor(scene) -> float:
    """The World's starting gust-variance stability factor for `scene` (raingen.physics.world)."""
    from raingen.physics import world as _world

    w0 = scene.weather
    ptype = str(w0.precipitation.type).lower()
    rate0 = 0.0 if ptype == "none" else max(float(w0.precipitation.rate_mm_h), 0.0)
    day0 = _world.is_daytime(float(w0.hour), float(getattr(w0, "latitude_deg", 50.0)), str(w0.season))
    stab0 = _world.stability_class(day0, float(w0.cloud_okta), max(float(w0.wind.speed_mps), 0.0), rate0)
    table = getattr(_world, "_STABILITY_TI", {"unstable": math.sqrt(2.0), "neutral": 1.0, "stable": math.sqrt(0.5)})
    return float(table.get(stab0, 1.0))


def lognormal_sigma(ti: float) -> float:
    """Log-std s of a mean-one log-normal multiplier with relative std `ti` (s^2 = ln(1 + ti^2))."""
    return math.sqrt(math.log1p(max(float(ti), 0.0) ** 2))


def remap_gust(g, ti_from: float, ti_to: float):
    """Rescale a mean-one log-normal gust multiplier from turbulence intensity ti_from to ti_to.

    g = exp(s Z - s^2/2) is mapped to exp(s' Z - s'^2/2) with the same Z, so the gust keeps its
    timing and only its relative excursion changes (e.g. ears -> crown height, where
    I_u = 1/ln(z/z0) is smaller). Works on scalars and arrays.
    """
    s0 = lognormal_sigma(ti_from)
    s1 = np.sqrt(np.log1p(np.maximum(np.asarray(ti_to, dtype=float), 0.0) ** 2))
    if s0 < 1e-6 or np.all(np.abs(s1 - s0) < 1e-9):
        return g
    z = (np.log(np.maximum(g, 1e-6)) + 0.5 * s0 * s0) / s0
    out = np.exp(s1 * z - 0.5 * s1 * s1)
    return float(out) if np.ndim(out) == 0 else out


def gust_power_mean(exponent: float, ti: float) -> float:
    """E[g^k] for a mean-one log-normal multiplier: exp(k (k - 1) s^2 / 2), s^2 = ln(1 + ti^2).

    Dividing a U^k level law by this calibrates it on the mean wind (the empirical laws of
    Curle-type and vegetation noise are fitted to Leq against the mean wind speed).
    """
    s2 = math.log1p(max(float(ti), 0.0) ** 2)
    k = float(exponent)
    return math.exp(0.5 * k * (k - 1.0) * s2)


class GustField:
    """Advected along-wind gust field shared by every wind emitter (Taylor's hypothesis).

    Re-runs the World's gust stream: a GustProcess on the same RNG layer ('wind.gust') with the
    same constructor arguments as raingen.physics.world.World (integral time, turbulence
    intensity incl. the initial stability factor, control rate = sr / block), stepped with the
    same per-block parameters taken from each block's WeatherState. `value` (= `at(0, U)`)
    therefore equals WeatherState.gust exactly. The block length is adopted from the first
    rendered block (`advance(..., n=ctx.n)`), so any engine block size stays in step.

    A point `upwind_m` metres upwind sees the listener's gust upwind_m / U seconds *earlier*
    (the GustProcess runs ahead by its lead time); a downwind point sees it later. Upwind
    trees therefore rustle first and the gust then sweeps past the listener.
    """

    LAYER = "wind.gust"            # == raingen.physics.world.World.GUST_LAYER

    def __init__(self, scene, rngf, sr: int = SAMPLE_RATE, block: int = BLOCK):
        from raingen.physics.world import World

        self.LAYER = str(getattr(World, "GUST_LAYER", GustField.LAYER))
        w = scene.weather.wind
        self.wind = w
        self.rngf = rngf
        self.sr = int(sr)
        self.z0 = scene_z0(scene)
        self.listener_h = float(scene.listener.height_m)
        u10 = max(float(w.speed_mps), 0.0)
        stab0 = initial_stability_factor(scene)
        try:
            # the World's own parameter code (run on a stand-in carrying its z0 and stability factor), so
            # the constructor arguments match World.gusts by construction
            shim = SimpleNamespace(z0=self.z0, _stab_factor=stab0)
            T0, ti0 = World._gust_params(shim, w, self.listener_h)
        except (AttributeError, TypeError):                              # pragma: no cover - older World
            T0 = gust_integral_time_s(w, u10, self.z0, self.listener_h)
            ti0 = ear_turbulence_intensity(w, self.z0, self.listener_h, stab0)
        self._T0, self._ti0 = float(T0), float(ti0)
        self.block = int(block)
        self.proc = self._make_proc(self.block)
        self.lead_s = float(self.proc.lead_s)
        self.history_s = float(self.proc.history_s)
        self.value = 1.0
        self.ti = float(self._ti0)
        self._last: Optional[int] = None

    def _make_proc(self, block: int) -> GustProcess:
        return GustProcess(self.rngf, self.LAYER, integral_time_s=self._T0, ti=self._ti0,
                           control_rate=self.sr / max(int(block), 1))

    def advance(self, block_index: int, weather=None, n: Optional[int] = None) -> None:
        b = int(block_index)
        if n is not None and int(n) != self.block:
            # the World steps its gusts once per engine block: follow its control rate
            self.block = int(n)
            self.proc = self._make_proc(self.block)
            self._last = None
        if self._last == b:
            return
        if weather is not None:
            T = gust_integral_time_s(self.wind, max(float(weather.wind_u10_mps), 0.0), self.z0, self.listener_h)
            self.ti = float(weather.turbulence_intensity)
            self.proc.set_params(T, self.ti)
        self.value = float(self.proc.step(b))
        self._last = b

    def at(self, upwind_m: float, u_mps: float) -> float:
        """Gust multiplier at a point `upwind_m` metres upwind (negative = downwind) of the listener."""
        return max(0.05, float(self.proc.at_position(float(upwind_m), max(float(u_mps), 0.5))))

    def at_many(self, upwind_m, u_mps) -> np.ndarray:
        """`at` for arrays of points (one value per point), vectorised over the process history.

        Same arithmetic as GustProcess.at_position / at_delay (linear interpolation of the control-rate
        history, clamped to what is available), so each value equals the scalar `at` bit for bit.
        """
        up = np.atleast_1d(np.asarray(upwind_m, dtype=float))
        u = np.broadcast_to(np.atleast_1d(np.asarray(u_mps, dtype=float)), up.shape)
        p = self.proc
        try:
            block, buf, n_hist, rate = p._block, p._buf, p._n_hist, p.control_rate
            lo, hi = p._available()
        except AttributeError:                                           # pragma: no cover - other GustProcess
            return np.array([self.at(a, b) for a, b in zip(up, u)])
        if block < 0:
            return np.ones(up.shape)
        delay = -up / np.maximum(np.maximum(u, 0.5), 0.5)
        pos = np.clip(block - delay * rate, float(lo), float(hi))
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        i1 = np.minimum(i0 + 1, hi)
        v0 = buf[i0 % n_hist]
        v1 = buf[i1 % n_hist]
        return np.maximum(0.05, v0 + (v1 - v0) * frac)


# --------------------------------------------------------------------------
# Vectorised synthesis helpers (shaped-noise banks, static SH encoding)
# --------------------------------------------------------------------------

def bandpass_mag2(f: np.ndarray, fc, q) -> np.ndarray:
    """|H|^2 of a 0 dB-peak analog band-pass, 1 / (1 + Q^2 (f/fc - fc/f)^2); broadcasts fc, q against f."""
    fc = np.maximum(np.asarray(fc, dtype=float), 1e-3)
    q = np.asarray(q, dtype=float)
    ff = np.maximum(f, 1e-3)
    x = q * (ff / fc - fc / ff)
    return 1.0 / (1.0 + x * x)


def bandpass_coefs(fc, q, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised RBJ 0 dB-peak band-pass: (b (..., 3), a (..., 3)) with a0 = 1."""
    fc = np.clip(np.asarray(fc, dtype=float), 5.0, 0.45 * sr)
    q = np.maximum(np.asarray(q, dtype=float), 0.05)
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    zero = np.zeros_like(alpha)
    b = np.stack([alpha / a0, zero, -alpha / a0], axis=-1)
    a = np.stack([np.ones_like(alpha), -2.0 * np.cos(w0) / a0, (1.0 - alpha) / a0], axis=-1)
    return b, a


def biquad_response(b: np.ndarray, a: np.ndarray, zinv: np.ndarray) -> np.ndarray:
    """Complex response of biquads (..., 3) on the unit-circle points z^-1 = `zinv` (F,) -> (..., F)."""
    z1 = zinv
    z2 = zinv * zinv
    num = b[..., 0:1] + b[..., 1:2] * z1 + b[..., 2:3] * z2
    den = a[..., 0:1] + a[..., 1:2] * z1 + a[..., 2:3] * z2
    return num / den


_U16_STD = math.sqrt((65536.0 ** 2 - 1.0) / 12.0)       # std of a uniform int16 (zero mean after +0.5)


def white_spectrum(rng: np.random.Generator, rows: int, bins: int) -> np.ndarray:
    """(rows, bins) complex white spectrum, zero-mean unit-variance parts, from uniform int16 draws.

    A time-domain sample of the inverse FFT sums ~`bins` independent parts, so it is Gaussian by the
    central limit theorem (excess kurtosis ~ -1.2 / bins) while the draws cost ~8x less than
    standard normals from the Philox streams.
    """
    x = np.frombuffer(rng.bytes(4 * rows * bins), dtype=np.int16).astype(np.float64)
    x += 0.5
    x *= 1.0 / _U16_STD
    return x.view(np.complex128).reshape(rows, bins)


class ShapedNoiseOLA:
    """Gaussian noise for V voices with a per-block magnitude spectrum each (vectorised).

    Each block draws one random frame of 2n samples per voice directly in the frequency
    domain (unit-variance white, `white_spectrum`), multiplies it by |H_v(f)| on the frame's rfft grid and
    overlap-adds it with a sine window at 50 % overlap (w^2 sums to 1, so independent
    frames keep the variance stationary). Spectral changes therefore crossfade over one
    block without clicks, and a whole bank of differently filtered voices costs one batched
    inverse FFT. Output variance of voice v is sum_k wgt_k |H_v,k|^2 (`power`), so a
    component spectrum B is normalised to unit variance by dividing by sqrt(power(B)).
    """

    def __init__(self, voices: int, sr: int = SAMPLE_RATE):
        self.V = int(voices)
        self.sr = int(sr)
        self.n = 0
        self.tail: Optional[np.ndarray] = None

    def setup(self, n: int) -> None:
        if n == self.n:
            return
        self.n = int(n)
        N = 2 * self.n
        self.N = N
        self.freqs = np.fft.rfftfreq(N, 1.0 / self.sr)
        F = self.freqs.size
        wgt = np.full(F, 2.0 / N)
        wgt[0] = 1.0 / N
        if N % 2 == 0:
            wgt[-1] = 1.0 / N
        self.wgt = wgt
        self.window = np.sin(np.pi * (np.arange(N) + 0.5) / N)
        self.zinv = np.exp(-2j * np.pi * self.freqs / self.sr)
        # unit-variance white frame: complex bins carry N/2 per part, DC and Nyquist are real with N
        # (irfft ignores their imaginary parts)
        scale = np.full(F, math.sqrt(N / 2.0))
        scale[0] = math.sqrt(N)
        if N % 2 == 0:
            scale[-1] = math.sqrt(N)
        self.scale = scale
        self.tail = None

    def power(self, mag2: np.ndarray) -> np.ndarray:
        """Output variance for squared magnitudes `mag2` (..., F) on this grid."""
        return mag2 @ self.wgt

    def process(self, rng: np.random.Generator, mag: Optional[np.ndarray], n: int) -> np.ndarray:
        """(V, n) noise for magnitudes `mag` (V, F) (None = silence; the previous frame still decays)."""
        self.setup(n)
        N = self.N
        if mag is None:
            out = np.zeros((self.V, n)) if self.tail is None else self.tail
            self.tail = None
            return out
        spec = white_spectrum(rng, self.V, self.freqs.size)
        spec *= mag * self.scale
        frame = sfft.irfft(spec, N, axis=1, overwrite_x=True)
        frame *= self.window
        out = frame[:, :n]
        if self.tail is not None:
            out += self.tail
        self.tail = frame[:, n:]
        return out

    @property
    def active(self) -> bool:
        return self.tail is not None


class FramedNoise:
    """Gaussian noise for V voices with a slowly varying magnitude spectrum, independent of the block length.

    Frames of 2 hop samples (sine window, 50 % overlap-add, w^2 sums to 1) are drawn in the frequency
    domain (`white_spectrum`) and shaped by |H_v(f)|; frame k covers stream samples [k hop, k hop + 2 hop)
    and `spectrum(k)` -> (rng, mag (V, F)) is asked once per frame, so a spectrum can follow a quantity at
    the frame centre (k + 1) hop and the stream is the same for any engine block length. The fine
    resolution sr / (2 hop) (5.9 Hz for hop 4096) suits low, narrow spectra such as thunder's sub-bass,
    which a block-sized frame (23 Hz bins at 1024) would smear. Variance per voice is `power(|H|^2)`.
    """

    def __init__(self, voices: int, hop: int = 4096, sr: int = SAMPLE_RATE):
        self.V = int(voices)
        self.hop = int(hop)
        self.sr = int(sr)
        N = 2 * self.hop
        self.N = N
        self.freqs = np.fft.rfftfreq(N, 1.0 / self.sr)
        F = self.freqs.size
        wgt = np.full(F, 2.0 / N)
        wgt[0] = 1.0 / N
        wgt[-1] = 1.0 / N
        self.wgt = wgt
        self.window = np.sin(np.pi * (np.arange(N) + 0.5) / N)
        scale = np.full(F, math.sqrt(N / 2.0))
        scale[0] = scale[-1] = math.sqrt(N)
        self.scale = scale
        self.buf = np.zeros((self.V, 0))
        self.tail = np.zeros((self.V, self.hop))
        self.k = 0                                  # next frame index

    def power(self, mag2: np.ndarray) -> np.ndarray:
        return mag2 @ self.wgt

    def process(self, n: int, spectrum) -> np.ndarray:
        """The next n samples (V, n); `spectrum(k)` returns (rng, mag (V, F)) for frame k."""
        parts = [self.buf]
        have = self.buf.shape[1]
        while have < n:
            rng, mag = spectrum(self.k)
            spec = white_spectrum(rng, self.V, self.freqs.size)
            spec *= mag * self.scale
            frame = sfft.irfft(spec, self.N, axis=1, overwrite_x=True)
            frame *= self.window
            parts.append(frame[:, : self.hop] + self.tail)
            self.tail = frame[:, self.hop:]
            have += self.hop
            self.k += 1
        allb = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        self.buf = allb[:, n:]
        return allb[:, :n]


class StaticSHEncoder:
    """V fixed mono sources (V, n) -> one (N_AMBI, n) bus, as raingen.spatial.encoder.Encoder would.

    Uses the encoder's own spread model (`spread_gains`: cap-weighted coherent part plus a
    spread^2 energy share rendered through the fixed decorrelator bank). Decorrelation is linear
    and the same FIR per channel serves every source, so the diffuse parts of all voices are
    summed per channel first and decorrelated once: one emission and one FFT pass per block
    instead of one encoder call (and decorrelator pass) per voice.
    """

    DIFFUSE_FLOOR = 0.1        # spreads up to this keep no diffuse part (<= 1 % of their energy, -0.04 dB)

    def __init__(self, az, el, spread, order: int = AMBI_ORDER, sr: int = SAMPLE_RATE):
        from raingen.spatial.encoder import DecorrelatorBank, spread_gains

        az = np.atleast_1d(np.asarray(az, dtype=float))
        el = np.broadcast_to(np.atleast_1d(np.asarray(el, dtype=float)), az.shape)
        sp = np.broadcast_to(np.atleast_1d(np.asarray(spread, dtype=float)), az.shape)
        n_ch = (int(order) + 1) ** 2
        g_coh = np.zeros((n_ch, az.size))
        g_dif = np.zeros((n_ch, az.size))
        for v in range(az.size):
            c, d = spread_gains(int(order), float(az[v]), float(el[v]), float(sp[v]))
            g_coh[:, v] = c
            # near-point sources: skip the decorrelator pass for a diffuse share of at most spread^2
            g_dif[:, v] = d if float(sp[v]) > self.DIFFUSE_FLOOR else 0.0
        self.g_coh = g_coh
        self.g_dif = g_dif[1:]
        self.channels = n_ch
        self.dec = DecorrelatorBank(n_ch - 1, sr) if float(np.max(self.g_dif, initial=0.0)) > 0.0 else None
        self.tail: Optional[np.ndarray] = None

    def process(self, y: np.ndarray) -> np.ndarray:
        n = y.shape[1]
        out = self.g_coh @ y
        if self.dec is None:
            return out
        L = self.dec.length
        d = self.g_dif @ y
        if self.tail is None and not np.any(d):
            return out
        if hasattr(self.dec, "process_multi"):
            wet, tail = self.dec.process_multi(d, self.tail)
        else:                                                            # pragma: no cover - older encoder
            nfft = 1 << max(1, int(n + L - 2).bit_length())
            full = sfft.irfft(self.dec.spectrum(nfft) * sfft.rfft(d, nfft, axis=1), nfft, axis=1)[:, : n + L - 1]
            if self.tail is not None:
                full[:, : L - 1] += self.tail
            wet, tail = full[:, :n], full[:, n:]
        out[1:] += wet
        self.tail = tail.copy() if np.any(tail) else None
        return out


class SparseModalBank:
    """Parallel damped modes driven by sparse impulses (strikes), exact across blocks.

    Each mode is the complex one-pole s[k] = p s[k-1] + x[k], p = r e^{i theta},
    r = 10^(-3 / (T60 sr)); the output is sum_m Im(s_m). Strikes are (offset, amps)
    with one amplitude per mode, so the strike position can weight the modes. The
    block is solved in closed form (carried ringing + one geometric run per strike),
    so cost scales with the number of strikes, not with the number of samples.
    """

    def __init__(self, freqs, t60s, sr: int = SAMPLE_RATE):
        f = np.asarray(freqs, dtype=float).ravel()
        t60 = np.broadcast_to(np.asarray(t60s, dtype=float), f.shape).astype(float)
        keep = (f > 5.0) & (f < 0.45 * sr)
        self.freqs = f[keep]
        self.t60 = np.maximum(t60[keep], 4.0 / sr)
        self.keep = keep
        r = np.exp(-math.log(1000.0) / (self.t60 * sr))
        self.logp = np.log(r) + 1j * 2.0 * np.pi * self.freqs / sr
        self.s = np.zeros(self.freqs.size, dtype=complex)
        self._n = -1
        self._P0: Optional[np.ndarray] = None

    @property
    def m(self) -> int:
        return int(self.freqs.size)

    def _powers(self, n: int) -> np.ndarray:
        if self._n != n:
            self._P0 = np.exp(self.logp[:, None] * np.arange(n + 1, dtype=float)[None, :])
            self._n = n
        return self._P0

    def silent(self, eps: float = 1e-9) -> bool:
        return self.m == 0 or float(np.max(np.abs(self.s), initial=0.0)) < eps

    def process(self, n: int, strikes=()) -> np.ndarray:
        """Render n samples; `strikes` is an iterable of (offset, amps[m]) in this block."""
        if self.m == 0:
            return np.zeros(n)
        strikes = list(strikes)
        if not strikes and self.silent():
            self.s[:] = 0.0
            return np.zeros(n)
        P = self._powers(n)
        S = self.s[:, None] * P[:, 1:]
        for off, amps in strikes:
            off = int(min(max(off, 0), n - 1))
            a = np.asarray(amps, dtype=float).ravel()
            if a.size == self.keep.size and a.size != self.m:
                a = a[self.keep]
            elif a.size != self.m:
                a = np.resize(a, self.m)
            S[:, off:] += a[:, None] * P[:, : n - off]
        self.s = S[:, -1].copy()
        return S.imag.sum(axis=0)


# --------------------------------------------------------------------------
# Wind bed
# --------------------------------------------------------------------------

BED_REF_DB = 56.0          # omni air-movement bed at a mean ear-height wind of 10 m/s (dB SPL)
BED_EXPONENT = 3.0         # p ~ U^3: compact aerodynamic dipoles, intensity ~ U^6 (Curle 1955)
RUMBLE_REF_DB = 46.0       # low-frequency pseudo-sound at 10 m/s (dB SPL), p' ~ rho U u' ~ U^2 (below the bed)
BED_ADVECT_M = 3.0         # each bed direction hears the turbulence of obstacles a few metres away
BED_FLUTTER = 0.15         # independent per-direction flutter (log-std per unit turbulence intensity)


def _bed_amplitude(u: float) -> float:
    """RMS pressure of the omni air-movement bed vs wind at ear height (~56 dB at 10 m/s, U^3)."""
    if u <= 0.0:
        return 0.0
    return db_spl(BED_REF_DB) * (u / 10.0) ** BED_EXPONENT * smoothstep(u, 0.3, 1.5)


def _debris_burst(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Skittering twigs/leaves/litter: a cluster of band-limited ticks plus a body knock."""
    dur = rng.uniform(0.08, 0.6)
    n_body = int(dur * sr)
    L = n_body + int(0.03 * sr)
    imp = np.zeros(L)
    k = int(rng.integers(3, 28))
    offs = np.sort(rng.integers(0, max(1, n_body), size=k))
    np.add.at(imp, offs, rng.lognormal(0.0, 0.6, size=k) * rng.choice([-1.0, 1.0], size=k))
    crisp = signal.sosfilt(bandpass_sos(rng.uniform(1500.0, 5500.0), 2.0, sr), imp)
    body = signal.sosfilt(bandpass_sos(rng.uniform(250.0, 800.0), 4.0, sr), imp)
    y = crisp + 0.6 * body
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


class WindGenerator:
    """Low broadband air movement plus a turbulence-driven SH-domain bed.

    Level and brightness follow WeatherState.wind_listener_mps and the gust
    multiplier; the bed is biased toward the windward azimuth and gusts sweep
    across its directions (advected through the shared GustField). From Beaufort
    8 loose debris skitters around the listener. Gust onsets are exported as
    EventMarker('gust') for captions and visual sync.
    """

    name = "wind"
    N_DIR = 8

    def __init__(self, scene, layout, rngf, gust_field: Optional[GustField] = None, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.scene = scene
        self.field = gust_field or GustField(scene, rngf, sr)
        k = np.arange(self.N_DIR)
        self.dir_az = 2.0 * np.pi * k / self.N_DIR
        self.dir_el = np.where(k % 2 == 0, 0.25, -0.1)
        self.Y = np.stack([sh_gains(a, e, spread=0.3) for a, e in zip(self.dir_az, self.dir_el)], axis=1)
        self._pink = np.vstack([slope_sos(-10.0, 20.0, 16000.0, sr), highpass_sos(18.0, sr)])
        self._fc = 0.0
        self.shape = StatefulSOS(np.vstack([self._pink, lowpass_sos(500.0, sr)]))
        self._norm = 1.0
        self._set_brightness(500.0)
        sos, self._rnorm, _ = cached_design("wind.rumble", 60.0, sr, lambda f: lowpass_sos(f, sr), grid=1.03)
        self.rumble = StatefulSOS(sos)
        self._rfc = 60.0
        self.bed_ramp = Ramp(np.zeros(self.N_DIR))
        self.rumble_ramp = Ramp(0.0)
        self.turb = np.zeros(self.N_DIR)
        self.debris = PendingMixer(N_AMBI)
        self._gusting = False
        self._last_gust_t = -1e9

    def _set_brightness(self, fc: float) -> None:
        if self._fc and abs(fc - self._fc) / self._fc < 0.03:
            return
        # designs are cached on a 2 % log-frequency grid: gusts revisit the same corners
        sos, norm, _ = cached_design("wind.bed", fc, self.sr,
                                     lambda f: np.vstack([self._pink, lowpass_sos(f, self.sr)]))
        self._fc = fc
        self.shape.set(sos)
        self._norm = norm

    def _gust_marker(self, ctx: BlockContext, u: float, gust: float, ti: float) -> None:
        """EventMarker('gust') when the gust multiplier crosses ~1.6 sigma (with hysteresis)."""
        on = 1.0 + 1.6 * ti
        off = 1.0 + 0.6 * ti
        if not self._gusting and gust >= on and u >= 4.0 and ctx.t0 - self._last_gust_t >= 4.0:
            w = ctx.weather
            u10g = float(w.wind_u10_mps) * gust
            ctx.markers.append(EventMarker(t=float(ctx.t0), kind="gust", data={
                "wind_mps": round(u, 2), "gust": round(gust, 3), "wind_10m_mps": round(u10g, 2),
                "beaufort": round(float(mps_to_beaufort(u10g)), 1),
                "from_deg": round(float(w.wind_from_deg), 1), "azimuth": round(float(w.wind_azimuth), 4)}))
            self._gusting = True
            self._last_gust_t = float(ctx.t0)
        elif self._gusting and gust <= off:
            self._gusting = False

    def render(self, ctx: BlockContext) -> list[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        self.field.advance(ctx.index, w, n)
        u = max(float(w.wind_listener_mps), 0.0)
        gust = max(float(w.gust), 0.05)
        u_mean = u / gust
        start = block_start_sample(ctx)
        ti = max(0.05, float(w.turbulence_intensity))
        self._gust_marker(ctx, u, gust, ti)
        if u < 0.2 and not self.debris.active and float(np.max(self.bed_ramp.value)) < 1e-7:
            return []
        rng = ctx.rng.get("wind.bed", ctx.index)

        # Ornstein-Uhlenbeck turbulence per direction (control rate, tau ~ 2 s)
        a = math.exp(-(n / sr) / 2.0)
        self.turb = a * self.turb + math.sqrt(1.0 - a * a) * rng.standard_normal(self.N_DIR)

        # each direction hears the obstacles a few metres away: upwind directions get the gust a
        # fraction of a second first, so gusts sweep across the bed while it stays locked to the
        # listener's gust; a small independent flutter per direction keeps the bed alive
        rel = np.cos(self.dir_az - w.wind_azimuth)
        weight = 0.35 + 0.65 * ((1.0 + rel) / 2.0) ** 2
        weight = weight / math.sqrt(float(np.sum(weight ** 2)))
        g_dir = self.field.at_many(BED_ADVECT_M * rel, u_mean) * np.exp(BED_FLUTTER * ti * self.turb)
        # Leq follows the mean-wind law: divide out E[(g e^{flutter ti Z})^3^2] of the log-normal gusts
        s2 = math.log1p(ti * ti)
        norm = math.exp(0.5 * BED_EXPONENT * (BED_EXPONENT - 1.0) * s2 + 0.5 * (BED_FLUTTER * BED_EXPONENT * ti) ** 2)
        amps = _bed_amplitude(u_mean) * g_dir ** BED_EXPONENT / math.sqrt(norm) * weight

        self._set_brightness(min(120.0 + 75.0 * u, 5000.0))
        noise = rng.standard_normal((self.N_DIR, n))
        shaped = self.shape.process(noise)
        shaped *= self.bed_ramp(amps / self._norm, n)
        bus = self.Y @ shaped

        # Low broadband air movement (pressure fluctuation felt more than heard), omni, ~U^2
        rfc = 40.0 + 6.0 * u
        if abs(rfc - self._rfc) / self._rfc > 0.05:
            self._rfc = rfc
            sos, self._rnorm, _ = cached_design("wind.rumble", rfc, sr, lambda f: lowpass_sos(f, sr), grid=1.03)
            self.rumble.set(sos)
        r_amp = (db_spl(RUMBLE_REF_DB) * (u_mean / 10.0) ** 2 * smoothstep(u_mean, 0.5, 3.0)
                 * gust ** 2 / math.sqrt(gust_power_mean(4.0, ti)))
        bus[0] += self.rumble.process(rng.standard_normal(n)) * self.rumble_ramp(r_amp / self._rnorm, n)

        # Debris from Beaufort 8 (17.2 m/s): twigs and litter skittering past
        if u > 15.0:
            drng = ctx.rng.get("wind.debris", ctx.index)
            rate = 0.8 * smoothstep(u, 15.0, 30.0) * (u / 17.2)
            for off in poisson_offsets(drng, rate, n, sr):
                burst = _debris_burst(drng, sr)
                dist = drng.uniform(2.0, 15.0)
                az = float(w.wind_azimuth + np.pi + drng.normal(0.0, 1.0))
                amp = db_spl(68.0) * (u / 20.0) ** 2 / dist
                y = sh_gains(az, -math.atan2(1.6, dist), spread=0.2)
                self.debris.add(start + int(off), np.outer(y, burst * amp))
        if self.debris.active:
            bus += self.debris.take(start, n)
        return [Emission(signal=bus, stem="wind", is_ambisonic=True, spread=1.0, source_id="wind.bed")]


# --------------------------------------------------------------------------
# Microphone wind rumble (optional artifact)
# --------------------------------------------------------------------------

WINDSCREENS = {
    # characteristic size D (m) of the turbulent interaction region, broadband attenuation (dB)
    "none": {"d_m": 0.021, "atten_db": 0.0},
    "foam": {"d_m": 0.06, "atten_db": 12.0},
    "blimp": {"d_m": 0.20, "atten_db": 22.0},
    "dead_cat": {"d_m": 0.25, "atten_db": 32.0},
    "fur": {"d_m": 0.08, "atten_db": 24.0},
}


class MicRumbleGenerator:
    """Wind noise on a microphone capsule (default-off 'recorded realism' artifact, stem 'wind').

    Spectrum: -6.7 dB/decade below fc = V/(3D), -26.7 dB/decade above; level ~ U^2
    (dynamic pressure) minus the windscreen attenuation. Emitted as decorrelated
    first-order components so a stereo decode yields uncorrelated L/R rumble.
    """

    name = "mic_rumble"

    def __init__(self, scene, rngf=None, sr: int = SAMPLE_RATE, windscreen: Optional[str] = None):
        self.sr = sr
        ws = windscreen or getattr(scene.weather.wind, "windscreen", "blimp") or "blimp"
        self.screen = WINDSCREENS.get(ws, WINDSCREENS["blimp"])
        self.tilt = slope_sos(-6.7, 5.0, 20000.0, sr)
        self._fc = 0.0
        self.filt = StatefulSOS(np.vstack([self.tilt, signal.butter(1, 100.0, "low", fs=sr, output="sos")]))
        self._norm = 1.0
        self.env_lp = StatefulSOS(lowpass_sos(3.0, sr, order=1))
        self.ramp = Ramp(0.0)
        # the capsule only sees wind where the listener does (no wind inside a car or a room)
        shelter = shelter_of(scene)
        self.exposure = {"porch": 0.4, "cave": 0.3, "underpass": 0.7, "umbrella": 0.9}.get(shelter, 1.0)
        if shelter in ("car", "tent", "room_closed", "attic"):
            self.exposure = 0.0
        elif shelter == "room_open":
            self.exposure = 0.05 * float(getattr(scene.place, "window_open", 0.0))

    def _design(self, fc: float) -> None:
        if self._fc and abs(fc - self._fc) / self._fc < 0.05:
            return
        sos, norm, _ = cached_design("wind.mic", fc, self.sr, lambda f: np.vstack(
            [self.tilt, signal.butter(1, clip_f(f, self.sr), "low", fs=self.sr, output="sos")]), grid=1.03)
        self._fc = fc
        self.filt.set(sos)
        self._norm = norm

    def render(self, ctx: BlockContext) -> list[Emission]:
        u = max(float(ctx.weather.wind_listener_mps), 0.0) * self.exposure
        n = ctx.n
        if u < 0.3 and float(self.ramp.value) < 1e-7:
            return []
        rng = ctx.rng.get("wind.mic_rumble", ctx.index)
        self._design(max(u, 0.3) / (3.0 * self.screen["d_m"]))
        amp = db_spl(95.0 - self.screen["atten_db"]) * (u / 10.0) ** 2 * smoothstep(u, 0.3, 1.5)
        noise = rng.standard_normal((3, n))
        x = self.filt.process(noise) / self._norm
        # turbulent intermittency: eddies hitting the capsule
        env = self.env_lp.process(rng.standard_normal(n)) * math.sqrt(self.sr / (math.pi * 3.0))
        env = np.exp(0.6 * np.clip(env, -3.0, 3.0) - 0.18)
        g = self.ramp(amp, n) * env
        bus = np.zeros((N_AMBI, n))
        bus[0] = x[0] * g
        bus[1] = 0.8 * x[1] * g    # ACN1 = Y (left/right): decorrelated capsules
        bus[3] = 0.4 * x[2] * g    # ACN3 = X
        return [Emission(signal=bus, stem="wind", is_ambisonic=True, spread=1.0, source_id="wind.mic_rumble")]
