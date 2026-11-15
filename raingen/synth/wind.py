"""Wind bed, microphone wind rumble, and the shared environment DSP toolkit.

Physics implemented here:

* Wind is heard through objects (MASTER_PLAN 3.6). The free-field "air movement"
  bed models turbulence scattering around the head and near obstacles: aerodynamic
  dipole sources radiate intensity ~U^6 (Curle 1955); for a bed dominated by
  low-Mach turbulent flow over many obstacles we use pressure ~U^2.5 with a
  brightness (spectral corner) rising with U.
* Gusts follow a normalized Kaimal process (raingen.physics.wind.GustProcess) and
  are *advected* (Taylor's frozen-turbulence hypothesis): a point s metres upwind
  of the listener experiences a gust s/U seconds before the listener does. The
  shared GustField re-runs the World's own gust stream (same RNG layer, same
  integral time and turbulence intensity), so the gust heard in upwind trees is
  exactly the one that reaches the listener's WeatherState.gust moments later.
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
from typing import Optional

import numpy as np
from scipy import signal

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


def lowpass_sos(fc: float, sr: int, order: int = 2) -> np.ndarray:
    return signal.butter(order, clip_f(fc, sr), "low", fs=sr, output="sos")


def highpass_sos(fc: float, sr: int, order: int = 2) -> np.ndarray:
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


def noise_rms_gain(sos: np.ndarray) -> float:
    """RMS gain of a filter for white noise input: sqrt((1/pi) int_0^pi |H(w)|^2 dw).

    Integrated on a log+linear frequency grid so very narrow / very low filters
    (a few Hz at 48 kHz) are measured accurately.
    """
    _, h = signal.sosfreqz(np.atleast_2d(sos), worN=_NRG_W)
    p = np.abs(h) ** 2
    var = float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(_NRG_W))) / math.pi
    return float(math.sqrt(max(var, 1e-24)))


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


class StatefulSOS:
    """sosfilt with state carried across blocks, for (n,) or (C, n) signals.

    Coefficients may be replaced between blocks (gradual parameter changes); the
    state is kept when the section count is unchanged.
    """

    def __init__(self, sos: np.ndarray):
        self.sos = np.atleast_2d(np.asarray(sos, dtype=float))
        self.zi: Optional[np.ndarray] = None

    def set(self, sos: np.ndarray) -> None:
        sos = np.atleast_2d(np.asarray(sos, dtype=float))
        if sos.shape != self.sos.shape:
            self.zi = None
        self.sos = sos

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        shape = (self.sos.shape[0],) + x.shape[:-1] + (2,)
        if self.zi is None or self.zi.shape != shape:
            self.zi = np.zeros(shape)
        y, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y

    def reset(self) -> None:
        self.zi = None


class Ramp:
    """Linear per-block parameter ramp (scalar or vector), avoids zipper noise."""

    def __init__(self, value=0.0):
        self.value = np.asarray(value, dtype=float)

    def __call__(self, target, n: int) -> np.ndarray:
        v1 = np.asarray(target, dtype=float)
        v0 = np.broadcast_to(self.value, v1.shape) if self.value.shape != v1.shape else self.value
        frac = np.arange(1, n + 1, dtype=float) / n
        out = v0[..., None] + (v1 - v0)[..., None] * frac
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


def ear_turbulence_intensity(wind, z0: float, listener_h: float) -> float:
    """Turbulence intensity at the ears; a user TI / gust factor refers to 10 m (neutral profile ratio)."""
    ti10_auto = float(turbulence_intensity_for(z0, 10.0))
    if getattr(wind, "turbulence_intensity", None) is not None:
        ti10 = float(wind.turbulence_intensity)
    elif getattr(wind, "gust_factor", None) is not None:
        ti10 = max(float(wind.gust_factor) - 1.0, 0.0) / 3.0
    else:
        ti10 = ti10_auto
    ti = float(turbulence_intensity_for(z0, max(float(listener_h), 0.3))) * ti10 / max(ti10_auto, 1e-6)
    return float(min(max(ti, 0.0), 0.8))


class GustField:
    """Advected along-wind gust field shared by every wind emitter (Taylor's hypothesis).

    Re-runs the World's gust stream (RNG layer 'wind.gust', same integral time and
    turbulence intensity, updated from each block's WeatherState), so `at(0, U)` equals
    WeatherState.gust. A point `upwind_m` metres upwind sees the listener's gust
    upwind_m / U seconds *earlier* (the GustProcess runs ahead by its lead time); a
    downwind point sees it later. Upwind trees therefore rustle first and the gust then
    sweeps past the listener.
    """

    LAYER = "wind.gust"

    def __init__(self, scene, rngf, sr: int = SAMPLE_RATE, block: int = BLOCK):
        w = scene.weather.wind
        self.wind = w
        self.z0 = scene_z0(scene)
        self.listener_h = float(scene.listener.height_m)
        u10 = max(float(w.speed_mps), 0.0)
        self.proc = GustProcess(rngf, self.LAYER,
                                integral_time_s=gust_integral_time_s(w, u10, self.z0, self.listener_h),
                                ti=ear_turbulence_intensity(w, self.z0, self.listener_h),
                                control_rate=sr / block)
        self.lead_s = float(self.proc.lead_s)
        self.history_s = float(self.proc.history_s)
        self.value = 1.0
        self._last: Optional[int] = None

    def advance(self, block_index: int, weather=None) -> None:
        b = int(block_index)
        if self._last == b:
            return
        if weather is not None:
            T = gust_integral_time_s(self.wind, max(float(weather.wind_u10_mps), 0.0), self.z0, self.listener_h)
            self.proc.set_params(T, float(weather.turbulence_intensity))
        self.value = float(self.proc.step(b))
        self._last = b

    def at(self, upwind_m: float, u_mps: float) -> float:
        """Gust multiplier at a point `upwind_m` metres upwind (negative = downwind) of the listener."""
        return max(0.05, float(self.proc.at_position(float(upwind_m), max(float(u_mps), 0.5))))


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

def _bed_amplitude(u: float) -> float:
    """RMS pressure of the omni air-movement bed vs wind at ear height (~56 dB at 10 m/s)."""
    if u <= 0.0:
        return 0.0
    return db_spl(56.0) * (u / 10.0) ** 2.5 * smoothstep(u, 0.3, 1.5)


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
        self.pink_hp = StatefulSOS(np.vstack([slope_sos(-10.0, 20.0, 16000.0, sr), highpass_sos(18.0, sr)]))
        self._fc = 0.0
        self.lp = StatefulSOS(lowpass_sos(500.0, sr))
        self._norm = 1.0
        self._set_brightness(500.0)
        self.rumble = StatefulSOS(lowpass_sos(60.0, sr))
        self._rfc = 60.0
        self._rnorm = noise_rms_gain(self.rumble.sos)
        self.bed_ramp = Ramp(np.zeros(self.N_DIR))
        self.rumble_ramp = Ramp(0.0)
        self.turb = np.zeros(self.N_DIR)
        self.debris = PendingMixer(N_AMBI)
        self._gusting = False
        self._last_gust_t = -1e9

    def _set_brightness(self, fc: float) -> None:
        if self._fc and abs(fc - self._fc) / self._fc < 0.03:
            return
        self._fc = fc
        self.lp.set(lowpass_sos(fc, self.sr))
        self._norm = noise_rms_gain(np.vstack([self.pink_hp.sos, self.lp.sos]))

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
        self.field.advance(ctx.index, w)
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

        # each direction hears the air ~30 m away: upwind directions get the gust first
        rel = np.cos(self.dir_az - w.wind_azimuth)
        weight = 0.35 + 0.65 * ((1.0 + rel) / 2.0) ** 2
        weight = weight / math.sqrt(float(np.sum(weight ** 2)))
        amps = np.empty(self.N_DIR)
        for k in range(self.N_DIR):
            gk = self.field.at(30.0 * rel[k], u_mean)
            uk = u_mean * gk * math.exp(0.3 * ti * self.turb[k])
            amps[k] = _bed_amplitude(uk) * weight[k]

        self._set_brightness(min(120.0 + 75.0 * u, 5000.0))
        noise = rng.standard_normal((self.N_DIR, n))
        shaped = self.lp.process(self.pink_hp.process(noise)) / self._norm
        bed = shaped * self.bed_ramp(amps, n)
        bus = self.Y @ bed

        # Low broadband air movement (pressure fluctuation felt more than heard), omni
        rfc = 40.0 + 6.0 * u
        if abs(rfc - self._rfc) / self._rfc > 0.05:
            self._rfc = rfc
            self.rumble.set(lowpass_sos(rfc, sr))
            self._rnorm = noise_rms_gain(self.rumble.sos)
        r_amp = db_spl(52.0) * (u / 10.0) ** 2 * smoothstep(u, 0.5, 3.0)
        bus[0] += self.rumble.process(rng.standard_normal(n)) / self._rnorm * self.rumble_ramp(r_amp, n)

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
        self._fc = fc
        self.filt.set(np.vstack([self.tilt, signal.butter(1, clip_f(fc, self.sr), "low", fs=self.sr, output="sos")]))
        self._norm = noise_rms_gain(self.filt.sos)

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
