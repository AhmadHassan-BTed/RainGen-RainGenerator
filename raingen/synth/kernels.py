"""Sound kernels for drop, bubble, splash and hail impacts, and the resonators they drive.

Physics implemented
- Liquid-drop force on a rigid surface (Gordillo, Sun & Cheng 2018, JFM 840): the force
  rises like sqrt(t), peaks at F_max ~ 0.83 rho D^2 V^2 at t ~ 0.18 D/V and decays over
  ~D/V. We use sqrt(t) up to the peak and a log-normal tail (sigma 0.8) after it; the
  pulse integral is ~0.81 of the drop momentum, as expected with lateral splash.
  Larger drops are therefore louder *and* darker (longer pulse, lower roll-off ~V/D).
- Hertz contact for hail / ice pellets (Hertz 1882; Johnson 1985, Contact Mechanics):
  K = 4/3 E* sqrt(R), delta_max = (5 m v^2 / 4K)^(2/5), F_max = K delta_max^(3/2),
  contact time 2.94 delta_max / v, force ~ sin(pi t / tau)^(3/2). When the mean contact
  pressure exceeds the ice crushing strength the stone behaves as a crushing projectile:
  duration ~0.6 D/v and impulse m v (1 + e).
- Air bubbles (Minnaert 1933): f a ~ 3.28 m/s at 1 atm (a = radius). Single-bubble voice
  after van den Doel (2005), as used by Moss et al. (2010):
  p(t) = eps a sin(2 pi f0 (t + xi beta t^2 / 2)) exp(-beta t),
  beta = 0.13/a + 0.0072 a^(-3/2) (viscous + thermal + radiation damping),
  xi ~ 0.1 the rising-pitch factor rated most realistic.
- Impact click on water (Medwin et al. 1992; Franz 1959): a 50-100 us broadband
  pulse; amplitude ~ U^3 (a dipole radiates ~dF/dt ~ rho D V^3 from the Gordillo force).
- Modal resonators: each surface mode is a damped sinusoid g r^n sin(n theta),
  r = 10^(-3/(T60 sr)), realised as a *parallel* bank of 2nd-order sections.

Every kernel is deterministic; any noise inside a kernel is drawn from RngFactory
streams with fixed keys, so kernels are identical on every run and platform.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from raingen.core.constants import P0, RHO_WATER, SAMPLE_RATE
from raingen.core.rng import RngFactory

from .materials import Material, get_material

_LN1000 = math.log(1000.0)          # 60 dB amplitude decay: r^(T60 sr) = 1e-3
_PULSE_SIGMA = 0.8                  # log-normal tail width of the drop force pulse
_PULSE_SPAN = 12.0                  # pulse kernel length in units of tau = D/V
_KERNEL_RNG = RngFactory(0x6B65726E)  # 'kern'

V_REF = 6.5                         # m/s, terminal speed of a 2 mm drop (Atlas 1973)
D_REF_MM = 2.0


# ---------------------------------------------------------------------------
# helpers

def _area_sample(fn, length: int, sr: float, os: int) -> np.ndarray:
    """Sample a continuous waveform by averaging over each output sample (anti-aliased)."""
    t = (np.arange(length * os, dtype=float) + 0.5) / (sr * os)
    return fn(t).reshape(length, os).mean(axis=1)


def _fade_tail(k: np.ndarray, frac: float = 0.1) -> np.ndarray:
    m = int(len(k) * frac)
    if m >= 2:
        k[-m:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, m))
    return k


def force_peak(d_mm, v_mps):
    """Gordillo et al. (2018) peak force F_max ~ 0.83 rho D^2 V^2 (N); vectorized."""
    d = np.asarray(d_mm, dtype=float) * 1e-3
    v = np.asarray(v_mps, dtype=float)
    return 0.83 * RHO_WATER * d * d * v * v


# ---------------------------------------------------------------------------
# drop force pulse

def pulse_shape(tau_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Force pulse of unit F_max with time scale tau = D/V (area-sampled, impulse-preserving)."""
    tau = max(float(tau_s), 2e-6)
    tp = 0.18 * tau
    length = max(2, int(math.ceil(_PULSE_SPAN * tau * sr)) + 1)
    os = 16 if tau * sr < 4.0 else 8
    two_s2 = 2.0 * _PULSE_SIGMA ** 2

    def f(t):
        x = t / tp
        lx = np.log(np.maximum(x, 1e-12))
        return np.where(x <= 1.0, np.sqrt(x), np.exp(-lx * lx / two_s2))

    return _area_sample(f, length, sr, os)


def impact_pulse(d_mm: float, v_mps: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Force (N per sample) of a liquid drop of diameter d_mm striking a rigid surface at v_mps."""
    d = max(float(d_mm), 0.01)
    v = max(float(v_mps), 0.05)
    return float(force_peak(d, v)) * pulse_shape(d * 1e-3 / v, sr)


# ---------------------------------------------------------------------------
# Hertz contact (hail, ice pellets, graupel)

def hertz_contact(d_mm, v_mps, *, rho: float = 917.0, e_proj: float = 9.0e9,
                  e_target=2.0e11, nu_proj: float = 0.33, nu_target: float = 0.3,
                  crush_mpa: float = 10.0, restitution=0.3):
    """(F_max N, contact time s) of a sphere hitting a half-space; crushing-limited for ice.

    Vectorized over d_mm / v_mps (and e_target / restitution); scalars in -> floats out.
    """
    scalar = np.ndim(d_mm) == 0 and np.ndim(v_mps) == 0
    d = np.maximum(np.asarray(d_mm, dtype=float), 0.05) * 1e-3
    v = np.maximum(np.asarray(v_mps, dtype=float), 0.05)
    r = 0.5 * d
    m = rho * np.pi * d ** 3 / 6.0
    e_star = 1.0 / ((1.0 - nu_proj ** 2) / e_proj + (1.0 - nu_target ** 2) / np.asarray(e_target, dtype=float))
    k = 4.0 / 3.0 * e_star * np.sqrt(r)
    delta = (5.0 * m * v * v / (4.0 * k)) ** 0.4
    f_max = k * delta ** 1.5
    tau = 2.94 * delta / v
    if crush_mpa and crush_mpa > 0.0:
        p_mean = f_max / (np.pi * r * delta)            # contact radius^2 = R delta
        crushed = p_mean > crush_mpa * 1e6
        tau_c = np.maximum(tau, 0.6 * d / v)
        f_c = m * v * (1.0 + np.asarray(restitution, dtype=float)) / (0.574 * tau_c)  # int sin^1.5 = 0.574
        f_max = np.where(crushed, f_c, f_max)
        tau = np.where(crushed, tau_c, tau)
    if scalar:
        return float(f_max), float(tau)
    return f_max, tau


def hertz_shape(tau_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Unit-peak Hertz force history sin(pi t / tau)^(3/2), area-sampled."""
    tau = max(float(tau_s), 1e-6)
    length = max(1, int(math.ceil(tau * sr)) + 1)
    os = 32 if tau * sr < 2.0 else 8

    def f(t):
        x = np.clip(t / tau, 0.0, 1.0)
        return np.where(t < tau, np.sin(np.pi * x) ** 1.5, 0.0)

    return _area_sample(f, length, sr, os)


def hertz_pulse(d_mm: float, v_mps: float, sr: int = SAMPLE_RATE, **kw) -> np.ndarray:
    f_max, tau = hertz_contact(float(d_mm), float(v_mps), **kw)
    return f_max * hertz_shape(tau, sr)


def fade_tail(k: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """Raised-cosine fade over the last `frac` of a kernel (in place, returned)."""
    return _fade_tail(k, frac)


# ---------------------------------------------------------------------------
# bubbles and clicks

def minnaert_frequency(radius_m, pressure_pa: float = P0):
    """Minnaert resonance f ~ 3.28/a * sqrt(P/P0) (Hz); vectorized."""
    a = np.maximum(np.asarray(radius_m, dtype=float), 1e-6)
    return 3.28 / a * math.sqrt(max(pressure_pa, 1.0) / P0)


def bubble_damping(radius_m):
    """van den Doel (2005) bubble decay rate beta (1/s) = 0.13/a + 0.0072 a^-1.5."""
    a = np.maximum(np.asarray(radius_m, dtype=float), 1e-6)
    return 0.13 / a + 0.0072 * a ** -1.5


def bubble_kernel(radius_m: float, sr: int = SAMPLE_RATE, xi: float = 0.1, eps: float = 0.05,
                  damping_extra: float = 0.0) -> np.ndarray:
    """Rising-pitch damped sinusoid of a freshly entrained bubble (van den Doel 2005).

    eps scales the initial wall excursion (A0 = eps * a, eps in 0.01..0.1); damping_extra is an
    additional dimensionless damping constant delta (beta += pi f0 delta), e.g. a shallow puddle
    bottom or a dirty surface. The waveform is band-limited: it fades out as the chirp
    approaches 0.45 sr, and is empty for bubbles that start above it.
    """
    a = max(float(radius_m), 1e-6)
    f0 = float(minnaert_frequency(a))
    f_lim = 0.45 * sr
    if f0 >= f_lim:
        return np.zeros(1)
    beta = float(bubble_damping(a)) + math.pi * f0 * max(float(damping_extra), 0.0)
    dur = 9.21 / beta                       # -80 dB
    xi = max(float(xi), 0.0)
    if xi > 0.0:                            # stop where the chirp crosses the band limit
        t_lim = (f_lim / f0 - 1.0) / (xi * beta)
        dur = min(dur, t_lim)
    length = max(4, int(math.ceil(dur * sr)) + 1)
    t = np.arange(length, dtype=float) / sr
    inst = f0 * (1.0 + xi * beta * t)
    phase = 2.0 * np.pi * f0 * (t + 0.5 * xi * beta * t * t)
    k = eps * a * np.exp(-beta * t) * np.sin(phase)
    # smooth band-limit taper from 0.40 sr to 0.45 sr of instantaneous frequency
    w = np.clip((f_lim - inst) / (0.05 * sr), 0.0, 1.0)
    k *= np.sin(0.5 * np.pi * w) ** 2
    return _fade_tail(k, 0.08)


def click_kernel(v_mps: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Impact click on water: 50-100 us bipolar pulse, peak (U/6.5 m/s)^3 (unit for a 2 mm drop)."""
    u = max(float(v_mps), 0.05)
    amp = (u / V_REF) ** 3
    width = float(np.clip(75e-6 * math.sqrt(V_REF / u), 50e-6, 100e-6))
    sigma = width / 4.0
    t0 = width
    length = int(math.ceil(2.0 * width * sr)) + 2

    def f(t):
        z = (t - t0) / sigma
        return -z * np.exp(0.5 - 0.5 * z * z)

    k = _area_sample(f, length, sr, 16)
    peak = np.max(np.abs(k))
    if peak > 0:
        k /= peak
    return amp * k


def splash_kernel(duration_s: float, sr: int = SAMPLE_RATE, variant: int = 0,
                  hp_hz: float = 1500.0, lp_hz: float = 14000.0) -> np.ndarray:
    """Unit-peak crown-splash / splat noise burst: 0.3 ms attack, exponential decay to -40 dB."""
    dur = max(float(duration_s), 5e-4)
    length = int(math.ceil(dur * sr)) + 1
    rng = _KERNEL_RNG.get(f"splash/{int(round(dur * 1e5))}", int(variant))
    noise = rng.standard_normal(length)
    t = np.arange(length) / sr
    env = (1.0 - np.exp(-t / 3e-4)) * np.exp(-t * (4.6 / dur))
    sos = []
    if hp_hz > 0:
        sos.append(signal.butter(2, hp_hz, "highpass", fs=sr, output="sos"))
    if lp_hz < 0.45 * sr:
        sos.append(signal.butter(2, lp_hz, "lowpass", fs=sr, output="sos"))
    k = noise * env
    if sos:
        k = signal.sosfilt(np.vstack(sos), k)
    k = _fade_tail(k, 0.15)
    peak = np.max(np.abs(k))
    return k / peak if peak > 0 else k


# ---------------------------------------------------------------------------
# resonators

def _mode_arrays(modes, sr, detune=1.0, damping=1.0):
    if isinstance(modes, (Material, str)):
        modes = get_material(modes).modes
    if not modes:
        z = np.zeros(0)
        return z, z, z
    arr = np.asarray(modes, dtype=float).reshape(-1, 3)
    f = arr[:, 0] * np.broadcast_to(np.asarray(detune, dtype=float), arr[:, 0].shape)
    t60 = arr[:, 1] * np.broadcast_to(np.asarray(damping, dtype=float), arr[:, 1].shape)
    g = arr[:, 2]
    keep = (f > 5.0) & (f < 0.45 * sr) & (t60 > 0)
    return f[keep], np.maximum(t60[keep], 2.0 / sr), g[keep]


def modal_sos(material, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Per-mode 2nd-order sections, shape (M, 6). PARALLEL bank: sum the rows' outputs.

    Row m has impulse response g r^n sin(n theta): b = [0, g r sin(theta), 0],
    a = [1, -2 r cos(theta), r^2], r = exp(-ln(1000)/(T60 sr)).
    """
    f, t60, g = _mode_arrays(material, sr)
    if f.size == 0:
        return np.zeros((0, 6))
    r = np.exp(-_LN1000 / (t60 * sr))
    th = 2.0 * np.pi * f / sr
    sos = np.zeros((f.size, 6))
    sos[:, 1] = g * r * np.sin(th)
    sos[:, 3] = 1.0
    sos[:, 4] = -2.0 * r * np.cos(th)
    sos[:, 5] = r * r
    return sos


def radiation_sos(material, sr: int = SAMPLE_RATE) -> np.ndarray:
    """2nd-order high-pass at radiation_hp_hz and low-pass at radiation_lp_hz (cascade)."""
    mat = get_material(material) if not isinstance(material, Material) else material
    secs = []
    if mat.radiation_hp_hz > 5.0:
        secs.append(signal.butter(2, min(mat.radiation_hp_hz, 0.4 * sr), "highpass", fs=sr, output="sos"))
    if mat.radiation_lp_hz < 0.45 * sr:
        secs.append(signal.butter(2, mat.radiation_lp_hz, "lowpass", fs=sr, output="sos"))
    return np.vstack(secs) if secs else np.zeros((0, 6))


class StatefulSOS:
    """Cascade of biquads with state carried across blocks (1-D or (C, n) input along the last axis)."""

    def __init__(self, sos):
        self.sos = np.asarray(sos, dtype=float).reshape(-1, 6)
        self.zi = None

    def reset(self):
        self.zi = None

    def set_sos(self, sos):
        sos = np.asarray(sos, dtype=float).reshape(-1, 6)
        if sos.shape != self.sos.shape:
            self.zi = None
        self.sos = sos

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.sos.shape[0] == 0:
            return x.copy()
        shape = (self.sos.shape[0],) + x.shape[:-1] + (2,)
        if self.zi is None or self.zi.shape != shape:
            self.zi = np.zeros(shape)
        y, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


class OverlapAdd:
    """Accumulator for convolution outputs that extend past the current block (tails never truncated)."""

    def __init__(self, channels: int | None = None):
        self.channels = channels
        self.buf = np.zeros((0,) if channels is None else (channels, 0))
        self.used = 0

    @property
    def active(self) -> bool:
        return self.used > 0

    def _grow(self, size: int):
        cap = self.buf.shape[-1]
        if size <= cap:
            return
        new_cap = max(size, 2 * cap, 4096)
        nb = np.zeros(self.buf.shape[:-1] + (new_cap,))
        nb[..., :cap] = self.buf
        self.buf = nb

    def add(self, y: np.ndarray, offset: int = 0):
        y = np.asarray(y, dtype=float)
        if y.shape[-1] == 0:
            return
        end = int(offset) + y.shape[-1]
        self._grow(end)
        self.buf[..., int(offset):end] += y
        self.used = max(self.used, end)

    def pop(self, n: int) -> np.ndarray:
        out = np.zeros(self.buf.shape[:-1] + (n,))
        if self.used == 0:
            return out
        m = min(n, self.used)
        out[..., :m] = self.buf[..., :m]
        rem = self.used - n
        if rem > 0:
            self.buf[..., :rem] = self.buf[..., n:self.used].copy()
            self.buf[..., rem:self.used] = 0.0
            self.used = rem
        else:
            self.buf[..., :self.used] = 0.0
            self.used = 0
        return out


class ModalBank:
    """Parallel bank of damped modes, run as an exact block recursion with carried state.

    Each mode is the complex one-pole s[k] = p s[k-1] + x[k], y = sum_m g_m Im(s_m), p = r e^{i theta};
    identical to running every row of `modal_sos` through sosfilt with zi, but vectorized:
    per block one FFT convolution with the bank's truncated impulse response plus a matrix
    product for the ringing carried in from previous blocks.
    """

    def __init__(self, modes, sr: int = SAMPLE_RATE, detune=1.0, damping=1.0, gain: float = 1.0):
        f, t60, g = _mode_arrays(modes, sr, detune, damping)
        self.sr = int(sr)
        self.freqs = f
        self.t60 = t60
        self.g = g * float(gain)
        r = np.exp(-_LN1000 / (t60 * sr)) if f.size else np.zeros(0)
        self.logp = np.log(r) + 1j * (2.0 * np.pi * f / sr) if f.size else np.zeros(0, complex)
        self.p = np.exp(self.logp)
        self.state = np.zeros(f.size, dtype=complex)
        self._n = -1
        self._P = None
        self._h = None

    @property
    def n_modes(self) -> int:
        return int(self.freqs.size)

    @property
    def sos(self) -> np.ndarray:
        return modal_sos([(f, t, g) for f, t, g in zip(self.freqs, self.t60, self.g)], self.sr)

    def reset(self):
        self.state[:] = 0.0

    def _prepare(self, n: int):
        if self._n == n:
            return
        k = np.arange(1, n + 1, dtype=float)
        self._P = np.exp(self.logp[:, None] * k[None, :])          # p^(k), k = 1..n
        h = np.zeros(n)
        if n > 1:
            h[1:] = self.g @ self._P[:, : n - 1].imag                 # h[j] = sum g Im(p^j)
        self._h = h
        self._n = n

    def impulse_response(self, length: int) -> np.ndarray:
        if self.n_modes == 0:
            return np.zeros(length)
        out = np.zeros(length)
        chunk = 8192
        for s in range(0, length, chunk):
            k = np.arange(s, min(length, s + chunk), dtype=float)
            out[s:s + k.size] = self.g @ np.exp(self.logp[:, None] * k[None, :]).imag
        return out

    def is_silent(self, eps: float = 1e-12) -> bool:
        if self.n_modes == 0:
            return True
        if float(np.max(np.abs(self.state) * np.abs(self.g))) < eps:
            self.state[:] = 0.0
            return True
        return False

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.shape[-1]
        if self.n_modes == 0 or n == 0:
            return np.zeros(n)
        if n == 1:
            y = np.array([float(self.g @ (self.p * self.state).imag)])
            self.state = self.p * self.state + x[0]
            return y
        self._prepare(n)
        has_in = bool(np.any(x))
        has_state = bool(np.any(self.state))
        y = np.zeros(n)
        if has_in:
            y += signal.fftconvolve(x, self._h)[:n]
        if has_state:
            y += ((self.g * self.state) @ self._P).imag
        new_state = self._P[:, n - 1] * self.state if has_state else np.zeros_like(self.state)
        if has_in:
            new_state = new_state + self._P[:, : n - 1] @ x[n - 2::-1] + x[n - 1]
        self.state = new_state
        return y


# ---------------------------------------------------------------------------
# batched rendering of event streams

_LEN_BUCKETS = (0, 96, 384, 1536, 6144, 24576, 1 << 62)


def convolve_trains(trains: np.ndarray, kernels: list) -> np.ndarray:
    """sum_b trains[b] * kernels[b] (full convolution), batched by kernel-length bucket via oaconvolve."""
    trains = np.atleast_2d(np.asarray(trains, dtype=float))
    m = trains.shape[1]
    lens = np.array([max(len(k), 1) for k in kernels])
    out = np.zeros(m + int(lens.max()) - 1)
    for lo, hi in zip(_LEN_BUCKETS[:-1], _LEN_BUCKETS[1:]):
        idx = np.nonzero((lens > lo) & (lens <= hi))[0]
        if idx.size == 0:
            continue
        L = int(lens[idx].max())
        K = np.zeros((idx.size, L))
        for j, i in enumerate(idx):
            kern = np.asarray(kernels[i], dtype=float)
            K[j, : kern.size] = kern
        if idx.size == 1:
            y = signal.oaconvolve(trains[idx[0]], K[0])
        else:
            y = signal.oaconvolve(trains[idx], K, axes=-1).sum(axis=0)
        out[: y.size] += y
    return out


def render_binned(offsets, amps, keys, kernel_fn, ola: OverlapAdd | None = None):
    """Per-bin impulse trains (np.add.at) convolved with kernel_fn(bin); full tails kept.

    Returns the full-length signal (or None when there are no events) and, if `ola` is
    given, accumulates it there (offset 0 = start of the current block).
    """
    offsets = np.asarray(offsets, dtype=np.int64).ravel()
    if offsets.size == 0:
        return None
    amps = np.broadcast_to(np.asarray(amps, dtype=float), offsets.shape)
    keys = np.broadcast_to(np.asarray(keys, dtype=np.int64), offsets.shape)
    uk, inv = np.unique(keys, return_inverse=True)
    inv = np.asarray(inv).ravel()
    width = int(offsets.max()) + 1
    trains = np.zeros((uk.size, width))
    np.add.at(trains, (inv, offsets), amps)
    y = convolve_trains(trains, [kernel_fn(int(k)) for k in uk])
    if ola is not None:
        ola.add(y)
    return y


class KernelBank:
    """Cache of kernels keyed by quantized parameters (log-spaced bins).

    - drop pulses: by tau = D/V in 1/6-octave bins (shape only; scale per event by F_max)
    - Hertz pulses: by contact time, 1/6-octave bins
    - bubbles: by radius in 1/36-octave bins (~33 cents) and xi in steps of 0.025 (eps = 1)
    - water clicks: by impact speed in 0.25 m/s steps
    - splash bursts: by duration in 1/3-octave bins from 1 ms, 8 noise variants
    """

    TAU_PER_OCT = 6
    RAD_PER_OCT = 36
    CLICK_STEP = 0.25
    XI_STEP = 0.025
    DUR_PER_OCT = 3
    SPLASH_VARIANTS = 8

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = int(sr)
        self._cache: dict = {}

    def _get(self, key, make):
        k = self._cache.get(key)
        if k is None:
            k = make()
            self._cache[key] = k
        return k

    # -- drop pulses --------------------------------------------------------
    def tau_bin(self, tau_s):
        return np.round(np.log2(np.maximum(np.asarray(tau_s, dtype=float), 2e-6) / 1e-5)
                        * self.TAU_PER_OCT).astype(np.int64)

    def tau_center(self, b: int) -> float:
        return 1e-5 * 2.0 ** (b / self.TAU_PER_OCT)

    def pulse(self, b: int) -> np.ndarray:
        return self._get(("pulse", int(b)), lambda: pulse_shape(self.tau_center(b), self.sr))

    def hertz(self, b: int) -> np.ndarray:
        return self._get(("hertz", int(b)), lambda: hertz_shape(self.tau_center(b), self.sr))

    def impact(self, d_mm: float, v_mps: float) -> np.ndarray:
        dq = round(float(d_mm) * 20.0) / 20.0
        vq = round(float(v_mps) * 10.0) / 10.0
        return self._get(("impact", dq, vq), lambda: impact_pulse(dq, vq, self.sr))

    # -- bubbles ------------------------------------------------------------
    def radius_bin(self, radius_m):
        return np.round(np.log2(np.maximum(np.asarray(radius_m, dtype=float), 1e-5) / 1e-5)
                        * self.RAD_PER_OCT).astype(np.int64)

    def radius_center(self, b: int) -> float:
        return 1e-5 * 2.0 ** (b / self.RAD_PER_OCT)

    def xi_bin(self, xi):
        return np.clip(np.round(np.asarray(xi, dtype=float) / self.XI_STEP), 0, 63).astype(np.int64)

    def bubble_shape(self, rb: int, xb: int = 4, damp_bin: int = 0) -> np.ndarray:
        """Unit-eps bubble for radius bin rb, xi bin xb, extra-damping bin (0.01 steps)."""
        return self._get(("bubble", int(rb), int(xb), int(damp_bin)),
                         lambda: bubble_kernel(self.radius_center(rb), self.sr, xi=xb * self.XI_STEP,
                                               eps=1.0, damping_extra=0.01 * damp_bin))

    def bubble(self, radius_m: float, xi: float = 0.1, eps: float = 0.05,
               damping_extra: float = 0.0) -> np.ndarray:
        rb = int(self.radius_bin(radius_m))
        xb = int(self.xi_bin(xi))
        db = int(round(max(damping_extra, 0.0) / 0.01))
        return eps * self.bubble_shape(rb, xb, db)

    # -- clicks -------------------------------------------------------------
    def click_bin(self, v_mps):
        return np.maximum(np.round(np.asarray(v_mps, dtype=float) / self.CLICK_STEP), 1).astype(np.int64)

    def click_shape(self, b: int) -> np.ndarray:
        return self._get(("click", int(b)), lambda: click_kernel(b * self.CLICK_STEP, self.sr))

    def click(self, v_mps: float) -> np.ndarray:
        return self.click_shape(int(self.click_bin(v_mps)))

    # -- splashes -----------------------------------------------------------
    def dur_bin(self, dur_s):
        return np.round(np.log2(np.maximum(np.asarray(dur_s, dtype=float), 5e-4) / 1e-3)
                        * self.DUR_PER_OCT).astype(np.int64)

    def dur_center(self, b: int) -> float:
        return 1e-3 * 2.0 ** (b / self.DUR_PER_OCT)

    def splash(self, dur_bin: int, variant: int = 0, bright: bool = True) -> np.ndarray:
        hp, lp = (1500.0, 14000.0) if bright else (300.0, 6000.0)
        return self._get(("splash", int(dur_bin), int(variant) % self.SPLASH_VARIANTS, bool(bright)),
                         lambda: splash_kernel(self.dur_center(dur_bin), self.sr,
                                               int(variant) % self.SPLASH_VARIANTS, hp, lp))
