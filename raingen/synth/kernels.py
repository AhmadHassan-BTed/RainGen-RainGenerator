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
  the force lasts about the time the stone takes to travel its own diameter, D/v, with
  impulse m v (1 + e), i.e. a peak of ~2.6 m v^2 / D at e = 0.5 (instrumented ice-sphere
  impacts: Kim & Kedward 2000; Tippmann, Kim & Rhymer 2013, peak ~2-2.6 m v^2 / D).
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

Batched rendering (the real-time path, MASTER_PLAN 9.5): kernels are addressed by one
int64 *code* (family << 40 | parameter bin), their spectra are cached per FFT size, and
`render_trains` convolves many (row, code) impulse or noise trains in one batched FFT.
`ModalBatch` runs one modal bank over many rows at once (same exact block recursion as
ModalBank), and `RowOLA` keeps every row's convolution tails across blocks.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft
from scipy import signal

from raingen.core.constants import P0, RHO_WATER, SAMPLE_RATE
from raingen.core.rng import RngFactory

from .materials import Material, get_material

_LN1000 = math.log(1000.0)          # 60 dB amplitude decay: r^(T60 sr) = 1e-3
_PULSE_SIGMA = 0.8                  # log-normal tail width of the drop force pulse
_PULSE_SPAN = 4.5                   # pulse kernel length in units of tau = D/V (log-normal tail < -70 dB)
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
        tau_c = np.maximum(tau, d / v)                  # crushing lasts the stone's own transit time
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
        np.copyto(self.zi, 0.0, where=np.abs(self.zi) < 1e-30)        # decaying states never go denormal
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
        self._nfft = fast_len(2 * n - 1)
        self._H = sfft.rfft(h, self._nfft)
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
            y += sfft.irfft(sfft.rfft(x, self._nfft) * self._H, self._nfft)[:n]
        if has_state:
            y += ((self.g * self.state) @ self._P).imag
        new_state = self._P[:, n - 1] * self.state if has_state else np.zeros_like(self.state)
        if has_in:
            new_state = new_state + self._P[:, : n - 1] @ x[n - 2::-1] + x[n - 1]
        np.copyto(new_state, 0.0, where=np.abs(new_state) < 1e-30)    # decayed modes never go denormal
        self.state = new_state
        return y


# ---------------------------------------------------------------------------
# batched rendering of event streams

_LEN_BUCKETS = (0, 96, 384, 1536, 6144, 24576, 1 << 62)
_FAST_LEN: dict = {}


def fast_len(m: int) -> int:
    """Cached scipy.fft.next_fast_len for real transforms."""
    v = _FAST_LEN.get(m)
    if v is None:
        v = int(sfft.next_fast_len(int(m), real=True))
        _FAST_LEN[m] = v
    return v


def convolve_trains(trains: np.ndarray, kernels: list) -> np.ndarray:
    """sum_b trains[b] * kernels[b] (full convolution), one batched FFT per kernel-length bucket."""
    trains = np.atleast_2d(np.asarray(trains, dtype=float))
    m = trains.shape[1]
    lens = np.array([max(len(k), 1) for k in kernels])
    out = np.zeros(m + int(lens.max()) - 1)
    for lo, hi in zip(_LEN_BUCKETS[:-1], _LEN_BUCKETS[1:]):
        idx = np.nonzero((lens > lo) & (lens <= hi))[0]
        if idx.size == 0:
            continue
        L = int(lens[idx].max())
        if idx.size == 1 and min(m, L) <= 64:
            y = np.convolve(trains[idx[0]], np.asarray(kernels[idx[0]], dtype=float))
            out[: y.size] += y
            continue
        nfft = fast_len(m + L - 1)
        K = np.zeros((idx.size, L))
        for j, i in enumerate(idx):
            kern = np.asarray(kernels[i], dtype=float)
            K[j, : kern.size] = kern
        spec = (sfft.rfft(trains[idx], nfft, axis=-1) * sfft.rfft(K, nfft, axis=-1)).sum(axis=0)
        y = sfft.irfft(spec, nfft)[: m + L - 1]
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
    kers = [np.asarray(kernel_fn(int(k)), dtype=float).ravel() for k in uk]
    lens = np.array([max(k.size, 1) for k in kers], dtype=np.int64)
    ev_len = lens[inv]
    if int(ev_len.sum()) <= 4 * uk.size * fast_len(width + int(lens.max()) - 1):
        # few events per kernel: expand them in the time domain (cost = their total length)
        flat = np.concatenate([k if k.size else np.zeros(1) for k in kers])
        k_off = np.cumsum(lens) - lens
        tot = int(ev_len.sum())
        local = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(ev_len) - ev_len, ev_len)
        y = np.bincount(np.repeat(offsets, ev_len) + local,
                        weights=np.repeat(amps, ev_len) * flat[np.repeat(k_off[inv], ev_len) + local],
                        minlength=width + int(lens.max()) - 1)
        if ola is not None:
            ola.add(y)
        return y
    trains = np.bincount(inv * width + offsets, weights=amps, minlength=uk.size * width).reshape(uk.size, width)
    y = convolve_trains(trains, kers)
    if ola is not None:
        ola.add(y)
    return y


FAM_PULSE, FAM_HERTZ, FAM_CLICK, FAM_BUBBLE, FAM_SPLASH, FAM_COLOR = 1, 2, 3, 4, 5, 6
_FAM_SHIFT = 40
_SUB_MASK = (1 << _FAM_SHIFT) - 1
_BIN_OFF = 256                       # keeps negative log-bins positive inside a code
_ROW_SHIFT = 44                      # (row << 44) | code is unique per (row, kernel)


class KernelBank:
    """Cache of kernels keyed by quantized parameters (log-spaced bins).

    - drop pulses: by tau = D/V in 1/4-octave bins (shape only; scale per event by F_max)
    - Hertz pulses: by contact time, 1/4-octave bins
    - bubbles: by radius in 1/36-octave bins (~33 cents) and xi in steps of 0.025 (eps = 1)
    - water clicks: by impact speed in 0.25 m/s steps
    - splash bursts: by duration in 1/3-octave bins from 1 ms, 8 noise variants

    Every kernel also has an int64 *code* (`pulse_code`, `click_code`, ...) for the batched
    renderer; `kernel(code)` returns it and `spectrum(code, nfft)` its cached rfft.
    """

    TAU_PER_OCT = 4
    RAD_PER_OCT = 36
    CLICK_STEP = 0.25
    XI_STEP = 0.025
    DUR_PER_OCT = 3
    SPLASH_VARIANTS = 8
    MAX_SPECTRA = 4096          # cached spectra before the cache is flushed

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = int(sr)
        self._cache: dict = {}
        self._spec: dict = {}
        self._len: dict = {}
        self._colors: list = []
        self._color_ids: dict = {}
        self._span: dict = {}               # code -> (offset, length) in the flat kernel table
        self._flat = np.zeros(1 << 16)
        self._flat_n = 0

    def _get(self, key, make):
        k = self._cache.get(key)
        if k is None:
            k = make()
            self._cache[key] = k
        return k

    # -- int64 kernel codes for the batched renderer ---------------------------
    def kernel(self, code: int) -> np.ndarray:
        fam, sub = int(code) >> _FAM_SHIFT, int(code) & _SUB_MASK
        if fam == FAM_PULSE:
            return self.pulse(sub - _BIN_OFF)
        if fam == FAM_HERTZ:
            return self.hertz(sub - _BIN_OFF)
        if fam == FAM_CLICK:
            return self.click_shape(sub)
        if fam == FAM_BUBBLE:
            return self.bubble_shape(sub // 512, (sub // 8) % 64, sub % 8)
        if fam == FAM_SPLASH:
            return self.splash((sub // 32) - _BIN_OFF, (sub // 2) % 16, bool(sub % 2))
        if fam == FAM_COLOR:
            return self._colors[sub]
        raise ValueError(f"unknown kernel family {fam}")

    def register(self, key, make) -> int:
        """Code of a derived kernel (e.g. a noise-colouring filter), built once by make() and cached by key."""
        code = self._color_ids.get(key)
        if code is None:
            self._colors.append(np.asarray(make(), dtype=float))
            code = (FAM_COLOR << _FAM_SHIFT) | (len(self._colors) - 1)
            self._color_ids[key] = code
        return code

    def spans(self, codes) -> tuple:
        """(offsets, lengths) of kernels `codes` in the flat table `self.flat` (appended on first use)."""
        codes = np.asarray(codes, dtype=np.int64).ravel()
        off = np.empty(codes.size, dtype=np.int64)
        ln = np.empty(codes.size, dtype=np.int64)
        for i, c in enumerate(codes.tolist()):
            sp = self._span.get(c)
            if sp is None:
                k = np.asarray(self.kernel(c), dtype=float)
                if k.size == 0:
                    k = np.zeros(1)
                end = self._flat_n + k.size
                if end > self._flat.size:
                    nb = np.zeros(max(2 * self._flat.size, end))
                    nb[: self._flat_n] = self._flat[: self._flat_n]
                    self._flat = nb
                self._flat[self._flat_n:end] = k
                sp = (self._flat_n, k.size)
                self._flat_n = end
                self._span[c] = sp
            off[i], ln[i] = sp
        return off, ln

    @property
    def flat(self) -> np.ndarray:
        """Every kernel used by `scatter_kernels`, back to back (see `spans`)."""
        return self._flat

    def kernel_len(self, code: int) -> int:
        n = self._len.get(code)
        if n is None:
            n = max(1, len(self.kernel(code)))
            self._len[code] = n
        return n

    def spectrum(self, code: int, nfft: int) -> np.ndarray:
        key = (int(code), int(nfft))
        s = self._spec.get(key)
        if s is None:
            if len(self._spec) >= self.MAX_SPECTRA:
                self._spec.clear()
            s = sfft.rfft(self.kernel(code), int(nfft))
            self._spec[key] = s
        return s

    def pooled_power(self, code: int, half_len: int) -> tuple:
        """(power spectrum on the rfft grid of size 2*half_len, energy) of a kernel, energy-preserving.

        The kernel's |rfft|^2 is taken on a fine grid (a power-of-two multiple of 2*half_len that holds
        the whole kernel) and averaged in groups of bins, so long ringing kernels keep their band energies.
        """
        key = ("pool", int(code), int(half_len))
        v = self._cache.get(key)
        if v is None:
            k = self.kernel(code)
            na = 2 * int(half_len)
            mult = 1
            while na * mult < len(k):
                mult *= 2
            pw = np.abs(sfft.rfft(k, na * mult)) ** 2
            half = mult // 2
            ext = np.concatenate([pw[1:half + 1][::-1], pw, pw[-1 - half:-1][::-1]])   # mirrored ends
            nb = na // 2 + 1
            pooled = ext[: nb * mult].reshape(nb, mult).mean(axis=1)
            v = (pooled, float(np.sum(k * k)))
            self._cache[key] = v
        return v

    def pulse_code(self, tau_s, family: int = 1):
        """Codes of drop (family FAM_PULSE) or Hertz (FAM_HERTZ) force pulses for durations tau_s."""
        return (np.int64(family) << _FAM_SHIFT) | (self.tau_bin(tau_s) + _BIN_OFF)

    def click_code(self, v_mps):
        return (np.int64(FAM_CLICK) << _FAM_SHIFT) | self.click_bin(v_mps)

    def bubble_code(self, radius_m, xi, damp_bin):
        sub = (self.radius_bin(radius_m) * 64 + self.xi_bin(xi)) * 8 + np.clip(damp_bin, 0, 7)
        return (np.int64(FAM_BUBBLE) << _FAM_SHIFT) | np.asarray(sub, dtype=np.int64)

    def splash_code(self, dur_s, variant, bright: bool = True):
        sub = ((self.dur_bin(dur_s) + _BIN_OFF) * 16 + np.asarray(variant, dtype=np.int64) % self.SPLASH_VARIANTS) * 2
        return (np.int64(FAM_SPLASH) << _FAM_SHIFT) | (sub + int(bool(bright)))

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


# ---------------------------------------------------------------------------
# batched rendering: many (row, kernel) trains in one FFT, row-wise overlap-add,
# one modal bank over many rows, and a deterministic white-noise table

_NOISE_BITS = 20                     # 2^20 samples (8 MB): ~400 rows of 2 x 1024-sample frames per block
_NOISE_EXT = 8192
_NOISE: np.ndarray | None = None


def noise_table() -> np.ndarray:
    """Deterministic unit-variance Gaussian white noise, 2^20 samples plus a wrapped extension."""
    global _NOISE
    if _NOISE is None:
        base = RngFactory(0x6E6F6973).get("kernels/noise_table", 0).standard_normal(1 << _NOISE_BITS)
        base -= base.mean()
        base /= base.std()
        _NOISE = np.concatenate([base, base[:_NOISE_EXT]])
    return _NOISE


def noise_rows(offsets, n: int) -> np.ndarray:
    """(k, n) white-noise segments starting at table positions `offsets` (any ints)."""
    tab = noise_table()
    size = 1 << _NOISE_BITS
    off = np.asarray(offsets, dtype=np.int64) & (size - 1)
    if n <= _NOISE_EXT:
        view = np.lib.stride_tricks.sliding_window_view(tab, n)
        return view[off]
    idx = (off[:, None] + np.arange(n, dtype=np.int64)[None, :]) & (size - 1)
    return tab[idx]


_LEN_EDGES = (64, 256, 1024, 4096, 16384, 1 << 62)


def render_trains(kb: KernelBank, n: int, rows=None, codes=None, offs=None, amps=None,
                  dense_rows=None, dense_codes=None, dense=None) -> list:
    """Batched sum of kernel trains per row.

    Events: amps[i] * kernel(codes[i]) placed at offs[i] in [0, n) of row rows[i].
    Dense trains: dense[j] (an (n,) noise train, already scaled) convolved with kernel(dense_codes[j])
    into row dense_rows[j]. Trains sharing a (row, code) are summed first, then every
    kernel-length bucket is convolved with one batched rfft / irfft using the cached kernel
    spectra. Returns [(rows_b, Y_b)], Y_b of shape (len(rows_b), n + L_b - 1), rows unique per item.
    """
    parts = []
    have_ev = rows is not None and np.size(rows) > 0
    have_dn = dense_rows is not None and np.size(dense_rows) > 0
    if not have_ev and not have_dn:
        return []
    if have_ev:
        rows = np.asarray(rows, dtype=np.int64).ravel()
        codes = np.asarray(codes, dtype=np.int64).ravel()
        parts.append((rows << _ROW_SHIFT) | codes)
    if have_dn:
        dense_rows = np.asarray(dense_rows, dtype=np.int64).ravel()
        dense_codes = np.asarray(dense_codes, dtype=np.int64).ravel()
        parts.append((dense_rows << _ROW_SHIFT) | dense_codes)
    pid = parts[0] if len(parts) == 1 else np.concatenate(parts)
    uniq, inv = np.unique(pid, return_inverse=True)
    inv = inv.ravel()
    u = uniq.size
    if have_ev:
        m = rows.size
        offs = np.clip(np.asarray(offs, dtype=np.int64).ravel(), 0, n - 1)
        trains = np.bincount(inv[:m] * n + offs, weights=np.broadcast_to(np.asarray(amps, dtype=float), (m,)),
                             minlength=u * n).reshape(u, n)
        inv_d = inv[m:]
    else:
        trains = np.zeros((u, n))
        inv_d = inv
    if have_dn:
        np.add.at(trains, inv_d, np.asarray(dense, dtype=float).reshape(-1, n))
    rows_u = uniq >> _ROW_SHIFT
    codes_u = uniq & ((1 << _ROW_SHIFT) - 1)
    uc, ui = np.unique(codes_u, return_inverse=True)
    ui = ui.ravel()
    lens_c = np.array([kb.kernel_len(int(c)) for c in uc], dtype=np.int64)
    lens = lens_c[ui]
    out = []
    lo = 0
    for hi in _LEN_EDGES:
        sel = np.nonzero((lens > lo) & (lens <= hi))[0]
        lo = hi
        if sel.size == 0:
            continue
        L = int(lens[sel].max())
        nfft = fast_len(n + L - 1)
        cs = np.unique(ui[sel])
        spec = {int(c): kb.spectrum(int(uc[c]), nfft) for c in cs}
        K = np.stack([spec[int(c)] for c in ui[sel]])
        X = sfft.rfft(trains[sel], nfft, axis=-1)
        X *= K
        r = rows_u[sel]
        starts = np.concatenate(([0], np.nonzero(np.diff(r))[0] + 1))
        if starts.size < sel.size:
            X = np.add.reduceat(X, starts, axis=0)
        y = sfft.irfft(X, nfft, axis=-1)[:, : n + L - 1]
        out.append((r[starts], y))
    return out


class RowOLA:
    """Overlap-add accumulator for many rows (tails of any length carried across blocks)."""

    def __init__(self, n_rows: int, capacity: int = 16384):
        self.n_rows = int(n_rows)
        self.buf = np.zeros((self.n_rows, max(int(capacity), 1024)))
        self.head = 0
        self.end = np.zeros(self.n_rows, dtype=np.int64)     # nonzero extent past head, per row

    def active_rows(self) -> np.ndarray:
        return self.end > 0

    @property
    def active(self) -> bool:
        return bool(np.any(self.end > 0))

    def _ensure(self, length: int):
        cap = self.buf.shape[1]
        if self.head + length <= cap:
            return
        keep = int(self.end.max()) if self.end.size else 0
        need = max(length, keep)
        if need > cap // 2:
            nb = np.zeros((self.n_rows, max(2 * need, cap)))
            nb[:, :keep] = self.buf[:, self.head:self.head + keep]
            self.buf = nb
        else:
            self.buf[:, :keep] = self.buf[:, self.head:self.head + keep]
            self.buf[:, keep:self.head + keep] = 0.0
        self.head = 0

    def add(self, rows, y: np.ndarray, offset: int = 0):
        """Add y (len(rows), L) to the given (unique) rows, starting `offset` samples after the head."""
        rows = np.asarray(rows, dtype=np.int64)
        length = int(offset) + y.shape[-1]
        if rows.size == 0 or y.shape[-1] == 0:
            return
        self._ensure(length)
        h = self.head + int(offset)
        if rows.size == self.n_rows and rows[0] == 0 and rows[-1] == self.n_rows - 1:
            self.buf[:, h:h + y.shape[-1]] += y          # every row (sorted, unique): a slice, no gather
            np.maximum(self.end, length, out=self.end)
            return
        self.buf[rows, h:h + y.shape[-1]] += y
        self.end[rows] = np.maximum(self.end[rows], length)

    def pop(self, n: int) -> np.ndarray:
        """The next n samples of every row (n_rows, n); the buffer advances by n."""
        self._ensure(n)
        h = self.head
        out = self.buf[:, h:h + n].copy()
        self.buf[:, h:h + n] = 0.0
        self.head = h + n
        self.end = np.maximum(self.end - n, 0)
        return out


class ModalBatch:
    """One ModalBank driven by many independent rows (each row keeps its own modal state).

    Same exact block recursion as ModalBank.process, batched: per block one rfft / irfft of all
    rows against the bank's truncated impulse response, plus two complex matrix products for the
    ringing carried in from (and into) neighbouring blocks.
    """

    def __init__(self, bank: ModalBank, n_rows: int):
        self.bank = bank
        self.n_rows = int(n_rows)
        self.state = np.zeros((self.n_rows, bank.n_modes), dtype=complex)
        self._n = -1
        self._P = self._R = self._H = None
        self._nfft = 0

    def reset(self):
        self.state[:] = 0.0

    def _prepare(self, n: int):
        if self._n == n:
            return
        b = self.bank
        k = np.arange(1, n + 1, dtype=float)
        P = np.exp(b.logp[:, None] * k[None, :])                    # p^k, k = 1..n   (M, n)
        h = np.zeros(n)
        if n > 1:
            h[1:] = b.g @ P[:, : n - 1].imag
        self._nfft = fast_len(2 * n - 1)
        self._H = sfft.rfft(h, self._nfft)
        self._P = P * b.g[:, None]                                   # gain folded in
        j = np.arange(n - 1, -1, -1, dtype=float)
        self._R = np.exp(b.logp[None, :] * j[:, None])               # p^(n-1-j)  (n, M)
        self._pn = P[:, n - 1]
        self._n = n

    def silent(self, eps: float = 1e-12) -> bool:
        if self.bank.n_modes == 0:
            return True
        s = np.abs(self.state) * np.abs(self.bank.g)[None, :]
        if float(s.max(initial=0.0)) < eps:
            self.state[:] = 0.0
            return True
        return False

    def fir(self, n: int) -> tuple:
        """(nfft, H): FFT size and spectrum of the bank's response within one block of n samples, so a
        caller can convolve the rows of several banks in one batched FFT (see `process(fir=...)`)."""
        if self.bank.n_modes == 0:
            return fast_len(2 * n - 1), None
        self._prepare(n)
        return self._nfft, self._H

    def process(self, X: np.ndarray, has_input: bool | None = None, fir: np.ndarray | None = None) -> np.ndarray:
        """X (n_rows, n) force -> (n_rows, n) modal response.

        `fir` (optional): the in-block convolution part irfft(rfft(X) H)[:, :n] computed by the caller."""
        X = np.asarray(X, dtype=float)
        n = X.shape[-1]
        if self.bank.n_modes == 0 or n == 0:
            return np.zeros_like(X)
        self._prepare(n)
        has_in = bool(np.any(X)) if has_input is None else bool(has_input)
        has_state = bool(np.any(self.state))
        Y = np.zeros(X.shape)
        if has_in:
            if fir is not None:
                Y += fir
            else:
                Y += sfft.irfft(sfft.rfft(X, self._nfft, axis=-1) * self._H, self._nfft, axis=-1)[:, :n]
        if has_state:
            Y += (self.state @ self._P).imag
            self.state = self.state * self._pn[None, :]
        if has_in:
            self.state = self.state + X @ self._R
        if has_state or has_in:
            np.copyto(self.state, 0.0, where=np.abs(self.state) < 1e-30)  # decayed modes never go denormal
        return Y


def scatter_kernels(kb: KernelBank, n: int, rows, codes, offs, amps):
    """Time-domain sum of amps[i] * kernel(codes[i]) at offs[i] in row rows[i] (no FFT).

    Cost is the total kernel length of the events, so it suits short kernels and sparse events; every
    event is expanded in one vectorized pass over the bank's flat kernel table. Returns (rows_used, Y):
    Y of shape (len(rows_used), n + longest kernel - 1); offsets are clipped to [0, n).
    """
    rows = np.asarray(rows, dtype=np.int64).ravel()
    m = rows.size
    codes = np.broadcast_to(np.asarray(codes, dtype=np.int64), (m,))
    offs = np.clip(np.broadcast_to(np.asarray(offs, dtype=np.int64), (m,)), 0, n - 1)
    amps = np.broadcast_to(np.asarray(amps, dtype=float), (m,))
    ur, rows = np.unique(rows, return_inverse=True)
    rows = rows.ravel()
    uc, inv = np.unique(codes, return_inverse=True)
    inv = inv.ravel()
    k_off, k_len = kb.spans(uc)
    W = n + int(k_len.max()) - 1
    L = k_len[inv]
    tot = int(L.sum())
    start = np.cumsum(L) - L
    local = np.arange(tot, dtype=np.int64) - np.repeat(start, L)
    idx = np.repeat(rows * W + offs, L) + local
    wts = np.repeat(amps, L) * kb.flat[np.repeat(k_off[inv], L) + local]
    Y = np.bincount(idx, weights=wts, minlength=ur.size * W).reshape(ur.size, W)
    return ur, Y
