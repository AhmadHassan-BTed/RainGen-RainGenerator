"""Event timing for precipitation: a doubly stochastic (Cox) process.

Rain is Poisson only below ~0.1 s / 0.5 m; above that, counts are clustered and
scale-invariant (Lovejoy & Schertzer 2008: number fluctuations ~ l^1/2). The engine
therefore draws impacts as a Cox process: Poisson within a block at an intensity that
is itself modulated by multiplicative 1/f-type noise. The slow, storm-wide part of that
modulation lives in raingen.physics.storm; `RateModulator` adds the local (per patch /
per stream) part from ~0.1 s to tens of seconds, so neighbouring surfaces do not swell
in lock-step.
"""

from __future__ import annotations

import math

import numpy as np

from raingen.core.constants import BLOCK, SAMPLE_RATE
from raingen.core.rng import RngFactory


def ramp_positions(u, m0, m1) -> np.ndarray:
    """Inverse CDF on [0, 1) of an intensity that ramps linearly from m0 to m1 (vectorized).

    Solves m0 x + (m1 - m0) x^2 / 2 = u (m0 + m1) / 2 for x; u, m0, m1 broadcast together.
    """
    u = np.asarray(u, dtype=float)
    m0 = np.maximum(np.asarray(m0, dtype=float), 0.0)
    m1 = np.maximum(np.asarray(m1, dtype=float), 0.0)
    a = m1 - m0
    tot = 0.5 * (m0 + m1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (np.sqrt(np.maximum(m0 * m0 + 2.0 * a * u * tot, 0.0)) - m0) / a
    flat = np.abs(a) <= 1e-9 * np.maximum(tot, 1e-300)
    x = np.where(flat, u, x)
    return np.clip(x, 0.0, np.nextafter(1.0, 0.0))


def cox_offsets(rng: np.random.Generator, rate_per_s: float, n: int, sr: int = SAMPLE_RATE,
                modulation=1.0) -> np.ndarray:
    """Sorted sample offsets in [0, n) of Poisson events at rate*modulation.

    `modulation` may be a scalar or an intensity curve over the block (any length >= 2,
    linearly resampled to n); in that case event times follow the curve exactly
    (inverse-CDF sampling of the integrated intensity; closed form for a two-point ramp).
    """
    n = int(n)
    if n <= 0 or not (rate_per_s > 0.0):
        return np.zeros(0, dtype=np.int64)
    mod = np.asarray(modulation, dtype=float)
    if mod.ndim == 0:
        lam = float(rate_per_s) * max(float(mod), 0.0) * n / sr
        if lam <= 0.0:
            return np.zeros(0, dtype=np.int64)
        k = int(rng.poisson(lam))
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        return np.sort(rng.integers(0, n, size=k)).astype(np.int64)

    mod = np.maximum(mod.ravel(), 0.0)
    if mod.size == 1:
        return cox_offsets(rng, rate_per_s, n, sr, float(mod[0]))
    if mod.size == 2:
        m0, m1 = float(mod[0]), float(mod[1])
        lam = float(rate_per_s) * 0.5 * (m0 + m1) * n / sr
        if lam <= 0.0:
            return np.zeros(0, dtype=np.int64)
        k = int(rng.poisson(lam))
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        x = ramp_positions(rng.random(k), m0, m1)
        return np.sort((x * n).astype(np.int64))
    if mod.size != n:
        mod = np.interp(np.linspace(0.0, mod.size - 1.0, n), np.arange(mod.size), mod)
    cum = np.cumsum(mod)
    total = cum[-1]
    lam = float(rate_per_s) * total / sr
    if lam <= 0.0:
        return np.zeros(0, dtype=np.int64)
    k = int(rng.poisson(lam))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    u = rng.uniform(0.0, total, size=k)
    offs = np.searchsorted(cum, u, side="right")
    return np.sort(np.clip(offs, 0, n - 1)).astype(np.int64)


def thinned_offsets(rng: np.random.Generator, rate_per_s: float, n: int, sr: int, cap_per_block: float,
                    modulation=1.0) -> tuple[np.ndarray, float]:
    """Cox offsets with an event budget: above `cap_per_block` expected events, draw the cap
    and return the energy-preserving amplitude factor sqrt(expected / cap) (granular density)."""
    mod = np.asarray(modulation, dtype=float)
    mean_mod = float(mod) if mod.ndim == 0 else float(np.mean(mod))
    expected = max(float(rate_per_s), 0.0) * n / sr * mean_mod
    if cap_per_block > 0 and expected > cap_per_block:
        scale = math.sqrt(expected / cap_per_block)
        rate = rate_per_s * cap_per_block / expected
        return cox_offsets(rng, rate, n, sr, modulation), scale
    return cox_offsets(rng, rate_per_s, n, sr, modulation), 1.0


class RateModulator:
    """Mean-one multiplicative 1/f-type modulation (sum of octave-spaced Ornstein-Uhlenbeck processes).

    Equal-variance OU components with time constants tau_min * 2^k give a ~1/f spectrum between
    1/(2 pi tau_max) and 1/(2 pi tau_min); exp(sigma * s - sigma^2 / 2) keeps the mean at 1.
    Deterministic: step(b) draws from RngFactory.get(layer, b); blocks must be stepped in order.
    """

    def __init__(self, rngf: RngFactory, layer: str, sigma: float = 0.25, tau_min_s: float = 0.1,
                 tau_max_s: float = 12.0, sr: int = SAMPLE_RATE):
        self.rngf = rngf
        self.layer = layer
        self.sigma = float(sigma)
        self.sr = int(sr)
        k = max(1, int(math.floor(math.log2(max(tau_max_s, tau_min_s) / tau_min_s))) + 1)
        self.taus = tau_min_s * 2.0 ** np.arange(k)
        self.x = rngf.get(layer + "/init", 0).standard_normal(k)  # stationary start
        self.last_block = None
        self.value = self._value()
        self.prev = self.value

    def _value(self) -> float:
        s = float(np.sum(self.x)) / math.sqrt(self.x.size)
        return math.exp(self.sigma * s - 0.5 * self.sigma ** 2)

    def step(self, block_index: int, n: int = BLOCK) -> float:
        if self.last_block == block_index:
            return self.value
        dt = n / self.sr
        a = np.exp(-dt / self.taus)
        z = self.rngf.get(self.layer, int(block_index)).standard_normal(self.x.size)
        self.x = a * self.x + np.sqrt(1.0 - a * a) * z
        self.prev = self.value
        self.value = self._value()
        self.last_block = block_index
        return self.value

    def curve(self, block_index: int, n: int = BLOCK) -> np.ndarray:
        """Two-point intensity curve (previous -> current value) for cox_offsets."""
        self.step(block_index, n)
        return np.array([self.prev, self.value])


class RateModulatorBank:
    """Independent RateModulators for many streams at once (one row per stream).

    Same model as RateModulator (equal-variance octave-spaced OU processes, exp(sigma s - sigma^2/2),
    mean 1), but all rows are stepped with one draw per block from RngFactory.get(layer, block),
    so a field of patches costs one generator construction per block instead of one per patch.
    """

    def __init__(self, rngf: RngFactory, layer: str, n_rows: int, sigma: float = 0.25,
                 tau_min_s: float = 0.1, tau_max_s: float = 12.0, sr: int = SAMPLE_RATE):
        self.rngf = rngf
        self.layer = layer
        self.n_rows = int(n_rows)
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (self.n_rows,)).copy()
        self.sr = int(sr)
        k = max(1, int(math.floor(math.log2(max(tau_max_s, tau_min_s) / tau_min_s))) + 1)
        self.taus = tau_min_s * 2.0 ** np.arange(k)
        self.x = rngf.get(layer + "/init", 0).standard_normal((self.n_rows, k))
        self.last_block = None
        self._a_n = -1
        self.value = self._value()
        self.prev = self.value.copy()

    def _value(self) -> np.ndarray:
        s = self.x.sum(axis=1) / math.sqrt(self.x.shape[1])
        return np.exp(self.sigma * s - 0.5 * self.sigma ** 2)

    def step(self, block_index: int, n: int = BLOCK) -> np.ndarray:
        if self.last_block == block_index:
            return self.value
        if self._a_n != n:
            self._a = np.exp(-(n / self.sr) / self.taus)
            self._b = np.sqrt(1.0 - self._a * self._a)
            self._a_n = n
        z = self.rngf.get(self.layer, int(block_index)).standard_normal(self.x.shape)
        self.x = self._a[None, :] * self.x + self._b[None, :] * z
        self.prev = self.value
        self.value = self._value()
        self.last_block = block_index
        return self.value
