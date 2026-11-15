"""Ambisonic encoder: Emission -> SN3D/ACN spherical-harmonic bus.

Point sources are plane-wave encoded, b = Y(az, el) x. Gains are ramped linearly over each
block per `source_id`, so moving sources never click.

`spread` (0 = point, 1 = fully diffuse) does two things:
  * the coherent part is the SH expansion of a uniform spherical cap of half-angle
    alpha = spread * pi (order weights c_n(alpha), which taper the higher orders toward W
    like a max-rE window and vanish for the whole sphere);
  * a fraction d = spread^2 of the energy is rendered incoherently: each higher-order
    channel receives its own decorrelated copy through a random-phase all-pass FIR (unit
    magnitude at every DFT bin, phase taken from exponentially decaying noise so the
    response stays short; Kendall 1995, Alary, Politis & Valimaki 2017) with the energy a
    diffuse field would put there (1/(2n+1) per SN3D channel; Gerzon / Zotter & Frank 2019).
W always carries the signal unchanged, so the pressure (and loudness) of a source does not
depend on its spread. Decorrelator state is carried across blocks per source (overlap-add).

Ambisonic emissions (`is_ambisonic=True`, already (N_in, n)) are truncated or zero-padded
to the bus order and passed through.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ..core.constants import AMBI_ORDER, SAMPLE_RATE
from ..core.rng import RngFactory
from ..core.types import Emission
from .sh import acn_order_index, cap_weights, n_channels, sh_real

_DECORRELATOR_SEED = 0x5EED_DEC0_A11


def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


class DecorrelatorBank:
    """One fixed random-phase all-pass FIR per channel, applied by FFT overlap-add with carried tails."""

    def __init__(self, channels: int, sr: int = SAMPLE_RATE, length_ms: float = 20.0,
                 rngf: Optional[RngFactory] = None, layer: str = "spatial.decorrelator"):
        self.channels = int(channels)
        self.length = max(16, int(round(length_ms * 1e-3 * sr)))
        rngf = rngf if rngf is not None else RngFactory(_DECORRELATOR_SEED)
        L = self.length
        t = np.arange(L, dtype=np.float64)
        env = np.exp(-t / (0.22 * L))
        firs = np.empty((self.channels, L))
        for k in range(self.channels):
            h = rngf.get(layer, k).standard_normal(L) * env
            # Flatten the magnitude on the length-L DFT grid: |H| = 1 exactly at every bin and,
            # because the decaying envelope keeps the phase smooth between bins, ~1 in between.
            H = np.fft.rfft(h)
            H /= np.maximum(np.abs(H), 1e-12)
            if L % 2 == 0:
                H[-1] = np.sign(H[-1].real) or 1.0
            H[0] = np.sign(H[0].real) or 1.0
            ap = np.fft.irfft(H, L)
            firs[k] = ap / math.sqrt(float(np.sum(ap * ap)))
        self.firs = firs
        self._spec: dict[int, np.ndarray] = {}

    def spectrum(self, nfft: int) -> np.ndarray:
        H = self._spec.get(nfft)
        if H is None:
            H = np.fft.rfft(self.firs, nfft, axis=1)
            self._spec[nfft] = H
        return H

    def process(self, x: np.ndarray, tail: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """x (n,) -> (wet (channels, n), new_tail (channels, L-1))."""
        n = x.shape[0]
        L = self.length
        nfft = _next_pow2(n + L - 1)
        X = np.fft.rfft(x, nfft)
        y = np.fft.irfft(self.spectrum(nfft) * X[None, :], nfft, axis=1)[:, : n + L - 1]
        if tail is not None:
            y[:, : L - 1] += tail
        return y[:, :n], y[:, n:].copy()


@dataclass
class _SourceState:
    g_coh: np.ndarray
    g_dif: np.ndarray
    tail: Optional[np.ndarray] = None


def spread_gains(order: int, az: float, el: float, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Target (coherent, diffuse) per-channel gains for a source of the given spread."""
    s = min(1.0, max(0.0, float(spread)))
    y = sh_real(order, az, el)
    nidx = acn_order_index(order)
    c = cap_weights(order, s * math.pi)[nidx]
    d = s * s
    g_coh = y * c
    g_coh[1:] *= math.sqrt(max(0.0, 1.0 - d))
    if d <= 0.0:
        return g_coh, np.zeros_like(g_coh)
    dif_energy = (1.0 - c * c) / (2.0 * nidx + 1.0) + (c * y) ** 2
    g_dif = np.sqrt(d * np.maximum(dif_energy, 0.0))
    g_dif[0] = 0.0
    return g_coh, g_dif


class Encoder:
    """Encode Emissions to an order-`order` AmbiX bus with per-source smoothing and spread."""

    def __init__(self, order: int = AMBI_ORDER, sr: int = SAMPLE_RATE, rngf: Optional[RngFactory] = None,
                 decorrelator_ms: float = 20.0, max_sources: int = 4096):
        self.order = int(order)
        self.sr = int(sr)
        self.channels = n_channels(self.order)
        self.max_sources = int(max_sources)
        self._states: "OrderedDict[str, _SourceState]" = OrderedDict()
        self._dec: Optional[DecorrelatorBank] = None
        self._rngf = rngf
        self._dec_ms = float(decorrelator_ms)
        self._ramp_cache: dict[int, np.ndarray] = {}

    # -- helpers -------------------------------------------------------
    @property
    def decorrelators(self) -> DecorrelatorBank:
        if self._dec is None:
            self._dec = DecorrelatorBank(self.channels - 1, self.sr, self._dec_ms, self._rngf)
        return self._dec

    def _ramp(self, n: int) -> np.ndarray:
        r = self._ramp_cache.get(n)
        if r is None:
            r = np.arange(1, n + 1, dtype=np.float64) / n
            self._ramp_cache[n] = r
        return r

    def _state(self, key: str, g_coh: np.ndarray, g_dif: np.ndarray) -> tuple[_SourceState, bool]:
        st = self._states.get(key)
        if st is not None:
            self._states.move_to_end(key)
            return st, False
        st = _SourceState(g_coh.copy(), g_dif.copy())
        self._states[key] = st
        while len(self._states) > self.max_sources:
            self._states.popitem(last=False)
        return st, True

    def forget(self, source_id: str) -> None:
        self._states.pop(source_id, None)

    def reset(self) -> None:
        self._states.clear()

    @staticmethod
    def _mono(signal, n: int) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        if x.shape[0] == n:
            return x
        out = np.zeros(n)
        m = min(n, x.shape[0])
        out[:m] = x[:m]
        return out

    # -- main ------------------------------------------------------------
    def encode(self, em: Emission, n: int) -> np.ndarray:
        """One Emission -> (N, n) SH signals."""
        N = self.channels
        if em.is_ambisonic:
            sig = np.asarray(em.signal, dtype=np.float64)
            if sig.ndim == 1:
                sig = sig[None, :]
            out = np.zeros((N, n))
            c = min(N, sig.shape[0])
            m = min(n, sig.shape[1])
            out[:c, :m] = sig[:c, :m]
            return out

        x = self._mono(em.signal, n)
        g_coh, g_dif = spread_gains(self.order, em.azimuth, em.elevation, em.spread)
        # Anonymous emissions are keyed by where they are, so several static anonymous sources
        # in one stem never share (and corrupt) one ramp/decorrelator state.
        key = em.source_id or (f"__anon__:{em.stem}:{float(em.azimuth):.3f}:"
                               f"{float(em.elevation):.3f}:{float(em.spread):.2f}")
        st, fresh = self._state(key, g_coh, g_dif)
        r = self._ramp(n)

        p_coh = st.g_coh
        if fresh or np.array_equal(p_coh, g_coh):
            out = g_coh[:, None] * x[None, :]
        else:
            out = (p_coh[:, None] + (g_coh - p_coh)[:, None] * r[None, :]) * x[None, :]

        p_dif = st.g_dif
        need_dif = bool(np.any(g_dif > 0.0) or np.any(p_dif > 0.0))
        if need_dif:
            wet, st.tail = self.decorrelators.process(x, st.tail)
            if fresh or np.array_equal(p_dif, g_dif):
                out[1:] += g_dif[1:, None] * wet
            else:
                out[1:] += (p_dif[1:, None] + (g_dif - p_dif)[1:, None] * r[None, :]) * wet
        else:
            st.tail = None

        st.g_coh = g_coh
        st.g_dif = g_dif
        return out

    def encode_block(self, emissions: Iterable[Emission], n: int,
                     stems: Optional[Iterable[str]] = None) -> dict[str, np.ndarray]:
        """Encode and sum a block of emissions per stem -> {stem: (N, n)}."""
        buses: dict[str, np.ndarray] = {}
        if stems is not None:
            for s in stems:
                buses[s] = np.zeros((self.channels, n))
        for em in emissions:
            b = self.encode(em, n)
            acc = buses.get(em.stem)
            if acc is None:
                buses[em.stem] = b
            else:
                acc += b
        return buses
