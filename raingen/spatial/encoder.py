"""Ambisonic encoder: Emission -> SN3D/ACN spherical-harmonic bus.

Point sources are plane-wave encoded, b = Y(az, el) x. Gains are ramped linearly over each
block per `source_id`, so moving sources never click; a source whose direction and spread
did not change reuses its gains (no SH evaluation, no ramp).

`spread` (0 = point, 1 = fully diffuse) does two things:
  * the coherent part is the SH expansion of a uniform spherical cap of half-angle
    alpha = spread * pi (order weights c_n(alpha), which taper the higher orders toward W
    like a max-rE window and vanish for the whole sphere);
  * a fraction d = spread^2 of the energy is rendered incoherently: each higher-order
    channel receives its own decorrelated copy through a random-phase all-pass FIR
    (Kendall 1995) with the energy a diffuse field would put there (1/(2n+1) per SN3D
    channel; Gerzon / Zotter & Frank 2019).
W always carries the signal unchanged, so the pressure (and loudness) of a source does not
depend on its spread. The gains of the decorrelated part are applied *before* the
decorrelators, so the decorrelation is one linear time-invariant filter per channel: per
source in `encode`, or once per stem for all sources in `encode_into` / `finish_block`
(identical output by linearity, ~20x cheaper). Decorrelator tails are carried across blocks
by overlap-add and flushed when a source stops being diffuse.

Ambisonic emissions (`is_ambisonic=True`, already (N_in, n)) are truncated or zero-padded
to the bus order and passed through.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np

from ..core.constants import AMBI_ORDER, SAMPLE_RATE
from ..core.rng import RngFactory
from ..core.types import Emission
from .sh import acn_order_index, cap_weights, n_channels, sh_real

_DECORRELATOR_SEED = 0x5EED_DEC0_A11
_ALLPASS_ITERS = 120          # L-BFGS iterations: magnitude ripple ~ +-0.05 dB at every frequency


def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


def design_allpass_firs(noise: np.ndarray, iters: int = _ALLPASS_ITERS, decay: float = 0.25,
                        orthogonal: bool = True) -> np.ndarray:
    """Random-phase FIRs (C, L) whose magnitude is flat at *every* frequency (true all-pass to ~0.05 dB).

    Start: exponentially decaying noise (envelope time constant decay * L, which sets the ~6 ms
    smear and the random phase), magnitude-normalised on the 2L-point DFT grid. Then minimise
    sum_k (|H_k|^2 - 1)^2 on that grid with L-BFGS. On M >= 2L - 1 points |H|^2 fixes the whole
    aperiodic autocorrelation, so a small residual there means the autocorrelation is a delta
    and |H(f)| ~ 1 between the grid points too (flattening only on the L-point grid, as a
    circular design does, leaves notches of -40 dB or more in the linear convolution).
    `orthogonal` adds the zero-lag cross-correlations h_i . h_j and h_i . delta = h_i[0] (with
    the dry W path), weighted like autocorrelation sidelobes, to the cost, so white noise
    through the bank gives exactly uncorrelated channels (the diagonal SN3D diffuse-field
    covariance) instead of chance correlations of ~0.15.
    """
    from scipy.optimize import minimize

    noise = np.atleast_2d(np.asarray(noise, dtype=np.float64))
    C, L = noise.shape
    M = 2 * L
    t = np.arange(L, dtype=np.float64)
    h0 = noise * np.exp(-t / (decay * L))[None, :]
    H = np.fft.rfft(h0, M, axis=1)
    H /= np.maximum(np.abs(H), 1e-12)
    h0 = np.fft.irfft(H, M, axis=1)[:, :L]
    wts = np.full(M // 2 + 1, 2.0)                  # rfft bins -> full-spectrum sum
    wts[0] = 1.0
    wts[-1] = 1.0

    def cost(v):
        h = v.reshape(C, L)
        Hv = np.fft.rfft(h, M, axis=1)
        e = Hv.real * Hv.real + Hv.imag * Hv.imag - 1.0
        val = float(np.sum(wts * e * e))
        grad = np.fft.irfft(e * Hv, M, axis=1)[:, :L] * (4.0 * M)
        if orthogonal:
            X = h @ h.T
            np.fill_diagonal(X, 0.0)
            val += float(M * (np.sum(X * X) + 2.0 * np.sum(h[:, 0] ** 2)))
            grad += (4.0 * M) * (X @ h)
            grad[:, 0] += (4.0 * M) * h[:, 0]
        return val, grad.ravel()

    res = minimize(cost, h0.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=int(iters), maxcor=20, gtol=0.0, ftol=0.0))
    h = res.x.reshape(C, L)
    return h / np.sqrt(np.sum(h * h, axis=1, keepdims=True))


@lru_cache(maxsize=8)
def _allpass_bank(channels: int, length: int, root_seed: int, layer: str) -> np.ndarray:
    rngf = RngFactory(root_seed)
    noise = np.stack([rngf.get(layer, k).standard_normal(length) for k in range(channels)])
    firs = design_allpass_firs(noise)
    firs.setflags(write=False)
    return firs


class DecorrelatorBank:
    """One fixed random-phase all-pass FIR per channel, applied by FFT overlap-add with carried tails.

    The filters are designed once per (channels, length, seed, layer) and shared process-wide.
    """

    def __init__(self, channels: int, sr: int = SAMPLE_RATE, length_ms: float = 20.0,
                 rngf: Optional[RngFactory] = None, layer: str = "spatial.decorrelator"):
        self.channels = int(channels)
        self.length = max(16, int(round(length_ms * 1e-3 * sr)))
        seed = int(rngf.root_seed) if rngf is not None else _DECORRELATOR_SEED
        self.firs = _allpass_bank(self.channels, self.length, seed, str(layer))
        self._spec: dict[int, np.ndarray] = {}

    def spectrum(self, nfft: int) -> np.ndarray:
        H = self._spec.get(nfft)
        if H is None:
            H = np.fft.rfft(self.firs, nfft, axis=1)
            self._spec[nfft] = H
        return H

    def _finish(self, Y: np.ndarray, n: int, nfft: int, tail: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        L = self.length
        y = np.fft.irfft(Y, nfft, axis=1)[:, : n + L - 1]
        if tail is not None:
            y[:, : L - 1] += tail
        return y[:, :n], y[:, n:].copy()

    def process(self, x: np.ndarray, tail: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """x (n,) into every channel -> (wet (channels, n), new_tail (channels, L-1))."""
        n = x.shape[0]
        nfft = _next_pow2(n + self.length - 1)
        X = np.fft.rfft(x, nfft)
        return self._finish(self.spectrum(nfft) * X[None, :], n, nfft, tail)

    def process_gains(self, x: np.ndarray, g0: np.ndarray, dg: Optional[np.ndarray], ramp: Optional[np.ndarray],
                      tail: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Channel k filters (g0_k + dg_k ramp(t)) x(t): the gains act before the all-pass."""
        n = x.shape[0]
        nfft = _next_pow2(n + self.length - 1)
        X = np.fft.rfft(x, nfft)
        S = g0[:, None] * X[None, :]
        if dg is not None:
            S += dg[:, None] * np.fft.rfft(x * ramp, nfft)[None, :]
        return self._finish(self.spectrum(nfft) * S, n, nfft, tail)

    def process_multi(self, xs: np.ndarray, tail: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """xs (channels, n), one input per channel -> (wet (channels, n), new_tail)."""
        n = xs.shape[1]
        nfft = _next_pow2(n + self.length - 1)
        X = np.fft.rfft(xs, nfft, axis=1)
        return self._finish(self.spectrum(nfft) * X, n, nfft, tail)


def _flush_tail(dst: np.ndarray, tail: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Add a decorrelator tail with no new input into dst (C, n); return what is left of it."""
    if tail is None:
        return None
    n = dst.shape[1]
    m = min(n, tail.shape[1])
    dst[:, :m] += tail[:, :m]
    return tail[:, n:].copy() if tail.shape[1] > n else None


@dataclass
class _SourceState:
    g_coh: np.ndarray
    g_dif: np.ndarray
    pos: tuple = ()
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
    """Encode Emissions to an order-`order` AmbiX bus with per-source smoothing and spread.

    Two ways in:
      * `encode(em, n)` -> (N, n) for one emission (decorrelated per source);
      * `encode_into(buses, stem, em, n)` for every emission of a block, then
        `finish_block(buses, n)` once: all sources of a stem are summed with BLAS and share
        one decorrelator pass. `encode_block` wraps the two. The buses are complete only
        after `finish_block`.
    """

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
        self._pending: dict[str, list] = {}
        self._stem_tails: dict[str, np.ndarray] = {}
        self._block_keys: dict[str, int] = {}

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
            r.setflags(write=False)
            self._ramp_cache[n] = r
        return r

    def forget(self, source_id: str) -> None:
        self._states.pop(source_id, None)

    def reset(self) -> None:
        self._states.clear()
        self._pending.clear()
        self._stem_tails.clear()
        self._block_keys.clear()

    @staticmethod
    def _mono(signal, n: int) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        if x.shape[0] == n:
            return x
        out = np.zeros(n)
        m = min(n, x.shape[0])
        out[:m] = x[:m]
        return out

    def _ambisonic(self, signal, n: int) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float64)
        if sig.ndim == 1:
            sig = sig[None, :]
        out = np.zeros((self.channels, n))
        c = min(self.channels, sig.shape[0])
        m = min(n, sig.shape[1])
        out[:c, :m] = sig[:c, :m]
        return out

    def _gains(self, em: Emission, batched: bool = False):
        """-> (state, start coh, end coh, start dif, end dif, moving). Static sources reuse their gains."""
        pos = (float(em.azimuth), float(em.elevation), float(em.spread))
        # Anonymous emissions are keyed by where they are, so several static anonymous sources
        # in one stem never share (and corrupt) one ramp/decorrelator state.
        key = em.source_id or f"__anon__:{em.stem}:{pos[0]:.3f}:{pos[1]:.3f}:{pos[2]:.2f}"
        if batched:
            # one id emitted twice in a block (e.g. two voices of one generator) gets one state per
            # occurrence: sharing would ramp the gains between the two positions every block
            k = self._block_keys.get(key, 0)
            self._block_keys[key] = k + 1
            if k:
                key = f"{key}#{k}"
        st = self._states.get(key)
        if st is not None:
            self._states.move_to_end(key)
            if st.pos == pos:
                return st, st.g_coh, st.g_coh, st.g_dif, st.g_dif, False
            g_coh, g_dif = spread_gains(self.order, pos[0], pos[1], pos[2])
            p_coh, p_dif = st.g_coh, st.g_dif
            st.g_coh, st.g_dif, st.pos = g_coh, g_dif, pos
            return st, p_coh, g_coh, p_dif, g_dif, True
        g_coh, g_dif = spread_gains(self.order, pos[0], pos[1], pos[2])
        st = _SourceState(g_coh, g_dif, pos)
        self._states[key] = st
        while len(self._states) > self.max_sources:
            self._states.popitem(last=False)
        return st, g_coh, g_coh, g_dif, g_dif, False

    # -- one emission ------------------------------------------------------
    def encode(self, em: Emission, n: int) -> np.ndarray:
        """One Emission -> (N, n) SH signals."""
        if em.is_ambisonic:
            return self._ambisonic(em.signal, n)
        x = self._mono(em.signal, n)
        st, p_coh, g_coh, p_dif, g_dif, moving = self._gains(em)
        if moving:
            r = self._ramp(n)
            out = np.multiply.outer(p_coh, x)
            out += np.multiply.outer(g_coh - p_coh, x * r)
        else:
            r = None
            out = np.multiply.outer(g_coh, x)
        if g_dif[1:].any() or p_dif[1:].any():
            dg = (g_dif - p_dif)[1:] if moving else None
            wet, st.tail = self.decorrelators.process_gains(x, p_dif[1:], dg, r, st.tail)
            out[1:] += wet
        else:
            st.tail = _flush_tail(out[1:], st.tail)
        return out

    # -- a whole block, batched per stem -----------------------------------------
    def encode_into(self, buses: dict, stem: str, em: Emission, n: int) -> None:
        """Queue one Emission for `stem`; `finish_block(buses, n)` adds the block to the buses."""
        bus = buses.get(stem)
        if bus is None:
            bus = buses[stem] = np.zeros((self.channels, n))
        if em.is_ambisonic:
            bus += self._ambisonic(em.signal, n)
            return
        x = np.array(self._mono(em.signal, n), copy=True)    # held until finish_block: never alias a generator buffer
        _, p_coh, g_coh, p_dif, g_dif, moving = self._gains(em, batched=True)
        self._pending.setdefault(stem, []).append((x, p_coh, g_coh, p_dif, g_dif, moving))

    def finish_block(self, buses: dict, n: int) -> dict:
        """Sum the queued emissions into `buses` (BLAS) and run one decorrelator pass per stem."""
        N = self.channels
        pending, self._pending = self._pending, {}
        self._block_keys.clear()
        for stem in list(pending) + [s for s in self._stem_tails if s not in pending]:
            bus = buses.get(stem)
            if bus is None:
                bus = buses[stem] = np.zeros((N, n))
            items = pending.get(stem)
            tail = self._stem_tails.pop(stem, None)
            if not items:
                tail = _flush_tail(bus[1:], tail)
                if tail is not None:
                    self._stem_tails[stem] = tail
                continue
            X = np.stack([it[0] for it in items])                     # (S, n)
            G0 = np.stack([it[1] for it in items])                    # (S, N)
            D0 = np.stack([it[3] for it in items])[:, 1:]             # (S, N-1)
            bus += G0.T @ X
            din = D0.T @ X if D0.any() else None
            if any(it[5] for it in items):
                r = self._ramp(n)
                XR = X * r[None, :]
                dG = np.stack([it[2] - it[1] for it in items])
                dD = np.stack([it[4] - it[3] for it in items])[:, 1:]
                bus += dG.T @ XR
                if dD.any():
                    din = dD.T @ XR if din is None else din + dD.T @ XR
            if din is None:
                tail = _flush_tail(bus[1:], tail)
            else:
                wet, tail = self.decorrelators.process_multi(din, tail)
                bus[1:] += wet
            if tail is not None:
                self._stem_tails[stem] = tail
        return buses

    def encode_block(self, emissions: Iterable[Emission], n: int, stems: Optional[Iterable[str]] = None,
                     stem_of: Optional[Callable[[str], str]] = None) -> dict[str, np.ndarray]:
        """Encode and sum a block of emissions per stem -> {stem: (N, n)}.

        `stem_of` maps an emission's stem name to a bus (e.g. routing unknown stems).
        """
        buses: dict[str, np.ndarray] = {}
        if stems is not None:
            for s in stems:
                buses[s] = np.zeros((self.channels, n))
        for em in emissions:
            self.encode_into(buses, stem_of(em.stem) if stem_of is not None else em.stem, em, n)
        return self.finish_block(buses, n)
