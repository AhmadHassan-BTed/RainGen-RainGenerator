"""Streaming look-ahead true-peak limiter.

Detection runs on the 8x (>= 4x) polyphase-oversampled signal (BS.1770 Annex 2 style), so
inter-sample peaks are caught: a 4x detector can under-read by up to ~0.7 dB on HF-rich
material such as rain. Required attenuation per sample a = max(0, ln(peak / ceiling))
(nepers) is:
  1. max-held over the look-ahead window (+ detector alignment slack),
  2. released exponentially, a[n] = max(held[n], beta a[n-1]), beta = exp(-1 / (t_r sr))
     (vectorised as a running max in the log domain),
  3. smoothed by two cascaded moving averages spanning the look-ahead (an S-shaped attack
     completed exactly when the peak arrives),
and applied to the audio delayed by the look-ahead. Because every value inside the smoothing
span already meets the peak's requirement, the average does too: no overshoot, no clipper.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .loudness import PolyphaseOversampler

_NEPER_TO_DB = 20.0 / math.log(10.0)


def _box(x: np.ndarray, hist: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    if L <= 1:
        return x, hist
    ext = np.concatenate([hist, x])
    cs = np.concatenate([[0.0], np.cumsum(ext)])
    out = (cs[L:] - cs[:-L]) / L
    return out, ext[ext.shape[0] - (L - 1):].copy()


class TruePeakLimiter:
    """Look-ahead limiter holding the true peak at or below `ceiling_dbtp`."""

    def __init__(self, sr: int, channels: int, ceiling_dbtp: float = -1.0, lookahead_ms: float = 5.0,
                 release_ms: float = 80.0, oversample: int = 8, taps_per_phase: int = 12,
                 safety_db: float = 0.05):
        self.sr = int(sr)
        self.channels = int(channels)
        self.ceiling_dbtp = float(ceiling_dbtp)
        self.lookahead_ms = float(lookahead_ms)
        self.release_ms = float(release_ms)
        self._ceil = 10.0 ** ((self.ceiling_dbtp - safety_db) / 20.0)
        L = max(2, int(round(self.lookahead_ms * 1e-3 * self.sr)))
        self._L = L
        self._L1 = L // 2
        self._L2 = L - self._L1
        self._os = PolyphaseOversampler(max(1, int(oversample)), self.channels, taps_per_phase)
        d = self._os.delay
        f, c = int(math.floor(d)), int(math.ceil(d))
        self._W = L + 2 + (c - f)
        self.latency = L + c
        self._ln_beta = -1.0 / max(1e-6, self.release_ms * 1e-3 * self.sr)
        self.gain_reduction_db = 0.0
        self.max_gain_reduction_db = 0.0
        self.reset()

    def reset(self) -> None:
        self._os.reset()
        self._a_hist = np.zeros(self._W - 1)
        self._a_rel = 0.0
        self._b1 = np.zeros(max(0, self._L1 - 1))
        self._b2 = np.zeros(max(0, self._L2 - 1))
        self._delay = np.zeros((self.channels, self.latency))

    def set_ceiling(self, ceiling_dbtp: float, safety_db: float = 0.05) -> None:
        self.ceiling_dbtp = float(ceiling_dbtp)
        self._ceil = 10.0 ** ((self.ceiling_dbtp - safety_db) / 20.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Limit a block (C, n) or (n,); output is delayed by `latency` samples."""
        x_in = np.asarray(x, dtype=np.float64)
        mono = x_in.ndim == 1
        x2 = x_in[None, :] if mono else x_in
        C, n = x2.shape
        if C != self.channels:
            raise ValueError(f"limiter expects {self.channels} channels, got {C}")
        if n == 0:
            return x_in.copy()

        up = self._os.process(x2)
        fac = self._os.factor
        pk = np.max(np.abs(up).reshape(C, n, fac), axis=(0, 2))
        pk = np.maximum(pk, np.max(np.abs(x2), axis=0))
        a = np.log(np.maximum(pk / self._ceil, 1.0))

        ext = np.concatenate([self._a_hist, a])
        held = sliding_window_view(ext, self._W).max(axis=1)
        self._a_hist = ext[ext.shape[0] - (self._W - 1):].copy()

        idx = np.arange(1, n + 1, dtype=np.float64)
        lnb = self._ln_beta
        with np.errstate(divide="ignore"):
            logs = np.log(np.maximum(held, 1e-12)) - idx * lnb
        init = math.log(max(self._a_rel, 1e-12))
        m = np.maximum.accumulate(np.concatenate([[init], logs]))[1:]
        a_r = np.maximum(np.exp(m + idx * lnb), held)
        a_r[a_r < 1e-9] = 0.0
        self._a_rel = float(a_r[-1])

        s1, self._b1 = _box(a_r, self._b1, self._L1)
        s2, self._b2 = _box(s1, self._b2, self._L2)
        g = np.exp(-s2)

        dl = np.concatenate([self._delay, x2], axis=1)
        y = dl[:, :n] * g[None, :]
        self._delay = dl[:, n:].copy()

        gr = float(np.max(s2)) * _NEPER_TO_DB
        self.gain_reduction_db = gr
        if gr > self.max_gain_reduction_db:
            self.max_gain_reduction_db = gr
        return y[0] if mono else y

    def flush(self) -> np.ndarray:
        """Drain the look-ahead delay line (returns `latency` samples)."""
        return self.process(np.zeros((self.channels, self.latency)))

    @property
    def gain_db(self) -> float:
        return -self.gain_reduction_db


def limit(x: np.ndarray, sr: int, ceiling_dbtp: float = -1.0, lookahead_ms: float = 5.0,
          release_ms: float = 80.0, block: Optional[int] = None) -> np.ndarray:
    """Offline helper: limit a whole signal (C, n), latency-compensated, same length."""
    x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
    lim = TruePeakLimiter(sr, x2.shape[0], ceiling_dbtp, lookahead_ms, release_ms)
    step = int(block or sr)
    parts = [lim.process(x2[:, s:s + step]) for s in range(0, x2.shape[1], step)]
    parts.append(lim.flush())
    y = np.concatenate(parts, axis=1)[:, lim.latency: lim.latency + x2.shape[1]]
    return y[0] if np.ndim(x) == 1 else y
