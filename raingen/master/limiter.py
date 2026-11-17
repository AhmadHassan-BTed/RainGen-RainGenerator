"""Streaming look-ahead true-peak limiter.

Detection runs on the 8x (>= 4x) polyphase-oversampled signal (BS.1770 Annex 2 style), so
inter-sample peaks are caught: a 4x detector can under-read by up to ~0.7 dB on HF-rich
material such as rain. The detector uses the same interpolator as LoudnessMeter (same
factor and taps: 8x, 48 taps per phase, flat to ~20 kHz), so the limited output never reads
above the ceiling on our own meter; a shorter detector than the meter's lets near-Nyquist
peaks through (a 12-tap detector overshot a 16-tap meter by up to 0.8 dB). Non-finite input
samples are zeroed. The sample-peak term is delayed to line up with the interpolator's group
delay. Only samples whose l1 bound can reach the ceiling are interpolated (identical
decisions, a few % of a rain bed), and an idle limiter passes the delayed input bit-exactly.
Required attenuation per sample a = max(0, ln(peak / ceiling)) (nepers) is:
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
from scipy.ndimage import maximum_filter1d

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
                 release_ms: float = 80.0, oversample: int = 8, taps_per_phase: Optional[int] = None,
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
        self._c = c                                    # sample-peak delay matching the interpolator
        self.latency = L + c
        self._ln_beta = -1.0 / max(1e-6, self.release_ms * 1e-3 * self.sr)
        self.gain_reduction_db = 0.0
        self.max_gain_reduction_db = 0.0
        self.reset()

    def reset(self) -> None:
        self._os.reset()
        self._a_hist = np.zeros(self._W - 1)
        self._pk_hist = np.zeros(self._c)
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

        if not np.all(np.isfinite(x2)):
            x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)

        # inter-sample peaks, interpolated exactly only where they could reach the ceiling
        pk = self._os.peak_envelope(x2, self._ceil)
        spk = np.max(np.abs(x2), axis=0)
        if self._c > 0:                                 # delay the sample peaks by the interpolator delay
            spk = np.concatenate([self._pk_hist, spk])
            self._pk_hist = spk[spk.shape[0] - self._c:].copy()
            spk = spk[:n]
        pk = np.maximum(pk, spk)
        a = np.log(np.maximum(pk / self._ceil, 1.0))

        dl = np.concatenate([self._delay, x2], axis=1)
        self._delay = dl[:, n:].copy()
        if self._a_rel == 0.0 and not a.any() and not self._a_hist.any() and not self._b1.any() \
                and not self._b2.any():
            # idle (nothing to do, no release in flight): the full path below gives g == 1.0
            # exactly here, so the output is the bit-exact delayed input
            self.gain_reduction_db = 0.0
            y = dl[:, :n].copy()
            return y[0] if mono else y

        # max over the last W requirements (O(n) running max, window fully inside ext)
        W = self._W
        ext = np.concatenate([self._a_hist, a])
        held = maximum_filter1d(ext, W)[W // 2: W // 2 + n]
        self._a_hist = ext[ext.shape[0] - (W - 1):].copy()

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
        y = dl[:, :n] * g[None, :]

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
