"""Loudness and true-peak metering per ITU-R BS.1770-5, EBU R128 / Tech 3341 / Tech 3342.

* K-weighting: the two BS.1770 biquads (high-shelf ~+4 dB above ~1.7 kHz, RLB high-pass
  ~38 Hz) re-derived for any sample rate from their analog prototypes by the bilinear
  transform with pre-warping (f0, gain, Q identified from the 48 kHz coefficients; this
  reproduces the published 48 kHz coefficients to ~1e-8).
* Channel weights G_i: 1.0, except 1.41 for loudspeakers at 60 <= |az| <= 120 deg with
  |el| < 30 deg (surrounds); LFE excluded.
* Gating blocks of 400 ms, 75 % overlap; absolute gate -70 LKFS; relative gate -10 LU.
  Integrated loudness L = -0.691 + 10 log10(mean_gated sum_i G_i z_ij).
* Momentary = last 400 ms, short-term = last 3 s (both ungated), updated every 100 ms.
* Loudness range (EBU Tech 3342): short-term values, absolute gate -70 LUFS, relative gate
  -20 LU below their energy mean, LRA = P95 - P10.
* True peak (BS.1770 Annex 2): polyphase FIR interpolation by >= 4x (default 8x), max |x|.
"""

from __future__ import annotations

import math
from array import array
from collections import deque
from typing import Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_LOG_OFFSET = -0.691


def k_weighting_sos(sr: float) -> np.ndarray:
    """BS.1770 K-weighting as a 2-section SOS for sample rate `sr`."""
    fs = float(sr)
    # stage 1: high shelf (head acoustic effect)
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = math.tan(math.pi * f0 / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    # stage 2: RLB high-pass
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = math.tan(math.pi * f0 / fs)
    den = 1.0 + K / Q + K * K
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2.0 * (K * K - 1.0) / den, (1.0 - K / Q + K * K) / den]
    return np.array([b1 + a1, b2 + a2], dtype=np.float64)


_LAYOUT_WEIGHTS = {
    "mono": [1.0],
    "stereo": [1.0, 1.0],
    "binaural": [1.0, 1.0],
    "5.1": [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
    "7.1": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.41, 1.41],          # L R C LFE Lrs Rrs Lss Rss (WAV order)
    "7.1.4": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.41, 1.41, 1.0, 1.0, 1.0, 1.0],
}


def weight_for_position(az_deg: Optional[float], el_deg: float = 0.0) -> float:
    """BS.1770-5 position-dependent channel weight (None azimuth = LFE -> 0)."""
    if az_deg is None:
        return 0.0
    a = abs(((az_deg + 180.0) % 360.0) - 180.0)
    return 1.41 if (abs(el_deg) < 30.0 and 60.0 <= a <= 120.0) else 1.0


def channel_weights(layout: str, channels: int, smpte_order: bool = False) -> np.ndarray:
    """Per-channel BS.1770 weights for a named output layout."""
    lay = str(layout).strip().lower()
    if lay in ("foa", "hoa3", "ambix", "ambisonic") or lay.startswith("hoa"):
        w = np.zeros(channels)
        w[0] = 1.0                                   # W only (omni pressure)
        return w
    if smpte_order and lay in ("7.1", "7.1.4"):
        base = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0] + ([1.0] * 4 if lay == "7.1.4" else [])
    else:
        base = _LAYOUT_WEIGHTS.get(lay, [1.0] * channels)
    w = np.ones(channels)
    m = min(channels, len(base))
    w[:m] = base[:m]
    return w


class PolyphaseOversampler:
    """Streaming polyphase FIR interpolator (Kaiser-windowed sinc), (C, n) -> (C, n * factor)."""

    def __init__(self, factor: int = 8, channels: int = 1, taps_per_phase: Optional[int] = None,
                 beta: float = 7.0):
        from scipy.signal import firwin

        self.factor = int(factor)
        self.channels = int(channels)
        self.tpp = int(taps_per_phase or (12 if self.factor <= 4 else 16))
        if self.factor <= 1:
            self.phases = np.ones((1, 1))
            self.tpp = 1
        else:
            h = firwin(self.factor * self.tpp, 1.0 / self.factor, window=("kaiser", beta)) * self.factor
            self.phases = np.ascontiguousarray(h.reshape(self.tpp, self.factor).T)   # (factor, tpp)
        self._kernel = np.ascontiguousarray(self.phases[:, ::-1].T)                  # (tpp, factor)
        self._hist = np.zeros((self.channels, self.tpp - 1))
        self.delay = (self.factor * self.tpp - 1) / 2.0 / self.factor              # input samples

    def reset(self) -> None:
        self._hist[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        C, n = x.shape
        if self.factor <= 1:
            return x.copy()
        if n == 0:
            return np.zeros((C, 0))
        ext = np.concatenate([self._hist, x], axis=1)
        win = sliding_window_view(ext, self.tpp, axis=1)        # (C, n, tpp)
        y = win @ self._kernel                                  # (C, n, factor)
        self._hist = ext[:, ext.shape[1] - (self.tpp - 1):].copy()
        return y.reshape(C, n * self.factor)

    def flush(self) -> np.ndarray:
        return self.process(np.zeros((self.channels, self.tpp)))


def _lufs(z: float) -> float:
    return _LOG_OFFSET + 10.0 * math.log10(z) if z > 0.0 else float("-inf")


class LoudnessMeter:
    """Streaming BS.1770-5 / EBU R128 meter: integrated, momentary, short-term, LRA, true peak."""

    def __init__(self, sr: int, channels: int, layout: str = "stereo",
                 weights: Optional[Sequence[float]] = None, true_peak: bool = True, oversample: int = 8):
        self.sr = int(sr)
        self.channels = int(channels)
        self.layout = layout
        self.weights = np.asarray(weights, dtype=np.float64) if weights is not None else \
            channel_weights(layout, self.channels)
        self._sos = k_weighting_sos(self.sr)
        self._zi = np.zeros((self._sos.shape[0], self.channels, 2))
        self.hop = max(1, int(round(0.1 * self.sr)))            # 100 ms sub-blocks
        self._acc = 0.0
        self._acc_n = 0
        self._sub: deque = deque(maxlen=30)
        self._blocks = array("d")                               # 400 ms gating-block energies
        self._st = array("d")                                   # 3 s short-term energies
        self._max_m = 0.0
        self._max_s = 0.0
        self.samples = 0
        self._tp = PolyphaseOversampler(oversample, self.channels) if true_peak else None
        self._tp_max = np.zeros(self.channels)
        self._sample_peak = np.zeros(self.channels)

    def reset(self) -> None:
        self.__init__(self.sr, self.channels, self.layout, self.weights, self._tp is not None,
                      self._tp.factor if self._tp is not None else 8)

    # -- streaming ---------------------------------------------------------
    def process(self, x: np.ndarray) -> None:
        from scipy.signal import sosfilt

        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[0] != self.channels:
            raise ValueError(f"meter expects {self.channels} channels, got {x.shape[0]}")
        n = x.shape[1]
        if n == 0:
            return
        y, self._zi = sosfilt(self._sos, x, axis=-1, zi=self._zi)
        e = np.einsum("c,cn->n", self.weights, y * y)
        self._accumulate(e)
        self.samples += n
        self._sample_peak = np.maximum(self._sample_peak, np.max(np.abs(x), axis=1))
        if self._tp is not None:
            up = self._tp.process(x)
            self._tp_max = np.maximum(self._tp_max, np.max(np.abs(up), axis=1))

    def _accumulate(self, e: np.ndarray) -> None:
        pos = 0
        n = e.shape[0]
        hop = self.hop
        while pos < n:
            take = min(hop - self._acc_n, n - pos)
            self._acc += float(np.sum(e[pos:pos + take]))
            self._acc_n += take
            pos += take
            if self._acc_n == hop:
                self._sub.append(self._acc / hop)
                self._acc = 0.0
                self._acc_n = 0
                self._on_subblock()

    def _on_subblock(self) -> None:
        k = len(self._sub)
        if k >= 4:
            m = (self._sub[-1] + self._sub[-2] + self._sub[-3] + self._sub[-4]) / 4.0
            self._blocks.append(m)
            if m > self._max_m:
                self._max_m = m
        if k == 30:
            s = sum(self._sub) / 30.0
            self._st.append(s)
            if s > self._max_s:
                self._max_s = s

    # -- readings ----------------------------------------------------------
    def integrated(self) -> float:
        """Gated integrated loudness (LUFS); -inf if nothing passes the gates."""
        if len(self._blocks) == 0:
            return float("-inf")
        z = np.array(self._blocks, dtype=np.float64)
        with np.errstate(divide="ignore"):
            l = _LOG_OFFSET + 10.0 * np.log10(z)
        abs_mask = l > -70.0
        if not np.any(abs_mask):
            return float("-inf")
        gamma_r = _lufs(float(np.mean(z[abs_mask]))) - 10.0
        mask = abs_mask & (l > gamma_r)
        if not np.any(mask):
            return float("-inf")
        return _lufs(float(np.mean(z[mask])))

    def _recent(self, count: int) -> float:
        vals = list(self._sub)[-count:]
        total = sum(vals)
        k = len(vals)
        if self._acc_n > 0 and k < count:
            total += self._acc / self._acc_n
            k += 1
        return _lufs(total / k) if k else float("-inf")

    def momentary(self) -> float:
        """Momentary loudness (last 400 ms), LUFS."""
        return self._recent(4)

    def short_term(self) -> float:
        """Short-term loudness (last 3 s), LUFS."""
        return self._recent(30)

    def max_momentary(self) -> float:
        return _lufs(self._max_m)

    def max_short_term(self) -> float:
        return _lufs(self._max_s)

    def lra(self) -> float:
        """Loudness range (LU) per EBU Tech 3342."""
        if len(self._st) == 0:
            return 0.0
        z = np.array(self._st, dtype=np.float64)
        with np.errstate(divide="ignore"):
            l = _LOG_OFFSET + 10.0 * np.log10(z)
        abs_mask = l > -70.0
        if not np.any(abs_mask):
            return 0.0
        gamma_r = _lufs(float(np.mean(z[abs_mask]))) - 20.0
        vals = l[abs_mask & (l > gamma_r)]
        if vals.size == 0:
            return 0.0
        return float(np.percentile(vals, 95.0) - np.percentile(vals, 10.0))

    def true_peak_db(self) -> float:
        """Maximum true peak so far over all channels, dBTP."""
        peaks = self._tp_max if self._tp is not None else self._sample_peak
        p = float(np.max(np.maximum(peaks, self._sample_peak))) if peaks.size else 0.0
        return 20.0 * math.log10(p) if p > 0.0 else float("-inf")

    def true_peak_db_per_channel(self) -> list[float]:
        peaks = np.maximum(self._tp_max, self._sample_peak)
        return [20.0 * math.log10(p) if p > 0 else float("-inf") for p in peaks]

    def sample_peak_db(self) -> float:
        p = float(np.max(self._sample_peak)) if self._sample_peak.size else 0.0
        return 20.0 * math.log10(p) if p > 0.0 else float("-inf")

    def summary(self) -> dict:
        """All readings, ready for bext / reports."""
        return {
            "integrated_lufs": self.integrated(),
            "lra_lu": self.lra(),
            "true_peak_dbtp": self.true_peak_db(),
            "max_momentary_lufs": self.max_momentary(),
            "max_short_term_lufs": self.max_short_term(),
            "sample_peak_dbfs": self.sample_peak_db(),
            "duration_s": self.samples / self.sr,
        }


def integrated_loudness(x: np.ndarray, sr: int, layout: str = "stereo") -> float:
    """Integrated loudness (LUFS) of a whole signal (C, n) or (n,)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    meter = LoudnessMeter(sr, x.shape[0], layout, true_peak=False)
    step = max(1, int(sr) * 10)
    for s in range(0, x.shape[1], step):
        meter.process(x[:, s:s + step])
    return meter.integrated()


def true_peak_db(x: np.ndarray, sr: int, oversample: int = 8) -> float:
    """True-peak level (dBTP) of a whole signal (C, n) or (n,), BS.1770 Annex 2 style."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.size == 0:
        return float("-inf")
    factor = int(oversample)
    if sr >= 176400:
        factor = max(1, factor // 4)
    elif sr >= 88200:
        factor = max(2, factor // 2)
    os_ = PolyphaseOversampler(factor, x.shape[0])
    peak = float(np.max(np.abs(x)))
    step = max(1, int(sr) * 10)
    for s in range(0, x.shape[1], step):
        peak = max(peak, float(np.max(np.abs(os_.process(x[:, s:s + step])))))
    peak = max(peak, float(np.max(np.abs(os_.flush()))))
    return 20.0 * math.log10(peak) if peak > 0.0 else float("-inf")
