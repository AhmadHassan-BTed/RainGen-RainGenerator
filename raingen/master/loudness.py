"""Loudness and true-peak metering per ITU-R BS.1770-5, EBU R128 / Tech 3341 / Tech 3342.

* K-weighting: the two BS.1770 biquads (high-shelf ~+4 dB above ~1.7 kHz, RLB high-pass
  ~38 Hz) re-derived for any sample rate from their analog prototypes by the bilinear
  transform with pre-warping (f0, gain, Q identified from the 48 kHz coefficients; this
  reproduces the published 48 kHz coefficients to ~1e-8).
* Channel weights G_i: 1.0, except 1.41 for loudspeakers at 60 <= |az| <= 120 deg with
  |el| < 30 deg (surrounds); LFE excluded.
* Gating blocks of 400 ms, 75 % overlap; absolute gate -70 LKFS; relative gate -10 LU.
  Integrated loudness L = -0.691 + 10 log10(mean_gated sum_i G_i z_ij).
  A programme shorter than one gating block has no BS.1770 value; the meter then reports a
  *provisional* reading (ungated mean square over all the audio so far, absolute gate only)
  and flags it (`integrated_is_provisional`), so short clips can still be normalised.
* Momentary = last 400 ms, short-term = last 3 s (both ungated), updated every 100 ms.
* Loudness range (EBU Tech 3342): short-term values, absolute gate -70 LUFS, relative gate
  -20 LU below their energy mean, LRA = P95 - P10.
* True peak (BS.1770 Annex 2): polyphase FIR interpolation by >= 4x (default 8x), max |x|.
  Every phase obeys |y_p[m]| <= l1 * max|x over its window| (l1 = largest phase L1 norm,
  2.55 for the 8x design), so peak readers interpolate exactly only the samples that could
  beat the running peak (meter) or the ceiling (limiter): a few % of a crackly rain bed.
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
    """Streaming polyphase FIR interpolator (Kaiser-windowed sinc), (C, n) -> (C, n * factor).

    The 8x default (48 taps per phase, Kaiser beta 10, cutoff at the input Nyquist) is flat to
    ~20 kHz at 48 kHz: a 16-tap design rolls off from ~17 kHz and under-reads the true peak of
    a 19 kHz tone by ~0.45 dB against a near-ideal reconstruction. 4x and below keep the
    BS.1770 Annex 2 size (12 taps per phase). Meter, limiter and true_peak_db() all use the 8x
    default at every sample rate.
    """

    def __init__(self, factor: int = 8, channels: int = 1, taps_per_phase: Optional[int] = None,
                 beta: Optional[float] = None):
        from scipy.signal import firwin

        self.factor = int(factor)
        self.channels = int(channels)
        self.tpp = int(taps_per_phase or (12 if self.factor <= 4 else 48))
        if beta is None:
            beta = 7.0 if self.factor <= 4 else 10.0
        if self.factor <= 1:
            self.phases = np.ones((1, 1))
            self.tpp = 1
        else:
            h = firwin(self.factor * self.tpp, 1.0 / self.factor, window=("kaiser", beta)) * self.factor
            self.phases = np.ascontiguousarray(h.reshape(self.tpp, self.factor).T)   # (factor, tpp)
        self._kernel = np.ascontiguousarray(self.phases[:, ::-1].T)                  # (tpp, factor)
        self._hist = np.zeros((self.channels, self.tpp - 1))
        self.delay = (self.factor * self.tpp - 1) / 2.0 / self.factor              # input samples
        # |y_p[m]| <= l1 * max|x over its tpp-sample window| for every phase p (slack covers rounding)
        self.l1 = float(np.max(np.sum(np.abs(self.phases), axis=1))) * (1.0 + 1e-9)
        self._toep: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._hist[:] = 0.0

    def _advance(self, x: np.ndarray) -> np.ndarray:
        """History + block (C, tpp - 1 + n); moves the stream on by the block."""
        ext = np.concatenate([self._hist, x], axis=1)
        self._hist = ext[:, ext.shape[1] - (self.tpp - 1):].copy()
        return ext

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        C, n = x.shape
        if self.factor <= 1:
            return x.copy()
        if n == 0:
            return np.zeros((C, 0))
        ext = self._advance(x)
        return self._interp(ext, n).reshape(C, n * self.factor)

    # Long blocks use a block-Toeplitz form: each row of one BLAS GEMM yields B outputs x factor
    # phases from B + tpp - 1 inputs. numpy cannot hand the overlapping sliding-window strides
    # to BLAS, so the direct product runs in its generic loop: ~5x slower per sample at 48k
    # samples, while at ~1k samples the GEMM is too short to win.
    _TOEPLITZ_B = 16
    _TOEPLITZ_MIN_N = 2048

    def _interp(self, ext: np.ndarray, n: int) -> np.ndarray:
        """Interpolated block (C, n, factor) from ext = history + block (C, tpp - 1 + n)."""
        if n < self._TOEPLITZ_MIN_N:
            return sliding_window_view(ext, self.tpp, axis=1) @ self._kernel
        B, t, F = self._TOEPLITZ_B, self.tpp, self.factor
        if self._toep is None:
            T = np.zeros((B, F, B + t - 1))
            for r in range(B):
                T[r, :, r:r + t] = self._kernel.T          # output r of a row reads inputs r .. r + t - 1
            self._toep = np.ascontiguousarray(T.reshape(B * F, B + t - 1).T)     # (B + t - 1, B * F)
        C = ext.shape[0]
        Q = -(-n // B)
        e = np.zeros((C, Q * B + t - 1))
        e[:, :ext.shape[1]] = ext
        rows = np.ascontiguousarray(sliding_window_view(e, B + t - 1, axis=1)[:, ::B][:, :Q])
        y = rows.reshape(C * Q, B + t - 1) @ self._toep
        return y.reshape(C, Q * B, F)[:, :n]

    def _full_fraction(self, n: int) -> float:
        """Candidate fraction above which interpolating the whole block is cheaper."""
        return 0.25 if n >= self._TOEPLITZ_MIN_N else 0.5

    def _window_max(self, a: np.ndarray, n: int) -> np.ndarray:
        """Upper bound (>=) of max |ext| over the tpp-sample window behind each of the n outputs.

        Maxima of 16-sample chunks, then over every chunk a window can touch (4 at 48 taps):
        a superset of each window, so still a rigorous bound, ~30x cheaper than a sliding
        maximum filter.
        """
        t, G = self.tpp, 16
        a = np.abs(a)
        L = a.shape[-1]
        Qc = -(-L // G)
        span = (G + t - 2) // G + 1                           # chunks touched by [m, m + t - 1]
        cm = np.zeros(a.shape[:-1] + (Qc + span - 1,))
        cm[..., :Qc] = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, Qc * G - L)]).reshape(
            a.shape[:-1] + (Qc, G)).max(axis=-1)
        s = cm[..., :Qc].copy()
        for k in range(1, span):
            np.maximum(s, cm[..., k:k + Qc], out=s)
        return np.repeat(s, G, axis=-1)[..., :n]

    def peak_envelope(self, x: np.ndarray, floor: float = 0.0) -> np.ndarray:
        """Per input sample, max |interpolated| over channels and phases: (C, n) -> (n,).

        Advances the stream exactly like process(). Exact wherever the value can exceed
        `floor`; elsewhere 0 (which is <= floor), decided by the l1 bound.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        n = x.shape[1]
        if n == 0:
            return np.zeros(0)
        if self.factor <= 1:
            return np.max(np.abs(x), axis=0)
        ext = self._advance(x)
        a = np.max(np.abs(ext), axis=0)
        if a.max() * self.l1 <= floor:                          # whole block safely below the floor
            return np.zeros(n)
        idx = np.flatnonzero(self._window_max(a, n) * self.l1 > floor)
        if idx.size == 0:
            return np.zeros(n)
        if idx.size > self._full_fraction(n) * n:               # busy block: interpolate everything
            return np.max(np.abs(self._interp(ext, n)), axis=(0, 2))
        win = sliding_window_view(ext, self.tpp, axis=1)
        out = np.zeros(n)
        out[idx] = np.max(np.abs(win[:, idx, :] @ self._kernel), axis=(0, 2))
        return out

    def channel_peaks(self, x: np.ndarray, floor=0.0) -> np.ndarray:
        """Per channel, max |interpolated| over the block: (C, n) -> (C,).

        Advances the stream exactly like process(). Exact for every channel whose peak can
        exceed its `floor` (scalar or per channel); otherwise a value <= floor.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        C, n = x.shape
        if n == 0:
            return np.zeros(C)
        if self.factor <= 1:
            return np.max(np.abs(x), axis=1)
        ext = self._advance(x)
        fl = np.asarray(floor, dtype=np.float64).reshape(-1, 1)            # scalar or (C,)
        out = np.zeros(C)
        # block bound first (one reduction), per-sample bounds only when it can be beaten
        if not np.any(np.max(np.abs(ext), axis=1, keepdims=True) * self.l1 > fl):
            return out
        cand = self._window_max(ext, n) * self.l1 > fl                    # (C, n)
        k = int(np.count_nonzero(cand))
        if k > self._full_fraction(n) * C * n:
            return np.max(np.abs(self._interp(ext, n)), axis=(1, 2))
        win = sliding_window_view(ext, self.tpp, axis=1)
        for c in np.flatnonzero(np.any(cand, axis=1)):
            out[c] = np.max(np.abs(win[c, np.flatnonzero(cand[c]), :] @ self._kernel))
        return out

    def flush(self) -> np.ndarray:
        return self.process(np.zeros((self.channels, self.tpp)))

    def tail(self) -> np.ndarray:
        """What flush() would return, without changing the stream state."""
        hist = self._hist.copy()
        out = self.flush()
        self._hist = hist
        return out


def _lufs(z: float) -> float:
    return _LOG_OFFSET + 10.0 * math.log10(z) if z > 0.0 else float("-inf")


class LoudnessMeter:
    """Streaming BS.1770-5 / EBU R128 meter: integrated, momentary, short-term, LRA, true peak."""

    _TP_BATCH = 8192                                            # samples per true-peak interpolation pass

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
        self._energy = 0.0                                      # sum of weighted z over all samples
        self._sub: deque = deque(maxlen=30)
        self._blocks = array("d")                               # 400 ms gating-block energies
        self._st = array("d")                                   # 3 s short-term energies
        self._max_m = 0.0
        self._max_s = 0.0
        self.samples = 0
        self._tp = PolyphaseOversampler(oversample, self.channels) if true_peak else None
        self._tp_max = np.zeros(self.channels)
        self._tp_pending: list = []                             # input awaiting true-peak interpolation
        self._tp_pending_n = 0
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
        # On digital silence the K-filter recursion settles into a subnormal limit cycle that
        # never reaches zero (every later block ~16x slower): flush states far below the gate.
        np.copyto(self._zi, 0.0, where=np.abs(self._zi) < 1e-30)
        e = np.einsum("c,cn->n", self.weights, y * y)
        self._energy += float(np.sum(e))
        self._accumulate(e)
        self.samples += n
        self._sample_peak = np.maximum(self._sample_peak, np.max(np.abs(x), axis=1))
        if self._tp is not None:
            # batched: one pass per >= _TP_BATCH samples puts the 8x product on BLAS (block
            # Toeplitz); readings drain the rest, so nothing is ever missed
            self._tp_pending.append(np.array(x, dtype=np.float64, copy=True))
            self._tp_pending_n += n
            if self._tp_pending_n >= self._TP_BATCH:
                self._tp_drain()

    def _tp_drain(self) -> None:
        if not self._tp_pending:
            return
        x = self._tp_pending[0] if len(self._tp_pending) == 1 else np.concatenate(self._tp_pending, axis=1)
        self._tp_pending = []
        self._tp_pending_n = 0
        # only samples that could beat the peak reported so far need the 8x interpolation
        floor = np.maximum(self._tp_max, self._sample_peak)
        self._tp_max = np.maximum(self._tp_max, self._tp.channel_peaks(x, floor))

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
    @property
    def integrated_is_provisional(self) -> bool:
        """True while the programme is shorter than one 400 ms gating block.

        integrated() then returns the provisional (ungated) reading, not a BS.1770 value.
        """
        return len(self._blocks) == 0 and self.samples > 0

    def provisional_integrated(self) -> float:
        """Ungated mean-square loudness over all the audio so far (absolute gate -70 LUFS only)."""
        if self.samples <= 0:
            return float("-inf")
        l = _lufs(self._energy / self.samples)
        return l if l > -70.0 else float("-inf")

    def integrated(self, provisional: bool = True) -> float:
        """Gated integrated loudness (LUFS); -inf if nothing passes the gates.

        Before the first complete 400 ms gating block exists, BS.1770 defines no value: with
        `provisional` (default) the ungated reading of provisional_integrated() is returned
        instead (check `integrated_is_provisional`); with provisional=False, -inf.
        """
        if len(self._blocks) == 0:
            return self.provisional_integrated() if provisional else float("-inf")
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
        """Mean square over the last `count` 100 ms sub-blocks (fewer + the partial one at the start)."""
        vals = list(self._sub)[-count:]
        k = len(vals)
        energy = sum(vals) * self.hop                          # sub-block means -> sample sums
        n = k * self.hop
        if self._acc_n > 0 and k < count:
            energy += self._acc                                 # weight the partial sub-block by its length
            n += self._acc_n
        return _lufs(energy / n) if n else float("-inf")

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

    def _peaks(self) -> np.ndarray:
        """Per-channel linear true peak (sample peak when true-peak metering is off)."""
        if self._tp is not None:
            self._tp_drain()
        peaks = np.maximum(self._tp_max, self._sample_peak)
        if self._tp is not None and self.samples > 0:
            # the interpolator still holds the last few samples: include their ringing
            peaks = np.maximum(peaks, np.max(np.abs(self._tp.tail()), axis=1))
        return peaks

    def true_peak_db(self) -> float:
        """Maximum true peak so far over all channels, dBTP."""
        peaks = self._peaks()
        p = float(np.max(peaks)) if peaks.size else 0.0
        return 20.0 * math.log10(p) if p > 0.0 else float("-inf")

    def true_peak_db_per_channel(self) -> list[float]:
        return [20.0 * math.log10(p) if p > 0 else float("-inf") for p in self._peaks()]

    def sample_peak_db(self) -> float:
        p = float(np.max(self._sample_peak)) if self._sample_peak.size else 0.0
        return 20.0 * math.log10(p) if p > 0.0 else float("-inf")

    def summary(self) -> dict:
        """All readings, ready for bext / reports."""
        return {
            "integrated_lufs": self.integrated(),
            "integrated_provisional": self.integrated_is_provisional,
            "lra_lu": self.lra(),
            "true_peak_dbtp": self.true_peak_db(),
            "max_momentary_lufs": self.max_momentary(),
            "max_short_term_lufs": self.max_short_term(),
            "sample_peak_dbfs": self.sample_peak_db(),
            "duration_s": self.samples / self.sr,
        }


def integrated_loudness(x: np.ndarray, sr: int, layout: str = "stereo", provisional: bool = True) -> float:
    """Integrated loudness (LUFS) of a whole signal (C, n) or (n,).

    Signals shorter than 400 ms get the provisional ungated reading (see
    LoudnessMeter.integrated); pass provisional=False for strict BS.1770 (-inf).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    meter = LoudnessMeter(sr, x.shape[0], layout, true_peak=False)
    step = max(1, int(sr) * 10)
    for s in range(0, x.shape[1], step):
        meter.process(x[:, s:s + step])
    return meter.integrated(provisional)


def true_peak_db(x: np.ndarray, sr: int, oversample: int = 8) -> float:
    """True-peak level (dBTP) of a whole signal (C, n) or (n,), BS.1770 Annex 2 style.

    Same interpolator as LoudnessMeter and TruePeakLimiter at every sample rate (8x, 48 taps
    per phase; plan sec. 10: >= 8x detection everywhere), so the three always agree. A 4x/12-tap
    reader at 96 kHz read up to 0.013 dB above the meter on the limiter's own output.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.size == 0:
        return float("-inf")
    os_ = PolyphaseOversampler(int(oversample), x.shape[0])
    peak = float(np.max(np.abs(x)))
    step = max(1, int(sr))                  # 1 s chunks: a busy 12-channel chunk is ~37 MB at 8x
    for s in range(0, x.shape[1], step):
        # only samples that could beat the peak found so far are interpolated exactly
        peak = max(peak, float(np.max(os_.channel_peaks(x[:, s:s + step], peak))))
    peak = max(peak, float(np.max(np.abs(os_.flush()))))
    return 20.0 * math.log10(peak) if peak > 0.0 else float("-inf")
