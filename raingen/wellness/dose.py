"""Hearing-dose meter per ITU-T H.870 (2022) safe-listening guidance (MASTER_PLAN 8.4, Appendix A).

Dose is accumulated as A-weighted sound exposure in Pa^2 h:

    E = sum p_A,rms^2 * dt   (dt in hours)

against a weekly allowance of 1.6 Pa^2 h for adults (80 dBA for 40 h, mode 1)
and 0.51 Pa^2 h for children / sensitive users (75 dBA for 40 h, mode 2), over
a rolling 7-day window kept in hourly bins. Digital level is mapped to SPL by a
calibration constant: the SPL (dB re 20 uPa) produced by a full-scale sine at
the current device volume. A-weighting is the IEC 61672-1 analog prototype
(poles 20.6, 107.7, 737.9, 12194 Hz) through the bilinear transform,
normalised to 0 dB at 1 kHz. The louder ear is metered (conservative).
When the allowance is exceeded `recommended_gain()` brings playback back to the
reference level; that limit sits outside the parameter graph (2.7).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from scipy import signal

from ..core.constants import SAMPLE_RATE

P_REF = 20e-6
ALLOWANCE_PA2H = {"adult": 1.6, "child": 0.51}
REFERENCE_DBA = {"adult": 80.0, "child": 75.0}
WEEK_H = 168


def _sos_gain(sos: np.ndarray, f: float, sr: float) -> float:
    zinv = np.exp(-1j * 2.0 * math.pi * f / sr)
    h = 1.0 + 0.0j
    for b0, b1, b2, a0, a1, a2 in sos:
        h *= (b0 + b1 * zinv + b2 * zinv * zinv) / (a0 + a1 * zinv + a2 * zinv * zinv)
    return float(abs(h))


def a_weighting_sos(sr: float = SAMPLE_RATE) -> np.ndarray:
    """IEC 61672-1 A-weighting as second-order sections, 0 dB at 1 kHz."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    w = lambda f: 2.0 * math.pi * f
    z = [0.0, 0.0, 0.0, 0.0]
    p = [-w(f1), -w(f1), -w(f2), -w(f3), -w(f4), -w(f4)]
    k = w(f4) ** 2
    zd, pd, kd = signal.bilinear_zpk(z, p, k, sr)
    sos = signal.zpk2sos(zd, pd, kd)
    g = _sos_gain(sos, 1000.0, sr)
    sos[0, :3] /= g
    return sos


def dose_pa2h(level_dba: float, hours: float) -> float:
    """Exposure of `hours` at `level_dba` in Pa^2 h."""
    p2 = (P_REF * 10.0 ** (level_dba / 20.0)) ** 2
    return p2 * hours


def hours_until(allowance_pa2h: float, level_dba: float) -> float:
    p2 = (P_REF * 10.0 ** (level_dba / 20.0)) ** 2
    return allowance_pa2h / p2 if p2 > 0 else math.inf


class DoseMeter:
    """Rolling 7-day A-weighted exposure meter (ITU-T H.870)."""

    def __init__(self, sr: int = SAMPLE_RATE, channels: int = 2, calibration_db_spl: float = 100.0,
                 profile: str = "adult", start_time_s: Optional[float] = None, window_h: int = WEEK_H):
        if profile not in ALLOWANCE_PA2H:
            raise ValueError(f"profile must be one of {sorted(ALLOWANCE_PA2H)}")
        self.sr = int(sr)
        self.channels = int(channels)
        self.calibration_db_spl = float(calibration_db_spl)
        self.profile = profile
        self.allowance = ALLOWANCE_PA2H[profile]
        self.reference_dba = REFERENCE_DBA[profile]
        self.window_h = int(window_h)
        self.start_time_s = float(start_time_s if start_time_s is not None else time.time())
        self._sos = a_weighting_sos(self.sr)
        self._zi: Optional[np.ndarray] = None
        self.bins: dict[int, float] = {}          # epoch hour -> Pa^2 h
        self.session_dose = 0.0
        self.session_seconds = 0.0
        self._recent: list[tuple[float, float]] = []   # (seconds, mean p^2) for short-term LAeq
        self._recent_s = 0.0
        self._processed = 0

    @property
    def pa_per_unit(self) -> float:
        """Pascals rms per digital rms unit: a full-scale sine (rms 1/sqrt 2) gives calibration_db_spl."""
        return P_REF * 10.0 ** (self.calibration_db_spl / 20.0) * math.sqrt(2.0)

    def set_calibration(self, db_spl_full_scale: float) -> None:
        self.calibration_db_spl = float(db_spl_full_scale)

    def process(self, x: np.ndarray, now_s: Optional[float] = None, volume_gain: float = 1.0) -> float:
        """Accumulate the dose for one block (C, n) or (n,); returns this block's Pa^2 h.

        `now_s` is the wall-clock epoch time at the block (defaults to start + audio time),
        `volume_gain` any post-meter playback gain (linear)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        C, n = x.shape
        if n == 0:
            return 0.0
        if self._zi is None or self._zi.shape[1] != C:
            self._zi = np.zeros((self._sos.shape[0], C, 2))
        xa, self._zi = signal.sosfilt(self._sos, x * float(volume_gain), axis=-1, zi=self._zi)
        ms = float(np.max(np.mean(xa * xa, axis=1)))
        p2 = ms * self.pa_per_unit ** 2
        dt_s = n / self.sr
        dose = p2 * dt_s / 3600.0
        t = float(now_s) if now_s is not None else self.start_time_s + self._processed / self.sr
        hour = int(t // 3600.0)
        self.bins[hour] = self.bins.get(hour, 0.0) + dose
        self.session_dose += dose
        self.session_seconds += dt_s
        self._processed += n
        self._recent.append((dt_s, p2))
        self._recent_s += dt_s
        while self._recent and self._recent_s - self._recent[0][0] >= 60.0:
            d, _ = self._recent.pop(0)
            self._recent_s -= d
        if len(self.bins) > self.window_h * 2:
            self._prune(hour)
        return dose

    def _prune(self, now_hour: int) -> None:
        for h in [h for h in self.bins if h <= now_hour - self.window_h]:
            del self.bins[h]

    def _now_hour(self, now_s: Optional[float]) -> int:
        t = float(now_s) if now_s is not None else self.start_time_s + self._processed / self.sr
        return int(t // 3600.0)

    def weekly_dose(self, now_s: Optional[float] = None) -> float:
        """Pa^2 h accumulated in the rolling window ending now."""
        h = self._now_hour(now_s)
        return float(sum(v for k, v in self.bins.items() if h - self.window_h < k <= h))

    def fraction(self, now_s: Optional[float] = None) -> float:
        return self.weekly_dose(now_s) / self.allowance

    def percent(self, now_s: Optional[float] = None) -> float:
        return 100.0 * self.fraction(now_s)

    def exceeded(self, now_s: Optional[float] = None) -> bool:
        return self.fraction(now_s) >= 1.0

    def laeq_db(self) -> float:
        """Session LAeq (dB SPL, A-weighted)."""
        if self.session_seconds <= 0:
            return -math.inf
        p2 = self.session_dose * 3600.0 / self.session_seconds
        return 10.0 * math.log10(p2 / P_REF ** 2 + 1e-30)

    def short_term_laeq_db(self) -> float:
        """LAeq over the last ~60 s of audio."""
        if self._recent_s <= 0:
            return -math.inf
        p2 = sum(d * v for d, v in self._recent) / self._recent_s
        return 10.0 * math.log10(p2 / P_REF ** 2 + 1e-30)

    def remaining_hours(self, level_dba: Optional[float] = None, now_s: Optional[float] = None) -> float:
        """Listening time left this week at `level_dba` (default: current short-term level)."""
        lvl = self.short_term_laeq_db() if level_dba is None else float(level_dba)
        left = max(0.0, self.allowance - self.weekly_dose(now_s))
        if not math.isfinite(lvl):
            return math.inf
        return hours_until(left, lvl)

    def recommended_gain(self, now_s: Optional[float] = None) -> float:
        """Linear playback gain: 1 below the allowance; above it, the gain that brings the
        short-term level down to the reference level (80 dBA adult / 75 dBA child)."""
        if not self.exceeded(now_s):
            return 1.0
        lvl = self.short_term_laeq_db()
        if not math.isfinite(lvl) or lvl <= self.reference_dba:
            return 1.0
        return float(10.0 ** ((self.reference_dba - lvl) / 20.0))

    def status(self, now_s: Optional[float] = None) -> dict:
        return {"profile": self.profile, "weekly_pa2h": self.weekly_dose(now_s), "allowance_pa2h": self.allowance,
                "percent": self.percent(now_s), "laeq_db": self.laeq_db(), "short_term_laeq_db": self.short_term_laeq_db(),
                "remaining_h": self.remaining_hours(None, now_s), "exceeded": self.exceeded(now_s),
                "recommended_gain": self.recommended_gain(now_s)}

    def to_dict(self) -> dict:
        return {"profile": self.profile, "calibration_db_spl": self.calibration_db_spl, "window_h": self.window_h,
                "bins": {str(k): v for k, v in self.bins.items()}}

    @classmethod
    def from_dict(cls, d: dict, sr: int = SAMPLE_RATE, channels: int = 2) -> "DoseMeter":
        m = cls(sr=sr, channels=channels, calibration_db_spl=d.get("calibration_db_spl", 100.0),
                profile=d.get("profile", "adult"), window_h=int(d.get("window_h", WEEK_H)))
        m.bins = {int(k): float(v) for k, v in (d.get("bins") or {}).items()}
        return m
