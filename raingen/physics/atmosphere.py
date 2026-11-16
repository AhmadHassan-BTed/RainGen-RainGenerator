"""Outdoor sound propagation: absorption, spreading, refraction, ground effect.

Physics implemented here:

* Atmospheric absorption, ISO 9613-1:1993 (exact formula, eqs. 3-5 and Annex B),
  with temperature, relative humidity and pressure. Reference: 20 C / 70 % RH gives
  4.98 dB/km at 1 kHz and 77.6 dB/km at 8 kHz.
* Geometric spreading: point (1/r), line (1/sqrt r), plane (none).
* Refraction by sound-speed gradients (temperature lapse + wind shear, c_eff = c(T) + u cos theta):
  upward refraction creates a shadow beyond r = sqrt(2c/a)(sqrt(h_s) + sqrt(h_r)) with
  a = (c / 2T) * lapse (e.g. a 5 km source in a 9.8 K/km lapse gives ~25 km);
  downward refraction (inversion, downwind) channels sound once the ray sag r^2/(8 R_c)
  exceeds the path height (R_c = c/a), turning spherical into cylindrical spreading.
* Ground effect: two-ray model with Delany & Bazley (1970) impedance (one-parameter,
  flow resistivity sigma; used only inside 0.01 <= rho0 f / sigma <= 1) and the
  spherical-wave boundary-loss factor F(w) (Attenborough et al., "Predicting Outdoor
  Sound", ch. 2), reduced to a low-shelf + interference-dip filter.
* Flash-to-bang time from c(T) (2.91 s/km at 20 C).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.signal import sosfilt
from scipy.special import wofz

from ..core.constants import SAMPLE_RATE, speed_of_sound

_PASSTHROUGH = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
_DENORMAL_FLUSH = 1e-30       # filter states below this are zeroed (subnormal stalls)


# ------------------------------------------------------------------ ISO 9613-1
def iso9613_alpha_db_per_m(f_hz, temp_c: float, rh_pct: float, pressure_hpa: float = 1013.25) -> np.ndarray:
    """Pure-tone atmospheric absorption coefficient alpha (dB/m), ISO 9613-1."""
    f = np.asarray(f_hz, dtype=float)
    T = float(temp_c) + 273.15
    T0 = 293.15
    T01 = 273.16
    p_rel = max(float(pressure_hpa), 1.0) / 1013.25
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    psat_rel = 10.0 ** C
    h = min(max(float(rh_pct), 0.0), 100.0) * psat_rel / p_rel     # molar concentration of water vapour, %
    fr_o = p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = p_rel * (T / T0) ** -0.5 * (9.0 + 280.0 * h * math.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    f2 = f * f
    alpha = 8.686 * f2 * (
        1.84e-11 / p_rel * (T / T0) ** 0.5
        + (T / T0) ** -2.5 * (
            0.01275 * math.exp(-2239.1 / T) / (fr_o + f2 / fr_o)
            + 0.1068 * math.exp(-3352.0 / T) / (fr_n + f2 / fr_n)
        )
    )
    return alpha


def air_absorption_db(f_hz, distance_m: float, temp_c: float, rh_pct: float,
                      pressure_hpa: float = 1013.25) -> np.ndarray:
    """Total absorption loss (dB, positive) over `distance_m`."""
    return iso9613_alpha_db_per_m(f_hz, temp_c, rh_pct, pressure_hpa) * max(float(distance_m), 0.0)


# ------------------------------------------------------------------ biquad helpers (RBJ cookbook)
def _high_shelf(f0: float, gain_db: float, sr: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt(2.0)          # shelf slope S = 1
    sa = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1) + (A - 1) * cw + sa)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - sa)
    a0 = (A + 1) - (A - 1) * cw + sa
    a1 = 2.0 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - sa
    return np.array([b0, b1, b2, a0, a1, a2]) / a0


def _low_shelf(f0: float, gain_db: float, sr: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt(2.0)
    sa = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1) - (A - 1) * cw + sa)
    b1 = 2.0 * A * ((A - 1) - (A + 1) * cw)
    b2 = A * ((A + 1) - (A - 1) * cw - sa)
    a0 = (A + 1) + (A - 1) * cw + sa
    a1 = -2.0 * ((A - 1) + (A + 1) * cw)
    a2 = (A + 1) + (A - 1) * cw - sa
    return np.array([b0, b1, b2, a0, a1, a2]) / a0


def _peaking(f0: float, gain_db: float, q: float, sr: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A
    return np.array([b0, b1, b2, a0, a1, a2]) / a0


def _sos_mag_db(sos: np.ndarray, f: np.ndarray, sr: float) -> np.ndarray:
    z1 = np.exp(-1j * 2.0 * np.pi * f / sr)
    z2 = z1 * z1
    h = np.ones_like(z1)
    for b0, b1, b2, a0, a1, a2 in np.atleast_2d(sos):
        h = h * (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return 20.0 * np.log10(np.abs(h) + 1e-300)


# ------------------------------------------------------------------ absorption filter design
_FIT_DB = 40.0               # match ISO 9613-1 (< 1 dB) wherever the loss is at most this
_DEEP_DB = 60.0              # beyond: penalise only too little attenuation (target min(loss, this))
_DEEP_SLOPE = 24.0           # dB/oct: the deep target rises from 40 dB no faster than the low-pass can
_DIST_STEP = 1.01            # 1 % distance quantization of the design cache
_BW4_Q = (0.5411961, 1.3065630)     # 4th-order Butterworth as two biquads
_LATTICE_OCT = 0.4           # shelf corners sit on 1000 Hz * 2^(0.4 k)
_GAIN_STEP_DB = 0.25         # tabulated shelf cuts 0 .. _GAIN_MAX_DB
_GAIN_MAX_DB = 80.0
_LP_STEP_OCT = 0.05          # tabulated low-pass corners 100 Hz * 2^(0.05 j)
_N_FIT_F = 96


def _shelf_bank(corners, cut_db, sr: float) -> np.ndarray:
    """RBJ high shelves (S = 1) cutting `cut_db` (>= 0) at `corners`, vectorized -> (..., 6)."""
    A = 10.0 ** (-np.asarray(cut_db, dtype=float) / 40.0)
    w0 = 2.0 * np.pi * np.asarray(corners, dtype=float) / sr
    cw, sw = np.cos(w0), np.sin(w0)
    sa = np.sqrt(A) * sw * math.sqrt(2.0)
    a0 = (A + 1) - (A - 1) * cw + sa
    rows = np.stack([A * ((A + 1) + (A - 1) * cw + sa), -2.0 * A * ((A - 1) + (A + 1) * cw),
                     A * ((A + 1) + (A - 1) * cw - sa), a0, 2.0 * ((A - 1) - (A + 1) * cw),
                     (A + 1) - (A - 1) * cw - sa], axis=-1)
    return rows / a0[..., None]


def _butter4_lowpass(fc, sr: float) -> np.ndarray:
    """4th-order Butterworth low-pass as two biquads; `fc` scalar -> (2, 6), array -> (..., 2, 6)."""
    w0 = 2.0 * np.pi * np.asarray(fc, dtype=float) / sr
    cw, sw = np.cos(w0), np.sin(w0)
    rows = []
    for q in _BW4_Q:
        al = sw / (2.0 * q)
        a0 = 1.0 + al
        rows.append(np.stack([(1 - cw) / 2, 1 - cw, (1 - cw) / 2, a0, -2 * cw, 1 - al], axis=-1) / a0[..., None])
    return np.stack(rows, axis=-2)


def _att_db(rows: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Attenuation (dB, > 0 = cut) of biquad rows (..., 6) at cos(w) = c1, cos(2w) = c2 -> (..., F)."""
    b0, b1, b2, a0, a1, a2 = (rows[..., i][..., None] for i in range(6))
    num = b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (b0 * b1 + b1 * b2) * c1 + 2.0 * b0 * b2 * c2
    den = a0 * a0 + a1 * a1 + a2 * a2 + 2.0 * (a0 * a1 + a1 * a2) * c1 + 2.0 * a0 * a2 * c2
    return 10.0 * np.log10(np.maximum(den, 1e-300) / np.maximum(num, 1e-300))


class _DesignTables:
    """Per-sample-rate lookup of shelf and low-pass responses on the fit frequency grid.

    The fitter evaluates the cascade hundreds of times; tabulating each lattice shelf's
    response against its cut (0.25 dB steps) and the low-pass against its corner (0.05 oct)
    turns every evaluation (and its exact Jacobian) into a gather plus linear interpolation.
    """

    def __init__(self, sr: int):
        self.sr = float(sr)
        self.f = np.geomspace(20.0, 0.45 * sr, _N_FIT_F)
        w = 2.0 * np.pi * self.f / sr
        c1, c2 = np.cos(w), np.cos(2.0 * w)
        k_lo = int(math.ceil(math.log2(25.0 / 1000.0) / _LATTICE_OCT))
        k_hi = int(math.floor(math.log2(0.42 * sr / 1000.0) / _LATTICE_OCT))
        self.corners = 1000.0 * 2.0 ** (_LATTICE_OCT * np.arange(k_lo, k_hi + 1))
        self.gains = np.arange(0.0, _GAIN_MAX_DB + 0.5 * _GAIN_STEP_DB, _GAIN_STEP_DB)
        rows = _shelf_bank(self.corners[:, None], self.gains[None, :], self.sr)        # (K, G, 6)
        self.shelf = _att_db(rows, c1, c2)                                               # (K, G, F)
        j_hi = int(math.floor(math.log2(0.46 * sr / 100.0) / _LP_STEP_OCT))
        self.lp_log = math.log(100.0) + _LP_STEP_OCT * math.log(2.0) * np.arange(j_hi + 1)
        self.lp = _att_db(_butter4_lowpass(np.exp(self.lp_log), self.sr), c1, c2).sum(axis=-2)   # (J, F)

    def shelves(self, idx: np.ndarray, g: np.ndarray):
        """(responses (K, F), d response / d gain (K, F)) of lattice shelves `idx` at cuts `g`."""
        x = np.clip(g, 0.0, _GAIN_MAX_DB) / _GAIN_STEP_DB
        i0 = np.minimum(x.astype(int), self.gains.size - 2)
        t = (x - i0)[:, None]
        lo = self.shelf[idx, i0]
        hi = self.shelf[idx, i0 + 1]
        return lo + (hi - lo) * t, (hi - lo) / _GAIN_STEP_DB

    def lowpass(self, log_fc: float):
        x = (log_fc - self.lp_log[0]) / (self.lp_log[1] - self.lp_log[0])
        x = min(max(x, 0.0), self.lp_log.size - 1.000001)
        j = int(x)
        t = x - j
        lo, hi = self.lp[j], self.lp[j + 1]
        return lo + (hi - lo) * t, (hi - lo) / (self.lp_log[1] - self.lp_log[0])


@lru_cache(maxsize=4)
def _design_tables(sr: int) -> _DesignTables:
    return _DesignTables(sr)


def _fit_absorption(loss_db: np.ndarray, tab: _DesignTables) -> np.ndarray:
    """Fit lattice high shelves (+ a Butterworth low-pass for the steep top) to `loss_db`.

    The ISO loss grows ~f^2, i.e. its slope in dB/octave keeps growing (~55 dB/oct at a
    40 dB loss), which no single second-order section can follow. Lattice corners are
    used every 0.4 octave where the loss exceeds 10 dB, every 0.8 octave down to 2 dB and
    every 1.6 octave below; their cuts (plus the low-pass corner when the top of the band
    is lost anyway) are solved by Levenberg-Marquardt on the tabulated exact responses.
    Inside the fit region (loss <= _FIT_DB) the residual is two-sided, above it only too
    little attenuation is penalised, against a target that ramps from 40 dB at 24 dB/oct
    (capped at min(loss, 60 dB)).
    """
    f, sr = tab.f, tab.sr
    fit_sel = loss_db <= _FIT_DB
    use_lp = bool(loss_db[-1] > _DEEP_DB)
    f_lo = float(np.interp(0.03, loss_db, f)) if loss_db[0] < 0.03 else float(f[0])
    f_hi = float(np.interp(_FIT_DB, loss_db, f)) if use_lp else float(tab.corners[-1])
    lat = tab.corners
    k_top = int(np.searchsorted(lat, f_hi * 2.0 ** (0.5 * _LATTICE_OCT)) - 1)
    k_top = min(max(k_top, 0), lat.size - 1)
    idx = [k_top]
    while lat[idx[-1]] > f_lo and idx[-1] > 0:
        lc = float(np.interp(lat[idx[-1]], f, loss_db))
        idx.append(max(idx[-1] - (1 if lc >= 10.0 else (2 if lc >= 2.0 else 4)), 0))
    idx = np.array(sorted(set(idx)), dtype=int)
    K = idx.size
    level = np.interp(np.log(lat[idx]), np.log(f), np.minimum(loss_db, _FIT_DB))
    g0 = np.maximum(np.diff(np.concatenate(([0.0], level))), 0.0)
    target = np.minimum(loss_db, _DEEP_DB)
    if use_lp:
        # past the knee the ISO loss climbs ~3x per octave (60+ dB/oct); asking the cascade to
        # follow it forces the error into the fitted band, so the deep target ramps up from
        # 40 dB at the 4th-order low-pass's own 24 dB/oct
        ramp = _FIT_DB + _DEEP_SLOPE * np.log2(np.maximum(f / f_hi, 1.0))
        target = np.where(fit_sel, target, np.minimum(target, ramp))
    w = np.where(fit_sel, 1.0, 0.25)
    lp_lo, lp_hi = float(tab.lp_log[0]), float(tab.lp_log[-1])

    def evaluate(gv, lc):
        P, dP = tab.shelves(idx, gv)
        a = P.sum(axis=0)
        dL = None
        if use_lp:
            L, dL = tab.lowpass(lc)
            a = a + L
        return a, dP, dL

    def residual(a):
        r = a - target
        return np.where(fit_sel, r, np.minimum(r, 0.0)) * w

    def solve(g, lpc):
        a, dP, dL = evaluate(g, lpc)
        r = residual(a)
        cost = float(r @ r)
        lam = 1e-3
        for _ in range(40):
            active = np.where(fit_sel, 1.0, (a < target).astype(float)) * w
            J = dP * active
            if use_lp:
                J = np.vstack([J, (dL * active)[None, :]])
            J = J.T
            JtJ, Jtr = J.T @ J, J.T @ r
            damp = np.diag(np.diag(JtJ)) + 1e-9 * np.eye(JtJ.shape[0])
            improved, rel = False, 0.0
            while lam < 1e9:
                dx = -np.linalg.solve(JtJ + lam * damp, Jtr)
                g2 = np.clip(g + dx[:K], 0.0, _GAIN_MAX_DB)
                l2 = min(max(lpc + dx[K], lp_lo), lp_hi) if use_lp else lpc
                a2, dP2, dL2 = evaluate(g2, l2)
                r2 = residual(a2)
                c2 = float(r2 @ r2)
                if c2 < cost:
                    rel = (cost - c2) / max(cost, 1e-12)
                    g, lpc, a, dP, dL, r, cost = g2, l2, a2, dP2, dL2, r2, c2
                    lam = max(lam / 5.0, 1e-9)
                    improved = True
                    break
                lam *= 5.0
            if not improved or rel < 2e-3 or cost < 1e-4:
                break
        return g, lpc, a, cost

    def score(a):
        # the contract: in-band error (to 10 kHz) first, then any shortfall in the deep band
        sel = fit_sel & (f <= 10000.0)
        e_fit = float(np.max(np.abs(a - loss_db)[sel], initial=0.0))
        short = float(np.max((target - a)[~fit_sel], initial=0.0))
        return e_fit + max(short - 2.0, 0.0)

    lp0 = math.log(min(1.4 * f_hi, 0.45 * sr)) if use_lp else 0.0
    g, lpc, a, _ = solve(g0, lp0)
    if use_lp:
        # the knee has local minima: if the fit is poor, restart the low-pass corner half an
        # octave lower and higher and keep the best in-band fit
        best = score(a)
        for mult in (1.0, 2.0):
            if best <= 0.8:
                break
            alt = solve(g0, min(max(math.log(mult * f_hi), lp_lo), lp_hi))
            s = score(alt[2])
            if s < best:
                best = s
                g, lpc, a, _ = alt
    keep = g > 0.02
    rows = _shelf_bank(lat[idx][keep], g[keep], sr) if np.any(keep) else _PASSTHROUGH.copy()
    if use_lp:
        rows = np.vstack([rows, _butter4_lowpass(math.exp(lpc), sr)])
    return rows


@lru_cache(maxsize=2048)
def _air_sos_cached(dq: int, tq: int, hq: int, sr: int) -> tuple:
    d = _DIST_STEP ** dq
    tab = _design_tables(int(sr))
    loss = iso9613_alpha_db_per_m(tab.f, tq * 0.5, float(hq)) * d
    if float(loss.max()) < 0.05:
        return tuple(map(tuple, _PASSTHROUGH))
    return tuple(map(tuple, _fit_absorption(loss, tab)))


def air_absorption_sos(distance_m: float, temp_c: float, rh_pct: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Biquad cascade matching the ISO 9613-1 loss over `distance_m` (see `_fit_absorption`).

    Within ~1 dB of alpha(f) * distance up to 10 kHz wherever that loss is <= 40 dB (worst
    1.1 dB over a 600-design scan); where the loss is larger the filter attenuates by at
    least ~38 dB, rising ~24 dB/oct towards min(loss, 60 dB). Passthrough under 1 m.
    Designs are cached on a 1 % distance / 0.5 C / 1 % RH grid.
    """
    d = float(distance_m)
    if d < 1.0:
        return _PASSTHROUGH.copy()
    dq = int(round(math.log(d) / math.log(_DIST_STEP)))
    tq = int(round(float(temp_c) * 2.0))
    hq = int(round(min(max(float(rh_pct), 1.0), 100.0)))
    return np.array(_air_sos_cached(dq, tq, hq, int(sr)), dtype=float)


class AirAbsorptionFilter:
    """Stateful ISO 9613-1 absorption filter for a moving / changing source.

    Redesigns only when distance or humidity change by more than 5 % (or temperature by
    more than 1 C); the new filter is warmed up on the previous block's input and
    crossfaded in over one block, so parameter changes never click.
    """

    _WARM = 1024

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = int(sr)
        self._sos = None
        self._zi = None
        self._params = None
        self._tail = None

    def reset(self) -> None:
        self._sos = None
        self._zi = None
        self._params = None
        self._tail = None

    def _changed(self, d: float, t: float, h: float) -> bool:
        if self._params is None:
            return True
        d0, t0, h0 = self._params
        if abs(d - d0) > 0.05 * max(d0, 1e-3):
            return True
        if abs(t - t0) > 1.0:
            return True
        return abs(h - h0) > 0.05 * max(h0, 1.0)

    def _zeros(self, sos: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.zeros((sos.shape[0],) + x.shape[:-1] + (2,))

    @staticmethod
    def _flush(zi: np.ndarray) -> np.ndarray:
        # on silence the recursive state decays into subnormals (~1e-308), which run ~100x
        # slower on x86; anything below 1e-30 is inaudible, so snap it to zero
        np.copyto(zi, 0.0, where=np.abs(zi) < _DENORMAL_FLUSH)
        return zi

    def process(self, x: np.ndarray, distance_m: float, temp_c: float, rh_pct: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] == 0:
            return x.copy()
        d, t, h = float(distance_m), float(temp_c), float(rh_pct)
        if self._zi is not None and self._zi.shape[1:-1] != x.shape[:-1]:
            self.reset()
        if self._sos is None:
            self._sos = air_absorption_sos(d, t, h, self.sr)
            self._zi = self._zeros(self._sos, x)
            self._params = (d, t, h)
            y, zi = sosfilt(self._sos, x, axis=-1, zi=self._zi)
            self._zi = self._flush(zi)
        elif self._changed(d, t, h):
            new_sos = air_absorption_sos(d, t, h, self.sr)
            new_zi = self._zeros(new_sos, x)
            if self._tail is not None and self._tail.shape[:-1] == x.shape[:-1]:
                _, new_zi = sosfilt(new_sos, self._tail, axis=-1, zi=new_zi)
            y_old, _ = sosfilt(self._sos, x, axis=-1, zi=self._zi)
            y_new, new_zi = sosfilt(new_sos, x, axis=-1, zi=new_zi)
            fade = np.linspace(0.0, 1.0, x.shape[-1])
            y = y_old * (1.0 - fade) + y_new * fade
            self._sos, self._zi, self._params = new_sos, self._flush(new_zi), (d, t, h)
        else:
            y, zi = sosfilt(self._sos, x, axis=-1, zi=self._zi)
            self._zi = self._flush(zi)
        self._tail = x[..., -self._WARM:].copy()
        return y


# ------------------------------------------------------------------ spreading
def spreading_gain(distance_m: float, ref_m: float = 1.0, kind: str = "point") -> float:
    """Amplitude gain relative to `ref_m`: point 1/r, line 1/sqrt(r), plane constant."""
    r = max(float(distance_m), 0.05)
    ref = max(float(ref_m), 1e-6)
    k = str(kind).lower()
    if k == "point":
        return ref / r
    if k == "line":
        return math.sqrt(ref / r)
    if k == "plane":
        return 1.0
    raise ValueError(f"unknown spreading kind {kind!r}")


# ------------------------------------------------------------------ refraction
_DTDZ_K_PER_M = {"stable": +0.06, "neutral": -0.0098, "unstable": -0.06}
_LN_10_OVER_Z0REF = math.log(10.0 / 0.03)


def shadow_distance_m(source_h_m: float, receiver_h_m: float, temp_c: float, lapse_k_per_km: float = 9.8) -> float:
    """Distance to the upward-refraction shadow zone: sqrt(2c/a)(sqrt h_s + sqrt h_r).

    a = (c / 2T) * lapse (1/s). Returns inf when there is no lapse (no shadow).
    """
    if lapse_k_per_km <= 0.0:
        return math.inf
    T = float(temp_c) + 273.15
    c = speed_of_sound(temp_c)
    a = c / (2.0 * T) * (lapse_k_per_km / 1000.0)
    return math.sqrt(2.0 * c / a) * (math.sqrt(max(source_h_m, 0.0)) + math.sqrt(max(receiver_h_m, 0.0)))


def sound_speed_gradient(wind_component_mps: float, stability: str, source_h_m: float,
                         receiver_h_m: float, temp_c: float) -> float:
    """Effective linear sound-speed gradient a = dc/dz (1/s) along the path.

    Temperature part from the stability class; wind part from the log-law shear
    du/dz = u / (z ln(10/z0)) evaluated at the geometric-mean path height.
    """
    T = float(temp_c) + 273.15
    c = speed_of_sound(temp_c)
    dtdz = _DTDZ_K_PER_M.get(str(stability).lower(), _DTDZ_K_PER_M["neutral"])
    a_t = c / (2.0 * T) * dtdz
    z_m = max(2.0, math.sqrt(max(source_h_m, 0.5) * max(receiver_h_m, 0.5)))
    a_w = float(wind_component_mps) / (z_m * _LN_10_OVER_Z0REF)
    return a_t + a_w


def refraction_gain_db(distance_m: float, wind_component_mps: float, stability: str,
                       source_h_m: float, receiver_h_m: float, temp_c: float) -> float:
    """Excess level (dB) from refraction, relative to a homogeneous atmosphere.

    `wind_component_mps` is the wind along the source->receiver direction (+ downwind).
    Downwind / inversion: up to +10 dB (spherical -> cylindrical spreading beyond the
    channel range sqrt(8 R_c h)). Upwind / lapse: shadow zone beyond the shadow
    distance, down to -20 dB (turbulent scattering caps the loss).
    """
    r = max(float(distance_m), 0.0)
    if r <= 1.0:
        return 0.0
    hs = max(float(source_h_m), 0.0)
    hr = max(float(receiver_h_m), 0.0)
    a = sound_speed_gradient(wind_component_mps, stability, hs, hr, temp_c)
    c = speed_of_sound(temp_c)
    if a < -1e-6:
        r_s = math.sqrt(2.0 * c / -a) * (math.sqrt(max(hs, 0.05)) + math.sqrt(max(hr, 0.05)))
        x = r / r_s
        if x <= 0.7:
            return 0.0
        return float(max(-20.0, -20.0 * (1.0 - math.exp(-2.0 * (x - 0.7)))))
    if a > 1e-6:
        r_curv = c / a
        h_m = max(0.5 * (hs + hr), 1.0)
        r_c = math.sqrt(8.0 * r_curv * h_m)
        if r <= r_c:
            return 0.0
        return float(min(10.0, 10.0 * math.log10(r / r_c)))
    return 0.0


def flash_to_bang_s(distance_km: float, temp_c: float) -> float:
    """Seconds between seeing a flash and hearing its thunder at `distance_km`."""
    return max(float(distance_km), 0.0) * 1000.0 / speed_of_sound(temp_c)


# ------------------------------------------------------------------ ground effect
# Effective flow resistivity, kPa s m^-2 (Embleton/Attenborough tables); inf = acoustically hard
GROUND_FLOW_RESISTIVITY = {
    "snow_fresh": 15.0,
    "snow_old": 60.0,
    "snow": 30.0,
    "forest_floor": 40.0,
    "leaves": 40.0,
    "needles": 30.0,
    "grass": 200.0,
    "lawn": 200.0,
    "soil": 500.0,
    "mud": 800.0,
    "sand": 400.0,
    "gravel": 400.0,
    "asphalt": 20000.0,
    "concrete": 50000.0,
    "tile": 50000.0,
    "water": math.inf,
    "ice": math.inf,
    "puddle": math.inf,
}


def _delany_bazley_z(f: np.ndarray, sigma_kpa: float) -> np.ndarray:
    """Normalized surface impedance (e^{-i w t} convention), clamped to the validity range."""
    x = np.clip(f / sigma_kpa, 8.3, 833.0)          # 0.01 <= rho0 f / sigma <= 1 with sigma in kPa s/m^2
    return 1.0 + 9.08 * x ** -0.75 + 1j * 11.9 * x ** -0.73


def ground_transfer(f_hz, distance_m: float, source_h_m: float, receiver_h_m: float,
                    ground: str, temp_c: float = 15.0) -> np.ndarray:
    """Complex direct+reflected field relative to the direct path, 1 + Q (r1/r2) e^{ik(r2-r1)}."""
    f = np.maximum(np.asarray(f_hz, dtype=float), 1.0)
    d = max(float(distance_m), 0.1)
    hs = max(float(source_h_m), 0.0)
    hr = max(float(receiver_h_m), 0.0)
    r1 = math.hypot(d, hs - hr)
    r2 = math.hypot(d, hs + hr)
    c = speed_of_sound(temp_c)
    k = 2.0 * np.pi * f / c
    sigma = GROUND_FLOW_RESISTIVITY.get(str(ground).lower(), 200.0)
    if not math.isfinite(sigma):
        q = np.ones_like(k, dtype=complex)
    else:
        sin_psi = (hs + hr) / r2 if r2 > 0 else 0.0
        z = _delany_bazley_z(f, sigma)
        rp = (z * sin_psi - 1.0) / (z * sin_psi + 1.0)
        beta = 1.0 / z
        w = 0.5 * (1.0 + 1j) * np.sqrt(k * r2) * (sin_psi + beta)
        with np.errstate(all="ignore"):
            fw = 1.0 + 1j * math.sqrt(math.pi) * w * wofz(w)
        fw = np.where(np.isfinite(fw), fw, 1.0)
        q = rp + (1.0 - rp) * fw
        mag = np.abs(q)
        q = np.where(mag > 1.2, q / np.maximum(mag, 1e-12) * 1.2, q)
    return 1.0 + q * (r1 / r2) * np.exp(1j * k * (r2 - r1))


def ground_effect_sos(distance_m: float, source_h_m: float, receiver_h_m: float, ground: str,
                      sr: int = SAMPLE_RATE, temp_c: float = 15.0) -> np.ndarray:
    """Low-shelf + first interference dip fitted to the two-ray ground transfer.

    The response is normalized to the incoherent (energy) sum, so above the first dip
    the filter is flat; higher-order comb ripples are omitted because turbulence and
    finite source size smear them in practice.
    """
    d = float(distance_m)
    if d < 1.0 or (source_h_m + receiver_h_m) <= 0.0:
        return _PASSTHROUGH.copy()
    f = np.geomspace(20.0, min(20000.0, 0.45 * sr), 240)
    h = ground_transfer(f, d, source_h_m, receiver_h_m, ground, temp_c)
    r1 = math.hypot(d, source_h_m - receiver_h_m)
    r2 = math.hypot(d, source_h_m + receiver_h_m)
    sigma = GROUND_FLOW_RESISTIVITY.get(str(ground).lower(), 200.0)
    if math.isfinite(sigma):
        z = _delany_bazley_z(f, sigma)
        sin_psi = (source_h_m + receiver_h_m) / r2
        q_mag = np.abs((z * sin_psi - 1.0) / (z * sin_psi + 1.0))
    else:
        q_mag = np.ones_like(f)
    incoherent = 10.0 * np.log10(1.0 + (q_mag * r1 / r2) ** 2)
    target = 20.0 * np.log10(np.abs(h) + 1e-12) - incoherent

    rows = []
    band = (f >= 40.0) & (f <= min(4000.0, 0.4 * sr))
    f_dip = None
    if np.any(band):
        idx_band = np.nonzero(band)[0]
        first = np.nonzero(band & (target < -3.0))[0]
        if first.size:
            # first interference notch: walk down from where the response first dips
            i_min = int(first[0])
            while i_min + 1 < len(f) and band[i_min + 1] and target[i_min + 1] < target[i_min]:
                i_min += 1
        else:
            i_min = int(idx_band[np.argmin(target[band])])
        depth = float(max(target[i_min], -20.0))
        if depth < -1.5:
            f_dip = float(f[i_min])
            half = 0.5 * depth
            lo_i = i_min
            while lo_i > 0 and target[lo_i] < half:
                lo_i -= 1
            hi_i = i_min
            while hi_i < len(f) - 1 and target[hi_i] < half:
                hi_i += 1
            bw = max(float(f[hi_i] - f[lo_i]), 1e-3)
            q = min(max(f_dip / bw, 0.4), 3.0)
            rows.append(_peaking(f_dip, depth, q, sr))
    f_shelf = 0.6 * f_dip if f_dip is not None else 150.0
    lf_sel = f <= max(0.5 * (f_dip if f_dip is not None else 200.0), 25.0)
    if np.any(lf_sel):
        lf_db = float(np.clip(np.mean(target[lf_sel]), -6.0, 6.0))
        if abs(lf_db) > 0.5:
            rows.insert(0, _low_shelf(max(f_shelf, 30.0), lf_db, sr))
    return np.array(rows) if rows else _PASSTHROUGH.copy()
