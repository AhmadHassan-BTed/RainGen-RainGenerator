"""Head-related impulse responses for binaural rendering.

Built-in model: the structural spherical-head model of Brown & Duda (1998), "A structural
model for binaural sound synthesis", IEEE Trans. Speech Audio Process. 6(5):476-488.
  * Head shadow, one pole / one zero:  H(w, t) = (1 + j a(t) w / 2w0) / (1 + j w / 2w0),
    w0 = c / r_head, a(t) = (1 + a_min/2) + (1 - a_min/2) cos(t / t_min * 180 deg),
    a_min = 0.1, t_min = 150 deg, t = angle between source and ear axis.
  * ITD per ear (Woodworth & Schlosberg ray model):
    tau(t) = -(r/c) cos t for t < 90 deg, (r/c)(t - 90 deg) otherwise.
  * Shoulder/torso echo, delay 1.2 ms (180 - th)/180 [1 - 0.00004 ((ph - 80) 180/(180 + th))^2],
    low-passed, faded out for sources below the shoulders.
  * Five pinna echoes tau_k = A_k cos(th/2) sin(D_k (90 - ph)) + B_k samples @ 44.1 kHz with
    rho = (0.5, -1, 0.5, -0.25, 0.25), A = (1, 5, 5, 5, 5), B = (2, 4, 7, 11, 13),
    D = (1, 0.5, 0.5, 0.5, 0.5) (Brown & Duda Table I), mirrored for rear sources, plus a
    rear pinna-shadow low-pass.
  * Ears 10 deg behind the interaural axis (+-100 deg azimuth).
The set is diffuse-field equalised with a minimum-phase filter (cepstral method) so that the
direction-averaged power per ear is 1/2, matching the loudspeaker decoders' energy.

SOFA (AES69) files are read lazily with h5py (preferred) or netCDF4.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import numpy as np

from ..core.constants import SAMPLE_RATE
from .sh import directions, fibonacci_sphere, unit_vectors

HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND = 343.0
EAR_AZ_DEG = 100.0
BULK_DELAY_S = 1.0e-3
_ALPHA_MIN = 0.1
_THETA_MIN = math.radians(150.0)
# (rho, A, B, D) per pinna echo, delays in samples at 44.1 kHz
_PINNA_ECHOES = (
    (0.5, 1.0, 2.0, 1.0),
    (-1.0, 5.0, 4.0, 0.5),
    (0.5, 5.0, 7.0, 0.5),
    (-0.25, 5.0, 11.0, 0.5),
    (0.25, 5.0, 13.0, 0.5),
)


def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


def smooth_fractional_octave(spec: np.ndarray, freqs: np.ndarray, fraction: float = 3.0) -> np.ndarray:
    """Moving average over +-1/(2*fraction) octave along the last axis (works for complex)."""
    F = freqs.shape[0]
    half = 2.0 ** (0.5 / fraction)
    f = np.maximum(freqs, freqs[1] if F > 1 else 1.0)
    lo = np.searchsorted(freqs, f / half, side="left")
    hi = np.searchsorted(freqs, f * half, side="right")
    lo = np.clip(lo, 0, F - 1)
    hi = np.clip(hi, lo + 1, F)
    cs = np.concatenate([np.zeros(spec.shape[:-1] + (1,), dtype=spec.dtype), np.cumsum(spec, axis=-1)], axis=-1)
    return (cs[..., hi] - cs[..., lo]) / (hi - lo)


def minimum_phase_spectrum(mag: np.ndarray, nfft: int) -> np.ndarray:
    """Minimum-phase complex spectrum (rfft bins) with the given magnitude (homomorphic method)."""
    mag = np.maximum(np.asarray(mag, dtype=np.float64), 1e-8)
    logm = np.log(mag)
    full = np.concatenate([logm, logm[..., -2:0:-1]], axis=-1) if nfft % 2 == 0 else \
        np.concatenate([logm, logm[..., :0:-1]], axis=-1)
    cep = np.fft.ifft(full, axis=-1).real
    w = np.zeros(nfft)
    w[0] = 1.0
    w[1:(nfft + 1) // 2] = 2.0
    if nfft % 2 == 0:
        w[nfft // 2] = 1.0
    return np.exp(np.fft.fft(cep * w, axis=-1))[..., : nfft // 2 + 1]


def _fade_out(ir: np.ndarray, length: int) -> np.ndarray:
    out = ir[..., :length].copy()
    nf = max(4, length // 8)
    out[..., -nf:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, nf)))
    return out


def spherical_head_response(az, el, freqs, *, head_radius_m: float = HEAD_RADIUS_M,
                            c: float = SPEED_OF_SOUND, pinna: float = 0.5, torso: float = 0.35,
                            bulk_delay_s: float = BULK_DELAY_S) -> np.ndarray:
    """Brown-Duda transfer functions -> complex array (P, 2, F) for ears (left, right)."""
    az = np.atleast_1d(np.asarray(az, dtype=np.float64))
    el = np.atleast_1d(np.asarray(el, dtype=np.float64))
    az, el = np.broadcast_arrays(az, el)
    u = unit_vectors(az, el)                                  # (P, 3)
    f = np.asarray(freqs, dtype=np.float64)[None, :]          # (1, F)
    w = 2.0 * math.pi * f
    w0 = c / head_radius_m
    P = u.shape[0]
    out = np.empty((P, 2, f.shape[1]), dtype=np.complex128)
    el_deg = np.degrees(el)
    for ear, sign in ((0, 1.0), (1, -1.0)):
        ear_az = sign * math.radians(EAR_AZ_DEG)
        ear_vec = np.array([math.cos(ear_az), math.sin(ear_az), 0.0])
        inc = np.arccos(np.clip(u @ ear_vec, -1.0, 1.0))[:, None]            # (P, 1)
        alpha = (1.0 + _ALPHA_MIN / 2.0) + (1.0 - _ALPHA_MIN / 2.0) * np.cos(inc / _THETA_MIN * math.pi)
        H = (1.0 + 1j * alpha * w / (2.0 * w0)) / (1.0 + 1j * w / (2.0 * w0))
        tau = np.where(inc < math.pi / 2.0, -(head_radius_m / c) * np.cos(inc),
                       (head_radius_m / c) * (inc - math.pi / 2.0))
        # interaural-polar coordinates seen from this ear
        lat = np.degrees(np.arcsin(np.clip(sign * u[:, 1], -1.0, 1.0)))[:, None]   # +90 at this ear
        pol = np.degrees(np.arctan2(u[:, 2], u[:, 0]))[:, None]                   # 0 front, 90 up, 180 back
        phi = np.where(np.abs(pol) <= 90.0, pol, np.sign(pol) * (180.0 - np.abs(pol)))
        if pinna > 0.0:
            acc = np.zeros_like(H)
            for rho, A, B, D in _PINNA_ECHOES:
                tk = (A * np.cos(np.radians(lat) / 2.0) * np.sin(np.radians(D * (90.0 - phi))) + B) / 44100.0
                acc += rho * np.exp(-1j * w * tk)
            H = H * (1.0 + pinna * acc)
        if torso > 0.0:
            th = np.clip(lat, -90.0, 90.0)
            ph = el_deg[:, None]
            t_sh = 1.2e-3 * (180.0 - np.abs(th)) / 180.0 * (1.0 - 0.00004 * ((ph - 80.0) * 180.0 / (180.0 + np.abs(th))) ** 2)
            t_sh = np.maximum(t_sh, 1.0e-4)
            above = np.clip((ph + 45.0) / 30.0, 0.0, 1.0)
            above = above * above * (3.0 - 2.0 * above)
            H = H * (1.0 + torso * above * np.exp(-1j * w * t_sh) / (1.0 + 1j * f / 3000.0))
        rear = np.clip(-u[:, 0], 0.0, 1.0)[:, None]
        H = H / (1.0 + 1j * rear * f / 6000.0)
        out[:, ear, :] = H * np.exp(-1j * w * (bulk_delay_s + tau))
    return out


class HRTFSet:
    """HRIRs measured or modelled on a set of directions (listener frame, radians)."""

    def __init__(self, az, el, ir, sr: int, weights=None, name: str = ""):
        self.az = np.asarray(az, dtype=np.float64).reshape(-1)
        self.el = np.asarray(el, dtype=np.float64).reshape(-1)
        self.ir = np.asarray(ir, dtype=np.float64)          # (P, 2, L)
        self.sr = int(sr)
        self.name = name
        if weights is None:
            weights = voronoi_weights(self.az, self.el)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.weights = w / np.sum(w)
        self._xyz = unit_vectors(self.az, self.el)

    @property
    def length(self) -> int:
        return int(self.ir.shape[-1])

    def nearest(self, az, el) -> np.ndarray:
        q = unit_vectors(np.atleast_1d(az), np.atleast_1d(el))
        return np.argmax(q @ self._xyz.T, axis=1)

    def hrirs(self, az, el) -> np.ndarray:
        """Nearest-neighbour HRIRs for the requested directions -> (P, 2, L)."""
        return self.ir[self.nearest(az, el)]

    def reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Whole set with quadrature weights, for diffuse-field targets."""
        return self.az, self.el, self.weights, self.ir


def voronoi_weights(az, el, probes: int = 6000) -> np.ndarray:
    """Approximate solid-angle weights by assigning a dense uniform probe grid to nearest points."""
    xyz = unit_vectors(np.asarray(az), np.asarray(el))
    pa, pe = fibonacci_sphere(probes)
    probe = unit_vectors(pa, pe)
    counts = np.zeros(xyz.shape[0])
    for s in range(0, probes, 512):
        idx = np.argmax(probe[s:s + 512] @ xyz.T, axis=1)
        counts += np.bincount(idx, minlength=xyz.shape[0])
    counts = np.maximum(counts, 0.5)
    return counts / counts.sum()


def _diffuse_eq(H: np.ndarray, weights: np.ndarray, freqs: np.ndarray, nfft: int) -> np.ndarray:
    """Minimum-phase diffuse-field equaliser (F,) for spectra H (P, 2, F)."""
    p = np.einsum("p,pef->f", weights, (H.real ** 2 + H.imag ** 2)) / 2.0
    p = smooth_fractional_octave(p, freqs, 3.0)
    return minimum_phase_spectrum(np.sqrt(0.5 / np.maximum(p, 1e-12)), nfft)


class BuiltinHRTF:
    """Analytic Brown-Duda spherical-head HRTFs, evaluated at any direction."""

    def __init__(self, sr: int = SAMPLE_RATE, length: int = 256, head_radius_m: float = HEAD_RADIUS_M,
                 c: float = SPEED_OF_SOUND, pinna: float = 0.5, torso: float = 0.35,
                 diffuse_eq: bool = True, reference_points: int = 480):
        self.sr = int(sr)
        self._length = int(length)
        self.kw = dict(head_radius_m=head_radius_m, c=c, pinna=pinna, torso=torso)
        self.nfft = _next_pow2(2 * self._length)
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.sr)
        self.name = "builtin-spherical-head"
        ra, re = fibonacci_sphere(reference_points)
        self._ref_az, self._ref_el = ra, re
        self._ref_w = np.full(ra.shape[0], 1.0 / ra.shape[0])
        H = spherical_head_response(ra, re, self.freqs, **self.kw)
        self._eq = _diffuse_eq(H, self._ref_w, self.freqs, self.nfft) if diffuse_eq else np.ones(self.freqs.shape[0])
        self._ref_ir = self._to_ir(H * self._eq[None, None, :])

    @property
    def length(self) -> int:
        return self._length

    def _to_ir(self, H: np.ndarray) -> np.ndarray:
        return _fade_out(np.fft.irfft(H, self.nfft, axis=-1), self._length)

    def response(self, az, el) -> np.ndarray:
        """Equalised complex responses (P, 2, F) on this object's FFT grid."""
        return spherical_head_response(az, el, self.freqs, **self.kw) * self._eq[None, None, :]

    def hrirs(self, az, el) -> np.ndarray:
        return self._to_ir(self.response(az, el))

    def reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._ref_az, self._ref_el, self._ref_w, self._ref_ir


def spherical_head_hrir(az, el, sr: int = SAMPLE_RATE, length: int = 256, **kw) -> np.ndarray:
    """Built-in HRIRs: scalar direction -> (2, L); arrays -> (P, 2, L)."""
    model = _builtin_cached(int(sr), int(length), tuple(sorted(kw.items())))
    ir = model.hrirs(az, el)
    if np.ndim(az) == 0 and np.ndim(el) == 0:
        return ir[0]
    return ir


@lru_cache(maxsize=8)
def _builtin_cached(sr: int, length: int, kw: tuple) -> BuiltinHRTF:
    return BuiltinHRTF(sr, length, **dict(kw))


def _attr_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, np.ndarray):
        v = v.ravel()
        return _attr_str(v[0]) if v.size else ""
    return str(v)


def _read_sofa_arrays(path: str) -> tuple[np.ndarray, float, np.ndarray, str, str]:
    try:
        import h5py  # type: ignore
    except ImportError:
        h5py = None
    if h5py is not None:
        with h5py.File(path, "r") as fh:
            ir = np.asarray(fh["Data.IR"], dtype=np.float64)
            fs = float(np.asarray(fh["Data.SamplingRate"]).ravel()[0])
            pos_ds = fh["SourcePosition"]
            pos = np.asarray(pos_ds, dtype=np.float64)
            ptype = _attr_str(pos_ds.attrs.get("Type", b"spherical"))
            units = _attr_str(pos_ds.attrs.get("Units", b"degree, degree, metre"))
        return ir, fs, pos, ptype, units
    try:
        import netCDF4  # type: ignore
    except ImportError as exc:
        raise ImportError("Reading SOFA HRTFs needs h5py or netCDF4: pip install h5py") from exc
    ds = netCDF4.Dataset(path, "r")
    try:
        ir = np.asarray(ds.variables["Data.IR"][:], dtype=np.float64)
        fs = float(np.asarray(ds.variables["Data.SamplingRate"][:]).ravel()[0])
        var = ds.variables["SourcePosition"]
        pos = np.asarray(var[:], dtype=np.float64)
        ptype = str(getattr(var, "Type", "spherical"))
        units = str(getattr(var, "Units", "degree, degree, metre"))
    finally:
        ds.close()
    return ir, fs, pos, ptype, units


def load_sofa(path: str, sr: Optional[int] = None, length: Optional[int] = None,
              diffuse_eq: bool = True) -> HRTFSet:
    """Load a SimpleFreeFieldHRIR SOFA file into an HRTFSet (resampled to `sr`, optionally DF-equalised)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ir, fs, pos, ptype, units = _read_sofa_arrays(path)
    if ir.ndim != 3 or ir.shape[1] < 2:
        raise ValueError(f"{path}: expected Data.IR with shape (M, 2, N), got {ir.shape}")
    ir = ir[:, :2, :]
    if pos.ndim == 1:
        pos = pos[None, :]
    if pos.shape[0] == 1 and ir.shape[0] > 1:
        pos = np.repeat(pos, ir.shape[0], axis=0)
    if ptype.lower().startswith("cart"):
        az, el = directions(pos[:, :3] / np.maximum(np.linalg.norm(pos[:, :3], axis=1, keepdims=True), 1e-12))
    else:
        if "rad" in units.lower():
            az, el = pos[:, 0], pos[:, 1]
        else:
            az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    sr = int(sr or fs)
    if int(round(fs)) != sr:
        from math import gcd

        from scipy.signal import resample_poly
        fsi = int(round(fs))
        g = gcd(sr, fsi)
        ir = resample_poly(ir, sr // g, fsi // g, axis=-1)
    L = int(length or min(ir.shape[-1], _next_pow2(int(0.0105 * sr))))
    if ir.shape[-1] < L:
        ir = np.concatenate([ir, np.zeros(ir.shape[:-1] + (L - ir.shape[-1],))], axis=-1)
    weights = voronoi_weights(az, el)
    if diffuse_eq:
        nfft = _next_pow2(2 * max(L, ir.shape[-1]))
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        H = np.fft.rfft(ir, nfft, axis=-1)
        eq = _diffuse_eq(H, weights, freqs, nfft)
        ir = np.fft.irfft(H * eq[None, None, :], nfft, axis=-1)
    ir = _fade_out(ir, L)
    return HRTFSet(az, el, ir, sr, weights=weights, name=os.path.basename(path))


def get_hrtf(spec="builtin", sr: int = SAMPLE_RATE, length: int = 256):
    """'builtin' -> BuiltinHRTF; a path to a .sofa file -> HRTFSet; HRTF objects pass through."""
    if isinstance(spec, (HRTFSet, BuiltinHRTF)):
        return spec
    if spec is None or str(spec).lower() in ("", "builtin", "default", "spherical", "spherical_head"):
        return _builtin_cached(int(sr), int(length), ())
    return load_sofa(str(spec), sr=sr)
