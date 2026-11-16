"""Real spherical harmonics in the AmbiX convention (ACN order, SN3D normalisation).

    Y_n^m(az, el) = N_n^|m| P_n^|m|(sin el) * { cos(m az)   m >= 0
                                               { sin(|m| az) m <  0
    N_n^|m| = sqrt((2 - delta_m0) (n - |m|)! / (n + |m|)!)     (Schmidt semi-normalised, SN3D)
    ACN index = n^2 + n + m

No Condon-Shortley phase (Nachbar, Zotter, Deleflie & Sontacchi 2011, "AmbiX - a suggested
ambisonics format"). Associated Legendre functions come from the standard stable three-term
recurrence, so any order works (the engine uses up to 7). With SN3D the addition theorem
reads sum_m Y_n^m(u) Y_n^m(v) = P_n(u.v), and a uniform diffuse field has per-channel
variance 1/(2n+1) relative to W.

Also here: max-rE order weights (Zotter & Frank 2012, "All-round ambisonic panning and
decoding"), spherical-cap order weights for source extent, uniform sphere samplings
(spherical Fibonacci, the 26-point degree-7 Lebedev rule) and SH-domain rotations.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

MAX_ORDER = 7


def n_channels(order: int) -> int:
    """Number of SH channels for a given order: (order + 1)^2."""
    return (int(order) + 1) ** 2


def order_of(channels: int) -> int:
    """Ambisonic order for a channel count (inverse of n_channels)."""
    order = int(round(math.sqrt(channels))) - 1
    if (order + 1) ** 2 != channels:
        raise ValueError(f"{channels} channels is not a full-sphere ambisonic channel count")
    return order


@lru_cache(maxsize=None)
def _acn_nm(order: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.empty(n_channels(order), dtype=np.int64)
    m = np.empty_like(n)
    for deg in range(order + 1):
        for mm in range(-deg, deg + 1):
            k = deg * deg + deg + mm
            n[k] = deg
            m[k] = mm
    n.setflags(write=False)
    m.setflags(write=False)
    return n, m


def acn_nm(order: int) -> tuple[np.ndarray, np.ndarray]:
    """(degree n, index m) for every ACN channel up to `order`."""
    return _acn_nm(int(order))


def acn_order_index(order: int) -> np.ndarray:
    """Degree n of each ACN channel, e.g. [0, 1, 1, 1, 2, ...]."""
    return _acn_nm(int(order))[0]


@lru_cache(maxsize=None)
def _sn3d_norm(order: int) -> tuple[tuple[float, ...], ...]:
    rows = []
    for n in range(order + 1):
        row = []
        for m in range(n + 1):
            delta = 1.0 if m == 0 else 0.0
            row.append(math.sqrt((2.0 - delta) * math.factorial(n - m) / math.factorial(n + m)))
        rows.append(tuple(row))
    return tuple(rows)


def _sh_terms(order: int, az, el, cos, sin) -> list:
    """Generic evaluation; works on Python floats (math) or numpy arrays (np)."""
    x = sin(el)
    c = cos(el)
    norm = _sn3d_norm(order)
    # P[l][m] without Condon-Shortley phase
    P = [[0.0] * (order + 1) for _ in range(order + 1)]
    pmm = 1.0
    for m in range(order + 1):
        if m > 0:
            pmm = pmm * (2 * m - 1) * c
        P[m][m] = pmm
        if m + 1 <= order:
            P[m + 1][m] = x * (2 * m + 1) * pmm
        for l in range(m + 2, order + 1):
            P[l][m] = ((2 * l - 1) * x * P[l - 1][m] - (l + m - 1) * P[l - 2][m]) / (l - m)
    cosm = [1.0] + [cos(m * az) for m in range(1, order + 1)]
    sinm = [0.0] + [sin(m * az) for m in range(1, order + 1)]
    out = [0.0] * n_channels(order)
    for n in range(order + 1):
        base = n * n + n
        out[base] = norm[n][0] * P[n][0]
        for m in range(1, n + 1):
            nm = norm[n][m] * P[n][m]
            out[base + m] = nm * cosm[m]
            out[base - m] = nm * sinm[m]
    return out


def sh_real(order: int, az, el) -> np.ndarray:
    """Real SN3D spherical harmonics in ACN order.

    Scalar az/el (radians, listener frame: az CCW from front, el up) -> shape ((order+1)^2,).
    Array az/el (broadcastable) -> shape ((order+1)^2, *broadcast_shape).
    """
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    if np.ndim(az) == 0 and np.ndim(el) == 0:
        return np.array(_sh_terms(order, float(az), float(el), math.cos, math.sin), dtype=np.float64)
    a, e = np.broadcast_arrays(np.asarray(az, dtype=np.float64), np.asarray(el, dtype=np.float64))
    terms = _sh_terms(order, a, e, np.cos, np.sin)
    out = np.empty((len(terms),) + a.shape, dtype=np.float64)
    for k, t in enumerate(terms):
        out[k] = t
    return out


def sh_matrix(order: int, az, el) -> np.ndarray:
    """SH evaluated on a list of directions -> shape (P, (order+1)^2)."""
    az = np.atleast_1d(np.asarray(az, dtype=np.float64))
    el = np.atleast_1d(np.asarray(el, dtype=np.float64))
    return sh_real(order, az, el).reshape(n_channels(order), -1).T


def sh_from_xyz(order: int, xyz) -> np.ndarray:
    """SH for unit vectors (..., 3) in the listener frame -> ((order+1)^2, ...)."""
    v = np.asarray(xyz, dtype=np.float64)
    az = np.arctan2(v[..., 1], v[..., 0])
    el = np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1]))
    return sh_real(order, az, el)


def legendre(nmax: int, x):
    """Legendre polynomials P_0..P_nmax at x (scalar or array) -> list."""
    out = [1.0 + 0.0 * x]
    if nmax >= 1:
        out.append(x + 0.0 * x)
    for n in range(2, nmax + 1):
        out.append(((2 * n - 1) * x * out[n - 1] - (n - 1) * out[n - 2]) / n)
    return out


def per_acn(order_weights, order: int) -> np.ndarray:
    """Expand per-degree weights (order+1,) to per-ACN-channel weights."""
    w = np.asarray(order_weights, dtype=np.float64)
    return w[acn_order_index(order)]


@lru_cache(maxsize=None)
def _maxre(order: int) -> tuple[float, ...]:
    if order <= 0:
        return (1.0,)
    # r_E = largest root of P_{N+1}; cos(137.9 deg / (N + 1.51)) is Zotter & Frank's approximation of it
    x = float(np.max(np.polynomial.legendre.legroots([0.0] * (order + 1) + [1.0]).real))
    return tuple(float(v) for v in legendre(order, x))


def maxre_weights(order: int) -> np.ndarray:
    """Per-degree max-rE weights g_n = P_n(r_E), r_E the largest root of P_{N+1} (Zotter & Frank 2012)."""
    return np.array(_maxre(int(order)), dtype=np.float64)


def maxre_energy_compensation(order: int) -> float:
    """Gain that restores diffuse-field energy after max-rE weighting (SN3D)."""
    n = np.arange(order + 1)
    g = maxre_weights(order)
    return float(np.sqrt(np.sum(2 * n + 1) / np.sum((2 * n + 1) * g * g)))


def cap_weights(order: int, half_angle: float) -> np.ndarray:
    """Per-degree SH coefficients of a uniform spherical cap of half-angle alpha (normalised so c_0 = 1).

    c_n = [P_{n-1}(cos a) - P_{n+1}(cos a)] / [(2n + 1)(1 - cos a)]; c_n -> 1 as a -> 0 and
    c_n -> 0 (n >= 1) for the whole sphere. Used to give a source angular extent.
    """
    a = float(half_angle)
    if a <= 1e-6:
        return np.ones(order + 1)
    x = math.cos(min(a, math.pi))
    P = legendre(order + 1, x)
    out = np.ones(order + 1)
    den = 1.0 - x
    for n in range(1, order + 1):
        out[n] = (P[n - 1] - P[n + 1]) / ((2 * n + 1) * den)
    return out


def sn3d_to_n3d(order: int) -> np.ndarray:
    """Per-ACN gains converting SN3D to N3D (sqrt(2n+1))."""
    return np.sqrt(2.0 * acn_order_index(order) + 1.0)


def unit_vectors(az, el) -> np.ndarray:
    """Directions (radians) -> unit vectors (..., 3) in the listener frame."""
    az = np.asarray(az, dtype=np.float64)
    el = np.asarray(el, dtype=np.float64)
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=-1)


def directions(xyz) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (..., 3) -> (az, el) radians."""
    v = np.asarray(xyz, dtype=np.float64)
    return np.arctan2(v[..., 1], v[..., 0]), np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1]))


@lru_cache(maxsize=32)
def _fibonacci(n: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / n
    golden = math.pi * (3.0 - math.sqrt(5.0))
    az = np.mod(i * golden + math.pi, 2.0 * math.pi) - math.pi
    el = np.arcsin(z)
    az.setflags(write=False)
    el.setflags(write=False)
    return az, el


def fibonacci_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Near-uniform spherical Fibonacci sampling with n points -> (az, el) radians."""
    az, el = _fibonacci(int(n))
    return az.copy(), el.copy()


def dense_grid(order: int) -> tuple[np.ndarray, np.ndarray]:
    """A dense near-uniform grid that integrates products of order-`order` SH accurately."""
    return fibonacci_sphere(max(240, 12 * n_channels(order)))


@lru_cache(maxsize=1)
def _lebedev26():
    pts, wts = [], []
    for ax in range(3):
        for s in (1.0, -1.0):
            v = [0.0, 0.0, 0.0]
            v[ax] = s
            pts.append(v)
            wts.append(1.0 / 21.0)
    r2 = 1.0 / math.sqrt(2.0)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for sa in (1.0, -1.0):
            for sb in (1.0, -1.0):
                v = [0.0, 0.0, 0.0]
                v[a] = sa * r2
                v[b] = sb * r2
                pts.append(v)
                wts.append(4.0 / 105.0)
    r3 = 1.0 / math.sqrt(3.0)
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                pts.append([sx * r3, sy * r3, sz * r3])
                wts.append(9.0 / 280.0)
    xyz = np.array(pts)
    az, el = directions(xyz)
    return az, el, np.array(wts)


def lebedev26() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """26-point Lebedev rule (exact to degree 7, enough for order-3 sampling): (az, el, weights summing to 1)."""
    az, el, w = _lebedev26()
    return az.copy(), el.copy(), w.copy()


def yaw_rotation_matrix(order: int, yaw: float) -> np.ndarray:
    """SH-domain rotation that turns the sound field by `yaw` radians about +z (sources move CCW)."""
    N = n_channels(order)
    R = np.zeros((N, N))
    R[0, 0] = 1.0
    for n in range(1, order + 1):
        base = n * n + n
        R[base, base] = 1.0
        for m in range(1, n + 1):
            c, s = math.cos(m * yaw), math.sin(m * yaw)
            ic, is_ = base + m, base - m
            # cos(m(az+y)) = cos(m az)cos(m y) - sin(m az)sin(m y); sin(m(az+y)) = sin cos + cos sin
            R[ic, ic] = c
            R[ic, is_] = -s
            R[is_, ic] = s
            R[is_, is_] = c
    return R


def _rot3(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    # positive pitch lifts the front (+x) towards +z
    Ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


@lru_cache(maxsize=256)
def _rotation_cached(order: int, yaw: float, pitch: float, roll: float) -> np.ndarray:
    az, el = fibonacci_sphere(max(64, 4 * n_channels(order) + 16))
    u = unit_vectors(az, el)
    ur = u @ _rot3(yaw, pitch, roll).T
    Y = sh_real(order, az, el)
    Yr = sh_from_xyz(order, ur)
    M = Yr @ np.linalg.pinv(Y)
    M[np.abs(M) < 1e-12] = 0.0
    M.setflags(write=False)
    return M


def rotation_matrix(order: int, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """SH-domain rotation of the sound field (yaw about +z, pitch lifting the front, roll about +x).

    Solved exactly by sampling: SH of degree <= N span an invariant subspace, so
    M = Y(R u) pinv(Y(u)) on any grid with enough points is the exact rotation.
    To compensate a head rotation, pass the negated head angles.
    """
    key = (int(order), round(float(yaw), 9), round(float(pitch), 9), round(float(roll), 9))
    if key[2] == 0.0 and key[3] == 0.0:
        return yaw_rotation_matrix(key[0], key[1])
    return np.array(_rotation_cached(*key))
