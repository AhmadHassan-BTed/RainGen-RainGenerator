"""Thunder: fractal lightning channels, N-wave synthesis, propagation, rumble and echoes.

Design: "signal-based, physics-inspired" (Reiss, Tez & Selfridge, AES 150, 2021: signal
models beat pure physics in listening tests; Fineberg, Walters & Reiss, AES 152, 2022).

Physics:
* Channel geometry: a multi-scale tortuous 3-D walk with a mean direction change of ~16 deg
  between segments (Few 1969; Hill 1968). Cloud-to-ground flashes start in the negative
  charge region (4.5-6.5 km), descend through a 1-3 km cloud base to the ground, with
  in-cloud horizontal branches and air-terminated downward branches; intracloud flashes run
  largely horizontally at 5-8 km. The full in-cloud channel lengthens the roll.
* Source: Few's relaxation radius R0 = sqrt(E0 / (pi p0)) for an energy per unit length
  E0 gives a source frequency f ~ c0 sqrt(p0 / E0) (~485 Hz at 50 kJ/m): bigger strokes
  are deeper. Each segment radiates a weak-shock N-wave whose period lengthens with
  distance as (r / R0)^(1/4) while its impulse is conserved, so its peak falls as
  r^-1 (R0 / r)^(1/4) (Few 1969; Ribner & Roy 1982, JASA 72(6):1911).
* A straight segment of length l seen at angle psi to its axis spreads its arrivals over
  l |cos psi| / c (Ribner & Roy): broadside segments clap, end-on segments roll. Each
  segment is the N-wave convolved with that arrival window.
* Propagation per segment: delay r / c(T) (flash-to-bang 2.91 s/km at 20 C), ISO 9613-1
  absorption at the segment's distance, spherical spreading, and refraction (lapse rate +
  wind shear) with the upward-refraction shadow beyond r_s = sqrt(2c/a)(sqrt h_s + sqrt h_r)
  (raingen.physics.atmosphere.refraction_gain_db; ~25 km for a 5 km source in a 9.8 K/km
  lapse). Lacroix et al. (2018) find no sharp spectral peaks, so none are imposed.
* Cloud-to-ground flashes carry 3-5 return strokes 40-80 ms apart along the main channel
  below the cloud base (Rakov & Uman, *Lightning: Physics and Effects*, 2003).
* Signal layer (Fineberg-style): a crack/tearing "strike" for near flashes, a "rumbler"
  (noise with the N-wave source spectrum, peak at f ~ 0.66 / T for the energy-weighted
  N-wave period T, falling as 1/f above, with growl modulation following the smoothed
  arrival energy) and a "deepener" (sub-bass afterglow), plus terrain/building echoes and a
  diffuse terrain tail. The rumbler and deepener get the same ISO 9613-1 absorption as the
  N-waves arriving at that instant (r = c (t - t_flash)), so distant thunder is dull.

Events last up to 40 s; each is rendered lazily into an accumulator that spans blocks. Stem
'thunder'.

Marker contract (for captions, visuals and render.thunder_one_shots). Every flash, storm-cell
or `strike()` cue, audible or not, emits exactly two markers:

* EventMarker('lightning') at t = t_flash (in the block containing t_flash), with the channel
  polylines (listener frame, metres), type 'cg' | 'ic', the strike/centre distance and
  bearing, and `flash_to_bang_s` = t_first - t_flash;
* EventMarker('thunder') at t = t_first = t_flash + r_min / c(T), the first arrival from the
  closest audible channel point (slant range r_min, reported as `distance_km`), emitted in
  the block containing t_first, with `audible` (False when every segment is below the
  audibility floor, e.g. beyond the refraction shadow), `peak_db_spl` and `duration_s`.

For a CG flash r_min is about the slant range to the ground strike point (the channel can
meander a little closer near the ground); for an IC flash the channel runs at 5-8 km, so the
thunder marker comes >= ~15 s after the flash even overhead. A render that must contain the
thunder marker of a flash at t_flash has to last at least t_flash + flash_to_bang_s.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import fft as sfft
from scipy import signal

from raingen.core.constants import AMBI_ORDER, BLOCK, N_AMBI, P0, SAMPLE_RATE, speed_of_sound
from raingen.core.types import BlockContext, Emission, EventMarker, bearing_to_azimuth
from raingen.physics.atmosphere import iso9613_alpha_db_per_m, shadow_distance_m, sound_speed_gradient
from raingen.spatial.sh import fibonacci_sphere, sh_real

from .places import Room
from .wind import FramedNoise, Ramp, StatefulSOS, highpass_sos, lowpass_sos, noise_rms_gain, sh_gains

MEAN_TURN_DEG = 16.0          # mean direction change between channel segments (Few 1969)
MAX_EVENT_S = 40.0
# Pa: calibration of the segment peak. Set so a typical CG flash (intensity 0.5, E0 ~ 70 kJ/m) heard
# from ~1 km gives a median 125 ms (fast) maximum of ~100-104 dB SPL over many channels: an acoustic
# efficiency of ~0.2 % of the energy of the ~3 km channel below the cloud base (Few 1969) spread over
# a few seconds gives ~95-100 dB Leq at 1 km. Thunder level data are sparse: treat as +-10 dB.
N_WAVE_K = 4.5
ENERGY_FACTOR = {"main": 1.0, "cloud": 0.55, "branch": 0.3, "ic": 0.5}
RUMBLER_POWER = 0.8           # rumbler power relative to the smoothed N-wave arrival power
DEEPENER_POWER = 0.45
BIN_S = 0.01
AFTERGLOW_TAU_S = 1.0         # power decay time constant of the deepener's afterglow (4.3 dB/s)
RUMBLE_HOP = 4096             # rumbler/deepener noise frames of 2 x 4096 samples: 5.9 Hz resolution below 60 Hz
CRACK_RANGE_M = 3500.0        # near-strike crack layer below this closest range
TEAR_RANGE_M = 1500.0         # the tearing clicks fade in smoothly below this closest range
N_BEAMS = 64                  # direction grid the segment N-waves are summed on (~25 deg spacing)


def stroke_energy_j_per_m(intensity: float) -> float:
    """Stroke intensity 0..1 -> energy per unit channel length E0 (1e4 .. 5e5 J/m, log scale)."""
    return 1.0e4 * 50.0 ** min(max(float(intensity), 0.0), 1.0)


def source_frequency_hz(e0_j_per_m: float, temp_c: float = 15.0) -> float:
    """Few's characteristic thunder frequency f ~ c0 sqrt(p0 / E0)."""
    return speed_of_sound(temp_c) * math.sqrt(P0 / max(e0_j_per_m, 1.0))


# --------------------------------------------------------------------------
# Channel geometry
# --------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-12 else np.array([0.0, 0.0, -1.0])


def _turn(rng: np.random.Generator, d: np.ndarray, angle: float) -> np.ndarray:
    """Rotate unit vector d by `angle` about a random axis perpendicular to it (Rodrigues)."""
    a = rng.standard_normal(3)
    a -= float(a @ d) * d
    a = _unit(a)
    return _unit(d * math.cos(angle) + np.cross(a, d) * math.sin(angle))


class _Normals:
    """Standard normals drawn from `rng` in chunks (the walks below need ~5 per step)."""

    def __init__(self, rng: np.random.Generator, chunk: int = 1280):
        self.rng = rng
        self.chunk = chunk
        self.buf = rng.standard_normal(chunk).tolist()
        self.i = 0

    def take(self, k: int) -> list:
        if self.i + k > len(self.buf):
            self.buf = self.buf[self.i:] + self.rng.standard_normal(self.chunk).tolist()
            self.i = 0
        out = self.buf[self.i:self.i + k]
        self.i += k
        return out


def _norm3(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 1e-12:
        return 0.0, 0.0, -1.0
    return x / r, y / r, z / r


def _turn3(d: tuple, angle: float, ax: float, ay: float, az: float) -> tuple[float, float, float]:
    """Rodrigues rotation of unit d by `angle` about the random axis (ax, ay, az) made perpendicular to d."""
    dx, dy, dz = d
    p = ax * dx + ay * dy + az * dz
    ax, ay, az = _norm3(ax - p * dx, ay - p * dy, az - p * dz)
    c, s_ = math.cos(angle), math.sin(angle)
    return _norm3(dx * c + (ay * dz - az * dy) * s_, dy * c + (az * dx - ax * dz) * s_, dz * c + (ax * dy - ay * dx) * s_)


def _walk(rng, start, end, step: float, steer: float, z_floor: float,
          turn_deg: float = MEAN_TURN_DEG, max_steps: int = 5000) -> np.ndarray:
    """Tortuous walk from start to end: half-normal turns (mean `turn_deg`) steered to the target.

    `rng` is a numpy Generator or a shared _Normals stream. Scalar float arithmetic: a channel
    has thousands of 3-D steps, where numpy's per-call overhead would dominate.
    """
    nrm = rng if isinstance(rng, _Normals) else _Normals(rng)
    sigma = math.radians(turn_deg) * math.sqrt(math.pi / 2.0)
    ex, ey, ez = (float(v) for v in end)
    px, py, pz = (float(v) for v in start)
    d = _norm3(ex - px, ey - py, ez - pz)
    pts = [(px, py, pz)]
    dist0 = math.sqrt((ex - px) ** 2 + (ey - py) ** 2 + (ez - pz) ** 2)
    # a walker that only steers gently can orbit its target forever: home in over the last few steps
    # and cap the walk at a few times the straight distance (tortuosity, not a random search)
    for _ in range(min(max_steps, int(4.0 * dist0 / step) + 16)):
        tx, ty, tz = ex - px, ey - py, ez - pz
        dist = math.sqrt(tx * tx + ty * ty + tz * tz)
        if dist <= step:
            break
        st = steer if dist > 3.0 * step else max(steer, 0.6)
        d = _norm3((1.0 - st) * d[0] + st * tx / dist, (1.0 - st) * d[1] + st * ty / dist,
                   (1.0 - st) * d[2] + st * tz / dist)
        ax, ay, az, g_turn, g_len = nrm.take(5)
        d = _turn3(d, abs(g_turn * sigma), ax, ay, az)
        L = min(step * math.exp(0.4 * g_len), dist)
        px, py, pz = px + d[0] * L, py + d[1] * L, pz + d[2] * L
        if pz < z_floor:
            pz = z_floor
            d = _norm3(d[0], d[1], abs(d[2]))
        pts.append((px, py, pz))
    pts.append((ex, ey, ez))
    return np.array(pts)


def _tortuous(rng, start, end, z_floor: float, macro: float = 450.0,
              micro: float = 35.0) -> np.ndarray:
    """Two-scale tortuosity: km-scale meander through waypoints, filled with ~35 m zig-zags."""
    nrm = rng if isinstance(rng, _Normals) else _Normals(rng)
    way = _walk(nrm, start, end, macro, 0.3, z_floor)
    return _fill(nrm, way, micro, z_floor)


def _fill(nrm, way: np.ndarray, micro: float, z_floor: float) -> np.ndarray:
    """Micro zig-zags (~35 m steps, 16 deg mean turns) between consecutive waypoints."""
    pts = [way[:1]]
    for a, b in zip(way[:-1], way[1:]):
        seg = _walk(nrm, a, b, micro, 0.18, z_floor)
        pts.append(seg[1:])
    return np.vstack(pts)


def _leader(rng, start, end, z_floor: float, macro: float = 450.0, micro: float = 35.0,
            wander: float = 0.3) -> np.ndarray:
    """The channel below the cloud base: a descending stepped leader that ends at the strike point.

    Macro waypoints are evenly spaced in height (the leader keeps descending: it never climbs or
    crawls along the ground) with a Brownian-bridge lateral meander of `wander` x the macro step
    per level, so the channel wanders a few hundred metres and lands exactly on `end`; micro
    zig-zags fill the gaps as for every other channel part.
    """
    nrm = rng if isinstance(rng, _Normals) else _Normals(rng)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    levels = max(1, int(math.ceil(max(a[2] - b[2], 1.0) / macro)))
    steps = np.array(nrm.take(2 * levels)).reshape(levels, 2) * (wander * macro)
    walk = np.cumsum(steps, axis=0)
    frac = np.arange(1, levels + 1) / levels
    way = a[None, :] + frac[:, None] * (b - a)[None, :]
    way[:, :2] += walk - frac[:, None] * walk[-1][None, :]
    way[-1] = b
    return _fill(nrm, np.vstack([a[None, :], way]), micro, z_floor)


def _free_walk(rng, start, d0, length: float, step: float, bias, bias_w: float,
               z_floor: float, z_ceiling: float = math.inf) -> np.ndarray:
    nrm = rng if isinstance(rng, _Normals) else _Normals(rng)
    sigma = math.radians(MEAN_TURN_DEG) * math.sqrt(math.pi / 2.0)
    px, py, pz = (float(v) for v in start)
    d = _norm3(*(float(v) for v in d0))
    bx, by, bz = _norm3(*(float(v) for v in bias))
    pts = [(px, py, pz)]
    total = 0.0
    while total < length:
        d = _norm3((1.0 - bias_w) * d[0] + bias_w * bx, (1.0 - bias_w) * d[1] + bias_w * by,
                   (1.0 - bias_w) * d[2] + bias_w * bz)
        ax, ay, az, g_turn, g_len = nrm.take(5)
        d = _turn3(d, abs(g_turn * sigma), ax, ay, az)
        L = step * math.exp(0.4 * g_len)
        px, py, pz = px + d[0] * L, py + d[1] * L, pz + d[2] * L
        total += L
        if pz < z_floor:
            break
        if pz > z_ceiling:
            pz = z_ceiling
            d = (d[0], d[1], -abs(d[2]))
        pts.append((px, py, pz))
    return np.array(pts)


def _lightning_tree(rng: np.random.Generator, sx: float, sy: float, cg: bool, z_ground: float,
                    cloud_base_km: Optional[float] = None, top_km: Optional[float] = None,
                    branches: bool = True) -> tuple[list[tuple[np.ndarray, str]], float]:
    """Polylines [(points (P,3), kind)] in the listener frame and the cloud-base height (listener z)."""
    zb_km = float(cloud_base_km) if cloud_base_km else float(rng.uniform(1.0, 3.0))
    z_b = z_ground + 1000.0 * zb_km
    out: list[tuple[np.ndarray, str]] = []
    if cg:
        zt_km = float(top_km) if top_km else float(rng.uniform(4.5, 6.5))
        z_t = max(z_ground + 1000.0 * zt_km, z_b + 1000.0)
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        off = float(rng.uniform(500.0, 3000.0))
        O = np.array([sx + off * math.cos(ang), sy + off * math.sin(ang), z_t])
        B = np.array([sx + float(rng.normal(0.0, 250.0)), sy + float(rng.normal(0.0, 250.0)), z_b])
        S = np.array([sx, sy, z_ground])
        upper = _tortuous(rng, O, B, z_floor=z_b - 300.0)
        lower = _leader(rng, B, S, z_floor=z_ground)
        main = np.vstack([upper, lower[1:]])
        out.append((main, "main"))
        if branches:
            for _ in range(1 + int(rng.poisson(1.5))):          # in-cloud horizontal "spider" branches
                i0 = int(rng.integers(0, max(1, len(upper) - 1)))
                a = float(rng.uniform(0.0, 2.0 * math.pi))
                h = np.array([math.cos(a), math.sin(a), 0.0])
                br = _free_walk(rng, upper[i0], h + np.array([0.0, 0.0, 0.1]), float(rng.uniform(1500.0, 6000.0)),
                                60.0, h, 0.12, z_floor=z_b - 200.0, z_ceiling=z_t + 1500.0)
                if len(br) > 2:
                    out.append((br, "cloud"))
            low_idx = np.nonzero(lower[:, 2] > z_ground + 300.0)[0]
            for _ in range(int(rng.poisson(3.0)) if low_idx.size > 2 else 0):   # air-terminated downward branches
                i0 = int(low_idx[int(rng.integers(0, low_idx.size - 1))])
                d_main = _unit(lower[min(i0 + 1, len(lower) - 1)] - lower[i0])
                d0 = _turn(rng, d_main, math.radians(float(rng.uniform(25.0, 50.0))))
                br = _free_walk(rng, lower[i0], d0, float(rng.uniform(100.0, 900.0)), 30.0,
                                np.array([0.0, 0.0, -1.0]), 0.08, z_floor=z_ground + 20.0)
                if len(br) > 2:
                    out.append((br, "branch"))
    else:
        zc = z_ground + 1000.0 * float(rng.uniform(5.0, 8.0))
        a = float(rng.uniform(0.0, 2.0 * math.pi))
        half = 1000.0 * float(rng.uniform(2.5, 7.5))
        h = np.array([math.cos(a), math.sin(a), 0.0])
        C = np.array([sx, sy, zc])
        E1 = C - half * h + np.array([0.0, 0.0, float(rng.normal(0.0, 400.0))])
        E2 = C + half * h + np.array([0.0, 0.0, float(rng.normal(0.0, 400.0))])
        main = _tortuous(rng, E1, E2, z_floor=z_ground + 3000.0)
        out.append((main, "ic"))
        if branches:
            for _ in range(2 + int(rng.poisson(2.0))):
                i0 = int(rng.integers(0, max(1, len(main) - 1)))
                b = float(rng.uniform(0.0, 2.0 * math.pi))
                hb = np.array([math.cos(b), math.sin(b), 0.0])
                br = _free_walk(rng, main[i0], hb, float(rng.uniform(1000.0, 5000.0)), 60.0, hb, 0.12,
                                z_floor=z_ground + 3000.0, z_ceiling=zc + 2000.0)
                if len(br) > 2:
                    out.append((br, "cloud"))
    return out, z_b


def _strike_xy(distance_km: float, bearing_deg: float, heading_deg: float) -> tuple[float, float]:
    az = bearing_to_azimuth(float(bearing_deg), float(heading_deg))
    d = 1000.0 * max(float(distance_km), 0.05)
    return d * math.cos(az), d * math.sin(az)


def lightning_channel(rng: np.random.Generator, distance_km: float, bearing_deg: float, heading_deg: float,
                      cg: bool, *, listener_height_m: float = 1.7, cloud_base_km: Optional[float] = None,
                      top_km: Optional[float] = None, branches: bool = True) -> np.ndarray:
    """Fractal 3-D lightning channel in the listener frame (metres), shape (M, 3).

    The main channel comes first; branches follow, each preceded by a row of NaN (the
    usual polyline separator: plotting the array draws the bolt with its branches).
    Ground is at z = -listener_height_m. CG: from the in-cloud charge region through a
    1-3 km cloud base to the ground strike point at (distance, bearing). IC: horizontal
    at 5-8 km centred over (distance, bearing).
    """
    sx, sy = _strike_xy(distance_km, bearing_deg, heading_deg)
    polys, _ = _lightning_tree(rng, sx, sy, bool(cg), -float(listener_height_m), cloud_base_km, top_km, branches)
    return join_polylines([p for p, _ in polys])


def join_polylines(polys) -> np.ndarray:
    parts = []
    for k, p in enumerate(polys):
        if k:
            parts.append(np.full((1, 3), np.nan))
        parts.append(np.asarray(p, dtype=float))
    return np.vstack(parts) if parts else np.zeros((0, 3))


def split_polylines(points: np.ndarray) -> list[np.ndarray]:
    """Inverse of `join_polylines`: split an (M, 3) array at NaN rows."""
    pts = np.asarray(points, dtype=float)
    bad = np.any(~np.isfinite(pts), axis=1)
    out, cur = [], []
    for row, b in zip(pts, bad):
        if b:
            if cur:
                out.append(np.array(cur))
            cur = []
        else:
            cur.append(row)
    if cur:
        out.append(np.array(cur))
    return out


# --------------------------------------------------------------------------
# One thunder event
# --------------------------------------------------------------------------

class _BeamAccum:
    """Overlap-add ring buffer of mono signals on a fixed set of directions, in absolute samples.

    Kernels are added lazily (only segments arriving within the next two blocks) and are at
    most a few N-wave periods long, so a short ring suffices; `pop` reads and clears n columns.
    """

    def __init__(self, rows: int, base: int, capacity: int = 16384):
        self.cap = 1 << max(10, int(capacity - 1).bit_length())
        self.buf = np.zeros((rows, self.cap))
        self.base = int(base)          # absolute sample index of the next column to pop
        self.used = 0                  # columns ahead of base holding data

    def _grow(self, need: int) -> None:
        cap = 1 << int(need - 1).bit_length()
        idx = (self.base + np.arange(self.used)) % self.cap
        nb = np.zeros((self.buf.shape[0], cap))
        nb[:, (self.base + np.arange(self.used)) % cap] = self.buf[:, idx]
        self.buf, self.cap = nb, cap

    def add(self, row: int, start: int, y: np.ndarray) -> None:
        off = int(start) - self.base
        if off < 0:                    # pragma: no cover - segments are scheduled ahead of time
            y = y[-off:]
            off = 0
        end = off + y.size
        if end > self.cap:
            self._grow(end)
        i0 = (self.base + off) % self.cap
        k = min(y.size, self.cap - i0)
        self.buf[row, i0:i0 + k] += y[:k]
        if k < y.size:
            self.buf[row, : y.size - k] += y[k:]
        self.used = max(self.used, end)

    def pop(self, n: int) -> np.ndarray:
        """The next n samples of every row (rows, n); advances base."""
        i0 = self.base % self.cap
        if i0 + n <= self.cap:
            out = self.buf[:, i0:i0 + n].copy()
            self.buf[:, i0:i0 + n] = 0.0
        else:
            idx = (i0 + np.arange(n)) % self.cap
            out = self.buf[:, idx]
            self.buf[:, idx] = 0.0
        self.base += n
        self.used = max(self.used - n, 0)
        return out


def refraction_gains_db(r_h: np.ndarray, wind_comp: np.ndarray, stability: str, h_src: np.ndarray,
                        h_rcv: float, temp_c: float) -> np.ndarray:
    """Vectorised raingen.physics.atmosphere.refraction_gain_db (same formula, per segment).

    The temperature gradient and the log-law shear constant are read back from
    sound_speed_gradient, so this follows the atmosphere module's constants.
    """
    r = np.asarray(r_h, dtype=float)
    hs = np.maximum(np.asarray(h_src, dtype=float), 0.0)
    hr = max(float(h_rcv), 0.0)
    a_t = sound_speed_gradient(0.0, stability, 10.0, 10.0, temp_c)
    inv_ln = (sound_speed_gradient(1.0, stability, 10.0, 10.0, temp_c) - a_t) * 10.0   # 1 / ln(10/z0_ref)
    z_m = np.maximum(2.0, np.sqrt(np.maximum(hs, 0.5) * max(hr, 0.5)))
    a = a_t + np.asarray(wind_comp, dtype=float) * inv_ln / z_m
    c = speed_of_sound(temp_c)
    g = np.zeros(np.broadcast(r, a).shape)
    neg = (a < -1e-6) & (r > 1.0)
    if np.any(neg):
        r_s = np.sqrt(2.0 * c / -a[neg]) * (np.sqrt(np.maximum(hs[neg], 0.05)) + math.sqrt(max(hr, 0.05)))
        x = r[neg] / r_s
        g[neg] = np.where(x <= 0.7, 0.0, np.maximum(-20.0, -20.0 * (1.0 - np.exp(-2.0 * (x - 0.7)))))
    pos = (a > 1e-6) & (r > 1.0)
    if np.any(pos):
        r_c = np.sqrt(8.0 * (c / a[pos]) * np.maximum(0.5 * (hs[pos] + hr), 1.0))
        g[pos] = np.where(r[pos] <= r_c, 0.0, np.minimum(10.0, 10.0 * np.log10(np.maximum(r[pos], 1e-9) / r_c)))
    return g


def absorption_gain(f_hz: np.ndarray, distance_m: float, temp_c: float, rh_pct: float) -> np.ndarray:
    """ISO 9613-1 amplitude transmission 10^(-alpha(f) d / 20) on a frequency grid."""
    return 10.0 ** (-iso9613_alpha_db_per_m(f_hz, temp_c, rh_pct) * max(float(distance_m), 0.0) / 20.0)


def _fft_absorb(x: np.ndarray, distance_m: float, temp_c: float, rh_pct: float, sr: int) -> np.ndarray:
    """Apply the ISO 9613-1 magnitude over `distance_m` to a finite signal (zero-phase, by FFT)."""
    N = int(sfft.next_fast_len(x.size + 256, real=True))
    X = sfft.rfft(x, N)
    X *= absorption_gain(np.fft.rfftfreq(N, 1.0 / sr), distance_m, temp_c, rh_pct)
    return sfft.irfft(X, N)[: x.size]


def absorption_lowpass_sos(distance_m: float, temp_c: float, rh_pct: float, sr: int) -> np.ndarray:
    """Closed-form stand-in for the ISO 9613-1 loss of a fixed path: a 2nd-order Butterworth low-pass
    at the frequency where alpha(f) d reaches 3 dB (the loss grows ~f^2, i.e. ~12 dB/octave there)."""
    f = np.geomspace(50.0, 0.45 * sr, 400)
    loss = iso9613_alpha_db_per_m(f, temp_c, rh_pct) * max(float(distance_m), 0.0)
    above = np.nonzero(loss >= 3.0)[0]
    if not above.size:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    return lowpass_sos(float(f[above[0]]), sr)


@lru_cache(maxsize=4096)
def _nwave_integral(nT: int, nr: int) -> np.ndarray:
    """Running sum I of a unit N-wave of nT samples whose shock rises over nr samples.

    The N-wave has zero mean, so I returns to zero at its end; a segment's arrival window of m
    samples smears the N-wave into (I(t) - I(t - m)) / m (Ribner & Roy), i.e. a positive bump
    at the first arrival and a negative one m samples later, each only ~nT long.
    """
    t = (np.arange(nT) + 0.5) / nT
    nw = 1.0 - 2.0 * t
    win = np.hanning(nr + 2)[1:-1]
    nw = np.convolve(nw, win / win.sum())
    cs = np.cumsum(nw)
    cs -= np.linspace(0.0, cs[-1], cs.size)       # remove the rounding residue: I(end) = 0 exactly
    cs.setflags(write=False)
    return cs


def _nwave_shape(nT: int, nr: int, m: int) -> np.ndarray:
    """Unit N-wave (nT samples, rise nr) smeared over an m-sample arrival window (dense form)."""
    I = _nwave_integral(nT, nr)
    if m <= 1:
        return np.diff(np.concatenate([[0.0], I]))
    out = np.zeros(I.size + m)
    out[:I.size] += I
    out[m:m + I.size] -= I
    return out / m


_BEAM_AZ, _BEAM_EL = fibonacci_sphere(N_BEAMS)
_BEAM_U = np.stack([np.cos(_BEAM_EL) * np.cos(_BEAM_AZ), np.cos(_BEAM_EL) * np.sin(_BEAM_AZ), np.sin(_BEAM_EL)], axis=1)
_BEAM_Y = np.asarray(sh_real(AMBI_ORDER, _BEAM_AZ, _BEAM_EL), dtype=float).reshape(N_AMBI, N_BEAMS)


class _ThunderEvent:
    """The acoustic consequence of one flash at one listener, rendered lazily block by block.

    Ribner & Roy: every straight channel segment radiates an N-wave that arrives over the
    window from its closest to its farthest point (r_near / c .. r_far / c). Segment N-waves
    are summed on a 64-direction beam grid (re-encoded to 3rd-order SH each block), and the
    ISO 9613-1 absorption of the sum is applied per block by a 50 %-overlap STFT at the
    distance r = c (t - t_flash) the arrivals of that instant have travelled (all arrivals of
    one instant come from the same range), so per-segment filter designs are never needed.
    """

    def __init__(self, eid: str, t_flash: float, cg: bool, polys, z_cloud_base: float, intensity: float,
                 weather, listener_h: float, sr: int, rng: np.random.Generator):
        self.id = eid
        self.sr = sr
        self.t_flash = float(t_flash)
        temp = float(weather.temperature_c)
        rh = float(weather.humidity_pct)
        self.temp, self.rh = temp, rh
        c = speed_of_sound(temp)
        self.c = c
        z_g = -float(listener_h)
        self.intensity = float(intensity)

        # ---- segment table (vectorized) ----
        a_list, b_list, f_list, main_low = [], [], [], []
        for pts, kind in polys:
            if len(pts) < 2:
                continue
            a_list.append(pts[:-1])
            b_list.append(pts[1:])
            mid_z = 0.5 * (pts[:-1, 2] + pts[1:, 2])
            fac = np.full(len(pts) - 1, ENERGY_FACTOR.get(kind, 0.4))
            if kind == "main":
                fac = np.where(mid_z > z_cloud_base, 0.7, 1.0)
            f_list.append(fac)
            main_low.append((mid_z < z_cloud_base) if kind == "main" else np.zeros(len(pts) - 1, bool))
        a = np.vstack(a_list)
        b = np.vstack(b_list)
        fac = np.concatenate(f_list)
        low = np.concatenate(main_low)
        mid = 0.5 * (a + b)
        vec = b - a
        seg_l = np.maximum(np.linalg.norm(vec, axis=1), 1.0)
        r = np.maximum(np.linalg.norm(mid, axis=1), 30.0)
        u = mid / r[:, None]
        # Ribner & Roy arrival window: from the segment's closest point to its farthest end
        tpar = np.clip(-np.sum(a * vec, axis=1) / np.maximum(np.sum(vec * vec, axis=1), 1e-9), 0.0, 1.0)
        r_near = np.maximum(np.linalg.norm(a + tpar[:, None] * vec, axis=1), 30.0)
        r_far = np.maximum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
        r_far = np.maximum(r_far, r_near)
        e0_base = stroke_energy_j_per_m(intensity)

        # return strokes along the lower main channel
        t_off = np.zeros(len(r))
        e_mult = np.ones(len(r))
        self.strokes = [0.0]
        if cg:
            n_str = min(1 + int(rng.poisson(2.5)), 8)
            tau = 0.0
            idx_low = np.nonzero(low)[0]
            extra = []
            for _ in range(n_str - 1):
                tau += float(np.clip(rng.lognormal(math.log(0.06), 0.5), 0.015, 0.2))
                self.strokes.append(tau)
                extra.append((tau, float(rng.uniform(0.3, 0.7))))
            if idx_low.size and extra:
                reps = [np.arange(len(r))]
                offs = [t_off]
                mults = [e_mult]
                for tau_s, e_s in extra:
                    reps.append(idx_low)
                    offs.append(np.full(idx_low.size, tau_s))
                    mults.append(np.full(idx_low.size, e_s))
                sel = np.concatenate(reps)
                t_off = np.concatenate(offs)
                e_mult = np.concatenate(mults)
                mid, seg_l, r, u, fac = mid[sel], seg_l[sel], r[sel], u[sel], fac[sel]
                r_near, r_far = r_near[sel], r_far[sel]

        E0 = e0_base * fac * e_mult
        R0 = np.sqrt(E0 / (math.pi * P0))
        T0 = np.sqrt(E0 / P0) / c                       # 1 / (c0 sqrt(p0/E0))
        T = np.clip(T0 * (r / R0) ** 0.25, T0, 0.08)
        A = N_WAVE_K * np.sqrt(E0) * (seg_l / r) * (R0 / r) ** 0.25
        dtau = (r_far - r_near) / c

        # refraction (lapse + wind shear, upward-refraction shadow) per segment
        az = np.arctan2(u[:, 1], u[:, 0])
        el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
        r_h = np.hypot(mid[:, 0], mid[:, 1])
        h_src = np.maximum(mid[:, 2] - z_g, 1.0)
        wcomp = float(weather.wind_u10_mps) * np.cos(az - float(weather.wind_azimuth))
        A = A * 10.0 ** (refraction_gains_db(r_h, wcomp, str(weather.stability), h_src, float(listener_h), temp) / 20.0)

        t_start = self.t_flash + r_near / c + t_off
        t_first = float(np.min(t_start))
        keep = (t_start <= t_first + MAX_EVENT_S) & (A > 2e-6)
        order = np.argsort(t_start[keep], kind="stable")
        idx = np.nonzero(keep)[0][order]
        self.A, self.T, self.dtau, self.r = A[idx], T[idx], dtau[idx], r[idx]
        self.az, self.el = az[idx], el[idx]
        self.r_near = r_near[idx]
        self.s_start = np.round(t_start[idx] * sr).astype(np.int64)
        self.beam = np.argmax(u[idx] @ _BEAM_U.T, axis=1) if idx.size else np.zeros(0, dtype=int)
        self.nT = np.maximum(np.ceil(self.T * sr).astype(np.int64), 2)
        rise = np.maximum(3.0 / sr, 2e-4 * np.sqrt(self.r / 1000.0))
        self.nr = np.maximum(3, np.round(rise * sr).astype(np.int64))
        self.m = np.maximum(1, np.round(self.dtau * sr).astype(np.int64))
        self.ptr = 0
        self._bumps()
        self.acc: Optional[_BeamAccum] = None
        self.audible = bool(idx.size) and float(np.max(self.A)) > 2e-5
        self.peak_pa = float(np.max(self.A)) if idx.size else 0.0
        self.r_min = float(np.min(self.r_near)) if idx.size else float(np.min(r_near))
        self.t_first = float(self.t_flash + self.r_min / c) if idx.size else float(t_first)
        i_near = int(np.argmin(self.r_near)) if idx.size else 0
        self.near_dir = (float(self.az[i_near]), float(self.el[i_near])) if idx.size else (0.0, 0.0)
        top_h = float(np.max(h_src))
        self.audible_range_km = shadow_distance_m(top_h, float(listener_h), temp) / 1000.0

        # ---- signal layer: energy envelope, direction track, rumbler, deepener ----
        self.end_sample = int(np.max(self.s_start + self.nT + self.nr + self.m)) if idx.size \
            else int(round(t_first * sr))
        self.t_origin = float(t_first)
        if idx.size:
            self._envelopes(rng)
            self._crack(rng)
        else:
            self.n_bins = 0
            self.crack = None
        self.dir_ramp = Ramp(sh_gains(self.near_dir[0], self.near_dir[1], spread=0.45))
        self.ola: Optional[FramedNoise] = None
        self.growl = None
        self._n = 0
        self._x_cur: Optional[np.ndarray] = None
        self._tail: Optional[np.ndarray] = None

    # -- precomputation -------------------------------------------------------
    def _envelopes(self, rng: np.random.Generator) -> None:
        sr = self.sr
        dur = np.maximum(self.T, self.dtau)
        # Pa^2 s of each segment's actual kernel before absorption: the signal layer gets the ISO 9613-1
        # absorption of its arrival range spectrally, frame by frame (`_rumble_frame`)
        e = self.A ** 2 * self._kernel_energy() / sr
        t0 = self.s_start / sr - self.t_origin
        b0 = np.maximum(np.floor(t0 / BIN_S).astype(int), 0)
        span = np.maximum(np.ceil(dur / BIN_S).astype(int), 1)
        tail_bins = int(9.0 / BIN_S)
        nb = int(np.max(b0 + span)) + tail_bins + 2
        diff = np.zeros(nb + 1)
        rate = e / (span * BIN_S)                        # each segment's energy spread over its duration
        np.add.at(diff, b0, rate)
        np.add.at(diff, b0 + span, -rate)
        P = np.cumsum(diff)[:nb]                         # arrival power, Pa^2
        P = np.maximum(P, 0.0)
        w_r = np.hanning(17)[1:-1]
        Pr = np.convolve(P, w_r / w_r.sum(), mode="same")
        w_d = np.hanning(63)[1:-1]
        Pd = np.convolve(P, w_d / w_d.sum(), mode="same")
        # afterglow: peak hold decaying with AFTERGLOW_TAU_S (power), hold[k] = max_j Pd[j] decay^(k - j)
        # (a running max in logs); short enough that the roll ends with the last arrivals (Ribner & Roy)
        ln_d = -BIN_S / AFTERGLOW_TAU_S
        kk = np.arange(nb) * ln_d
        Pd = np.exp(kk + np.maximum.accumulate(np.log(np.maximum(Pd, 1e-300)) - kk))
        self.env_r = np.sqrt(RUMBLER_POWER * Pr)
        self.env_d = np.sqrt(DEEPENER_POWER * Pd)
        # energy-weighted direction track (the roll moves across the sky)
        vx = np.zeros(nb + 1)
        vy = np.zeros(nb + 1)
        vz = np.zeros(nb + 1)
        ux = np.cos(self.el) * np.cos(self.az)
        uy = np.cos(self.el) * np.sin(self.az)
        uz = np.sin(self.el)
        for comp, arr in ((ux, vx), (uy, vy), (uz, vz)):
            np.add.at(arr, b0, rate * comp)
            np.add.at(arr, b0 + span, -rate * comp)
        w_v = np.hanning(53)[1:-1]
        V = [np.convolve(np.cumsum(arr)[:nb], w_v, mode="same") for arr in (vx, vy, vz)]
        last = np.array([math.cos(self.near_dir[1]) * math.cos(self.near_dir[0]),
                         math.cos(self.near_dir[1]) * math.sin(self.near_dir[0]), math.sin(self.near_dir[1])])
        dirs = np.stack(V, axis=1)
        norms = np.linalg.norm(dirs, axis=1)
        good = norms > 1e-18
        # hold the last direction through silent stretches
        idx_last = np.maximum.accumulate(np.where(good, np.arange(nb), -1))
        held = np.where(good[:, None], dirs / np.maximum(norms, 1e-30)[:, None], 0.0)
        dirs = np.where((idx_last >= 0)[:, None], held[np.maximum(idx_last, 0)], last[None, :])
        self.dir_az = np.arctan2(dirs[:, 1], dirs[:, 0])
        self.dir_el = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
        self.n_bins = nb
        self.env_end_sample = int(round((self.t_origin + nb * BIN_S) * sr))
        self.end_sample = max(self.end_sample + int(0.5 * sr), self.env_end_sample)
        # the rumbler's source spectrum: an N-wave of period T peaks at x = pi f T ~ 2.08 (f ~ 0.66 / T)
        # and falls as 1/f above; T is the energy-weighted period, lengthened with range (Few 1969)
        t_mean = float(np.sum(self.T * e) / max(float(np.sum(e)), 1e-30))
        self.f_nwave = float(np.clip(0.66 / max(t_mean, 1e-4), 15.0, 400.0))

    def _kernel_energy(self) -> np.ndarray:
        """Sum of squares of every segment's unit kernel: the N-wave smeared over its arrival window.

        For m > 1 the kernel is (I(t) - I(t - m)) / m, so sum k^2 = (2 sum I^2 - 2 sum I(t) I(t - m)) / m^2
        (the overlap term vanishes once m exceeds the N-wave): a long window keeps ~2 nT / (15 m) of the
        N-wave's energy, far less than an incoherent T / dtau share.
        """
        out = np.empty(self.nT.size)
        for i in range(self.nT.size):
            I = _nwave_integral(int(self.nT[i]), int(self.nr[i]))
            m = int(self.m[i])
            if m <= 1:
                d = np.diff(I, prepend=0.0)
                out[i] = float(d @ d)
            else:
                cross = float(I[m:] @ I[:-m]) if m < I.size else 0.0
                out[i] = (2.0 * float(I @ I) - 2.0 * cross) / (m * m)
        return out

    def _crack(self, rng: np.random.Generator) -> None:
        """Near-strike crack and tearing (the 'strike' layer)."""
        sr = self.sr
        self.crack = None
        rmin = self.r_min
        if rmin >= CRACK_RANGE_M:
            return
        L = int(0.6 * sr)
        t = np.arange(L) / sr
        tau = 0.05 + 0.1 * rmin / CRACK_RANGE_M
        burst = rng.standard_normal(L) * np.exp(-t / tau) * (1.0 - np.exp(-t / 0.0015))
        burst = signal.sosfilt(highpass_sos(600.0, sr), burst)
        tear = 3.0 * max(1.0 - rmin / TEAR_RANGE_M, 0.0) ** 1.5
        if tear > 0.0:                                   # tearing: dense clicks from the nearest channel parts
            k = int(rng.integers(40, 150))
            offs = np.sort(rng.integers(0, int(0.12 * sr), size=k))
            imp = np.zeros(L)
            np.add.at(imp, offs, rng.lognormal(0.0, 0.6, size=k) * rng.choice([-1.0, 1.0], size=k))
            burst += tear * signal.sosfilt(np.vstack([highpass_sos(1800.0, sr), lowpass_sos(7000.0, sr)]), imp)
        burst = _fft_absorb(burst, rmin, self.temp, self.rh, sr)
        pk = float(np.max(np.abs(burst)))
        if pk <= 0.0:
            return
        near = self.A[self.r_near < 1.2 * rmin]
        a_peak = float(np.max(near)) if near.size else self.peak_pa
        self.crack = burst / pk * 0.5 * a_peak * (1.0 - rmin / CRACK_RANGE_M)
        self.crack_start = int(round((self.t_flash + rmin / self.c) * sr))
        self.crack_Y = sh_gains(self.near_dir[0], self.near_dir[1], spread=0.1)
        self.end_sample = max(self.end_sample, self.crack_start + L)

    # -- per block --------------------------------------------------------------
    def _rumble_filters(self, start: int) -> None:
        sr = self.sr
        self.ola = FramedNoise(2, RUMBLE_HOP, sr)          # voice 0 rumbler, voice 1 deepener
        self._ola_start = int(start)                       # absolute sample of the noise stream's start
        f = np.maximum(self.ola.freqs, 1e-3)
        hp_r = (f / 22.0) ** 4 / (1.0 + (f / 22.0) ** 4)                  # 2nd-order Butterworth |H|^2
        b_r = hp_r / (1.0 + (f / self.f_nwave) ** 2)                      # N-wave tail: amplitude ~ 1/f
        hp_d = (f / 12.0) ** 4 / (1.0 + (f / 12.0) ** 4)
        b_d = hp_d / (1.0 + (f / 55.0) ** 8)                               # sub-bass afterglow below 55 Hz
        src = np.stack([b_r, b_d])
        self._src_mag = np.sqrt(src / self.ola.power(src)[:, None])
        self._alpha_ola = iso9613_alpha_db_per_m(self.ola.freqs, self.temp, self.rh)
        self.growl = StatefulSOS(lowpass_sos(6.0, sr))
        self.growl_g = noise_rms_gain(self.growl.sos)

    def _rumble_frame(self, ctx: BlockContext, k: int) -> tuple:
        """(rng, (2, F) magnitudes) of noise frame k of the rumbler and deepener.

        Source spectra are normalised to unit power, then the ISO 9613-1 absorption over the range the
        arrivals at the frame centre have travelled, r = c (t - t_flash), is applied, so the absorbed
        energy is really lost (distant thunder is quieter as well as duller).
        """
        t_mid = (self._ola_start + (k + 1) * self.ola.hop) / self.sr
        r_m = max(self.c * (t_mid - self.t_flash), 0.0)
        mag = self._src_mag * (10.0 ** (-self._alpha_ola * r_m / 20.0))[None, :]
        return ctx.rng.get(f"thunder.rumble.{self.id}", k), mag

    def _stft_setup(self, n: int) -> None:
        self._n = n
        N = 2 * n
        self._win = np.sin(np.pi * (np.arange(N) + 0.5) / N)
        self._alpha = iso9613_alpha_db_per_m(np.fft.rfftfreq(N, 1.0 / self.sr), self.temp, self.rh)
        self._x_cur = None
        self._tail = None

    def _bumps(self) -> None:
        """Time-ordered kernel list: per segment the +I/m bump at its first arrival and the -I/m bump
        m samples later (or one dense N-wave when its arrival window is a single sample)."""
        dense = self.m <= 1
        seg = np.concatenate([np.arange(self.s_start.size), np.nonzero(~dense)[0]])
        start = np.concatenate([self.s_start, (self.s_start + self.m)[~dense]])
        sign = np.concatenate([np.where(dense, 0.0, 1.0), np.full(int(np.sum(~dense)), -1.0)])
        order = np.argsort(start, kind="stable")
        self.b_seg, self.b_start, self.b_sign = seg[order], start[order], sign[order]
        self.b_ptr = 0

    def _schedule(self, until: int) -> None:
        """Add every kernel starting before absolute sample `until` to the beam accumulator."""
        j = int(np.searchsorted(self.b_start, until, side="left"))
        for k in range(self.b_ptr, j):
            i = int(self.b_seg[k])
            I = _nwave_integral(int(self.nT[i]), int(self.nr[i]))
            sg = float(self.b_sign[k])
            if sg == 0.0:
                self.acc.add(int(self.beam[i]), int(self.b_start[k]), np.diff(I, prepend=0.0) * float(self.A[i]))
            else:
                self.acc.add(int(self.beam[i]), int(self.b_start[k]), I * (sg * float(self.A[i]) / float(self.m[i])))
        self.b_ptr = max(self.b_ptr, j)
        self.ptr = int(np.searchsorted(self.s_start, until, side="left"))

    def _nwaves(self, start: int, n: int) -> np.ndarray:
        """The absorbed N-wave sum for [start, start + n) as an SH block (one block of look-ahead)."""
        if self.acc is None and self.b_start.size and start + 2 * n <= int(self.b_start[0]):
            return np.zeros((N_AMBI, n))                  # still travelling: nothing has arrived yet
        if n != self._n:
            self._stft_setup(n)
        if self.acc is None:
            self.acc = _BeamAccum(N_BEAMS, start)
        if self._x_cur is None:
            self.acc.base = max(self.acc.base, start) if self.acc.used == 0 else self.acc.base
            self._schedule(start + n)
            self._x_cur = _BEAM_Y @ self.acc.pop(n)
            self._tail = np.zeros((N_AMBI, n))
        self._schedule(start + 2 * n)
        x_next = _BEAM_Y @ self.acc.pop(n)
        if not np.any(self._x_cur) and not np.any(x_next) and not np.any(self._tail):
            self._x_cur = x_next
            return np.zeros((N_AMBI, n))
        frame = np.concatenate([self._x_cur, x_next], axis=1) * self._win
        r_now = max(self.c * ((start + n) / self.sr - self.t_flash), 1.0)
        spec = sfft.rfft(frame, axis=1) * (10.0 ** (-self._alpha * r_now / 20.0))
        proc = sfft.irfft(spec, 2 * n, axis=1) * self._win
        out = self._tail + proc[:, :n]
        self._tail = proc[:, n:]
        self._x_cur = x_next
        return out

    def render(self, ctx: BlockContext, start: int, n: int) -> np.ndarray:
        sr = self.sr
        end = start + n
        if self.s_start.size and (self.b_ptr < self.b_start.size or self.acc is not None):
            out = self._nwaves(start, n)
        else:
            out = np.zeros((N_AMBI, n))
        if self.crack is not None:
            a = max(start, self.crack_start)
            b = min(end, self.crack_start + self.crack.size)
            if b > a:
                out[:, a - start:b - start] += np.outer(self.crack_Y, self.crack[a - self.crack_start:b - self.crack_start])
        if self.n_bins and start < self.env_end_sample and end > int(self.t_origin * sr):
            if self.ola is None:
                self._rumble_filters(start)
            tb = ((start + np.arange(n)) / sr - self.t_origin) / BIN_S
            grid = np.arange(self.n_bins)
            env_r = np.interp(tb, grid, self.env_r, left=0.0, right=0.0)
            env_d = np.interp(tb, grid, self.env_d, left=0.0, right=0.0)
            noise = self.ola.process(n, lambda k: self._rumble_frame(ctx, k))
            growl = self.growl.process(ctx.rng.get(f"thunder.growl.{self.id}", ctx.index).standard_normal(n)) \
                / self.growl_g
            rum = noise[0] * env_r * np.maximum(1.0 + 0.6 * growl, 0.0)
            deep = noise[1] * env_d
            kb = int(min(max(tb[n // 2], 0), self.n_bins - 1))
            Yt = sh_gains(float(self.dir_az[kb]), float(self.dir_el[kb]), spread=0.45)
            out += self.dir_ramp(Yt, n) * (rum + deep)[None, :]
        return out

    def finished(self, next_start: int) -> bool:
        pending = self.acc is not None and (self.acc.used > 0 or (self._tail is not None and np.any(self._tail))
                                            or (self._x_cur is not None and np.any(self._x_cur)))
        return self.b_ptr >= self.b_start.size and not pending and next_start >= self.end_sample


# --------------------------------------------------------------------------
# Terrain and building echoes
# --------------------------------------------------------------------------

class _EchoField:
    """Discrete reflections (mountainsides, buildings, walls) and a diffuse terrain tail."""

    MAX_DELAY_S = 8.0

    def __init__(self, scene, layout, rngf, sr: int):
        self.sr = sr
        cfg = rngf.get("thunder.echo.cfg", 0)
        preset = str(scene.place.preset or "").lower()
        taps = []                      # (distance to reflector m, gain, az, el)
        if any(k in preset for k in ("mountain", "ridge", "valley", "hill", "alp", "canyon", "fjord", "cliff")):
            for _ in range(4):
                d = float(cfg.uniform(400.0, 2500.0))
                taps.append((d, float(cfg.uniform(0.35, 0.6)) * 5000.0 / (5000.0 + 2.0 * d),
                             float(cfg.uniform(-math.pi, math.pi)), float(cfg.uniform(0.0, 0.15))))
            tail = (4.0, 0.25)
        elif any(k in preset for k in ("city", "street", "alley", "town", "downtown", "urban", "suburb", "village")):
            for _ in range(4):
                d = float(cfg.uniform(30.0, 200.0))
                taps.append((d, float(cfg.uniform(0.3, 0.45)), float(cfg.uniform(-math.pi, math.pi)), 0.05))
            tail = (2.5, 0.2)
        elif any(k in preset for k in ("forest", "wood", "jungle", "rainforest", "cabin", "camp")):
            for _ in range(2):
                d = float(cfg.uniform(60.0, 200.0))
                taps.append((d, 0.12, float(cfg.uniform(-math.pi, math.pi)), 0.1))
            tail = (1.6, 0.22)
        else:
            d = float(cfg.uniform(800.0, 2500.0))
            taps.append((d, 0.12 * 5000.0 / (5000.0 + 2.0 * d), float(cfg.uniform(-math.pi, math.pi)), 0.03))
            tail = (1.0, 0.1)
        for p in layout or []:
            prm = p.params or {}
            if prm.get("reflector") and p.distance < 80.0:
                taps.append((p.distance, 0.45 if prm.get("facade") else 0.35, p.azimuth, p.elevation))
        self.taps = []
        for d, g, az, el in taps:
            delay = int(round(2.0 * d / 343.0 * sr))
            if delay < 1 or delay > int(self.MAX_DELAY_S * sr):
                continue
            sos = np.vstack([absorption_lowpass_sos(2.0 * d, 15.0, 70.0, sr), lowpass_sos(3500.0, sr)])
            self.taps.append((delay, float(g), sh_gains(az, el, spread=0.3), StatefulSOS(sos)))
        max_d = max([t[0] for t in self.taps], default=0)
        self.size = max_d + 4 * BLOCK + 1
        self.ring = np.zeros(self.size)
        self.w = 0
        self.tail_wet = tail[1]
        self.tail = Room(tail[0], sr, AMBI_ORDER, hf_ratio=0.3, size_m=25.0, directional_input=0.1)
        self.tail_rt = tail[0]
        self.quiet = 10 ** 9
        self.max_delay = max_d

    def process(self, bus: np.ndarray) -> Optional[np.ndarray]:
        n = bus.shape[1]
        loud = bool(np.any(bus[0] != 0.0))
        self.quiet = 0 if loud else self.quiet + n
        if self.quiet > self.max_delay + int((1.5 * self.tail_rt + 0.5) * self.sr):
            return None
        if n > self.size - self.max_delay:
            grow = np.zeros(self.max_delay + 2 * n + 1)
            idx = (self.w - np.arange(self.size, 0, -1)) % self.size
            grow[-self.size:] = self.ring[idx]
            self.ring, self.size, self.w = grow, grow.size, 0
        k = np.arange(n)
        self.ring[(self.w + k) % self.size] = bus[0]
        out = np.zeros((N_AMBI, n))
        for delay, g, Y, lp in self.taps:
            x = self.ring[(self.w - delay + k) % self.size]
            out += np.outer(Y, lp.process(x) * g)
        self.w = (self.w + n) % self.size
        out += self.tail.process(bus)[:N_AMBI] * self.tail_wet
        return out


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

class ThunderGenerator:
    """Thunder from the storm cells in WeatherState.storms (stem 'thunder', SH bus).

    Flashes are drawn exactly as StormSystem.flashes_between does (same RNG layer
    'storm.flash.<cell>' per block slot, Poisson at the cell's flash rate), so audio,
    markers and World.flashes_between agree. Each flash scatters within the cell (~1.2 km),
    is CG with probability cg_fraction, and gets its own channel and event.
    """

    name = "thunder"

    def __init__(self, scene, layout, rngf, sr: int = SAMPLE_RATE, block: Optional[int] = None, max_events: int = 16):
        self.sr = int(sr)
        # flash slots must match the World's StormSystem block: adopted from the first rendered block
        # unless given (the World is built with the engine's block length)
        self._block_fixed = block is not None
        self.block_s = int(block if block is not None else BLOCK) / float(sr)
        self.heading = float(scene.listener.heading_deg)
        self.listener_h = float(scene.listener.height_m)
        self.events: list[_ThunderEvent] = []
        self.max_events = int(max_events)
        self.pending_markers: list[EventMarker] = []
        self.echo = _EchoField(scene, layout, rngf, self.sr)

    def _flashes(self, ctx: BlockContext) -> list:
        w = ctx.weather
        t0 = float(ctx.t0)
        t1 = t0 + ctx.n / float(ctx.sr)
        bd = self.block_s
        k0 = int(math.floor(t0 / bd))
        k1 = int(math.floor((t1 - 1e-12) / bd))
        out = []
        for cell in w.storms or []:
            lam = max(float(cell.flash_rate_per_min), 0.0) / 60.0 * bd
            if lam <= 0.0:
                continue
            for k in range(k0, k1 + 1):
                rng = ctx.rng.get(f"storm.flash.{cell.id}", k)
                count = int(rng.poisson(lam))
                if count == 0:
                    continue
                times = k * bd + rng.uniform(0.0, bd, count)
                gains = rng.lognormal(0.0, 0.3, count)
                for j, (tf, gk) in enumerate(zip(times, gains)):
                    tf = float(tf)
                    if t0 <= tf < t1:
                        si = float(min(max(cell.stroke_intensity * gk, 0.02), 1.0))
                        out.append((tf, cell, si, k, j))
        out.sort(key=lambda e: e[0])
        return out

    def _spawn(self, ctx: BlockContext, tf: float, cell, si: float, k: int, j: int) -> None:
        rng = ctx.rng.get(f"thunder.flash.{cell.id}.{j}", k)
        cg = bool(rng.random() < float(cell.cg_fraction))
        cx, cy = _strike_xy(cell.distance_km, cell.bearing_deg, self.heading)
        sx = cx + float(rng.normal(0.0, 1200.0))
        sy = cy + float(rng.normal(0.0, 1200.0))
        self._launch(ctx, f"{cell.id}.{k}.{j}", str(cell.id), tf, sx, sy, cg, si, rng)

    def strike(self, ctx: BlockContext, t_flash: float, distance_km: float, bearing_deg: float = 0.0,
               intensity: float = 0.5, cg: bool = True, flash_id: str = "cue", seed: int = 0):
        """Launch one flash at an explicit ground point (cue sheets, one-shots, the Composer).

        `distance_km` / `bearing_deg` (true) place the strike point (CG) or the channel centre (IC);
        the channel itself is drawn from the 'thunder.cue.<flash_id>' stream. Markers are emitted
        like a storm-cell flash. Returns the event (None when it is inaudible).
        """
        rng = ctx.rng.get(f"thunder.cue.{flash_id}", int(seed))
        sx, sy = _strike_xy(distance_km, bearing_deg, self.heading)
        si = float(min(max(intensity, 0.02), 1.0))
        return self._launch(ctx, str(flash_id), "cue", float(t_flash), sx, sy, bool(cg), si, rng)

    def _launch(self, ctx: BlockContext, eid: str, cell_id: str, tf: float, sx: float, sy: float, cg: bool,
                si: float, rng: np.random.Generator):
        dh = math.hypot(sx, sy)
        if dh < 250.0:
            s = 250.0 / max(dh, 1e-6)
            sx, sy = sx * s if dh > 1e-6 else 250.0, sy * s
            dh = 250.0
        z_g = -self.listener_h
        polys, z_b = _lightning_tree(rng, sx, sy, cg, z_g)
        ev = _ThunderEvent(eid, tf, cg, polys, z_b, si, ctx.weather, self.listener_h, self.sr, rng)
        az = math.atan2(sy, sx)
        bearing = (self.heading - math.degrees(az)) % 360.0
        channel = [np.round(p[:: (1 if kind in ("main", "ic") else 2)], 1).tolist() for p, kind in polys]
        ctx.markers.append(EventMarker(t=tf, kind="lightning", data={
            "id": eid, "cell": cell_id, "type": "cg" if cg else "ic",
            "distance_km": round(dh / 1000.0, 3), "bearing_deg": round(bearing, 1), "azimuth": round(az, 4),
            "intensity": round(si, 3), "energy_j_per_m": round(stroke_energy_j_per_m(si), 1),
            "strokes_s": [round(s, 4) for s in ev.strokes], "cloud_base_m": round(z_b - z_g, 1),
            "flash_to_bang_s": round(ev.t_first - tf, 4), "audible_range_km": round(ev.audible_range_km, 2),
            "channel": channel, "frame": "listener: +x forward, +y left, +z up, metres from the ears"}))
        # every flash gets its thunder marker at the first arrival, audible or not (module docstring)
        self.pending_markers.append(EventMarker(t=ev.t_first, kind="thunder", data={
            "id": eid, "cell": cell_id, "flash_t": round(tf, 4), "type": "cg" if cg else "ic",
            "distance_km": round(ev.r_min / 1000.0, 4), "azimuth": round(ev.near_dir[0], 4),
            "elevation": round(ev.near_dir[1], 4), "audible": bool(ev.audible),
            "peak_db_spl": round(20.0 * math.log10(max(ev.peak_pa, 1e-9) / 20e-6), 1),
            "duration_s": round(max(ev.end_sample / self.sr - ev.t_first, 0.0), 2)}))
        self.pending_markers.sort(key=lambda m: m.t)
        if not ev.audible:
            return None
        self.events.append(ev)
        if len(self.events) > self.max_events:
            self.events.sort(key=lambda e: e.peak_pa)
            self.events.pop(0)
            self.events.sort(key=lambda e: e.t_flash)
        return ev

    def render(self, ctx: BlockContext) -> list[Emission]:
        n, sr = ctx.n, ctx.sr
        if not self._block_fixed:
            self.block_s = int(n) / float(sr)
            self._block_fixed = True
        start = int(round(ctx.t0 * sr))
        for tf, cell, si, k, j in self._flashes(ctx):
            self._spawn(ctx, tf, cell, si, k, j)
        t1 = ctx.t0 + n / float(sr)
        due = [m for m in self.pending_markers if m.t < t1]
        if due:
            self.pending_markers = [m for m in self.pending_markers if m.t >= t1]
            ctx.markers.extend(due)
        bus = np.zeros((N_AMBI, n))
        alive = []
        for ev in self.events:
            bus += ev.render(ctx, start, n)
            if not ev.finished(start + n):
                alive.append(ev)
        self.events = alive
        echo = self.echo.process(bus)
        if echo is not None:
            bus += echo
        elif not self.events:
            return []
        return [Emission(signal=bus, stem="thunder", is_ambisonic=True, spread=1.0, source_id="thunder")]


__all__ = ["lightning_channel", "join_polylines", "split_polylines", "ThunderGenerator", "stroke_energy_j_per_m",
           "source_frequency_hz", "MEAN_TURN_DEG"]
