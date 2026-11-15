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
  (band noise with growl modulation following the smoothed arrival energy) and a
  "deepener" (sub-bass afterglow), plus terrain/building echoes and a diffuse terrain tail.

Events last up to 40 s; each is rendered lazily into an accumulator that spans blocks.
Markers: EventMarker('lightning') at the flash with the channel polylines (listener frame,
metres) and EventMarker('thunder') at the first arrival. Stem 'thunder'.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import signal

from raingen.core.constants import AMBI_ORDER, BLOCK, N_AMBI, P0, SAMPLE_RATE, speed_of_sound
from raingen.core.types import BlockContext, Emission, EventMarker, bearing_to_azimuth
from raingen.physics.atmosphere import (air_absorption_sos, iso9613_alpha_db_per_m, refraction_gain_db,
                                        shadow_distance_m)
from raingen.spatial.sh import sh_real

from .places import Room
from .wind import Ramp, StatefulSOS, highpass_sos, lowpass_sos, noise_rms_gain, sh_gains

MEAN_TURN_DEG = 16.0          # mean direction change between channel segments (Few 1969)
MAX_EVENT_S = 40.0
N_WAVE_K = 2.6                # Pa: calibration of the segment peak (3 Pa broadside at 1 km, 30 m, E0 = 70 kJ/m)
ENERGY_FACTOR = {"main": 1.0, "cloud": 0.55, "branch": 0.3, "ic": 0.5}
RUMBLER_POWER = 0.8           # rumbler power relative to the smoothed N-wave arrival power
DEEPENER_POWER = 0.45
BIN_S = 0.01


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


def _walk(rng: np.random.Generator, start, end, step: float, steer: float, z_floor: float,
          turn_deg: float = MEAN_TURN_DEG, max_steps: int = 5000) -> np.ndarray:
    """Tortuous walk from start to end: half-normal turns (mean `turn_deg`) steered to the target."""
    sigma = math.radians(turn_deg) * math.sqrt(math.pi / 2.0)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    p = start.copy()
    d = _unit(end - p)
    pts = [p.copy()]
    for _ in range(max_steps):
        to = end - p
        dist = float(np.linalg.norm(to))
        if dist <= step:
            break
        d = _unit((1.0 - steer) * d + steer * to / dist)
        d = _turn(rng, d, abs(float(rng.normal(0.0, sigma))))
        L = min(step * float(rng.lognormal(0.0, 0.4)), dist)
        p = p + d * L
        if p[2] < z_floor:
            p[2] = z_floor
            d[2] = abs(d[2])
            d = _unit(d)
        pts.append(p.copy())
    pts.append(end.copy())
    return np.array(pts)


def _tortuous(rng: np.random.Generator, start, end, z_floor: float, macro: float = 450.0,
              micro: float = 35.0) -> np.ndarray:
    """Two-scale tortuosity: km-scale meander through waypoints, filled with ~35 m zig-zags."""
    way = _walk(rng, start, end, macro, 0.3, z_floor)
    pts = [way[0]]
    for a, b in zip(way[:-1], way[1:]):
        seg = _walk(rng, a, b, micro, 0.18, z_floor)
        pts.extend(seg[1:])
    return np.array(pts)


def _free_walk(rng: np.random.Generator, start, d0, length: float, step: float, bias, bias_w: float,
               z_floor: float, z_ceiling: float = math.inf) -> np.ndarray:
    sigma = math.radians(MEAN_TURN_DEG) * math.sqrt(math.pi / 2.0)
    p = np.asarray(start, dtype=float).copy()
    d = _unit(np.asarray(d0, dtype=float))
    bias = _unit(np.asarray(bias, dtype=float))
    pts = [p.copy()]
    total = 0.0
    while total < length:
        d = _unit((1.0 - bias_w) * d + bias_w * bias)
        d = _turn(rng, d, abs(float(rng.normal(0.0, sigma))))
        L = step * float(rng.lognormal(0.0, 0.4))
        p = p + d * L
        total += L
        if p[2] < z_floor:
            break
        if p[2] > z_ceiling:
            p[2] = z_ceiling
            d[2] = -abs(d[2])
        pts.append(p.copy())
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
        lower = _tortuous(rng, B, S, z_floor=z_ground)
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

class _Accum:
    """Multichannel overlap-add accumulator addressed relative to the current block."""

    def __init__(self, channels: int):
        self.buf = np.zeros((channels, 0))
        self.used = 0

    def add(self, offset: int, y: np.ndarray) -> None:
        if offset < 0:
            y = y[:, -offset:]
            offset = 0
        end = offset + y.shape[1]
        if end > self.buf.shape[1]:
            nb = np.zeros((self.buf.shape[0], max(end, 2 * self.buf.shape[1], 8192)))
            nb[:, : self.buf.shape[1]] = self.buf
            self.buf = nb
        self.buf[:, offset:end] += y
        self.used = max(self.used, end)

    def pop(self, n: int) -> np.ndarray:
        out = np.zeros((self.buf.shape[0], n))
        if self.used == 0:
            return out
        m = min(n, self.used)
        out[:, :m] = self.buf[:, :m]
        rem = self.used - n
        if rem > 0:
            self.buf[:, :rem] = self.buf[:, n:self.used].copy()
            self.buf[:, rem:self.used] = 0.0
            self.used = rem
        else:
            self.buf[:, : self.used] = 0.0
            self.used = 0
        return out


class _ThunderEvent:
    """The acoustic consequence of one flash at one listener, rendered lazily block by block."""

    def __init__(self, eid: str, t_flash: float, cg: bool, polys, z_cloud_base: float, intensity: float,
                 weather, listener_h: float, sr: int, rng: np.random.Generator):
        self.id = eid
        self.sr = sr
        self.t_flash = float(t_flash)
        temp = float(weather.temperature_c)
        rh = float(weather.humidity_pct)
        self.temp, self.rh = temp, rh
        c = speed_of_sound(temp)
        z_g = -float(listener_h)

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
        cospsi = np.abs(np.sum(vec * u, axis=1)) / seg_l
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
                mid, seg_l, r, u, cospsi, fac = mid[sel], seg_l[sel], r[sel], u[sel], cospsi[sel], fac[sel]

        E0 = e0_base * fac * e_mult
        R0 = np.sqrt(E0 / (math.pi * P0))
        T0 = np.sqrt(E0 / P0) / c                       # 1 / (c0 sqrt(p0/E0))
        T = np.clip(T0 * (r / R0) ** 0.25, T0, 0.08)
        A = N_WAVE_K * np.sqrt(E0) * (seg_l / r) * (R0 / r) ** 0.25
        dtau = seg_l * cospsi / c

        # refraction (lapse + wind shear, upward-refraction shadow) per segment
        az = np.arctan2(u[:, 1], u[:, 0])
        el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
        r_h = np.hypot(mid[:, 0], mid[:, 1])
        h_src = np.maximum(mid[:, 2] - z_g, 1.0)
        wcomp = float(weather.wind_u10_mps) * np.cos(az - float(weather.wind_azimuth))
        stab = str(weather.stability)
        g_db = np.array([refraction_gain_db(float(rh_i), float(wc), stab, float(hs), float(listener_h), temp)
                         for rh_i, wc, hs in zip(r_h, wcomp, h_src)])
        A = A * 10.0 ** (g_db / 20.0)

        t_arr = self.t_flash + r / c + t_off
        t_start = t_arr - 0.5 * dtau
        t_first = float(np.min(t_start))
        keep = (t_start <= t_first + MAX_EVENT_S) & (A > 2e-6)
        order = np.argsort(t_start[keep], kind="stable")
        idx = np.nonzero(keep)[0][order]
        self.A, self.T, self.dtau, self.r = A[idx], T[idx], dtau[idx], r[idx]
        self.az, self.el = az[idx], el[idx]
        self.s_start = np.round(t_start[idx] * sr).astype(np.int64)
        self.Y = np.asarray(sh_real(AMBI_ORDER, self.az, self.el), dtype=float).reshape(N_AMBI, -1) if idx.size \
            else np.zeros((N_AMBI, 0))
        self.ptr = 0
        self.acc = _Accum(N_AMBI)
        self.audible = bool(idx.size) and float(np.max(self.A)) > 2e-5
        self.peak_pa = float(np.max(self.A)) if idx.size else 0.0
        self.r_min = float(np.min(self.r)) if idx.size else float(np.min(r))
        self.t_first = float(t_first)
        i_near = int(np.argmin(self.r)) if idx.size else 0
        self.near_dir = (float(self.az[i_near]), float(self.el[i_near])) if idx.size else (0.0, 0.0)
        top_h = float(np.max(h_src))
        self.audible_range_km = shadow_distance_m(top_h, float(listener_h), temp) / 1000.0

        # ---- signal layer: energy envelope, direction track, rumbler, deepener ----
        self.end_sample = int(self.s_start[-1]) if idx.size else int(round(t_first * sr))
        self.t_origin = t_first
        if idx.size:
            self._envelopes(rng)
            self._crack(rng)
        else:
            self.n_bins = 0
            self.crack = None
        self.dir_ramp = Ramp(sh_gains(self.near_dir[0], self.near_dir[1], spread=0.45))
        self.rumbler = None
        self.deep = None
        self.growl = None

    # -- precomputation -------------------------------------------------------
    def _envelopes(self, rng: np.random.Generator) -> None:
        sr = self.sr
        dur = np.maximum(self.T, self.dtau)
        alpha = iso9613_alpha_db_per_m(1.0 / self.T, self.temp, self.rh)
        e = self.A ** 2 * (self.T / 3.0) * np.minimum(1.0, 1.5 * self.T / np.maximum(self.dtau, 1e-9))
        e = e * 10.0 ** (-alpha * self.r / 10.0)
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
        decay = math.exp(-BIN_S / 2.5)
        hold = 0.0
        for k in range(nb):                              # afterglow: peak hold with ~2.5 s decay
            hold = max(Pd[k], hold * decay)
            Pd[k] = hold
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
        for k in range(nb):                              # hold the last direction through silent stretches
            if norms[k] > 1e-18:
                last = dirs[k] / norms[k]
            dirs[k] = last
        self.dir_az = np.arctan2(dirs[:, 1], dirs[:, 0])
        self.dir_el = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
        self.n_bins = nb
        self.env_end_sample = int(round((self.t_origin + nb * BIN_S) * sr))
        self.end_sample = max(self.end_sample + int(0.5 * sr), self.env_end_sample)
        # the rumbler's bandwidth narrows with distance (absorption strips the top first)
        r_mean = float(np.sum(self.r * e) / max(float(np.sum(e)), 1e-30))
        self.f_hi = float(np.clip(2500.0 * (1000.0 / max(r_mean, 100.0)) ** 0.7, 120.0, 2500.0))

    def _crack(self, rng: np.random.Generator) -> None:
        """Near-strike crack and tearing (the 'strike' layer)."""
        sr = self.sr
        self.crack = None
        rmin = self.r_min
        if rmin >= 3500.0:
            return
        L = int(0.6 * sr)
        t = np.arange(L) / sr
        tau = 0.05 + 0.1 * rmin / 3500.0
        burst = rng.standard_normal(L) * np.exp(-t / tau) * (1.0 - np.exp(-t / 0.0015))
        burst = signal.sosfilt(highpass_sos(600.0, sr), burst)
        if rmin < 800.0:                                 # tearing: dense clicks from the nearest channel parts
            k = int(rng.integers(40, 150))
            offs = np.sort(rng.integers(0, int(0.12 * sr), size=k))
            imp = np.zeros(L)
            np.add.at(imp, offs, rng.lognormal(0.0, 0.6, size=k) * rng.choice([-1.0, 1.0], size=k))
            burst += 3.0 * signal.sosfilt(np.vstack([highpass_sos(1800.0, sr), lowpass_sos(7000.0, sr)]), imp)
        burst = signal.sosfilt(air_absorption_sos(rmin, self.temp, self.rh, sr), burst)
        pk = float(np.max(np.abs(burst)))
        if pk <= 0.0:
            return
        near = self.A[self.r < 1.2 * rmin]
        a_peak = float(np.max(near)) if near.size else self.peak_pa
        self.crack = burst / pk * 0.5 * a_peak * (1.0 - rmin / 3500.0)
        self.crack_start = int(round((self.t_flash + rmin / speed_of_sound(self.temp)) * sr))
        self.crack_Y = sh_gains(self.near_dir[0], self.near_dir[1], spread=0.1)
        self.end_sample = max(self.end_sample, self.crack_start + L)

    def _kernel(self, i: int) -> np.ndarray:
        """Band-limited N-wave smeared over the segment's arrival window, absorbed over its path."""
        sr = self.sr
        T = float(self.T[i])
        nT = max(int(math.ceil(T * sr)), 2)
        t = (np.arange(nT) + 0.5) / sr
        nw = 1.0 - 2.0 * t / T
        rise = max(3.0 / sr, 2e-4 * math.sqrt(float(self.r[i]) / 1000.0))
        nr = max(3, int(round(rise * sr)))
        win = np.hanning(nr + 2)[1:-1]
        nw = np.convolve(nw, win / win.sum())
        m = max(1, int(round(float(self.dtau[i]) * sr)))
        if m > 1:
            xp = np.concatenate([np.zeros(m), nw, np.zeros(m - 1)])
            cs = np.cumsum(xp)
            nw = (cs[m:] - cs[:-m]) / m
        nw = np.concatenate([nw, np.zeros(384)])
        nw = signal.sosfilt(air_absorption_sos(float(self.r[i]), self.temp, self.rh, sr), nw)
        return nw * float(self.A[i])

    # -- per block --------------------------------------------------------------
    def _rumble_filters(self) -> None:
        sr = self.sr
        self.rumbler = StatefulSOS(np.vstack([highpass_sos(22.0, sr), lowpass_sos(self.f_hi, sr)]))
        self.rumbler_g = noise_rms_gain(self.rumbler.sos)
        self.deep = StatefulSOS(np.vstack([highpass_sos(12.0, sr), lowpass_sos(55.0, sr, order=4)]))
        self.deep_g = noise_rms_gain(self.deep.sos)
        self.growl = StatefulSOS(lowpass_sos(6.0, sr))
        self.growl_g = noise_rms_gain(self.growl.sos)

    def render(self, ctx: BlockContext, start: int, n: int) -> np.ndarray:
        sr = self.sr
        end = start + n
        while self.ptr < self.s_start.size and int(self.s_start[self.ptr]) < end:
            i = self.ptr
            k = self._kernel(i)
            self.acc.add(int(self.s_start[i]) - start, np.outer(self.Y[:, i], k))
            self.end_sample = max(self.end_sample, int(self.s_start[i]) + k.size)
            self.ptr += 1
        out = self.acc.pop(n)
        if self.crack is not None:
            a = max(start, self.crack_start)
            b = min(end, self.crack_start + self.crack.size)
            if b > a:
                out[:, a - start:b - start] += np.outer(self.crack_Y, self.crack[a - self.crack_start:b - self.crack_start])
        if self.n_bins and start < self.env_end_sample and end > int(self.t_origin * sr):
            if self.rumbler is None:
                self._rumble_filters()
            rng = ctx.rng.get(f"thunder.rumble.{self.id}", ctx.index)
            tb = ((start + np.arange(n)) / sr - self.t_origin) / BIN_S
            grid = np.arange(self.n_bins)
            env_r = np.interp(tb, grid, self.env_r, left=0.0, right=0.0)
            env_d = np.interp(tb, grid, self.env_d, left=0.0, right=0.0)
            growl = self.growl.process(rng.standard_normal(n)) / self.growl_g
            rum = self.rumbler.process(rng.standard_normal(n)) / self.rumbler_g * env_r * np.maximum(1.0 + 0.6 * growl, 0.0)
            deep = self.deep.process(rng.standard_normal(n)) / self.deep_g * env_d
            kb = int(min(max(tb[n // 2], 0), self.n_bins - 1))
            Yt = sh_gains(float(self.dir_az[kb]), float(self.dir_el[kb]), spread=0.45)
            out += self.dir_ramp(Yt, n) * (rum + deep)[None, :]
        return out

    def finished(self, next_start: int) -> bool:
        return self.ptr >= self.s_start.size and self.acc.used == 0 and next_start >= self.end_sample


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
            sos = np.vstack([air_absorption_sos(2.0 * d, 15.0, 70.0, sr), lowpass_sos(3500.0, sr)])
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

    def __init__(self, scene, layout, rngf, sr: int = SAMPLE_RATE, block: int = BLOCK, max_events: int = 16):
        self.sr = int(sr)
        self.block_s = int(block) / float(sr)
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
        dh = math.hypot(sx, sy)
        if dh < 250.0:
            s = 250.0 / max(dh, 1e-6)
            sx, sy = sx * s if dh > 1e-6 else 250.0, sy * s
            dh = 250.0
        z_g = -self.listener_h
        polys, z_b = _lightning_tree(rng, sx, sy, cg, z_g)
        eid = f"{cell.id}.{k}.{j}"
        ev = _ThunderEvent(eid, tf, cg, polys, z_b, si, ctx.weather, self.listener_h, self.sr, rng)
        az = math.atan2(sy, sx)
        bearing = (self.heading - math.degrees(az)) % 360.0
        channel = [np.round(p[:: (1 if kind in ("main", "ic") else 2)], 1).tolist() for p, kind in polys]
        ctx.markers.append(EventMarker(t=tf, kind="lightning", data={
            "id": eid, "cell": cell.id, "type": "cg" if cg else "ic",
            "distance_km": round(dh / 1000.0, 3), "bearing_deg": round(bearing, 1), "azimuth": round(az, 4),
            "intensity": round(si, 3), "energy_j_per_m": round(stroke_energy_j_per_m(si), 1),
            "strokes_s": [round(s, 4) for s in ev.strokes], "cloud_base_m": round(z_b - z_g, 1),
            "flash_to_bang_s": round(ev.t_first - tf, 3), "audible_range_km": round(ev.audible_range_km, 2),
            "channel": channel, "frame": "listener: +x forward, +y left, +z up, metres from the ears"}))
        if not ev.audible:
            return
        self.pending_markers.append(EventMarker(t=ev.t_first, kind="thunder", data={
            "id": eid, "cell": cell.id, "flash_t": round(tf, 4), "type": "cg" if cg else "ic",
            "distance_km": round(ev.r_min / 1000.0, 3), "azimuth": round(ev.near_dir[0], 4),
            "elevation": round(ev.near_dir[1], 4), "peak_db_spl": round(20.0 * math.log10(max(ev.peak_pa, 1e-9) / 20e-6), 1),
            "duration_s": round(max(ev.end_sample / self.sr - ev.t_first, 0.0), 2)}))
        self.events.append(ev)
        if len(self.events) > self.max_events:
            self.events.sort(key=lambda e: e.peak_pa)
            self.events.pop(0)
            self.events.sort(key=lambda e: e.t_flash)

    def render(self, ctx: BlockContext) -> list[Emission]:
        n, sr = ctx.n, ctx.sr
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
