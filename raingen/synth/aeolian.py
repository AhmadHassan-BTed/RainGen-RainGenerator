"""Aeolian tones, edge/gap whistles, wind chimes and flapping fabric.

Physics:
* Vortex shedding from a cylinder (wire, guy line, railing bar): f = St U_n / d with
  St ~ 0.2 for 300 < Re < 2e5 (Strouhal 1878; Roshko 1954). U_n is the wind component
  normal to the axis (independence principle); shedding stops below Re ~ 40.
  A 5 mm wire at 10 m/s gives 400 Hz; 25-30 mm power conductors moan at 67-100 Hz.
* Level from Curle's (1955) dipole analogy (Phillips 1956): fluctuating lift per unit
  length F' = 0.5 rho U^2 d C_L' correlated over l_c ~ 4 d. Summing independent
  correlation lengths along a wire of length L at perpendicular distance r:
      p_rms^2 = l_c St^2 rho^2 U^6 C_L'^2 / (16 c^2) * (2 / r) atan(L / 2r)
  (a 5 mm wire at 12 m/s heard from 15 m: ~45 dB SPL).
* Lock-in (aeolian harp; Blevins, *Flow-Induced Vibration*): when St U/d falls within a
  few per cent of a string mode f_n = n/(2L) sqrt(T/mu), the wire vibrates, the spanwise
  correlation grows and the tone locks onto f_n, ~6-10 dB louder. Free-stream turbulence
  narrows the lock-in band, so steady winds sing and gusty winds waver.
* Beaufort ladder (WMO): wires and railings whistle from force 6 (10.8 m/s at 10 m).
  Wet high-voltage conductors add corona crackle and hum at twice the mains frequency.
* Edge/gap whistles: the jet through a slot of width h (speed v_j = U sqrt(Cp) from the
  facade pressure coefficient) has an edge-tone frequency ~0.3 v_j / h; where it meets
  an acoustic resonance (window frame, door seal, roof vent) it locks and whistles
  (Howe, *Acoustics of Fluid-Structure Interactions*); otherwise it howls broadband.
* Wind chimes: free-free tubes whose bending modes follow beta_k^2, i.e. ratios
  1 : 2.757 : 5.404 : 8.933 (Euler-Bernoulli beam), struck by a wind-driven clapper.
* Flags and fabric flutter at a Strouhal number ~0.2 on the chord (Shelley & Zhang 2011)
  and snap at the trailing edge once the flapping saturates; halyards slap the pole.

All voices produce sound pressure (Pa) at the listener; exterior voices use stem 'wind',
voices of the listener's own enclosure (window gaps, door seals, tent fabric) 'interior'.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import signal

from raingen.core.constants import RHO_AIR, SAMPLE_RATE, speed_of_sound
from raingen.core.types import BlockContext, Emission, SourcePatch
from raingen.physics.atmosphere import AirAbsorptionFilter
from raingen.physics.wind import wind_at_height

from .wind import (GustField, PendingMixer, Ramp, SparseModalBank, StatefulSOS, bandpass_rms_gain, bandpass_sos,
                   block_start_sample, db_spl, highpass_sos, lowpass_sos, noise_rms_gain, poisson_offsets, scene_z0,
                   shelter_of, smoothstep, upwind_distance)

NU_AIR = 1.5e-5            # kinematic viscosity of air, m^2/s
ST_CYL = 0.2               # Strouhal number of a circular cylinder (subcritical Re)
CL_RMS = 0.25              # rms fluctuating lift coefficient in atmospheric turbulence
CHIME_RATIOS = (1.0, 2.757, 5.404, 8.933)
_SCALES = {
    "pentatonic": (1.0, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2.0, 9 / 4, 5 / 2, 3.0),
    "major": (1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2.0, 9 / 4),
    "minor": (1.0, 9 / 8, 6 / 5, 4 / 3, 3 / 2, 8 / 5, 9 / 5, 2.0, 9 / 4),
    "minor_pentatonic": (1.0, 6 / 5, 4 / 3, 3 / 2, 9 / 5, 2.0, 12 / 5, 8 / 3, 3.0),
    "whole_tone": tuple(2.0 ** (k / 6.0) for k in range(9)),
}
_WIRE_KINDS = ("wire", "power_line", "cable", "guy_line", "rope")
_BAR_KINDS = ("railing", "fence", "bars", "grille", "balustrade", "rail")
_GAP_KINDS = ("gap", "vent", "slot", "edge", "chimney", "corner")
_WOOD_MATERIALS = ("wood", "bamboo", "timber")


def _num(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _mains_hz(scene) -> float:
    region = str((scene.place.biophony or {}).get("region", "")).lower()
    return 60.0 if region in ("north_america",) else 50.0


class _Voice:
    """Common geometry, local wind and propagation of one aeolian emitter."""

    def __init__(self, vid: str, position, height_m: float, stem: str, sr: int, z0: float,
                 wind_exposure: float = 1.0, spread: float = 0.1):
        self.id = vid
        self.position = tuple(float(v) for v in position)
        self.height = max(float(height_m), 0.3)
        self.stem = stem
        self.sr = sr
        self.z0 = z0
        self.wind_exposure = float(wind_exposure)
        self.spread = float(spread)
        x, y, z = self.position
        self.dist = max(0.3, math.sqrt(x * x + y * y + z * z))
        self.az = math.atan2(y, x)
        self.el = math.atan2(z, math.hypot(x, y))
        self.air = AirAbsorptionFilter(sr)

    def local_wind(self, w, field: GustField) -> tuple[float, float, float]:
        """(wind at the emitter's height incl. advected gust, 10 m-equivalent wind, gust multiplier)."""
        u_h = float(wind_at_height(w.wind_u10_mps, self.height, self.z0))
        g = field.at(upwind_distance(self.position, w.wind_azimuth), max(u_h, 0.5))
        return u_h * g * self.wind_exposure, float(w.wind_u10_mps) * g * self.wind_exposure, g

    def emit(self, y: np.ndarray, w, spread: Optional[float] = None) -> Emission:
        if self.dist > 3.0 and self.stem != "interior":
            y = self.air.process(y, self.dist, w.temperature_c, w.humidity_pct)
        return Emission(signal=y, azimuth=self.az, elevation=self.el, distance=self.dist,
                        spread=self.spread if spread is None else float(spread), stem=self.stem,
                        source_id=f"aeolian.{self.id}")


# --------------------------------------------------------------------------
# Cylinders: wires, guy lines, power lines, railing bars
# --------------------------------------------------------------------------

class _WireVoice(_Voice):
    """K parallel cylinders shedding vortices (one wire, or the bars of a railing)."""

    def __init__(self, vid, position, height_m, stem, sr, z0, cfg: np.random.Generator, *,
                 diameter_m: float, length_m: float, n_elements: int = 1, f1_hz: float = 0.0,
                 lock_db: float = 8.0, axis_az: Optional[float] = None, gate=(7.5, 11.0),
                 power_line: bool = False, corona_db: float = 0.0, mains_hz: float = 50.0,
                 wind_exposure: float = 1.0):
        super().__init__(vid, position, height_m, stem, sr, z0, wind_exposure)
        K = max(1, int(n_elements))
        self.K = K
        jit = cfg.uniform(0.85, 1.15, size=K) if K > 1 else np.ones(1)
        self.d = max(float(diameter_m), 5e-4) * jit
        self.L = max(float(length_m), 0.1)
        self.f1 = float(f1_hz) * (cfg.uniform(0.97, 1.03, size=K) if K > 1 else np.ones(1))
        self.lock_db = float(lock_db)
        self.axis_az = float(axis_az) if axis_az is not None else self.az + 0.5 * math.pi
        self.gate = gate
        self.power_line = bool(power_line)
        self.corona_db = float(corona_db)
        self.mains = float(mains_hz)
        self.spread = min(0.9, 0.7 * (2.0 / math.pi) * math.atan(self.L / (2.0 * self.dist))) if K == 1 else 0.25
        self.f = np.full(K, 200.0)
        self._started = False
        self.phase = cfg.uniform(0.0, 2.0 * math.pi, size=K)
        self.psi = cfg.uniform(0.0, 2.0 * math.pi, size=(2, K))
        self.jit = np.zeros(K)
        self.lock = np.zeros(K)
        self.mode = np.zeros(K, dtype=int)
        self.am_lp = StatefulSOS(lowpass_sos(10.0, sr))
        self.am_norm = noise_rms_gain(self.am_lp.sos)
        self.band = [StatefulSOS(bandpass_sos(200.0, 5.0, sr)) for _ in range(K)]
        self.band_fc = np.full(K, 200.0)
        self.band_g = np.full(K, bandpass_rms_gain(200.0, 5.0, sr))
        self.amp = Ramp(np.zeros(K))
        self.bb_amp = Ramp(np.zeros(K))
        self.corona_amp = Ramp(0.0)
        self.hum_phase = 0.0
        self.crackle_hp = StatefulSOS(np.vstack([highpass_sos(2500.0, sr), lowpass_sos(9000.0, sr)]))
        self.crackle_g = 1.0

    def _corona(self, ctx: BlockContext, n: int) -> np.ndarray:
        w = ctx.weather
        wet = smoothstep(w.rain_rate_mm_h, 0.2, 6.0) + 0.35 * smoothstep(w.humidity_pct, 88.0, 100.0)
        amp = db_spl(self.corona_db) * min(wet, 1.2) * spreading(self.dist, 15.0)
        g = self.corona_amp(amp, n)
        if float(np.max(g)) < 1e-9:
            return np.zeros(n)
        rng = ctx.rng.get(f"aeolian.corona.{self.id}", ctx.index)
        t = np.arange(1, n + 1) / self.sr
        ph = self.hum_phase + 2.0 * math.pi * 2.0 * self.mains * t
        hum = 0.8 * np.sin(ph) + 0.35 * np.sin(2.0 * ph + 0.7) + 0.2 * np.sin(3.0 * ph + 1.9)
        self.hum_phase = float(ph[-1] % (2.0 * math.pi))
        imp = np.zeros(n)
        offs = poisson_offsets(rng, 60.0 + 400.0 * min(wet, 1.0), n, self.sr)
        if offs.size:
            np.add.at(imp, offs, rng.lognormal(0.0, 0.8, size=offs.size) * rng.choice([-1.0, 1.0], size=offs.size))
        crackle = self.crackle_hp.process(imp) * 6.0
        return g * (0.6 * hum + crackle)

    def render(self, ctx: BlockContext, field: GustField) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        dt = n / sr
        u_loc, u_bft, _ = self.local_wind(w, field)
        normal = max(0.3, abs(math.sin(w.wind_azimuth - self.axis_az)))
        un = u_loc * normal
        gate = smoothstep(u_bft, self.gate[0], self.gate[1])
        idle = float(np.max(self.amp.value)) < 1e-9 and float(np.max(self.bb_amp.value)) < 1e-9
        if gate <= 0.0 and idle and not self.power_line:
            return None
        rng = ctx.rng.get(f"aeolian.{self.id}", ctx.index)
        ti = max(float(w.turbulence_intensity), 0.02)

        a = math.exp(-dt / 0.25)
        self.jit = a * self.jit + math.sqrt(1.0 - a * a) * rng.standard_normal(self.K)
        fs = ST_CYL * max(un, 0.05) / self.d * (1.0 + 0.35 * ti * self.jit)
        fs = np.clip(fs, 20.0, 0.4 * sr)

        # lock-in onto string modes (band shrinks with turbulence, hysteresis while locked)
        f_target = fs.copy()
        if np.any(self.f1 > 0.0):
            f1 = np.maximum(self.f1, 1e-3)
            nmode = np.maximum(1, np.rint(fs / f1)).astype(int)
            fn = nmode * f1
            det = (fs - fn) / fn
            bw = 0.07 * (1.0 - min(ti / 0.5, 0.8))
            bw = np.where(nmode == self.mode, 1.6 * bw, bw)
            locked = (np.abs(det) < bw) & (self.f1 > 0.0)
            tgt = locked.astype(float)
            rate = np.where(tgt > self.lock, 1.0 - math.exp(-dt / 0.3), 1.0 - math.exp(-dt / 0.1))
            self.lock = self.lock + (tgt - self.lock) * rate
            self.mode = np.where(locked, nmode, self.mode)
            f_target = np.where(locked, fn * (1.0 + 0.1 * det), fs)
            f_target = self.lock * f_target + (1.0 - self.lock) * fs
        f_new = f_target
        if not self._started:
            self.f = f_new.copy()
            self._started = True
        frac = np.arange(1, n + 1, dtype=float) / n
        f_inst = self.f[:, None] + (f_new - self.f)[:, None] * frac[None, :]
        self.f = f_new.copy()
        ph = self.phase[:, None] + 2.0 * math.pi * np.cumsum(f_inst, axis=1) / sr
        self.phase = ph[:, -1] % (2.0 * math.pi)
        tone = math.sqrt(2.0) * (np.sin(ph) + 0.12 * np.sin(2.0 * ph + self.psi[0][:, None])
                                 + 0.07 * np.sin(3.0 * ph + self.psi[1][:, None])) / 1.01

        # turbulent amplitude flicker of the shedding: deep when unlocked, steadier when locked
        am = self.am_lp.process(rng.standard_normal((self.K, n))) / self.am_norm
        depth = (0.55 * (1.0 - self.lock) + 0.12)[:, None]
        tone *= np.maximum(1.0 + depth * am, 0.0)

        # Phillips/Curle level along the cylinder (perpendicular distance r)
        c = speed_of_sound(w.temperature_c)
        r = self.dist
        l_c = 4.0 * self.d
        geo = np.sqrt(l_c * (2.0 / r) * math.atan(self.L / (2.0 * r)))
        re = un * self.d / NU_AIR
        active = np.clip((re - 40.0) / 110.0, 0.0, 1.0)
        p_rms = geo * ST_CYL * RHO_AIR * un ** 3 * CL_RMS / (4.0 * c) * active
        p_rms = p_rms * 10.0 ** (self.lock_db * self.lock / 20.0) * gate / math.sqrt(self.K)
        g = self.amp(p_rms, n)
        y = np.sum(g * tone, axis=0)

        # broadband shedding noise around the Strouhal frequency (unlocked, turbulent)
        bb_t = p_rms * 0.6 * (1.0 - self.lock)
        if float(np.max(bb_t)) > 1e-9 or float(np.max(self.bb_amp.value)) > 1e-9:
            gb = self.bb_amp(bb_t, n)
            noise = rng.standard_normal((self.K, n))
            for k in range(self.K):
                if abs(f_new[k] - self.band_fc[k]) / self.band_fc[k] > 0.03:
                    self.band_fc[k] = f_new[k]
                    self.band[k].set(bandpass_sos(f_new[k], 5.0, sr))
                    self.band_g[k] = bandpass_rms_gain(f_new[k], 5.0, sr)
                y = y + gb[k] * self.band[k].process(noise[k]) / self.band_g[k]

        if self.power_line and self.corona_db > 0.0:
            y = y + self._corona(ctx, n)
        return self.emit(y, w)


def spreading(dist: float, ref_m: float) -> float:
    return ref_m / max(dist, 0.3)


# --------------------------------------------------------------------------
# Edge / gap / slot whistles
# --------------------------------------------------------------------------

class _GapVoice(_Voice):
    """Flow through a slot or past an edge exciting a resonance: whistle + broadband howl."""

    def __init__(self, vid, position, height_m, stem, sr, z0, cfg: np.random.Generator, *,
                 f_res: float, q: float = 12.0, gap_m: float = 0.004, cp: float = 0.6,
                 onset_mps: float = 6.0, level_db: float = 50.0, howl_db: Optional[float] = None,
                 wind_exposure: float = 1.0, spread: float = 0.15):
        super().__init__(vid, position, height_m, stem, sr, z0, wind_exposure, spread)
        self.f_res = float(f_res)
        self.q = float(q)
        self.gap = max(float(gap_m), 5e-4)
        self.cp = max(float(cp), 0.05)
        self.onset = float(onset_mps)
        self.level_db = float(level_db)
        self.howl_db = self.level_db - 8.0 if howl_db is None else float(howl_db)
        self.f = self.f_res
        self.phase = cfg.uniform(0.0, 2.0 * math.pi)
        self.jit = 0.0
        self.env_lp = StatefulSOS(lowpass_sos(3.0, sr))
        self.env_norm = noise_rms_gain(self.env_lp.sos)
        self.howl = StatefulSOS(np.vstack([bandpass_sos(self.f_res, 2.0, sr), bandpass_sos(self.f_res * 0.5, 0.8, sr)]))
        self.howl_g = noise_rms_gain(self.howl.sos)
        self.whistle_amp = Ramp(0.0)
        self.howl_amp = Ramp(0.0)

    def render(self, ctx: BlockContext, field: GustField) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        dt = n / sr
        u_loc, _, _ = self.local_wind(w, field)
        vj = u_loc * math.sqrt(self.cp)
        drive = smoothstep(vj, self.onset * math.sqrt(self.cp), 1.6 * self.onset * math.sqrt(self.cp))
        if drive <= 0.0 and float(self.whistle_amp.value) < 1e-9 and float(self.howl_amp.value) < 1e-9:
            return None
        rng = ctx.rng.get(f"aeolian.{self.id}", ctx.index)
        a = math.exp(-dt / 0.4)
        self.jit = a * self.jit + math.sqrt(1.0 - a * a) * float(rng.standard_normal())
        fe = max(0.3 * max(vj, 0.1) / self.gap, 1.0)
        lr = math.log(fe / self.f_res)
        coupling = math.exp(-(lr / 0.35) ** 2)
        f_new = self.f_res * (1.0 + 0.04 * math.tanh(3.0 * lr) + 0.008 * self.jit)
        f_inst = self.f + (f_new - self.f) * np.arange(1, n + 1) / n
        self.f = f_new
        ph = self.phase + 2.0 * math.pi * np.cumsum(f_inst) / sr
        self.phase = float(ph[-1] % (2.0 * math.pi))
        # whistles switch on and off with the gusts (intermittent jet attachment)
        env = self.env_lp.process(rng.standard_normal(n)) / self.env_norm
        inter = np.clip(0.55 + 0.45 * env, 0.0, 1.5)
        ref = 10.0 * math.sqrt(self.cp)
        wa = db_spl(self.level_db) * (vj / ref) ** 2.5 * coupling * drive * spreading(self.dist, 1.0)
        ha = db_spl(self.howl_db) * (vj / ref) ** 3 * drive * spreading(self.dist, 1.0)
        tone = math.sqrt(2.0) * (np.sin(ph) + 0.15 * np.sin(2.0 * ph + 1.1))
        y = self.whistle_amp(wa, n) * tone * inter
        y = y + self.howl_amp(ha, n) * self.howl.process(rng.standard_normal(n)) / self.howl_g * np.sqrt(inter)
        return self.emit(y, w)


# --------------------------------------------------------------------------
# Wind chimes
# --------------------------------------------------------------------------

class _ChimeVoice(_Voice):
    """Free-free tubes struck by a wind-driven clapper."""

    def __init__(self, vid, position, height_m, stem, sr, z0, cfg: np.random.Generator, *,
                 n_tubes: int = 5, base_hz: float = 523.25, scale: str = "pentatonic",
                 material: str = "metal_sheet", level_db: float = 72.0, wind_exposure: float = 1.0):
        super().__init__(vid, position, height_m, stem, sr, z0, wind_exposure, spread=0.05)
        ratios = _SCALES.get(str(scale).lower(), _SCALES["pentatonic"])
        nt = int(min(max(n_tubes, 1), len(ratios)))
        self.n_tubes = nt
        self.wood = str(material).lower() in _WOOD_MATERIALS
        f0 = np.array(ratios[:nt]) * float(base_hz) * cfg.uniform(0.995, 1.005, size=nt)
        freqs, t60s = [], []
        for fi in f0:
            for k, ratio in enumerate(CHIME_RATIOS):
                f = fi * ratio * float(cfg.uniform(0.997, 1.003))
                if self.wood:
                    t60 = (0.35, 0.18, 0.1, 0.06)[k] * (500.0 / fi) ** 0.3
                else:
                    t60 = 6.0 * (523.0 / fi) ** 0.35 * (fi / f) ** 0.6
                freqs.append(f)
                t60s.append(t60)
        self.bank = SparseModalBank(freqs, t60s, sr)
        self.mode_gain = np.tile(np.array([1.0, 0.55, 0.35, 0.2]), nt)
        if self.wood:
            self.mode_gain = np.tile(np.array([1.0, 0.35, 0.15, 0.06]), nt)
        self.mode_gain = self.mode_gain[self.bank.keep] if self.bank.keep.size == self.mode_gain.size else self.mode_gain
        self.tube_of_mode = np.repeat(np.arange(nt), len(CHIME_RATIOS))[self.bank.keep]
        self.level = db_spl(level_db - (8.0 if self.wood else 0.0))
        self.carry: list[tuple[int, int, float]] = []     # double hits spilling into the next block

    def render(self, ctx: BlockContext, field: GustField) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        u_loc, _, g = self.local_wind(w, field)
        start = block_start_sample(ctx)
        rate = 3.5 * smoothstep(u_loc, 0.8, 3.5) * (max(u_loc, 0.1) / 5.0) ** 0.6 * (1.0 + 1.5 * max(g - 1.0, 0.0))
        rate = min(rate, 10.0)
        if rate <= 0.0 and self.bank.silent() and not self.carry:
            return None
        rng = ctx.rng.get(f"aeolian.{self.id}", ctx.index)
        strikes = []
        pending = [c for c in self.carry if c[0] < start + n]
        self.carry = [c for c in self.carry if c[0] >= start + n]
        for t_abs, tube, amp in pending:
            strikes.append((t_abs - start, self._amps(rng, tube, amp)))
        for off in poisson_offsets(rng, rate, n, sr):
            tube = int(rng.integers(0, self.n_tubes))
            amp = float(rng.lognormal(0.0, 0.35)) * min(1.0, max(u_loc, 0.1) / 6.0) ** 0.7
            strikes.append((int(off), self._amps(rng, tube, amp)))
            if rng.random() < 0.3:          # the clapper bounces onto a neighbouring tube
                off2 = int(off) + int(rng.uniform(0.015, 0.06) * sr)
                tube2 = int(min(max(tube + int(rng.choice([-1, 1])), 0), self.n_tubes - 1))
                amp2 = amp * float(rng.uniform(0.3, 0.7))
                if off2 < n:
                    strikes.append((off2, self._amps(rng, tube2, amp2)))
                else:
                    self.carry.append((start + off2, tube2, amp2))
        strikes.sort(key=lambda s: s[0])
        y = self.bank.process(n, strikes) * self.level * spreading(self.dist, 1.0)
        return self.emit(y, w)

    def _amps(self, rng: np.random.Generator, tube: int, amp: float) -> np.ndarray:
        a = np.zeros(self.bank.m)
        sel = self.tube_of_mode == tube
        a[sel] = amp * self.mode_gain[sel] * rng.lognormal(0.0, 0.25, size=int(np.sum(sel)))
        return a


# --------------------------------------------------------------------------
# Flags, fabric flutter, halyards
# --------------------------------------------------------------------------

class _FlagVoice(_Voice):
    """Flapping fabric: flutter rustle, trailing-edge snaps and (for flags) halyard slaps."""

    def __init__(self, vid, position, height_m, stem, sr, z0, cfg: np.random.Generator, *,
                 chord_m: float = 1.2, strouhal: float = 0.2, rustle_db: float = 52.0, snap_db: float = 66.0,
                 snap_range=(6.0, 13.0), onset_mps: float = 2.0, halyard: bool = False,
                 wind_exposure: float = 1.0, spread: float = 0.2):
        super().__init__(vid, position, height_m, stem, sr, z0, wind_exposure, spread)
        self.chord = min(max(float(chord_m), 0.2), 4.0)
        self.st = float(strouhal)
        self.rustle_db = float(rustle_db)
        self.snap_db = float(snap_db)
        self.snap_range = snap_range
        self.onset = float(onset_mps)
        self.phase = float(cfg.uniform(0.0, 1.0))
        self.f = 0.5
        self.jit = 0.0
        self.fc = 700.0
        self.rustle = StatefulSOS(bandpass_sos(self.fc, 0.8, sr))
        self.rustle_g = noise_rms_gain(self.rustle.sos)
        self.rustle_amp = Ramp(0.0)
        self.snaps = PendingMixer(1)
        self.halyard = bool(halyard)
        if self.halyard:
            f_pole = float(cfg.uniform(700.0, 1300.0))
            ratios = np.array(CHIME_RATIOS) * cfg.uniform(0.98, 1.02, size=4)
            self.pole = SparseModalBank(f_pole * ratios, np.array([1.6, 0.9, 0.5, 0.3]), sr)
            self.pole_gain = np.array([1.0, 0.6, 0.35, 0.2])[self.pole.keep]
        else:
            self.pole = None

    def _snap(self, rng: np.random.Generator, sr: int) -> np.ndarray:
        L = int(rng.uniform(0.004, 0.016) * sr)
        t = np.arange(L) / sr
        tau = L / sr / 3.0
        burst = rng.standard_normal(L) * np.exp(-t / tau)
        fc = rng.uniform(1200.0, 3200.0)
        burst = signal.sosfilt(np.vstack([bandpass_sos(fc, 0.9, sr), highpass_sos(250.0, sr)]), burst)
        pk = float(np.max(np.abs(burst)))
        return burst / pk if pk > 0 else burst

    def render(self, ctx: BlockContext, field: GustField) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        dt = n / sr
        u, _, _ = self.local_wind(w, field)
        active = smoothstep(u, self.onset, self.onset * 2.0)
        pole_idle = self.pole is None or self.pole.silent()
        if active <= 0.0 and float(self.rustle_amp.value) < 1e-9 and not self.snaps.active and pole_idle:
            return None
        rng = ctx.rng.get(f"aeolian.{self.id}", ctx.index)
        start = block_start_sample(ctx)
        a = math.exp(-dt / 0.6)
        self.jit = a * self.jit + math.sqrt(1.0 - a * a) * float(rng.standard_normal())
        f_new = max(self.st * u / self.chord * (1.0 + 0.15 * self.jit), 0.05)
        f_inst = self.f + (f_new - self.f) * np.arange(1, n + 1) / n
        self.f = f_new
        ph = self.phase + np.cumsum(f_inst) / sr
        wraps = np.nonzero(np.diff(np.floor(np.concatenate(([self.phase], ph)))) > 0)[0]
        self.phase = float(ph[-1] % 1.0)

        # flutter rustle, swelling twice per flap cycle
        fc = 450.0 + 55.0 * u
        if abs(fc - self.fc) / self.fc > 0.04:
            self.fc = fc
            self.rustle.set(bandpass_sos(fc, 0.8, sr))
            self.rustle_g = noise_rms_gain(self.rustle.sos)
        motion = 0.35 + 0.65 * np.sin(2.0 * math.pi * ph) ** 2
        amp = db_spl(self.rustle_db) * (max(u, 0.0) / 10.0) ** 2 * active * spreading(self.dist, 1.0)
        y = self.rustle_amp(amp, n) * self.rustle.process(rng.standard_normal(n)) / self.rustle_g * motion

        # trailing-edge snaps once the flapping saturates
        p_snap = smoothstep(u, self.snap_range[0], self.snap_range[1])
        if p_snap > 0.0:
            s_amp = db_spl(self.snap_db) * (u / 10.0) ** 2 * spreading(self.dist, 1.0)
            for off in wraps:
                if rng.random() < p_snap * (0.6 + 0.4 * rng.random()):
                    self.snaps.add(start + int(off), self._snap(rng, sr) * s_amp * float(rng.lognormal(0.0, 0.3)))
                    if rng.random() < 0.25 * p_snap:     # a second, smaller crack within the same flap
                        d2 = int(rng.uniform(0.02, 0.08) * sr)
                        self.snaps.add(start + int(off) + d2, self._snap(rng, sr) * 0.5 * s_amp)
        if self.snaps.active:
            y = y + self.snaps.take(start, n)

        # halyard slapping the (metal) pole
        if self.pole is not None:
            rate = 1.4 * smoothstep(u, 3.0, 9.0) * (1.0 + 0.5 * abs(self.jit))
            strikes = []
            for off in poisson_offsets(rng, rate, n, sr):
                amp_s = float(rng.lognormal(0.0, 0.4)) * min(u / 8.0, 1.5)
                strikes.append((int(off), amp_s * self.pole_gain * rng.lognormal(0.0, 0.3, size=self.pole.m)))
            ring = self.pole.process(n, strikes)
            y = y + ring * db_spl(64.0) * spreading(self.dist, 1.0)
        return self.emit(y, w)


# --------------------------------------------------------------------------
# Car body buffeting
# --------------------------------------------------------------------------

class _BuffetVoice(_Voice):
    """Gust pressure loading on a car body: low thumps and rocking felt inside the cabin."""

    def __init__(self, vid, position, height_m, stem, sr, z0, level_db: float = 62.0):
        super().__init__(vid, position, height_m, stem, sr, z0, 1.0, spread=0.9)
        self.level_db = float(level_db)
        self.lp = StatefulSOS(np.vstack([lowpass_sos(38.0, sr), highpass_sos(8.0, sr)]))
        self.lp_g = noise_rms_gain(self.lp.sos)
        self.amp = Ramp(0.0)
        self.g_prev = 1.0

    def render(self, ctx: BlockContext, field: GustField) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        u, _, g = self.local_wind(w, field)
        dgdt = abs(g - self.g_prev) / (n / sr)
        self.g_prev = g
        if u < 4.0 and float(self.amp.value) < 1e-9:
            return None
        rng = ctx.rng.get(f"aeolian.{self.id}", ctx.index)
        amp = db_spl(self.level_db) * (u / 15.0) ** 2 * smoothstep(u, 4.0, 9.0) * (0.5 + min(dgdt, 2.0))
        y = self.amp(amp, n) * self.lp.process(rng.standard_normal(n)) / self.lp_g
        return Emission(signal=y, azimuth=0.0, elevation=0.0, distance=1.0, spread=0.9, stem=self.stem,
                        source_id=f"aeolian.{self.id}")


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def _string_f1(length_m: float, tension_n: float, diameter_m: float, density: float) -> float:
    mu = density * math.pi * diameter_m * diameter_m / 4.0
    return math.sqrt(max(tension_n, 1.0) / max(mu, 1e-6)) / (2.0 * max(length_m, 0.1))


class AeolianGenerator:
    """Every wind-excited object: wires, railings, gaps and edges, chimes, flags, shelter fabric.

    Built from the layout (kinds wire / railing / fence / gap / vent / chime / flag) plus the
    listener's shelter (tent guy lines and fly sheet, umbrella canopy, car door seals and
    body buffeting, window gaps, attic vents) and street-canyon corners in town presets.
    """

    name = "aeolian"

    def __init__(self, scene, layout, rngf, gust_field: Optional[GustField] = None, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.field = gust_field or GustField(scene, rngf, sr)
        self.z0 = scene_z0(scene)
        self.h = float(scene.listener.height_m)
        self.voices: list[_Voice] = []
        self.mains = _mains_hz(scene)
        for patch in layout or []:
            v = self._from_patch(patch, rngf)
            if v is not None:
                self.voices.append(v)
        self._shelter_voices(scene, layout or [], rngf)

    # -- construction --------------------------------------------------------
    def _height(self, patch: SourcePatch) -> float:
        p = patch.params or {}
        if "height_m" in p:
            return _num(p["height_m"], 2.0)
        return max(patch.position[2] + self.h, 0.3)

    def _from_patch(self, patch: SourcePatch, rngf) -> Optional[_Voice]:
        kind = str(patch.kind or "").lower()
        p = patch.params or {}
        ptype = str(p.get("type", kind)).lower()
        cfg = rngf.get(f"aeolian.cfg.{patch.id}", 0)
        h = self._height(patch)
        expo = _num(p.get("wind_exposure", 1.0), 1.0)
        axis = None
        if "axis_deg" in p:
            axis = math.radians(_num(p["axis_deg"], 0.0))
        if kind in _WIRE_KINDS or ptype in _WIRE_KINDS:
            power = bool(p.get("power_line", ptype == "power_line"))
            d = _num(p.get("diameter_m", 0.025 if power else 0.005), 0.005)
            L = _num(p.get("length_m", 40.0), 40.0)
            if "f1_hz" in p:
                f1 = _num(p["f1_hz"], 0.0)
            else:
                tension = _num(p.get("tension_n", 15000.0 if power else 900.0), 900.0)
                density = 3500.0 if power else 7850.0       # ACSR conductor vs steel wire
                f1 = _string_f1(L, tension, d, density)
            kv = _num(p.get("kv", 132.0 if power else 0.0), 0.0)
            corona = (36.0 + 12.0 * smoothstep(kv, 60.0, 220.0)) if (power and kv > 1.0) else 0.0
            return _WireVoice(patch.id, patch.position, h, "wind", self.sr, self.z0, cfg, diameter_m=d, length_m=L,
                              f1_hz=f1, lock_db=6.0 if L > 10.0 else 10.0, axis_az=axis, power_line=power,
                              corona_db=corona, mains_hz=self.mains, wind_exposure=expo)
        if kind in _BAR_KINDS or ptype in _BAR_KINDS:
            d = _num(p.get("diameter_m", 0.004 if ptype == "fence" else 0.025), 0.02)
            nb = int(_num(p.get("n_bars", 6), 6))
            return _WireVoice(patch.id, patch.position, h, "wind", self.sr, self.z0, cfg, diameter_m=d,
                              length_m=_num(p.get("bar_length_m", 1.0), 1.0), n_elements=nb, f1_hz=0.0,
                              lock_db=0.0, axis_az=axis if axis is not None else 0.0,
                              gate=(7.0, 10.5), wind_exposure=expo)
        if kind in _GAP_KINDS or ptype in _GAP_KINDS:
            gap = _num(p.get("gap_mm", 4.0), 4.0) / 1000.0
            f_res = _num(p.get("freq_hz", p.get("f_res_hz", 0.3 * 10.0 * math.sqrt(0.6) / gap)), 600.0)
            stem = "interior" if patch.interior else "wind"
            return _GapVoice(patch.id, patch.position, h, stem, self.sr, self.z0, cfg, f_res=f_res,
                             q=_num(p.get("q", 10.0), 10.0), gap_m=gap, onset_mps=_num(p.get("onset_mps", 6.0), 6.0),
                             level_db=_num(p.get("level_db", 50.0), 50.0), wind_exposure=expo)
        if kind == "chime" or ptype == "chime":
            return _ChimeVoice(patch.id, patch.position, h, "wind", self.sr, self.z0, cfg,
                               n_tubes=int(_num(p.get("n_tubes", 5), 5)),
                               base_hz=_num(p.get("base_freq_hz", 523.25), 523.25),
                               scale=str(p.get("scale", "pentatonic")), material=patch.material,
                               level_db=_num(p.get("level_db", 72.0), 72.0), wind_exposure=expo)
        if kind == "flag" or ptype == "flag":
            area = _num(p.get("area_m2", patch.area_m2), 1.5)
            pole_h = _num(p.get("pole_height_m", h), h)
            return _FlagVoice(patch.id, patch.position, max(pole_h, 1.0), "wind", self.sr, self.z0, cfg,
                              chord_m=math.sqrt(max(area, 0.05) * 1.5), halyard=bool(p.get("halyard", True)),
                              wind_exposure=expo)
        return None

    def _shelter_voices(self, scene, layout, rngf) -> None:
        shelter = shelter_of(scene)
        sr, z0, h = self.sr, self.z0, self.h
        place = scene.place

        def cfg(tag: str) -> np.random.Generator:
            return rngf.get(f"aeolian.cfg.shelter.{tag}", 0)

        if shelter == "tent":
            f1 = _string_f1(1.8, 60.0, 0.003, 1150.0)
            for k, (x, y) in enumerate(((1.6, 1.3), (-1.6, -1.3))):
                c = cfg(f"guy{k}")
                self.voices.append(_WireVoice(f"shelter.tent.guy{k}", (x, y, 0.4 - h), 0.8, "wind", sr, z0, c,
                                              diameter_m=0.003, length_m=1.8, f1_hz=f1 * float(c.uniform(0.9, 1.1)),
                                              lock_db=10.0, gate=(5.5, 9.0)))
            self.voices.append(_FlagVoice("shelter.tent.fly", (0.0, 0.45, 0.35), max(h + 0.35, 1.0), "interior", sr,
                                          z0, cfg("fly"), chord_m=1.0, strouhal=0.3, rustle_db=58.0, snap_db=70.0,
                                          snap_range=(6.0, 12.0), onset_mps=2.5, wind_exposure=0.8, spread=0.7))
        elif shelter == "umbrella":
            self.voices.append(_FlagVoice("shelter.umbrella.canopy", (0.0, 0.0, 0.35), h + 0.35, "wind", sr, z0,
                                          cfg("umbrella"), chord_m=0.6, strouhal=0.35, rustle_db=50.0, snap_db=68.0,
                                          snap_range=(7.0, 14.0), onset_mps=3.0, spread=0.6))
        elif shelter == "car":
            c = cfg("seal")
            self.voices.append(_GapVoice("shelter.car.door_seal", (0.2, 0.6, 0.0), 1.2, "interior", sr, z0, c,
                                         f_res=float(c.uniform(950.0, 1500.0)), q=18.0, gap_m=0.0008, cp=0.5,
                                         onset_mps=9.0, level_db=40.0, spread=0.3))
            self.voices.append(_BuffetVoice("shelter.car.buffet", (0.0, 0.0, 0.0), 1.2, "interior", sr, z0))
        elif shelter in ("room_open", "room_closed"):
            wo = float(getattr(place, "window_open", 0.0) or 0.0)
            c = cfg("window")
            if 0.0 < wo <= 0.3:
                gap = 0.002 + 0.05 * wo
                self.voices.append(_GapVoice("shelter.room.window_gap", (2.0, 0.0, 0.0), h, "interior", sr, z0, c,
                                             f_res=float(c.uniform(350.0, 900.0)), q=12.0, gap_m=gap, cp=0.7,
                                             onset_mps=5.0, level_db=52.0 - 20.0 * wo, spread=0.2))
            else:
                self.voices.append(_GapVoice("shelter.room.window_seal", (2.0, 0.0, 0.0), h, "interior", sr, z0, c,
                                             f_res=float(c.uniform(600.0, 1200.0)), q=20.0, gap_m=0.0007, cp=0.7,
                                             onset_mps=11.0, level_db=36.0, spread=0.2))
        elif shelter == "attic":
            c = cfg("vent")
            self.voices.append(_GapVoice("shelter.attic.vent", (0.0, 0.0, 1.6), h + 1.6, "interior", sr, z0, c,
                                         f_res=float(c.uniform(170.0, 320.0)), q=5.0, gap_m=0.01, cp=0.8,
                                         onset_mps=7.0, level_db=50.0, spread=0.6))
        preset = str(place.preset or "").lower()
        if any(k in preset for k in ("city", "street", "alley", "downtown", "urban", "town")):
            facades = [p for p in layout if (p.params or {}).get("facade") and p.kind == "surface"]
            pos = facades[0].position if facades else (8.0, 6.0, 2.0)
            c = cfg("corner")
            self.voices.append(_GapVoice("scenery.corner_howl", pos, max(pos[2] + h, 3.0), "wind", sr, z0, c,
                                         f_res=float(c.uniform(220.0, 480.0)), q=3.0, gap_m=0.05, cp=0.9,
                                         onset_mps=6.0, level_db=58.0, spread=0.5))

    # -- render --------------------------------------------------------------
    def render(self, ctx: BlockContext) -> list[Emission]:
        self.field.advance(ctx.index, ctx.weather)
        out = []
        for v in self.voices:
            em = v.render(ctx, self.field)
            if em is not None:
                out.append(em)
        return out
