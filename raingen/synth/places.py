"""Places and shelters: envelope transmission loss, room reverberation, car wipers.

Physics:
* Transmission loss of a shelter envelope (Fahy, *Sound and Structural Vibration*;
  Bies & Hansen, *Engineering Noise Control*): a limp leaf of surface mass m obeys the
  mass law, normal incidence TL_0 = 10 log10(1 + (pi f m / rho c)^2), field incidence
  ~ TL_0 - 5 dB (= 20 log10(m f) - 47 for heavy leaves). Leaks dominate a good leaf:
  tau = (1 - s) 10^(-TL_mass/10) + s for an open-area fraction s (seals, vents, the
  permeable weave of tent fabric). Double glazing adds a mass-air-mass dip at
  f0 = 60 sqrt((m1 + m2) / (m1 m2 d)). An opened window adds its open area to s
  (window_open blends toward the open aperture). Open structures (porch, cave mouth,
  underpass) only shield, a few dB rising with frequency.
* Transmitted sound is mostly diffuse: higher spherical-harmonic orders are tapered,
  and an open window re-radiates as an aperture in its own direction.
* Rooms: a feedback delay network (Jot & Chaigne 1991; Rocchesso & Smith 1997) with an
  orthogonal Hadamard feedback matrix and per-line gains g = 10^(-3 d / (RT60 sr)); a
  one-pole absorption filter in the loop shortens the high-frequency decay (Jot's
  b = ln(10)/4 log10(g) (1 - 1/alpha^2), alpha = RT60(Nyquist)/RT60(DC)). Line k drives
  ACN channel k with gain sqrt(L / (2n + 1)), the SN3D energy split of a diffuse field.
* Car wipers (FMVSS 104 S4.1.1): at least two frequencies, >= 45 and >= 20 cycles/min,
  differing by >= 15; here slow 40, fast 62, intermittent one wipe every ~7.5 s. Rubber
  on glass chatters by stick-slip when the glass is barely wet and swishes when it is
  wet; the linkage thunks and the blade flips at each reversal.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import signal

from raingen.core.constants import AMBI_ORDER, RHO_AIR, SAMPLE_RATE
from raingen.core.types import BlockContext, Emission
from raingen.physics.layout import normalize_shelter
from raingen.spatial.sh import sh_real

from .wind import (PendingMixer, Ramp, SparseModalBank, StatefulSOS, bandpass_sos, block_start_sample, db_spl,
                   graphic_eq_sos, highpass_sos, noise_rms_gain, shelter_of, smoothstep)

BANDS_HZ = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
EXTERIOR_STEMS = ("rain_far", "wind", "thunder", "ambience", "aftermath")
ENCLOSED_SHELTERS = ("car", "tent", "room_open", "room_closed", "attic")
_RHO_C = RHO_AIR * 343.0

# Envelope models: surface mass (kg/m^2), leak/open-area fraction, double-leaf dip,
# geometric shielding (dB at 1 kHz) and per-order SH taper of the transmitted field.
SHELTER_ENVELOPE = {
    "car": {"mass": 8.0, "leak": 2.0e-4, "taper": (1.0, 0.55, 0.3, 0.15)},
    "tent": {"mass": 0.15, "leak": 0.12, "taper": (1.0, 0.9, 0.75, 0.6)},
    "room_closed": {"mass": 20.0, "leak": 5.0e-5, "dip_hz": 215.0, "dip_db": 7.0, "taper": (1.0, 0.35, 0.15, 0.08)},
    "room_open": {"mass": 20.0, "leak": 5.0e-5, "dip_hz": 215.0, "dip_db": 7.0, "taper": (1.0, 0.35, 0.15, 0.08)},
    "attic": {"mass": 9.0, "leak": 1.0e-3, "taper": (1.0, 0.45, 0.2, 0.1)},
    "porch": {"shield_db": 1.5, "taper": (1.0, 0.95, 0.9, 0.85)},
    "cave": {"shield_db": 3.0, "taper": (1.0, 0.8, 0.6, 0.45)},
    "underpass": {"shield_db": 3.0, "taper": (1.0, 0.85, 0.7, 0.55)},
}
WINDOW_AREA_SHARE = 0.25        # share of the transmitting envelope taken by an opened window
WINDOW_AZIMUTH = 0.0            # the room window is ahead of the listener (raingen.physics.layout)


def mass_law_tl_db(f_hz, mass_kg_m2: float, leak: float = 0.0, dip_hz: float = 0.0, dip_db: float = 0.0) -> np.ndarray:
    """Field-incidence mass-law transmission loss with leakage and an optional mass-air-mass dip."""
    f = np.asarray(f_hz, dtype=float)
    x = math.pi * f * max(mass_kg_m2, 0.0) / _RHO_C
    tl0 = 10.0 * np.log10(1.0 + x * x)
    tl = np.maximum(tl0 - 5.0 * np.clip(tl0 / 15.0, 0.0, 1.0), 0.0)
    if dip_hz > 0.0 and dip_db > 0.0:
        tl = np.maximum(tl - dip_db * np.exp(-(np.log2(f / dip_hz) / 0.6) ** 2), 0.0)
    s = min(max(float(leak), 0.0), 1.0)
    tau = (1.0 - s) * 10.0 ** (-tl / 10.0) + s
    return -10.0 * np.log10(np.maximum(tau, 1e-12))


def shelter_tl_db(shelter: str, window_open: float = 0.0, f_hz=BANDS_HZ) -> np.ndarray:
    """Transmission loss (dB, positive = attenuation) of the shelter envelope at f_hz."""
    f = np.asarray(f_hz, dtype=float)
    env = SHELTER_ENVELOPE.get(shelter)
    if env is None:
        return np.zeros_like(f)
    if "shield_db" in env:
        return env["shield_db"] * np.clip((f / 1000.0) ** 0.35, 0.3, 2.5)
    leak = env["leak"]
    if shelter in ("room_open", "room_closed"):
        leak = leak + WINDOW_AREA_SHARE * min(max(float(window_open), 0.0), 1.0)
    return mass_law_tl_db(f, env["mass"], leak, env.get("dip_hz", 0.0), env.get("dip_db", 0.0))


def _effective_window_open(shelter: str, window_open: float) -> float:
    wo = min(max(float(window_open or 0.0), 0.0), 1.0)
    if shelter == "room_open" and wo <= 0.0:
        return 0.5                      # "room, window open" with no amount given: half open
    return wo


def room_for_shelter(shelter: str, window_open: float = 0.0) -> tuple[float, float]:
    """(RT60 s, wet gain) of the listener's enclosure; (0, 0) when outdoors."""
    s = shelter_of_name(shelter)
    wo = _effective_window_open(s, window_open)
    table = {
        "tent": (0.12, 0.06),
        "car": (0.08, 0.25),
        "porch": (0.35, 0.08),
        "room_closed": (0.45, 0.22),
        "room_open": (0.45, 0.20),
        "attic": (0.7, 0.3),
        "cave": (1.8, 0.35),
        "underpass": (1.4, 0.3),
    }
    rt, wet = table.get(s, (0.0, 0.0))
    if s in ("room_open", "room_closed") and wo > 0.0:
        rt *= 1.0 - 0.15 * wo           # the open window is a perfect absorber
        wet *= 1.0 - 0.2 * wo
    return float(rt), float(wet)


def shelter_of_name(name: str) -> str:
    return normalize_shelter(name or "open")


# --------------------------------------------------------------------------
# Envelope transmission
# --------------------------------------------------------------------------

class ShelterTransfer:
    """Exterior stems as heard from inside the listener's shelter.

    `process(stem, bus)` applies the envelope transmission loss (mass law + leaks, window
    blend) and the spatial transform (order taper, window aperture) to exterior stems:
    rain_far, wind, thunder, ambience, aftermath, and rain_near for enclosures (in a car
    or tent the near ground rain is outside too). Interior stems and every stem in an
    open shelter (open, tree, umbrella) pass unchanged. Filters keep state per stem.
    `room` / `room_wet` give the enclosure's reverberation for the caller to mix.
    """

    def __init__(self, scene, sr: int = SAMPLE_RATE, order: int = AMBI_ORDER):
        self.sr = int(sr)
        self.order = int(order)
        self.n_ch = (self.order + 1) ** 2
        self.acn_order = np.floor(np.sqrt(np.arange(self.n_ch))).astype(int)
        self.shelter = shelter_of(scene)
        self._filters: dict[str, StatefulSOS] = {}
        self.window_open = -1.0
        self.room: Optional[Room] = None
        self.room_wet = 0.0
        self.set_window_open(float(getattr(scene.place, "window_open", 0.0) or 0.0))

    @property
    def active(self) -> bool:
        return self.shelter in SHELTER_ENVELOPE

    def stems(self) -> tuple:
        if not self.active:
            return ()
        if self.shelter in ENCLOSED_SHELTERS:
            return EXTERIOR_STEMS + ("rain_near",)
        return EXTERIOR_STEMS

    def set_window_open(self, window_open: float) -> None:
        """Redesign for a new window opening (state is kept, so changes do not click)."""
        wo = _effective_window_open(self.shelter, window_open)
        if abs(wo - self.window_open) < 1e-3:
            return
        self.window_open = wo
        self.tl_db = shelter_tl_db(self.shelter, wo)
        if self.active and float(np.max(self.tl_db)) > 0.05:
            self.sos = graphic_eq_sos(BANDS_HZ, -self.tl_db, self.sr, q=1.2, iterations=5)
        else:
            self.sos = None
        for f in self._filters.values():
            if self.sos is not None:
                f.set(self.sos)
        env = SHELTER_ENVELOPE.get(self.shelter, {})
        taper = np.array(env.get("taper", (1.0, 1.0, 1.0, 1.0)), dtype=float)
        taper = np.concatenate([taper, np.full(max(0, self.order + 1 - taper.size), taper[-1])])[: self.order + 1]
        self.aperture = 0.0
        if self.shelter in ("room_open", "room_closed") and wo > 0.0:
            # through an open window the exterior arrives from the window, not through the walls
            self.aperture = 0.7 * math.sqrt(wo)
            taper = taper * (1.0 - 0.5 * math.sqrt(wo)) + 0.5 * math.sqrt(wo) * np.array(
                [1.0] + [0.0] * self.order)[: self.order + 1]
        self.channel_gain = taper[self.acn_order]
        y = np.asarray(sh_real(self.order, WINDOW_AZIMUTH, 0.0), dtype=float)
        self.aperture_y = y[: self.n_ch]
        rt, wet = room_for_shelter(self.shelter, wo)
        self.room_wet = wet
        if rt > 0.0 and wet > 0.0:
            if self.room is None or abs(self.room.rt60 - rt) > 0.02:
                self.room = Room(rt, self.sr, self.order, hf_ratio=0.45 if self.shelter != "cave" else 0.6)
        else:
            self.room = None

    def process(self, stem: str, bus: np.ndarray) -> np.ndarray:
        bus = np.asarray(bus, dtype=float)
        if stem not in self.stems():
            return bus
        out = bus
        if self.sos is not None:
            filt = self._filters.get(stem)
            if filt is None:
                filt = StatefulSOS(self.sos)
                self._filters[stem] = filt
            out = filt.process(bus)
        c = min(out.shape[0], self.n_ch)
        res = np.zeros_like(out)
        res[:c] = out[:c] * self.channel_gain[:c, None]
        if self.aperture > 0.0 and c > 1:
            res[1:c] += self.aperture * self.aperture_y[1:c, None] * out[0][None, :]
        return res

    def room_process(self, bus: np.ndarray) -> np.ndarray:
        """The enclosure's reverberation of `bus`, already scaled by `room_wet` (zeros outdoors)."""
        bus = np.asarray(bus, dtype=float)
        if self.room is None:
            return np.zeros_like(bus)
        return self.room.process(bus) * self.room_wet


# --------------------------------------------------------------------------
# SH-domain FDN reverb
# --------------------------------------------------------------------------

def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h[:n, :n] / math.sqrt(n)


def _primes_near(values) -> list[int]:
    out = []
    used = set()
    for v in values:
        k = max(int(round(v)), 2)
        while True:
            if k not in used and all(k % p for p in range(2, int(math.isqrt(k)) + 1)):
                break
            k += 1
        used.add(k)
        out.append(k)
    return out


class Room:
    """Spherical-harmonic-domain feedback delay network reverberator.

    `process(bus)` takes an (N, n) AmbiX bus and returns the reverberant part only
    (mix it with the `wet` gain from room_for_shelter). The tail is diffuse: L decorrelated
    lines, line k feeding ACN channel k with the SN3D diffuse-field energy split. Blocks of
    any length are processed in sub-blocks no longer than the shortest delay, so the
    recursion is exact and vectorized.
    """

    def __init__(self, rt60_s: float, sr: int = SAMPLE_RATE, order: int = AMBI_ORDER, *,
                 hf_ratio: float = 0.5, size_m: Optional[float] = None, lines: Optional[int] = None,
                 directional_input: float = 0.25):
        self.sr = int(sr)
        self.order = int(order)
        self.n_ch = (self.order + 1) ** 2
        self.L = int(lines) if lines else max(8, 1 << int(math.ceil(math.log2(self.n_ch))))
        self.rt60 = max(float(rt60_s), 0.02)
        size = float(size_m) if size_m else min(max(2.0 + 8.0 * self.rt60, 1.5), 30.0)
        d_lo = size / 343.0 * self.sr
        d_hi = 3.0 * d_lo
        targets = d_lo * (d_hi / d_lo) ** (np.arange(self.L) / max(self.L - 1, 1))
        self.delays = np.array(_primes_near(targets), dtype=int)
        self.min_delay = int(self.delays.min())
        self.size = int(self.delays.max()) + 1
        self.buf = np.zeros((self.L, self.size))
        self.w = 0
        self.A = _hadamard(self.L)
        rng = np.random.default_rng(0x6F646E66)            # fixed sign pattern (no runtime randomness)
        self.A = self.A * np.where(rng.random(self.L) < 0.5, -1.0, 1.0)[None, :]
        self.g = 10.0 ** (-3.0 * self.delays / (self.rt60 * self.sr))
        alpha = min(max(float(hf_ratio), 0.05), 1.0)
        g_mean = 10.0 ** (-3.0 * float(np.mean(self.delays)) / (self.rt60 * self.sr))
        self.b = (math.log(10.0) / 4.0) * math.log10(g_mean) * (1.0 - 1.0 / (alpha * alpha))
        self.b = min(max(self.b, 0.0), 0.95)
        self.damp_zi = np.zeros((self.L, 1))
        acn = np.floor(np.sqrt(np.arange(self.n_ch))).astype(int)
        w_in = np.where(acn == 0, 1.0, float(directional_input))
        self.B = _hadamard(self.L)[:, : self.n_ch] * w_in[None, :]
        out_g = np.zeros(self.L)
        out_g[: self.n_ch] = np.sqrt(self.L / (2.0 * acn + 1.0))
        self.out_g = out_g
        self.energy = 0.0

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.damp_zi[:] = 0.0
        self.energy = 0.0

    def process(self, bus: np.ndarray) -> np.ndarray:
        bus = np.asarray(bus, dtype=float)
        if bus.ndim == 1:
            bus = bus[None, :]
        c = min(bus.shape[0], self.n_ch)
        n = bus.shape[1]
        out = np.zeros((max(bus.shape[0], self.n_ch), n))
        in_energy = float(np.sum(bus[:c] ** 2))
        if in_energy == 0.0 and self.energy < 1e-14:
            return out
        x = self.B[:, :c] @ bus[:c]                        # (L, n) line inputs
        y_all = np.zeros((self.L, n))
        rows = np.arange(self.L)[:, None]
        pos = 0
        while pos < n:
            m = min(n - pos, self.min_delay)
            k = np.arange(m)
            ridx = (self.w - self.delays[:, None] + k[None, :]) % self.size
            y = self.buf[rows, ridx]
            y, self.damp_zi = signal.lfilter([1.0 - self.b], [1.0, -self.b], y, axis=-1, zi=self.damp_zi)
            y = y * self.g[:, None]
            y_all[:, pos:pos + m] = y
            widx = (self.w + k) % self.size
            self.buf[:, widx] = x[:, pos:pos + m] + self.A @ y
            self.w = (self.w + m) % self.size
            pos += m
        out[: self.n_ch] = y_all[: self.n_ch] * self.out_g[: self.n_ch, None]
        self.energy = float(np.sum(self.buf ** 2))
        return out


# --------------------------------------------------------------------------
# Car wipers
# --------------------------------------------------------------------------

WIPER_CPM = {"slow": 40.0, "low": 40.0, "fast": 62.0, "high": 62.0}
INTERMITTENT_PERIOD_S = 7.5


class CarWipersGenerator:
    """Windscreen wipers heard from the driver's seat (stem 'interior').

    Modes: off | intermittent | slow | fast | auto (speed from the rain rate). One cycle is
    park -> top -> park with sinusoidal blade speed; reversals thunk (linkage) and click
    (blade flip); the blade swishes on wet glass and chatters on barely wet glass; the
    motor hums with its load.
    """

    name = "car_wipers"

    def __init__(self, scene, layout=None, rngf=None, sr: int = SAMPLE_RATE):
        self.sr = int(sr)
        self.mode = "off"
        self.set_speed(str(getattr(scene.place, "wipers", "off") or "off"))
        ws = None
        for p in layout or []:
            if p.id.startswith("shelter.car.windshield"):
                ws = p
                break
        self.pos = tuple(ws.position) if ws is not None else (0.75, -0.3, 0.05)
        x, y, z = self.pos
        self.az = math.atan2(y, x)
        self.el = math.atan2(z, math.hypot(x, y))
        self.dist = max(0.3, math.sqrt(x * x + y * y + z * z))
        cfg = rngf.get("wipers.cfg", 0) if rngf is not None else np.random.default_rng(7)
        self.f_motor = float(cfg.uniform(85.0, 115.0))
        self.f_squeal = float(cfg.uniform(1300.0, 2300.0))
        self.theta = 0.0              # cycle phase 0..1 (0 = parked)
        self.running = False
        self.dwell = 0.0
        self.motor_phase = 0.0
        self.swish = StatefulSOS(bandpass_sos(2200.0, 0.7, self.sr))
        self.swish_fc = 2200.0
        self.swish_g = noise_rms_gain(self.swish.sos)
        self.slosh = StatefulSOS(bandpass_sos(550.0, 1.0, self.sr))
        self.slosh_g = noise_rms_gain(self.slosh.sos)
        self.squeal = StatefulSOS(bandpass_sos(self.f_squeal, 5.0, self.sr))
        self.squeal_g = noise_rms_gain(self.squeal.sos)
        self.slip_phase = 0.0
        self.thunk = SparseModalBank([110.0 * float(cfg.uniform(0.9, 1.1)), 240.0, 430.0], [0.09, 0.06, 0.04], self.sr)
        self.clicks = PendingMixer(1)
        self.motor_amp = Ramp(0.0)
        self.blade_amp = Ramp(0.0)
        self.hp = StatefulSOS(highpass_sos(40.0, self.sr))

    def set_speed(self, mode: str) -> None:
        m = str(mode or "off").lower()
        if m in ("int", "interval", "intermittent", "delay"):
            m = "intermittent"
        if m not in ("off", "intermittent", "slow", "low", "fast", "high", "auto"):
            m = "off"
        self.mode = m

    def _cpm(self, rain: float) -> tuple[float, bool]:
        """(cycles per minute while sweeping, intermittent?)"""
        m = self.mode
        if m == "auto":
            if rain < 0.05:
                return 0.0, False
            if rain < 1.0:
                return WIPER_CPM["slow"], True
            return (WIPER_CPM["slow"], False) if rain < 6.0 else (WIPER_CPM["fast"], False)
        if m == "intermittent":
            return WIPER_CPM["slow"], True
        return WIPER_CPM.get(m, 0.0), False

    def render(self, ctx: BlockContext) -> list[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        dt = n / sr
        rain = float(w.rain_rate_mm_h) if w.precip_type not in ("snow", "none") else 0.3 * float(w.rain_rate_mm_h)
        cpm, inter = self._cpm(rain)

        # cycle clock: sweep, or dwell parked between intermittent wipes. Switched off
        # mid-sweep, the blades finish the cycle at the slow rate and park.
        eff_cpm = cpm if cpm > 0.0 else WIPER_CPM["slow"]
        park_after = inter or cpm <= 0.0
        if cpm > 0.0 and not self.running:
            self.dwell -= dt
            if not inter or self.dwell <= 0.0:
                self.running = True
        idle = (not self.running and self.theta == 0.0 and float(self.motor_amp.value) < 1e-9
                and float(self.blade_amp.value) < 1e-9 and not self.clicks.active and self.thunk.silent())
        if idle:
            return []
        rng = ctx.rng.get("wipers", ctx.index)
        start = block_start_sample(ctx)
        rate = eff_cpm / 60.0 / sr
        was_running = self.running
        old_theta = self.theta
        theta = np.full(n, self.theta)
        if self.running:
            th = self.theta + rate * np.arange(1, n + 1)
            if park_after and th[-1] >= 1.0:
                cut = int(np.argmax(th >= 1.0))
                th[cut:] = 0.0
                self.running = False
                self.dwell = (INTERMITTENT_PERIOD_S - 60.0 / eff_cpm) if inter else 0.0
                self.theta = 0.0
            else:
                self.theta = float(th[-1] % 1.0)
            theta = th
        prev = np.concatenate(([old_theta if was_running else theta[0]], theta[:-1]))
        speed = np.abs(np.sin(2.0 * math.pi * theta)) * (theta > 0.0)
        moving = float(np.mean(theta > 0.0))

        # glass wetness: water collected on the glass during one wipe period
        period = 60.0 / max(cpm, 1.0) if not inter else INTERMITTENT_PERIOD_S
        film_mm = rain / 3600.0 * period
        wet = max(smoothstep(film_mm, 0.0002, 0.003), 0.6 * float(w.wetness.get("glass", 0.0)) * (1.0 if inter else 0.4))
        wet = min(wet, 1.0)

        # reversals: top (theta crosses 0.5) and park (theta wraps / returns to 0)
        half = np.nonzero((np.floor(prev * 2.0) != np.floor(theta * 2.0)) & (np.abs(theta - prev) < 0.5))[0]
        park = np.nonzero((prev > 0.9) & (theta < 0.1))[0]
        strikes = []
        for off in np.concatenate([half, park]):
            a = float(rng.lognormal(0.0, 0.2))
            strikes.append((int(off), a * np.array([1.0, 0.5, 0.25])))
            click = self._click(rng, sr) * db_spl(52.0) * a
            self.clicks.add(start + int(off) + int(0.004 * sr), click)
        strikes.sort(key=lambda s: s[0])
        y_blade = self.thunk.process(n, strikes) * db_spl(60.0)
        if self.clicks.active:
            y_blade = y_blade + self.clicks.take(start, n)

        # blade on glass: swish when wet, stick-slip chatter/squeal when nearly dry
        fc = 1600.0 + 1400.0 * wet
        if abs(fc - self.swish_fc) / self.swish_fc > 0.05:
            self.swish_fc = fc
            self.swish.set(bandpass_sos(fc, 0.7, sr))
            self.swish_g = noise_rms_gain(self.swish.sos)
        blade_target = db_spl(50.0) if (self.running or moving > 0.0) else 0.0
        env = self.blade_amp(blade_target, n) * speed
        swish = self.swish.process(rng.standard_normal(n)) / self.swish_g * (0.35 + 0.65 * wet)
        slosh = self.slosh.process(rng.standard_normal(n)) / self.slosh_g * min(rain / 20.0, 1.0)
        dry = (1.0 - wet) ** 1.5
        f_slip = 280.0 + 500.0 * speed
        ph = self.slip_phase + np.cumsum(f_slip) / sr
        self.slip_phase = float(ph[-1] % 1.0)
        slips = np.nonzero(np.diff(np.floor(np.concatenate(([ph[0] - f_slip[0] / sr], ph)))) > 0)[0]
        imp = np.zeros(n)
        p_slip = 0.5 + 0.5 * dry
        if slips.size:
            imp[slips] = rng.lognormal(0.0, 0.4, size=slips.size) * (rng.random(slips.size) < p_slip)
        slip_rate = max(float(np.mean(f_slip)) * p_slip, 1.0)
        chatter = self.squeal.process(imp) / (self.squeal_g * math.sqrt(slip_rate / sr * 1.377))
        y_blade = y_blade + env * (swish + 0.8 * slosh + 1.6 * dry * chatter)

        # motor and gearbox: pitch sags slightly with the load at the reversals
        motor_target = db_spl(42.0) if (self.running or moving > 0.0) else 0.0
        load = 0.75 + 0.25 * speed
        f_m = self.f_motor * (eff_cpm / 40.0) ** 0.3 * (1.0 - 0.04 * (1.0 - speed))
        mph = self.motor_phase + 2.0 * math.pi * np.cumsum(np.broadcast_to(f_m, (n,))) / sr
        self.motor_phase = float(mph[-1] % (2.0 * math.pi))
        hum = (np.sin(mph) + 0.5 * np.sin(2.0 * mph + 0.4) + 0.3 * np.sin(3.0 * mph + 1.2)
               + 0.12 * np.sin(7.0 * mph + 2.1))
        y_motor = self.motor_amp(motor_target, n) * load * hum / 1.2

        y = self.hp.process(y_blade + y_motor)
        return [Emission(signal=y, azimuth=self.az, elevation=self.el, distance=self.dist, spread=0.45,
                         stem="interior", source_id="car.wipers")]

    def _click(self, rng: np.random.Generator, sr: int) -> np.ndarray:
        L = int(0.004 * sr)
        t = np.arange(L) / sr
        b = rng.standard_normal(L) * np.exp(-t / 0.0008)
        b = signal.sosfilt(bandpass_sos(rng.uniform(2500.0, 4000.0), 1.5, sr), b)
        pk = float(np.max(np.abs(b)))
        return b / pk if pk > 0 else b


__all__ = [
    "ShelterTransfer", "Room", "room_for_shelter", "CarWipersGenerator", "shelter_tl_db", "mass_law_tl_db",
    "EXTERIOR_STEMS", "ENCLOSED_SHELTERS", "BANDS_HZ",
]
