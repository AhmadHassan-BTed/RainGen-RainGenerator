"""Wellness modes and the "Keep it soothing" governor (MASTER_PLAN 8.0-8.3, 2.3).

`apply_mode(scene, mode)` sets the scene guard-rails of each mode:

* **calm** (shown as *Relax*): full ranges; 40-55 dBA; thunder distant only
  (>= 8 km, low-passed <= 600 Hz, attack >= 200 ms, <= 1 per 2 min);
  event-to-bed <= 6 dB.
* **sleep**: ~35-42 dBA (a hypothesis, not proven REM-safe: Basner et al.
  2026); rain tops out at "heavy", wind <= force 5, thunder >= 8 km and <= 1 per
  2-5 min (low-passed <= 400 Hz, attack >= 300 ms); timer 45 min with a 15 min
  "Fade out as I fall asleep" (-12 dB, darker tilt, events -> 0); a dense,
  fused texture (event-to-bed <= 3 dB); "No sudden loud sounds" always on.
* **focus**: -5 dB/octave tilt (125 Hz-8 kHz), 42-48 dBA, steady state only,
  salient events <= 0.5/min, gust swells <= +-2 dB, no thunder, no birds,
  pitched drips or rhythmic gutters (the irrelevant-sound effect; Hongisto 2017).
* **asmr**: opt-in, binaural, close triggers (window tap, umbrella, tin, fabric,
  drips, leaves; each removable) at 5-50 cm moving at 2-8 cm/s; exempt from the
  soothing limits except the startle limiter, with a Tingle <-> Comfort control
  (the Comfort end applies the soothing sharpness and attack targets; Honda 2020).

`CalmGovernor` is deterministic and feed-forward: it analyses the *incoming*
audio on a fixed 100 ms hop aligned to absolute sample time (never to host
blocks), so its decisions do not depend on the block size. Per hop it measures
a loudness proxy (the BS.1770 sum of channel mean squares with its -0.691 dB
offset, not K-weighted), a sharpness proxy (DIN 45692 weighting g(z) over Bark bands with a
Zwicker-like N' ~ E^0.23 specific loudness, no masking spread), the PSD tilt
(dB/octave over 125 Hz-8 kHz; pink noise = -3) and a fluctuation proxy (1-20 Hz
modulation of the 10 ms envelope over 2.56 s). It then slews a broadband gain
(never a boost) and a 1 kHz high-shelf tilt correction (slew-limited, never
periodic), and applies a look-ahead event-to-bed limiter (startle guard). It
never optimises toward the Aures/Zwicker "pleasantness" score.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

from ..core.constants import SAMPLE_RATE
from ..scene.schema import Scene, SceneObject
from ..scene.units import beaufort_to_mps
from ..scene.vibe import turbulence_from_gust

MODES = ("none", "sleep", "focus", "calm", "asmr")
MODE_ALIASES = {"relax": "calm", "relaxed": "calm", "cinematic": "none", "off": "none", "": "none",
                "studio": "none", "game": "none", "music": "none"}
MODE_LABELS = {"none": "Off", "sleep": "Sleep", "focus": "Focus", "calm": "Relax", "asmr": "ASMR"}

_NO_THUNDER_LIMIT = {"allowed": True, "min_distance_km": 0.3, "max_per_min": 20.0, "lowpass_hz": None, "attack_ms": None}

GOVERNOR_SETTINGS: dict[str, dict[str, Any]] = {
    "none": {
        "soothing": False, "level_dba": None, "ceiling_lufs": None, "tilt_db_oct": None, "max_sharpness_acum": None,
        "event_to_bed_db": None, "event_attack_ms": None, "fluct_limit_db": None, "gust_depth_db": None,
        "max_gust_factor": None, "max_beaufort": None, "salient_events_per_min": None,
        "thunder": dict(_NO_THUNDER_LIMIT),
        "timer_min": 0.0, "descent_min": 0.0, "descent_db": 0.0, "descent_tilt_db": 0.0,
        "startle_guard": False, "volume_db": -12.0, "rate_range": None, "character": None,
        "remove_emitters": (), "biophony": {}, "slew_db_per_s": 0.5,
    },
    "calm": {
        "soothing": True, "level_dba": (40.0, 55.0), "ceiling_lufs": -20.0, "tilt_db_oct": (-6.0, -3.0),
        "max_sharpness_acum": 1.5, "event_to_bed_db": 6.0, "event_attack_ms": 50.0, "fluct_limit_db": 3.0,
        "gust_depth_db": 3.0, "max_gust_factor": None, "max_beaufort": None, "salient_events_per_min": 2.0,
        "thunder": {"allowed": True, "min_distance_km": 8.0, "max_per_min": 0.5, "lowpass_hz": 600.0, "attack_ms": 200.0},
        "timer_min": 0.0, "descent_min": 0.0, "descent_db": 0.0, "descent_tilt_db": 0.0,
        "startle_guard": False, "volume_db": -10.0, "rate_range": None, "character": None,
        "remove_emitters": (), "biophony": {}, "slew_db_per_s": 0.5,
    },
    "sleep": {
        "soothing": True, "level_dba": (35.0, 42.0), "ceiling_lufs": -30.0, "tilt_db_oct": (-6.0, -4.5),
        "max_sharpness_acum": 1.3, "event_to_bed_db": 3.0, "event_attack_ms": 80.0, "fluct_limit_db": 1.0,
        "gust_depth_db": 2.0, "max_gust_factor": 1.3, "max_beaufort": 5.0, "salient_events_per_min": 0.2,
        "thunder": {"allowed": True, "min_distance_km": 8.0, "max_per_min": 0.3, "lowpass_hz": 400.0, "attack_ms": 300.0},
        "timer_min": 45.0, "descent_min": 15.0, "descent_db": -12.0, "descent_tilt_db": -4.0,
        "startle_guard": True, "volume_db": -18.0, "rate_range": (1.0, 20.0), "character": "continuous",
        "remove_emitters": ("chime", "flag", "wire"),
        "biophony": {"birds": "off", "insects": "off", "frogs": "off", "traffic": 0.0}, "slew_db_per_s": 0.25,
    },
    "focus": {
        "soothing": True, "level_dba": (42.0, 48.0), "ceiling_lufs": -24.0, "tilt_db_oct": (-5.5, -4.5),
        "max_sharpness_acum": 1.5, "event_to_bed_db": 4.0, "event_attack_ms": 50.0, "fluct_limit_db": 1.5,
        "gust_depth_db": 2.0, "max_gust_factor": 1.25, "max_beaufort": None, "salient_events_per_min": 0.5,
        "thunder": {"allowed": False, "min_distance_km": 15.0, "max_per_min": 0.0, "lowpass_hz": 400.0, "attack_ms": 300.0},
        "timer_min": 0.0, "descent_min": 0.0, "descent_db": 0.0, "descent_tilt_db": 0.0,
        "startle_guard": True, "volume_db": -12.0, "rate_range": (2.0, 20.0), "character": "continuous",
        "remove_emitters": ("gutter", "downspout", "eave", "chime", "flag", "wire"),
        "biophony": {"birds": "off", "insects": "off", "frogs": "off", "traffic": 0.0}, "slew_db_per_s": 0.25,
    },
    "asmr": {
        # Comfort-end values; CalmGovernor interpolates toward the Tingle end with `tingle`.
        "soothing": False, "level_dba": (35.0, 50.0), "ceiling_lufs": -22.0, "tilt_db_oct": (-6.0, -3.0),
        "max_sharpness_acum": 1.5, "event_to_bed_db": 6.0, "event_attack_ms": 10.0, "fluct_limit_db": None,
        "gust_depth_db": None, "max_gust_factor": None, "max_beaufort": None, "salient_events_per_min": None,
        "thunder": dict(_NO_THUNDER_LIMIT),
        "timer_min": 0.0, "descent_min": 0.0, "descent_db": 0.0, "descent_tilt_db": 0.0,
        "startle_guard": True, "volume_db": -12.0, "rate_range": None, "character": None,
        "remove_emitters": ("flag", "wire"), "biophony": {"traffic": 0.0}, "slew_db_per_s": 0.5,
        "triggers": ("window_tap", "umbrella", "tin", "fabric", "drips", "leaves"),
        "near_field_m": (0.05, 0.50), "motion_cm_s": (2.0, 8.0),
        "tingle_end": {"max_sharpness_acum": 2.2, "event_to_bed_db": 12.0, "tilt_db_oct": (-4.5, -1.5)},
    },
}

ASMR_TRIGGERS = GOVERNOR_SETTINGS["asmr"]["triggers"]


def canonical_mode(mode: Optional[str]) -> str:
    """'Relax' -> 'calm', 'Cinematic' -> 'none' (no soothing limits), etc."""
    m = (mode or "none").strip().lower()
    m = MODE_ALIASES.get(m, m)
    if m not in GOVERNOR_SETTINGS:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES} (or 'relax', 'cinematic')")
    return m


def _ear_frame(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lx, ly, lz = (float(v) for v in scene.listener.position)
    ear = np.array([lx, ly, lz + float(scene.listener.height_m)])
    h = math.radians(float(scene.listener.heading_deg))
    fwd = np.array([math.sin(h), math.cos(h), 0.0])        # world: x east, y north
    left = np.array([-math.cos(h), math.sin(h), 0.0])
    return ear, fwd, left


def _asmr_objects(scene: Scene, triggers: tuple, tingle: float) -> list[SceneObject]:
    """Near-field trigger objects in world coordinates around the listener's ears."""
    ear, fwd, left = _ear_frame(scene)
    up = np.array([0.0, 0.0, 1.0])
    crisp = {"tingle": float(tingle), "crispness": 0.3 + 0.7 * float(tingle)}
    spec = {
        # name: (object type, material, position, size, slow motion cm/s)
        "window_tap": ("window", "glass", ear + 0.28 * fwd + 0.05 * left, (0.6, 0.0, 0.8), 3.0),
        "umbrella": ("umbrella", "umbrella", ear + 0.30 * up, (0.9, 0.9, 0.0), 0.0),
        "tin": ("surface", "tin_roof", ear + 0.22 * left - 0.08 * up, (0.3, 0.3, 0.0), 4.0),
        "fabric": ("surface", "fabric", ear - 0.20 * left + 0.05 * fwd, (0.4, 0.4, 0.0), 2.5),
        "drips": ("surface", "bucket", ear + 0.25 * fwd - 0.25 * left - 0.35 * up, (0.25, 0.25, 0.0), 0.0),
        "leaves": ("surface", "leaves", ear - 0.25 * fwd + 0.15 * left - 0.30 * up, (0.4, 0.4, 0.0), 5.0),
    }
    out = []
    for name in triggers:
        if name not in spec:
            continue
        typ, mat, pos, size, motion = spec[name]
        if name == "umbrella" and scene.place.shelter == "umbrella":
            continue                     # the shelter itself provides the membrane
        dist = float(np.linalg.norm(pos - ear))
        params = {"asmr": True, "trigger": name, "motion_cm_s": motion, "near_field": True,
                  "distance_m": dist, "experimental_near_field": dist < 0.25, **crisp}
        if name == "drips":
            params.update({"drip_rate_hz": 0.8, "fall_height_m": 0.4})
        out.append(SceneObject(id=f"asmr_{name}", type=typ, material=mat,
                               position=(float(pos[0]), float(pos[1]), float(pos[2])),
                               size=size, rotation_deg=float(scene.listener.heading_deg), params=params))
    return out


def _guard_summary(mode: str, s: dict) -> dict:
    """What the generators and the UI badge need to know about the active limits."""
    th = s["thunder"]
    pair = lambda v: list(v) if isinstance(v, tuple) else v      # JSON-native, so scenes round-trip equal
    return {
        "mode": mode, "label": MODE_LABELS[mode], "soothing": bool(s["soothing"]),
        "level_dba": pair(s["level_dba"]), "tilt_db_oct": pair(s["tilt_db_oct"]),
        "event_to_bed_db": s["event_to_bed_db"], "event_attack_ms": s["event_attack_ms"],
        "gust_depth_db": s["gust_depth_db"], "salient_events_per_min": s["salient_events_per_min"],
        "thunder_lowpass_hz": th["lowpass_hz"], "thunder_attack_ms": th["attack_ms"],
        "thunder_max_per_min": th["max_per_min"] if th["allowed"] else 0.0,
        "no_pitched_drips": mode == "focus", "startle_guard": bool(s["startle_guard"]),
    }


def apply_mode(scene: Scene, mode: str) -> Scene:
    """Copy of `scene` with the wellness mode's guard-rails applied (MASTER_PLAN 8.3, 2.3).

    The mode only tightens the weather (it never makes it wilder); its limits and the
    timer land in `scene.vibe` ('guard', 'timer_min', 'descent_min', 'volume_db',
    'startle_guard'), and ASMR trigger objects are added to `scene.place.objects`.
    """
    mode = canonical_mode(mode)
    s = GOVERNOR_SETTINGS[mode]
    prev_mode = scene.mode
    out = scene.copy()
    out.mode = mode
    vibe = dict(out.vibe or {})
    out.place.objects = [o for o in out.place.objects if not (isinstance(o.params, dict) and o.params.get("asmr"))]
    if prev_mode == "sleep" and mode != "sleep" and float(vibe.get("timer_min") or 0.0) == GOVERNOR_SETTINGS["sleep"]["timer_min"]:
        vibe["timer_min"] = 0.0                       # the Sleep default timer leaves with Sleep
    vibe["guard"] = _guard_summary(mode, s)
    if mode == "none":
        vibe["timer_min"] = float(vibe.get("timer_min", 0.0) or 0.0)
        vibe["descent_min"] = 0.0
        vibe["startle_guard"] = False
        out.vibe = vibe
        return out

    w = out.weather
    pr = w.precipitation
    wind = w.wind
    th = w.thunder

    # rain: a dense, steady texture inside the mode's range
    if s["rate_range"] is not None and pr.type != "none":
        lo, hi = s["rate_range"]
        if pr.type == "drizzle":
            pr.rate_mm_h = float(min(max(pr.rate_mm_h, 0.2), 1.0))
        else:
            pr.rate_mm_h = float(min(max(pr.rate_mm_h, lo), hi))
    if s["character"] is not None:
        pr.character = s["character"]
    elif s["soothing"] and pr.character == "showery":
        pr.shower_period_min = max(pr.shower_period_min, 10.0)       # slow swells only (< 0.25 Hz macro)
    if mode in ("sleep", "focus"):
        if pr.drops == "fat":
            pr.drops = "normal"
        if pr.type in ("hail", "graupel"):
            pr.type = "rain"
        if pr.type == "showers":
            pr.type = "rain"

    # wind: strength cap and gust depth
    if s["max_beaufort"] is not None:
        wind.speed_mps = float(min(wind.speed_mps, beaufort_to_mps(s["max_beaufort"])))
    if s["max_gust_factor"] is not None:
        g = wind.gust_factor
        wind.gust_factor = float(min(g, s["max_gust_factor"])) if g is not None else float(s["max_gust_factor"])
        wind.turbulence_intensity = turbulence_from_gust(wind.gust_factor)
        rate = wind.gust_rate_per_min
        wind.gust_rate_per_min = float(min(rate, 1.0)) if rate is not None else 0.5
    if mode in ("sleep", "focus"):
        wind.mic_rumble = False

    # thunder: distant only and rare, or off
    tl = s["thunder"]
    if th.enabled:
        if not tl["allowed"]:
            th.enabled = False
        else:
            th.distance_km = float(max(th.distance_km, tl["min_distance_km"]))
            th.flashes_per_min = float(min(th.flashes_per_min, tl["max_per_min"]))
            if s["soothing"]:
                th.approach_kmh = min(0.0, th.approach_kmh)          # staying put or moving away
                th.stroke_intensity = float(min(th.stroke_intensity, 0.4))
                th.cg_fraction = float(min(th.cg_fraction, 0.3))

    # changing-state sources
    if s["remove_emitters"]:
        out.place.emitters = [e for e in out.place.emitters if e.get("type") not in s["remove_emitters"]]
    if mode == "sleep":
        out.place.emitters = [dict(e) for e in out.place.emitters]
        for e in out.place.emitters:
            if e.get("type") in ("gutter", "downspout", "eave"):
                e["distance_m"] = float(max(float(e.get("distance_m", 4.0)), 6.0))
    bio = dict(out.place.biophony)
    bio.update(s["biophony"])
    out.place.biophony = bio

    # ASMR: near-field binaural triggers
    if mode == "asmr":
        out.listener.render = "binaural"
        if out.place.shelter == "open":
            out.place.shelter = "umbrella"
        tingle = vibe.get("tingle")
        tingle = float(min(1.0, max(0.0, float(tingle) if tingle is not None else 0.5)))
        exclude = set(vibe.get("asmr_exclude", []))
        triggers = tuple(t for t in s["triggers"] if t not in exclude)
        out.place.objects = list(out.place.objects) + _asmr_objects(out, triggers, tingle)
        vibe["tingle"] = tingle

    if mode == "sleep":
        vibe["volume_db"] = float(min(float(vibe.get("volume_db", s["volume_db"])), s["volume_db"]))
        vibe["timer_min"] = float(vibe.get("timer_min") or s["timer_min"])
    else:
        vibe.setdefault("volume_db", s["volume_db"])
        vibe["timer_min"] = float(vibe.get("timer_min", 0.0) or 0.0)
    vibe["descent_min"] = float(s["descent_min"])
    vibe["startle_guard"] = bool(s["startle_guard"])
    out.vibe = vibe
    return out


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------
def _bark(f: np.ndarray) -> np.ndarray:
    """Zwicker & Terhardt 1980 critical-band rate (Bark)."""
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _din45692_g(z: np.ndarray) -> np.ndarray:
    """DIN 45692 sharpness weighting g(z)."""
    return np.where(z <= 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)


def _high_shelf_sos(sr: float, f0: float, gain_db: float, slope: float = 0.5) -> np.ndarray:
    """RBJ audio-EQ-cookbook high shelf as one SOS section (gentle slope for spectral tilt)."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    sa = 2.0 * math.sqrt(a) * alpha
    b0 = a * ((a + 1) + (a - 1) * cw + sa)
    b1 = -2 * a * ((a - 1) + (a + 1) * cw)
    b2 = a * ((a + 1) + (a - 1) * cw - sa)
    a0 = (a + 1) - (a - 1) * cw + sa
    a1 = 2 * ((a - 1) - (a + 1) * cw)
    a2 = (a + 1) - (a - 1) * cw - sa
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


class CalmGovernor:
    """"Keep it soothing": hop-aligned loudness / sharpness / modulation regulator.

    process(x) takes (C, n) or (n,) float audio and returns the same shape, delayed by
    `latency` samples (the limiter look-ahead). All analysis and control decisions
    happen on 100 ms hops of absolute time, so the output does not depend on how the
    stream is split into blocks. `metrics` holds the latest measurements for meters
    and badges; `suggestions()` gives macro nudges for the engine's control pre-pass.
    """

    HOP_S = 0.100
    FRAME_S = 0.010
    HIST_FRAMES = 256            # 2.56 s envelope history -> 0.39 Hz modulation resolution
    LOOKAHEAD_S = 0.010
    HOLD_S = 0.060               # limiter hold before release
    FINAL_FADE_S = 10.0          # after the timer: silence over this long

    def __init__(self, sr: int = SAMPLE_RATE, mode: str = "calm", settings: Optional[dict] = None,
                 timer_min: Optional[float] = None, descent_min: Optional[float] = None,
                 tingle: Optional[float] = None):
        self.sr = int(sr)
        self.mode = canonical_mode(mode)
        self._init_args = (settings, timer_min, descent_min, tingle)
        s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in GOVERNOR_SETTINGS[self.mode].items()}
        if settings:
            s.update(settings)
        if self.mode == "asmr":
            tg = float(min(1.0, max(0.0, tingle if tingle is not None else s.get("tingle", 0.5))))
            end = s.get("tingle_end", {})
            s["tingle"] = tg
            s["max_sharpness_acum"] = s["max_sharpness_acum"] + tg * (end.get("max_sharpness_acum", s["max_sharpness_acum"]) - s["max_sharpness_acum"])
            s["event_to_bed_db"] = s["event_to_bed_db"] + tg * (end.get("event_to_bed_db", s["event_to_bed_db"]) - s["event_to_bed_db"])
            lo0, hi0 = s["tilt_db_oct"]
            lo1, hi1 = end.get("tilt_db_oct", (lo0, hi0))
            s["tilt_db_oct"] = (lo0 + tg * (lo1 - lo0), hi0 + tg * (hi1 - hi0))
            s["correction_strength"] = 1.0 - 0.7 * tg
        self.settings = s
        self.timer_s = float(timer_min if timer_min is not None else s["timer_min"]) * 60.0
        self.descent_s = float(descent_min if descent_min is not None else s["descent_min"]) * 60.0
        self.active = self.mode != "none"
        self.limiter_on = self.active and s["event_to_bed_db"] is not None
        self.hop = max(64, int(round(self.HOP_S * self.sr)))
        self.frame = max(16, int(round(self.FRAME_S * self.sr)))
        self.latency = int(round(self.LOOKAHEAD_S * self.sr)) if self.limiter_on else 0
        self._hold = int(round(self.HOLD_S * self.sr))
        self.elapsed = 0                 # samples consumed (int, absolute time base)
        self.finished = False
        self.metrics: dict[str, float] = {}
        # analysis (fixed hop -> one band layout)
        self._hop_buf = np.zeros(self.hop)
        self._hop_pow = 0.0              # sum over channels and samples of x^2 (BS.1770 channel sum)
        self._hop_fill = 0
        self._layout = self._band_layout(self.hop)
        self._bands: Optional[np.ndarray] = None
        self._loud_db: Optional[float] = None
        self._frame_carry = np.zeros(0)
        self._env_hist = np.full(self.HIST_FRAMES, np.nan)
        self._hist_fill = 0
        self._fluct_db = 0.0
        self._bed_db: Optional[float] = None
        self._cap_db = float(s["event_to_bed_db"]) if s["event_to_bed_db"] is not None else 0.0
        self._ev_peak = -np.inf
        # control state
        self._gain_db = 0.0
        self._g0 = 0.0
        self._g1 = 0.0
        self._tilt_db = 0.0
        self._shelf_db = 0.0
        self._shelf_sos = _high_shelf_sos(self.sr, 1000.0, 0.0)
        self._channels: Optional[int] = None
        self._shelf_zi: Optional[np.ndarray] = None
        self._fast_zi: Optional[np.ndarray] = None
        self._min_carry: Optional[np.ndarray] = None
        self._ma_carry: Optional[np.ndarray] = None
        self._delay: Optional[np.ndarray] = None

    @classmethod
    def from_scene(cls, scene: Scene, sr: int = SAMPLE_RATE) -> "CalmGovernor":
        """Governor configured from a scene's mode, timer and Tingle setting."""
        vibe = scene.vibe or {}
        timer = vibe.get("timer_min")
        descent = vibe.get("descent_min")
        return cls(sr, scene.mode, timer_min=float(timer) if timer is not None else None,
                   descent_min=float(descent) if descent is not None else None, tingle=vibe.get("tingle"))

    # ------------------------------------------------------------- analysis
    def _band_layout(self, n: int) -> tuple:
        f = np.fft.rfftfreq(n, 1.0 / self.sr)
        zb = np.minimum(23, np.floor(_bark(f)).astype(int))
        valid = (f >= 20.0) & (f <= min(16000.0, 0.45 * self.sr))
        octs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        octs = octs[octs * math.sqrt(2.0) < 0.5 * self.sr]
        oct_idx = np.full(f.shape, -1)
        for k, fc in enumerate(octs):
            oct_idx[(f >= fc / math.sqrt(2.0)) & (f < fc * math.sqrt(2.0))] = k
        win = np.hanning(n)
        return zb, valid, oct_idx, octs, win, float(np.mean(win ** 2))

    def _analyse(self, mono: np.ndarray, ms: float) -> None:
        """Spectral metrics from the hop's channel average; loudness from `ms`, the per-sample
        sum of channel mean squares (BS.1770-4 eq. 2 with unit channel weights)."""
        zb, valid, oct_idx, octs, win, wnorm = self._layout
        n = mono.size
        spec = np.abs(np.fft.rfft(mono * win)) ** 2 / (n * wnorm + 1e-30)
        bark_e = np.bincount(zb[valid], weights=spec[valid], minlength=24)[:24]
        oct_e = np.bincount(oct_idx[oct_idx >= 0], weights=spec[oct_idx >= 0], minlength=len(octs))[:len(octs)]
        cur = np.concatenate([bark_e, oct_e])
        alpha = 1.0 - math.exp(-self.HOP_S / 1.0)          # 1 s spectral smoothing
        if self._bands is None:
            self._bands = cur.copy()
        else:
            self._bands += alpha * (cur - self._bands)
        be = np.maximum(self._bands[:24], 1e-20)
        oe = np.maximum(self._bands[24:], 1e-20)
        # sharpness proxy: S = 0.11 * sum N'(z) g(z) z dz / N  (DIN 45692 form)
        nprime = be ** 0.23
        top = float(np.max(nprime))
        nprime = np.where(nprime > top * 1e-3, nprime, 0.0)
        z = np.arange(24) + 0.5
        total = float(np.sum(nprime))
        sharp = 0.11 * float(np.sum(nprime * _din45692_g(z) * z)) / total if total > 0 else 0.0
        # PSD tilt: octave-band power density vs log2 f (pink = -3 dB/oct)
        if oe.size >= 3:
            lvl = 10.0 * np.log10(oe) - 10.0 * np.log10(octs * (math.sqrt(2.0) - 1.0 / math.sqrt(2.0)))
            tilt = float(np.polyfit(np.log2(octs), lvl, 1)[0])
        else:
            tilt = -3.0
        ldb = 10.0 * math.log10(ms + 1e-20) - 0.691
        a3 = 1.0 - math.exp(-self.HOP_S / 3.0)              # 3 s loudness smoothing
        self._loud_db = ldb if self._loud_db is None else self._loud_db + a3 * (ldb - self._loud_db)
        self.metrics.update({"sharpness_acum": sharp, "tilt_db_oct": tilt, "loudness_db": self._loud_db})

    def _modulation(self, mono: np.ndarray) -> float:
        """Update the 10 ms envelope history; return the bed level (30th percentile, dB)."""
        buf = np.concatenate([self._frame_carry, mono])
        nf = buf.size // self.frame
        if nf:
            frames = buf[: nf * self.frame].reshape(nf, self.frame)
            db = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)[-self.HIST_FRAMES:]
            k = db.size
            self._env_hist = np.roll(self._env_hist, -k)
            self._env_hist[-k:] = db
            self._hist_fill = min(self.HIST_FRAMES, self._hist_fill + k)
        self._frame_carry = buf[nf * self.frame:]
        if self._hist_fill == 0:
            return -120.0
        h = self._env_hist[-self._hist_fill:]
        bed = float(np.percentile(h, 30))
        if self._hist_fill >= self.HIST_FRAMES // 2:
            nh = h.size
            idx = np.arange(nh)
            w = np.hanning(nh)
            X = np.fft.rfft((h - np.polyval(np.polyfit(idx, h, 1), idx)) * w)
            f = np.fft.rfftfreq(nh, self.FRAME_S)
            band = (f >= 1.0) & (f <= 20.0)
            pw = np.abs(X) ** 2
            var = 2.0 * float(np.sum(pw[band])) / (nh * nh * float(np.mean(w ** 2)))
            self._fluct_db = math.sqrt(max(0.0, var))       # rms envelope swing in the fluctuation band
            bandp = pw[band]
            peak_ratio = float(np.max(bandp) / (np.median(bandp) + 1e-20)) if bandp.size else 0.0
            self.metrics.update({"fluctuation_db": self._fluct_db, "modulation_peak_ratio": peak_ratio})
        return bed

    # ------------------------------------------------------------- control
    def _descent_u(self, t: np.ndarray) -> np.ndarray:
        """0 before the Descent, smoothstep to 1 at the end of the timer."""
        if self.timer_s <= 0.0:
            return np.zeros_like(t)
        start = self.timer_s - self.descent_s
        if self.descent_s > 0.0:
            u = np.clip((t - start) / self.descent_s, 0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)
        return (t >= start).astype(float)

    def _end_hop(self) -> None:
        """Analyse the completed hop and set the controls for the next one."""
        s = self.settings
        mono = self._hop_buf
        self._analyse(mono, self._hop_pow / self.hop)
        self._hop_pow = 0.0
        self._bed_db = self._modulation(mono)
        t_end = (self.elapsed) / self.sr
        u = float(self._descent_u(np.array([t_end]))[0])
        strength = float(s.get("correction_strength", 1.0))

        target_tilt = 0.0
        tilt = self.metrics.get("tilt_db_oct", -3.0)
        if s["tilt_db_oct"] is not None:
            lo, hi = s["tilt_db_oct"]
            if tilt > hi:
                target_tilt += 6.0 * (hi - tilt)            # too bright: cut highs
            elif tilt < lo:
                target_tilt += 3.0 * (lo - tilt)            # too dull: allow a little back
        sharp = self.metrics.get("sharpness_acum", 0.0)
        if s["max_sharpness_acum"] is not None and sharp > s["max_sharpness_acum"]:
            target_tilt -= 8.0 * (sharp - s["max_sharpness_acum"])
        target_tilt = float(min(4.0, max(-15.0, strength * target_tilt))) + s["descent_tilt_db"] * u
        step = s["slew_db_per_s"] * self.HOP_S
        self._tilt_db += float(np.clip(target_tilt - self._tilt_db, -step, step))
        if abs(self._tilt_db - self._shelf_db) >= 0.05:
            self._shelf_sos = _high_shelf_sos(self.sr, 1000.0, self._tilt_db)
            self._shelf_db = self._tilt_db

        # broadband gain: never boost; bring an over-loud stream down to the ceiling, recover slowly
        target_gain = 0.0
        if s["ceiling_lufs"] is not None and self._loud_db is not None:
            over = (self._loud_db + self._gain_db) - s["ceiling_lufs"]
            target_gain = min(0.0, self._gain_db - over) if over > 0.5 else min(0.0, self._gain_db + 0.25 * self.HOP_S)
        self._gain_db += float(np.clip(target_gain - self._gain_db, -step, step))
        self._g0, self._g1 = self._g1, self._gain_db

        # event-to-bed cap: tighter while the texture fluctuates, -> 0 through the Descent
        if self.limiter_on:
            cap = float(s["event_to_bed_db"]) * (1.0 - u)
            fl = s["fluct_limit_db"]
            if fl is not None and self._fluct_db > fl:
                cap -= min(3.0, self._fluct_db - fl)
            self._cap_db = max(0.0, cap)
        if np.isfinite(self._ev_peak) and self._bed_db is not None:
            self.metrics["event_to_bed_db"] = float(self._ev_peak - self._bed_db)
        self._ev_peak = -np.inf
        self.metrics.update({"gain_db": self._gain_db, "tilt_correction_db": self._tilt_db,
                             "bed_db": self._bed_db if self._bed_db is not None else -120.0,
                             "event_cap_db": self._cap_db, "elapsed_s": self.elapsed / self.sr,
                             "descent": u})

    def _ensure_state(self, C: int) -> None:
        if self._channels == C:
            return
        self._channels = C
        self._shelf_zi = np.zeros((C, 2))
        self._fast_zi = None
        D = self.latency
        self._min_carry = np.zeros(D + self._hold)
        self._ma_carry = np.zeros(D)
        self._delay = np.zeros((C, D))

    def _limiter_gain(self, y: np.ndarray) -> np.ndarray:
        """Look-ahead event-to-bed gain (dB, <= 0) for the samples of y; delays y in place via the delay line."""
        m = y.shape[1]
        D = self.latency
        ymono = y.mean(axis=0)
        a_fast = math.exp(-1.0 / (0.005 * self.sr))
        if self._fast_zi is None:
            self._fast_zi = np.zeros(1)              # settles within the first hop, before the bed is known
        env, self._fast_zi = signal.lfilter([1.0 - a_fast], [1.0, -a_fast], ymono ** 2, zi=self._fast_zi)
        np.copyto(self._fast_zi, 0.0, where=np.abs(self._fast_zi) < 1e-30)
        env_db = 10.0 * np.log10(env + 1e-20)
        self._ev_peak = max(self._ev_peak, float(np.max(env_db)))
        if self._bed_db is None:
            g_db = np.zeros(m)
        else:
            limit = self._bed_db + 0.5 * self._tilt_db + self._cap_db
            excess = env_db - limit
            knee = 2.0
            g_db = np.where(excess <= -knee / 2, 0.0,
                            np.where(excess >= knee / 2, -excess, -((excess + knee / 2) ** 2) / (2.0 * knee)))
        # minimum over [look-ahead + hold], then a (D+1)-tap moving average: the full reduction is
        # reached exactly when the event leaves the delay line, and released after the hold.
        # O(n) running minimum (van Herk / Gil-Werman); the average is a cumulative-sum difference
        # over at most one hop plus D samples, so its rounding stays near 1e-12 dB at any block size.
        W = D + self._hold + 1
        if W > 1:
            ext = np.concatenate([self._min_carry, g_db])
            gmin = minimum_filter1d(ext, W, mode="nearest")[W // 2: W // 2 + m]
            self._min_carry = ext[-(W - 1):]
        else:
            gmin = g_db
        if D > 0:
            ext2 = np.concatenate([self._ma_carry, gmin])
            cs = np.cumsum(ext2)
            g_s = (cs[D:] - np.concatenate(([0.0], cs[:-(D + 1)]))) / (D + 1)
            self._ma_carry = ext2[-D:]
        else:
            g_s = gmin
        return g_s

    def _render(self, seg: np.ndarray) -> np.ndarray:
        """Process a segment that lies inside one hop."""
        s = self.settings
        m = seg.shape[1]
        if self._shelf_db == 0.0 and not self._shelf_zi.any():
            y = seg                                  # flat shelf with a clear state: the filter is the identity
        else:
            sec = self._shelf_sos[0]
            y, self._shelf_zi = signal.lfilter(sec[:3], sec[3:], seg, axis=-1, zi=self._shelf_zi)
            np.copyto(self._shelf_zi, 0.0, where=np.abs(self._shelf_zi) < 1e-30)
        total = self._g0 + (self._g1 - self._g0) * ((self._hop_fill + np.arange(m)) / self.hop)
        if self.limiter_on:
            total = total + self._limiter_gain(y)
            D = self.latency
            if D > 0:
                yd = np.concatenate([self._delay, y], axis=1)
                self._delay = yd[:, -D:].copy()
                y = yd[:, :m]
        if self.timer_s > 0.0:
            t = (self.elapsed + np.arange(m)) / self.sr
            total = total + s["descent_db"] * self._descent_u(t)
            over = t - self.timer_s
            total = total + np.where(over > 0.0, -120.0 * np.minimum(1.0, over / self.FINAL_FADE_S), 0.0)
            gain = 10.0 ** (total / 20.0)
            gain[over >= self.FINAL_FADE_S] = 0.0            # the -120 dB fade ends in true digital silence
            if over[-1] >= self.FINAL_FADE_S:
                self.finished = True
            return y * gain[None, :]
        return y * (10.0 ** (total / 20.0))[None, :]

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        C, n = x.shape
        if n == 0 or not self.active:
            self.elapsed += n
            return x[0] if squeeze else x
        if self.finished:
            self.elapsed += n
            y = np.zeros_like(x)
            return y[0] if squeeze else y
        self._ensure_state(C)
        out = np.empty_like(x)
        pos = 0
        while pos < n:
            take = min(n - pos, self.hop - self._hop_fill)
            seg = x[:, pos:pos + take]
            out[:, pos:pos + take] = self._render(seg)
            self._hop_buf[self._hop_fill:self._hop_fill + take] = seg.mean(axis=0)
            self._hop_pow += float(np.einsum("ij,ij->", seg, seg))
            self._hop_fill += take
            self.elapsed += take
            pos += take
            if self._hop_fill == self.hop:
                self._end_hop()
                self._hop_fill = 0
        return out[0] if squeeze else out

    def suggestions(self) -> dict[str, float]:
        """Macro-level nudges for the control pre-pass (weather macros only, never the user's settings)."""
        s = self.settings
        out: dict[str, float] = {}
        fl = s.get("fluct_limit_db")
        if fl and self._fluct_db > fl:
            out["density_scale"] = float(max(0.6, 1.0 - 0.1 * (self._fluct_db - fl)))
            out["gust_depth_scale"] = 0.8
        if s.get("max_sharpness_acum") and self.metrics.get("sharpness_acum", 0.0) > s["max_sharpness_acum"] + 0.3:
            out["drop_size_shift"] = -0.2          # toward finer drops / farther surfaces
        return out

    def badge(self) -> Optional[str]:
        """Short visible note when the governor is acting (MASTER_PLAN 2.7), else None."""
        if not self.active:
            return None
        if self.finished:
            return "Timer finished"
        if self.metrics.get("descent", 0.0) > 0.0:
            return "Fading out as you fall asleep"
        if self.metrics.get("gain_db", 0.0) < -1.0:
            return f"Volume eased for {MODE_LABELS[self.mode]}"
        if self.metrics.get("tilt_correction_db", 0.0) < -1.0:
            return f"Tone softened for {MODE_LABELS[self.mode]}"
        return None

    def reset(self) -> None:
        settings, timer_min, descent_min, tingle = self._init_args
        self.__init__(self.sr, self.mode, settings, timer_min, descent_min, tingle)
