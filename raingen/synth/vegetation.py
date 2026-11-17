"""Wind in vegetation: species spectra, Beaufort ladder and advected gusts.

Physics:
* Fégeant (1999) and Bolin (2006, 2009): vegetation noise level scales as
  L ~ 10*log10(U^(2*chi)) with chi = 1.5..2.7 depending on species; Bolin's conifer
  stands measured 23-35 log10(U) (chi ~ 1.2-1.75). Calibration (level_db = SPL 10 m from
  one crown at a crown-height mean wind of 5 m/s) is anchored on Bolin's KTH data: a
  single 18 m spruce heard from 3 m at 6.3 m/s gave 47.2 dB(A), a 120 m x 30 m spruce edge
  heard from 29 m at 5.1 m/s gave 55.2 dB(A) (both ~44 dB at 10 m from one crown at 5 m/s).
  Levels follow the *mean* wind: the gust-weighted average E[g^(2 chi)] of the log-normal
  gust multiplier is divided out, as the empirical laws are fitted to Leq vs mean wind.
* Crowns sit higher than the ears, where turbulence intensity I_u = 1/ln(z/z0) is smaller:
  the listener's gust multiplier is remapped to the crown's intensity with the same timing
  (raingen.synth.wind.remap_gust). With it the level fluctuates by sigma_L ~ 8.7 chi s
  ~ 3-5 dB, matching Bolin's measured 2.6-5.2 dB(A).
* Conifers: a log-Gaussian "whoosh" centred on the needle Strouhal frequency
  f = St*U/d_n (St ~ 0.2, needle diameter d_n ~ 1-1.5 mm): Gaussian, hiss-like.
* Broadleaf trees: rustle = dense granular leaf-to-leaf impacts (sparse, crisp grains whose
  band rises with the impact speed) plus foliage swish modulated by crown sway.
* Bare (winter) branches: aeolian whistles f = 0.2*U/d for twigs d = 3..10 mm.
* Beaufort ladder (WMO land descriptions, defined on the 10 m wind): leaves rustle
  from force 2, twigs/small branches move from force 3-4, branches whistle at
  force 5-6, trunks creak and branches break from force 8. The ladder switches on
  the 10 m-equivalent wind including the gust at the plant; levels use the log-law
  wind at crown height.
* Gusts are advected (Taylor): a stand of trees s metres upwind hears each gust
  s/U seconds before the listener (GustProcess.at_position), so gusts are heard
  sweeping across the scene.

Synthesis is vectorised over every plant of the scene: one frequency-domain shaped-noise
bank (swish, needles, whistles and ISO 9613-1 air absorption folded into one magnitude
spectrum per plant), one batched FFT convolution for the leaf grains, and one static SH
encode of all plants (the encoder's spread model), so the whole vegetation is a single
ambisonic emission on stem 'wind'.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import fft as sfft
from scipy import signal

from raingen.core.constants import SAMPLE_RATE
from raingen.core.types import BlockContext, Emission, SourcePatch
from raingen.physics.atmosphere import iso9613_alpha_db_per_m
from raingen.physics.wind import turbulence_intensity_for, wind_at_height

from .wind import (GustField, PendingMixer, ShapedNoiseOLA, StaticSHEncoder, bandpass_coefs, bandpass_mag2,
                   bandpass_sos, biquad_response, block_start_sample, db_spl, ramp_frac, remap_gust, scene_z0,
                   shelter_of, smoothstep)

# level_db: SPL (dB re 20 uPa) 10 m from the crown centre of one plant at a crown-height mean wind
# of 5 m/s (Bolin 2009 KTH data for conifers; the others relative to it, uncertain to +-5 dB).
SPECIES = {
    "deciduous": {"chi": 2.0, "level_db": 45.0, "tick": 1.0, "tick_rate": 2500.0, "swish": 0.55,
                  "needle": 0.0, "whistle": 0.12, "height": 10.0, "tree": True},
    "conifer": {"chi": 1.5, "level_db": 44.0, "tick": 0.04, "tick_rate": 600.0, "swish": 0.3,
                "needle": 1.0, "needle_d_mm": 1.2, "whistle": 0.18, "height": 14.0, "tree": True},
    "bush": {"chi": 1.8, "level_db": 37.0, "tick": 0.8, "tick_rate": 1800.0, "swish": 0.6,
             "needle": 0.0, "whistle": 0.05, "height": 1.5, "tree": False},
    "grass": {"chi": 1.6, "level_db": 32.0, "tick": 0.3, "tick_rate": 3000.0, "swish": 1.0,
              "needle": 0.0, "whistle": 0.0, "height": 0.4, "tree": False},
    "palm": {"chi": 2.0, "level_db": 44.0, "tick": 0.6, "tick_rate": 900.0, "swish": 0.7,
             "needle": 0.0, "whistle": 0.05, "height": 9.0, "tree": True},
    "reeds": {"chi": 1.7, "level_db": 34.0, "tick": 0.5, "tick_rate": 2000.0, "swish": 0.9,
              "needle": 0.4, "needle_d_mm": 5.0, "whistle": 0.0, "height": 1.8, "tree": False},
}
LEVEL_REF_M = 10.0          # distance of the level_db anchor
LEVEL_REF_MPS = 5.0         # crown-height mean wind of the level_db anchor
TICK_M2 = math.exp(2.0 * 0.7 ** 2)      # E[a^2] of the lognormal(0, 0.7) grain amplitudes
TICK_TAIL = 512             # samples of grain ringing carried between blocks (>= 140 dB down)

_ALIASES = {
    "tree": "deciduous", "broadleaf": "deciduous", "oak": "deciduous", "beech": "deciduous",
    "birch": "deciduous", "maple": "deciduous", "poplar": "deciduous", "willow": "deciduous",
    "pine": "conifer", "spruce": "conifer", "fir": "conifer", "needles": "conifer", "cedar": "conifer",
    "hedge": "bush", "shrub": "bush", "leaves": "deciduous", "meadow": "grass", "bamboo": "reeds",
}


def species_of(patch: SourcePatch) -> str:
    p = patch.params or {}
    for key in ("species", "type", "kind"):
        sp = str(p.get(key, "")).lower()
        if sp in SPECIES:
            return sp
        if sp in _ALIASES:
            return _ALIASES[sp]
    mat = (patch.material or "").lower()
    if mat in SPECIES:
        return mat
    return _ALIASES.get(mat, "deciduous")


def leaf_factor(species: str, season: str) -> float:
    """Foliage present (0 bare .. 1 full) by season."""
    season = (season or "summer").lower()
    if species == "deciduous":
        return {"spring": 0.8, "summer": 1.0, "autumn": 0.7, "winter": 0.05}.get(season, 1.0)
    if species in ("grass", "reeds"):
        return {"winter": 0.5, "autumn": 0.85}.get(season, 1.0)
    if species == "bush":
        return {"winter": 0.3, "autumn": 0.75}.get(season, 1.0)
    return 1.0


def _creak(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Trunk/branch creak: stick-slip pulse train with gliding rate through wood modes."""
    dur = rng.uniform(0.3, 1.4)
    L = int(dur * sr)
    tail = int(0.25 * sr)
    t = np.arange(L) / sr
    f0, f1 = rng.uniform(22.0, 60.0), rng.uniform(40.0, 120.0)
    if rng.random() < 0.5:
        f0, f1 = f1, f0
    rate = f0 + (f1 - f0) * (t / dur) ** rng.uniform(0.6, 1.6)
    phase = np.cumsum(rate) / sr
    idx = np.nonzero(np.diff(np.floor(phase)) > 0)[0] + 1
    imp = np.zeros(L + tail)
    env = np.sin(np.pi * idx / L) ** 0.7
    imp[idx] = env * (1.0 + 0.3 * rng.standard_normal(idx.size))
    base = rng.uniform(160.0, 480.0)
    y = np.zeros_like(imp)
    for ratio, gain in ((1.0, 1.0), (1.63, 0.6), (2.41, 0.35), (3.7, 0.2)):
        f = base * ratio * rng.uniform(0.97, 1.03)
        t60 = rng.uniform(0.04, 0.12)
        q = math.pi * f * t60 / 6.91
        y += gain * signal.sosfilt(bandpass_sos(f, q, sr), imp)
    peak = float(np.max(np.abs(y)))
    return y / peak if peak > 0 else y


def _smooth(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorised raingen.synth.wind.smoothstep."""
    t = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rfft_weights(N: int) -> np.ndarray:
    """Parseval weights on an N-point rfft grid: variance = sum_k w_k |X_k|^2 / N for unit-variance bins."""
    F = N // 2 + 1
    w = np.full(F, 2.0 / N)
    w[0] = 1.0 / N
    if N % 2 == 0:
        w[-1] = 1.0 / N
    return w


class _VegVoice:
    """Configuration of one vegetation patch (a tree, a stand, a hedge or a lawn)."""

    def __init__(self, patch: SourcePatch, species: str, scene, rngf, sr: int, level_scale: float = 1.0):
        self.patch = patch
        self.species = species
        self.spec = SPECIES[species]
        self.sr = sr
        p = patch.params or {}
        cfg = rngf.get(f"veg.cfg.{patch.id}", 0)
        self.count = max(1, int(p.get("count", 1)))
        self.height = float(p.get("height_m", self.spec["height"]))
        self.exposure = float(p.get("wind_exposure", 1.0))
        self.level_scale = float(level_scale) * float(p.get("level", 1.0))
        self.season = scene.weather.season
        self.leaf = leaf_factor(species, self.season)
        if "leaf_fraction" in p and species not in ("grass", "reeds"):
            self.leaf = min(max(float(p["leaf_fraction"]), 0.02), 1.0)
        autumn_crisp = 1.25 if (species == "deciduous" and self.season == "autumn") else 1.0
        self.f_crisp0 = cfg.uniform(3200.0, 5200.0) * autumn_crisp
        self.f_body = cfg.uniform(1100.0, 1900.0)
        self.sway_f = cfg.uniform(0.18, 0.5) if self.spec["tree"] else cfg.uniform(0.6, 1.6)
        self.sway_phase = cfg.uniform(0.0, 2.0 * np.pi)
        self.d_needle = float(p.get("needle_d_mm", self.spec.get("needle_d_mm", 1.2))) / 1000.0
        twig0 = float(p.get("twig_diameter_m", 0.004))
        self.twig_d = np.array([0.75, 1.25, 2.0]) * twig0 * cfg.uniform(0.85, 1.2, size=3)
        area = max(float(patch.area_m2), 1.0)
        self.radius = float(p.get("crown_radius_m", p.get("canopy_radius_m", math.sqrt(area / math.pi))))
        self.fixed_spread = p.get("spread")
        self.dist = float(patch.distance)
        self.spread = float(self.fixed_spread) if self.fixed_spread is not None else \
            min(max(self.radius / self.dist, 0.05), 0.95)
        self.creaks = PendingMixer(1)


class VegetationGenerator:
    """All vegetation patches of the layout (plus a ground grass bed from the surface mix).

    Every plant is a voice of one vectorised bank (see the module docstring); the output is
    a single SH emission on stem 'wind'. Works for any block length.
    """

    name = "vegetation"

    def __init__(self, scene, layout, rngf, gust_field: Optional[GustField] = None, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.field = gust_field or GustField(scene, rngf, sr)
        self.z0 = scene_z0(scene)
        self.listener_h = float(scene.listener.height_m)
        self.voices: list[_VegVoice] = []
        has_grass = False
        for patch in layout or []:
            if patch.kind != "vegetation":
                continue
            sp = species_of(patch)
            has_grass = has_grass or sp == "grass"
            self.voices.append(_VegVoice(patch, sp, scene, rngf, sr))
        surfaces = dict(scene.place.surfaces or {})
        total = float(sum(float(v) for v in surfaces.values())) or 1.0
        share = float(surfaces.get("grass", 0.0)) / total
        if not has_grass and share >= 0.1 and shelter_of(scene) not in ("room_closed", "car", "attic"):
            bed = SourcePatch(id="grass_bed", kind="vegetation", material="grass",
                              position=(0.3, 0.0, -float(scene.listener.height_m) + 0.2), area_m2=60.0,
                              params={"species": "grass", "spread": 1.0, "height_m": 0.4})
            self.voices.append(_VegVoice(bed, "grass", scene, rngf, sr, level_scale=math.sqrt(share)))
        self._build()

    # -- static tables -----------------------------------------------------
    def _build(self) -> None:
        vs = self.voices
        V = len(vs)
        self.V = V
        if not V:
            return

        def arr(f):
            return np.array([f(v) for v in vs], dtype=float)

        self.chi = arr(lambda v: v.spec["chi"])
        dist = arr(lambda v: v.dist)
        radius = arr(lambda v: v.radius)
        self.dist = dist
        # amplitude at the listener per unit (crown-wind / 5 m/s)^chi: anchor at 10 m, 1/r beyond the crown
        r_eff = np.hypot(dist, 0.5 * radius)
        self.level = arr(lambda v: db_spl(v.spec["level_db"]) * math.sqrt(v.count) * v.level_scale) \
            * LEVEL_REF_M / r_eff
        self.tick_w = arr(lambda v: v.spec["tick"] * v.leaf)
        self.tick_rate = arr(lambda v: v.spec["tick_rate"] * max(v.leaf, 0.02) * math.sqrt(v.count))
        self.swish_w = arr(lambda v: v.spec["swish"] * (0.25 + 0.75 * v.leaf))
        self.needle_w = arr(lambda v: v.spec["needle"])
        self.whistle_w = arr(lambda v: v.spec["whistle"] * (1.0 + 3.0 * (1.0 - v.leaf)))
        self.grassy = np.array([v.species in ("grass", "reeds") for v in vs])
        self.tree = np.array([bool(v.spec["tree"]) for v in vs])
        self.exposure = arr(lambda v: v.exposure)
        self.level_scale = arr(lambda v: v.level_scale)
        self.count = arr(lambda v: v.count)
        self.f_crisp0 = arr(lambda v: v.f_crisp0)
        self.f_body = arr(lambda v: v.f_body)
        self.sway_f = arr(lambda v: v.sway_f)
        self.sway_phase = arr(lambda v: v.sway_phase)
        self.d_needle = arr(lambda v: v.d_needle)
        self.twig_d = np.stack([v.twig_d for v in vs])                    # (V, 3)
        self.pos = np.array([v.patch.position for v in vs], dtype=float)
        self.material = [v.patch.material for v in vs]
        # crown-height wind per unit 10 m wind (log law is linear in u10) and turbulence ratio to the ears
        z_crown = np.maximum(0.7 * arr(lambda v: v.height), 0.5)
        self.u_ratio = np.array([wind_at_height(1.0, float(z), self.z0) for z in z_crown])
        ti_ear = float(turbulence_intensity_for(self.z0, max(self.listener_h, 0.3)))
        self.ti_ratio = np.array([float(turbulence_intensity_for(self.z0, float(z))) for z in z_crown]) \
            / max(ti_ear, 1e-6)
        self.encoder = StaticSHEncoder([v.patch.azimuth for v in vs], [v.patch.elevation for v in vs],
                                       [v.spread for v in vs], sr=self.sr)
        self.noise = ShapedNoiseOLA(V, self.sr)
        self.whistle_jit = np.zeros((V, 3))
        self._n = 0
        self._air_key = None
        self._air_noise: Optional[np.ndarray] = None
        self._air_tick: Optional[np.ndarray] = None
        self._fc_crisp = np.zeros(V)
        self._kernel: Optional[np.ndarray] = None
        self._body: Optional[np.ndarray] = None
        self._carry: Optional[np.ndarray] = None
        self._g_prev = np.zeros(V)
        mats = sorted(set(self.material))
        self._mat_names = mats
        self._mat_idx = np.array([mats.index(m) for m in self.material], dtype=np.int64)

    def _grids(self, n: int) -> None:
        if n == self._n:
            return
        self._n = n
        self.noise.setup(n)
        self.f_noise = self.noise.freqs
        self.N_tick = int(sfft.next_fast_len(n + TICK_TAIL, real=True))
        self.f_tick = np.fft.rfftfreq(self.N_tick, 1.0 / self.sr)
        self.w_tick = _rfft_weights(self.N_tick)
        self.zinv_tick = np.exp(-2j * np.pi * self.f_tick / self.sr)
        self._air_key = None
        self._fc_crisp[:] = 0.0
        self._kernel = None
        self._body = None
        self._carry = None

    def _air(self, temp_c: float, rh: float) -> None:
        """ISO 9613-1 absorption magnitudes per plant on both grids (re-evaluated on 1 C / 5 % RH changes)."""
        key = (round(temp_c), round(rh / 5.0))
        if key == self._air_key:
            return
        self._air_key = key
        d = np.where(self.dist > 3.0, self.dist, 0.0)[:, None]
        self._air_noise = 10.0 ** (-iso9613_alpha_db_per_m(self.f_noise, temp_c, rh)[None, :] * d / 20.0)
        self._air_tick = 10.0 ** (-iso9613_alpha_db_per_m(self.f_tick, temp_c, rh)[None, :] * d / 20.0)
        self._fc_crisp[:] = 0.0                         # tick kernels carry the absorption: rebuild

    def _tick_kernels(self, f_crisp: np.ndarray) -> np.ndarray:
        """Unit-white-gain grain kernels 0.7 crisp + 0.5 body band-pass (x air absorption), rebuilt on 5 % changes.

        The body band of each plant is fixed: its response is computed once per grid.
        """
        stale = np.abs(f_crisp - self._fc_crisp) > 0.05 * np.maximum(self._fc_crisp, 1.0)
        if self._kernel is None:
            self._kernel = np.zeros((self.V, self.f_tick.size), dtype=complex)
            bb, ab = bandpass_coefs(self.f_body, 1.8, self.sr)
            self._body = 0.5 * biquad_response(bb, ab, self.zinv_tick)
            stale[:] = True
        if np.any(stale):
            idx = np.nonzero(stale)[0]
            bc, ac = bandpass_coefs(f_crisp[idx], 1.4, self.sr)
            k = 0.7 * biquad_response(bc, ac, self.zinv_tick) + self._body[idx]
            gain = np.sqrt((k.real ** 2 + k.imag ** 2) @ self.w_tick)
            self._kernel[idx] = k * (self._air_tick[idx] / gain[:, None])
            self._fc_crisp[idx] = f_crisp[idx]
        return self._kernel

    def render(self, ctx: BlockContext) -> list[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        self.field.advance(ctx.index, w, n)
        if not self.V:
            return []
        self._grids(n)
        u10 = max(float(w.wind_u10_mps), 0.0)
        u_h = u10 * self.u_ratio
        s_up = self.pos[:, 0] * math.cos(w.wind_azimuth) + self.pos[:, 1] * math.sin(w.wind_azimuth)
        ti_ear = max(float(w.turbulence_intensity), 0.0)
        g_ear = self.field.at_many(s_up, np.maximum(u_h, 0.5))
        ti_c = np.clip(ti_ear * self.ti_ratio, 0.0, 0.8)
        g = remap_gust(g_ear, ti_ear, ti_c)
        u = u_h * g * self.exposure                       # instantaneous crown wind
        u_bft = u10 * g * self.exposure                   # 10 m-equivalent wind at this plant
        active = (u >= 0.5) | (u_bft >= 1.2)
        creaking = any(v.creaks.active for v in self.voices)
        if not np.any(active) and not np.any(self._g_prev > 1e-8) and not self.noise.active \
                and self._carry is None and not creaking:
            return []
        rng = ctx.rng.get("veg", ctx.index)

        on_leaf = _smooth(u_bft, 1.3, 3.0)          # Beaufort 2 (1.6-3.3 m/s): leaves rustle
        on_twig = _smooth(u_bft, 3.2, 6.5)          # Beaufort 3-4: twigs in constant motion
        on_whistle = _smooth(u_bft, 7.0, 11.0)      # Beaufort 5-6: branches move, whistle
        on_twig = np.where(self.grassy, on_leaf, on_twig)
        # Leq follows the mean-wind law: divide out E[g^(2 chi)] = exp(chi (2 chi - 1) s^2)
        s2 = np.log1p(ti_c * ti_c)
        u_mean = u_h * self.exposure
        A = self.level * (np.maximum(u_mean, 0.0) / LEVEL_REF_MPS) ** self.chi * g ** self.chi \
            * np.exp(-0.5 * self.chi * (2.0 * self.chi - 1.0) * s2)
        A *= _smooth(u, 0.8, 2.0) * active
        wet_all = w.wetness or {}
        wet_leaves = float(wet_all.get("leaves", 0.0))
        wet = np.array([float(wet_all.get(m, wet_leaves)) for m in self._mat_names])[self._mat_idx]

        # crown sway (slow, linear over the block)
        t0, t1 = float(ctx.t0), float(ctx.t0) + n / sr
        depth = 0.25 + 0.45 * on_twig
        sw0 = 1.0 + depth * np.sin(2.0 * np.pi * self.sway_f * t0 + self.sway_phase)
        sw1 = 1.0 + depth * np.sin(2.0 * np.pi * self.sway_f * t1 + self.sway_phase)
        frac = ramp_frac(n)

        # ---- noise layers: swish, needle whoosh and bare-twig whistles in one spectrum per plant ----
        self._air(float(w.temperature_c), float(w.humidity_pct))
        f = self.f_noise
        swish_g = self.swish_w * on_twig * (1.0 + 0.3 * wet) * A
        needle_g = self.needle_w * on_leaf * A
        a = math.exp(-(n / sr) / 0.7)
        self.whistle_jit = a * self.whistle_jit + math.sqrt(1.0 - a * a) * rng.standard_normal((self.V, 3))
        wh_on = self.whistle_w * on_whistle
        wh_g = (wh_on / math.sqrt(3.0))[:, None] * (0.5 + 0.5 * np.abs(self.whistle_jit)) * A[:, None]
        mag2 = None
        if np.any(swish_g > 0.0):
            b = bandpass_mag2(f[None, :], (250.0 + 90.0 * u)[:, None], 0.7)
            mag2 = (swish_g ** 2 / self.noise.power(b))[:, None] * b
        if np.any(needle_g > 0.0):
            f_n = np.clip(0.2 * u / self.d_needle, 150.0, 12000.0)
            b = bandpass_mag2(f[None, :], f_n[:, None], 1.0) ** 2           # two cascaded resonances
            term = (needle_g ** 2 / self.noise.power(b))[:, None] * b
            mag2 = term if mag2 is None else mag2 + term
        if np.any(wh_g > 0.0):
            for j in range(3):
                f_j = np.clip(0.2 * u / self.twig_d[:, j] * (1.0 + 0.03 * self.whistle_jit[:, j]), 60.0, 9000.0)
                b = bandpass_mag2(f[None, :], f_j[:, None], 15.0)
                term = (wh_g[:, j] ** 2 / self.noise.power(b))[:, None] * b
                mag2 = term if mag2 is None else mag2 + term
        mag = None if mag2 is None else np.sqrt(mag2) * self._air_noise
        y = self.noise.process(rng, mag, n)
        if mag is not None or self.noise.active:
            m0 = 1.0 + 0.75 * (sw0 - 1.0)
            m1 = 1.0 + 0.75 * (sw1 - 1.0)
            y = y * (m0[:, None] + (m1 - m0)[:, None] * frac[None, :])

        # ---- leaf grains: sparse signed impulses x sway, batched FFT convolution with carried tails ----
        rate = np.minimum(15000.0, self.tick_rate * (np.maximum(u, 0.1) / 5.0) ** 1.3) * on_leaf * active
        tick_g = self.tick_w * on_leaf * (1.0 - 0.4 * wet) * A
        lam = rate * n / sr
        counts = rng.poisson(lam) if np.any(lam > 0.0) else np.zeros(self.V, dtype=np.int64)
        total = int(np.sum(counts))
        if total > 0 or self._carry is not None:
            # grains brighten with the impact speed (shorter contact): band x (U/5)^0.3, darker when wet
            f_crisp = self.f_crisp0 * (1.0 - 0.35 * wet) * np.clip((np.maximum(u, 0.1) / 5.0) ** 0.3, 0.6, 1.6)
            K = self._tick_kernels(f_crisp)
            N = self.N_tick
            if total > 0:
                vi = np.repeat(np.arange(self.V), counts)
                pos = rng.integers(0, n, size=total)
                amp = rng.lognormal(0.0, 0.7, size=total) * rng.choice([-1.0, 1.0], size=total)
                sway_at = sw0[vi] + (sw1 - sw0)[vi] * (pos + 1) / n
                norm = tick_g / np.sqrt(np.maximum(rate, 1.0) / sr * TICK_M2)
                imp = np.bincount(vi * n + pos, weights=amp * sway_at * norm[vi], minlength=self.V * n)
                ticks = sfft.irfft(sfft.rfft(imp.reshape(self.V, n), N, axis=1) * K, N, axis=1)
            else:
                ticks = np.zeros((self.V, N))
            if self._carry is not None:
                c = self._carry.shape[1]
                if c > N:
                    ticks = np.concatenate([ticks, np.zeros((self.V, c - N))], axis=1)
                ticks[:, :c] += self._carry
            y = y + ticks[:, :n]
            carry = ticks[:, n:]
            self._carry = carry.copy() if np.max(np.abs(carry), initial=0.0) > 1e-12 else None

        # ---- creaks from Beaufort 8 (trees only) ----
        start = block_start_sample(ctx)
        for v_i, v in enumerate(self.voices):
            if self.tree[v_i] and u_bft[v_i] > 13.5:
                c_rate = 0.05 * v.count * smoothstep(float(u_bft[v_i]), 14.5, 23.0)
                lam_c = c_rate * n / sr
                k = int(rng.poisson(lam_c)) if lam_c > 0.0 else 0
                for off in np.sort(rng.integers(0, n, size=k)):
                    amp_c = db_spl(72.0) * (u_bft[v_i] / 20.0) * self.level_scale[v_i] / max(self.dist[v_i], 1.0)
                    v.creaks.add(start + int(off), _creak(rng, sr) * amp_c)
            if v.creaks.active:
                y[v_i] = y[v_i] + v.creaks.take(start, n)

        self._g_prev = A
        bus = self.encoder.process(y)
        return [Emission(signal=bus, stem="wind", is_ambisonic=True, spread=1.0, source_id="vegetation")]
