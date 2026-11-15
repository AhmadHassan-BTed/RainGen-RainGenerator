"""Wind in vegetation: species spectra, Beaufort ladder and advected gusts.

Physics:
* Fégeant (1999) and Bolin (2006, 2009): vegetation noise level scales as
  L ~ 10*log10(U^(2*chi)) with chi = 1.5..2.7 depending on species.
* Conifers: a log-Gaussian "whoosh" centred on the needle Strouhal frequency
  f = St*U/d_n (St ~ 0.2, needle diameter d_n ~ 1-1.5 mm).
* Broadleaf trees: rustle = dense granular leaf-to-leaf impacts (grains) plus
  foliage swish modulated by crown sway.
* Bare (winter) branches: aeolian whistles f = 0.2*U/d for twigs d = 3..10 mm.
* Beaufort ladder (WMO land descriptions, defined on the 10 m wind): leaves rustle
  from force 2, twigs/small branches move from force 3-4, branches whistle at
  force 5-6, trunks creak and branches break from force 8. The ladder switches on
  the 10 m-equivalent wind including the gust at the plant; levels use the log-law
  wind at crown height.
* Gusts are advected (Taylor): a stand of trees s metres upwind hears each gust
  s/U seconds before the listener (GustProcess.at_position), so gusts are heard
  sweeping across the scene.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import signal

from raingen.core.constants import SAMPLE_RATE
from raingen.core.types import BlockContext, Emission, SourcePatch
from raingen.physics.atmosphere import AirAbsorptionFilter, spreading_gain
from raingen.physics.wind import wind_at_height

from .wind import (GustField, PendingMixer, Ramp, StatefulSOS, bandpass_rms_gain, bandpass_sos, block_start_sample,
                   db_spl, noise_rms_gain, poisson_offsets, scene_z0, shelter_of, smoothstep, upwind_distance)

# level_db: SPL at 1 m from the crown centre for one plant at U = 5 m/s
SPECIES = {
    "deciduous": {"chi": 2.0, "level_db": 64.0, "tick": 1.0, "tick_rate": 2500.0, "swish": 0.55,
                  "needle": 0.0, "whistle": 0.12, "height": 10.0, "tree": True},
    "conifer": {"chi": 2.3, "level_db": 62.0, "tick": 0.12, "tick_rate": 600.0, "swish": 0.3,
                "needle": 1.0, "needle_d_mm": 1.2, "whistle": 0.18, "height": 14.0, "tree": True},
    "bush": {"chi": 1.8, "level_db": 56.0, "tick": 0.8, "tick_rate": 1800.0, "swish": 0.6,
             "needle": 0.0, "whistle": 0.05, "height": 1.5, "tree": False},
    "grass": {"chi": 1.6, "level_db": 50.0, "tick": 0.3, "tick_rate": 3000.0, "swish": 1.0,
              "needle": 0.0, "whistle": 0.0, "height": 0.4, "tree": False},
    "palm": {"chi": 2.0, "level_db": 62.0, "tick": 0.6, "tick_rate": 900.0, "swish": 0.7,
             "needle": 0.0, "whistle": 0.05, "height": 9.0, "tree": True},
    "reeds": {"chi": 1.7, "level_db": 52.0, "tick": 0.5, "tick_rate": 2000.0, "swish": 0.9,
              "needle": 0.4, "needle_d_mm": 5.0, "whistle": 0.0, "height": 1.8, "tree": False},
}

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


class _VegVoice:
    """One vegetation patch (a tree, a stand, a hedge or a lawn)."""

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
        self.whistle_jit = np.zeros(3)
        area = max(float(patch.area_m2), 1.0)
        self.radius = float(p.get("crown_radius_m", p.get("canopy_radius_m", math.sqrt(area / math.pi))))
        self.fixed_spread = p.get("spread")

        self.crisp = StatefulSOS(bandpass_sos(self.f_crisp0, 1.4, sr))
        self.body = StatefulSOS(bandpass_sos(self.f_body, 1.8, sr))
        self._fc_crisp = self.f_crisp0
        self._g_tick = math.sqrt(0.49 * noise_rms_gain(self.crisp.sos) ** 2 + 0.25 * noise_rms_gain(self.body.sos) ** 2)
        self.swish = StatefulSOS(bandpass_sos(500.0, 0.7, sr))
        self._fc_swish, self._g_swish = 500.0, noise_rms_gain(self.swish.sos)
        self.needle = StatefulSOS(np.vstack([bandpass_sos(800.0, 1.0, sr)] * 2))
        self._fc_needle, self._g_needle = 800.0, noise_rms_gain(self.needle.sos)
        self.whistles = [StatefulSOS(bandpass_sos(400.0, 15.0, sr)) for _ in range(3)]
        self._fc_wh = np.full(3, 400.0)
        self._g_wh = np.full(3, bandpass_rms_gain(400.0, 15.0, sr))
        self.gains = Ramp(np.zeros(6))
        self.creaks = PendingMixer(1)
        self.air = AirAbsorptionFilter(sr)

    def _retune(self, filt: StatefulSOS, sos: np.ndarray) -> float:
        filt.set(sos)
        return noise_rms_gain(sos)

    def render(self, ctx: BlockContext, field: GustField, z0: float) -> Optional[Emission]:
        w = ctx.weather
        n, sr = ctx.n, ctx.sr
        u_h = float(wind_at_height(w.wind_u10_mps, max(0.7 * self.height, 0.5), z0))
        s = upwind_distance(self.patch.position, w.wind_azimuth)
        g_loc = field.at(s, max(u_h, 0.5))
        u = u_h * g_loc * self.exposure
        u_bft = float(w.wind_u10_mps) * g_loc * self.exposure     # 10 m-equivalent wind at this plant
        if u < 0.5 and u_bft < 1.2 and float(np.max(self.gains.value)) < 1e-8 and not self.creaks.active:
            return None
        rng = ctx.rng.get(f"veg.{self.patch.id}", ctx.index)
        spec = self.spec
        wet = float(w.wetness.get(self.patch.material, w.wetness.get("leaves", 0.0))) if w.wetness else 0.0

        on_leaf = smoothstep(u_bft, 1.3, 3.0)       # Beaufort 2 (1.6-3.3 m/s): leaves rustle
        on_twig = smoothstep(u_bft, 3.2, 6.5)       # Beaufort 3-4: twigs in constant motion
        on_whistle = smoothstep(u_bft, 7.0, 11.0)   # Beaufort 5-6: branches move, whistle
        if self.species in ("grass", "reeds"):
            on_twig = on_leaf
        A = db_spl(spec["level_db"]) * (max(u, 0.0) / 5.0) ** spec["chi"] * math.sqrt(self.count)
        A *= smoothstep(u, 0.8, 2.0) * self.level_scale

        t = (ctx.t0 + np.arange(n) / sr)
        sway_depth = 0.25 + 0.45 * on_twig
        sway = 1.0 + sway_depth * np.sin(2.0 * np.pi * self.sway_f * t + self.sway_phase)

        # leaf ticks: dense granular impacts
        f_crisp = self.f_crisp0 * (1.0 - 0.35 * wet)
        if abs(f_crisp - self._fc_crisp) / self._fc_crisp > 0.03:
            self._fc_crisp = f_crisp
            self.crisp.set(bandpass_sos(f_crisp, 1.4, sr))
            self._g_tick = math.sqrt(0.49 * noise_rms_gain(self.crisp.sos) ** 2 + 0.25 * noise_rms_gain(self.body.sos) ** 2)
        rate = min(15000.0, spec["tick_rate"] * max(self.leaf, 0.02) * math.sqrt(self.count) * (max(u, 0.1) / 5.0) ** 1.3)
        rate *= on_leaf
        imp = np.zeros(n)
        offs = poisson_offsets(rng, rate, n, sr)
        if offs.size:
            np.add.at(imp, offs, rng.lognormal(0.0, 0.7, size=offs.size) * rng.choice([-1.0, 1.0], size=offs.size))
            imp *= sway
        tick = 0.7 * self.crisp.process(imp) + 0.5 * self.body.process(imp)
        tick /= self._g_tick * math.sqrt(max(rate, 1.0) / sr * 2.66)

        # foliage / branch swish
        fc_sw = 300.0 + 45.0 * u
        if abs(fc_sw - self._fc_swish) / self._fc_swish > 0.03:
            self._fc_swish = fc_sw
            self._g_swish = self._retune(self.swish, bandpass_sos(fc_sw, 0.7, sr))
        swish = self.swish.process(rng.standard_normal(n)) / self._g_swish * sway

        # needle Strouhal whoosh (log-Gaussian-like peak from two cascaded resonances)
        needle = np.zeros(n)
        if spec["needle"] > 0.0:
            f_n = min(max(0.2 * u / self.d_needle, 150.0), 12000.0)
            if abs(f_n - self._fc_needle) / self._fc_needle > 0.03:
                self._fc_needle = f_n
                self.needle.set(np.vstack([bandpass_sos(f_n, 1.0, sr)] * 2))
                self._g_needle = noise_rms_gain(self.needle.sos)
            needle = self.needle.process(rng.standard_normal(n)) / self._g_needle * np.sqrt(sway)

        # bare-branch aeolian whistles f = 0.2 U / d with turbulent jitter
        a = math.exp(-(n / sr) / 0.7)
        self.whistle_jit = a * self.whistle_jit + math.sqrt(1 - a * a) * rng.standard_normal(3)
        whistles = np.zeros((3, n))
        w_on = spec["whistle"] * on_whistle * (1.0 + 3.0 * (1.0 - self.leaf))
        if w_on > 1e-4:
            noise3 = rng.standard_normal((3, n))
            for j in range(3):
                f_j = min(max(0.2 * u / self.twig_d[j] * (1.0 + 0.03 * self.whistle_jit[j]), 60.0), 9000.0)
                if abs(f_j - self._fc_wh[j]) / self._fc_wh[j] > 0.015:
                    self._fc_wh[j] = f_j
                    self.whistles[j].set(bandpass_sos(f_j, 15.0, sr))
                    self._g_wh[j] = bandpass_rms_gain(f_j, 15.0, sr)
                whistles[j] = self.whistles[j].process(noise3[j]) / self._g_wh[j]

        leaf_w = spec["tick"] * self.leaf * on_leaf * (1.0 - 0.4 * wet)
        swish_w = spec["swish"] * (0.25 + 0.75 * self.leaf) * on_twig * (1.0 + 0.3 * wet)
        needle_w = spec["needle"] * on_leaf
        wh = w_on / math.sqrt(3.0) * (0.5 + 0.5 * np.abs(self.whistle_jit))
        g = self.gains(A * np.array([leaf_w, swish_w, needle_w, wh[0], wh[1], wh[2]]), n)
        y = g[0] * tick + g[1] * swish + g[2] * needle + g[3] * whistles[0] + g[4] * whistles[1] + g[5] * whistles[2]

        # creaks from Beaufort 8 (trees only)
        if spec["tree"] and u_bft > 13.5:
            crng = ctx.rng.get(f"veg.creak.{self.patch.id}", ctx.index)
            c_rate = 0.05 * self.count * smoothstep(u_bft, 14.5, 23.0)
            start = block_start_sample(ctx)
            for off in poisson_offsets(crng, c_rate, n, sr):
                self.creaks.add(start + int(off), _creak(crng, sr) * db_spl(72.0) * (u_bft / 20.0) * self.level_scale)
        if self.creaks.active:
            y = y + self.creaks.take(block_start_sample(ctx), n)

        dist = self.patch.distance
        y = y * spreading_gain(dist)
        if dist > 3.0:
            y = self.air.process(y, dist, w.temperature_c, w.humidity_pct)
        spread = float(self.fixed_spread) if self.fixed_spread is not None else min(max(self.radius / dist, 0.05), 0.95)
        return Emission(signal=y, azimuth=self.patch.azimuth, elevation=self.patch.elevation, distance=dist,
                        spread=spread, stem="wind", source_id=f"veg.{self.patch.id}")


class VegetationGenerator:
    """All vegetation patches of the layout (plus a ground grass bed from the surface mix)."""

    name = "vegetation"

    def __init__(self, scene, layout, rngf, gust_field: Optional[GustField] = None, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.field = gust_field or GustField(scene, rngf, sr)
        self.z0 = scene_z0(scene)
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

    def render(self, ctx: BlockContext) -> list[Emission]:
        self.field.advance(ctx.index, ctx.weather)
        out = []
        for v in self.voices:
            em = v.render(ctx, self.field, self.z0)
            if em is not None:
                out.append(em)
        return out
