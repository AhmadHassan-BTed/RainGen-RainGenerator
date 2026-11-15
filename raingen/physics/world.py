"""Per-block weather resolution: Scene (human units) -> WeatherState (physics).

Each block the World:
  1. advances the storm system (arc envelope, showers, Cox 1/f modulation, moving
     thunder cells) -> instantaneous rain multiplier and Dm factor (size sorting);
  2. advances the Kaimal gust process (peak n = f z/U ~ 0.045 evaluated at the ears;
     variance scaled by stability, unstable ~4x stable after Bolin) and converts the
     10 m report to wind at the ears with the log law (u(z) = u10 ln(z/z0)/ln(10/z0));
  3. resolves the drop-size distribution at the instantaneous rate (Nw solved for
     rate closure) and the five acoustic class fluxes;
  4. fills and drains the wetness stores (film, puddles, canopy, roof/gutter);
  5. derives day/night from solar geometry (Cooper 1969 declination, sunrise hour
     angle cos w0 = -tan(phi) tan(delta)) and a Pasquill-style stability class
     (clear calm nights -> stable inversion, sunny light-wind days -> unstable,
     overcast / windy / rainy -> neutral) for the refraction model;
  6. applies a modest convective outflow cooling near active thunder cells.
"""

from __future__ import annotations

import math

from ..core.constants import BLOCK, SAMPLE_RATE, speed_of_sound
from ..core.types import WeatherState, bearing_to_azimuth
from .dsd import CLASS_NAMES, class_flux, dsd_from_scene, dsd_with_dm
from .layout import build_layout, exposure_by_material
from .storm import StormSystem
from .wetness import WetnessStores
from .wind import GustProcess, kaimal_integral_time_s, terrain_z0, turbulence_intensity_for, wind_at_height

_LIQUID_FRACTION = {
    "none": 0.0, "drizzle": 1.0, "rain": 1.0, "showers": 1.0, "freezing_rain": 1.0,
    "freezing_drizzle": 1.0, "sleet": 0.5, "snow": 0.0, "graupel": 0.3, "hail": 0.5,
}
_NO_DROP_TYPES = ("none", "snow", "graupel")
_SEASON_DOY = {"spring": 80, "summer": 172, "autumn": 266, "fall": 266, "winter": 355}
_CHARACTER_MEAN = {"showery": 0.4, "intermittent": 0.7}
_STABILITY_TI = {"unstable": math.sqrt(2.0), "neutral": 1.0, "stable": math.sqrt(0.5)}
_STABILITY_TAU_S = 20.0


def daylight_hours(latitude_deg: float, season: str) -> tuple[float, float]:
    """(sunrise, sunset) in local solar hours for a representative day of the season."""
    doy = _SEASON_DOY.get(str(season).lower(), 172)
    if latitude_deg < 0.0:
        doy = (doy + 182) % 365          # seasons are local: southern summer is December
    decl = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + doy) / 365.0)
    phi = math.radians(min(max(float(latitude_deg), -89.9), 89.9))
    x = min(max(-math.tan(phi) * math.tan(decl), -1.0), 1.0)
    w0 = math.degrees(math.acos(x))
    return 12.0 - w0 / 15.0, 12.0 + w0 / 15.0


def is_daytime(hour: float, latitude_deg: float, season: str) -> bool:
    rise, sset = daylight_hours(latitude_deg, season)
    hr = float(hour) % 24.0
    return rise <= hr < sset


def stability_class(is_day: bool, cloud_okta: float, wind_mps: float, rain_mm_h: float) -> str:
    """Pasquill-Gifford-style near-ground stability: 'unstable' | 'neutral' | 'stable'."""
    if wind_mps >= 6.0 or rain_mm_h >= 2.0 or cloud_okta >= 7.0:
        return "neutral"
    if is_day:
        return "unstable" if (cloud_okta <= 4.0 and wind_mps < 5.0) else "neutral"
    return "stable" if (cloud_okta <= 6.0 and wind_mps < 5.0) else "neutral"


class World:
    """Physically resolved weather, one `step` per block."""

    def __init__(self, scene, rngf, sr: int = SAMPLE_RATE, block: int = BLOCK):
        self.scene = scene
        self.rngf = rngf
        self.sr = int(sr)
        self.block = int(block)
        place = scene.place
        self.z0 = float(place.z0_m) if place.z0_m is not None else terrain_z0(place.preset)
        self.layout = build_layout(scene)
        self.exposure = exposure_by_material(self.layout)
        self.exposure["canopy"] = self._canopy_cover()
        self.exposure["roof"] = 1.0
        # drip edges (eaves, rims, leaf tips) keep their own film store so they go on
        # dripping from storage after the rain stops; edges are exposed to the sky
        edge_mats = set()
        for p in self.layout:
            for key in ("edge_material", "roof_material"):
                m = p.params.get(key)
                if m:
                    edge_mats.add(str(m))
        for m in edge_mats:
            self.exposure.setdefault(m, 1.0)
        materials = sorted({p.material for p in self.layout} | edge_mats | {"puddle", "water"})
        self.storms = StormSystem(scene, rngf, sr=self.sr, block=self.block)
        w0 = scene.weather
        rate0 = 0.0 if str(w0.precipitation.type).lower() == "none" else max(float(w0.precipitation.rate_mm_h), 0.0)
        day0 = is_daytime(float(w0.hour), float(getattr(w0, "latitude_deg", 50.0)), str(w0.season))
        stab0 = stability_class(day0, float(w0.cloud_okta), max(float(w0.wind.speed_mps), 0.0), rate0)
        self._stab_factor = _STABILITY_TI.get(stab0, 1.0)
        T, ti = self._gust_params(scene.weather.wind, scene.listener.height_m)
        self.gusts = GustProcess(rngf, "wind.gust", integral_time_s=T, ti=ti, control_rate=self.sr / self.block)
        self.wetness = WetnessStores(materials, aftermath_tau_min=place.aftermath_tau_min,
                                     canopy_capacity_mm=self._canopy_capacity(), roof_area_m2=self._roof_area())
        self._primed = False
        self.state: WeatherState | None = None
        self.dsd = None

    # ---------------------------------------------------------------- setup helpers
    def _canopy_cover(self) -> float:
        shelter = str(self.scene.place.shelter).lower()
        if shelter == "tree" or any(p.params.get("feed") == "canopy" for p in self.layout):
            return 1.0
        if any(p.kind == "vegetation" and p.params.get("species") not in (None, "grass") for p in self.layout):
            return 0.5
        return 0.0

    def _canopy_capacity(self) -> float:
        lais = [float(p.params.get("lai", 0.0)) for p in self.layout
                if p.kind == "vegetation" and p.params.get("species") not in (None, "grass")]
        overhead = [float(p.params.get("lai", 0.0)) for p in self.layout if p.params.get("overhead")]
        pool = overhead or lais
        if not pool:
            return 1.0
        lai = sum(pool) / len(pool)
        return min(max(0.25 * lai, 0.2), 3.0)       # storage capacity S ~ 0.2-0.3 x LAI (mm)

    def _roof_area(self) -> float:
        catch = [float(p.params.get("catchment_m2", 0.0)) for p in self.layout
                 if p.kind == "gutter" and p.params.get("catchment_m2")]
        roofs = sum(p.area_m2 for p in self.layout
                    if p.material in ("tin_roof", "tile") and not p.params.get("ground"))
        area = max(catch + [min(roofs, 200.0)]) if (catch or roofs) else 30.0
        return max(area, 5.0)

    def _gust_params(self, wind, listener_h: float) -> tuple[float, float]:
        """(integral time s, turbulence intensity) of the gust process at the ears.

        Kaimal's peak n = f z / U ~ 0.045 is evaluated at the listener's height with the
        log-law wind there (gusts are shorter near the ground: ~6 s periods at 2 m and
        7 m/s vs ~22 s at 10 m and 10 m/s). A user-set gust factor / TI refers to 10 m
        and is carried down with the neutral profile ratio I(z)/I(10). With automatic
        turbulence, stability scales the variance (unstable ~4x stable, Bolin).
        """
        u10 = max(float(wind.speed_mps), 0.0)
        z_ear = max(float(listener_h), 0.3)
        if wind.integral_time_s:
            T = float(wind.integral_time_s)
        elif wind.gust_rate_per_min:
            T = 60.0 / (2.0 * math.pi * max(float(wind.gust_rate_per_min), 0.05))
        else:
            u_ear = wind_at_height(u10, z_ear, self.z0)
            T = kaimal_integral_time_s(u_ear, max(z_ear, 1.0))
        ti10_auto = turbulence_intensity_for(self.z0, 10.0)
        auto = False
        if wind.turbulence_intensity is not None:
            ti10 = float(wind.turbulence_intensity)
        elif wind.gust_factor is not None:
            ti10 = max(float(wind.gust_factor) - 1.0, 0.0) / 3.0
        else:
            ti10 = ti10_auto
            auto = True
        ti_h = turbulence_intensity_for(self.z0, z_ear) * ti10 / max(ti10_auto, 1e-6)
        if auto:
            ti_h *= self._stab_factor
        return T, float(min(max(ti_h, 0.0), 0.8))

    def _update_stability_factor(self, stability: str, dt: float) -> None:
        # std factor: variance x2 unstable, x0.5 stable -> unstable/stable = 4 (Bolin)
        target = _STABILITY_TI.get(stability, 1.0)
        a = 1.0 - math.exp(-max(dt, 0.0) / _STABILITY_TAU_S)
        self._stab_factor += (target - self._stab_factor) * a

    def _prime(self, sc, base_rate: float, liquid: float) -> None:
        """Start the wetness stores in a state consistent with the scene's story."""
        w = sc.weather
        p = w.precipitation
        arc = str(w.storm_arc).lower()
        ptype = str(p.type).lower()
        character = "showery" if ptype == "showers" else str(p.character).lower()
        if ptype == "none" or base_rate <= 0.0:
            if arc == "clearing":                     # it has just rained
                self.wetness.equilibrate(3.0, self.exposure, minutes=60.0)
            return
        if arc in ("arriving", "passing"):
            return                                    # the storm is only arriving: start dry
        avg = base_rate * liquid * _CHARACTER_MEAN.get(character, 1.0)
        if avg > 0.0:
            self.wetness.equilibrate(avg, self.exposure)

    # ---------------------------------------------------------------- per block
    def step(self, block_index: int, t0: float, n: int, scene_at_t=None) -> WeatherState:
        sc = scene_at_t if scene_at_t is not None else self.scene
        w = sc.weather
        p = w.precipitation
        wind = w.wind
        place = sc.place
        L = sc.listener
        dt = max(int(n), 0) / float(self.sr)

        rain_mult, cells, env = self.storms.step(block_index, t0, dt, sc)
        ptype = str(p.type or "rain").lower()
        base_rate = 0.0 if ptype == "none" else max(float(p.rate_mm_h), 0.0)
        u10 = max(float(wind.speed_mps), 0.0)

        # sky and near-ground stability (decided on the slow envelope, not the Cox flicker)
        hour = float(w.hour) % 24.0
        season = str(w.season).lower()
        day = is_daytime(hour, float(getattr(w, "latitude_deg", 50.0)), season)
        stability = stability_class(day, float(w.cloud_okta), u10, base_rate * env)
        self._update_stability_factor(stability, dt)

        # wind at the ears with gusts
        T, ti = self._gust_params(wind, L.height_m)
        self.gusts.set_params(T, ti)
        gust = self.gusts.step(block_index)
        u_h = wind_at_height(u10, float(L.height_m), self.z0)
        wind_listener = u_h * gust
        wind_az = bearing_to_azimuth(float(wind.from_deg), float(L.heading_deg))

        # precipitation: arc x showers x Cox x gust curtains
        curtain = min(max(1.0 + 0.3 * (gust - 1.0), 0.5), 1.8)
        rate = base_rate * rain_mult * curtain
        dsd = dsd_from_scene(p, rate)
        dm_factor = self.storms.dm_factor
        if rate > 0.0 and abs(dm_factor - 1.0) > 1e-3 and p.dm_mm is None and p.nw is None and ptype != "drizzle":
            dsd = dsd_with_dm(dsd, dsd.dm_mm * dm_factor, rate)
        if ptype in _NO_DROP_TYPES or rate <= 0.0:
            flux = {name: 0.0 for name in CLASS_NAMES}
        else:
            flux = class_flux(dsd, getattr(p, "class_boundaries", "medwin") or "medwin")
        self.dsd = dsd

        # wetness memory
        liquid = _LIQUID_FRACTION.get(ptype, 1.0)
        if not self._primed:
            self._prime(sc, base_rate, liquid)
            self._primed = True
        self.wetness.aftermath_tau_min = max(float(place.aftermath_tau_min), 0.25)
        self.wetness.update(dt, rate * liquid, self.exposure)
        wet, depth = self.wetness.snapshot()

        # convective outflow cooling under a nearby thunder cell
        temp = float(w.temperature_c)
        if cells and bool(getattr(w.thunder, "enabled", False)):
            nearest = min(c.distance_km for c in cells)
            temp -= 3.0 * math.exp(-nearest / 3.0) * min(self.storms.thunder_envelope, 1.0)

        if ptype == "hail":
            hail = float(p.hail_size_mm)
        elif ptype == "graupel":
            hail = 3.0              # soft-hail pellets, 2-5 mm (WMO)
        elif ptype == "sleet":
            hail = 3.5              # ice pellets (US sleet) < 5 mm
        else:
            hail = 0.0

        state = WeatherState(
            t=float(t0),
            precip_type=ptype,
            rain_rate_mm_h=float(rate),
            dsd=dsd,
            class_flux=flux,
            hail_size_mm=hail,
            wind_u10_mps=u10,
            wind_listener_mps=float(wind_listener),
            gust=float(gust),
            wind_from_deg=float(wind.from_deg) % 360.0,
            wind_azimuth=float(wind_az),
            turbulence_intensity=float(ti),
            temperature_c=float(temp),
            humidity_pct=float(w.humidity_pct),
            pressure_hpa=float(w.pressure_hpa),
            speed_of_sound=float(speed_of_sound(temp)),
            hour=hour,
            season=season,
            is_day=bool(day),
            storms=cells,
            wetness=wet,
            puddle_depth_mm=depth,
            canopy_storage_mm=float(self.wetness.canopy_mm),
            roof_flow_l_s=float(self.wetness.roof_flow_l_s),
            stability=stability,
            intensity_envelope=float(env),
        )
        self.state = state
        return state

    # ---------------------------------------------------------------- conveniences
    def flashes_between(self, t0: float, t1: float) -> list:
        return self.storms.flashes_between(t0, t1)

    def drip_feed(self, feed: str, exposure: float = 1.0) -> tuple[float, float]:
        """(inflow_mm_h, storage_mm) to drive a DripNode for a patch's params['feed'].

        'rain': the current rain on its catchment; 'canopy': canopy release rate plus
        canopy storage; 'roof': gutter flow spread over the roof catchment plus the
        roof film; 'storage': seepage from the scene's wetness (keeps dripping after rain).
        """
        st = self.state
        rate = st.rain_rate_mm_h if st is not None else 0.0
        liquid = _LIQUID_FRACTION.get(st.precip_type, 1.0) if st is not None else 1.0
        f = str(feed or "rain").lower()
        if f == "canopy":
            return self.wetness.canopy_drip_mm_h, self.wetness.canopy_mm
        if f == "roof":
            area = max(self.wetness.roof_area_m2, 1.0)
            return self.wetness.roof_flow_l_s * 3600.0 / area, 0.0
        if f == "storage":
            films = [self.wetness.film_mm(m) for m in self.wetness.materials]
            store = max(films) if films else 0.0
            return 0.0, store * 4.0
        return rate * liquid * max(float(exposure), 0.0), 0.0
