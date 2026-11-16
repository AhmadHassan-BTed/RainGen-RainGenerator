"""Live weather providers (MASTER_PLAN 2.8 "Weather outside now").

* `OpenMeteoProvider` - Open-Meteo forecast API `current=` conditions
  (https://open-meteo.com/en/docs). Keyless on the free endpoint; the free tier
  is **non-commercial** and attributed (CC BY 4.0). Commercial use needs an API
  key on the customer endpoint (bring your own key) or a self-hosted instance
  (pass `base_url`). The model is left on "best match" because some named
  models are being discontinued. Caveats: precipitation fields are sums over
  the *preceding* interval reported by the API (`current.interval`, up to one
  hour) and gusts are the preceding-interval maximum, so they are converted to a
  mean rate and a gust factor; `lightning_potential` exists only for Central
  Europe (ICON-D2), so thunder comes from the WMO weather code.
* `MetarProvider` - aviationweather.gov Data API
  (``/api/data/metar?ids=KJFK&format=raw``) with a descriptive User-Agent, as the
  service requests; the nearest station to a coordinate is found with a bbox query.
* Place names are geocoded with the Open-Meteo geocoding API.

Privacy: only an approximate location is ever sent to a weather service
(coordinates rounded to 0.1 deg, about 10 km; MASTER_PLAN 2.8). Pass
``coarse_location=False`` to a provider only for a deliberate, user-approved
precise lookup.

Grid means sound static, so `scene_from_live` adds stochastic sub-grid detail:
showery cells for convective codes, gust rate and timing, a storm-cell bearing,
distance and approach. All randomness comes from `RngFactory(seed).get(...)`, so a
given observation always maps to the same scene. The engine is expected to
slew live updates over 30-120 s (2.7).
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Optional, Union

from .. import __version__
from ..core.rng import RngFactory
from ..scene.metar import scene_from_metar
from ..scene.schema import Scene
from ..scene.units import beaufort_to_mps

USER_AGENT = f"raingen/{__version__} (weather sound engine; https://github.com/AhmadHassan-BTed/RainGen-RainGenerator)"


class LiveWeatherError(RuntimeError):
    pass


# WMO Code Table 4677 subset used by Open-Meteo (28 codes) plus 97 "heavy thunderstorm" = 29.
# type, representative rate (mm/h water equivalent), character, thunder, extra
WMO_CODES: dict[int, dict[str, Any]] = {
    0: {"label": "clear sky", "type": "none", "rate": 0.0, "character": "continuous", "thunder": False, "okta": 0},
    1: {"label": "mainly clear", "type": "none", "rate": 0.0, "character": "continuous", "thunder": False, "okta": 2},
    2: {"label": "partly cloudy", "type": "none", "rate": 0.0, "character": "continuous", "thunder": False, "okta": 4},
    3: {"label": "overcast", "type": "none", "rate": 0.0, "character": "continuous", "thunder": False, "okta": 8},
    45: {"label": "fog", "type": "none", "rate": 0.0, "character": "continuous", "thunder": False, "okta": 8, "fog": True},
    48: {"label": "depositing rime fog", "type": "none", "rate": 0.0, "character": "continuous", "thunder": False, "okta": 8, "fog": True},
    51: {"label": "light drizzle", "type": "drizzle", "rate": 0.08, "character": "continuous", "thunder": False, "okta": 8},
    53: {"label": "moderate drizzle", "type": "drizzle", "rate": 0.3, "character": "continuous", "thunder": False, "okta": 8},
    55: {"label": "dense drizzle", "type": "drizzle", "rate": 0.8, "character": "continuous", "thunder": False, "okta": 8},
    56: {"label": "light freezing drizzle", "type": "freezing_rain", "rate": 0.2, "character": "continuous", "thunder": False, "okta": 8, "drops": "fine"},
    57: {"label": "dense freezing drizzle", "type": "freezing_rain", "rate": 0.6, "character": "continuous", "thunder": False, "okta": 8, "drops": "fine"},
    61: {"label": "slight rain", "type": "rain", "rate": 1.0, "character": "continuous", "thunder": False, "okta": 8},
    63: {"label": "moderate rain", "type": "rain", "rate": 5.0, "character": "continuous", "thunder": False, "okta": 8},
    65: {"label": "heavy rain", "type": "rain", "rate": 15.0, "character": "continuous", "thunder": False, "okta": 8},
    66: {"label": "light freezing rain", "type": "freezing_rain", "rate": 1.0, "character": "continuous", "thunder": False, "okta": 8},
    67: {"label": "heavy freezing rain", "type": "freezing_rain", "rate": 5.0, "character": "continuous", "thunder": False, "okta": 8},
    71: {"label": "slight snow fall", "type": "snow", "rate": 0.5, "character": "continuous", "thunder": False, "okta": 8},
    73: {"label": "moderate snow fall", "type": "snow", "rate": 1.5, "character": "continuous", "thunder": False, "okta": 8},
    75: {"label": "heavy snow fall", "type": "snow", "rate": 4.0, "character": "continuous", "thunder": False, "okta": 8},
    77: {"label": "snow grains", "type": "snow", "rate": 0.2, "character": "continuous", "thunder": False, "okta": 8, "drops": "fine"},
    80: {"label": "slight rain showers", "type": "showers", "rate": 2.0, "character": "showery", "thunder": False, "okta": 6},
    81: {"label": "moderate rain showers", "type": "showers", "rate": 8.0, "character": "showery", "thunder": False, "okta": 7},
    82: {"label": "violent rain showers", "type": "showers", "rate": 40.0, "character": "showery", "thunder": False, "okta": 8, "drops": "fat"},
    85: {"label": "slight snow showers", "type": "snow", "rate": 0.8, "character": "showery", "thunder": False, "okta": 6},
    86: {"label": "heavy snow showers", "type": "snow", "rate": 3.0, "character": "showery", "thunder": False, "okta": 8},
    95: {"label": "thunderstorm", "type": "showers", "rate": 12.0, "character": "showery", "thunder": True, "okta": 8, "drops": "fat"},
    96: {"label": "thunderstorm with slight hail", "type": "hail", "rate": 15.0, "character": "showery", "thunder": True, "okta": 8, "hail_mm": 8.0},
    97: {"label": "heavy thunderstorm", "type": "showers", "rate": 35.0, "character": "showery", "thunder": True, "okta": 8, "drops": "fat", "heavy": True},
    99: {"label": "thunderstorm with heavy hail", "type": "hail", "rate": 30.0, "character": "showery", "thunder": True, "okta": 8, "hail_mm": 25.0, "heavy": True},
}


def _http_json(url: str, timeout_s: float, user_agent: str = USER_AGENT) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise LiveWeatherError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).netloc}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LiveWeatherError(f"cannot reach {urllib.parse.urlsplit(url).netloc}: {e}") from e
    except json.JSONDecodeError as e:
        raise LiveWeatherError(f"bad JSON from {urllib.parse.urlsplit(url).netloc}") from e


def _http_text(url: str, timeout_s: float, user_agent: str = USER_AGENT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise LiveWeatherError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).netloc}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LiveWeatherError(f"cannot reach {urllib.parse.urlsplit(url).netloc}: {e}") from e


def geocode(name: str, timeout_s: float = 10.0, language: str = "en") -> dict:
    """Place name -> {'lat', 'lon', 'name', 'country', 'timezone'} via the Open-Meteo geocoding API."""
    q = urllib.parse.urlencode({"name": name, "count": 1, "language": language, "format": "json"})
    data = _http_json(f"https://geocoding-api.open-meteo.com/v1/search?{q}", timeout_s)
    results = (data or {}).get("results") or []
    if not results:
        raise LiveWeatherError(f"no place found for {name!r}")
    r = results[0]
    return {"lat": float(r["latitude"]), "lon": float(r["longitude"]), "name": r.get("name", name),
            "country": r.get("country", ""), "timezone": r.get("timezone", ""), "elevation_m": r.get("elevation")}


def _parse_location(location: Union[str, tuple, list, dict], timeout_s: float) -> dict:
    if isinstance(location, dict):
        return {"lat": float(location["lat"]), "lon": float(location["lon"]), "name": location.get("name", "")}
    if isinstance(location, (tuple, list)) and len(location) >= 2:
        return {"lat": float(location[0]), "lon": float(location[1]), "name": f"{location[0]:.3f},{location[1]:.3f}"}
    text = str(location).strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if m:
        return {"lat": float(m.group(1)), "lon": float(m.group(2)), "name": text}
    return geocode(text, timeout_s)


class OpenMeteoProvider:
    """Open-Meteo current conditions. Free endpoint: non-commercial, CC BY 4.0 attribution."""

    name = "open-meteo"
    FREE_URL = "https://api.open-meteo.com/v1/forecast"
    CUSTOMER_URL = "https://customer-api.open-meteo.com/v1/forecast"
    CURRENT = ("precipitation", "rain", "showers", "snowfall", "weather_code", "wind_speed_10m", "wind_gusts_10m",
               "wind_direction_10m", "temperature_2m", "relative_humidity_2m", "cloud_cover", "is_day")
    ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout_s: float = 10.0,
                 user_agent: str = USER_AGENT, coarse_location: bool = True):
        self.api_key = api_key
        self.base_url = base_url or (self.CUSTOMER_URL if api_key else self.FREE_URL)
        self.timeout_s = float(timeout_s)
        self.user_agent = user_agent
        self.coarse_location = bool(coarse_location)

    def url(self, lat: float, lon: float) -> str:
        lat, lon = _coarse(lat, lon) if self.coarse_location else (float(lat), float(lon))
        q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "current": ",".join(self.CURRENT),
             "wind_speed_unit": "ms", "temperature_unit": "celsius", "precipitation_unit": "mm", "timezone": "auto"}
        if self.api_key:
            q["apikey"] = self.api_key
        return f"{self.base_url}?{urllib.parse.urlencode(q)}"

    def current(self, lat: float, lon: float) -> dict:
        """Normalised observation: rates in mm/h, wind in m/s, local time, raw payload under 'raw'."""
        data = _http_json(self.url(lat, lon), self.timeout_s, self.user_agent)
        cur = (data or {}).get("current")
        if not cur:
            raise LiveWeatherError("Open-Meteo response has no 'current' block")
        interval_s = float(cur.get("interval") or 3600.0)
        per_h = 3600.0 / max(60.0, interval_s)

        def f(key: str, default: float = 0.0) -> float:
            v = cur.get(key)
            return float(v) if v is not None else default

        t_local = cur.get("time")
        try:
            dt = datetime.fromisoformat(t_local) if t_local else None
        except ValueError:
            dt = None
        return {
            "provider": self.name,
            "lat": float(data.get("latitude", lat)), "lon": float(data.get("longitude", lon)),
            "time": t_local, "hour": (dt.hour + dt.minute / 60.0) if dt else None, "month": dt.month if dt else None,
            "interval_s": interval_s,
            "precip_mm_h": f("precipitation") * per_h,
            "rain_mm_h": f("rain") * per_h,
            "showers_mm_h": f("showers") * per_h,
            "snow_water_mm_h": f("snowfall") * 10.0 / 7.0 * per_h,   # cm of snow -> mm water (Open-Meteo: divide by 7)
            "weather_code": int(f("weather_code", 0)),
            "wind_mps": f("wind_speed_10m"),
            "gust_mps": f("wind_gusts_10m"),
            "wind_from_deg": f("wind_direction_10m", 270.0),
            "temperature_c": f("temperature_2m", 12.0),
            "humidity_pct": f("relative_humidity_2m", 80.0),
            "cloud_pct": f("cloud_cover", 100.0),
            "is_day": bool(int(f("is_day", 1))),
            "attribution": self.ATTRIBUTION if not self.api_key else "Open-Meteo.com",
            "raw": data,
        }


class MetarProvider:
    """aviationweather.gov METAR (US NWS Aviation Weather Center Data API)."""

    name = "metar"
    URL = "https://aviationweather.gov/api/data/metar"

    def __init__(self, timeout_s: float = 10.0, user_agent: str = USER_AGENT, coarse_location: bool = True):
        self.timeout_s = float(timeout_s)
        self.user_agent = user_agent
        self.coarse_location = bool(coarse_location)

    def fetch_raw(self, station: str) -> str:
        q = urllib.parse.urlencode({"ids": station.strip().upper(), "format": "raw"})
        text = _http_text(f"{self.URL}?{q}", self.timeout_s, self.user_agent)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise LiveWeatherError(f"no METAR for {station!r}")
        return lines[0]

    def nearest(self, lat: float, lon: float, radius_deg: float = 1.0) -> tuple[str, str]:
        """(ICAO id, raw METAR) of the reporting station nearest to (lat, lon)."""
        if self.coarse_location:
            lat, lon = _coarse(lat, lon)
        bbox = f"{lat - radius_deg:.3f},{lon - radius_deg:.3f},{lat + radius_deg:.3f},{lon + radius_deg:.3f}"
        q = urllib.parse.urlencode({"bbox": bbox, "format": "json"})
        data = _http_json(f"{self.URL}?{q}", self.timeout_s, self.user_agent) or []
        best = None
        for row in data:
            try:
                slat, slon = float(row["lat"]), float(row["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            d = _haversine_km(lat, lon, slat, slon)
            if row.get("rawOb") and (best is None or d < best[0]):
                best = (d, row.get("icaoId", ""), row["rawOb"])
        if best is None:
            if radius_deg < 4.0:
                return self.nearest(lat, lon, radius_deg * 2.0)
            raise LiveWeatherError("no METAR station near that location")
        return best[1], best[2]


def _coarse(lat: float, lon: float) -> tuple[float, float]:
    """Round a coordinate to 0.1 deg (~10 km) before it leaves the machine."""
    return round(float(lat), 1), round(float(lon), 1)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _season(month: Optional[int], lat: float) -> Optional[str]:
    if month is None:
        return None
    north = ("winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer",
             "autumn", "autumn", "autumn", "winter")[month - 1]
    if lat < 0:
        return {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}[north]
    return north


def scene_from_observation(obs: dict, base: Optional[Scene] = None, name: str = "") -> Scene:
    """Map a normalised Open-Meteo observation to a Scene, adding sub-grid variability."""
    scene = base.copy() if base is not None else Scene()
    code = int(obs.get("weather_code", 0))
    info = WMO_CODES.get(code, WMO_CODES[3 if code < 50 else 63])
    seed_key = f"{obs.get('time')}|{obs.get('lat', 0):.3f}|{obs.get('lon', 0):.3f}|{code}"
    rng = RngFactory(scene.seed).get("live.subgrid|" + seed_key, 0)
    w = scene.weather
    pr = w.precipitation

    measured = float(obs.get("rain_mm_h", 0.0)) + float(obs.get("showers_mm_h", 0.0))
    snow = float(obs.get("snow_water_mm_h", 0.0))
    total = max(float(obs.get("precip_mm_h", 0.0)), measured + snow)
    ptype = info["type"]
    if ptype == "none" and total >= 0.05:
        ptype = "snow" if snow > measured else ("showers" if obs.get("showers_mm_h", 0.0) > obs.get("rain_mm_h", 0.0) else "rain")
    if ptype == "none":
        pr.type = "none"
        pr.rate_mm_h = 0.0
    else:
        pr.type = ptype
        rate = total if total >= 0.05 else info["rate"]
        # a 1 h grid mean of a showery field under-represents the cells: keep the mean, add showery structure
        pr.rate_mm_h = float(min(150.0, max(0.02, rate)))
        pr.character = info["character"]
        if obs.get("showers_mm_h", 0.0) > 0.5 * max(1e-6, obs.get("rain_mm_h", 0.0)):
            pr.character = "showery"
        if pr.character == "showery":
            pr.shower_period_min = float(rng.uniform(8.0, 20.0))
        elif pr.rate_mm_h < 1.0 and ptype in ("rain", "drizzle"):
            pr.character = "intermittent" if rng.random() < 0.5 else "continuous"
        if "drops" in info:
            pr.drops = info["drops"]
        if ptype == "hail":
            pr.hail_size_mm = float(info.get("hail_mm", 10.0) * rng.uniform(0.8, 1.25))

    # wind: mean, gust factor from the preceding-interval maximum, sub-grid gust rate
    u = float(obs.get("wind_mps", 0.0))
    g = float(obs.get("gust_mps", 0.0))
    wind = w.wind
    wind.speed_mps = u
    wind.from_deg = float(obs.get("wind_from_deg", wind.from_deg)) % 360.0
    if u > 0.5 and g > u:
        wind.gust_factor = float(min(2.5, max(1.1, g / u)))
    else:
        wind.gust_factor = None
    wind.gust_rate_per_min = float(rng.uniform(0.8, 3.0)) if u > 2.0 else None

    # thunder from the weather code (no lightning field outside ICON-D2)
    th = w.thunder
    if info["thunder"]:
        th.enabled = True
        heavy = bool(info.get("heavy"))
        lo, hi = (1.0, 6.0) if heavy else (2.0, 12.0)
        th.distance_km = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        th.bearing_deg = float((wind.from_deg + rng.uniform(-60.0, 60.0)) % 360.0)
        th.approach_kmh = float(rng.uniform(-10.0, 35.0))
        th.flashes_per_min = float(rng.uniform(3.0, 8.0) if heavy else rng.uniform(0.8, 4.0))
        th.stroke_intensity = float(rng.uniform(0.6, 0.9) if heavy else rng.uniform(0.35, 0.7))
        w.storm_arc = "steady"
    else:
        th.enabled = False

    w.temperature_c = float(obs.get("temperature_c", w.temperature_c))
    w.humidity_pct = float(min(100.0, max(10.0, obs.get("humidity_pct", w.humidity_pct))))
    w.cloud_okta = float(min(8.0, max(0.0, obs.get("cloud_pct", 100.0) / 12.5)))
    if info.get("fog"):
        w.visibility_km = 0.2
        w.humidity_pct = 100.0
    if obs.get("hour") is not None:
        w.hour = float(obs["hour"])
    if obs.get("lat") is not None:
        w.latitude_deg = float(obs["lat"])
        season = _season(obs.get("month"), float(obs["lat"]))
        if season:
            w.season = season
    label = name or obs.get("name") or f"{obs.get('lat', 0):.2f},{obs.get('lon', 0):.2f}"
    scene.name = f"Live: {label} ({info['label']})"
    return scene


def scene_from_live(location: Union[str, tuple, list, dict], base: Optional[Scene] = None,
                    provider: Union[str, OpenMeteoProvider, MetarProvider] = "open-meteo",
                    api_key: Optional[str] = None, timeout_s: float = 10.0) -> Scene:
    """Scene from current weather at `location` (place name, 'lat,lon', (lat, lon), or an ICAO id
    for the METAR provider)."""
    if isinstance(provider, str):
        key = provider.strip().lower().replace("_", "-")
        if key in ("open-meteo", "openmeteo", "om"):
            provider = OpenMeteoProvider(api_key=api_key, timeout_s=timeout_s)
        elif key in ("metar", "aviationweather", "awc"):
            provider = MetarProvider(timeout_s=timeout_s)
        else:
            raise ValueError(f"unknown provider {provider!r}")
    if isinstance(provider, MetarProvider):
        if isinstance(location, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{3}", location.strip()):
            station, raw = location.strip().upper(), provider.fetch_raw(location)
        else:
            loc = _parse_location(location, timeout_s)
            station, raw = provider.nearest(loc["lat"], loc["lon"])
        scene = scene_from_metar(raw, base)
        rng = RngFactory(scene.seed).get(f"live.metar|{raw}", 0)
        if scene.weather.precipitation.character == "showery":
            scene.weather.precipitation.shower_period_min = float(rng.uniform(8.0, 20.0))
        if scene.weather.wind.speed_mps > 2.0 and scene.weather.wind.gust_rate_per_min is None:
            scene.weather.wind.gust_rate_per_min = float(rng.uniform(0.8, 3.0))
        scene.name = f"Live METAR {station}" + (f": {scene.name.split(': ', 1)[1]}" if ": " in scene.name else "")
        return scene
    loc = _parse_location(location, timeout_s)
    obs = provider.current(loc["lat"], loc["lon"])
    return scene_from_observation(obs, base, name=loc.get("name", ""))


def demo_observation(code: int = 63, wind_mps: float = beaufort_to_mps(4)) -> dict:
    """An offline observation (for docs, tests and the UI when the network is unavailable)."""
    info = WMO_CODES.get(code, WMO_CODES[63])
    return {"provider": "demo", "lat": 51.5, "lon": -0.12, "time": "2026-11-16T22:00", "hour": 22.0, "month": 11,
            "interval_s": 3600.0, "precip_mm_h": info["rate"], "rain_mm_h": info["rate"] if info["type"] != "showers" else 0.0,
            "showers_mm_h": info["rate"] if info["type"] == "showers" else 0.0, "snow_water_mm_h": 0.0,
            "weather_code": code, "wind_mps": wind_mps, "gust_mps": wind_mps * 1.6, "wind_from_deg": 240.0,
            "temperature_c": 11.0, "humidity_pct": 92.0, "cloud_pct": 100.0, "is_day": False}
