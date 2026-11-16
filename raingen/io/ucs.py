"""Universal Category System (UCS v8.2.1-style) naming and metadata for weather renders.

Filename: CatID_FXName_CreatorID_SourceID[_UserData], e.g.
    RAINMetl_Heavy rain on tin roof close, wind W 30kmh_RGN_raingen
The CatID comes from the dominant sound mechanism: the surface the rain is heard on picks
the RAIN subcategory (metal, glass, water, vegetation, concrete, cloth, plastic, wood,
interior); wind, thunder, aftermath and ambience stems get their own CatIDs. FXName is
generated from the scene in plain words (WMO rain-intensity terms, 8-point compass, km/h).
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

UCS_VERSION = "8.2.1"

# CatID -> (Category, SubCategory), exactly as in the UCS v8.2.1 Full List (the weather-relevant
# subset). Note the WEATHER CatIDs are not prefixed: HAIL, STORM, THUN, and WTHR is WEATHER-MISC.
UCS_CATIDS: dict[str, tuple[str, str]] = {
    "RAIN": ("RAIN", "GENERAL"),
    "RAINClth": ("RAIN", "CLOTH"),
    "RAINConc": ("RAIN", "CONCRETE"),
    "RAINGlas": ("RAIN", "GLASS"),
    "RAINInt": ("RAIN", "INTERIOR"),
    "RAINMetl": ("RAIN", "METAL"),
    "RAINPlas": ("RAIN", "PLASTIC"),
    "RAINVege": ("RAIN", "VEGETATION"),
    "RAINWatr": ("RAIN", "WATER"),
    "RAINWood": ("RAIN", "WOOD"),
    "WIND": ("WIND", "GENERAL"),
    "WINDDsgn": ("WIND", "DESIGNED"),
    "WINDGust": ("WIND", "GUST"),
    "WINDInt": ("WIND", "INTERIOR"),
    "WINDTonl": ("WIND", "TONAL"),
    "WINDTurb": ("WIND", "TURBULENT"),
    "WINDVege": ("WIND", "VEGETATION"),
    "HAIL": ("WEATHER", "HAIL"),
    "STORM": ("WEATHER", "STORM"),
    "THUN": ("WEATHER", "THUNDER"),
    "WTHR": ("WEATHER", "MISC"),
    "WATRDrip": ("WATER", "DRIP"),
    "WATRDran": ("WATER", "DRAIN"),
    "WATRFlow": ("WATER", "FLOW"),
    "WATRLap": ("WATER", "LAP"),
    "WATRMisc": ("WATER", "MISC"),
    "WATRPour": ("WATER", "POUR"),
    "WATRSplsh": ("WATER", "SPLASH"),
    "AMBAlpn": ("AMBIENCE", "ALPINE"),
    "AMBBird": ("AMBIENCE", "BIRDSONG"),
    "AMBDsrt": ("AMBIENCE", "DESERT"),
    "AMBFarm": ("AMBIENCE", "FARM"),
    "AMBForst": ("AMBIENCE", "FOREST"),
    "AMBGras": ("AMBIENCE", "GRASSLAND"),
    "AMBHome": ("AMBIENCE", "RESIDENTIAL"),
    "AMBInsc": ("AMBIENCE", "INSECT"),
    "AMBLake": ("AMBIENCE", "LAKESIDE"),
    "AMBMisc": ("AMBIENCE", "MISC"),
    "AMBPark": ("AMBIENCE", "PARK"),
    "AMBRurl": ("AMBIENCE", "RURAL"),
    "AMBSea": ("AMBIENCE", "SEASIDE"),
    "AMBSubn": ("AMBIENCE", "SUBURBAN"),
    "AMBSwmp": ("AMBIENCE", "SWAMP"),
    "AMBTndra": ("AMBIENCE", "TUNDRA"),
    "AMBTown": ("AMBIENCE", "TOWN"),
    "AMBTraf": ("AMBIENCE", "TRAFFIC"),
    "AMBTrop": ("AMBIENCE", "TROPICAL"),
    "AMBUndr": ("AMBIENCE", "UNDERGROUND"),
    "AMBUrbn": ("AMBIENCE", "URBAN"),
}

CATID_ALIASES = {
    "THUNDER": "THUN", "WEATHER/THUN": "THUN", "WTHRTHUN": "THUN", "WTHRSTRM": "STORM",
    "WTHRHAIL": "HAIL", "WEATHER": "WTHR", "WTHRMISC": "WTHR", "DRIP": "WATRDrip", "GUST": "WINDGust",
}


def canonical_catid(catid: str) -> str:
    """A UCS CatID from a CatID or a legacy/alias spelling ('WTHRThun' -> 'THUN')."""
    c = str(catid).strip()
    if c in UCS_CATIDS:
        return c
    return CATID_ALIASES.get(c.upper(), c)


MATERIAL_CATID = {
    "metal_sheet": "RAINMetl", "tin_roof": "RAINMetl", "car_body": "RAINMetl", "metal_gutter": "RAINMetl",
    "bucket": "RAINMetl", "glass": "RAINGlas", "window": "RAINGlas", "puddle": "RAINWatr", "water": "RAINWatr",
    "grass": "RAINVege", "leaves": "RAINVege", "needles": "RAINVege", "concrete": "RAINConc",
    "asphalt": "RAINConc", "tile": "RAINConc", "gravel": "RAINConc", "fabric": "RAINClth",
    "umbrella": "RAINClth", "tent": "RAINClth", "canvas": "RAINClth", "plastic": "RAINPlas",
    "wood": "RAINWood", "soil": "RAIN", "snow": "RAIN",
}

MATERIAL_WORDS = {
    "metal_sheet": "metal sheet", "tin_roof": "tin roof", "car_body": "car roof", "metal_gutter": "gutter",
    "bucket": "bucket", "glass": "glass", "window": "window", "puddle": "puddles", "water": "water",
    "grass": "grass", "leaves": "leaves", "needles": "pine needles", "concrete": "pavement",
    "asphalt": "road", "tile": "tiles", "gravel": "gravel", "fabric": "fabric", "umbrella": "umbrella",
    "tent": "tent", "canvas": "canvas", "plastic": "plastic", "wood": "wood", "soil": "soil", "snow": "snow",
}

# shelter -> (CatID or None for dominant surface, phrase, closeness). UCS RAIN-INTERIOR is
# "rain from inside house or barn, or vehicle", so attic, car and rooms are RAINInt; tent and
# umbrella fabric is RAIN-CLOTH ("patio umbrella, awning").
SHELTER_INFO = {
    "umbrella": ("RAINClth", "on umbrella", "close"),
    "tent": ("RAINClth", "on tent", "close"),
    "car": ("RAINInt", "on car roof", "from inside"),
    "porch": (None, "from porch", ""),
    "attic": ("RAINInt", "on tin roof", "from inside"),
    "room_closed": ("RAINInt", "on window", "from inside"),
    "room_open": ("RAINInt", "through open window", ""),
    "tree": ("RAINVege", "under tree", ""),
    "cave": ("RAINConc", "at cave mouth", ""),
    "underpass": ("RAINConc", "under bridge", ""),
}

_OBJECT_MATERIAL = {"car": "car_body", "umbrella": "umbrella", "tent": "tent", "window": "window",
                    "roof": "tin_roof", "gutter": "metal_gutter", "downspout": "metal_gutter", "water": "water",
                    "tree": "leaves", "grass": "grass"}

COMPASS8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
COMPASS8_WORDS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")

_PRESET_AMB = {"forest": "AMBForst", "woods": "AMBForst", "conifer_forest": "AMBForst",
               "deciduous_forest": "AMBForst", "jungle": "AMBTrop", "rainforest": "AMBTrop",
               "tropical": "AMBTrop", "city": "AMBUrbn", "city_street": "AMBUrbn", "street": "AMBUrbn",
               "urban": "AMBUrbn", "town": "AMBTown", "village": "AMBTown", "suburb": "AMBSubn",
               "garden": "AMBSubn", "park": "AMBPark", "beach": "AMBSea", "coast": "AMBSea", "sea": "AMBSea",
               "lake": "AMBLake", "lakeside": "AMBLake", "swamp": "AMBSwmp", "marsh": "AMBSwmp",
               "field": "AMBGras", "meadow": "AMBGras", "moor": "AMBGras", "moorland": "AMBGras",
               "prairie": "AMBGras", "farm": "AMBFarm", "mountain": "AMBAlpn", "alpine": "AMBAlpn",
               "desert": "AMBDsrt", "tundra": "AMBTndra", "cabin": "AMBRurl", "countryside": "AMBRurl",
               "cave": "AMBUndr", "underpass": "AMBUrbn", "alley": "AMBUrbn", "mountain_ridge": "AMBAlpn", "bedroom": "AMBHome", "house": "AMBHome"}


def compass8(bearing_deg: float, words: bool = False) -> str:
    i = int(math.floor(((float(bearing_deg) % 360.0) + 22.5) / 45.0)) % 8
    return COMPASS8_WORDS[i] if words else COMPASS8[i]


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def rain_intensity_word(rate_mm_h: float) -> str:
    """WMO-style intensity word for FXNames (slight < 2.5, moderate < 10, heavy < 50, violent)."""
    r = float(rate_mm_h)
    if r < 0.5:
        return "Very light"
    if r < 2.5:
        return "Light"
    if r < 10.0:
        return "Moderate"
    if r < 50.0:
        return "Heavy"
    return "Torrential"


def precipitation_phrase(scene) -> Optional[str]:
    p = scene.weather.precipitation
    ptype = str(_get(p, "type", "rain")).lower()
    rate = float(_get(p, "rate_mm_h", 0.0) or 0.0)
    if ptype == "none" or rate <= 0.0:
        return None
    if ptype == "drizzle":
        return "Heavy drizzle" if rate >= 0.5 else "Drizzle"
    if ptype == "showers":
        return f"{rain_intensity_word(rate)} rain showers"
    if ptype == "freezing_rain":
        return "Freezing rain"
    if ptype == "sleet":
        return "Sleet"
    if ptype == "snow":
        return "Heavy snow" if rate >= 4.0 else "Light snow"
    if ptype == "graupel":
        return "Graupel"
    if ptype == "hail":
        size = float(_get(p, "hail_size_mm", 13.0) or 13.0)
        if size < 10.0:
            return "Pea hail"
        if size < 25.0:
            return "Hail"
        if size < 45.0:
            return "Large hail"
        return "Giant hail"
    return f"{rain_intensity_word(rate)} rain"


def _listener_xy(scene) -> tuple[float, float]:
    pos = _get(scene.listener, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    return float(pos[0]), float(pos[1])


def dominant_surface(scene) -> str:
    """Material that dominates what the listener hears rain on."""
    lx, ly = _listener_xy(scene)
    best, best_d = None, 4.0
    for obj in (_get(scene.place, "objects", []) or []):
        otype = str(_get(obj, "type", "")).lower()
        mat = _get(obj, "material", "") or _OBJECT_MATERIAL.get(otype, "")
        if not mat or otype in ("storm", "wire", "chime", "flag"):
            continue
        pos = _get(obj, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
        d = math.hypot(float(pos[0]) - lx, float(pos[1]) - ly)
        if d < best_d:
            best, best_d = str(mat), d
    if best:
        return best
    surfaces = _get(scene.place, "surfaces", {}) or {}
    if surfaces:
        return str(max(surfaces.items(), key=lambda kv: float(kv[1]))[0])
    return "concrete"


def _rain_catid(scene) -> tuple[str, str, str]:
    """(CatID, surface phrase, closeness) for rain as heard at the listener."""
    shelter = str(_get(scene.place, "shelter", "open")).lower()
    info = SHELTER_INFO.get(shelter)
    if info is not None and info[0] is not None:
        return info
    mat = dominant_surface(scene)
    cat = MATERIAL_CATID.get(mat, "RAIN")
    phrase = f"on {MATERIAL_WORDS.get(mat, mat.replace('_', ' '))}"
    if shelter == "porch":
        phrase = f"from porch, {phrase}"
    preset = str(_get(scene.place, "preset", "")).lower()
    if ("forest" in preset or preset in ("woods", "jungle")) and cat in ("RAIN", "RAINVege"):
        cat, phrase = "RAINVege", ("in jungle" if preset == "jungle" else "in forest")
    closeness = ""
    vibe = _get(scene, "vibe", {}) or {}
    c = vibe.get("closeness") if isinstance(vibe, dict) else None
    if c is not None:
        closeness = "close" if float(c) >= 0.66 else ("distant" if float(c) <= 0.33 else "")
    return cat, phrase, closeness


def _wind_phrase(scene, force: bool = False) -> Optional[str]:
    w = scene.weather.wind
    v = float(_get(w, "speed_mps", 0.0) or 0.0)
    if v < 2.5 and not force:
        return None
    kmh = int(5 * round(v * 3.6 / 5.0))
    gf = _get(w, "gust_factor", None)
    gusty = " gusty" if (gf is not None and float(gf) >= 1.6) or v >= 10.8 else ""
    return f"wind {compass8(float(_get(w, 'from_deg', 270.0)))} {kmh}kmh{gusty}"


def _thunder_phrase(scene) -> Optional[str]:
    th = scene.weather.thunder
    if not _get(th, "enabled", False):
        return None
    d = float(_get(th, "distance_km", 8.0))
    if d < 2.0:
        return "close thunder"
    if d < 8.0:
        return "thunder"
    return "distant thunder"


def _is_storm(scene) -> bool:
    th = scene.weather.thunder
    rate = float(_get(scene.weather.precipitation, "rate_mm_h", 0.0) or 0.0)
    return bool(_get(th, "enabled", False)) and float(_get(th, "distance_km", 8.0)) < 5.0 and rate >= 10.0


def _sanitize(text: str, maxlen: int = 100) -> str:
    t = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", str(text)).replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    return t[:maxlen].rstrip(" ,.")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _code(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9\-]", "-", str(text)).strip("-")
    return t or "X"


def ucs_catid(scene, stem: Optional[str] = None) -> str:
    """CatID for a full mix (stem None) or one stem."""
    if stem:
        stem = stem.lower()
    ptype = str(_get(scene.weather.precipitation, "type", "rain")).lower()
    rate = float(_get(scene.weather.precipitation, "rate_mm_h", 0.0) or 0.0)
    wind_v = float(_get(scene.weather.wind, "speed_mps", 0.0) or 0.0)
    if stem == "thunder":
        return "THUN"
    if stem == "wind":
        veg = _get(scene.place, "vegetation", {}) or {}
        trees = float(veg.get("deciduous", 0) or 0) + float(veg.get("conifer", 0) or 0) if isinstance(veg, dict) else 0.0
        gf = _get(scene.weather.wind, "gust_factor", None)
        if wind_v >= 20.8:                       # Beaufort 9+: UCS WIND-TURBULENT "huge storm winds"
            return "WINDTurb"
        if wind_v >= 10.8 or (gf is not None and float(gf) >= 1.6):
            return "WINDGust"
        if trees >= 3:
            return "WINDVege"
        return "WIND"
    if stem == "aftermath":
        return "WATRDrip"
    if stem == "ambience":
        preset = str(_get(scene.place, "preset", "")).lower()
        return _PRESET_AMB.get(preset, "AMBRurl")
    if stem == "rain_far":
        return "RAIN"
    if stem == "interior":
        shelter = str(_get(scene.place, "shelter", "open")).lower()
        return {"tent": "RAINClth", "umbrella": "RAINClth"}.get(shelter, "RAINInt")
    # full mix or rain_near
    if ptype == "none" or rate <= 0.0:
        if stem == "rain_near":
            return "RAIN"
        if _get(scene.weather.thunder, "enabled", False):
            return "THUN"
        if wind_v >= 5.0:
            return "WIND"
        preset = str(_get(scene.place, "preset", "")).lower()
        return _PRESET_AMB.get(preset, "AMBMisc")
    if ptype == "hail":
        return "HAIL"
    if ptype in ("snow", "sleet", "graupel", "ice_pellets", "freezing_rain"):
        return "WTHR"
    if stem is None and _is_storm(scene):
        return "STORM"
    return _rain_catid(scene)[0]


def ucs_fxname(scene, stem: Optional[str] = None) -> str:
    """Human FXName, e.g. 'Heavy rain on tin roof close, wind W 30kmh'."""
    stem_l = stem.lower() if stem else None
    precip = precipitation_phrase(scene)
    th = scene.weather.thunder
    if stem_l == "thunder":
        d = float(_get(th, "distance_km", 8.0))
        b = compass8(float(_get(th, "bearing_deg", 270.0)))
        roll = " rolling" if d >= 3.0 else " crack"
        return _sanitize(f"Thunder {d:.0f}km {b}{roll}")
    if stem_l == "wind":
        return _sanitize(_cap(_wind_phrase(scene, force=True) or "wind"))
    if stem_l == "aftermath":
        emit = _get(scene.place, "emitters", []) or []
        has_gutter = any(str(_get(e, "type", "")).lower() in ("gutter", "downspout") for e in emit)
        return _sanitize("Drips and runoff" + (" from gutter" if has_gutter else ""))
    if stem_l == "ambience":
        preset = str(_get(scene.place, "preset", "outdoor")).replace("_", " ")
        hour = float(_get(scene.weather, "hour", 12.0) or 12.0)
        tod = "night" if (hour < 6.0 or hour >= 21.0) else ("dawn" if hour < 9.0 else ("evening" if hour >= 18.0 else "day"))
        return _sanitize(f"{preset.capitalize()} {tod} ambience" + (" in rain" if precip else ""))
    if stem_l == "rain_far":
        return _sanitize(f"{precip or 'Rain'} distant wash")
    if precip is None:
        parts = [_cap(_wind_phrase(scene, force=True) or "calm")]
        tp = _thunder_phrase(scene)
        if tp:
            parts.append(tp)
        return _sanitize(", ".join(parts))
    _, phrase, closeness = _rain_catid(scene)
    head = f"{precip} {phrase}" + (f" {closeness}" if closeness else "")
    if stem_l in ("rain_near", "interior"):
        return _sanitize(head)
    parts = [head]
    wp = _wind_phrase(scene)
    if wp:
        parts.append(wp)
    tp = _thunder_phrase(scene)
    if tp:
        parts.append(tp)
    return _sanitize(", ".join(parts))


def ucs_filename(scene, stem: Optional[str] = None, creator: str = "RGN", source: str = "raingen",
                 ext: str = "") -> str:
    """CatID_FXName_CreatorID_SourceID[_stem][ext]."""
    name = f"{ucs_catid(scene, stem)}_{ucs_fxname(scene, stem)}_{_code(creator)}_{_code(source)}"
    if stem:
        name += f"_{_code(stem)}"
    if ext:
        name += ext if ext.startswith(".") else f".{ext}"
    return name


def _description(scene, stem: Optional[str]) -> str:
    from .. import __version__ as engine_version

    w = scene.weather
    p = w.precipitation
    parts = []
    precip = precipitation_phrase(scene)
    if precip:
        dm = _get(p, "dm_mm", None)
        drops = _get(p, "drops", "normal")
        parts.append(f"Physically modelled {precip.lower()} ({float(p.rate_mm_h):.1f} mm/h, {drops} drops"
                     + (f", Dm {float(dm):.2f} mm" if dm else "") + ")")
        _, phrase, _ = _rain_catid(scene)
        parts.append(f"heard {phrase}")
    else:
        parts.append("Physically modelled weather without precipitation")
    shelter = str(_get(scene.place, "shelter", "open"))
    parts.append(f"listener {shelter.replace('_', ' ')} at {_get(scene.place, 'preset', 'outdoors')}")
    v = float(_get(w.wind, "speed_mps", 0.0) or 0.0)
    if v > 0.3:
        parts.append(f"wind from {compass8(float(w.wind.from_deg), words=True)} at {v * 3.6:.0f} km/h")
    th = w.thunder
    if _get(th, "enabled", False):
        parts.append(f"thunder {float(th.distance_km):.1f} km {compass8(float(th.bearing_deg), words=True)}, "
                     f"{float(th.flashes_per_min):.1f} flashes/min")
    parts.append(f"{float(w.temperature_c):.0f} C, {float(w.humidity_pct):.0f}% RH, {float(w.hour):04.1f} h, {w.season}")
    if stem:
        parts.append(f"stem: {stem}")
    parts.append(f"seed 0x{int(scene.seed):X}, raingen {engine_version}")
    return "; ".join(parts) + "."


def _keywords(scene, stem: Optional[str], catid: str) -> list[str]:
    kw: list[str] = ["weather", "procedural", "raingen"]
    cat, sub = UCS_CATIDS.get(catid, ("", ""))
    kw += [cat.lower(), sub.lower()]
    precip = precipitation_phrase(scene)
    if precip:
        kw += precip.lower().split()
        kw.append(dominant_surface(scene).replace("_", " "))
    shelter = str(_get(scene.place, "shelter", "open"))
    if shelter != "open":
        kw.append(shelter.replace("_", " "))
    kw.append(str(_get(scene.place, "preset", "")))
    if float(_get(scene.weather.wind, "speed_mps", 0.0) or 0.0) >= 2.5:
        kw += ["wind", compass8(float(scene.weather.wind.from_deg), words=True)]
    if _get(scene.weather.thunder, "enabled", False):
        kw += ["thunder", "storm", "lightning"]
    kw.append(str(_get(scene.weather, "season", "")))
    hour = float(_get(scene.weather, "hour", 12.0) or 12.0)
    kw.append("night" if (hour < 6.0 or hour >= 21.0) else "day")
    if stem:
        kw.append(stem.replace("_", " "))
    seen, out = set(), []
    for k in kw:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def ucs_metadata(scene, stem: Optional[str] = None, creator: str = "RGN", source: str = "raingen",
                 designer: str = "raingen", library: str = "raingen procedural weather") -> dict:
    """UCS metadata dict (CatID, Category, SubCategory, FXName, Description, Keywords, Library, Designer...)."""
    catid = ucs_catid(scene, stem)
    cat, sub = UCS_CATIDS.get(catid, ("", ""))
    fx = ucs_fxname(scene, stem)
    kws = _keywords(scene, stem, catid)
    return {
        "CatID": catid,
        "Category": cat,
        "SubCategory": sub,
        "CategoryFull": f"{cat}-{sub}" if cat else "",
        "FXName": fx,
        "TrackTitle": fx,
        "Description": _description(scene, stem),
        "Keywords": ", ".join(kws),
        "KeywordList": kws,
        "Library": library,
        "Designer": designer,
        "Manufacturer": "raingen",
        "CreatorID": _code(creator),
        "SourceID": _code(source),
        "Source": source,
        "UserData": stem or "",
        "Stem": stem or "mix",
        "UCSVersion": UCS_VERSION,
        "Filename": ucs_filename(scene, stem, creator, source),
    }
