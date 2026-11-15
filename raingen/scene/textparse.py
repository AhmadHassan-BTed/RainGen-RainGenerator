"""'Describe your weather': a deterministic vocabulary parser (MASTER_PLAN 2.8).

Known vocabulary first, no language model: rain words and explicit rates
(WMO intensity words, drizzle / downpour / cloudburst, '8 mm/h'), wind
(Beaufort names, 'force 6', '30 mph gusts', 'from the west', 'westerly'),
thunder ('distant', 'close', '3 km', '9 seconds', 'overhead', 'approaching'),
places and shelters ('tin roof', 'tent', 'car', 'umbrella', 'window', 'forest',
'city'), surfaces, time of day, seasons, temperatures, wellness modes ('for
sleep / focus / asmr') and METAR present-weather groups ('+TSRA', 'VCTS',
'-FZDZ'). A whole METAR pasted as text is routed to the METAR parser.

`parse_text` returns what was recognised (for the "shown to the user before
applying" step); `scene_from_text` applies it on top of a base scene.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from . import units
from .metar import looks_like_metar, parse_weather_group, scene_from_metar, weather_assignments
from .presets import apply_place
from .schema import Scene

_NUM = r"(\d+(?:[.,]\d+)?)"
_SPEED_UNIT = r"(mph|km/h|kmh|kph|km\s*per\s*hour|kilometers per hour|kilometres per hour|kts?|knots?|kn|m/s|mps|meters per second|metres per second)"
_DIST_UNIT = r"(km|kilometres?|kilometers?|mi|miles?|nm|nautical miles?|m|metres?|meters?)"


@dataclass
class ParseResult:
    """What the parser understood. `assignments` maps scene paths to values."""
    text: str
    assignments: dict = field(default_factory=dict)
    place: Optional[str] = None
    surfaces_add: dict = field(default_factory=dict)
    emitters_add: list = field(default_factory=list)
    matched: list = field(default_factory=list)          # (phrase, meaning)
    metar: bool = False

    def set(self, path: str, value: Any, phrase: str, force: bool = False) -> bool:
        if path in self.assignments and not force:
            return False
        self.assignments[path] = value
        self.matched.append((phrase, f"{path} = {value}"))
        return True

    def has(self, path: str) -> bool:
        return path in self.assignments


def _find(pattern: str, text: str) -> Optional[re.Match]:
    return re.search(pattern, text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def _metar_groups(raw: str, r: ParseResult) -> None:
    groups = []
    for tok in raw.split():
        tok = tok.strip(",.;:()")
        if len(tok) < 2 or tok != tok.upper() or not re.search(r"[A-Z]", tok):
            continue
        if not re.search(r"(DZ|RA|SN|SG|PL|GR|GS|TS|SH|FZ|SQ|UP)", tok):
            continue
        g = parse_weather_group(tok)
        if g is not None:
            groups.append(g)
            r.matched.append((tok, "METAR weather group"))
    if groups:
        for path, value in weather_assignments(groups).items():
            r.set(path, value, groups[0]["raw"])


def _explicit_rate(t: str, r: ParseResult) -> None:
    m = _find(_NUM + r"\s*(mm\s*/\s*h(?:r|our)?|mm\s*per\s*hour|mm\s*an\s*hour|mmph|mm/hr|in(?:ches|ch)?\s*(?:/|per|an)\s*h(?:r|our)?|in/h)", t)
    if m:
        unit = m.group(2).replace(" ", "")
        unit = "in/h" if unit.startswith("in") else "mm/h"
        r.set("weather.precipitation.rate_mm_h", units.parse_rate(f"{m.group(1)} {unit}"), m.group(0))


_PRECIP_RULES: list[tuple[str, str, Optional[float], dict]] = [
    # pattern, type, rate (None = keep), extras
    (r"\bfreezing\s+drizzle\b", "freezing_rain", 0.4, {"weather.precipitation.drops": "fine", "weather.temperature_c": -1.0}),
    (r"\bfreezing\s+rain\b|\bglaze\b|\bice\s+storm\b", "freezing_rain", 2.0, {"weather.temperature_c": -1.0}),
    (r"\bsleet\b|\bice\s+pellets\b|\bwintry\s+mix\b|\brain\s+and\s+snow\b", "sleet", 2.0, {"weather.temperature_c": 0.5}),
    (r"\bgraupel\b|\bsnow\s+pellets\b", "graupel", 2.0, {"weather.temperature_c": 0.0}),
    (r"\bhail\w*\b", "hail", 15.0, {}),
    (r"\bblizzard\b", "snow", 4.0, {"weather.wind.speed_mps": units.beaufort_to_mps(8), "weather.temperature_c": -6.0}),
    (r"\bheavy\s+snow\w*\b", "snow", 3.0, {"weather.temperature_c": -3.0}),
    (r"\b(?:light\s+snow\w*|snow\s+flurr\w*|flurr\w*)\b", "snow", 0.5, {"weather.temperature_c": -2.0}),
    (r"\bsnow\w*\b", "snow", 1.5, {"weather.temperature_c": -2.0}),
    (r"\bcloudburst\b", "rain", 120.0, {"weather.precipitation.drops": "fat"}),
    (r"\b(?:torrential|deluge|monsoon|biblical|tropical\s+downpour)\b", "rain", 60.0, {}),
    (r"\b(?:downpour|pouring|bucketing|lashing|teeming|hammering|pelting|pissing)\b", "rain", 25.0, {}),
    (r"\b(?:heavy|hard)\s+showers?\b", "showers", 15.0, {"weather.precipitation.character": "showery"}),
    (r"\b(?:heavy|hard|driving|intense)\s+rain\w*\b", "rain", 20.0, {}),
    (r"\bheavy\s+drizzle\b", "drizzle", 0.8, {"weather.precipitation.drops": "fine"}),
    (r"\blight\s+drizzle\b|\bmizzle\b|\bscotch\s+mist\b|\bmisty\s+rain\b", "drizzle", 0.1, {"weather.precipitation.drops": "fine"}),
    (r"\bdrizzl\w*\b", "drizzle", 0.3, {"weather.precipitation.drops": "fine"}),
    (r"\b(?:light|gentle|soft|slight|fine|little|bit\s+of)\s+(?:rain|showers?)\b|\bspitting\b|\bsprinkl\w*\b", "rain", 1.2, {}),
    (r"\b(?:moderate|steady|constant|persistent|continuous)\s+rain\b", "rain", 5.0, {"weather.precipitation.character": "continuous"}),
    (r"\bshowers?\b|\bshowery\b|\bpassing\s+showers\b|\bscattered\s+showers\b", "showers", 6.0, {"weather.precipitation.character": "showery"}),
    (r"\brain\w*\b|\bwet\b|\bpitter[- ]patter\b", "rain", None, {}),
]


def _precip(t: str, r: ParseResult, base: Scene) -> None:
    if _find(r"\b(?:no\s+rain|dry\s+weather|rain\s+(?:has\s+)?stopped|stopped\s+raining|after\s+the\s+(?:rain|storm|downpour)|"
             r"aftermath|drips?\s+only|just\s+(?:the\s+)?drips|wind\s+only|no\s+precipitation|bone\s+dry)\b", t):
        m = _find(r"\b(?:after\s+the\s+(?:rain|storm|downpour)|aftermath|drips?\s+only|just\s+(?:the\s+)?drips|rain\s+(?:has\s+)?stopped|stopped\s+raining)\b", t)
        r.set("weather.precipitation.type", "none", "no rain")
        r.set("weather.precipitation.rate_mm_h", 0.0, "no rain")
        if m:
            r.set("weather.storm_arc", "clearing", m.group(0))
        return
    for pattern, ptype, rate, extras in _PRECIP_RULES:
        m = _find(pattern, t)
        if not m:
            continue
        r.set("weather.precipitation.type", ptype, m.group(0))
        if rate is not None:
            r.set("weather.precipitation.rate_mm_h", rate, m.group(0))
        elif not r.has("weather.precipitation.rate_mm_h"):
            if base.weather.precipitation.type in ("none", "drizzle") or base.weather.precipitation.rate_mm_h < 0.5:
                r.set("weather.precipitation.rate_mm_h", 4.0, m.group(0))
        for path, val in extras.items():
            r.set(path, val, m.group(0))
        break
    m = _find(r"\b(?:fat|big|large|heavy|huge|plump)\s+(?:rain)?drops\b|\bfirst\s+(?:fat\s+)?drops\b", t)
    if m:
        r.set("weather.precipitation.drops", "fat", m.group(0))
    m = _find(r"\b(?:fine|tiny|small|misty)\s+(?:rain)?drops\b|\bmist(?:y)?\b|\bfine\s+rain\b", t)
    if m:
        r.set("weather.precipitation.drops", "fine", m.group(0))
    m = _find(r"\b(?:intermittent|on\s+and\s+off|on-and-off|patchy|spells?\s+of|sporadic|comes\s+and\s+goes)\b", t)
    if m:
        r.set("weather.precipitation.character", "intermittent", m.group(0))
    m = _find(r"\b(?:steady|constant|continuous|unchanging|even)\b", t)
    if m:
        r.set("weather.precipitation.character", "continuous", m.group(0))
    m = _find(r"\b(pea|marble|penny|nickel|quarter|walnut|golf[- ]?ball|tennis[- ]?ball|baseball|softball)[- ]?(?:sized?)?\s+hail", t)
    if m:
        r.set("weather.precipitation.hail_size_mm", units.parse_hail_size_mm(m.group(1).replace("-", " ")), m.group(0))
    m = _find(_NUM + r"\s*(mm|cm|in(?:ch(?:es)?)?)\s+hail", t)
    if m:
        r.set("weather.precipitation.hail_size_mm", units.parse_hail_size_mm(f"{m.group(1)} {m.group(2)}"), m.group(0))
    if _find(r"\bfirst\s+(?:fat\s+)?drops\b|\bleading\s+edge\b|\bstarting\s+to\s+rain\b|\bbeginning\s+to\s+rain\b", t):
        r.set("weather.storm_arc", "arriving", "first drops")


_BFT_NAMES = sorted(units._BEAUFORT_ALIASES.items(), key=lambda kv: -len(kv[0]))
_WIND_WORDS: list[tuple[str, float, Optional[float]]] = [
    # pattern, beaufort, gust factor
    (r"\b(?:windless|no\s+wind|still\s+air|dead\s+calm|calm\s+(?:wind|air|conditions)|no\s+breeze|perfectly\s+still)\b", 0.3, None),
    (r"\b(?:howling|roaring|screaming|raging)\s+winds?\b|\bwinds?\s+(?:howling|roaring|screaming|raging)\b", 8.0, 1.8),
    (r"\bvery\s+windy\b|\bstrong\s+winds?\b|\bhigh\s+winds?\b", 7.0, None),
    (r"\bsquall(?:s|y)?\b", 7.0, 2.0),
    (r"\bblustery\b", 6.0, 1.8),
    (r"\bwindy\b|\bwindswept\b", 6.0, None),
    (r"\bbreezy\b", 4.0, None),
    (r"\blight\s+winds?\b|\bslight\s+breeze\b", 2.0, None),
    (r"\bwind\b|\bbreeze\b", 3.0, None),
]


def _speed_mentions(t: str) -> list[tuple[float, str, int, int]]:
    out = []
    for m in re.finditer(_NUM + r"\s*" + _SPEED_UNIT + r"\b", t, flags=re.IGNORECASE):
        unit = re.sub(r"\s+", " ", m.group(2).lower())
        unit = {"km per hour": "km/h", "kilometers per hour": "km/h", "kilometres per hour": "km/h",
                "meters per second": "m/s", "metres per second": "m/s"}.get(unit, unit)
        try:
            v = units.parse_speed(f"{m.group(1)} {unit}")
        except ValueError:
            continue
        out.append((v, m.group(0), m.start(), m.end()))
    return out


def _wind(t: str, r: ParseResult) -> None:
    gust_mps: Optional[float] = None
    for v, phrase, a, b in _speed_mentions(t):
        before = t[max(0, a - 24):a].lower()
        after = t[b:b + 14].lower()
        if re.search(r"approach|moving|travell?ing|heading", before):
            r.set("weather.thunder.approach_kmh", v * 3.6, phrase)
            continue
        if re.match(r"^\s*(?:wind\s+)?gusts?\b", after) or re.search(r"gust\w*\s+(?:to|of|up\s+to|at|near)?\s*$", before):
            gust_mps = v if gust_mps is None else max(gust_mps, v)
            r.matched.append((phrase, "gust speed"))
            continue
        r.set("weather.wind.speed_mps", v, phrase)
    m = _find(r"\b(?:force|bft|beaufort|f)\s*" + _NUM + r"\b", t)
    if m:
        r.set("weather.wind.speed_mps", units.beaufort_to_mps(units._num(m.group(1))), m.group(0), force=True)
    if not r.has("weather.wind.speed_mps"):
        for name, force in _BFT_NAMES:
            if name in ("storm", "calm"):
                continue                       # too ambiguous in free text; handled by other rules
            m = _find(r"\b" + re.escape(name) + r"\b", t)
            if m:
                r.set("weather.wind.speed_mps", units.beaufort_to_mps(force), m.group(0))
                break
    gust_factor: Optional[float] = None
    for pattern, bft, g in _WIND_WORDS:
        m = _find(pattern, t)
        if m:
            r.set("weather.wind.speed_mps", units.beaufort_to_mps(bft), m.group(0))
            if g is not None:
                gust_factor = g
            break
    if _find(r"\bgusty\b|\bgusts?\b|\bgusting\b", t) and gust_factor is None:
        gust_factor = 1.8
    if gust_mps is not None:
        mean = r.assignments.get("weather.wind.speed_mps")
        if mean is None:
            mean = gust_mps / 1.6
            r.set("weather.wind.speed_mps", mean, "gusts (mean from typical gust factor 1.6)")
        gust_factor = max(1.05, min(2.5, gust_mps / max(0.5, mean)))
    if gust_factor is not None:
        r.set("weather.wind.gust_factor", float(gust_factor), "gusts")
    # direction
    for pattern in (
        r"\b(?:wind|winds|breeze|gusts?|blowing|coming)?\s*from\s+the\s+([a-z]+(?:[- ][a-z]+){0,2})",
        r"\bfrom\s+([nsew]{1,3}\b|\d{1,3}\s*(?:°|deg(?:rees)?\b)?)",
        r"\b((?:north|south|east|west)(?:[- ]?(?:north|south|east|west)){0,2})(?:erly)?\s+(?:winds?|breeze|gale)\b",
        r"\b((?:north|south|east|west)(?:[- ]?(?:north|south|east|west)){0,2}erly)\b",
        r"\bwinds?\s+(?:from\s+)?([nsew]{1,3})\b",
    ):
        m = _find(pattern, t)
        if not m:
            continue
        words = m.group(1).strip().split()
        for k in range(len(words), 0, -1):          # 'the west tonight' -> try 'west tonight', then 'west'
            try:
                deg = units.parse_direction_deg(" ".join(words[:k]))
            except ValueError:
                continue
            r.set("weather.wind.from_deg", deg, m.group(0))
            break
        if r.has("weather.wind.from_deg"):
            break


def _thunder(t: str, r: ParseResult, temp_c: float) -> None:
    th = _find(r"\bthunder\w*\b|\blightning\b|\belectrical\s+storm\b|\bt-?storms?\b|\bstorm\s+cell\b", t)
    storm = _find(r"\bstorm(?:y)?\b", t)
    snowy = _find(r"\bsnow\s*storm\b|\bwind\s*storm\b|\bdust\s*storm\b|\bafter\s+the\s+storm\b|\bstorm\s+force\b", t)
    if not th and storm and not snowy:
        r.set("weather.precipitation.rate_mm_h", 18.0, storm.group(0))
        if not r.has("weather.precipitation.type"):
            r.set("weather.precipitation.type", "rain", storm.group(0))
        r.set("weather.wind.speed_mps", units.beaufort_to_mps(6), storm.group(0))
        r.set("weather.wind.gust_factor", 1.8, storm.group(0))
    if not th:
        return
    r.set("weather.thunder.enabled", True, th.group(0))
    if not r.has("weather.precipitation.type") and not _find(r"\bdry\s+(?:thunder|lightning)\b|\bheat\s+lightning\b", t):
        r.set("weather.precipitation.type", "rain", th.group(0))
        r.set("weather.precipitation.rate_mm_h", 15.0, th.group(0))
        r.set("weather.precipitation.drops", "fat", th.group(0))
    # explicit distance or flash-to-bang count
    for m in re.finditer(_NUM + r"\s*" + _DIST_UNIT + r"\b(?!\s*/)", t, flags=re.IGNORECASE):
        if not _find(r"thunder|lightning|storm|strike|bolt", t[max(0, m.start() - 40):m.end() + 40]):
            continue
        unit = m.group(2).lower()
        if unit in ("m", "metre", "metres", "meter", "meters"):
            unit = "m"
        try:
            r.set("weather.thunder.distance_km", units.parse_distance_km(f"{m.group(1)} {unit}", temp_c), m.group(0))
            break
        except ValueError:
            continue
    if not r.has("weather.thunder.distance_km"):
        for m in re.finditer(_NUM + r"\s*(?:s|sec|secs|seconds)\b", t, flags=re.IGNORECASE):
            if _find(r"thunder|lightning|flash|bang|count", t[max(0, m.start() - 40):m.end() + 40]):
                r.set("weather.thunder.distance_km", units.seconds_to_km(units._num(m.group(1)), temp_c),
                      m.group(0) + " flash-to-bang")
                break
    words = [
        (r"\b(?:overhead|right\s+above|directly\s+above|on\s+top\s+of\s+(?:us|me))\b", 0.8),
        (r"\b(?:very\s+close|cracking|crashing|deafening|right\s+here)\b", 1.2),
        (r"\b(?:close|nearby|near)\s+(?:thunder|lightning|strikes?)\b|\bthunder(?:storm)?\s+(?:close|nearby|near)\b|\bclose\s+by\b", 2.0),
        (r"\b(?:very\s+distant|far,?\s+far|barely\s+audible)\b", 20.0),
        (r"\b(?:faint)\s+(?:thunder|rumbl\w*)\b", 18.0),
        (r"\b(?:distant|far(?:[- ]off|\s+away)?|remote)\s+(?:thunder|rumbl\w*|storm|lightning)\b|\bthunder\s+in\s+the\s+distance\b|\brumbl\w*\b", 12.0),
    ]
    for pattern, km in words:
        m = _find(pattern, t)
        if m:
            r.set("weather.thunder.distance_km", km, m.group(0))
            break
    if _find(r"\b(?:approaching|rolling\s+in|coming\s+in|moving\s+in|building|incoming|closing\s+in|getting\s+closer)\b", t):
        r.set("weather.thunder.approach_kmh", 20.0, "approaching")
        r.set("weather.storm_arc", "arriving", "approaching")
    elif _find(r"\b(?:passing|moving\s+away|receding|departing|moving\s+off|dying\s+down|fading)\b", t):
        r.set("weather.thunder.approach_kmh", -20.0, "receding")
        r.set("weather.storm_arc", "passing", "receding")
    m = _find(r"\b(?:frequent|constant|non-?stop|continuous)\s+(?:thunder|lightning|flashes)\b", t)
    if m:
        r.set("weather.thunder.flashes_per_min", 8.0, m.group(0))
    m = _find(r"\b(?:occasional|rare|the\s+odd|sporadic)\s+(?:thunder|lightning|rumbles?|flashes)\b", t)
    if m:
        r.set("weather.thunder.flashes_per_min", 0.6, m.group(0))


_SHELTER_RULES: list[tuple[str, str, dict]] = [
    (r"\b(?:in|inside)\s+(?:a|the|my|our)?\s*(?:car|vehicle|truck|van|taxi|cab)\b|\bcar\s+roof\b|\bwind(?:shield|screen)\b|\bwipers?\b|\bparked\s+car\b", "car", {}),
    (r"\btent\b|\bcamping\b|\bbivvy\b|\bbivouac\b", "tent", {}),
    (r"\bumbrella\b|\bbrolly\b", "umbrella", {}),
    (r"\b(?:tin|metal|corrugated|iron|zinc|steel)\s+roof\b|\battic\b|\bloft\b|\bgarret\b", "attic", {}),
    (r"\bporch\b|\bverandah?\b|\bterrace\b|\bgazebo\b|\bawning\b|\bbalcony\b|\bcovered\s+deck\b|\blanai\b|\bbarn\s+door\b|\bshed\s+door\b", "porch", {}),
    (r"\b(?:open\s+window|window\s+(?:is\s+)?(?:open|ajar|cracked)|ajar|cracked\s+window)\b", "room_open", {"place.window_open": 0.5}),
    (r"\bwindow\w*\b|\bbedroom\b|\bindoors\b|\binside\b|\bin\s+bed\b|\bliving\s+room\b|\bskylight\b|\bconservatory\b|\bcozy\s+room\b|\bcosy\s+room\b", "room_closed", {"place.window_open": 0.0}),
    (r"\bunder\s+(?:a|the|an)?\s*(?:big\s+|old\s+|large\s+)?(?:tree|oak|pine|canopy)\b", "tree", {}),
    (r"\bcave\b|\brock\s+overhang\b", "cave", {}),
    (r"\bunderpass\b|\bunder\s+(?:a|the)\s+bridge\b|\btunnel\b|\bflyover\b", "underpass", {}),
    (r"\bout\s+in\s+(?:it|the\s+rain|the\s+open)\b|\boutside\b|\bin\s+the\s+open\b|\bexposed\b|\bunsheltered\b", "open", {}),
]

_PLACE_RULES: list[tuple[str, str]] = [
    (r"\bjungle\b|\brain\s*forest\b|\btropic(?:s|al\s+forest)\b", "jungle"),
    (r"\b(?:pine|conifer\w*|spruce|fir|taiga|boreal)\b", "conifer_forest"),
    (r"\bforest\b|\bwoods?\b|\bwoodland\b", "deciduous_forest"),
    (r"\bcabin\b|\bhut\b|\bbothy\b|\blodge\b", "cabin"),
    (r"\balley\w*\b|\bbackstreet\b|\blane\s*way\b", "alley"),
    (r"\bcity\b|\bstreet\b|\burban\b|\bdowntown\b|\bpavement\b|\bsidewalk\b|\btokyo\b|\blondon\b|\bnew\s+york\b|\bparis\b", "city_street"),
    (r"\bsuburb\w*\b|\bneighbou?rhood\b|\bhouse\b|\bhome\b", "suburb"),
    (r"\bgarden\b|\bback\s*yard\b|\byard\b|\blawn\b", "garden"),
    (r"\bmeadow\b|\bfield\b|\bprairie\b|\bgrassland\b|\bsteppe\b", "meadow"),
    (r"\bfarm\b|\bbarn\b|\branch\b|\bpaddock\b", "farm"),
    (r"\bbeach\b|\bcoast\w*\b|\bsea\b|\bocean\b|\bharbou?r\b|\bshore\b|\bpier\b|\bcliff\b", "coast"),
    (r"\blake\b|\bpond\b|\briver\b|\bstream\b|\bloch\b|\bjetty\b", "lake"),
    (r"\bmountain\w*\b|\bridge\b|\bhill\w*\b|\bhighland\w*\b|\bmoor\w*\b|\bsummit\b|\bpeak\b", "mountain_ridge"),
    (r"\bdesert\b|\bdunes?\b|\bcanyon\b|\bmesa\b|\barid\b", "desert"),
]

_SURFACE_RULES: list[tuple[str, str, float]] = [
    (r"\bpuddles?\b", "puddle", 20.0),
    (r"\bgravel\b|\bshingle\b|\bpebbles\b", "gravel", 30.0),
    (r"\bleaves\b|\bleafy\b|\bfoliage\b", "leaves", 30.0),
    (r"\b(?:pine\s+)?needles\b", "needles", 30.0),
    (r"\basphalt\b|\btarmac\b|\broad\b|\bpavement\b", "asphalt", 25.0),
    (r"\bconcrete\b|\bsidewalk\b|\bpatio\b|\bstone\s+steps?\b", "concrete", 25.0),
    (r"\btiles?\b|\btiled\b|\bslate\b", "tile", 20.0),
    (r"\bglass\b|\bgreenhouse\b|\bconservatory\b|\bskylight\b", "glass", 20.0),
    (r"\bcorrugated\b|\bmetal\s+sheet\w*\b|\bshed\b|\bdumpsters?\b|\bbins?\b", "metal_sheet", 15.0),
    (r"\bwood(?:en)?\b|\bdeck(?:ing)?\b|\bboardwalk\b|\bplanks?\b", "wood", 20.0),
    (r"\blake\b|\bpond\b|\briver\b|\bstream\b|\bsea\b|\bocean\b|\bwater\b", "water", 30.0),
    (r"\bgrass\w*\b|\blawn\b|\bmeadow\b", "grass", 25.0),
    (r"\bmud\w*\b|\bsoil\b|\bdirt\b|\bearth\b", "soil", 25.0),
    (r"\btarp\w*\b|\bcanvas\b", "canvas", 15.0),
    (r"\bplastic\b|\bcrates?\b", "plastic", 10.0),
    (r"\bbuckets?\b|\bbarrels?\b|\bwatering\s+can\b", "bucket", 5.0),
    (r"\bparked\s+cars?\b|\bcar\s+bonnets?\b|\bcar\s+hoods?\b", "car_body", 10.0),
]

_EMITTER_RULES: list[tuple[str, str, float]] = [
    (r"\bgutters?\b|\bdownspouts?\b|\bdrain\s*pipes?\b|\brain\s*spouts?\b", "gutter", 3.0),
    (r"\beaves?\b|\bdripping\s+(?:roof|edge)\b|\broof\s+drips?\b", "eave", 2.0),
    (r"\b(?:wind\s+)?chimes?\b", "chime", 3.0),
    (r"\bflags?\b|\bbunting\b", "flag", 10.0),
    (r"\bpower\s+lines?\b|\bwires?\b|\bcables?\b|\brigging\b", "wire", 12.0),
]

_TIME_WORDS: list[tuple[str, float]] = [
    (r"\bmidnight\b", 0.0), (r"\bsmall\s+hours\b|\bwee\s+hours\b|\bpre-?dawn\b", 3.5),
    (r"\bdawn\b|\bsunrise\b|\bdaybreak\b|\bfirst\s+light\b", 6.0),
    (r"\bmorning\b", 8.5), (r"\bnoon\b|\bmidday\b", 12.0), (r"\bafternoon\b", 15.0),
    (r"\bdusk\b|\bsunset\b|\btwilight\b|\bgolden\s+hour\b", 20.0), (r"\bevening\b", 19.5),
    (r"\bnight\w*\b|\bovernight\b|\bbedtime\b|\bdark\b", 23.0), (r"\bdaytime\b|\bby\s+day\b", 13.0),
]

_SEASONS = [
    (r"\bspring\w*\b", "spring"), (r"\bsummer\w*\b", "summer"),
    (r"\bautumn\w*\b|\bin\s+(?:the\s+)?fall\b|\bfall\s+(?:rain|storm|evening|night|day|leaves|colou?rs)\b", "autumn"),
    (r"\bwinter\w*\b|\bwintry\b", "winter"),
]
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")

_TEMP_WORDS: list[tuple[str, float, Optional[float]]] = [
    (r"\bsweltering\b|\bscorching\b|\bsultry\b", 33.0, 70.0),
    (r"\bhot\b", 30.0, None), (r"\btropical\b", 27.0, 95.0), (r"\bhumid\b|\bmuggy\b|\bsticky\b|\bclose\s+air\b", 25.0, 95.0),
    (r"\bbalmy\b", 24.0, None), (r"\bwarm\b", 22.0, None), (r"\bmild\b", 14.0, None),
    (r"\bcool\b", 10.0, None), (r"\bchilly\b|\bnippy\b|\braw\b", 6.0, None), (r"\bcold\b", 3.0, None),
    (r"\bfrosty\b|\bicy\b", -2.0, None), (r"\bfreezing\b(?!\s+(?:rain|drizzle|fog))", -3.0, None),
]

_MODE_RULES: list[tuple[str, str]] = [
    (r"\basmr\b|\btingles?\b", "asmr"),
    (r"\bfor\s+sleep\w*\b|\bsleep\w*\b|\bto\s+sleep\b|\bfall(?:ing)?\s+asleep\b|\bnap\b|\binsomni\w*", "sleep"),
    (r"\bfocus\w*\b|\bstudy\w*\b|\bconcentrat\w*\b|\bdeep\s+work\b|\bfor\s+work\b|\bworking\b|\bcoding\b|\breading\b", "focus"),
    (r"\brelax\w*\b|\bunwind\w*\b|\bcalming\b|\bfor\s+calm\b|\bmeditat\w*\b|\bchill\b|\bde-?stress\w*\b", "calm"),
]

_RENDER_RULES: list[tuple[str, str]] = [
    (r"\b7\.1\.4\b|\batmos\b", "7.1.4"), (r"\b5\.1\b|\bsurround\b", "5.1"),
    (r"\bthird[- ]order\b|\bhoa\b", "hoa3"), (r"\bfirst[- ]order\b|\bfoa\b|\bambisonics?\b|\b360\b|\bvr\b", "foa"),
    (r"\bbinaural\b|\bheadphones?\b|\bearbuds?\b", "binaural"),
]


def _shelter(t: str, r: ParseResult) -> None:
    for pattern, shelter, extras in _SHELTER_RULES:
        m = _find(pattern, t)
        if m:
            r.set("place.shelter", shelter, m.group(0))
            for path, val in extras.items():
                r.set(path, val, m.group(0))
            if shelter == "car":
                w = _find(r"\b(?:fast|rapid|high[- ]speed)\s+wipers?\b|\bwipers?\s+(?:on\s+)?(?:fast|high)\b", t)
                if w:
                    r.set("place.wipers", "fast", w.group(0))
                elif _find(r"\bwipers?\b", t) and not _find(r"\bwipers?\s+off\b|\bno\s+wipers?\b", t):
                    r.set("place.wipers", "slow", "wipers")
            if shelter == "tent" and _find(r"\blying\b|\bsleeping\s+bag\b|\bin\s+bed\b|\bcamp\s*bed\b", t):
                r.set("listener.height_m", 0.4, "lying down")
            break


def _place(t: str, r: ParseResult) -> None:
    for pattern, name in _PLACE_RULES:
        m = _find(pattern, t)
        if m:
            r.place = name
            r.matched.append((m.group(0), f"place = {name}"))
            break
    for pattern, mat, weight in _SURFACE_RULES:
        m = _find(pattern, t)
        if m:
            r.surfaces_add[mat] = r.surfaces_add.get(mat, 0.0) + weight
            r.matched.append((m.group(0), f"surface + {mat}"))
    for pattern, kind, dist in _EMITTER_RULES:
        m = _find(pattern, t)
        if m:
            r.emitters_add.append({"type": kind, "distance_m": dist})
            r.matched.append((m.group(0), f"emitter {kind}"))
    if _find(r"\bbare\s+(?:trees|branches)\b|\bleafless\b", t):
        r.set("weather.season", "winter", "bare trees")
    m = _find(r"\b(?:lots\s+of|many|tall)\s+trees\b", t)
    if m:
        r.set("place.vegetation.deciduous", 12, m.group(0))


def _time_season_temp(t: str, r: ParseResult, base: Scene) -> None:
    m = _find(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", t)
    if m:
        r.set("weather.hour", units.parse_hour(m.group(0)), m.group(0))
    else:
        m = _find(r"\b(?:at\s+)([01]?\d|2[0-3]):([0-5]\d)\b", t)
        if m:
            r.set("weather.hour", int(m.group(1)) + int(m.group(2)) / 60.0, m.group(0))
    if not r.has("weather.hour"):
        for pattern, hour in _TIME_WORDS:
            m = _find(pattern, t)
            if m:
                r.set("weather.hour", hour, m.group(0))
                break
    for pattern, season in _SEASONS:
        m = _find(pattern, t)
        if m:
            r.set("weather.season", season, m.group(0))
            break
    if not r.has("weather.season"):
        for i, month in enumerate(_MONTHS):
            m = _find(r"\b" + month + r"\b", t)
            if m:
                north = ("winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer",
                         "autumn", "autumn", "autumn", "winter")[i]
                if base.weather.latitude_deg < 0:
                    north = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}[north]
                r.set("weather.season", north, m.group(0))
                break
    m = _find(r"(-?\d+(?:[.,]\d+)?)\s*(?:°|º|deg(?:rees?)?)?\s*(c|f|celsius|fahrenheit|centigrade)\b", t)
    if m and not _find(r"\b(?:force|bft|f)\s*$", t[:m.start()]):
        r.set("weather.temperature_c", units.parse_temperature_c(f"{m.group(1)} {m.group(2)}"), m.group(0), force=True)
    else:
        for pattern, temp, rh in _TEMP_WORDS:
            m = _find(pattern, t)
            if m:
                r.set("weather.temperature_c", temp, m.group(0))
                if rh is not None:
                    r.set("weather.humidity_pct", rh, m.group(0))
                break
    m = _find(r"\b(?:humid|muggy|sticky|damp|clammy)\b", t)
    if m:
        r.set("weather.humidity_pct", 95.0, m.group(0))
    m = _find(r"\bdry\s+air\b|\barid\b", t)
    if m:
        r.set("weather.humidity_pct", 35.0, m.group(0))


def _mode_render(t: str, r: ParseResult) -> None:
    for pattern, mode in _MODE_RULES:
        m = _find(pattern, t)
        if m:
            r.set("mode", mode, m.group(0))
            break
    for pattern, layout in _RENDER_RULES:
        m = _find(pattern, t)
        if m:
            r.set("listener.render", layout, m.group(0))
            break
    if r.assignments.get("mode") == "asmr":
        r.set("listener.render", "binaural", "asmr")


def parse_text(text: str, base: Optional[Scene] = None) -> ParseResult:
    """Recognise known weather vocabulary in `text` (deterministic, no network)."""
    base = base if base is not None else Scene()
    r = ParseResult(text=text)
    t = " " + re.sub(r"\s+", " ", text.strip()) + " "
    _metar_groups(text, r)
    _explicit_rate(t, r)
    _precip(t, r, base)
    _wind(t, r)
    _time_season_temp(t, r, base)
    temp = r.assignments.get("weather.temperature_c", base.weather.temperature_c)
    _thunder(t, r, float(temp))
    _shelter(t, r)
    _place(t, r)
    _mode_render(t, r)
    # phase sanity: rain words below freezing become snow unless explicitly freezing rain
    ptype = r.assignments.get("weather.precipitation.type")
    if ptype in ("rain", "showers", "drizzle") and r.has("weather.temperature_c") and r.assignments["weather.temperature_c"] <= -2.0:
        r.set("weather.precipitation.type", "snow", "below freezing", force=True)
        rate = r.assignments.get("weather.precipitation.rate_mm_h", base.weather.precipitation.rate_mm_h)
        r.set("weather.precipitation.rate_mm_h", min(5.0, float(rate) * 0.3), "snow water equivalent", force=True)
    return r


def _set(scene: Scene, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = scene
    for part in parts[:-1]:
        obj = obj.setdefault(part, {}) if isinstance(obj, dict) else getattr(obj, part)
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    else:
        setattr(obj, parts[-1], value)


def apply_parse(result: ParseResult, base: Optional[Scene] = None) -> Scene:
    """Apply a ParseResult on top of `base` (copy)."""
    scene = base.copy() if base is not None else Scene()
    if result.place:
        scene = apply_place(scene, result.place, keep_shelter=True)
    if result.surfaces_add:
        surf = dict(scene.place.surfaces)
        for mat, w in result.surfaces_add.items():
            surf[mat] = surf.get(mat, 0.0) + w
        total = sum(v for v in surf.values() if v > 0) or 1.0
        scene.place.surfaces = {k: round(100.0 * v / total, 2) for k, v in surf.items() if v > 0}
    if result.emitters_add:
        have = {e.get("type") for e in scene.place.emitters}
        for e in result.emitters_add:
            if e["type"] not in have:
                scene.place.emitters.append(dict(e))
                have.add(e["type"])
    for path, value in result.assignments.items():
        _set(scene, path, value)
    p = scene.weather.precipitation
    if p.type == "drizzle":
        p.rate_mm_h = min(p.rate_mm_h, 1.0)
    if p.type == "snow" and "weather.temperature_c" not in result.assignments and scene.weather.temperature_c > 1.0:
        scene.weather.temperature_c = -1.0
    if p.type == "showers" and p.character == "continuous" and "weather.precipitation.character" not in result.assignments:
        p.character = "showery"
    if scene.place.shelter == "car" and scene.listener.height_m > 1.4 and "listener.height_m" not in result.assignments:
        scene.listener.height_m = 1.2
    if scene.place.shelter == "underpass" and "place.preset" not in result.assignments and not result.place:
        scene = apply_place(scene, "city_street", keep_shelter=True)
    if scene.mode and scene.mode != "none":
        from ..wellness.modes import apply_mode
        scene = apply_mode(scene, scene.mode)
    name = re.sub(r"\s+", " ", result.text.strip())
    scene.name = (name[:1].upper() + name[1:])[:96] if name else scene.name
    return scene


def scene_from_text(text: str, base: Optional[Scene] = None) -> Scene:
    """'Describe your weather' -> Scene. A pasted METAR is parsed as a METAR."""
    if looks_like_metar(text):
        return scene_from_metar(text, base)
    return apply_parse(parse_text(text, base), base)
