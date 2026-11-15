"""Scene presets: the Tier-0 scene cards (MASTER_PLAN 2.3-2.4) and place templates (2.6).

Each preset is a complete, physically sensible Scene with a title, tags in four
families (weather:, place:, mood:, use: sleep/focus/film/game/asmr/music) and a
one-line description. Places carry surface mixes, vegetation, emitters, ground
type, roughness length z0 (Davenport/WMO roughness classes: open water 0.0002 m,
grass 0.03 m, crops/garden 0.1 m, suburbs 0.5 m, forest/city 1.0+ m) and
region-correct biophony. Presets that name a wellness mode pass through
`raingen.wellness.modes.apply_mode` so their guard-rails are always honoured.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from .schema import Scene

DEFAULT_PRESET = "steady-rain-leaves-night"

# ---------------------------------------------------------------------------
# Place templates
# ---------------------------------------------------------------------------
PLACE_TEMPLATES: dict[str, dict] = {
    "garden": {
        "preset": "garden", "ground": "grass", "z0_m": 0.1,
        "surfaces": {"grass": 40.0, "soil": 20.0, "leaves": 20.0, "puddle": 10.0, "concrete": 10.0},
        "vegetation": {"deciduous": 3, "conifer": 0, "distance_m": 12.0},
        "emitters": [{"type": "gutter", "distance_m": 4.0}, {"type": "eave", "distance_m": 2.5}],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.05, "region": "temperate_europe"},
    },
    "deciduous_forest": {
        "preset": "deciduous_forest", "ground": "forest_floor", "z0_m": 1.0,
        "surfaces": {"leaves": 55.0, "soil": 25.0, "puddle": 10.0, "grass": 10.0},
        "vegetation": {"deciduous": 14, "conifer": 2, "distance_m": 6.0},
        "emitters": [],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0, "region": "temperate_europe"},
    },
    "conifer_forest": {
        "preset": "conifer_forest", "ground": "forest_floor", "z0_m": 1.2,
        "surfaces": {"needles": 55.0, "soil": 25.0, "puddle": 8.0, "grass": 7.0, "leaves": 5.0},
        "vegetation": {"deciduous": 1, "conifer": 16, "distance_m": 6.0},
        "emitters": [],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "off", "traffic": 0.0, "region": "boreal"},
    },
    "meadow": {
        "preset": "meadow", "ground": "grass", "z0_m": 0.03,
        "surfaces": {"grass": 75.0, "soil": 15.0, "puddle": 10.0},
        "vegetation": {"deciduous": 1, "conifer": 0, "distance_m": 40.0},
        "emitters": [],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0, "region": "temperate_europe"},
    },
    "coast": {
        "preset": "coast", "ground": "water", "z0_m": 0.005,
        "surfaces": {"water": 40.0, "gravel": 25.0, "grass": 15.0, "concrete": 10.0, "puddle": 10.0},
        "vegetation": {"deciduous": 0, "conifer": 0, "distance_m": 50.0},
        "emitters": [{"type": "wire", "distance_m": 15.0}, {"type": "flag", "distance_m": 20.0}],
        "biophony": {"birds": "auto", "insects": "off", "frogs": "off", "traffic": 0.0, "region": "temperate_europe"},
    },
    "lake": {
        "preset": "lake", "ground": "water", "z0_m": 0.002,
        "surfaces": {"water": 45.0, "grass": 25.0, "soil": 15.0, "leaves": 10.0, "puddle": 5.0},
        "vegetation": {"deciduous": 4, "conifer": 2, "distance_m": 15.0},
        "emitters": [],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0, "region": "temperate_europe"},
    },
    "city_street": {
        "preset": "city_street", "ground": "asphalt", "z0_m": 1.0,
        "surfaces": {"asphalt": 45.0, "concrete": 30.0, "puddle": 15.0, "metal_sheet": 5.0, "glass": 5.0},
        "vegetation": {"deciduous": 2, "conifer": 0, "distance_m": 15.0},
        "emitters": [{"type": "gutter", "distance_m": 3.0}, {"type": "eave", "distance_m": 2.0},
                     {"type": "gutter", "distance_m": 7.0}],
        "biophony": {"birds": "auto", "insects": "off", "frogs": "off", "traffic": 0.55, "region": "temperate_europe"},
    },
    "alley": {
        "preset": "alley", "ground": "asphalt", "z0_m": 1.0,
        "surfaces": {"concrete": 40.0, "asphalt": 25.0, "puddle": 15.0, "metal_sheet": 10.0, "plastic": 5.0, "wood": 5.0},
        "vegetation": {"deciduous": 0, "conifer": 0, "distance_m": 30.0},
        "emitters": [{"type": "gutter", "distance_m": 1.5}, {"type": "eave", "distance_m": 1.2},
                     {"type": "gutter", "distance_m": 3.5}],
        "biophony": {"birds": "off", "insects": "off", "frogs": "off", "traffic": 0.25, "region": "temperate_europe"},
    },
    "suburb": {
        "preset": "suburb", "ground": "grass", "z0_m": 0.5,
        "surfaces": {"grass": 30.0, "asphalt": 20.0, "concrete": 20.0, "puddle": 10.0, "leaves": 10.0, "tile": 10.0},
        "vegetation": {"deciduous": 4, "conifer": 1, "distance_m": 15.0},
        "emitters": [{"type": "gutter", "distance_m": 5.0}, {"type": "eave", "distance_m": 3.0}],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "off", "traffic": 0.2, "region": "temperate_europe"},
    },
    "mountain_ridge": {
        "preset": "mountain_ridge", "ground": "soil", "z0_m": 0.03,
        "surfaces": {"gravel": 40.0, "grass": 30.0, "soil": 20.0, "puddle": 5.0, "snow": 5.0},
        "vegetation": {"deciduous": 0, "conifer": 3, "distance_m": 30.0},
        "emitters": [{"type": "wire", "distance_m": 25.0}],
        "biophony": {"birds": "auto", "insects": "off", "frogs": "off", "traffic": 0.0, "region": "temperate_europe"},
    },
    "desert": {
        "preset": "desert", "ground": "soil", "z0_m": 0.01,
        "surfaces": {"soil": 60.0, "gravel": 30.0, "concrete": 5.0, "wood": 5.0},
        "vegetation": {"deciduous": 0, "conifer": 0, "distance_m": 60.0},
        "emitters": [],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "off", "traffic": 0.0, "region": "arid"},
    },
    "jungle": {
        "preset": "jungle", "ground": "forest_floor", "z0_m": 1.5,
        "surfaces": {"leaves": 60.0, "soil": 20.0, "puddle": 15.0, "water": 5.0},
        "vegetation": {"deciduous": 25, "conifer": 0, "distance_m": 4.0},
        "emitters": [],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0, "region": "tropical"},
    },
    "farm": {
        "preset": "farm", "ground": "soil", "z0_m": 0.1,
        "surfaces": {"grass": 35.0, "soil": 30.0, "puddle": 15.0, "metal_sheet": 10.0, "wood": 10.0},
        "vegetation": {"deciduous": 3, "conifer": 0, "distance_m": 20.0},
        "emitters": [{"type": "gutter", "distance_m": 6.0}, {"type": "wire", "distance_m": 10.0}],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.05, "region": "temperate_europe"},
    },
    "cabin": {
        "preset": "cabin", "ground": "forest_floor", "z0_m": 0.8,
        "surfaces": {"wood": 30.0, "needles": 25.0, "soil": 25.0, "puddle": 10.0, "grass": 10.0},
        "vegetation": {"deciduous": 2, "conifer": 8, "distance_m": 10.0},
        "emitters": [{"type": "gutter", "distance_m": 3.0}, {"type": "eave", "distance_m": 2.0}],
        "biophony": {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0, "region": "north_america"},
    },
}

PLACE_ALIASES = {
    "forest": "deciduous_forest", "woods": "deciduous_forest", "woodland": "deciduous_forest",
    "pine_forest": "conifer_forest", "conifer": "conifer_forest", "taiga": "conifer_forest",
    "field": "meadow", "beach": "coast", "sea": "coast", "harbour": "coast", "harbor": "coast",
    "city": "city_street", "street": "city_street", "urban": "city_street", "ridge": "mountain_ridge",
    "mountain": "mountain_ridge", "rainforest": "jungle", "backyard": "garden",
}


def place_template(name: str) -> dict:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = PLACE_ALIASES.get(key, key)
    if key not in PLACE_TEMPLATES:
        raise KeyError(f"unknown place {name!r}; known: {sorted(PLACE_TEMPLATES)}")
    return copy.deepcopy(PLACE_TEMPLATES[key])


def apply_place(scene: Scene, name: str, keep_shelter: bool = True) -> Scene:
    """Return a copy of `scene` with the place template `name` applied."""
    tpl = place_template(name)
    out = scene.copy()
    p = out.place
    p.preset = tpl["preset"]
    p.ground = tpl["ground"]
    p.z0_m = tpl["z0_m"]
    p.surfaces = dict(tpl["surfaces"])
    p.vegetation = dict(tpl["vegetation"])
    p.emitters = [dict(e) for e in tpl["emitters"]]
    p.biophony = dict(tpl["biophony"])
    if not keep_shelter:
        p.shelter = "open"
    return out


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("surfaces", "vegetation", "biophony"):
            out[k] = _deep_merge(out[k], v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = copy.deepcopy(v)
    return out


PRESETS: dict[str, dict] = {}


def _define(name: str, title: str, tags: list[str], description: str, place: str, scene: dict,
            surfaces: Optional[dict] = None) -> None:
    """Register a preset. `scene` is a partial to_dict()-style override; `surfaces`
    replaces the template's surface mix when given."""
    PRESETS[name] = {"name": name, "title": title, "tags": tags, "description": description,
                     "place": place, "scene": scene, "surfaces": surfaces}


_define(
    "steady-rain-leaves-night", "Steady rain on leaves, night",
    ["weather:rain", "place:forest", "mood:cozy", "use:sleep", "use:focus", "use:film"],
    "Moderate stratiform rain on a broadleaf canopy and leaf litter, light breeze, no thunder. The default.",
    "deciduous_forest",
    {"weather": {"precipitation": {"type": "rain", "rate_mm_h": 4.0, "drops": "normal", "character": "continuous"},
                 "wind": {"speed_mps": 2.4, "from_deg": 240.0},
                 "temperature_c": 11.0, "humidity_pct": 95.0, "hour": 23.0, "season": "autumn", "cloud_okta": 8.0}},
)
_define(
    "tin-roof-cabin-storm", "Tin-roof cabin in a storm",
    ["weather:thunderstorm", "place:cabin", "mood:dramatic", "mood:cozy", "use:film", "use:game"],
    "Inside a cabin under a corrugated tin roof: heavy continental rain drums overhead, thunder 3 km out, gusty wind in the pines.",
    "cabin",
    {"place": {"shelter": "attic"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 22.0, "drops": "fat", "character": "showery", "shower_period_min": 14.0},
                 "wind": {"speed_mps": 9.0, "from_deg": 250.0, "gust_factor": 1.7},
                 "thunder": {"enabled": True, "distance_km": 3.0, "bearing_deg": 260.0, "approach_kmh": 10.0, "flashes_per_min": 3.0,
                             "stroke_intensity": 0.7, "cg_fraction": 0.5},
                 "temperature_c": 17.0, "humidity_pct": 96.0, "hour": 21.5, "season": "summer"}},
)
_define(
    "tent-in-a-storm", "Tent in a storm",
    ["weather:thunderstorm", "place:forest", "mood:adventurous", "use:asmr", "use:film", "use:sleep"],
    "Lying in a nylon tent among conifers: fabric patter over your head, gusts shaking the flysheet, thunder rolling 6 km away.",
    "conifer_forest",
    {"place": {"shelter": "tent"},
     "listener": {"height_m": 0.4, "render": "binaural"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 9.0, "drops": "normal", "character": "showery", "shower_period_min": 10.0},
                 "wind": {"speed_mps": 10.0, "from_deg": 270.0, "gust_factor": 1.8},
                 "thunder": {"enabled": True, "distance_km": 6.0, "bearing_deg": 280.0, "flashes_per_min": 1.5},
                 "temperature_c": 9.0, "humidity_pct": 97.0, "hour": 1.0, "season": "summer"}},
)
_define(
    "car-in-a-thunderstorm", "Car in a thunderstorm",
    ["weather:thunderstorm", "place:city", "mood:cozy", "use:film", "use:asmr"],
    "Parked car, heavy rain hammering the thin steel roof and windshield, slow wipers, wet-tyre hiss passing by, thunder 4 km away.",
    "city_street",
    {"place": {"shelter": "car", "wipers": "slow"},
     "listener": {"height_m": 1.2},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 25.0, "drops": "fat", "character": "showery", "shower_period_min": 12.0},
                 "wind": {"speed_mps": 6.0, "from_deg": 230.0},
                 "thunder": {"enabled": True, "distance_km": 4.0, "bearing_deg": 210.0, "flashes_per_min": 2.5},
                 "temperature_c": 19.0, "humidity_pct": 95.0, "hour": 20.5, "season": "summer"}},
)
_define(
    "umbrella-city-street", "Umbrella on a city street",
    ["weather:rain", "place:city", "mood:urban", "use:film", "use:game", "use:asmr"],
    "Standing under an umbrella on a wet pavement: taut membrane patter overhead, gutters and downspouts, distant traffic.",
    "city_street",
    {"place": {"shelter": "umbrella"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 6.0, "drops": "normal"},
                 "wind": {"speed_mps": 3.0, "from_deg": 225.0},
                 "temperature_c": 12.0, "humidity_pct": 92.0, "hour": 18.5, "season": "autumn"}},
)
_define(
    "porch-summer-thunderstorm", "Porch, summer thunderstorm",
    ["weather:thunderstorm", "place:suburb", "mood:cozy", "use:sleep", "use:music", "use:film"],
    "From a covered porch: a warm-evening thunderstorm rolls in, downpour on the lawn and the roof edge, frogs before and after.",
    "suburb",
    {"place": {"shelter": "porch", "surfaces": {"wood": 15.0}},
     "weather": {"precipitation": {"type": "showers", "rate_mm_h": 18.0, "drops": "fat", "character": "showery", "shower_period_min": 18.0},
                 "wind": {"speed_mps": 5.0, "from_deg": 260.0, "gust_factor": 1.6},
                 "thunder": {"enabled": True, "distance_km": 7.0, "bearing_deg": 250.0, "approach_kmh": 15.0, "flashes_per_min": 2.0},
                 "temperature_c": 24.0, "humidity_pct": 90.0, "hour": 20.0, "season": "summer", "storm_arc": "arriving", "arc_minutes": 25.0}},
)
_define(
    "tokyo-night-alley", "Tokyo night alley",
    ["weather:rain", "place:alley", "mood:urban", "mood:melancholy", "use:film", "use:game", "use:music"],
    "Narrow alley at night: rain on corrugated awnings, plastic crates and puddles, a close downspout, the city hum beyond.",
    "alley",
    {"place": {"biophony": {"region": "east_asia", "traffic": 0.35}},
     "listener": {"height_m": 1.65},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 7.0, "drops": "normal"},
                 "wind": {"speed_mps": 1.5, "from_deg": 180.0},
                 "temperature_c": 16.0, "humidity_pct": 94.0, "hour": 23.5, "season": "spring", "latitude_deg": 35.7}},
    surfaces={"concrete": 30.0, "asphalt": 20.0, "puddle": 20.0, "metal_sheet": 15.0, "plastic": 10.0, "glass": 5.0},
)
_define(
    "tropical-monsoon-veranda", "Tropical monsoon veranda",
    ["weather:monsoon", "place:jungle", "mood:lush", "use:sleep", "use:focus", "use:music"],
    "Wooden veranda at the edge of a rainforest: torrential maritime rain of countless mid-size drops, frogs and insects, warm air.",
    "jungle",
    {"place": {"shelter": "porch", "surfaces": {"wood": 15.0}},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 45.0, "drops": "fine", "character": "showery", "shower_period_min": 25.0},
                 "wind": {"speed_mps": 3.0, "from_deg": 200.0},
                 "temperature_c": 27.0, "humidity_pct": 98.0, "hour": 16.0, "season": "summer", "latitude_deg": 12.0}},
)
_define(
    "scottish-highland-drizzle", "Scottish highland drizzle",
    ["weather:drizzle", "place:mountain", "mood:melancholy", "use:focus", "use:film"],
    "Fine drizzle drifting across open moorland, a fresh westerly over the heather, no shelter, nothing but the hill.",
    "mountain_ridge",
    {"weather": {"precipitation": {"type": "drizzle", "rate_mm_h": 0.3, "drops": "fine"},
                 "wind": {"speed_mps": 8.0, "from_deg": 260.0},
                 "temperature_c": 8.0, "humidity_pct": 99.0, "hour": 14.0, "season": "autumn", "latitude_deg": 57.0,
                 "visibility_km": 1.5}},
    surfaces={"grass": 55.0, "soil": 20.0, "gravel": 15.0, "puddle": 10.0},
)
_define(
    "conifer-forest-gale", "Conifer forest in a gale",
    ["weather:gale", "place:forest", "mood:wild", "use:film", "use:game"],
    "Force 8 through spruce and pine: needle whoosh swelling with each gust front, creaks, rain curtains sweeping through.",
    "conifer_forest",
    {"weather": {"precipitation": {"type": "rain", "rate_mm_h": 5.0, "drops": "normal", "character": "intermittent"},
                 "wind": {"speed_mps": 18.5, "from_deg": 290.0, "gust_factor": 2.0, "gust_rate_per_min": 3.0},
                 "temperature_c": 6.0, "humidity_pct": 90.0, "hour": 15.0, "season": "winter"}},
)
_define(
    "desert-flash-storm", "Desert flash storm",
    ["weather:thunderstorm", "place:desert", "mood:dramatic", "use:film", "use:game"],
    "Sudden convective cell over hot desert: fat drops smack dry ground and gravel, a gust front ahead, thunder close.",
    "desert",
    {"weather": {"precipitation": {"type": "showers", "rate_mm_h": 30.0, "drops": "fat", "character": "showery", "shower_period_min": 8.0},
                 "wind": {"speed_mps": 12.0, "from_deg": 200.0, "gust_factor": 2.0},
                 "thunder": {"enabled": True, "distance_km": 2.5, "bearing_deg": 210.0, "flashes_per_min": 4.0, "stroke_intensity": 0.8, "cg_fraction": 0.6},
                 "temperature_c": 31.0, "humidity_pct": 40.0, "hour": 17.0, "season": "summer", "latitude_deg": 33.0,
                 "storm_arc": "arriving", "arc_minutes": 10.0}},
)
_define(
    "coastal-squall", "Coastal squall",
    ["weather:squall", "place:coast", "mood:wild", "use:film", "use:game"],
    "A squall line comes ashore: rain on shingle and the sea surface, rigging wires singing, gusts to gale force.",
    "coast",
    {"weather": {"precipitation": {"type": "showers", "rate_mm_h": 15.0, "drops": "normal", "character": "showery", "shower_period_min": 6.0},
                 "wind": {"speed_mps": 15.0, "from_deg": 240.0, "gust_factor": 1.9, "gust_rate_per_min": 4.0},
                 "thunder": {"enabled": True, "distance_km": 9.0, "bearing_deg": 240.0, "flashes_per_min": 1.0},
                 "temperature_c": 13.0, "humidity_pct": 93.0, "hour": 11.0, "season": "autumn"}},
)
_define(
    "attic-tin-roof", "Attic under a tin roof",
    ["weather:rain", "place:attic", "mood:cozy", "use:sleep", "use:asmr", "use:music"],
    "In the attic, a metre and a half below corrugated tin: every fat drop pings, steady rain becomes a warm metallic roar.",
    "suburb",
    {"place": {"shelter": "attic"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 8.0, "drops": "normal"},
                 "wind": {"speed_mps": 2.0, "from_deg": 250.0},
                 "temperature_c": 14.0, "humidity_pct": 90.0, "hour": 22.0, "season": "autumn"}},
)
_define(
    "window-rain-bedroom", "Rain on the bedroom window",
    ["weather:rain", "place:room", "mood:cozy", "use:sleep"],
    "Tucked in bed, window shut: soft rain on the glass and the roof above, the garden muffled outside. Sleep mode, timer on.",
    "garden",
    {"place": {"shelter": "room_closed", "window_open": 0.0},
     "mode": "sleep",
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 4.0, "drops": "normal", "character": "continuous"},
                 "wind": {"speed_mps": 2.0, "from_deg": 250.0},
                 "temperature_c": 10.0, "humidity_pct": 95.0, "hour": 23.5, "season": "autumn"}},
)
_define(
    "focus-steady-roof", "Focus: steady rain on the roof",
    ["weather:rain", "place:room", "mood:steady", "use:focus"],
    "Dense, unchanging rain on a tiled roof heard from indoors. No birds, no plinks, no rhythm: just masking. Focus mode.",
    "suburb",
    {"place": {"shelter": "room_closed", "emitters": []},
     "mode": "focus",
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 8.0, "drops": "fine", "character": "continuous"},
                 "wind": {"speed_mps": 2.5, "from_deg": 250.0, "gust_factor": 1.15},
                 "temperature_c": 13.0, "humidity_pct": 90.0, "hour": 14.0, "season": "autumn"}},
)
_define(
    "asmr-umbrella-close", "ASMR: umbrella, close",
    ["weather:rain", "place:garden", "mood:intimate", "use:asmr"],
    "Binaural, a hand's breadth from a taut umbrella: crisp patter, window taps, drips into a bucket. Every trigger is removable.",
    "garden",
    {"place": {"shelter": "umbrella"},
     "mode": "asmr",
     "listener": {"render": "binaural"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 3.0, "drops": "normal"},
                 "wind": {"speed_mps": 1.0, "from_deg": 240.0},
                 "temperature_c": 15.0, "humidity_pct": 90.0, "hour": 21.0, "season": "spring"}},
)
_define(
    "after-the-storm-drips", "After the storm: drips",
    ["weather:aftermath", "place:forest", "mood:peaceful", "use:sleep", "use:focus", "use:film"],
    "The rain has stopped. The canopy keeps releasing big drips, gutters run down, the first bird returns.",
    "deciduous_forest",
    {"place": {"aftermath_tau_min": 15.0, "emitters": [{"type": "gutter", "distance_m": 6.0}, {"type": "eave", "distance_m": 3.0}]},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 2.0, "drops": "normal"},
                 "wind": {"speed_mps": 1.2, "from_deg": 300.0},
                 "temperature_c": 14.0, "humidity_pct": 98.0, "hour": 7.0, "season": "spring",
                 "storm_arc": "clearing", "arc_minutes": 10.0, "cloud_okta": 5.0}},
)
_define(
    "first-fat-drops", "First fat drops",
    ["weather:arriving", "place:garden", "mood:anticipation", "use:film", "use:game"],
    "The leading edge of a cell: sparse, heavy drops smack the patio and leaves before the downpour, distant thunder approaching.",
    "garden",
    {"weather": {"precipitation": {"type": "showers", "rate_mm_h": 12.0, "drops": "fat", "character": "showery"},
                 "wind": {"speed_mps": 6.0, "from_deg": 250.0, "gust_factor": 1.8},
                 "thunder": {"enabled": True, "distance_km": 12.0, "bearing_deg": 250.0, "approach_kmh": 25.0, "flashes_per_min": 1.0},
                 "temperature_c": 23.0, "humidity_pct": 80.0, "hour": 17.5, "season": "summer",
                 "storm_arc": "arriving", "arc_minutes": 15.0}},
)
_define(
    "hailstorm-car", "Hailstorm on a car",
    ["weather:hail", "place:city", "mood:intense", "use:film", "use:game"],
    "Inside a car while marble-size hail hammers the roof and glass: hard Hertzian impacts, bounces, rain between bursts.",
    "city_street",
    {"place": {"shelter": "car"},
     "listener": {"height_m": 1.2},
     "weather": {"precipitation": {"type": "hail", "rate_mm_h": 30.0, "hail_size_mm": 15.0, "character": "showery", "shower_period_min": 6.0},
                 "wind": {"speed_mps": 9.0, "from_deg": 240.0, "gust_factor": 1.9},
                 "thunder": {"enabled": True, "distance_km": 3.0, "bearing_deg": 240.0, "flashes_per_min": 3.0},
                 "temperature_c": 16.0, "humidity_pct": 85.0, "hour": 16.0, "season": "summer"}},
)
_define(
    "sleet-city", "Sleet in the city",
    ["weather:sleet", "place:city", "mood:bleak", "use:film", "use:game"],
    "Wet sleet ticking on parked cars, awnings and slush, raw wind between buildings, traffic hiss on wet roads.",
    "city_street",
    {"weather": {"precipitation": {"type": "sleet", "rate_mm_h": 2.5, "drops": "normal"},
                 "wind": {"speed_mps": 7.0, "from_deg": 20.0, "gust_factor": 1.7},
                 "temperature_c": 0.5, "humidity_pct": 95.0, "hour": 17.0, "season": "winter"}},
    surfaces={"asphalt": 35.0, "concrete": 25.0, "puddle": 15.0, "snow": 10.0, "car_body": 10.0, "metal_sheet": 5.0},
)
_define(
    "snow-hush-forest", "Snow hush in the forest",
    ["weather:snow", "place:forest", "mood:peaceful", "use:sleep", "use:focus"],
    "Fresh snow on a spruce forest floor: the world goes quiet, the porous cover swallows distant sound, flakes whisper on needles.",
    "conifer_forest",
    {"place": {"ground": "snow_fresh"},
     "weather": {"precipitation": {"type": "snow", "rate_mm_h": 1.0, "drops": "fine"},
                 "wind": {"speed_mps": 1.0, "from_deg": 330.0},
                 "temperature_c": -4.0, "humidity_pct": 90.0, "hour": 22.0, "season": "winter", "latitude_deg": 60.0}},
    surfaces={"snow": 75.0, "needles": 15.0, "soil": 10.0},
)
_define(
    "freezing-rain-ice", "Freezing rain and ice",
    ["weather:freezing_rain", "place:forest", "mood:eerie", "use:film", "use:game"],
    "Supercooled rain glazing every twig: brittle clicking on iced branches, occasional snaps, a glassy hiss on the ground.",
    "deciduous_forest",
    {"weather": {"precipitation": {"type": "freezing_rain", "rate_mm_h": 2.0, "drops": "normal"},
                 "wind": {"speed_mps": 4.0, "from_deg": 60.0},
                 "temperature_c": -1.0, "humidity_pct": 97.0, "hour": 6.0, "season": "winter"}},
    surfaces={"leaves": 30.0, "soil": 20.0, "snow": 20.0, "puddle": 10.0, "concrete": 20.0},
)
_define(
    "wind-only-ridge", "Wind on the ridge",
    ["weather:wind", "place:mountain", "mood:vast", "use:film", "use:game", "use:music"],
    "No rain at all: a strong westerly over an exposed ridge, wires moaning, gusts arriving from far upwind.",
    "mountain_ridge",
    {"weather": {"precipitation": {"type": "none", "rate_mm_h": 0.0},
                 "wind": {"speed_mps": 14.0, "from_deg": 270.0, "gust_factor": 1.6, "gust_rate_per_min": 2.0},
                 "temperature_c": 5.0, "humidity_pct": 60.0, "hour": 13.0, "season": "autumn", "cloud_okta": 4.0}},
)
_define(
    "cricket-warm-night-rain", "Crickets and warm night rain",
    ["weather:rain", "place:meadow", "mood:peaceful", "use:sleep", "use:music"],
    "A warm summer night: light rain on grass, crickets chirping at the Dolbear rate for 22 °C, frogs from a ditch.",
    "meadow",
    {"weather": {"precipitation": {"type": "rain", "rate_mm_h": 1.2, "drops": "normal"},
                 "wind": {"speed_mps": 1.0, "from_deg": 200.0},
                 "temperature_c": 22.0, "humidity_pct": 92.0, "hour": 23.0, "season": "summer"},
     "place": {"biophony": {"insects": "on", "frogs": "on"}}},
)
_define(
    "lake-rain-plinks", "Rain on a still lake",
    ["weather:rain", "place:lake", "mood:peaceful", "use:sleep", "use:music", "use:asmr"],
    "Light rain on calm water: small drops ring at 15 kHz, large drops plink at 2-10 kHz, from a jetty at the shore.",
    "lake",
    {"weather": {"precipitation": {"type": "rain", "rate_mm_h": 2.0, "drops": "normal", "water_quality": 1.0},
                 "wind": {"speed_mps": 0.8, "from_deg": 220.0},
                 "temperature_c": 15.0, "humidity_pct": 95.0, "hour": 20.0, "season": "summer"}},
    surfaces={"water": 70.0, "wood": 10.0, "grass": 10.0, "leaves": 10.0},
)
_define(
    "farm-barn-downpour", "Barn door downpour",
    ["weather:rain", "place:farm", "mood:rustic", "use:film", "use:game"],
    "Sheltering in an open barn door: downpour on metal sheeting and the yard, a full gutter overflowing, wind in the wires.",
    "farm",
    {"place": {"shelter": "porch"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 20.0, "drops": "normal"},
                 "wind": {"speed_mps": 7.0, "from_deg": 240.0},
                 "temperature_c": 15.0, "humidity_pct": 93.0, "hour": 18.0, "season": "summer"}},
)
_define(
    "underpass-shelter", "Under the underpass",
    ["weather:rain", "place:city", "mood:urban", "use:film", "use:game"],
    "Sheltering beneath a concrete underpass: curtains of runoff pour off the edge, rain beyond, traffic rumbling overhead.",
    "city_street",
    {"place": {"shelter": "underpass"},
     "weather": {"precipitation": {"type": "rain", "rate_mm_h": 12.0, "drops": "normal"},
                 "wind": {"speed_mps": 4.0, "from_deg": 250.0},
                 "temperature_c": 11.0, "humidity_pct": 94.0, "hour": 19.0, "season": "autumn"}},
)
_define(
    "jungle-dawn-rain", "Jungle dawn rain",
    ["weather:rain", "place:jungle", "mood:lush", "use:focus", "use:music"],
    "First light in the rainforest: warm steady rain on huge leaves, canopy drips, the dawn chorus fighting through.",
    "jungle",
    {"weather": {"precipitation": {"type": "rain", "rate_mm_h": 10.0, "drops": "fine"},
                 "wind": {"speed_mps": 1.0, "from_deg": 90.0},
                 "temperature_c": 25.0, "humidity_pct": 99.0, "hour": 6.0, "season": "summer", "latitude_deg": 3.0}},
)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def preset(name: str) -> Scene:
    """Build a fresh Scene for preset `name`."""
    key = _normalize_name(name)
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {', '.join(sorted(PRESETS))}")
    spec = PRESETS[key]
    base = apply_place(Scene(), spec["place"])
    d = base.to_dict()
    d = _deep_merge(d, copy.deepcopy(spec["scene"]))
    if spec.get("surfaces"):
        d["place"]["surfaces"] = dict(spec["surfaces"])
    d["name"] = spec["title"]
    scene = Scene.from_dict(d)
    scene.listener.position = tuple(scene.listener.position)
    if scene.mode and scene.mode != "none":
        from ..wellness.modes import apply_mode
        scene = apply_mode(scene, scene.mode)
    return scene


def _tag_match(tags: list[str], tag: str) -> bool:
    t = tag.strip().lower()
    for x in tags:
        if x == t or x.split(":", 1)[-1] == t:
            return True
    return False


def list_presets(tag: Optional[str] = None) -> list[dict]:
    """Cards for the scene browser: name, title, tags, description, place (filtered by tag)."""
    out = []
    for name, spec in PRESETS.items():
        if tag is not None and not _tag_match(spec["tags"], tag):
            continue
        out.append({"name": name, "title": spec["title"], "tags": list(spec["tags"]),
                    "description": spec["description"], "place": spec["place"],
                    "mode": spec["scene"].get("mode", "none")})
    return out


def preset_tags() -> dict[str, list[str]]:
    """All tags grouped by family (weather, place, mood, use)."""
    fam: dict[str, set] = {}
    for spec in PRESETS.values():
        for t in spec["tags"]:
            k, _, v = t.partition(":")
            fam.setdefault(k, set()).add(v)
    return {k: sorted(v) for k, v in sorted(fam.items())}


def surprise(seed: int, tag: Optional[str] = None) -> Scene:
    """'Surprise me': a preset chosen deterministically from a seed, re-seeded."""
    from ..core.rng import RngFactory
    names = [p["name"] for p in list_presets(tag)] or list(PRESETS)
    rng = RngFactory(seed).get("presets.surprise", 0)
    scene = preset(names[int(rng.integers(0, len(names)))])
    scene.seed = int(rng.integers(0, 2 ** 63 - 1))
    return scene
