"""The parameter manifest: one source of truth for every scene parameter (MASTER_PLAN 2.12).

Each `ParamSpec` carries id (dotted scene path), human label and unit, range,
default, scale (lin | log | beaufort | choice), tier (0 Vibe, 1 Weather,
2 Place & Listener, 3 Engineer), smoothing time (time to 99.5 %, tau = T/5.3;
rain 3 s, wind 1 s, distance 2 s, macros 0.25 s - 2.6), an honest audibility
tag (7.8 audibility table), the OSC address, a MIDI hint (14-bit CC pairs for
log / Beaufort controls because a 7-bit log rain slider jumps ~7 % per step),
and Wwise (RTPC / Switch / State) and FMOD (continuous / discrete / labeled)
types. JSON Schema, the OSC namespace, Wwise/FMOD manifests and the Markdown
reference are all generated from this list, so GUI, API and plugins cannot
disagree.

The *perceptual domain* used for offsets, MIDI and sliders is: log2 of the value
for 'log' (one unit = a doubling), the Beaufort number for 'beaufort', the
choice index for 'choice', and the plain value for 'lin'.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from . import units
from .schema import SCHEMA_VERSION, Scene

TIER_NAMES = {0: "Vibe", 1: "Weather", 2: "Place & Listener", 3: "Engineer"}
AUDIBILITY = ("direct", "indirect", "distant_only", "context_only")

SMOOTH_RAIN = 3.0
SMOOTH_WIND = 1.0
SMOOTH_DISTANCE = 2.0
SMOOTH_MACRO = 0.25
SMOOTH_LIVE = 60.0

PRECIP_TYPES = ("none", "drizzle", "rain", "showers", "freezing_rain", "sleet", "snow", "graupel", "hail")
SHELTERS = ("open", "tree", "umbrella", "tent", "porch", "car", "room_open", "room_closed", "attic", "cave", "underpass")
PLACES = ("garden", "deciduous_forest", "conifer_forest", "meadow", "coast", "lake", "city_street",
          "suburb", "alley", "mountain_ridge", "desert", "jungle", "farm", "cabin")
GROUNDS = ("grass", "soil", "asphalt", "water", "snow_fresh", "snow_old", "forest_floor")
SURFACE_MATERIALS = ("leaves", "grass", "soil", "puddle", "asphalt", "concrete", "gravel", "metal_sheet",
                     "tile", "glass", "fabric", "wood", "car_body", "snow", "needles", "water",
                     "plastic", "canvas")
REGIONS = ("temperate_europe", "north_america", "east_asia", "tropical", "mediterranean", "arid", "boreal", "oceania")
RENDER_LAYOUTS = ("stereo", "binaural", "5.1", "7.1.4", "foa", "hoa3", "7.1", "mono")   # new layouts appended: indices stay stable
# The delivery profiles of raingen.master.destinations, new ones appended so choice indices stay stable.
# Listed here rather than imported: importing raingen.master pulls in scipy and the limiter, and the
# scene layer must stay cheap to import (the test suite checks the two lists stay equal).
DESTINATIONS = ("raw", "youtube", "spotify", "apple_music", "podcast", "sleep", "broadcast_eu", "broadcast_us",
                "streaming_broadcaster", "post_stems", "game", "vr", "online_streaming")
MODES = ("none", "sleep", "focus", "calm", "asmr")

# Human words accepted for choice parameters (UI labels and spoken names -> stored value).
CHOICE_ALIASES: dict[str, dict[str, str]] = {
    "mode": {"relax": "calm", "relaxed": "calm", "cinematic": "none", "off": "none"},
    "place.shelter": {"out_in_it": "open", "outside": "open", "under_a_tree": "tree", "window_open": "room_open",
                      "window_shut": "room_closed", "window_closed": "room_closed", "indoors": "room_closed",
                      "tin_roof": "attic", "veranda": "porch", "vehicle": "car"},
    "weather.precipitation.type": {"ice_pellets": "sleet", "freezing_drizzle": "freezing_rain", "dry": "none",
                                   "snow_pellets": "graupel"},
    "weather.precipitation.drops": {"medium": "normal", "big": "fat", "small": "fine"},
    "weather.precipitation.character": {"steady": "continuous", "on_and_off": "intermittent"},
}


@dataclass(frozen=True)
class ParamSpec:
    id: str                                   # dotted scene path
    label: str
    unit: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    default: Any = None
    scale: str = "lin"                        # lin | log | beaufort | choice
    tier: int = 1
    smoothing_s: float = 0.0
    audibility: str = "direct"                # direct | indirect | distant_only | context_only
    audibility_hint: str = ""
    osc: str = ""
    osc_type: str = "f"
    midi: str = "cc7"                         # cc7 | cc14 | toggle | program | none
    wwise: Optional[str] = "RTPC"             # RTPC | Switch | State | None (not exported)
    fmod: Optional[str] = "continuous"        # continuous | discrete | labeled | None
    choices: tuple = ()
    type: str = "float"                       # float | int | bool | str | vec3
    nullable: bool = False                    # None means "auto"
    wrap: bool = False                        # circular quantity (degrees, hour of day)
    zero_ok: bool = False                     # 0 is valid below a log range ("no rain"); it is never clamped up
    description: str = ""
    ucs: str = ""                             # UCS hint for file naming / metadata
    group: str = ""

    @property
    def is_numeric(self) -> bool:
        return self.type in ("float", "int") and self.scale != "choice"

    @property
    def tier_name(self) -> str:
        return TIER_NAMES.get(self.tier, str(self.tier))


# ---------------------------------------------------------------------------
# Defaults from the schema
# ---------------------------------------------------------------------------
_DEFAULT_SCENE = Scene()
_VIBE_DEFAULTS = {"intensity": 40.0, "closeness": 50.0, "wildness": 30.0, "volume_db": -12.0, "timer_min": 0.0}


def _schema_default(path: str) -> Any:
    if path.startswith("vibe."):
        return _VIBE_DEFAULTS.get(path.split(".", 1)[1])
    try:
        return _DEFAULT_SCENE.get(path)
    except (KeyError, AttributeError):
        if path.startswith("place.surfaces."):
            return 0.0
        return None


def _smoothing_for(path: str, scale: str, type_: str) -> float:
    if scale == "choice" or type_ in ("bool", "str"):
        return 0.0
    if path.startswith("vibe."):
        return SMOOTH_MACRO
    if "distance" in path or path.endswith("_km") or path == "listener.position":
        return SMOOTH_DISTANCE
    if path.startswith("weather.wind."):
        return SMOOTH_WIND
    if path.startswith("weather.precipitation."):
        return SMOOTH_RAIN
    if path.startswith("weather.thunder."):
        return SMOOTH_DISTANCE
    if path.startswith("weather."):
        return SMOOTH_RAIN
    if path.startswith("listener."):
        return SMOOTH_MACRO
    if path.startswith("place."):
        return SMOOTH_DISTANCE
    return 0.0


def _p(id: str, label: str, unit: str = "", lo: Optional[float] = None, hi: Optional[float] = None, *,
       scale: str = "lin", tier: int = 1, audibility: str = "direct", hint: str = "", choices: tuple = (),
       type: str = "float", nullable: bool = False, wrap: bool = False, zero_ok: bool = False,
       wwise: Optional[str] = "auto",
       fmod: Optional[str] = "auto", midi: Optional[str] = None, smoothing: Optional[float] = None,
       default: Any = "schema", description: str = "", ucs: str = "") -> ParamSpec:
    if choices and scale == "lin":
        scale = "choice"
    if type == "bool":
        scale = "choice"
        choices = (False, True)
    if default == "schema":
        default = _schema_default(id)
    if smoothing is None:
        smoothing = _smoothing_for(id, scale, type)
    osc = "/rg/" + id.replace(".", "/")
    osc_type = {"float": "f", "int": "i", "bool": "T", "str": "s", "vec3": "fff"}[type]
    if scale == "choice" and type == "str":
        osc_type = "s"
    if midi is None:
        if type == "bool":
            midi = "toggle"
        elif scale == "choice":
            midi = "program"
        elif type in ("str", "vec3"):
            midi = "none"
        elif scale in ("log", "beaufort") or (hi is not None and lo is not None and type == "int" and hi - lo > 127):
            midi = "cc14"
        else:
            midi = "cc7"
    if wwise == "auto":
        if type in ("str", "vec3") and not choices:
            wwise = None
        elif scale == "choice":
            wwise = "Switch" if id.startswith(("place.", "listener.")) else "State"
        else:
            wwise = "RTPC"
    if fmod == "auto":
        if type in ("str", "vec3") and not choices:
            fmod = None
        elif scale == "choice":
            fmod = "labeled"
        elif type == "int":
            fmod = "discrete"
        else:
            fmod = "continuous"
    group = id.rsplit(".", 1)[0] if "." in id else id
    return ParamSpec(id=id, label=label, unit=unit, min=lo, max=hi, default=default, scale=scale, tier=tier,
                     smoothing_s=float(smoothing), audibility=audibility, audibility_hint=hint, osc=osc,
                     osc_type=osc_type, midi=midi, wwise=wwise, fmod=fmod, choices=tuple(choices), type=type,
                     nullable=nullable, wrap=wrap, zero_ok=zero_ok, description=description, ucs=ucs, group=group)


_H_DIRECT = "heard directly"
_H_RATE = "level, density and timbre: more large drops add low-mid punch"
_H_DROPS = "fine-rain hiss vs fat-drop splats; how many plinks on water"
_H_WIND = "objects come alive by Beaufort force; rain slants; small-drop plinks on water vanish at 8-10 m/s"
_H_GUST = "swells that travel across the scene; rain curtains"
_H_DIR = "which side gusts come from, which windows get hit; distant sounds louder downwind"
_H_TEMP = "indirect but real: rain vs sleet vs snow, cricket chirp rate, flash-to-bang timing, night-time sound carry"
_H_HUM = "brightness of distant sounds only (within 10 m it changes 8 kHz by at most 2 dB)"
_H_PRESS = "nothing audible (980-1030 hPa changes absorption by <= 0.05 %); storm-approach logic only"
_H_VISUAL = "visuals and context only"
_H_TIME = "birds, insects, traffic, leaves on or off, night-time sound carry"
_H_ENGINE = "engineering parameter: changes the physical model"
_H_RENDER = "output format; does not change the weather"


def _build_params() -> list[ParamSpec]:
    P: list[ParamSpec] = []
    add = P.append

    # ---------------- Tier 0: Vibe -------------------------------------------------
    add(_p("vibe.intensity", "Intensity", "", 0, 100, tier=0, hint="walks the storm path: drips, drizzle, rain, downpour, thunderstorm",
           description="0 dry/drips only, 20 drizzle 0.3 mm/h, 40 moderate 5, 60 heavy 20, 80 thunderstorm 50, 100 cloudburst 100 mm/h"))
    add(_p("vibe.closeness", "Closeness / Shelter", "", 0, 100, tier=0, hint="from out in it to indoors with the window shut",
           description="0 open, 25 umbrella or tree, 50 porch or tent, 75 room with window open, 100 window shut"))
    add(_p("vibe.wildness", "Calm ↔ Wild", "", 0, 100, tier=0, hint="gustiness, shower contrast and thunder rate",
           description="gust factor 1.15-1.9, gust rate, shower-cell contrast, thunder rate"))
    add(_p("vibe.volume_db", "Volume", "dB", -60, 0, tier=0, hint="output level behind the limiter and dose meter"))
    add(_p("vibe.timer_min", "Sleep timer", "min", 0, 120, tier=0, hint="0 = off; fades out with a Descent", smoothing=0.0))
    add(_p("mode", "Mode", "", choices=MODES, type="str", tier=0, wwise="State",
           hint="wellness modes apply the Calm Governor guard-rails"))

    # ---------------- Tier 1: Weather ----------------------------------------------
    pr = "weather.precipitation."
    add(_p(pr + "type", "Precipitation", "", choices=PRECIP_TYPES, type="str", tier=1, hint=_H_DIRECT, ucs="RAIN"))
    add(_p(pr + "rate_mm_h", "Amount", "mm/h", 0.02, 150.0, scale="log", tier=1, hint=_H_RATE, zero_ok=True,
           description="WMO detents: slight < 2.5, moderate 2.5-10, heavy 10-50, intense >= 50, cloudburst >= 100; "
                       "0 = no rain (0.02 mm/h is the WMO detection threshold and the log slider's floor)"))
    add(_p(pr + "drops", "Drop character", "", choices=("fine", "normal", "fat"), type="str", tier=1, hint=_H_DROPS,
           description="moves Dm along the stratiform / maritime / continental clusters (Bringi 2003)"))
    add(_p(pr + "character", "Character", "", choices=("continuous", "intermittent", "showery"), type="str", tier=1,
           hint="steady texture vs cells that come and go"))
    add(_p(pr + "shower_period_min", "Shower cell period", "min", 5, 30, tier=1, hint="how often shower cells pass"))
    add(_p(pr + "hail_size_mm", "Hail size", "mm", 6, 44, scale="log", tier=1, hint="pea 6 mm ... golf ball 44 mm"))

    wd = "weather.wind."
    add(_p(wd + "speed_mps", "Wind", "m/s", 0.0, 40.0, scale="beaufort", tier=1, hint=_H_WIND, ucs="WIND",
           description="reported mean at 10 m; shown as Beaufort with its land description"))
    add(_p(wd + "from_deg", "Wind from", "deg", 0, 360, tier=1, wrap=True, audibility="indirect", hint=_H_DIR,
           description="meteorological 'from', degrees true"))
    add(_p(wd + "gust_factor", "Gusts", "x", 1.0, 2.5, tier=1, nullable=True, hint=_H_GUST,
           description="peak/mean gust factor; auto from place roughness (G = 1 + 3 I)"))
    add(_p(wd + "gust_rate_per_min", "Gust rate", "/min", 0.0, 10.0, tier=1, nullable=True, hint=_H_GUST))

    th = "weather.thunder."
    add(_p(th + "enabled", "Thunder", "", type="bool", tier=1, hint=_H_DIRECT, ucs="THUN"))
    add(_p(th + "distance_km", "Thunder distance", "km", 0.3, 25.0, scale="log", tier=1,
           hint="crack and clap up close, only rumble far away (air absorption)",
           description="also shown as flash-to-bang seconds"))
    add(_p(th + "bearing_deg", "Storm bearing", "deg", 0, 360, tier=1, wrap=True, hint="direction of the storm cell"))
    add(_p(th + "approach_kmh", "Storm approach", "km/h", -60, 60, tier=1, hint="+ approaching, - receding"))
    add(_p(th + "flashes_per_min", "Flashes", "/min", 0, 20, tier=1, hint="how often thunder arrives"))

    add(_p("weather.temperature_c", "Temperature", "°C", -30, 45, tier=1, audibility="indirect", hint=_H_TEMP))
    add(_p("weather.humidity_pct", "Humidity", "%", 10, 100, tier=1, audibility="distant_only", hint=_H_HUM))
    add(_p("weather.hour", "Time of day", "h", 0, 24, tier=1, wrap=True, audibility="indirect", hint=_H_TIME))
    add(_p("weather.season", "Season", "", choices=("spring", "summer", "autumn", "winter"), type="str", tier=1,
           audibility="indirect", hint=_H_TIME))
    add(_p("weather.storm_arc", "Storm arc", "", choices=("arriving", "steady", "passing", "clearing"), type="str",
           tier=1, hint="sparse big drops, downpour, stratiform tail, drips"))
    add(_p("weather.arc_minutes", "Arc length", "min", 0.5, 120, scale="log", tier=1, hint="how long the arc phase lasts",
           description="minutes the current arc phase lasts; a whole arc is usually 5-120 min (2.5), so a short "
                       "storm's phases can be a few minutes each (the storm model's floor is 0.5 min)"))
    add(_p("weather.cloud_okta", "Cloud cover", "okta", 0, 8, tier=1, audibility="context_only", hint=_H_VISUAL))
    add(_p("weather.visibility_km", "Visibility", "km", 0.05, 50, scale="log", tier=1, audibility="context_only", hint=_H_VISUAL))
    add(_p("weather.pressure_hpa", "Pressure", "hPa", 870, 1085, tier=1, audibility="context_only", hint=_H_PRESS))

    # ---------------- Tier 2: Place & Listener -------------------------------------
    add(_p("place.preset", "Place", "", choices=PLACES, type="str", tier=2, wwise="State",
           hint="sets roughness, vegetation, ambience beds and reverb"))
    add(_p("place.shelter", "Shelter", "", choices=SHELTERS, type="str", tier=2, hint="impact panels near the ear plus a filtered exterior"))
    for mat in SURFACE_MATERIALS:
        add(_p(f"place.surfaces.{mat}", f"Surface: {mat.replace('_', ' ')}", "%", 0, 100, tier=2,
               hint="share of ground coverage around the listener", ucs="RAIN"))
    add(_p("place.vegetation.deciduous", "Broadleaf trees", "", 0, 50, type="int", tier=2, hint="rustle (Bft 2+), canopy drips"))
    add(_p("place.vegetation.conifer", "Conifers", "", 0, 50, type="int", tier=2, hint="needle 'whoosh' (Strouhal), canopy drips"))
    add(_p("place.vegetation.distance_m", "Tree distance", "m", 1, 200, scale="log", tier=2, hint="how far the trees stand"))
    add(_p("place.z0_m", "Roughness length z0", "m", 0.0002, 2.0, scale="log", tier=3, nullable=True, hint=_H_ENGINE,
           description="auto from place; sets wind at your ears and gust factor"))
    add(_p("place.ground", "Ground", "", choices=GROUNDS, type="str", tier=2, audibility="distant_only",
           hint="soft ground gives a 200-600 Hz dip for distant sound"))
    add(_p("place.biophony.birds", "Birds", "", choices=("auto", "off", "on"), type="str", tier=2, hint="go quiet in heavy rain, rebound afterwards"))
    add(_p("place.biophony.insects", "Insects", "", choices=("auto", "off", "on"), type="str", tier=2, hint="crickets follow temperature (Dolbear)"))
    add(_p("place.biophony.frogs", "Frogs", "", choices=("auto", "off", "on"), type="str", tier=2, hint="chorus on warm wet evenings"))
    add(_p("place.biophony.traffic", "Traffic", "", 0, 1, tier=2, hint="distant traffic hiss, wetter roads hiss more"))
    add(_p("place.biophony.region", "Biophony region", "", choices=REGIONS, type="str", tier=2, wwise="State",
           hint="region- and season-correct species"))
    add(_p("place.aftermath_tau_min", "Drip decay", "min", 1, 30, scale="log", tier=2, hint="how long drips and runoff last after rain"))
    add(_p("place.window_open", "Window open", "", 0, 1, tier=2, hint="room shelters: exterior through the window gap"))
    add(_p("place.wipers", "Wipers", "", choices=("off", "slow", "fast"), type="str", tier=2, hint="car shelter: 20-60 cycles/min (FMVSS 104)"))

    add(_p("listener.height_m", "Listener height", "m", 0.3, 30, scale="log", tier=2,
           hint="sets impact footprint (50 % within radius h) and wind at your ears"))
    add(_p("listener.heading_deg", "Heading", "deg", 0, 360, tier=2, wrap=True, hint="direction you face"))
    add(_p("listener.position", "Listener position", "m", type="vec3", tier=2, hint="world metres: x east, y north, z ground",
           wwise=None, fmod=None))
    add(_p("listener.render", "Output layout", "", choices=RENDER_LAYOUTS, type="str", tier=2, hint=_H_RENDER, wwise=None))
    add(_p("listener.width", "Width", "", 0, 1.5, tier=2, hint="stereo/binaural image width"))
    add(_p("listener.hrtf", "HRTF", "", type="str", tier=3, hint="builtin spherical head or a .sofa file", wwise=None, fmod=None))

    # ---------------- Tier 3: Engineer ---------------------------------------------
    add(_p(pr + "dsd_model", "DSD family", "", choices=("gamma", "marshall_palmer"), type="str", tier=3, hint=_H_ENGINE))
    add(_p(pr + "mu", "DSD shape μ", "", -2, 10, tier=3, nullable=True, hint=_H_ENGINE))
    add(_p(pr + "dm_mm", "Mass-weighted diameter Dm", "mm", 0.3, 4.0, scale="log", tier=3, nullable=True, hint=_H_DROPS))
    add(_p(pr + "nw", "Intercept Nw", "m^-3 mm^-1", 10, 1e6, scale="log", tier=3, nullable=True, hint=_H_ENGINE))
    add(_p(pr + "d_min_mm", "Min drop size", "mm", 0.1, 1.0, tier=3, hint=_H_ENGINE))
    add(_p(pr + "d_max_mm", "Max drop size", "mm", 3.0, 8.0, tier=3, hint="drops break up above ~6 mm"))
    add(_p(pr + "class_boundaries", "Drop-class boundaries", "", choices=("medwin", "nystuen"), type="str", tier=3, hint=_H_ENGINE))
    add(_p(pr + "water_quality", "Water quality", "", 0, 1, tier=3, hint="1 clean .. 0 soapy: surfactant stops bubble plinks"))

    add(_p(wd + "turbulence_intensity", "Turbulence intensity", "", 0.05, 0.5, tier=3, nullable=True, hint=_H_GUST))
    add(_p(wd + "spectrum", "Gust spectrum", "", choices=("kaimal", "von_karman"), type="str", tier=3, hint=_H_ENGINE))
    add(_p(wd + "integral_time_s", "Integral time scale", "s", 1, 60, scale="log", tier=3, nullable=True, hint=_H_ENGINE))
    add(_p(wd + "mic_rumble", "Mic wind rumble", "", type="bool", tier=3, hint="recorded-realism artifact, not environmental sound"))
    add(_p(wd + "windscreen", "Windscreen", "", choices=("none", "foam", "blimp", "dead_cat"), type="str", tier=3,
           hint="only with mic rumble"))
    add(_p(th + "stroke_intensity", "Stroke intensity", "", 0, 1, tier=3, hint="bigger strokes are deeper (Few relaxation radius)"))
    add(_p(th + "cg_fraction", "Cloud-to-ground share", "", 0, 1, tier=3, hint="CG channels crack, IC channels roll"))
    add(_p("weather.latitude_deg", "Latitude", "deg", -90, 90, tier=3, audibility="indirect", hint="season and daylight logic"))

    add(_p("render.sample_rate", "Sample rate", "Hz", choices=(44100, 48000, 96000), type="int", tier=3, wwise=None, hint=_H_RENDER))
    add(_p("render.quality", "Quality tier", "", choices=("mobile", "game", "studio", "cinematic"), type="str", tier=3, wwise=None, hint=_H_RENDER))
    add(_p("render.ambi_order", "Ambisonic order", "", choices=(1, 2, 3), type="int", tier=3, wwise=None, hint=_H_RENDER))
    add(_p("render.k_objects", "Object voices K", "", 8, 256, type="int", tier=3, wwise=None, hint="near-field voices before the granular layer"))
    add(_p("render.bit_depth", "Bit depth", "bit", choices=(16, 24, 32), type="int", tier=3, wwise=None, hint=_H_RENDER))
    add(_p("render.destination", "Destination", "", choices=DESTINATIONS, type="str", tier=3, wwise=None, hint=_H_RENDER))
    add(_p("render.stems", "Stems", "", type="bool", tier=3, wwise=None, hint=_H_RENDER))
    add(_p("seed", "Seed", "", 0, float(2 ** 64 - 1), type="int", tier=3, wwise=None, fmod=None, midi="none",
           audibility="direct", hint="same scene + seed = the same audio, bit for bit"))

    # fix audibility for render/engineer entries without a meaningful tag
    out = []
    for s in P:
        if s.id.startswith("render.") or s.id in ("listener.render", "listener.hrtf"):
            s = _replace(s, audibility="context_only")
        out.append(s)
    return out


def _replace(spec: ParamSpec, **kw) -> ParamSpec:
    d = {f: getattr(spec, f) for f in spec.__dataclass_fields__}
    d.update(kw)
    return ParamSpec(**d)


PARAMS: list[ParamSpec] = _build_params()
_BY_ID: dict[str, ParamSpec] = {p.id: p for p in PARAMS}
_BY_OSC: dict[str, ParamSpec] = {p.osc: p for p in PARAMS}


def get_param(path: str) -> ParamSpec:
    """The ParamSpec for a dotted scene path (KeyError if the path is not a parameter)."""
    try:
        return _BY_ID[path]
    except KeyError:
        if path.startswith("place.surfaces."):
            mat = path.rsplit(".", 1)[1]
            spec = _p(path, f"Surface: {mat.replace('_', ' ')}", "%", 0, 100, tier=2,
                      hint="share of ground coverage around the listener", default=0.0)
            _BY_ID[path] = spec
            return spec
        raise KeyError(f"unknown parameter {path!r}") from None


def find_param(path: str) -> Optional[ParamSpec]:
    try:
        return get_param(path)
    except KeyError:
        return None


def param_by_osc(address: str) -> Optional[ParamSpec]:
    return _BY_OSC.get(address)


def params_for_tier(tier: int) -> list[ParamSpec]:
    return [p for p in PARAMS if p.tier == tier]


# ---------------------------------------------------------------------------
# Value domains
# ---------------------------------------------------------------------------
def _log_floor(spec: ParamSpec) -> float:
    return spec.min if (spec.min is not None and spec.min > 0) else 1e-6


def to_perceptual(spec: ParamSpec, value: Any) -> float:
    """Plain value -> perceptual domain (log2 for 'log', Beaufort for 'beaufort', index for 'choice')."""
    if spec.scale == "choice":
        if value in spec.choices:
            return float(spec.choices.index(value))
        return 0.0
    v = float(value)
    if spec.scale == "log":
        return math.log2(max(v, _log_floor(spec)))
    if spec.scale == "beaufort":
        return units.mps_to_beaufort(v)
    return v


def from_perceptual(spec: ParamSpec, p: float) -> Any:
    """Perceptual domain -> plain value (inverse of to_perceptual)."""
    if spec.scale == "choice":
        i = int(round(p))
        return spec.choices[max(0, min(len(spec.choices) - 1, i))] if spec.choices else None
    if spec.scale == "log":
        v = 2.0 ** float(p)
    elif spec.scale == "beaufort":
        v = units.beaufort_to_mps(p)
    else:
        v = float(p)
    if spec.type == "int":
        return int(round(v))
    return v


def clamp(spec: ParamSpec, value: Any) -> Any:
    """Clamp a numeric value to the spec range (wrapping circular quantities)."""
    if value is None or not spec.is_numeric:
        return value
    v = float(value)
    if spec.zero_ok and v <= 0.0:
        return 0.0                                # "no rain" stays exactly zero
    if spec.wrap and spec.min is not None and spec.max is not None:
        span = spec.max - spec.min
        v = spec.min + ((v - spec.min) % span)
    else:
        if spec.min is not None:
            v = max(spec.min, v)
        if spec.max is not None:
            v = min(spec.max, v)
    return int(round(v)) if spec.type == "int" else v


def normalize(spec: ParamSpec, value: Any) -> float:
    """Plain value -> 0..1 slider / MIDI position (in the perceptual domain)."""
    if spec.scale == "choice":
        n = max(1, len(spec.choices) - 1)
        return to_perceptual(spec, value) / n
    if spec.min is None or spec.max is None or value is None:
        return 0.0
    lo, hi = to_perceptual(spec, spec.min), to_perceptual(spec, spec.max)
    if hi == lo:
        return 0.0
    return float(min(1.0, max(0.0, (to_perceptual(spec, value) - lo) / (hi - lo))))


def denormalize(spec: ParamSpec, x: float) -> Any:
    """0..1 slider / MIDI position -> plain value."""
    x = float(min(1.0, max(0.0, x)))
    if spec.scale == "choice":
        n = max(1, len(spec.choices) - 1)
        return from_perceptual(spec, x * n)
    if spec.min is None or spec.max is None:
        return x
    lo, hi = to_perceptual(spec, spec.min), to_perceptual(spec, spec.max)
    return clamp(spec, from_perceptual(spec, lo + x * (hi - lo)))


def midi_to_value(spec: ParamSpec, msb: int, lsb: Optional[int] = None) -> Any:
    """7-bit (msb only) or 14-bit (msb, lsb) controller value -> plain value."""
    if lsb is None:
        return denormalize(spec, (msb & 0x7F) / 127.0)
    return denormalize(spec, (((msb & 0x7F) << 7) | (lsb & 0x7F)) / 16383.0)


def coerce(spec: ParamSpec, value: Any, temp_c: float = 20.0) -> Any:
    """Convert a user/OSC/MIDI value into the stored type and unit (strings may carry units)."""
    if spec.nullable and (value is None or (isinstance(value, str) and value.strip().lower() in ("auto", "none", ""))):
        return None
    if spec.type == "bool":
        return units.parse_bool(value)
    if spec.type == "vec3":
        if isinstance(value, str):
            parts = [float(units._num(v)) for v in value.replace(";", ",").split(",")] if "," in value else [float(v) for v in value.split()]
        else:
            parts = [float(v) for v in value]
        if len(parts) != 3:
            raise ValueError(f"{spec.id} needs three numbers (x, y, z)")
        return tuple(parts)
    if spec.scale == "choice":
        if value in spec.choices:
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace(" ", "_").replace("-", "_")
            key = CHOICE_ALIASES.get(spec.id, {}).get(key, key)
            for c in spec.choices:
                if str(c).lower().replace("-", "_") == key:
                    return c
            try:
                num = float(value)
            except ValueError:
                raise ValueError(f"{value!r} is not one of {spec.choices} for {spec.id}") from None
            value = num
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value in spec.choices:
                return spec.choices[spec.choices.index(value)]
            i = int(round(value))
            if 0 <= i < len(spec.choices):
                return spec.choices[i]
        raise ValueError(f"{value!r} is not one of {spec.choices} for {spec.id}")
    if spec.type == "str":
        return str(value)
    v = units.parse_value(spec.id, value, temp_c) if isinstance(value, str) else value
    v = float(v)
    return int(round(v)) if spec.type == "int" else v


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def _json_type(spec: ParamSpec) -> dict:
    if spec.type == "bool":
        t: dict = {"type": "boolean"}
    elif spec.type == "vec3":
        t = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    elif spec.scale == "choice":
        t = {"enum": list(spec.choices)}
    elif spec.type == "str":
        t = {"type": "string"}
    else:
        t = {"type": "integer" if spec.type == "int" else "number"}
        if spec.min is not None:
            t["minimum"] = 0 if spec.zero_ok else spec.min
            if spec.zero_ok:
                t["x-log-floor"] = spec.min
        if spec.max is not None:
            t["maximum"] = spec.max
    if spec.nullable:
        if "type" in t:
            t["type"] = [t["type"], "null"]
        elif "enum" in t:
            t["enum"] = t["enum"] + [None]
    t["title"] = spec.label
    desc = spec.description or spec.audibility_hint
    if desc:
        t["description"] = desc
    if spec.default is not None or spec.nullable:
        d = spec.default
        t["default"] = list(d) if isinstance(d, tuple) else d
    t["x-unit"] = spec.unit
    t["x-scale"] = spec.scale
    t["x-tier"] = spec.tier
    t["x-smoothing-s"] = spec.smoothing_s
    t["x-audibility"] = spec.audibility
    t["x-osc"] = spec.osc
    return t


def _ensure_obj(node: dict, key: str) -> dict:
    props = node.setdefault("properties", {})
    child = props.get(key)
    if child is None:
        child = {"type": "object", "properties": {}}
        props[key] = child
    return child


def to_json_schema() -> dict:
    """JSON Schema (draft 2020-12) for the open `rainscene` format, generated from PARAMS."""
    root: dict = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:raingen:rainscene:{SCHEMA_VERSION}",
        "title": "rainscene",
        "description": "Open, versioned scene format for physically based weather sound (raingen).",
        "type": "object",
        "properties": {
            "rainscene": {"const": SCHEMA_VERSION, "description": "format version"},
            "name": {"type": "string"},
        },
    }
    for spec in PARAMS:
        parts = spec.id.split(".")
        node = root
        for part in parts[:-1]:
            node = _ensure_obj(node, part)
        node.setdefault("properties", {})[parts[-1]] = _json_type(spec)

    place = root["properties"]["place"]
    place["properties"]["surfaces"]["additionalProperties"] = {"type": "number", "minimum": 0, "maximum": 100}
    place["properties"]["emitters"] = {
        "type": "array",
        "items": {"type": "object", "required": ["type"], "properties": {
            "type": {"enum": ["gutter", "downspout", "eave", "wire", "chime", "flag"]},
            "distance_m": {"type": "number", "minimum": 0.2},
            "bearing_deg": {"type": "number"},
            "count": {"type": "integer", "minimum": 1},
        }},
    }
    place["properties"]["objects"] = {
        "type": "array",
        "items": {"type": "object", "required": ["id", "type"], "properties": {
            "id": {"type": "string"},
            "type": {"enum": ["surface", "water", "tree", "grass", "gutter", "downspout", "eave", "wire", "chime",
                              "flag", "car", "tent", "umbrella", "roof", "window", "storm", "wall"]},
            "material": {"type": "string"},
            "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "size": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "rotation_deg": {"type": "number"},
            "params": {"type": "object"},
        }},
    }
    root["properties"]["timeline"] = {
        "type": "array",
        "items": {"type": "object", "required": ["at", "path", "value"], "properties": {
            "at": {"type": ["number", "string"], "description": "seconds, mm:ss, hh:mm:ss or hh:mm:ss:ff"},
            "path": {"type": "string"},
            "value": {},
            "curve": {"enum": ["linear", "log", "hold", "smooth"]},
        }},
    }
    root["properties"]["overrides"] = {
        "type": "object",
        "description": "Auto/Offset/Pinned state per parameter path",
        "additionalProperties": {"type": "object", "properties": {
            "state": {"enum": ["auto", "offset", "pinned"]},
            "offset": {"type": "number", "description": "perceptual delta: log2 ratio, Beaufort steps or plain units"},
            "ratio": {"type": "number", "minimum": 0.25, "maximum": 4.0,
                      "description": "log-scaled parameters: the adjustment as a ratio (x0.25 .. x4)"},
            "value": {},
        }},
    }
    return root


def osc_namespace() -> dict[str, dict]:
    """OSC address -> description for every parameter and control command."""
    ns: dict[str, dict] = {}
    for spec in PARAMS:
        entry = {"id": spec.id, "type": spec.osc_type, "unit": spec.unit, "min": spec.min, "max": spec.max,
                 "default": list(spec.default) if isinstance(spec.default, tuple) else spec.default,
                 "smoothing_s": spec.smoothing_s, "kind": "parameter"}
        if spec.choices:
            entry["choices"] = list(spec.choices)
        ns[spec.osc] = entry
        if spec.is_numeric and spec.min is not None and spec.max is not None:
            ns[spec.osc + "/norm"] = {"id": spec.id, "type": "f", "min": 0.0, "max": 1.0, "kind": "normalized",
                                      "scale": spec.scale}
    ns["/rg/preset"] = {"type": "s", "kind": "command", "description": "load a preset by name"}
    ns["/rg/text"] = {"type": "s", "kind": "command", "description": "describe your weather in words"}
    ns["/rg/metar"] = {"type": "s", "kind": "command", "description": "apply a METAR report"}
    ns["/rg/pin"] = {"type": "s", "optional": "value (any type, in the parameter's units)", "kind": "command",
                     "description": "lock a parameter (path [, value]); without a value it locks the current value"}
    ns["/rg/offset"] = {"type": "sf", "kind": "command", "description": "offset a parameter (path, perceptual delta)"}
    ns["/rg/relink"] = {"type": "s", "kind": "command", "description": "return a parameter to Auto"}
    return dict(sorted(ns.items()))


def _wwise_name(spec: ParamSpec) -> str:
    return "RG_" + spec.id.replace(".", "_")


def wwise_manifest() -> dict:
    """Wwise RTPC / Switch Group / State Group definitions in human units."""
    rtpcs, switches, states = [], [], []
    for spec in PARAMS:
        if spec.wwise == "RTPC" and spec.is_numeric:
            rtpcs.append({"name": _wwise_name(spec), "path": spec.id, "min": spec.min, "max": spec.max,
                          "default": spec.default if spec.default is not None else spec.min,
                          "unit": spec.unit, "scale": spec.scale, "slew_s": spec.smoothing_s,
                          "description": spec.audibility_hint})
        elif spec.wwise == "Switch":
            switches.append({"group": _wwise_name(spec), "path": spec.id,
                             "switches": [str(c) for c in spec.choices], "default": str(spec.default)})
        elif spec.wwise == "State":
            states.append({"group": _wwise_name(spec), "path": spec.id,
                           "states": [str(c) for c in spec.choices], "default": str(spec.default)})
        elif spec.wwise == "RTPC" and spec.type == "bool":
            rtpcs.append({"name": _wwise_name(spec), "path": spec.id, "min": 0, "max": 1,
                          "default": int(bool(spec.default)), "unit": "", "scale": "lin", "slew_s": 0.0})
    from .presets import list_presets  # local import: presets depend on the manifest indirectly
    states.append({"group": "RG_Preset", "path": "preset", "states": [p["name"] for p in list_presets()],
                   "default": "steady-rain-leaves-night"})
    return {"generator": "raingen", "schema": SCHEMA_VERSION, "RTPCs": rtpcs, "Switches": switches, "States": states}


def fmod_manifest() -> dict:
    """FMOD Studio parameter definitions (continuous / discrete / labeled) with seek speeds."""
    params = []
    for spec in PARAMS:
        if spec.fmod is None:
            continue
        entry: dict = {"name": spec.id.replace(".", "/"), "path": spec.id, "type": spec.fmod,
                       "global": spec.id.startswith(("weather.", "vibe.", "mode")), "unit": spec.unit}
        if spec.fmod == "labeled":
            entry["labels"] = [str(c) for c in spec.choices]
            entry["min"], entry["max"] = 0, max(0, len(spec.choices) - 1)
            entry["default"] = spec.choices.index(spec.default) if spec.default in spec.choices else 0
        else:
            entry["min"], entry["max"] = spec.min, spec.max
            entry["default"] = spec.default if spec.default is not None else spec.min
            if spec.smoothing_s > 0 and spec.min is not None and spec.max is not None:
                entry["seek_speed"] = (spec.max - spec.min) / spec.smoothing_s
        params.append(entry)
    return {"generator": "raingen", "schema": SCHEMA_VERSION, "parameters": params}


def _range_txt(spec: ParamSpec) -> str:
    if spec.choices:
        return " / ".join(str(c) for c in spec.choices)
    if spec.min is None or spec.max is None:
        return ""
    return f"{spec.min:g} – {spec.max:g}"


def markdown_reference() -> str:
    """Human-readable parameter reference grouped by tier."""
    lines = ["# raingen parameter reference", "",
             f"Generated from the manifest (rainscene v{SCHEMA_VERSION}). Stored units: mm/h, m/s, °C, hPa, m/km, degrees true *from*.", "",
             "Audibility: **direct** = heard directly; **indirect** = changes other sounds; "
             "**distant_only** = only audible on distant sources; **context_only** = visuals/metadata only.", ""]
    for tier in sorted(TIER_NAMES):
        specs = params_for_tier(tier)
        if not specs:
            continue
        lines += [f"## Tier {tier}: {TIER_NAMES[tier]}", "",
                  "| Parameter | Label | Range | Default | Unit | Scale | Smoothing | Audibility | OSC |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in specs:
            default = "auto" if s.default is None and s.nullable else s.default
            aud = s.audibility + (f": {s.audibility_hint}" if s.audibility_hint else "")
            lines.append(f"| `{s.id}` | {s.label} | {_range_txt(s)} | {default} | {s.unit} | {s.scale} | "
                         f"{s.smoothing_s:g} s | {aud.replace('|', '/')} | `{s.osc} {s.osc_type}` |")
        lines.append("")
    return "\n".join(lines)


def manifest_json(indent: int = 2) -> str:
    """The full manifest as JSON (for code generators in other languages)."""
    rows = []
    for s in PARAMS:
        d = {f: getattr(s, f) for f in s.__dataclass_fields__}
        d["choices"] = list(s.choices)
        if isinstance(d["default"], tuple):
            d["default"] = list(d["default"])
        rows.append(d)
    return json.dumps({"schema": SCHEMA_VERSION, "params": rows}, indent=indent, default=str)
