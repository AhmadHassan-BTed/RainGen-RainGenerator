"""The Scene: user intent in human units (the open `rainscene` format, v1).

A Scene is plain data. It round-trips to JSON/YAML and is embedded in every
exported file so any render can be reproduced exactly. Physical resolution
(drop-size distributions, gusts, wetness...) happens in `raingen.physics`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional

SCHEMA_VERSION = 1


@dataclass
class Precipitation:
    type: str = "rain"              # none|drizzle|rain|showers|freezing_rain|sleet|snow|graupel|hail
    rate_mm_h: float = 4.0          # 0.02 .. 150 (log)
    drops: str = "normal"           # fine|normal|fat  -> Dm along stratiform/maritime/continental
    character: str = "continuous"   # continuous|intermittent|showery
    shower_period_min: float = 12.0
    hail_size_mm: float = 13.0
    # Engineer tier (None = auto)
    dsd_model: str = "gamma"        # gamma|marshall_palmer
    mu: Optional[float] = None
    dm_mm: Optional[float] = None
    nw: Optional[float] = None
    d_min_mm: float = 0.3
    d_max_mm: float = 6.0
    class_boundaries: str = "medwin"  # medwin (0.8/1.1/2.2/3.5) | nystuen (0.8/1.2/2.0/3.5)
    water_quality: float = 1.0      # 1 clean .. 0 soapy/dirty (suppresses bubble plinks)


@dataclass
class Wind:
    speed_mps: float = 4.4          # reported mean at 10 m (Beaufort 3)
    from_deg: float = 270.0         # meteorological 'from', degrees true
    gust_factor: Optional[float] = None   # None = from place roughness
    gust_rate_per_min: Optional[float] = None
    turbulence_intensity: Optional[float] = None
    spectrum: str = "kaimal"        # kaimal|von_karman
    integral_time_s: Optional[float] = None
    mic_rumble: bool = False        # optional 'recorded realism' artifact
    windscreen: str = "blimp"       # none|foam|blimp|dead_cat


@dataclass
class Thunder:
    enabled: bool = False
    distance_km: float = 8.0        # also shown as flash-to-bang seconds
    bearing_deg: float = 270.0
    approach_kmh: float = 0.0       # + approaching, - receding
    flashes_per_min: float = 2.0
    stroke_intensity: float = 0.5   # 0..1
    cg_fraction: float = 0.4


@dataclass
class Weather:
    precipitation: Precipitation = field(default_factory=Precipitation)
    wind: Wind = field(default_factory=Wind)
    thunder: Thunder = field(default_factory=Thunder)
    temperature_c: float = 12.0
    humidity_pct: float = 90.0
    pressure_hpa: float = 1013.0
    hour: float = 22.0              # local time of day 0..24
    season: str = "autumn"
    latitude_deg: float = 50.0
    storm_arc: str = "steady"       # arriving|steady|passing|clearing
    arc_minutes: float = 20.0
    cloud_okta: float = 8.0         # visual/context only
    visibility_km: float = 5.0      # visual/context only


@dataclass
class SceneObject:
    """Something placed in the 3-D environment (from presets or the 4D composer)."""
    id: str
    type: str                       # surface|water|tree|grass|gutter|downspout|eave|wire|chime|flag|car|tent|umbrella|roof|window|storm|wall
    material: str = ""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)   # world metres (x east, y north, z up)
    size: tuple[float, float, float] = (1.0, 1.0, 0.0)
    rotation_deg: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class Place:
    preset: str = "garden"
    shelter: str = "open"           # open|tree|umbrella|tent|porch|car|room_open|room_closed|attic|cave|underpass
    surfaces: dict = field(default_factory=lambda: {"grass": 40.0, "soil": 20.0, "leaves": 20.0, "puddle": 10.0, "concrete": 10.0})
    vegetation: dict = field(default_factory=lambda: {"deciduous": 3, "conifer": 0, "distance_m": 12.0})
    emitters: list = field(default_factory=lambda: [{"type": "gutter", "distance_m": 4.0}, {"type": "eave", "distance_m": 2.5}])
    objects: list[SceneObject] = field(default_factory=list)
    z0_m: Optional[float] = None    # roughness length (None = from preset)
    ground: str = "grass"           # for propagation: grass|soil|asphalt|water|snow_fresh|snow_old|forest_floor
    biophony: dict = field(default_factory=lambda: {"birds": "auto", "insects": "auto", "frogs": "auto", "traffic": 0.0, "region": "temperate_europe"})
    aftermath_tau_min: float = 8.0
    window_open: float = 0.0        # 0..1 for room shelters
    wipers: str = "off"             # car: off|slow|fast


@dataclass
class Listener:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)   # world metres; z = ground level at feet
    height_m: float = 1.7
    heading_deg: float = 0.0        # direction the listener faces, deg true
    render: str = "stereo"          # stereo|binaural|5.1|7.1.4|foa|hoa3|7.1|mono
    width: float = 1.0
    hrtf: str = "builtin"           # builtin spherical-head model | path to .sofa


@dataclass
class Keyframe:
    at: float                       # seconds
    path: str                       # dotted path, e.g. "weather.precipitation.rate_mm_h"
    value: Any
    curve: str = "linear"           # linear|log|hold|smooth


@dataclass
class RenderSettings:
    sample_rate: int = 48_000
    quality: str = "studio"         # mobile|game|studio|cinematic
    ambi_order: int = 3
    k_objects: int = 64
    bit_depth: int = 24             # 16|24|32 (32 = float)
    destination: str = "raw"        # raw|youtube|spotify|apple_music|podcast|sleep|broadcast_eu|broadcast_us|streaming_broadcaster|online_streaming|post_stems|game|vr
    stems: bool = False


@dataclass
class Scene:
    version: int = SCHEMA_VERSION
    name: str = "Untitled weather"
    seed: int = 0x5EEDCAFE
    weather: Weather = field(default_factory=Weather)
    place: Place = field(default_factory=Place)
    listener: Listener = field(default_factory=Listener)
    mode: str = "none"              # none|sleep|focus|calm|asmr
    timeline: list[Keyframe] = field(default_factory=list)
    render: RenderSettings = field(default_factory=RenderSettings)
    # Auto/Offset/Pinned state for the parameter graph: path -> {"state": ..., "offset": ..., "value": ...}
    overrides: dict = field(default_factory=dict)
    vibe: dict = field(default_factory=dict)   # Tier-0 macro values (intensity, closeness, wildness) if driven by vibe

    # ---- helpers -------------------------------------------------------
    def copy(self) -> "Scene":
        return copy.deepcopy(self)

    def get(self, path: str) -> Any:
        obj: Any = self
        for part in path.split("."):
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        return obj

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        obj: Any = self
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        if isinstance(obj, dict):
            obj[parts[-1]] = value
        else:
            setattr(obj, parts[-1], value)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rainscene"] = d.pop("version")
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        """Build a Scene from a `to_dict()`-style mapping. Unknown keys are ignored at every level;
        the format version may be given as 'rainscene' (the file key) or 'version'. The input is
        deep-copied, so the scene never shares mutable dicts or lists with the caller."""
        d = copy.deepcopy(dict(d))
        if "rainscene" in d:
            d["version"] = d.pop("rainscene")
        return _build(cls, d)

    @classmethod
    def from_json(cls, text: str) -> "Scene":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str) -> "Scene":
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional dependency
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_json(text)

    def save(self, path: str) -> None:
        if path.endswith((".yaml", ".yml")):
            import yaml
            text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        else:
            text = self.to_json()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)


def _build(tp, data):
    """Recursively build dataclasses from dicts, ignoring unknown keys."""
    if not is_dataclass(tp) or not isinstance(data, dict):
        return data
    kwargs = {}
    for f in fields(tp):
        name = f.name
        if name not in data:
            continue
        val = data[name]
        sub = _FIELD_TYPES.get((tp.__name__, name))
        if sub is not None and val is None:
            continue                        # an empty YAML section ("weather:") keeps the defaults
        if sub is not None:
            if isinstance(val, list):
                val = [_build(sub, v) for v in val]
            else:
                val = _build(sub, val)
        elif isinstance(val, list) and name in ("position", "size"):
            val = tuple(val)
        kwargs[name] = val
    return tp(**kwargs)


# Nested dataclass fields (the annotations are strings under `from __future__ import annotations`,
# so the mapping is explicit rather than introspected).
_FIELD_TYPES = {
    ("Scene", "weather"): Weather,
    ("Scene", "place"): Place,
    ("Scene", "listener"): Listener,
    ("Scene", "timeline"): Keyframe,
    ("Scene", "render"): RenderSettings,
    ("Weather", "precipitation"): Precipitation,
    ("Weather", "wind"): Wind,
    ("Weather", "thunder"): Thunder,
    ("Place", "objects"): SceneObject,
}
