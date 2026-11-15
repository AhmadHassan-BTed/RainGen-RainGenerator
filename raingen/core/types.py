"""Shared engine data contracts.

These dataclasses are the seams between modules:

    Scene (user intent, human units)          raingen.scene.schema
      -> World.step()  -> WeatherState         raingen.physics.world   (per block)
      -> build_layout() -> list[SourcePatch]   raingen.physics.layout  (geometry)
      -> Generator.render(ctx) -> list[Emission]   raingen.synth.*
      -> Encoder -> SH bus per stem -> Decoder -> Master -> Writer

Coordinate system (listener-centred, metres): +x forward (listener heading),
+y left, +z up. Azimuth is measured counter-clockwise from +x (0 = front,
+pi/2 = left), elevation up from the horizontal plane. This matches AmbiX.
Compass bearings (deg true, clockwise from north) are converted with
`bearing_to_azimuth`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from .rng import RngFactory


def bearing_to_azimuth(bearing_deg: float, heading_deg: float) -> float:
    """Compass bearing (deg true, clockwise) of a direction -> listener azimuth (rad, CCW, 0=front)."""
    return math.radians(-(bearing_deg - heading_deg))


@dataclass
class DSDParams:
    """Normalized-gamma drop-size distribution N(D) = Nw f(mu) (D/Dm)^mu exp(-(4+mu) D/Dm)."""
    nw: float = 8000.0          # m^-3 mm^-1
    dm_mm: float = 1.4          # mass-weighted mean diameter
    mu: float = 3.0
    d_min_mm: float = 0.3
    d_max_mm: float = 6.0


@dataclass
class StormCellState:
    id: str
    distance_km: float
    bearing_deg: float
    flash_rate_per_min: float
    stroke_intensity: float     # 0..1
    cg_fraction: float          # share of cloud-to-ground flashes
    rain_contribution: float    # 0..1 how much this cell drives local rain


@dataclass
class WeatherState:
    """Physically resolved weather at the start of one block (what generators read)."""
    t: float                               # seconds since render start
    precip_type: str                       # none|drizzle|rain|showers|freezing_rain|sleet|snow|graupel|hail
    rain_rate_mm_h: float                  # instantaneous (after showers/arc/gust modulation)
    dsd: DSDParams
    class_flux: dict[str, float]           # impacts / m^2 / s per acoustic class:
                                           # 'minuscule','small','mid','large','very_large'
    hail_size_mm: float
    wind_u10_mps: float                    # reported mean wind at 10 m
    wind_listener_mps: float               # instantaneous wind at listener height (incl. gust)
    gust: float                            # instantaneous gust multiplier (1 = mean)
    wind_from_deg: float                   # meteorological 'from', deg true
    wind_azimuth: float                    # azimuth (rad, listener frame) the wind comes FROM
    turbulence_intensity: float
    temperature_c: float
    humidity_pct: float
    pressure_hpa: float
    speed_of_sound: float
    hour: float                            # local time of day 0..24
    season: str                            # spring|summer|autumn|winter
    is_day: bool
    storms: list[StormCellState] = field(default_factory=list)
    wetness: dict[str, float] = field(default_factory=dict)        # material -> 0..1 (film)
    puddle_depth_mm: dict[str, float] = field(default_factory=dict)
    canopy_storage_mm: float = 0.0
    roof_flow_l_s: float = 0.0
    stability: str = "neutral"             # unstable|neutral|stable (refraction regime)
    intensity_envelope: float = 1.0        # slow storm/shower envelope 0..1+ (for beds)


@dataclass
class SourcePatch:
    """A surface (or object) that rain/wind act upon, placed around the listener."""
    id: str
    kind: str                     # 'surface' | 'water' | 'membrane' | 'vegetation' | 'wire' | 'gutter' | 'drip_edge' | 'vehicle_body'
    material: str                 # key into raingen.synth.materials.MATERIALS
    position: tuple[float, float, float]   # listener frame, metres
    area_m2: float = 1.0
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    exposure: float = 1.0         # 0..1 how much open sky / rain reaches it
    sheltered_by: Optional[str] = None     # id of a covering patch (canopy, roof)
    interior: bool = False        # True if the patch radiates into the listener's enclosure
    params: dict = field(default_factory=dict)  # free extra parameters (e.g. species, diameter_m, length_m)

    @property
    def distance(self) -> float:
        x, y, z = self.position
        return max(0.05, math.sqrt(x * x + y * y + z * z))

    @property
    def azimuth(self) -> float:
        x, y, _ = self.position
        return math.atan2(y, x)

    @property
    def elevation(self) -> float:
        x, y, z = self.position
        return math.atan2(z, math.hypot(x, y))


@dataclass
class Emission:
    """Audio produced by a generator for one block, as heard at the listener position.

    Level, propagation (distance, air absorption, shelter transmission) are already
    applied by the generator; the encoder only adds direction and spread.
    """
    signal: np.ndarray            # (n,) mono, or (N_AMBI, n) if is_ambisonic
    azimuth: float = 0.0
    elevation: float = 0.0
    distance: float = 1.0
    spread: float = 0.0           # 0 = point source, 1 = fully diffuse
    stem: str = "rain_near"
    is_ambisonic: bool = False
    source_id: str = ""


@dataclass
class EventMarker:
    """Discrete events exported for captions, cue sheets and visual sync."""
    t: float
    kind: str                     # 'lightning' | 'thunder' | 'gust' | 'drip' | ...
    data: dict = field(default_factory=dict)


@dataclass
class BlockContext:
    sr: int
    n: int                        # frames in this block
    index: int                    # block index from render start
    t0: float                     # block start time (s)
    weather: WeatherState
    rng: RngFactory
    listener_height_m: float = 1.7
    shelter: str = "open"
    markers: list[EventMarker] = field(default_factory=list)  # generators append here


class Generator(Protocol):
    """Anything that produces sound each block."""
    name: str

    def render(self, ctx: BlockContext) -> list[Emission]:
        ...
