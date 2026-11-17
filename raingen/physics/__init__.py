"""Physics layer: drop-size distributions, atmosphere, wind, wetness, storms, world, layout.

Turns a Scene (human units) into per-block physical state (WeatherState) and the
geometry generators act upon (SourcePatch list). No audio is produced here.
"""

from .atmosphere import (
    AirAbsorptionFilter,
    air_absorption_db,
    air_absorption_sos,
    flash_to_bang_s,
    ground_effect_sos,
    iso9613_alpha_db_per_m,
    refraction_gain_db,
    shadow_distance_m,
    spreading_gain,
)
from .dsd import (
    CLASS_NAMES,
    class_bounds,
    class_flux,
    dsd_from_scene,
    dsd_with_dm,
    fall_speed,
    fall_speed_from_height,
    number_density,
    rain_rate,
    sample_diameters,
    total_flux,
)
from .layout import (
    build_layout,
    exposure_by_material,
    listener_to_world,
    world_to_listener,
)
from .storm import StormSystem
from .wetness import DripNode, WetnessStores, tate_drop_diameter_mm
from .wind import (
    BEAUFORT,
    GustProcess,
    beaufort_description,
    beaufort_to_mps,
    gust_factor_for,
    mps_to_beaufort,
    terrain_z0,
    turbulence_intensity_for,
    wind_at_height,
)
from .world import World

__all__ = [
    "AirAbsorptionFilter", "air_absorption_db", "air_absorption_sos", "flash_to_bang_s", "ground_effect_sos",
    "iso9613_alpha_db_per_m", "refraction_gain_db", "shadow_distance_m", "spreading_gain",
    "CLASS_NAMES", "class_bounds", "class_flux", "dsd_from_scene", "dsd_with_dm", "fall_speed",
    "fall_speed_from_height", "number_density", "rain_rate", "sample_diameters", "total_flux",
    "build_layout", "exposure_by_material", "listener_to_world", "world_to_listener",
    "StormSystem", "DripNode", "WetnessStores", "tate_drop_diameter_mm",
    "BEAUFORT", "GustProcess", "beaufort_description", "beaufort_to_mps", "gust_factor_for",
    "mps_to_beaufort", "terrain_z0", "turbulence_intensity_for", "wind_at_height",
    "World",
]
