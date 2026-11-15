"""Environment generators for a scene: wind, plants, aeolian tones, thunder, life, traffic, shelter.

MASTER_PLAN 3.6-3.9: wind is heard through objects (vegetation species spectra, wires,
gaps, chimes, flags, debris), all driven by one advected Kaimal gust field (Taylor's
hypothesis) that is the World's own gust stream; thunder comes from fractal channels of
the storm cells in WeatherState.storms; insects, birds and frogs follow temperature, time,
season, rain and wind; traffic follows the daily curve and road wetness; shelters add their
own sounds (wipers in a car, window gaps, tent fabric).

Stems: 'wind' (bed, vegetation, aeolian, mic rumble), 'thunder', 'ambience' (biophony,
traffic), 'interior' (wipers, gap whistles and fabric of the listener's own shelter).
Exterior stems are filtered for the listener's shelter by places.ShelterTransfer, which the
renderer applies to the stem buses (`make_shelter_transfer`).
"""

from __future__ import annotations

from raingen.core.constants import SAMPLE_RATE

from .aeolian import AeolianGenerator
from .biophony import BiophonyGenerator, TrafficGenerator
from .places import CarWipersGenerator, Room, ShelterTransfer, room_for_shelter
from .thunder import ThunderGenerator
from .vegetation import VegetationGenerator
from .wind import GustField, MicRumbleGenerator, WindGenerator, shelter_of


def _traffic_level(scene) -> float:
    try:
        return float((scene.place.biophony or {}).get("traffic", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def make_environment_generators(scene, layout, rngf, sr: int = SAMPLE_RATE) -> list:
    """All non-precipitation generators for `scene` around `layout` (deterministic per seed).

    Order: wind bed, vegetation, aeolian emitters, optional mic rumble, thunder, biophony,
    traffic, car wipers. Generators whose scene has nothing for them to do are left out,
    except thunder (storm cells can appear mid-render through the timeline).
    """
    if layout is None:
        from raingen.physics.layout import build_layout
        layout = build_layout(scene)
    layout = list(layout)
    field = GustField(scene, rngf, sr)
    gens: list = [WindGenerator(scene, layout, rngf, gust_field=field, sr=sr)]

    veg = VegetationGenerator(scene, layout, rngf, gust_field=field, sr=sr)
    if veg.voices:
        gens.append(veg)
    aeo = AeolianGenerator(scene, layout, rngf, gust_field=field, sr=sr)
    if aeo.voices:
        gens.append(aeo)
    if bool(getattr(scene.weather.wind, "mic_rumble", False)):
        gens.append(MicRumbleGenerator(scene, rngf, sr=sr))

    gens.append(ThunderGenerator(scene, layout, rngf, sr=sr))

    bio = BiophonyGenerator(scene, layout, rngf, sr=sr)
    if bio.insects or bio.cicadas or bio.birds or bio.owls or bio.frogs or bio.chorus is not None:
        gens.append(bio)
    if _traffic_level(scene) > 0.0:
        gens.append(TrafficGenerator(scene, layout, rngf, sr=sr))
    if shelter_of(scene) == "car":
        gens.append(CarWipersGenerator(scene, layout, rngf, sr=sr))
    return gens


def make_shelter_transfer(scene, sr: int = SAMPLE_RATE) -> ShelterTransfer:
    """The exterior-stem transfer for the listener's shelter (identity outdoors)."""
    return ShelterTransfer(scene, sr)


def make_room(scene, sr: int = SAMPLE_RATE):
    """(Room, wet) for the listener's enclosure, or (None, 0.0) outdoors."""
    rt, wet = room_for_shelter(shelter_of(scene), float(getattr(scene.place, "window_open", 0.0) or 0.0))
    if rt <= 0.0 or wet <= 0.0:
        return None, 0.0
    return Room(rt, sr), wet


__all__ = ["make_environment_generators", "make_shelter_transfer", "make_room"]
