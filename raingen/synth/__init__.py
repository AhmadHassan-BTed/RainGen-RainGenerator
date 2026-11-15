"""Sound generators: rain, drips, gutters, hail, snow, wind, vegetation, aeolian tones,
thunder, biophony, traffic and shelters, plus the kernels and materials they share.

`make_generators(scene, layout, rngf)` (raingen.synth.registry) builds every generator a
scene needs. Names are resolved lazily (PEP 562) so importing one submodule, e.g.
`raingen.synth.materials`, never pulls in the whole synthesis stack and the physics and
synthesis packages can import each other's submodules without cycles.
"""

from __future__ import annotations

import importlib

_LAZY = {
    "make_generators": ("registry", "make_generators"),
    "generator_names": ("registry", "generator_names"),
    "describe_generators": ("registry", "describe_generators"),
    "make_rain_generators": ("rain", "make_rain_generators"),
    "make_environment_generators": ("environment", "make_environment_generators"),
    "make_shelter_transfer": ("environment", "make_shelter_transfer"),
    "make_room": ("environment", "make_room"),
    "RainPatchGenerator": ("rain", "RainPatchGenerator"),
    "RainBedGenerator": ("rain", "RainBedGenerator"),
    "DripGenerator": ("rain", "DripGenerator"),
    "GutterGenerator": ("rain", "GutterGenerator"),
    "HailGenerator": ("rain", "HailGenerator"),
    "SnowGenerator": ("rain", "SnowGenerator"),
    "WindGenerator": ("wind", "WindGenerator"),
    "MicRumbleGenerator": ("wind", "MicRumbleGenerator"),
    "VegetationGenerator": ("vegetation", "VegetationGenerator"),
    "AeolianGenerator": ("aeolian", "AeolianGenerator"),
    "ThunderGenerator": ("thunder", "ThunderGenerator"),
    "lightning_channel": ("thunder", "lightning_channel"),
    "BiophonyGenerator": ("biophony", "BiophonyGenerator"),
    "TrafficGenerator": ("biophony", "TrafficGenerator"),
    "ShelterTransfer": ("places", "ShelterTransfer"),
    "Room": ("places", "Room"),
    "room_for_shelter": ("places", "room_for_shelter"),
    "CarWipersGenerator": ("places", "CarWipersGenerator"),
    "Material": ("materials", "Material"),
    "MATERIALS": ("materials", "MATERIALS"),
    "get_material": ("materials", "get_material"),
    "KernelBank": ("kernels", "KernelBank"),
    "cox_offsets": ("events", "cox_offsets"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'raingen.synth' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
