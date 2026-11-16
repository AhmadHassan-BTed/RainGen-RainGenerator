"""raingen: physically based weather & environment sound engine (Python reference implementation).

Describe the weather the way you would experience it and get endless, never-repeating,
studio-grade sound: rain from drop-size distributions striking real materials, wind heard
through objects, thunder from fractal lightning channels, shelters, rooms and the aftermath,
rendered to stereo, binaural, surround or ambisonics with broadcast-correct loudness.

    import raingen as rg
    rg.render("heavy rain on a tent, gusty, distant thunder", minutes=10, out="tent.flac",
              destination="youtube")

The public API lives in raingen.api; names are imported lazily (PEP 562) so `import raingen`
is cheap and the engine's subpackages can import `raingen.__version__` without cycles.
"""

from __future__ import annotations

import importlib

__version__ = "3.0.0.dev0"

_API_NAMES = (
    "Scene", "Weather", "Precipitation", "Wind", "Thunder", "Place", "SceneObject", "Listener", "Keyframe",
    "RenderSettings", "Engine", "Renderer", "RenderReport", "SceneParseError",
    "preset", "list_presets", "scene_from_text", "scene_from_metar", "scene_from_live", "load_scene",
    "render", "render_file", "render_array", "play", "beaufort", "variations", "describe", "one_shots",
    "apply_vibe", "apply_mode", "manifest", "destinations", "PARAMS", "get_param", "to_json_schema",
    "osc_namespace", "wwise_manifest", "fmod_manifest", "markdown_reference", "duration_seconds",
)
_OTHER = {
    "RngFactory": ("raingen.core.rng", "RngFactory"),
    "STEMS": ("raingen.core.constants", "STEMS"),
    "SAMPLE_RATE": ("raingen.core.constants", "SAMPLE_RATE"),
    "DESTINATIONS": ("raingen.master.destinations", "DESTINATIONS"),
    "PRESETS": ("raingen.scene.presets", "PRESETS"),
    "serve_osc": ("raingen.control.osc", "serve_osc"),
}

# Subpackages and modules reachable as attributes after a bare `import raingen` (raingen.api ...).
# `render` is listed with the API names instead: the raingen.render submodule is callable and is
# itself raingen.api.render, so the package attribute is the same object whichever import binds it.
_SUBMODULES = ("api", "cli", "control", "core", "engine", "io", "live", "master", "physics", "scene",
               "spatial", "synth", "wellness")

__all__ = ["__version__", *_API_NAMES, *_OTHER]


def __getattr__(name: str):
    if name in _SUBMODULES:
        # importing a submodule binds it on this package, so later lookups never get here
        return importlib.import_module(f"raingen.{name}")
    if name in _API_NAMES:
        value = getattr(importlib.import_module("raingen.api"), name)
    elif name in _OTHER:
        module, attr = _OTHER[name]
        value = getattr(importlib.import_module(module), attr)
    else:
        raise AttributeError(f"module 'raingen' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
