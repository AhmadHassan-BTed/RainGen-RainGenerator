"""Generator registry: every sound source a scene needs, in one deterministic list.

`make_generators(scene, layout, rngf)` is the single entry point the renderer and the
real-time engine use. It concatenates the precipitation family (raingen.synth.rain:
rain patches with level-of-detail assignment after Zhang, Raghuvanshi, Snyder & Marschner
2019, the diffuse bed, drips, gutters, hail, snow) and the environment family
(raingen.synth.environment: wind bed, vegetation, aeolian tones, mic rumble, thunder,
biophony, traffic, wipers). Order is fixed (rain first, then environment) and every
generator draws only from `rngf.get(layer, block)` streams keyed by its own layer name,
so adding or removing one generator never changes the sound of another (MASTER_PLAN 9.4).
"""

from __future__ import annotations

from typing import Iterable, Optional

from raingen.core.constants import SAMPLE_RATE
from raingen.core.rng import RngFactory

__all__ = ["make_generators", "generator_names", "describe_generators"]


def make_generators(scene, layout: Optional[list] = None, rngf: Optional[RngFactory] = None, *,
                    precip_types=None) -> list:
    """Rain + environment generators for `scene` around `layout`.

    layout        the SourcePatch list from raingen.physics.layout.build_layout (built here if None)
    rngf          the scene's RngFactory (RngFactory(scene.seed) if None)
    precip_types  forwarded to make_rain_generators: None = the scene's precipitation type plus any
                  timeline keyframes on it; 'all' = prepare every type (live control)
    """
    from raingen.physics.layout import build_layout

    from .environment import make_environment_generators
    from .rain import make_rain_generators

    if layout is None:
        layout = build_layout(scene)
    layout = list(layout)
    if rngf is None:
        rngf = RngFactory(int(scene.seed))
    gens = list(make_rain_generators(scene, layout, rngf, precip_types=precip_types))
    gens.extend(make_environment_generators(scene, layout, rngf, sr=SAMPLE_RATE))
    return gens


def generator_names(generators: Iterable) -> list[str]:
    """The `name` of each generator (class name when a generator has none)."""
    return [str(getattr(g, "name", type(g).__name__)) for g in generators]


def describe_generators(generators: Iterable) -> dict[str, int]:
    """Count of generators per class, e.g. {'RainPatchGenerator': 41, 'ThunderGenerator': 1}."""
    counts: dict[str, int] = {}
    for g in generators:
        k = type(g).__name__
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))
