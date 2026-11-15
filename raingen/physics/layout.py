"""Scene geometry: the surfaces, shelters, plants and emitters around the listener.

Frames. World: x east, y north, z up (metres; z = 0 is the ground under the
listener's feet when listener.position z = 0). Listener: +x forward along the
heading, +y left, +z up, origin at the ears (listener.position + height). Azimuth
is counter-clockwise from +x, matching AmbiX and `bearing_to_azimuth`.

Level-of-detail footprint (MASTER_PLAN 7.5): for ground impacts heard at height h,
50 % of the intensity comes from within radius h and 90 % from within 3h, so the
ground is cut into rings at ~0.5-1.5 / 1.5-4 / 4-10 / 10-30 m x 8 azimuth sectors,
fine near the ears and coarse far away. Materials are allocated deterministically
from `place.surfaces` (largest remainder with error diffusion between rings, laid
out in contiguous runs so a lawn is a lawn and not a checkerboard).

Exposure. Rain on a tilted surface scales with the projected area (n_z); vertical
faces receive wind-driven rain in proportion to U/v_t (ISO 15927-3 driving-rain
index, ~0.2 U R^0.88) on the windward side only, with a small eddy term on the
leeward side. Shelter footprints (umbrella, canopy, tent, car, porch, buildings,
cave, underpass) remove or reduce exposure of what lies beneath them; forest
canopies intercept 10-40 % of the rain (the rest arrives as throughfall and drips).

Everything is deterministic: the only randomness is RngFactory(scene.seed).get('layout', 0)
drawn in a fixed order.
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.rng import RngFactory
from ..core.types import SourcePatch, bearing_to_azimuth
from .wind import terrain_z0, wind_at_height

RINGS = ((0.5, 1.5), (1.5, 4.0), (4.0, 10.0), (10.0, 30.0))
N_SECTORS = 8
SECTOR_WIDTH = 2.0 * math.pi / N_SECTORS

_MATERIAL_ALIASES = {
    "puddles": "puddle", "pond": "water", "lake": "water", "sea": "water", "ocean": "water",
    "river": "water", "stream": "water", "pool": "water",
    "mud": "soil", "dirt": "soil", "earth": "soil", "sand": "soil", "clay": "soil",
    "forest_floor": "leaves", "leaf_litter": "leaves", "leaf": "leaves", "foliage": "leaves",
    "needle": "needles", "pine_needles": "needles", "lawn": "grass", "meadow": "grass", "moss": "grass",
    "metal": "metal_sheet", "steel": "metal_sheet", "iron": "metal_sheet", "aluminium": "metal_sheet",
    "aluminum": "metal_sheet", "corrugated": "tin_roof", "tin": "tin_roof", "roof": "tile",
    "car": "car_body", "vehicle": "car_body", "tent_fabric": "tent", "tarp": "canvas", "tarpaulin": "canvas",
    "stone": "concrete", "rock": "concrete", "pavement": "concrete", "paving": "concrete",
    "sidewalk": "concrete", "cobbles": "tile", "cobblestone": "tile", "brick": "tile", "slate": "tile",
    "road": "asphalt", "tarmac": "asphalt", "decking": "wood", "deck": "wood", "timber": "wood",
    "snow_fresh": "snow", "snow_old": "snow", "ice": "glass", "gutter": "metal_gutter",
    "glass_pane": "glass", "windowpane": "window", "plexiglass": "plastic", "polycarbonate": "plastic",
}

_SHELTER_ALIASES = {
    "": "open", "none": "open", "outside": "open", "outdoors": "open",
    "under_tree": "tree", "forest": "tree", "canopy": "tree",
    "vehicle": "car", "in_car": "car",
    "veranda": "porch", "verandah": "porch", "awning": "porch",
    "room": "room_closed", "indoors": "room_closed", "house": "room_closed", "inside": "room_closed",
    "window": "room_open", "room_window_open": "room_open",
    "tin_roof": "attic", "loft": "attic", "barn": "attic", "shed": "attic",
    "cave_mouth": "cave", "bridge": "underpass", "tunnel": "underpass",
}

_SPECIES = {
    "deciduous": dict(height=12.0, canopy_r=4.0, material="leaves", lai=4.0, evergreen=False,
                      twig_d=0.004, branch_d=0.03, leaf_size_m=0.08),
    "conifer": dict(height=18.0, canopy_r=2.5, material="needles", lai=6.0, evergreen=True,
                    twig_d=0.003, branch_d=0.025, leaf_size_m=0.001),
    "shrub": dict(height=2.0, canopy_r=1.0, material="leaves", lai=3.0, evergreen=False,
                  twig_d=0.003, branch_d=0.012, leaf_size_m=0.04),
    "hedge": dict(height=1.8, canopy_r=0.8, material="leaves", lai=5.0, evergreen=True,
                  twig_d=0.003, branch_d=0.01, leaf_size_m=0.03),
    "palm": dict(height=10.0, canopy_r=3.0, material="leaves", lai=2.5, evergreen=True,
                 twig_d=0.01, branch_d=0.05, leaf_size_m=1.5),
    "bamboo": dict(height=6.0, canopy_r=1.5, material="leaves", lai=5.0, evergreen=True,
                   twig_d=0.005, branch_d=0.03, leaf_size_m=0.15),
    "grass": dict(height=0.6, canopy_r=2.0, material="grass", lai=2.0, evergreen=True,
                  twig_d=0.002, branch_d=0.004, leaf_size_m=0.2),
}
_SPECIES_ALIASES = {
    "broadleaf": "deciduous", "oak": "deciduous", "beech": "deciduous", "birch": "deciduous",
    "maple": "deciduous", "ash": "deciduous", "poplar": "deciduous", "willow": "deciduous",
    "pine": "conifer", "spruce": "conifer", "fir": "conifer", "cedar": "conifer", "larch": "conifer",
    "evergreen": "conifer", "bush": "shrub", "bushes": "shrub", "shrubs": "shrub",
    "hedges": "hedge", "palms": "palm", "reeds": "grass", "tall_grass": "grass", "grass_tall": "grass",
}
_LEAF_FRACTION = {"spring": 0.7, "summer": 1.0, "autumn": 0.6, "winter": 0.05}

# height (relative to the ears) below which a shelter's footprint covers things
_SHELTER_TOP = {"umbrella": 0.35, "tree": 3.5, "tent": 0.5, "car": 0.3, "porch": 2.5,
                "room_open": 3.0, "room_closed": 3.0, "attic": 1.5, "cave": 6.0, "underpass": 3.0}
_INTERIOR_SHELTERS = ("tent", "car", "room_open", "room_closed", "attic", "cave")


def _norm_name(name) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_material(name) -> str:
    """Map free-form surface names onto material keys ('puddles' -> 'puddle', ...)."""
    key = _norm_name(name)
    if not key:
        return "concrete"
    return _MATERIAL_ALIASES.get(key, key)


def normalize_shelter(name) -> str:
    key = _norm_name(name)
    return _SHELTER_ALIASES.get(key, key)


def _species_key(name) -> str:
    key = _norm_name(name)
    return _SPECIES_ALIASES.get(key, key)


# ------------------------------------------------------------------ frames
def _heading_vectors(heading_deg: float):
    h = math.radians(float(heading_deg))
    return (math.sin(h), math.cos(h)), (-math.cos(h), math.sin(h))


def _ears(listener):
    pos = tuple(listener.position) + (0.0, 0.0, 0.0)
    return float(pos[0]), float(pos[1]), float(pos[2]) + float(listener.height_m)


def world_to_listener(pos_world, listener) -> tuple[float, float, float]:
    """World (x east, y north, z up) -> listener frame (+x forward, +y left, +z up, ears origin)."""
    p = tuple(pos_world) + (0.0, 0.0, 0.0)
    ex, ey, ez = _ears(listener)
    dx, dy, dz = float(p[0]) - ex, float(p[1]) - ey, float(p[2]) - ez
    fwd, left = _heading_vectors(listener.heading_deg)
    return (dx * fwd[0] + dy * fwd[1], dx * left[0] + dy * left[1], dz)


def listener_to_world(pos_listener, listener) -> tuple[float, float, float]:
    """Inverse of `world_to_listener`."""
    x, y, z = (tuple(pos_listener) + (0.0, 0.0, 0.0))[:3]
    ex, ey, ez = _ears(listener)
    fwd, left = _heading_vectors(listener.heading_deg)
    return (ex + x * fwd[0] + y * left[0], ey + x * fwd[1] + y * left[1], ez + z)


# ------------------------------------------------------------------ context
class _Ctx:
    def __init__(self, scene):
        self.scene = scene
        self.listener = scene.listener
        self.h = max(float(scene.listener.height_m), 0.1)
        self.fwd, self.left = _heading_vectors(scene.listener.heading_deg)
        wind = scene.weather.wind
        self.u10 = max(float(wind.speed_mps), 0.0)
        fr = math.radians(float(wind.from_deg))
        self.wf = (math.sin(fr), math.cos(fr))          # unit vector pointing to where the wind comes from
        place = scene.place
        self.z0 = float(place.z0_m) if place.z0_m is not None else terrain_z0(place.preset)
        self.season = _norm_name(scene.weather.season) or "summer"
        self.leaf_fraction = _LEAF_FRACTION.get(self.season, 1.0)
        self.shelter = normalize_shelter(place.shelter)
        self.preset = _norm_name(place.preset)
        self.rng = RngFactory(scene.seed).get("layout", 0)
        self.ground_patches: list[SourcePatch] = []
        veg = place.vegetation or {}
        dec = _num(veg.get("deciduous", 0))
        con = _num(veg.get("conifer", 0))
        self.canopy_species = "conifer" if con > dec else "deciduous"

    def to_world_dir(self, x: float, y: float):
        return (x * self.fwd[0] + y * self.left[0], x * self.fwd[1] + y * self.left[1])

    def from_world_dir(self, e: float, n: float):
        return (e * self.fwd[0] + n * self.fwd[1], e * self.left[0] + n * self.left[1])

    def upwind(self, x: float, y: float) -> float:
        wx, wy = self.to_world_dir(x, y)
        return wx * self.wf[0] + wy * self.wf[1]

    def exposure_for_normal(self, normal, height_above_ground: float) -> float:
        nx, ny, nz = _unit(normal)
        horizontal = max(nz, 0.0)
        wx, wy = self.to_world_dir(nx, ny)
        hn = math.hypot(wx, wy)
        vertical = 0.0
        if hn > 1e-6:
            cos_w = (wx * self.wf[0] + wy * self.wf[1]) / hn
            u = wind_at_height(self.u10, max(height_above_ground, 0.5), self.z0)
            drf = min(1.0, u / 6.0)                     # horizontal / vertical rain flux ~ U / v_t
            vertical = drf * max(cos_w, 0.0) + 0.04
        e = horizontal + (1.0 - abs(nz)) * vertical
        return float(min(max(e, 0.0), 1.0))


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _unit(v):
    x, y, z = (float(c) for c in (tuple(v) + (0.0, 0.0, 0.0))[:3])
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0, 0.0, 1.0
    return x / n, y / n, z / n


def _patch(id, kind, material, pos, area=1.0, normal=(0.0, 0.0, 1.0), exposure=1.0,
           sheltered_by=None, interior=False, params=None) -> SourcePatch:
    return SourcePatch(
        id=str(id), kind=str(kind), material=str(material),
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        area_m2=float(max(area, 1e-4)), normal=tuple(float(c) for c in _unit(normal)),
        exposure=float(min(max(exposure, 0.0), 1.0)), sheltered_by=sheltered_by,
        interior=bool(interior), params=dict(params or {}),
    )


# ------------------------------------------------------------------ shelter footprints
def _shelter_cover(ctx: _Ctx, x: float, y: float):
    """(cover 0..1, covering patch id, floor material or None) for a ground point."""
    s = ctx.shelter
    r = math.hypot(x, y)
    if s == "umbrella":
        if r <= 0.55:
            return 0.9, "shelter.umbrella", None
    elif s == "tree":
        if r <= 4.0:
            return _canopy_interception(ctx, ctx.canopy_species), "shelter.tree.canopy", None
    elif s == "tent":
        if abs(x) <= 1.1 and abs(y) <= 0.75:
            return 1.0, "shelter.tent.roof_l", "fabric"
    elif s == "car":
        if -2.4 <= x <= 2.0 and -1.25 <= y <= 0.65:
            return 1.0, "shelter.car.roof", "asphalt"
    elif s == "porch":
        if x < -1.0 and abs(y) <= 8.0:
            return 1.0, "shelter.porch.wall", None
        if -1.0 <= x <= 2.0 and abs(y) <= 3.0:
            return 0.95, "shelter.porch.roof_c", "wood"
    elif s in ("room_open", "room_closed"):
        if -8.0 <= x <= 2.2 and abs(y) <= 6.0:
            return 1.0, "shelter.room.roof", None
    elif s == "attic":
        if abs(x) <= 5.0 and abs(y) <= 4.0:
            return 1.0, "shelter.attic.roof_lf", None
    elif s == "cave":
        if x < 3.0:
            return 1.0, "shelter.cave.ceiling", ("gravel" if abs(y) <= 2.5 else None)
    elif s == "underpass":
        if abs(x) <= 8.0 and abs(y) <= 6.0:
            return 1.0, "shelter.underpass.deck", "asphalt"
    return 0.0, None, None


def _canopy_interception(ctx: _Ctx, species: str) -> float:
    """Fraction of rain intercepted by a canopy overhead (10-40 %)."""
    prof = _SPECIES.get(species, _SPECIES["deciduous"])
    leaf = 1.0 if prof["evergreen"] else ctx.leaf_fraction
    return 0.10 + 0.30 * min(max(leaf * prof["lai"] / 5.0, 0.0), 1.0)


# ------------------------------------------------------------------ ground rings
def _surface_shares(place) -> list[tuple[str, float]]:
    shares: dict[str, float] = {}
    for k, v in (place.surfaces or {}).items():
        val = _num(v)
        if val <= 0.0:
            continue
        m = normalize_material(k)
        shares[m] = shares.get(m, 0.0) + val
    if not shares:
        shares = {normalize_material(getattr(place, "ground", "grass") or "grass"): 1.0}
    total = sum(shares.values())
    return sorted(((m, v / total) for m, v in shares.items()), key=lambda e: (-e[1], e[0]))


def _allocate_ring_counts(shares, n_rings: int, n_sectors: int) -> list[dict]:
    """Sector counts per ring: largest remainder with error diffusion across rings."""
    carry = {m: 0.0 for m, _ in shares}
    out = []
    for _ in range(n_rings):
        want = {m: f * n_sectors + carry[m] for m, f in shares}
        counts = {m: int(math.floor(max(want[m], 0.0))) for m, _ in shares}
        remaining = n_sectors - sum(counts.values())
        if remaining > 0:
            order = sorted(counts, key=lambda m: (-(want[m] - counts[m]), m))
            for i in range(remaining):
                counts[order[i % len(order)]] += 1
        elif remaining < 0:
            order = sorted((m for m in counts if counts[m] > 0), key=lambda m: (want[m] - counts[m], m))
            i = 0
            while remaining < 0 and order:
                m = order[i % len(order)]
                if counts[m] > 0:
                    counts[m] -= 1
                    remaining += 1
                i += 1
        for m in counts:
            carry[m] = max(min(want[m] - counts[m], 1.5), -1.5)
        out.append(counts)
    return out


def _build_ground(ctx: _Ctx) -> list[SourcePatch]:
    place = ctx.scene.place
    shares = _surface_shares(place)
    ring_counts = _allocate_ring_counts(shares, len(RINGS), N_SECTORS)
    patches = []
    for i, ((r1, r2), counts) in enumerate(zip(RINGS, ring_counts)):
        mats = [m for m, _ in shares if counts.get(m, 0) > 0]
        perm = ctx.rng.permutation(len(mats)) if mats else []
        rot = int(ctx.rng.integers(0, N_SECTORS))
        seq = []
        for idx in perm:
            m = mats[int(idx)]
            seq.extend([m] * counts[m])
        while len(seq) < N_SECTORS:
            seq.append(shares[0][0])
        r_c = (2.0 / 3.0) * (r2 ** 3 - r1 ** 3) / (r2 ** 2 - r1 ** 2)
        r_c *= math.sin(SECTOR_WIDTH / 2.0) / (SECTOR_WIDTH / 2.0)
        area = 0.5 * SECTOR_WIDTH * (r2 * r2 - r1 * r1)
        for j in range(N_SECTORS):
            mat = seq[(j - rot) % N_SECTORS]
            az = j * SECTOR_WIDTH
            if az > math.pi:
                az -= 2.0 * math.pi
            x, y = r_c * math.cos(az), r_c * math.sin(az)
            # sample the shelter footprint over the patch
            covers, floors, covering = [], [], None
            for fr in (1.0 / 6.0, 0.5, 5.0 / 6.0):
                rr = r1 + (r2 - r1) * fr
                for fa in (0.125, 0.375, 0.625, 0.875):
                    aa = az - SECTOR_WIDTH / 2.0 + SECTOR_WIDTH * fa
                    c, cid, floor = _shelter_cover(ctx, rr * math.cos(aa), rr * math.sin(aa))
                    covers.append(c)
                    floors.append(floor)
                    if cid is not None:
                        covering = cid
            cover = sum(covers) / len(covers)
            floor_hits = [f for f in floors if f is not None]
            if len(floor_hits) * 2 >= len(floors):
                mat = max(sorted(set(floor_hits)), key=floor_hits.count)
            kind = "water" if mat in ("water", "puddle") else "surface"
            patches.append(_patch(
                f"ground.r{i}.s{j}", kind, mat, (x, y, -ctx.h), area=area,
                exposure=1.0 - cover, sheltered_by=covering if cover > 0.5 else None,
                params={"ground": True, "ring": i, "sector": j, "r_inner": r1, "r_outer": r2,
                        "az_center": az, "az_width": SECTOR_WIDTH, "height_m": 0.0,
                        "radial_extent_m": r2 - r1, "cover": cover},
            ))
    ctx.ground_patches = patches
    return patches


def ground_material_at(patches, x: float, y: float, default: str = "soil") -> str:
    """Material of the ground ring patch under listener-frame point (x, y)."""
    r = math.hypot(x, y)
    i = len(RINGS) - 1
    for k, (r1, r2) in enumerate(RINGS):
        if r < r2:
            i = k
            break
    j = int(round(math.atan2(y, x) / SECTOR_WIDTH)) % N_SECTORS
    pid = f"ground.r{i}.s{j}"
    for p in patches:
        if p.id == pid:
            return p.material
    return default


def _drip_target(ctx: _Ctx, x: float, y: float, prefer_puddle: bool = False) -> str:
    mat = ground_material_at(ctx.ground_patches, x, y)
    if prefer_puddle and mat in ("soil", "grass", "leaves", "needles", "gravel"):
        return "puddle"
    return mat


def _drip(ctx: _Ctx, pid: str, edge_xyz, catchment_m2: float, edge_radius_mm: float, feed: str,
          source: str, edge_material: str, prefer_puddle: bool = False, target: Optional[str] = None,
          interior: bool = False, impact_z: Optional[float] = None, extra: Optional[dict] = None,
          n_points: int = 1, length_m: float = 0.2) -> SourcePatch:
    """A drip point (or a short run of `n_points` drip points along `length_m` of edge).

    The patch sits where the drops land; the edge itself is in params['edge_position'].
    `catchment_m2` is per drip point.
    """
    ex, ey, ez = edge_xyz
    tgt = target or _drip_target(ctx, ex, ey, prefer_puddle)
    z_imp = -ctx.h if impact_z is None else impact_z
    params = {"catchment_m2": float(catchment_m2), "edge_radius_mm": float(edge_radius_mm),
              "fall_height_m": float(max(ez - z_imp, 0.05)), "feed": feed, "source": source,
              "edge_material": edge_material, "roof_material": edge_material,
              "edge_position": (float(ex), float(ey), float(ez)),
              "target_material": tgt, "n_points": int(max(n_points, 1)), "length_m": float(max(length_m, 0.05))}
    if extra:
        params.update(extra)
    return _patch(pid, "drip_edge", tgt, (ex, ey, z_imp), area=0.01, exposure=1.0,
                  interior=interior, params=params)


# ------------------------------------------------------------------ shelters
def _vert(ctx: _Ctx, normal, z_rel: float) -> float:
    return ctx.exposure_for_normal(normal, z_rel + ctx.h)


def _build_shelter(ctx: _Ctx) -> list[SourcePatch]:
    s = ctx.shelter
    h = ctx.h
    place = ctx.scene.place
    out: list[SourcePatch] = []
    if s == "umbrella":
        r = 0.55
        out.append(_patch("shelter.umbrella", "membrane", "umbrella", (0.0, 0.0, 0.35),
                          area=math.pi * r * r, exposure=1.0,
                          params={"shelter": "umbrella", "radius_m": r, "height_m": h + 0.35,
                                  "near_field": True, "ribs": 8}))
        for k in range(8):
            a = (k + 0.5) * math.pi / 4.0
            out.append(_drip(ctx, f"shelter.umbrella.drip{k}", (r * math.cos(a), r * math.sin(a), 0.12),
                             catchment_m2=math.pi * r * r / 8.0, edge_radius_mm=1.2, feed="rain",
                             source="shelter.umbrella", edge_material="umbrella",
                             n_points=1, length_m=0.05, extra={"rib": k}))
    elif s == "tree":
        sp = ctx.canopy_species
        prof = _SPECIES[sp]
        leaf = 1.0 if prof["evergreen"] else ctx.leaf_fraction
        mat = prof["material"] if leaf >= 0.2 else "wood"
        z_c = max(1.5, 5.0 - h)
        out.append(_patch("shelter.tree.canopy", "vegetation", mat, (0.5, 0.3, z_c),
                          area=math.pi * 16.0, exposure=1.0,
                          params={"species": sp, "canopy_radius_m": 4.0, "height_m": 9.0, "overhead": True,
                                  "canopy": True, "fall_height_m": z_c + h, "edge_radius_mm": 2.2,
                                  "catchment_m2": 0.01, "n_points": 8,
                                  "leaf_on": leaf >= 0.2, "leaf_fraction": leaf, "lai": prof["lai"] * leaf,
                                  "twig_diameter_m": prof["twig_d"], "branch_diameter_m": prof["branch_d"],
                                  "leaf_size_m": prof["leaf_size_m"],
                                  "interception": _canopy_interception(ctx, sp)}))
        for k in range(6):
            rr = float(ctx.rng.uniform(0.6, 3.6))
            aa = float(ctx.rng.uniform(-math.pi, math.pi))
            out.append(_drip(ctx, f"shelter.tree.drip{k}", (rr * math.cos(aa), rr * math.sin(aa), 2.5 - h),
                             catchment_m2=float(ctx.rng.uniform(0.01, 0.03)), edge_radius_mm=1.5,
                             feed="canopy", source="shelter.tree.canopy", edge_material=mat,
                             n_points=2, length_m=0.6))
    elif s == "tent":
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            n = (0.0, 0.71 * sgn, 0.71)
            out.append(_patch(f"shelter.tent.roof_{side}", "membrane", "tent", (0.0, 0.45 * sgn, 0.35),
                              area=2.4, normal=n, exposure=_vert(ctx, n, 0.35), interior=True,
                              params={"shelter": "tent", "tension": "medium", "panel": "roof"}))
        for side, sgn in (("front", 1.0), ("back", -1.0)):
            n = (sgn, 0.0, 0.0)
            out.append(_patch(f"shelter.tent.{side}", "membrane", "tent", (1.1 * sgn, 0.0, 0.0),
                              area=1.1, normal=n, exposure=_vert(ctx, n, 0.0), interior=True,
                              params={"shelter": "tent", "tension": "low", "panel": "wall"}))
        for k, (dx, sgn) in enumerate(((0.6, 1.0), (-0.6, 1.0), (0.6, -1.0), (-0.6, -1.0))):
            out.append(_drip(ctx, f"shelter.tent.drip{k}", (dx, 0.95 * sgn, 0.6 - h),
                             catchment_m2=0.15, edge_radius_mm=1.5, feed="rain",
                             source=f"shelter.tent.roof_{'l' if sgn > 0 else 'r'}", edge_material="tent",
                             prefer_puddle=True, n_points=2, length_m=0.8))
    elif s == "car":
        yc = -0.3
        panels = [
            ("shelter.car.roof", "vehicle_body", "car_body", (-0.2, yc, 0.3), 1.8, (0.0, 0.0, 1.0), {}),
            ("shelter.car.windshield", "surface", "glass", (0.75, yc, 0.05), 1.1, (0.6, 0.0, 0.8),
             {"wipers": str(getattr(place, "wipers", "off"))}),
            ("shelter.car.rear_window", "surface", "glass", (-1.5, yc, 0.0), 0.8, (-0.7, 0.0, 0.7), {}),
            ("shelter.car.hood", "vehicle_body", "car_body", (1.6, yc, -0.3), 1.6, (0.1, 0.0, 0.99), {}),
            ("shelter.car.trunk", "vehicle_body", "car_body", (-2.1, yc, -0.3), 0.8, (-0.1, 0.0, 0.99), {}),
            ("shelter.car.window_l", "surface", "glass", (0.0, 0.55, 0.0), 0.5, (0.0, 1.0, 0.1), {}),
            ("shelter.car.window_r", "surface", "glass", (0.0, -1.15, 0.0), 0.5, (0.0, -1.0, 0.1), {}),
        ]
        for pid, kind, mat, pos, area, n, extra in panels:
            params = {"shelter": "car"}
            params.update(extra)
            out.append(_patch(pid, kind, mat, pos, area=area, normal=n, exposure=_vert(ctx, n, pos[2]),
                              interior=True, params=params))
    elif s == "porch":
        roof_mat = "tin_roof"
        n_roof = (0.15, 0.0, 0.99)
        for tag, yy in (("l", 2.0), ("c", 0.0), ("r", -2.0)):
            out.append(_patch(f"shelter.porch.roof_{tag}", "surface", roof_mat, (0.5, yy, 2.5), area=6.0,
                              normal=n_roof, exposure=_vert(ctx, n_roof, 2.5),
                              params={"shelter": "porch", "panel": "roof"}))
        for k, yy in enumerate((-2.25, -0.75, 0.75, 2.25)):
            out.append(_drip(ctx, f"shelter.porch.drip{k}", (2.1, yy, 2.4),
                             catchment_m2=0.08, edge_radius_mm=2.0, feed="roof",
                             source="shelter.porch.roof_c", edge_material=roof_mat, prefer_puddle=True,
                             n_points=3, length_m=1.5))
        n_wall = (1.0, 0.0, 0.0)
        out.append(_patch("shelter.porch.wall", "surface", "wood", (-1.0, 0.0, 0.0), area=15.0, normal=n_wall,
                          exposure=_vert(ctx, n_wall, 0.0) * 0.3,
                          params={"shelter": "porch", "reflector": True}))
    elif s in ("room_open", "room_closed"):
        n_win = (1.0, 0.0, 0.0)
        out.append(_patch("shelter.room.window", "surface", "window", (2.0, 0.0, 0.0), area=1.5, normal=n_win,
                          exposure=max(_vert(ctx, n_win, 0.0), 0.05), interior=True,
                          params={"shelter": s, "window_open": float(getattr(place, "window_open", 0.0)),
                                  "glazing": "double"}))
        out.append(_patch("shelter.room.roof", "surface", "tile", (-1.0, 0.0, 3.0), area=40.0,
                          exposure=1.0, interior=True,
                          params={"shelter": s, "through_ceiling": True}))
        out.append(_patch("shelter.room.sill", "surface", "metal_sheet", (2.15, 0.0, -0.65), area=0.25,
                          exposure=0.8, params={"shelter": s, "sill": True}))
    elif s == "attic":
        for tag, xx, yy in (("lf", 1.5, 1.0), ("lb", -1.5, 1.0), ("rf", 1.5, -1.0), ("rb", -1.5, -1.0)):
            n = (0.0, 0.5 if yy > 0 else -0.5, 0.87)
            out.append(_patch(f"shelter.attic.roof_{tag}", "surface", "tin_roof", (xx, yy, 1.5),
                              area=6.0, normal=n, exposure=_vert(ctx, n, 1.5), interior=True,
                              params={"shelter": "attic", "panel": "roof", "span_m": 1.2}))
    elif s == "cave":
        for k, yy in enumerate((-1.5, -0.5, 0.5, 1.5)):
            out.append(_drip(ctx, f"shelter.cave.lip{k}", (3.1, yy, 2.5 - h),
                             catchment_m2=0.5, edge_radius_mm=3.0, feed="rain",
                             source="shelter.cave.ceiling", edge_material="concrete", prefer_puddle=True,
                             n_points=2, length_m=1.0))
        out.append(_drip(ctx, "shelter.cave.seep0", (-1.0, 0.8, 1.0), catchment_m2=0.02, edge_radius_mm=1.5,
                         feed="storage", source="shelter.cave.ceiling", edge_material="concrete",
                         target="water", interior=True, n_points=1, length_m=0.05))
        out.append(_drip(ctx, "shelter.cave.seep1", (0.5, -1.2, 0.8), catchment_m2=0.02, edge_radius_mm=1.5,
                         feed="storage", source="shelter.cave.ceiling", edge_material="concrete",
                         target="water", interior=True, n_points=1, length_m=0.05))
        for pid, pos, n in (("shelter.cave.ceiling", (0.0, 0.0, 1.5), (0.0, 0.0, -1.0)),
                            ("shelter.cave.wall_l", (0.0, 2.5, 0.0), (0.0, -1.0, 0.0)),
                            ("shelter.cave.wall_r", (0.0, -2.5, 0.0), (0.0, 1.0, 0.0))):
            out.append(_patch(pid, "surface", "concrete", pos, area=15.0, normal=n, exposure=0.0, interior=True,
                              params={"shelter": "cave", "reflector": True}))
    elif s == "underpass":
        out.append(_patch("shelter.underpass.deck", "surface", "concrete", (0.0, 0.0, 3.0), area=192.0,
                          exposure=1.0, interior=True,
                          params={"shelter": "underpass", "thickness_m": 0.6}))
        for side in (1.0, -1.0):
            for k, yy in enumerate((-4.5, -1.5, 1.5, 4.5)):
                tag = "f" if side > 0 else "b"
                out.append(_drip(ctx, f"shelter.underpass.drip_{tag}{k}", (8.2 * side, yy, 2.6),
                                 catchment_m2=1.0, edge_radius_mm=2.5, feed="rain",
                                 source="shelter.underpass.deck", edge_material="concrete", target="puddle",
                                 n_points=3, length_m=3.0))
        for tag, sgn in (("l", 1.0), ("r", -1.0)):
            n = (0.0, -sgn, 0.0)
            out.append(_patch(f"shelter.underpass.wall_{tag}", "surface", "concrete", (0.0, 6.0 * sgn, 0.5),
                              area=60.0, normal=n, exposure=0.0, interior=True,
                              params={"shelter": "underpass", "reflector": True}))
    return out


# ------------------------------------------------------------------ vegetation
def _tree_patch(ctx: _Ctx, pid: str, species: str, x: float, y: float, z_ground_rel: float,
                height: Optional[float] = None, extra: Optional[dict] = None) -> SourcePatch:
    prof = _SPECIES.get(species, _SPECIES["deciduous"])
    ht = float(height) if height else prof["height"]
    leaf = 1.0 if prof["evergreen"] else ctx.leaf_fraction
    mat = prof["material"] if leaf >= 0.2 else "wood"
    canopy_r = prof["canopy_r"] * (ht / prof["height"]) ** 0.5
    z = z_ground_rel + 0.6 * ht
    params = {"species": species, "height_m": ht, "canopy_radius_m": canopy_r, "leaf_on": leaf >= 0.2,
              "leaf_fraction": leaf, "lai": prof["lai"] * leaf, "twig_diameter_m": prof["twig_d"],
              "branch_diameter_m": prof["branch_d"], "leaf_size_m": prof["leaf_size_m"],
              "distance_m": math.hypot(x, y)}
    if extra:
        params.update(extra)
    return _patch(pid, "vegetation", mat, (x, y, z), area=2.0 * math.pi * canopy_r * canopy_r,
                  exposure=1.0, params=params)


def _build_vegetation(ctx: _Ctx) -> list[SourcePatch]:
    veg = ctx.scene.place.vegetation or {}
    dist = max(_num(veg.get("distance_m", veg.get("distance", 12.0)), 12.0), 1.0)
    groups = []
    for k in sorted(veg):
        if _norm_name(k) in ("distance_m", "distance", "spread_m", "density"):
            continue
        n = int(round(_num(veg[k])))
        if n > 0:
            groups.append((_species_key(k), n))
    out = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    base = float(ctx.rng.uniform(-math.pi, math.pi))
    idx = 0
    for species, count in groups:
        sp = species if species in _SPECIES else "deciduous"
        for c in range(count):
            az = base + idx * golden + float(ctx.rng.uniform(-0.2, 0.2))
            r = dist * float(ctx.rng.uniform(0.75, 1.35))
            x, y = r * math.cos(az), r * math.sin(az)
            out.append(_tree_patch(ctx, f"veg.{species}.{c}", sp, x, y, -ctx.h,
                                   extra={"species_name": species}))
            idx += 1
    return out


# ------------------------------------------------------------------ emitters
_EMITTER_HEIGHT = {"gutter": 2.7, "downspout": 0.25, "eave": 2.6, "wire": 6.0, "power_line": 9.0,
                   "chime": 2.2, "flag": 6.0, "bucket": 0.35}
_BUILDING_SHELTERS = ("porch", "room_open", "room_closed", "attic")


def _emitter_azimuth(ctx: _Ctx, em: dict) -> float:
    if "bearing_deg" in em:
        return bearing_to_azimuth(_num(em["bearing_deg"]), float(ctx.listener.heading_deg))
    if "azimuth_deg" in em:
        return math.radians(_num(em["azimuth_deg"]))
    if ctx.shelter in _BUILDING_SHELTERS:
        return float(ctx.rng.uniform(-0.35 * math.pi, 0.35 * math.pi))
    return float(ctx.rng.uniform(-math.pi, math.pi))


def _build_emitters(ctx: _Ctx) -> list[SourcePatch]:
    out = []
    h = ctx.h
    for i, raw in enumerate(ctx.scene.place.emitters or []):
        em = dict(raw) if isinstance(raw, dict) else dict(getattr(raw, "__dict__", {}))
        etype = _norm_name(em.get("type", "surface"))
        if etype in ("power", "powerline"):
            etype = "power_line"
        dist = max(_num(em.get("distance_m", 4.0), 4.0), 0.3)
        count = max(int(_num(em.get("count", 1), 1)), 1)
        height = _num(em.get("height_m", _EMITTER_HEIGHT.get(etype, 1.0)), 1.0)
        base_az = _emitter_azimuth(ctx, em)
        extra = {k: v for k, v in em.items() if k not in ("type",)}
        for c in range(count):
            # extra instances alternate either side of the first, ~3 m apart
            az = base_az + ((c + 1) // 2) * (1.0 if c % 2 else -1.0) * (3.0 / max(dist, 1.0))
            x, y = dist * math.cos(az), dist * math.sin(az)
            z = height - h
            pid = f"emitter.{etype}{i}" + (f".{c}" if count > 1 else "")
            params = {"type": etype, "height_m": height, "distance_m": dist}
            params.update(extra)
            if etype == "gutter":
                length = _num(em.get("length_m", 6.0), 6.0)
                params.setdefault("length_m", length)
                params.setdefault("width_m", 0.12)
                params.setdefault("catchment_m2", 25.0)
                params.setdefault("overflow_rate_mm_h", 25.0)
                out.append(_patch(pid, "gutter", normalize_material(em.get("material", "metal_gutter")),
                                  (x, y, z), area=0.12 * length, exposure=1.0, params=params))
            elif etype == "downspout":
                params.setdefault("pipe_length_m", max(height + 2.5, 1.0))
                params.setdefault("length_m", params["pipe_length_m"])
                params.setdefault("diameter_m", 0.08)
                params.setdefault("catchment_m2", 25.0)
                params.setdefault("target_material", _drip_target(ctx, x, y, prefer_puddle=False))
                out.append(_patch(pid, "gutter", normalize_material(em.get("material", "metal_gutter")),
                                  (x, y, z), area=0.05, exposure=0.0, params=params))
            elif etype == "eave":
                n_pts = max(int(_num(em.get("drips", 4), 4)), 1)
                tx, ty = -math.sin(az), math.cos(az)         # along the eave, perpendicular to the radius
                spacing = _num(em.get("spacing_m", 1.2), 1.2)
                for k in range(n_pts):
                    off = (k - 0.5 * (n_pts - 1)) * spacing
                    out.append(_drip(ctx, f"{pid}.drip{k}", (x + tx * off, y + ty * off, z),
                                     catchment_m2=_num(em.get("catchment_m2", 0.08), 0.08),
                                     edge_radius_mm=_num(em.get("edge_radius_mm", 2.0), 2.0),
                                     feed="roof", source=pid,
                                     edge_material=normalize_material(em.get("material", "tile")),
                                     prefer_puddle=True, extra={"type": "eave", "emitter": pid},
                                     n_points=max(1, int(round(spacing * 2.0))), length_m=spacing))
            elif etype in ("wire", "power_line"):
                params.setdefault("diameter_m", 0.02 if etype == "power_line" else 0.005)
                params.setdefault("length_m", 40.0)
                params.setdefault("power_line", etype == "power_line")
                wire_area = _num(params["diameter_m"], 0.005) * _num(params["length_m"], 40.0)
                out.append(_patch(pid, "wire", normalize_material(em.get("material", "metal_sheet")),
                                  (x, y, z), area=wire_area, normal=(0.0, 0.0, 1.0), exposure=1.0, params=params))
            elif etype == "chime":
                params.setdefault("n_tubes", 5)
                params.setdefault("base_freq_hz", 523.25)
                params.setdefault("scale", "pentatonic")
                params.setdefault("tube_diameter_m", 0.02)
                out.append(_patch(pid, "chime", normalize_material(em.get("material", "metal_sheet")),
                                  (x, y, z), area=0.05, exposure=0.3, params=params))
            elif etype == "flag":
                params.setdefault("area_m2", 1.5)
                params.setdefault("pole_height_m", height)
                out.append(_patch(pid, "flag", normalize_material(em.get("material", "fabric")),
                                  (x, y, z), area=_num(params["area_m2"], 1.5), normal=(0.0, 0.0, 1.0),
                                  exposure=0.2, params=params))
            elif etype == "bucket":
                out.append(_patch(pid, "surface", "bucket", (x, y, z), area=0.07, exposure=1.0, params=params))
            elif etype in ("tree", "trees"):
                sp = _species_key(em.get("species", "deciduous"))
                tree_extra = {k: v for k, v in params.items() if k != "height_m"}
                out.append(_tree_patch(ctx, pid, sp if sp in _SPECIES else "deciduous", x, y, -h,
                                       height=em.get("height_m"), extra=tree_extra))
            else:
                out.append(_patch(pid, etype or "surface", normalize_material(em.get("material", "concrete")),
                                  (x, y, z), area=_num(em.get("area_m2", 1.0), 1.0), exposure=1.0, params=params))
    return out


# ------------------------------------------------------------------ user objects
def _object_normal_world(obj, default_vertical: bool, ctx: _Ctx, pos_l) -> tuple:
    params = obj.params or {}
    if "normal" in params:
        return _unit(params["normal"])
    if not default_vertical:
        return (0.0, 0.0, 1.0)
    if abs(float(obj.rotation_deg or 0.0)) > 1e-9:
        r = math.radians(float(obj.rotation_deg))
        return (math.sin(r), math.cos(r), 0.0)
    # face the listener
    wx, wy = ctx.to_world_dir(-pos_l[0], -pos_l[1])
    n = math.hypot(wx, wy)
    return (wx / n, wy / n, 0.0) if n > 1e-6 else (1.0, 0.0, 0.0)


def _normal_to_listener(ctx: _Ctx, n_world) -> tuple:
    nx, ny = ctx.from_world_dir(n_world[0], n_world[1])
    return (nx, ny, n_world[2])


def _apply_shelter_to_object(ctx: _Ctx, p: SourcePatch) -> SourcePatch:
    top = _SHELTER_TOP.get(ctx.shelter)
    if top is None or p.position[2] >= top:
        return p
    cover, cid, _ = _shelter_cover(ctx, p.position[0], p.position[1])
    if cover > 0.0:
        p.exposure = float(p.exposure * (1.0 - cover))
        if cover > 0.5:
            p.sheltered_by = cid
            if ctx.shelter in _INTERIOR_SHELTERS:
                p.interior = True
    return p


def _build_objects(ctx: _Ctx) -> list[SourcePatch]:
    out: list[SourcePatch] = []
    L = ctx.listener
    h = ctx.h
    lz = float((tuple(L.position) + (0.0, 0.0, 0.0))[2])
    ground_l = -(lz + h)        # world ground plane (z = 0) in the listener frame
    for obj in ctx.scene.place.objects or []:
        otype = _norm_name(getattr(obj, "type", "surface"))
        if otype == "storm":
            continue
        pos = world_to_listener(obj.position, L)
        size = tuple(obj.size) + (0.0, 0.0, 0.0)
        sx, sy, sz = max(float(size[0]), 0.0), max(float(size[1]), 0.0), max(float(size[2]), 0.0)
        oparams = dict(obj.params or {})
        base = {"type": otype, "object_id": obj.id, "size": (sx, sy, sz),
                "rotation_deg": float(obj.rotation_deg or 0.0)}
        base.update(oparams)
        pid = f"obj.{obj.id}"
        x, y, z = pos
        # trees, cars, tents and umbrellas are placed by their base: `z` is their foot
        z_base = z
        mat_raw = getattr(obj, "material", "") or ""
        new: list[SourcePatch] = []
        if otype in ("surface", "roof", "bucket"):
            default_mat = {"surface": "concrete", "roof": "tile", "bucket": "bucket"}[otype]
            n_w = _object_normal_world(obj, False, ctx, pos)
            n_l = _normal_to_listener(ctx, n_w)
            area = sx * sy if sx * sy > 0 else (0.07 if otype == "bucket" else 1.0)
            mat = normalize_material(mat_raw) if mat_raw else default_mat
            new.append(_patch(pid, "surface", mat, pos, area=area, normal=n_l,
                              exposure=_vert(ctx, n_l, z), params=base))
        elif otype == "water":
            mat = normalize_material(mat_raw) if mat_raw else "water"
            if mat not in ("water", "puddle"):
                mat = "water"
            area = sx * sy if sx * sy > 0 else 4.0
            new.append(_patch(pid, "water", mat, pos, area=area, exposure=1.0, params=base))
        elif otype in ("tree", "grass"):
            if otype == "grass":
                sp = "grass"
            else:
                sp = _species_key(oparams.get("species", mat_raw or "deciduous"))
                if sp not in _SPECIES:
                    sp = "deciduous"
            height = _num(oparams.get("height_m", sz if sz > 0 else 0.0), 0.0) or None
            tp = _tree_patch(ctx, pid, sp, x, y, z_base, height=height, extra=base)
            if otype == "grass" and sx * sy > 0:
                tp.area_m2 = float(sx * sy)
            new.append(tp)
            if otype == "tree" and math.hypot(x, y) < 8.0:
                prof = _SPECIES[sp]
                r_c = tp.params["canopy_radius_m"]
                for k in range(3):
                    a = 2.0 * math.pi * k / 3.0 + 0.5
                    rr = 0.6 * r_c
                    ex, ey = x + rr * math.cos(a), y + rr * math.sin(a)
                    new.append(_drip(ctx, f"{pid}.drip{k}", (ex, ey, z_base + 0.35 * tp.params["height_m"]),
                                     catchment_m2=0.02, edge_radius_mm=1.5, feed="canopy", source=pid,
                                     edge_material=tp.material, impact_z=z_base,
                                     extra={"species": sp, "leaf_size_m": prof["leaf_size_m"]},
                                     n_points=2, length_m=0.6))
        elif otype in ("gutter", "downspout"):
            length = sx if sx > 0 else (6.0 if otype == "gutter" else 2.5)
            base.setdefault("length_m" if otype == "gutter" else "pipe_length_m", length)
            base.setdefault("catchment_m2", 25.0)
            base.setdefault("diameter_m", 0.08)
            mat = normalize_material(mat_raw) if mat_raw else "metal_gutter"
            if otype == "downspout":
                base.setdefault("length_m", length)
                base.setdefault("target_material", _drip_target(ctx, x, y))
            new.append(_patch(pid, "gutter", mat, pos, area=0.12 * length,
                              exposure=1.0 if otype == "gutter" else 0.0, params=base))
        elif otype == "eave":
            length = sx if sx > 0 else 4.0
            n_pts = max(2, int(round(length / 1.2)))
            r = math.radians(float(obj.rotation_deg or 0.0))
            ax_e, ax_n = math.cos(r), -math.sin(r)        # eave runs along the object's local x (east at 0 deg)
            tx, ty = ctx.from_world_dir(ax_e, ax_n)
            mat = normalize_material(mat_raw) if mat_raw else "tile"
            for k in range(n_pts):
                off = (k - 0.5 * (n_pts - 1)) * (length / n_pts)
                new.append(_drip(ctx, f"{pid}.drip{k}", (x + tx * off, y + ty * off, z),
                                 catchment_m2=_num(oparams.get("catchment_m2", 0.08), 0.08),
                                 edge_radius_mm=_num(oparams.get("edge_radius_mm", 2.0), 2.0),
                                 feed="roof", source=pid, edge_material=mat, prefer_puddle=True,
                                 impact_z=ground_l, extra={"type": "eave", "object_id": obj.id},
                                 n_points=max(1, int(round(2.0 * length / n_pts))), length_m=length / n_pts))
        elif otype == "wire":
            base.setdefault("diameter_m", 0.005)
            base.setdefault("length_m", sx if sx > 0 else 30.0)
            mat = normalize_material(mat_raw) if mat_raw else "metal_sheet"
            wire_area = _num(base["diameter_m"], 0.005) * _num(base["length_m"], 30.0)
            new.append(_patch(pid, "wire", mat, pos, area=wire_area, params=base))
        elif otype == "chime":
            base.setdefault("n_tubes", 5)
            base.setdefault("base_freq_hz", 523.25)
            base.setdefault("scale", "pentatonic")
            mat = normalize_material(mat_raw) if mat_raw else "metal_sheet"
            new.append(_patch(pid, "chime", mat, pos, area=0.05, exposure=0.3, params=base))
        elif otype == "flag":
            area = sx * sy if sx * sy > 0 else 1.5
            base.setdefault("area_m2", area)
            mat = normalize_material(mat_raw) if mat_raw else "fabric"
            new.append(_patch(pid, "flag", mat, pos, area=area, exposure=0.2, params=base))
        elif otype == "car":
            for sub, kind, mat, dz, dxf, area, n in (
                ("roof", "vehicle_body", "car_body", 1.45, 0.0, 1.8, (0.0, 0.0, 1.0)),
                ("windshield", "surface", "glass", 1.2, 0.9, 1.1, None),
                ("hood", "vehicle_body", "car_body", 0.95, 1.8, 1.6, (0.0, 0.0, 1.0)),
            ):
                r = math.radians(float(obj.rotation_deg or 0.0))
                fe, fn = math.sin(r), math.cos(r)            # car's forward axis (world)
                lx_, ly_ = ctx.from_world_dir(fe * dxf, fn * dxf)
                if n is None:
                    nl = _normal_to_listener(ctx, (fe * 0.6, fn * 0.6, 0.8))
                else:
                    nl = n
                p_sub = (x + lx_, y + ly_, z_base + dz)
                new.append(_patch(f"{pid}.{sub}", kind, mat, p_sub, area=area, normal=nl,
                                  exposure=_vert(ctx, nl, dz), params=dict(base, part=sub)))
        elif otype == "tent":
            area = sx * sy if sx * sy > 0 else 4.0
            mat = normalize_material(mat_raw) if mat_raw else "tent"
            new.append(_patch(pid, "membrane", mat, (x, y, z_base + (sz if sz > 0 else 1.0)),
                              area=area * 1.3, exposure=1.0, params=base))
        elif otype == "umbrella":
            radius = 0.5 * sx if sx > 0 else 1.2
            mat = normalize_material(mat_raw) if mat_raw else ("canvas" if radius > 0.8 else "umbrella")
            new.append(_patch(pid, "membrane", mat, (x, y, z_base + (sz if sz > 0 else 2.2)),
                              area=math.pi * radius * radius, exposure=1.0, params=dict(base, radius_m=radius)))
        elif otype in ("window", "wall"):
            n_w = _object_normal_world(obj, True, ctx, pos)
            n_l = _normal_to_listener(ctx, n_w)
            height = sz if sz > 0 else (sy if sy > 0 else (1.2 if otype == "window" else 3.0))
            area = max(sx, 0.5) * height
            mat = normalize_material(mat_raw) if mat_raw else ("window" if otype == "window" else "concrete")
            if otype == "wall":
                base.setdefault("reflector", True)
            new.append(_patch(pid, "surface", mat, pos, area=area, normal=n_l,
                              exposure=_vert(ctx, n_l, z), params=base))
        else:
            area = sx * sy if sx * sy > 0 else 1.0
            mat = normalize_material(mat_raw) if mat_raw else "concrete"
            new.append(_patch(pid, otype or "surface", mat, pos, area=area,
                              exposure=_vert(ctx, (0.0, 0.0, 1.0), z), params=base))
        for p in new:
            out.append(_apply_shelter_to_object(ctx, p))
    return out


# ------------------------------------------------------------------ preset scenery
def _has_object(ctx: _Ctx, *types) -> bool:
    want = set(types)
    return any(_norm_name(getattr(o, "type", "")) in want for o in (ctx.scene.place.objects or []))


def _build_preset_features(ctx: _Ctx) -> list[SourcePatch]:
    p = ctx.preset
    h = ctx.h
    out: list[SourcePatch] = []
    if any(k in p for k in ("city", "street", "alley", "downtown", "urban")) and not _has_object(ctx, "wall"):
        width = 4.0 if "alley" in p else 12.0
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            n = (0.0, -sgn, 0.0)
            for k, xx in enumerate((-15.0, 0.0, 15.0)):
                out.append(_patch(f"scenery.facade_{side}{k}", "surface", "concrete",
                                  (xx, sgn * width / 2.0, 4.0 - h), area=120.0, normal=n,
                                  exposure=_vert(ctx, n, 4.0) * 0.5,
                                  params={"reflector": True, "facade": True, "street_width_m": width}))
                out.append(_patch(f"scenery.window_{side}{k}", "surface", "window",
                                  (xx + 3.0, sgn * (width / 2.0 - 0.05), 2.5 - h), area=2.0, normal=n,
                                  exposure=_vert(ctx, n, 2.5), params={"facade": True}))
        if "alley" in p:
            out.append(_patch("scenery.dumpster_lid", "surface", "metal_sheet", (3.0, 1.4, 1.2 - h), area=2.0,
                              exposure=1.0, params={"object": "dumpster"}))
        elif not _has_object(ctx, "car") and ctx.shelter != "car":
            kerb = -(width / 2.0 - 2.0)
            out.append(_patch("scenery.parked_car.roof", "vehicle_body", "car_body", (4.0, kerb, 1.45 - h),
                              area=1.8, exposure=1.0, params={"object": "parked_car"}))
            nl = (0.6, 0.0, 0.8)
            out.append(_patch("scenery.parked_car.windshield", "surface", "glass", (5.0, kerb, 1.2 - h),
                              area=1.1, normal=nl, exposure=_vert(ctx, nl, 1.2), params={"object": "parked_car"}))
    if any(k in p for k in ("coast", "sea", "beach", "ocean", "harbour", "harbor")) and not _has_object(ctx, "water"):
        out.append(_patch("scenery.sea", "water", "water", (40.0, 0.0, -h), area=2000.0, exposure=1.0,
                          params={"sea": True, "deep": True, "surf": "beach" in p or "coast" in p, "extent_m": 60.0}))
    elif any(k in p for k in ("lake", "pond", "river", "loch")) and not _has_object(ctx, "water"):
        flowing = "river" in p
        out.append(_patch("scenery.water", "water", "water", (20.0, 0.0, -h), area=600.0, exposure=1.0,
                          params={"flowing": flowing, "deep": True, "extent_m": 30.0}))
    if any(k in p for k in ("suburb", "garden", "backyard")) and ctx.shelter not in _BUILDING_SHELTERS \
            and not _has_object(ctx, "wall"):
        n = (1.0, 0.0, 0.0)
        out.append(_patch("scenery.house_wall", "surface", "concrete", (-8.0, 0.0, 2.5 - h), area=50.0, normal=n,
                          exposure=_vert(ctx, n, 2.5), params={"reflector": True, "house": True}))
    if "farm" in p and not _has_object(ctx, "roof"):
        out.append(_patch("scenery.barn_roof", "surface", "tin_roof", (-20.0, 15.0, 5.0 - h), area=120.0,
                          exposure=1.0, params={"barn": True}))
        n = (1.0, 0.0, 0.0)
        out.append(_patch("scenery.barn_wall", "surface", "wood", (-20.0, 15.0, 2.0 - h), area=60.0, normal=n,
                          exposure=_vert(ctx, n, 2.0), params={"barn": True, "reflector": True}))
    if "camp" in p and ctx.shelter != "tent" and not _has_object(ctx, "tent"):
        out.append(_patch("scenery.tent", "membrane", "tent", (6.0, 4.0, 1.0 - h), area=6.0, exposure=1.0,
                          params={"neighbour": True}))
    for patch in out:
        _apply_shelter_to_object(ctx, patch)
    return out


# ------------------------------------------------------------------ public
def build_layout(scene) -> list[SourcePatch]:
    """All source patches around the listener for `scene` (deterministic per seed).

    Order: ground rings, shelter panels (+ their drip edges), vegetation, place
    emitters, preset scenery, user objects. Every patch carries params['upwind_m']
    (metres upwind of the listener, for Taylor-advected gusts) and
    params['world_position'].
    """
    ctx = _Ctx(scene)
    patches: list[SourcePatch] = []
    patches.extend(_build_ground(ctx))
    patches.extend(_build_shelter(ctx))
    patches.extend(_build_vegetation(ctx))
    patches.extend(_build_emitters(ctx))
    patches.extend(_build_preset_features(ctx))
    patches.extend(_build_objects(ctx))
    seen: dict[str, int] = {}
    for p in patches:
        n = seen.get(p.id, 0)
        if n:
            p.id = f"{p.id}#{n}"
        seen[p.id.split("#")[0]] = n + 1
        x, y, z = p.position
        p.params.setdefault("upwind_m", float(ctx.upwind(x, y)))
        p.params.setdefault("world_position", listener_to_world(p.position, scene.listener))
        p.params.setdefault("shelter", ctx.shelter)
    return patches


def exposure_by_material(patches) -> dict[str, float]:
    """Area-weighted mean exposure per material (input for WetnessStores.update)."""
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for p in patches:
        if p.kind == "drip_edge":
            continue
        num[p.material] = num.get(p.material, 0.0) + p.exposure * p.area_m2
        den[p.material] = den.get(p.material, 0.0) + p.area_m2
    return {m: (num[m] / den[m] if den[m] > 0 else 0.0) for m in num}
