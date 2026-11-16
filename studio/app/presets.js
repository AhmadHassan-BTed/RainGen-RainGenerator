/**
 * Scene presets, the Listener vibe macros and storm-arc automation for the composer.
 *
 * Vibe anchors (interpolated in log(rate), Beaufort and log(distance)):
 *   Intensity 0 dry/drips · 20 drizzle 0.3 mm/h Bft 2 · 40 moderate 5 mm/h Bft 3 ·
 *   60 heavy 20 mm/h Bft 5 · 80 thunderstorm 40 mm/h Bft 7, thunder 5 km ·
 *   100 extreme 80 mm/h Bft 9, thunder 1.5 km.
 *   Closeness 0 out in it · 25 umbrella/tree · 50 porch/tent · 75 indoors window
 *   open · 100 indoors window shut.
 *   Calm-Wild: gust factor 1.15-1.9, gust rate, shower-cell contrast, thunder rate.
 * Storm arcs follow the life of a convective cell: sparse fat drops on arrival,
 * the downpour, then a lighter stratiform tail as it passes or clears.
 */
import { defaultScene, deepMerge, normalizeScene, clone } from './scene-store.js';

/* ------------------------------------------------------------------ materials */

export const MATERIALS = {
  tin_roof: { label: 'Tin roof', color: '#8b939c', roughness: 0.35, metalness: 0.75 },
  metal: { label: 'Metal', color: '#9aa3ab', roughness: 0.3, metalness: 0.8 },
  glass: { label: 'Glass', color: '#a9d2e3', roughness: 0.05, metalness: 0.1, opacity: 0.35 },
  concrete: { label: 'Concrete', color: '#8e8b84', roughness: 0.9, metalness: 0 },
  asphalt: { label: 'Asphalt', color: '#3b3d42', roughness: 0.85, metalness: 0 },
  puddle: { label: 'Puddle', color: '#2c4a5e', roughness: 0.05, metalness: 0.2, opacity: 0.85 },
  water: { label: 'Water', color: '#1f4b66', roughness: 0.04, metalness: 0.25, opacity: 0.9 },
  leaves: { label: 'Leaves', color: '#6f7f2f', roughness: 0.85, metalness: 0 },
  needles: { label: 'Needles', color: '#3f5a2c', roughness: 0.9, metalness: 0 },
  wood: { label: 'Wood', color: '#8a5a2e', roughness: 0.75, metalness: 0 },
  fabric: { label: 'Fabric', color: '#3f6e8c', roughness: 0.95, metalness: 0 },
  tent_fabric: { label: 'Tent fabric', color: '#d08a2a', roughness: 0.9, metalness: 0 },
  umbrella_fabric: { label: 'Umbrella', color: '#2b3f66', roughness: 0.8, metalness: 0 },
  car_body: { label: 'Car body', color: '#3b5f86', roughness: 0.25, metalness: 0.7 },
  tile: { label: 'Tile', color: '#a65a3a', roughness: 0.7, metalness: 0 },
  slate: { label: 'Slate', color: '#4c5259', roughness: 0.6, metalness: 0.05 },
  brick: { label: 'Brick', color: '#8d4b3b', roughness: 0.85, metalness: 0 },
  plaster: { label: 'Plaster', color: '#cfc7b8', roughness: 0.9, metalness: 0 },
  gravel: { label: 'Gravel', color: '#9b958b', roughness: 1, metalness: 0 },
  grass: { label: 'Grass', color: '#4f7c35', roughness: 0.95, metalness: 0 },
  soil: { label: 'Soil / mud', color: '#5b4231', roughness: 0.95, metalness: 0 },
  snow: { label: 'Snow', color: '#e9eef3', roughness: 0.9, metalness: 0 },
  canvas: { label: 'Canvas', color: '#a3916d', roughness: 0.95, metalness: 0 },
};

export const MATERIALS_BY_TYPE = {
  surface: ['tin_roof', 'glass', 'concrete', 'asphalt', 'puddle', 'leaves', 'needles', 'wood', 'fabric', 'car_body', 'tile', 'gravel', 'grass', 'soil', 'snow', 'metal', 'slate'],
  water: ['water', 'puddle'],
  roof: ['tin_roof', 'tile', 'slate', 'wood', 'metal', 'canvas'],
  window: ['glass'],
  wall: ['brick', 'concrete', 'wood', 'plaster', 'metal'],
  tree: ['leaves', 'needles'],
  grass: ['grass'],
  gutter: ['metal'],
  downspout: ['metal'],
  eave: ['wood', 'tin_roof', 'tile', 'metal'],
  wire: ['metal'],
  chime: ['metal', 'wood'],
  flag: ['fabric', 'canvas'],
  car: ['car_body'],
  tent: ['tent_fabric', 'fabric', 'canvas'],
  umbrella: ['umbrella_fabric', 'fabric'],
  storm: [''],
};

/* ------------------------------------------------------------------ object factories */

const tan = (deg) => Math.tan((deg * Math.PI) / 180);

export const make = {
  tree: (id, x, y, h = 9, species = 'deciduous') => ({
    id, type: 'tree', material: species === 'conifer' ? 'needles' : 'leaves', position: [x, y, 0],
    size: species === 'conifer' ? [h * 0.38, h * 0.38, h] : [h * 0.7, h * 0.7, h], rotation_deg: 0, params: { species, height: h },
  }),
  surface: (id, material, x, y, w, d, z = 0, rot = 0) => ({ id, type: 'surface', material, position: [x, y, z], size: [w, d, 0.03], rotation_deg: rot, params: {} }),
  water: (id, x, y, w, d, depth = 0.6) => ({ id, type: 'water', material: 'water', position: [x, y, 0], size: [w, d, depth], rotation_deg: 0, params: { depth_m: depth } }),
  roof: (id, material, x, y, z, w, d, rot = 0, style = 'gable', pitch = 30) => ({
    id, type: 'roof', material, position: [x, y, z], size: [w, d, style === 'gable' ? (w / 2) * tan(pitch) : w * tan(pitch) * 0.35], rotation_deg: rot, params: { style, pitch_deg: pitch },
  }),
  wall: (id, material, x, y, w, h, rot = 0, t = 0.25) => ({ id, type: 'wall', material, position: [x, y, 0], size: [w, t, h], rotation_deg: rot, params: {} }),
  window: (id, x, y, z, w, h, rot = 0, open = 0) => ({ id, type: 'window', material: 'glass', position: [x, y, z], size: [w, 0.08, h], rotation_deg: rot, params: { open } }),
  gutter: (id, x, y, z, len, rot = 0) => ({ id, type: 'gutter', material: 'metal', position: [x, y, z], size: [len, 0.15, 0.12], rotation_deg: rot, params: { length_m: len } }),
  downspout: (id, x, y, h) => ({ id, type: 'downspout', material: 'metal', position: [x, y, 0], size: [0.1, 0.1, h], rotation_deg: 0, params: { height_m: h } }),
  eave: (id, x, y, z, len, rot = 0, material = 'wood') => ({ id, type: 'eave', material, position: [x, y, z], size: [len, 0.3, 0.06], rotation_deg: rot, params: { drips_per_m: 3 } }),
  wire: (id, x, y, z, len, rot = 0, d = 0.012) => ({ id, type: 'wire', material: 'metal', position: [x, y, z], size: [len, 0.05, 0.05], rotation_deg: rot, params: { diameter_m: d } }),
  chime: (id, x, y, z) => ({ id, type: 'chime', material: 'metal', position: [x, y, z], size: [0.3, 0.3, 0.6], rotation_deg: 0, params: { tubes: 5 } }),
  flag: (id, x, y, h = 6) => ({ id, type: 'flag', material: 'fabric', position: [x, y, 0], size: [1.5, 0.02, h], rotation_deg: 0, params: {} }),
  car: (id, x, y, rot = 0) => ({ id, type: 'car', material: 'car_body', position: [x, y, 0], size: [4.4, 1.8, 1.45], rotation_deg: rot, params: {} }),
  tent: (id, x, y, rot = 0) => ({ id, type: 'tent', material: 'tent_fabric', position: [x, y, 0], size: [2.2, 2.0, 1.3], rotation_deg: rot, params: {} }),
  umbrella: (id, x, y, r = 0.55, h = 2.0) => ({ id, type: 'umbrella', material: 'umbrella_fabric', position: [x, y, 0], size: [r * 2, r * 2, h], rotation_deg: 0, params: { radius_m: r } }),
  grass: (id, x, y, w, d) => ({ id, type: 'grass', material: 'grass', position: [x, y, 0], size: [w, d, 0.35], rotation_deg: 0, params: {} }),
  porch: (id, x, y, w = 6, d = 3, z = 2.6, material = 'wood', rot = 0) => ({
    id, type: 'roof', material, position: [x, y, z], size: [w, d, 0.35], rotation_deg: rot, params: { style: 'porch', pitch_deg: 8, posts: true },
  }),
  storm: (id, track, speed = 25, fpm = 3, intensity = 0.6, radius = 4) => ({
    id, type: 'storm', material: '', position: [track[0][0], track[0][1], 2000], size: [radius * 2000, radius * 2000, 8000], rotation_deg: 0,
    params: { track, speed_kmh: speed, flashes_per_min: fpm, intensity, radius_km: radius },
  }),
};

/* ------------------------------------------------------------------ presets */

const W = (o) => o;

function P(id, name, meta, scene) {
  return { id, name, ...meta, scene: { name, ...scene } };
}

export const PRESETS = [
  P('leaves-night', 'Steady rain on leaves, night', { tags: ['Calm', 'Sleep', 'Forest'], blurb: 'Moderate rain through a broadleaf canopy, the default.', hue: ['#0e1a14', '#28452d'] }, {
    seed: 0x5eedcafe,
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 4, drops: 'normal', character: 'continuous' }, wind: { speed_mps: 3.2, from_deg: 250, gust_factor: 1.35 }, temperature_c: 11, humidity_pct: 94, hour: 23, season: 'autumn' }),
    place: { preset: 'deciduous_forest', shelter: 'open', surfaces: { leaves: 55, grass: 20, soil: 15, puddle: 10 }, objects: [
      make.tree('oak-1', 5, 6, 12), make.tree('oak-2', -7, 4, 10), make.tree('beech-1', 2, -9, 11), make.tree('oak-3', -11, -6, 9), make.tree('beech-2', 13, -3, 13),
      make.surface('leaf-litter', 'leaves', 0, 0, 14, 14), make.surface('puddle-1', 'puddle', 2.5, -2, 1.6, 1.1), make.grass('meadow', -2, 12, 16, 8),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 0 },
    mode: 'calm',
  }),
  P('tin-roof-cabin-storm', 'Tin roof cabin storm', { tags: ['Storm', 'Cozy', 'Film'], blurb: 'Rain drumming on corrugated metal, thunder rolling over the pines.', hue: ['#13171d', '#46505c'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 18, drops: 'fat', character: 'intermittent' }, wind: { speed_mps: 8.5, from_deg: 290, gust_factor: 1.6 }, thunder: { enabled: true, distance_km: 4, bearing_deg: 300, approach_kmh: 10, flashes_per_min: 3, stroke_intensity: 0.7 }, temperature_c: 14, humidity_pct: 96, hour: 21, season: 'autumn', storm_arc: 'passing' }),
    place: { preset: 'conifer_forest', shelter: 'attic', surfaces: { metal: 50, wood: 15, needles: 25, puddle: 10 }, objects: [
      make.roof('cabin-roof', 'tin_roof', 0, 0, 2.6, 6, 5, 0, 'gable', 28),
      make.wall('cabin-n', 'wood', 0, 2.5, 6, 2.6), make.wall('cabin-s', 'wood', 0, -2.5, 6, 2.6), make.wall('cabin-e', 'wood', 3, 0, 5, 2.6, 90), make.wall('cabin-w', 'wood', -3, 0, 5, 2.6, 90),
      make.window('cabin-window', -3, 0.6, 1.1, 1.2, 1.0, 90, 0.15), make.gutter('cabin-gutter', 0, -2.75, 2.55, 6), make.downspout('cabin-downspout', 3, -2.75, 2.55),
      make.tree('pine-1', -9, 7, 18, 'conifer'), make.tree('pine-2', 8, 9, 16, 'conifer'), make.tree('pine-3', 11, -6, 20, 'conifer'), make.tree('pine-4', -10, -9, 17, 'conifer'),
      make.surface('needle-floor', 'needles', 0, 0, 30, 30, 0), make.surface('yard-puddle', 'puddle', 1, -5, 2, 1.2),
      make.storm('storm-west', [[-9000, 2500], [-2500, 3000], [4000, 1500]], 28, 3, 0.7, 5),
    ] },
    listener: { position: [0.5, 0.3, 0], height_m: 1.6, heading_deg: 270 },
  }),
  P('tent-in-a-storm', 'Tent in a storm', { tags: ['Storm', 'Outdoors', 'ASMR'], blurb: 'Fabric drumhead overhead, gusts slamming the fly sheet.', hue: ['#1a1408', '#6b4a1a'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 10, drops: 'normal', character: 'intermittent' }, wind: { speed_mps: 11, from_deg: 260, gust_factor: 1.7 }, thunder: { enabled: true, distance_km: 6, bearing_deg: 240, approach_kmh: 0, flashes_per_min: 2 }, temperature_c: 9, humidity_pct: 95, hour: 2, season: 'summer' }),
    place: { preset: 'mountain_ridge', shelter: 'tent', surfaces: { fabric: 45, grass: 30, gravel: 15, soil: 10 }, objects: [
      make.tent('tent', 0, 0, 80), make.grass('ridge-grass', 0, 0, 24, 18), make.surface('scree', 'gravel', 9, 6, 8, 6),
      make.tree('lone-pine', -12, 8, 9, 'conifer'), make.wire('guy-line', 1.6, 0, 0.5, 1.4, 20, 0.003),
    ] },
    listener: { position: [0, 0, 0], height_m: 0.45, heading_deg: 80 },
  }),
  P('car-thunderstorm', 'Car in a thunderstorm', { tags: ['Storm', 'Urban', 'Film'], blurb: 'Steel roof, glass, wipers and the city glowing through the downpour.', hue: ['#0b1220', '#2c3e5c'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 25, drops: 'fat', character: 'showery', shower_period_min: 10 }, wind: { speed_mps: 7, from_deg: 230, gust_factor: 1.5 }, thunder: { enabled: true, distance_km: 2.5, bearing_deg: 200, approach_kmh: 0, flashes_per_min: 4, stroke_intensity: 0.75 }, temperature_c: 19, humidity_pct: 95, hour: 20.5, season: 'summer' }),
    place: { preset: 'city_street', shelter: 'car', wipers: 'slow', surfaces: { car_body: 35, glass: 15, asphalt: 35, puddle: 15 }, objects: [
      make.car('car', 0, 0, 0), make.surface('road', 'asphalt', 0, 0, 12, 60), make.surface('kerb-puddle', 'puddle', 4.5, 3, 2.5, 1.2),
      make.wall('block-east', 'concrete', 9, 0, 40, 14, 90, 0.5), make.wall('block-west', 'brick', -9, 0, 40, 12, 90, 0.5),
      make.car('parked-1', 4.2, -8, 0), make.car('parked-2', -4.2, 9, 180),
      make.storm('storm-sw', [[-6000, -6000], [-1500, -2000], [3000, 2500]], 22, 4, 0.8, 4),
    ] },
    listener: { position: [-0.4, 0.2, 0], height_m: 1.15, heading_deg: 0 },
  }),
  P('umbrella-city-street', 'Umbrella on a city street', { tags: ['Urban', 'Walk', 'Calm'], blurb: 'Patter on nylon, reflections off shopfronts, tyres hissing past.', hue: ['#151021', '#3d2f5e'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 6, drops: 'normal', character: 'continuous' }, wind: { speed_mps: 3, from_deg: 200, gust_factor: 1.4 }, temperature_c: 13, humidity_pct: 92, hour: 19, season: 'autumn' }),
    place: { preset: 'city_street', shelter: 'umbrella', surfaces: { fabric: 30, asphalt: 30, concrete: 25, puddle: 15 }, objects: [
      make.umbrella('umbrella', 0, 0, 0.55, 2.0), make.surface('pavement', 'concrete', 0, 0, 4, 40), make.surface('street', 'asphalt', 6, 0, 8, 40),
      make.wall('shopfront', 'brick', -2.4, 0, 40, 10, 90, 0.4), make.window('shop-window', -2.15, 2, 0.6, 3, 2.2, 90),
      make.surface('puddle-a', 'puddle', 1.2, 3, 1.4, 0.8), make.car('passing-car', 6, -10, 0), make.eave('awning-drip', -1.6, 6, 2.6, 3, 90, 'canvas'),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 0 },
  }),
  P('porch-summer-thunderstorm', 'Porch summer thunderstorm', { tags: ['Storm', 'Cozy', 'Calm'], blurb: 'Fat warm drops on a wooden porch roof, wind chimes, rolling thunder.', hue: ['#1b1406', '#5e4a1d'] }, {
    weather: W({ precipitation: { type: 'showers', rate_mm_h: 30, drops: 'fat', character: 'showery', shower_period_min: 14 }, wind: { speed_mps: 6, from_deg: 250, gust_factor: 1.6 }, thunder: { enabled: true, distance_km: 3, bearing_deg: 260, approach_kmh: 15, flashes_per_min: 3, stroke_intensity: 0.65 }, temperature_c: 24, humidity_pct: 88, hour: 17, season: 'summer', storm_arc: 'arriving' }),
    place: { preset: 'garden', shelter: 'porch', surfaces: { wood: 30, grass: 35, soil: 15, puddle: 10, concrete: 10 }, objects: [
      make.porch('porch-roof', 0, 0, 7, 3, 2.7, 'wood'), make.wall('house-front', 'plaster', 0, 1.6, 7, 3), make.surface('deck', 'wood', 0, 0, 7, 3, 0.3),
      make.chime('chime', 2.6, -1.2, 2.2), make.gutter('porch-gutter', 0, -1.55, 2.6, 7), make.downspout('porch-downspout', -3.5, -1.55, 2.6),
      make.grass('lawn', 0, -9, 20, 12), make.tree('maple', -7, -10, 11), make.tree('elm', 8, -13, 13),
      make.storm('summer-cell', [[-7000, -1500], [-2500, -500], [2000, 1200]], 25, 3, 0.65, 3),
    ] },
    listener: { position: [0, 0.2, 0.3], height_m: 1.2, heading_deg: 180 },
  }),
  P('tokyo-night-alley', 'Tokyo night alley', { tags: ['Urban', 'Night', 'Focus'], blurb: 'Fine rain between narrow walls, drips from eaves and humming wires.', hue: ['#120914', '#4c1d4f'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 5, drops: 'fine', character: 'continuous' }, wind: { speed_mps: 2, from_deg: 180, gust_factor: 1.3 }, temperature_c: 22, humidity_pct: 97, hour: 1, season: 'summer' }),
    place: { preset: 'alley', shelter: 'open', surfaces: { asphalt: 40, concrete: 20, metal: 15, puddle: 15, tile: 10 }, objects: [
      make.wall('alley-w', 'concrete', -1.8, 0, 30, 9, 90, 0.4), make.wall('alley-e', 'plaster', 1.8, 0, 30, 7, 90, 0.4),
      make.eave('eave-w', -1.3, 2, 3.2, 10, 90, 'tile'), make.eave('eave-e', 1.3, -3, 2.9, 8, 90, 'metal'),
      make.downspout('ds-1', -1.55, 5, 6), make.downspout('ds-2', 1.55, -6, 5), make.gutter('gutter-e', 1.5, -3, 2.9, 8, 90),
      make.wire('overhead-1', 0, 4, 5.5, 3.6, 0, 0.01), make.wire('overhead-2', 0, -2, 6, 3.6, 10, 0.014),
      make.surface('alley-floor', 'asphalt', 0, 0, 3.2, 30), make.surface('alley-puddle', 'puddle', 0.4, 2.5, 1.2, 0.7), make.surface('ac-unit', 'metal', 1.4, 1, 0.8, 0.5, 2.2),
    ] },
    listener: { position: [0, -1, 0], height_m: 1.65, heading_deg: 0 },
  }),
  P('tropical-monsoon-veranda', 'Tropical monsoon veranda', { tags: ['Tropical', 'Heavy', 'Calm'], blurb: 'Warm torrential rain on broad leaves and a tiled veranda roof.', hue: ['#06160f', '#145c3c'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 60, drops: 'normal', character: 'showery', shower_period_min: 20 }, wind: { speed_mps: 5, from_deg: 200, gust_factor: 1.4 }, thunder: { enabled: true, distance_km: 10, bearing_deg: 150, flashes_per_min: 1.5 }, temperature_c: 27, humidity_pct: 98, hour: 15, season: 'summer' }),
    place: { preset: 'jungle', shelter: 'porch', surfaces: { leaves: 45, tile: 20, soil: 20, puddle: 15 }, objects: [
      make.porch('veranda', 0, 0, 8, 3.5, 2.8, 'tile'), make.surface('veranda-floor', 'tile', 0, 0, 8, 3.5, 0.4), make.wall('villa', 'plaster', 0, 1.9, 8, 3.2),
      make.tree('banana-1', -5, -5, 6), make.tree('palm-1', 4, -7, 12), make.tree('ficus', -9, -12, 15), make.tree('banana-2', 7, -3, 5),
      make.surface('garden-soil', 'soil', 0, -8, 18, 10), make.water('lily-pond', -3, -9, 4, 3, 0.5), make.eave('veranda-eave', 0, -1.9, 2.8, 8, 0, 'tile'),
    ] },
    listener: { position: [0, 0.5, 0.4], height_m: 1.25, heading_deg: 180 },
  }),
  P('highland-drizzle', 'Highland drizzle', { tags: ['Drizzle', 'Wind', 'Focus'], blurb: 'Fine drizzle carried on a steady westerly over open moor.', hue: ['#10161a', '#43525a'] }, {
    weather: W({ precipitation: { type: 'drizzle', rate_mm_h: 0.4, drops: 'fine', character: 'continuous' }, wind: { speed_mps: 9, from_deg: 240, gust_factor: 1.45 }, temperature_c: 7, humidity_pct: 99, hour: 11, season: 'autumn' }),
    place: { preset: 'mountain_ridge', shelter: 'open', surfaces: { grass: 60, gravel: 20, soil: 15, puddle: 5 }, objects: [
      make.grass('moor', 0, 0, 40, 40), make.surface('track', 'gravel', 3, 0, 2.5, 40), make.wire('fence-wire', -4, 2, 1.0, 12, 90, 0.004),
      make.wire('fence-wire-2', -4, 2, 0.6, 12, 90, 0.004), make.surface('peat-pool', 'puddle', -8, -6, 3, 2),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 240 },
  }),
  P('conifer-gale', 'Conifer gale', { tags: ['Wind', 'Storm', 'Film'], blurb: 'A force-8 gale roaring through tall spruce, rain in squalls.', hue: ['#0a120c', '#2a3f2e'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 6, drops: 'normal', character: 'showery', shower_period_min: 8 }, wind: { speed_mps: 18, from_deg: 280, gust_factor: 1.85 }, temperature_c: 5, humidity_pct: 90, hour: 14, season: 'autumn' }),
    place: { preset: 'conifer_forest', shelter: 'open', surfaces: { needles: 60, soil: 25, puddle: 10, gravel: 5 }, objects: [
      make.tree('spruce-1', -6, 5, 22, 'conifer'), make.tree('spruce-2', 4, 8, 24, 'conifer'), make.tree('spruce-3', 9, -2, 20, 'conifer'),
      make.tree('spruce-4', -10, -6, 25, 'conifer'), make.tree('spruce-5', 2, -9, 21, 'conifer'), make.tree('spruce-6', -14, 10, 23, 'conifer'),
      make.tree('spruce-7', 15, 12, 22, 'conifer'), make.surface('forest-floor', 'needles', 0, 0, 40, 40), make.wire('powerline', 0, 20, 9, 60, 0, 0.025),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 280 },
  }),
  P('after-the-storm-drips', 'After-the-storm drips', { tags: ['Aftermath', 'Calm', 'Sleep'], blurb: 'Rain has stopped; trees, eaves and gutters keep dripping at dawn.', hue: ['#121a22', '#53697a'] }, {
    weather: W({ precipitation: { type: 'none', rate_mm_h: 0.02, drops: 'fat', character: 'intermittent' }, wind: { speed_mps: 1.5, from_deg: 300, gust_factor: 1.3 }, thunder: { enabled: true, distance_km: 20, bearing_deg: 80, approach_kmh: -20, flashes_per_min: 0.5, stroke_intensity: 0.4 }, temperature_c: 13, humidity_pct: 98, hour: 6, season: 'spring', storm_arc: 'clearing' }),
    place: { preset: 'garden', shelter: 'porch', surfaces: { leaves: 30, grass: 30, puddle: 20, wood: 10, concrete: 10 }, objects: [
      make.porch('porch-roof', 0, 0, 6, 3, 2.6, 'wood'), make.eave('porch-eave', 0, -1.5, 2.6, 6), make.gutter('gutter', 0, -1.6, 2.6, 6), make.downspout('downspout', 3, -1.6, 2.6),
      make.tree('dripping-oak', -5, -6, 12), make.tree('birch', 6, -8, 10), make.surface('puddle-1', 'puddle', 0, -3, 2.2, 1.4), make.surface('puddle-2', 'puddle', -3, -5, 1.4, 1.0),
      make.water('water-butt', 3.3, -1.3, 0.6, 0.6, 0.9),
    ] },
    listener: { position: [0, 0.3, 0], height_m: 1.6, heading_deg: 180 },
  }),
  P('first-fat-drops', 'First fat drops', { tags: ['Arriving', 'Summer', 'Film'], blurb: 'The leading edge of a cell: sparse heavy drops slapping hot concrete.', hue: ['#1d1408', '#7a5223'] }, {
    weather: W({ precipitation: { type: 'showers', rate_mm_h: 0.8, drops: 'fat', character: 'showery', shower_period_min: 18 }, wind: { speed_mps: 6, from_deg: 250, gust_factor: 1.7 }, thunder: { enabled: true, distance_km: 9, bearing_deg: 250, approach_kmh: 20, flashes_per_min: 2 }, temperature_c: 27, humidity_pct: 70, hour: 16, season: 'summer', storm_arc: 'arriving' }),
    place: { preset: 'suburb', shelter: 'open', surfaces: { concrete: 45, soil: 20, grass: 20, metal: 10, leaves: 5 }, objects: [
      make.surface('patio', 'concrete', 0, 0, 8, 6), make.surface('bin-lid', 'metal', 3, 2, 0.6, 0.6, 1.0), make.grass('lawn', 0, -8, 14, 8), make.tree('cherry', 6, -5, 7),
      make.car('driveway-car', -6, 2, 90), make.storm('approaching-cell', [[-12000, -3000], [-5000, -1200], [1000, 300]], 30, 2, 0.6, 5),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 250 },
  }),
  P('hailstorm-on-a-car', 'Hailstorm on a car', { tags: ['Hail', 'Storm', 'Film'], blurb: 'Marble-size hail hammering steel and glass, thunder close by.', hue: ['#0d1015', '#3e4a5a'] }, {
    weather: W({ precipitation: { type: 'hail', rate_mm_h: 40, drops: 'fat', character: 'showery', shower_period_min: 6, hail_size_mm: 19 }, wind: { speed_mps: 12, from_deg: 270, gust_factor: 1.8 }, thunder: { enabled: true, distance_km: 2, bearing_deg: 280, flashes_per_min: 5, stroke_intensity: 0.85 }, temperature_c: 16, humidity_pct: 85, hour: 17.5, season: 'summer' }),
    place: { preset: 'city_street', shelter: 'car', wipers: 'off', surfaces: { car_body: 40, glass: 20, asphalt: 30, puddle: 10 }, objects: [
      make.car('car', 0, 0, 0), make.surface('parking-lot', 'asphalt', 0, 0, 30, 30), make.car('car-2', 3, 1, 0), make.car('car-3', -3, -1, 0),
      make.storm('hail-cell', [[-5000, 0], [-500, 500], [4000, 1000]], 35, 5, 0.9, 3),
    ] },
    listener: { position: [-0.4, 0.2, 0], height_m: 1.15, heading_deg: 270 },
  }),
  P('snow-hush-forest', 'Snow hush forest', { tags: ['Snow', 'Winter', 'Sleep'], blurb: 'Soft snowfall on a snow-covered forest floor; the world goes quiet.', hue: ['#11161d', '#7d8fa3'] }, {
    weather: W({ precipitation: { type: 'snow', rate_mm_h: 1.5, drops: 'fine', character: 'continuous' }, wind: { speed_mps: 1.5, from_deg: 20, gust_factor: 1.25 }, temperature_c: -3, humidity_pct: 85, hour: 18, season: 'winter' }),
    place: { preset: 'conifer_forest', shelter: 'open', surfaces: { snow: 85, needles: 10, wood: 5 }, objects: [
      make.surface('snow-cover', 'snow', 0, 0, 40, 40), make.tree('fir-1', -5, 6, 14, 'conifer'), make.tree('fir-2', 6, 4, 16, 'conifer'), make.tree('fir-3', 2, -8, 13, 'conifer'),
      make.tree('fir-4', -9, -5, 15, 'conifer'), make.tree('bare-birch', 9, -7, 10), make.surface('log', 'wood', -2, 2, 2.5, 0.4, 0.2, 30),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 0 },
    mode: 'sleep',
  }),
  P('window-rain-bedroom', 'Window rain bedroom (sleep)', { tags: ['Sleep', 'Indoors', 'Calm'], blurb: 'Wind-driven rain ticking on the bedroom window, muffled and close.', hue: ['#0b0d18', '#262c4f'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 5, drops: 'normal', character: 'continuous' }, wind: { speed_mps: 4.5, from_deg: 270, gust_factor: 1.3 }, temperature_c: 10, humidity_pct: 93, hour: 23.5, season: 'autumn' }),
    place: { preset: 'suburb', shelter: 'room_closed', window_open: 0, surfaces: { glass: 40, tile: 30, grass: 20, puddle: 10 }, objects: [
      make.wall('bedroom-w', 'plaster', -2, 0, 4, 2.6, 90), make.wall('bedroom-n', 'plaster', 0, 2, 4, 2.6), make.wall('bedroom-s', 'plaster', 0, -2, 4, 2.6), make.wall('bedroom-e', 'plaster', 2, 0, 4, 2.6, 90),
      make.window('bedroom-window', -2, 0, 1.0, 1.4, 1.2, 90, 0), make.roof('house-roof', 'tile', 0, 0, 2.6, 4.5, 4.5, 0, 'gable', 35),
      make.tree('garden-tree', -8, 3, 9), make.grass('garden', -8, 0, 8, 10),
    ] },
    listener: { position: [0.8, 0, 0], height_m: 0.7, heading_deg: 270 },
    mode: 'sleep',
  }),
  P('asmr-umbrella-close', 'ASMR umbrella close', { tags: ['ASMR', 'Close', 'Calm'], blurb: 'Close, crisp patter on a small umbrella right above your head.', hue: ['#161018', '#4f3352'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 3, drops: 'fat', character: 'continuous' }, wind: { speed_mps: 1, from_deg: 180, gust_factor: 1.2 }, temperature_c: 15, humidity_pct: 92, hour: 20, season: 'spring' }),
    place: { preset: 'garden', shelter: 'umbrella', surfaces: { fabric: 60, grass: 20, puddle: 20 }, objects: [
      make.umbrella('close-umbrella', 0, 0, 0.45, 1.85), make.surface('puddle', 'puddle', 0.6, 0.5, 1.2, 0.9), make.grass('lawn', 0, 0, 8, 8),
    ] },
    listener: { position: [0, 0, 0], height_m: 1.6, heading_deg: 0 },
    mode: 'asmr',
  }),
  P('lake-dawn-rain', 'Rain on a lake at dawn', { tags: ['Water', 'Calm', 'Focus'], blurb: 'Light rain of small drops ringing on still water from a wooden jetty.', hue: ['#0b1622', '#3a6a8a'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 2, drops: 'fine', character: 'continuous' }, wind: { speed_mps: 1, from_deg: 90, gust_factor: 1.2 }, temperature_c: 12, humidity_pct: 96, hour: 5.6, season: 'summer' }),
    place: { preset: 'lake', shelter: 'open', surfaces: { puddle: 70, wood: 15, grass: 15 }, objects: [
      make.water('lake', 0, 25, 80, 40, 3), make.surface('jetty', 'wood', 0, 4, 1.6, 8, 0.4), make.tree('willow', -6, 0, 9), make.grass('shore', 0, -6, 30, 8),
    ] },
    listener: { position: [0, 6, 0.4], height_m: 1.2, heading_deg: 0 },
  }),
  P('coastal-squall', 'Coastal squall', { tags: ['Wind', 'Storm', 'Film'], blurb: 'A squall line coming off the sea: horizontal rain, flag snapping, wires moaning.', hue: ['#0a141c', '#2f5470'] }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 15, drops: 'normal', character: 'showery', shower_period_min: 7 }, wind: { speed_mps: 15, from_deg: 230, gust_factor: 1.75 }, thunder: { enabled: true, distance_km: 7, bearing_deg: 220, approach_kmh: 25, flashes_per_min: 2 }, temperature_c: 12, humidity_pct: 92, hour: 13, season: 'autumn' }),
    place: { preset: 'coast', shelter: 'open', surfaces: { gravel: 35, grass: 30, concrete: 20, puddle: 15 }, objects: [
      make.water('sea', -20, -30, 80, 40, 10), make.surface('promenade', 'concrete', 0, 0, 30, 5), make.flag('harbour-flag', 3, 2, 7), make.wire('stay-wire', 3, 2.5, 3.5, 4, 70, 0.006),
      make.surface('shingle', 'gravel', 0, -6, 30, 6), make.grass('dune', 0, 8, 30, 8),
      make.storm('squall-line', [[-8000, -8000], [-2000, -2000], [4000, 4000]], 40, 2, 0.6, 8),
    ] },
    listener: { position: [0, 1, 0], height_m: 1.7, heading_deg: 230 },
  }),
  P('walk-in-from-the-storm', 'Walk in from the storm (demo)', { tags: ['Demo', 'Storm', 'Film'], blurb: 'Guided walk: meadow, under the oak, onto the porch, indoors, close the window.', hue: ['#0c121a', '#3b4f67'], demo: true }, {
    weather: W({ precipitation: { type: 'rain', rate_mm_h: 12, drops: 'normal', character: 'intermittent' }, wind: { speed_mps: 8, from_deg: 250, gust_factor: 1.6 }, thunder: { enabled: true, distance_km: 5, bearing_deg: 240, approach_kmh: 10, flashes_per_min: 3, stroke_intensity: 0.7 }, temperature_c: 16, humidity_pct: 94, hour: 19.5, season: 'summer' }),
    place: { preset: 'meadow', shelter: 'open', window_open: 1, surfaces: { grass: 45, leaves: 15, wood: 15, puddle: 15, tile: 10 }, objects: [
      make.grass('meadow', -30, -4, 30, 22), make.tree('oak', -14, 2, 12), make.surface('path', 'gravel', -12, 3, 26, 1.6, 0, 12),
      make.porch('porch', 2, 7.5, 6, 3, 2.6, 'tin_roof'), make.surface('porch-deck', 'wood', 2, 7.5, 6, 3, 0.25),
      make.wall('house-s-left', 'wood', 0, 9, 2, 2.8), make.wall('house-s-right', 'wood', 4.1, 9, 1.8, 2.8),
      make.wall('house-n', 'wood', 2, 15, 6, 2.8), make.wall('house-e', 'wood', 5, 12, 6, 2.8, 90), make.wall('house-w', 'wood', -1, 12, 6, 2.8, 90),
      make.window('west-window', -1, 12, 1.0, 1.3, 1.1, 90, 1), make.roof('house-roof', 'tile', 2, 12, 2.8, 6.4, 6.4, 0, 'gable', 32),
      make.gutter('porch-gutter', 2, 5.95, 2.55, 6), make.surface('puddle', 'puddle', -4, 5, 2, 1.3), make.chime('porch-chime', 4.3, 6.6, 2.1),
      make.storm('storm-cell', [[-9000, -2500], [-3500, 500], [2500, 3500]], 30, 3, 0.75, 5),
    ] },
    listener: { position: [-34, -6, 0], height_m: 1.7, heading_deg: 70 },
  }),
];

export const PRESET_BY_ID = new Map(PRESETS.map((p) => [p.id, p]));

/** Full, normalised scene for a preset id. */
export function presetScene(id) {
  const p = PRESET_BY_ID.get(id) || PRESETS[0];
  const s = normalizeScene(deepMerge(defaultScene(), clone(p.scene)));
  s.vibe = {};
  s.overrides = {};
  s.presetId = p.id;
  return s;
}

/* ------------------------------------------------------------------ guided demo */

export const DEMO_WALK = {
  presetId: 'walk-in-from-the-storm',
  duration: 60,
  waypoints: [
    { t: 0, pos: [-34, -6], look: [-14, 2], shelter: 'open', caption: 'Out on the meadow: rain all around, the storm to the west-south-west.' },
    { t: 11, pos: [-15.5, 1.2], look: [-10, 6], shelter: 'tree', caption: 'Under the oak: the canopy catches drops and releases fatter drips.' },
    { t: 19, pos: [-8, 4.2], look: [2, 7.5], shelter: 'open', caption: 'Back out in it, heading for the porch.' },
    { t: 28, pos: [1.6, 7.3], look: [2, 12], shelter: 'porch', caption: 'On the porch: rain drums on the tin roof above.' },
    { t: 35, pos: [1.6, 9.2], look: [1.6, 13], shelter: 'room_open', windowOpen: 1, caption: 'Through the door: indoors, the window is open.' },
    { t: 44, pos: [0.3, 12], look: [-6, 12], shelter: 'room_open', windowOpen: 1, caption: 'At the west window, facing the storm.' },
    { t: 50, pos: [0.3, 12], look: [-6, 12], shelter: 'room_closed', windowOpen: 0, caption: 'Window closed: the storm becomes a hush.' },
    { t: 60, pos: [1.8, 12.6], look: [-6, 11], shelter: 'room_closed', windowOpen: 0, caption: '' },
  ],
};

/* ------------------------------------------------------------------ vibe macros */

const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (x) => Math.max(0, Math.min(1, x));

function interp(x, anchors, log = false) {
  if (x <= anchors[0][0]) return anchors[0][1];
  for (let i = 1; i < anchors.length; i++) {
    const [x1, y1] = anchors[i];
    const [x0, y0] = anchors[i - 1];
    if (x <= x1) {
      const t = (x - x0) / (x1 - x0);
      return log ? Math.exp(lerp(Math.log(y0), Math.log(y1), t)) : lerp(y0, y1, t);
    }
  }
  return anchors[anchors.length - 1][1];
}

function invInterp(y, anchors, log = false) {
  const f = log ? Math.log : (v) => v;
  const yy = f(y);
  if (yy <= f(anchors[0][1])) return anchors[0][0];
  for (let i = 1; i < anchors.length; i++) {
    const y0 = f(anchors[i - 1][1]);
    const y1 = f(anchors[i][1]);
    if (yy <= y1) return lerp(anchors[i - 1][0], anchors[i][0], (yy - y0) / (y1 - y0 || 1));
  }
  return anchors[anchors.length - 1][0];
}

const RATE_ANCHORS = [[0, 0.02], [20, 0.3], [40, 5], [60, 20], [80, 40], [100, 80]];
const BFT_ANCHORS = [[0, 1], [20, 2], [40, 3], [60, 5], [80, 7], [100, 9]];
const THUNDER_KM_ANCHORS = [[45, 25], [60, 15], [80, 5], [100, 1.5]];
const FROZEN = ['snow', 'sleet', 'hail', 'graupel', 'freezing_rain'];
const mpsFromBft = (b) => 0.836 * Math.pow(Math.max(0, b), 1.5);

export const VIBE_DEFAULT = { intensity: 40, closeness: 50, wildness: 30 };

/** Driver values per macro group: { intensity: {path: value}, closeness: {...}, wildness: {...} }. */
export function vibeDrivers(vibe, scene) {
  const base = vibe.base || {};
  const I = vibe.intensity ?? VIBE_DEFAULT.intensity;
  const C = vibe.closeness ?? VIBE_DEFAULT.closeness;
  const Wd = vibe.wildness ?? VIBE_DEFAULT.wildness;

  const rate = interp(I, RATE_ANCHORS, true);
  const bft = interp(I, BFT_ANCHORS);
  const baseType = base.type || 'rain';
  let type;
  if (I < 6) type = 'none';
  else if (FROZEN.includes(baseType)) type = baseType;
  else if (rate < 0.6) type = 'drizzle';
  else type = baseType === 'showers' ? 'showers' : 'rain';
  const rateScaled = type === 'snow' ? rate * 0.3 : rate;
  const thunderOn = I >= 68 || (!!base.thunder && I >= 45);
  const km = interp(Math.max(45, I), THUNDER_KM_ANCHORS, true);

  const intensity = {
    'weather.precipitation.type': type,
    'weather.precipitation.rate_mm_h': Math.max(0.02, Math.min(150, rateScaled)),
    'weather.wind.speed_mps': mpsFromBft(bft),
    'weather.thunder.enabled': thunderOn,
    'weather.thunder.distance_km': km,
  };

  let shelter;
  const hasUmbrella = scene.place.objects.some((o) => o.type === 'umbrella');
  if (C < 12.5) shelter = 'open';
  else if (C < 37.5) shelter = ['umbrella', 'tree'].includes(base.shelter) ? base.shelter : hasUmbrella ? 'umbrella' : 'tree';
  else if (C < 62.5) shelter = ['tent', 'car', 'porch', 'attic', 'cave', 'underpass'].includes(base.shelter) ? base.shelter : 'porch';
  else if (C < 97.5) shelter = 'room_open';
  else shelter = 'room_closed';
  const windowOpen = C >= 62.5 ? clamp01((100 - C) / 25) : scene.place.window_open;
  const closeness = { 'place.shelter': shelter, 'place.window_open': windowOpen };

  const w = Wd / 100;
  const wildness = {
    'weather.wind.gust_factor': 1.15 + 0.75 * w,
    'weather.wind.gust_rate_per_min': 0.5 + 3.5 * w,
    'weather.precipitation.character': w < 0.35 ? 'continuous' : w < 0.7 ? 'intermittent' : 'showery',
    'weather.precipitation.shower_period_min': Math.round(30 - 24 * w),
    'weather.thunder.flashes_per_min': Math.round((0.5 + 7.5 * w) * 10) / 10,
  };
  return { intensity, closeness, wildness };
}

/** Best vibe-slider position for a scene (inverse of the anchors), so the sliders open where the scene is. */
export function inferVibe(scene) {
  const p = scene.weather.precipitation;
  let I = p.type === 'none' ? 0 : invInterp(p.type === 'snow' ? p.rate_mm_h / 0.3 : p.rate_mm_h, RATE_ANCHORS, true);
  I = Math.max(p.type === 'none' ? 0 : 8, Math.min(100, I));
  const sh = scene.place.shelter;
  let C;
  if (sh === 'open') C = 0;
  else if (sh === 'tree' || sh === 'umbrella') C = 25;
  else if (sh === 'room_open') C = 100 - 25 * clamp01(scene.place.window_open ?? 1);
  else if (sh === 'room_closed') C = 100;
  else C = 50;
  const gf = scene.weather.wind.gust_factor ?? 1.4;
  const Wd = Math.max(0, Math.min(100, ((gf - 1.15) / 0.75) * 100));
  return { intensity: Math.round(I), closeness: Math.round(C), wildness: Math.round(Wd) };
}

/* ------------------------------------------------------------------ storm arcs */

const R = 'weather.precipitation.rate_mm_h';
const WS = 'weather.wind.speed_mps';
const TK = 'weather.thunder.distance_km';
const TE = 'weather.thunder.enabled';
const DR = 'weather.precipitation.drops';
const PT = 'weather.precipitation.type';
const AR = 'weather.storm_arc';
const AP = 'weather.thunder.approach_kmh';

const kf = (at, path, value, curve = 'linear') => ({ at: Math.round(at * 100) / 100, path, value, curve });

export const STORM_ARCS = {
  arriving: (s, D) => [
    kf(0, AR, 'arriving', 'hold'),
    kf(0, PT, 'rain', 'hold'),
    kf(0, R, 0.3, 'log'), kf(D * 0.35, R, 2, 'log'), kf(D, R, 25, 'log'),
    kf(0, DR, 'fat', 'hold'), kf(D * 0.55, DR, 'normal', 'hold'),
    kf(0, WS, mpsFromBft(2), 'smooth'), kf(D * 0.7, WS, mpsFromBft(6), 'smooth'), kf(D, WS, mpsFromBft(5), 'smooth'),
    kf(0, TE, true, 'hold'), kf(0, TK, 18, 'log'), kf(D, TK, 2, 'log'),
    kf(0, AP, 20, 'hold'),
  ],
  steady: (s, D) => [
    kf(0, AR, 'steady', 'hold'),
    kf(0, R, s.weather.precipitation.rate_mm_h, 'linear'), kf(D, R, s.weather.precipitation.rate_mm_h, 'linear'),
    kf(0, WS, s.weather.wind.speed_mps, 'linear'), kf(D, WS, s.weather.wind.speed_mps, 'linear'),
    kf(0, AP, 0, 'hold'),
  ],
  passing: (s, D) => [
    kf(0, AR, 'passing', 'hold'),
    kf(0, PT, 'rain', 'hold'),
    kf(0, R, 1, 'log'), kf(D * 0.45, R, 30, 'log'), kf(D, R, 1, 'log'),
    kf(0, DR, 'fat', 'hold'), kf(D * 0.3, DR, 'normal', 'hold'), kf(D * 0.75, DR, 'fine', 'hold'),
    kf(0, WS, mpsFromBft(2), 'smooth'), kf(D * 0.42, WS, mpsFromBft(6), 'smooth'), kf(D, WS, mpsFromBft(3), 'smooth'),
    kf(0, TE, true, 'hold'), kf(0, TK, 12, 'log'), kf(D * 0.45, TK, 1.5, 'log'), kf(D, TK, 12, 'log'),
    kf(0, AP, 20, 'hold'), kf(D * 0.45, AP, -20, 'hold'),
  ],
  clearing: (s, D) => [
    kf(0, AR, 'clearing', 'hold'),
    kf(0, PT, 'rain', 'hold'), kf(D * 0.85, PT, 'none', 'hold'),
    kf(0, R, 15, 'log'), kf(D * 0.6, R, 0.3, 'log'), kf(D * 0.85, R, 0.02, 'log'),
    kf(0, DR, 'normal', 'hold'), kf(D * 0.4, DR, 'fine', 'hold'),
    kf(0, WS, mpsFromBft(5), 'smooth'), kf(D, WS, mpsFromBft(2), 'smooth'),
    kf(0, TE, true, 'hold'), kf(0, TK, 3, 'log'), kf(D * 0.8, TK, 20, 'log'), kf(D * 0.8, TE, false, 'hold'),
    kf(0, AP, -20, 'hold'),
  ],
};

/* ------------------------------------------------------------------ scene physics helpers */

/** Impacts per m^2 per s for drops >= 0.5 mm from a Marshall-Palmer drop-size distribution, log-log interpolated. */
const FLUX_TABLE = [[0.5, 450], [1, 750], [2.5, 1380], [5, 2100], [10, 3100], [25, 5000], [50, 7100], [100, 9700]];

export function impactFlux(rate) {
  if (!(rate > 0)) return 0;
  const T = FLUX_TABLE;
  let i = 1;
  while (i < T.length - 1 && rate > T[i][0]) i++;
  const [r0, f0] = T[i - 1];
  const [r1, f1] = T[i];
  const t = (Math.log(rate) - Math.log(r0)) / (Math.log(r1) - Math.log(r0));
  return Math.exp(Math.log(f0) + t * (Math.log(f1) - Math.log(f0)));
}

/** Drop-character factor: maritime (fine) ~3x the impacts per mm/h of continental (fat), Bringi et al. 2003. */
const DROP_FACTOR = { fine: 1.7, normal: 1, fat: 0.55 };
const TYPE_FACTOR = { none: 0, drizzle: 1.2, rain: 1, showers: 1, freezing_rain: 1, sleet: 0.6, snow: 0.5, graupel: 0.4, hail: 0.02 };

/** World (x east, y north) -> object-local (lx along width, ly along depth); rotation_deg is clockwise. */
export function toLocal(o, x, y) {
  const th = ((o.rotation_deg || 0) * Math.PI) / 180;
  const dx = x - o.position[0];
  const dy = y - o.position[1];
  return [dx * Math.cos(th) - dy * Math.sin(th), dx * Math.sin(th) + dy * Math.cos(th)];
}

function insideFootprint(o, x, y, margin = 0) {
  if (o.type === 'umbrella' || o.type === 'tree') {
    const r = (o.size[0] || 1) / 2 + margin;
    return Math.hypot(x - o.position[0], y - o.position[1]) <= r;
  }
  const [lx, ly] = toLocal(o, x, y);
  return Math.abs(lx) <= o.size[0] / 2 + margin && Math.abs(ly) <= o.size[1] / 2 + margin;
}

function coverTop(o) {
  if (o.type === 'roof') return o.position[2];
  if (o.type === 'tree') return o.size[2] * 0.35;
  return o.position[2] + o.size[2];
}

function interception(o, scene) {
  switch (o.type) {
    case 'roof': case 'tent': case 'car': return 1;
    case 'umbrella': return 0.97;
    case 'tree': {
      const winter = scene.weather.season === 'winter';
      const conifer = o.params?.species === 'conifer';
      return conifer ? 0.4 : winter ? 0.1 : 0.3;
    }
    default: return 0;
  }
}

/** Fraction of the open-field rain reaching a point at height zTop (canopy interception 10-40 %, roofs 100 %). */
export function exposureAt(x, y, zTop, scene, excludeId) {
  let frac = 1;
  for (const c of scene.place.objects) {
    if (c.id === excludeId) continue;
    const ic = interception(c, scene);
    if (ic <= 0) continue;
    if (coverTop(c) <= zTop + 0.05) continue;
    if (!insideFootprint(c, x, y)) continue;
    frac *= 1 - ic;
  }
  return frac;
}

export function rainExposure(o, scene) {
  if (o.type === 'storm') return 0;
  const w = scene.weather.wind;
  if (o.type === 'wall' || o.type === 'window') {
    const th = ((o.rotation_deg || 0) * Math.PI) / 180;
    const normalBearing = (th * 180) / Math.PI;
    const rel = ((w.from_deg - normalBearing) * Math.PI) / 180;
    const facing = Math.abs(Math.cos(rel));
    const driven = Math.min(1, w.speed_mps / 12) * facing;
    return Math.max(0.05, Math.min(1, 0.08 + 0.92 * driven)) * exposureAt(o.position[0], o.position[1], o.position[2] + o.size[2], scene, o.id);
  }
  if (o.type === 'wire' || o.type === 'chime' || o.type === 'flag') return 1;
  const top = o.type === 'roof' ? o.position[2] + o.size[2] : o.position[2] + (o.size[2] || 0);
  return exposureAt(o.position[0], o.position[1], top, scene, o.id);
}

export function catchArea(o) {
  const [w, d, h] = o.size;
  switch (o.type) {
    case 'wall': case 'window': return w * h;
    case 'tree': return Math.PI * (w / 2) ** 2;
    case 'umbrella': return Math.PI * (w / 2) ** 2;
    case 'gutter': return w * 0.15;
    case 'eave': return w * 0.3;
    case 'wire': return w * Math.max(0.002, o.params?.diameter_m || 0.01);
    case 'chime': return 0.05;
    case 'flag': return w * 1.0 * 0.2;
    case 'downspout': return 0.02;
    default: return Math.max(0.01, w * d);
  }
}

/** Estimated impacts per second on an object (before the engine reports live 'impact-stats'). */
export function estimateImpactRate(o, scene) {
  const p = scene.weather.precipitation;
  const R = p.type === 'none' ? 0 : p.rate_mm_h;
  const f = impactFlux(R) * (TYPE_FACTOR[p.type] ?? 1) * (DROP_FACTOR[p.drops] ?? 1);
  return f * catchArea(o) * rainExposure(o, scene);
}

/** Shelter implied by standing at world (x, y): "transitions happen because you moved". */
export function detectShelter(x, y, scene) {
  const objs = scene.place.objects;
  const find = (type, pred = () => true) => objs.find((o) => o.type === type && pred(o) && insideFootprint(o, x, y, 0.1));
  if (find('car')) return 'car';
  if (find('tent')) return 'tent';
  if (find('umbrella')) return 'umbrella';
  const roof = find('roof');
  if (roof) {
    if (roof.params?.style === 'porch') return 'porch';
    if (roof.material === 'tin_roof' || roof.material === 'metal') return 'attic';
    const windows = objs.filter((o) => o.type === 'window' && insideFootprint(roof, o.position[0], o.position[1], 1.0));
    const open = windows.length
      ? Math.max(...windows.map((w) => (w.params?.open ?? scene.place.window_open ?? 0)))
      : scene.place.window_open ?? 0;
    return open > 0.05 ? 'room_open' : 'room_closed';
  }
  if (find('tree')) return 'tree';
  return 'open';
}

/** Storm cell position (world x, y) after t seconds along its track at speed_kmh. */
export function stormPosition(o, t) {
  const tr = o.params?.track;
  if (!Array.isArray(tr) || tr.length === 0) return [o.position[0], o.position[1]];
  if (tr.length === 1) return [tr[0][0], tr[0][1]];
  let s = Math.max(0, ((o.params.speed_kmh ?? 25) / 3.6) * Math.max(0, t));
  for (let i = 1; i < tr.length; i++) {
    const seg = Math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1]);
    if (s <= seg && seg > 0) {
      const k = s / seg;
      return [tr[i - 1][0] + (tr[i][0] - tr[i - 1][0]) * k, tr[i - 1][1] + (tr[i][1] - tr[i - 1][1]) * k];
    }
    s -= seg;
  }
  const last = tr[tr.length - 1];
  return [last[0], last[1]];
}

/** Distance (km) and bearing (deg true) from the listener to a world point. */
export function rangeBearing(scene, x, y) {
  const L = scene.listener.position;
  const dx = x - L[0];
  const dy = y - L[1];
  const km = Math.hypot(dx, dy) / 1000;
  const bearing = (((Math.atan2(dx, dy) * 180) / Math.PI) + 360) % 360;
  return { km, bearing };
}
