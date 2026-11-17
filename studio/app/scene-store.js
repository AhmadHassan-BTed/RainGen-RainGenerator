/**
 * Scene state for the 4D Environment Composer: the open `rainscene` v1 JSON
 * (the same file the Python CLI reads), a parameter manifest (path, label,
 * unit, range, scale and smoothing group of every control), the Auto / Offset /
 * Pinned model (resolved = pinned ? user : clamp(driver + offset), offsets
 * stored in perceptual units), undo/redo with one-gesture-one-step,
 * localStorage autosave, user presets, URL-hash sharing and JSON import/export.
 * Every storage access is wrapped: private browsing or blocked storage only
 * turns persistence off.
 *
 * World coordinates: metres, x east, y north, z up.
 */

export const SCHEMA_VERSION = 1;

/* ------------------------------------------------------------------ defaults */

export function defaultScene() {
  return {
    rainscene: SCHEMA_VERSION,
    name: 'Untitled weather',
    seed: 0x5eedcafe,
    weather: {
      precipitation: {
        type: 'rain',
        rate_mm_h: 4,
        drops: 'normal',
        character: 'continuous',
        shower_period_min: 12,
        hail_size_mm: 13,
        dsd_model: 'gamma',
        mu: null,
        dm_mm: null,
        class_boundaries: 'medwin',
        water_quality: 1,
      },
      wind: { speed_mps: 4.4, from_deg: 270, gust_factor: 1.4, gust_rate_per_min: 1.5 },
      thunder: {
        enabled: false,
        distance_km: 8,
        bearing_deg: 270,
        approach_kmh: 0,
        flashes_per_min: 2,
        stroke_intensity: 0.5,
        rumble: 'auto',
      },
      temperature_c: 12,
      humidity_pct: 90,
      hour: 22,
      season: 'autumn',
      storm_arc: 'steady',
      arc_minutes: 20,
    },
    place: {
      preset: 'garden',
      shelter: 'open',
      surfaces: { grass: 40, soil: 20, leaves: 20, puddle: 10, concrete: 10 },
      window_open: 0,
      wipers: 'off',
      objects: [],
      biophony: {
        birds: 'auto', crows: 'auto', owls: 'auto', insects: 'auto', crickets: 'auto', katydids: 'auto',
        cicadas: 'auto', frogs: 'auto', dogs: 'auto', traffic: 0, stream: 0, chimes: 0,
      },
    },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 0, render: 'binaural' },
    mode: 'none',
    timeline: [],
    overrides: {},
    vibe: {},
    studio: {
      duration: 300,
      loop: [0, 300],
      loopOn: false,
      smoothing: { rain: 1.5, wind: 1, distance: 2, other: 0.25 },
      units: 'metric',
      labels: 'wmo',
    },
  };
}

/* ------------------------------------------------------------------ manifest */

const opts = (...pairs) => pairs.map((p) => (Array.isArray(p) ? p : [p, p[0].toUpperCase() + p.slice(1).replace(/_/g, ' ')]));

export const PRECIP_TYPES = [
  ['none', 'None (dry, drips only)'], ['drizzle', 'Drizzle'], ['rain', 'Rain'], ['showers', 'Showers'],
  ['freezing_rain', 'Freezing rain'], ['sleet', 'Sleet / ice pellets'], ['snow', 'Snow'],
  ['graupel', 'Graupel'], ['hail', 'Hail'],
];

export const PLACE_PRESETS = opts(
  'garden', ['deciduous_forest', 'Deciduous forest'], ['conifer_forest', 'Conifer forest'], 'meadow', 'coast', 'lake',
  ['city_street', 'City street'], 'suburb', 'alley', ['mountain_ridge', 'Mountain ridge'], 'desert', 'jungle', 'farm',
);

export const SHELTERS = [
  ['open', 'Open (out in it)'], ['tree', 'Under a tree'], ['umbrella', 'Umbrella'], ['tent', 'Tent'],
  ['porch', 'Porch'], ['car', 'Car'], ['room_open', 'Room, window open'], ['room_closed', 'Room, window closed'],
  ['attic', 'Attic / tin roof'], ['cave', 'Cave mouth'], ['underpass', 'Underpass'],
];

export const SURFACE_KEYS = [
  'leaves', 'grass', 'soil', 'puddle', 'asphalt', 'concrete', 'gravel', 'metal', 'tile', 'glass', 'fabric', 'wood', 'car_body', 'snow', 'needles',
];

/**
 * Sounds that come with rain. Each living category is "auto" (the engine's ecology
 * decides from season, hour, temperature and rain) or a level 0..1 (0 = off).
 * [key, label, one-line ecology hint, level used when a row is switched On].
 */
export const BIOPHONY = [
  ['birds', 'Songbirds', 'Songbirds sing most at dawn, hush in heavy rain and start again as it clears.', 0.6],
  ['crows', 'Crows', 'Crows call by day, during and after rain.', 0.5],
  ['owls', 'Owls', 'Owls call at night, most on calm nights; heavy rain quiets them.', 0.5],
  ['crickets', 'Crickets', 'Crickets chirp on warm evenings, faster as it warms; heavy rain quiets them.', 0.6],
  ['katydids', 'Katydids', 'Katydids rasp through warm late-summer nights.', 0.5],
  ['cicadas', 'Cicadas', 'Cicadas sing on hot summer days and stop when it rains.', 0.5],
  ['frogs', 'Frogs', 'Frogs call most on warm nights during and after rain.', 0.6],
  ['dogs', 'Distant dogs', 'Far-off dogs bark now and then, more in the evening; thunder can set them off.', 0.4],
];
/** Plain 0..1 levels of the place (0 = off). [key, label, hint, level used when switched On]. */
export const AMBIENT_LEVELS = [
  ['traffic', 'Wet-road traffic', 'Tyres hiss and splash on wet roads; the wetter the road, the louder the hiss.', 0.4],
  ['stream', 'Nearby stream', 'A brook close by swells and grows louder as the rain goes on.', 0.5],
  ['chimes', 'Wind chimes', 'Chimes ring when gusts pass and fall still in a lull.', 0.4],
];
/** Legacy `insects` key: a master over these three rows. */
export const INSECT_KEYS = ['crickets', 'katydids', 'cicadas'];
export const RUMBLE_HINT = 'Far-off storms keep grumbling between the nearer strikes.';

const pct = (v) => (v === 'auto' ? 'Auto' : typeof v === 'number' ? (v <= 0 ? 'Off' : `${Math.round(v * 100)} %`) : String(v));

/**
 * "auto" or a level 0..1. Accepts the older "on" / "off" words (on = full level),
 * booleans and numeric strings; anything else falls back to `d`.
 */
export function autoLevel(v, d = 'auto') {
  if (v === undefined || v === null) return d;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'number') return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : d;
  if (typeof v === 'string') {
    const t = v.trim().toLowerCase();
    if (t === 'auto') return 'auto';
    if (t === 'on' || t === 'true' || t === 'yes') return 1;
    if (t === 'off' || t === 'false' || t === 'no' || t === 'none') return 0;
    const n = parseFloat(t);
    if (Number.isFinite(n)) return Math.min(1, Math.max(0, n));
  }
  return d;
}

/** Effective value of one sound row: an Auto insect row follows a non-auto legacy `insects` master. */
export function soundValue(scene, key) {
  if (key === 'rumble') return autoLevel(scene?.weather?.thunder?.rumble);
  const bio = scene?.place?.biophony || {};
  if (AMBIENT_LEVELS.some((a) => a[0] === key)) {
    const lv = autoLevel(bio[key], 0);
    return lv === 'auto' ? 0 : lv;
  }
  const v = autoLevel(bio[key]);
  if (v === 'auto' && INSECT_KEYS.includes(key)) {
    const m = autoLevel(bio.insects);
    if (m !== 'auto') return m;
  }
  return v;
}

/** Store path of a sound row key. */
export function soundPath(key) {
  return key === 'rumble' ? 'weather.thunder.rumble' : `place.biophony.${key}`;
}

/**
 * setMany() patch for one sound row. Editing an insect row while the legacy master
 * holds them releases the master into the three rows, so nothing is applied twice.
 */
export function soundPatch(scene, key, value) {
  const out = { [soundPath(key)]: value };
  const bio = scene?.place?.biophony || {};
  if (INSECT_KEYS.includes(key) && autoLevel(bio.insects) !== 'auto') {
    for (const k of INSECT_KEYS) if (k !== key) out[soundPath(k)] = soundValue(scene, k);
    out['place.biophony.insects'] = 'auto';
  }
  return out;
}

/**
 * Parameter manifest. scale: 'linear' | 'log' | 'beaufort'. smoothing group keys
 * into scene.studio.smoothing (seconds to 99.5 %).
 */
export const PARAMS = [
  // Weather
  { path: 'weather.precipitation.type', label: 'Precipitation', kind: 'enum', options: PRECIP_TYPES, group: 'weather', section: 'Precipitation' },
  { path: 'weather.precipitation.rate_mm_h', label: 'Amount', kind: 'number', scale: 'log', min: 0.02, max: 150, unit: 'mm/h', unitSpoken: 'millimetres per hour', group: 'weather', section: 'Precipitation', smoothing: 'rain' },
  { path: 'weather.precipitation.drops', label: 'Drop character', kind: 'enum', options: [['fine', 'Fine'], ['normal', 'Normal'], ['fat', 'Fat']], group: 'weather', section: 'Precipitation', help: 'Moves Dm along the stratiform / maritime / continental clusters.' },
  { path: 'weather.precipitation.character', label: 'Character', kind: 'enum', options: [['continuous', 'Continuous'], ['intermittent', 'Intermittent'], ['showery', 'Showery']], group: 'weather', section: 'Precipitation' },
  { path: 'weather.precipitation.shower_period_min', label: 'Shower cell period', kind: 'number', min: 5, max: 30, step: 1, unit: 'min', unitSpoken: 'minutes', group: 'weather', section: 'Precipitation', showIf: (s) => s.weather.precipitation.character !== 'continuous' },
  { path: 'weather.precipitation.hail_size_mm', label: 'Hail size', kind: 'number', min: 6, max: 44, step: 1, unit: 'mm', unitSpoken: 'millimetres', group: 'weather', section: 'Precipitation', showIf: (s) => s.weather.precipitation.type === 'hail' },
  { path: 'weather.wind.speed_mps', label: 'Wind', kind: 'number', scale: 'beaufort', min: 0, max: 34.8, unit: 'm/s', group: 'weather', section: 'Wind', smoothing: 'wind' },
  { path: 'weather.wind.gust_factor', label: 'Gust factor', kind: 'number', min: 1, max: 2, step: 0.01, unit: '×', unitSpoken: 'times', group: 'weather', section: 'Wind', smoothing: 'wind' },
  { path: 'weather.wind.from_deg', label: 'Wind from', kind: 'compass', min: 0, max: 360, step: 1, unit: '°', group: 'weather', section: 'Wind', smoothing: 'wind' },
  { path: 'weather.thunder.enabled', label: 'Thunder', kind: 'bool', group: 'weather', section: 'Thunder' },
  { path: 'weather.thunder.distance_km', label: 'Thunder distance', kind: 'number', scale: 'log', min: 0.3, max: 25, unit: 'km', unitSpoken: 'kilometres', group: 'weather', section: 'Thunder', smoothing: 'distance' },
  { path: 'weather.thunder.bearing_deg', label: 'Thunder bearing', kind: 'compass', min: 0, max: 360, step: 1, unit: '°', group: 'weather', section: 'Thunder', smoothing: 'distance' },
  { path: 'weather.thunder.approach_kmh', label: 'Approach', kind: 'number', min: -60, max: 60, step: 1, unit: 'km/h', unitSpoken: 'kilometres per hour', group: 'weather', section: 'Thunder' },
  { path: 'weather.thunder.flashes_per_min', label: 'Flashes', kind: 'number', min: 0, max: 20, step: 0.1, unit: '/min', unitSpoken: 'per minute', group: 'weather', section: 'Thunder' },
  { path: 'weather.thunder.stroke_intensity', label: 'Stroke intensity', kind: 'number', min: 0, max: 1, step: 0.01, unit: '', group: 'weather', section: 'Thunder' },
  { path: 'weather.thunder.rumble', label: 'Cloud rumble', kind: 'autolevel', min: 0, max: 1, step: 0.01, unit: '', default: 'auto', onLevel: 0.5, group: 'weather', section: 'Thunder', help: RUMBLE_HINT, format: pct },
  { path: 'weather.temperature_c', label: 'Temperature', kind: 'number', min: -30, max: 45, step: 0.5, unit: '°C', unitSpoken: 'degrees Celsius', group: 'weather', section: 'Air & time', tag: 'indirect: crickets, sleet/snow, flash-to-bang' },
  { path: 'weather.humidity_pct', label: 'Humidity', kind: 'number', min: 10, max: 100, step: 1, unit: '%', unitSpoken: 'percent', group: 'weather', section: 'Air & time', tag: 'distant brightness only' },
  { path: 'weather.hour', label: 'Time of day', kind: 'hour', min: 0, max: 24, step: 0.25, unit: 'h', group: 'weather', section: 'Air & time' },
  { path: 'weather.season', label: 'Season', kind: 'enum', options: opts('spring', 'summer', 'autumn', 'winter'), group: 'weather', section: 'Air & time' },
  { path: 'weather.storm_arc', label: 'Storm arc', kind: 'enum', options: opts('arriving', 'steady', 'passing', 'clearing'), group: 'weather', section: 'Air & time' },
  { path: 'weather.arc_minutes', label: 'Arc length', kind: 'number', min: 5, max: 120, step: 1, unit: 'min', unitSpoken: 'minutes', group: 'weather', section: 'Air & time' },

  // Place & listener
  { path: 'place.preset', label: 'Place', kind: 'enum', options: PLACE_PRESETS, group: 'place', section: 'Place' },
  { path: 'place.shelter', label: 'Shelter', kind: 'enum', options: SHELTERS, group: 'place', section: 'Place' },
  { path: 'place.window_open', label: 'Window open', kind: 'number', min: 0, max: 1, step: 0.01, unit: '', group: 'place', section: 'Place', format: (v) => `${Math.round(v * 100)} %` },
  ...BIOPHONY.map(([k, label, help, onLevel]) => ({ path: `place.biophony.${k}`, label, kind: 'autolevel', min: 0, max: 1, step: 0.01, unit: '', default: 'auto', onLevel, group: 'place', section: 'Sounds of the place', help, format: pct })),
  ...AMBIENT_LEVELS.map(([k, label, help, onLevel]) => ({ path: `place.biophony.${k}`, label, kind: 'level', min: 0, max: 1, step: 0.01, unit: '', default: 0, onLevel, group: 'place', section: 'Sounds of the place', help, format: pct })),
  { path: 'place.wipers', label: 'Wipers', kind: 'enum', options: opts('off', 'slow', 'fast'), group: 'place', section: 'Place', showIf: (s) => s.place.shelter === 'car' || s.place.objects.some((o) => o.type === 'car') },
  { path: 'listener.height_m', label: 'Listener height', kind: 'number', scale: 'log', min: 0.3, max: 30, unit: 'm', unitSpoken: 'metres', group: 'listener', section: 'Listener' },
  { path: 'listener.heading_deg', label: 'Heading', kind: 'compass', min: 0, max: 360, step: 1, unit: '°', group: 'listener', section: 'Listener' },
  { path: 'listener.render', label: 'Render', kind: 'enum', options: [['binaural', 'Binaural (headphones)'], ['stereo', 'Stereo (speakers)']], group: 'listener', section: 'Listener' },
  { path: 'mode', label: 'Mode', kind: 'enum', options: [['none', 'None'], ['sleep', 'Sleep'], ['focus', 'Focus'], ['calm', 'Relax'], ['asmr', 'ASMR']], group: 'listener', section: 'Listener' },

  // Engineer: physics, randomness and smoothing
  { path: 'weather.precipitation.mu', label: 'DSD shape μ', kind: 'number', min: -2, max: 10, step: 0.1, unit: '', group: 'engineer', section: 'Engineer', nullable: true },
  { path: 'weather.precipitation.dm_mm', label: 'Dm override', kind: 'number', min: 0.5, max: 4, step: 0.05, unit: 'mm', unitSpoken: 'millimetres', group: 'engineer', section: 'Engineer', nullable: true },
  { path: 'weather.precipitation.class_boundaries', label: 'Class boundaries', kind: 'enum', options: [['medwin', 'Medwin (0.8/1.1/2.2/3.5 mm)'], ['nystuen', 'Nystuen (0.8/1.2/2.0/3.5 mm)']], group: 'engineer', section: 'Engineer' },
  { path: 'weather.precipitation.water_quality', label: 'Water quality', kind: 'number', min: 0, max: 1, step: 0.01, unit: '', group: 'engineer', section: 'Engineer', help: '1 clean ... 0 soapy (surfactant stops bubble plinks).' },
  { path: 'weather.wind.gust_rate_per_min', label: 'Gust rate', kind: 'number', min: 0.2, max: 6, step: 0.1, unit: '/min', unitSpoken: 'per minute', group: 'engineer', section: 'Engineer' },
];

export const PARAM_BY_PATH = new Map(PARAMS.map((p) => [p.path, p]));

/** Spec lookup that also covers surface-mix and object paths. */
export function paramSpec(path) {
  const s = PARAM_BY_PATH.get(path);
  if (s) return s;
  if (path.startsWith('place.surfaces.')) {
    const k = path.split('.')[2];
    return { path, label: `Surface: ${k.replace(/_/g, ' ')}`, kind: 'number', min: 0, max: 100, step: 1, unit: '%', unitSpoken: 'percent', group: 'place' };
  }
  if (path === 'listener.position') return { path, label: 'Listener position', kind: 'vec3', group: 'listener' };
  const m = /^place\.objects\.([^.]+)\.(.+)$/.exec(path);
  if (m) {
    const sub = m[2];
    if (sub === 'position') return { path, label: `${m[1]} position`, kind: 'vec3', group: 'object' };
    if (sub === 'rotation_deg') return { path, label: `${m[1]} rotation`, kind: 'number', min: -360, max: 360, unit: '°', group: 'object' };
    return { path, label: `${m[1]} ${sub.replace(/^params\./, '')}`, kind: 'number', min: 0, max: 100, group: 'object' };
  }
  return { path, label: path.split('.').pop().replace(/_/g, ' '), kind: 'number', min: 0, max: 100, group: 'other' };
}

/* ------------------------------------------------------------------ utilities */

export function clone(v) {
  if (v === undefined) return undefined;
  if (typeof structuredClone === 'function') {
    try { return structuredClone(v); } catch (_) { /* functions etc. */ }
  }
  return JSON.parse(JSON.stringify(v));
}

export function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null || typeof a !== 'object') {
    return typeof a === 'number' && typeof b === 'number' && Number.isNaN(a) && Number.isNaN(b);
  }
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!deepEqual(a[k], b[k])) return false;
  return true;
}

const REPLACE_KEYS = new Set(['surfaces', 'objects', 'timeline', 'params', 'track', 'position', 'size', 'overrides', 'vibe', 'loop']);

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

export function deepMerge(base, patch) {
  if (patch === undefined) return base;
  if (patch === null || typeof patch !== 'object' || Array.isArray(patch)) return clone(patch);
  const out = isPlainObject(base) ? { ...base } : {};
  for (const [k, v] of Object.entries(patch)) {
    const cur = out[k];
    // A value of the wrong shape (a string where a section belongs, an object
    // where a list belongs) is ignored so a damaged file cannot break the scene.
    if (isPlainObject(cur) && !isPlainObject(v)) continue;
    if (Array.isArray(cur) && !Array.isArray(v)) continue;
    if (REPLACE_KEYS.has(k) || v === null || typeof v !== 'object' || Array.isArray(v)) out[k] = clone(v);
    else out[k] = deepMerge(cur, v);
  }
  return out;
}

function step(cur, key) {
  if (cur == null) return undefined;
  if (Array.isArray(cur)) {
    const byId = cur.find((o) => o && typeof o === 'object' && o.id === key);
    if (byId) return byId;
    return /^\d+$/.test(key) ? cur[Number(key)] : undefined;
  }
  return cur[key];
}

/** Dotted path get; arrays are addressed by element id (or numeric index). */
export function getPath(obj, path) {
  if (!path) return obj;
  let cur = obj;
  for (const k of path.split('.')) {
    cur = step(cur, k);
    if (cur === undefined) return undefined;
  }
  return cur;
}

export function setPath(obj, path, value) {
  const parts = path.split('.');
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    let next = step(cur, parts[i]);
    if (next === undefined || next === null) {
      if (Array.isArray(cur)) return false;
      next = {};
      cur[parts[i]] = next;
    }
    cur = next;
  }
  const last = parts[parts.length - 1];
  if (Array.isArray(cur)) {
    const idx = cur.findIndex((o) => o && o.id === last);
    if (idx >= 0) cur[idx] = value;
    else if (/^\d+$/.test(last)) cur[Number(last)] = value;
    else return false;
  } else {
    cur[last] = value;
  }
  return true;
}

let uidCounter = 0;
export function uid(prefix = 'o') {
  uidCounter = (uidCounter + 1) % 1296;
  const t = Date.now().toString(36).slice(-4);
  return `${prefix}-${t}${uidCounter.toString(36).padStart(2, '0')}`;
}

function num(v, d) {
  const n = typeof v === 'string' ? parseFloat(v) : v;
  return Number.isFinite(n) ? n : d;
}

function vec3(v, d = [0, 0, 0]) {
  if (!Array.isArray(v)) return d.slice();
  return [num(v[0], d[0]), num(v[1], d[1]), num(v[2], d[2])];
}

export function normalizeObject(o) {
  const type = String(o.type || 'surface');
  const out = {
    id: String(o.id || uid(type)),
    type,
    material: String(o.material || ''),
    position: vec3(o.position),
    size: vec3(o.size, [1, 1, 0]),
    rotation_deg: num(o.rotation_deg, 0),
    params: o.params && typeof o.params === 'object' ? clone(o.params) : {},
  };
  if (o.label) out.label = String(o.label);
  return out;
}

/**
 * Give every number / boolean / string leaf the type of its default. A numeric
 * null stays null: Python scene files use it for "automatic".
 */
function coerceLeaves(def, val) {
  for (const [k, d] of Object.entries(def)) {
    if (!(k in val)) continue;
    const v = val[k];
    if (typeof d === 'number') {
      if (v !== null && (typeof v !== 'number' || !Number.isFinite(v))) val[k] = num(v, d);
    } else if (typeof d === 'boolean') {
      if (typeof v !== 'boolean') val[k] = v === 'true' || v === 1;
    } else if (typeof d === 'string') {
      if (d === 'auto') continue; // "auto" or a number: normalised by normalizeSounds()
      if (typeof v !== 'string') val[k] = d;
    } else if (isPlainObject(d) && isPlainObject(v) && !REPLACE_KEYS.has(k)) {
      coerceLeaves(d, v);
    }
  }
}

/**
 * Sounds of the place and the cloud rumble: every row becomes "auto" or 0..1. A legacy
 * `insects` master (Python presets, older files) is expanded into the crickets / katydids /
 * cicadas rows when the file sets none of them, then released to "auto".
 */
function normalizeSounds(s, src) {
  const bio = s.place.biophony;
  const srcBio = isPlainObject(src?.place?.biophony) ? src.place.biophony : {};
  for (const [k] of BIOPHONY) bio[k] = autoLevel(bio[k]);
  for (const [k] of AMBIENT_LEVELS) {
    const v = autoLevel(bio[k], 0);
    bio[k] = v === 'auto' ? 0 : v;
  }
  const master = autoLevel(bio.insects);
  if (master !== 'auto' && INSECT_KEYS.every((k) => srcBio[k] === undefined)) {
    for (const k of INSECT_KEYS) bio[k] = master;
    bio.insects = 'auto';
  } else {
    bio.insects = master;
  }
  s.weather.thunder.rumble = autoLevel(s.weather.thunder.rumble);
}

/** Normalise any rainscene-like object (accepts Python `version` key too). */
export function normalizeScene(input) {
  const src = input && typeof input === 'object' ? clone(input) : {};
  if (src.version !== undefined && src.rainscene === undefined) src.rainscene = src.version;
  delete src.version;
  const s = deepMerge(defaultScene(), src);
  coerceLeaves(defaultScene(), s);
  s.rainscene = SCHEMA_VERSION;
  s.name = typeof s.name === 'string' && s.name.trim() ? s.name.slice(0, 120) : 'Untitled weather';
  s.seed = Math.max(0, Math.floor(num(s.seed, 0x5eedcafe))) % 2 ** 52;
  const p = s.weather.precipitation;
  p.rate_mm_h = Math.min(150, Math.max(0.02, num(p.rate_mm_h, 4)));
  const w = s.weather.wind;
  w.speed_mps = Math.min(40, Math.max(0, num(w.speed_mps, 4.4)));
  w.from_deg = ((num(w.from_deg, 270) % 360) + 360) % 360;
  if (w.gust_factor === null || w.gust_factor === undefined || w.gust_factor === 'auto') w.gust_factor = 1.4;
  normalizeSounds(s, src);
  const seen = new Set();
  s.place.objects = (Array.isArray(s.place.objects) ? s.place.objects : []).filter(isPlainObject).map((o) => {
    const n = normalizeObject(o);
    while (seen.has(n.id)) n.id = uid(n.type);
    seen.add(n.id);
    return n;
  });
  s.listener.position = vec3(s.listener.position);
  s.timeline = (Array.isArray(s.timeline) ? s.timeline : [])
    .filter((k) => isPlainObject(k) && typeof k.path === 'string')
    .map((k) => ({ id: k.id || uid('k'), at: Math.max(0, num(typeof k.at === 'string' ? parseTimecode(k.at) : k.at, 0)), path: k.path, value: clone(k.value), curve: ['linear', 'log', 'hold', 'smooth'].includes(k.curve) ? k.curve : 'linear' }))
    .sort((a, b) => a.at - b.at);
  if (!s.overrides || typeof s.overrides !== 'object') s.overrides = {};
  if (!s.vibe || typeof s.vibe !== 'object') s.vibe = {};
  const st = s.studio;
  st.duration = Math.max(10, num(st.duration, 300));
  if (!Array.isArray(st.loop) || st.loop.length !== 2) st.loop = [0, st.duration];
  st.loop = [Math.max(0, num(st.loop[0], 0)), Math.max(0, num(st.loop[1], st.duration))];
  return s;
}

/** True when parsed JSON plausibly is a rainscene (or a Python scene file). */
export function looksLikeScene(data) {
  return isPlainObject(data) && (data.rainscene !== undefined || data.version !== undefined || isPlainObject(data.weather));
}

export function parseTimecode(str) {
  if (typeof str === 'number') return str;
  const parts = String(str).trim().split(':').map((x) => parseFloat(x));
  if (parts.some((x) => !Number.isFinite(x))) return 0;
  return parts.reduce((acc, x) => acc * 60 + x, 0);
}

/* ------------------------------------------------------------------ base64url */

export function b64urlEncode(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function b64urlDecode(s) {
  let t = s.replace(/-/g, '+').replace(/_/g, '/');
  while (t.length % 4) t += '=';
  const bin = atob(t);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

/* ------------------------------------------------------------------ offsets */

function toScale(spec, v) {
  if (spec.scale === 'log') return Math.log10(Math.max(1e-6, v));
  if (spec.scale === 'beaufort') return Math.pow(Math.max(0, v) / 0.836, 2 / 3);
  return v;
}

function fromScale(spec, x) {
  if (spec.scale === 'log') return Math.pow(10, x);
  if (spec.scale === 'beaufort') return 0.836 * Math.pow(Math.max(0, x), 1.5);
  return x;
}

function clampSpec(spec, v) {
  if (typeof v !== 'number' || !spec) return v;
  let out = v;
  if (spec.min !== undefined) out = Math.max(spec.min, out);
  if (spec.max !== undefined) out = Math.min(spec.max, out);
  return out;
}

/* ------------------------------------------------------------------ store */

export class SceneStore {
  constructor({ storageKey = 'raingen.studio' } = {}) {
    this.storageKey = storageKey;
    this.scene = defaultScene();
    this.selection = null;
    this._listeners = new Map();
    this._undo = [];
    this._redo = [];
    this._gesture = null;
    this._last = { key: null, t: 0 };
    this._saveTimer = 0;
    this._driverFn = null;
    this.drivers = {};
  }

  /* ---------------- events */
  on(type, cb) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(cb);
    return () => this.off(type, cb);
  }

  off(type, cb) {
    this._listeners.get(type)?.delete(cb);
  }

  _emit(type, detail) {
    const set = this._listeners.get(type);
    if (!set) return;
    for (const cb of [...set]) {
      try { cb(detail); } catch (err) { console.error(`[scene-store] ${type} listener failed`, err); }
    }
  }

  _emitHistory() {
    this._emit('history', { canUndo: this._undo.length > 0, canRedo: this._redo.length > 0 });
  }

  /* ---------------- access */
  get(path) {
    return getPath(this.scene, path);
  }

  getObject(id) {
    return this.scene.place.objects.find((o) => o.id === id) || null;
  }

  /* ---------------- history */
  _snapshot() {
    return JSON.stringify(this.scene);
  }

  _checkpoint(key) {
    if (this._gesture) {
      this._gesture.changed = true;
      return;
    }
    const now = performance.now();
    if (key && key === this._last.key && now - this._last.t < 700) {
      this._last.t = now;
      return;
    }
    this._pushUndo(this._snapshot());
    this._last = { key, t: now };
  }

  _pushUndo(snap) {
    this._undo.push(snap);
    if (this._undo.length > 200) this._undo.shift();
    this._redo.length = 0;
    this._emitHistory();
  }

  beginGesture(label = 'edit') {
    if (this._gesture) {
      this._gesture.depth++;
      return;
    }
    this._gesture = { label, depth: 1, pre: this._snapshot(), changed: false };
    this._emit('gesture', { active: true, label });
  }

  endGesture() {
    const g = this._gesture;
    if (!g) return;
    g.depth--;
    if (g.depth > 0) return;
    this._gesture = null;
    if (g.changed && g.pre !== this._snapshot()) this._pushUndo(g.pre);
    this._last = { key: null, t: 0 };
    this._scheduleSave();
    this._emit('gesture', { active: false, label: g.label, changed: g.changed });
  }

  /** True while a drag (or another multi-step gesture) is in progress. */
  get gestureActive() {
    return !!this._gesture;
  }

  get canUndo() {
    return this._undo.length > 0;
  }

  get canRedo() {
    return this._redo.length > 0;
  }

  undo() {
    if (this._gesture) this.endGesture();
    const snap = this._undo.pop();
    if (!snap) return false;
    this._redo.push(this._snapshot());
    this._restore(snap);
    return true;
  }

  redo() {
    const snap = this._redo.pop();
    if (!snap) return false;
    this._undo.push(this._snapshot());
    this._restore(snap);
    return true;
  }

  _restore(snap) {
    this.scene = normalizeScene(JSON.parse(snap));
    this._recomputeDrivers();
    if (this.selection && !this.getObject(this.selection)) this.select(null);
    this._last = { key: null, t: 0 };
    this._emit('change', { paths: ['*'], source: 'history', kind: 'restore' });
    this._emitHistory();
    this._scheduleSave();
  }

  /* ---------------- edits */
  /**
   * Set one value. opts: source ('user' | 'automation' | 'walk' | 'demo' | 'vibe' | ...),
   * history (record an undo step), coalesce (merge rapid edits with the same key).
   */
  set(path, value, { source = 'user', history = true, coalesce = path } = {}) {
    const old = getPath(this.scene, path);
    if (deepEqual(old, value)) return false;
    if (history) this._checkpoint(coalesce);
    if (source === 'user') this._noteUserEdit(path, value);
    setPath(this.scene, path, clone(value));
    this._emit('change', { paths: [path], source, kind: path.startsWith('place.objects.') ? 'object' : 'set' });
    if (history || source === 'user') this._scheduleSave();
    return true;
  }

  setMany(map, { source = 'user', history = true, coalesce = null } = {}) {
    const changed = [];
    let checkpointed = false;
    for (const [path, value] of Object.entries(map)) {
      if (deepEqual(getPath(this.scene, path), value)) continue;
      if (history && !checkpointed) {
        this._checkpoint(coalesce || `many:${Object.keys(map).join(',')}`);
        checkpointed = true;
      }
      if (source === 'user') this._noteUserEdit(path, value);
      setPath(this.scene, path, clone(value));
      changed.push(path);
    }
    if (changed.length) {
      this._emit('change', { paths: changed, source, kind: 'set' });
      this._scheduleSave();
    }
    return changed;
  }

  /** Values from timeline automation: no undo, no override bookkeeping. */
  applyAutomation(values) {
    const changed = [];
    for (const [path, value] of Object.entries(values)) {
      const cur = getPath(this.scene, path);
      if (cur === undefined) continue;
      if (deepEqual(cur, value)) continue;
      if (typeof cur === 'number' && typeof value === 'number' && Math.abs(cur - value) < 1e-6 * (1 + Math.abs(value))) continue;
      setPath(this.scene, path, clone(value));
      changed.push(path);
    }
    if (changed.length) this._emit('change', { paths: changed, source: 'automation', kind: 'set' });
    return changed;
  }

  /* ---------------- whole scene */
  load(scene, { history = true, source = 'load', label = '' } = {}) {
    const next = normalizeScene(scene);
    if (history) {
      if (this._gesture) this._gesture.changed = true;
      else this._pushUndo(this._snapshot());
    }
    this.scene = next;
    this._recomputeDrivers();
    this.selection = null;
    this._last = { key: null, t: 0 };
    this._emit('change', { paths: ['*'], source, kind: 'load', label });
    this._emit('select', { id: null });
    this._scheduleSave();
  }

  rename(name) {
    this.set('name', String(name || 'Untitled weather').slice(0, 120), { coalesce: 'name' });
  }

  /* ---------------- objects */
  addObject(obj, { select = true, history = true } = {}) {
    const o = normalizeObject(obj);
    while (this.getObject(o.id)) o.id = uid(o.type);
    if (history) this._checkpoint(null);
    this.scene.place.objects.push(o);
    this._emit('change', { paths: [`place.objects.${o.id}`], source: 'user', kind: 'object-add', id: o.id });
    this._scheduleSave();
    if (select) this.select(o.id);
    return o.id;
  }

  updateObject(id, patch, { history = true, source = 'user', coalesce = `obj:${id}` } = {}) {
    const o = this.getObject(id);
    if (!o) return false;
    const merged = normalizeObject(deepMerge(o, patch));
    merged.id = id;
    if (patch.params) merged.params = { ...o.params, ...patch.params };
    if (deepEqual(o, merged)) return false;
    if (history) this._checkpoint(coalesce);
    const idx = this.scene.place.objects.indexOf(o);
    this.scene.place.objects[idx] = merged;
    this._emit('change', { paths: [`place.objects.${id}`], source, kind: 'object', id });
    if (history) this._scheduleSave();
    return true;
  }

  removeObject(id, { history = true } = {}) {
    const idx = this.scene.place.objects.findIndex((o) => o.id === id);
    if (idx < 0) return false;
    if (history) this._checkpoint(null);
    this.scene.place.objects.splice(idx, 1);
    const prefix = `place.objects.${id}.`;
    this.scene.timeline = this.scene.timeline.filter((k) => !k.path.startsWith(prefix));
    this._emit('change', { paths: [`place.objects.${id}`, 'timeline'], source: 'user', kind: 'object-remove', id });
    if (this.selection === id) this.select(null);
    this._scheduleSave();
    return true;
  }

  duplicateObject(id) {
    const o = this.getObject(id);
    if (!o) return null;
    const c = clone(o);
    c.id = uid(o.type);
    c.position = [o.position[0] + 1, o.position[1] + 1, o.position[2]];
    if (o.type === 'storm' && Array.isArray(c.params.track)) {
      c.params.track = c.params.track.map(([x, y]) => [x + 500, y + 500]);
    }
    if (o.label) c.label = `${o.label} copy`;
    return this.addObject(c);
  }

  select(id) {
    if (this.selection === id) return;
    this.selection = id;
    this._emit('select', { id });
  }

  /* ---------------- timeline */
  addKeyframe({ path, at, value, curve = 'linear' }, { history = true, coalesce = null } = {}) {
    if (history) this._checkpoint(coalesce);
    const tl = this.scene.timeline;
    const existing = tl.find((k) => k.path === path && Math.abs(k.at - at) < 0.02);
    let id;
    if (existing) {
      existing.value = clone(value);
      existing.curve = curve || existing.curve;
      id = existing.id;
    } else {
      id = uid('k');
      tl.push({ id, at: Math.max(0, at), path, value: clone(value), curve });
      tl.sort((a, b) => a.at - b.at);
    }
    this._emit('change', { paths: ['timeline'], source: 'user', kind: 'timeline' });
    if (history) this._scheduleSave();
    return id;
  }

  updateKeyframe(id, patch, { history = true, coalesce = `kf:${id}` } = {}) {
    const k = this.scene.timeline.find((x) => x.id === id);
    if (!k) return false;
    if (history) this._checkpoint(coalesce);
    if (patch.at !== undefined) k.at = Math.max(0, patch.at);
    if (patch.value !== undefined) k.value = clone(patch.value);
    if (patch.curve !== undefined) k.curve = patch.curve;
    this.scene.timeline.sort((a, b) => a.at - b.at);
    this._emit('change', { paths: ['timeline'], source: 'user', kind: 'timeline' });
    if (history) this._scheduleSave();
    return true;
  }

  removeKeyframe(id) {
    const idx = this.scene.timeline.findIndex((x) => x.id === id);
    if (idx < 0) return false;
    this._checkpoint(null);
    this.scene.timeline.splice(idx, 1);
    this._emit('change', { paths: ['timeline'], source: 'user', kind: 'timeline' });
    this._scheduleSave();
    return true;
  }

  removeTrack(path) {
    const before = this.scene.timeline.length;
    if (!this.scene.timeline.some((k) => k.path === path)) return false;
    this._checkpoint(null);
    this.scene.timeline = this.scene.timeline.filter((k) => k.path !== path);
    this._emit('change', { paths: ['timeline'], source: 'user', kind: 'timeline' });
    this._scheduleSave();
    return before !== this.scene.timeline.length;
  }

  replaceTimeline(keyframes, { label = 'timeline' } = {}) {
    this._checkpoint(null);
    this.scene.timeline = keyframes.map((k) => ({ id: k.id || uid('k'), at: k.at, path: k.path, value: clone(k.value), curve: k.curve || 'linear' })).sort((a, b) => a.at - b.at);
    this._emit('change', { paths: ['timeline'], source: 'user', kind: 'timeline', label });
    this._scheduleSave();
  }

  tracks() {
    const out = [];
    for (const k of this.scene.timeline) if (!out.includes(k.path)) out.push(k.path);
    return out;
  }

  hasTrack(path) {
    return this.scene.timeline.some((k) => k.path === path);
  }

  setStudio(key, value, { history = false } = {}) {
    this.set(`studio.${key}`, value, { source: 'studio', history });
  }

  /* ---------------- Auto / Offset / Pinned */
  setDriverFn(fn) {
    this._driverFn = fn;
    this._recomputeDrivers();
  }

  _recomputeDrivers() {
    this.drivers = {};
    if (!this._driverFn || !this.scene.vibe || !Array.isArray(this.scene.vibe.touched)) return;
    try {
      const groups = this._driverFn(this.scene.vibe, this.scene) || {};
      for (const g of this.scene.vibe.touched) Object.assign(this.drivers, groups[g] || {});
    } catch (err) {
      console.error('[scene-store] driver function failed', err);
    }
  }

  /** Driver value (where Auto would put the parameter), or undefined. */
  driverValue(path) {
    return this.drivers[path];
  }

  overrideOf(path) {
    const o = this.scene.overrides[path];
    const driven = Object.prototype.hasOwnProperty.call(this.drivers, path);
    if (this.hasTrack(path)) return { state: 'automation', driven, offset: 0 };
    if (!o) return { state: 'auto', driven, offset: 0 };
    return { state: o.state, driven, offset: o.offset || 0, value: o.value };
  }

  describeOverride(path) {
    const o = this.overrideOf(path);
    const spec = paramSpec(path);
    if (o.state === 'automation') return { state: 'automation', text: 'Anim', title: 'Driven by a timeline track' };
    if (o.state === 'pinned') return { state: 'pinned', text: 'Pinned', title: 'Ignores scene macros. Alt-click to re-link.' };
    if (o.state === 'offset') {
      let txt;
      if (spec.scale === 'log') txt = `×${(10 ** o.offset).toFixed(2)}`;
      else if (spec.scale === 'beaufort') txt = `${o.offset >= 0 ? '+' : ''}${o.offset.toFixed(1)} Bft`;
      else txt = `${o.offset >= 0 ? '+' : ''}${(+o.offset.toFixed(2))}${spec.unit ? ' ' + spec.unit : ''}`;
      return { state: 'offset', text: `Offset ${txt}`, title: 'Follows the macro plus your offset. Alt-click to re-link.' };
    }
    return { state: 'auto', text: 'Auto', title: o.driven ? 'Follows the Listener macros' : 'Scene value (no macro driving it yet)' };
  }

  _noteUserEdit(path, value) {
    if (!Object.prototype.hasOwnProperty.call(this.drivers, path)) {
      return;
    }
    const spec = paramSpec(path);
    const prev = this.scene.overrides[path];
    if (prev && prev.state === 'pinned') {
      this.scene.overrides[path] = { state: 'pinned', value: clone(value) };
      return;
    }
    if (typeof value === 'number' && typeof this.drivers[path] === 'number') {
      const offset = toScale(spec, value) - toScale(spec, this.drivers[path]);
      this.scene.overrides[path] = { state: 'offset', offset };
    } else {
      this.scene.overrides[path] = { state: 'pinned', value: clone(value) };
    }
  }

  /** Cycle Auto -> Offset -> Pinned -> Auto (enums: Auto <-> Pinned). */
  cycleOverride(path) {
    const cur = this.overrideOf(path);
    const value = this.get(path);
    const numeric = typeof value === 'number';
    let next;
    if (cur.state === 'auto') next = numeric && cur.driven ? 'offset' : 'pinned';
    else if (cur.state === 'offset') next = 'pinned';
    else next = 'auto';
    this._checkpoint(null);
    if (next === 'auto') {
      delete this.scene.overrides[path];
      this._applyDriver(path);
    } else if (next === 'offset') {
      const d = this.drivers[path];
      const spec = paramSpec(path);
      this.scene.overrides[path] = { state: 'offset', offset: typeof d === 'number' ? toScale(spec, value) - toScale(spec, d) : 0 };
    } else {
      this.scene.overrides[path] = { state: 'pinned', value: clone(value) };
    }
    this._emit('change', { paths: [path, `overrides`], source: 'user', kind: 'override' });
    this._scheduleSave();
    return next;
  }

  /** Alt-click: drop offset/pin and follow the driver again. */
  relink(path) {
    if (!this.scene.overrides[path]) return false;
    this._checkpoint(null);
    delete this.scene.overrides[path];
    this._applyDriver(path);
    this._emit('change', { paths: [path, 'overrides'], source: 'user', kind: 'override' });
    this._scheduleSave();
    return true;
  }

  _applyDriver(path) {
    if (!Object.prototype.hasOwnProperty.call(this.drivers, path)) return;
    setPath(this.scene, path, clone(this.drivers[path]));
  }

  _resolved(path, driver) {
    const o = this.scene.overrides[path];
    const spec = paramSpec(path);
    if (o && o.state === 'pinned') return o.value !== undefined ? o.value : getPath(this.scene, path);
    if (o && o.state === 'offset' && typeof driver === 'number') {
      return clampSpec(spec, fromScale(spec, toScale(spec, driver) + (o.offset || 0)));
    }
    return driver;
  }

  /**
   * Move the Listener vibe macros. `partial` = { intensity?, closeness?, wildness? }.
   * `base` captures the scene's own character the first time a macro is touched.
   */
  setVibe(partial, { history = true, base = null } = {}) {
    if (history) this._checkpoint('vibe');
    const v = { ...(this.scene.vibe || {}) };
    if (!v.base) {
      const s = this.scene;
      v.base = base || {
        type: s.weather.precipitation.type,
        shelter: s.place.shelter,
        thunder: !!s.weather.thunder.enabled,
        drops: s.weather.precipitation.drops,
      };
    }
    const touched = new Set(Array.isArray(v.touched) ? v.touched : []);
    for (const [k, val] of Object.entries(partial)) {
      if (typeof val !== 'number') continue;
      v[k] = Math.max(0, Math.min(100, val));
      touched.add(k);
    }
    v.touched = [...touched];
    this.scene.vibe = v;
    this._recomputeDrivers();
    const changed = ['vibe'];
    for (const [path, driver] of Object.entries(this.drivers)) {
      if (this.hasTrack(path)) continue;
      const val = this._resolved(path, driver);
      if (!deepEqual(getPath(this.scene, path), val)) {
        setPath(this.scene, path, clone(val));
        changed.push(path);
      }
    }
    this._emit('change', { paths: changed, source: 'vibe', kind: 'set' });
    this._scheduleSave();
  }

  /* ---------------- persistence */
  _scheduleSave() {
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => this.saveAutosave(), 600);
  }

  saveAutosave() {
    try {
      localStorage.setItem(`${this.storageKey}.autosave`, JSON.stringify(this.scene));
    } catch (_) { /* storage unavailable or full */ }
  }

  loadAutosave() {
    try {
      const txt = localStorage.getItem(`${this.storageKey}.autosave`);
      return txt ? normalizeScene(JSON.parse(txt)) : null;
    } catch (_) {
      return null;
    }
  }

  userPresets() {
    try {
      const txt = localStorage.getItem(`${this.storageKey}.presets`);
      const arr = txt ? JSON.parse(txt) : [];
      return Array.isArray(arr) ? arr.filter((p) => p && p.name && p.scene) : [];
    } catch (_) {
      return [];
    }
  }

  _writePresets(list) {
    try {
      localStorage.setItem(`${this.storageKey}.presets`, JSON.stringify(list));
      this._emit('presets', { list });
      return true;
    } catch (_) {
      return false;
    }
  }

  saveUserPreset(name) {
    const nm = String(name || this.scene.name || 'My weather').trim().slice(0, 80) || 'My weather';
    const list = this.userPresets().filter((p) => p.name !== nm);
    const scene = clone(this.scene);
    scene.name = nm;
    list.unshift({ name: nm, savedAt: new Date().toISOString(), scene });
    return this._writePresets(list.slice(0, 100)) ? nm : null;
  }

  deleteUserPreset(name) {
    return this._writePresets(this.userPresets().filter((p) => p.name !== name));
  }

  loadUserPreset(name) {
    const p = this.userPresets().find((x) => x.name === name);
    if (!p) return false;
    this.load(p.scene, { label: name });
    return true;
  }

  /* ---------------- interchange */
  toJSON(indent = 2) {
    const s = clone(this.scene);
    const ordered = { rainscene: SCHEMA_VERSION, ...s };
    return JSON.stringify(ordered, null, indent);
  }

  /** Load a rainscene JSON string; throws an Error with a reader-friendly message. */
  importJSON(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (_) {
      throw new Error('it is not valid JSON (the file may be damaged or cut short)');
    }
    if (!looksLikeScene(data)) {
      throw new Error('it is JSON, but not a rainscene file (no "rainscene" or "weather" section)');
    }
    try {
      this.load(data, { label: typeof data.name === 'string' ? data.name : 'imported' });
    } catch (_) {
      throw new Error('some of its values could not be read');
    }
  }

  toHash() {
    return `#scene=${b64urlEncode(JSON.stringify(this.scene))}`;
  }

  shareURL() {
    const u = new URL(location.href);
    u.hash = this.toHash().slice(1);
    return u.toString();
  }

  /** True when the URL hash carries a scene=… share payload (readable or not). */
  static hasSceneHash(hash) {
    return /(?:^#|&)scene=/.test(hash || '');
  }

  /** Scene from a share-link hash, or null when there is none or it cannot be read. */
  static decodeHash(hash) {
    const m = /(?:^#|&)scene=([A-Za-z0-9_-]+)(?:&|$)/.exec(hash || '');
    if (!m) return null;
    try {
      const data = JSON.parse(b64urlDecode(m[1]));
      return looksLikeScene(data) ? normalizeScene(data) : null;
    } catch (_) {
      return null;
    }
  }
}
