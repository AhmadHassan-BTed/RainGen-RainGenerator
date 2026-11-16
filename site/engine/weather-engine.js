/**
 * WeatherEngine: the real-time browser port of the raingen weather & environment sound
 * engine.
 *
 * - Control pre-pass at 20 Hz in f64 on the main thread: normalized-gamma drop-size
 *   distribution per drop regime and storm arc (Bringi et al. 2003), Medwin class impact
 *   fluxes, Atlas fall speeds, shower-cell envelopes, wetness stores, normalized Kaimal gusts
 *   advected across the scene by Taylor's hypothesis, log-law wind at each emitter height,
 *   ISO 9613-1 absorption cutoffs.
 * - Level of detail: resonant objects and 8 azimuth x 3 distance ground patches within
 *   ~3-9 listener heights are rain-surface worklets driven as Cox processes; the far field is a
 *   decorrelated 4-channel bed whose level follows the near field (monopole-like water surfaces
 *   carry a larger share).
 * - Spatial rendering: every voice -> PannerNode (HRTF in binaural mode, equal-power in
 *   stereo, inverse distance law); world (x east, y north, z up) -> Web Audio (x, z, -y).
 * - Places and shelters: near interior panels (umbrella, tent, car, porch, attic, room
 *   window) plus an exterior heard through an envelope transmission loss (low-pass, shelf, loss;
 *   window opening blends) and a generated room impulse response.
 * - Thunder: flashes scheduled per storm with counter-based thinning; fractal
 *   channels and rolls are synthesized ahead of their arrival, never in the audio callback;
 *   'flash' events are rate-limited for photosensitive safety (merged strokes, <= 3/s).
 * - Master: stems -> safety limiter (-1 dBFS threshold) and a soft clip whose output never
 *   exceeds -1 dBFS -> analyser; momentary loudness from BS.1770 K-weighting (two biquads)
 *   for the meter.
 * - Determinism: randomness is counter-based on the scene seed; offline renders drive the
 *   same graph with automation curves computed by the same control model.
 */

import {
  clamp,
  seedKey,
  hashString,
  CounterRng,
  fractalNoise1D,
  speedOfSound,
  absorptionCutoffHz,
  resolvePrecipitation,
  impactEnergyRate,
  placePreset,
  PLACE_PRESETS,
  windAtHeight,
  gustFactorFromRoughness,
  kaimalPeakPeriod,
  GustProcess,
  stormArcEnvelope,
  showerEnvelope,
  wetnessEquilibrium,
  stepWetness,
  MATERIALS,
  buildMaterial,
  patchMaterials,
  resolveMaterialName,
  bearingVector,
  bearingTo,
  alongTrack,
  insideFootprint,
  localToWorld,
  lapseRateKPerKm,
  DEG,
} from './physics.js';
import { shelterName, shelterInfo, reverbSpecFor, transmissionFor, ReverbUnit } from './reverb.js';
import {
  FLASH_CELL_SEC,
  FLASH_CEILING_PER_MIN,
  flashCandidates,
  buildChannel,
  nearestDistance,
  synthesizeThunder,
  createThunderBuffers,
} from './thunder.js';
import { compassName } from './units.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PROC = { rain: 'rg-rain-surface', wind: 'rg-wind', bed: 'rg-bed', bus: 'rg-weather' };
const MODULES = ['./worklets/rain-surface-processor.js', './worklets/wind-processor.js', './worklets/bed-processor.js'];
const CONTROL_HZ = 20;
const OFFLINE_CHUNK_SEC = 4;
const STEMS = ['rain', 'objects', 'wind', 'thunder', 'bed'];
const SECTORS = 8;
const RINGS = [
  { r0: 0.3, r1: 2.0, maxEv: 2500, modes: 12, bubbles: 48 },
  { r0: 2.0, r1: 5.5, maxEv: 1200, modes: 8, bubbles: 32 },
  { r0: 5.5, r1: 15.0, maxEv: 600, modes: 6, bubbles: 24 },
];
const GUST_LOOKAHEAD = 8;
const GUST_HISTORY = 40;
const MAX_AUDIBLE_M = 35000;
const MAX_THUNDER_VOICES = 8;
const TARGET_RAIN_DB = -25; // near rain field at 5 mm/h, open garden, before stems
const TARGET_WIND_DB = -26; // wind 'air' layer at 8 m/s (scale 1)
const CAL_AREA = 4;
const RN = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5'];
const DN = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'];
// Rain surfaces carry only their own drive; the weather they share (class impact rates and
// diameters, wetness, puddles, liquid rate) is one set of parameters on the rg-weather bus node.
const PARAM_NAMES = {
  rain: ['intensity', 'wind', 'gain'],
  wind: ['wind', 'gain', 'aux'],
  bed: ['level', 'cutoff', 'drizzle', 'density', 'intensity', 'gain', 'depth'],
  bus: [...RN, ...DN, 'wetness', 'puddle', 'liquid'],
};
const RANGE = {
  rain: { intensity: [0, 100], wind: [0, 200], gain: [0, 1000] },
  bus: { wetness: [0, 1], puddle: [0, 2000], liquid: [0, 2000] },
  wind: { wind: [0, 200], gain: [0, 1000], aux: [-1000, 1000] },
  bed: { level: [0, 100], cutoff: [100, 24000], drizzle: [0, 100], density: [0, 100], intensity: [0, 100], gain: [0, 1000], depth: [0, 12] },
};
// Level of wind-driven objects relative to the 'air' layer, at 1 m from their effective
// radius (extended sources: a whole crown, a whole span).
const TYPE_GAIN = { air: 1, tree: 12, grass: 3, wire: 5, flag: 3, chime: 2, wiper: 1.5 };
const MODE_GAINS = {
  none: {},
  sleep: { thunder: -14, wind: -6, objects: -1, bed: 1, master: -3 },
  focus: { thunder: -12, wind: -5, bed: 2 },
  calm: { thunder: -8, wind: -3 },
  asmr: { objects: 3, rain: 1, bed: -4, wind: -8, thunder: -10 },
};
const SMOOTH_MODE = {
  'weather.precipitation.rate_mm_h': 'log',
  'weather.thunder.distance_km': 'log',
  'weather.wind.from_deg': 'circ',
  'weather.thunder.bearing_deg': 'circ',
  'weather.hour': 'circ24',
};
const SMOOTH_GROUP = {
  'weather.precipitation.rate_mm_h': 'rain',
  'weather.precipitation.water_quality': 'rain',
  'weather.wind.speed_mps': 'wind',
  'weather.wind.from_deg': 'wind',
  'weather.wind.gust_factor': 'wind',
  'weather.wind.gust_rate_per_min': 'wind',
  'weather.thunder.distance_km': 'distance',
  'weather.thunder.bearing_deg': 'distance',
};
const SMOOTH_PATHS = [
  'weather.precipitation.rate_mm_h',
  'weather.precipitation.shower_period_min',
  'weather.precipitation.water_quality',
  'weather.precipitation.hail_size_mm',
  'weather.wind.speed_mps',
  'weather.wind.from_deg',
  'weather.wind.gust_factor',
  'weather.wind.gust_rate_per_min',
  'weather.thunder.distance_km',
  'weather.thunder.bearing_deg',
  'weather.thunder.approach_kmh',
  'weather.thunder.flashes_per_min',
  'weather.thunder.stroke_intensity',
  'weather.thunder.cg_fraction',
  'weather.temperature_c',
  'weather.humidity_pct',
  'weather.hour',
  'weather.arc_minutes',
  'place.window_open',
  'listener.height_m',
];
const SMOOTHABLE = new Set(SMOOTH_PATHS);
const FORESTS = {
  deciduous_forest: { species: 'deciduous', ic: 0.25 },
  conifer_forest: { species: 'conifer', ic: 0.35 },
  jungle: { species: 'deciduous', ic: 0.4 },
};
const TYPE_WORDS = {
  none: 'dry',
  drizzle: 'drizzle',
  rain: 'rain',
  showers: 'showers',
  freezing_rain: 'freezing rain',
  freezing_drizzle: 'freezing drizzle',
  sleet: 'sleet',
  ice_pellets: 'ice pellets',
  snow: 'snow',
  snow_showers: 'snow showers',
  graupel: 'graupel',
  hail: 'hail',
};

const REF_PRECIP = resolvePrecipitation('rain', 5, { drops: 'normal' });
const E_REF = Math.max(1e-12, impactEnergyRate(REF_PRECIP.classes.slice(0, 5)));

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const num = (x, d) => {
  if (x === null || x === undefined || x === '') return d;
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
};
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const obj = (x) => (isObj(x) ? x : {});
const str = (x, d) => (typeof x === 'string' && x.length ? x : d);
const dbToGain = (db) => Math.pow(10, db / 20);
const qDb = (q) => 20 * Math.log10(q); // Web Audio low/high-pass Q is expressed in dB
const wrap360 = (d) => ((num(d, 0) % 360) + 360) % 360;
const angDiff = (a, b) => ((((a - b) % 360) + 540) % 360) - 180;
const web = (p) => [p[0], p[2], -p[1]];
const nullableNum = (x) => (x === null || x === undefined || x === '' || !Number.isFinite(Number(x)) ? null : Number(x));
const nowMs = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());

function vec3(v, d = [0, 0, 0]) {
  const a = Array.isArray(v) ? v : [];
  return [num(a[0], d[0]), num(a[1], d[1]), num(a[2], d[2])];
}

function sameVec(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => x === b[i]);
}

function roundDeep(v) {
  if (typeof v === 'number') return Math.round(v * 100) / 100;
  if (Array.isArray(v)) return v.map(roundDeep);
  return v;
}

function cloneScene(s) {
  try {
    if (typeof structuredClone === 'function') return structuredClone(s);
  } catch (e) {
    /* fall through */
  }
  return JSON.parse(JSON.stringify(s, (k, v) => (typeof v === 'bigint' ? v.toString() : v)));
}

function parseTime(x) {
  if (typeof x === 'number') return x;
  if (typeof x === 'string') {
    const parts = x.trim().split(':').map(Number);
    if (parts.length && parts.every(Number.isFinite)) return parts.slice(0, 3).reduce((a, b) => a * 60 + b, 0);
  }
  return NaN;
}

// ---------------------------------------------------------------------------
// Scene: defaults, normalization, paths, timeline
// ---------------------------------------------------------------------------

/** A complete 'rainscene' v1 scene with every field at its default. */
export function defaultScene() {
  return {
    rainscene: 1,
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
        nw: null,
        d_min_mm: 0.3,
        d_max_mm: 6,
        class_boundaries: 'medwin',
        water_quality: 1,
      },
      wind: { speed_mps: 4.4, from_deg: 270, gust_factor: null, gust_rate_per_min: null },
      thunder: { enabled: false, distance_km: 8, bearing_deg: 270, approach_kmh: 0, flashes_per_min: 2, stroke_intensity: 0.5, cg_fraction: 0.4 },
      temperature_c: 12,
      humidity_pct: 90,
      hour: 22,
      season: 'autumn',
      storm_arc: 'steady',
      arc_minutes: 20,
      cloud_okta: 8,
    },
    place: {
      preset: 'garden',
      shelter: 'open',
      surfaces: { grass: 40, soil: 20, leaves: 20, puddle: 10, concrete: 10 },
      window_open: 0,
      wipers: 'off',
      objects: [],
      z0_m: null,
      aftermath_tau_min: 8,
      vegetation: null,
      emitters: null,
    },
    listener: { position: [0, 0, 0], height_m: 1.7, heading_deg: 0, render: 'binaural' },
    mode: 'none',
    timeline: [],
    studio: { loop: [0, 300], loopOn: false, smoothing: null },
  };
}

function normalizeObject(o, i, used) {
  if (!isObj(o)) return null;
  let id = o.id !== undefined && o.id !== null && String(o.id).length ? String(o.id) : `obj${i}`;
  while (used.has(id)) id += '_';
  used.add(id);
  const size = vec3(o.size, [1, 1, 0]).map((v) => Math.abs(v));
  return {
    id,
    type: str(o.type, 'surface').toLowerCase(),
    material: typeof o.material === 'string' ? o.material : '',
    position: vec3(o.position),
    size,
    rotation_deg: num(o.rotation_deg, 0),
    params: isObj(o.params) ? { ...o.params } : {},
  };
}

function normalizeSurfaces(surfaces, preset) {
  const out = {};
  let any = false;
  for (const [k, v] of Object.entries(obj(surfaces))) {
    const x = num(v, 0);
    if (x > 0) {
      out[k] = x;
      any = true;
    } else if (x === 0) out[k] = 0;
  }
  if (!any) return { ...placePreset(preset).surfaces };
  return out;
}

/** Fill every field of a (possibly partial) rainscene with validated values. */
export function normalizeScene(input) {
  const D = defaultScene();
  const s = obj(input);
  const w = obj(s.weather);
  const p = obj(w.precipitation);
  const wi = obj(w.wind);
  const th = obj(w.thunder);
  const pl = obj(s.place);
  const li = obj(s.listener);
  const st = obj(s.studio);
  const preset = str(pl.preset, D.place.preset);
  const used = new Set();
  const objects = (Array.isArray(pl.objects) ? pl.objects : []).map((o, i) => normalizeObject(o, i, used)).filter(Boolean);
  const timeline = (Array.isArray(s.timeline) ? s.timeline : [])
    .filter((k) => isObj(k) && typeof k.path === 'string' && k.path.length)
    .map((k) => ({
      id: k.id,
      at: Math.max(0, num(parseTime(k.at), 0)),
      path: k.path,
      value: k.value,
      curve: ['linear', 'log', 'hold', 'smooth'].includes(k.curve) ? k.curve : 'linear',
    }))
    .sort((a, b) => a.at - b.at);
  const thEnabled = th.enabled === true || th.enabled === 'true' || th.enabled === 1;
  const loop = Array.isArray(st.loop) && st.loop.length >= 2 ? [Math.max(0, num(st.loop[0], 0)), Math.max(0, num(st.loop[1], 300))] : [0, 300];
  return {
    rainscene: 1,
    name: str(s.name, D.name),
    seed: s.seed === undefined || s.seed === null || s.seed === '' ? D.seed : s.seed,
    weather: {
      precipitation: {
        type: str(p.type, 'rain').toLowerCase(),
        rate_mm_h: clamp(num(p.rate_mm_h, 4), 0, 500),
        drops: ['fine', 'normal', 'fat'].includes(p.drops) ? p.drops : 'normal',
        character: ['continuous', 'intermittent', 'showery'].includes(p.character) ? p.character : 'continuous',
        shower_period_min: clamp(num(p.shower_period_min, 12), 1, 240),
        hail_size_mm: clamp(num(p.hail_size_mm, 13), 5, 100),
        dsd_model: p.dsd_model === 'marshall_palmer' ? 'marshall_palmer' : 'gamma',
        mu: nullableNum(p.mu),
        dm_mm: nullableNum(p.dm_mm),
        nw: nullableNum(p.nw),
        d_min_mm: clamp(num(p.d_min_mm, 0.3), 0.05, 2),
        d_max_mm: clamp(num(p.d_max_mm, 6), 1, 9),
        class_boundaries: p.class_boundaries === 'nystuen' ? 'nystuen' : 'medwin',
        water_quality: clamp(num(p.water_quality, 1), 0, 1),
      },
      wind: {
        speed_mps: clamp(num(wi.speed_mps, 4.4), 0, 90),
        from_deg: wrap360(num(wi.from_deg, 270)),
        gust_factor: nullableNum(wi.gust_factor),
        gust_rate_per_min: nullableNum(wi.gust_rate_per_min),
      },
      thunder: {
        enabled: thEnabled,
        distance_km: clamp(num(th.distance_km, 8), 0.05, 100),
        bearing_deg: wrap360(num(th.bearing_deg, 270)),
        approach_kmh: clamp(num(th.approach_kmh, 0), -300, 300),
        flashes_per_min: clamp(num(th.flashes_per_min, 2), 0, FLASH_CEILING_PER_MIN),
        stroke_intensity: clamp(num(th.stroke_intensity, 0.5), 0, 1),
        cg_fraction: clamp(num(th.cg_fraction, 0.4), 0, 1),
      },
      temperature_c: clamp(num(w.temperature_c, 12), -60, 60),
      humidity_pct: clamp(num(w.humidity_pct, 90), 1, 100),
      hour: ((num(w.hour, 22) % 24) + 24) % 24,
      season: ['spring', 'summer', 'autumn', 'winter'].includes(w.season) ? w.season : 'autumn',
      storm_arc: ['arriving', 'steady', 'passing', 'clearing'].includes(w.storm_arc) ? w.storm_arc : 'steady',
      arc_minutes: clamp(num(w.arc_minutes, 20), 1, 600),
      cloud_okta: clamp(num(w.cloud_okta, 8), 0, 8),
    },
    place: {
      preset,
      shelter: shelterName(pl.shelter),
      surfaces: normalizeSurfaces(pl.surfaces, preset),
      window_open: clamp(num(pl.window_open, 0), 0, 1),
      wipers: ['off', 'slow', 'fast', 'intermittent'].includes(pl.wipers) ? pl.wipers : 'off',
      objects,
      z0_m: nullableNum(pl.z0_m),
      aftermath_tau_min: clamp(num(pl.aftermath_tau_min, 8), 0.5, 600),
      vegetation: isObj(pl.vegetation) ? { ...pl.vegetation } : null,
      emitters: Array.isArray(pl.emitters) ? pl.emitters.filter(isObj).map((e) => ({ ...e })) : null,
    },
    listener: {
      position: vec3(li.position),
      height_m: clamp(num(li.height_m, 1.7), 0.1, 200),
      heading_deg: wrap360(num(li.heading_deg, 0)),
      render: li.render === 'stereo' || (typeof li.render === 'string' && li.render !== 'binaural' && li.render.length) ? 'stereo' : 'binaural',
    },
    mode: ['none', 'sleep', 'focus', 'calm', 'asmr'].includes(s.mode) ? s.mode : 'none',
    timeline,
    studio: { loop, loopOn: !!st.loopOn, smoothing: isObj(st.smoothing) ? { ...st.smoothing } : null },
  };
}

function pathParts(path) {
  return String(path || '')
    .split('.')
    .filter((x) => x.length);
}

function findObject(list, k) {
  const byId = list.find((o) => o && String(o.id) === k);
  if (byId) return byId;
  return /^\d+$/.test(k) ? list[Number(k)] : undefined;
}

/** Read a dotted scene path; 'place.objects.<id>...' addresses objects by id (or index). */
export function getScenePath(scene, path) {
  const parts = pathParts(path);
  let cur = scene;
  for (let i = 0; i < parts.length; i++) {
    if (cur === null || cur === undefined) return undefined;
    const k = parts[i];
    if (Array.isArray(cur)) cur = parts[i - 1] === 'objects' ? findObject(cur, k) : cur[Number(k)];
    else if (typeof cur === 'object') cur = cur[k];
    else return undefined;
  }
  return cur;
}

/** Write a dotted scene path in place; returns false if the path cannot be created. */
export function setScenePath(scene, path, value) {
  const parts = pathParts(path);
  if (!parts.length || !scene || typeof scene !== 'object') return false;
  let cur = scene;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    let next;
    if (Array.isArray(cur)) next = parts[i - 1] === 'objects' ? findObject(cur, k) : cur[Number(k)];
    else next = cur[k];
    if (next === null || next === undefined || typeof next !== 'object') {
      if (Array.isArray(cur)) return false;
      next = {};
      cur[k] = next;
    }
    cur = next;
  }
  const last = parts[parts.length - 1];
  if (Array.isArray(cur)) {
    if (parts[parts.length - 2] === 'objects') return false;
    cur[Number(last)] = value;
  } else cur[last] = value;
  return true;
}

const TRACK_CACHE = new WeakMap();

function timelineTracks(tl) {
  if (!Array.isArray(tl) || !tl.length) return null;
  let m = TRACK_CACHE.get(tl);
  if (!m) {
    m = new Map();
    for (const k of tl) {
      if (!k || typeof k.path !== 'string') continue;
      if (!m.has(k.path)) m.set(k.path, []);
      m.get(k.path).push(k);
    }
    for (const a of m.values()) a.sort((x, y) => x.at - y.at);
    TRACK_CACHE.set(tl, m);
  }
  return m;
}

function interp(a, b, u, curve) {
  if (typeof a === 'number' && typeof b === 'number') {
    if (curve === 'hold') return a;
    let k = u;
    if (curve === 'smooth') k = u * u * (3 - 2 * u);
    if (curve === 'log' && a > 0 && b > 0) return a * Math.pow(b / a, k);
    return a + (b - a) * k;
  }
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length && curve !== 'hold') {
    return a.map((x, i) => (typeof x === 'number' && typeof b[i] === 'number' ? interp(x, b[i], u, curve) : x));
  }
  return a;
}

function evalTrack(keys, t) {
  if (t <= keys[0].at) return keys[0].value;
  const last = keys[keys.length - 1];
  if (t >= last.at) return last.value;
  let lo = 0;
  let hi = keys.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keys[mid].at <= t) lo = mid;
    else hi = mid;
  }
  const a = keys[lo];
  const b = keys[hi];
  const span = b.at - a.at;
  return interp(a.value, b.value, span > 0 ? (t - a.at) / span : 1, a.curve || 'linear');
}

/** Values of every automated path at time t (seconds on the timeline). */
function evalTimeline(tl, t) {
  const out = new Map();
  const m = timelineTracks(tl);
  if (!m) return out;
  for (const [p, keys] of m) out.set(p, evalTrack(keys, t));
  return out;
}

/** The base scene with timeline values applied (copy-on-write of the touched branches). */
function overlay(base, tlMap) {
  if (!tlMap || !tlMap.size) return base;
  const s = {
    ...base,
    weather: {
      ...base.weather,
      precipitation: { ...base.weather.precipitation },
      wind: { ...base.weather.wind },
      thunder: { ...base.weather.thunder },
    },
    place: { ...base.place, surfaces: { ...base.place.surfaces } },
    listener: { ...base.listener },
  };
  let copied = false;
  for (const [path, value] of tlMap) {
    if (path.startsWith('place.objects.') && !copied) {
      s.place.objects = base.place.objects.map((o) => ({ ...o, position: o.position.slice(), size: o.size.slice(), params: { ...o.params } }));
      copied = true;
    }
    setScenePath(s, path, value);
  }
  if (s.place.shelter !== base.place.shelter) s.place.shelter = shelterName(s.place.shelter);
  return s;
}

function rampFor(path, scene) {
  const g = SMOOTH_GROUP[path] || 'other';
  const d = { rain: 1.5, wind: 1, distance: 2, other: 0.25 };
  const sm = scene && scene.studio && scene.studio.smoothing;
  return num(sm && sm[g], d[g]);
}

// ---------------------------------------------------------------------------
// Control model: smoothed scene values plus evolving state (gusts, wetness)
// ---------------------------------------------------------------------------

class ControlModel {
  constructor(seed) {
    this.live = new Map();
    this.wet = { film: 0, puddle: 0 };
    this.tau = 0;
    this.primed = false;
    this._pKey = '';
    this._pVal = null;
    this.setSeed(seed);
  }

  setSeed(seed) {
    this.seed = seed;
    this.key = seedKey(seed);
    this.gust = new GustProcess(seed, 'gust');
    this.gHist = new Float64Array(Math.ceil((GUST_HISTORY + GUST_LOOKAHEAD) * CONTROL_HZ) + 8).fill(1);
    this.gCount = 0;
    this.gAcc = 0;
    this.gustPrimed = false;
  }

  /** Smooth a numeric path from `from` to `to` over ~rampSec (99 %). */
  setTarget(path, from, to, rampSec) {
    const mode = SMOOTH_MODE[path] || 'lin';
    const ramp = num(rampSec, 0.5);
    let e = this.live.get(path);
    if (!e) {
      e = { v: num(from, to), target: to, tau: Math.max(0.005, ramp / 5), mode };
      this.live.set(path, e);
    } else {
      e.target = to;
      e.tau = Math.max(0.005, ramp / 5);
    }
    if (!(ramp > 0)) e.v = to;
  }

  stepLive(dt) {
    if (!(dt > 0)) return;
    for (const [p, e] of this.live) {
      const k = 1 - Math.exp(-dt / e.tau);
      if (e.mode === 'log' && e.v > 0 && e.target > 0) e.v = Math.exp(Math.log(e.v) + (Math.log(e.target) - Math.log(e.v)) * k);
      else if (e.mode === 'circ') e.v = wrap360(e.v + angDiff(e.target, e.v) * k);
      else if (e.mode === 'circ24') {
        const d = ((((e.target - e.v) % 24) + 36) % 24) - 12;
        e.v = (((e.v + d * k) % 24) + 24) % 24;
      } else e.v += (e.target - e.v) * k;
      const err = e.mode === 'circ' ? Math.abs(angDiff(e.target, e.v)) : Math.abs(e.target - e.v);
      if (err <= 1e-6 * (Math.abs(e.target) + 1e-3)) this.live.delete(p);
    }
  }

  _stepGust(dt, U, G, ratePerMin, maxSteps = 400) {
    const h = 1 / CONTROL_HZ;
    const natural = 60 / kaimalPeakPeriod(U, 10);
    const speed = ratePerMin > 0 ? clamp(ratePerMin / natural, 0.25, 4) : 1;
    this.gAcc += dt;
    let n = 0;
    while (this.gAcc >= h && n++ < maxSteps) {
      this.gAcc -= h;
      this.gHist[this.gCount % this.gHist.length] = this.gust.step(h * speed, U, G);
      this.gCount++;
    }
    if (this.gAcc > h) this.gAcc = h;
  }

  /** Gust multiplier at relative time `delta` (s; positive = ahead of the listener). */
  gustAt(delta = 0) {
    const d = clamp(delta, -GUST_HISTORY + 0.5, GUST_LOOKAHEAD);
    const pos = this.gCount - 1 - (GUST_LOOKAHEAD - d) * CONTROL_HZ;
    const L = this.gHist.length;
    const i = Math.floor(pos);
    const f = pos - i;
    const a = this.gHist[((i % L) + L) % L];
    const b = this.gHist[(((i + 1) % L) + L) % L];
    return a + (b - a) * f;
  }

  /** Reference (10 m) wind at a world position, with the gust advected along the wind. */
  windRefAt(pos, ctl, L) {
    if (!pos || !L) return ctl.windRef;
    const b = bearingVector(ctl.fromDeg);
    const upwind = (pos[0] - L.pos[0]) * b[0] + (pos[1] - L.pos[1]) * b[1];
    return ctl.U10 * this.gustAt(upwind / Math.max(1, 0.8 * ctl.U10));
  }

  _precip(type, rate, o) {
    const k = `${type}|${rate.toPrecision(4)}|${o.drops}|${o.dmScale.toFixed(3)}|${o.mu}|${o.dm}|${o.nw}|${o.model}|${o.boundaries}|${o.hailSizeMm}|${o.dMin}|${o.dMax}`;
    if (k !== this._pKey || !this._pVal) {
      const opts = { drops: o.drops, dmScale: o.dmScale, model: o.model, boundaries: o.boundaries, hailSizeMm: o.hailSizeMm, dMin: o.dMin, dMax: o.dMax };
      if (o.mu !== null) opts.mu = o.mu;
      if (o.dm !== null) opts.dm = o.dm;
      if (o.nw !== null) opts.nw = o.nw;
      this._pKey = k;
      this._pVal = resolvePrecipitation(type, rate, opts);
    }
    return this._pVal;
  }

  /** Advance the evolving state (gusts, wetness) by dt and return the control snapshot for scene time t. */
  compute(S, t, dt) {
    this.stepLive(dt);
    this.tau += Math.max(0, dt);
    const tl = evalTimeline(S.timeline, t);
    const live = this.live;
    const V = (p) => (tl.has(p) ? tl.get(p) : live.has(p) ? live.get(p).v : getScenePath(S, p));
    const PP = 'weather.precipitation.';
    const type = String(V(PP + 'type') || 'rain').toLowerCase();
    const arc = stormArcEnvelope(String(V('weather.storm_arc') || 'steady'), t, num(V('weather.arc_minutes'), 20));
    let character = String(V(PP + 'character') || 'continuous');
    if ((type === 'showers' || type === 'snow_showers') && character === 'continuous') character = 'showery';
    const shower = showerEnvelope((this.key ^ 0x5a17c0de) >>> 0, t, character, num(V(PP + 'shower_period_min'), 12));
    const rateBase = Math.max(0, num(V(PP + 'rate_mm_h'), 4));
    const rate = type === 'none' ? 0 : rateBase * (tl.has(PP + 'rate_mm_h') ? 1 : arc.rate) * shower;
    const drops = String(V(PP + 'drops') || 'normal');
    const precip = this._precip(type, rate, {
      drops,
      dmScale: tl.has(PP + 'drops') ? 1 : arc.dm,
      mu: nullableNum(V(PP + 'mu')),
      dm: nullableNum(V(PP + 'dm_mm')),
      nw: nullableNum(V(PP + 'nw')),
      model: V(PP + 'dsd_model') === 'marshall_palmer' ? 'marshall_palmer' : 'gamma',
      boundaries: V(PP + 'class_boundaries') === 'nystuen' ? 'nystuen' : 'medwin',
      hailSizeMm: num(V(PP + 'hail_size_mm'), 13),
      dMin: num(V(PP + 'd_min_mm'), 0.3),
      dMax: num(V(PP + 'd_max_mm'), 6),
    });
    const kinds = precip.kinds;
    const flux = new Float64Array(6);
    const dcls = new Float64Array(6);
    let totalFlux = 0;
    for (let c = 0; c < 6; c++) {
      const k = precip.classes[c] || { flux: 0, d: 1 };
      flux[c] = Math.max(0, num(k.flux, 0));
      dcls[c] = Math.max(0, num(k.d, 1));
      if (c < 5) totalFlux += flux[c];
    }

    const presetName = String(V('place.preset') || 'garden');
    const preset = placePreset(presetName);
    const presetKey = Object.keys(PLACE_PRESETS).find((k) => PLACE_PRESETS[k] === preset) || 'garden';
    const z0 = S.place.z0_m !== null && S.place.z0_m !== undefined ? clamp(num(S.place.z0_m, preset.z0), 0.0001, 3) : preset.z0;
    const U10 = Math.max(0, num(V('weather.wind.speed_mps'), 4.4)) * (tl.has('weather.wind.speed_mps') ? 1 : arc.wind);
    const gfRaw = nullableNum(V('weather.wind.gust_factor'));
    let G = gfRaw === null ? gustFactorFromRoughness(z0) : clamp(gfRaw, 1, 3);
    const hour = num(V('weather.hour'), 12);
    const lapse = lapseRateKPerKm(hour, U10, num(S.weather.cloud_okta, 8));
    // Turbulence variance ~4x higher in unstable (sunny) than stable (night) air (Bolin).
    G = 1 + (G - 1) * (lapse > 9 ? 1.35 : lapse < 0 ? 0.6 : 1);
    const gr = num(V('weather.wind.gust_rate_per_min'), 0);
    if (!this.gustPrimed) {
      this.gustPrimed = true;
      let left = GUST_LOOKAHEAD + GUST_HISTORY;
      while (left > 0) {
        const step = Math.min(left, 15);
        this._stepGust(step, U10, G, gr, 100000);
        left -= step;
      }
    }
    this._stepGust(Math.max(0, dt), U10, G, gr);
    const gust = this.gustAt(0);
    const fromDeg = wrap360(num(V('weather.wind.from_deg'), 270) + 8 * fractalNoise1D((this.key ^ 0x3c6ef372) >>> 0, this.tau / 45, 3));

    const tempC = num(V('weather.temperature_c'), 12);
    const rh = clamp(num(V('weather.humidity_pct'), 90), 1, 100);
    const liquid = kinds[0] === 2 ? (tempC > 1 ? 0.4 * rate : 0) : kinds[0] === 1 || kinds[0] === 3 ? 0.6 * rate : rate;
    if (!this.primed) {
      const eq = wetnessEquilibrium(liquid);
      this.wet.film = eq.film;
      this.wet.puddle = eq.puddle;
      this.primed = true;
    } else if (dt > 0) {
      stepWetness(this.wet, liquid, dt, num(S.place.aftermath_tau_min, 8));
    }

    return {
      t,
      dt,
      tl,
      V,
      type,
      kinds,
      kindsKey: kinds.join(''),
      character,
      drops,
      rate,
      liquid,
      flux,
      dcls,
      totalFlux,
      energyRate: precip.energyRate,
      hailFlux: flux[5],
      film: this.wet.film,
      puddle: this.wet.puddle,
      U10,
      G,
      gust,
      windRef: U10 * gust,
      fromDeg,
      z0,
      preset,
      presetKey,
      lapse,
      tempC,
      rh,
      hour,
      season: String(V('weather.season') || 'autumn'),
      thunderMul: tl.has('weather.thunder.flashes_per_min') || tl.has('weather.thunder.distance_km') ? 1 : arc.thunder,
      arc,
      waterQuality: clamp(num(V(PP + 'water_quality'), 1), 0, 1),
      boundaries: V(PP + 'class_boundaries') === 'nystuen' ? 'nystuen' : 'medwin',
      dMin: num(V(PP + 'd_min_mm'), 0.3),
      dMax: num(V(PP + 'd_max_mm'), 6),
      windowOpen: clamp(num(V('place.window_open'), 0), 0, 1),
      shelter: shelterName(V('place.shelter')),
      mode: String(V('mode') || 'none'),
    };
  }
}

function listenerAt(ctl, base) {
  const tl = ctl.tl;
  const tp = tl.get('listener.position');
  const pos = Array.isArray(tp) ? vec3(tp, base.position) : base.position.slice();
  const heading = tl.has('listener.heading_deg') ? num(tl.get('listener.heading_deg'), base.headingDeg) : base.headingDeg;
  const h = clamp(num(ctl.V('listener.height_m'), 1.7), 0.1, 200);
  return { pos, ear: [pos[0], pos[1], pos[2] + h], heading: wrap360(heading), pitch: num(base.pitchDeg, 0), height: h };
}

function anchorFor(prev, L, shelter) {
  if (!prev || prev.shelter !== shelter || Math.hypot(L.pos[0] - prev.pos[0], L.pos[1] - prev.pos[1]) > 1.5) {
    return { shelter, pos: L.pos.slice(), heading: L.heading, key: `${shelter}|${L.pos.map((v) => v.toFixed(2)).join(',')}|${L.heading.toFixed(1)}` };
  }
  return prev;
}

function planSignature(ver, ctl, L, anchor, quality) {
  let tlPart = '';
  for (const [p, v] of ctl.tl) {
    if (p.startsWith('listener.')) continue;
    if (typeof v === 'number' && !p.startsWith('place.')) continue;
    tlPart += `${p}=${JSON.stringify(roundDeep(v))};`;
  }
  return [
    ver,
    ctl.kindsKey,
    ctl.shelter,
    ctl.season,
    ctl.presetKey,
    Math.round(ctl.waterQuality * 20),
    ctl.boundaries,
    ctl.dMin,
    ctl.dMax,
    ctl.kinds[0] === 2 && ctl.tempC < 1.5 ? 'snowcover' : '',
    Math.round(L.pos[0] * 2),
    Math.round(L.pos[1] * 2),
    Math.round(L.ear[2] * 10),
    anchor ? anchor.key : '',
    quality,
    tlPart,
  ].join('|');
}

// ---------------------------------------------------------------------------
// Geometry of objects
// ---------------------------------------------------------------------------

const COVER_TYPES = new Set(['roof', 'tent', 'car', 'umbrella', 'tree']);

function treeHeight(o) {
  return Math.max(1, num(o.params.height, o.size[2] || 9));
}

function footprintHas(o, x, y, margin = 0) {
  if (o.type === 'umbrella' || o.type === 'tree') {
    const r = Math.max(o.size[0], o.size[1], 0.2) / 2 + margin;
    return Math.hypot(x - o.position[0], y - o.position[1]) <= r;
  }
  return insideFootprint([x, y], o, margin);
}

/** Height of the rain-catching cover of an object. */
function coverZ(o) {
  const z = o.position[2];
  if (o.type === 'roof') return z;
  if (o.type === 'tree') return z + 0.35 * treeHeight(o);
  return z + o.size[2];
}

function interception(o, season) {
  switch (o.type) {
    case 'roof':
    case 'tent':
    case 'car':
      return 1;
    case 'umbrella':
      return 0.97;
    case 'tree':
      return o.params.species === 'conifer' ? 0.4 : season === 'winter' ? 0.1 : 0.3;
    default:
      return 0;
  }
}

/** Fraction of open-field rain reaching (x, y) at height zTop (canopy 10-40 %, roofs 100 %). */
function exposureAt(objects, x, y, zTop, excludeId, season) {
  let f = 1;
  for (const c of objects) {
    if (c.id === excludeId) continue;
    const ic = interception(c, season);
    if (ic <= 0) continue;
    if (coverZ(c) <= zTop + 0.05) continue;
    if (!footprintHas(c, x, y)) continue;
    f *= 1 - ic;
  }
  return f;
}

function coveredBy(objects, x, y, zTop, season) {
  return objects.some((c) => interception(c, season) > 0 && coverZ(c) > zTop + 0.05 && footprintHas(c, x, y));
}

function listenerInside(o, ear) {
  if (!COVER_TYPES.has(o.type)) return false;
  if (!footprintHas(o, ear[0], ear[1], 0.05)) return false;
  const z = o.position[2];
  if (o.type === 'roof') return ear[2] < z + Math.max(0.05, o.size[2]) + 0.3 && ear[2] > z - 1.5;
  if (o.type === 'tree') return ear[2] < z + treeHeight(o);
  return ear[2] < z + o.size[2] + 0.3 && ear[2] > z - 1;
}

/** Forward bearing of a car: along its longer horizontal axis. */
function carForward(o) {
  return o.size[0] >= o.size[1] ? wrap360(90 + o.rotation_deg) : wrap360(o.rotation_deg);
}

function surfaceMix(surfaces) {
  const m = new Map();
  let tot = 0;
  for (const [k, v] of Object.entries(surfaces || {})) {
    const x = Math.max(0, num(v, 0));
    if (!(x > 0)) continue;
    const n = resolveMaterialName(k);
    m.set(n, (m.get(n) || 0) + x);
    tot += x;
  }
  if (!(tot > 0)) return [['grass', 1]];
  return [...m.entries()].map(([k, v]) => [k, v / tot]).sort((a, b) => b[1] - a[1]);
}

function waterFraction(mix) {
  let f = 0;
  for (const [n, x] of mix) if (MATERIALS[n] && MATERIALS[n].kind === 'water') f += x;
  return f;
}

// ---------------------------------------------------------------------------
// Voice specs and the scene plan
// ---------------------------------------------------------------------------

function keysFor(spec) {
  const k = {};
  for (const n of PARAM_NAMES[spec.proc]) k[n] = `${spec.key}|${n}`;
  if (spec.pan === 'panner') {
    k.px = `${spec.key}|px`;
    k.py = `${spec.key}|py`;
    k.pz = `${spec.key}|pz`;
  } else if (spec.pan === 'quad') {
    k.qx = [0, 1, 2, 3].map((i) => `${spec.key}|qx${i}`);
    k.qy = [0, 1, 2, 3].map((i) => `${spec.key}|qy${i}`);
    k.qz = [0, 1, 2, 3].map((i) => `${spec.key}|qz${i}`);
  } else if (spec.pan === 'stereo') {
    k.pan = `${spec.key}|pan`;
  }
  return k;
}

function keyList(keys) {
  const out = [];
  for (const v of Object.values(keys || {})) {
    if (Array.isArray(v)) out.push(...v);
    else out.push(v);
  }
  return out;
}

function matSpec(pc, name, salt, fraction, o = {}) {
  const key = `${name}|${salt}|${fraction.toFixed(4)}|${num(o.area, 0).toFixed(2)}|${o.maxModes ?? 16}|${o.depthMm ?? ''}`;
  const cache = pc.env.matCache;
  let m = cache.get(key);
  if (!m) {
    const opts = { seed: pc.env.seed, salt, fraction, maxModes: o.maxModes };
    if (o.area !== undefined) opts.area = o.area;
    if (o.depthMm !== undefined && o.depthMm !== null) opts.depthMm = o.depthMm;
    m = buildMaterial(name, opts);
    if (cache.size > 4000) cache.clear();
    cache.set(key, m);
  }
  return m;
}

/** A rain-surface voice (rain on a patch or object, a drip line, or a gutter/downspout flow). */
function rainSpec(pc, key, objectId, o) {
  const area = Math.max(0.01, num(o.area, 1));
  // Live CPU budget: every voice runs a per-sample modal bank on the single audio thread,
  // so lower tiers trade modal detail and event density for real-time headroom.
  const q = pc.env.quality;
  const modes0 = o.modes ?? 12;
  const modes = q === 'low' ? Math.min(modes0, 4) : q === 'balanced' ? Math.max(3, Math.ceil(modes0 / 2)) : modes0;
  const evBudget = q === 'low' ? 0.4 : q === 'balanced' ? 0.6 : 1;
  const materials = (o.mats || [['concrete', 1]]).slice(0, 4).map(([n, f, extra], i) =>
    matSpec(pc, n, `${key}:${i}`, Math.max(0.001, num(f, 1)), { area: extra && extra.area !== undefined ? extra.area : area, maxModes: modes, depthMm: extra ? extra.depthMm : undefined }),
  );
  const ref = clamp(num(o.ref, Math.sqrt(area / Math.PI)), 0.2, 4);
  const config = {
    id: key,
    kind: o.kind || 'rain',
    area: +area.toFixed(4),
    exposure: +clamp(num(o.exposure, 1), 0, 4).toFixed(4),
    maxEventsPerSec: Math.max(50, Math.round(num(o.maxEv, 1500) * pc.env.evScale * evBudget)),
    maxModes: modes,
    maxBubbles: o.bubbles ?? 32,
    coxDepth: 0.35,
    waterQuality: Math.round(pc.ctl.waterQuality * 20) / 20,
    seed: pc.env.seedKey,
    sharedSeed: pc.env.sharedKey,
    precipKinds: Array.from(pc.ctl.kinds),
    boundaries: pc.ctl.boundaries,
    dMin: pc.ctl.dMin,
    dMax: pc.ctl.dMax,
    materials,
    drip: o.drip || null,
    flow: o.flow || null,
    prefill: true,
  };
  for (const f of ['wetMul', 'puddleMul', 'flowMul']) if (o[f] !== undefined && o[f] !== null) config[f] = num(o[f], 1);
  const { exposure, area: a2, ...rest } = config;
  const spec = {
    key,
    objectId,
    proc: 'rain',
    stem: o.stem || 'objects',
    route: o.route || 'ext',
    channels: 1,
    pan: 'panner',
    posMode: o.posMode || 'world',
    pos: o.pos || null,
    rel: o.rel || null,
    ref,
    rolloff: 1,
    hrtf: o.hrtf !== false,
    config,
    structKey: JSON.stringify(rest),
    ctl: {
      gain: num(o.gain, 1) / ref,
      driven: o.driven || null,
      windH: num(o.windH, 1),
      intensity: num(o.intensity, 1),
    },
  };
  spec.keys = keysFor(spec);
  return spec;
}

/** A wind-processor voice (air, vegetation, grass, wire, flag, chime, wiper). */
function windSpec(pc, key, objectId, o) {
  const config = { id: key, mode: o.mode, seed: pc.env.seedKey, windScale: +num(o.windScale, 1).toFixed(4), level: 1 };
  for (const f of ['species', 'leafOn', 'height', 'area', 'diameter', 'length', 'notes', 'speed', 'cpm']) {
    if (o[f] !== undefined && o[f] !== null) config[f] = o[f];
  }
  const pan = o.pan || 'panner';
  const ref = pan === 'panner' ? clamp(num(o.ref, 1), 0.2, 10) : 1;
  const spec = {
    key,
    objectId,
    proc: 'wind',
    stem: o.stem || 'wind',
    route: o.route || 'ext',
    channels: o.channels || 1,
    pan,
    posMode: o.posMode || 'world',
    pos: o.pos || null,
    rel: o.rel || null,
    ref,
    rolloff: 1,
    hrtf: o.hrtf !== false,
    config,
    structKey: JSON.stringify(config),
    ctl: { gain: num(o.gain, 1) / ref, aux: o.aux || null },
  };
  spec.keys = keysFor(spec);
  return spec;
}

/** Geometry of ground patch (ring, sector) for a listener at height h. */
function patchGeometry(ri, s, h) {
  const R = RINGS[ri];
  const rc = Math.sqrt((R.r0 * R.r0 + R.r1 * R.r1) / 2);
  const b = (s + 0.5 + (ri % 2) * 0.5) * (360 / SECTORS) * DEG;
  const area = (Math.PI * (R.r1 * R.r1 - R.r0 * R.r0)) / SECTORS;
  const d = Math.hypot(rc, h);
  // Impacts radiate like vertical dipoles from hard ground (cos theta), monopole-like from water.
  const dir = 0.35 + 0.65 * (h / d);
  const ref = clamp(Math.sqrt(area / Math.PI), 0.3, 3);
  return { rel: [rc * Math.sin(b), rc * Math.cos(b), 0], area, d, dir, ref };
}

/** Sum over the ground field of area * (gain / distance)^2: the field's power relative to 1 m^2 at 1 m. */
function fieldSum(h, nRings = RINGS.length) {
  let s = 0;
  for (let ri = 0; ri < nRings; ri++) {
    for (let k = 0; k < SECTORS; k++) {
      const g = patchGeometry(ri, k, h);
      const a = g.dir / Math.max(g.d, g.ref);
      s += g.area * a * a;
    }
  }
  return s;
}

const FIELD_REF = fieldSum(1.7);

function dripMats(pc) {
  const top = pc.mix.filter(([n]) => !(MATERIALS[n] && MATERIALS[n].kind === 'water')).slice(0, 2);
  let s = 0;
  for (const [, f] of top) s += f;
  const out = top.map(([n, f]) => [n, (0.7 * f) / (s || 1)]);
  out.push(['puddle', 0.3]);
  return out;
}

function roofCatchment(pc, o) {
  let best = null;
  let bestD = 8;
  for (const r of pc.objects) {
    if (r.type !== 'roof' || r.id === o.id) continue;
    const d = Math.hypot(r.position[0] - o.position[0], r.position[1] - o.position[1]) - Math.max(r.size[0], r.size[1]) / 2;
    if (d < bestD) {
      bestD = d;
      best = r;
    }
  }
  return best ? Math.max(2, (best.size[0] * best.size[1]) / 2) : null;
}

/** Shelter surfaces implied by place.shelter when the scene has no matching object around the listener. */
function shelterObjects(S, shelter, env, objects) {
  const A = env.anchor;
  const L = env.listener;
  const ear = L.ear;
  const h = A.heading;
  const fwd = bearingVector(h);
  const right = bearingVector(h + 90);
  const base = A.pos;
  const out = [];
  const has = (pred) => objects.some((o) => pred(o) && listenerInside(o, ear));
  const earAbove = Math.max(0.3, ear[2] - base[2]);
  switch (shelter) {
    case 'umbrella':
      if (!has((o) => o.type === 'umbrella'))
        out.push({ id: '~umbrella', type: 'umbrella', material: 'umbrella', position: [L.pos[0], L.pos[1], L.pos[2]], size: [1.1, 1.1, ear[2] - L.pos[2] + 0.35], rotation_deg: h, params: { radius_m: 0.55 } });
      break;
    case 'tent':
      if (!has((o) => o.type === 'tent'))
        out.push({ id: '~tent', type: 'tent', material: 'fabric', position: base.slice(), size: [2.2, 2.0, Math.max(1.1, earAbove + 0.55)], rotation_deg: h, params: {} });
      break;
    case 'car':
      if (!has((o) => o.type === 'car'))
        out.push({ id: '~car', type: 'car', material: 'car_body', position: base.slice(), size: [4.4, 1.8, Math.max(1.4, earAbove + 0.35)], rotation_deg: h - 90, params: { wipers: S.place.wipers } });
      break;
    case 'porch':
      if (!has((o) => o.type === 'roof')) {
        const z = base[2] + Math.max(2.4, earAbove + 0.8);
        out.push({ id: '~porch', type: 'roof', material: 'tin_roof', position: [base[0] - fwd[0] * 0.5, base[1] - fwd[1] * 0.5, z], size: [5, 3.2, 0.3], rotation_deg: h, params: { style: 'porch', pitch_deg: 8 } });
        out.push({ id: '~porch-eave', type: 'eave', material: 'tin_roof', position: [base[0] + fwd[0] * 1.1, base[1] + fwd[1] * 1.1, z], size: [5, 0.3, 0.06], rotation_deg: h, params: { drips_per_m: 3, depth_m: 1.6 } });
      }
      break;
    case 'attic':
      if (!has((o) => o.type === 'roof'))
        out.push({ id: '~attic', type: 'roof', material: 'tin_roof', position: [base[0], base[1], ear[2] + 0.7], size: [6, 5, 1.0], rotation_deg: h, params: { style: 'gable' } });
      break;
    case 'room_open':
    case 'room_closed':
      if (!objects.some((o) => o.type === 'window' && Math.hypot(o.position[0] - ear[0], o.position[1] - ear[1]) < 8))
        out.push({ id: '~window', type: 'window', material: 'glass', position: [base[0] + fwd[0] * 1.6, base[1] + fwd[1] * 1.6, ear[2] - 0.7], size: [1.2, 0.08, 1.3], rotation_deg: h, params: { open: S.place.window_open }, _int: true });
      break;
    case 'tree':
      if (!has((o) => o.type === 'tree')) {
        const conifer = placePreset(S.place.preset).vegetation === 'conifer';
        out.push({ id: '~tree', type: 'tree', material: conifer ? 'needles' : 'leaves', position: [base[0] + right[0] * 1.2, base[1] + right[1] * 1.2, base[2]], size: [8, 8, 11], rotation_deg: 0, params: { species: conifer ? 'conifer' : 'deciduous', height: 11 } });
      }
      break;
    case 'cave':
      out.push({ id: '~cave-mouth', type: 'eave', material: 'stone', position: [base[0] + fwd[0] * 3, base[1] + fwd[1] * 3, base[2] + 3], size: [4, 0.3, 0.1], rotation_deg: h, params: { drips_per_m: 2, depth_m: 3, ground: 'stone' }, _int: true });
      out.push({ id: '~cave-seep', type: 'eave', material: 'stone', position: [base[0] + right[0] * 1.8, base[1] + right[1] * 1.8, base[2] + 2.5], size: [0.6, 0.2, 0.1], rotation_deg: h, params: { drips_per_m: 6, catchment_m2: 2, tau_s: 300, ground: 'stone' }, _int: true });
      break;
    case 'underpass':
      for (const sgn of [-1, 1]) {
        out.push({ id: `~underpass-${sgn < 0 ? 'back' : 'front'}`, type: 'eave', material: 'concrete', position: [base[0] + fwd[0] * 4 * sgn, base[1] + fwd[1] * 4 * sgn, base[2] + 3.6], size: [6, 0.3, 0.1], rotation_deg: h, params: { drips_per_m: 2.5, depth_m: 2, ground: 'concrete' }, _int: true });
      }
      break;
    default:
      break;
  }
  return out;
}

/** Implicit objects from the Python schema's place.vegetation and place.emitters. */
function schemaObjects(S, env) {
  const out = [];
  const A = env.anchor;
  const rng = new CounterRng(env.seed, 'schema-objects', 0);
  const veg = S.place.vegetation;
  if (veg) {
    const dist = clamp(num(veg.distance_m, 12), 2, 200);
    const nd = clamp(Math.round(num(veg.deciduous, 0)), 0, 6);
    const nc = clamp(Math.round(num(veg.conifer, 0)), 0, 6);
    const n = Math.min(6, nd + nc);
    for (let i = 0; i < n; i++) {
      const conifer = i >= Math.min(nd, 6 - nc);
      const b = A.heading + (360 * (i + 0.5)) / Math.max(1, n) + 25 * rng.normal();
      const r = dist * (0.8 + 0.4 * rng.next());
      const v = bearingVector(b);
      const H = conifer ? 14 + 6 * rng.next() : 9 + 5 * rng.next();
      out.push({ id: `~veg${i}`, type: 'tree', material: conifer ? 'needles' : 'leaves', position: [A.pos[0] + v[0] * r, A.pos[1] + v[1] * r, A.pos[2]], size: conifer ? [H * 0.38, H * 0.38, H] : [H * 0.7, H * 0.7, H], rotation_deg: 0, params: { species: conifer ? 'conifer' : 'deciduous', height: H } });
    }
  }
  if (Array.isArray(S.place.emitters)) {
    S.place.emitters.slice(0, 8).forEach((e, i) => {
      const type = String(e.type || '').toLowerCase();
      if (!['gutter', 'eave', 'downspout'].includes(type)) return;
      const r = clamp(num(e.distance_m, 4), 0.5, 100);
      const b = num(e.bearing_deg, A.heading + 40 + 70 * i);
      const v = bearingVector(b);
      const pos = [A.pos[0] + v[0] * r, A.pos[1] + v[1] * r, A.pos[2] + (type === 'downspout' ? 0 : 2.6)];
      out.push({ id: `~em${i}`, type, material: type === 'eave' ? 'tile' : 'metal', position: pos, size: type === 'downspout' ? [0.1, 0.1, 2.6] : [4, 0.2, 0.1], rotation_deg: b + 90, params: { ...e } });
    });
  }
  return out;
}

function objectVoices(pc, o, voices) {
  const L = pc.env.listener;
  const ear = L.ear;
  const season = pc.season;
  const inside = listenerInside(o, ear);
  const nearRoom = (pc.shelter === 'room_open' || pc.shelter === 'room_closed') && o.type === 'window' && Math.hypot(o.position[0] - ear[0], o.position[1] - ear[1]) < 8;
  const route = inside || nearRoom || o._int === true ? 'int' : 'ext';
  const id = o.id;
  const k = (s) => `o:${id}:${s}`;
  const mat = (dflt) => (o.material && o.material.length ? o.material : dflt);
  const [w, d, hgt] = o.size;
  const [x, y, z] = o.position;
  const groundZ = L.pos[2];
  const expo = (px, py, top) => exposureAt(pc.objects, px, py, top, id, season);
  const add = (spec) => voices.set(spec.key, spec);
  const objId = id;
  switch (o.type) {
    case 'storm':
      return;
    case 'tree': {
      const H = treeHeight(o);
      const conifer = o.params.species === 'conifer' || /needle/.test(o.material);
      const R = Math.max(w, d, 2) / 2;
      const area = Math.PI * R * R;
      const leafOn = conifer || season !== 'winter';
      const crownZ = z + 0.62 * H;
      add(rainSpec(pc, k('rain'), objId, { pos: [x, y, crownZ], area: area * (leafOn ? 1 : 0.25), mats: [[conifer ? 'needles' : 'leaves', 1]], exposure: expo(x, y, z + H), route, maxEv: 1500, modes: 8, ref: clamp(R, 0.5, 4), windH: 0.7 * H }));
      add(windSpec(pc, k('wind'), objId, { mode: 'vegetation', pos: [x, y, crownZ], species: conifer ? 'conifer' : 'deciduous', leafOn, height: H, windScale: windAtHeight(1, 0.75 * H, pc.ctl.z0), gain: TYPE_GAIN.tree, ref: clamp(R, 1, 6), route }));
      const ic = interception(o, season);
      const dripPos = inside ? [L.pos[0], L.pos[1], groundZ] : [x, y, groundZ];
      add(rainSpec(pc, k('drip'), objId, { kind: 'drip', pos: dripPos, area: 1, mats: dripMats(pc), route, maxEv: 400, modes: 6, ref: clamp(R * 0.8, 0.5, 4), drip: { points: 18, tipRadiusMm: 2.2, fallHeight: Math.max(1, 0.4 * H), catchment: +(area * ic).toFixed(3), tau: 90 } }));
      return;
    }
    case 'grass': {
      const area = Math.max(0.1, w * d);
      add(rainSpec(pc, k('rain'), objId, { pos: [x, y, z + hgt], area, mats: [[mat('grass'), 1]], exposure: expo(x, y, z + hgt), route, maxEv: 800, modes: 4 }));
      add(windSpec(pc, k('wind'), objId, { mode: 'grass', pos: [x, y, z + hgt * 0.5], area, windScale: windAtHeight(1, Math.max(0.3, hgt), pc.ctl.z0), gain: TYPE_GAIN.grass, ref: clamp(Math.sqrt(area / Math.PI), 0.5, 6), route }));
      return;
    }
    case 'water':
    case 'surface': {
      const name = mat(o.type === 'water' ? 'water' : 'concrete');
      const rn = resolveMaterialName(name);
      let depthMm;
      if (o.params.depth_mm !== undefined) depthMm = num(o.params.depth_mm, undefined);
      else if (o.params.depth_m !== undefined) depthMm = num(o.params.depth_m, 0) * 1000;
      else if (o.type === 'water' && rn === 'water') depthMm = hgt > 0.01 ? hgt * 1000 : 1000;
      const area = Math.max(0.05, w * d);
      const top = z + (o.type === 'water' ? 0 : hgt);
      add(rainSpec(pc, k('rain'), objId, { pos: [x, y, top], area, mats: [[name, 1, { depthMm }]], exposure: expo(x, y, top), route, maxEv: 1500, modes: 12, windH: 0.3 }));
      return;
    }
    case 'roof': {
      const name = mat('tile');
      const area = Math.max(0.2, w * d);
      const top = z + Math.max(0.05, hgt);
      if (route === 'int' && area > 6) {
        for (const qx of [-1, 1]) {
          for (const qy of [-1, 1]) {
            const p = localToWorld(o, (qx * w) / 4, (qy * d) / 4, Math.max(0.05, hgt) * 0.5);
            const dist = Math.hypot(p[0] - ear[0], p[1] - ear[1], p[2] - ear[2]);
            add(rainSpec(pc, k(`q${qx}${qy}`), objId, { pos: p, area: area / 4, mats: [[name, 1]], exposure: expo(p[0], p[1], top), route, maxEv: 1500, modes: 14, ref: clamp(dist * 0.9, 0.3, 2), windH: top - groundZ }));
          }
        }
      } else {
        const p = localToWorld(o, 0, 0, Math.max(0.05, hgt) * 0.5);
        add(rainSpec(pc, k('rain'), objId, { pos: p, area, mats: [[name, 1]], exposure: expo(p[0], p[1], top), route, maxEv: 1800, modes: 14, windH: top - groundZ }));
      }
      return;
    }
    case 'wall':
    case 'window': {
      const glass = o.type === 'window';
      const name = mat(glass ? 'glass' : 'concrete');
      const area = Math.max(0.1, w * Math.max(hgt, 0.1));
      const p = [x, y, z + hgt / 2];
      add(rainSpec(pc, k('rain'), objId, {
        pos: p,
        area,
        mats: [[name, 1]],
        exposure: expo(x, y, z + hgt),
        route,
        maxEv: glass ? 1200 : 800,
        modes: glass ? 12 : 8,
        ref: route === 'int' ? clamp(Math.hypot(p[0] - ear[0], p[1] - ear[1], p[2] - ear[2]) * 0.9, 0.3, 1.5) : undefined,
        driven: { normal: wrap360(o.rotation_deg), twoSided: true, min: glass ? 0.12 : 0.03, h: Math.max(0.5, z + hgt / 2 - groundZ) },
        windH: Math.max(0.5, z + hgt / 2 - groundZ),
      }));
      return;
    }
    case 'gutter': {
      const len = Math.max(0.3, num(o.params.length_m, Math.max(w, d)));
      const catchment = num(o.params.catchment_m2, roofCatchment(pc, o) ?? len * 2.5);
      add(rainSpec(pc, k('flow'), objId, { kind: 'flow', pos: [x, y, z], area: 0.2, mats: [[mat('metal'), 0.7], ['water', 0.3, { depthMm: 20 }]], route, maxEv: 600, modes: 8, ref: 0.5, flow: { catchment, tau: 30, pipeLength: 0, fallHeight: 0.15 } }));
      return;
    }
    case 'downspout': {
      const H = Math.max(0.3, num(o.params.height_m, hgt || 2.5));
      const catchment = num(o.params.catchment_m2, roofCatchment(pc, o) ?? 20);
      add(rainSpec(pc, k('flow'), objId, { kind: 'flow', pos: [x, y, z + 0.3], area: 0.2, mats: [['concrete', 0.6], ['puddle', 0.4]], route, maxEv: 600, modes: 8, ref: 0.4, flow: { catchment, tau: 45, pipeLength: H, fallHeight: 0.25 } }));
      return;
    }
    case 'eave': {
      const len = Math.max(0.3, Math.max(w, d));
      const dpm = Math.max(0.2, num(o.params.drips_per_m, 3));
      const ground = o.params.ground ? [[String(o.params.ground), 0.7], ['puddle', 0.3]] : dripMats(pc);
      add(rainSpec(pc, k('drip'), objId, {
        kind: 'drip',
        pos: [x, y, groundZ],
        area: 1,
        mats: ground,
        route,
        maxEv: 400,
        modes: 6,
        ref: clamp(len / 4, 0.4, 3),
        drip: { points: clamp(Math.round(dpm * len), 2, 48), tipRadiusMm: 1.6, fallHeight: Math.max(0.3, z - groundZ), catchment: num(o.params.catchment_m2, len * num(o.params.depth_m, 2.5)), tau: num(o.params.tau_s, 25) },
      }));
      return;
    }
    case 'wire': {
      const len = Math.max(1, w, d);
      add(windSpec(pc, k('wind'), objId, { mode: 'wire', pos: [x, y, z + hgt / 2], diameter: clamp(num(o.params.diameter_m, 0.012), 0.0005, 0.5), length: len, windScale: windAtHeight(1, Math.max(0.5, z), pc.ctl.z0), gain: TYPE_GAIN.wire, ref: clamp(len / 4, 0.5, 8), route }));
      return;
    }
    case 'chime': {
      const n = clamp(Math.round(num(o.params.tubes, 5)), 1, 6);
      let notes = Array.isArray(o.params.notes) ? o.params.notes.map((v) => num(v, 880)).slice(0, 6) : null;
      if (!notes) {
        const rng = new CounterRng(pc.env.seed, `chime:${id}`, 0);
        const base = 600 + 500 * rng.next();
        notes = [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2].slice(0, n).map((r) => +(base * r).toFixed(2));
      }
      add(windSpec(pc, k('wind'), objId, { mode: 'chime', pos: [x, y, z + hgt / 2], notes, windScale: windAtHeight(1, Math.max(0.5, z), pc.ctl.z0), gain: TYPE_GAIN.chime, ref: 0.5, route }));
      return;
    }
    case 'flag': {
      add(windSpec(pc, k('wind'), objId, { mode: 'flag', pos: [x, y, z + Math.max(1, hgt)], length: Math.max(0.3, w || 1.5), windScale: windAtHeight(1, Math.max(1, z + hgt), pc.ctl.z0), gain: TYPE_GAIN.flag, ref: 1, route }));
      return;
    }
    case 'car': {
      const top = z + Math.max(0.5, hgt);
      const wipers = String(o.params.wipers ?? (inside ? pc.S.place.wipers : 'off'));
      if (inside) {
        const fwd = bearingVector(carForward(o));
        add(rainSpec(pc, k('roof'), objId, { pos: [ear[0], ear[1], top - 0.02], area: 2.4, mats: [[mat('car_body'), 1]], exposure: expo(ear[0], ear[1], top), route: 'int', maxEv: 2500, modes: 14, ref: clamp((top - ear[2]) * 0.9, 0.2, 1), windH: top - groundZ }));
        add(rainSpec(pc, k('windscreen'), objId, { pos: [ear[0] + fwd[0] * 0.85, ear[1] + fwd[1] * 0.85, ear[2] + 0.1], area: 1.2, mats: [['glass', 1]], exposure: 0.75 * expo(ear[0] + fwd[0], ear[1] + fwd[1], top), route: 'int', maxEv: 1500, modes: 12, ref: 0.35, windH: top - groundZ }));
        add(rainSpec(pc, k('rear'), objId, { pos: [ear[0] - fwd[0] * 1.4, ear[1] - fwd[1] * 1.4, ear[2] + 0.05], area: 0.9, mats: [['glass', 1]], exposure: 0.6 * expo(ear[0] - fwd[0], ear[1] - fwd[1], top), route: 'int', maxEv: 800, modes: 10, ref: 0.5, windH: top - groundZ }));
        if (wipers !== 'off') add(windSpec(pc, k('wiper'), objId, { mode: 'wiper', pos: [ear[0] + fwd[0] * 0.95, ear[1] + fwd[1] * 0.95, ear[2] - 0.25], speed: wipers, gain: TYPE_GAIN.wiper, ref: 0.4, route: 'int', aux: 'wet' }));
      } else {
        const area = Math.max(1, w * d * 0.85);
        add(rainSpec(pc, k('body'), objId, { pos: [x, y, top], area, mats: [[mat('car_body'), 0.65], ['glass', 0.35]], exposure: expo(x, y, top), route, maxEv: 1500, modes: 14, ref: clamp(Math.sqrt(area / Math.PI), 0.5, 2), windH: top - groundZ }));
        if (wipers !== 'off') add(windSpec(pc, k('wiper'), objId, { mode: 'wiper', pos: [x, y, z + hgt * 0.7], speed: wipers, gain: TYPE_GAIN.wiper * 0.5, ref: 1, route, aux: 'wet' }));
      }
      return;
    }
    case 'tent': {
      const top = z + Math.max(0.5, hgt);
      const name = mat('fabric');
      if (inside) {
        const ridgeX = w >= d;
        for (const side of [-1, 1]) {
          const p = localToWorld(o, ridgeX ? 0 : (side * w) / 4, ridgeX ? (side * d) / 4 : 0, Math.max(0.5, hgt) * 0.7);
          const dist = Math.hypot(p[0] - ear[0], p[1] - ear[1], p[2] - ear[2]);
          add(rainSpec(pc, k(side < 0 ? 'left' : 'right'), objId, { pos: p, area: Math.max(0.5, w * d * 0.55), mats: [[name, 1]], exposure: expo(p[0], p[1], top), route: 'int', maxEv: 2200, modes: 14, ref: clamp(dist * 0.9, 0.25, 1), windH: top - groundZ }));
        }
      } else {
        const area = Math.max(0.5, w * d * 1.15);
        add(rainSpec(pc, k('fabric'), objId, { pos: [x, y, z + Math.max(0.5, hgt) * 0.7], area, mats: [[name, 1]], exposure: expo(x, y, top), route, maxEv: 1500, modes: 14, ref: clamp(Math.sqrt(area / Math.PI), 0.4, 2), windH: top - groundZ }));
      }
      return;
    }
    case 'umbrella': {
      const r = Math.max(0.2, num(o.params.radius_m, Math.max(w, d) / 2 || 0.55));
      const area = Math.PI * r * r;
      const top = coverZ(o);
      add(rainSpec(pc, k('canopy'), objId, { pos: [x, y, top], area, mats: [[mat('umbrella'), 1]], exposure: expo(x, y, top), route, maxEv: 2500, modes: 14, ref: route === 'int' ? clamp((top - ear[2]) * 0.9, 0.2, 1) : clamp(r, 0.3, 1), windH: top - groundZ }));
      [0, 120, 240].forEach((b, i) => {
        const v = bearingVector(o.rotation_deg + b);
        add(rainSpec(pc, k(`rim${i}`), objId, { kind: 'drip', pos: [x + v[0] * r, y + v[1] * r, groundZ], area: 1, mats: dripMats(pc), route, maxEv: 300, modes: 6, ref: 0.35, drip: { points: 3, tipRadiusMm: 1.8, fallHeight: Math.max(0.3, top - groundZ), catchment: +((area / 3) * 0.97).toFixed(3), tau: 6 } }));
      });
      return;
    }
    default: {
      const area = Math.max(0.05, w * d);
      const top = z + hgt;
      add(rainSpec(pc, k('rain'), objId, { pos: [x, y, top], area, mats: [[mat('plastic'), 1]], exposure: expo(x, y, top), route, maxEv: 1000, modes: 10 }));
    }
  }
}

/** Build the voice plan (pure data) for the scene at one instant. */
function planScene(S, ctl, env) {
  const voices = new Map();
  const shelter = ctl.shelter;
  const sInfo = shelterInfo(shelter);
  const L = env.listener;
  const season = ctl.season;
  const presetKey = ctl.presetKey;
  const preset = ctl.preset;
  const objects = S.place.objects.slice();
  for (const o of schemaObjects(S, env)) objects.push(o);
  for (const o of shelterObjects(S, shelter, env, objects)) objects.push(o);

  let surfaces = S.place.surfaces;
  if (ctl.kinds[0] === 2 && ctl.tempC < 1.5) {
    surfaces = { snow: 70 };
    for (const [k, v] of Object.entries(S.place.surfaces)) surfaces[k] = (surfaces[k] || 0) + 0.3 * num(v, 0);
  }
  const mix = surfaceMix(surfaces);
  const forest = FORESTS[presetKey] || null;
  const forestIc = forest ? (forest.species === 'deciduous' && season === 'winter' ? forest.ic * 0.35 : forest.ic) : 0;
  const pc = { S, ctl, env, objects, shelter, sInfo, season, mix, preset, presetKey };

  // Ground patches: 8 azimuth sectors x 3 distance rings around the listener.
  const h = Math.max(0.2, L.ear[2] - L.pos[2]);
  const nRings = env.quality === 'low' ? 2 : RINGS.length;
  for (let ri = 0; ri < nRings; ri++) {
    const ring = RINGS[ri];
    for (let s = 0; s < SECTORS; s++) {
      const g = patchGeometry(ri, s, h);
      const key = `g${ri}.${s}`;
      const wx = L.pos[0] + g.rel[0];
      const wy = L.pos[1] + g.rel[1];
      let exposure = coveredBy(objects, wx, wy, L.pos[2], season) ? exposureAt(objects, wx, wy, L.pos[2], null, season) : ri === 0 ? sInfo.nearExposure : 1;
      exposure *= 1 - forestIc;
      const pm = patchMaterials(surfaces, env.seed, key);
      voices.set(
        key,
        rainSpec(pc, key, 'ground', {
          posMode: 'listener',
          rel: g.rel,
          area: g.area,
          mats: pm,
          exposure,
          stem: 'rain',
          route: 'ext',
          maxEv: ring.maxEv,
          modes: ring.modes,
          bubbles: ring.bubbles,
          gain: g.dir,
          ref: g.ref,
          windH: 0.3,
          hrtf: env.quality === 'high' || (env.quality === 'balanced' ? ri < 2 : ri === 0),
        }),
      );
    }
  }

  // Far-field bed: decorrelated 4 channels panned to the quadrants around the listener.
  const wf = waterFraction(mix);
  const bedCfg = { seed: (env.seedKey ^ hashString('bed')) >>> 0 };
  const bed = {
    key: 'bed',
    objectId: null,
    proc: 'bed',
    stem: 'bed',
    route: 'ext',
    channels: 4,
    pan: 'quad',
    posMode: 'listener',
    pos: null,
    rel: null,
    quad: [45, 135, 225, 315].map((b) => {
      const v = bearingVector(b);
      return [v[0] * 25, v[1] * 25, 0];
    }),
    ref: 1,
    rolloff: 0,
    hrtf: env.quality !== 'low',
    config: bedCfg,
    structKey: JSON.stringify(bedCfg),
    ctl: { farShare: (0.3 + 0.9 * wf) * num(preset.far, 0.85) * (nRings < RINGS.length ? 1.6 : 1) },
  };
  bed.keys = keysFor(bed);
  voices.set('bed', bed);

  // Wind 'air' at the ear: two decorrelated channels, balance toward the upwind side.
  voices.set('air', windSpec(pc, 'air', null, { mode: 'air', channels: 2, pan: 'stereo', windScale: windAtHeight(1, h, ctl.z0), gain: TYPE_GAIN.air }));

  // Place vegetation heard around the listener when the scene has no trees of its own.
  const veg = preset.vegetation || 'none';
  const anyTree = objects.some((o) => o.type === 'tree' && Math.hypot(o.position[0] - L.pos[0], o.position[1] - L.pos[1]) < 60);
  const rng = new CounterRng(env.seed, 'ambience', 0);
  if (veg === 'grass') {
    [60, 240].forEach((b, i) => {
      const v = bearingVector(b + 30 * rng.normal());
      voices.set(`amb.grass${i}`, windSpec(pc, `amb.grass${i}`, null, { mode: 'grass', posMode: 'listener', rel: [v[0] * 7, v[1] * 7, 0.2], area: 60, windScale: windAtHeight(1, 0.4, ctl.z0), gain: TYPE_GAIN.grass, ref: 3 }));
    });
  } else if (veg !== 'none' && !anyTree) {
    const n = forest ? 5 : 3;
    const H = forest ? 18 : 11;
    for (let i = 0; i < n; i++) {
      const b = (360 * i) / n + 20 + 25 * rng.normal();
      const r = (forest ? 12 : 22) + 10 * rng.next();
      const v = bearingVector(b);
      const conifer = veg === 'conifer';
      voices.set(
        `amb.tree${i}`,
        windSpec(pc, `amb.tree${i}`, null, {
          mode: 'vegetation',
          posMode: 'listener',
          rel: [v[0] * r, v[1] * r, 0.6 * H],
          species: conifer ? 'conifer' : 'deciduous',
          leafOn: conifer || season !== 'winter',
          height: H,
          windScale: windAtHeight(1, 0.75 * H, ctl.z0),
          gain: TYPE_GAIN.tree * (forest ? 1.3 : 1),
          ref: 5,
        }),
      );
    }
  }
  if (forest) {
    const conifer = forest.species === 'conifer';
    const leafOn = conifer || season !== 'winter';
    voices.set('canopy', rainSpec(pc, 'canopy', null, { posMode: 'listener', rel: [1.5, 1, 9], area: leafOn ? 60 : 18, mats: [[conifer ? 'needles' : 'leaves', 1]], exposure: 1, stem: 'objects', route: 'ext', maxEv: 1500, modes: 8, ref: 3, windH: 8 }));
    voices.set('canopy.drip', rainSpec(pc, 'canopy.drip', null, { kind: 'drip', posMode: 'listener', rel: [0, 0, 0], area: 1, mats: dripMats(pc), stem: 'objects', route: 'ext', maxEv: 500, modes: 6, ref: 2, drip: { points: 20, tipRadiusMm: 2.2, fallHeight: 7, catchment: +(60 * forestIc).toFixed(3), tau: 120 } }));
  }

  for (const o of objects) objectVoices(pc, o, voices);

  let windowOpen = 0;
  if (shelter === 'room_open' || shelter === 'room_closed') {
    for (const o of objects) {
      // Explicit windows only: the implicit one follows place.window_open through the control model.
      if (o.type === 'window' && !String(o.id).startsWith('~') && Math.hypot(o.position[0] - L.ear[0], o.position[1] - L.ear[1]) < 8) {
        windowOpen = Math.max(windowOpen, clamp(num(o.params.open, 0), 0, 1));
      }
    }
  }
  const reverbSpec = reverbSpecFor(shelter, S.place.preset);
  return {
    voices,
    shelter,
    windowOpen,
    reverbSpec,
    reverbKey: JSON.stringify(reverbSpec || null),
    S,
    objects,
    fieldSum: fieldSum(h, nRings),
  };
}

// ---------------------------------------------------------------------------
// Per-tick parameters
// ---------------------------------------------------------------------------

/**
 * Keep at most `cap` voices for live playback: listener-attached voices first, then the nearest.
 * Every voice runs on the one audio thread, so a scene with many objects can outgrow it.
 */
function capVoices(plan, cap, L) {
  if (!(cap > 0) || plan.voices.size <= cap) return;
  const lp = L.pos || [0, 0, 0];
  const ranked = [];
  for (const [key, spec] of plan.voices) {
    if (spec.posMode === 'listener') {
      ranked.push([key, -1]);
      continue;
    }
    const p = specPos(spec, L);
    ranked.push([key, Math.hypot(p[0] - lp[0], p[1] - lp[1], p[2] - lp[2])]);
  }
  ranked.sort((a, b) => a[1] - b[1]);
  for (let i = cap; i < ranked.length; i++) plan.voices.delete(ranked[i][0]);
}

function specPos(spec, L) {
  if (spec.posMode === 'listener') {
    const r = spec.rel || [0, 0, 0];
    return [L.pos[0] + r[0], L.pos[1] + r[1], L.pos[2] + r[2]];
  }
  return spec.pos || L.pos;
}

function clampParam(proc, name, v) {
  const r = RANGE[proc][name] || (name[0] === 'r' ? [0, 1e7] : [0, 100]);
  if (!Number.isFinite(v)) return r[0];
  return v < r[0] ? r[0] : v > r[1] ? r[1] : v;
}

function drivenFactor(dr, ctl, model, pos, L) {
  const U = windAtHeight(model.windRefAt(pos, ctl, L), dr.h, ctl.z0);
  const rel = Math.cos((ctl.fromDeg - dr.normal) * DEG);
  const facing = dr.twoSided ? Math.abs(rel) : Math.max(0, rel);
  // Rain slants at tan(alpha) = U / v_fall (~6.5 m/s): vertical faces catch U/6.5 of the flux.
  return clamp(dr.min + (U / 6.5) * facing, dr.min, 1.6);
}

const CUTOFF_CACHE = new Map();
function bedCutoff(tempC, rh) {
  const k = `${Math.round(tempC)}|${Math.round(rh / 2)}`;
  let v = CUTOFF_CACHE.get(k);
  if (v === undefined) {
    v = absorptionCutoffHz(60, tempC, rh, 6, 20000);
    if (CUTOFF_CACHE.size > 500) CUTOFF_CACHE.clear();
    CUTOFF_CACHE.set(k, v);
  }
  return v;
}

/** Worklet parameter values for one voice (k-rate AudioParams), clamped to their ranges. */
function voiceParams(spec, ctl, model, env) {
  const o = spec._out || (spec._out = {});
  const L = env.listener;
  const pos = specPos(spec, L);
  if (spec.proc === 'rain') {
    const c = spec.ctl;
    o.intensity = spec.config.kind === 'rain' ? c.intensity * (c.driven ? drivenFactor(c.driven, ctl, model, pos, L) : 1) : 1;
    o.wind = windAtHeight(model.windRefAt(pos, ctl, L), c.windH, ctl.z0);
    o.gain = c.gain;
  } else if (spec.proc === 'wind') {
    o.wind = model.windRefAt(pos, ctl, L);
    o.gain = spec.ctl.gain;
    o.aux = spec.ctl.aux === 'wet' ? ctl.film : 0;
  } else {
    const cal = env.cal;
    const eRel = Math.max(0, num(ctl.energyRate, 0)) / cal.eRef;
    const kindF = ctl.kinds[0] === 2 ? 0.06 : ctl.kinds[0] === 3 ? 0.35 : ctl.kinds[0] === 1 ? 0.7 : 1;
    const hail = ctl.hailFlux > 0 ? 1 + Math.min(2, 0.3 * Math.sqrt(ctl.hailFlux)) : 1;
    const fieldRms = Math.sqrt(cal.msPerM2 * env.fieldSum);
    o.level = (fieldRms * Math.sqrt(eRel * spec.ctl.farShare) * kindF * hail) / cal.bedUnit / Math.SQRT2;
    let dz = 0;
    if (ctl.type === 'drizzle' || ctl.type === 'freezing_drizzle') dz = 0.3 * Math.sqrt(ctl.rate / 0.5);
    else if (ctl.kinds[0] === 0 && ctl.drops === 'fine') dz = 0.08 * Math.sqrt(ctl.rate / 5);
    o.drizzle = (dz * fieldRms) / cal.drizzleUnit / Math.SQRT2;
    o.cutoff = bedCutoff(ctl.tempC, ctl.rh);
    o.density = clamp(Math.sqrt(ctl.totalFlux / 2000), 0.15, 4);
    o.intensity = 1;
    o.gain = 1;
    o.depth = ctl.character === 'showery' ? 3 : ctl.character === 'intermittent' ? 2.2 : 1.5;
  }
  for (const n of PARAM_NAMES[spec.proc]) o[n] = clampParam(spec.proc, n, o[n]);
  return o;
}

function initParams(spec, ctl, model, env) {
  const o = voiceParams(spec, ctl, model, env);
  const out = {};
  for (const n of PARAM_NAMES[spec.proc]) out[n] = o[n];
  return out;
}

/** The shared weather on the rg-weather bus: class impact rates and diameters, wetness, puddles, liquid rate. */
function busParams(ctl, emit) {
  for (let i = 0; i < 6; i++) {
    emit(`bus|${RN[i]}`, clampParam('bus', RN[i], ctl.flux[i]));
    emit(`bus|${DN[i]}`, clampParam('bus', DN[i], ctl.dcls[i]));
  }
  emit('bus|wetness', clampParam('bus', 'wetness', ctl.film));
  emit('bus|puddle', clampParam('bus', 'puddle', ctl.puddle));
  emit('bus|liquid', clampParam('bus', 'liquid', ctl.liquid));
}

function airPan(ctl, L) {
  return clamp(0.3 * Math.sin(angDiff(ctl.fromDeg, L.heading) * DEG), -0.3, 0.3);
}

/** Emit every voice parameter and panner position for this tick. */
function emitParams(plan, ctl, model, env, emit) {
  if (!plan) return;
  const L = env.listener;
  for (const spec of plan.voices.values()) {
    const o = voiceParams(spec, ctl, model, env);
    const K = spec.keys;
    for (const n of PARAM_NAMES[spec.proc]) emit(K[n], o[n]);
    if (spec.pan === 'panner') {
      const p = web(specPos(spec, L));
      emit(K.px, p[0]);
      emit(K.py, p[1]);
      emit(K.pz, p[2]);
    } else if (spec.pan === 'quad') {
      for (let i = 0; i < 4; i++) {
        const q = spec.quad[i];
        emit(K.qx[i], L.pos[0] + q[0]);
        emit(K.qy[i], L.pos[2] + q[2]);
        emit(K.qz[i], -(L.pos[1] + q[1]));
      }
    } else if (spec.pan === 'stereo') emit(K.pan, airPan(ctl, L));
  }
}

function stemCal(cal, s) {
  if (s === 'wind') return cal.wind;
  if (s === 'thunder') return 1;
  return cal.rain;
}

// Playback trim: the physical calibration puts moderate rain near -27 dBFS RMS, inaudible on
// laptop speakers; +10 dB lands it near -17 dB RMS while the -1 dBFS limiter guards storms.
const OUTPUT_TRIM_DB = 10;

/** Transmission loss, stem gains (calibration x mode x mix) and master gain. */
function masterParams(plan, ctl, cal, mix, masterDb, sr, emit) {
  const shelter = plan ? plan.shelter : ctl.shelter;
  const openness = Math.max(ctl.windowOpen, plan ? plan.windowOpen : 0);
  const tl = transmissionFor(shelter, openness);
  const ny = 0.45 * sr;
  emit('tl|lpC', Math.min(ny, tl.closed.lp));
  emit('tl|shC', tl.closed.shelfDb);
  emit('tl|gC', dbToGain(tl.closed.gainDb) * (1 - tl.openness));
  emit('tl|lpO', Math.min(ny, tl.open.lp));
  emit('tl|shO', tl.open.shelfDb);
  emit('tl|gO', dbToGain(tl.open.gainDb) * tl.openness);
  const mode = MODE_GAINS[ctl.mode] || MODE_GAINS.none;
  for (const s of STEMS) emit(`stem|${s}`, stemCal(cal, s) * dbToGain((mode[s] || 0) + (mix[s] || 0)));
  emit('master|gain', dbToGain(masterDb + (mode.master || 0) + OUTPUT_TRIM_DB));
}

/** AudioListener position (ear) and orientation from heading/pitch, in Web Audio coordinates. */
function listenerParams(L, emit) {
  const e = web(L.ear);
  emit('L|px', e[0]);
  emit('L|py', e[1]);
  emit('L|pz', e[2]);
  const h = L.heading * DEG;
  const p = clamp(num(L.pitch, 0), -89, 89) * DEG;
  const f = web([Math.sin(h) * Math.cos(p), Math.cos(h) * Math.cos(p), Math.sin(p)]);
  const u = web([-Math.sin(h) * Math.sin(p), -Math.cos(h) * Math.sin(p), Math.cos(p)]);
  emit('L|fx', f[0]);
  emit('L|fy', f[1]);
  emit('L|fz', f[2]);
  emit('L|ux', u[0]);
  emit('L|uy', u[1]);
  emit('L|uz', u[2]);
}

// ---------------------------------------------------------------------------
// Thunder sources and flashes
// ---------------------------------------------------------------------------

/** Active thunder sources: the scene's weather.thunder cell and every 'storm' object. */
function thunderSources(S, ctl, L, t, includeDisabled = false) {
  const out = [];
  const V = ctl.V;
  const en = V('weather.thunder.enabled');
  const enabled = en === true || en === 'true' || en === 1;
  if (enabled || includeDisabled) {
    const d0 = num(V('weather.thunder.distance_km'), 8);
    const appr = num(V('weather.thunder.approach_kmh'), 0);
    const dkm = clamp(ctl.tl.has('weather.thunder.distance_km') ? d0 : d0 - (appr * t) / 3600, 0.3, 60);
    const v = bearingVector(num(V('weather.thunder.bearing_deg'), 270));
    out.push({
      id: 'wx',
      objectId: null,
      center: [L.pos[0] + v[0] * dkm * 1000, L.pos[1] + v[1] * dkm * 1000],
      ratePerMin: enabled ? num(V('weather.thunder.flashes_per_min'), 2) * ctl.thunderMul : 0,
      intensity: clamp(num(V('weather.thunder.stroke_intensity'), 0.5), 0, 1),
      cg: clamp(num(V('weather.thunder.cg_fraction'), 0.4), 0, 1),
      spreadM: Math.max(300, 0.12 * dkm * 1000),
    });
  }
  for (const o of S.place.objects) {
    if (o.type !== 'storm') continue;
    const P = o.params || {};
    const track = Array.isArray(P.track) && P.track.length ? P.track : null;
    const c = track ? alongTrack(track, (num(P.speed_kmh, 25) / 3.6) * Math.max(0, t)) : [o.position[0], o.position[1]];
    out.push({
      id: `storm:${o.id}`,
      objectId: o.id,
      center: c,
      ratePerMin: P.enabled === false ? 0 : clamp(num(P.flashes_per_min, 3), 0, FLASH_CEILING_PER_MIN),
      intensity: clamp(num(P.intensity, 0.6), 0, 1),
      cg: clamp(num(P.cg_fraction, 0.45), 0, 1),
      spreadM: Math.max(200, num(P.radius_km, 4) * 500),
    });
  }
  return out;
}

/** Geometry and timing of one flash (deterministic per seed and key). */
function prepareFlash(seed, src, key, ear, tempC, forceCg = false) {
  const rng = new CounterRng(seed, `flash:${key}`, 0);
  const cg = forceCg || rng.next() < src.cg;
  const strike = [src.center[0] + src.spreadM * rng.normal(), src.center[1] + src.spreadM * rng.normal()];
  const intensity = clamp(src.intensity * Math.exp(0.3 * rng.normal()), 0.05, 1);
  const chan = buildChannel({ seed, key, kind: cg ? 'cg' : 'ic', strike, intensity });
  const nearestM = Math.max(1, nearestDistance(chan, ear));
  const dKm = nearestM / 1000;
  return {
    id: key,
    key,
    kind: chan.kind,
    chan,
    strike,
    intensity,
    nearestM,
    delaySec: nearestM / speedOfSound(tempC),
    distanceKm: dKm,
    bearingDeg: bearingTo(ear, strike),
    brightness: clamp(1.15 - dKm / 20, 0.12, 1) * (0.55 + 0.45 * intensity) * (cg ? 1 : 0.55),
    stormId: src.objectId,
  };
}

function synthRateFor(nearestM, ctxRate) {
  return Math.min(ctxRate, nearestM < 1500 ? 48000 : nearestM < 5000 ? 24000 : 16000);
}

function dirWords(fl) {
  return fl.distanceKm < 1 ? 'overhead' : `to the ${compassName(fl.bearingDeg)}`;
}

function flashCaption(fl) {
  return fl.kind === 'ic' ? `[Sheet lightning ${dirWords(fl)}]` : `[Lightning ${dirWords(fl)}]`;
}

function thunderCaption(fl) {
  const km = fl.distanceKm;
  const d = km < 10 ? km.toFixed(1) : String(Math.round(km));
  if (km < 1.2) return `[Thunder cracks ${dirWords(fl)}]`;
  if (km < 6) return `[Thunder, ${d} km ${dirWords(fl)}]`;
  return `[Distant thunder rumbles, ${d} km ${dirWords(fl)}]`;
}

// ---------------------------------------------------------------------------
// Audio graph (one per AudioContext / OfflineAudioContext)
// ---------------------------------------------------------------------------

function softClipCurve(n = 4096) {
  const c = new Float32Array(n);
  // Linear to -2 dBFS, then a tanh knee that never passes -1 dBFS (the compressor's
  // 2 ms attack lets transients through; this catches them below the stated ceiling).
  const k = 0.794;
  const top = 0.891;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    const a = Math.abs(x);
    c[i] = a <= k ? x : Math.sign(x) * (k + (top - k) * Math.tanh((a - k) / (top - k)));
  }
  return c;
}

let BUS_SEQ = 0;

class SceneGraph {
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.offline = !!o.offline;
    this.renderMode = o.renderMode === 'stereo' ? 'stereo' : 'binaural';
    this.onStats = o.onStats || null;
    this.statsOn = !!(this.onStats && o.stats);
    this.busId = `wx${++BUS_SEQ}`;
    this.bus = null;
    this.voices = new Map();
    this.pmap = new Map();
    this.last = new Map();
    this.thunders = new Set();
    this.dead = [];
    this.analyser = null;
    this.meter = null;
    this._build(o.meter !== false && !this.offline);
  }

  _gain(v = 1) {
    const g = this.ctx.createGain();
    g.gain.value = v;
    return g;
  }

  _biquad(type, f, q, gain = 0) {
    const b = this.ctx.createBiquadFilter();
    b.type = type;
    b.frequency.value = f;
    b.Q.value = q;
    b.gain.value = gain;
    return b;
  }

  _reg(key, param) {
    if (param) this.pmap.set(key, param);
  }

  _build(withMeter) {
    const ctx = this.ctx;
    const ny = 0.45 * ctx.sampleRate;
    // The weather bus: one node holds the parameters every rain surface shares. Its (silent)
    // output feeds each surface's input, so it renders before them in every quantum.
    try {
      this.bus = new AudioWorkletNode(ctx, PROC.bus, { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: { bus: this.busId } });
      for (const n of PARAM_NAMES.bus) this._reg(`bus|${n}`, this.bus.parameters.get(n));
    } catch (e) {
      this.bus = null;
    }
    this.ext = {};
    this.int = {};
    this.extBus = this._gain();
    this.intBus = this._gain();
    this.post = this._gain();
    this.pre = this._gain();
    for (const s of STEMS) {
      this.ext[s] = this._gain(0);
      this.int[s] = this._gain(0);
      this.ext[s].connect(this.extBus);
      this.int[s].connect(this.intBus);
      this._reg(`stem|${s}`, [this.ext[s].gain, this.int[s].gain]);
    }
    // Envelope transmission loss: closed and open paths, blended by window opening.
    const lpC = this._biquad('lowpass', Math.min(20000, ny), qDb(Math.SQRT1_2));
    const shC = this._biquad('highshelf', 1500, 0, 0);
    const gC = this._gain(0);
    const lpO = this._biquad('lowpass', Math.min(20000, ny), qDb(Math.SQRT1_2));
    const shO = this._biquad('highshelf', 1500, 0, 0);
    const gO = this._gain(1);
    this.extBus.connect(lpC);
    lpC.connect(shC);
    shC.connect(gC);
    gC.connect(this.post);
    this.extBus.connect(lpO);
    lpO.connect(shO);
    shO.connect(gO);
    gO.connect(this.post);
    this.intBus.connect(this.post);
    this._reg('tl|lpC', lpC.frequency);
    this._reg('tl|shC', shC.gain);
    this._reg('tl|gC', gC.gain);
    this._reg('tl|lpO', lpO.frequency);
    this._reg('tl|shO', shO.gain);
    this._reg('tl|gO', gO.gain);
    this.tlNodes = [lpC, shC, gC, lpO, shO, gO];
    try {
      this.reverb = new ReverbUnit(ctx);
      this.post.connect(this.reverb.input);
      this.reverb.output.connect(this.pre);
    } catch (e) {
      this.reverb = null;
    }
    this.post.connect(this.pre);
    this.master = this._gain(1);
    this.pre.connect(this.master);
    this._reg('master|gain', this.master.gain);
    let node = this.master;
    try {
      const lim = ctx.createDynamicsCompressor();
      lim.threshold.value = -1;
      lim.knee.value = 0;
      lim.ratio.value = 20;
      lim.attack.value = 0.002;
      lim.release.value = 0.15;
      node.connect(lim);
      node = lim;
      this.limiter = lim;
    } catch (e) {
      this.limiter = null;
    }
    try {
      const ws = ctx.createWaveShaper();
      ws.curve = softClipCurve();
      ws.oversample = 'none';
      node.connect(ws);
      node = ws;
      this.clip = ws;
    } catch (e) {
      this.clip = null;
    }
    this.out = this._gain(1);
    node.connect(this.out);
    this.out.connect(ctx.destination);
    this._regListener();
    if (withMeter) this._buildMeter();
  }

  _buildMeter() {
    const ctx = this.ctx;
    try {
      this.analyser = ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.8;
      this.out.connect(this.analyser);
      // BS.1770 K-weighting: +4 dB high shelf at 1682 Hz, then 38 Hz high-pass (Q 0.5).
      const hs = this._biquad('highshelf', 1681.97, 0, 4.0);
      const hp = this._biquad('highpass', 38.13, qDb(0.5));
      this.out.connect(hs);
      hs.connect(hp);
      const sp = ctx.createChannelSplitter(2);
      hp.connect(sp);
      const size = 32768;
      const mk = (fft) => {
        const a = ctx.createAnalyser();
        a.fftSize = fft;
        a.smoothingTimeConstant = 0;
        return a;
      };
      const kL = mk(size);
      const kR = mk(size);
      sp.connect(kL, 0);
      sp.connect(kR, 1);
      const sp2 = ctx.createChannelSplitter(2);
      this.out.connect(sp2);
      const pL = mk(8192);
      const pR = mk(8192);
      sp2.connect(pL, 0);
      sp2.connect(pR, 1);
      this.meter = { kL, kR, pL, pR, bufK: new Float32Array(size), bufP: new Float32Array(8192), win: Math.min(size, Math.round(0.4 * ctx.sampleRate)), nodes: [hs, hp, sp, sp2] };
    } catch (e) {
      this.meter = null;
    }
  }

  _regListener() {
    const l = this.ctx.listener;
    if (!l) return;
    if (l.positionX) {
      this._reg('L|px', l.positionX);
      this._reg('L|py', l.positionY);
      this._reg('L|pz', l.positionZ);
      this._reg('L|fx', l.forwardX);
      this._reg('L|fy', l.forwardY);
      this._reg('L|fz', l.forwardZ);
      this._reg('L|ux', l.upX);
      this._reg('L|uy', l.upY);
      this._reg('L|uz', l.upZ);
      return;
    }
    const st = { p: [0, 0, 0], f: [0, 0, -1], u: [0, 1, 0] };
    const setP = (i) => (v) => {
      st.p[i] = v;
      try {
        l.setPosition(st.p[0], st.p[1], st.p[2]);
      } catch (e) {
        /* unsupported */
      }
    };
    const setO = (arr, i) => (v) => {
      st[arr][i] = v;
      try {
        l.setOrientation(st.f[0], st.f[1], st.f[2], st.u[0], st.u[1], st.u[2]);
      } catch (e) {
        /* unsupported */
      }
    };
    this._reg('L|px', setP(0));
    this._reg('L|py', setP(1));
    this._reg('L|pz', setP(2));
    this._reg('L|fx', setO('f', 0));
    this._reg('L|fy', setO('f', 1));
    this._reg('L|fz', setO('f', 2));
    this._reg('L|ux', setO('u', 0));
    this._reg('L|uy', setO('u', 1));
    this._reg('L|uz', setO('u', 2));
  }

  /** Real-time parameter update with smoothing (setTargetAtTime). */
  set(key, value, when, tau = 0.05) {
    if (!Number.isFinite(value)) return;
    const p = this.pmap.get(key);
    if (!p) return;
    const last = this.last.get(key);
    if (last !== undefined && Math.abs(value - last) <= 1e-5 * Math.max(Math.abs(value), Math.abs(last)) + 1e-9) return;
    this.last.set(key, value);
    const list = Array.isArray(p) ? p : [p];
    for (const q of list) {
      if (typeof q === 'function') {
        q(value);
        continue;
      }
      try {
        if (last === undefined) q.setValueAtTime(value, when);
        else q.setTargetAtTime(value, when, tau);
      } catch (e) {
        try {
          q.value = value;
        } catch (e2) {
          /* ignore */
        }
      }
    }
  }

  /** Offline automation: a value curve sampled every dt from t0. */
  curve(key, arr, t0, dt) {
    const p = this.pmap.get(key);
    if (!p || !arr.length) return;
    let prev = 0;
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) arr[i] = prev;
      prev = arr[i];
    }
    let constant = true;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] !== arr[0]) {
        constant = false;
        break;
      }
    }
    const list = Array.isArray(p) ? p : [p];
    for (const q of list) {
      if (typeof q === 'function') {
        q(arr[arr.length - 1]);
        continue;
      }
      try {
        if (constant || arr.length < 2) q.setValueAtTime(arr[0], t0);
        else q.setValueCurveAtTime(arr, t0, (arr.length - 1) * dt);
      } catch (e) {
        try {
          q.setValueAtTime(arr[arr.length - 1], t0);
        } catch (e2) {
          /* ignore */
        }
      }
    }
  }

  _panModel(spec) {
    return this.renderMode === 'binaural' && spec.hrtf !== false ? 'HRTF' : 'equalpower';
  }

  _panner(spec) {
    const p = this.ctx.createPanner();
    try {
      p.panningModel = this._panModel(spec);
      p.distanceModel = 'inverse';
      p.refDistance = clamp(num(spec.ref, 1), 0.05, 1e4);
      p.rolloffFactor = Math.max(0, num(spec.rolloff, 1));
      p.maxDistance = 1e6;
      p.coneInnerAngle = 360;
      p.coneOuterAngle = 360;
      p.coneOuterGain = 1;
    } catch (e) {
      /* keep defaults */
    }
    return p;
  }

  _regPos(kx, ky, kz, p) {
    if (p.positionX) {
      this._reg(kx, p.positionX);
      this._reg(ky, p.positionY);
      this._reg(kz, p.positionZ);
      return;
    }
    const s = [0, 0, 0];
    const f = (i) => (v) => {
      s[i] = v;
      try {
        p.setPosition(s[0], s[1], s[2]);
      } catch (e) {
        /* ignore */
      }
    };
    this._reg(kx, f(0));
    this._reg(ky, f(1));
    this._reg(kz, f(2));
  }

  static _setPos(p, w) {
    if (p.positionX) {
      p.positionX.value = w[0];
      p.positionY.value = w[1];
      p.positionZ.value = w[2];
    } else {
      try {
        p.setPosition(w[0], w[1], w[2]);
      } catch (e) {
        /* ignore */
      }
    }
  }

  addVoice(spec, init, when, L) {
    const ctx = this.ctx;
    let node;
    const rain = spec.proc === 'rain';
    try {
      node = new AudioWorkletNode(ctx, PROC[spec.proc], {
        numberOfInputs: rain ? 1 : 0,
        numberOfOutputs: 1,
        outputChannelCount: [spec.channels],
        processorOptions: rain ? { ...spec.config, bus: this.busId, stats: this.statsOn } : spec.config,
        parameterData: init,
      });
      if (rain && this.bus) this.bus.connect(node);
    } catch (e) {
      if (!this._warned) {
        this._warned = true;
        console.warn('[WeatherEngine] could not create voice', spec.key, e);
      }
      return null;
    }
    const fade = this._gain(this.offline ? 1 : 0);
    const bus = (spec.route === 'int' ? this.int : this.ext)[spec.stem] || this.ext.objects;
    const v = { spec, node, fade, panners: [], extra: [], keys: keyList(spec.keys) };
    const K = spec.keys;
    if (spec.pan === 'quad') {
      const sp = ctx.createChannelSplitter(spec.channels);
      node.connect(sp);
      v.extra.push(sp);
      for (let i = 0; i < Math.min(4, spec.channels); i++) {
        const p = this._panner(spec);
        const q = spec.quad[i];
        SceneGraph._setPos(p, [L.pos[0] + q[0], L.pos[2] + q[2], -(L.pos[1] + q[1])]);
        sp.connect(p, i);
        p.connect(fade);
        this._regPos(K.qx[i], K.qy[i], K.qz[i], p);
        v.panners.push(p);
      }
      fade.connect(bus);
    } else if (spec.pan === 'panner') {
      const p = this._panner(spec);
      SceneGraph._setPos(p, web(specPos(spec, L)));
      node.connect(fade);
      fade.connect(p);
      p.connect(bus);
      this._regPos(K.px, K.py, K.pz, p);
      v.panners.push(p);
    } else if (spec.pan === 'stereo') {
      node.connect(fade);
      let sp = null;
      try {
        sp = ctx.createStereoPanner();
      } catch (e) {
        sp = null;
      }
      if (sp) {
        fade.connect(sp);
        sp.connect(bus);
        this._reg(K.pan, sp.pan);
        v.extra.push(sp);
      } else fade.connect(bus);
    } else {
      node.connect(fade);
      fade.connect(bus);
    }
    for (const n of PARAM_NAMES[spec.proc]) {
      const prm = node.parameters && node.parameters.get(n);
      if (prm) {
        this._reg(K[n], prm);
        if (init && Number.isFinite(init[n])) this.last.set(K[n], init[n]);
      }
    }
    if (!this.offline) {
      fade.gain.setValueAtTime(0, when);
      fade.gain.setTargetAtTime(1, when, 0.08);
      if (spec.proc === 'rain' && this.onStats) {
        node.port.onmessage = (e) => {
          const d = e.data;
          if (d && d.type === 'stats') this.onStats(spec, num(d.ratePerSec, 0));
        };
      }
    }
    node.onprocessorerror = () => console.warn('[WeatherEngine] processor error in', spec.key);
    this.voices.set(spec.key, v);
    return v;
  }

  removeVoice(key, when) {
    const v = this.voices.get(key);
    if (!v) return;
    this.voices.delete(key);
    for (const k of v.keys) {
      this.pmap.delete(k);
      this.last.delete(k);
    }
    const kill = () => {
      try {
        v.node.port.onmessage = null;
        v.node.disconnect();
        if (this.bus && v.spec.proc === 'rain') this.bus.disconnect(v.node);
      } catch (e) {
        /* ignore */
      }
      for (const n of [...v.panners, ...v.extra, v.fade]) {
        try {
          n.disconnect();
        } catch (e) {
          /* ignore */
        }
      }
    };
    try {
      v.fade.gain.cancelScheduledValues(when);
      v.fade.gain.setTargetAtTime(0, when, this.offline ? 0.02 : 0.06);
    } catch (e) {
      /* ignore */
    }
    if (this.offline) this.dead.push(kill);
    else setTimeout(kill, 600);
  }

  /** Turn the rain surfaces' impact-rate reports (4 per second each) on or off. */
  setStats(on) {
    const want = !!(on && this.onStats);
    if (want === this.statsOn) return;
    this.statsOn = want;
    for (const v of this.voices.values()) {
      if (v.spec.proc !== 'rain') continue;
      try {
        v.node.port.postMessage({ type: 'stats', on: want });
      } catch (e) {
        /* ignore */
      }
    }
  }

  post(key, msg) {
    const v = this.voices.get(key);
    if (!v) return;
    try {
      v.node.port.postMessage(msg);
    } catch (e) {
      /* ignore */
    }
  }

  updatePanners(key, spec) {
    const v = this.voices.get(key);
    if (!v) return;
    for (const p of v.panners) {
      try {
        p.refDistance = clamp(num(spec.ref, 1), 0.05, 1e4);
        p.rolloffFactor = Math.max(0, num(spec.rolloff, 1));
        const m = this._panModel(spec);
        if (p.panningModel !== m) p.panningModel = m;
      } catch (e) {
        /* ignore */
      }
    }
  }

  setRenderMode(mode) {
    this.renderMode = mode === 'stereo' ? 'stereo' : 'binaural';
    for (const v of this.voices.values()) {
      for (const p of v.panners) {
        try {
          p.panningModel = this._panModel(v.spec);
        } catch (e) {
          /* ignore */
        }
      }
    }
    for (const t of this.thunders) {
      try {
        t.panner.panningModel = this.renderMode === 'binaural' ? 'HRTF' : 'equalpower';
      } catch (e) {
        /* ignore */
      }
    }
  }

  setReverb(spec, seed, when) {
    if (!this.reverb) return;
    try {
      this.reverb.setSpec(spec, seed, { when, offline: this.offline, fade: 0.8 });
    } catch (e) {
      /* ignore */
    }
  }

  /** Play a synthesized roll: direct part through a panner at the channel, rumble unpanned. */
  playThunder(bufs, syn, when, posWorld, gain) {
    const ctx = this.ctx;
    const now = ctx.currentTime;
    let start = when;
    let offset = 0;
    if (start < now + 0.005) {
      offset = now + 0.005 - start;
      start = now + 0.005;
    }
    if (offset >= bufs.direct.duration - 0.05) return false;
    const g = this._gain(gain);
    const panner = this._panner({ ref: 1, rolloff: 0, hrtf: true });
    SceneGraph._setPos(panner, web(posWorld));
    const s1 = ctx.createBufferSource();
    s1.buffer = bufs.direct;
    s1.connect(g);
    g.connect(panner);
    panner.connect(this.ext.thunder);
    const g2 = this._gain(gain);
    const s2 = ctx.createBufferSource();
    s2.buffer = bufs.rumble;
    s2.connect(g2);
    g2.connect(this.ext.thunder);
    const rec = { panner, nodes: [s1, s2, g, g2, panner], nearestM: syn.nearestM };
    this.thunders.add(rec);
    const done = () => {
      if (!this.thunders.has(rec)) return;
      this.thunders.delete(rec);
      for (const n of rec.nodes) {
        try {
          n.disconnect();
        } catch (e) {
          /* ignore */
        }
      }
    };
    s1.onended = done;
    try {
      s1.start(start, offset);
      s2.start(start, Math.min(offset, Math.max(0, bufs.rumble.duration - 0.01)));
    } catch (e) {
      done();
      return false;
    }
    return true;
  }

  dispose() {
    for (const key of [...this.voices.keys()]) {
      const v = this.voices.get(key);
      try {
        v.node.port.onmessage = null;
        v.node.disconnect();
        if (this.bus && v.spec.proc === 'rain') this.bus.disconnect(v.node);
      } catch (e) {
        /* ignore */
      }
      for (const n of [...v.panners, ...v.extra, v.fade]) {
        try {
          n.disconnect();
        } catch (e) {
          /* ignore */
        }
      }
    }
    this.voices.clear();
    for (const k of this.dead) k();
    this.dead = [];
    for (const t of this.thunders) {
      for (const n of t.nodes) {
        try {
          n.disconnect();
        } catch (e) {
          /* ignore */
        }
      }
    }
    this.thunders.clear();
    if (this.bus) {
      try {
        this.bus.port.postMessage({ type: 'dispose' });
        this.bus.disconnect();
      } catch (e) {
        /* ignore */
      }
      this.bus = null;
    }
    if (this.reverb) this.reverb.disconnect();
    const all = [this.extBus, this.intBus, this.post, this.pre, this.master, this.limiter, this.clip, this.out, this.analyser, ...(this.tlNodes || [])];
    for (const s of STEMS) all.push(this.ext[s], this.int[s]);
    if (this.meter) all.push(this.meter.kL, this.meter.kR, this.meter.pL, this.meter.pR, ...this.meter.nodes);
    for (const n of all) {
      if (!n) continue;
      try {
        n.disconnect();
      } catch (e) {
        /* ignore */
      }
    }
    this.pmap.clear();
    this.last.clear();
  }
}

/** Diff a new plan against the voices alive in a graph. */
function applyPlan(graph, prev, plan, ctl, model, env, when) {
  for (const [key, spec] of plan.voices) {
    const v = graph.voices.get(key);
    if (!v) {
      graph.addVoice(spec, initParams(spec, ctl, model, env), when, env.listener);
      continue;
    }
    const old = v.spec;
    if (old.proc !== spec.proc || old.channels !== spec.channels || old.route !== spec.route || old.stem !== spec.stem || old.pan !== spec.pan) {
      graph.removeVoice(key, when);
      graph.addVoice(spec, initParams(spec, ctl, model, env), when, env.listener);
      continue;
    }
    if (old.structKey !== spec.structKey) {
      graph.post(key, { type: 'config', config: spec.config });
    } else if (spec.proc === 'rain') {
      if (old.config.exposure !== spec.config.exposure) graph.post(key, { type: 'exposure', value: spec.config.exposure });
      if (old.config.area !== spec.config.area) graph.post(key, { type: 'area', value: spec.config.area });
    }
    if (old.ref !== spec.ref || old.rolloff !== spec.rolloff || old.hrtf !== spec.hrtf) graph.updatePanners(key, spec);
    v.spec = spec;
  }
  for (const key of [...graph.voices.keys()]) if (!plan.voices.has(key)) graph.removeVoice(key, when);
  if (!prev || prev.reverbKey !== plan.reverbKey) graph.setReverb(plan.reverbSpec, env.seedKey, when);
}

// ---------------------------------------------------------------------------
// Worklet modules and level calibration
// ---------------------------------------------------------------------------

const LOADED = new WeakMap();

function moduleUrls(base) {
  return MODULES.map((m) => new URL(m, base || import.meta.url).href);
}

async function loadModules(ctx, base) {
  let p = LOADED.get(ctx);
  if (!p) {
    if (!ctx.audioWorklet) throw new Error('AudioWorklet is not supported in this browser.');
    p = Promise.all(moduleUrls(base).map((u) => ctx.audioWorklet.addModule(u)));
    LOADED.set(ctx, p);
  }
  return p;
}

const FALLBACK_CAL = { rain: 260, wind: 0.9, msPerM2: 1.1e-8, bedUnit: 0.18, drizzleUnit: 0.14, eRef: E_REF, fieldRef: FIELD_REF, ok: false };
const CAL_CACHE = new Map();

/**
 * Measure the worklets' output per physical unit once per sample rate (a 1.25 s offline render
 * of a reference rain patch, the wind 'air' layer and the bed), so the near rain field at 5 mm/h
 * lands at TARGET_RAIN_DB and every physical relationship (rate, area, distance) is preserved.
 */
function calibrate(sr, base) {
  const key = Math.round(sr);
  if (CAL_CACHE.has(key)) return CAL_CACHE.get(key);
  const p = (async () => {
    const cal = { ...FALLBACK_CAL };
    const OAC = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
    if (!OAC || typeof AudioWorkletNode === 'undefined') return cal;
    try {
      const len = Math.round(sr * 1.25);
      let oc;
      try {
        oc = new OAC({ numberOfChannels: 4, length: len, sampleRate: sr });
      } catch (e) {
        oc = new OAC(4, len, sr);
      }
      if (!oc.audioWorklet) return cal;
      await loadModules(oc, base);
      const wx = { wetness: 0.5, puddle: 0, liquid: 0 };
      REF_PRECIP.classes.forEach((c, i) => {
        wx[RN[i]] = c.flux;
        wx[DN[i]] = c.d;
      });
      const mk = (proc, ch, po, data, inputs = 0) => new AudioWorkletNode(oc, PROC[proc], { numberOfInputs: inputs, numberOfOutputs: 1, outputChannelCount: [ch], processorOptions: po, parameterData: data });
      const mats = [buildMaterial('concrete', { seed: 1, salt: 'cal', fraction: 0.5, maxModes: 12 }), buildMaterial('grass', { seed: 1, salt: 'cal', fraction: 0.5, maxModes: 12 })];
      const bus = mk('bus', 1, { bus: 'cal' }, wx);
      const rain = mk('rain', 1, { id: 'cal-rain', kind: 'rain', area: CAL_AREA, exposure: 1, maxEventsPerSec: 3000, maxModes: 12, coxDepth: 0, seed: 12345, sharedSeed: 54321, materials: mats, bus: 'cal', stats: false }, { intensity: 1, wind: 0, gain: 1 }, 1);
      bus.connect(rain);
      const air = mk('wind', 2, { id: 'cal-air', mode: 'air', seed: 777, windScale: 1 }, { wind: 8, gain: 1, aux: 0 });
      const bed = mk('bed', 2, { seed: 99 }, { level: 1, cutoff: 8000, drizzle: 0, density: 1, intensity: 1, gain: 1, depth: 0 });
      const drz = mk('bed', 2, { seed: 98 }, { level: 0, cutoff: 8000, drizzle: 1, density: 1, intensity: 1, gain: 1, depth: 0 });
      const merger = oc.createChannelMerger(4);
      rain.connect(merger, 0, 0);
      [air, bed, drz].forEach((n, i) => {
        const sp = oc.createChannelSplitter(2);
        n.connect(sp);
        sp.connect(merger, 0, i + 1);
      });
      merger.connect(oc.destination);
      const buf = await oc.startRendering();
      const rms = (ch) => {
        const d = buf.getChannelData(ch);
        let s = 0;
        let n = 0;
        for (let i = Math.floor(0.25 * sr); i < d.length; i++) {
          s += d[i] * d[i];
          n++;
        }
        return Math.sqrt(s / Math.max(1, n));
      };
      const r = [rms(0), rms(1), rms(2), rms(3)];
      if (r[0] > 1e-9) {
        cal.msPerM2 = (r[0] * r[0]) / CAL_AREA;
        cal.rain = clamp(dbToGain(TARGET_RAIN_DB) / Math.sqrt(cal.msPerM2 * FIELD_REF), 0.5, 1e5);
        cal.ok = true;
      }
      if (r[1] > 1e-9) cal.wind = clamp(dbToGain(TARGET_WIND_DB) / r[1], 0.005, 200);
      if (r[2] > 1e-9) cal.bedUnit = r[2];
      if (r[3] > 1e-9) cal.drizzleUnit = r[3];
    } catch (e) {
      console.warn('[WeatherEngine] level calibration failed, using defaults', e);
    }
    return cal;
  })();
  CAL_CACHE.set(key, p);
  return p;
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

/** Encode an AudioBuffer as PCM WAV (24-bit default) with TPDF dither; returns a Blob. */
export function encodeWav(buffer, bitDepth = 24, seed = 1) {
  const ch = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const n = buffer.length;
  const bps = bitDepth === 16 ? 2 : 3;
  const dataBytes = n * ch * bps;
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const ws = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  ws(0, 'RIFF');
  v.setUint32(4, 36 + dataBytes, true);
  ws(8, 'WAVE');
  ws(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, ch, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * ch * bps, true);
  v.setUint16(32, ch * bps, true);
  v.setUint16(34, bps * 8, true);
  ws(36, 'data');
  v.setUint32(40, dataBytes, true);
  const parts = [header];
  const chans = [];
  for (let c = 0; c < ch; c++) chans.push(buffer.getChannelData(c));
  const full = bps === 3 ? 8388607 : 32767;
  let s = (seedKey(seed) | 1) | 0;
  const rnd = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
  const FR = 65536;
  for (let i0 = 0; i0 < n; i0 += FR) {
    const m = Math.min(FR, n - i0);
    const out = new Uint8Array(m * ch * bps);
    let o = 0;
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < ch; c++) {
        let x = chans[c][i0 + i] * full + (rnd() - rnd());
        if (!(x === x)) x = 0;
        let q = Math.round(x);
        if (q > full) q = full;
        else if (q < -full - 1) q = -full - 1;
        out[o++] = q & 255;
        out[o++] = (q >> 8) & 255;
        if (bps === 3) out[o++] = (q >> 16) & 255;
      }
    }
    parts.push(out);
  }
  return new Blob(parts, { type: 'audio/wav' });
}

// ---------------------------------------------------------------------------
// Offline render job
// ---------------------------------------------------------------------------

class OfflineJob {
  constructor(engine, octx, cal, o) {
    this.ctx = octx;
    this.cal = cal;
    this.o = o;
    this.scene = normalizeScene(cloneScene(engine.scene));
    this.base = { position: engine._listener.position.slice(), headingDeg: engine._listener.headingDeg, pitchDeg: engine._listener.pitchDeg };
    this.model = new ControlModel(this.scene.seed);
    this.graph = new SceneGraph(octx, { offline: true, renderMode: o.renderMode, meter: false });
    this.quality = engine._quality;
    this.mix = { ...engine._mix };
    this.masterDb = engine._masterDb;
    this.dt = 1 / CONTROL_HZ;
    this.plan = null;
    this.sig = '';
    this.anchor = null;
    this.matCache = new Map();
    this.seen = new Set();
    this.cands = new Map();
    this.pending = null;
    this.rel = 0;
  }

  _env(L) {
    const sk = seedKey(this.scene.seed);
    return {
      seed: this.scene.seed,
      seedKey: sk,
      sharedKey: (sk ^ hashString('cox')) >>> 0,
      listener: L,
      anchor: this.anchor,
      quality: this.quality,
      evScale: this.quality === 'low' ? 0.5 : this.quality === 'high' ? 1.5 : 1,
      matCache: this.matCache,
      cal: this.cal,
      fieldSum: this.plan ? this.plan.fieldSum : FIELD_REF,
    };
  }

  _progress(f) {
    if (typeof this.o.onProgress === 'function') {
      try {
        this.o.onProgress(clamp(f, 0, 1));
      } catch (e) {
        /* ignore */
      }
    }
  }

  async run() {
    const { dur, startSec } = this.o;
    this.model.compute(this.scene, 0, 0);
    let t = 0;
    while (t + this.dt <= startSec + 1e-9) {
      t += this.dt;
      this.model.compute(this.scene, t, this.dt);
    }
    const canSuspend = typeof this.ctx.suspend === 'function';
    this._chunk(true);
    if (canSuspend) this._scheduleNext();
    else {
      while (this.rel < dur - 1e-9) this._chunk(false);
    }
    this._progress(0.02);
    const buf = await new Promise((resolve, reject) => {
      let settled = false;
      this.ctx.oncomplete = (e) => {
        if (!settled) {
          settled = true;
          resolve(e.renderedBuffer);
        }
      };
      try {
        const r = this.ctx.startRendering();
        if (r && typeof r.then === 'function') {
          r.then(
            (b) => {
              if (!settled) {
                settled = true;
                resolve(b);
              }
            },
            (err) => {
              if (!settled) {
                settled = true;
                reject(err);
              }
            },
          );
        }
      } catch (err) {
        reject(err);
      }
    });
    this.graph.dispose();
    this._progress(1);
    return buf;
  }

  _scheduleNext() {
    const dur = this.o.dur;
    const at = this.rel;
    if (at >= dur - 256 / this.ctx.sampleRate) return;
    let p;
    try {
      p = this.ctx.suspend(at);
    } catch (e) {
      while (this.rel < dur - 1e-9) this._chunk(false);
      return;
    }
    p.then(
      () => {
        try {
          this._chunk(true);
          this._progress(at / dur);
          this._scheduleNext();
        } catch (e) {
          console.error('[WeatherEngine] offline chunk failed', e);
        }
        this.ctx.resume();
      },
      (e) => console.error('[WeatherEngine] offline suspend failed', e),
    );
  }

  _chunk(structural) {
    const dt = this.dt;
    const maxTicks = Math.max(2, Math.round(OFFLINE_CHUNK_SEC * CONTROL_HZ));
    const rel0 = this.rel;
    const dur = this.o.dur;
    const rows = new Map();
    let n = 0;
    for (let i = 0; i < maxTicks; i++) {
      const rel = rel0 + i * dt;
      if (rel > dur + dt) break;
      const t = this.o.startSec + rel;
      let ctl;
      if (i === 0 && this.pending) {
        ctl = this.pending;
        this.pending = null;
      } else ctl = this.model.compute(this.scene, t, rel0 === 0 && i === 0 ? 0 : dt);
      const L = listenerAt(ctl, this.base);
      this.anchor = anchorFor(this.anchor, L, ctl.shelter);
      const sig = planSignature(0, ctl, L, this.anchor, this.quality);
      if (!this.plan || (structural && sig !== this.sig)) {
        if (i > 0) {
          this.pending = ctl;
          break;
        }
        this.sig = sig;
        const S = overlay(this.scene, ctl.tl);
        const env = this._env(L);
        const plan = planScene(S, ctl, env);
        env.fieldSum = plan.fieldSum;
        applyPlan(this.graph, this.plan, plan, ctl, this.model, env, rel);
        this.plan = plan;
      }
      const env = this._env(L);
      const emit = (k, v) => {
        let a = rows.get(k);
        if (!a) {
          a = new Float32Array(maxTicks);
          rows.set(k, a);
        }
        a[i] = v;
      };
      emitParams(this.plan, ctl, this.model, env, emit);
      busParams(ctl, emit);
      masterParams(this.plan, ctl, this.cal, this.mix, this.masterDb, this.ctx.sampleRate, emit);
      listenerParams(L, emit);
      this._flashes(ctl, L, t, t + dt);
      n = i + 1;
    }
    for (const [k, a] of rows) this.graph.curve(k, a.subarray(0, n), rel0, dt);
    this.rel = rel0 + Math.max(1, n) * dt;
  }

  _flashes(ctl, L, t0, t1) {
    const S = this.plan ? this.plan.S : this.scene;
    for (const src of thunderSources(S, ctl, L, t0)) {
      const p = src.ratePerMin / FLASH_CEILING_PER_MIN;
      if (!(p > 0)) continue;
      for (let cell = Math.floor(t0 / FLASH_CELL_SEC); cell <= Math.floor(t1 / FLASH_CELL_SEC); cell++) {
        const ck = `${src.id}#${cell}`;
        let cands = this.cands.get(ck);
        if (!cands) {
          cands = flashCandidates(this.scene.seed, src.id, cell);
          if (this.cands.size > 64) this.cands.clear();
          this.cands.set(ck, cands);
        }
        for (const c of cands) {
          if (c.t < t0 || c.t >= t1 || c.u >= p || this.seen.has(c.key)) continue;
          this.seen.add(c.key);
          const fl = prepareFlash(this.scene.seed, src, c.key, L.ear, ctl.tempC);
          if (fl.nearestM >= MAX_AUDIBLE_M) continue;
          const syn = synthesizeThunder(fl.chan, L.ear, {
            seed: this.scene.seed,
            key: fl.key,
            tempC: ctl.tempC,
            humidityPct: ctl.rh,
            intensity: fl.intensity,
            lapseKPerKm: ctl.lapse,
            windMps: ctl.U10,
            windFromDeg: ctl.fromDeg,
            sampleRate: synthRateFor(fl.nearestM, this.ctx.sampleRate),
          });
          if (!syn) continue;
          const bufs = createThunderBuffers(this.ctx, syn);
          this.graph.playThunder(bufs, syn, c.t - this.o.startSec + syn.offsetSec, syn.centroid, syn.gain);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public engine
// ---------------------------------------------------------------------------

/**
 * Real-time weather & environment sound engine for the browser.
 *
 * Events (on/off): 'flash' {id, t, channel, branches, kind, distanceKm, bearingDeg, delaySec,
 * brightness, limited, stormId}, 'thunder' {id, t, distanceKm, bearingDeg, stormId},
 * 'gust' {speedMps, fromDeg, meanMps, gustFactor}, 'impact-stats' {objectId, ratePerSec},
 * 'meter' {lufsMomentary, peakDb, t}, 'caption' {text, t}, 'state' {state}.
 * Times `t` are seconds on the scene timeline.
 */
export class WeatherEngine {
  constructor(opts = {}) {
    const o = obj(opts);
    this.opts = { latencyHint: 'playback', ...o };
    this._quality = ['low', 'balanced', 'high'].includes(o.quality) ? o.quality : WeatherEngine._autoQuality();
    this.scene = normalizeScene(o.scene);
    this._events = new Map();
    this._ctx = null;
    this._graph = null;
    this._model = new ControlModel(this.scene.seed);
    this._plan = null;
    this._planSig = '';
    this._ver = 1;
    this._lastReplanMs = -1e9;
    this._renderMode = o.renderMode === 'stereo' || o.renderMode === 'binaural' ? o.renderMode : null;
    this._masterDb = num(o.masterGainDb, 0);
    this._mix = { rain: 0, objects: 0, wind: 0, thunder: 0, bed: 0 };
    this._safety = { maxFlashesPerSec: 3 };
    this._listener = { position: this.scene.listener.position.slice(), headingDeg: this.scene.listener.heading_deg, pitchDeg: 0 };
    this._anchor = null;
    this._t = 0;
    this._lastCtxT = 0;
    this._flashCheckT = 0;
    this._flashSeen = new Set();
    this._cands = new Map();
    this._flashTimes = [];
    this._lastFlashMs = -1e9;
    this._timers = new Set();
    this._tickTimer = null;
    this._meterTimer = null;
    this._stats = new Map();
    this._statsAt = new Map();
    this._gustAcc = 0;
    this._lastGustCaption = -1e9;
    this._lastType = this.scene.weather.precipitation.type;
    this._matCache = new Map();
    this._manualCount = 0;
    this._cal = { ...FALLBACK_CAL };
    this._starting = null;
    this._lastCtl = null;
    this._lastL = null;
    this._tickErr = false;
    // Live voice budget; the load governor in _tickNow lowers it if the audio clock falls behind.
    this._voiceCap = this._quality === 'low' ? 8 : this._quality === 'high' ? 24 : 12;
    this._loadWin = null;
  }

  static _autoQuality() {
    try {
      const nav = globalThis.navigator;
      const cores = nav && nav.hardwareConcurrency ? nav.hardwareConcurrency : 8;
      const mobile = nav && /Mobi|Android|iPhone|iPad/i.test(nav.userAgent || '');
      if (mobile || cores <= 2) return 'low';
    } catch (e) {
      /* ignore */
    }
    return 'balanced';
  }

  /** True while the AudioContext is rendering. */
  get running() {
    return !!this._ctx && this._ctx.state === 'running';
  }

  /** The live AudioContext (null before start()). */
  get context() {
    return this._ctx;
  }

  /** AnalyserNode on the master output (null before start()). */
  get analyser() {
    return this._graph ? this._graph.analyser : null;
  }

  // ---------------- events

  on(event, cb) {
    if (typeof cb !== 'function') return () => {};
    if (!this._events.has(event)) this._events.set(event, new Set());
    this._events.get(event).add(cb);
    if (event === 'impact-stats' && this._graph) this._graph.setStats(true);
    return () => this.off(event, cb);
  }

  off(event, cb) {
    const s = this._events.get(event);
    if (!s) return;
    if (cb) s.delete(cb);
    else s.clear();
    if (event === 'impact-stats' && this._graph) this._graph.setStats(this._has('impact-stats'));
  }

  _emit(event, payload) {
    const s = this._events.get(event);
    if (!s || !s.size) return;
    for (const cb of [...s]) {
      try {
        cb(payload);
      } catch (e) {
        console.error(`[WeatherEngine] '${event}' listener failed`, e);
      }
    }
  }

  _has(event) {
    const s = this._events.get(event);
    return !!(s && s.size);
  }

  // ---------------- lifecycle

  /** Create the AudioContext (call from a user gesture), load the worklets and start rendering. */
  async start() {
    if (this._ctx && this._ctx.state !== 'closed') {
      if (this._ctx.state === 'suspended') {
        try {
          await this._ctx.resume();
        } catch (e) {
          /* needs a gesture */
        }
      }
      return;
    }
    if (this._starting) return this._starting;
    this._starting = (async () => {
      const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!AC) throw new Error('Web Audio is not available in this browser.');
      // A context created inside the user's gesture (see unlock.js) is the only kind Safari/iOS will play.
      let ctx = this.opts.context && this.opts.context.state !== 'closed' ? this.opts.context : null;
      const ctxOpts = { latencyHint: this.opts.latencyHint };
      if (num(this.opts.sampleRate, 0) > 0) ctxOpts.sampleRate = num(this.opts.sampleRate, 48000);
      if (!ctx) {
        try {
          ctx = new AC(ctxOpts);
        } catch (e) {
          ctx = new AC();
        }
      }
      if (!ctx.audioWorklet || typeof AudioWorkletNode === 'undefined') {
        try {
          await ctx.close();
        } catch (e) {
          /* ignore */
        }
        throw new Error('AudioWorklet is not supported in this browser.');
      }
      const resumeP = ctx.state === 'suspended' ? ctx.resume().catch(() => {}) : null;
      try {
        await loadModules(ctx, this.opts.moduleBase);
      } catch (e) {
        try {
          await ctx.close();
        } catch (e2) {
          /* ignore */
        }
        throw e;
      }
      try {
        this._cal = await calibrate(ctx.sampleRate, this.opts.moduleBase);
      } catch (e) {
        this._cal = { ...FALLBACK_CAL };
      }
      if (resumeP) await resumeP;
      this._ctx = ctx;
      this._graph = new SceneGraph(ctx, {
        renderMode: this._renderModeNow(),
        onStats: (spec, rate) => this._onStats(spec, rate),
        stats: this._has('impact-stats'),
      });
      this._lastCtxT = ctx.currentTime;
      this._plan = null;
      this._planSig = '';
      this._flashCheckT = this._t;
      this._flashSeen.clear();
      this._tickErr = false;
      ctx.onstatechange = () => this._emit('state', { state: ctx.state });
      this._tickTimer = setInterval(() => this._tickNow(), 1000 / CONTROL_HZ);
      this._meterTimer = setInterval(() => this._meterNow(), 100);
    })();
    try {
      await this._starting;
    } finally {
      this._starting = null;
    }
  }

  /** Stop rendering and release the AudioContext; start() may be called again. */
  stop() {
    if (this._tickTimer) clearInterval(this._tickTimer);
    if (this._meterTimer) clearInterval(this._meterTimer);
    this._tickTimer = null;
    this._meterTimer = null;
    for (const t of this._timers) clearTimeout(t);
    this._timers.clear();
    if (this._graph) {
      try {
        this._graph.dispose();
      } catch (e) {
        /* ignore */
      }
      this._graph = null;
    }
    if (this._ctx) {
      const c = this._ctx;
      this._ctx = null;
      try {
        const p = c.close();
        if (p && p.catch) p.catch(() => {});
      } catch (e) {
        /* ignore */
      }
    }
    this._plan = null;
    this._planSig = '';
    this._stats.clear();
  }

  /** If the audio clock runs slower than real time (the graph is too heavy), shed the farthest voices. */
  _governLoad(ctx, now) {
    if (ctx.state !== 'running') {
      this._loadWin = null;
      return;
    }
    const t = nowMs();
    const w = this._loadWin;
    // Skip the first seconds after start or a rebuild: worklet JIT warm-up runs slow by itself.
    if (!w) {
      this._loadWin = { t: t + 8000, c: now };
      return;
    }
    if (t < w.t) {
      w.c = now;
      return;
    }
    if (t - w.t < 3000) return;
    const speed = (now - w.c) / ((t - w.t) / 1000);
    this._loadWin = { t, c: now };
    if (speed < 0.93 && this._voiceCap > 6) {
      this._voiceCap = Math.max(6, Math.round(this._voiceCap * 0.7));
      this._ver++;
      this._loadWin = { t: t + 8000, c: now };
      this._emit('load', { speed, voiceCap: this._voiceCap });
    }
  }

  async suspend() {
    if (this._ctx && this._ctx.state === 'running') {
      try {
        await this._ctx.suspend();
      } catch (e) {
        /* ignore */
      }
    }
  }

  async resume() {
    if (!this._ctx) return this.start();
    if (this._ctx.state === 'suspended') {
      try {
        await this._ctx.resume();
      } catch (e) {
        /* needs a gesture */
      }
    }
    this._lastCtxT = this._ctx.currentTime;
  }

  // ---------------- scene

  /** Replace the scene; numeric weather changes glide, objects are diffed into voices. */
  setScene(scene) {
    const next = normalizeScene(scene);
    const prev = this.scene;
    for (const p of SMOOTH_PATHS) {
      const a = getScenePath(prev, p);
      const b = getScenePath(next, p);
      if (typeof a === 'number' && typeof b === 'number' && a !== b) {
        const live = this._model.live.get(p);
        this._model.setTarget(p, live ? live.v : a, b, rampFor(p, next));
      }
    }
    if (String(prev.seed) !== String(next.seed)) {
      const wet = this._model.wet;
      const live = this._model.live;
      this._model = new ControlModel(next.seed);
      this._model.wet = wet;
      this._model.live = live;
      this._model.primed = true;
      this._matCache.clear();
    }
    const lp = prev.listener;
    const ln = next.listener;
    if (!sameVec(lp.position, ln.position) || lp.heading_deg !== ln.heading_deg) {
      this._listener = { position: ln.position.slice(), headingDeg: ln.heading_deg, pitchDeg: this._listener.pitchDeg };
    }
    this.scene = next;
    if (lp.render !== ln.render && !this._renderMode && this._graph) this._graph.setRenderMode(ln.render);
    this._ver++;
  }

  /**
   * Set one scene parameter by dotted path ('weather.precipitation.rate_mm_h',
   * 'place.objects.<id>.position', 'mix.thunder' in dB, ...). Numbers glide over rampSec.
   */
  setParam(path, value, rampSec = 0.5) {
    const p = String(path || '');
    if (!p) return false;
    if (p.startsWith('mix.')) {
      const s = p.slice(4).replace(/_db$/, '');
      if (s === 'master') this._masterDb = clamp(num(value, 0), -80, 24);
      else if (STEMS.includes(s)) this._mix[s] = clamp(num(value, 0), -80, 24);
      else return false;
      return true;
    }
    if (p === 'listener.position') {
      this.setListener({ position: value });
      return true;
    }
    if (p === 'listener.heading_deg') {
      this.setListener({ headingDeg: value });
      return true;
    }
    if (p === 'listener.render') {
      setScenePath(this.scene, p, value === 'stereo' ? 'stereo' : 'binaural');
      if (!this._renderMode && this._graph) this._graph.setRenderMode(this.scene.listener.render);
      return true;
    }
    const cur = getScenePath(this.scene, p);
    let v = value;
    if (typeof cur === 'number' && typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) v = Number(v);
    if (typeof cur === 'boolean' && typeof v === 'string') v = v === 'true';
    if (SMOOTHABLE.has(p) && typeof v === 'number' && Number.isFinite(v)) {
      const live = this._model.live.get(p);
      const from = live ? live.v : typeof cur === 'number' ? cur : v;
      setScenePath(this.scene, p, v);
      this.scene = normalizeScene(this.scene);
      const nv = getScenePath(this.scene, p);
      this._model.setTarget(p, from, typeof nv === 'number' ? nv : v, num(rampSec, 0.5));
      return true;
    }
    if (p === 'seed') {
      this.setScene({ ...this.scene, seed: v });
      return true;
    }
    const ok = setScenePath(this.scene, p, v);
    if (!ok) return false;
    this.scene = normalizeScene(this.scene);
    this._ver++;
    return true;
  }

  /** Move or turn the listener: position [x, y, z] (feet, world metres), heading/pitch in degrees. */
  setListener(l = {}) {
    const o = obj(l);
    if (Array.isArray(o.position)) this._listener.position = vec3(o.position, this._listener.position);
    if (o.headingDeg !== undefined && Number.isFinite(Number(o.headingDeg))) this._listener.headingDeg = wrap360(Number(o.headingDeg));
    if (o.pitchDeg !== undefined && Number.isFinite(Number(o.pitchDeg))) this._listener.pitchDeg = clamp(Number(o.pitchDeg), -89, 89);
    if (o.heightM !== undefined && Number.isFinite(Number(o.heightM))) this.setParam('listener.height_m', Number(o.heightM), 0.3);
    this.scene.listener.position = this._listener.position.slice();
    this.scene.listener.heading_deg = this._listener.headingDeg;
    if (this._graph && this._ctx && this._lastCtl) {
      const L = listenerAt(this._lastCtl, this._listener);
      const now = this._ctx.currentTime;
      listenerParams(L, (k, v) => this._graph.set(k, v, now, 0.03));
    }
  }

  /** Add or replace one scene object (matched by id). */
  upsertObject(object) {
    if (!isObj(object)) return false;
    const list = this.scene.place.objects;
    const id = object.id !== undefined && object.id !== null ? String(object.id) : `obj${list.length}`;
    const idx = list.findIndex((o) => o.id === id);
    const used = new Set(list.map((o) => o.id));
    used.delete(id);
    const n = normalizeObject({ ...object, id }, idx < 0 ? list.length : idx, used);
    if (!n) return false;
    if (idx < 0) list.push(n);
    else list[idx] = n;
    this._ver++;
    return true;
  }

  /** Remove one scene object by id. */
  removeObject(id) {
    const k = String(id);
    const list = this.scene.place.objects;
    const n = list.length;
    this.scene.place.objects = list.filter((o) => o.id !== k);
    if (this.scene.place.objects.length === n) return false;
    this._ver++;
    return true;
  }

  /** Jump the scene timeline (automation, storm arc, storm tracks) to tSec. */
  setTime(tSec) {
    this._t = Math.max(0, num(tSec, 0));
    this._flashCheckT = this._t;
    this._flashSeen.clear();
  }

  setMasterGainDb(db) {
    this._masterDb = clamp(num(db, 0), -80, 24);
  }

  /** 'binaural' (HRTF panners, headphones) or 'stereo' (equal-power, speakers). */
  setRenderMode(mode) {
    this._renderMode = mode === 'stereo' ? 'stereo' : 'binaural';
    if (this._graph) this._graph.setRenderMode(this._renderMode);
  }

  /** Photosensitive safety: at most maxFlashesPerSec full-brightness flashes per second (<= 3). */
  setVisualSafety(o = {}) {
    this._safety.maxFlashesPerSec = clamp(num(obj(o).maxFlashesPerSec, 3), 0, 3);
  }

  _renderModeNow() {
    return this._renderMode || this.scene.listener.render || 'binaural';
  }

  // ---------------- control loop

  _env(L) {
    const sk = seedKey(this.scene.seed);
    return {
      seed: this.scene.seed,
      seedKey: sk,
      sharedKey: (sk ^ hashString('cox')) >>> 0,
      listener: L,
      anchor: this._anchor,
      quality: this._quality,
      evScale: this._quality === 'low' ? 0.5 : this._quality === 'high' ? 1.5 : 1,
      matCache: this._matCache,
      cal: this._cal,
      fieldSum: this._plan ? this._plan.fieldSum : FIELD_REF,
    };
  }

  _applyLoop() {
    const st = this.scene.studio;
    if (!st || !st.loopOn || !Array.isArray(st.loop)) return;
    const a = num(st.loop[0], 0);
    const b = num(st.loop[1], 0);
    if (b - a > 1 && this._t >= b) this.setTime(a + ((this._t - a) % (b - a)));
  }

  _tickNow() {
    const ctx = this._ctx;
    const g = this._graph;
    if (!ctx || !g) return;
    try {
      const now = ctx.currentTime;
      this._governLoad(ctx, now);
      let dt = now - this._lastCtxT;
      this._lastCtxT = now;
      if (!(dt > 0) || ctx.state !== 'running') dt = 0;
      dt = Math.min(dt, 0.5);
      this._t += dt;
      this._applyLoop();
      const t = this._t;
      const ctl = this._model.compute(this.scene, t, dt);
      const L = listenerAt(ctl, this._listener);
      this._lastCtl = ctl;
      this._lastL = L;
      this._anchor = anchorFor(this._anchor, L, ctl.shelter);
      const sig = planSignature(this._ver, ctl, L, this._anchor, this._quality);
      const ms = nowMs();
      if (!this._plan || (sig !== this._planSig && ms - this._lastReplanMs > 200)) {
        this._planSig = sig;
        this._lastReplanMs = ms;
        const S = overlay(this.scene, ctl.tl);
        const env = this._env(L);
        const plan = planScene(S, ctl, env);
        capVoices(plan, this._voiceCap, env.listener);
        env.fieldSum = plan.fieldSum;
        applyPlan(g, this._plan, plan, ctl, this._model, env, now);
        this._plan = plan;
      }
      const env = this._env(L);
      const emit = (k, v) => g.set(k, v, now, 0.05);
      emitParams(this._plan, ctl, this._model, env, emit);
      busParams(ctl, emit);
      masterParams(this._plan, ctl, this._cal, this._mix, this._masterDb, ctx.sampleRate, emit);
      listenerParams(L, (k, v) => g.set(k, v, now, 0.03));
      this._thunderTick(ctl, L, t, now);
      this._eventsTick(ctl, dt);
    } catch (e) {
      if (!this._tickErr) {
        this._tickErr = true;
        console.error('[WeatherEngine] control tick failed', e);
      }
    }
  }

  _eventsTick(ctl, dt) {
    this._gustAcc += dt;
    if (this._gustAcc >= 0.25) {
      this._gustAcc = 0;
      this._emit('gust', { speedMps: ctl.windRef, fromDeg: ctl.fromDeg, meanMps: ctl.U10, gustFactor: ctl.G });
      if (ctl.gust > 1.3 && ctl.windRef > 10.8 && this._t - this._lastGustCaption > 25) {
        this._lastGustCaption = this._t;
        this._emit('caption', { text: `[Strong gust from the ${compassName(ctl.fromDeg)}]`, t: this._t });
      }
    }
    if (ctl.type !== this._lastType) {
      const a = TYPE_WORDS[this._lastType] || this._lastType;
      const b = TYPE_WORDS[ctl.type] || ctl.type;
      const text = ctl.type === 'none' ? `[The ${a} stops]` : this._lastType === 'none' ? `[${b[0].toUpperCase()}${b.slice(1)} begins]` : `[${a[0].toUpperCase()}${a.slice(1)} turns to ${b}]`;
      this._lastType = ctl.type;
      this._emit('caption', { text, t: this._t });
    }
  }

  _onStats(spec, rate) {
    const id = spec.objectId || 'ground';
    if (String(id).startsWith('~')) return;
    let m = this._stats.get(id);
    if (!m) {
      m = new Map();
      this._stats.set(id, m);
    }
    m.set(spec.key, rate);
    const t = nowMs();
    if (t - (this._statsAt.get(id) || 0) < 230) return;
    this._statsAt.set(id, t);
    let s = 0;
    for (const [k, v] of m) {
      if (!this._graph || !this._graph.voices.has(k)) m.delete(k);
      else s += v;
    }
    this._emit('impact-stats', { objectId: id, ratePerSec: s });
  }

  _meterNow() {
    const m = this._graph && this._graph.meter;
    if (!m || !this.running || !this._has('meter')) return;
    try {
      const msTail = (an) => {
        an.getFloatTimeDomainData(m.bufK);
        let s = 0;
        const n = m.win;
        for (let i = m.bufK.length - n; i < m.bufK.length; i++) s += m.bufK[i] * m.bufK[i];
        return s / n;
      };
      const ms = msTail(m.kL) + msTail(m.kR);
      let pk = 0;
      for (const an of [m.pL, m.pR]) {
        an.getFloatTimeDomainData(m.bufP);
        for (let i = 0; i < m.bufP.length; i++) {
          const a = Math.abs(m.bufP[i]);
          if (a > pk) pk = a;
        }
      }
      this._emit('meter', {
        lufsMomentary: Math.max(-120, -0.691 + 10 * Math.log10(ms + 1e-20)),
        peakDb: Math.max(-120, 20 * Math.log10(pk + 1e-12)),
        t: this._t,
      });
    } catch (e) {
      /* analyser unavailable */
    }
  }

  // ---------------- timers on the audio clock

  _at(ctxTime, fn) {
    const ctx = this._ctx;
    if (!ctx) return;
    let ms = (ctxTime - ctx.currentTime) * 1000;
    try {
      const ts = ctx.getOutputTimestamp ? ctx.getOutputTimestamp() : null;
      if (ts && ts.contextTime > 0 && ts.performanceTime > 0) ms = (ctxTime - ts.contextTime) * 1000 + ts.performanceTime - nowMs();
    } catch (e) {
      /* keep the estimate */
    }
    const id = setTimeout(() => {
      this._timers.delete(id);
      try {
        fn();
      } catch (e) {
        console.error('[WeatherEngine]', e);
      }
    }, Math.max(0, ms));
    this._timers.add(id);
  }

  _defer(fn) {
    const id = setTimeout(() => {
      this._timers.delete(id);
      try {
        fn();
      } catch (e) {
        console.error('[WeatherEngine]', e);
      }
    }, 0);
    this._timers.add(id);
  }

  // ---------------- thunder

  _thunderTick(ctl, L, t, now) {
    const t0 = this._flashCheckT;
    const t1 = t + 0.25;
    if (!(t1 > t0)) return;
    this._flashCheckT = t1;
    const S = this._plan ? this._plan.S : this.scene;
    for (const src of thunderSources(S, ctl, L, t)) {
      const p = src.ratePerMin / FLASH_CEILING_PER_MIN;
      if (!(p > 0)) continue;
      for (let cell = Math.floor(t0 / FLASH_CELL_SEC); cell <= Math.floor(t1 / FLASH_CELL_SEC); cell++) {
        const ck = `${src.id}#${cell}`;
        let cands = this._cands.get(ck);
        if (!cands) {
          cands = flashCandidates(this.scene.seed, src.id, cell);
          if (this._cands.size > 64) this._cands.clear();
          this._cands.set(ck, cands);
        }
        for (const c of cands) {
          if (c.t < t0 || c.t >= t1 || c.u >= p || this._flashSeen.has(c.key)) continue;
          this._flashSeen.add(c.key);
          this._fire(src, c.t, c.key, now + Math.max(0, c.t - t), ctl, L, false);
        }
      }
    }
    if (this._flashSeen.size > 400) this._flashSeen = new Set([...this._flashSeen].slice(-100));
  }

  _limit(b, mode) {
    const t = nowMs();
    let br = clamp(num(b, 1), 0, 1);
    if (mode === 'sleep') br *= 0.35;
    else if (mode === 'calm') br *= 0.6;
    if (t - this._lastFlashMs < 150) return { brightness: Math.min(br, 0.6), limited: true, merged: true };
    const max = clamp(num(this._safety.maxFlashesPerSec, 3), 0, 3);
    this._flashTimes = this._flashTimes.filter((x) => t - x < 1000);
    if (max <= 0 || this._flashTimes.length >= max) return { brightness: Math.min(br, 0.25), limited: true, merged: false };
    this._flashTimes.push(t);
    this._lastFlashMs = t;
    return { brightness: br, limited: false, merged: false };
  }

  _fire(src, tFlash, key, ctxT, ctl, L, forceCg) {
    const fl = prepareFlash(this.scene.seed, src, key, L.ear, ctl.tempC, forceCg);
    const evt = {
      id: fl.id,
      t: tFlash,
      ctxTime: ctxT,
      channel: fl.chan.channel,
      branches: fl.chan.branches,
      kind: fl.kind,
      strokes: fl.chan.strokes.length,
      distanceKm: fl.distanceKm,
      bearingDeg: fl.bearingDeg,
      delaySec: fl.delaySec,
      brightness: fl.brightness,
      limited: false,
      stormId: fl.stormId,
    };
    const show = () => {
      const b = this._limit(fl.brightness, ctl.mode);
      evt.brightness = b.brightness;
      evt.limited = b.limited;
      evt.merged = b.merged;
      this._emit('flash', evt);
      this._emit('caption', { text: flashCaption(fl), t: tFlash });
    };
    if (!this._ctx || !this._graph) {
      show();
      return evt;
    }
    this._at(ctxT, show);
    if (fl.nearestM < MAX_AUDIBLE_M) {
      const ear = L.ear.slice();
      this._defer(() => this._synthAndPlay(fl, ctxT, ctl, ear));
      this._at(ctxT + fl.delaySec, () => {
        this._emit('thunder', { id: fl.id, t: tFlash + fl.delaySec, distanceKm: fl.distanceKm, bearingDeg: fl.bearingDeg, stormId: fl.stormId });
        this._emit('caption', { text: thunderCaption(fl), t: tFlash + fl.delaySec });
      });
    }
    return evt;
  }

  _synthAndPlay(fl, ctxT, ctl, ear) {
    const g = this._graph;
    const ctx = this._ctx;
    if (!g || !ctx) return;
    if (g.thunders.size >= MAX_THUNDER_VOICES) {
      let farthest = 0;
      for (const r of g.thunders) farthest = Math.max(farthest, r.nearestM);
      if (fl.nearestM >= farthest) return;
    }
    const syn = synthesizeThunder(fl.chan, ear, {
      seed: this.scene.seed,
      key: fl.key,
      tempC: ctl.tempC,
      humidityPct: ctl.rh,
      intensity: fl.intensity,
      lapseKPerKm: ctl.lapse,
      windMps: ctl.U10,
      windFromDeg: ctl.fromDeg,
      sampleRate: synthRateFor(fl.nearestM, ctx.sampleRate),
    });
    if (!syn || this._graph !== g) return;
    const bufs = createThunderBuffers(ctx, syn);
    g.playThunder(bufs, syn, ctxT + syn.offsetSec, syn.centroid, syn.gain);
  }

  /**
   * Trigger a lightning strike now. With stormId, at that storm object; with distanceKm and/or
   * bearingDeg, there; otherwise at the scene's thunder cell. Returns the flash description
   * {id, t, channel, distanceKm, delaySec, ...}; audio plays when the engine is running.
   */
  strikeLightning(o = {}) {
    const opts = obj(o);
    const ctl = this._lastCtl || this._model.compute(this.scene, this._t, 0);
    const L = this._lastL || listenerAt(ctl, this._listener);
    const S = this._plan ? this._plan.S : this.scene;
    const srcs = thunderSources(S, ctl, L, this._t, true);
    let src = null;
    if (opts.stormId !== undefined && opts.stormId !== null) src = srcs.find((s) => s.objectId === String(opts.stormId)) || null;
    const wx = srcs.find((s) => s.id === 'wx');
    if (!src && (opts.distanceKm !== undefined || opts.bearingDeg !== undefined)) {
      const dkm = clamp(num(opts.distanceKm, num(ctl.V('weather.thunder.distance_km'), 8)), 0.05, 60);
      const v = bearingVector(num(opts.bearingDeg, num(ctl.V('weather.thunder.bearing_deg'), 270)));
      src = {
        id: 'manual',
        objectId: null,
        center: [L.pos[0] + v[0] * dkm * 1000, L.pos[1] + v[1] * dkm * 1000],
        ratePerMin: 0,
        intensity: wx ? wx.intensity : 0.6,
        cg: 1,
        spreadM: Math.min(150, dkm * 20),
      };
    }
    if (!src) src = wx;
    const key = `manual:${this._manualCount++}`;
    const ctxT = this._ctx ? this._ctx.currentTime + 0.02 : 0;
    const evt = this._fire(src, this._t, key, ctxT, ctl, L, true);
    return evt;
  }

  // ---------------- offline

  /**
   * Render `seconds` of the current scene (timeline from opts.startSec, default 0) through the
   * same graph in an OfflineAudioContext; resolves to a 24-bit stereo/binaural WAV Blob.
   */
  async renderOffline(seconds, opts = {}) {
    const o = obj(opts);
    const dur = num(seconds, 0);
    if (!(dur > 0)) throw new RangeError('renderOffline: seconds must be a positive number.');
    const sr = Math.round(clamp(num(o.sampleRate, 48000), 8000, 192000));
    const maxSec = Math.floor((4294967295 - 64) / (sr * 2 * 3));
    if (dur > Math.min(maxSec, 3 * 3600)) throw new RangeError(`renderOffline: at most ${Math.min(maxSec, 3 * 3600)} s at ${sr} Hz in one file.`);
    const OAC = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
    if (!OAC || typeof AudioWorkletNode === 'undefined') throw new Error('Offline rendering is not supported in this browser.');
    const length = Math.ceil(dur * sr);
    let octx;
    try {
      octx = new OAC({ numberOfChannels: 2, length, sampleRate: sr });
    } catch (e) {
      octx = new OAC(2, length, sr);
    }
    if (!octx.audioWorklet) throw new Error('AudioWorklet is not available for offline rendering.');
    await loadModules(octx, this.opts.moduleBase);
    let cal;
    try {
      cal = await calibrate(sr, this.opts.moduleBase);
    } catch (e) {
      cal = { ...FALLBACK_CAL };
    }
    const mode = o.renderMode === 'stereo' || o.renderMode === 'binaural' ? o.renderMode : this._renderModeNow();
    const job = new OfflineJob(this, octx, cal, { dur, startSec: Math.max(0, num(o.startSec, 0)), onProgress: o.onProgress, renderMode: mode });
    const buffer = await job.run();
    return encodeWav(buffer, o.bitDepth === 16 ? 16 : 24, this.scene.seed);
  }
}

export default WeatherEngine;
