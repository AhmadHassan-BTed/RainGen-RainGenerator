/**
 * Accessibility and human-unit helpers for the raingen 4D Environment Composer.
 *
 * - Spoken values: every control reads in human units ("force 5, fresh
 *   breeze, 30 km/h from the west"), never as a bare number.
 * - FlashLimiter implements the WCAG 2.3.1 general flash threshold (no more than
 *   three flashes in any one-second period) and merges multi-stroke lightning
 *   (return strokes 40-60 ms apart) into a single visual flash. With
 *   prefers-reduced-motion the flash becomes a soft glow.
 * - Colours for meters and badges use the Okabe & Ito (2008) colour-universal
 *   palette and are always paired with a shape or text.
 * - Beaufort uses the WMO relation v = 0.836 * B^1.5 m/s; flash-to-bang uses
 *   c(T) = 331.3 * sqrt(1 + T/273.15) m/s.
 */
import * as Units from '../../engine/units.js';

export const OKABE_ITO = Object.freeze({
  black: '#000000',
  orange: '#E69F00',
  sky: '#56B4E9',
  green: '#009E73',
  yellow: '#F0E442',
  blue: '#0072B2',
  vermillion: '#D55E00',
  purple: '#CC79A7',
});

/* ------------------------------------------------------------------ numbers */

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export function fmtNum(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  if (digits === undefined) {
    const a = Math.abs(v);
    digits = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 1 : a >= 0.1 ? 2 : 3;
  }
  const s = Number(v).toFixed(digits);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

export function speedOfSound(tempC) {
  return 331.3 * Math.sqrt(1 + tempC / 273.15);
}

/* ------------------------------------------------------------------ compass */

const COMPASS_WORDS = [
  'north', 'north-north-east', 'north-east', 'east-north-east',
  'east', 'east-south-east', 'south-east', 'south-south-east',
  'south', 'south-south-west', 'south-west', 'west-south-west',
  'west', 'west-north-west', 'north-west', 'north-north-west',
];
const COMPASS_ABBR = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

export function wrapDeg(d) {
  return ((d % 360) + 360) % 360;
}

function compassIndex(deg) {
  return Math.round(wrapDeg(deg) / 22.5) % 16;
}

export function compassWord(deg) {
  return COMPASS_WORDS[compassIndex(deg)];
}

export function compassAbbr(deg) {
  try {
    const s = typeof Units.compass16 === 'function' ? Units.compass16(deg) : null;
    if (typeof s === 'string' && s.length) return s;
  } catch (_) { /* fall through */ }
  return COMPASS_ABBR[compassIndex(deg)];
}

/** Direction relative to a listener heading: front / right / behind / left. */
export function relativeDirection(bearingDeg, headingDeg) {
  const r = wrapDeg(bearingDeg - headingDeg);
  if (r < 30 || r > 330) return 'in front';
  if (r < 60) return 'front-right';
  if (r <= 120) return 'to the right';
  if (r < 150) return 'behind-right';
  if (r <= 210) return 'behind';
  if (r < 240) return 'behind-left';
  if (r <= 300) return 'to the left';
  return 'front-left';
}

/* ------------------------------------------------------------------ Beaufort */

const BFT_FALLBACK = [
  ['Calm', 'Smoke rises vertically; leaves are still.'],
  ['Light air', 'Smoke drifts; wind vanes do not move.'],
  ['Light breeze', 'Wind felt on the face; leaves rustle.'],
  ['Gentle breeze', 'Leaves and small twigs in constant motion.'],
  ['Moderate breeze', 'Dust and loose paper lift; small branches move.'],
  ['Fresh breeze', 'Small trees in leaf begin to sway.'],
  ['Strong breeze', 'Large branches move; wires whistle; umbrellas are hard to use.'],
  ['Near gale', 'Whole trees in motion; walking into the wind is an effort.'],
  ['Gale', 'Twigs break off trees; progress on foot is impeded.'],
  ['Strong gale', 'Slight structural damage; roof tiles lift.'],
  ['Storm', 'Trees uprooted; considerable structural damage.'],
  ['Violent storm', 'Widespread damage.'],
  ['Hurricane', 'Devastation.'],
];

/** Continuous Beaufort number from 10 m wind (WMO: v = 0.836 B^1.5). */
export function bftFromMps(mps) {
  return clamp(Math.pow(Math.max(0, mps) / 0.836, 2 / 3), 0, 12);
}

export function mpsFromBft(b) {
  return 0.836 * Math.pow(clamp(b, 0, 12), 1.5);
}

export function bftInfo(b) {
  const f = clamp(Math.round(b), 0, 12);
  let e = null;
  const T = Units.BEAUFORT;
  try {
    if (Array.isArray(T)) {
      e = T.find((x) => x && (x.force === f || x.number === f || x.bft === f || (Array.isArray(x) && x[0] === f))) ?? T[f];
    } else if (T && typeof T === 'object') {
      e = T[f] ?? T[String(f)];
    }
  } catch (_) {
    e = null;
  }
  const fb = BFT_FALLBACK[f];
  const pick = (...vals) => vals.find((v) => typeof v === 'string' && v.length);
  const name = (e && pick(e.name, e.label, e.description_short, Array.isArray(e) ? e[1] : null)) || fb[0];
  const land = (e && pick(e.land, e.landDescription, e.land_description, e.description, e.desc, Array.isArray(e) ? e[2] : null)) || fb[1];
  return { force: f, name, land };
}

/* ------------------------------------------------------------------ rain */

export function rainWord(mmh, set = 'wmo') {
  try {
    if (typeof Units.rainLabel === 'function') {
      const s = Units.rainLabel(mmh, set);
      if (typeof s === 'string' && s.length) return s;
      if (s && typeof s === 'object' && typeof s.label === 'string') return s.label;
    }
  } catch (_) { /* fall through */ }
  if (mmh <= 0.02) return 'none';
  if (set === 'ams') {
    if (mmh < 2.5) return 'light';
    if (mmh < 7.6) return 'moderate';
    if (mmh < 50) return 'heavy';
    return 'violent';
  }
  if (mmh < 0.6) return 'drizzle-light';
  if (mmh < 2.5) return 'slight';
  if (mmh < 10) return 'moderate';
  if (mmh < 50) return 'heavy';
  if (mmh < 100) return 'intense';
  return 'cloudburst';
}

/* ------------------------------------------------------------------ thunder */

export function flashBang(km, tempC = 20) {
  try {
    if (typeof Units.flashToBangSeconds === 'function') {
      const s = Units.flashToBangSeconds(km, tempC);
      if (Number.isFinite(s)) return s;
    }
  } catch (_) { /* fall through */ }
  return (km * 1000) / speedOfSound(tempC);
}

export function kmFromFlashBang(sec, tempC = 20) {
  return (sec * speedOfSound(tempC)) / 1000;
}

/* ------------------------------------------------------------------ speed */

export function speeds(mps) {
  return {
    ms: fmtNum(mps, 1),
    kmh: fmtNum(mps * 3.6, 0),
    mph: fmtNum(mps * 2.23694, 0),
    kn: fmtNum(mps * 1.94384, 0),
  };
}

export function fmtSpeed(mps, system = 'metric') {
  try {
    if (typeof Units.formatSpeed === 'function') {
      const s = Units.formatSpeed(mps, system);
      if (typeof s === 'string' && s.length) return s;
    }
  } catch (_) { /* fall through */ }
  const s = speeds(mps);
  if (system === 'imperial' || system === 'mph') return `${s.mph} mph`;
  if (system === 'knots' || system === 'kn') return `${s.kn} kn`;
  if (system === 'si' || system === 'ms') return `${s.ms} m/s`;
  return `${s.kmh} km/h`;
}

/* ------------------------------------------------------------------ time */

export function fmtHour(h) {
  const hh = Math.floor(wrapDeg(h * 15) / 15);
  const mm = Math.round((h - Math.floor(h)) * 60) % 60;
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}

export function hourWord(h) {
  const x = ((h % 24) + 24) % 24;
  if (x < 4.5) return 'night';
  if (x < 7) return 'dawn';
  if (x < 11) return 'morning';
  if (x < 14) return 'midday';
  if (x < 17.5) return 'afternoon';
  if (x < 20) return 'dusk';
  if (x < 22.5) return 'evening';
  return 'night';
}

/* ------------------------------------------------------------------ spoken values */

function pathGet(obj, path) {
  let cur = obj;
  for (const k of path.split('.')) {
    if (cur == null) return undefined;
    cur = Array.isArray(cur) ? cur.find((o) => o && o.id === k) ?? cur[Number(k)] : cur[k];
  }
  return cur;
}

/** Human value text (aria-valuetext) for a parameter spec and value. */
export function valueText(spec, v, scene) {
  const p = spec.path;
  const T = scene ? pathGet(scene, 'weather.temperature_c') ?? 20 : 20;
  const heading = scene ? pathGet(scene, 'listener.heading_deg') ?? 0 : 0;
  if (v === null || v === undefined) return 'automatic';
  switch (p) {
    case 'weather.precipitation.rate_mm_h':
      return `${fmtNum(v)} millimetres per hour, ${rainWord(v, scene?.studio?.labels || 'wmo')}`;
    case 'weather.wind.speed_mps': {
      const b = bftFromMps(v);
      const info = bftInfo(b);
      const from = scene ? ` from the ${compassWord(pathGet(scene, 'weather.wind.from_deg') ?? 270)}` : '';
      return `force ${fmtNum(b, 1)}, ${info.name.toLowerCase()}, ${speeds(v).kmh} kilometres per hour${from}`;
    }
    case 'weather.wind.from_deg':
      return `from the ${compassWord(v)}, ${Math.round(wrapDeg(v))} degrees, ${relativeDirection(v, heading)} of the listener`;
    case 'weather.thunder.bearing_deg':
      return `to the ${compassWord(v)}, ${Math.round(wrapDeg(v))} degrees, ${relativeDirection(v, heading)}`;
    case 'listener.heading_deg':
      return `facing ${compassWord(v)}, ${Math.round(wrapDeg(v))} degrees`;
    case 'weather.thunder.distance_km':
      return `${fmtNum(v)} kilometres, flash-to-bang ${fmtNum(flashBang(v, T), 1)} seconds`;
    case 'weather.thunder.approach_kmh':
      return v > 0.5 ? `approaching at ${fmtNum(v, 0)} kilometres per hour` : v < -0.5 ? `receding at ${fmtNum(-v, 0)} kilometres per hour` : 'stationary';
    case 'weather.temperature_c':
      return `${fmtNum(v, 1)} degrees Celsius`;
    case 'weather.humidity_pct':
      return `${fmtNum(v, 0)} percent`;
    case 'weather.hour':
      return `${fmtHour(v)}, ${hourWord(v)}`;
    case 'place.window_open':
      return v <= 0.01 ? 'closed' : `${fmtNum(v * 100, 0)} percent open`;
    case 'weather.wind.gust_factor':
      return `gusts ${fmtNum(v, 2)} times the mean wind`;
    default:
      break;
  }
  if (spec.kind === 'bool') return v ? 'on' : 'off';
  if (spec.kind === 'enum') {
    const opt = (spec.options || []).find((o) => o[0] === v);
    return opt ? opt[1] : String(v);
  }
  if (typeof v === 'number') {
    const unit = spec.unitSpoken ?? spec.unit ?? '';
    return `${fmtNum(v)}${unit ? ' ' + unit : ''}`;
  }
  return String(v);
}

export function spokenValue(spec, v, scene) {
  return `${spec.label}: ${valueText(spec, v, scene)}`;
}

/* ------------------------------------------------------------------ live regions */

function liveRegion(politeness) {
  const id = `a11y-live-${politeness}`;
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    el.className = 'sr-only';
    el.setAttribute('aria-live', politeness);
    el.setAttribute('aria-atomic', 'true');
    document.body.appendChild(el);
  }
  return el;
}

let announceTimer = 0;
export function announce(message, politeness = 'polite') {
  const el = liveRegion(politeness);
  el.textContent = '';
  clearTimeout(announceTimer);
  announceTimer = setTimeout(() => { el.textContent = message; }, 40);
}

/* ------------------------------------------------------------------ motion */

const rmQuery = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;

export function prefersReducedMotion() {
  return !!(rmQuery && rmQuery.matches);
}

export function onReducedMotionChange(cb) {
  if (!rmQuery) return () => {};
  const h = () => cb(rmQuery.matches);
  rmQuery.addEventListener ? rmQuery.addEventListener('change', h) : rmQuery.addListener(h);
  return () => (rmQuery.removeEventListener ? rmQuery.removeEventListener('change', h) : rmQuery.removeListener(h));
}

/**
 * Visual flash limiter (independent of audio).
 * admit() returns { mode: 'flash' | 'merge' | 'glow', brightness }.
 */
export class FlashLimiter {
  constructor({ maxPerSec = 3, mergeMs = 120 } = {}) {
    this.maxPerSec = maxPerSec;
    this.mergeMs = mergeMs;
    this.onsets = [];
    this.lastOnset = -Infinity;
    this.forceGlow = false;
  }

  setMax(n) {
    this.maxPerSec = clamp(n, 0, 3);
  }

  admit(now, brightness = 1, reducedMotion = false) {
    const b = clamp(brightness, 0, 1);
    if (reducedMotion || this.forceGlow || this.maxPerSec <= 0) {
      return { mode: 'glow', brightness: Math.min(b, 0.35) };
    }
    if (now - this.lastOnset < this.mergeMs) {
      return { mode: 'merge', brightness: b };
    }
    this.onsets = this.onsets.filter((t) => now - t < 1000);
    if (this.onsets.length >= this.maxPerSec) {
      return { mode: 'glow', brightness: Math.min(b, 0.3) };
    }
    this.onsets.push(now);
    this.lastOnset = now;
    return { mode: 'flash', brightness: b };
  }
}

/* ------------------------------------------------------------------ inputs */

/**
 * Keyboard operation for native range inputs: arrows = fine step (native),
 * PageUp/PageDown = coarse step, Home/End = min/max. Keeps aria-valuetext fresh.
 */
export function enhanceRange(input, { coarse, valueTextFn } = {}) {
  const step = () => {
    const s = parseFloat(input.step);
    return Number.isFinite(s) && s > 0 ? s : 1;
  };
  const span = () => parseFloat(input.max) - parseFloat(input.min);
  input.addEventListener('keydown', (e) => {
    let delta = 0;
    const c = coarse ?? Math.max(step() * 10, span() / 10);
    if (e.key === 'PageUp') delta = c;
    else if (e.key === 'PageDown') delta = -c;
    else return;
    e.preventDefault();
    const v = clamp(parseFloat(input.value) + delta, parseFloat(input.min), parseFloat(input.max));
    input.value = String(v);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  const refresh = () => {
    if (valueTextFn) input.setAttribute('aria-valuetext', valueTextFn(parseFloat(input.value)));
  };
  input.addEventListener('input', refresh);
  refresh();
  return refresh;
}

/** Make an arbitrary element behave as an ARIA slider (used by the compass dial and orb). */
export function makeSliderRole(el, { label, min, max, get, set, fine = 1, coarse = 10, valueTextFn, wrap = false }) {
  el.setAttribute('role', 'slider');
  el.setAttribute('tabindex', '0');
  el.setAttribute('aria-label', label);
  el.setAttribute('aria-valuemin', String(min));
  el.setAttribute('aria-valuemax', String(max));
  const refresh = () => {
    const v = get();
    el.setAttribute('aria-valuenow', String(Math.round(v * 100) / 100));
    if (valueTextFn) el.setAttribute('aria-valuetext', valueTextFn(v));
  };
  el.addEventListener('keydown', (e) => {
    let v = get();
    const span = max - min;
    switch (e.key) {
      case 'ArrowRight': case 'ArrowUp': v += fine; break;
      case 'ArrowLeft': case 'ArrowDown': v -= fine; break;
      case 'PageUp': v += coarse; break;
      case 'PageDown': v -= coarse; break;
      case 'Home': v = min; break;
      case 'End': v = max; break;
      default: return;
    }
    e.preventDefault();
    if (wrap) v = min + ((((v - min) % span) + span) % span);
    else v = clamp(v, min, max);
    set(v, { keyboard: true });
    refresh();
  });
  refresh();
  return refresh;
}

/** True when keyboard focus is in a text-entry control (shortcuts must not fire). */
export function isTyping(target) {
  if (!target) return false;
  const t = target.tagName;
  if (t === 'TEXTAREA') return true;
  if (t === 'SELECT') return true;
  if (t === 'INPUT') {
    const type = (target.type || 'text').toLowerCase();
    return !['range', 'checkbox', 'radio', 'button', 'submit', 'color'].includes(type);
  }
  return !!target.isContentEditable;
}

/* ------------------------------------------------------------------ captions */

export class CaptionsPanel {
  /**
   * @param {HTMLElement} list  element with role="log" aria-live="polite"
   */
  constructor(list, { max = 300, fmtTime = (t) => fmtNum(t, 1) + ' s' } = {}) {
    this.list = list;
    this.max = max;
    this.fmtTime = fmtTime;
    this.items = [];
    if (list) {
      list.setAttribute('role', 'log');
      list.setAttribute('aria-live', 'polite');
      list.setAttribute('aria-relevant', 'additions');
    }
  }

  add({ text, t = 0, kind = 'event' }) {
    if (!text) return;
    const last = this.items[this.items.length - 1];
    if (last && last.text === text && Math.abs(last.t - t) < 0.75) return;
    const item = { text, t, kind };
    this.items.push(item);
    if (this.items.length > this.max) this.items.shift();
    if (!this.list) return;
    const li = document.createElement('li');
    li.className = `caption caption-${kind}`;
    const time = document.createElement('time');
    time.textContent = this.fmtTime(t);
    const span = document.createElement('span');
    span.textContent = text;
    li.append(time, span);
    this.list.appendChild(li);
    while (this.list.children.length > this.max) this.list.firstElementChild.remove();
    this.list.scrollTop = this.list.scrollHeight;
  }

  hasRecent(re, t, window = 1.5) {
    return this.items.some((i) => re.test(i.text) && Math.abs(i.t - t) <= window);
  }

  clear() {
    this.items = [];
    if (this.list) this.list.textContent = '';
  }

  /** WebVTT export of the caption log (each cue shown for 3 s). */
  toVTT() {
    const ts = (t) => {
      const ms = Math.max(0, Math.round(t * 1000));
      const h = Math.floor(ms / 3600000);
      const m = Math.floor((ms % 3600000) / 60000);
      const s = Math.floor((ms % 60000) / 1000);
      const r = ms % 1000;
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(r).padStart(3, '0')}`;
    };
    const lines = ['WEBVTT', ''];
    this.items.forEach((it, i) => {
      lines.push(String(i + 1), `${ts(it.t)} --> ${ts(it.t + 3)}`, it.text, '');
    });
    return lines.join('\n');
  }
}

/* ------------------------------------------------------------------ dialogs */

export function openDialog(dlg, returnFocusTo = document.activeElement) {
  if (!dlg) return;
  if (typeof dlg.showModal === 'function') {
    if (!dlg.open) dlg.showModal();
  } else {
    dlg.setAttribute('open', '');
  }
  dlg.addEventListener('close', () => {
    if (returnFocusTo && typeof returnFocusTo.focus === 'function') returnFocusTo.focus();
  }, { once: true });
  const first = dlg.querySelector('[autofocus], input, select, textarea, button');
  if (first) first.focus();
}

export function closeDialog(dlg) {
  if (!dlg) return;
  if (typeof dlg.close === 'function') dlg.close();
  else dlg.removeAttribute('open');
}
