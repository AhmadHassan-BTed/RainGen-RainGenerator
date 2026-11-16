/**
 * Timeline: the fourth dimension of the 4D Environment Composer.
 *
 * A dope sheet in the grammar of Unreal Sequencer / Houdini / Pro Tools lanes:
 * one lane per animated parameter path (the same canonical dotted id
 * everywhere), keyframes you can drag, a playhead that scrubs the world, a
 * loop region, storm-arc presets and record (latch) mode.
 *
 * Automation priority: automation lane -> pin -> driver + offset, so while a
 * lane is active it wins; a lane being recorded is "latched" and the user's
 * live gesture wins instead.
 * Curves: linear; log (geometric x(u) = a (b/a)^u, perceptually even for mm/h
 * and km, which the ear judges by ratio); hold; smooth (Hermite smoothstep).
 * Compass angles interpolate along the shortest arc.
 * Timecode: mm:ss.cc and SMPTE HH:MM:SS:FF at 30 fps non-drop (SMPTE ST 12-1).
 * Every drag has a keyboard alternative: the Points table, P adds a point at
 * the playhead, Alt+Left/Right jumps between points, Ctrl+G goes to a time.
 */
import { PARAMS, paramSpec, clone } from './scene-store.js';
import { STORM_ARCS } from './presets.js';
import { objectLabel } from './library.js';
import { valueText, fmtNum, announce, clamp, wrapDeg, bftFromMps, compassAbbr, fmtHour } from './a11y.js';

export const FPS = 30;
const CURVES = [['linear', 'Linear'], ['log', 'Log'], ['hold', 'Hold'], ['smooth', 'Smooth']];
const ARC_PATHS = [
  'weather.storm_arc', 'weather.precipitation.type', 'weather.precipitation.rate_mm_h', 'weather.precipitation.drops',
  'weather.wind.speed_mps', 'weather.thunder.enabled', 'weather.thunder.distance_km', 'weather.thunder.approach_kmh',
];
const ARC_NAMES = [['arriving', 'Arriving'], ['steady', 'Steady'], ['passing', 'Passing'], ['clearing', 'Clearing']];
const TICK_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const PARAM_PATHS = new Set(PARAMS.map((p) => p.path));

/* ------------------------------------------------------------------ time formatting */

/** "mm:ss.cc" (minutes keep counting past 60). */
export function fmtClock(t) {
  const s = Math.max(0, Number(t) || 0);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  const whole = Math.floor(sec);
  const cs = Math.min(99, Math.floor((sec - whole) * 100));
  return `${String(m).padStart(2, '0')}:${String(whole).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

/** SMPTE "HH:MM:SS:FF" (non-drop). */
export function fmtSMPTE(t, fps = FPS) {
  const frames = Math.floor(Math.max(0, Number(t) || 0) * fps + 1e-6);
  const ff = frames % fps;
  const S = Math.floor(frames / fps);
  const hh = Math.floor(S / 3600);
  const mm = Math.floor((S % 3600) / 60);
  const ss = S % 60;
  return [hh, mm, ss, ff].map((x) => String(x).padStart(2, '0')).join(':');
}

/** Spoken form: "1 minute 2.4 seconds". */
export function spokenTime(t) {
  const s = Math.max(0, Number(t) || 0);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  const secTxt = `${fmtNum(sec, 1)} second${Math.abs(sec - 1) < 0.05 ? '' : 's'}`;
  return m > 0 ? `${m} minute${m === 1 ? '' : 's'} ${secTxt}` : secTxt;
}

/**
 * Parse "75", "75s", "1m30s", "1:15.5" (m:s), "0:01:15" (h:m:s) or SMPTE "00:01:15:12".
 * Returns NaN when unreadable.
 */
export function parseTime(str, fps = FPS) {
  const s = String(str ?? '').trim().toLowerCase().replace(',', '.');
  if (!s) return NaN;
  let m = /^(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?$/.exec(s);
  if (m) return parseFloat(m[1]);
  m = /^(?:(\d+(?:\.\d+)?)\s*h)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in)?)?\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec)?)?$/.exec(s);
  if (m && (m[1] || m[2] || m[3])) return (parseFloat(m[1] || 0) * 3600) + (parseFloat(m[2] || 0) * 60) + parseFloat(m[3] || 0);
  const parts = s.split(':');
  if (parts.some((p) => !/^\d+(?:\.\d+)?$/.test(p))) return NaN;
  const n = parts.map(parseFloat);
  if (n.length === 4) return n[0] * 3600 + n[1] * 60 + n[2] + n[3] / fps;
  if (n.length === 3) return n[0] * 3600 + n[1] * 60 + n[2];
  if (n.length === 2) return n[0] * 60 + n[1];
  return NaN;
}

function fmtTick(t, step) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  if (step < 1) return `${m}:${s.toFixed(1).padStart(4, '0')}`;
  if (step >= 3600) return `${Math.round(t / 3600)}h`;
  return `${m}:${String(Math.round(s)).padStart(2, '0')}`;
}

/* ------------------------------------------------------------------ evaluation */

function isAngular(path) {
  const spec = PARAM_PATHS.has(path) ? paramSpec(path) : null;
  return !!(spec && spec.kind === 'compass');
}

/** Interpolate between two keyframe values with the outgoing curve of the first. */
export function interpolate(a, b, u, curve = 'linear', angular = false) {
  if (curve === 'hold') return clone(a);
  if (typeof a === 'number' && typeof b === 'number') {
    const k = curve === 'smooth' ? u * u * (3 - 2 * u) : u;
    if (angular) {
      const d = ((((b - a) % 360) + 540) % 360) - 180;
      return wrapDeg(a + d * k);
    }
    if (curve === 'log' && a > 0 && b > 0) return a * Math.pow(b / a, k);
    return a + (b - a) * k;
  }
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    return a.map((x, i) => interpolate(x, b[i], u, curve === 'log' ? 'linear' : curve, false));
  }
  return clone(a);
}

/** Value of one sorted track at time t. */
export function evaluateTrack(keys, t, angular = false) {
  if (!keys || !keys.length) return undefined;
  if (t <= keys[0].at) return clone(keys[0].value);
  const last = keys[keys.length - 1];
  if (t >= last.at) return clone(last.value);
  let lo = 0;
  let hi = keys.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keys[mid].at <= t) lo = mid;
    else hi = mid;
  }
  const k0 = keys[lo];
  const k1 = keys[hi];
  const span = k1.at - k0.at;
  const u = span > 0 ? (t - k0.at) / span : 1;
  return interpolate(k0.value, k1.value, u, k0.curve, angular);
}

/** Map path -> sorted keyframes. */
export function groupTimeline(timeline) {
  const m = new Map();
  for (const k of timeline || []) {
    if (!m.has(k.path)) m.set(k.path, []);
    m.get(k.path).push(k);
  }
  for (const arr of m.values()) arr.sort((a, b) => a.at - b.at);
  return m;
}

/** All automated values at time t: { path: value }. `skip` is a Set of paths to leave alone. */
export function evaluateTimeline(timeline, t, skip = null) {
  const out = {};
  for (const [path, keys] of groupTimeline(timeline)) {
    if (skip && skip.has(path)) continue;
    out[path] = evaluateTrack(keys, t, isAngular(path));
  }
  return out;
}

export function defaultCurve(path) {
  const spec = paramSpec(path);
  if (spec.kind === 'enum' || spec.kind === 'bool') return 'hold';
  if (spec.kind === 'vec3' || /\.params\.track$/.test(path)) return 'smooth';
  if (spec.scale === 'log') return 'log';
  return 'linear';
}

/* ------------------------------------------------------------------ labels */

function trackLabel(path, store) {
  if (PARAM_PATHS.has(path)) return paramSpec(path).label;
  if (path === 'listener.position') return 'Listener position';
  if (path.startsWith('place.surfaces.')) return `Surface: ${path.split('.')[2].replace(/_/g, ' ')}`;
  const m = /^place\.objects\.([^.]+)\.(.+)$/.exec(path);
  if (m) {
    const o = store.getObject(m[1]);
    const name = o ? objectLabel(o) : m[1];
    const sub = m[2] === 'params.track' ? 'track' : m[2] === 'rotation_deg' ? 'rotation' : m[2].replace(/^params\./, '').replace(/_/g, ' ');
    return `${name} · ${sub}`;
  }
  if (path.startsWith('studio.')) return path.slice(7);
  return path;
}

function shortValue(path, v) {
  if (v === null || v === undefined) return '–';
  const spec = paramSpec(path);
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (Array.isArray(v)) {
    if (v.length && Array.isArray(v[0])) return `${v.length} pts`;
    return `(${v.map((x) => fmtNum(Number(x) || 0, 1)).join(', ')})`;
  }
  if (spec.kind === 'enum') {
    const o = (spec.options || []).find((x) => x[0] === v);
    return o ? o[1] : String(v);
  }
  if (typeof v !== 'number') return String(v);
  if (spec.scale === 'beaufort') return `F${fmtNum(bftFromMps(v), 1)} · ${Math.round(v * 3.6)} km/h`;
  if (spec.kind === 'compass') return `${Math.round(wrapDeg(v))}° ${compassAbbr(v)}`;
  if (spec.kind === 'hour') return fmtHour(v);
  if (spec.format) return spec.format(v);
  return `${fmtNum(v)}${spec.unit ? ' ' + spec.unit : ''}`;
}

function spokenValue(path, v, scene) {
  const spec = paramSpec(path);
  if (PARAM_PATHS.has(path)) return valueText(spec, v, scene);
  return shortValue(path, v);
}

function isNumericTrack(keys) {
  return keys.length > 0 && keys.every((k) => typeof k.value === 'number');
}

function normFor(path, keys) {
  const spec = paramSpec(path);
  if (PARAM_PATHS.has(path) && spec.min !== undefined && spec.max !== undefined) {
    if (spec.scale === 'log') {
      const lo = Math.log(spec.min);
      const hi = Math.log(spec.max);
      return (v) => (Math.log(Math.max(spec.min, v)) - lo) / (hi - lo || 1);
    }
    if (spec.scale === 'beaufort') return (v) => bftFromMps(v) / 12;
    return (v) => (v - spec.min) / (spec.max - spec.min || 1);
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const k of keys) {
    lo = Math.min(lo, k.value);
    hi = Math.max(hi, k.value);
  }
  if (!(hi > lo)) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.1;
  return (v) => (v - (lo - pad)) / ((hi + pad) - (lo - pad));
}

/* ------------------------------------------------------------------ dom helper */

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith('--')) el.style.setProperty(sk, String(sv));
        else el.style[sk] = sv;
      }
    }
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const round2 = (x) => Math.round(x * 100) / 100;

/* ------------------------------------------------------------------ class */

export class Timeline {
  /**
   * @param {HTMLElement} root
   * @param {{store: any, getTime?: () => number, isPlaying?: () => boolean, onSeek: (t: number) => void,
   *          onTogglePlay?: () => void, onLayout?: () => void}} opts
   */
  constructor(root, opts) {
    this.root = root;
    this.store = opts.store;
    this.opts = opts;
    this.time = 0;
    this.view = null;
    this.selectedTrack = null;
    this.selectedKey = null;
    this.recording = false;
    this.latched = new Set();
    this._recLast = new Map();
    this._take = false;
    this._groups = null;
    this._lastApplied = -1;
    this._rows = new Map();
    this.pointsOpen = false;
    this.playing = false;
    this._readoutQueued = false;
    this._build();
    this.store.on('change', (e) => this._onChange(e));
    this._ro = new ResizeObserver(() => {
      this._renderRuler();
      this._renderCurves();
    });
    this._ro.observe(this.rulerEl);
    this.render();
  }

  get duration() {
    return Math.max(10, this.store.scene.studio.duration || 300);
  }

  _getGroups() {
    if (!this._groups) this._groups = groupTimeline(this.store.scene.timeline);
    return this._groups;
  }

  _viewRange() {
    if (!this.view) return [0, this.duration];
    const [a, b] = this.view;
    return b - a > 0.2 ? [a, b] : [0, this.duration];
  }

  _x(t) {
    const [a, b] = this._viewRange();
    return (t - a) / (b - a);
  }

  _tFromClientX(clientX, el = this.rulerEl) {
    const r = el.getBoundingClientRect();
    const [a, b] = this._viewRange();
    const u = clamp((clientX - r.left) / Math.max(1, r.width), 0, 1);
    return a + u * (b - a);
  }

  /* ------------------------------------------------------------------ build */
  _build() {
    const root = this.root;
    root.textContent = '';
    root.classList.add('tl');

    const btn = (cls, label, text, onclick, extra = {}) => h('button', { type: 'button', class: `tl-btn ${cls}`, 'aria-label': label, title: label, onclick, ...extra }, text);

    this.playBtn = btn('tl-play', 'Play (Space)', '▶', () => this.opts.onTogglePlay?.());
    this.recBtn = btn('tl-rec', 'Record moves and edits while playing', '● Rec', () => this.toggleRecord(), { 'aria-pressed': 'false' });
    this.loopBtn = btn('tl-loop-btn', 'Loop region (L)', '⟲ Loop', () => this.toggleLoop(), { 'aria-pressed': 'false' });
    this.tcMain = h('output', { class: 'tc-main', 'aria-label': 'Playhead time', 'aria-live': 'off' }, '00:00.00');
    this.tcSmpte = h('output', { class: 'tc-smpte', 'aria-label': 'SMPTE timecode', 'aria-live': 'off' }, '00:00:00:00');
    this.gotoInput = h('input', { type: 'text', class: 'tl-goto', placeholder: 'go to 1:30 or 00:01:30:00', 'aria-label': 'Go to time', 'aria-keyshortcuts': 'Control+G', spellcheck: 'false' });
    this.gotoInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const t = parseTime(this.gotoInput.value);
        if (Number.isFinite(t)) {
          this.opts.onSeek?.(clamp(t, 0, this.duration * 4));
          announce(`Playhead at ${spokenTime(t)}`);
          this.gotoInput.value = '';
        } else announce('Time not understood. Try 1:30, 90s or 00:01:30:00.');
      } else if (e.key === 'Escape') {
        this.gotoInput.value = '';
        this.rulerEl.focus();
      }
    });
    this.durInput = h('input', { type: 'number', class: 'num tl-dur', min: 10, max: 14400, step: 10, 'aria-label': 'Timeline length in seconds' });
    this.durInput.addEventListener('change', () => {
      const v = clamp(parseFloat(this.durInput.value) || 300, 10, 14400);
      const st = this.store.scene.studio;
      this.store.beginGesture('length');
      this.store.set('studio.duration', v, { source: 'studio' });
      if (st.loop[1] > v || st.loop[1] === st.duration) this.store.set('studio.loop', [Math.min(st.loop[0], v - 1), v], { source: 'studio' });
      this.store.endGesture();
      this.view = null;
      this.render();
    });

    const arcs = h('div', { class: 'tl-arcs', role: 'group', 'aria-label': 'Storm arc presets' },
      h('span', { class: 'tl-arcs-label', 'aria-hidden': 'true' }, 'Storm arc'),
      ...ARC_NAMES.map(([k, lab]) => h('button', {
        type: 'button', class: 'tl-btn tl-arc', 'data-arc': k, 'aria-pressed': 'false',
        title: `${lab}: write rain, wind and thunder lanes over the arc length`, onclick: () => this.applyArc(k),
      }, lab)));

    this.addSel = h('select', { class: 'tl-add', 'aria-label': 'Add an animation track' });
    this.addSel.addEventListener('change', () => {
      const p = this.addSel.value;
      this.addSel.value = '';
      if (p) this.animate(p);
    });
    this.curveSel = h('select', { class: 'tl-curve', 'aria-label': 'Curve of the selected point' }, ...CURVES.map(([v, l]) => h('option', { value: v }, l)));
    this.curveSel.addEventListener('change', () => {
      if (this.selectedKey) this.store.updateKeyframe(this.selectedKey, { curve: this.curveSel.value }, { coalesce: null });
    });
    this.addKeyBtn = btn('tl-addkey', 'Add point at playhead (P)', '◆+', () => this.addKeyAtPlayhead(), { 'aria-keyshortcuts': 'P' });
    this.delKeyBtn = btn('tl-delkey', 'Delete selected point (Delete)', '◆−', () => this.deleteSelectedKey());
    this.pointsBtn = btn('tl-points-btn', 'Points table: edit points as text', '☰ Points', () => this.togglePoints(), { 'aria-expanded': 'false', 'aria-controls': 'tl-points' });
    const zoomIn = btn('tl-zoom', 'Zoom in (Ctrl+wheel)', '+', () => this.zoom(0.5));
    const zoomOut = btn('tl-zoom', 'Zoom out', '−', () => this.zoom(2));
    const zoomFit = btn('tl-zoom', 'Fit whole timeline', 'Fit', () => { this.view = null; this.render(); });
    this.collapseBtn = btn('tl-collapse', 'Collapse timeline', '▾', () => this.toggleCollapsed(), { 'aria-expanded': 'true' });

    const toolbar = h('div', { class: 'tl-toolbar', role: 'toolbar', 'aria-label': 'Timeline tools' },
      h('div', { class: 'tl-group' },
        btn('tl-rew', 'Go to start (Home)', '⏮', () => this.opts.onSeek?.(this._loopStart())),
        this.playBtn, this.recBtn, this.loopBtn),
      h('div', { class: 'tl-group tl-tc' }, this.tcMain, this.tcSmpte),
      h('div', { class: 'tl-group' }, this.gotoInput, h('label', { class: 'tl-dur-label' }, 'Length ', this.durInput, ' s')),
      arcs,
      h('div', { class: 'tl-group' }, this.addSel, this.curveSel, this.addKeyBtn, this.delKeyBtn, this.pointsBtn),
      h('div', { class: 'tl-group tl-end' }, zoomOut, zoomIn, zoomFit, this.collapseBtn),
    );

    /* ruler */
    this.ticksEl = h('div', { class: 'tl-ticks', 'aria-hidden': 'true' });
    this.loopEl = h('div', { class: 'tl-loop', 'aria-hidden': 'true' });
    this.headEl = h('div', { class: 'tl-ph-handle', 'aria-hidden': 'true' });
    this.rulerEl = h('div', {
      class: 'tl-ruler', role: 'slider', tabindex: '0', 'aria-label': 'Playhead. Arrows move 1 second, Page Up/Down 10 seconds, Shift-drag sets the loop.',
      'aria-valuemin': '0',
    }, this.loopEl, this.ticksEl, this.headEl);
    this._bindRuler();

    this.tracksEl = h('div', { class: 'tl-tracks', role: 'list', 'aria-label': 'Animation tracks' });
    this.emptyEl = h('p', { class: 'tl-empty' },
      'No animation yet. Right-click any control in the inspector and choose Animate, pick a track from “Add track”, or press a storm-arc preset. ',
      'Record (● Rec) captures moves of objects and the listener while playing.');
    this.bodyEl = h('div', { class: 'tl-body' },
      h('div', { class: 'tl-ruler-row' }, h('div', { class: 'tl-corner', 'aria-hidden': 'true' }, 'Tracks'), this.rulerEl),
      this.tracksEl,
      this.emptyEl,
    );
    this.bodyEl.addEventListener('wheel', (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        this.zoom(e.deltaY > 0 ? 1.25 : 0.8, this._tFromClientX(e.clientX));
      } else if (e.shiftKey) {
        e.preventDefault();
        this.pan((e.deltaY || e.deltaX) > 0 ? 0.1 : -0.1);
      }
    }, { passive: false });

    this.pointsEl = h('section', { class: 'tl-points', id: 'tl-points', 'aria-label': 'Points table', hidden: true });

    root.append(toolbar, this.bodyEl, this.pointsEl);
    root.addEventListener('keydown', (e) => this._onKey(e));
  }

  /* ------------------------------------------------------------------ ruler & scrubbing */
  _bindRuler() {
    const el = this.rulerEl;
    let drag = null;
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      el.setPointerCapture(e.pointerId);
      const t = this._snapT(this._tFromClientX(e.clientX), e);
      if (e.shiftKey) {
        drag = { kind: 'loop', from: t };
        this._setLoopPreview(t, t);
      } else {
        drag = { kind: 'scrub' };
        this.opts.onSeek?.(t);
      }
    });
    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const t = this._snapT(this._tFromClientX(e.clientX), e);
      if (drag.kind === 'scrub') this.opts.onSeek?.(t);
      else {
        drag.to = t;
        this._setLoopPreview(Math.min(drag.from, t), Math.max(drag.from, t));
      }
    });
    const end = (e) => {
      if (!drag) return;
      if (drag.kind === 'loop' && drag.to !== undefined && Math.abs(drag.to - drag.from) > 0.2) {
        const a = round2(Math.min(drag.from, drag.to));
        const b = round2(Math.max(drag.from, drag.to));
        this.store.beginGesture('loop');
        this.store.set('studio.loop', [a, b], { source: 'studio' });
        this.store.set('studio.loopOn', true, { source: 'studio' });
        this.store.endGesture();
        announce(`Loop ${fmtClock(a)} to ${fmtClock(b)}`);
      }
      drag = null;
      try { el.releasePointerCapture(e.pointerId); } catch (_) { /* released */ }
      this._renderRuler();
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('keydown', (e) => {
      let t = this.time;
      switch (e.key) {
        case 'ArrowRight': t += e.shiftKey ? 10 : 1; break;
        case 'ArrowLeft': t -= e.shiftKey ? 10 : 1; break;
        case 'PageUp': t += 10; break;
        case 'PageDown': t -= 10; break;
        case 'Home': t = 0; break;
        case 'End': t = this.duration; break;
        default: return;
      }
      if (e.altKey) return;
      e.preventDefault();
      e.stopPropagation();
      this.opts.onSeek?.(clamp(t, 0, this.duration * 4));
    });
  }

  _snapT(t, e) {
    if (e && e.altKey) return Math.max(0, t);
    const [a, b] = this._viewRange();
    const grid = (b - a) > 600 ? 1 : (b - a) > 60 ? 0.1 : 1 / FPS;
    return Math.max(0, Math.round(t / grid) * grid);
  }

  _setLoopPreview(a, b) {
    const x0 = this._x(a);
    const x1 = this._x(b);
    this.loopEl.style.left = `${clamp(x0, 0, 1) * 100}%`;
    this.loopEl.style.width = `${Math.max(0, clamp(x1, 0, 1) - clamp(x0, 0, 1)) * 100}%`;
    this.loopEl.classList.add('on', 'preview');
  }

  _loopStart() {
    const st = this.store.scene.studio;
    return st.loopOn ? st.loop[0] : 0;
  }

  _renderRuler() {
    const r = this.rulerEl.getBoundingClientRect();
    const width = Math.max(1, r.width);
    const [a, b] = this._viewRange();
    const span = b - a;
    const step = TICK_STEPS.find((s) => (span / s) * 80 <= width) || 3600;
    this.ticksEl.textContent = '';
    const first = Math.ceil(a / step) * step;
    const frag = document.createDocumentFragment();
    for (let t = first; t <= b + 1e-9; t += step) {
      const x = (t - a) / span;
      const major = Math.abs((t / (step * 5)) - Math.round(t / (step * 5))) < 1e-6;
      frag.append(h('span', { class: `tl-tick${major ? ' major' : ''}`, style: { left: `${x * 100}%` } }, h('span', { class: 'tl-tick-label' }, fmtTick(t, step))));
      const minor = step / 5;
      if (step >= 0.5) {
        for (let k = 1; k < 5; k++) {
          const tm = t + k * minor;
          if (tm > b) break;
          frag.append(h('span', { class: 'tl-tick minor', style: { left: `${((tm - a) / span) * 100}%` } }));
        }
      }
    }
    this.ticksEl.append(frag);
    const st = this.store.scene.studio;
    const [la, lb] = st.loop || [0, this.duration];
    const x0 = clamp(this._x(la), 0, 1);
    const x1 = clamp(this._x(lb), 0, 1);
    this.loopEl.classList.remove('preview');
    this.loopEl.classList.toggle('on', !!st.loopOn);
    this.loopEl.style.left = `${x0 * 100}%`;
    this.loopEl.style.width = `${Math.max(0, x1 - x0) * 100}%`;
    this.rulerEl.setAttribute('aria-valuemax', String(Math.round(this.duration)));
    this._updatePlayhead();
  }

  _updatePlayhead() {
    const x = this._x(this.time);
    const inView = x >= 0 && x <= 1;
    this.bodyEl.style.setProperty('--ph', String(clamp(x, 0, 1)));
    this.bodyEl.classList.toggle('ph-out', !inView);
    this.headEl.hidden = !inView;
    // aria values only while stopped, so a focused ruler is not read out 60 times a second
    if (!this.playing) {
      this.rulerEl.setAttribute('aria-valuenow', String(round2(this.time)));
      this.rulerEl.setAttribute('aria-valuetext', `${spokenTime(this.time)}, timecode ${fmtSMPTE(this.time)}`);
    }
    this.tcMain.textContent = fmtClock(this.time);
    this.tcSmpte.textContent = fmtSMPTE(this.time);
    if (this.playing && !inView && this.view) {
      const [a, b] = this._viewRange();
      const span = b - a;
      if (this.time > b || this.time < a) {
        this.view = [Math.max(0, this.time - span * 0.1), Math.max(0, this.time - span * 0.1) + span];
        this._renderRuler();
        this._renderTracks();
      }
    }
  }

  zoom(factor, center = this.time) {
    const [a, b] = this._viewRange();
    const span = clamp((b - a) * factor, 1, Math.max(this.duration, this.time + 10) * 1.05);
    const k = (center - a) / (b - a || 1);
    let start = center - k * span;
    start = clamp(start, 0, Math.max(0, Math.max(this.duration, this.time) - span * 0.25));
    this.view = span >= this.duration && start <= 0 ? null : [start, start + span];
    this._renderRuler();
    this._renderTracks();
  }

  pan(fraction) {
    const [a, b] = this._viewRange();
    const span = b - a;
    const start = Math.max(0, a + span * fraction);
    this.view = [start, start + span];
    this._renderRuler();
    this._renderTracks();
  }

  /* ------------------------------------------------------------------ state from outside */
  /** Move the playhead (main calls this every frame while playing). */
  setTime(t, { apply = true, force = false } = {}) {
    this.time = Math.max(0, Number(t) || 0);
    this._updatePlayhead();
    if (apply) this.apply(force);
    this._queueReadouts();
  }

  setPlaying(on) {
    this.playing = !!on;
    this.playBtn.textContent = on ? '❚❚' : '▶';
    this.playBtn.setAttribute('aria-label', on ? 'Pause (Space)' : 'Play (Space)');
    this.playBtn.title = this.playBtn.getAttribute('aria-label');
    if (!on) {
      this.endTake();
      this._updatePlayhead();
    }
    this.root.classList.toggle('is-playing', !!on);
  }

  /** Apply automation at the current time (throttled to ~30 Hz unless forced). */
  apply(force = false) {
    if (!force && Math.abs(this.time - this._lastApplied) < 1 / 30) return;
    this._lastApplied = this.time;
    const groups = this._getGroups();
    if (!groups.size) return;
    const vals = {};
    for (const [path, keys] of groups) {
      if (this.latched.has(path)) continue;
      const v = evaluateTrack(keys, this.time, isAngular(path));
      if (v !== undefined) vals[path] = v;
    }
    this.store.applyAutomation(vals);
  }

  /* ------------------------------------------------------------------ store events */
  _onChange(e) {
    if (e.kind === 'load' || e.kind === 'restore') {
      this._groups = null;
      this.latched.clear();
      this._recLast.clear();
      this.selectedKey = null;
      if (this.selectedTrack && !this.store.hasTrack(this.selectedTrack)) this.selectedTrack = null;
      this.view = null;
      this.render();
      this._lastApplied = -1;
      queueMicrotask(() => this.apply(true));
      return;
    }
    if (e.kind === 'timeline' || e.paths.includes('timeline')) {
      this._groups = null;
      if (this.selectedKey && !this.store.scene.timeline.some((k) => k.id === this.selectedKey)) this.selectedKey = null;
      if (this.selectedTrack && !this.store.hasTrack(this.selectedTrack)) this.selectedTrack = null;
      this._renderTracks();
      this._syncToolbar();
      if (this.pointsOpen) this._renderPoints();
      if (!this.recording) {
        this._lastApplied = -1;
        queueMicrotask(() => this.apply(true));
      }
    } else if (e.paths.some((p) => p.startsWith('studio'))) {
      this._syncToolbar();
      this._renderRuler();
      this._renderTracks();
    } else if (e.kind === 'object-remove' || e.kind === 'object-add' || (e.kind === 'object' && e.source === 'user' && this._rows.size)) {
      this._renderTracks();
      if (e.kind !== 'object') this._fillAddSelect();
    }
    this._maybeRecord(e);
    this._queueReadouts();
  }

  _queueReadouts() {
    if (this._readoutQueued) return;
    this._readoutQueued = true;
    requestAnimationFrame(() => {
      this._readoutQueued = false;
      for (const [path, row] of this._rows) row.val.textContent = shortValue(path, this.store.get(path));
    });
  }

  /* ------------------------------------------------------------------ recording */
  toggleRecord(on = !this.recording) {
    this.recording = !!on;
    this.recBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    this.root.classList.toggle('is-recording', this.recording);
    if (!on) this.endTake();
    announce(on ? 'Record armed: moves and edits made while playing become keyframes.' : 'Record off');
  }

  _maybeRecord(e) {
    if (!this.recording || !this.opts.isPlaying?.()) return;
    if (e.source !== 'user' && e.source !== 'walk') return;
    if (e.kind === 'timeline' || e.kind === 'load' || e.kind === 'restore' || e.kind === 'override') return;
    const paths = new Set();
    for (const p of e.paths) {
      if (p === 'listener.position' || p === 'listener.heading_deg') paths.add(p);
      else if (/^place\.objects\.[^.]+$/.test(p) && e.kind === 'object') {
        const id = p.split('.')[2];
        const o = this.store.getObject(id);
        if (!o) continue;
        if (o.type === 'storm') paths.add(`${p}.params.track`);
        else {
          paths.add(`${p}.position`);
          if (this.store.hasTrack(`${p}.rotation_deg`)) paths.add(`${p}.rotation_deg`);
        }
      } else if (/^place\.objects\.[^.]+\.(position|rotation_deg|params\..+)$/.test(p)) paths.add(p);
      else if (PARAM_PATHS.has(p)) paths.add(p);
    }
    for (const p of paths) this._recordPoint(p);
  }

  _recordPoint(path) {
    const v = this.store.get(path);
    if (v === undefined) return;
    const discrete = typeof v !== 'number' && !Array.isArray(v);
    const last = this._recLast.get(path);
    this.latched.add(path);
    if (!discrete && last && Math.abs(this.time - last.t) < 0.2) {
      last.pending = true;
      return;
    }
    if (!this._take) {
      this.store.beginGesture('record take');
      this._take = true;
    }
    this._recLast.set(path, { t: this.time, pending: false });
    this.store.addKeyframe({ path, at: round2(this.time), value: v, curve: defaultCurve(path) }, { history: true, coalesce: `rec:${path}` });
    if (this.time > this.duration - 1) {
      this.store.set('studio.duration', Math.ceil((this.time + 30) / 10) * 10, { source: 'studio', history: false });
    }
  }

  /** Finish a record take: write the final pose of every latched path, close the undo step. */
  endTake() {
    if (this._take) {
      for (const path of this.latched) {
        const last = this._recLast.get(path);
        const v = this.store.get(path);
        if (last && last.pending && v !== undefined) {
          this.store.addKeyframe({ path, at: round2(this.time), value: v, curve: defaultCurve(path) }, { history: true, coalesce: `rec:${path}` });
        }
      }
      this._take = false;
      this.store.endGesture();
    }
    this.latched.clear();
    this._recLast.clear();
  }

  /* ------------------------------------------------------------------ editing */
  /** "Animate" from the inspector: add a point at the playhead with the current value. */
  animate(path) {
    const v = this.store.get(path);
    if (v === undefined) {
      announce(`Cannot animate ${path}: no such value`);
      return null;
    }
    const isNew = !this.store.hasTrack(path);
    const id = this.store.addKeyframe({ path, at: round2(this.time), value: v, curve: defaultCurve(path) }, { coalesce: null });
    this.selectedTrack = path;
    this.selectedKey = id;
    this.toggleCollapsed(true);
    this._renderTracks();
    this._syncToolbar();
    if (this.pointsOpen) this._renderPoints();
    const label = trackLabel(path, this.store);
    announce(`${isNew ? 'New track' : 'Point added'}: ${label} at ${fmtClock(this.time)}`);
    requestAnimationFrame(() => this._rows.get(path)?.keyEls.get(id)?.focus());
    return id;
  }

  addKeyAtPlayhead(path = this.selectedTrack) {
    if (!path) {
      announce('Select a track first (click its name), or use Add track.');
      return null;
    }
    const keys = this._getGroups().get(path) || [];
    const v = keys.length ? evaluateTrack(keys, this.time, isAngular(path)) : this.store.get(path);
    const cur = this.store.get(path);
    const value = this.latched.has(path) || !keys.length ? cur : v;
    const before = [...keys].reverse().find((k) => k.at <= this.time);
    const id = this.store.addKeyframe({ path, at: round2(this.time), value, curve: before?.curve || defaultCurve(path) }, { coalesce: null });
    this.selectedKey = id;
    this._renderTracks();
    this._syncToolbar();
    announce(`Point added at ${fmtClock(this.time)}`);
    return id;
  }

  deleteSelectedKey() {
    if (!this.selectedKey) return;
    const k = this.store.scene.timeline.find((x) => x.id === this.selectedKey);
    if (!k) return;
    const siblings = (this._getGroups().get(k.path) || []).filter((x) => x.id !== k.id);
    const next = siblings.find((x) => x.at >= k.at) || siblings[siblings.length - 1];
    this.store.removeKeyframe(k.id);
    this.selectedKey = next ? next.id : null;
    announce('Point deleted');
    if (next) requestAnimationFrame(() => this._rows.get(k.path)?.keyEls.get(next.id)?.focus());
  }

  _selectKey(id, path, { focus = false } = {}) {
    this.selectedKey = id;
    this.selectedTrack = path;
    for (const [p, row] of this._rows) {
      row.el.classList.toggle('selected', p === path);
      row.name.setAttribute('aria-pressed', p === path ? 'true' : 'false');
      for (const [kid, el] of row.keyEls) el.classList.toggle('selected', kid === id);
    }
    this._syncToolbar();
    if (this.pointsOpen) this._renderPoints();
    if (focus) this._rows.get(path)?.keyEls.get(id)?.focus();
  }

  _jumpKey(dir) {
    const path = this.selectedTrack;
    if (!path) return;
    const keys = this._getGroups().get(path) || [];
    if (!keys.length) return;
    const eps = 1e-3;
    const k = dir > 0 ? keys.find((x) => x.at > this.time + eps) : [...keys].reverse().find((x) => x.at < this.time - eps);
    if (!k) return;
    this.opts.onSeek?.(k.at);
    this._selectKey(k.id, path, { focus: true });
    announce(`Point at ${fmtClock(k.at)}: ${spokenValue(path, k.value, this.store.scene)}`);
  }

  _nudgeKeyValue(k, dir, big) {
    if (typeof k.value === 'number') {
      const spec = paramSpec(k.path);
      let v;
      if (spec.scale === 'log') v = k.value * Math.pow(big ? 1.25 : 1.05, dir);
      else {
        const span = (spec.max ?? 100) - (spec.min ?? 0);
        const step = spec.step && spec.step !== 'any' ? spec.step : span / 100;
        v = k.value + dir * (big ? step * 10 : step);
      }
      if (spec.kind === 'compass') v = wrapDeg(v);
      else if (PARAM_PATHS.has(k.path)) v = clamp(v, spec.min ?? -Infinity, spec.max ?? Infinity);
      this.store.updateKeyframe(k.id, { value: round2(v) }, { coalesce: `kfv:${k.id}` });
      announce(spokenValue(k.path, round2(v), this.store.scene));
    } else if (typeof k.value === 'boolean') {
      this.store.updateKeyframe(k.id, { value: !k.value }, { coalesce: null });
    } else if (typeof k.value === 'string') {
      const spec = paramSpec(k.path);
      const opts = (spec.options || []).map((o) => o[0]);
      const i = opts.indexOf(k.value);
      if (i >= 0 && opts.length) {
        const nv = opts[(i + dir + opts.length) % opts.length];
        this.store.updateKeyframe(k.id, { value: nv }, { coalesce: null });
        announce(spokenValue(k.path, nv, this.store.scene));
      }
    }
  }

  _onKey(e) {
    const t = e.target;
    const tag = t?.tagName;
    if (tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (tag === 'INPUT' && t.type !== 'range' && t.type !== 'checkbox') return;
    if (e.ctrlKey || e.metaKey) return;
    if (e.altKey && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
      e.preventDefault();
      e.stopPropagation();
      this._jumpKey(e.key === 'ArrowRight' ? 1 : -1);
      return;
    }
    if (e.altKey) return;
    const key = e.key.toLowerCase();
    if (key === 'p') {
      e.preventDefault();
      e.stopPropagation();
      this.addKeyAtPlayhead();
    } else if (key === 'i' || key === 'o') {
      e.preventDefault();
      e.stopPropagation();
      const st = this.store.scene.studio;
      const loop = (st.loop || [0, this.duration]).slice();
      if (key === 'i') loop[0] = Math.min(round2(this.time), loop[1] - 0.5);
      else loop[1] = Math.max(round2(this.time), loop[0] + 0.5);
      this.store.set('studio.loop', loop, { source: 'studio', coalesce: 'loop' });
      announce(`Loop ${fmtClock(loop[0])} to ${fmtClock(loop[1])}`);
    } else if (key === 'l') {
      e.preventDefault();
      e.stopPropagation();
      this.toggleLoop();
    }
  }

  toggleLoop() {
    const on = !this.store.scene.studio.loopOn;
    this.store.set('studio.loopOn', on, { source: 'studio', coalesce: null });
    announce(on ? `Loop on, ${fmtClock(this.store.scene.studio.loop[0])} to ${fmtClock(this.store.scene.studio.loop[1])}` : 'Loop off');
  }

  /** Toggle, or force with expanded = true / false. */
  toggleCollapsed(expanded) {
    const app = document.getElementById('app');
    const collapsed = expanded === undefined ? !app?.classList.contains('timeline-collapsed') : !expanded;
    app?.classList.toggle('timeline-collapsed', collapsed);
    this.collapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    this.collapseBtn.textContent = collapsed ? '▴' : '▾';
    this.collapseBtn.setAttribute('aria-label', collapsed ? 'Expand timeline' : 'Collapse timeline');
    this.collapseBtn.title = this.collapseBtn.getAttribute('aria-label');
    requestAnimationFrame(() => {
      this._renderRuler();
      this._renderCurves();
      this.opts.onLayout?.();
    });
  }

  togglePoints(force) {
    this.pointsOpen = force === undefined ? !this.pointsOpen : !!force;
    this.pointsEl.hidden = !this.pointsOpen;
    this.pointsBtn.setAttribute('aria-expanded', this.pointsOpen ? 'true' : 'false');
    if (this.pointsOpen) this._renderPoints();
  }

  focusGoto() {
    this.toggleCollapsed(true);
    this.gotoInput.focus();
    this.gotoInput.select();
  }

  /** Storm-arc preset: rewrite the rain / wind / thunder lanes over the arc length (shapes in presets.js STORM_ARCS). */
  applyArc(name) {
    const fn = STORM_ARCS[name];
    if (!fn) return;
    const s = this.store.scene;
    const D = clamp(Math.round((s.weather.arc_minutes || 20) * 60), 30, 14400);
    const keys = fn(s, D);
    const keep = s.timeline.filter((k) => !ARC_PATHS.includes(k.path));
    this.store.beginGesture(`storm arc ${name}`);
    if (this.duration < D) {
      this.store.set('studio.duration', D, { source: 'studio' });
      this.store.set('studio.loop', [0, D], { source: 'studio' });
    }
    this.store.replaceTimeline([...keep, ...keys], { label: `Storm arc: ${name}` });
    this.store.endGesture();
    this.view = null;
    this.selectedTrack = 'weather.precipitation.rate_mm_h';
    this.selectedKey = null;
    this.render();
    this.opts.onSeek?.(0);
    announce(`Storm arc ${name} over ${fmtNum(D / 60, 0)} minutes: rain, wind and thunder lanes written. Press Play.`);
  }

  /* ------------------------------------------------------------------ render */
  render() {
    this._syncToolbar();
    this._renderRuler();
    this._renderTracks();
    if (this.pointsOpen) this._renderPoints();
  }

  _syncToolbar() {
    const s = this.store.scene;
    const st = s.studio;
    if (document.activeElement !== this.durInput) this.durInput.value = String(Math.round(this.duration));
    this.loopBtn.setAttribute('aria-pressed', st.loopOn ? 'true' : 'false');
    const hasArc = this.store.hasTrack('weather.storm_arc');
    this.root.querySelectorAll('.tl-arc').forEach((b) => b.setAttribute('aria-pressed', hasArc && s.weather.storm_arc === b.dataset.arc ? 'true' : 'false'));
    const k = this.selectedKey ? s.timeline.find((x) => x.id === this.selectedKey) : null;
    this.curveSel.disabled = !k;
    this.delKeyBtn.disabled = !k;
    if (k) this.curveSel.value = k.curve;
    this.addKeyBtn.disabled = !this.selectedTrack;
    this._fillAddSelect();
  }

  _fillAddSelect() {
    const s = this.store.scene;
    const sel = this.addSel;
    if (document.activeElement === sel) return;
    sel.textContent = '';
    sel.append(h('option', { value: '' }, '+ Add track…'));
    const groups = [
      ['Weather', PARAMS.filter((p) => p.group === 'weather')],
      ['Place & listener', PARAMS.filter((p) => p.group === 'place' || p.group === 'listener')],
      ['Engineer', PARAMS.filter((p) => p.group === 'engineer')],
    ];
    for (const [label, specs] of groups) {
      const og = h('optgroup', { label });
      for (const p of specs) if (!this.store.hasTrack(p.path)) og.append(h('option', { value: p.path }, p.label));
      if (label === 'Place & listener' && !this.store.hasTrack('listener.position')) og.append(h('option', { value: 'listener.position' }, 'Listener position'));
      if (og.children.length) sel.append(og);
    }
    const surf = h('optgroup', { label: 'Surface mix' });
    for (const k of Object.keys(s.place.surfaces || {})) {
      const p = `place.surfaces.${k}`;
      if (!this.store.hasTrack(p)) surf.append(h('option', { value: p }, `Surface: ${k.replace(/_/g, ' ')}`));
    }
    if (surf.children.length) sel.append(surf);
    const objs = h('optgroup', { label: 'Objects' });
    for (const o of s.place.objects) {
      const p = o.type === 'storm' ? `place.objects.${o.id}.params.track` : `place.objects.${o.id}.position`;
      if (!this.store.hasTrack(p)) objs.append(h('option', { value: p }, `${objectLabel(o)} · ${o.type === 'storm' ? 'track' : 'position'}`));
      if (o.type === 'window' && !this.store.hasTrack(`place.objects.${o.id}.params.open`)) objs.append(h('option', { value: `place.objects.${o.id}.params.open` }, `${objectLabel(o)} · open`));
      if (o.type === 'storm' && !this.store.hasTrack(`place.objects.${o.id}.params.flashes_per_min`)) objs.append(h('option', { value: `place.objects.${o.id}.params.flashes_per_min` }, `${objectLabel(o)} · flashes per minute`));
    }
    if (objs.children.length) sel.append(objs);
  }

  _makeRow(path) {
    const name = h('button', { type: 'button', class: 'tl-name', 'aria-pressed': 'false' });
    const val = h('output', { class: 'tl-val', 'aria-live': 'off' });
    const addHere = h('button', { type: 'button', class: 'icon-btn tl-keyhere', title: 'Add point at playhead (P)' }, '◆');
    const remove = h('button', { type: 'button', class: 'icon-btn tl-remove', title: 'Remove track' }, '×');
    const keysEl = h('div', { class: 'tl-keys' });
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', 'tl-curve-svg');
    svg.setAttribute('viewBox', '0 0 1000 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('aria-hidden', 'true');
    const poly = document.createElementNS(SVG_NS, 'polyline');
    poly.setAttribute('vector-effect', 'non-scaling-stroke');
    svg.append(poly);
    const lane = h('div', { class: 'tl-lane' }, svg, keysEl);
    const head = h('div', { class: 'tl-head' }, name, val, addHere, remove);
    const el = h('div', { class: 'tl-track', role: 'listitem', 'data-path': path }, head, lane);
    const row = { el, name, val, addHere, remove, lane, keysEl, poly, keyEls: new Map(), path };

    name.addEventListener('click', () => {
      const keys = this._getGroups().get(path) || [];
      const k = keys.find((x) => x.id === this.selectedKey) || keys[0];
      this._selectKey(k ? k.id : null, path);
    });
    addHere.addEventListener('click', () => {
      this.selectedTrack = path;
      this.addKeyAtPlayhead(path);
    });
    remove.addEventListener('click', () => {
      this.store.removeTrack(path);
      this.latched.delete(path);
      announce(`Track removed: ${trackLabel(path, this.store)}`);
    });

    let scrub = null;
    lane.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || e.target.closest('.kf')) return;
      lane.setPointerCapture(e.pointerId);
      scrub = true;
      this.selectedTrack = path;
      this._selectKey(this.selectedKey && (this._getGroups().get(path) || []).some((k) => k.id === this.selectedKey) ? this.selectedKey : null, path);
      this.opts.onSeek?.(this._snapT(this._tFromClientX(e.clientX, lane), e));
    });
    lane.addEventListener('pointermove', (e) => {
      if (scrub) this.opts.onSeek?.(this._snapT(this._tFromClientX(e.clientX, lane), e));
    });
    const endScrub = (e) => {
      scrub = null;
      try { lane.releasePointerCapture(e.pointerId); } catch (_) { /* released */ }
    };
    lane.addEventListener('pointerup', endScrub);
    lane.addEventListener('pointercancel', endScrub);
    lane.addEventListener('dblclick', (e) => {
      if (e.target.closest('.kf')) return;
      const t = this._snapT(this._tFromClientX(e.clientX, lane), e);
      const keys = this._getGroups().get(path) || [];
      const v = keys.length ? evaluateTrack(keys, t, isAngular(path)) : this.store.get(path);
      const before = [...keys].reverse().find((k) => k.at <= t);
      const id = this.store.addKeyframe({ path, at: round2(t), value: v, curve: before?.curve || defaultCurve(path) }, { coalesce: null });
      this._selectKey(id, path, { focus: true });
    });
    return row;
  }

  _makeKey(row, id) {
    const b = h('button', { type: 'button', class: 'kf', 'data-id': id });
    let drag = null;
    b.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      const k = this.store.scene.timeline.find((x) => x.id === id);
      if (!k) return;
      if (e.altKey && e.shiftKey) {
        this.store.removeKeyframe(id);
        return;
      }
      this._selectKey(id, row.path);
      b.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, at: k.at, moved: false };
      this.store.beginGesture('move point');
    });
    b.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      if (!drag.moved && Math.abs(dx) < 3) return;
      drag.moved = true;
      const r = row.lane.getBoundingClientRect();
      const [a, bb] = this._viewRange();
      const at = clamp(this._snapT(drag.at + (dx / Math.max(1, r.width)) * (bb - a), e), 0, this.duration * 4);
      this.store.updateKeyframe(id, { at: round2(at) }, { coalesce: `kf:${id}` });
      b.title = fmtClock(at);
    });
    const end = (e) => {
      if (!drag) return;
      const moved = drag.moved;
      drag = null;
      try { b.releasePointerCapture(e.pointerId); } catch (_) { /* released */ }
      this.store.endGesture();
      if (!moved) {
        const k = this.store.scene.timeline.find((x) => x.id === id);
        if (k) this.opts.onSeek?.(k.at);
      }
    };
    b.addEventListener('pointerup', end);
    b.addEventListener('pointercancel', end);
    b.addEventListener('focus', () => {
      if (this.selectedKey !== id) this._selectKey(id, row.path);
    });
    b.addEventListener('keydown', (e) => {
      const k = this.store.scene.timeline.find((x) => x.id === id);
      if (!k) return;
      if (e.altKey) return;
      let handled = true;
      switch (e.key) {
        case 'ArrowLeft':
        case 'ArrowRight': {
          const d = (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 1 : 0.1);
          this.store.updateKeyframe(id, { at: round2(Math.max(0, k.at + d)) }, { coalesce: `kfk:${id}` });
          announce(`Point at ${fmtClock(Math.max(0, k.at + d))}`);
          break;
        }
        case 'PageUp':
        case 'PageDown': {
          const d = (e.key === 'PageUp' ? 1 : -1) * 10;
          this.store.updateKeyframe(id, { at: round2(Math.max(0, k.at + d)) }, { coalesce: `kfk:${id}` });
          break;
        }
        case 'ArrowUp':
        case 'ArrowDown':
          this._nudgeKeyValue(k, e.key === 'ArrowUp' ? 1 : -1, e.shiftKey);
          break;
        case 'Delete':
        case 'Backspace':
          this.deleteSelectedKey();
          break;
        case 'Enter':
          this.opts.onSeek?.(k.at);
          break;
        case 'c':
        case 'C': {
          const order = CURVES.map((c) => c[0]);
          const next = order[(order.indexOf(k.curve) + 1) % order.length];
          this.store.updateKeyframe(id, { curve: next }, { coalesce: null });
          announce(`Curve ${next}`);
          break;
        }
        default:
          handled = false;
      }
      if (handled) {
        e.preventDefault();
        e.stopPropagation();
      }
    });
    row.keysEl.append(b);
    row.keyEls.set(id, b);
    return b;
  }

  _renderTracks() {
    const groups = this._getGroups();
    const paths = [...groups.keys()];
    for (const [p, row] of this._rows) {
      if (!groups.has(p)) {
        row.el.remove();
        this._rows.delete(p);
      }
    }
    paths.forEach((p, idx) => {
      let row = this._rows.get(p);
      if (!row) {
        row = this._makeRow(p);
        this._rows.set(p, row);
      }
      if (this.tracksEl.children[idx] !== row.el) this.tracksEl.insertBefore(row.el, this.tracksEl.children[idx] || null);
      this._fillRow(row, p, groups.get(p));
    });
    this.emptyEl.hidden = paths.length > 0;
    this.root.classList.toggle('has-tracks', paths.length > 0);
    this._queueReadouts();
  }

  _fillRow(row, path, keys) {
    const label = trackLabel(path, this.store);
    const sel = path === this.selectedTrack;
    row.name.textContent = label;
    row.name.title = path;
    row.name.setAttribute('aria-pressed', sel ? 'true' : 'false');
    row.name.setAttribute('aria-label', `Track ${label}, ${keys.length} point${keys.length === 1 ? '' : 's'}. Select track.`);
    row.addHere.setAttribute('aria-label', `Add point to ${label} at playhead`);
    row.remove.setAttribute('aria-label', `Remove track ${label}`);
    row.el.classList.toggle('selected', sel);
    row.el.classList.toggle('latched', this.latched.has(path));
    const ids = new Set(keys.map((k) => k.id));
    for (const [id, el] of row.keyEls) {
      if (!ids.has(id)) {
        el.remove();
        row.keyEls.delete(id);
      }
    }
    const discrete = keys.some((k) => typeof k.value === 'string' || typeof k.value === 'boolean');
    for (const k of keys) {
      const el = row.keyEls.get(k.id) || this._makeKey(row, k.id);
      const x = this._x(k.at);
      el.hidden = x < -0.001 || x > 1.001;
      el.style.left = `${clamp(x, 0, 1) * 100}%`;
      el.dataset.curve = k.curve;
      el.classList.toggle('selected', k.id === this.selectedKey);
      const vtxt = spokenValue(path, k.value, this.store.scene);
      el.setAttribute('aria-label', `${label} point at ${fmtClock(k.at)}: ${vtxt}, ${k.curve} curve. Arrows move it, Up/Down change the value, C cycles the curve, Delete removes.`);
      el.title = `${fmtClock(k.at)} · ${shortValue(path, k.value)} · ${k.curve}`;
      el.textContent = '';
      if (discrete) el.append(h('span', { class: 'kf-label', 'aria-hidden': 'true' }, shortValue(path, k.value)));
    }
    this._renderCurve(row, path, keys);
  }

  _renderCurves() {
    const groups = this._getGroups();
    for (const [p, row] of this._rows) if (groups.has(p)) this._renderCurve(row, p, groups.get(p));
  }

  _renderCurve(row, path, keys) {
    if (!isNumericTrack(keys)) {
      row.poly.setAttribute('points', '');
      return;
    }
    const [a, b] = this._viewRange();
    const width = Math.max(40, row.lane.clientWidth || 600);
    const n = clamp(Math.round(width / 4), 40, 300);
    const norm = normFor(path, keys);
    const ang = isAngular(path);
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const t = a + ((b - a) * i) / n;
      const v = evaluateTrack(keys, t, ang);
      const y = 100 - clamp(norm(v), 0, 1) * 90 - 5;
      pts.push(`${((i / n) * 1000).toFixed(1)},${y.toFixed(1)}`);
    }
    row.poly.setAttribute('points', pts.join(' '));
  }

  /* ------------------------------------------------------------------ points table */
  _renderPoints() {
    const box = this.pointsEl;
    const focus = document.activeElement && box.contains(document.activeElement)
      ? { id: document.activeElement.dataset.kid, field: document.activeElement.dataset.field } : null;
    box.textContent = '';
    const path = this.selectedTrack;
    const head = h('div', { class: 'tl-points-head' },
      h('h3', {}, path ? `Points: ${trackLabel(path, this.store)}` : 'Points'),
      h('button', { type: 'button', class: 'btn btn-small', onclick: () => this.addKeyAtPlayhead(), disabled: !path }, 'Add point at playhead (P)'),
      h('button', { type: 'button', class: 'icon-btn', 'aria-label': 'Close points table', onclick: () => this.togglePoints(false) }, '×'));
    box.append(head);
    if (!path) {
      box.append(h('p', { class: 'hint' }, 'Select a track to list its points.'));
      return;
    }
    const keys = this._getGroups().get(path) || [];
    const spec = paramSpec(path);
    const table = h('table', { class: 'tl-points-table' },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'Time'), h('th', { scope: 'col' }, 'Value'), h('th', { scope: 'col' }, 'Curve'), h('th', { scope: 'col' }, h('span', { class: 'sr-only' }, 'Actions')))));
    const tbody = h('tbody');
    for (const k of keys) {
      const timeIn = h('input', { type: 'text', class: 'mono', value: fmtClock(k.at), 'aria-label': `Time of point ${fmtClock(k.at)}`, 'data-kid': k.id, 'data-field': 'at', spellcheck: 'false' });
      timeIn.addEventListener('change', () => {
        const t = parseTime(timeIn.value);
        if (Number.isFinite(t)) this.store.updateKeyframe(k.id, { at: round2(Math.max(0, t)) }, { coalesce: null });
        else timeIn.value = fmtClock(k.at);
      });
      let valIn;
      if (typeof k.value === 'boolean') {
        valIn = h('input', { type: 'checkbox', 'aria-label': `Value at ${fmtClock(k.at)}`, 'data-kid': k.id, 'data-field': 'value' });
        valIn.checked = k.value;
        valIn.addEventListener('change', () => this.store.updateKeyframe(k.id, { value: valIn.checked }, { coalesce: null }));
      } else if (typeof k.value === 'string' && spec.options) {
        valIn = h('select', { 'aria-label': `Value at ${fmtClock(k.at)}`, 'data-kid': k.id, 'data-field': 'value' }, ...spec.options.map(([v, l]) => h('option', { value: v }, l)));
        valIn.value = k.value;
        valIn.addEventListener('change', () => this.store.updateKeyframe(k.id, { value: valIn.value }, { coalesce: null }));
      } else if (typeof k.value === 'number') {
        valIn = h('input', { type: 'number', step: 'any', class: 'num', value: String(round2(k.value)), 'aria-label': `Value at ${fmtClock(k.at)}${spec.unitSpoken ? ' in ' + spec.unitSpoken : spec.unit ? ' in ' + spec.unit : ''}`, 'data-kid': k.id, 'data-field': 'value' });
        valIn.addEventListener('change', () => {
          let v = parseFloat(String(valIn.value).replace(',', '.'));
          if (!Number.isFinite(v)) return;
          if (PARAM_PATHS.has(path)) v = spec.kind === 'compass' ? wrapDeg(v) : clamp(v, spec.min ?? -Infinity, spec.max ?? Infinity);
          this.store.updateKeyframe(k.id, { value: v }, { coalesce: null });
        });
      } else {
        valIn = h('input', { type: 'text', class: 'mono', value: JSON.stringify(k.value), 'aria-label': `Value at ${fmtClock(k.at)} as JSON`, 'data-kid': k.id, 'data-field': 'value', spellcheck: 'false' });
        valIn.addEventListener('change', () => {
          try {
            const v = JSON.parse(valIn.value);
            if (Array.isArray(k.value) === Array.isArray(v)) this.store.updateKeyframe(k.id, { value: v }, { coalesce: null });
          } catch (_) {
            valIn.value = JSON.stringify(k.value);
            announce('Value not valid JSON');
          }
        });
      }
      const curveIn = h('select', { 'aria-label': `Curve after ${fmtClock(k.at)}`, 'data-kid': k.id, 'data-field': 'curve' }, ...CURVES.map(([v, l]) => h('option', { value: v }, l)));
      curveIn.value = k.curve;
      curveIn.addEventListener('change', () => this.store.updateKeyframe(k.id, { curve: curveIn.value }, { coalesce: null }));
      const go = h('button', { type: 'button', class: 'icon-btn', 'aria-label': `Go to ${fmtClock(k.at)}`, title: 'Go to point', 'data-kid': k.id, 'data-field': 'go', onclick: () => this.opts.onSeek?.(k.at) }, '⇥');
      const del = h('button', { type: 'button', class: 'icon-btn', 'aria-label': `Delete point at ${fmtClock(k.at)}`, title: 'Delete point', 'data-kid': k.id, 'data-field': 'del', onclick: () => this.store.removeKeyframe(k.id) }, '×');
      const tr = h('tr', { class: k.id === this.selectedKey ? 'selected' : '' }, h('td', {}, timeIn), h('td', {}, valIn), h('td', {}, curveIn), h('td', { class: 'tl-points-actions' }, go, del));
      tbody.append(tr);
    }
    table.append(tbody);
    box.append(table);
    if (!keys.length) box.append(h('p', { class: 'hint' }, 'No points on this track.'));
    if (focus && focus.id) box.querySelector(`[data-kid="${focus.id}"][data-field="${focus.field}"]`)?.focus();
  }
}
