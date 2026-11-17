/**
 * Inspector panel: human-unit controls over the shared parameter graph.
 * With nothing selected it shows WEATHER, PLACE & LISTENER and a collapsed
 * ENGINEER section; with an object selected it shows that object's material,
 * size, params, rain exposure and impact rate.
 * Every control carries an Auto / Offset / Pinned provenance badge (shape + text),
 * Alt-click re-links, right-click (or Shift+F10) offers "Animate".
 */
import {
  PARAMS, paramSpec, SURFACE_KEYS, BIOPHONY, AMBIENT_LEVELS, INSECT_KEYS, soundValue, soundPatch, autoLevel,
} from './scene-store.js';
import {
  valueText, enhanceRange, makeSliderRole, bftFromMps, mpsFromBft, bftInfo, speeds, flashBang, kmFromFlashBang,
  fmtNum, compassWord, compassAbbr, relativeDirection, fmtHour, hourWord, rainWord, announce, clamp, wrapDeg,
} from './a11y.js';
import { MATERIALS, MATERIALS_BY_TYPE, estimateImpactRate, rainExposure, stormPosition, rangeBearing } from './presets.js';
import { objectLabel } from './library.js';

/* ------------------------------------------------------------------ dom helper */

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'html') el.innerHTML = v;
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

const idFor = (path) => `insp-${path.replace(/[^a-zA-Z0-9]+/g, '-')}`;
const LOG_STEPS = 1000;
const WMO_DETENTS = [[0.1, 'drizzle'], [0.5, ''], [2.5, 'moderate'], [10, 'heavy'], [50, 'intense'], [100, 'cloudburst']];

function toSlider(spec, v) {
  if (spec.scale === 'log') return Math.round((Math.log(v / spec.min) / Math.log(spec.max / spec.min)) * LOG_STEPS);
  if (spec.scale === 'beaufort') return Math.round(bftFromMps(v) * 10) / 10;
  return v;
}

function fromSlider(spec, x) {
  if (spec.scale === 'log') {
    const v = spec.min * Math.pow(spec.max / spec.min, x / LOG_STEPS);
    return v < 1 ? Math.round(v * 100) / 100 : v < 10 ? Math.round(v * 10) / 10 : Math.round(v);
  }
  if (spec.scale === 'beaufort') return Math.round(mpsFromBft(x) * 100) / 100;
  return x;
}

/* ------------------------------------------------------------------ inspector */

export class Inspector {
  /**
   * @param {HTMLElement} root
   * @param {{store: any, getTime: () => number, onAnimate: (path: string) => void,
   *          getLiveRate: (id: string) => number|undefined, onStrike: () => void, onFrame: () => void}} opts
   */
  constructor(root, opts) {
    this.root = root;
    this.store = opts.store;
    this.opts = opts;
    this.controls = new Map();
    this.visibility = [];
    this.lastFocusedPath = null;
    this._menu = null;
    this._dragging = false;
    this.render();
    this.store.on('change', (e) => this._onChange(e));
    this.store.on('select', () => this.render());
    document.addEventListener('pointerdown', (e) => {
      if (this._menu && !this._menu.contains(e.target)) this._closeMenu();
    });
  }

  _onChange(e) {
    if (e.kind === 'load' || e.kind === 'restore') {
      this.render();
      return;
    }
    const sel = this.store.selection;
    if (sel && e.paths.some((p) => p.startsWith(`place.objects.${sel}`))) {
      if (e.kind === 'object-remove') this.render();
      else this._refreshObject();
      return;
    }
    if (sel) {
      this._refreshObject();
      return;
    }
    if (e.paths.includes('*')) {
      this.render();
      return;
    }
    this.refreshAll();
  }

  render() {
    this._closeMenu();
    const activeId = document.activeElement && this.root.contains(document.activeElement) ? document.activeElement.id : null;
    this.root.textContent = '';
    this.controls.clear();
    this.visibility = [];
    const sel = this.store.selection;
    const o = sel ? this.store.getObject(sel) : null;
    if (o) this._renderObject(o);
    else this._renderWeather();
    this.refreshAll();
    if (activeId) document.getElementById(activeId)?.focus();
  }

  refreshAll() {
    const s = this.store.scene;
    for (const [path, c] of this.controls) c.update(this.store.get(path), s);
    for (const v of this.visibility) v.el.hidden = !v.test(s);
  }

  /* ------------------------------------------------------------------ weather */
  _renderWeather() {
    const s = this.store.scene;
    const head = h('div', { class: 'panel-head' },
      h('h2', { id: 'inspector-title', text: 'Weather' }),
      h('span', { class: 'tier-chip', text: 'Human units' }),
    );
    const name = h('input', {
      type: 'text', class: 'scene-name', value: s.name, 'aria-label': 'Scene name', maxlength: 120,
      onchange: (e) => this.store.rename(e.target.value),
    });
    this.root.append(head, name);

    const sec = (title, open = true, cls = '') => {
      const d = h('details', { class: `insp-section ${cls}`, open: open ? true : undefined });
      d.append(h('summary', {}, h('span', { text: title })));
      this.root.append(d);
      return d;
    };

    const precip = sec('Precipitation');
    precip.append(this._row(paramSpec('weather.precipitation.type')));
    precip.append(this._row(paramSpec('weather.precipitation.rate_mm_h')));
    precip.append(this._row(paramSpec('weather.precipitation.drops')));
    precip.append(this._row(paramSpec('weather.precipitation.character')));
    precip.append(this._row(paramSpec('weather.precipitation.shower_period_min')));
    precip.append(this._row(paramSpec('weather.precipitation.hail_size_mm')));

    const wind = sec('Wind');
    wind.append(this._row(paramSpec('weather.wind.speed_mps')));
    wind.append(this._row(paramSpec('weather.wind.gust_factor')));
    wind.append(this._row(paramSpec('weather.wind.from_deg')));

    const thunder = sec('Thunder');
    thunder.append(this._row(paramSpec('weather.thunder.enabled')));
    thunder.append(this._soundRow(paramSpec('weather.thunder.rumble')));
    const tbody = h('div', { class: 'thunder-body' });
    tbody.append(
      this._row(paramSpec('weather.thunder.distance_km')),
      this._row(paramSpec('weather.thunder.bearing_deg')),
      this._row(paramSpec('weather.thunder.approach_kmh')),
      this._row(paramSpec('weather.thunder.flashes_per_min')),
      this._row(paramSpec('weather.thunder.stroke_intensity')),
      h('button', { type: 'button', class: 'btn btn-strike', onclick: () => this.opts.onStrike?.(), 'aria-keyshortcuts': 'T' }, 'Strike lightning now'),
    );
    thunder.append(tbody);
    this.visibility.push({ el: tbody, test: (sc) => sc.weather.thunder.enabled });

    const air = sec('Air & time');
    for (const p of ['weather.temperature_c', 'weather.humidity_pct', 'weather.hour', 'weather.season', 'weather.storm_arc', 'weather.arc_minutes']) air.append(this._row(paramSpec(p)));

    const place = sec('Place & listener', true, 'tier-2');
    place.append(h('p', { class: 'tier-note', text: 'Where you are and how you listen' }));
    for (const p of ['place.preset', 'place.shelter']) place.append(this._row(paramSpec(p)));
    place.append(this._surfaceMix());
    for (const p of ['place.window_open', 'place.wipers', 'listener.height_m', 'listener.heading_deg', 'listener.render', 'mode']) place.append(this._row(paramSpec(p)));

    const sounds = sec('Sounds of the place', true, 'tier-2 insp-sounds');
    sounds.append(h('p', { class: 'tier-note', text: 'Life, water and wind around you. Auto lets the ecology decide from season, hour, temperature and rain.' }));
    sounds.append(this._insectMasterNote());
    for (const [k] of [...BIOPHONY, ...AMBIENT_LEVELS]) sounds.append(this._soundRow(paramSpec(`place.biophony.${k}`)));

    const eng = sec('Engineer', false, 'tier-3');
    eng.append(h('p', { class: 'tier-note', text: 'Physics, randomness and smoothing' }));
    eng.append(this._seedRow());
    eng.append(this._nullableRow(paramSpec('weather.precipitation.mu'), 3));
    eng.append(this._nullableRow(paramSpec('weather.precipitation.dm_mm'), 1.6));
    for (const p of ['weather.precipitation.class_boundaries', 'weather.precipitation.water_quality', 'weather.wind.gust_rate_per_min']) eng.append(this._row(paramSpec(p)));
    eng.append(this._smoothingRows());
    eng.append(this._unitsRow());
  }

  /* ------------------------------------------------------------------ generic row */
  _row(spec, { extraReadout } = {}) {
    const path = spec.path;
    const id = idFor(path);
    const row = h('div', { class: `row row-${spec.kind}`, 'data-path': path });
    const label = h('label', { for: id, class: 'row-label', text: spec.label });
    const badge = h('button', { type: 'button', class: 'badge', 'data-state': 'auto' });
    badge.addEventListener('click', (e) => {
      if (e.altKey) this.store.relink(path);
      else {
        const st = this.store.cycleOverride(path);
        announce(`${spec.label}: ${st}`);
      }
    });
    label.addEventListener('click', (e) => {
      if (e.altKey) {
        e.preventDefault();
        this.store.relink(path);
        announce(`${spec.label} re-linked to Auto`);
      }
    });
    const headEl = h('div', { class: 'row-head' }, label, badge);
    const ctl = h('div', { class: 'row-ctl' });
    const read = h('div', { class: 'row-read', 'aria-hidden': 'true' });
    row.append(headEl, ctl);
    let update;
    switch (spec.kind) {
      case 'enum': update = this._enumCtl(spec, ctl, id); break;
      case 'bool': update = this._boolCtl(spec, ctl, id); break;
      case 'compass': update = this._compassCtl(spec, ctl, id, read); break;
      case 'hour':
      case 'number':
      default: update = this._numberCtl(spec, ctl, id, read); break;
    }
    row.append(read);
    if (spec.tag) row.append(h('p', { class: 'row-tag', text: spec.tag }));
    if (spec.help) row.append(h('p', { class: 'row-help', text: spec.help }));
    if (extraReadout) row.append(extraReadout);
    row.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this._openMenu(spec, e.clientX, e.clientY, row);
    });
    row.addEventListener('keydown', (e) => {
      if ((e.key === 'F10' && e.shiftKey) || e.key === 'ContextMenu') {
        e.preventDefault();
        const r = row.getBoundingClientRect();
        this._openMenu(spec, r.left + 20, r.top + 20, row);
      }
    });
    row.addEventListener('focusin', () => { this.lastFocusedPath = path; });
    if (spec.showIf) this.visibility.push({ el: row, test: spec.showIf });
    this.controls.set(path, {
      update: (v, s) => {
        update(v, s);
        const d = this.store.describeOverride(path);
        badge.dataset.state = d.state;
        badge.textContent = d.text;
        badge.title = d.title;
        badge.setAttribute('aria-label', `${spec.label} provenance: ${d.text}. Click to cycle Auto, Offset, Pinned; Alt-click re-links.`);
        const drv = this.store.driverValue(path);
        row.dataset.ghost = d.state === 'pinned' && drv !== undefined ? valueText(spec, drv, s) : '';
      },
      row,
    });
    return row;
  }

  _commit(path, value, opts = {}) {
    this.store.set(path, value, { coalesce: path, ...opts });
  }

  _gestureOn(el, path) {
    el.addEventListener('pointerdown', () => {
      this._dragging = true;
      this.store.beginGesture(path);
      const end = () => {
        this._dragging = false;
        this.store.endGesture();
        window.removeEventListener('pointerup', end);
        window.removeEventListener('pointercancel', end);
      };
      window.addEventListener('pointerup', end);
      window.addEventListener('pointercancel', end);
    });
  }

  _numberCtl(spec, ctl, id, read) {
    const path = spec.path;
    const isLog = spec.scale === 'log';
    const isBft = spec.scale === 'beaufort';
    const range = h('input', {
      type: 'range', id, class: 'slider',
      min: isLog ? 0 : isBft ? 0 : spec.min,
      max: isLog ? LOG_STEPS : isBft ? 12 : spec.max,
      step: isLog ? 1 : isBft ? 0.1 : spec.step ?? 'any',
      'aria-label': spec.label,
    });
    const num = h('input', { type: 'number', class: 'num', step: 'any', 'aria-label': `${spec.label} value${spec.unit ? ' in ' + (spec.unitSpoken || spec.unit) : ''}` });
    const unit = h('span', { class: 'unit', text: isBft ? (this.store.scene.studio.units === 'imperial' ? 'mph' : this.store.scene.studio.units === 'knots' ? 'kn' : 'km/h') : spec.unit || '' });
    const scene = () => this.store.scene;
    const vt = (x) => valueText(spec, fromSlider(spec, x), scene());
    enhanceRange(range, { coarse: isLog ? 100 : isBft ? 1 : undefined, valueTextFn: vt });
    this._gestureOn(range, path);
    range.addEventListener('input', () => this._commit(path, fromSlider(spec, parseFloat(range.value))));
    const speedFactor = () => ({ imperial: 2.23694, knots: 1.94384 }[this.store.scene.studio.units] ?? 3.6);
    num.addEventListener('change', () => {
      let v = parseFloat(String(num.value).replace(',', '.'));
      if (!Number.isFinite(v)) return;
      if (isBft) v = v / speedFactor();
      v = clamp(v, spec.min, spec.max);
      this._commit(path, v, { coalesce: null });
    });
    const wrap = h('div', { class: 'ctl-line' }, range, h('span', { class: 'num-wrap' }, num, unit));
    ctl.append(wrap);

    if (isLog && path === 'weather.precipitation.rate_mm_h') {
      const ticks = h('div', { class: 'detents', 'aria-hidden': 'true' });
      for (const [v, lab] of WMO_DETENTS) {
        const x = (Math.log(v / spec.min) / Math.log(spec.max / spec.min)) * 100;
        // the slider is narrow, so the closely spaced heavy/intense ticks carry their names as tooltips only
        const shown = lab === 'drizzle' || lab === 'moderate' || lab === 'cloudburst' ? lab : '';
        ticks.append(h('span', { class: 'detent', style: { left: `${x}%` }, title: lab ? `${lab}: ${v} mm/h` : `${v} mm/h`, text: shown }));
      }
      ctl.append(ticks);
    }
    let caption = null;
    if (isBft) {
      caption = h('p', { class: 'bft-caption', 'aria-live': 'polite' });
      ctl.append(caption);
    }
    let fbInput = null;
    if (path === 'weather.thunder.distance_km') {
      fbInput = h('input', { type: 'number', class: 'num num-fb', step: '0.1', min: 0.5, max: 80, 'aria-label': 'Flash-to-bang seconds' });
      fbInput.addEventListener('change', () => {
        const sec = parseFloat(fbInput.value);
        if (!Number.isFinite(sec)) return;
        this._commit(path, clamp(kmFromFlashBang(sec, this.store.scene.weather.temperature_c), spec.min, spec.max), { coalesce: null });
      });
      ctl.append(h('div', { class: 'fb-line' }, h('span', { class: 'fb-label', text: 'Flash-to-bang' }), fbInput, h('span', { class: 'unit', text: 's' })));
    }

    return (v, s) => {
      if (v === null || v === undefined) v = spec.min;
      if (!(this._dragging && document.activeElement === range)) range.value = String(toSlider(spec, v));
      range.setAttribute('aria-valuetext', valueText(spec, v, s));
      if (document.activeElement !== num) {
        num.value = isBft ? fmtNum(v * speedFactor(), 0) : fmtNum(v, spec.step && spec.step < 0.1 ? 2 : undefined);
      }
      unit.textContent = isBft ? ({ imperial: 'mph', knots: 'kn' }[s.studio.units] || 'km/h') : spec.unit || '';
      read.textContent = this._readout(spec, v, s);
      if (caption) {
        const b = bftFromMps(v);
        const info = bftInfo(b);
        caption.textContent = `Force ${fmtNum(b, 1)}, ${info.name}: ${info.land}`;
      }
      if (fbInput && document.activeElement !== fbInput) fbInput.value = fmtNum(flashBang(v, s.weather.temperature_c), 1);
    };
  }

  _readout(spec, v, s) {
    switch (spec.path) {
      case 'weather.precipitation.rate_mm_h': {
        const lab = rainWord(v, s.studio.labels || 'wmo');
        return `${fmtNum(v)} mm/h · ${lab}`;
      }
      case 'weather.wind.speed_mps': {
        const sp = speeds(v);
        return `${sp.kmh} km/h · ${sp.mph} mph · ${sp.ms} m/s · ${sp.kn} kn`;
      }
      case 'weather.thunder.distance_km':
        return `${fmtNum(v)} km · ${fmtNum(flashBang(v, s.weather.temperature_c), 1)} s flash-to-bang at ${fmtNum(s.weather.temperature_c, 0)} °C`;
      case 'weather.temperature_c':
        return `${fmtNum(v, 1)} °C · ${fmtNum(v * 1.8 + 32, 0)} °F · sound ${fmtNum(331.3 * Math.sqrt(1 + v / 273.15), 1)} m/s`;
      case 'weather.humidity_pct': {
        const T = s.weather.temperature_c;
        const a = 17.625;
        const b = 243.04;
        const g = Math.log(Math.max(1, v) / 100) + (a * T) / (b + T);
        const dew = (b * g) / (a - g);
        return `${fmtNum(v, 0)} % · dew point ${fmtNum(dew, 1)} °C`;
      }
      case 'weather.hour':
        return `${fmtHour(v)} · ${hourWord(v)}`;
      case 'place.window_open':
        return v <= 0.01 ? 'closed' : `${Math.round(v * 100)} % open`;
      case 'weather.thunder.approach_kmh':
        return v > 0.5 ? `approaching ${fmtNum(v, 0)} km/h` : v < -0.5 ? `receding ${fmtNum(-v, 0)} km/h` : 'stationary';
      case 'weather.wind.gust_factor':
        return `gusts to ${speeds(s.weather.wind.speed_mps * v).kmh} km/h`;
      case 'listener.height_m': {
        return `${fmtNum(v, 1)} m · footprint 50 % within ${fmtNum(v, 1)} m, 90 % within ${fmtNum(3 * v, 1)} m`;
      }
      default:
        return spec.format ? spec.format(v) : `${fmtNum(v)}${spec.unit ? ' ' + spec.unit : ''}`;
    }
  }

  _enumCtl(spec, ctl, id) {
    const path = spec.path;
    if ((spec.options || []).length <= 4) {
      const group = h('div', { class: 'segmented', role: 'radiogroup', id, 'aria-label': spec.label });
      const inputs = spec.options.map(([val, lab], i) => {
        const rid = `${id}-${i}`;
        const inp = h('input', { type: 'radio', name: id, id: rid, value: val });
        inp.addEventListener('change', () => {
          if (inp.checked) this._commit(path, val, { coalesce: null });
        });
        group.append(inp, h('label', { for: rid, text: lab }));
        return inp;
      });
      ctl.append(group);
      return (v) => inputs.forEach((inp) => { inp.checked = inp.value === v; });
    }
    const sel = h('select', { id, 'aria-label': spec.label });
    for (const [val, lab] of spec.options) sel.append(h('option', { value: val, text: lab }));
    sel.addEventListener('change', () => this._commit(path, sel.value, { coalesce: null }));
    ctl.append(sel);
    return (v) => {
      if (document.activeElement !== sel) sel.value = v;
    };
  }

  _boolCtl(spec, ctl, id) {
    const inp = h('input', { type: 'checkbox', id, role: 'switch', class: 'switch', 'aria-label': spec.label });
    inp.addEventListener('change', () => this._commit(spec.path, inp.checked, { coalesce: null }));
    ctl.append(inp, h('span', { class: 'switch-text', 'aria-hidden': 'true' }));
    return (v) => {
      inp.checked = !!v;
      inp.nextSibling.textContent = v ? 'On' : 'Off';
    };
  }

  _compassCtl(spec, ctl, id, read) {
    const path = spec.path;
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '-60 -60 120 120');
    svg.setAttribute('class', 'compass');
    svg.setAttribute('id', id);
    svg.setAttribute('focusable', 'false');
    const ring = document.createElementNS(ns, 'circle');
    ring.setAttribute('r', '50');
    ring.setAttribute('class', 'compass-ring');
    svg.append(ring);
    for (let i = 0; i < 16; i++) {
      const a = (i * 22.5 * Math.PI) / 180;
      const t = document.createElementNS(ns, 'line');
      const r0 = i % 4 === 0 ? 40 : 45;
      t.setAttribute('x1', String(Math.sin(a) * r0));
      t.setAttribute('y1', String(-Math.cos(a) * r0));
      t.setAttribute('x2', String(Math.sin(a) * 50));
      t.setAttribute('y2', String(-Math.cos(a) * 50));
      t.setAttribute('class', 'compass-tick');
      svg.append(t);
    }
    for (const [txt, a] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
      const tx = document.createElementNS(ns, 'text');
      const r = (a * Math.PI) / 180;
      tx.setAttribute('x', String(Math.sin(r) * 30));
      tx.setAttribute('y', String(-Math.cos(r) * 30 + 4));
      tx.setAttribute('class', 'compass-letter');
      tx.textContent = txt;
      svg.append(tx);
    }
    const heading = document.createElementNS(ns, 'path');
    heading.setAttribute('d', 'M0,-56 L5,-47 L-5,-47 Z');
    heading.setAttribute('class', 'compass-heading');
    svg.append(heading);
    const needle = document.createElementNS(ns, 'g');
    needle.setAttribute('class', 'compass-needle');
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', '0');
    line.setAttribute('y1', '-46');
    line.setAttribute('x2', '0');
    line.setAttribute('y2', '0');
    const head = document.createElementNS(ns, 'circle');
    head.setAttribute('cx', '0');
    head.setAttribute('cy', '-46');
    head.setAttribute('r', '6');
    needle.append(line, head);
    svg.append(needle);
    const hub = document.createElementNS(ns, 'circle');
    hub.setAttribute('r', '3');
    hub.setAttribute('class', 'compass-hub');
    svg.append(hub);

    const wrap = h('div', { class: 'compass-wrap' });
    wrap.append(svg);
    const num = h('input', { type: 'number', class: 'num', min: 0, max: 359, step: 1, 'aria-label': `${spec.label} in degrees` });
    num.addEventListener('change', () => {
      const v = parseFloat(num.value);
      if (Number.isFinite(v)) this._commit(path, wrapDeg(v), { coalesce: null });
    });
    const side = h('div', { class: 'compass-side' }, h('span', { class: 'num-wrap' }, num, h('span', { class: 'unit', text: '°' })));
    wrap.append(side);
    ctl.append(wrap);

    const get = () => this.store.get(path) ?? 0;
    const set = (v) => this._commit(path, Math.round(wrapDeg(v)));
    const refreshAria = makeSliderRole(svg, {
      label: spec.label, min: 0, max: 360, get, set, fine: 1, coarse: 22.5, wrap: true,
      valueTextFn: (v) => valueText(spec, v, this.store.scene),
    });
    const angleFrom = (e) => {
      const r = svg.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      return wrapDeg((Math.atan2(dx, -dy) * 180) / Math.PI);
    };
    svg.addEventListener('pointerdown', (e) => {
      svg.setPointerCapture(e.pointerId);
      svg.focus();
      this.store.beginGesture(path);
      const snap = (a) => (e.shiftKey ? Math.round(a / 22.5) * 22.5 : a);
      set(snap(angleFrom(e)));
      const move = (ev) => set(ev.shiftKey ? Math.round(angleFrom(ev) / 22.5) * 22.5 : angleFrom(ev));
      const up = () => {
        svg.removeEventListener('pointermove', move);
        svg.removeEventListener('pointerup', up);
        svg.removeEventListener('pointercancel', up);
        this.store.endGesture();
      };
      svg.addEventListener('pointermove', move);
      svg.addEventListener('pointerup', up);
      svg.addEventListener('pointercancel', up);
    });

    return (v, s) => {
      const deg = v ?? 0;
      needle.setAttribute('transform', `rotate(${deg})`);
      const hd = s.listener.heading_deg || 0;
      heading.setAttribute('transform', `rotate(${path === 'listener.heading_deg' ? deg : hd})`);
      heading.style.display = path === 'listener.heading_deg' ? 'none' : '';
      if (document.activeElement !== num) num.value = String(Math.round(deg));
      refreshAria();
      const abbr = compassAbbr(deg);
      if (path === 'weather.wind.from_deg') read.textContent = `From ${abbr} (${compassWord(deg)}) · blowing towards ${compassAbbr(deg + 180)} · ${relativeDirection(deg, hd)} of you`;
      else if (path === 'weather.thunder.bearing_deg') read.textContent = `Storm to the ${compassWord(deg)} · ${relativeDirection(deg, hd)}`;
      else read.textContent = `Facing ${compassWord(deg)} (${abbr})`;
    };
  }

  _surfaceMix() {
    const box = h('div', { class: 'row surface-mix', 'data-path': 'place.surfaces' });
    box.append(h('div', { class: 'row-head' }, h('span', { class: 'row-label', id: 'surface-mix-label', text: 'Surface mix' }), h('span', { class: 'sum-chip', 'aria-live': 'polite' })));
    const list = h('div', { class: 'mix-list', role: 'group', 'aria-labelledby': 'surface-mix-label' });
    const addSel = h('select', { 'aria-label': 'Add a surface to the mix' });
    box.append(list, addSel);

    const build = () => {
      list.textContent = '';
      const mix = this.store.scene.place.surfaces || {};
      const keys = Object.keys(mix).filter((k) => mix[k] > 0);
      for (const k of keys) {
        const id = idFor(`place.surfaces.${k}`);
        const r = h('input', { type: 'range', id, min: 0, max: 100, step: 1, class: 'slider', 'aria-label': `${k.replace('_', ' ')} share` });
        enhanceRange(r, { coarse: 10, valueTextFn: (v) => `${Math.round(v)} percent of the ground` });
        this._gestureOn(r, 'place.surfaces');
        r.addEventListener('input', () => this._setShare(k, parseFloat(r.value)));
        const pct = h('span', { class: 'mix-pct', 'aria-hidden': 'true' });
        const rm = h('button', { type: 'button', class: 'icon-btn', 'aria-label': `Remove ${k.replace('_', ' ')}`, text: '×', onclick: () => this._setShare(k, 0) });
        const sw = h('span', { class: 'swatch', 'aria-hidden': 'true', style: { '--swatch': MATERIALS[k]?.color || MATERIALS[k === 'puddle' ? 'puddle' : 'concrete'].color } });
        const rowEl = h('div', { class: 'mix-row', 'data-key': k }, sw, h('label', { for: id, text: (MATERIALS[k]?.label || k).replace('_', ' ') }), r, pct, rm);
        rowEl.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          this._openMenu(paramSpec(`place.surfaces.${k}`), e.clientX, e.clientY, rowEl);
        });
        list.append(rowEl);
      }
      addSel.textContent = '';
      addSel.append(h('option', { value: '', text: '+ Add surface…' }));
      for (const k of SURFACE_KEYS) if (!keys.includes(k)) addSel.append(h('option', { value: k, text: (MATERIALS[k]?.label || k).replace('_', ' ') }));
    };
    addSel.addEventListener('change', () => {
      if (addSel.value) this._setShare(addSel.value, 15);
    });
    build();
    this.controls.set('place.surfaces', {
      update: (mix) => {
        const keys = Object.keys(mix || {}).filter((k) => mix[k] > 0);
        const shown = [...list.querySelectorAll('.mix-row')].map((r) => r.dataset.key);
        if (keys.join() !== shown.join()) build();
        let sum = 0;
        list.querySelectorAll('.mix-row').forEach((rowEl) => {
          const k = rowEl.dataset.key;
          const v = mix[k] || 0;
          sum += v;
          const r = rowEl.querySelector('input');
          if (document.activeElement !== r) r.value = String(Math.round(v));
          r.setAttribute('aria-valuetext', `${Math.round(v)} percent of the ground`);
          rowEl.querySelector('.mix-pct').textContent = `${Math.round(v)} %`;
        });
        box.querySelector('.sum-chip').textContent = `Σ ${Math.round(sum)} %`;
      },
      row: box,
    });
    return box;
  }

  /** Change one share and rescale the others so the mix sums to 100. */
  _setShare(key, value) {
    const mix = { ...(this.store.scene.place.surfaces || {}) };
    const v = clamp(value, 0, 100);
    const others = Object.keys(mix).filter((k) => k !== key && mix[k] > 0);
    const rest = others.reduce((a, k) => a + mix[k], 0);
    const target = 100 - v;
    if (others.length === 0) {
      mix[key] = 100;
    } else {
      for (const k of others) mix[k] = rest > 0 ? (mix[k] / rest) * target : target / others.length;
      mix[key] = v;
    }
    for (const k of Object.keys(mix)) {
      mix[k] = Math.round(mix[k] * 10) / 10;
      if (mix[k] <= 0) delete mix[k];
    }
    this.store.set('place.surfaces', mix, { coalesce: 'place.surfaces' });
  }

  _seedRow() {
    const id = 'insp-seed';
    const inp = h('input', { type: 'text', id, class: 'mono seed', pattern: '[0-9a-fA-F]{1,13}', spellcheck: 'false', 'aria-label': 'Seed in hexadecimal' });
    inp.addEventListener('change', () => {
      const v = parseInt(inp.value.replace(/^0x/i, ''), 16);
      if (Number.isFinite(v)) this.store.set('seed', v % 2 ** 52, { coalesce: null });
    });
    const btn = h('button', {
      type: 'button', class: 'btn btn-small', text: 'New seed',
      onclick: () => {
        const a = new Uint32Array(2);
        crypto.getRandomValues(a);
        const v = (a[0] % 2 ** 20) * 2 ** 32 + a[1];
        this.store.set('seed', v, { coalesce: null });
        announce(`New seed ${v.toString(16)}`);
      },
    });
    const row = h('div', { class: 'row' }, h('div', { class: 'row-head' }, h('label', { for: id, class: 'row-label', text: 'Seed (hex)' })), h('div', { class: 'ctl-line' }, inp, btn),
      h('p', { class: 'row-help', text: 'The seed fixes the random detail: the same scene and seed give the same render.' }));
    this.controls.set('seed', { update: (v) => { if (document.activeElement !== inp) inp.value = (v >>> 0 === v ? v.toString(16) : Math.floor(v).toString(16)).toUpperCase(); }, row });
    return row;
  }

  /**
   * Compact sound row: Auto checkbox (living sounds and cloud rumble) plus a 0..1 level
   * slider where 0 is off. Writes go through soundPatch() so a legacy insects master is released.
   */
  _soundRow(spec) {
    const path = spec.path;
    const key = path === 'weather.thunder.rumble' ? 'rumble' : path.split('.').pop();
    const id = idFor(path);
    const helpId = `${id}-help`;
    const hasAuto = spec.kind === 'autolevel';
    const row = h('div', { class: 'row row-sound', 'data-path': path });
    const head = h('div', { class: 'row-head' }, h('label', { for: id, class: 'row-label', text: spec.label }));
    const set = (v, coalesce = path) => this.store.setMany(soundPatch(this.store.scene, key, v), { coalesce });
    let auto = null;
    if (hasAuto) {
      auto = h('input', { type: 'checkbox', id: `${id}-auto`, 'aria-describedby': helpId });
      auto.addEventListener('change', () => {
        set(auto.checked ? 'auto' : spec.onLevel ?? 0.5, null);
        announce(`${spec.label}: ${auto.checked ? 'automatic' : 'set by hand'}`);
      });
      head.append(h('label', { for: auto.id, class: 'auto-toggle', title: 'Let the ecology decide' }, auto, ' Auto'));
    }
    const range = h('input', { type: 'range', id, class: 'slider', min: 0, max: 1, step: 0.01, 'aria-describedby': helpId });
    const out = h('output', { class: 'sound-read', for: id, 'aria-hidden': 'true' });
    const words = (v) => (v === 'auto' ? 'automatic' : v <= 0 ? 'off' : `${Math.round(v * 100)} percent`);
    enhanceRange(range, { coarse: 0.1, valueTextFn: (x) => words(x) });
    this._gestureOn(range, path);
    range.addEventListener('input', () => set(Math.round(parseFloat(range.value) * 100) / 100));
    row.append(head, h('div', { class: 'row-ctl' }, h('div', { class: 'ctl-line' }, range, out)));
    if (spec.help) row.append(h('p', { class: 'row-help', id: helpId, text: spec.help }));
    row.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this._openMenu(spec, e.clientX, e.clientY, row);
    });
    row.addEventListener('keydown', (e) => {
      if ((e.key === 'F10' && e.shiftKey) || e.key === 'ContextMenu') {
        e.preventDefault();
        const r = row.getBoundingClientRect();
        this._openMenu(spec, r.left + 20, r.top + 20, row);
      }
    });
    row.addEventListener('focusin', () => { this.lastFocusedPath = path; });
    this.controls.set(path, {
      update: (_v, s) => {
        const v = soundValue(s, key);
        const isAuto = v === 'auto';
        if (auto) auto.checked = isAuto;
        row.classList.toggle('is-auto', isAuto);
        range.disabled = isAuto;
        if (!(this._dragging && document.activeElement === range)) range.value = String(isAuto ? spec.onLevel ?? 0.5 : v);
        range.setAttribute('aria-valuetext', words(v));
        out.textContent = isAuto ? 'Auto' : v <= 0 ? 'Off' : `${Math.round(v * 100)} %`;
      },
      row,
    });
    return row;
  }

  /** Shown only while an older scene's `insects` master holds crickets, katydids and cicadas. */
  _insectMasterNote() {
    const note = h('p', { class: 'row-help insect-master', hidden: true });
    this.controls.set('place.biophony.insects', {
      update: (v) => {
        const m = autoLevel(v);
        note.hidden = m === 'auto';
        if (m !== 'auto') note.textContent = `This scene sets all insects to ${m <= 0 ? 'off' : `${Math.round(m * 100)} %`}. Changing ${INSECT_KEYS.join(', ')} takes over from it.`;
      },
      row: note,
    });
    return note;
  }

  _nullableRow(spec, fallback) {
    const row = this._row(spec);
    const auto = h('input', { type: 'checkbox', id: `${idFor(spec.path)}-auto` });
    const lab = h('label', { for: auto.id, class: 'auto-toggle' }, auto, ' Auto');
    auto.addEventListener('change', () => this.store.set(spec.path, auto.checked ? null : fallback, { coalesce: null }));
    row.querySelector('.row-head').append(lab);
    const ctl = this.controls.get(spec.path);
    const prev = ctl.update;
    ctl.update = (v, s) => {
      auto.checked = v === null || v === undefined;
      row.classList.toggle('is-auto', auto.checked);
      row.querySelectorAll('.row-ctl input').forEach((i) => { i.disabled = auto.checked; });
      prev(v ?? fallback, s);
    };
    return row;
  }

  _smoothingRows() {
    const box = h('div', { class: 'row smoothing' }, h('div', { class: 'row-head' }, h('span', { class: 'row-label', text: 'Smoothing (s to 99.5 %)' })));
    const grid = h('div', { class: 'smooth-grid' });
    for (const k of ['rain', 'wind', 'distance', 'other']) {
      const id = `insp-smooth-${k}`;
      const inp = h('input', { type: 'number', id, min: 0, max: 60, step: 0.05, class: 'num' });
      inp.addEventListener('change', () => {
        const v = parseFloat(inp.value);
        if (Number.isFinite(v)) this.store.set(`studio.smoothing.${k}`, clamp(v, 0, 60), { coalesce: null });
      });
      grid.append(h('label', { for: id, text: k }), inp);
      this.controls.set(`studio.smoothing.${k}`, { update: (v) => { if (document.activeElement !== inp) inp.value = String(v ?? 0.25); }, row: box });
    }
    box.append(grid);
    return box;
  }

  _unitsRow() {
    const box = h('div', { class: 'row units' });
    const u = h('select', { id: 'insp-units', 'aria-label': 'Speed units' },
      h('option', { value: 'metric', text: 'km/h' }), h('option', { value: 'imperial', text: 'mph' }), h('option', { value: 'knots', text: 'knots' }));
    const l = h('select', { id: 'insp-labels', 'aria-label': 'Rain label set' },
      h('option', { value: 'wmo', text: 'WMO labels' }), h('option', { value: 'ams', text: 'AMS / US labels' }));
    u.addEventListener('change', () => this.store.set('studio.units', u.value, { coalesce: null }));
    l.addEventListener('change', () => this.store.set('studio.labels', l.value, { coalesce: null }));
    box.append(h('div', { class: 'row-head' }, h('span', { class: 'row-label', text: 'Display units' })), h('div', { class: 'ctl-line' }, u, l));
    this.controls.set('studio.units', { update: (v) => { u.value = v || 'metric'; }, row: box });
    this.controls.set('studio.labels', { update: (v) => { l.value = v || 'wmo'; }, row: box });
    return box;
  }

  /* ------------------------------------------------------------------ object inspector */
  _renderObject(o) {
    const store = this.store;
    const id = o.id;
    const head = h('div', { class: 'panel-head' },
      h('h2', { id: 'inspector-title', text: objectLabel(o) }),
      h('span', { class: 'tier-chip', text: o.type }),
    );
    const name = h('input', {
      type: 'text', class: 'scene-name', value: objectLabel(o), 'aria-label': 'Object name',
      onchange: (e) => store.updateObject(id, { label: e.target.value }, { coalesce: null }),
    });
    const actions = h('div', { class: 'obj-actions' },
      h('button', { type: 'button', class: 'btn btn-small', text: 'Frame', 'aria-keyshortcuts': 'F', onclick: () => this.opts.onFrame?.() }),
      h('button', { type: 'button', class: 'btn btn-small', text: 'Duplicate', 'aria-keyshortcuts': 'Control+D', onclick: () => store.duplicateObject(id) }),
      h('button', { type: 'button', class: 'btn btn-small btn-danger', text: 'Delete', 'aria-keyshortcuts': 'Delete', onclick: () => store.removeObject(id) }),
      h('button', { type: 'button', class: 'btn btn-small', text: 'Done', onclick: () => store.select(null) }),
    );
    this.root.append(head, name, actions);

    const stats = h('div', { class: 'obj-stats', 'aria-live': 'off' });
    this.root.append(stats);
    this._objStats = stats;

    const sec = h('details', { class: 'insp-section', open: true }, h('summary', {}, h('span', { text: 'Object' })));
    this.root.append(sec);

    const mats = MATERIALS_BY_TYPE[o.type] || [];
    if (mats.length > 1 || (mats.length === 1 && mats[0])) {
      const mid = 'insp-obj-material';
      const sel = h('select', { id: mid });
      for (const m of mats) sel.append(h('option', { value: m, text: MATERIALS[m]?.label || m }));
      sel.addEventListener('change', () => store.updateObject(id, { material: sel.value }, { coalesce: null }));
      sec.append(h('div', { class: 'row' }, h('div', { class: 'row-head' }, h('label', { for: mid, class: 'row-label', text: 'Material' })), sel));
      this.controls.set(`place.objects.${id}.material`, { update: (v) => { if (document.activeElement !== sel) sel.value = v; }, row: sec });
    }

    if (o.type !== 'storm') {
      sec.append(this._vecRow(o, 'position', 'Position (m)', ['x east', 'y north', 'z up'], 0.1));
      sec.append(this._numObjRow(o, 'rotation_deg', 'Rotation', '°', -360, 360, 1));
      sec.append(this._vecRow(o, 'size', 'Size (m)', ['width', 'depth', 'height'], 0.05));
    }

    const P = (key, label, unit, min, max, step, opts = {}) => sec.append(this._paramRow(o, key, label, unit, min, max, step, opts));
    switch (o.type) {
      case 'tree':
        sec.append(this._paramEnum(o, 'species', 'Species', [['deciduous', 'Deciduous (rustle)'], ['conifer', 'Conifer (whoosh)']]));
        P('height', 'Height', 'm', 2, 40, 0.5);
        break;
      case 'wire':
        P('diameter_m', 'Diameter', 'm', 0.002, 0.05, 0.001, {
          note: (ob, s) => `Aeolian tone f = 0.2·U/d ≈ ${fmtNum((0.2 * Math.max(0.1, s.weather.wind.speed_mps)) / Math.max(0.001, ob.params.diameter_m || 0.01), 0)} Hz at the current wind`,
        });
        break;
      case 'window':
        P('open', 'Open', '', 0, 1, 0.01, { note: () => 'Opening lets the exterior in (window transmission loss falls).' });
        break;
      case 'roof':
        sec.append(this._paramEnum(o, 'style', 'Style', [['gable', 'Gable'], ['porch', 'Porch (shed on posts)'], ['shed', 'Shed']]));
        P('pitch_deg', 'Pitch', '°', 2, 60, 1);
        break;
      case 'chime':
        P('tubes', 'Tubes', '', 3, 8, 1);
        break;
      case 'eave':
        P('drips_per_m', 'Drip points per metre', '/m', 0.5, 10, 0.5, { note: () => 'Integrate-and-fire drips: interval = catchment volume ÷ inflow (Tate drops 3-5.5 mm).' });
        break;
      case 'water':
        P('depth_m', 'Depth', 'm', 0.01, 10, 0.01);
        break;
      case 'umbrella':
        P('radius_m', 'Radius', 'm', 0.3, 1.5, 0.05);
        break;
      case 'gutter':
        P('length_m', 'Length', 'm', 0.5, 30, 0.5, { note: () => 'Fed by the roof it sits under: the flow builds and fades over about half a minute as the rain changes.' });
        break;
      case 'downspout':
        P('height_m', 'Height', 'm', 0.5, 20, 0.1);
        break;
      case 'storm':
        P('speed_kmh', 'Speed', 'km/h', 0, 120, 1);
        P('flashes_per_min', 'Flashes', '/min', 0, 20, 0.1);
        P('intensity', 'Stroke intensity', '', 0, 1, 0.01);
        P('radius_km', 'Cell radius', 'km', 0.5, 25, 0.5);
        sec.append(this._trackTools(o));
        break;
      default:
        break;
    }

    const anim = h('div', { class: 'obj-anim' },
      h('button', {
        type: 'button', class: 'btn btn-small', text: o.type === 'storm' ? 'Keyframe track' : 'Keyframe position',
        onclick: () => this.opts.onAnimate?.(o.type === 'storm' ? `place.objects.${id}.params.track` : `place.objects.${id}.position`),
      }),
    );
    this.root.append(anim);
    this._refreshObject();
  }

  _vecRow(o, key, label, axes, step) {
    const id = o.id;
    const box = h('fieldset', { class: 'row vec-row' }, h('legend', { class: 'row-label', text: label }));
    const inputs = axes.map((ax, i) => {
      const inp = h('input', { type: 'number', step, class: 'num', 'aria-label': `${label} ${ax}` });
      inp.addEventListener('change', () => {
        const cur = this.store.getObject(id);
        if (!cur) return;
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) return;
        const arr = cur[key].slice();
        arr[i] = key === 'size' ? Math.max(0.01, v) : v;
        const patch = { [key]: arr };
        if (key === 'size' && cur.type === 'tree' && i === 2) patch.params = { height: arr[2] };
        this.store.updateObject(id, patch, { coalesce: null });
      });
      box.append(h('label', { class: 'vec-label' }, h('span', { text: ax.split(' ')[0] }), inp));
      return inp;
    });
    box.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      if (key === 'position') this._openMenu({ path: `place.objects.${id}.position`, label: `${objectLabel(o)} position`, kind: 'vec3' }, e.clientX, e.clientY, box);
    });
    this.controls.set(`place.objects.${id}.${key}`, {
      update: (v) => inputs.forEach((inp, i) => { if (document.activeElement !== inp && v) inp.value = fmtNum(v[i], 2); }),
      row: box,
    });
    return box;
  }

  _numObjRow(o, key, label, unit, min, max, step) {
    const id = o.id;
    const iid = `insp-obj-${key}`;
    const inp = h('input', { type: 'number', id: iid, min, max, step, class: 'num' });
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (Number.isFinite(v)) this.store.updateObject(id, { [key]: wrapDeg(v) }, { coalesce: null });
    });
    const row = h('div', { class: 'row' }, h('div', { class: 'row-head' }, h('label', { for: iid, class: 'row-label', text: label })), h('span', { class: 'num-wrap' }, inp, h('span', { class: 'unit', text: unit })));
    row.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this._openMenu({ path: `place.objects.${id}.${key}`, label: `${objectLabel(o)} ${label}`, kind: 'number', min, max }, e.clientX, e.clientY, row);
    });
    this.controls.set(`place.objects.${id}.${key}`, { update: (v) => { if (document.activeElement !== inp) inp.value = fmtNum(v ?? 0, 1); }, row });
    return row;
  }

  _paramRow(o, key, label, unit, min, max, step, { note } = {}) {
    const id = o.id;
    const path = `place.objects.${id}.params.${key}`;
    const iid = idFor(path);
    const range = h('input', { type: 'range', id: iid, min, max, step, class: 'slider', 'aria-label': label });
    const num = h('input', { type: 'number', min, max, step, class: 'num', 'aria-label': `${label} value` });
    enhanceRange(range, { valueTextFn: (v) => `${fmtNum(v)} ${unit}` });
    this._gestureOn(range, path);
    const commit = (v) => this.store.updateObject(id, { params: { [key]: clamp(v, min, max) } }, { coalesce: path });
    range.addEventListener('input', () => commit(parseFloat(range.value)));
    num.addEventListener('change', () => {
      const v = parseFloat(num.value);
      if (Number.isFinite(v)) commit(v);
    });
    const noteEl = note ? h('p', { class: 'row-help' }) : null;
    const row = h('div', { class: 'row' },
      h('div', { class: 'row-head' }, h('label', { for: iid, class: 'row-label', text: label })),
      h('div', { class: 'ctl-line' }, range, h('span', { class: 'num-wrap' }, num, h('span', { class: 'unit', text: unit }))),
      noteEl,
    );
    row.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this._openMenu({ path, label: `${objectLabel(o)} ${label}`, kind: 'number', min, max }, e.clientX, e.clientY, row);
    });
    this.controls.set(path, {
      update: (v, s) => {
        const val = v ?? min;
        if (document.activeElement !== range) range.value = String(val);
        if (document.activeElement !== num) num.value = fmtNum(val, step < 0.01 ? 3 : step < 1 ? 2 : 0);
        range.setAttribute('aria-valuetext', `${fmtNum(val)} ${unit}`);
        if (noteEl) noteEl.textContent = note(this.store.getObject(id) || o, s);
      },
      row,
    });
    return row;
  }

  _paramEnum(o, key, label, options) {
    const id = o.id;
    const path = `place.objects.${id}.params.${key}`;
    const iid = idFor(path);
    const sel = h('select', { id: iid });
    for (const [v, l] of options) sel.append(h('option', { value: v, text: l }));
    sel.addEventListener('change', () => {
      const patch = { params: { [key]: sel.value } };
      if (key === 'species') patch.material = sel.value === 'conifer' ? 'needles' : 'leaves';
      this.store.updateObject(id, patch, { coalesce: null });
    });
    const row = h('div', { class: 'row' }, h('div', { class: 'row-head' }, h('label', { for: iid, class: 'row-label', text: label })), sel);
    this.controls.set(path, { update: (v) => { if (document.activeElement !== sel) sel.value = v ?? options[0][0]; }, row });
    return row;
  }

  _trackTools(o) {
    const id = o.id;
    const store = this.store;
    const box = h('div', { class: 'row track-tools' });
    const info = h('p', { class: 'row-help track-info' });
    const get = () => store.getObject(id);
    box.append(
      h('div', { class: 'row-head' }, h('span', { class: 'row-label', text: 'Track' })),
      info,
      h('div', { class: 'ctl-line wrap' },
        h('button', { type: 'button', class: 'btn btn-small', text: 'Reverse', onclick: () => { const ob = get(); if (ob) store.updateObject(id, { params: { track: ob.params.track.slice().reverse() } }, { coalesce: null }); } }),
        h('button', {
          type: 'button', class: 'btn btn-small', text: 'Add point',
          onclick: () => {
            const ob = get();
            if (!ob) return;
            const tr = ob.params.track;
            const a = tr[tr.length - 2] || tr[0];
            const b = tr[tr.length - 1];
            const next = [Math.round(b[0] + (b[0] - a[0]) * 0.5), Math.round(b[1] + (b[1] - a[1]) * 0.5)];
            store.updateObject(id, { params: { track: [...tr, next] } }, { coalesce: null });
          },
        }),
        h('button', { type: 'button', class: 'btn btn-small', text: 'Remove last', onclick: () => { const ob = get(); if (ob && ob.params.track.length > 2) store.updateObject(id, { params: { track: ob.params.track.slice(0, -1) } }, { coalesce: null }); } }),
        h('button', {
          type: 'button', class: 'btn btn-small', text: 'Aim at listener',
          onclick: () => {
            const ob = get();
            if (!ob) return;
            const L = store.scene.listener.position;
            const st = ob.params.track[0];
            const dx = L[0] - st[0];
            const dy = L[1] - st[1];
            store.updateObject(id, { params: { track: [st, [Math.round(L[0]), Math.round(L[1])], [Math.round(L[0] + dx), Math.round(L[1] + dy)]] } }, { coalesce: null });
          },
        }),
      ),
      h('p', { class: 'row-help', text: 'Drag the orange handles in the viewport (Top view is easiest). Double-click the ground to append a point, Alt-click a handle to delete it.' }),
    );
    this._trackInfo = info;
    return box;
  }

  /** Live stats for the selected object (called on changes and by main at ~4 Hz). */
  updateLive() {
    const sel = this.store.selection;
    if (!sel || !this._objStats) return;
    const o = this.store.getObject(sel);
    if (!o) return;
    const s = this.store.scene;
    if (o.type === 'storm') {
      const t = this.opts.getTime?.() ?? 0;
      const [x, y] = stormPosition(o, t);
      const rb = rangeBearing(s, x, y);
      const fb = flashBang(rb.km, s.weather.temperature_c);
      this._objStats.textContent = '';
      this._objStats.append(
        statChip('Distance', `${fmtNum(rb.km, 1)} km`),
        statChip('Bearing', `${compassAbbr(rb.bearing)} ${Math.round(rb.bearing)}°`),
        statChip('Flash-to-bang', `${fmtNum(fb, 1)} s`),
      );
      if (this._trackInfo) {
        const tr = o.params.track || [];
        let len = 0;
        for (let i = 1; i < tr.length; i++) len += Math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1]);
        const dur = len / Math.max(0.1, (o.params.speed_kmh ?? 25) / 3.6);
        this._trackInfo.textContent = `${tr.length} points · ${fmtNum(len / 1000, 1)} km · crosses in ${fmtNum(dur / 60, 1)} min at ${fmtNum(o.params.speed_kmh ?? 25, 0)} km/h`;
      }
      return;
    }
    const exp = rainExposure(o, s);
    const est = estimateImpactRate(o, s);
    const live = this.opts.getLiveRate?.(o.id);
    this._objStats.textContent = '';
    this._objStats.append(
      statChip('Rain exposure', `${Math.round(exp * 100)} %`),
      statChip('Impacts (est.)', `${fmtRate(est)}/s`),
      statChip('Live', live !== undefined ? `${fmtRate(live)}/s` : '–'),
    );
  }

  _refreshObject() {
    const sel = this.store.selection;
    if (!sel) return;
    const o = this.store.getObject(sel);
    if (!o) {
      this.render();
      return;
    }
    const s = this.store.scene;
    for (const [path, c] of this.controls) c.update(this.store.get(path), s);
    this.updateLive();
  }

  /* ------------------------------------------------------------------ context menu */
  _openMenu(spec, x, y, anchor) {
    this._closeMenu();
    const path = spec.path;
    const t = this.opts.getTime?.() ?? 0;
    const mm = Math.floor(t / 60);
    const ss = Math.floor(t % 60);
    const stamp = `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
    const items = [
      [`Animate · add keyframe at ${stamp}`, () => this.opts.onAnimate?.(path)],
    ];
    if (PARAMS.some((p) => p.path === path)) {
      items.push(['Pin (ignore macros)', () => {
        const cur = this.store.overrideOf(path).state;
        if (cur !== 'pinned') {
          while (this.store.overrideOf(path).state !== 'pinned') this.store.cycleOverride(path);
        }
      }]);
      items.push(['Re-link to Auto (Alt-click)', () => this.store.relink(path)]);
    }
    if (this.store.hasTrack(path)) items.push(['Remove animation track', () => this.store.removeTrack(path)]);
    items.push(['Copy value', () => {
      const v = this.store.get(path);
      navigator.clipboard?.writeText(typeof v === 'object' ? JSON.stringify(v) : String(v)).catch(() => {});
      announce('Value copied');
    }]);
    const menu = h('div', { class: 'ctx-menu', role: 'menu', 'aria-label': `${spec.label} actions` });
    items.forEach(([label, fn]) => {
      const b = h('button', { type: 'button', role: 'menuitem', class: 'ctx-item', text: label, tabindex: -1 });
      b.addEventListener('click', () => {
        this._closeMenu();
        fn();
        anchor?.querySelector('input, select, button, [tabindex]')?.focus();
      });
      menu.append(b);
    });
    menu.addEventListener('keydown', (e) => {
      const btns = [...menu.querySelectorAll('[role="menuitem"]')];
      const i = btns.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); btns[(i + 1) % btns.length].focus(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); btns[(i - 1 + btns.length) % btns.length].focus(); }
      else if (e.key === 'Escape' || e.key === 'Tab') { e.preventDefault(); this._closeMenu(); anchor?.querySelector('input, select, button')?.focus(); }
    });
    document.body.append(menu);
    const r = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(x, window.innerWidth - r.width - 8)}px`;
    menu.style.top = `${Math.min(y, window.innerHeight - r.height - 8)}px`;
    this._menu = menu;
    menu.querySelector('[role="menuitem"]')?.focus();
  }

  _closeMenu() {
    if (this._menu) {
      this._menu.remove();
      this._menu = null;
    }
  }
}

function statChip(label, value) {
  return h('div', { class: 'stat' }, h('span', { class: 'stat-label', text: label }), h('span', { class: 'stat-value', text: value }));
}

function fmtRate(r) {
  if (!(r > 0)) return '0';
  if (r >= 10000) return `${fmtNum(r / 1000, 0)}k`;
  if (r >= 1000) return `${fmtNum(r / 1000, 1)}k`;
  return fmtNum(r, r < 10 ? 1 : 0);
}
