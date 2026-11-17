/**
 * Listener workspace: a full-bleed world with a minimal overlay. Three big
 * sliders drive the vibe macros through the scene-store's Auto / Offset /
 * Pinned graph (anchors in presets.js: Intensity 0 dry . 20 drizzle 0.3 mm/h .
 * 40 moderate 5 mm/h . 60 heavy 20 mm/h . 80 thunderstorm 40 mm/h . 100
 * extreme 80 mm/h; Closeness in five stops "Out in it" -> "Indoors, window
 * shut"; Calm-Wild drives gust factor 1.15-1.9, shower contrast and flash
 * rate), scene cards with the user's own presets first, "Describe your
 * weather" (the deterministic parser in describe.js), the Weather Orb
 * (radius = intensity, angle = character, with inertia), a sleep timer with
 * "Fade out as I fall asleep", the Visuals Full / Calm / Off switch and
 * "Sounds around you" (birds, insects, frogs, dogs, cloud rumble, stream,
 * chimes and wet-road traffic: Auto / On / Off with a level when On).
 */
import { PRESETS, presetScene, inferVibe, VIBE_DEFAULT } from './presets.js';
import { parseDescription } from './describe.js';
import { BIOPHONY, INSECT_KEYS, paramSpec, soundValue, soundPatch, soundPath } from './scene-store.js';
import {
  rainWord, bftFromMps, bftInfo, fmtNum, flashBang, compassWord, announce, clamp, enhanceRange,
  prefersReducedMotion, speeds,
} from './a11y.js';

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
function s(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

/* ------------------------------------------------------------------ words */

const SHELTER_WORDS = {
  open: 'Out in it',
  tree: 'Under a tree',
  umbrella: 'Under an umbrella',
  tent: 'In the tent',
  porch: 'On the porch',
  car: 'In the car',
  attic: 'Under a tin roof',
  cave: 'In a cave mouth',
  underpass: 'Under a bridge',
  room_open: 'Indoors, window open',
  room_closed: 'Indoors, window shut',
};

const CLOSE_STOPS = [
  [0, 'Out in it'], [25, 'Under a tree or umbrella'], [50, 'Porch or tent'], [75, 'Indoors, window open'], [100, 'Indoors, window shut'],
];

const MODES = [['calm', 'Relax'], ['sleep', 'Sleep'], ['focus', 'Focus'], ['asmr', 'ASMR'], ['none', 'None (full studio mix)']];
const SLEEP_CHOICES = [[0, 'Off'], [15, '15 min'], [30, '30 min'], [45, '45 min'], [60, '1 hour'], [90, '1½ hours'], [120, '2 hours']];

function characterWord(w) {
  if (w < 20) return 'calm, steady';
  if (w < 40) return 'gently gusty';
  if (w < 60) return 'gusty, on and off';
  if (w < 80) return 'blustery, showery';
  return 'wild, squally';
}

function capital(sx) {
  return sx ? sx[0].toUpperCase() + sx.slice(1) : sx;
}

/** One-line plain caption of a scene: "Moderate rain · 5 mm/h · gentle breeze from the west · thunder 8 km". */
export function sceneSummary(scene) {
  const w = scene.weather;
  const p = w.precipitation;
  const parts = [];
  if (p.type === 'none') parts.push('Rain stopped · drips only');
  else if (p.type === 'snow') parts.push(`Snow · ${fmtNum(p.rate_mm_h)} mm/h water`);
  else if (p.type === 'hail') parts.push(`Hail ${fmtNum(p.hail_size_mm, 0)} mm · ${fmtNum(p.rate_mm_h)} mm/h`);
  else if (p.type === 'drizzle') parts.push(`Drizzle · ${fmtNum(p.rate_mm_h)} mm/h`);
  else if (p.type === 'sleet' || p.type === 'freezing_rain' || p.type === 'graupel') parts.push(`${capital(p.type.replace('_', ' '))} · ${fmtNum(p.rate_mm_h)} mm/h`);
  else parts.push(`${capital(rainWord(p.rate_mm_h, scene.studio?.labels || 'wmo'))} · ${fmtNum(p.rate_mm_h)} mm/h`);
  const info = bftInfo(bftFromMps(w.wind.speed_mps));
  parts.push(`${info.name.toLowerCase()} from the ${compassWord(w.wind.from_deg)}`);
  if (w.thunder.enabled) parts.push(`thunder ${fmtNum(w.thunder.distance_km)} km (${fmtNum(flashBang(w.thunder.distance_km, w.temperature_c), 0)} s)`);
  parts.push(SHELTER_WORDS[scene.place.shelter] || scene.place.shelter);
  return parts.join(' · ');
}

/* ------------------------------------------------------------------ class */

export class ListenerMode {
  /**
   * @param {HTMLElement} root
   * @param {{store: any, play?: () => Promise<any>, isPlaying?: () => boolean, onVisuals?: (m: string) => void,
   *          getVisuals?: () => string, getVolume?: () => number, setVolume?: (pct: number) => void,
   *          onSleepFade?: (f: number) => void, onSleepEnd?: () => void, onDemo?: () => void,
   *          toast?: (msg: string, opts?: any) => void, onSavePreset?: () => void, onShare?: () => void}} opts
   */
  constructor(root, opts) {
    this.root = root;
    this.store = opts.store;
    this.opts = opts;
    this.sheet = null;
    this.sliders = {};
    this.sleep = { minutes: 0, fade: true, endAt: 0, total: 0, timer: 0 };
    this.orb = null;
    this._queued = false;
    this._build();
    this.store.on('change', (e) => this._onChange(e));
    this.store.on('presets', () => this._renderCards());
    this.sync();
  }

  /* ------------------------------------------------------------------ build */
  _build() {
    const root = this.root;
    root.textContent = '';
    root.classList.add('lm');

    this.nameEl = h('h2', { class: 'lm-scene-name', id: 'lm-scene-name' });
    this.summaryEl = h('p', { class: 'lm-summary' });
    const head = h('div', { class: 'lm-head' },
      h('div', { class: 'lm-title' }, h('p', { class: 'lm-kicker' }, 'Now playing'), this.nameEl, this.summaryEl),
      h('div', { class: 'lm-head-actions' },
        h('button', { type: 'button', class: 'btn btn-small', onclick: () => this.opts.onSavePreset?.(), 'aria-keyshortcuts': 'Control+S' }, 'Save'),
        h('button', { type: 'button', class: 'btn btn-small', onclick: () => this.opts.onShare?.() }, 'Share')),
    );

    const sliders = h('div', { class: 'lm-sliders' },
      this._bigSlider('intensity', 'How much rain', 'Rain stopped', 'Cloudburst', 1),
      this._bigSlider('closeness', 'Where you are', 'Out in it', 'Indoors, shut', 25),
      this._bigSlider('wildness', 'Calm ↔ Wild', 'Calm', 'Wild', 1),
      this._volumeSlider(),
    );

    this.sheetBtns = {};
    const sheetBtn = (key, label) => {
      const b = h('button', { type: 'button', class: 'lm-tab', 'aria-expanded': 'false', 'aria-controls': `lm-sheet-${key}`, onclick: () => this.toggleSheet(key) }, label);
      this.sheetBtns[key] = b;
      return b;
    };
    this.threeD = h('input', { type: 'checkbox', role: 'switch', id: 'lm-3d', class: 'switch' });
    this.threeD.addEventListener('change', () => {
      this.store.set('listener.render', this.threeD.checked ? 'binaural' : 'stereo', { coalesce: null });
      announce(this.threeD.checked ? '3-D sound on: for headphones' : '3-D sound off: stereo for speakers');
    });
    this.sleepChip = h('p', { class: 'lm-sleep-chip', hidden: true });
    const bar = h('div', { class: 'lm-bar', role: 'toolbar', 'aria-label': 'Listening options' },
      sheetBtn('scenes', 'Scenes'), sheetBtn('orb', 'Weather Orb'), sheetBtn('options', 'Sleep & display'), sheetBtn('sounds', 'Sounds around you'),
      h('label', { class: 'lm-switch', for: 'lm-3d' }, this.threeD, h('span', {}, '3-D sound (headphones)')),
    );

    this.panel = h('div', { class: 'lm-panel', role: 'region', 'aria-labelledby': 'lm-scene-name' }, head, sliders, bar, this.sleepChip);

    this.sheets = {
      scenes: this._buildScenesSheet(),
      orb: this._buildOrbSheet(),
      options: this._buildOptionsSheet(),
      sounds: this._buildSoundsSheet(),
    };
    root.append(this.panel, ...Object.values(this.sheets));
    root.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.sheet) {
        e.stopPropagation();
        const k = this.sheet;
        this.toggleSheet(k, false);
        this.sheetBtns[k]?.focus();
      }
    });
  }

  _bigSlider(key, label, lo, hi, step) {
    const id = `lm-${key}`;
    const input = h('input', { type: 'range', id, class: 'lm-range', min: 0, max: 100, step, value: VIBE_DEFAULT[key] });
    const cap = h('output', { class: 'lm-cap', for: id, 'aria-live': 'off' });
    const row = h('div', { class: `lm-slider lm-slider-${key}` },
      h('div', { class: 'lm-slider-head' }, h('label', { for: id, class: 'lm-label' }, label), cap),
      input,
      h('div', { class: 'lm-ends', 'aria-hidden': 'true' }, h('span', {}, lo), h('span', {}, hi)),
    );
    if (key === 'closeness') {
      const ticks = h('div', { class: 'lm-stops', 'aria-hidden': 'true' });
      for (const [v] of CLOSE_STOPS) ticks.append(h('span', { class: 'lm-stop', style: { left: `${v}%` } }));
      row.insertBefore(ticks, row.querySelector('.lm-ends'));
    }
    enhanceRange(input, { coarse: key === 'closeness' ? 25 : 10, valueTextFn: (v) => this._valueText(key, v) });
    input.addEventListener('pointerdown', () => {
      this.store.beginGesture(`vibe-${key}`);
      const end = () => {
        this.store.endGesture();
        window.removeEventListener('pointerup', end);
        window.removeEventListener('pointercancel', end);
      };
      window.addEventListener('pointerup', end);
      window.addEventListener('pointercancel', end);
    });
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      this.store.setVibe({ [key]: v });
      this._refreshCaptions();
      if (this.orb && key !== 'closeness') this._orbFromVibe();
    });
    this.sliders[key] = { input, cap };
    return row;
  }

  _volumeSlider() {
    const id = 'lm-volume';
    const input = h('input', { type: 'range', id, class: 'lm-range lm-volume', min: 0, max: 100, step: 1, value: this.opts.getVolume?.() ?? 50 });
    const cap = h('output', { class: 'lm-cap', for: id, 'aria-live': 'off' });
    enhanceRange(input, { coarse: 10, valueTextFn: (v) => `${Math.round(v)} percent` });
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      this.opts.setVolume?.(v);
      cap.textContent = `${Math.round(v)} %`;
    });
    this.volume = { input, cap };
    cap.textContent = `${Math.round(parseFloat(input.value))} %`;
    return h('div', { class: 'lm-slider lm-slider-volume' },
      h('div', { class: 'lm-slider-head' }, h('label', { for: id, class: 'lm-label' }, 'Volume'), cap), input);
  }

  _sheet(key, title, ...body) {
    const close = h('button', { type: 'button', class: 'icon-btn', 'aria-label': `Close ${title}`, onclick: () => { this.toggleSheet(key, false); this.sheetBtns[key]?.focus(); } }, '×');
    return h('section', { class: `lm-sheet lm-sheet-${key}`, id: `lm-sheet-${key}`, 'aria-label': title, hidden: true },
      h('div', { class: 'lm-sheet-head' }, h('h3', {}, title), close), ...body);
  }

  _buildScenesSheet() {
    this.describeInput = h('input', {
      type: 'search', id: 'lm-describe', class: 'lm-describe-input', autocomplete: 'off', spellcheck: 'true',
      placeholder: 'e.g. heavy rain on a tin roof, force 6 from the west, thunder 3 km away, night',
    });
    this.chipsEl = h('div', { class: 'lm-chips', 'aria-live': 'polite' });
    const form = h('form', { class: 'lm-describe', role: 'search' },
      h('label', { for: 'lm-describe', class: 'lm-label' }, 'Describe your weather'),
      h('div', { class: 'lm-describe-row' }, this.describeInput, h('button', { type: 'submit', class: 'btn btn-primary btn-small' }, 'Apply')),
      h('p', { class: 'hint' }, 'Understands English: rain words, mm/h, Beaufort names, “force 6”, “30 mph gusts”, “from the west”, thunder distance or seconds, surfaces, shelters, time and season. Also filters the cards.'),
      this.chipsEl,
    );
    let t = 0;
    this.describeInput.addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(() => {
        this._previewDescribe();
        this._renderCards();
      }, 120);
    });
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      this._applyDescribe();
    });
    this.userCards = h('ul', { class: 'lm-cards lm-cards-user', 'aria-label': 'My presets' });
    this.userEmpty = h('p', { class: 'hint lm-user-empty' }, 'Nothing saved yet. Change any scene and press Save to keep it here.');
    this.cards = h('ul', { class: 'lm-cards', 'aria-label': 'Scenes' });
    for (const ul of [this.userCards, this.cards]) ul.addEventListener('keydown', (e) => this._cardKeys(e, ul));
    this._renderCards();
    return this._sheet('scenes', 'Scenes', form,
      h('h4', { class: 'lm-sub' }, 'My presets'), this.userEmpty, this.userCards,
      h('h4', { class: 'lm-sub' }, 'Scenes'), this.cards);
  }

  _buildOrbSheet() {
    const svg = s('svg', { viewBox: '-120 -120 240 240', class: 'orb' });
    const defs = s('defs');
    const grad = s('radialGradient', { id: 'orb-grad' });
    // CSS custom properties only resolve in style, not in SVG presentation attributes
    grad.append(s('stop', { offset: '0', style: 'stop-color: var(--orb-in)' }), s('stop', { offset: '1', style: 'stop-color: var(--orb-out)' }));
    defs.append(grad);
    svg.append(defs, s('circle', { r: 100, class: 'orb-disc', fill: 'url(#orb-grad)' }));
    for (const r of [25, 50, 75]) svg.append(s('circle', { r, class: 'orb-ring' }));
    svg.append(s('line', { x1: 0, y1: -100, x2: 0, y2: 100, class: 'orb-axis' }));
    const label = (x, y, txt, cls = '') => {
      const tx = s('text', { x, y, class: `orb-label ${cls}`, 'text-anchor': 'middle' });
      tx.textContent = txt;
      svg.append(tx);
    };
    label(0, -106, 'Calm');
    label(0, 116, 'Wild');
    label(0, 4, 'Dry', 'orb-centre');
    label(-60, 4, 'drizzle', 'orb-small');
    label(-100 + 18, 4, 'downpour', 'orb-small orb-edge');
    const trail = s('circle', { r: 16, class: 'orb-trail' });
    const puck = s('circle', { r: 11, class: 'orb-puck', tabindex: 0, role: 'slider', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': 'Weather Orb: distance from the centre is how much rain, top is calm, bottom is wild. Up and Down change the rain, Left and Right change the character.' });
    svg.append(trail, puck);
    const drift = h('input', { type: 'checkbox', id: 'lm-orb-drift' });
    const readout = h('p', { class: 'lm-orb-readout', 'aria-hidden': 'true' });
    this.orb = {
      svg, puck, trail, drift, readout, x: 0, y: -0.4, vx: 0, vy: 0, tx: 0, ty: 0,
      dragging: false, raf: 0, last: 0, lastPush: 0, gesture: false, samples: [], noiseT: 0,
    };
    this._bindOrb();
    this._orbFromVibe();
    return this._sheet('orb', 'Weather Orb', h('div', { class: 'lm-orb-wrap' }, svg),
      readout,
      h('label', { class: 'lm-check', for: 'lm-orb-drift' }, drift, ' Drift (the weather wanders on its own)'),
      h('p', { class: 'hint' }, 'Drag or flick the orb: it keeps moving with a little inertia. Distance from the centre = how much rain; top = calm, bottom = wild. The three sliders are the number-field alternative.'));
  }

  _buildOptionsSheet() {
    this.sleepSel = h('select', { id: 'lm-sleep' }, ...SLEEP_CHOICES.map(([v, l]) => h('option', { value: v }, l)));
    this.sleepFade = h('input', { type: 'checkbox', id: 'lm-sleep-fade', checked: true });
    this.sleepStatus = h('p', { class: 'lm-sleep-status', 'aria-live': 'polite' }, 'Sleep timer off.');
    this.sleepSel.addEventListener('change', () => this.setSleep(parseInt(this.sleepSel.value, 10) || 0));
    this.sleepFade.addEventListener('change', () => {
      this.sleep.fade = this.sleepFade.checked;
      if (!this.sleep.fade) this.opts.onSleepFade?.(1);
    });

    this.modeSel = h('select', { id: 'lm-mode' }, ...MODES.map(([v, l]) => h('option', { value: v }, l)));
    this.modeSel.addEventListener('change', () => {
      this.store.set('mode', this.modeSel.value, { coalesce: null });
      const label = MODES.find((m) => m[0] === this.modeSel.value)?.[1];
      this.opts.toast?.(`Mode: ${label}.`);
    });

    const vis = h('div', { class: 'segmented lm-visuals', role: 'radiogroup', 'aria-label': 'Visuals' });
    this.visInputs = [['full', 'Full'], ['calm', 'Calm'], ['off', 'Off']].map(([v, l]) => {
      const id = `lm-vis-${v}`;
      const inp = h('input', { type: 'radio', name: 'lm-visuals', id, value: v });
      inp.addEventListener('change', () => {
        if (inp.checked) this.opts.onVisuals?.(v);
      });
      vis.append(inp, h('label', { for: id }, l));
      return inp;
    });
    this.setVisualsUI(this.opts.getVisuals?.() || 'full');

    return this._sheet('options', 'Sleep & display',
      h('div', { class: 'lm-field' }, h('label', { for: 'lm-sleep', class: 'lm-label' }, 'Stop after'), this.sleepSel),
      h('label', { class: 'lm-check', for: 'lm-sleep-fade' }, this.sleepFade, ' Fade out as I fall asleep (up to the last 15 minutes)'),
      this.sleepStatus,
      h('div', { class: 'lm-field' }, h('label', { for: 'lm-mode', class: 'lm-label' }, 'Mode'), this.modeSel),
      h('p', { class: 'hint' }, 'Sleep pulls thunder and wind well back, softens lightning and lowers the overall level; Focus and Relax pull thunder and wind back; ASMR brings close detail forward.'),
      h('div', { class: 'lm-field' }, h('span', { class: 'lm-label', id: 'lm-vis-label' }, 'Visuals'), vis),
      h('p', { class: 'hint' }, 'Calm softens animation and turns lightning into a gentle glow. Off stops the 3-D view (audio keeps playing). Flashes never exceed 3 per second.'));
  }

  /* ------------------------------------------------------------------ sounds around you */
  _buildSoundsSheet() {
    this.soundRows = {};
    this.soundMem = {};
    const groups = [
      ['birds', 'Birds & animals', ['birds', 'crows', 'owls', 'frogs', 'dogs']],
      ['insects', 'Insects', INSECT_KEYS],
      ['place', 'Sky, water & town', ['rumble', 'stream', 'chimes', 'traffic']],
    ];
    const body = [h('p', { class: 'hint' }, 'Auto lets nature decide from the season, the hour, the temperature and the rain. On sets a level by hand; Off silences it.')];
    for (const [gid, title, keys] of groups) {
      const hid = `lm-snd-g-${gid}`;
      const list = h('div', { class: 'lm-snd-list', role: 'group', 'aria-labelledby': hid });
      for (const k of keys) list.append(this._soundRow(k));
      body.push(h('h4', { class: 'lm-sub', id: hid }, title), list);
    }
    body.push(h('div', { class: 'lm-snd-foot' },
      h('button', { type: 'button', class: 'btn btn-small', onclick: () => this._allSoundsAuto() }, 'Set living sounds to Auto')));
    return this._sheet('sounds', 'Sounds around you', ...body);
  }

  _soundRow(key) {
    const spec = paramSpec(soundPath(key));
    const id = `lm-snd-${key}`;
    const nameId = `${id}-name`;
    const hintId = `${id}-hint`;
    const states = spec.kind === 'autolevel' ? [['auto', 'Auto'], ['on', 'On'], ['off', 'Off']] : [['on', 'On'], ['off', 'Off']];
    const seg = h('div', { class: 'segmented lm-snd-seg', role: 'radiogroup', 'aria-labelledby': nameId, 'aria-describedby': hintId });
    const radios = {};
    for (const [st, lab] of states) {
      const rid = `${id}-${st}`;
      const inp = h('input', { type: 'radio', name: id, id: rid, value: st });
      inp.addEventListener('change', () => {
        if (inp.checked) this._setSoundState(key, st);
      });
      seg.append(inp, h('label', { for: rid }, lab));
      radios[st] = inp;
    }
    const range = h('input', { type: 'range', id: `${id}-level`, class: 'lm-range lm-snd-range', min: 0.05, max: 1, step: 0.01, 'aria-label': `${spec.label} level` });
    const cap = h('output', { class: 'lm-cap', for: range.id, 'aria-hidden': 'true' });
    enhanceRange(range, { coarse: 0.1, valueTextFn: (x) => `${Math.round(x * 100)} percent` });
    range.addEventListener('pointerdown', () => {
      this.store.beginGesture(soundPath(key));
      const end = () => {
        this.store.endGesture();
        window.removeEventListener('pointerup', end);
        window.removeEventListener('pointercancel', end);
      };
      window.addEventListener('pointerup', end);
      window.addEventListener('pointercancel', end);
    });
    range.addEventListener('input', () => {
      const v = Math.round(parseFloat(range.value) * 100) / 100;
      this.soundMem[key] = v;
      cap.textContent = `${Math.round(v * 100)} %`;
      this.store.setMany(soundPatch(this.store.scene, key, v), { coalesce: soundPath(key) });
    });
    const level = h('div', { class: 'lm-snd-level', hidden: true }, range, cap);
    const row = h('div', { class: 'lm-snd-row', 'data-key': key },
      h('div', { class: 'lm-snd-head' }, h('span', { class: 'lm-snd-name', id: nameId }, spec.label), seg),
      h('p', { class: 'lm-snd-hint', id: hintId }, spec.help || ''),
      level);
    this.soundRows[key] = { spec, radios, range, cap, level, row };
    return row;
  }

  _setSoundState(key, st) {
    const spec = this.soundRows[key].spec;
    const cur = soundValue(this.store.scene, key);
    if (typeof cur === 'number' && cur > 0) this.soundMem[key] = cur;
    const v = st === 'auto' ? 'auto' : st === 'off' ? 0 : this.soundMem[key] ?? spec.onLevel ?? 0.5;
    this.store.setMany(soundPatch(this.store.scene, key, v), { coalesce: null });
    this._syncSounds();
    announce(`${spec.label}: ${st === 'auto' ? 'automatic' : st === 'off' ? 'off' : `on, ${Math.round(v * 100)} percent`}`);
  }

  _allSoundsAuto() {
    const map = { 'place.biophony.insects': 'auto', 'weather.thunder.rumble': 'auto' };
    for (const [k] of BIOPHONY) map[soundPath(k)] = 'auto';
    this.store.setMany(map, { coalesce: null });
    this._syncSounds();
    announce('Birds, animals, insects and the cloud rumble are on Auto');
  }

  _syncSounds() {
    if (!this.soundRows) return;
    const sc = this.store.scene;
    for (const [key, r] of Object.entries(this.soundRows)) {
      const v = soundValue(sc, key);
      const st = v === 'auto' ? 'auto' : v > 0 ? 'on' : 'off';
      for (const [k, inp] of Object.entries(r.radios)) inp.checked = k === st;
      r.row.dataset.state = st;
      r.level.hidden = st !== 'on';
      if (st === 'on') {
        if (document.activeElement !== r.range) r.range.value = String(v);
        r.range.setAttribute('aria-valuetext', `${Math.round(v * 100)} percent`);
        r.cap.textContent = `${Math.round(v * 100)} %`;
      }
    }
  }

  /* ------------------------------------------------------------------ sheets */
  toggleSheet(key, force) {
    const open = force === undefined ? this.sheet !== key : !!force;
    for (const [k, el] of Object.entries(this.sheets)) {
      const on = open && k === key;
      el.hidden = !on;
      this.sheetBtns[k]?.setAttribute('aria-expanded', on ? 'true' : 'false');
    }
    this.sheet = open ? key : null;
    this.root.classList.toggle('sheet-open', !!this.sheet);
    if (open) {
      if (key === 'orb') this._orbFromVibe();
      if (key === 'scenes') this._renderCards();
      if (key === 'sounds') this._syncSounds();
      requestAnimationFrame(() => {
        const first = key === 'scenes' ? this.describeInput : key === 'orb' ? this.orb.puck
          : key === 'sounds' ? this.sheets.sounds.querySelector('.lm-snd-seg input:checked') : this.sleepSel;
        first?.focus({ preventScroll: true });
      });
    }
  }

  show(on) {
    this.root.hidden = !on;
    if (on) this.sync();
  }

  /* ------------------------------------------------------------------ sync */
  _onChange(e) {
    if (e.kind === 'load' || e.kind === 'restore') {
      this.sync();
      return;
    }
    if (this._queued) return;
    this._queued = true;
    requestAnimationFrame(() => {
      this._queued = false;
      this._refreshCaptions();
      this.nameEl.textContent = this.store.scene.name;
      this.threeD.checked = this.store.scene.listener.render === 'binaural';
      if (document.activeElement !== this.modeSel) this.modeSel.value = this.store.scene.mode || 'none';
      if (this.sheet === 'sounds') this._syncSounds();
    });
  }

  _vibe() {
    const s = this.store.scene;
    const inferred = inferVibe(s);
    const v = s.vibe || {};
    return {
      intensity: v.intensity ?? inferred.intensity,
      closeness: v.closeness ?? inferred.closeness,
      wildness: v.wildness ?? inferred.wildness,
    };
  }

  sync() {
    const s = this.store.scene;
    this.nameEl.textContent = s.name;
    const v = this._vibe();
    for (const k of ['intensity', 'closeness', 'wildness']) {
      const inp = this.sliders[k].input;
      if (document.activeElement !== inp) inp.value = String(k === 'closeness' ? Math.round(v[k] / 25) * 25 : Math.round(v[k]));
    }
    this.threeD.checked = s.listener.render === 'binaural';
    this.modeSel.value = s.mode || 'none';
    this._refreshCaptions();
    this._orbFromVibe();
    this._renderCards();
    this._syncSounds();
  }

  _refreshCaptions() {
    const s = this.store.scene;
    this.summaryEl.textContent = sceneSummary(s);
    for (const k of ['intensity', 'closeness', 'wildness']) {
      const { input, cap } = this.sliders[k];
      const v = parseFloat(input.value);
      cap.textContent = this._caption(k, v);
      input.setAttribute('aria-valuetext', this._valueText(k, v));
    }
  }

  _caption(key, v) {
    const s = this.store.scene;
    const w = s.weather;
    if (key === 'intensity') {
      const p = w.precipitation;
      const rain = p.type === 'none' ? 'Rain stopped' : p.type === 'snow' ? `Snow · ${fmtNum(p.rate_mm_h)} mm/h` : `${capital(rainWord(p.rate_mm_h, s.studio?.labels || 'wmo'))} · ${fmtNum(p.rate_mm_h)} mm/h`;
      const wind = `F${Math.round(bftFromMps(w.wind.speed_mps))}`;
      return `${rain} · ${wind}${w.thunder.enabled ? ` · thunder ${fmtNum(w.thunder.distance_km)} km` : ''}`;
    }
    if (key === 'closeness') return SHELTER_WORDS[s.place.shelter] || CLOSE_STOPS.reduce((a, b) => (Math.abs(b[0] - v) < Math.abs(a[0] - v) ? b : a))[1];
    const gust = speeds(w.wind.speed_mps * (w.wind.gust_factor || 1.4)).kmh;
    return `${capital(characterWord(v))} · gusts to ${gust} km/h`;
  }

  _valueText(key, v) {
    const s = this.store.scene;
    if (key === 'intensity') {
      const p = s.weather.precipitation;
      return `${Math.round(v)}: ${p.type === 'none' ? 'rain stopped' : `${rainWord(p.rate_mm_h)}, ${fmtNum(p.rate_mm_h)} millimetres an hour`}`;
    }
    if (key === 'closeness') return `${Math.round(v)}: ${(SHELTER_WORDS[s.place.shelter] || '').toLowerCase()}`;
    return `${Math.round(v)}: ${characterWord(v)}`;
  }

  setVolumeUI(pct) {
    if (!this.volume) return;
    if (document.activeElement !== this.volume.input) this.volume.input.value = String(Math.round(pct));
    this.volume.cap.textContent = `${Math.round(pct)} %`;
    this.volume.input.setAttribute('aria-valuetext', `${Math.round(pct)} percent`);
  }

  setVisualsUI(mode) {
    for (const inp of this.visInputs || []) inp.checked = inp.value === mode;
  }

  /* ------------------------------------------------------------------ cards */
  _renderCards() {
    if (!this.cards) return;
    const q = (this.describeInput?.value || '').toLowerCase().trim();
    const words = q.split(/[^a-z0-9]+/).filter((w) => w.length >= 3);
    const score = (text) => (words.length ? words.reduce((a, w) => a + (text.includes(w) ? 1 : 0), 0) : 1);

    const user = this.store.userPresets();
    this.userCards.textContent = '';
    this.userEmpty.hidden = user.length > 0;
    user.forEach((p, i) => {
      const card = h('button', { type: 'button', class: 'lm-card lm-card-user', 'data-index': i, tabindex: i === 0 ? 0 : -1 },
        h('span', { class: 'lm-card-name' }, p.name),
        h('span', { class: 'lm-card-blurb' }, sceneSummary(normalizeLite(p.scene))),
        h('span', { class: 'lm-card-tags' }, h('span', { class: 'tag' }, 'My preset'), p.savedAt ? h('span', { class: 'tag tag-quiet' }, new Date(p.savedAt).toLocaleDateString()) : null));
      card.addEventListener('click', () => this._chooseUser(p));
      const del = h('button', { type: 'button', class: 'icon-btn lm-card-del', 'aria-label': `Delete preset ${p.name}`, title: 'Delete preset' }, '×');
      del.addEventListener('click', () => this._deleteUser(p));
      this.userCards.append(h('li', { class: 'lm-card-wrap' }, card, del));
    });

    const list = PRESETS
      .map((p) => ({ p, sc: score(`${p.name} ${p.blurb} ${p.tags.join(' ')}`.toLowerCase()) }))
      .filter((x) => x.sc > 0);
    const shown = list.length ? list.sort((a, b) => b.sc - a.sc) : PRESETS.map((p) => ({ p, sc: 1 }));
    this.cards.textContent = '';
    const current = this.store.scene.presetId;
    shown.forEach(({ p }, i) => {
      const card = h('button', {
        type: 'button', class: `lm-card${p.demo ? ' lm-card-demo' : ''}`, 'data-id': p.id, tabindex: i === 0 ? 0 : -1,
        'aria-current': p.id === current ? 'true' : undefined,
        style: { '--c1': p.hue[0], '--c2': p.hue[1] },
      },
      h('span', { class: 'lm-card-name' }, p.name),
      h('span', { class: 'lm-card-blurb' }, p.blurb),
      h('span', { class: 'lm-card-tags' }, ...p.tags.map((t) => h('span', { class: 'tag' }, t))));
      card.addEventListener('click', () => this._choosePreset(p));
      this.cards.append(h('li', { class: 'lm-card-wrap' }, card));
    });
  }

  _cardKeys(e, ul) {
    const cards = [...ul.querySelectorAll('.lm-card')];
    const i = cards.indexOf(document.activeElement);
    if (i < 0) return;
    let j = i;
    const cols = Math.max(1, cards.filter((c) => c.offsetTop === cards[0].offsetTop).length);
    switch (e.key) {
      case 'ArrowRight': j = Math.min(cards.length - 1, i + 1); break;
      case 'ArrowLeft': j = Math.max(0, i - 1); break;
      case 'ArrowDown': j = Math.min(cards.length - 1, i + cols); break;
      case 'ArrowUp': j = Math.max(0, i - cols); break;
      case 'Home': j = 0; break;
      case 'End': j = cards.length - 1; break;
      default: return;
    }
    e.preventDefault();
    e.stopPropagation();
    cards.forEach((c, k) => { c.tabIndex = k === j ? 0 : -1; });
    cards[j].focus();
  }

  _choosePreset(p) {
    if (p.demo) {
      this.toggleSheet('scenes', false);
      this.opts.onDemo?.();
      return;
    }
    this.store.load(presetScene(p.id), { label: p.name });
    if (this.store.scene.mode === 'sleep' && !this.sleep.endAt) this.setSleep(45);
    this.opts.play?.();
    announce(`Playing ${p.name}`);
  }

  _chooseUser(p) {
    if (this.store.loadUserPreset(p.name)) {
      this.opts.play?.();
      announce(`Playing ${p.name}`);
    }
  }

  _deleteUser(p) {
    const all = this.store.userPresets();
    const idx = all.findIndex((x) => x.name === p.name);
    this.store.deleteUserPreset(p.name);
    this.opts.toast?.(`Deleted “${p.name}”.`, {
      action: {
        label: 'Undo',
        fn: () => {
          const list = this.store.userPresets();
          list.splice(Math.max(0, idx), 0, p);
          this.store._writePresets(list);
        },
      },
    });
  }

  /* ------------------------------------------------------------------ describe */
  _previewDescribe() {
    const txt = this.describeInput.value.trim();
    this.chipsEl.textContent = '';
    if (!txt) return;
    const res = parseDescription(txt, this.store.scene);
    if (!res.understood) {
      this.chipsEl.append(h('p', { class: 'hint' }, 'Nothing understood yet. Try “heavy rain on a tin roof, force 6 from the west, thunder 3 km away, night”.'));
      return;
    }
    this.chipsEl.append(h('span', { class: 'lm-chips-label' }, 'Understood:'),
      ...res.chips.map((c) => h('span', { class: 'chip' }, h('b', {}, c.label), ` ${c.text}`)));
  }

  _applyDescribe() {
    const txt = this.describeInput.value.trim();
    if (!txt) return;
    const res = parseDescription(txt, this.store.scene);
    if (!res.understood) {
      this._previewDescribe();
      announce('Nothing understood. Nothing changed.');
      return;
    }
    this.store.beginGesture('describe');
    this.store.setMany(res.patch, { coalesce: null });
    this.store.rename(capital(txt).slice(0, 80));
    this.store.endGesture();
    this.sync();
    this.opts.play?.();
    const summary = res.chips.map((c) => `${c.label} ${c.text}`).join('; ');
    announce(`Applied: ${summary}`);
    this.opts.toast?.(`Applied: ${res.chips.map((c) => c.text).join(' · ')}`, { action: { label: 'Undo', fn: () => this.store.undo() } });
  }

  /* ------------------------------------------------------------------ orb */
  _orbFromVibe() {
    const o = this.orb;
    if (!o || o.dragging || Math.hypot(o.vx, o.vy) > 0.01) return;
    const v = this._vibe();
    const r = clamp(v.intensity / 100, 0, 1);
    const th = Math.acos(clamp(1 - v.wildness / 50, -1, 1));
    const side = o.x < -1e-3 ? -1 : 1;
    o.x = side * r * Math.sin(th);
    o.y = -r * Math.cos(th);
    this._orbDraw();
  }

  _orbValues() {
    const o = this.orb;
    const r = clamp(Math.hypot(o.x, o.y), 0, 1);
    const th = Math.atan2(o.x, -o.y);
    return { intensity: Math.round(r * 100), wildness: Math.round(((1 - Math.cos(th)) / 2) * 100) };
  }

  _orbDraw() {
    const o = this.orb;
    const cx = o.x * 100;
    const cy = o.y * 100;
    o.puck.setAttribute('cx', cx.toFixed(2));
    o.puck.setAttribute('cy', cy.toFixed(2));
    o.trail.setAttribute('cx', (cx - o.vx * 6).toFixed(2));
    o.trail.setAttribute('cy', (cy - o.vy * 6).toFixed(2));
    const v = this._orbValues();
    o.puck.setAttribute('aria-valuenow', String(v.intensity));
    const s = this.store.scene;
    const p = s.weather.precipitation;
    const text = `Intensity ${v.intensity}: ${p.type === 'none' ? 'rain stopped' : `${rainWord(p.rate_mm_h)}, ${fmtNum(p.rate_mm_h)} millimetres an hour`}; character ${v.wildness}: ${characterWord(v.wildness)}`;
    o.puck.setAttribute('aria-valuetext', text);
    o.readout.textContent = `Rain ${v.intensity} · ${capital(characterWord(v.wildness))}`;
  }

  _orbPush(force = false) {
    const o = this.orb;
    const now = performance.now();
    if (!force && now - o.lastPush < 110) return;
    o.lastPush = now;
    const v = this._orbValues();
    const cur = this.store.scene.vibe || {};
    if (!force && cur.intensity === v.intensity && cur.wildness === v.wildness) return;
    this.store.setVibe({ intensity: v.intensity, wildness: v.wildness });
    for (const k of ['intensity', 'wildness']) {
      const inp = this.sliders[k].input;
      if (document.activeElement !== inp) inp.value = String(v[k]);
    }
    this._refreshCaptions();
  }

  _bindOrb() {
    const o = this.orb;
    const toUnit = (e) => {
      const r = o.svg.getBoundingClientRect();
      const scale = 240 / Math.max(1, r.width);
      const x = ((e.clientX - r.left) * scale - 120) / 100;
      const y = ((e.clientY - r.top) * scale - 120) / 100;
      const d = Math.hypot(x, y);
      return d > 1 ? [x / d, y / d] : [x, y];
    };
    o.svg.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      o.svg.setPointerCapture(e.pointerId);
      o.puck.focus({ preventScroll: true });
      [o.tx, o.ty] = toUnit(e);
      o.dragging = true;
      o.samples = [{ t: performance.now(), x: o.tx, y: o.ty }];
      if (!o.gesture) {
        this.store.beginGesture('orb');
        o.gesture = true;
      }
      if (prefersReducedMotion()) {
        o.x = o.tx;
        o.y = o.ty;
      }
      this._orbStart();
    });
    o.svg.addEventListener('pointermove', (e) => {
      if (!o.dragging) return;
      [o.tx, o.ty] = toUnit(e);
      o.samples.push({ t: performance.now(), x: o.tx, y: o.ty });
      if (o.samples.length > 6) o.samples.shift();
      if (prefersReducedMotion()) {
        o.x = o.tx;
        o.y = o.ty;
      }
    });
    const release = (e) => {
      if (!o.dragging) return;
      o.dragging = false;
      try { o.svg.releasePointerCapture(e.pointerId); } catch (_) { /* released */ }
      const a = o.samples[0];
      const b = o.samples[o.samples.length - 1];
      const dt = Math.max(0.016, (b.t - a.t) / 1000);
      if (!prefersReducedMotion() && o.samples.length > 1) {
        o.vx = clamp((b.x - a.x) / dt, -3, 3) * 0.6;
        o.vy = clamp((b.y - a.y) / dt, -3, 3) * 0.6;
      } else {
        o.vx = 0;
        o.vy = 0;
      }
      this._orbStart();
    };
    o.svg.addEventListener('pointerup', release);
    o.svg.addEventListener('pointercancel', release);
    o.drift.addEventListener('change', () => {
      if (o.drift.checked) this._orbStart();
      announce(o.drift.checked ? 'Drift on: the weather wanders slowly' : 'Drift off');
    });
    o.puck.addEventListener('keydown', (e) => {
      let { intensity, wildness } = this._orbValues();
      let th = Math.atan2(o.x, -o.y);
      let r = Math.hypot(o.x, o.y);
      switch (e.key) {
        case 'ArrowUp': r += 0.05; break;
        case 'ArrowDown': r -= 0.05; break;
        case 'PageUp': r += 0.2; break;
        case 'PageDown': r -= 0.2; break;
        case 'ArrowRight': th += (15 * Math.PI) / 180; break;
        case 'ArrowLeft': th -= (15 * Math.PI) / 180; break;
        case 'Home': r = 0; break;
        case 'End': r = 1; break;
        default: return;
      }
      e.preventDefault();
      e.stopPropagation();
      r = clamp(r, 0, 1);
      o.vx = 0;
      o.vy = 0;
      o.x = r * Math.sin(th);
      o.y = -r * Math.cos(th);
      ({ intensity, wildness } = this._orbValues());
      this.store.setVibe({ intensity, wildness });
      this._orbDraw();
      this._refreshCaptions();
      for (const k of ['intensity', 'wildness']) this.sliders[k].input.value = String(k === 'intensity' ? intensity : wildness);
    });
  }

  _orbStart() {
    const o = this.orb;
    if (o.raf) return;
    o.last = performance.now();
    const step = (now) => {
      const dt = Math.min(0.05, (now - o.last) / 1000);
      o.last = now;
      const moving = this._orbStep(dt);
      this._orbDraw();
      this._orbPush(false);
      if (moving) o.raf = requestAnimationFrame(step);
      else {
        o.raf = 0;
        this._orbPush(true);
        if (o.gesture) {
          o.gesture = false;
          this.store.endGesture();
        }
      }
    };
    o.raf = requestAnimationFrame(step);
  }

  /** Spring-damper toward the pointer while dragging; friction (inertia) after release; boundary bounce. */
  _orbStep(dt) {
    const o = this.orb;
    const reduced = prefersReducedMotion();
    if (o.dragging) {
      if (reduced) {
        o.vx = 0;
        o.vy = 0;
      } else {
        const k = 140;
        const c = 2 * Math.sqrt(k) * 0.9;
        o.vx += (k * (o.tx - o.x) - c * o.vx) * dt;
        o.vy += (k * (o.ty - o.y) - c * o.vy) * dt;
      }
    } else {
      const fr = Math.exp(-2.4 * dt);
      o.vx *= fr;
      o.vy *= fr;
      if (o.drift.checked && !reduced) {
        o.noiseT += dt;
        const n = o.noiseT;
        o.vx += (Math.sin(n * 0.31) * 0.6 + Math.sin(n * 0.17 + 1.3) * 0.4) * 0.02 * dt;
        o.vy += (Math.cos(n * 0.23 + 0.7) * 0.6 + Math.sin(n * 0.11 + 2.1) * 0.4) * 0.02 * dt;
        const r = Math.hypot(o.x, o.y);
        if (r > 0.85) {
          o.vx -= o.x * 0.05 * dt;
          o.vy -= o.y * 0.05 * dt;
        }
      }
    }
    o.x += o.vx * dt;
    o.y += o.vy * dt;
    const r = Math.hypot(o.x, o.y);
    if (r > 1) {
      const nx = o.x / r;
      const ny = o.y / r;
      o.x = nx;
      o.y = ny;
      const vr = o.vx * nx + o.vy * ny;
      if (vr > 0) {
        o.vx -= 1.4 * vr * nx;
        o.vy -= 1.4 * vr * ny;
      }
    }
    const speed = Math.hypot(o.vx, o.vy);
    return o.dragging || speed > 0.004 || (o.drift.checked && !reduced);
  }

  /* ------------------------------------------------------------------ sleep timer */
  setSleep(minutes) {
    clearInterval(this.sleep.timer);
    this.sleep.timer = 0;
    this.sleep.minutes = minutes;
    if (this.sleepSel && String(this.sleepSel.value) !== String(minutes)) this.sleepSel.value = String(minutes);
    if (!minutes) {
      this.sleep.endAt = 0;
      this.opts.onSleepFade?.(1);
      this.sleepStatus.textContent = 'Sleep timer off.';
      this.sleepChip.hidden = true;
      return;
    }
    this.sleep.total = minutes * 60;
    this.sleep.endAt = Date.now() + minutes * 60000;
    this.sleep.fade = this.sleepFade.checked;
    const at = new Date(this.sleep.endAt);
    const hhmm = `${String(at.getHours()).padStart(2, '0')}:${String(at.getMinutes()).padStart(2, '0')}`;
    this.sleepStatus.textContent = `Stops at ${hhmm}${this.sleep.fade ? ', fading out over the last minutes' : ''}.`;
    announce(`Sleep timer: stops at ${hhmm}`);
    this.sleep.timer = setInterval(() => this._sleepTick(), 1000);
    this._sleepTick();
  }

  _sleepTick() {
    const sl = this.sleep;
    if (!sl.endAt) return;
    const rem = (sl.endAt - Date.now()) / 1000;
    const at = new Date(sl.endAt);
    const hhmm = `${String(at.getHours()).padStart(2, '0')}:${String(at.getMinutes()).padStart(2, '0')}`;
    if (rem <= 0) {
      clearInterval(sl.timer);
      sl.timer = 0;
      sl.endAt = 0;
      this.sleepSel.value = '0';
      this.sleepStatus.textContent = 'Sleep timer finished. Sweet dreams.';
      this.sleepChip.hidden = true;
      this.opts.onSleepEnd?.();
      return;
    }
    const F = Math.min(900, sl.total / 2);
    const fading = sl.fade && rem < F;
    // equal-loudness-ish fade: amplitude follows (rem/F)^1.5 so the last minutes fall away gently
    const f = fading ? Math.pow(clamp(rem / F, 0, 1), 1.5) : 1;
    this.opts.onSleepFade?.(f);
    const mins = Math.ceil(rem / 60);
    this.sleepChip.hidden = false;
    this.sleepChip.textContent = `Stops at ${hhmm} · ${mins} min left${fading ? ' · fading' : ''}`;
  }
}

/** Minimal shape guard for summarising a saved preset without normalising it. */
function normalizeLite(scene) {
  const d = {
    weather: { precipitation: { type: 'rain', rate_mm_h: 4, hail_size_mm: 13 }, wind: { speed_mps: 4, from_deg: 270 }, thunder: { enabled: false, distance_km: 8 }, temperature_c: 12 },
    place: { shelter: 'open' }, studio: { labels: 'wmo' },
  };
  const s = scene && typeof scene === 'object' ? scene : {};
  const w = s.weather || {};
  return {
    weather: {
      precipitation: { ...d.weather.precipitation, ...(w.precipitation || {}) },
      wind: { ...d.weather.wind, ...(w.wind || {}) },
      thunder: { ...d.weather.thunder, ...(w.thunder || {}) },
      temperature_c: w.temperature_c ?? 12,
    },
    place: { ...d.place, ...(s.place || {}) },
    studio: { ...d.studio, ...(s.studio || {}) },
  };
}
