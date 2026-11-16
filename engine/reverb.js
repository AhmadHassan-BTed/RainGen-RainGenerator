/**
 * Shelter and place acoustics.
 *
 * - Impulse responses: exponentially decaying noise (Moorer 1979) with frequency-dependent
 *   decay (a low-pass whose cutoff falls over the tail, i.e. high frequencies die first, as
 *   with air and wall absorption in Sabine/Eyring rooms), plus discrete early reflections and
 *   flutter echoes between parallel walls every 2w/c (urban canyons, underpasses).
 * - Envelope transmission loss: exterior sound reaching a shelter interior through glazing,
 *   car bodies, tent fabric or a roof is modelled as low-pass + high-shelf + broadband loss
 *   (mass-law-like slope); a window opening blends toward a much lighter 'open' curve.
 */

import { seedKey, clamp } from './physics.js';

const ROOM_ER = [
  [5, 0.4],
  [9, 0.3],
  [14, 0.25],
  [21, 0.15],
];

/** Shelter descriptors: interior reverb, exterior transmission (closed/open) and near-ground rain exposure. */
export const SHELTERS = {
  open: { label: 'Out in it', reverb: null, tl: null, nearExposure: 1, enclosed: false, defaultOpen: 1 },
  tree: {
    label: 'Under a tree',
    reverb: { rt60: 0.25, preMs: 5, wet: 0.07, hf: 0.5, er: [[6, 0.25], [11, 0.18]] },
    tl: null,
    nearExposure: 0.7,
    enclosed: false,
    defaultOpen: 1,
  },
  umbrella: {
    label: 'Under an umbrella',
    reverb: { rt60: 0.1, preMs: 1, wet: 0.04, hf: 0.6, er: [[1.5, 0.3]] },
    tl: null,
    nearExposure: 0.92,
    enclosed: false,
    defaultOpen: 1,
  },
  tent: {
    label: 'In a tent',
    reverb: { rt60: 0.14, preMs: 2, wet: 0.1, hf: 0.35, er: [[2.5, 0.35], [4.2, 0.25], [6, 0.15]] },
    tl: { closed: { lp: 5500, shelfDb: -4, gainDb: -2 }, open: { lp: 12000, shelfDb: -1, gainDb: 0 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 0,
  },
  porch: {
    label: 'On a porch',
    reverb: { rt60: 0.45, preMs: 6, wet: 0.1, hf: 0.5, er: [[7, 0.3], [13, 0.2]] },
    tl: null,
    nearExposure: 0.35,
    enclosed: false,
    defaultOpen: 1,
  },
  car: {
    label: 'In a car',
    reverb: { rt60: 0.08, preMs: 1, wet: 0.18, hf: 0.4, er: [[1.6, 0.5], [2.9, 0.4], [4.1, 0.3], [5.3, 0.2]] },
    tl: { closed: { lp: 650, shelfDb: -14, gainDb: -15 }, open: { lp: 6000, shelfDb: -3, gainDb: -3 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 0,
  },
  room_open: {
    label: 'Indoors, window open',
    reverb: { rt60: 0.55, preMs: 8, wet: 0.2, hf: 0.45, er: ROOM_ER },
    tl: { closed: { lp: 450, shelfDb: -18, gainDb: -24 }, open: { lp: 5000, shelfDb: -4, gainDb: -7 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 1,
  },
  room_closed: {
    label: 'Indoors, window shut',
    reverb: { rt60: 0.55, preMs: 8, wet: 0.2, hf: 0.45, er: ROOM_ER },
    tl: { closed: { lp: 450, shelfDb: -18, gainDb: -24 }, open: { lp: 5000, shelfDb: -4, gainDb: -7 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 0,
  },
  attic: {
    label: 'In the attic',
    reverb: { rt60: 0.7, preMs: 4, wet: 0.22, hf: 0.4, er: [[3, 0.4], [6, 0.3], [10, 0.2]] },
    tl: { closed: { lp: 600, shelfDb: -16, gainDb: -20 }, open: { lp: 4000, shelfDb: -6, gainDb: -8 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 0,
  },
  cave: {
    label: 'At a cave mouth',
    reverb: { rt60: 2.6, preMs: 18, wet: 0.45, hf: 0.35, er: [[12, 0.4], [23, 0.35], [37, 0.3], [52, 0.2]] },
    tl: { closed: { lp: 3000, shelfDb: -6, gainDb: -6 }, open: { lp: 3000, shelfDb: -6, gainDb: -6 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 1,
  },
  underpass: {
    label: 'In an underpass',
    reverb: { rt60: 1.8, preMs: 10, wet: 0.35, hf: 0.5, flutter: { periodMs: 70, count: 14, decay: 0.78, gain: 0.35 } },
    tl: { closed: { lp: 9000, shelfDb: -2, gainDb: -3 }, open: { lp: 9000, shelfDb: -2, gainDb: -3 } },
    nearExposure: 0,
    enclosed: true,
    defaultOpen: 1,
  },
};

const SHELTER_ALIASES = {
  none: 'open',
  outside: 'open',
  outdoors: 'open',
  room: 'room_closed',
  indoors: 'room_closed',
  window_open: 'room_open',
  window: 'room_open',
  cabin: 'room_closed',
  vehicle: 'car',
  tin_roof: 'attic',
  roof: 'attic',
  veranda: 'porch',
  bridge: 'underpass',
  tunnel: 'underpass',
};

export function shelterName(name) {
  const k = String(name || 'open').toLowerCase().replace(/[\s-]+/g, '_');
  if (SHELTERS[k]) return k;
  if (SHELTER_ALIASES[k]) return SHELTER_ALIASES[k];
  return 'open';
}

export function shelterInfo(name) {
  return SHELTERS[shelterName(name)];
}

/** Outdoor acoustics per place preset (used when the listener is out in it). */
export const OUTDOOR = {
  garden: { rt60: 0.5, preMs: 12, wet: 0.05, hf: 0.45 },
  deciduous_forest: { rt60: 1.4, preMs: 15, wet: 0.11, hf: 0.3 },
  conifer_forest: { rt60: 1.6, preMs: 15, wet: 0.12, hf: 0.3 },
  meadow: { rt60: 0.25, preMs: 8, wet: 0.025, hf: 0.5 },
  coast: { rt60: 0.3, preMs: 10, wet: 0.03, hf: 0.5 },
  lake: { rt60: 0.6, preMs: 20, wet: 0.05, hf: 0.45, er: [[40, 0.2]] },
  city_street: { rt60: 1.0, preMs: 10, wet: 0.14, hf: 0.45, flutter: { periodMs: 87, count: 10, decay: 0.7, gain: 0.3 } },
  suburb: { rt60: 0.6, preMs: 12, wet: 0.07, hf: 0.45 },
  alley: { rt60: 0.9, preMs: 6, wet: 0.2, hf: 0.45, flutter: { periodMs: 30, count: 18, decay: 0.8, gain: 0.35 } },
  mountain_ridge: { rt60: 0.4, preMs: 10, wet: 0.08, hf: 0.4, er: [[420, 0.25], [910, 0.12]] },
  desert: { rt60: 0.2, preMs: 8, wet: 0.02, hf: 0.5 },
  jungle: { rt60: 1.2, preMs: 12, wet: 0.1, hf: 0.3 },
  farm: { rt60: 0.5, preMs: 12, wet: 0.05, hf: 0.45, er: [[60, 0.2]] },
};

export function outdoorSpec(preset) {
  const k = String(preset || 'garden').toLowerCase().replace(/[\s-]+/g, '_');
  return OUTDOOR[k] || (k.includes('forest') ? OUTDOOR.deciduous_forest : k.includes('city') ? OUTDOOR.city_street : OUTDOOR.garden);
}

/** The reverb spec the listener hears for a shelter at a place. */
export function reverbSpecFor(shelter, preset) {
  const s = shelterInfo(shelter);
  return s.reverb || outdoorSpec(preset);
}

/** Exterior transmission description for a shelter and window opening (0..1). */
export function transmissionFor(shelter, windowOpen = 0) {
  const s = shelterInfo(shelter);
  const name = shelterName(shelter);
  if (!s.tl) return { openness: 1, closed: { lp: 20000, shelfDb: 0, gainDb: 0 }, open: { lp: 20000, shelfDb: 0, gainDb: 0 } };
  let openness = clamp(Number(windowOpen) || 0, 0, 1);
  if (name === 'room_open' && !(openness > 0)) openness = 1;
  if (name === 'cave' || name === 'underpass') openness = 1;
  return { openness, closed: s.tl.closed, open: s.tl.open || s.tl.closed };
}

/** Generate a stereo impulse response AudioBuffer for a reverb spec. Energy-normalized per channel. */
export function createImpulseResponse(ctx, spec, seed = 1) {
  const sr = ctx.sampleRate;
  const sp = spec || { rt60: 0.3, preMs: 5 };
  const rt = clamp(Number(sp.rt60) || 0.3, 0.03, 8);
  const pre = Math.max(0, (Number(sp.preMs) || 0) / 1000);
  let lastEarly = 0;
  for (const e of sp.er || []) lastEarly = Math.max(lastEarly, (Number(e[0]) || 0) / 1000);
  if (sp.flutter) lastEarly = Math.max(lastEarly, pre + ((sp.flutter.periodMs || 50) * (sp.flutter.count || 8)) / 1000);
  const dur = Math.min(8, Math.max(pre + rt * 1.25 + 0.02, lastEarly + 0.06));
  const len = Math.max(64, Math.ceil(dur * sr));
  const buf = ctx.createBuffer(2, len, sr);
  const key = seedKey(seed);
  const fcStart = Math.min(0.45 * sr, 15000);
  const fcEnd = fcStart * clamp(Number(sp.hf) || 0.4, 0.05, 1) * 0.2;
  const preI = Math.floor(pre * sr);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    let s = ((key ^ (ch ? 0x9e3779b9 : 0x85ebca6b)) | 1) | 0;
    let y = 0;
    for (let i = preI; i < len; i++) {
      const t = (i - preI) / sr;
      const env = Math.exp((-6.907755 * t) / rt);
      const fc = fcStart * Math.pow(fcEnd / fcStart, Math.min(1, t / rt));
      const a = 1 - Math.exp((-2 * Math.PI * fc) / sr);
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      y += a * (s / 2147483648 - y);
      const fade = t < 0.002 ? t / 0.002 : 1;
      d[i] = y * env * fade;
    }
    let sign = ch ? -1 : 1;
    for (const e of sp.er || []) {
      const ms = Number(e[0]) || 0;
      const g = Number(e[1]) || 0;
      const idx = Math.round((ms / 1000) * (1 + (ch ? 0.04 : -0.03)) * sr);
      if (idx > 0 && idx < len) d[idx] += g * sign;
      sign = -sign;
    }
    if (sp.flutter) {
      const per = (sp.flutter.periodMs || 50) / 1000;
      const cnt = sp.flutter.count || 8;
      const dec = sp.flutter.decay || 0.75;
      const g0 = sp.flutter.gain || 0.3;
      for (let k = 1; k <= cnt; k++) {
        const idx = Math.round((pre + k * per) * sr) + (ch ? 3 : 0);
        if (idx >= len) break;
        const g = g0 * Math.pow(dec, k) * (k % 2 === ch ? 1 : 0.5);
        d[idx] += g;
        if (idx + 1 < len) d[idx + 1] += 0.5 * g;
      }
    }
    let e = 0;
    for (let i = 0; i < len; i++) e += d[i] * d[i];
    const norm = e > 0 ? 1 / Math.sqrt(e) : 0;
    for (let i = 0; i < len; i++) d[i] *= norm;
  }
  return buf;
}

/**
 * Crossfading convolution reverb: input -> two convolver slots -> wet output.
 * setSpec() builds a new impulse response in the idle slot and crossfades to it.
 */
export class ReverbUnit {
  constructor(ctx) {
    this.ctx = ctx;
    this.input = ctx.createGain();
    this.output = ctx.createGain();
    this.output.gain.value = 0;
    this.slots = [0, 1].map(() => {
      const g = ctx.createGain();
      g.gain.value = 0;
      g.connect(this.output);
      return { conv: null, gain: g };
    });
    this.active = -1;
    this.key = '';
    this.wet = 0;
  }

  setSpec(spec, seed = 1, { when, fade = 0.6, offline = false } = {}) {
    const ctx = this.ctx;
    const t = Math.max(when ?? ctx.currentTime, ctx.currentTime);
    const wet = spec && spec.wet > 0 ? spec.wet : 0;
    const key = spec ? `${JSON.stringify(spec)}|${seed}` : 'none';
    try {
      this.output.gain.cancelScheduledValues(t);
      this.output.gain.setTargetAtTime(wet, t, offline ? 0.01 : 0.2);
    } catch (e) {
      this.output.gain.value = wet;
    }
    this.wet = wet;
    if (key === this.key || !spec || !(wet > 0)) {
      if (!spec) this.key = key;
      return;
    }
    this.key = key;
    let conv;
    try {
      conv = ctx.createConvolver();
      conv.normalize = false;
      conv.buffer = createImpulseResponse(ctx, spec, seed);
    } catch (e) {
      return;
    }
    const next = this.active === 0 ? 1 : 0;
    const slot = this.slots[next];
    if (slot.conv) {
      try {
        this.input.disconnect(slot.conv);
      } catch (e) {
        /* already disconnected */
      }
      try {
        slot.conv.disconnect();
      } catch (e) {
        /* already disconnected */
      }
    }
    slot.conv = conv;
    this.input.connect(conv);
    conv.connect(slot.gain);
    const f = offline || this.active < 0 ? 0.005 : fade;
    const g = slot.gain.gain;
    g.cancelScheduledValues(t);
    g.setValueAtTime(0, t);
    g.linearRampToValueAtTime(1, t + f);
    if (this.active >= 0) {
      const old = this.slots[this.active];
      const og = old.gain.gain;
      og.cancelScheduledValues(t);
      og.setValueAtTime(og.value, t);
      og.linearRampToValueAtTime(0, t + f);
      if (!offline && typeof setTimeout === 'function') {
        const oldConv = old.conv;
        setTimeout(() => {
          if (old.conv === oldConv && this.slots[this.active] !== old && oldConv) {
            try {
              this.input.disconnect(oldConv);
              oldConv.disconnect();
            } catch (e) {
              /* ignore */
            }
            old.conv = null;
          }
        }, (f + 0.2) * 1000);
      }
    }
    this.active = next;
  }

  disconnect() {
    try {
      this.input.disconnect();
      this.output.disconnect();
      for (const s of this.slots) {
        if (s.conv) s.conv.disconnect();
        s.gain.disconnect();
      }
    } catch (e) {
      /* ignore */
    }
  }
}
