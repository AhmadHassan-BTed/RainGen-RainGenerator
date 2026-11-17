import { unlockAudio, releaseAudio } from '../engine/unlock.js?v=20261117a';
/* raingen landing page. Progressive enhancement: the page reads fine without it. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const root = document.documentElement;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const motionMQ = window.matchMedia('(prefers-reduced-motion: reduce)');
let reduceMotion = motionMQ.matches;

/* ---------- reveal on scroll ---------- */
(function reveal() {
  if (reduceMotion || !('IntersectionObserver' in window)) return;
  root.classList.add('rv');
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); }
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0.12 });
  $$('.reveal').forEach((el) => io.observe(el));
})();

/* ---------- header turns solid once the stage scrolls away ---------- */
(function header() {
  const h = $('.site-header');
  if (!h) return;
  const on = () => h.classList.toggle('is-solid', window.scrollY > 40);
  window.addEventListener('scroll', on, { passive: true });
  on();
})();

/* ---------- units ---------- */
const TEMP_C = 16; // hero scene air temperature
const BFT = ['Calm', 'Light air', 'Light breeze', 'Gentle breeze', 'Moderate breeze', 'Fresh breeze', 'Strong breeze',
  'Near gale', 'Gale', 'Strong gale', 'Storm', 'Violent storm', 'Hurricane force'];
const mpsFromBeaufort = (b) => 0.836 * Math.pow(Math.max(0, b), 1.5);
const soundSpeed = (tc) => 331.3 * Math.sqrt(1 + tc / 273.15);
const secPerKm = (tc) => 1000 / soundSpeed(tc);
const C16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const C16_WORDS = {
  N: 'north', NNE: 'north-northeast', NE: 'northeast', ENE: 'east-northeast', E: 'east', ESE: 'east-southeast',
  SE: 'southeast', SSE: 'south-southeast', S: 'south', SSW: 'south-southwest', SW: 'southwest', WSW: 'west-southwest',
  W: 'west', WNW: 'west-northwest', NW: 'northwest', NNW: 'north-northwest'
};
const compass16 = (deg) => C16[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16];
function rainLabel(r) {
  if (r <= 0) return 'dry';
  if (r < 2.5) return 'slight';
  if (r < 10) return 'moderate';
  if (r < 50) return 'heavy';
  if (r < 100) return 'intense';
  return 'cloudburst';
}
const fmt = (n, d = 1) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtSig = (n) => (n >= 100 ? Math.round(n).toLocaleString('en-US') : n >= 1 ? fmt(n, 1) : fmt(n, 2));
const logMap = (p, lo, hi) => Math.pow(10, Math.log10(lo) + (p / 1000) * (Math.log10(hi) - Math.log10(lo)));
function syncFill(input) {
  const min = Number(input.min || 0), max = Number(input.max || 100);
  input.style.setProperty('--fill', (((Number(input.value) - min) / (max - min)) * 100).toFixed(2) + '%');
}
function bindRange(input, fn) {
  const run = () => { syncFill(input); fn(Number(input.value)); };
  input.addEventListener('input', run);
  run();
}

const heroState = { rate: 5, windMps: 4.3, fromDeg: 270, thunderOn: true, thunderKm: 6, shelter: 'porch' };

/* ================================================================== *
 * Hero stage: depth-layered rain under a cloud base, lightning glow   *
 * ================================================================== */
class Sky {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.drops = [];
    this.splashes = [];
    this.flash = null;
    this.lastFlashAt = -1e9;
    this.running = false;
    this.last = 0;
    this.onTick = null;
    this.frame = this.frame.bind(this);
    this.resize();
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    const reseed = Math.abs(w - (this.w || 0)) > 40 || !this.drops.length;
    this.w = w; this.h = h;
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    if (reseed) { this.drops = []; this.update(); } else this.draw(performance.now());
  }
  target() {
    const f = clamp((Math.log10(Math.max(0.1, heroState.rate)) + 1) / 3, 0, 1);
    const area = clamp((this.w * this.h) / (1440 * 900), 0.3, 1.6);
    return Math.round((50 + 260 * f + 560 * f * f) * area);
  }
  spawn(d, anywhere) {
    const z = Math.random();
    const layer = z < 0.62 ? 0 : z < 0.9 ? 1 : 2;
    d.layer = layer;
    d.v = [430, 760, 1250][layer] * (0.85 + Math.random() * 0.3);
    d.len = [9, 20, 44][layer] * (0.75 + Math.random() * 0.5);
    d.gy = this.h * ([0.8, 0.9, 1.02][layer] + Math.random() * 0.08);
    d.x = Math.random() * (this.w + 240) - 200;
    d.y = anywhere ? Math.random() * d.gy : -Math.random() * 120;
    return d;
  }
  update() {
    const n = this.target();
    while (this.drops.length < n) this.drops.push(this.spawn({}, true));
    if (this.drops.length > n) this.drops.length = n;
    if (!this.running) this.draw(performance.now());
  }
  triggerFlash(km, bright) {
    const now = performance.now();
    if (now - this.lastFlashAt < 1200) return; // never more than one flash event per 1.2 s
    this.lastFlashAt = now;
    const b = clamp(bright ?? 0.6, 0.2, 1);
    this.flash = { t0: now, x: this.w * (0.5 + Math.random() * 0.42), y: this.h * (0.04 + Math.random() * 0.1), b,
      r: Math.max(this.w, this.h) * (0.38 + 0.3 * b) };
    if (!this.running) {
      this.draw(now);
      clearTimeout(this.ft);
      this.ft = setTimeout(() => { this.flash = null; this.draw(performance.now()); }, 1800);
    }
  }
  flashLevel(now) {
    const f = this.flash;
    if (!f) return 0;
    if (reduceMotion || !this.running) return 0.3 * f.b; // a soft, steady glow
    const t = (now - f.t0) / 1000;
    if (t > 2.4) { this.flash = null; return 0; }
    // two strokes, 150 ms apart, then a slow decay: two flashes per event at most
    const a = t < 0.07 ? 1 : t < 0.15 ? 0.3 : t < 0.24 ? 0.75 : 0.75 * Math.exp(-(t - 0.24) / 0.4);
    return a * f.b;
  }
  draw(now) {
    const { ctx, w, h } = this;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    let g = ctx.createRadialGradient(w * 0.72, -h * 0.15, 0, w * 0.72, -h * 0.15, Math.max(w, h) * 0.95);
    g.addColorStop(0, 'rgba(30,48,66,0.95)');
    g.addColorStop(0.42, 'rgba(10,18,28,0.75)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    const fl = this.flashLevel(now);
    if (fl > 0.003) {
      const f = this.flash;
      g = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.r);
      g.addColorStop(0, `rgba(244,241,232,${0.5 * fl})`);
      g.addColorStop(0.3, `rgba(190,208,226,${0.16 * fl})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    }
    const slant = clamp(heroState.windMps / 9, 0, 2.4) * 0.5;
    const alpha = [0.17, 0.27, 0.36], width = [0.6, 0.9, 1.3];
    ctx.lineCap = 'round';
    for (let L = 0; L < 3; L++) {
      ctx.strokeStyle = `rgba(205,226,246,${Math.min(0.9, alpha[L] + fl * 0.35)})`;
      ctx.lineWidth = width[L];
      ctx.beginPath();
      for (const d of this.drops) {
        if (d.layer !== L) continue;
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(d.x - slant * d.len, d.y - d.len);
      }
      ctx.stroke();
    }
    ctx.lineWidth = 0.8;
    for (const s of this.splashes) {
      const a = (1 - s.t) * (s.layer === 2 ? 0.4 : 0.22);
      const rx = (s.layer === 2 ? 9 : 5) * (0.3 + s.t);
      ctx.strokeStyle = `rgba(205,226,246,${a})`;
      ctx.beginPath();
      ctx.ellipse(s.x, s.y, rx, rx * 0.22, 0, Math.PI, 2 * Math.PI);
      ctx.stroke();
    }
  }
  frame(now) {
    if (!this.running) return;
    const dt = Math.min(0.05, (now - (this.last || now)) / 1000);
    this.last = now;
    const slant = clamp(heroState.windMps / 9, 0, 2.4) * 0.5;
    for (const d of this.drops) {
      d.y += d.v * dt;
      d.x += slant * d.v * dt;
      if (d.y > d.gy) {
        if (d.layer && this.splashes.length < 140 && d.x > 0 && d.x < this.w) this.splashes.push({ x: d.x, y: d.gy, t: 0, layer: d.layer });
        this.spawn(d, false);
      }
      if (d.x > this.w + 60) d.x -= this.w + 260;
    }
    for (let i = this.splashes.length - 1; i >= 0; i--) {
      this.splashes[i].t += dt * 3.2;
      if (this.splashes[i].t >= 1) this.splashes.splice(i, 1);
    }
    this.draw(now);
    if (this.onTick) this.onTick(now);
    this.raf = requestAnimationFrame(this.frame);
  }
  start() { if (this.running) return; this.running = true; this.last = 0; this.raf = requestAnimationFrame(this.frame); }
  stop() { this.running = false; cancelAnimationFrame(this.raf); this.draw(performance.now()); }
}

const skyCanvas = $('#hero-sky');
const sky = skyCanvas && skyCanvas.getContext ? new Sky(skyCanvas) : null;
(function skyLifecycle() {
  if (!sky) return;
  let visible = true;
  const sync = () => { if (visible && !document.hidden && !reduceMotion) sky.start(); else sky.stop(); };
  if ('IntersectionObserver' in window) new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }).observe($('.hero'));
  document.addEventListener('visibilitychange', sync);
  let rq = 0;
  const onResize = () => { if (!rq) rq = requestAnimationFrame(() => { rq = 0; sky.resize(); }); };
  if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(skyCanvas); else window.addEventListener('resize', onResize);
  motionMQ.addEventListener?.('change', (e) => { reduceMotion = e.matches; sync(); });
  sync();
})();

/* ---------- flash-to-bang counter shared by the stage and the engine ---------- */
const flashCounter = $('#flash-counter');
const flashText = $('#flash-text');
const captions = $('#captions');
let pending = null;
let counterTimer = 0;
let nextSimFlashAt = performance.now() + 6000;
let engineCaptions = false;
let engine = null, listening = false, busy = false, enginePromise = null;

function idleCounterText() {
  if (!heroState.thunderOn) { flashText.textContent = 'Thunder is off.'; return; }
  const s = heroState.thunderKm * secPerKm(TEMP_C);
  flashText.textContent = `Watch for a flash, then count: thunder at ${fmt(heroState.thunderKm)} km arrives ${fmt(s)} s later.`;
}
const setCaption = (text) => { captions.textContent = text; };

function onFlash({ id = null, km, delay, bright, bearing = 300, sim = false }) {
  sky?.triggerFlash(km, bright ?? clamp(1.15 - km / 20, 0.25, 1));
  pending = { id, t0: performance.now(), delay, km, bearing, sim };
  flashCounter.classList.add('is-counting');
  flashCounter.classList.remove('is-arrived');
  if (!engineCaptions) setCaption(`[Lightning, ${C16_WORDS[compass16(bearing)]}]`);
  clearInterval(counterTimer);
  counterTimer = setInterval(tickCounter, 100);
  tickCounter();
}
function tickCounter() {
  if (!pending) return;
  const t = (performance.now() - pending.t0) / 1000;
  if (pending.sim && t >= pending.delay) { onThunder({ km: pending.km, bearing: pending.bearing }); return; }
  if (!pending.sim && t > pending.delay + 6) {
    clearInterval(counterTimer); pending = null; flashCounter.classList.remove('is-counting'); idleCounterText(); return;
  }
  flashText.textContent = `Flash. Counting ${fmt(Math.min(t, pending.delay))} s, thunder due at ${fmt(pending.delay)} s.`;
}
function onThunder({ km, bearing = 300 }) {
  clearInterval(counterTimer);
  const delay = pending ? pending.delay : km * secPerKm(TEMP_C);
  pending = null;
  flashCounter.classList.remove('is-counting');
  flashCounter.classList.add('is-arrived');
  flashText.textContent = `Thunder after ${fmt(delay)} s: ${fmt(km)} km away at ${TEMP_C} °C.`;
  if (!engineCaptions) setCaption(`[Thunder, ${fmt(km)} km, ${C16_WORDS[compass16(bearing)]}]`);
  setTimeout(() => { if (!pending) { flashCounter.classList.remove('is-arrived'); idleCounterText(); } }, 7000);
}
function scheduleSimFlash(now) {
  const wait = (-Math.log(1 - Math.random()) / (1.5 / 60)) * 1000;
  nextSimFlashAt = now + Math.max(9000, wait);
}
if (sky) sky.onTick = (now) => {
  if (listening || !heroState.thunderOn || reduceMotion) return;
  if (now >= nextSimFlashAt && !pending) {
    const km = heroState.thunderKm;
    onFlash({ km, delay: km * secPerKm(TEMP_C), sim: true });
    scheduleSimFlash(now);
  }
};

/* ---------- the listen console, wired to the real engine ---------- */
const rainIn = $('#c-rain'), windIn = $('#c-wind'), thIn = $('#c-thunder'), thOn = $('#c-thunder-on'), shelterIn = $('#c-shelter');

function pushParam(path, value, ramp = 0.6) {
  if (!engine || !listening) return;
  try { engine.setParam(path, value, ramp); } catch (e) { console.warn('setParam failed', path, e); }
}
bindRange(rainIn, (p) => {
  heroState.rate = logMap(p, 0.1, 100);
  $('#c-rain-out').textContent = `${fmtSig(heroState.rate)} mm/h · ${rainLabel(heroState.rate)}`;
  rainIn.setAttribute('aria-valuetext', `${fmtSig(heroState.rate)} millimetres per hour, ${rainLabel(heroState.rate)}`);
  sky?.update();
  pushParam('weather.precipitation.rate_mm_h', heroState.rate, 1.5);
});
bindRange(windIn, (b) => {
  heroState.windMps = mpsFromBeaufort(b);
  const name = BFT[Math.round(b)];
  $('#c-wind-out').textContent = `Force ${fmt(b)} · ${name.toLowerCase()}`;
  windIn.setAttribute('aria-valuetext', `Force ${fmt(b)}, ${name}, ${fmt(heroState.windMps)} metres per second from the west`);
  sky?.update();
  pushParam('weather.wind.speed_mps', heroState.windMps, 1.0);
});
bindRange(thIn, (p) => {
  heroState.thunderKm = logMap(p, 0.5, 25);
  const s = heroState.thunderKm * secPerKm(TEMP_C);
  $('#c-thunder-out').textContent = `${fmt(heroState.thunderKm)} km · ${fmt(s)} s`;
  thIn.setAttribute('aria-valuetext', `${fmt(s)} seconds between flash and thunder, ${fmt(heroState.thunderKm)} kilometres away`);
  if (!pending) idleCounterText();
  pushParam('weather.thunder.distance_km', heroState.thunderKm, 2.0);
});
thOn.addEventListener('change', () => {
  heroState.thunderOn = thOn.checked;
  thIn.disabled = !thOn.checked;
  if (!pending) idleCounterText();
  nextSimFlashAt = performance.now() + 4000;
  pushParam('weather.thunder.enabled', heroState.thunderOn, 0);
});
shelterIn.addEventListener('change', () => {
  heroState.shelter = shelterIn.value;
  if (engine && listening) { try { engine.setScene(buildScene()); } catch (e) { console.warn(e); } }
});

function buildScene() {
  const s = heroState;
  const objects = [
    { id: 'oak', type: 'tree', material: 'leaves', position: [-9, 14, 0], size: [8, 8, 14], rotation_deg: 0, params: { species: 'deciduous', height: 14 } },
    { id: 'pine', type: 'tree', material: 'leaves', position: [12, 22, 0], size: [5, 5, 18], rotation_deg: 0, params: { species: 'conifer', height: 18 } },
    { id: 'puddle', type: 'water', material: 'puddle', position: [3, -4, 0], size: [1.6, 1.1, 0.01], rotation_deg: 0, params: { depth_mm: 8 } },
    { id: 'power-line', type: 'wire', material: '', position: [0, 25, 7], size: [40, 0.03, 0], rotation_deg: 90, params: { diameter_m: 0.03 } }
  ];
  if (s.shelter === 'porch' || s.shelter === 'attic') {
    objects.push({ id: 'roof', type: 'roof', material: 'tin_roof', position: [0, 0.5, s.shelter === 'attic' ? 3.2 : 2.6], size: [4, 3, 0.05], rotation_deg: 0, params: {} });
    objects.push({ id: 'downspout', type: 'downspout', material: '', position: [2, 1.8, 0], size: [0.08, 0.08, 2.6], rotation_deg: 0, params: {} });
  }
  if (s.shelter === 'umbrella') objects.push({ id: 'umbrella', type: 'umbrella', material: 'fabric', position: [0, 0, 1.95], size: [1.0, 1.0, 0.3], rotation_deg: 0, params: {} });
  if (s.shelter === 'tent') objects.push({ id: 'tent', type: 'tent', material: 'fabric', position: [0, 0, 0], size: [2.2, 1.6, 1.2], rotation_deg: 0, params: {} });
  if (s.shelter === 'car') objects.push({ id: 'car', type: 'car', material: 'car_body', position: [0, 0, 0], size: [1.8, 4.4, 1.45], rotation_deg: 0, params: {} });
  return {
    rainscene: 1,
    name: 'Landing preview',
    seed: 0x5eedcafe,
    weather: {
      precipitation: { type: 'rain', rate_mm_h: s.rate, drops: 'normal', character: 'continuous', shower_period_min: 12, hail_size_mm: 13 },
      wind: { speed_mps: s.windMps, from_deg: s.fromDeg, gust_factor: 1.5 },
      thunder: { enabled: s.thunderOn, distance_km: s.thunderKm, bearing_deg: 300, approach_kmh: 0, flashes_per_min: 1.5, stroke_intensity: 0.6 },
      temperature_c: TEMP_C, humidity_pct: 88, hour: 21, season: 'autumn', storm_arc: 'steady', arc_minutes: 20
    },
    place: {
      preset: 'garden',
      shelter: s.shelter,
      surfaces: { grass: 35, leaves: 25, puddle: 15, concrete: 10, wood: 15 },
      window_open: s.shelter === 'room_open' ? 0.5 : 0,
      wipers: s.shelter === 'car' ? 'slow' : 'off',
      objects
    },
    listener: { position: [0, 0, 0], height_m: s.shelter === 'tent' ? 0.6 : 1.6, heading_deg: 0, render: 'binaural' },
    mode: 'none',
    timeline: []
  };
}

const listenBtn = $('#listen'), listenLabel = $('#listen-label'), listenStatus = $('#listen-status');

// The first play fades in over 3 s to a modest level, because the system volume is unknown.
const FADE_FROM_DB = -48, FADE_TO_DB = -4, FADE_SEC = 3;
let fadeTimer = 0;
function fadeInMaster() {
  clearInterval(fadeTimer);
  if (!engine || !engine.setMasterGainDb) return;
  const t0 = performance.now();
  const a0 = Math.pow(10, FADE_FROM_DB / 20), a1 = Math.pow(10, FADE_TO_DB / 20);
  fadeTimer = setInterval(() => {
    const p = clamp((performance.now() - t0) / (FADE_SEC * 1000), 0, 1);
    let ok = true;
    try { engine.setMasterGainDb(20 * Math.log10(a0 + p * (a1 - a0))); } catch (e) { ok = false; }
    if (p >= 1 || !ok) clearInterval(fadeTimer);
  }, 50);
}
function loadEngine() {
  if (!enginePromise) enginePromise = import('../engine/weather-engine.js?v=20261117a').catch((err) => { enginePromise = null; throw err; });
  return enginePromise;
}
const prefetch = () => { loadEngine().catch(() => {}); };
listenBtn.addEventListener('pointerenter', prefetch, { once: true });
listenBtn.addEventListener('focus', prefetch, { once: true });
listenBtn.addEventListener('touchstart', prefetch, { once: true, passive: true });

function setStatus(text, isError = false) {
  listenStatus.textContent = text;
  listenStatus.classList.toggle('is-error', isError);
}
function setListening(on) {
  listening = on;
  listenBtn.setAttribute('aria-pressed', String(on));
  listenLabel.textContent = on ? 'Listening' : 'Listen';
  $('.visually-hidden', listenBtn).textContent = on ? 'Pause' : 'Listen';
  $('.i-play', listenBtn).hidden = on;
  $('.i-pause', listenBtn).hidden = !on;
  $('#meter').hidden = !on;
  if (!on) nextSimFlashAt = performance.now() + 6000;
}
function wireEngine(e) {
  e.on('flash', (ev) => {
    const km = ev.distanceKm ?? heroState.thunderKm;
    onFlash({ id: ev.id, km, delay: ev.delaySec ?? km * secPerKm(TEMP_C), bright: ev.brightness, bearing: 300, sim: false });
  });
  e.on('thunder', (ev) => onThunder({ km: ev.distanceKm ?? heroState.thunderKm, bearing: ev.bearingDeg ?? 300 }));
  e.on('caption', (ev) => { if (ev && ev.text) { engineCaptions = true; setCaption(ev.text); } });
  e.on('meter', (ev) => {
    if (!ev) return;
    const l = Number(ev.lufsMomentary), pk = Number(ev.peakDb);
    $('#meter-text').textContent = `${Number.isFinite(l) ? fmt(l) : '--'} LUFS-M · peak ${Number.isFinite(pk) ? fmt(pk) : '--'} dB`;
    $('#meter-fill').style.width = `${clamp(((Number.isFinite(l) ? l : -70) + 60) / 60, 0, 1) * 100}%`;
  });
}
listenBtn.addEventListener('click', async () => {
  if (busy) return;
  if (listening) {
    clearInterval(fadeTimer);
    releaseAudio();
    try { await engine.suspend(); } catch (e) { /* already suspended */ }
    setListening(false);
    setStatus('Paused. Press to resume.');
    return;
  }
  // Must run before any await: Safari/iOS only start audio created inside the click itself.
  const unlocked = unlockAudio(engine ? engine.context : null);
  busy = true;
  listenBtn.setAttribute('aria-busy', 'true');
  setStatus('Starting the engine…');
  try {
    if (!engine) {
      const mod = await loadEngine();
      const Ctor = mod.WeatherEngine || mod.default;
      const e = new Ctor({ context: unlocked });
      await e.start();
      if (e.context && e.context.state === 'suspended') await e.context.resume();
      e.setVisualSafety?.({ maxFlashesPerSec: 3 });
      e.setRenderMode?.('binaural');
      e.setMasterGainDb?.(FADE_FROM_DB);
      e.setScene(buildScene());
      e.setListener?.({ position: [0, 0, 0], headingDeg: 0, pitchDeg: 0 });
      wireEngine(e);
      engine = e;
    } else {
      engine.setMasterGainDb?.(FADE_FROM_DB);
      await engine.resume();
    }
    if (engine.context && engine.context.state !== 'running') {
      setStatus('Your browser is holding the sound. Press Listen again; on iPhone, also check the volume.', true);
      return;
    }
    fadeInMaster();
    setListening(true);
    setStatus('Synthesized live in your browser. Move the controls.');
  } catch (err) {
    console.error(err);
    setStatus('The audio engine could not start in this browser. Try the Composer.', true);
  } finally {
    busy = false;
    listenBtn.removeAttribute('aria-busy');
  }
});
window.addEventListener('pagehide', () => { try { engine?.stop(); } catch (e) { /* closing */ } });

idleCounterText();
scheduleSimFlash(performance.now() - 30000);
nextSimFlashAt = Math.max(nextSimFlashAt, performance.now() + 5000);

/* ================================================================== *
 * Sound in space: a listener under a hemisphere. Rain impacts are     *
 * points of light, wind streams from its bearing, and the thunder     *
 * wavefront arrives after the flash. Scrolling adds speaker layouts.  *
 * ================================================================== */
const SPEAKERS = {
  L: [-30, 0], R: [30, 0], C: [0, 0], LFE: [0, -1], Ls: [-110, 0], Rs: [110, 0],
  Lss: [-90, 0], Rss: [90, 0], Lrs: [-150, 0], Rrs: [150, 0],
  Ltf: [-45, 45], Rtf: [45, 45], Ltr: [-135, 45], Rtr: [135, 45]
};
const LAYOUTS = [[], ['L', 'R'], ['L', 'R', 'C', 'LFE', 'Ls', 'Rs'],
  ['L', 'R', 'C', 'LFE', 'Lss', 'Rss', 'Lrs', 'Rrs', 'Ltf', 'Rtf', 'Ltr', 'Rtr']];
const D2R = Math.PI / 180;
const onSphere = (az, el, r = 1) => [Math.sin(az * D2R) * Math.cos(el * D2R) * r, Math.sin(el * D2R) * r, -Math.cos(az * D2R) * Math.cos(el * D2R) * r];

class Dome {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.step = 0;
    this.spk = Object.fromEntries(Object.keys(SPEAKERS).map((k) => [k, 0]));
    this.impacts = Array.from({ length: 120 }, () => this.newImpact({}, true));
    this.wind = Array.from({ length: 30 }, () => this.newGust({}, true));
    this.t0 = performance.now();
    this.nextBolt = 1.5;
    this.bolt = null;
    this.pulse = -1;
    this.running = false;
    this.frame = this.frame.bind(this);
    this.resize();
  }
  newImpact(p, anywhere) {
    const r = Math.sqrt(Math.random()) * 0.98, a = Math.random() * Math.PI * 2;
    p.x = Math.cos(a) * r; p.z = Math.sin(a) * r; p.y = Math.random() < 0.15 ? Math.random() * 0.25 : 0;
    p.life = 0.5 + Math.random() * 1.1; p.t = anywhere ? Math.random() * p.life : 0;
    p.accent = Math.random() < 0.18; p.ring = Math.random() < 0.3;
    return p;
  }
  newGust(g, anywhere) {
    g.y = 0.06 + Math.random() * 0.55;
    g.z = (Math.random() * 2 - 1) * Math.sqrt(Math.max(0, 1 - g.y * g.y)) * 0.92;
    const span = Math.sqrt(Math.max(0, 1 - g.y * g.y - g.z * g.z));
    g.x = anywhere ? (Math.random() * 2 - 1) * span : -span;
    g.v = 0.22 + Math.random() * 0.22;
    return g;
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = Math.max(1, r.width); this.h = Math.max(1, r.height);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const R = Math.min(this.w * 0.44, this.h * 0.62);
    this.dist = 3.4; this.f = R * this.dist;
    this.cx = this.w / 2; this.cy = this.h * 0.64;
    this.draw(performance.now());
  }
  setStep(s) {
    if (s === this.step) return;
    this.step = s;
    if (!this.running) { for (const k in this.spk) this.spk[k] = LAYOUTS[s].includes(k) ? 1 : 0; this.draw(performance.now()); }
  }
  proj(x, y, z) {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const X = x * cy - z * sy, Z = x * sy + z * cy;
    const cp = Math.cos(0.44), sp = Math.sin(0.44);
    const Y2 = y * cp - Z * sp, Z2 = y * sp + Z * cp;
    const s = this.f / (this.dist - Z2);
    return [this.cx + X * s, this.cy - Y2 * s, s];
  }
  poly(pts, close) {
    const c = this.ctx;
    c.beginPath();
    pts.forEach((p, i) => { const q = this.proj(p[0], p[1], p[2]); if (i) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]); });
    if (close) c.closePath();
    c.stroke();
  }
  ring(r, y) {
    const pts = [];
    for (let i = 0; i < 96; i++) { const a = (i / 96) * Math.PI * 2; pts.push([Math.cos(a) * r, y, Math.sin(a) * r]); }
    this.poly(pts, true);
  }
  label(text, x, y, a) {
    const c = this.ctx;
    c.fillStyle = `rgba(245,247,250,${0.5 * a})`;
    c.fillText(text, x, y);
  }
  draw(now) {
    const c = this.ctx, t = reduceMotion ? 1.6 : (now - this.t0) / 1000;
    this.yaw = reduceMotion ? -0.18 : 0.32 * Math.sin(t * 0.07) - 0.12;
    c.clearRect(0, 0, this.w, this.h);
    c.lineWidth = 1;
    c.font = '500 10px ui-monospace, "SF Mono", Menlo, Consolas, monospace';
    c.textAlign = 'center';
    if ('letterSpacing' in c) c.letterSpacing = '1.5px';

    // the hemisphere: ground rings, latitudes and meridians
    c.strokeStyle = 'rgba(255,255,255,0.2)';
    this.ring(1, 0);
    c.strokeStyle = 'rgba(255,255,255,0.08)';
    this.ring(0.66, 0); this.ring(0.33, 0);
    this.ring(Math.cos(30 * D2R), Math.sin(30 * D2R));
    this.ring(Math.cos(60 * D2R), Math.sin(60 * D2R));
    c.strokeStyle = 'rgba(255,255,255,0.06)';
    for (let m = 0; m < 12; m++) {
      const pts = [];
      for (let e = 0; e <= 90; e += 6) pts.push(onSphere(m * 30, e));
      this.poly(pts, false);
    }
    this.poly([[-1, 0, 0], [1, 0, 0]], false);
    this.poly([[0, 0, -1], [0, 0, 1]], false);

    const spkOn = LAYOUTS[this.step].length > 0;
    const dim = spkOn ? 0.45 : 1;

    // rain impacts
    for (const p of this.impacts) {
      const k = 1 - p.t / p.life;
      const q = this.proj(p.x, p.y, p.z);
      const a = Math.pow(k, 1.6) * dim;
      c.fillStyle = p.accent ? `rgba(143,211,255,${0.9 * a})` : `rgba(245,247,250,${0.75 * a})`;
      c.beginPath();
      c.arc(q[0], q[1], Math.max(0.8, q[2] * 0.006), 0, Math.PI * 2);
      c.fill();
      if (p.ring) {
        const rr = (1 - k) * 0.07 * q[2];
        c.strokeStyle = `rgba(245,247,250,${0.35 * a})`;
        c.beginPath();
        c.ellipse(q[0], q[1], rr, rr * 0.42, 0, 0, Math.PI * 2);
        c.stroke();
      }
    }

    // wind streams from the west, flowing east
    c.lineCap = 'round';
    for (const g of this.wind) {
      const inside = clamp(1 - (g.x * g.x + g.y * g.y + g.z * g.z), 0, 1);
      const a = Math.sqrt(inside) * 0.5 * dim;
      if (a < 0.01) continue;
      const q0 = this.proj(g.x - 0.14, g.y, g.z), q1 = this.proj(g.x, g.y, g.z);
      c.strokeStyle = `rgba(205,226,246,${a})`;
      c.beginPath(); c.moveTo(q0[0], q0[1]); c.lineTo(q1[0], q1[1]); c.stroke();
    }
    const wq = this.proj(-0.97, 0.05, 0.1);
    c.textAlign = 'left';
    this.label('WIND 270°', wq[0], wq[1] - 8, 1);
    c.textAlign = 'center';

    // lightning and its wavefront
    const b = this.bolt;
    if (b) {
      const bt = t - b.t;
      const bq = this.proj(b.p[0], b.p[1], b.p[2]);
      const fl = reduceMotion ? 0.35 : bt < 0.08 ? 1 : Math.exp(-(bt - 0.08) / 0.35);
      if (fl > 0.01) {
        const gr = c.createRadialGradient(bq[0], bq[1], 0, bq[0], bq[1], 60);
        gr.addColorStop(0, `rgba(244,241,232,${0.9 * fl})`);
        gr.addColorStop(1, 'rgba(244,241,232,0)');
        c.fillStyle = gr;
        c.fillRect(bq[0] - 60, bq[1] - 60, 120, 120);
      }
      this.label('FLASH 300°', bq[0], bq[1] - 16, 1);
      const R = bt * 0.36;
      const gx = b.p[0], gz = b.p[2], reach = Math.hypot(gx, gz);
      if (R > 0 && R < reach + 1.1) {
        const fade = clamp(1 - (R - reach) / 1.1, 0, 1);
        c.strokeStyle = `rgba(143,211,255,${0.75 * fade})`;
        c.lineWidth = 1.2;
        c.beginPath();
        let pen = false;
        for (let i = 0; i <= 160; i++) {
          const a = (i / 160) * Math.PI * 2;
          const x = gx + Math.cos(a) * R, z = gz + Math.sin(a) * R;
          if (x * x + z * z > 1) { pen = false; continue; }
          const q = this.proj(x, 0, z);
          if (pen) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]);
          pen = true;
        }
        c.stroke();
        c.lineWidth = 1;
        if (R >= reach && this.pulse < b.t) this.pulse = t;
      }
    }

    // the listener
    const o = this.proj(0, 0, 0);
    const pt = t - this.pulse;
    if (this.pulse > 0 && pt < 1.4) {
      const pr = 6 + pt * 40;
      c.strokeStyle = `rgba(143,211,255,${0.8 * (1 - pt / 1.4)})`;
      c.beginPath(); c.arc(o[0], o[1], pr, 0, Math.PI * 2); c.stroke();
    }
    c.strokeStyle = 'rgba(245,247,250,0.55)';
    c.beginPath(); c.ellipse(o[0], o[1], 0.07 * o[2], 0.03 * o[2], 0, 0, Math.PI * 2); c.stroke();
    const nose = this.proj(0, 0, -0.11);
    c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(nose[0], nose[1]); c.stroke();
    c.fillStyle = '#f5f7fa';
    c.beginPath(); c.arc(o[0], o[1], 3.5, 0, Math.PI * 2); c.fill();
    this.label('LISTENER', o[0], o[1] + 22, 1);

    // speakers for the current layout
    for (const k in SPEAKERS) {
      const a = this.spk[k];
      if (a < 0.01) continue;
      const [az, el] = SPEAKERS[k];
      const p = el < 0 ? [0.22, 0, -0.62] : onSphere(az, el, 0.94);
      const q = this.proj(p[0], p[1], p[2]);
      c.strokeStyle = `rgba(245,247,250,${0.16 * a})`;
      c.beginPath(); c.moveTo(q[0], q[1]); c.lineTo(o[0], o[1]); c.stroke();
      c.fillStyle = `rgba(245,247,250,${a})`;
      const sz = k === 'LFE' ? 9 : 7;
      c.fillRect(q[0] - sz / 2, q[1] - sz / 2, sz, sz);
      this.label(k.toUpperCase(), q[0], q[1] - 12, a * 1.6);
    }
  }
  frame(now) {
    if (!this.running) return;
    const dt = Math.min(0.05, (now - (this.last || now)) / 1000);
    this.last = now;
    const t = (now - this.t0) / 1000;
    for (const p of this.impacts) { p.t += dt; if (p.t >= p.life) this.newImpact(p, false); }
    for (const g of this.wind) {
      g.x += g.v * dt;
      if (g.x * g.x + g.y * g.y + g.z * g.z > 1.02 && g.x > 0) this.newGust(g, false);
    }
    if (t >= this.nextBolt) { this.bolt = { t, p: onSphere(300, 22) }; this.nextBolt = t + 7.5; }
    const target = LAYOUTS[this.step];
    for (const k in this.spk) this.spk[k] += ((target.includes(k) ? 1 : 0) - this.spk[k]) * Math.min(1, dt * 4);
    this.draw(now);
    this.raf = requestAnimationFrame(this.frame);
  }
  start() { if (this.running) return; this.running = true; this.last = 0; this.raf = requestAnimationFrame(this.frame); }
  stop() { this.running = false; cancelAnimationFrame(this.raf); }
}

(function space() {
  const section = $('#space'), canvas = $('#space-canvas');
  if (!section || !canvas || !canvas.getContext) return;
  const dome = new Dome(canvas);
  if (reduceMotion) dome.bolt = { t: 0, p: onSphere(300, 22) };
  const items = $$('#layouts li');
  let visible = false;
  const sync = () => { if (visible && !document.hidden && !reduceMotion) dome.start(); else { dome.stop(); dome.draw(performance.now()); } };
  const onScroll = () => {
    const r = section.getBoundingClientRect();
    const span = Math.max(1, r.height - window.innerHeight);
    const p = clamp(-r.top / span, 0, 0.999);
    const step = Math.floor(p * 4);
    if (step !== dome.step) items.forEach((li, i) => li.classList.toggle('is-active', i === step));
    dome.setStep(step);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  if ('IntersectionObserver' in window) new IntersectionObserver(([e]) => { visible = e.isIntersecting; sync(); }).observe(section);
  document.addEventListener('visibilitychange', sync);
  let rq = 0;
  const onResize = () => { if (!rq) rq = requestAnimationFrame(() => { rq = 0; dome.resize(); }); };
  if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(canvas); else window.addEventListener('resize', onResize);
  motionMQ.addEventListener?.('change', sync);
})();

/* ================================================================== *
 * Code examples: highlighting, tabs, copy                             *
 * ================================================================== */
(function code() {
  const RULES = {
    python: /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(import|from|as|for|in|with|def|return|if|else|elif|True|False|None|and|or|not|class|lambda|yield|async|await)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g,
    js: /(\/\/[^\n]*)|(`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(import|from|const|let|await|async|new|return|if|else|function|true|false|null|of)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_$][\w$]*)(?=\()/g,
    bash: /(#[^\n]*)|("(?:\\.|[^"\\])*"|'[^']*')|(^|\s)(--?[A-Za-z][\w-]*)|\b(raingen)\b|\b(\d+(?:\.\d+)?)\b/gm,
    json: /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g
  };
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;
  const tok = {
    python: (m) => m[1] ? span('tok-c', m[0]) : m[2] ? span('tok-s', m[0]) : m[3] ? span('tok-k', m[0]) : m[4] ? span('tok-n', m[0]) : m[5] ? span('tok-f', m[0]) : esc(m[0]),
    bash: (m) => m[1] ? span('tok-c', m[0]) : m[2] ? span('tok-s', m[0]) : m[4] ? esc(m[3]) + span('tok-p', m[4]) : m[5] ? span('tok-f', m[0]) : m[6] ? span('tok-n', m[0]) : esc(m[0]),
    json: (m) => m[1] ? (m[2] ? span('tok-f', m[1]) + esc(m[2]) : span('tok-s', m[1])) : m[3] ? span('tok-k', m[0]) : m[4] ? span('tok-n', m[0]) : esc(m[0])
  };
  tok.js = tok.python;
  const highlight = (src, lang) => {
    const re = RULES[lang];
    if (!re) return esc(src);
    re.lastIndex = 0;
    let out = '', last = 0, m;
    while ((m = re.exec(src))) {
      if (!m[0].length) { re.lastIndex++; continue; }
      out += esc(src.slice(last, m.index)) + tok[lang](m);
      last = m.index + m[0].length;
    }
    return out + esc(src.slice(last));
  };
  $$('code[data-lang]').forEach((el) => { el.innerHTML = highlight(el.textContent, el.dataset.lang); });

  const tabs = $$('[role="tab"]');
  const select = (tab, focus = false) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
    if (focus) tab.focus();
  };
  // every panel starts visible so the page reads without JS
  select(tabs.find((t) => t.getAttribute('aria-selected') === 'true') || tabs[0]);
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t));
    t.addEventListener('keydown', (e) => {
      let j = null;
      if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = tabs.length - 1;
      if (j != null) { e.preventDefault(); select(tabs[j], true); }
    });
  });

  $$('.copy-btn').forEach((b) => b.addEventListener('click', async () => {
    const text = document.getElementById(b.dataset.copy).textContent;
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
      ta.remove();
    }
    b.textContent = ok ? 'Copied' : 'Press Ctrl+C';
    setTimeout(() => { b.textContent = 'Copy'; }, 1600);
  }));
})();
