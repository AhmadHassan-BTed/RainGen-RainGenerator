/**
 * Bootstrap of the raingen 4D Environment Composer.
 *
 * Wires one shared parameter graph (scene-store.js: one set of settings behind
 * both workspaces) to the views (Listener overlay, Library, 3-D viewport,
 * Inspector, Timeline, Captions) and to the Web Audio engine
 * (../../engine/weather-engine.js, used through its public API only).
 *
 * Behaviour:
 * - No autoplay: one big "Press to hear the rain" gesture starts the
 *   AudioContext, then the scene fades in over 3 s to a safe level.
 * - Every store change is forwarded to the engine with the parameter's smoothing
 *   time ("time to 99.5 %", Engineer > Smoothing in the inspector).
 * - Flash safety (WCAG 2.3.1) sits outside the parameter graph: the engine is
 *   told maxFlashesPerSec = 3 and the UI limits again.
 * - Flash-to-bang counts real seconds between the 'flash' and 'thunder' events;
 *   distance = c(T) * t with c(T) = 331.3 sqrt(1 + T/273.15) m/s.
 * - Degrades with a plain message instead of failing: no WebGL -> audio-only;
 *   no AudioWorklet or a blocked AudioContext -> the editor still works and
 *   exports scenes for the Python CLI; an unreadable share link or file -> the
 *   current scene stays and a toast says why.
 */
import { SceneStore, clone, paramSpec } from './scene-store.js';
import { unlockAudio } from '../../engine/unlock.js?v=20261117a';
import { presetScene, vibeDrivers, DEMO_WALK, stormPosition, rangeBearing } from './presets.js';
import { Library, instantiate, objectLabel } from './library.js';
import { Inspector } from './inspector.js';
import { Timeline, fmtClock, fmtSMPTE } from './timeline.js';
import { ListenerMode } from './listener-mode.js';
import {
  announce, CaptionsPanel, FlashLimiter, prefersReducedMotion, onReducedMotionChange, isTyping, openDialog,
  closeDialog, compassWord, flashBang, kmFromFlashBang, fmtNum, clamp, speeds,
} from './a11y.js';

const ENGINE_URL = '../../engine/weather-engine.js?v=20261117a';
const BRIDGE_URL = '../../engine/native-bridge.js?v=20261117b';
const LS = 'raingen.studio';
const DEG = Math.PI / 180;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const prefs = {
  get(k, d) {
    try {
      const v = localStorage.getItem(`${LS}.${k}`);
      return v === null ? d : JSON.parse(v);
    } catch (_) {
      return d;
    }
  },
  set(k, v) {
    try { localStorage.setItem(`${LS}.${k}`, JSON.stringify(v)); } catch (_) { /* storage blocked */ }
  },
};

/* ------------------------------------------------------------------ state */

const storedLayers = prefs.get('layers', {});

const ui = {
  workspace: prefs.get('workspace', 'listener') === 'studio' ? 'studio' : 'listener',
  visuals: ['full', 'calm', 'off'].includes(prefs.get('visuals', 'full')) ? prefs.get('visuals', 'full') : 'full',
  volume: clamp(Number(prefs.get('volume', 70)) || 0, 0, 100),
  singleKeys: prefs.get('singleKeys', true) !== false,
  layers: storedLayers && typeof storedLayers === 'object' && !Array.isArray(storedLayers) ? storedLayers : {},
  sleepFade: 1,
  fadeInDb: 0,
};

const transport = { playing: false, t: 0, anchorT: 0, anchorPerf: 0 };
const nowS = () => performance.now() / 1000;

const AUDIO_SUPPORTED = typeof window.AudioContext === 'function' || typeof window.webkitAudioContext === 'function';
const WORKLET_SUPPORTED = AUDIO_SUPPORTED && typeof window.AudioWorkletNode === 'function' && window.isSecureContext !== false;

/*
 * Native mode: served by `raingen studio` (a local Python server), the sound comes from the
 * Rust core through ../../engine/native-bridge.js (same interface as WeatherEngine). `raingen
 * studio` opens the page with ?engine=native; on a local host the page also asks
 * /native/hello. Anywhere else (GitHub Pages) nothing changes. ?engine=browser forces Web Audio.
 */
const ENGINE_FLAG = (() => {
  try {
    return new URLSearchParams(location.search).get('engine');
  } catch (_) {
    return null;
  }
})();
const LOCAL_HOST = /^(127\.0\.0\.1|localhost|\[::1\])$/.test(location.hostname);
const native = { on: ENGINE_FLAG === 'native', pending: false, ready: Promise.resolve(false), cls: null, hello: null };
if (ENGINE_FLAG === 'native' || (LOCAL_HOST && ENGINE_FLAG !== 'browser' && location.protocol === 'http:')) {
  native.pending = true;
  native.ready = import(BRIDGE_URL)
    .then(async (m) => {
      native.cls = m.NativeBridge || m.default;
      native.hello = await native.cls.detect();
      native.on = !!native.hello;
      return native.on;
    })
    .catch((err) => {
      console.warn('[studio] native bridge unavailable:', err && err.message ? err.message : err);
      native.on = false;
      return false;
    })
    .finally(() => {
      native.pending = false;
      queueMicrotask(() => onNativeDetected());
    });
}

/** Sound is possible: the native engine (raingen studio) or Web Audio with AudioWorklet. */
function audioCapable() {
  return native.on || WORKLET_SUPPORTED;
}

const audio = {
  cls: null, loading: null, engine: null, started: false, state: 'idle',
  liveRates: new Map(), lastPitch: 0, warnedSilent: false, warnedBlocked: false, startP: null, startAt: 0, sceneTimer: 0,
  timeTimer: 0, lastTimePush: 0,
};

const f2b = { pending: new Map(), seen: new Set(), last: null };
const glowLimiter = new FlashLimiter({ maxPerSec: 3 });
const phoneMQ = typeof matchMedia === 'function' ? matchMedia('(max-width: 759px)') : { matches: false, addEventListener() {} };

let viewport = null;
let lastGustCaption = 0;
let ownHash = '';
let renderURL = null;

/* ------------------------------------------------------------------ store */

const store = new SceneStore({ storageKey: LS });
store.setDriverFn(vibeDrivers);
const sharedScene = SceneStore.decodeHash(location.hash);
const brokenShareLink = !sharedScene && SceneStore.hasSceneHash(location.hash);
const initialScene = sharedScene || store.loadAutosave() || presetScene('leaves-night');
store.load(initialScene, { history: false, source: 'init', label: initialScene.name });
if (sharedScene) ownHash = location.hash;
if (brokenShareLink) clearHash();

/* ------------------------------------------------------------------ small helpers */

function safe(fn) {
  try {
    return fn();
  } catch (err) {
    console.warn('[studio] engine call failed:', err);
    return undefined;
  }
}

async function safeAsync(fn) {
  try {
    return await fn();
  } catch (err) {
    console.warn('[studio] engine call failed:', err);
    return undefined;
  }
}

/** Drop the #… part of the address without adding a history entry. */
function clearHash() {
  try {
    history.replaceState(null, '', location.pathname + location.search);
  } catch (_) { /* sandboxed frame: leave the address as it is */ }
  ownHash = '';
}

const BROKEN_LINK_MESSAGE = 'This share link is damaged or incomplete, so its scene could not be opened. Ask for the link again, or open the scene as a JSON file with Import.';

function slug(name) {
  return String(name || 'weather').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'weather';
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) e.append(kid instanceof Node ? kid : String(kid));
  return e;
}

/** Non-modal toast; opts.action = {label, fn}. */
function toast(message, { action = null, timeout = 6000 } = {}) {
  const host = $('#toasts');
  if (!host) return;
  const t = el('div', { class: 'toast' }, el('span', { class: 'toast-text' }, message));
  if (action) {
    t.append(el('button', {
      type: 'button', class: 'toast-action',
      onclick: () => {
        try { action.fn(); } finally { t.remove(); }
      },
    }, action.label));
  }
  t.append(el('button', { type: 'button', class: 'toast-close', 'aria-label': 'Dismiss', onclick: () => t.remove() }, '×'));
  host.append(t);
  while (host.children.length > 3) host.firstElementChild.remove();
  setTimeout(() => t.remove(), action ? Math.max(timeout, 9000) : timeout);
}

function hasWebGL() {
  try {
    if (!window.WebGLRenderingContext) return false;
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    if (!gl) return false;
    // release the probe context at once: browsers cap the number of live contexts
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  } catch (_) {
    return false;
  }
}

/* ------------------------------------------------------------------ captions & subtitles */

const captions = new CaptionsPanel($('#captions'), { fmtTime: (t) => fmtClock(t) });
let subtitleTimer = 0;

function showSubtitle(text) {
  const s = $('#subtitle');
  if (!s || !text) return;
  s.textContent = text;
  s.classList.add('on');
  clearTimeout(subtitleTimer);
  subtitleTimer = setTimeout(() => s.classList.remove('on'), 4500);
}

function addCaption(text, kind = 'event') {
  captions.add({ text, t: transport.t, kind });
  if (ui.workspace === 'listener' || kind === 'note') showSubtitle(text);
}

/* ------------------------------------------------------------------ audio engine */

function setAudioState(state) {
  audio.state = state;
  const chip = $('#audio-status');
  if (!chip) return;
  chip.dataset.state = state;
  const txt = native.on ? nativeChipText(state) : {
    idle: 'Audio off', starting: 'Starting…', live: transport.playing ? 'Live' : 'Paused', blocked: 'Sound blocked', unsupported: 'No audio', error: 'Audio error',
  }[state] || state;
  if (chip.textContent !== txt) chip.textContent = txt;
}

/** 'Rust engine · 101 voices' (the sound is played by the raingen studio process). */
function nativeChipText(state) {
  const e = audio.engine && audio.engine.native ? audio.engine : null;
  const info = (e && (e._stats || e.info)) || native.hello || {};
  const kind = info.engine === 'python' ? 'Python engine' : 'Rust engine';
  const n = e ? e.voices : 0;
  if (state === 'starting') return `${kind} · starting…`;
  if (state === 'error') return `${kind} · error`;
  if (state === 'live') return transport.playing ? `${kind} · ${n ? `${n} voices` : 'live'}` : `${kind} · paused`;
  return `${kind} · ready`;
}

/** Called once the /native/hello probe has answered (or failed). */
function onNativeDetected() {
  viewport?.setLite(native.on);
  if (!native.on) {
    if (ENGINE_FLAG === 'native') {
      toast('The raingen studio server is not answering, so the browser audio engine is used instead.');
    }
    if (!audio.started) setAudioState(audioCapable() ? 'idle' : 'unsupported');
    return;
  }
  const chip = $('#audio-status');
  if (chip) chip.title = `Sound from the ${native.hello.engine === 'python' ? 'Python' : 'native Rust'} core (raingen studio ${native.hello.version || ''}, ${native.hello.layout || 'binaural'})`;
  if (!audio.started) setAudioState('idle');
  const h = native.hello;
  // the scene named on the command line (`raingen studio tent-in-a-storm`) opens once per server run
  if (h.scene && h.sceneId && prefs.get('nativeSceneId', '') !== h.sceneId && !sharedScene) {
    prefs.set('nativeSceneId', h.sceneId);
    store.load(h.scene, { label: h.scene.name || 'raingen scene' });
    seek(0);
    toast(`Opened “${store.scene.name}” from raingen studio.`);
  }
}

/** Plain-language reason an audio start failed (the raw error goes to the console). */
function audioErrorText(err) {
  const name = err && err.name;
  const msg = String((err && err.message) || err || '');
  if (name === 'NotAllowedError') return 'The browser blocked sound on this page. Press Play to allow it.';
  if (/AudioWorklet|Web Audio/i.test(msg)) return 'This browser cannot run the audio engine (it needs AudioWorklet on an https page).';
  if (/fetch|import|load|network/i.test(msg) || name === 'TypeError') return 'The audio engine could not be loaded. Check your connection and press Play again.';
  return 'The audio engine could not start.';
}

function loadEngineModule() {
  if (audio.cls) return Promise.resolve(audio.cls);
  if (!WORKLET_SUPPORTED) return Promise.reject(new Error('AudioWorklet is not available (it needs a secure https page)'));
  if (!audio.loading) {
    audio.loading = import(ENGINE_URL).then((m) => {
      const C = m.WeatherEngine || m.default;
      if (typeof C !== 'function') throw new Error('weather-engine.js does not export WeatherEngine');
      audio.cls = C;
      return C;
    }, (err) => {
      audio.loading = null;
      throw err;
    });
  }
  return audio.loading;
}

async function ensureEngine(context = null) {
  if (audio.engine) return audio.engine;
  if (native.pending) await native.ready;
  const C = native.on ? native.cls : await loadEngineModule();
  const e = new C({ context, hello: native.hello });
  bindEngine(e);
  audio.engine = e;
  return e;
}

function engineRunning() {
  try {
    return !!audio.engine?.running;
  } catch (_) {
    return false;
  }
}

function renderModeOf(scene) {
  return scene.listener.render === 'stereo' ? 'stereo' : 'binaural';
}

function pushListener(pitch = audio.lastPitch) {
  if (!audio.started) return;
  const L = store.scene.listener;
  safe(() => audio.engine.setListener({ position: L.position.slice(), headingDeg: L.heading_deg || 0, pitchDeg: pitch || 0 }));
}

function pushWholeScene() {
  if (!audio.started) return;
  clearTimeout(audio.sceneTimer);
  const e = audio.engine;
  safe(() => e.setScene(clone(store.scene)));
  safe(() => e.setRenderMode(renderModeOf(store.scene)));
  pushListener();
}

function scheduleSceneSync() {
  if (!audio.started) return;
  clearTimeout(audio.sceneTimer);
  audio.sceneTimer = setTimeout(() => safe(() => audio.engine.setScene(clone(store.scene))), 250);
}

function engineSetTime(t, force = false) {
  if (!audio.started) return;
  clearTimeout(audio.timeTimer);
  const now = performance.now();
  if (force || now - audio.lastTimePush > 90) {
    audio.lastTimePush = now;
    safe(() => audio.engine.setTime(t));
  } else {
    audio.timeTimer = setTimeout(() => {
      audio.lastTimePush = performance.now();
      safe(() => audio.engine.setTime(transport.t));
    }, 100);
  }
}

function volToDb(pct) {
  return pct <= 0 ? -120 : 40 * Math.log10(pct / 100);
}

function applyGain() {
  if (!audio.started) return;
  const db = volToDb(ui.volume) + 20 * Math.log10(Math.max(1e-4, ui.sleepFade)) + ui.fadeInDb;
  safe(() => audio.engine.setMasterGainDb(Math.max(-120, db)));
}

/** First-play safety: ramp the master in over 3 s from -48 dB. */
function fadeIn(sec = 3) {
  const t0 = performance.now();
  ui.fadeInDb = -48;
  applyGain();
  const iv = setInterval(() => {
    const k = clamp((performance.now() - t0) / (sec * 1000), 0, 1);
    ui.fadeInDb = -48 * (1 - k) * (1 - k);
    applyGain();
    if (k >= 1) {
      ui.fadeInDb = 0;
      applyGain();
      clearInterval(iv);
    }
  }, 50);
}

const SLOW_START_MS = 6000;

/** Resolves to 'ok', 'timeout' (after ms) or { error }; never rejects. */
function settle(promise, ms) {
  return Promise.race([
    Promise.resolve(promise).then(() => 'ok', (error) => ({ error })),
    new Promise((resolve) => { setTimeout(() => resolve('timeout'), ms); }),
  ]);
}

/** Browsers only let an AudioContext start after the visitor has interacted with the page. */
function mayStartAudio() {
  const ua = navigator.userActivation;
  return !ua || ua.isActive || ua.hasBeenActive;
}

function inGesture() {
  const ua = navigator.userActivation;
  return !ua || ua.isActive;
}

function audioFailed(err) {
  console.warn('[studio] audio start failed:', err);
  setAudioState(err && err.name === 'NotAllowedError' ? 'blocked' : 'error');
  toast(native.on
    ? `The Rust engine could not start: ${err && err.message ? err.message : err}`
    : `${audioErrorText(err)} You can still build, animate and export scenes.`);
  return false;
}

/**
 * Start the engine once. Resolves true (sound running), false (no sound; the
 * reason is shown) or null (this attempt was replaced by a newer one).
 */
function startAudio() {
  if (audio.started) return Promise.resolve(true);
  if (audio.startP) {
    // A start that has hung this long had its AudioContext refused by the browser;
    // a new press of Play retires it and starts again inside this gesture.
    if (native.on || performance.now() - audio.startAt < SLOW_START_MS || !inGesture()) return audio.startP;
    audio.engine = null;
  }
  const p = startAudioOnce().finally(() => {
    if (audio.startP === p) audio.startP = null;
  });
  audio.startP = p;
  audio.startAt = performance.now();
  return p;
}

async function startAudioOnce() {
  if (native.pending) await native.ready;
  if (native.on) return startNativeAudio();
  if (!WORKLET_SUPPORTED) {
    setAudioState('unsupported');
    return false;
  }
  if (!mayStartAudio()) {
    setAudioState('blocked');
    warnBlocked();
    return false;
  }
  audio.warnedBlocked = false;
  // Before any await: Safari/iOS only start audio created inside the user's gesture.
  const unlocked = unlockAudio(audio.engine ? audio.engine.context : null);
  setAudioState('starting');
  let e;
  try {
    e = await ensureEngine(unlocked);
  } catch (err) {
    return audioFailed(err);
  }
  const begun = e.start();
  const slow = setTimeout(() => {
    if (audio.engine === e && !audio.started) toast('Sound is taking a while to start. If you hear nothing, press Play again.');
  }, SLOW_START_MS);
  const res = await begun.then(() => 'ok', (error) => ({ error }));
  clearTimeout(slow);
  if (audio.engine !== e) {
    // replaced by a newer attempt while this one hung: make sure it stays silent
    if (res === 'ok') safe(() => e.stop());
    return null;
  }
  if (res !== 'ok') return audioFailed(res.error);
  if (e.context && e.context.state === 'suspended') await settle(e.resume(), 1500);
  audio.started = true;
  audio.warnedBlocked = false;
  pushWholeScene();
  safe(() => e.setVisualSafety({ maxFlashesPerSec: 3 }));
  safe(() => e.setTime(transport.t));
  fadeIn(3);
  setAudioState(engineRunning() ? 'live' : 'blocked');
  if (!engineRunning()) warnBlocked();
  return true;
}

/**
 * Native mode: the raingen studio process plays the sound, so there is no AudioContext to
 * unlock and no gesture to wait for; the scene goes with the start request.
 */
async function startNativeAudio() {
  setAudioState('starting');
  let e;
  try {
    e = await ensureEngine();
  } catch (err) {
    return audioFailed(err);
  }
  safe(() => e.setScene(clone(store.scene)));
  safe(() => e.setRenderMode(renderModeOf(store.scene)));
  const res = await settle(e.start(), 90000);
  if (audio.engine !== e) return null;
  if (res !== 'ok') return audioFailed(res === 'timeout' ? new Error('the engine did not start in time') : res.error);
  audio.started = true;
  pushWholeScene();
  safe(() => e.setVisualSafety({ maxFlashesPerSec: 3 }));
  safe(() => e.setTime(transport.t));
  if (transport.t > 0) safe(() => e.seek(transport.t));
  fadeIn(3);
  setAudioState('live');
  return true;
}

/** The browser keeps the AudioContext suspended until a click or key press (autoplay policy). */
function warnBlocked() {
  if (audio.warnedBlocked) return;
  audio.warnedBlocked = true;
  toast('The browser is holding the sound back until you allow it.', {
    action: { label: 'Start sound', fn: () => allowSound() },
  });
}

/** Called from a click: resume the suspended engine, or start playback with sound. */
async function allowSound() {
  if (audio.started && audio.engine && transport.playing) {
    await settle(audio.engine.resume(), 3000);
    setAudioState(engineRunning() ? 'live' : 'blocked');
    return;
  }
  await play();
}

function bindEngine(e) {
  const on = (ev, fn) => safe(() => e.on(ev, (d) => {
    try { fn(d || {}); } catch (err) { console.error(`[studio] ${ev} handler failed`, err); }
  }));
  on('flash', onFlash);
  on('thunder', onThunder);
  on('gust', onGust);
  on('impact-stats', onImpactStats);
  on('meter', onMeter);
  on('caption', onEngineCaption);
  if (e.native) {
    on('stats', () => {
      if (audio.engine === e) setAudioState(audio.state);
    });
    on('error', (d) => {
      console.warn('[studio] native engine:', d.where, d.message);
      if (audio.engine === e) toast(`Rust engine: ${d.message || 'error'}`);
    });
    on('server', ({ up }) => {
      if (audio.engine !== e || !audio.started) return;
      if (up) {
        setAudioState('live');
      } else {
        setAudioState('error');
        toast('The raingen studio process is not answering. Start `raingen studio` again and reload this window.');
      }
    });
  }
  on('state', ({ state }) => {
    if (!audio.started || audio.engine !== e) return;
    if (state === 'running') {
      audio.warnedBlocked = false;
      setAudioState('live');
    } else if (state !== 'closed' && transport.playing) {
      // suspended or interrupted by the browser while playing (not by our own pause)
      setAudioState('blocked');
      warnBlocked();
    }
  });
}

/* ------------------------------------------------------------------ engine events */

function onFlash(evt) {
  const s = store.scene;
  const L = s.listener.position;
  let bearing = evt.bearingDeg;
  if (!Number.isFinite(bearing) && Array.isArray(evt.channel) && evt.channel.length) {
    let low = evt.channel[0];
    for (const p of evt.channel) if ((p[2] ?? 0) < (low[2] ?? 0)) low = p;
    bearing = ((Math.atan2(low[0] - L[0], low[1] - L[1]) / DEG) + 360) % 360;
  }
  if (!Number.isFinite(bearing)) bearing = s.weather.thunder.bearing_deg;
  const ev = { brightness: 1, ...evt, bearingDeg: bearing };
  if (ev.id !== undefined) {
    if (f2b.seen.has(ev.id)) return;
    f2b.seen.add(ev.id);
    if (f2b.seen.size > 300) f2b.seen.delete(f2b.seen.values().next().value);
    f2b.pending.set(ev.id, { start: performance.now(), delaySec: Number(ev.delaySec) || 0, km: ev.distanceKm, bearing });
  }
  if (ui.visuals === 'off') return;
  if (viewport) viewport.onFlash(ev);
  else screenGlow(ev);
}

function screenGlow(ev) {
  const g = $('#flash-glow');
  if (!g) return;
  const v = glowLimiter.admit(performance.now(), ev.brightness ?? 1, prefersReducedMotion() || ui.visuals === 'calm');
  if (v.mode === 'merge') return;
  g.style.setProperty('--flash', String(v.mode === 'flash' ? clamp(v.brightness, 0.2, 1) : Math.min(v.brightness, 0.35)));
  g.classList.remove('flash', 'glow');
  void g.offsetWidth;
  g.classList.add(v.mode === 'flash' ? 'flash' : 'glow');
}

function onThunder(evt) {
  const s = store.scene;
  const T = s.weather.temperature_c;
  const p = evt.id !== undefined ? f2b.pending.get(evt.id) : null;
  if (p) f2b.pending.delete(evt.id);
  let sec = p ? (performance.now() - p.start) / 1000 : null;
  let km = Number.isFinite(evt.distanceKm) ? evt.distanceKm : p?.km;
  if (!Number.isFinite(km) && sec !== null) km = kmFromFlashBang(sec, T);
  if (sec === null && Number.isFinite(km)) sec = flashBang(km, T);
  if (Number.isFinite(km)) f2b.last = { sec, km, at: performance.now() };
  viewport?.onThunder(evt);
  const bearing = Number.isFinite(evt.bearingDeg) ? evt.bearingDeg : p?.bearing ?? s.weather.thunder.bearing_deg;
  if (!captions.hasRecent(/thunder/i, transport.t, 1.5) && Number.isFinite(km)) {
    addCaption(`[Thunder, ${fmtNum(km, 1)} km, ${compassWord(bearing)}]`, 'thunder');
  }
}

function onGust(evt) {
  viewport?.onGust(evt);
  const w = store.scene.weather.wind;
  const v = Number(evt.speedMps);
  if (!Number.isFinite(v)) return;
  const now = performance.now();
  if (v > Math.max(8, w.speed_mps * 1.35) && now - lastGustCaption > 12000) {
    lastGustCaption = now;
    addCaption(`[Gust, ${speeds(v).kmh} km/h from the ${compassWord(Number.isFinite(evt.fromDeg) ? evt.fromDeg : w.from_deg)}]`, 'gust');
  }
}

let liveTimer = 0;
function onImpactStats(evt) {
  if (!evt.objectId) return;
  audio.liveRates.set(evt.objectId, Number(evt.ratePerSec) || 0);
  viewport?.onImpactStats({ objectId: evt.objectId, ratePerSec: Number(evt.ratePerSec) || 0 });
  if (!liveTimer && store.selection === evt.objectId) {
    liveTimer = setTimeout(() => {
      liveTimer = 0;
      inspector.updateLive();
    }, 250);
  }
}

function onMeter({ lufsMomentary, peakDb }) {
  const L = Number.isFinite(lufsMomentary) ? lufsMomentary : -70;
  const P = Number.isFinite(peakDb) ? peakDb : -120;
  const m = $('#meter');
  if (!m) return;
  const pct = clamp((L + 60) / 60, 0, 1);
  $('#meter-fill').style.width = `${(pct * 100).toFixed(1)}%`;
  const state = P > -1 ? 'clip' : L > -9 ? 'hot' : L > -16 ? 'loud' : 'ok';
  m.dataset.state = state;
  const word = { ok: 'OK', loud: 'Loud', hot: 'Hot', clip: 'Peak!' }[state];
  const txt = `${L > -69 ? fmtNum(L, 0) : '−∞'} LUFS · ${word}`;
  const te = $('#meter-text');
  if (te.textContent !== txt) te.textContent = txt;
  m.setAttribute('aria-valuenow', String(Math.round(clamp(L, -60, 0))));
  m.setAttribute('aria-valuetext', `${L > -69 ? Math.round(L) : 'silent'} LUFS momentary, peak ${P > -100 ? Math.round(P) : 'minus infinity'} dB, ${word}`);
}

function onEngineCaption({ text }) {
  if (!text) return;
  if (/thunder/i.test(text) && captions.hasRecent(/^\[Thunder/i, transport.t, 1.5)) return;
  addCaption(String(text), 'engine');
}

/* ------------------------------------------------------------------ store -> engine */

function rampFor(path, source) {
  if (source === 'automation') return 0.08;
  if (source === 'walk' || source === 'demo') return path === 'place.shelter' ? 1.5 : 0.3;
  if (path === 'place.shelter') return 2.5;
  const sm = store.scene.studio.smoothing || {};
  const spec = paramSpec(path);
  const g = spec.smoothing || 'other';
  // Live edits must be heard at once: cap the glide at 0.4 s (still click-free).
  return Math.min(0.4, Math.max(0.02, Number(sm[g] ?? 0.25)));
}

const SCENE_SYNC_PATHS = /^(timeline|overrides|vibe|name|seed|presetId|studio)(\.|$)/;
/*
 * Native mode: the live server drops the timeline, overrides, vibe macros and studio settings
 * (the Composer plays them as parameters), so editing those must not rebuild the engine.
 */
const NATIVE_NO_SYNC_PATHS = /^(timeline|overrides|vibe|name|presetId|studio)(\.|$)/;
const drag = { objects: false };

function nativeEngine() {
  return audio.started && audio.engine && audio.engine.native ? audio.engine : null;
}

/* End of a drag (gizmo, storm track, ...): the moved objects reach the native engine as one scene. */
store.on('gesture', ({ active }) => {
  if (active) {
    drag.objects = false;
    return;
  }
  const e = nativeEngine();
  if (!e || !drag.objects) return;
  drag.objects = false;
  clearTimeout(audio.sceneTimer);
  safe(() => e.setScene(clone(store.scene)));
  safe(() => e.commit());
});

store.on('change', (e) => {
  if (e.paths.includes('name') || e.kind === 'load' || e.kind === 'restore') {
    document.title = `${store.scene.name} · raingen 4D Composer`;
    const gs = $('#gate-scene');
    if (gs) gs.textContent = store.scene.name;
    updateMediaSession();
  }
  if (!audio.started || !audio.engine) return;
  const eng = audio.engine;
  if (e.kind === 'load' || e.kind === 'restore' || e.paths.includes('*')) {
    pushWholeScene();
    return;
  }
  const objIds = new Set();
  let listenerMoved = false;
  for (const p of e.paths) {
    if (p.startsWith('place.objects.')) {
      objIds.add(p.split('.')[2]);
      continue;
    }
    if (p === 'listener.position' || p === 'listener.heading_deg') {
      listenerMoved = true;
      continue;
    }
    if (p === 'listener.render') {
      safe(() => eng.setRenderMode(renderModeOf(store.scene)));
      continue;
    }
    if (SCENE_SYNC_PATHS.test(p)) {
      if (!(eng.native && NATIVE_NO_SYNC_PATHS.test(p))) scheduleSceneSync();
      continue;
    }
    const v = store.get(p);
    if (v === undefined) {
      scheduleSceneSync();
      continue;
    }
    safe(() => eng.setParam(p, clone(v), rampFor(p, e.source)));
  }
  if (objIds.size && store.gestureActive) drag.objects = true;
  for (const id of objIds) {
    const o = store.getObject(id);
    if (o) safe(() => eng.upsertObject(clone(o)));
    else safe(() => eng.removeObject(id));
  }
  if (listenerMoved && e.source !== 'walk') pushListener(e.source === 'demo' ? audio.lastPitch : 0);
});

store.on('history', ({ canUndo, canRedo }) => {
  $('#btn-undo').disabled = !canUndo;
  $('#btn-redo').disabled = !canRedo;
});

/* ------------------------------------------------------------------ transport */

function updateTransportUI() {
  const b = $('#btn-play');
  b.classList.toggle('is-playing', transport.playing);
  b.setAttribute('aria-label', transport.playing ? 'Pause' : 'Play');
  b.title = transport.playing ? 'Pause (Space)' : 'Play (Space)';
  document.documentElement.classList.toggle('is-playing', transport.playing);
  if (audio.state === 'live') setAudioState('live');
  try {
    if ('mediaSession' in navigator) navigator.mediaSession.playbackState = transport.playing ? 'playing' : 'paused';
  } catch (_) { /* unsupported */ }
}

async function play() {
  if (transport.playing) return;
  hideGate();
  const ok = await startAudio();
  if (ok === null || transport.playing) return;
  if (!ok && audio.state === 'blocked') {
    // keep the transport stopped: the next press of Play is the gesture that lets sound start
    updateTransportUI();
    return;
  }
  if (ok && audio.engine) {
    if (!engineRunning()) await settle(audio.engine.resume(), 3000);
    if (!engineRunning()) {
      setAudioState('blocked');
      warnBlocked();
      updateTransportUI();
      return;
    }
    if (audio.state === 'blocked') setAudioState('live');
    engineSetTime(transport.t, true);
  }
  transport.playing = true;
  transport.anchorT = transport.t;
  transport.anchorPerf = nowS();
  timeline.setPlaying(true);
  updateTransportUI();
  if (!ok && !audio.warnedSilent) {
    audio.warnedSilent = true;
    toast(WORKLET_SUPPORTED
      ? 'Playing the timeline silently: the audio engine is not available.'
      : 'Sound is not available here (the audio engine needs AudioWorklet on an https page). The world and timeline still play; export the scene for the Python CLI.');
  }
}

async function pause() {
  if (!transport.playing) return;
  transport.playing = false;
  timeline.setPlaying(false);
  updateTransportUI();
  if (audio.started) await safeAsync(() => audio.engine.suspend());
}

function togglePlay() {
  if (transport.playing) pause();
  else play();
}

function stop() {
  pause();
  const st = store.scene.studio;
  seek(st.loopOn ? st.loop[0] : 0);
}

function seek(t) {
  transport.t = Math.max(0, Number(t) || 0);
  transport.anchorT = transport.t;
  transport.anchorPerf = nowS();
  timeline.setTime(transport.t, { apply: true, force: true });
  viewport?.setTime(transport.t);
  const ne = nativeEngine();
  if (ne && typeof ne.seek === 'function') safe(() => ne.seek(transport.t));
  else engineSetTime(transport.t);
  updateClock();
}

function advance() {
  if (!transport.playing) return;
  const st = store.scene.studio;
  const t = transport.anchorT + (nowS() - transport.anchorPerf);
  if (st.loopOn && st.loop[1] - st.loop[0] >= 0.5 && t >= st.loop[1]) {
    seek(st.loop[0]);
    engineSetTime(st.loop[0], true);
    return;
  }
  transport.t = t;
  timeline.setTime(t);
  viewport?.setTime(t);
}

let lastClockText = '';
function updateClock() {
  const tc = fmtClock(transport.t);
  if (tc !== lastClockText) {
    lastClockText = tc;
    $('#clock-tc').textContent = tc;
    $('#clock-smpte').textContent = fmtSMPTE(transport.t);
  }
}

function expectedThunder() {
  const s = store.scene;
  const storms = s.place.objects.filter((o) => o.type === 'storm');
  let best = null;
  for (const o of storms) {
    const [x, y] = stormPosition(o, transport.t);
    const rb = rangeBearing(s, x, y);
    if (!best || rb.km < best.km) best = rb;
  }
  if (best) return { km: best.km, sec: flashBang(best.km, s.weather.temperature_c) };
  if (s.weather.thunder.enabled) return { km: s.weather.thunder.distance_km, sec: flashBang(s.weather.thunder.distance_km, s.weather.temperature_c) };
  return null;
}

let lastF2B = '';
function updateF2B() {
  const box = $('#f2b');
  const out = $('#f2b-value');
  const now = performance.now();
  let counting = null;
  for (const [id, p] of f2b.pending) {
    if (now - p.start > (p.delaySec + 12) * 1000) f2b.pending.delete(id);
    else if (!counting || p.start > counting.start) counting = p;
  }
  let text;
  let label;
  let state;
  if (counting) {
    const sec = (now - counting.start) / 1000;
    text = `${sec.toFixed(1)} s…`;
    label = `Counting since the flash: ${Math.floor(sec)} seconds`;
    state = 'counting';
  } else if (f2b.last && now - f2b.last.at < 15000) {
    text = `${f2b.last.sec.toFixed(1)} s · ${fmtNum(f2b.last.km, 1)} km`;
    label = `Last thunder: ${f2b.last.sec.toFixed(1)} seconds after the flash, ${fmtNum(f2b.last.km, 1)} kilometres away`;
    state = 'heard';
  } else {
    const ex = expectedThunder();
    text = ex ? `≈${fmtNum(ex.sec, 0)} s · ${fmtNum(ex.km, ex.km < 10 ? 1 : 0)} km` : '—';
    label = ex ? `Storm about ${fmtNum(ex.km, 1)} kilometres away, thunder about ${Math.round(ex.sec)} seconds after each flash` : 'No thunder';
    state = ex ? 'idle' : 'none';
  }
  if (text !== lastF2B) {
    lastF2B = text;
    out.textContent = text;
    box.dataset.state = state;
    box.setAttribute('aria-label', `Flash-to-bang: ${label}`);
  }
}

function frame() {
  advance();
  updateClock();
  updateF2B();
  requestAnimationFrame(frame);
}

/* ------------------------------------------------------------------ volume & sleep */

function setVolume(pct, { from = null } = {}) {
  ui.volume = clamp(Number(pct) || 0, 0, 100);
  prefs.set('volume', ui.volume);
  applyGain();
  if (from !== 'listener') listener.setVolumeUI(ui.volume);
  const vs = $('#vol-studio');
  if (from !== 'studio' && vs) vs.value = String(Math.round(ui.volume));
  const db = volToDb(ui.volume);
  $('#vol-studio-out').textContent = ui.volume <= 0 ? 'muted' : `${Math.round(ui.volume)} % · ${fmtNum(db, 0)} dB`;
  vs?.setAttribute('aria-valuetext', ui.volume <= 0 ? 'muted' : `${Math.round(ui.volume)} percent, ${fmtNum(db, 0)} decibels`);
}

/* ------------------------------------------------------------------ visuals */

function setVisuals(mode) {
  ui.visuals = ['full', 'calm', 'off'].includes(mode) ? mode : 'full';
  if (ui.visuals !== 'off') ui.lastOnVisuals = ui.visuals;
  prefs.set('visuals', ui.visuals);
  const sb = $('#btn-scene');
  if (sb) {
    const off = ui.visuals === 'off';
    sb.setAttribute('aria-pressed', String(off));
    sb.title = off ? 'Scene is off: turn the 3-D scene back on' : 'Scene off: stop 3-D drawing, all CPU to the sound';
    sb.setAttribute('aria-label', sb.title);
  }
  viewport?.setVisualsMode(ui.visuals);
  listener?.setVisualsUI(ui.visuals);
  for (const i of $$('input[name="vp-visuals"]')) i.checked = i.value === ui.visuals;
  document.documentElement.dataset.visuals = ui.visuals;
}

/* ------------------------------------------------------------------ workspace */

function setWorkspace(ws, { quiet = false } = {}) {
  let target = ws === 'studio' ? 'studio' : 'listener';
  if (target === 'studio' && phoneMQ.matches) {
    toast('Studio needs a wider screen. On phones the Listener workspace has everything you need to listen.');
    target = 'listener';
  }
  ui.workspace = target;
  prefs.set('workspace', target);
  const app = $('#app');
  app.dataset.workspace = target;
  for (const b of $$('[data-ws]')) {
    const on = b.dataset.ws === target;
    b.setAttribute('aria-checked', on ? 'true' : 'false');
    b.tabIndex = on ? 0 : -1;
  }
  const studio = target === 'studio';
  document.documentElement.classList.toggle('ws-studio', studio);
  $('#library').hidden = !studio;
  $('#inspector').hidden = !studio;
  $('#timeline').hidden = !studio;
  listener.show(!studio);
  if (!studio) {
    store.select(null);
    if (viewport && viewport.cameraMode === 'walk') viewport.exitWalk();
    viewport?.frameListener();
  }
  requestAnimationFrame(() => viewport?.resize());
  if (!quiet) announce(studio ? 'Studio workspace' : 'Listener workspace');
}

/* ------------------------------------------------------------------ actions */

async function strike() {
  const s = store.scene;
  const sel = store.selection ? store.getObject(store.selection) : null;
  const args = {};
  const storms = s.place.objects.filter((o) => o.type === 'storm');
  let target = sel && sel.type === 'storm' ? sel : null;
  if (!target && storms.length) {
    let bestD = Infinity;
    for (const o of storms) {
      const [x, y] = stormPosition(o, transport.t);
      const d = rangeBearing(s, x, y).km;
      if (d < bestD) {
        bestD = d;
        target = o;
      }
    }
  }
  if (target) {
    const [x, y] = stormPosition(target, transport.t);
    const rb = rangeBearing(s, x, y);
    args.stormId = target.id;
    args.distanceKm = Math.max(0.3, rb.km);
    args.bearingDeg = rb.bearing;
  } else {
    args.distanceKm = s.weather.thunder.distance_km;
    args.bearingDeg = s.weather.thunder.bearing_deg;
  }
  if (!transport.playing) await play();
  if (audio.started) {
    const res = safe(() => audio.engine.strikeLightning(args));
    Promise.resolve(res).then((r) => {
      if (r && typeof r === 'object') onFlash({ brightness: 1, ...r, bearingDeg: Number.isFinite(r.bearingDeg) ? r.bearingDeg : args.bearingDeg });
    }).catch((err) => console.warn('[studio] strike failed', err));
  } else {
    const T = s.weather.temperature_c;
    const id = `ui-${Date.now()}`;
    const delay = flashBang(args.distanceKm, T);
    onFlash({ id, t: transport.t, channel: null, distanceKm: args.distanceKm, delaySec: delay, brightness: 1, bearingDeg: args.bearingDeg });
    setTimeout(() => onThunder({ id, t: transport.t, distanceKm: args.distanceKm, bearingDeg: args.bearingDeg }), delay * 1000);
  }
}

async function startDemo() {
  if (!viewport) {
    toast('The guided walk needs the 3-D view, and WebGL is not available here.');
    return;
  }
  if (store.scene.presetId !== DEMO_WALK.presetId) store.load(presetScene(DEMO_WALK.presetId), { label: 'Walk in from the storm' });
  seek(0);
  await play();
  viewport.startDemo();
  announce('Guided walk started: meadow, under the oak, the porch, indoors. Press Escape to stop.');
}

function addFromLibrary(key) {
  let id = null;
  if (viewport) id = viewport.placeInFront(key);
  else {
    const s = store.scene;
    const L = s.listener.position;
    let pos;
    if (key === 'storm') {
      const b = s.weather.wind.from_deg * DEG;
      pos = [Math.round(L[0] + Math.sin(b) * 6000), Math.round(L[1] + Math.cos(b) * 6000), 0];
    } else {
      const hd = (s.listener.heading_deg || 0) * DEG;
      pos = [Math.round((L[0] + Math.sin(hd) * 4) * 100) / 100, Math.round((L[1] + Math.cos(hd) * 4) * 100) / 100, 0];
    }
    const obj = instantiate(key, pos, s);
    id = obj ? store.addObject(obj) : null;
  }
  const o = id ? store.getObject(id) : null;
  if (o) announce(`Added ${objectLabel(o)}${o.type === 'storm' ? ' upwind' : ' in front of the listener'}`);
}

function exportJSON() {
  const file = `${slug(store.scene.name)}.rainscene.json`;
  download(new Blob([store.toJSON()], { type: 'application/json' }), file);
  toast(`Exported ${file}. The Python CLI renders it offline too: raingen render ${file}`);
}

const MAX_IMPORT_BYTES = 20 * 1024 * 1024;

async function importFile(file) {
  if (!file) return;
  if (file.size > MAX_IMPORT_BYTES) {
    toast(`Could not open ${file.name}: it is too large to be a scene file.`);
    return;
  }
  let text;
  try {
    text = await file.text();
  } catch (_) {
    toast(`Could not read ${file.name}. Try saving it again and re-importing.`);
    return;
  }
  try {
    store.importJSON(text);
  } catch (err) {
    toast(`Could not open ${file.name}: ${err && err.message ? err.message : 'it is not a rainscene file'}. Your current scene is unchanged.`);
    return;
  }
  seek(0);
  toast(`Opened “${store.scene.name}”.`, { action: { label: 'Undo', fn: () => store.undo() } });
}

/** Save the current scene to My presets and say whether it worked. */
function savePreset(name) {
  const nm = store.saveUserPreset(name);
  if (nm) toast(`Saved “${nm}” to My presets.`);
  else toast('Could not save: browser storage is full or blocked (private browsing?). Use Export to keep the scene as a file.');
  return nm;
}

async function share() {
  const url = store.shareURL();
  try {
    history.replaceState(null, '', url);
    ownHash = location.hash;
  } catch (_) { /* sandboxed */ }
  const input = $('#share-url');
  input.value = url;
  $('#share-note').textContent = `${url.length.toLocaleString()} characters. The whole scene travels inside the link; nothing is uploaded.`;
  $('#share-status').textContent = '';
  openDialog($('#dlg-share'), document.activeElement);
  input.select();
  try {
    await navigator.clipboard.writeText(url);
    $('#share-status').textContent = 'Link copied to the clipboard.';
  } catch (_) {
    $('#share-status').textContent = 'Select the link and copy it.';
  }
}

function openSaveDialog() {
  const dlg = $('#dlg-save');
  const name = $('#save-name');
  name.value = store.scene.name;
  openDialog(dlg, document.activeElement);
  name.focus();
  name.select();
}

function openRenderDialog() {
  const st = store.scene.studio;
  const secs = st.loopOn ? Math.round(st.loop[1] - st.loop[0]) : Math.min(Math.round(st.duration), 600);
  $('#render-seconds').value = String(Math.max(1, secs));
  $('#render-status').textContent = native.on ? 'Renders through the Rust engine (raingen studio).' : WORKLET_SUPPORTED ? '' : 'Rendering needs AudioWorklet (a modern browser on an https page).';
  $('#render-go').disabled = !audioCapable();
  $('#render-progress').value = 0;
  openDialog($('#dlg-render'), document.activeElement);
}

async function doRender() {
  const go = $('#render-go');
  const status = $('#render-status');
  const prog = $('#render-progress');
  const link = $('#render-link');
  const secs = clamp(parseFloat($('#render-seconds').value) || 60, 1, 3600);
  const sr = parseInt($('#render-rate').value, 10) || 48000;
  go.disabled = true;
  link.hidden = true;
  prog.value = 0;
  status.textContent = `Rendering ${fmtClock(secs)} at ${sr / 1000} kHz…`;
  const t0 = performance.now();
  try {
    const e = await ensureEngine();
    safe(() => e.setScene(clone(store.scene)));
    const blob = await e.renderOffline(secs, {
      sampleRate: sr,
      onProgress: (p) => {
        const v = typeof p === 'number' ? (p > 1.0001 ? p / 100 : p) : Number(p?.fraction ?? p?.progress ?? 0);
        prog.value = clamp(v || 0, 0, 1);
        status.textContent = `Rendering… ${Math.round(prog.value * 100)} %`;
      },
    });
    if (!(blob instanceof Blob)) throw new Error('the engine returned no audio');
    if (renderURL) URL.revokeObjectURL(renderURL);
    renderURL = URL.createObjectURL(blob);
    const file = `${slug(store.scene.name)}.wav`;
    link.href = renderURL;
    link.download = file;
    link.textContent = `Download ${file} (${fmtNum(blob.size / 1048576, 1)} MB)`;
    link.hidden = false;
    prog.value = 1;
    status.textContent = `Done in ${fmtNum((performance.now() - t0) / 1000, 1)} s.`;
    link.click();
    announce('Render finished; download started.');
  } catch (err) {
    console.error('[studio] render failed', err);
    status.textContent = `Render failed: ${err && err.message ? err.message : err}`;
  } finally {
    go.disabled = false;
  }
}

/* ------------------------------------------------------------------ gate */

function showGate() {
  const g = $('#gate');
  $('#gate-scene').textContent = store.scene.name;
  if (!audioCapable()) {
    $('#gate-note').textContent = AUDIO_SUPPORTED
      ? 'The audio engine cannot run here: it needs a browser with AudioWorklet, on an https page. You can still build the world, animate it and export it for the Python CLI.'
      : 'This browser has no Web Audio. You can still build the world and export it for the Python CLI.';
    $('#gate-play span').textContent = 'Enter the composer';
  }
  g.hidden = false;
  document.documentElement.classList.add('gate-open');
  requestAnimationFrame(() => $('#gate-play').focus());
}

function hideGate() {
  const g = $('#gate');
  if (g.hidden) return;
  g.hidden = true;
  document.documentElement.classList.remove('gate-open');
}

/* ------------------------------------------------------------------ media session */

function updateMediaSession() {
  try {
    if (!('mediaSession' in navigator) || typeof window.MediaMetadata !== 'function') return;
    navigator.mediaSession.metadata = new window.MediaMetadata({ title: store.scene.name, artist: 'raingen', album: '4D Environment Composer' });
  } catch (_) { /* unsupported */ }
}

/* ------------------------------------------------------------------ views */

const listener = new ListenerMode($('#listener-ui'), {
  store,
  play: () => play(),
  isPlaying: () => transport.playing,
  onVisuals: (m) => setVisuals(m),
  getVisuals: () => ui.visuals,
  getVolume: () => ui.volume,
  setVolume: (v) => setVolume(v, { from: 'listener' }),
  onSleepFade: (f) => {
    ui.sleepFade = clamp(f, 0, 1);
    applyGain();
  },
  onSleepEnd: async () => {
    await pause();
    ui.sleepFade = 1;
    applyGain();
    toast('Sleep timer: playback stopped.');
  },
  onDemo: () => startDemo(),
  toast,
  onSavePreset: () => openSaveDialog(),
  onShare: () => share(),
});

const library = new Library($('#library'), { store, onAdd: (key) => addFromLibrary(key) });

const inspector = new Inspector($('#inspector'), {
  store,
  getTime: () => transport.t,
  onAnimate: (path) => timeline.animate(path),
  getLiveRate: (id) => (audio.liveRates.has(id) ? audio.liveRates.get(id) : viewport?.liveRates.get(id)),
  onStrike: () => strike(),
  onFrame: () => viewport?.frameSelected(),
});

const timeline = new Timeline($('#timeline-main'), {
  store,
  getTime: () => transport.t,
  isPlaying: () => transport.playing,
  onSeek: (t) => seek(t),
  onTogglePlay: () => togglePlay(),
  onLayout: () => viewport?.resize(),
});

/* ------------------------------------------------------------------ viewport */

function showVpFallback(message) {
  const fb = $('#vp-fallback');
  fb.textContent = '';
  fb.append(
    el('p', { class: 'vp-fallback-title' }, 'Audio-only mode'),
    el('p', {}, message),
    el('p', { class: 'hint' }, 'Scenes, sliders, the inspector, the timeline and rendering all still work.'),
  );
  fb.hidden = false;
  document.documentElement.classList.add('no-webgl');
  $('#vp-toolbar').querySelectorAll('button, input, summary').forEach((b) => {
    if (b.tagName === 'SUMMARY') b.parentElement.hidden = true;
    else b.disabled = true;
  });
}

function onModeChange(mode) {
  for (const b of $$('[data-cam]')) b.setAttribute('aria-pressed', b.dataset.cam === mode ? 'true' : 'false');
  $('#btn-demo').setAttribute('aria-pressed', mode === 'demo' ? 'true' : 'false');
  $('#walk-hint').hidden = mode !== 'walk';
  $('#app').dataset.camera = mode;
  if (mode === 'walk') announce('Walk mode: W A S D to walk, click the view to look around, Shift to run, Escape to leave.');
}

async function initViewport() {
  if (!hasWebGL()) {
    showVpFallback('This browser or device cannot use WebGL (or has it turned off), so the 3-D view cannot be drawn.');
    return;
  }
  try {
    const mod = await import('./viewport.js');
    viewport = new mod.Viewport($('#viewport'), {
      store,
      onListenerLive: ({ position, headingDeg, pitchDeg }) => {
        audio.lastPitch = pitchDeg || 0;
        if (audio.started) safe(() => audio.engine.setListener({ position, headingDeg, pitchDeg: pitchDeg || 0 }));
      },
      onCaption: (text) => addCaption(text, 'note'),
      onDemoEnd: () => {
        toast('Guided walk finished: indoors, window shut. Press Walk to explore on your own.');
        announce('Guided walk finished');
      },
      onModeChange: (mode) => onModeChange(mode),
    });
    for (const [name, on] of Object.entries(ui.layers || {})) {
      viewport.setLayer(name, !!on);
      const cb = document.querySelector(`[data-layer="${name}"]`);
      if (cb) cb.checked = !!on;
    }
    viewport.setLite(native.on);
    viewport.setVisualsMode(ui.visuals);
    viewport.setReducedMotion(prefersReducedMotion());
    viewport.setTime(transport.t);
    viewport.autoShelter = $('#opt-autoshelter').checked;
    if (ui.workspace === 'listener') viewport.frameListener();
    document.documentElement.classList.add('has-webgl');
  } catch (err) {
    console.warn('[studio] 3-D view unavailable:', err);
    viewport = null;
    $('#viewport').textContent = '';
    const offline = /fetch|import|module|specifier/i.test(String((err && err.message) || ''));
    showVpFallback(offline
      ? 'The 3-D view could not be loaded (check your connection and reload), so the composer is running without it.'
      : 'This device could not start the 3-D view (WebGL failed to initialise), so the composer is running without it.');
  }
}

/* ------------------------------------------------------------------ bindings */

function bindTopbar() {
  $('#btn-play').addEventListener('click', () => togglePlay());
  $('#btn-stop').addEventListener('click', () => stop());
  $('#btn-undo').addEventListener('click', () => store.undo());
  $('#btn-redo').addEventListener('click', () => store.redo());
  $('#btn-save-preset').addEventListener('click', () => openSaveDialog());
  $('#btn-render').addEventListener('click', () => openRenderDialog());
  $('#btn-export').addEventListener('click', () => exportJSON());
  $('#btn-import').addEventListener('click', () => $('#file-import').click());
  $('#file-import').addEventListener('change', (e) => {
    importFile(e.target.files && e.target.files[0]);
    e.target.value = '';
  });
  $('#btn-share').addEventListener('click', () => share());
  $('#btn-help').addEventListener('click', () => openDialog($('#dlg-help'), document.activeElement));
  const vs = $('#vol-studio');
  vs.addEventListener('input', () => setVolume(parseFloat(vs.value), { from: 'studio' }));

  const wsBtns = $$('[data-ws]');
  for (const b of wsBtns) {
    b.addEventListener('click', () => setWorkspace(b.dataset.ws));
    b.addEventListener('keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
      e.preventDefault();
      const i = wsBtns.indexOf(b);
      const n = wsBtns[(i + (e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1) + wsBtns.length) % wsBtns.length];
      setWorkspace(n.dataset.ws);
      n.focus();
    });
  }

  $('#btn-captions-vtt').addEventListener('click', () => download(new Blob([captions.toVTT()], { type: 'text/vtt' }), `${slug(store.scene.name)}.vtt`));
  $('#btn-captions-clear').addEventListener('click', () => captions.clear());
}

function bindViewportToolbar() {
  for (const b of $$('[data-cam]')) b.addEventListener('click', () => viewport?.setCameraMode(b.dataset.cam));
  $('#btn-demo').addEventListener('click', () => startDemo());
  for (const b of $$('[data-gizmo]')) {
    b.addEventListener('click', () => {
      viewport?.setGizmoMode(b.dataset.gizmo);
      for (const x of $$('[data-gizmo]')) x.setAttribute('aria-pressed', x === b ? 'true' : 'false');
    });
  }
  $('#btn-snap').addEventListener('click', () => {
    const on = viewport ? viewport.toggleSnap() : false;
    $('#btn-snap').setAttribute('aria-pressed', on ? 'true' : 'false');
    announce(on ? 'Snapping on: 0.5 metres and 15 degrees' : 'Snapping off');
  });
  $('#btn-frame').addEventListener('click', () => viewport?.frameSelected());
  for (const cb of $$('[data-layer]')) {
    cb.addEventListener('change', () => {
      viewport?.setLayer(cb.dataset.layer, cb.checked);
      ui.layers = { ...(ui.layers || {}), [cb.dataset.layer]: cb.checked };
      prefs.set('layers', ui.layers);
    });
  }
  $('#opt-autoshelter').addEventListener('change', (e) => {
    if (viewport) viewport.autoShelter = e.target.checked;
  });
  for (const r of $$('input[name="vp-visuals"]')) r.addEventListener('change', () => r.checked && setVisuals(r.value));
  $('#btn-scene')?.addEventListener('click', () => setVisuals(ui.visuals === 'off' ? (ui.lastOnVisuals || 'full') : 'off'));
}

function bindDialogs() {
  $('#gate-play').addEventListener('click', () => {
    hideGate();
    play();
  });
  $('#gate-silent').addEventListener('click', () => {
    hideGate();
    $('#btn-play').focus();
  });
  $('#gate').addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      hideGate();
      $('#btn-play').focus();
    } else if (e.key === 'Tab') {
      const f = [$('#gate-play'), $('#gate-silent')];
      const i = f.indexOf(document.activeElement);
      e.preventDefault();
      f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length].focus();
    }
  });

  $('#render-go').addEventListener('click', () => doRender());
  $('#render-use-timeline').addEventListener('click', () => { $('#render-seconds').value = String(Math.round(store.scene.studio.duration)); });
  $('#render-use-loop').addEventListener('click', () => {
    const [a, b] = store.scene.studio.loop;
    $('#render-seconds').value = String(Math.max(1, Math.round(b - a)));
  });

  $('#share-copy').addEventListener('click', async () => {
    const input = $('#share-url');
    input.select();
    try {
      await navigator.clipboard.writeText(input.value);
      $('#share-status').textContent = 'Link copied to the clipboard.';
    } catch (_) {
      $('#share-status').textContent = 'Copy is blocked here: select the link and press Ctrl+C.';
    }
  });
  $('#share-json').addEventListener('click', () => exportJSON());

  $('#save-form').addEventListener('submit', (e) => {
    const sub = e.submitter;
    if (sub && sub.value !== 'save') return;
    e.preventDefault();
    closeDialog($('#dlg-save'));
    savePreset($('#save-name').value);
  });

  const single = $('#opt-single-keys');
  single.checked = ui.singleKeys;
  single.addEventListener('change', () => {
    ui.singleKeys = single.checked;
    prefs.set('singleKeys', ui.singleKeys);
  });
}

function bindDropImport() {
  window.addEventListener('dragover', (e) => {
    if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  });
  window.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    e.preventDefault();
    if (/\.json$/i.test(f.name) || f.type === 'application/json') importFile(f);
    else toast(`${f.name} is not a rainscene JSON file.`);
  });
}

function spaceTarget(t) {
  if (!t || t === document.body || t === document.documentElement) return true;
  if (t.id === 'stage' || t.classList?.contains('vp-canvas')) return true;
  if (t.matches?.('.kf, .tl-ruler')) return true;
  if (t.closest?.('#timeline') && !t.matches?.('button, input, select, textarea, a, [role="slider"], summary')) return true;
  if (t.closest?.('#listener-ui') && !t.matches?.('button, input, select, textarea, a, [role="slider"], [tabindex]')) return true;
  return false;
}

function inCanvasContext(t) {
  return !t || t === document.body || t === document.documentElement || t.id === 'stage' || t.classList?.contains('vp-canvas');
}

function bindKeys() {
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented) return;
    const typing = isTyping(e.target);
    const mod = e.ctrlKey || e.metaKey;
    const studio = ui.workspace === 'studio';
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (mod && !e.altKey) {
      if (key === '1' || key === '2') {
        e.preventDefault();
        setWorkspace(key === '1' ? 'listener' : 'studio');
        return;
      }
      if (key === 'z' && !typing) {
        e.preventDefault();
        if (e.shiftKey) store.redo();
        else store.undo();
        return;
      }
      if (key === 'y' && !typing) {
        e.preventDefault();
        store.redo();
        return;
      }
      if (key === 's') {
        e.preventDefault();
        openSaveDialog();
        return;
      }
      if (key === 'g' && studio) {
        e.preventDefault();
        timeline.focusGoto();
        return;
      }
      if (key === 'd' && studio && store.selection && !typing) {
        e.preventDefault();
        const id = store.duplicateObject(store.selection);
        if (id) announce(`Duplicated as ${objectLabel(store.getObject(id))}`);
        return;
      }
      return;
    }
    if (typing) return;
    if (e.key === 'Escape') {
      if (!$('#gate').hidden || document.querySelector('dialog[open]')) return;
      if (viewport?.cameraMode === 'demo') {
        viewport.stopDemo();
        announce('Guided walk stopped');
      } else if (viewport?.cameraMode === 'walk') viewport.exitWalk();
      else if (store.selection) store.select(null);
      return;
    }
    if (e.key === 'F1' || (e.key === '?' && !e.altKey)) {
      e.preventDefault();
      openDialog($('#dlg-help'), document.activeElement);
      return;
    }
    if (e.key === ' ' || e.code === 'Space') {
      if (spaceTarget(e.target)) {
        e.preventDefault();
        togglePlay();
      }
      return;
    }
    if (e.altKey || !ui.singleKeys) return;
    const canvas = inCanvasContext(e.target);
    if (key === 't' && (canvas || e.target.closest?.('#timeline, #inspector, #listener-ui'))) {
      e.preventDefault();
      strike();
      return;
    }
    if (key === 'Home' && (canvas || e.target.closest?.('#timeline'))) {
      e.preventDefault();
      seek(0);
      return;
    }
    if (key === 'l' && canvas && studio) {
      e.preventDefault();
      timeline.toggleLoop();
      return;
    }
    if (!studio || !canvas || !viewport) return;
    const step = e.shiftKey ? 5 : 0.5;
    switch (key) {
      case '1': viewport.setCameraMode('director'); break;
      case '2': viewport.setCameraMode('top'); break;
      case '3': viewport.setCameraMode('walk'); break;
      case 'w': case 'e': case 'r': {
        if (viewport.cameraMode === 'walk') return;
        const m = { w: 'translate', e: 'rotate', r: 'scale' }[key];
        viewport.setGizmoMode(m);
        for (const x of $$('[data-gizmo]')) x.setAttribute('aria-pressed', x.dataset.gizmo === m ? 'true' : 'false');
        announce(`${{ translate: 'Move', rotate: 'Rotate', scale: 'Scale' }[m]} gizmo`);
        break;
      }
      case 'f': viewport.frameSelected(); break;
      case 'n': $('#btn-snap').click(); break;
      case 'Delete':
      case 'Backspace':
        if (store.selection) {
          const o = store.getObject(store.selection);
          store.removeObject(store.selection);
          if (o) toast(`Removed ${objectLabel(o)}.`, { action: { label: 'Undo', fn: () => store.undo() } });
        }
        break;
      case 'ArrowUp': viewport.nudge(0, step); break;
      case 'ArrowDown': viewport.nudge(0, -step); break;
      case 'ArrowLeft': viewport.nudge(-step, 0); break;
      case 'ArrowRight': viewport.nudge(step, 0); break;
      case '[': viewport.nudge(0, 0, -15); break;
      case ']': viewport.nudge(0, 0, 15); break;
      default: return;
    }
    e.preventDefault();
  });
}

function bindEnvironment() {
  onReducedMotionChange((on) => viewport?.setReducedMotion(on));
  const onPhone = () => {
    if (phoneMQ.matches && ui.workspace === 'studio') setWorkspace('listener', { quiet: true });
    $('#ws-studio').setAttribute('aria-disabled', phoneMQ.matches ? 'true' : 'false');
  };
  if (phoneMQ.addEventListener) phoneMQ.addEventListener('change', onPhone);
  onPhone();
  window.addEventListener('hashchange', () => {
    if (location.hash === ownHash) return;
    const sc = SceneStore.decodeHash(location.hash);
    if (!sc) {
      if (SceneStore.hasSceneHash(location.hash)) {
        clearHash();
        toast(BROKEN_LINK_MESSAGE);
      }
      return;
    }
    ownHash = location.hash;
    store.load(sc, { label: sc.name });
    seek(0);
    toast(`Opened the shared scene “${sc.name}”.`, { action: { label: 'Save to My presets', fn: () => savePreset(sc.name) } });
  });
  window.addEventListener('beforeunload', () => store.saveAutosave());
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) store.saveAutosave();
  });
  // keep the timeline moving while the tab is hidden (rAF pauses, audio does not)
  setInterval(() => {
    if (document.hidden) advance();
  }, 250);
  // live stats for the selected object (storms move with time)
  setInterval(() => {
    if (store.selection && ui.workspace === 'studio') inspector.updateLive();
  }, 500);
  try {
    if ('mediaSession' in navigator) {
      navigator.mediaSession.setActionHandler('play', () => play());
      navigator.mediaSession.setActionHandler('pause', () => pause());
      navigator.mediaSession.setActionHandler('stop', () => stop());
    }
  } catch (_) { /* unsupported */ }
}

/* ------------------------------------------------------------------ boot */

function boot() {
  bindTopbar();
  bindViewportToolbar();
  bindDialogs();
  bindDropImport();
  bindKeys();
  bindEnvironment();
  setVolume(ui.volume);
  setVisuals(ui.visuals);
  setWorkspace(ui.workspace, { quiet: true });
  document.title = `${store.scene.name} · raingen 4D Composer`;
  updateMediaSession();
  setAudioState(audioCapable() ? 'idle' : 'unsupported');
  updateTransportUI();
  timeline.setTime(0, { apply: true, force: true });
  if (WORKLET_SUPPORTED && !native.on) loadEngineModule().catch((err) => console.warn('[studio] engine module not loaded yet:', err && err.message ? err.message : err));
  initViewport();
  showGate();
  if (sharedScene) toast(`Opened the shared scene “${sharedScene.name}”.`, { action: { label: 'Save to My presets', fn: () => savePreset(sharedScene.name) } });
  if (brokenShareLink) toast(BROKEN_LINK_MESSAGE, { timeout: 12000 });
  requestAnimationFrame(frame);
  void library;
  document.documentElement.dataset.ready = 'true';
}

boot();
