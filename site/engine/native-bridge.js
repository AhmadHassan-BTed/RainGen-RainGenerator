/**
 * NativeBridge: the browser WeatherEngine interface, played by the native (Rust) core.
 *
 * Used by the 4D Composer when the page is served by `raingen studio` (a local Python server):
 * every engine call becomes a small JSON POST to /native/*, and the server's Server-Sent Events
 * (/native/events) come back as the same events WeatherEngine emits ('flash', 'thunder', 'gust',
 * 'caption', 'meter', 'state', ...). The sound comes from the Python process, so no AudioContext,
 * no worklets and no autoplay unlock are involved.
 *
 * Traffic is shaped so live use never stalls the engine:
 * - parameters, volume and render mode are coalesced per path and sent at most every 50 ms
 *   (20 Hz) in one ordered batch; batches are serialized so a scene never overtakes a later
 *   parameter;
 * - object edits (each one is a structural change on the server) go at most 5 times a second,
 *   latest state per object wins; `commit()` sends them at once (end of a drag);
 * - the listener pose has its own lane, /native/listener at most 30 times a second, latest pose
 *   wins, one request in flight; it never rides with a scene or an object;
 * - seeks have their own lane too (/native/seek, latest time wins), so scrubbing stays smooth.
 */

const FLUSH_MS = 50;
const OBJECT_MS = 200;
const LISTENER_MS = 34;                    // <= 30 Hz even with timer jitter
const SEEK_MS = 80;
const STRIKE_FLASH_WAIT_MS = 600;

function clone(v) {
  return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
}

function finite(...vals) {
  for (const v of vals) {
    if (v === null || v === undefined || v === '') continue;
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function setPath(obj, path, value) {
  const parts = String(path).split('.').filter(Boolean);
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    let next;
    if (Array.isArray(cur)) next = cur.find((o) => o && String(o.id) === p) ?? cur[Number(p)];
    else next = cur[p];
    if (next === null || typeof next !== 'object') {
      if (Array.isArray(cur)) return false;
      next = {};
      cur[p] = next;
    }
    cur = next;
  }
  if (Array.isArray(cur)) return false;
  cur[parts[parts.length - 1]] = clone(value);
  return true;
}

function emptyQueue(objects = new Map()) {
  return { scene: null, params: new Map(), objects, volume: null, renderMode: null };
}

/**
 * One latest-wins request lane: at most one request in flight, at most one every `minMs`; the
 * body is built when it is sent, so a burst of pushes costs one request with the newest state.
 */
class Lane {
  constructor(bridge, path, minMs) {
    this.bridge = bridge;
    this.path = path;
    this.minMs = minMs;
    this.build = null;
    this.timer = 0;
    this.busy = false;
    this.last = -Infinity;
    this.warned = false;
  }

  push(build) {
    this.build = build;
    this._kick();
  }

  _kick() {
    if (this.timer || this.busy || !this.build || !this.bridge._started) return;
    const wait = Math.max(0, this.minMs - (performance.now() - this.last));
    this.timer = setTimeout(() => {
      this.timer = 0;
      this._send();
    }, wait);
  }

  _send() {
    const build = this.build;
    if (!build || !this.bridge._started) return;
    this.build = null;
    this.busy = true;
    this.last = performance.now();
    let body;
    try {
      body = build();
    } catch (_) {
      body = null;
    }
    if (!body) {
      this.busy = false;
      return;
    }
    this.bridge._post(this.path, body)
      .then(() => { this.warned = false; })
      .catch((err) => {
        if (!this.warned) console.warn(`[NativeBridge] ${this.path} failed:`, err && err.message ? err.message : err);
        this.warned = true;
      })
      .finally(() => {
        this.busy = false;
        this._kick();
      });
  }

  cancel() {
    clearTimeout(this.timer);
    this.timer = 0;
    this.build = null;
  }
}

/*
 * Parameters that rebuild the engine's voices: held until the control settles (pointer released,
 * or 250 ms without a change from keys / menus) so a drag across several shelters costs one
 * rebuild. A precipitation type change is structural only across families: none / drizzle /
 * rain / showers share their voices and glide.
 */
const STRUCTURAL_PARAM = /^place\.(shelter|preset)$/;
const PRECIP_TYPE = 'weather.precipitation.type';
const LIQUID = new Set(['none', 'drizzle', 'rain', 'showers']);
const precipFamily = (t) => (LIQUID.has(String(t)) ? 'liquid' : String(t));
const SETTLE_MS = 250;
const RELEASE_MS = 120;
const MAX_HOLD_MS = 5000;

export class NativeBridge {
  /** opts: { base: '' (same origin), hello: the /native/hello reply if already fetched } */
  constructor(opts = {}) {
    this.base = String(opts.base || '');
    this.scene = opts.scene ? clone(opts.scene) : null;
    this.info = opts.hello || null;
    this._events = new Map();
    this._started = false;
    this._suspended = false;
    this._dead = false;
    this._es = null;
    this._renderMode = null;
    this._masterDb = -48;
    this._t = 0;
    this._safety = { maxFlashesPerSec: 3 };
    this._flashTimes = [];
    this._lastFlashAt = -Infinity;
    this._pending = emptyQueue();
    this._flushTimer = 0;
    this._flushAt = 0;
    this._lastFlush = 0;
    this._lastObjects = -Infinity;
    this._objectsNow = false;
    this._chain = Promise.resolve();
    this._stats = null;
    // the live listener pose (its own 30 Hz lane) and a counter that says it moved
    this._pose = null;
    this._poseSeq = 0;
    this._listenerLane = new Lane(this, '/native/listener', LISTENER_MS);
    this._seekLane = new Lane(this, '/native/seek', SEEK_MS);
    // structural changes held until the control settles (see STRUCTURAL_PARAM)
    this._held = new Map();
    this._heldTimer = 0;
    this._heldSince = 0;
    this._pointerDown = false;
    if (typeof window !== 'undefined' && window.addEventListener) {
      const down = () => { this._pointerDown = true; };
      const up = () => {
        this._pointerDown = false;
        if (this._held.size) this._armHeld(RELEASE_MS);
      };
      window.addEventListener('pointerdown', down, true);
      for (const ev of ['pointerup', 'pointercancel', 'blur']) window.addEventListener(ev, up, true);
    }
  }

  _isStructural(p, oldValue, value) {
    if (STRUCTURAL_PARAM.test(p)) return true;
    if (p !== PRECIP_TYPE) return false;
    return this._held.has(p) || precipFamily(oldValue) !== precipFamily(value);
  }

  _armHeld(ms) {
    clearTimeout(this._heldTimer);
    this._heldTimer = setTimeout(() => this._releaseHeld(), ms);
  }

  _releaseHeld() {
    this._heldTimer = 0;
    if (!this._held.size) return;
    if (this._pointerDown && performance.now() - this._heldSince < MAX_HOLD_MS) {
      this._armHeld(SETTLE_MS);                 // still dragging: the pointer-up releases it
      return;
    }
    for (const [p, e] of this._held) this._pending.params.set(p, e);
    this._held.clear();
    this._scheduleFlush();
  }

  _dropHeld() {
    clearTimeout(this._heldTimer);
    this._heldTimer = 0;
    this._held.clear();
  }

  /** GET /native/hello: the server's reply, or null when the page is not served by raingen studio. */
  static async detect(base = '', timeoutMs = 1500) {
    const ctl = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = setTimeout(() => ctl && ctl.abort(), timeoutMs);
    try {
      const r = await fetch(`${base}/native/hello`, { cache: 'no-store', signal: ctl ? ctl.signal : undefined });
      if (!r.ok) return null;
      const j = await r.json();
      return j && j.app === 'raingen studio' ? j : null;
    } catch (_) {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  /* ---------------- WeatherEngine getters */

  get running() {
    return this._started && !this._suspended && !this._dead;
  }

  /** No AudioContext: the sound is played by the raingen process. */
  get context() {
    return null;
  }

  get analyser() {
    return null;
  }

  get native() {
    return true;
  }

  get voices() {
    const s = this._stats || this.info || {};
    return Number(s.voices) || 0;
  }

  get engineKind() {
    const s = this._stats || this.info || {};
    return s.engine || 'native';
  }

  /* ---------------- events */

  on(event, cb) {
    if (typeof cb !== 'function') return () => {};
    if (!this._events.has(event)) this._events.set(event, new Set());
    this._events.get(event).add(cb);
    return () => this.off(event, cb);
  }

  off(event, cb) {
    const s = this._events.get(event);
    if (!s) return;
    if (cb) s.delete(cb);
    else s.clear();
  }

  _emit(event, payload) {
    const s = this._events.get(event);
    if (!s || !s.size) return;
    for (const cb of [...s]) {
      try {
        cb(payload);
      } catch (e) {
        console.error(`[NativeBridge] '${event}' listener failed`, e);
      }
    }
  }

  /** Photosensitive safety (WCAG 2.3.1): at most maxFlashesPerSec full flashes per second. */
  _onFlash(d) {
    const now = performance.now();
    this._lastFlashAt = now;
    this._flashTimes = this._flashTimes.filter((t) => now - t < 1000);
    const cap = this._safety.maxFlashesPerSec;
    const ev = { ...d, limited: false };
    if (this._flashTimes.length >= cap) {
      ev.limited = true;
      ev.brightness = Math.min(Number(ev.brightness) || 1, 0.35);
    } else {
      this._flashTimes.push(now);
    }
    this._emit('flash', ev);
  }

  _openEvents() {
    if (this._es || typeof EventSource !== 'function') return;
    const es = new EventSource(`${this.base}/native/events`);
    this._es = es;
    const json = (fn) => (ev) => {
      let d = {};
      try {
        d = JSON.parse(ev.data || '{}');
      } catch (_) {
        return;
      }
      fn(d || {});
    };
    es.addEventListener('hello', json((d) => {
      this._dead = false;
      this.info = { ...(this.info || {}), ...d };
      this._emit('stats', this.info);
    }));
    es.addEventListener('stats', json((d) => {
      this._stats = d;
      this._emit('stats', d);
    }));
    es.addEventListener('meter', json((d) => this._emit('meter', d)));
    es.addEventListener('caption', json((d) => this._emit('caption', d)));
    es.addEventListener('gust', json((d) => this._emit('gust', d)));
    es.addEventListener('thunder', json((d) => this._emit('thunder', d)));
    es.addEventListener('marker', json((d) => this._emit('marker', d)));
    es.addEventListener('render', json((d) => this._emit('render', d)));
    es.addEventListener('engine-error', json((d) => this._emit('error', d)));
    es.addEventListener('flash', json((d) => this._onFlash(d)));
    es.addEventListener('state', json((d) => {
      if (d.state === 'stopped' && this._started) {
        this._started = false;
        this._emit('state', { state: 'closed' });
      }
    }));
    es.onerror = () => {
      // the server went away (EventSource keeps retrying; 'hello' arrives again when it is back)
      if (!this._dead) {
        this._dead = true;
        this._emit('server', { up: false });
      }
    };
    es.onopen = () => {
      if (this._dead) this._emit('server', { up: true });
      this._dead = false;
    };
  }

  _closeEvents() {
    if (this._es) {
      try {
        this._es.close();
      } catch (_) { /* closed */ }
      this._es = null;
    }
  }

  /* ---------------- transport */

  async _post(path, body, { raw = false } = {}) {
    const r = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
      cache: 'no-store',
    });
    if (raw) return r;
    let j = null;
    try {
      j = await r.json();
    } catch (_) {
      j = null;
    }
    if (!r.ok || !j || j.ok === false) throw new Error((j && j.error) || `${path}: HTTP ${r.status}`);
    return j;
  }

  /** Serialize requests: each POST starts after the previous one finished. */
  _enqueue(path, body) {
    const p = this._chain.then(() => this._post(path, body));
    this._chain = p.catch((err) => {
      console.warn(`[NativeBridge] ${path} failed:`, err && err.message ? err.message : err);
    });
    return p;
  }

  _hasPending() {
    const q = this._pending;
    return !!(q.scene || q.params.size || q.objects.size || q.volume || q.renderMode);
  }

  /** Flush after `wait` ms (default: the 20 Hz batch rate); an earlier request moves the timer up. */
  _scheduleFlush(wait = Math.max(0, FLUSH_MS - (performance.now() - this._lastFlush))) {
    if (!this._started) return;
    const at = performance.now() + Math.max(0, wait);
    if (this._flushTimer) {
      if (at >= this._flushAt - 1) return;
      clearTimeout(this._flushTimer);
    }
    this._flushAt = at;
    this._flushTimer = setTimeout(() => {
      this._flushTimer = 0;
      this._flush();
    }, Math.max(0, wait));
  }

  _flush() {
    if (!this._started || !this._hasPending()) return this._chain;
    const q = this._pending;
    const now = performance.now();
    // object edits rebuild voices on the server: at most every OBJECT_MS, unless a scene goes too
    const objectsDue = q.objects.size > 0 && (!!q.scene || this._objectsNow || now - this._lastObjects >= OBJECT_MS);
    const body = {};
    if (q.scene) {
      body.scene = q.scene;
      this._poseIntoScene(body.scene);
    }
    if (q.renderMode) body.renderMode = q.renderMode;
    if (objectsDue) {
      body.objects = [...q.objects.values()];
      this._lastObjects = now;
    }
    if (q.params.size) body.params = [...q.params.entries()].map(([path, v]) => ({ path, value: v.value, ramp: v.ramp }));
    if (q.volume) body.volume = q.volume;
    const held = objectsDue ? new Map() : q.objects;
    this._pending = emptyQueue(held);
    this._objectsNow = false;
    if (held.size) this._scheduleFlush(OBJECT_MS - (now - this._lastObjects));
    if (!Object.keys(body).length) return this._chain;
    body.t = this._t;
    this._lastFlush = now;
    const p = this._enqueue('/native/batch', body);
    if (body.scene) {
      // the scene carried the pose of this moment; a pose that moved on since goes again after it
      const seq = this._poseSeq;
      p.then(() => {
        if (this._poseSeq !== seq) this._pushPose();
      }, () => {});
    }
    return p;
  }

  /** A scene sent to the server carries the live listener pose, not the store's last copy. */
  _poseIntoScene(scene) {
    const P = this._pose;
    if (!P || !scene || typeof scene !== 'object') return;
    const L = scene.listener && typeof scene.listener === 'object' ? scene.listener : (scene.listener = {});
    if (P.position) L.position = P.position.slice();
    if (P.headingDeg !== undefined) L.heading_deg = P.headingDeg;
  }

  /** Send the queued edits now (objects included), e.g. at the end of a drag. */
  commit() {
    if (!this._started) return this._chain;
    clearTimeout(this._flushTimer);
    this._flushTimer = 0;
    this._objectsNow = true;
    return this._flush();
  }

  /* ---------------- lifecycle */

  async start() {
    if (this._started) {
      if (this._suspended) await this.resume();
      return;
    }
    this._openEvents();
    const r = await this._post('/native/start', {
      scene: this.scene || undefined,
      renderMode: this._renderMode || undefined,
      volumeDb: this._masterDb,
    });
    this.info = { ...(this.info || {}), ...r };
    this._stats = null;
    this._started = true;
    this._suspended = false;
    this._dead = false;
    // anything queued before start is already in the scene the server started with
    this._pending = emptyQueue();
    this._emit('state', { state: 'running' });
  }

  stop() {
    clearTimeout(this._flushTimer);
    this._flushTimer = 0;
    this._dropHeld();
    this._listenerLane.cancel();
    this._seekLane.cancel();
    this._started = false;
    this._suspended = false;
    this._enqueue('/native/stop', {});
    this._closeEvents();
  }

  async suspend() {
    if (!this._started || this._suspended) return;
    this._suspended = true;
    await this.commit();
    await this._enqueue('/native/suspend', {}).catch(() => {});
  }

  async resume() {
    if (!this._started) return this.start();
    if (!this._suspended) return;
    await this._enqueue('/native/resume', {});
    this._suspended = false;
    this._emit('state', { state: 'running' });
  }

  /* ---------------- scene */

  setScene(scene) {
    if (!scene || typeof scene !== 'object') return false;
    this.scene = clone(scene);
    if (!this._started) return true;
    const q = this._pending;
    this._dropHeld();                           // the whole scene supersedes held changes
    q.scene = clone(scene);
    q.params.clear();
    q.objects.clear();
    this._scheduleFlush();
    return true;
  }

  setParam(path, value, rampSec = 0.5) {
    const p = String(path || '');
    if (!p) return false;
    const old = p === PRECIP_TYPE && this.scene ? this.scene.weather?.precipitation?.type : undefined;
    if (this.scene && !p.startsWith('mix.')) setPath(this.scene, p, value);
    if (!this._started) return true;
    const entry = { value: clone(value), ramp: Number.isFinite(Number(rampSec)) ? Number(rampSec) : 0.5 };
    if (this._isStructural(p, old, value)) {
      if (!this._held.size) this._heldSince = performance.now();
      this._held.set(p, entry);
      this._armHeld(SETTLE_MS);
      return true;
    }
    this._pending.params.set(p, entry);
    this._scheduleFlush();
    return true;
  }

  /**
   * The live listener pose: its own lane to /native/listener (at most 30 Hz, latest pose wins).
   * It never goes through a scene or a batch, so moving and turning never rebuild voices.
   */
  setListener(l = {}) {
    const o = l && typeof l === 'object' ? l : {};
    const P = this._pose ? { ...this._pose } : {};
    if (Array.isArray(o.position)) {
      const pos = o.position.slice(0, 3).map(Number);
      if (pos.length >= 2 && pos.every(Number.isFinite)) {
        if (pos.length < 3) pos.push(P.position ? P.position[2] : 0);
        P.position = pos;
      }
    }
    const h = finite(o.headingDeg, o.heading_deg);
    if (h !== undefined) P.headingDeg = ((h % 360) + 360) % 360;
    const p = finite(o.pitchDeg, o.pitch_deg);
    if (p !== undefined) P.pitchDeg = Math.max(-90, Math.min(90, p));
    const hm = finite(o.heightM, o.height_m);
    if (hm !== undefined) P.heightM = hm;
    this._pose = P;
    this._poseSeq++;
    if (this.scene && this.scene.listener) {
      if (P.position) this.scene.listener.position = P.position.slice();
      if (P.headingDeg !== undefined) this.scene.listener.heading_deg = P.headingDeg;
      if (P.heightM !== undefined) this.scene.listener.height_m = P.heightM;
    }
    if (!this._started) return;
    this._pushPose();
  }

  _pushPose() {
    if (!this._pose) return;
    this._listenerLane.push(() => {
      const P = this._pose;
      if (!P) return null;
      const body = {};
      if (P.position) body.position = P.position.slice();
      if (P.headingDeg !== undefined) {
        body.heading_deg = P.headingDeg;
        body.headingDeg = P.headingDeg;
      }
      if (P.pitchDeg !== undefined) {
        body.pitch_deg = P.pitchDeg;
        body.pitchDeg = P.pitchDeg;
      }
      if (P.heightM !== undefined) {
        body.height_m = P.heightM;
        body.heightM = P.heightM;
      }
      return body;
    });
  }

  upsertObject(object) {
    if (!object || typeof object !== 'object') return false;
    const id = object.id !== undefined && object.id !== null ? String(object.id) : `obj${Date.now()}`;
    const o = { ...clone(object), id };
    if (this.scene && this.scene.place) {
      const list = Array.isArray(this.scene.place.objects) ? this.scene.place.objects : (this.scene.place.objects = []);
      const i = list.findIndex((x) => x && String(x.id) === id);
      if (i < 0) list.push(clone(o));
      else list[i] = clone(o);
    }
    if (!this._started) return true;
    this._pending.objects.set(id, { op: 'upsert', object: o });
    this._scheduleFlush();
    return true;
  }

  removeObject(id) {
    const k = String(id);
    if (this.scene && this.scene.place && Array.isArray(this.scene.place.objects)) {
      this.scene.place.objects = this.scene.place.objects.filter((o) => !o || String(o.id) !== k);
    }
    if (!this._started) return true;
    this._pending.objects.set(k, { op: 'remove', id: k });
    this._scheduleFlush();
    return true;
  }

  /** The Composer plays its own automation (as parameter changes); the time rides along with the next batch. */
  setTime(tSec) {
    const t = Number(tSec);
    this._t = Number.isFinite(t) && t > 0 ? t : 0;
  }

  /** Jump the scene clock to `tSec` without restarting the sound (/native/seek, latest time wins). */
  seek(tSec) {
    this.setTime(tSec);
    if (!this._started) return;
    this._seekLane.push(() => ({ t: this._t }));
  }

  setMasterGainDb(db) {
    const v = Number(db);
    this._masterDb = Number.isFinite(v) ? Math.max(-120, Math.min(12, v)) : 0;
    if (!this._started) return;
    this._pending.volume = { db: this._masterDb, ramp: 0.06 };
    this._scheduleFlush();
  }

  setRenderMode(mode) {
    this._renderMode = mode === 'stereo' ? 'stereo' : 'binaural';
    if (!this._started) return;
    this._pending.renderMode = this._renderMode;
    this._scheduleFlush();
  }

  setVisualSafety(o = {}) {
    const n = Number(o && o.maxFlashesPerSec);
    this._safety.maxFlashesPerSec = Number.isFinite(n) ? Math.max(0, Math.min(3, n)) : 3;
  }

  /**
   * A manual strike now (/native/strike): the server plays the crack and the thunder at the right
   * delay and sends 'flash' and 'thunder' events like any other strike. Resolves to null (the flash
   * arrives as an event); a server that sends no flash event gets one drawn from its reply.
   * args: { distanceKm?, bearingDeg?, intensity? }
   */
  strikeLightning(args = {}) {
    if (!this._started) return null;
    const a = args && typeof args === 'object' ? args : {};
    const body = {};
    const km = finite(a.distanceKm, a.distance_km);
    if (km !== undefined && km > 0) body.distance_km = km;
    const b = finite(a.bearingDeg, a.bearing_deg);
    if (b !== undefined) body.bearing_deg = ((b % 360) + 360) % 360;
    const k = finite(a.intensity);
    if (k !== undefined) body.intensity = Math.max(0, Math.min(1, k));
    const sent = performance.now();
    return this._post('/native/strike', body).then((r) => {
      setTimeout(() => {
        if (this._lastFlashAt >= sent || !this._started) return;
        const f = (r && typeof r.flash === 'object' && r.flash) || r || {};
        this._onFlash({
          id: f.id !== undefined && f.id !== null ? String(f.id) : `strike-${Math.round(sent)}`,
          t: this._t,
          distanceKm: finite(f.distanceKm, f.distance_km, body.distance_km),
          bearingDeg: finite(f.bearingDeg, f.bearing_deg, body.bearing_deg),
          delaySec: finite(f.delaySec, f.flash_to_bang_s, f.delay_s),
          brightness: 1,
          channel: null,
          source: 'native',
        });
      }, Math.max(0, STRIKE_FLASH_WAIT_MS - (performance.now() - sent)));
      return null;
    }, (err) => {
      console.warn('[NativeBridge] /native/strike failed:', err && err.message ? err.message : err);
      this._emit('error', { where: 'strike', message: err && err.message ? err.message : String(err) });
      return null;
    });
  }

  /* ---------------- offline */

  /** Render `seconds` through the native core (Python Renderer, mastered); resolves to a WAV Blob. */
  async renderOffline(seconds, opts = {}) {
    const o = opts && typeof opts === 'object' ? opts : {};
    const dur = Number(seconds);
    if (!(dur > 0)) throw new RangeError('renderOffline: seconds must be a positive number.');
    this._openEvents();
    const off = typeof o.onProgress === 'function'
      ? this.on('render', (d) => {
        if (Number.isFinite(Number(d.fraction))) o.onProgress(Number(d.fraction));
      })
      : () => {};
    try {
      const r = await this._post('/native/render', {
        scene: this.scene || undefined,
        seconds: dur,
        sampleRate: Number(o.sampleRate) || 48000,
        renderMode: o.renderMode === 'stereo' || o.renderMode === 'binaural' ? o.renderMode : this._renderMode || undefined,
      }, { raw: true });
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try {
          const j = await r.json();
          if (j && j.error) msg = j.error;
        } catch (_) { /* not JSON */ }
        throw new Error(msg);
      }
      const buf = await r.arrayBuffer();
      return new Blob([buf], { type: 'audio/wav' });
    } finally {
      off();
    }
  }
}

export default NativeBridge;
