/**
 * NativeBridge: the browser WeatherEngine interface, played by the native (Rust) core.
 *
 * Used by the 4D Composer when the page is served by `raingen studio` (a local Python server):
 * every engine call becomes a small JSON POST to /native/*, and the server's Server-Sent Events
 * (/native/events) come back as the same events WeatherEngine emits ('flash', 'thunder', 'gust',
 * 'caption', 'meter', 'state', ...). The sound comes from the Python process, so no AudioContext,
 * no worklets and no autoplay unlock are involved.
 *
 * Live calls (params, listener, objects, volume) are coalesced per path and sent at most every
 * 50 ms (20 Hz) in one ordered batch; requests are serialized so a scene never overtakes a later
 * parameter.
 */

const FLUSH_MS = 50;

function clone(v) {
  return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
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
    this._pending = { scene: null, params: new Map(), objects: new Map(), listener: null, volume: null, renderMode: null };
    this._flushTimer = 0;
    this._lastFlush = 0;
    this._chain = Promise.resolve();
    this._stats = null;
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
    es.addEventListener('flash', json((d) => {
      // photosensitive safety (WCAG 2.3.1): at most maxFlashesPerSec full flashes per second
      const now = performance.now();
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
    }));
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
    return !!(q.scene || q.params.size || q.objects.size || q.listener || q.volume || q.renderMode);
  }

  _scheduleFlush() {
    if (!this._started || this._flushTimer) return;
    const wait = Math.max(0, FLUSH_MS - (performance.now() - this._lastFlush));
    this._flushTimer = setTimeout(() => {
      this._flushTimer = 0;
      this._flush();
    }, wait);
  }

  _flush() {
    if (!this._started || !this._hasPending()) return this._chain;
    const q = this._pending;
    const body = {};
    if (q.scene) body.scene = q.scene;
    if (q.renderMode) body.renderMode = q.renderMode;
    if (q.objects.size) body.objects = [...q.objects.values()];
    if (q.params.size) body.params = [...q.params.entries()].map(([path, v]) => ({ path, value: v.value, ramp: v.ramp }));
    if (q.listener) body.listener = q.listener;
    if (q.volume) body.volume = q.volume;
    body.t = this._t;
    this._pending = { scene: null, params: new Map(), objects: new Map(), listener: null, volume: null, renderMode: null };
    this._lastFlush = performance.now();
    return this._enqueue('/native/batch', body);
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
    this._pending = { scene: null, params: new Map(), objects: new Map(), listener: null, volume: null, renderMode: null };
    this._emit('state', { state: 'running' });
  }

  stop() {
    clearTimeout(this._flushTimer);
    this._flushTimer = 0;
    this._started = false;
    this._suspended = false;
    this._enqueue('/native/stop', {});
    this._closeEvents();
  }

  async suspend() {
    if (!this._started || this._suspended) return;
    this._suspended = true;
    await this._flush();
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
    q.scene = clone(scene);
    q.params.clear();
    q.objects.clear();
    this._scheduleFlush();
    return true;
  }

  setParam(path, value, rampSec = 0.5) {
    const p = String(path || '');
    if (!p) return false;
    if (this.scene && !p.startsWith('mix.')) setPath(this.scene, p, value);
    if (!this._started) return true;
    this._pending.params.set(p, { value: clone(value), ramp: Number.isFinite(Number(rampSec)) ? Number(rampSec) : 0.5 });
    this._scheduleFlush();
    return true;
  }

  setListener(l = {}) {
    const o = l && typeof l === 'object' ? l : {};
    const L = this._pending.listener || {};
    if (Array.isArray(o.position)) L.position = o.position.slice(0, 3).map(Number);
    if (o.headingDeg !== undefined && Number.isFinite(Number(o.headingDeg))) L.headingDeg = Number(o.headingDeg);
    if (o.pitchDeg !== undefined && Number.isFinite(Number(o.pitchDeg))) L.pitchDeg = Number(o.pitchDeg);
    if (o.heightM !== undefined && Number.isFinite(Number(o.heightM))) L.heightM = Number(o.heightM);
    if (this.scene && this.scene.listener) {
      if (L.position) this.scene.listener.position = L.position.slice();
      if (L.headingDeg !== undefined) this.scene.listener.heading_deg = L.headingDeg;
    }
    if (!this._started) return;
    this._pending.listener = L;
    this._scheduleFlush();
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

  /** Manual strikes are synthesized by the browser engine only; the native core plays the scene's own storm. */
  strikeLightning() {
    return null;
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
