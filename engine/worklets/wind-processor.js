/**
 * rg-wind: wind heard through objects (MASTER_PLAN §3.6, §5.2).
 *
 * Modes (processorOptions.mode):
 * - air:        broadband wind roar, band centres rising with U, intensity ~ U^(2 chi), chi = 2,
 *               with 1/f turbulence modulation; 2 decorrelated channels.
 * - vegetation: species spectra after Fegeant (1999) and Bolin (2006/2009): broadleaf rustle as
 *               granular leaf-collision ticks (onset near Beaufort 2) plus swish; conifer
 *               log-Gaussian whoosh centred on the needle Strouhal frequency f = 0.2 U / d;
 *               bare winter branches whistle at 0.2 U / d (d = 3-6 mm twigs) and creak from
 *               Beaufort 8. Level ~ U^(2 chi), chi = 1.5-2.7.
 * - grass:      blade hiss and fine ticks.
 * - wire:       aeolian tone f = 0.2 U / d (Strouhal) with lock-in to the span's string
 *               harmonics; dipole level ~ U^6 (intensity); Reynolds-number gated.
 * - flag:       flapping at f ~ 0.3 U / L with cloth snaps, flutter noise and halyard pings.
 * - chime:      free-free tube partials 1 : 2.756 : 5.404 struck at a rate rising with U.
 * - wiper:      car wipers (20-60 cycles/min): rubber swish on wet glass, dry squeak,
 *               reversal thunk and motor hum.
 *
 * The 'wind' AudioParam carries the 10 m reference wind (mean + advected gust); the
 * processor applies its own height factor (config.windScale) from the log law.
 */

const TWO_PI = Math.PI * 2;
const RING = 8192;
const RMASK = RING - 1;

function mix32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

function hashStr(s) {
  const str = String(s);
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return mix32(h);
}

function num(x, d) {
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
}

function clamp(x, a, b) {
  return x < a ? a : x > b ? b : x;
}

function smooth(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

class Rng {
  constructor(a, b) {
    this.k0 = mix32((a ^ 0x85ebca6b) >>> 0);
    this.k1 = mix32((b ^ 0xc2b2ae35) >>> 0);
    this.c = 0;
    this.spare = NaN;
  }

  next() {
    this.c = (this.c + 1) >>> 0;
    return (mix32((mix32((this.c ^ this.k0) >>> 0) + this.k1) >>> 0) + 0.5) / 4294967296;
  }

  normal() {
    if (this.spare === this.spare) {
      const s = this.spare;
      this.spare = NaN;
      return s;
    }
    const u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(TWO_PI * v);
    return r * Math.cos(TWO_PI * v);
  }

  poisson(mean) {
    if (!(mean > 0)) return 0;
    if (mean < 30) {
      const L = Math.exp(-mean);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= this.next();
      } while (p > L);
      return k - 1;
    }
    return Math.max(0, Math.round(mean + Math.sqrt(mean) * this.normal()));
  }
}

// --- RBJ biquads, state layout [b0, b1, b2, a1, a2, s1, s2] (transposed direct form II) ---
function bqNew() {
  return new Float64Array(7);
}
function bqBand(q, f, Q, sr) {
  const w = (TWO_PI * clamp(f, 10, 0.45 * sr)) / sr;
  const al = Math.sin(w) / (2 * Q);
  const a0 = 1 + al;
  q[0] = al / a0;
  q[1] = 0;
  q[2] = -al / a0;
  q[3] = (-2 * Math.cos(w)) / a0;
  q[4] = (1 - al) / a0;
}
function bqLow(q, f, Q, sr) {
  const w = (TWO_PI * clamp(f, 10, 0.45 * sr)) / sr;
  const al = Math.sin(w) / (2 * Q);
  const cw = Math.cos(w);
  const a0 = 1 + al;
  q[0] = (1 - cw) / 2 / a0;
  q[1] = (1 - cw) / a0;
  q[2] = (1 - cw) / 2 / a0;
  q[3] = (-2 * cw) / a0;
  q[4] = (1 - al) / a0;
}
function bqHigh(q, f, Q, sr) {
  const w = (TWO_PI * clamp(f, 10, 0.45 * sr)) / sr;
  const al = Math.sin(w) / (2 * Q);
  const cw = Math.cos(w);
  const a0 = 1 + al;
  q[0] = (1 + cw) / 2 / a0;
  q[1] = -(1 + cw) / a0;
  q[2] = (1 + cw) / 2 / a0;
  q[3] = (-2 * cw) / a0;
  q[4] = (1 - al) / a0;
}
function bq(q, x) {
  const y = q[0] * x + q[5];
  q[5] = q[1] * x - q[3] * y + q[6];
  q[6] = q[2] * x - q[4] * y;
  return y;
}

/** Energy-normalized 2-pole resonator bank (same form as the rain-surface modes). */
class Res {
  constructor(n) {
    this.n = n;
    this.c1 = new Float64Array(n);
    this.c2 = new Float64Array(n);
    this.b = new Float64Array(n);
    this.y1 = new Float64Array(n);
    this.y2 = new Float64Array(n);
  }

  set(k, f, t60, g, sr) {
    const ff = clamp(f, 20, 0.45 * sr);
    const r = Math.exp(-6.907755 / (Math.max(0.001, t60) * sr));
    const w = (TWO_PI * ff) / sr;
    this.c1[k] = 2 * r * Math.cos(w);
    this.c2[k] = r * r;
    this.b[k] = g * Math.sin(w) * Math.sqrt(2 * (1 - r * r));
  }

  tick(k, x) {
    const y = this.c1[k] * this.y1[k] - this.c2[k] * this.y2[k] + this.b[k] * x;
    this.y2[k] = this.y1[k];
    this.y1[k] = y;
    return y;
  }
}

const MODES = { air: 0, vegetation: 1, grass: 2, wire: 3, flag: 4, chime: 5, wiper: 6 };

class WindProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'wind', defaultValue: 0, minValue: 0, maxValue: 200, automationRate: 'k-rate' },
      { name: 'gain', defaultValue: 1, minValue: 0, maxValue: 1000, automationRate: 'k-rate' },
      { name: 'aux', defaultValue: 0, minValue: -1000, maxValue: 1000, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();
    this.sr = sampleRate;
    this.pos = 0;
    this.U = -1;
    this.lastCfg = {};
    this.configure((options && options.processorOptions) || {});
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d && d.type === 'config') this.configure({ ...this.lastCfg, ...(d.config || {}) });
    };
  }

  configure(cfg) {
    const c = cfg || {};
    this.lastCfg = c;
    const sr = this.sr;
    this.id = String(c.id ?? '');
    this.mode = MODES[c.mode] !== undefined ? MODES[c.mode] : 0;
    this.seed = num(c.seed, 1) >>> 0;
    this.rng = new Rng(this.seed, hashStr(this.id + ':' + (c.mode || 'air')));
    this.ns0 = (mix32((this.seed ^ hashStr(this.id + ':n0')) >>> 0) | 1) | 0;
    this.ns1 = (mix32((this.seed ^ hashStr(this.id + ':n1')) >>> 0) | 1) | 0;
    this.windScale = Math.max(0, num(c.windScale, 1));
    this.species = String(c.species || 'deciduous');
    this.leafOn = c.leafOn !== false;
    this.height = clamp(num(c.height, 10), 0.3, 60);
    this.size = clamp(this.height / 12, 0.25, 3);
    this.area = Math.max(0.1, num(c.area, 10));
    this.diameter = clamp(num(c.diameter, 0.006), 0.0005, 0.5);
    this.length = Math.max(0.2, num(c.length, 10));
    this.level = Math.max(0, num(c.level, 1));
    this.dt = 128 / sr;
    // modulation (OU) states
    this.ou = new Float64Array(6);
    this.ouCoef = (tau) => Math.exp(-this.dt / tau);
    this.coefU = 1 - Math.exp(-this.dt / 0.12);
    this.lastU = -1;
    this.ring = [new Float32Array(RING), new Float32Array(RING), new Float32Array(RING)];
    this.f = [];
    for (let i = 0; i < 12; i++) this.f.push(bqNew());
    this.brown = [0, 0];

    if (this.mode === 3) {
      this.fw = 0;
      this.ft = 0;
      this.ph = 0;
      this.lockG = 0.45;
      this.f1 = Math.max(2, num(c.f1, 15 * Math.sqrt(20 / this.length)));
    } else if (this.mode === 4) {
      this.flapPh = 0;
      this.res = new Res(2);
      this.res.set(0, 1150 * (1 + 0.05 * this.rng.normal()), 0.25, 1, sr);
      this.res.set(1, 2900 * (1 + 0.05 * this.rng.normal()), 0.18, 0.6, sr);
      bqHigh(this.f[2], 800, 0.7, sr);
      bqBand(this.f[1], 900, 0.7, sr);
    } else if (this.mode === 5) {
      const ratios = [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3];
      const notes = Array.isArray(c.notes) && c.notes.length ? c.notes.map((x) => num(x, 880)) : null;
      const base = 600 + 500 * this.rng.next();
      this.tubes = notes ? notes.slice(0, 6) : ratios.map((r) => base * r);
      const parts = [1, 2.756, 5.404];
      const t60 = [3.5, 1.6, 0.7];
      const g = [1, 0.55, 0.3];
      this.res = new Res(this.tubes.length * 3);
      for (let i = 0; i < this.tubes.length; i++) {
        for (let p = 0; p < 3; p++) this.res.set(i * 3 + p, this.tubes[i] * parts[p], t60[p], g[p], sr);
      }
    } else if (this.mode === 6) {
      const sp = String(c.speed || 'slow');
      this.cpm = num(c.cpm, sp === 'fast' ? 55 : sp === 'intermittent' ? 12 : sp === 'off' ? 0 : 28);
      this.wph = 0;
      this.half = 0;
      this.res = new Res(1);
      this.res.set(0, 110, 0.06, 1, sr);
      bqBand(this.f[0], 1700, 0.9, sr);
      this.motorPh = 0;
      this.sqPh = 0;
    } else if (this.mode === 1) {
      this.twigs = [];
      for (let i = 0; i < 4; i++) this.twigs.push({ d: (3 + 3 * this.rng.next()) * 1e-3, ph: this.rng.next() * TWO_PI });
      this.creak = null;
      bqLow(this.f[7], 300, 0.7, sr);
    } else if (this.mode === 2) {
      bqHigh(this.f[0], 2500, 0.7, sr);
      bqLow(this.f[1], 9000, 0.7, sr);
      bqBand(this.f[2], 5000, 1.5, sr);
    }
  }

  ouStep(i, tau) {
    const a = Math.exp(-this.dt / tau);
    this.ou[i] = this.ou[i] * a + Math.sqrt(1 - a * a) * this.rng.normal();
    return this.ou[i];
  }

  process(inputs, outputs, parameters) {
    const out = outputs[0];
    if (!out || !out[0]) return true;
    const N = out[0].length;
    const sr = this.sr;
    this.dt = N / sr;
    const windRef = Math.max(0, parameters.wind ? parameters.wind[0] : 0);
    const gain = Math.max(0, parameters.gain ? parameters.gain[0] : 1) * this.level;
    const aux = parameters.aux ? parameters.aux[0] : 0;
    const target = windRef * this.windScale;
    if (this.U < 0) this.U = target;
    else this.U += (target - this.U) * (1 - Math.exp(-this.dt / 0.12));
    const U = this.U;
    const L = out[0];
    const R = out.length > 1 ? out[1] : null;
    switch (this.mode) {
      case 0:
        this.air(N, U, L, R);
        break;
      case 1:
        if (this.species === 'conifer') this.conifer(N, U, L);
        else if (!this.leafOn || this.species === 'bare') this.bare(N, U, L);
        else this.broadleaf(N, U, L);
        break;
      case 2:
        this.grass(N, U, L);
        break;
      case 3:
        this.wire(N, U, L);
        break;
      case 4:
        this.flag(N, U, L);
        break;
      case 5:
        this.chime(N, U, L);
        break;
      case 6:
        this.wiper(N, aux, L);
        break;
      default:
        L.fill(0);
    }
    for (let i = 0; i < N; i++) L[i] *= gain;
    if (R && this.mode === 0) {
      for (let i = 0; i < N; i++) R[i] *= gain;
    } else if (R) {
      R.set(L);
    }
    for (let c = 2; c < out.length; c++) out[c].set(L);
    this.pos = (this.pos + N) & RMASK;
    return true;
  }

  air(N, U, L, R) {
    const sr = this.sr;
    if (Math.abs(U - this.lastU) > 0.005 * (this.lastU + 0.5)) {
      this.lastU = U;
      const f1 = 60 + 22 * U;
      const f2 = 220 + 85 * U;
      bqBand(this.f[0], f1, 0.7, sr);
      bqBand(this.f[1], f2, 1.1, sr);
      bqBand(this.f[2], f1 * 1.07, 0.7, sr);
      bqBand(this.f[3], f2 * 0.94, 1.1, sr);
    }
    const amp = 0.18 * Math.pow(U / 10, 2);
    const rum = 0.25 * Math.pow(U / 10, 2.5);
    const m0 = Math.exp(0.25 * (0.6 * this.ouStep(0, 0.35) + 0.4 * this.ouStep(1, 1.4)));
    const m1 = Math.exp(0.25 * (0.6 * this.ouStep(2, 0.35) + 0.4 * this.ouStep(3, 1.4)));
    let n0 = this.ns0;
    let n1 = this.ns1;
    let b0 = this.brown[0];
    let b1 = this.brown[1];
    const fa = this.f[0];
    const fb = this.f[1];
    const fc = this.f[2];
    const fd = this.f[3];
    for (let i = 0; i < N; i++) {
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      n1 ^= n1 << 13;
      n1 ^= n1 >>> 17;
      n1 ^= n1 << 5;
      const w0 = n0 / 2147483648;
      const w1 = n1 / 2147483648;
      b0 = 0.996 * b0 + 0.06 * w0;
      b1 = 0.996 * b1 + 0.06 * w1;
      L[i] = m0 * (amp * (0.9 * bq(fa, w0) + 0.6 * bq(fb, w0)) + rum * b0);
      if (R) R[i] = m1 * (amp * (0.9 * bq(fc, w1) + 0.6 * bq(fd, w1)) + rum * b1);
    }
    this.ns0 = n0;
    this.ns1 = n1;
    this.brown[0] = b0;
    this.brown[1] = b1;
  }

  ticks(N, rate, ampScale, probs) {
    const rng = this.rng;
    const n = rng.poisson(Math.min(8000, rate) * (N / this.sr));
    for (let k = 0; k < n; k++) {
      const off = Math.floor(rng.next() * N);
      const a = ampScale * (0.3 + rng.next()) * (rng.next() < 0.5 ? -1 : 1);
      const u = rng.next();
      const band = u < probs[0] ? 0 : u < probs[0] + probs[1] ? 1 : 2;
      const r = this.ring[band];
      const j = (this.pos + off) & RMASK;
      r[j] += a;
      r[(j + 1) & RMASK] -= 0.6 * a;
    }
  }

  broadleaf(N, U, L) {
    const sr = this.sr;
    if (Math.abs(U - this.lastU) > 0.005 * (this.lastU + 0.5)) {
      this.lastU = U;
      bqBand(this.f[0], 1800, 2, sr);
      bqBand(this.f[1], 3600, 2.5, sr);
      bqBand(this.f[2], 6800, 3, sr);
      bqBand(this.f[3], 650 + 55 * U, 0.6, sr);
    }
    const sz = this.size;
    const rate = sz * 900 * Math.pow(Math.max(0, U - 1.5), 1.6);
    this.ticks(N, rate, 0.08 * Math.pow(U / 8, 1.2), [0.4, 0.4]);
    const swish = 0.12 * sz * Math.pow(U / 10, 2.2) * Math.exp(0.35 * this.ouStep(0, 1.5));
    const low = 0.05 * sz * Math.pow(U / 10, 2.5) * Math.exp(0.3 * this.ouStep(1, 3));
    let n0 = this.ns0;
    const r0 = this.ring[0];
    const r1 = this.ring[1];
    const r2 = this.ring[2];
    for (let i = 0; i < N; i++) {
      const j = (this.pos + i) & RMASK;
      const t0 = r0[j];
      const t1 = r1[j];
      const t2 = r2[j];
      r0[j] = 0;
      r1[j] = 0;
      r2[j] = 0;
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      L[i] = bq(this.f[0], t0) + bq(this.f[1], t1) + bq(this.f[2], t2) + swish * bq(this.f[3], w) + low * bq(this.f[7], w);
    }
    this.ns0 = n0;
  }

  conifer(N, U, L) {
    const sr = this.sr;
    const fs = clamp((0.2 * U) / 0.0012, 150, 4500);
    if (Math.abs(U - this.lastU) > 0.005 * (this.lastU + 0.5)) {
      this.lastU = U;
      bqBand(this.f[0], fs, 1.4, sr);
      bqBand(this.f[1], fs, 1.4, sr);
      bqBand(this.f[2], fs / 3, 0.7, sr);
    }
    const sz = this.size;
    const a = 0.25 * sz * Math.pow(U / 10, 2.4) * Math.exp(0.35 * this.ouStep(0, 1.8));
    let n0 = this.ns0;
    for (let i = 0; i < N; i++) {
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      L[i] = a * (1.6 * bq(this.f[1], bq(this.f[0], w)) + 0.4 * bq(this.f[2], w));
    }
    this.ns0 = n0;
  }

  bare(N, U, L) {
    const sr = this.sr;
    const rng = this.rng;
    if (Math.abs(U - this.lastU) > 0.005 * (this.lastU + 0.5)) {
      this.lastU = U;
      for (let k = 0; k < 4; k++) bqBand(this.f[k], clamp((0.2 * Math.max(U, 0.5)) / this.twigs[k].d, 60, 0.4 * sr), 18, sr);
      bqBand(this.f[4], 650 + 55 * U, 0.6, sr);
    }
    const sz = this.size;
    const wh = 0.06 * sz * Math.pow(U / 12, 3) * smooth(4, 8, U);
    const am = [];
    for (let k = 0; k < 4; k++) am.push(wh * Math.exp(0.5 * this.ouStep(k, 0.8 + 0.5 * k)));
    const swish = 0.05 * sz * Math.pow(U / 10, 2.2) * Math.exp(0.35 * this.ouStep(4, 1.5));
    if (!this.creak && rng.next() < 0.25 * smooth(14, 20, U) * this.dt) {
      const fc = 180 + 220 * rng.next();
      bqBand(this.f[5], fc, 6, sr);
      this.creak = { left: Math.round((0.4 + 0.8 * rng.next()) * sr), len: 0, rate: 25 + 35 * rng.next(), ph: 0 };
      this.creak.len = this.creak.left;
    }
    let n0 = this.ns0;
    const cr = this.creak;
    for (let i = 0; i < N; i++) {
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      let y = swish * bq(this.f[4], w);
      for (let k = 0; k < 4; k++) y += am[k] * bq(this.f[k], w) * 4;
      if (cr && cr.left > 0) {
        cr.ph += cr.rate / sr;
        let x = 0;
        if (cr.ph >= 1) {
          cr.ph -= 1;
          x = 0.8 * Math.sin((Math.PI * (cr.len - cr.left)) / cr.len);
        }
        y += 0.3 * bq(this.f[5], x);
        cr.left--;
      }
      L[i] = y;
    }
    if (cr && cr.left <= 0) this.creak = null;
    this.ns0 = n0;
  }

  grass(N, U, L) {
    const sq = Math.sqrt(this.area / 10);
    const hiss = 0.1 * sq * Math.pow(U / 8, 2) * smooth(1.5, 3, U) * Math.exp(0.3 * this.ouStep(0, 1.2));
    this.ticks(N, 300 * sq * Math.pow(Math.max(0, U - 2), 1.2), 0.03 * Math.pow(U / 8, 1.1), [0, 0]);
    let n0 = this.ns0;
    const r2 = this.ring[2];
    for (let i = 0; i < N; i++) {
      const j = (this.pos + i) & RMASK;
      const t = r2[j];
      r2[j] = 0;
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      L[i] = hiss * bq(this.f[1], bq(this.f[0], w)) + bq(this.f[2], t);
    }
    this.ns0 = n0;
  }

  wire(N, U, L) {
    const sr = this.sr;
    const d = this.diameter;
    const fsh = clamp((0.2 * U) / d, 15, 0.4 * sr);
    this.fw += (fsh - this.fw) * (1 - Math.exp(-this.dt / 0.4));
    const f1 = this.f1;
    const n = Math.max(1, Math.round(this.fw / f1));
    const fn = n * f1;
    const lock = Math.abs(this.fw - fn) < 0.15 * f1;
    const tgtF = lock ? fn : this.fw;
    this.ft += (tgtF - this.ft) * (1 - Math.exp(-this.dt / 0.08));
    this.lockG += ((lock ? 1 : 0.45) - this.lockG) * (1 - Math.exp(-this.dt / 0.25));
    const Re = (U * d) / 1.5e-5;
    const reGate = smooth(40, 150, Re) * (1 - smooth(1e5, 3e5, Re));
    const amp = 0.05 * Math.pow(U / 10, 3) * Math.sqrt(this.length / 10) * smooth(1.5, 3.5, U) * reGate * this.lockG * Math.exp(0.4 * this.ouStep(0, 0.5));
    bqBand(this.f[0], this.ft, 25, sr);
    let ph = this.ph;
    const k = (TWO_PI * this.ft) / sr;
    let n0 = this.ns0;
    for (let i = 0; i < N; i++) {
      ph += k;
      if (ph > TWO_PI) ph -= TWO_PI;
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      L[i] = amp * (Math.sin(ph) + 0.15 * Math.sin(2 * ph) + 2.5 * bq(this.f[0], w));
    }
    this.ph = ph;
    this.ns0 = n0;
  }

  flag(N, U, L) {
    const sr = this.sr;
    const rng = this.rng;
    const fFlap = clamp((0.3 * U) / this.length, 0.2, 8);
    const flutter = 0.1 * Math.pow(U / 10, 2) * smooth(1, 3, U);
    const snapP = smooth(4, 10, U);
    const r0 = this.ring[0];
    let ph = this.flapPh;
    const before = Math.floor(ph);
    ph += fFlap * this.dt;
    if (Math.floor(ph) !== before && rng.next() < snapP) {
      const off = Math.floor(rng.next() * N);
      const amp = 0.25 * Math.pow(U / 10, 2) * (0.5 + rng.next());
      const tau = 0.006 * sr;
      const kk = Math.exp(-1 / tau);
      let e = amp;
      for (let i = 0; i < 4 * tau; i++) {
        r0[(this.pos + off + i) & RMASK] += e;
        e *= kk;
      }
    }
    const pingRate = 0.4 * smooth(5, 12, U);
    let ping = 0;
    let pingOff = -1;
    if (rng.next() < pingRate * this.dt) {
      ping = 0.6 * (0.4 + rng.next());
      pingOff = Math.floor(rng.next() * N);
    }
    let n0 = this.ns0;
    const ph0 = this.flapPh;
    for (let i = 0; i < N; i++) {
      const j = (this.pos + i) & RMASK;
      const env = r0[j];
      r0[j] = 0;
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      const p = ph0 + (fFlap * i) / sr;
      const mod = 0.6 + 0.4 * (0.5 + 0.5 * Math.sin(TWO_PI * 2 * p));
      const x = i === pingOff ? ping : 0;
      L[i] = flutter * mod * bq(this.f[1], w) + bq(this.f[2], env * w) + this.res.tick(0, x) + this.res.tick(1, x);
    }
    this.flapPh = ph % 1000;
    this.ns0 = n0;
  }

  chime(N, U, L) {
    const rng = this.rng;
    const rate = Math.min(10, 0.7 * Math.pow(Math.max(0, U - 1.5), 1.3));
    const n = rng.poisson(rate * this.dt);
    const hits = [];
    for (let k = 0; k < n; k++) {
      hits.push({ off: Math.floor(rng.next() * N), tube: Math.floor(rng.next() * this.tubes.length), v: 0.08 * (0.3 + 0.7 * rng.next() * Math.min(1, U / 10)) });
    }
    const res = this.res;
    const nt = this.tubes.length;
    for (let i = 0; i < N; i++) {
      let y = 0;
      for (let t = 0; t < nt; t++) {
        let x = 0;
        for (let h = 0; h < hits.length; h++) if (hits[h].off === i && hits[h].tube === t) x += hits[h].v;
        y += res.tick(t * 3, x) + res.tick(t * 3 + 1, x) + res.tick(t * 3 + 2, x);
        if (x !== 0) y += 0.3 * x;
      }
      L[i] = y;
    }
    for (let k = 0; k < res.n; k++) {
      if (Math.abs(res.y1[k]) < 1e-18) res.y1[k] = 0;
      if (Math.abs(res.y2[k]) < 1e-18) res.y2[k] = 0;
    }
  }

  wiper(N, wetness, L) {
    const sr = this.sr;
    if (!(this.cpm > 0)) {
      L.fill(0);
      return;
    }
    const period = 60 / this.cpm;
    const halfP = period / 2;
    const sweepT = Math.min(halfP * 0.85, 0.75);
    const wet = clamp(wetness, 0, 1);
    const squeak = 0.02 * Math.max(0, 1 - wet / 0.35);
    let n0 = this.ns0;
    for (let i = 0; i < N; i++) {
      this.wph += 1 / sr;
      let x = 0;
      if (this.wph >= halfP) {
        this.wph -= halfP;
        this.half ^= 1;
        x = 0.5;
      }
      const u = this.wph / sweepT;
      const env = u < 1 ? Math.pow(Math.sin(Math.PI * u), 0.7) : 0;
      n0 ^= n0 << 13;
      n0 ^= n0 >>> 17;
      n0 ^= n0 << 5;
      const w = n0 / 2147483648;
      let y = 0.06 * env * (0.7 + 0.6 * wet) * bq(this.f[0], w);
      if (squeak > 0 && (u < 0.15 || (u > 0.85 && u < 1))) {
        this.sqPh += (TWO_PI * (1500 + 700 * u + 30 * Math.sin(TWO_PI * 7 * this.wph))) / sr;
        y += squeak * Math.sin(this.sqPh) * Math.sin(Math.PI * (u < 0.15 ? u / 0.15 : (1 - u) / 0.15));
      }
      if (env > 0) {
        this.motorPh += (TWO_PI * 98) / sr;
        y += 0.006 * (Math.sin(this.motorPh) + 0.5 * Math.sin(2 * this.motorPh));
      }
      y += this.res.tick(0, x);
      L[i] = y;
    }
    if (this.sqPh > 1e6) this.sqPh %= TWO_PI;
    if (this.motorPh > 1e6) this.motorPh %= TWO_PI;
    this.ns0 = n0;
  }
}

registerProcessor('rg-wind', WindProcessor);
