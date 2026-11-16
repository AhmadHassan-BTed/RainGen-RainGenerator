/**
 * rg-bed: far-field rain bed and drizzle hiss.
 *
 * The far field (beyond ~12 listener heights) holds >10^5 impacts per second, so it is
 * rendered as a statistically exact diffuse texture: every output channel is an independent
 * (decorrelated) realization of dense Poisson micro-impacts plus pink-tilted noise, shaped by
 * a ground-impact spectrum (high-pass ~150 Hz, broad ~2.5 kHz bump) and low-passed at the
 * ISO 9613-1 air-absorption cutoff for the bed distance ('cutoff' param, computed by the
 * engine from temperature and humidity). Drizzle adds a fine high hiss (3.5-11 kHz) of tiny
 * crackles. Slow multiplicative 1/f modulation (Ornstein-Uhlenbeck sum, 0.7-45 s, +-depth dB)
 * keeps the bed alive with no periodic modulation in the 1-20 Hz fluctuation band.
 */

const TWO_PI = Math.PI * 2;
const RING = 4096;
const RMASK = RING - 1;
const MAXCH = 8;

function mix32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

function num(x, d) {
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
}

function clamp(x, a, b) {
  return x < a ? a : x > b ? b : x;
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

function bqLow(q, f, Q, sr) {
  const w = (TWO_PI * clamp(f, 20, 0.45 * sr)) / sr;
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
  const w = (TWO_PI * clamp(f, 20, 0.45 * sr)) / sr;
  const al = Math.sin(w) / (2 * Q);
  const cw = Math.cos(w);
  const a0 = 1 + al;
  q[0] = (1 + cw) / 2 / a0;
  q[1] = -(1 + cw) / a0;
  q[2] = (1 + cw) / 2 / a0;
  q[3] = (-2 * cw) / a0;
  q[4] = (1 - al) / a0;
}

function bqPeak(q, f, Q, dB, sr) {
  const A = Math.pow(10, dB / 40);
  const w = (TWO_PI * clamp(f, 20, 0.45 * sr)) / sr;
  const al = Math.sin(w) / (2 * Q);
  const cw = Math.cos(w);
  const a0 = 1 + al / A;
  q[0] = (1 + al * A) / a0;
  q[1] = (-2 * cw) / a0;
  q[2] = (1 - al * A) / a0;
  q[3] = (-2 * cw) / a0;
  q[4] = (1 - al / A) / a0;
}

function bq(q, x) {
  const y = q[0] * x + q[5];
  q[5] = q[1] * x - q[3] * y + q[6];
  q[6] = q[2] * x - q[4] * y;
  return y;
}

class Chan {
  constructor(seed, idx, sr) {
    this.ns = (mix32((seed ^ Math.imul(idx + 1, 0x9e3779b9)) >>> 0) | 1) | 0;
    this.rng = new Rng(seed, 0x51ed270b + idx * 7919);
    this.pk = new Float64Array(7);
    this.hp = new Float64Array(7);
    this.pe = new Float64Array(7);
    this.lp1 = new Float64Array(7);
    this.lp2 = new Float64Array(7);
    this.dh = new Float64Array(7);
    this.dl = new Float64Array(7);
    bqHigh(this.hp, 150, 0.6, sr);
    bqPeak(this.pe, 2500, 0.7, 4, sr);
    bqHigh(this.dh, 3500, 0.7, sr);
    bqLow(this.dl, 11000, 0.7, sr);
    this.ou = new Float64Array(4);
    this.dou = new Float64Array(2);
    for (let k = 0; k < 4; k++) this.ou[k] = this.rng.normal();
    this.crackle = new Float32Array(RING);
    this.drizzleCr = new Float32Array(RING);
  }
}

class BedProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'level', defaultValue: 0, minValue: 0, maxValue: 100, automationRate: 'k-rate' },
      { name: 'cutoff', defaultValue: 8000, minValue: 100, maxValue: 24000, automationRate: 'k-rate' },
      { name: 'drizzle', defaultValue: 0, minValue: 0, maxValue: 100, automationRate: 'k-rate' },
      { name: 'density', defaultValue: 1, minValue: 0, maxValue: 100, automationRate: 'k-rate' },
      { name: 'intensity', defaultValue: 0, minValue: 0, maxValue: 100, automationRate: 'k-rate' },
      { name: 'gain', defaultValue: 1, minValue: 0, maxValue: 1000, automationRate: 'k-rate' },
      { name: 'depth', defaultValue: 1.5, minValue: 0, maxValue: 12, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();
    const po = (options && options.processorOptions) || {};
    this.sr = sampleRate;
    this.seed = num(po.seed, 7) >>> 0;
    this.chans = [];
    for (let i = 0; i < MAXCH; i++) this.chans.push(new Chan(this.seed, i, this.sr));
    this.cut = -1;
    this.pos = 0;
    this.taus = [0.7, 2.8, 11, 45];
    this.wts = [0.35, 0.45, 0.6, 0.75];
    let s = 0;
    for (const w of this.wts) s += w * w;
    this.wnorm = 1 / Math.sqrt(s);
    this.ouN = 0;
    this.ouA = new Float64Array(5);
    this.ouS = new Float64Array(5);
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d && d.type === 'config' && d.config && d.config.seed !== undefined) {
        this.seed = num(d.config.seed, this.seed) >>> 0;
        this.chans = [];
        for (let i = 0; i < MAXCH; i++) this.chans.push(new Chan(this.seed, i, this.sr));
        this.cut = -1;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const out = outputs[0];
    if (!out || !out[0]) return true;
    const N = out[0].length;
    const sr = this.sr;
    const dt = N / sr;
    const P = parameters;
    const level = Math.max(0, P.level ? P.level[0] : 0);
    const cutoff = clamp(P.cutoff ? P.cutoff[0] : 8000, 100, 0.45 * sr);
    const drizzle = Math.max(0, P.drizzle ? P.drizzle[0] : 0);
    const density = Math.max(0, P.density ? P.density[0] : 1);
    const intensity = Math.max(0, P.intensity ? P.intensity[0] : 0);
    const gain = Math.max(0, P.gain ? P.gain[0] : 1);
    const depth = clamp(P.depth ? P.depth[0] : 1.5, 0, 12);
    const bedAmp = level * intensity * gain;
    const drzAmp = drizzle * intensity * gain;
    const nch = Math.min(out.length, MAXCH);
    if (bedAmp <= 1e-9 && drzAmp <= 1e-9) {
      for (let c = 0; c < out.length; c++) out[c].fill(0);
      this.pos = (this.pos + N) & RMASK;
      return true;
    }
    if (Math.abs(cutoff - this.cut) > 0.01 * this.cut) {
      this.cut = cutoff;
      for (const ch of this.chans) {
        bqLow(ch.lp1, cutoff, 0.6, sr);
        bqLow(ch.lp2, Math.min(0.45 * sr, cutoff * 1.4), 0.55, sr);
      }
    }
    const crRate = 1500 * density;
    const drRate = 3000;
    // OU decay / drive coefficients depend only on the quantum length: computed once per length.
    if (N !== this.ouN) {
      this.ouN = N;
      for (let k = 0; k < 5; k++) {
        const a = Math.exp(-dt / (k < 4 ? this.taus[k] : 3));
        this.ouA[k] = a;
        this.ouS[k] = Math.sqrt(1 - a * a);
      }
    }
    const ouA = this.ouA;
    const ouS = this.ouS;
    for (let c = 0; c < nch; c++) {
      const ch = this.chans[c];
      const rng = ch.rng;
      let z = 0;
      for (let k = 0; k < 4; k++) {
        ch.ou[k] = ch.ou[k] * ouA[k] + ouS[k] * rng.normal();
        z += this.wts[k] * ch.ou[k];
      }
      z *= this.wnorm;
      const mod = Math.pow(10, (depth * z) / 20);
      ch.dou[0] = ch.dou[0] * ouA[4] + ouS[4] * rng.normal();
      const dmod = Math.pow(10, (0.8 * depth * ch.dou[0]) / 20);
      if (bedAmp > 1e-9) {
        const n = rng.poisson(crRate * dt);
        for (let k = 0; k < n; k++) {
          const off = Math.floor(rng.next() * N);
          const a = -Math.log(rng.next()) * (rng.next() < 0.5 ? -1 : 1);
          const j = (this.pos + off) & RMASK;
          ch.crackle[j] += a;
          ch.crackle[(j + 1) & RMASK] += 0.5 * a;
        }
      }
      if (drzAmp > 1e-9) {
        const n = rng.poisson(drRate * dt);
        for (let k = 0; k < n; k++) {
          const off = Math.floor(rng.next() * N);
          ch.drizzleCr[(this.pos + off) & RMASK] += -Math.log(rng.next()) * (rng.next() < 0.5 ? -1 : 1);
        }
      }
      const o = out[c];
      const pk = ch.pk;
      let ns = ch.ns;
      const g = bedAmp * mod;
      const gd = drzAmp * dmod;
      const crN = 1 / Math.sqrt(Math.max(1, crRate / sr) * 2);
      for (let i = 0; i < N; i++) {
        ns ^= ns << 13;
        ns ^= ns >>> 17;
        ns ^= ns << 5;
        const w = ns / 2147483648;
        // Paul Kellet's pink filter
        pk[0] = 0.99886 * pk[0] + w * 0.0555179;
        pk[1] = 0.99332 * pk[1] + w * 0.0750759;
        pk[2] = 0.969 * pk[2] + w * 0.153852;
        pk[3] = 0.8665 * pk[3] + w * 0.3104856;
        pk[4] = 0.55 * pk[4] + w * 0.5329522;
        pk[5] = -0.7616 * pk[5] - w * 0.016898;
        const pink = (pk[0] + pk[1] + pk[2] + pk[3] + pk[4] + pk[5] + pk[6] + w * 0.5362) * 0.11;
        pk[6] = w * 0.115926;
        const j = (this.pos + i) & RMASK;
        const cr = ch.crackle[j] * crN;
        ch.crackle[j] = 0;
        let y = 0;
        if (g > 0) {
          const x = bq(ch.pe, bq(ch.hp, 0.6 * pink + 0.35 * w + 0.8 * cr));
          y = g * bq(ch.lp2, bq(ch.lp1, x));
        }
        if (gd > 0) {
          const dc = ch.drizzleCr[j];
          ch.drizzleCr[j] = 0;
          y += gd * bq(ch.dl, bq(ch.dh, 0.4 * w + 0.15 * dc));
        } else {
          ch.drizzleCr[j] = 0;
        }
        o[i] = y;
      }
      ch.ns = ns;
    }
    for (let c = nch; c < out.length; c++) out[c].set(out[c % nch]);
    this.pos = (this.pos + N) & RMASK;
    return true;
  }
}

registerProcessor('rg-bed', BedProcessor);
