/**
 * rg-rain-surface: precipitation impacts on one surface patch or object.
 * Physics:
 *
 * - Events: one doubly stochastic (Cox) process per drop class: Poisson inside a render
 *   quantum, rate modulated by multiplicative 1/f noise (Lovejoy & Schertzer 2008) and by
 *   the gust ratio (wind-driven rain curtains).
 * - Solids: liquid-drop force pulse rising like sqrt(t) to F_max ~ 0.83 rho D^2 V^2 at
 *   ~0.18 D/V, width ~ D/V (Gordillo, Sun & Cheng 2018), driving energy-normalized 2-pole
 *   modal resonators (two excitation buses model impact-position dependence); the directly
 *   radiated click is ~ dF/dt. Ice: Hertz contact, t_c = 2.87 (m^2 / (R E*^2 v))^(1/5).
 * - Water: 50-100 us impact click (amplitude ~ U^3 through dF/dt) plus a probabilistic
 *   Minnaert bubble, f a = 3.28 m/s, p(t) = A sin(2 pi f(t) t) e^(-beta t),
 *   f(t) = f0 (1 + xi beta t), xi = 0.1, beta = 0.043 f + 0.0014 f^1.5 (van den Doel 2005).
 *   Entrainment per class after Medwin, Nystuen et al. (1992); small-drop bubbles vanish
 *   above 6-10 m/s wind (Nystuen).
 * - Wetness: a film softens the click and adds slap noise; puddles route impacts to the
 *   water model, with bubbles once deeper than ~2-6 mm (Yu & Hopkins 2018).
 * - Splashes: drops on dry hard surfaces shed micro-impacts 5-50 ms later.
 * - Drips: integrate-and-fire catchments, Tate's-law drop volume, chaotic jitter at high
 *   flow (Shaw 1984). Flow: gutter/downspout bubble cloud through open-pipe resonances
 *   f_n = n c / 2L.
 * - Level of detail: above maxEventsPerSec the remaining events are rendered as
 *   variance-matched noise excitation (Campbell's theorem), keeping the energy exact.
 *
 * Static configuration arrives in processorOptions (and 'config' port messages); every
 * time-varying quantity is a k-rate AudioParam so offline renders are deterministic. The
 * weather shared by every surface (class impact rates and diameters, film wetness, puddle
 * depth, liquid rate) lives on one rg-weather bus node: its output feeds each surface's input,
 * so the bus is rendered first in every quantum and the surfaces read its values directly.
 * A surface only carries its own intensity, wind and gain.
 */

const TWO_PI = Math.PI * 2;
const RING = 8192;
const RMASK = RING - 1;
const QCAP = 2048;
const MAXB = 96;
const OUT_SCALE = 0.0012;
const BOUNDS = { medwin: [0.8, 1.1, 2.2, 3.5], nystuen: [0.8, 1.2, 2.0, 3.5] };
// Per class: 0 minuscule, 1 small, 2 mid, 3 large, 4 very large, 5 hail/ice, 6 drip
const P_ENTRAIN = [0, 0.9, 0.03, 0.35, 0.6, 0.45, 0.75];
const BUB_RMIN = [0.13, 0.13, 0.33, 0.33, 0.55, 0.6, 0.8];
const BUB_RMAX = [0.25, 0.25, 1.6, 1.6, 3.3, 4.0, 3.0];
const BUB_ALPHA = [2, 2, 2.5, 2.5, 2.2, 2.0, 2.2];
const BUB_AMP = [0, 0.12, 0.08, 0.6, 0.9, 0.7, 0.8];
const V2 = 9.65 - 10.3 * Math.exp(-1.2);
const FREF = 4 * V2 * V2;
// Terminal fall speed of liquid drops (Atlas et al. 1973 from 0.5 mm, Stokes drag below 0.1 mm,
// linear bridge between), tabulated every 5 um up to 10 mm (interpolation error < 1e-5 m/s).
const VT_STEP = 0.005;
const VT_N = 2001;
const VT = new Float64Array(VT_N + 1);
{
  const k = (9.81 * 1e-6 * (998 - 1.204)) / (18 * 1.81e-5);
  const vS = k * 0.01;
  const vA = 9.65 - 10.3 * Math.exp(-0.3);
  for (let i = 0; i <= VT_N; i++) {
    const d = i * VT_STEP;
    VT[i] = d >= 0.5 ? 9.65 - 10.3 * Math.exp(-0.6 * d) : d >= 0.1 ? vS + ((vA - vS) * (d - 0.1)) / 0.4 : k * d * d;
  }
}
function liquidFallSpeed(D) {
  const x = D / VT_STEP;
  if (!(x > 0)) return 0.1;
  if (x >= VT_N) return 9.65 - 10.3 * Math.exp(-0.6 * D);
  const i = x | 0;
  const v = VT[i] + (VT[i + 1] - VT[i]) * (x - i);
  return v > 0.1 ? v : 0.1;
}
// Shared weather per bus id: r0..r5, d0..d5, wetness, puddle, liquid.
const BUSES = new Map();
const BUS_SIZE = 15;
const F2MM_N = 0.83 * 998 * 4e-6 * V2 * V2;
const KIND = { solid: 0, water: 1, membrane: 2, soft: 3 };
const RN = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5'];
const DN = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'];

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

function lattice(key, i) {
  return (mix32((key ^ mix32((i | 0) + 0x5bd1e995)) >>> 0) / 4294967295) * 2 - 1;
}

function vnoise(key, x) {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * f * (f * (f * 6 - 15) + 10);
  return lattice(key, i) * (1 - u) + lattice(key, i + 1) * u;
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

  powerLaw(a, b, alpha) {
    const u = this.next();
    const e = 1 - alpha;
    if (Math.abs(e) < 1e-6) return a * Math.pow(b / a, u);
    const pa = Math.pow(a, e);
    const pb = Math.pow(b, e);
    return Math.pow(pa + u * (pb - pa), 1 / e);
  }
}

/** RBJ band-pass (constant 0 dB peak) coefficients into a Float64Array(7) [b0,b1,b2,a1,a2,s1,s2]. */
function setBandpass(bq, f, Q, sr) {
  const w = (TWO_PI * clamp(f, 10, 0.45 * sr)) / sr;
  const al = Math.sin(w) / (2 * Q);
  const a0 = 1 + al;
  bq[0] = al / a0;
  bq[1] = 0;
  bq[2] = -al / a0;
  bq[3] = (-2 * Math.cos(w)) / a0;
  bq[4] = (1 - al) / a0;
}

class Mat {
  constructor(spec, sr, maxModes) {
    const s = spec || {};
    this.name = String(s.name || 'concrete');
    this.kind = KIND[s.kind] !== undefined ? KIND[s.kind] : 0;
    this.fraction = Math.max(0, num(s.fraction, 1));
    this.modal = Math.max(0, num(s.modal, 0.02));
    this.click = Math.max(0, num(s.click, 1));
    this.noise = Math.max(0, num(s.noise, 0));
    this.noiseDecay = Math.max(0.0003, num(s.noiseDecay, 0.002));
    this.splash = Math.max(0, num(s.splash, 0));
    this.puddleable = !!s.puddleable;
    this.wetDamp = clamp(num(s.wetDamp, 0), 0, 0.95);
    this.hardness = clamp(num(s.hardness, 0.5), 0.01, 1);
    this.depthMm = Math.max(0, num(s.depthMm, this.kind === 1 ? 1000 : 0));
    this.waterClick = Math.max(0, num(s.waterClick, 1));
    this.bubble = Math.max(0, num(s.bubble, 1));
    this.spatial = clamp(num(s.spatial, 0.8), 0, 1);
    this.share = this.fraction;
    const fl = Array.isArray(s.modes) ? s.modes : [];
    const f = [];
    const t = [];
    const g = [];
    for (let i = 0; i + 2 < fl.length && f.length < maxModes; i += 3) {
      const fr = num(fl[i], 0);
      if (fr > 20 && fr < 0.45 * sr) {
        f.push(fr);
        t.push(Math.max(0.0012, num(fl[i + 1], 0.01)));
        g.push(num(fl[i + 2], 0.2));
      }
    }
    this.n = f.length;
    this.f = Float64Array.from(f);
    this.t60 = Float64Array.from(t);
    this.g = Float64Array.from(g);
    this.c1 = new Float64Array(this.n);
    this.c2 = new Float64Array(this.n);
    this.b = new Float64Array(this.n);
    this.y1 = new Float64Array(this.n);
    this.y2 = new Float64Array(this.n);
    this.bus = new Uint8Array(this.n);
    for (let k = 0; k < this.n; k++) this.bus[k] = k & 1;
    this.ringA = new Float32Array(RING);
    this.ringB = new Float32Array(RING);
    this.ringN = new Float32Array(RING);
    this.prevF = 0;
    const lp = num(s.lpHz, 0);
    this.lpA = lp > 20 && lp < 0.45 * sr ? 1 - Math.exp((-TWO_PI * lp) / sr) : 0;
    this.lpY = 0;
    this.resLp = 0;
    this.wetSet = -1;
    this.update(0, sr);
  }

  update(wet, sr) {
    this.wetSet = wet;
    for (let k = 0; k < this.n; k++) {
      const f = this.f[k];
      const t60 = Math.max(0.0012, this.t60[k] * (1 - this.wetDamp * wet * (f / (f + 2000))));
      const r = Math.exp(-6.907755 / (t60 * sr));
      const w = (TWO_PI * f) / sr;
      this.c1[k] = 2 * r * Math.cos(w);
      this.c2[k] = r * r;
      // Energy-normalized: impulse-response energy = (modal * g)^2 independent of f and T60.
      this.b[k] = this.modal * this.g[k] * Math.sin(w) * Math.sqrt(2 * (1 - r * r));
    }
  }

  silence() {
    this.y1.fill(0);
    this.y2.fill(0);
    this.prevF = 0;
    this.lpY = 0;
    this.resLp = 0;
  }
}

class RainSurfaceProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'intensity', defaultValue: 0, minValue: 0, maxValue: 100, automationRate: 'k-rate' },
      { name: 'wind', defaultValue: 0, minValue: 0, maxValue: 200, automationRate: 'k-rate' },
      { name: 'gain', defaultValue: 1, minValue: 0, maxValue: 1000, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();
    const po = (options && options.processorOptions) || {};
    this.sr = sampleRate;
    this.statsOn = po.stats !== false;
    this.pos = 0;
    this.abs = 0;
    this.qT = new Float64Array(QCAP);
    this.qD = new Float32Array(QCAP);
    this.qV = new Float32Array(QCAP);
    this.qC = new Int8Array(QCAP);
    this.qK = new Int8Array(QCAP);
    this.qM = new Int8Array(QCAP);
    this.qMicro = new Uint8Array(QCAP);
    this.qN = 0;
    this.bOn = new Uint8Array(MAXB);
    this.bStart = new Float64Array(MAXB);
    // Rising-pitch bubble as a rotating phasor z (sin = Im z) whose step w itself rotates by dw.
    this.bZr = new Float64Array(MAXB);
    this.bZi = new Float64Array(MAXB);
    this.bWr = new Float64Array(MAXB);
    this.bWi = new Float64Array(MAXB);
    this.bDr = new Float64Array(MAXB);
    this.bDi = new Float64Array(MAXB);
    this.bF = new Float64Array(MAXB);
    this.bDf = new Float64Array(MAXB);
    this.bEnv = new Float64Array(MAXB);
    this.bDec = new Float64Array(MAXB);
    this.bAmp = new Float64Array(MAXB);
    this.activeBubbles = 0;
    this.ringW = new Float32Array(RING);
    this.wPrev = 0;
    this.rates = new Float64Array(6);
    this.dpar = new Float64Array(6);
    this.lam = new Float64Array(6);
    this.cx = new Float64Array(3);
    this.coxA = new Float64Array(3);
    this.coxS = new Float64Array(3);
    this.pipe = [];
    this.flowBp = new Float64Array(7);
    this._alloc(128);
    this.dcX = 0;
    this.dcY = 0;
    this.quiet = 0;
    this.evCount = 0;
    this.statCount = 0;
    this.physRate = 0;
    this.residVar = 0;
    this.residWater = 0;
    this.flowLevel = 0;
    this.store = 0;
    this.dribAcc = 0;
    this.streamExcess = 0;
    this.blockEvents = 0;
    this.wind = 0;
    this.wet = 0;
    this.puddleDepth = 0;
    this.puddleCover = 0;
    this.wResLast = 0;
    this.lastCfg = {};
    this.configure(po);
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  _alloc(n) {
    // Excitation scratch shares the rings' element type so the inner loops stay monomorphic.
    this.exA = new Float32Array(n);
    this.exB = new Float32Array(n);
    this.exN = new Float32Array(n);
    this.acc = new Float64Array(n);
    this.mixBuf = new Float64Array(n);
  }

  configure(cfg) {
    const c = cfg || {};
    this.lastCfg = c;
    const sr = this.sr;
    this.id = String(c.id ?? '');
    this.busId = String(c.bus ?? 'main');
    this.wetMul = Math.max(0, num(c.wetMul, 1));
    this.puddleMul = Math.max(0, num(c.puddleMul, 1));
    this.flowMul = Math.max(0, num(c.flowMul, 1));
    this.kind = c.kind === 'drip' ? 1 : c.kind === 'flow' ? 2 : 0;
    this.area = Math.max(0, num(c.area, 1));
    this.exposure = clamp(num(c.exposure, 1), 0, 4);
    this.maxEv = Math.max(50, num(c.maxEventsPerSec, 3000));
    this.maxModes = clamp(Math.round(num(c.maxModes, 16)), 0, 24);
    this.maxBub = clamp(Math.round(num(c.maxBubbles, 48)), 4, MAXB);
    this.coxDepth = clamp(num(c.coxDepth, 0.35), 0, 1.5);
    this.waterQuality = clamp(num(c.waterQuality, 1), 0, 1);
    this.seed = num(c.seed, 1) >>> 0;
    this.sharedKey = num(c.sharedSeed, this.seed) >>> 0;
    this.rng = new Rng(this.seed, hashStr(this.id || 'surface'));
    this.ns = (mix32((this.seed ^ hashStr(this.id + ':noise')) >>> 0) | 1) | 0;
    const pk = Array.isArray(c.precipKinds) ? c.precipKinds : [0, 0, 0, 0, 0, 1];
    this.kinds = new Int8Array(6);
    for (let i = 0; i < 6; i++) this.kinds[i] = clamp(Math.round(num(pk[i], i === 5 ? 1 : 0)), 0, 3);
    const b = BOUNDS[c.boundaries] || BOUNDS.medwin;
    const dMin = clamp(num(c.dMin, 0.3), 0.05, 2);
    const dMax = clamp(num(c.dMax, 6), b[3] + 0.1, 9);
    this.lo = [dMin, b[0], b[1], b[2], b[3], 5, 3];
    this.hi = [b[0], b[1], b[2], b[3], dMax, 80, 6];
    const specs = Array.isArray(c.materials) && c.materials.length ? c.materials.slice(0, 4) : [{ name: 'concrete', kind: 'solid', modal: 0.02, click: 1, splash: 0.8 }];
    this.mats = specs.map((s) => new Mat(s, sr, this.maxModes));
    let tot = 0;
    for (const m of this.mats) tot += m.fraction;
    if (!(tot > 0)) {
      for (const m of this.mats) m.fraction = 1;
      tot = this.mats.length;
    }
    this.cum = new Float64Array(this.mats.length);
    let acc = 0;
    for (let i = 0; i < this.mats.length; i++) {
      this.mats[i].fraction /= tot;
      acc += this.mats[i].fraction;
      this.cum[i] = acc;
    }
    this.cum[this.mats.length - 1] = 1;
    let wc = 0;
    let bg = 0;
    let nw = 0;
    for (const m of this.mats) {
      if (m.kind === 1 || m.puddleable) {
        wc += m.waterClick;
        bg += m.bubble;
        nw++;
      }
    }
    this.wClick = nw ? wc / nw : 1;
    this.bubbleGain = nw ? bg / nw : 1;
    const taus = [0.12, 0.5, 2.0];
    const dt = 128 / sr;
    for (let k = 0; k < 3; k++) {
      this.coxA[k] = Math.exp(-dt / taus[k]);
      this.coxS[k] = Math.sqrt(1 - this.coxA[k] * this.coxA[k]);
    }
    this.coxDt = dt;
    this.windAvg = -1;
    this.resA = 1 - Math.exp((-TWO_PI * Math.min(2500, 0.2 * sr)) / sr);
    this.statsEvery = Math.round(sr / 4);
    this.refArea = 0.393 * (0.002 / V2) * sr;
    this.needPrefill = c.prefill !== false;

    // Drips: integrate-and-fire points
    const dr = c.drip || {};
    const points = clamp(Math.round(num(dr.points, 8)), 1, 64);
    const rTip = clamp(num(dr.tipRadiusMm, 1.5), 0.2, 5) * 1e-3;
    let vol = (0.6 * 2 * Math.PI * rTip * 0.0728) / (998 * 9.81);
    let D = 1000 * Math.cbrt((6 * vol) / Math.PI);
    if (D < 3 || D > 5.5) {
      D = clamp(D, 3, 5.5);
      vol = (Math.PI / 6) * Math.pow(D * 1e-3, 3);
    }
    const fall = Math.max(0.02, num(dr.fallHeight, 2));
    const vt = Math.max(0.1, 9.65 - 10.3 * Math.exp(-0.6 * D));
    const w = new Float64Array(points);
    let ws = 0;
    for (let i = 0; i < points; i++) {
      w[i] = Math.exp(0.5 * this.rng.normal());
      ws += w[i];
    }
    for (let i = 0; i < points; i++) w[i] /= ws;
    const vd = new Float64Array(points).fill(vol);
    this.drip = {
      points,
      catchment: Math.max(0.0005, num(dr.catchment, 1)),
      tau: Math.max(1, num(dr.tau, 60)),
      vol,
      D,
      vImpact: vt * Math.sqrt(1 - Math.exp((-2 * 9.81 * fall) / (vt * vt))),
      w,
      acc: new Float64Array(points),
      vd,
    };

    // Flow: gutters and downspouts
    const fo = c.flow || {};
    const pipeL = Math.max(0, num(fo.pipeLength, 0));
    const ffall = Math.max(0.05, num(fo.fallHeight, 0.3));
    this.flowCfg = {
      catchment: Math.max(0.01, num(fo.catchment, 20)),
      tau: Math.max(2, num(fo.tau, 45)),
      pipeLength: pipeL,
    };
    this.flowDripVol = vol;
    this.flowDripD = D;
    this.flowFallV = Math.sqrt(2 * 9.81 * ffall);
    this.pipe = [];
    if (this.kind === 2) {
      if (pipeL > 0.2) {
        for (let n = 1; n <= 4; n++) {
          const bq = new Float64Array(7);
          setBandpass(bq, (n * 343) / (2 * pipeL), 9, sr);
          this.pipe.push(bq);
        }
      } else {
        const a = new Float64Array(7);
        setBandpass(a, 620, 2.2, sr);
        const b2 = new Float64Array(7);
        setBandpass(b2, 1550, 2.5, sr);
        this.pipe.push(a, b2);
      }
      setBandpass(this.flowBp, 1100, 0.6, sr);
    }
    this.quiet = 0;
  }

  onMessage(d) {
    if (!d || typeof d !== 'object') return;
    switch (d.type) {
      case 'config':
        this.configure({ ...this.lastCfg, ...(d.config || {}) });
        break;
      case 'exposure':
        this.exposure = clamp(num(d.value, this.exposure), 0, 4);
        this.lastCfg = { ...this.lastCfg, exposure: this.exposure };
        break;
      case 'area':
        this.area = Math.max(0, num(d.value, this.area));
        this.lastCfg = { ...this.lastCfg, area: this.area };
        break;
      case 'stats':
        this.statsOn = !!d.on;
        break;
      case 'reset':
        this.qN = 0;
        this.bOn.fill(0);
        this.activeBubbles = 0;
        this.ringW.fill(0);
        for (const m of this.mats) {
          m.ringA.fill(0);
          m.ringB.fill(0);
          m.ringN.fill(0);
          m.silence();
        }
        break;
      default:
        break;
    }
  }

  pickMat() {
    const u = this.rng.next();
    const cum = this.cum;
    for (let i = 0; i < cum.length; i++) if (u < cum[i]) return i;
    return cum.length - 1;
  }

  fallSpeed(c, kind, D) {
    if (c === 5 && kind === 1) return 11.45 * Math.sqrt(D / 10);
    if (kind === 2) return 0.85 + 0.3 * this.rng.next();
    if (kind === 3) return 1.4 + 0.3 * D;
    return liquidFallSpeed(D);
  }

  enqueue(tAbs, D, V, c, kind, mi, micro) {
    if (this.qN >= QCAP) return;
    const q = this.qN++;
    this.qT[q] = tAbs;
    this.qD[q] = D;
    this.qV[q] = V;
    this.qC[q] = c;
    this.qK[q] = kind;
    this.qM[q] = mi;
    this.qMicro[q] = micro ? 1 : 0;
  }

  drainQueue(N) {
    const end = this.abs + N;
    for (let q = 0; q < this.qN; q++) {
      if (this.qT[q] >= end) continue;
      const off = Math.max(0, Math.min(N - 1, Math.floor(this.qT[q] - this.abs)));
      let mi = this.qM[q];
      if (mi >= this.mats.length) mi = -2;
      this.impact(off, this.qC[q], this.qD[q], this.qV[q], this.qK[q], mi, this.qMicro[q] === 1);
      const last = --this.qN;
      if (q !== last) {
        this.qT[q] = this.qT[last];
        this.qD[q] = this.qD[last];
        this.qV[q] = this.qV[last];
        this.qC[q] = this.qC[last];
        this.qK[q] = this.qK[last];
        this.qM[q] = this.qM[last];
        this.qMicro[q] = this.qMicro[last];
        q--;
      }
    }
  }

  impact(off, c, D, V, kind, matIdx, micro) {
    this.blockEvents++;
    this.evCount++;
    let mi = matIdx;
    if (mi === -2) mi = this.pickMat();
    if (kind === 2) {
      const m = this.mats[mi < 0 ? 0 : mi];
      if (m.kind !== 1) this.noiseBurst(m, off, 0.02 * (D / 3), 0.0025);
      return;
    }
    let water = mi < 0;
    if (!water) {
      const m = this.mats[mi];
      if (m.kind === 1) water = true;
      else if (m.puddleable && this.puddleCover > 0 && this.rng.next() < this.puddleCover) water = true;
    }
    if (water) {
      const depth = mi >= 0 && this.mats[mi].kind === 1 ? this.mats[mi].depthMm : Math.max(this.puddleDepth, mi < 0 ? 20 : 0);
      this.waterImpact(off, c, D, V, kind, depth, micro);
    } else {
      this.solidImpact(this.mats[mi], mi, off, c, D, V, kind, micro);
    }
  }

  solidImpact(m, mi, off, c, D, V, kind, micro) {
    const sr = this.sr;
    const rng = this.rng;
    let A;
    let L;
    let hertz = false;
    if (kind === 1 || kind === 3) {
      hertz = true;
      const Dm = D * 1e-3;
      const R = Dm / 2;
      const mass = ((917 * Math.PI) / 6) * Dm * Dm * Dm;
      const Es = 9e9 * m.hardness * m.hardness + 2e7;
      let tc = 2.87 * Math.pow((mass * mass) / (R * Es * Es * Math.max(0.1, V)), 0.2);
      if (kind === 3) tc *= 4;
      L = clamp(Math.round(tc * sr), 2, 96);
      const J = 1.3 * mass * V;
      let rel = (J * sr) / F2MM_N / (1.6 * this.refArea);
      // Level compression above a 2 mm raindrop: hail would otherwise sit ~+50 dB over rain.
      if (rel > 1) rel = Math.pow(rel, 0.55);
      if (kind === 3) rel *= 0.35;
      A = (rel * this.refArea) / (0.6366 * L);
    } else {
      A = (D * D * V * V) / FREF;
      L = clamp(Math.round(((D * 1e-3) / Math.max(0.05, V)) * sr), 2, 480);
    }
    const w = 0.5 + m.spatial * (rng.next() - 0.5);
    const wa = A * w;
    const wb = A - wa;
    const base = this.pos + off;
    const rA = m.ringA;
    const rB = m.ringB;
    if (hertz) {
      for (let i = 0; i < L; i++) {
        const s = Math.sin((Math.PI * (i + 0.5)) / L);
        const j = (base + i) & RMASK;
        rA[j] += wa * s;
        rB[j] += wb * s;
      }
    } else {
      const ip = Math.max(1, Math.round(0.18 * L));
      const inv = 1 / Math.max(1, L - ip);
      for (let i = 1; i < L; i++) {
        let s;
        if (i <= ip) s = Math.sqrt(i / ip);
        else {
          const u = 1 - (i - ip) * inv;
          s = u * u;
        }
        const j = (base + i) & RMASK;
        rA[j] += wa * s;
        rB[j] += wb * s;
      }
    }
    const nAmp = A * (m.noise + 0.25 * this.wet * (m.kind === 0 ? 1 : 0.5));
    if (nAmp > 1e-6) this.noiseBurst(m, off, nAmp, m.noiseDecay * Math.sqrt(Math.max(0.3, D) / 2) * (1 + this.wet));
    if (!micro && kind === 0 && m.splash > 0 && D > 0.8) {
      const h = 0.5 * D;
      const k = rng.poisson(m.splash * 1.6 * h * Math.sqrt(h) * (1 + 0.5 * this.wet));
      for (let s = 0; s < k; s++) {
        const delay = 0.005 * Math.exp(2.302585092994046 * rng.next());
        this.enqueue(this.abs + off + delay * sr, 0.25 + 0.5 * rng.next(), 1 + 2 * rng.next(), c, 0, mi, true);
      }
    }
    if (hertz && !micro && kind === 1 && D > 4) {
      const e = 0.3 + 0.2 * rng.next();
      const vb = V * e;
      const tFly = Math.min(0.6, (vb / 9.81) * (0.5 + rng.next()));
      this.enqueue(this.abs + off + tFly * sr, D * 0.9, vb, c, kind, mi, true);
    }
  }

  noiseBurst(m, off, amp, decay) {
    const tau = Math.max(0.0003, decay) * this.sr;
    const Ln = Math.min(480, Math.ceil(tau * 4));
    const k = Math.exp(-1 / tau);
    let e = amp;
    const j0 = (this.pos + off) & RMASK;
    const n1 = Math.min(Ln, RING - j0);
    const r = m.ringN;
    for (let i = 0; i < n1; i++) {
      r[j0 + i] += e;
      e *= k;
    }
    for (let i = 0; i < Ln - n1; i++) {
      r[i] += e;
      e *= k;
    }
  }

  waterImpact(off, c, D, V, kind, depth, micro) {
    const sr = this.sr;
    const rng = this.rng;
    const L = clamp(Math.round(sr * (0.00005 + 0.000025 * D)), 2, 16);
    let A = (D * D * V * V) / FREF;
    if (kind === 1) A = Math.min(40, A * 3);
    const base = this.pos + off;
    const r = this.ringW;
    for (let i = 0; i < L; i++) r[(base + i) & RMASK] += A * Math.sin((Math.PI * (i + 0.5)) / L);
    let p = micro ? 0.2 : P_ENTRAIN[c] || 0;
    if (c === 1 && !micro) p *= 1 - smooth(6, 10, this.wind);
    p *= smooth(1.5, 6, depth) * this.waterQuality;
    if (p > 0 && rng.next() < p) {
      const start = this.abs + off + Math.round(sr * (0.0003 + 0.0012 * rng.next()));
      const cc = micro ? 1 : c;
      const rmm = rng.powerLaw(BUB_RMIN[cc], BUB_RMAX[cc], BUB_ALPHA[cc]);
      const mid = 0.5 * (BUB_RMIN[cc] + BUB_RMAX[cc]);
      const amp = (BUB_AMP[cc] || 0.1) * (micro ? 0.3 : 1) * Math.exp(0.35 * rng.normal()) * Math.sqrt(rmm / mid);
      this.spawnBubble(start, rmm, amp * this.bubbleGain);
    }
    if (!micro && kind === 0 && D > 2.4) {
      const k = rng.poisson(1.5 * (D / 3) * (D / 3));
      for (let s = 0; s < k; s++) {
        this.enqueue(this.abs + off + (0.02 + 0.07 * rng.next()) * sr, 0.6 + 0.6 * rng.next(), 1 + 1.5 * rng.next(), 1, 0, -1, true);
      }
    }
  }

  spawnBubble(startAbs, rmm, amp) {
    const f0 = 3.28 / (rmm * 1e-3);
    if (f0 > 0.45 * this.sr || f0 < 60 || !(amp > 0)) return;
    const beta = 0.043 * f0 + 0.0014 * Math.pow(f0, 1.5);
    let v = -1;
    for (let i = 0; i < this.maxBub; i++) {
      if (!this.bOn[i]) {
        v = i;
        break;
      }
    }
    if (v < 0) {
      let best = Infinity;
      for (let i = 0; i < this.maxBub; i++) {
        const e = this.bAmp[i] * this.bEnv[i];
        if (e < best) {
          best = e;
          v = i;
        }
      }
      if (best > amp) return;
    } else {
      this.activeBubbles++;
    }
    const k = TWO_PI / this.sr;
    const df = (f0 * 0.1 * beta) / this.sr;
    this.bOn[v] = 1;
    this.bStart[v] = startAbs;
    this.bZr[v] = 1;
    this.bZi[v] = 0;
    this.bWr[v] = Math.cos(k * f0);
    this.bWi[v] = Math.sin(k * f0);
    this.bDr[v] = Math.cos(k * df);
    this.bDi[v] = Math.sin(k * df);
    this.bF[v] = f0;
    this.bDf[v] = df;
    this.bEnv[v] = 1;
    this.bDec[v] = Math.exp(-beta / this.sr);
    this.bAmp[v] = amp;
  }

  genRain(N, intensity, wind) {
    const sr = this.sr;
    const dt = N / sr;
    const rng = this.rng;
    const cx = this.cx;
    for (let k = 0; k < 3; k++) cx[k] = cx[k] * this.coxA[k] + this.coxS[k] * rng.normal();
    const t = typeof currentTime === 'number' ? currentTime : this.abs / sr;
    const slow = 0.6 * vnoise(this.sharedKey, t / 6.4) + 0.4 * vnoise((this.sharedKey ^ 0x9e3779b9) >>> 0, t / 25.6);
    const z = 0.45 * cx[0] + 0.4 * cx[1] + 0.35 * cx[2] + 0.55 * slow;
    const sd = this.coxDepth;
    const cox = Math.exp(sd * z - 0.35 * sd * sd);
    if (this.windAvg < 0) this.windAvg = wind;
    else this.windAvg += (wind - this.windAvg) * (1 - Math.exp(-dt / 30));
    const ratio = this.windAvg > 0.5 ? wind / this.windAvg : 1;
    const gm = clamp(1 + 0.35 * (ratio - 1), 0.5, 2);
    const scale = this.area * this.exposure * intensity * cox * gm;
    let total = 0;
    for (let c = 0; c < 6; c++) {
      const l = Math.max(0, this.rates[c]) * scale;
      this.lam[c] = l;
      total += l;
    }
    this.physRate = total;
    if (!(total > 0)) return;
    const keep = total > this.maxEv ? this.maxEv / total : 1;
    let resid = 0;
    for (let c = 0; c < 6; c++) {
      const lam = this.lam[c];
      if (!(lam > 0)) continue;
      const kind = this.kinds[c];
      const lo = this.lo[c];
      const hi = this.hi[c];
      const dMean = this.dpar[c] > 0 ? this.dpar[c] : 0.5 * (lo + hi);
      if (keep < 1 && kind !== 2) {
        const v = this.fallSpeed(c, kind, dMean);
        const A = Math.min(40, (dMean * dMean * v * v) / FREF);
        const Lp = clamp(((dMean * 1e-3) / v) * sr, 2, 480);
        resid += ((lam * (1 - keep)) / sr) * A * A * 0.25 * Lp;
      }
      const n = rng.poisson(lam * keep * dt);
      for (let e = 0; e < n; e++) {
        const off = Math.floor(rng.next() * N);
        let D = dMean * Math.exp(0.12 * rng.normal());
        if (c < 5) D = clamp(D, lo, hi);
        else D = Math.max(1, D);
        this.impact(off, c, D, this.fallSpeed(c, kind, D), kind, -2, false);
      }
    }
    if (resid > 0) {
      let waterShare = 0;
      for (const m of this.mats) waterShare += m.kind === 1 ? m.fraction : m.puddleable ? m.fraction * this.puddleCover : 0;
      this.residVar = resid;
      this.residWater = resid * waterShare * 0.5;
    }
  }

  genDrips(N, inflowMmh) {
    const d = this.drip;
    const sr = this.sr;
    const dt = N / sr;
    const rng = this.rng;
    const inflow = (inflowMmh / 3.6e6) * d.catchment;
    if (this.needPrefill) {
      this.store = inflow * d.tau;
      for (let i = 0; i < d.points; i++) d.acc[i] = rng.next() * d.vd[i];
      this.needPrefill = false;
    }
    this.store += inflow * dt;
    const out = this.store * (1 - Math.exp(-dt / d.tau));
    this.store -= out;
    let phys = 0;
    for (let i = 0; i < d.points; i++) {
      d.acc[i] += out * d.w[i];
      const ratePt = ((this.store / d.tau) * d.w[i]) / d.vol;
      phys += ratePt;
      let fired = 0;
      while (d.acc[i] >= d.vd[i] && fired < 3) {
        d.acc[i] -= d.vd[i];
        fired++;
        const jit = 0.04 + 0.3 * smooth(1, 8, ratePt);
        d.vd[i] = d.vol * Math.max(0.3, 1 + jit * rng.normal());
        const off = Math.floor(rng.next() * N);
        this.impact(off, 6, d.D * Math.exp(0.04 * rng.normal()), d.vImpact * (1 + 0.03 * rng.normal()), 0, -2, false);
      }
      if (d.acc[i] > 3 * d.vd[i]) {
        this.streamExcess += d.acc[i] - d.vd[i];
        d.acc[i] = 0.5 * d.vd[i];
      }
    }
    this.physRate = phys;
    if (this.streamExcess > 0) {
      const trickle = this.streamExcess / dt;
      this.streamExcess = 0;
      const n = rng.poisson(Math.min(400, trickle * 4e6) * dt);
      for (let k = 0; k < n; k++) {
        this.spawnBubble(this.abs + Math.floor(rng.next() * N), rng.powerLaw(0.5, 3, 2.4), 0.2 * Math.exp(0.4 * rng.normal()));
      }
    }
  }

  genFlow(N, inflowMmh) {
    const fl = this.flowCfg;
    const sr = this.sr;
    const dt = N / sr;
    const rng = this.rng;
    const inflow = (inflowMmh / 3.6e6) * fl.catchment;
    if (this.needPrefill) {
      this.store = inflow * fl.tau;
      this.needPrefill = false;
    }
    this.store += inflow * dt;
    const out = this.store * (1 - Math.exp(-dt / fl.tau));
    this.store -= out;
    const Q = out / dt;
    const qLs = Q * 1000;
    if (qLs < 0.004) {
      this.flowLevel = 0;
      this.dribAcc += out;
      this.physRate = Q / this.flowDripVol;
      if (this.dribAcc >= this.flowDripVol) {
        this.dribAcc -= this.flowDripVol;
        this.impact(Math.floor(rng.next() * N), 6, this.flowDripD, Math.min(this.flowFallV, 9), 0, -2, false);
      }
      return;
    }
    const rate = Math.min(1500, 700 * Math.pow(qLs, 0.7));
    this.physRate = rate;
    const n = rng.poisson(rate * dt);
    for (let k = 0; k < n; k++) {
      this.spawnBubble(this.abs + Math.floor(rng.next() * N), rng.powerLaw(0.5, 5, 2.6), 0.25 * Math.exp(0.4 * rng.normal()));
    }
    const nImp = rng.poisson(Math.min(300, 120 * qLs) * dt);
    for (let k = 0; k < nImp; k++) {
      this.impact(Math.floor(rng.next() * N), 4, 3 + 2 * rng.next(), Math.min(this.flowFallV, 9), 0, -2, true);
    }
    this.flowLevel = 0.04 * Math.sqrt(qLs);
    this.blockEvents++;
  }

  process(inputs, outputs, parameters) {
    const out = outputs[0];
    if (!out || !out[0]) return true;
    const ch0 = out[0];
    const N = ch0.length;
    if (N > this.mixBuf.length) this._alloc(N);
    const sr = this.sr;
    const P = parameters;
    const S = BUSES.get(this.busId);
    if (S) {
      for (let c = 0; c < 6; c++) {
        this.rates[c] = S[c];
        this.dpar[c] = S[6 + c];
      }
    } else {
      this.rates.fill(0);
      this.dpar.fill(0);
    }
    const intensity = Math.max(0, P.intensity ? P.intensity[0] : 0);
    const wind = Math.max(0, P.wind ? P.wind[0] : 0);
    const gain = Math.max(0, P.gain ? P.gain[0] : 1);
    const wet = S ? clamp(S[12] * this.wetMul, 0, 1) : 0;
    const puddle = S ? Math.max(0, S[13] * this.puddleMul) : 0;
    const flow = S ? Math.max(0, S[14] * this.flowMul) : 0;
    this.wind = wind;
    this.wet = wet;
    this.puddleDepth = puddle;
    this.puddleCover = puddle > 0.05 ? 1 - Math.exp(-puddle / 6) : 0;
    const mats = this.mats;
    const nMats = mats.length;
    for (let mi = 0; mi < nMats; mi++) {
      const m = mats[mi];
      if (Math.abs(m.wetSet - wet) > 0.02) m.update(wet, sr);
      m.share = m.kind === 1 ? 0 : m.puddleable ? m.fraction * (1 - this.puddleCover) : m.fraction;
    }
    this.blockEvents = 0;
    this.residVar = 0;
    this.residWater = 0;
    if (this.kind === 0) this.genRain(N, intensity, wind);
    else if (this.kind === 1) this.genDrips(N, flow * intensity);
    else this.genFlow(N, flow * intensity);
    this.drainQueue(N);

    const idle = this.blockEvents === 0 && this.activeBubbles === 0 && this.residVar <= 0 && this.flowLevel <= 0;
    if (idle && this.quiet > 12) {
      for (let c = 0; c < out.length; c++) out[c].fill(0);
      this.pos = (this.pos + N) & RMASK;
      this.abs += N;
      this.postStats(N);
      return true;
    }

    const acc = this.acc;
    const mix = this.mixBuf;
    mix.fill(0, 0, N);
    let ns = this.ns;
    const base = this.pos;
    // The rings are read in place when this quantum does not wrap around them (always, for
    // 128-frame quanta), and cleared with one fill afterwards.
    const direct = base + N <= RING;

    for (let mi = 0; mi < nMats; mi++) {
      const m = mats[mi];
      if (m.kind === 1) continue;
      let xa;
      let xb;
      let xn;
      let o;
      if (direct) {
        xa = m.ringA;
        xb = m.ringB;
        xn = m.ringN;
        o = base;
      } else {
        xa = this.exA;
        xb = this.exB;
        xn = this.exN;
        o = 0;
        const rA = m.ringA;
        const rB = m.ringB;
        const rNz = m.ringN;
        for (let i = 0; i < N; i++) {
          const j = (base + i) & RMASK;
          xa[i] = rA[j];
          rA[j] = 0;
          xb[i] = rB[j];
          rB[j] = 0;
          xn[i] = rNz[j];
          rNz[j] = 0;
        }
      }
      // Events above maxEventsPerSec: variance-matched low-passed noise on both buses.
      const rv = this.residVar * m.share;
      if (rv > 0) {
        const a = this.resA;
        const s = Math.sqrt((rv * (2 - a)) / a) * 1.732;
        let lp = m.resLp;
        for (let i = 0; i < N; i++) {
          ns ^= ns << 13;
          ns ^= ns >>> 17;
          ns ^= ns << 5;
          lp += a * (s * (ns / 2147483648) - lp);
          xa[o + i] += 0.5 * lp;
          xb[o + i] += 0.5 * lp;
        }
        m.resLp = lp;
      }
      // Modal bank: even modes ride bus A, odd modes bus B; two modes per pass over the block.
      acc.fill(0, 0, N);
      const nm = m.n;
      const C1 = m.c1;
      const C2 = m.c2;
      const B = m.b;
      const Y1 = m.y1;
      const Y2 = m.y2;
      let k = 0;
      for (; k + 1 < nm; k += 2) {
        const a1 = C1[k];
        const a2 = C2[k];
        const ab = B[k];
        const b1 = C1[k + 1];
        const b2 = C2[k + 1];
        const bb = B[k + 1];
        let p1 = Y1[k];
        let p2 = Y2[k];
        let q1 = Y1[k + 1];
        let q2 = Y2[k + 1];
        for (let i = 0; i < N; i++) {
          const ya = a1 * p1 - a2 * p2 + ab * xa[o + i];
          const yb = b1 * q1 - b2 * q2 + bb * xb[o + i];
          p2 = p1;
          p1 = ya;
          q2 = q1;
          q1 = yb;
          acc[i] += ya + yb;
        }
        Y1[k] = p1 < 1e-18 && p1 > -1e-18 ? 0 : p1;
        Y2[k] = p2 < 1e-18 && p2 > -1e-18 ? 0 : p2;
        Y1[k + 1] = q1 < 1e-18 && q1 > -1e-18 ? 0 : q1;
        Y2[k + 1] = q2 < 1e-18 && q2 > -1e-18 ? 0 : q2;
      }
      if (k < nm) {
        const a1 = C1[k];
        const a2 = C2[k];
        const ab = B[k];
        let p1 = Y1[k];
        let p2 = Y2[k];
        for (let i = 0; i < N; i++) {
          const ya = a1 * p1 - a2 * p2 + ab * xa[o + i];
          p2 = p1;
          p1 = ya;
          acc[i] += ya;
        }
        Y1[k] = p1 < 1e-18 && p1 > -1e-18 ? 0 : p1;
        Y2[k] = p2 < 1e-18 && p2 > -1e-18 ? 0 : p2;
      }
      // Directly radiated click (dF/dt), wet-slap / rustle noise bursts, material low-pass.
      const cg = m.click * (1 - 0.45 * wet);
      let prev = m.prevF;
      let lpY = m.lpY;
      const lpA = m.lpA;
      for (let i = 0; i < N; i++) {
        const F = xa[o + i] + xb[o + i];
        let y = acc[i] + cg * (F - prev);
        prev = F;
        const e = xn[o + i];
        if (e !== 0) {
          ns ^= ns << 13;
          ns ^= ns >>> 17;
          ns ^= ns << 5;
          y += e * (ns / 2147483648);
        }
        if (lpA > 0) {
          lpY += lpA * (y - lpY);
          y = lpY;
        }
        mix[i] += y;
      }
      m.prevF = prev;
      m.lpY = lpY < 1e-18 && lpY > -1e-18 ? 0 : lpY;
      if (direct) {
        xa.fill(0, o, o + N);
        xb.fill(0, o, o + N);
        xn.fill(0, o, o + N);
      }
    }

    // Water clicks (water materials and puddled patches share one radiating surface).
    {
      const rW = this.ringW;
      let wPrev = this.wPrev;
      const wc = this.wClick;
      if (direct) {
        for (let i = 0; i < N; i++) {
          const F = rW[base + i];
          mix[i] += wc * (F - wPrev);
          wPrev = F;
        }
        rW.fill(0, base, base + N);
      } else {
        for (let i = 0; i < N; i++) {
          const j = (base + i) & RMASK;
          const F = rW[j];
          rW[j] = 0;
          mix[i] += wc * (F - wPrev);
          wPrev = F;
        }
      }
      this.wPrev = wPrev;
      if (this.residWater > 0) {
        const s = Math.sqrt(this.residWater) * 1.2;
        let last = this.wResLast;
        for (let i = 0; i < N; i++) {
          ns ^= ns << 13;
          ns ^= ns >>> 17;
          ns ^= ns << 5;
          const w = ns / 2147483648;
          mix[i] += s * (w - last);
          last = w;
        }
        this.wResLast = last;
      }
    }

    // Minnaert bubbles with rising pitch: sin(phase) from a rotating phasor (no per-sample sin).
    if (this.activeBubbles > 0) {
      const absStart = this.abs;
      let active = 0;
      for (let v = 0; v < this.maxBub; v++) {
        if (!this.bOn[v]) continue;
        let i0 = this.bStart[v] - absStart;
        if (i0 >= N) {
          active++;
          continue;
        }
        if (i0 < 0) i0 = 0;
        i0 = Math.floor(i0);
        let zr = this.bZr[v];
        let zi = this.bZi[v];
        let wr = this.bWr[v];
        let wi = this.bWi[v];
        const dr = this.bDr[v];
        const di = this.bDi[v];
        let env = this.bEnv[v];
        const dec = this.bDec[v];
        const amp = this.bAmp[v];
        for (let i = i0; i < N; i++) {
          const t = zr * wr - zi * wi;
          zi = zr * wi + zi * wr;
          zr = t;
          const u = wr * dr - wi * di;
          wi = wr * di + wi * dr;
          wr = u;
          mix[i] += amp * env * zi;
          env *= dec;
        }
        const f = this.bF[v] + this.bDf[v] * (N - i0);
        if (env < 1e-4 || f > 0.48 * sr) {
          this.bOn[v] = 0;
        } else {
          const gz = 1 / Math.sqrt(zr * zr + zi * zi);
          const gw = 1 / Math.sqrt(wr * wr + wi * wi);
          this.bZr[v] = zr * gz;
          this.bZi[v] = zi * gz;
          this.bWr[v] = wr * gw;
          this.bWi[v] = wi * gw;
          this.bF[v] = f;
          this.bEnv[v] = env;
          active++;
        }
      }
      this.activeBubbles = active;
    }

    // Gutter / downspout: turbulent noise and open-pipe resonances.
    if (this.kind === 2) {
      const bp = this.flowBp;
      const lvl = this.flowLevel;
      if (lvl > 0) {
        for (let i = 0; i < N; i++) {
          ns ^= ns << 13;
          ns ^= ns >>> 17;
          ns ^= ns << 5;
          const x = lvl * (ns / 2147483648);
          const y = bp[0] * x + bp[5];
          bp[5] = bp[1] * x - bp[3] * y + bp[6];
          bp[6] = bp[2] * x - bp[4] * y;
          mix[i] += y * 3;
        }
      }
      if (this.pipe.length) {
        for (let i = 0; i < N; i++) {
          const x = mix[i];
          let s = 0;
          for (let p = 0; p < this.pipe.length; p++) {
            const q = this.pipe[p];
            const y = q[0] * x + q[5];
            q[5] = q[1] * x - q[3] * y + q[6];
            q[6] = q[2] * x - q[4] * y;
            s += y;
          }
          mix[i] = 0.6 * x + 0.8 * s;
        }
        // Resonators ringing down on silence would otherwise decay into subnormal numbers.
        for (let p = 0; p < this.pipe.length; p++) {
          const q = this.pipe[p];
          if (q[5] < 1e-20 && q[5] > -1e-20) q[5] = 0;
          if (q[6] < 1e-20 && q[6] > -1e-20) q[6] = 0;
        }
      }
    }

    this.ns = ns;
    const g = gain * OUT_SCALE;
    let x1 = this.dcX;
    let y1 = this.dcY;
    let energy = 0;
    for (let i = 0; i < N; i++) {
      const x = mix[i] * g;
      const y = x - x1 + 0.9985 * y1;
      x1 = x;
      y1 = y;
      ch0[i] = y;
      energy += y * y;
    }
    this.dcX = x1;
    this.dcY = y1 < 1e-20 && y1 > -1e-20 ? 0 : y1;
    for (let c = 1; c < out.length; c++) out[c].set(ch0);

    if (idle && energy < 1e-13) {
      this.quiet++;
      if (this.quiet === 12) {
        for (let mi = 0; mi < nMats; mi++) mats[mi].silence();
        this.dcX = 0;
        this.dcY = 0;
        this.wPrev = 0;
      }
    } else {
      this.quiet = 0;
    }
    this.pos = (this.pos + N) & RMASK;
    this.abs += N;
    this.postStats(N);
    return true;
  }

  postStats(N) {
    this.statCount += N;
    if (this.statCount < this.statsEvery) return;
    const secs = this.statCount / this.sr;
    if (this.statsOn) {
      try {
        this.port.postMessage({ type: 'stats', id: this.id, ratePerSec: this.physRate, rendered: this.evCount / secs });
      } catch (e) {
        this.statsOn = false;
      }
    }
    this.statCount = 0;
    this.evCount = 0;
  }
}

/**
 * rg-weather: the weather bus. Its k-rate parameters are the drop-class impact rates r0..r5
 * (m^-2 s^-1), the class mean diameters d0..d5 (mm), film wetness (0-1), puddle depth (mm)
 * and the liquid rain rate (mm/h) feeding drips and gutters. Each quantum it publishes them
 * for every rg-rain-surface node on the same bus; it outputs silence.
 */
class WeatherBusProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    const d = [];
    for (let i = 0; i < 6; i++) d.push({ name: RN[i], defaultValue: 0, minValue: 0, maxValue: 1e7, automationRate: 'k-rate' });
    for (let i = 0; i < 6; i++) d.push({ name: DN[i], defaultValue: 0, minValue: 0, maxValue: 100, automationRate: 'k-rate' });
    d.push({ name: 'wetness', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' });
    d.push({ name: 'puddle', defaultValue: 0, minValue: 0, maxValue: 2000, automationRate: 'k-rate' });
    d.push({ name: 'liquid', defaultValue: 0, minValue: 0, maxValue: 2000, automationRate: 'k-rate' });
    return d;
  }

  constructor(options) {
    super();
    const po = (options && options.processorOptions) || {};
    this.busId = String(po.bus ?? 'main');
    this.state = new Float64Array(BUS_SIZE);
    this.alive = true;
    BUSES.set(this.busId, this.state);
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'dispose') {
        this.alive = false;
        if (BUSES.get(this.busId) === this.state) BUSES.delete(this.busId);
      }
    };
  }

  process(inputs, outputs, parameters) {
    const s = this.state;
    for (let i = 0; i < 6; i++) {
      const r = parameters[RN[i]];
      const d = parameters[DN[i]];
      s[i] = r && r.length ? r[0] : 0;
      s[6 + i] = d && d.length ? d[0] : 0;
    }
    s[12] = parameters.wetness ? parameters.wetness[0] : 0;
    s[13] = parameters.puddle ? parameters.puddle[0] : 0;
    s[14] = parameters.liquid ? parameters.liquid[0] : 0;
    if (BUSES.get(this.busId) !== s && this.alive) BUSES.set(this.busId, s);
    return this.alive;
  }
}

registerProcessor('rg-rain-surface', RainSurfaceProcessor);
registerProcessor('rg-weather', WeatherBusProcessor);
