/**
 * Weather physics for the browser engine, matching the raingen Python reference engine.
 *
 * - Fall speed: Atlas, Srivastava & Sekhon (1973), v = 9.65 - 10.3 exp(-0.6 D) m/s from 0.5 mm,
 *   Stokes drag below 0.1 mm and a linear bridge between. Hail: Matson & Huggins (1980),
 *   v = 11.45 sqrt(D_cm) m/s.
 * - Drop-size distribution: normalized gamma (Testud et al. 2001; Bringi et al. 2003),
 *   N(D) = Nw f(mu) (D/Dm)^mu exp(-(4+mu) D/Dm); Marshall & Palmer (1948) as an option.
 *   Rain rate R = 6 pi 1e-4 ∫ D^3 v(D) N(D) dD (mm/h, D in mm); Nw closes the rate.
 * - Acoustic drop classes: Medwin, Nystuen et al. (1992) boundaries 0.8/1.1/2.2/3.5 mm
 *   (or Nystuen 0.8/1.2/2.0/3.5 mm). Class impact flux = ∫ N(D) v(D) dD (m^-2 s^-1).
 * - Drop impact force: Gordillo, Sun & Cheng (2018), F_max ~ 0.83 rho D^2 V^2, width ~ D/V.
 * - Wind: log law u(z) = u10 ln(z/z0)/ln(10/z0); Kaimal spectrum peak n = 0.045 U/z,
 *   gust process normalized to the turbulence intensity (Simiu & Scanlan).
 * - Air absorption: ISO 9613-1:1993 (pure-tone atmospheric absorption).
 * - Drips: Tate's law m g = 2 pi r sigma with Harkins-Brown factor ~0.6.
 *
 * All randomness is counter-based (splitmix/lowbias32-style hashing keyed by seed,
 * layer name and counter), so the scene seed reproduces every draw on every platform.
 */

export const RHO_WATER = 998.0;
export const RHO_ICE = 917.0;
export const SIGMA_WATER = 0.0728;
export const G = 9.81;
export const P0 = 101325.0;

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const lerp = (a, b, t) => a + (b - a) * t;
export function smoothstep(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0 || 1e-9), 0, 1);
  return t * t * (3 - 2 * t);
}
const num = (x, d) => {
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
};

// ---------------------------------------------------------------------------
// Deterministic counter-based random numbers
// ---------------------------------------------------------------------------

/** 32-bit integer finalizer (lowbias32, C. Wellons); a bijective avalanche mix. */
export function mix32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

/** Stable 32-bit hash of a string (FNV-1a followed by mix32). */
export function hashString(s) {
  const str = String(s);
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return mix32(h);
}

/** Fold any seed (number, bigint, hex/decimal string, other string) into a 32-bit key. */
export function seedKey(seed) {
  let lo = 0;
  let hi = 0;
  if (typeof seed === 'bigint') {
    const s = seed < 0n ? -seed : seed;
    lo = Number(s & 0xffffffffn);
    hi = Number((s >> 32n) & 0xffffffffn) ^ Number((s >> 64n) & 0xffffffffn) ^ Number((s >> 96n) & 0xffffffffn);
  } else if (typeof seed === 'string') {
    const t = seed.trim();
    if (/^0x[0-9a-f]+$/i.test(t)) {
      const hex = t.slice(2).replace(/^0+/, '') || '0';
      lo = parseInt(hex.slice(-8), 16) >>> 0;
      const rest = hex.slice(0, -8);
      for (let i = rest.length; i > 0; i -= 8) hi ^= parseInt(rest.slice(Math.max(0, i - 8), i), 16) >>> 0;
    } else if (/^\d+$/.test(t)) {
      return seedKey(t.length > 15 ? BigInt(t) : Number(t));
    } else {
      return hashString(t);
    }
  } else {
    const n = Math.abs(num(seed, 0));
    lo = n % 4294967296 >>> 0;
    hi = Math.floor(n / 4294967296) >>> 0;
  }
  return mix32((lo ^ mix32((hi + 0x9e3779b9) >>> 0)) >>> 0);
}

/**
 * Counter-based RNG: value(i) = mix(mix(i ^ k0) + k1), with (k0, k1) derived from
 * (seed, layer, block). Independent of creation order; same inputs give the same stream.
 */
export class CounterRng {
  constructor(seed = 0, layer = '', block = 0) {
    const sk = seedKey(seed);
    const lk = hashString(layer);
    const b = Math.abs(num(block, 0));
    this.k0 = mix32((sk ^ lk) >>> 0);
    this.k1 = mix32((this.k0 ^ mix32(((b >>> 0) + 0x632be5ab) >>> 0) ^ (Math.floor(b / 4294967296) >>> 0)) >>> 0);
    this.c = 0;
    this._spare = NaN;
  }

  u32() {
    this.c = (this.c + 1) >>> 0;
    return mix32((mix32((this.c ^ this.k0) >>> 0) + this.k1) >>> 0);
  }

  /** Uniform in the open interval (0, 1). */
  next() {
    return (this.u32() + 0.5) / 4294967296;
  }

  uniform(a = 0, b = 1) {
    return a + (b - a) * this.next();
  }

  int(n) {
    return Math.floor(this.next() * n);
  }

  chance(p) {
    return this.next() < p;
  }

  pick(arr) {
    return arr[this.int(arr.length)];
  }

  normal() {
    if (!Number.isNaN(this._spare)) {
      const s = this._spare;
      this._spare = NaN;
      return s;
    }
    const u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this._spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  exponential(rate = 1) {
    return -Math.log(this.next()) / rate;
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

  /** Power-law (x^-alpha) sample in [a, b]. */
  powerLaw(a, b, alpha) {
    const u = this.next();
    if (Math.abs(alpha - 1) < 1e-6) return a * Math.pow(b / a, u);
    const e = 1 - alpha;
    const pa = Math.pow(a, e);
    const pb = Math.pow(b, e);
    return Math.pow(pa + u * (pb - pa), 1 / e);
  }
}

/** Hashed lattice value in [-1, 1] for integer i. */
function lattice(key, i) {
  return (mix32((key ^ mix32((i | 0) + 0x5bd1e995)) >>> 0) / 4294967295) * 2 - 1;
}

/** Smooth 1-D value noise in [-1, 1], a pure function of (key, x). */
export function valueNoise1D(key, x) {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * f * (f * (f * 6 - 15) + 10);
  return lattice(key, i) * (1 - u) + lattice(key, i + 1) * u;
}

/** 1/f-like fractal value noise in about [-1, 1]. */
export function fractalNoise1D(key, x, octaves = 4, gain = 0.6) {
  let s = 0;
  let a = 1;
  let n = 0;
  let f = 1;
  for (let k = 0; k < octaves; k++) {
    s += a * valueNoise1D((key + Math.imul(k, 0x9e3779b9)) >>> 0, x * f + k * 17.31);
    n += a;
    a *= gain;
    f *= 2;
  }
  return s / n;
}

// ---------------------------------------------------------------------------
// Basic atmosphere
// ---------------------------------------------------------------------------

/** Speed of sound (m/s) at temperature tC. */
export function speedOfSound(tC = 20) {
  return 331.3 * Math.sqrt(Math.max(0.05, 1 + num(tC, 20) / 273.15));
}

/**
 * ISO 9613-1 pure-tone atmospheric absorption coefficient in dB per metre.
 * f in Hz, tC in Celsius, rh in percent, pKpa ambient pressure.
 */
export function airAbsorptionDbPerM(f, tC = 20, rh = 70, pKpa = 101.325) {
  const T = num(tC, 20) + 273.15;
  const T0 = 293.15;
  const T01 = 273.16;
  const pr = num(pKpa, 101.325) / 101.325;
  const C = -6.8346 * Math.pow(T01 / T, 1.261) + 4.6151;
  const psat = Math.pow(10, C);
  const h = (clamp(num(rh, 70), 1, 100) * psat) / pr;
  const frO = pr * (24 + (4.04e4 * h * (0.02 + h)) / (0.391 + h));
  const frN = pr * Math.pow(T / T0, -0.5) * (9 + 280 * h * Math.exp(-4.17 * (Math.pow(T / T0, -1 / 3) - 1)));
  const f2 = f * f;
  return (
    8.686 *
    f2 *
    ((1.84e-11 / pr) * Math.sqrt(T / T0) +
      Math.pow(T / T0, -2.5) *
        ((0.01275 * Math.exp(-2239.1 / T)) / (frO + f2 / frO) + (0.1068 * Math.exp(-3352.0 / T)) / (frN + f2 / frN)))
  );
}

/** Total air absorption (dB) over a path. */
export function airAbsorptionDb(f, distanceM, tC = 20, rh = 70) {
  return airAbsorptionDbPerM(f, tC, rh) * Math.max(0, distanceM);
}

/**
 * Frequency (Hz) at which the ISO 9613-1 path absorption reaches `db` dB:
 * used as the cutoff of a gentle low-pass that stands in for air absorption.
 */
export function absorptionCutoffHz(distanceM, tC = 20, rh = 70, db = 3, fMax = 20000) {
  const d = Math.max(0.1, num(distanceM, 1));
  if (airAbsorptionDb(fMax, d, tC, rh) < db) return fMax;
  let lo = Math.log(20);
  let hi = Math.log(fMax);
  if (airAbsorptionDb(20, d, tC, rh) >= db) return 20;
  for (let i = 0; i < 30; i++) {
    const mid = 0.5 * (lo + hi);
    if (airAbsorptionDb(Math.exp(mid), d, tC, rh) < db) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

// ---------------------------------------------------------------------------
// Drops: fall speed, force, DSD, classes
// ---------------------------------------------------------------------------

const RHO_AIR = 1.204; // kg/m^3 at 20 C, 1013 hPa
const MU_AIR = 1.81e-5; // dynamic viscosity of air, Pa s (~15 C)
const D_STOKES_MM = 0.1; // below: Stokes regime
const D_ATLAS_MM = 0.5; // above: Atlas et al. (1973)
const STOKES_K = (G * 1e-6 * (RHO_WATER - RHO_AIR)) / (18 * MU_AIR); // v = K d_mm^2
const V_STOKES_EDGE = STOKES_K * D_STOKES_MM * D_STOKES_MM;
const V_ATLAS_EDGE = 9.65 - 10.3 * Math.exp(-0.6 * D_ATLAS_MM);
const BRIDGE_SLOPE = (V_ATLAS_EDGE - V_STOKES_EDGE) / (D_ATLAS_MM - D_STOKES_MM);

/**
 * Terminal fall speed (m/s) of a raindrop of diameter dMm: Atlas, Srivastava & Sekhon (1973)
 * v = 9.65 - 10.3 exp(-0.6 D) from 0.5 mm, Stokes drag below 0.1 mm and a linear bridge in
 * between (within 3 % of Gunn & Kinzer 1949). Sea-level air by default; `rhoAir` (kg/m^3)
 * applies the Foote & du Toit (1969) correction v = v0 (rho0 / rho)^0.4.
 */
export function fallSpeed(dMm, rhoAir) {
  const d = num(dMm, 1);
  let v;
  if (d >= D_ATLAS_MM) v = Math.max(0, 9.65 - 10.3 * Math.exp(-0.6 * d));
  else if (d >= D_STOKES_MM) v = V_STOKES_EDGE + BRIDGE_SLOPE * (d - D_STOKES_MM);
  else v = d > 0 ? STOKES_K * d * d : 0;
  if (rhoAir !== undefined && rhoAir !== null) v *= Math.pow(RHO_AIR / Math.max(num(rhoAir, RHO_AIR), 1e-3), 0.4);
  return v;
}

/** Terminal fall speed (m/s) for diameter in mm: the Atlas et al. (1973) law of fallSpeed(). */
export function atlasFallSpeed(dMm) {
  return Math.max(0.05, fallSpeed(Math.max(0, num(dMm, 1))));
}

/** Hail terminal speed (m/s), Matson & Huggins (1980): 11.45 sqrt(D_cm). */
export function hailFallSpeed(dMm) {
  return 11.45 * Math.sqrt(Math.max(0.5, num(dMm, 10)) / 10);
}

/** Impact speed (m/s) of a drop released from rest and falling h metres (with drag toward terminal). */
export function dripImpactSpeed(dMm, fallM) {
  const vt = atlasFallSpeed(dMm);
  const h = Math.max(0, num(fallM, 1));
  return vt * Math.sqrt(1 - Math.exp((-2 * G * h) / (vt * vt)));
}

const V2MM = atlasFallSpeed(2);
const FORCE_REF = 4 * V2MM * V2MM;

/** Peak impact force relative to a 2 mm drop at terminal speed (F ~ 0.83 rho D^2 V^2). */
export function impactAmplitude(dMm, v) {
  return (dMm * dMm * v * v) / FORCE_REF;
}

/** Peak impact force in newtons, Gordillo et al. (2018). */
export function impactForceN(dMm, v) {
  const D = dMm * 1e-3;
  return 0.83 * RHO_WATER * D * D * v * v;
}

/** Drop diameter (mm) from Tate's law for a drip tip of radius rTipMm (Harkins-Brown factor 0.6). */
export function tateDropDiameterMm(rTipMm = 1.5) {
  const r = clamp(num(rTipMm, 1.5), 0.2, 5) * 1e-3;
  const vol = (0.6 * 2 * Math.PI * r * SIGMA_WATER) / (RHO_WATER * G);
  return clamp(1000 * Math.cbrt((6 * vol) / Math.PI), 3, 5.5);
}

/** Lanczos approximation of the gamma function. */
export function gammaFn(x) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gammaFn(1 - x));
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
}

/** f(mu) of the normalized gamma DSD: 6/4^4 (4+mu)^(mu+4) / Gamma(mu+4). */
export function fMu(mu) {
  return (6 / 256) * Math.exp((mu + 4) * Math.log(4 + mu) - logGamma(mu + 4));
}

/** ln Gamma(x) for x > 0 (Lanczos, g = 7). */
function logGamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const y = x - 1;
  let a = c[0];
  const t = y + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (y + i);
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Drop-size regimes: Dm = a R^b mm with a soft saturation at `cap` (sharpness p), mu = 3.
 * 'fine' and 'fat' sit on the Bringi et al. (2003) maritime and continental convective cluster
 * centres at ~30 mm/h (Dm 1.62 / 2.40 mm, log10 Nw 4.25 / 3.48); 'normal' is stratiform /
 * mid-latitude rain (D0 ~1.7-2.0 mm in heavy rain). In heavy rain extra rate comes mostly from
 * more drops, so Dm saturates; fine < normal < fat holds at every rate.
 */
export const DROP_REGIMES = {
  fine: { a: 0.85, b: 0.206, cap: 1.85, p: 8, mu: 3, label: 'maritime convective' },
  normal: { a: 1.0, b: 0.2, cap: 2.25, p: 4, mu: 3, label: 'stratiform' },
  fat: { a: 1.39, b: 0.17, cap: 2.9, p: 8, mu: 3, label: 'continental convective' },
};
export const DROP_CHARACTER = DROP_REGIMES;
const DM_FLOOR_MM = 0.35;

export const CLASS_BOUNDARIES = {
  medwin: [0.8, 1.1, 2.2, 3.5],
  nystuen: [0.8, 1.2, 2.0, 3.5],
};
export const CLASS_NAMES = ['minuscule', 'small', 'mid', 'large', 'very_large'];

const RATE_CONST = 6 * Math.PI * 1e-4; // mm^3 m^-3 (m/s) -> mm/h
const TABLE_INTERVALS = 768;
const GRID_BREAKS = [D_STOKES_MM, D_ATLAS_MM, 0.8, 1.1, 1.2, 2.0, 2.2, 3.5];
const GRID_CACHE = new Map();

/**
 * Integration grid on [dMin, dMax]: 768 uniform intervals plus every class edge of both
 * boundary schemes and the fall-speed breakpoints, so class fluxes are exact trapezoid sums.
 * Holds d, ln d, d^3 and v(d); cached per cutoff pair.
 */
function dsdGrid(dMin, dMax) {
  const key = `${dMin}|${dMax}`;
  let g = GRID_CACHE.get(key);
  if (g) return g;
  const pts = [];
  const step = (dMax - dMin) / TABLE_INTERVALS;
  for (let i = 0; i < TABLE_INTERVALS; i++) pts.push(dMin + i * step);
  pts.push(dMax);
  for (const b of GRID_BREAKS) if (b > dMin && b < dMax) pts.push(b);
  pts.sort((a, b) => a - b);
  const d = [];
  for (const p of pts) if (!d.length || p !== d[d.length - 1]) d.push(p);
  const n = d.length;
  g = { n, d: Float64Array.from(d), lnd: new Float64Array(n), d3: new Float64Array(n), v: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    const x = g.d[i];
    g.lnd[i] = Math.log(x);
    g.d3[i] = x * x * x;
    g.v[i] = fallSpeed(x);
  }
  if (GRID_CACHE.size > 32) GRID_CACHE.clear();
  GRID_CACHE.set(key, g);
  return g;
}

/**
 * Trapezoid integrals of the unit-Nw shape f(mu) (D/Dm)^mu exp(-(4+mu) D/Dm) on the grid:
 * rate = Int D^3 v N dD; with `bins` (4 class edges) also per-class flux = Int v N dD and
 * first moment Int D v N dD written into `out`.
 */
function shapeIntegrals(dm, mu, dMin, dMax, bins, out) {
  const g = dsdGrid(dMin, dMax);
  const { n, d, lnd, d3, v } = g;
  const fm = fMu(mu);
  const lndm = Math.log(dm);
  const k = (4 + mu) / dm;
  let rate = 0;
  let pf = fm * Math.exp(mu * (lnd[0] - lndm) - k * d[0]) * v[0];
  if (out) {
    out.flux.fill(0);
    out.m1.fill(0);
  }
  for (let i = 1; i < n; i++) {
    const f = fm * Math.exp(mu * (lnd[i] - lndm) - k * d[i]) * v[i];
    const h = 0.5 * (d[i] - d[i - 1]);
    rate += h * (pf * d3[i - 1] + f * d3[i]);
    if (out) {
      const mid = 0.5 * (d[i] + d[i - 1]);
      const c = mid < bins[0] ? 0 : mid < bins[1] ? 1 : mid < bins[2] ? 2 : mid < bins[3] ? 3 : 4;
      out.flux[c] += h * (pf + f);
      out.m1[c] += h * (pf * d[i - 1] + f * d[i]);
    }
    pf = f;
  }
  return rate;
}

function nullable(x) {
  if (x === null || x === undefined || x === '') return null;
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}

/** Dm (mm) such that the DSD with fixed Nw carries `rate` mm/h (rate grows monotonically with Dm). */
function solveDm(nw, mu, dMin, dMax, rate) {
  let lo = Math.log(0.1);
  let hi = Math.log(8);
  for (let i = 0; i < 48; i++) {
    const mid = 0.5 * (lo + hi);
    if (nw * RATE_CONST * shapeIntegrals(Math.exp(mid), mu, dMin, dMax) < rate) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

/** Number density N(D) in m^-3 mm^-1 (zero outside the cutoffs). */
export function dsdDensity(dsd, D) {
  if (!dsd || !(dsd.nw > 0) || !(D > 0) || D < dsd.dMin || D > dsd.dMax) return 0;
  const x = D / dsd.dm;
  return dsd.nw * dsd.fmu * Math.exp(dsd.mu * Math.log(x) - (4 + dsd.mu) * x);
}

/** Rain rate (mm/h) carried by a DSD: 6 pi 1e-4 Int D^3 v(D) N(D) dD. */
export function rainRateFromDsd(dsd) {
  if (!dsd || !(dsd.nw > 0) || !(dsd.dm > 0) || !(dsd.dMax > dsd.dMin)) return 0;
  return RATE_CONST * dsd.nw * shapeIntegrals(dsd.dm, dsd.mu, dsd.dMin, dsd.dMax);
}

/**
 * Build the normalized-gamma DSD (Testud et al. 2001; Bringi et al. 2003) that carries `rate` mm/h.
 *
 * Dm follows the drop regime and saturates (DROP_REGIMES); drizzle uses Dm 0.25-0.5 mm on
 * 0.1-1.2 mm drops; Marshall & Palmer (1948) uses Lambda = 4.1 R^-0.21 (mu = 0). Nw is then solved
 * so that rainRateFromDsd(dsd) == rate (closure). Overrides: `mu` (clamped to -3..20), `dm`, `nw`
 * (with nw pinned and dm free, Dm is solved instead). `dmScale` moves Dm (size sorting along a
 * storm arc) with Nw re-solved, unless dm or nw is pinned or the type is drizzle.
 *
 * opts: { type, drops: 'fine'|'normal'|'fat', model: 'gamma'|'marshall_palmer', mu, dm, nw, dMin, dMax, dmScale }
 */
export function dsdFromRate(rate, opts = {}) {
  const o = opts || {};
  const R = Math.max(0, num(rate, 0));
  const type = String(o.type || 'rain').toLowerCase();
  const drizzle = type === 'drizzle' || type === 'freezing_drizzle';
  const model = String(o.model || 'gamma').toLowerCase().replace(/-/g, '_');
  const mp = model === 'marshall_palmer' || model === 'mp' || model === 'marshallpalmer';
  let dMin = Math.max(num(o.dMin, 0.3), 0.01);
  let dMax = Math.max(num(o.dMax, 6), dMin + 0.05);
  const Rc = Math.max(R, 1e-3);
  let dm;
  let mu;
  if (drizzle) {
    dm = clamp(0.35 * Math.pow(Rc / 0.2, 0.08), 0.25, 0.5);
    mu = 3;
    dMin = Math.min(dMin, 0.1);
    dMax = Math.max(Math.min(dMax, 1.2), dMin + 0.05);
  } else if (mp) {
    dm = 4 / (4.1 * Math.pow(Rc, -0.21));
    mu = 0;
  } else {
    const r = DROP_REGIMES[o.drops] || DROP_REGIMES.normal;
    const x = r.a * Math.pow(Rc, r.b);
    dm = Math.max(DM_FLOOR_MM, x / Math.pow(1 + Math.pow(x / r.cap, r.p), 1 / r.p));
    mu = r.mu;
  }
  const muO = nullable(o.mu);
  const dmO = nullable(o.dm);
  const nwO = nullable(o.nw);
  if (muO !== null) mu = muO;
  mu = clamp(mu, -3, 20);
  if (dmO !== null) dm = Math.max(dmO, 0.05);
  const dmScale = clamp(num(o.dmScale, 1), 0.3, 3);
  if (R > 0 && Math.abs(dmScale - 1) > 1e-3 && dmO === null && nwO === null && !drizzle) dm = Math.max(dm * dmScale, 0.05);

  let nw = 0;
  if (R > 0 && type !== 'none') {
    if (nwO !== null) {
      nw = Math.max(nwO, 0);
      if (dmO === null && nw > 0) dm = solveDm(nw, mu, dMin, dMax, R);
    } else {
      const r1 = RATE_CONST * shapeIntegrals(dm, mu, dMin, dMax);
      nw = r1 > 0 ? R / r1 : 0;
    }
  }
  const fmu = fMu(mu);
  return { model: mp ? 'marshall_palmer' : 'gamma', nw, dm, mu, fmu, n0: nw * fmu, lambda: (4 + mu) / dm, dMin, dMax, rate: R };
}

/**
 * Impact flux (m^-2 s^-1), flux-weighted mean diameter (mm) and fall speed per acoustic class:
 * Medwin, Nystuen et al. (1992) boundaries 0.8 / 1.1 / 2.2 / 3.5 mm, or Nystuen (2001)
 * 0.8 / 1.2 / 2.0 / 3.5 mm. Returns 5 entries: minuscule, small, mid, large, very_large.
 */
export function classFlux(dsd, boundaries = 'medwin') {
  const b = CLASS_BOUNDARIES[boundaries] || CLASS_BOUNDARIES.medwin;
  const edges = [dsd.dMin, b[0], b[1], b[2], b[3], dsd.dMax];
  let acc = null;
  if (dsd.nw > 0 && dsd.dm > 0 && dsd.dMax > dsd.dMin) {
    acc = { flux: new Float64Array(5), m1: new Float64Array(5) };
    shapeIntegrals(dsd.dm, dsd.mu, dsd.dMin, dsd.dMax, b, acc);
  }
  const out = [];
  for (let c = 0; c < 5; c++) {
    const lo = Math.max(edges[c], dsd.dMin);
    const hi = Math.min(edges[c + 1], dsd.dMax);
    const mid = 0.5 * (edges[c] + edges[c + 1]);
    const flux = acc && hi > lo ? dsd.nw * acc.flux[c] : 0;
    if (!(flux > 0)) {
      out.push({ name: CLASS_NAMES[c], lo: edges[c], hi: edges[c + 1], flux: 0, d: clamp(mid, 0.1, 6), v: atlasFallSpeed(mid) });
      continue;
    }
    const d = acc.flux[c] > 1e-300 ? clamp(acc.m1[c] / acc.flux[c], lo, hi) : 0.5 * (lo + hi);
    out.push({ name: CLASS_NAMES[c], lo, hi, flux, d, v: atlasFallSpeed(d) });
  }
  return out;
}

/** Sum of flux x (relative force)^2: a proxy for the acoustic energy rate of the rain field per m^2. */
export function impactEnergyRate(classes) {
  let e = 0;
  for (const c of classes || []) {
    if (!c || !(c.flux > 0)) continue;
    const a = impactAmplitude(c.d, c.v);
    e += c.flux * a * a;
  }
  return e;
}

/** Total impact flux (m^-2 s^-1) at or above dMinMm. */
export function totalFlux(classes, dMinMm = 0) {
  let s = 0;
  for (const c of classes || []) if (c && c.hi > dMinMm) s += c.flux;
  return s;
}

// ---------------------------------------------------------------------------
// Precipitation types -> per-class descriptors fed to the rain-surface worklet
// kinds: 0 liquid, 1 ice (Hertz contact), 2 snow (soft), 3 graupel (soft ice)
// ---------------------------------------------------------------------------

/**
 * Resolve precipitation into 6 worklet classes [5 drop classes + hail/ice].
 * Returns { classes: [{flux, d, v}] x6, kinds: [k] x6, dsd, energyRate, rate }.
 */
export function resolvePrecipitation(type, rate, opts = {}) {
  const t = String(type || 'rain').toLowerCase();
  const R = Math.max(0, num(rate, 0));
  const o = { ...opts, type: t };
  let kinds = [0, 0, 0, 0, 0, 1];
  let liquidRate = R;
  let hailFlux = 0;
  let hailD = clamp(num(opts.hailSizeMm, 13), 5, 80);
  if (t === 'none' || R <= 0) {
    liquidRate = 0;
  } else if (t === 'hail') {
    liquidRate = R * 0.6;
    const vol = (Math.PI / 6) * Math.pow(hailD * 1e-3, 3);
    hailFlux = (R * 0.4) / 3.6e6 / vol;
  } else if (t === 'sleet' || t === 'ice_pellets') {
    kinds = [1, 1, 1, 1, 1, 1];
    o.dMin = Math.max(num(opts.dMin, 0.5), 0.5);
    o.dMax = Math.min(num(opts.dMax, 5), 5);
  } else if (t === 'snow' || t === 'snow_showers') {
    kinds = [2, 2, 2, 2, 2, 2];
    o.dm = num(opts.dm, 2.5);
  } else if (t === 'graupel') {
    kinds = [3, 3, 3, 3, 3, 3];
    o.dm = num(opts.dm, 2.2);
  }
  const dsd = dsdFromRate(liquidRate, o);
  const cls = classFlux(dsd, opts.boundaries);
  const classes = cls.map((c) => ({ flux: c.flux, d: c.d, v: c.v, lo: c.lo, hi: c.hi }));
  if (kinds[0] === 2) {
    // Snowflakes: ~1 m/s, liquid-equivalent flux of large, light aggregates.
    for (const c of classes) {
      c.v = 1.0;
      c.flux *= 0.6;
    }
  } else if (kinds[0] === 3) {
    for (const c of classes) c.v = 1.4 + 0.3 * c.d;
  }
  classes.push({ flux: hailFlux, d: hailD, v: hailFallSpeed(hailD), lo: 5, hi: 80 });
  return { classes, kinds, dsd, energyRate: impactEnergyRate(classes.slice(0, 5)), rate: R };
}

// ---------------------------------------------------------------------------
// Wind
// ---------------------------------------------------------------------------

/** Place presets: roughness length z0 (m), far-field rain-bed factor, defaults. */
export const PLACE_PRESETS = {
  garden: { z0: 0.03, far: 0.85, surfaces: { grass: 40, soil: 20, leaves: 20, puddle: 10, concrete: 10 }, vegetation: 'deciduous' },
  deciduous_forest: { z0: 0.8, far: 0.7, surfaces: { leaves: 55, soil: 25, grass: 10, puddle: 10 }, vegetation: 'deciduous' },
  conifer_forest: { z0: 0.9, far: 0.7, surfaces: { needles: 55, soil: 25, puddle: 10, grass: 10 }, vegetation: 'conifer' },
  meadow: { z0: 0.03, far: 0.8, surfaces: { grass: 75, soil: 15, puddle: 10 }, vegetation: 'grass' },
  coast: { z0: 0.0005, far: 1.0, surfaces: { stone: 30, gravel: 30, water: 30, puddle: 10 }, vegetation: 'none' },
  lake: { z0: 0.0002, far: 1.0, surfaces: { water: 55, grass: 20, soil: 15, stone: 10 }, vegetation: 'deciduous' },
  city_street: { z0: 1.0, far: 1.1, surfaces: { asphalt: 45, concrete: 30, puddle: 20, metal: 5 }, vegetation: 'none' },
  suburb: { z0: 0.4, far: 0.95, surfaces: { grass: 30, concrete: 25, asphalt: 20, puddle: 15, tile: 10 }, vegetation: 'deciduous' },
  alley: { z0: 1.2, far: 0.9, surfaces: { concrete: 50, puddle: 25, metal: 15, asphalt: 10 }, vegetation: 'none' },
  mountain_ridge: { z0: 0.1, far: 0.7, surfaces: { stone: 50, gravel: 25, grass: 20, puddle: 5 }, vegetation: 'conifer' },
  desert: { z0: 0.005, far: 0.8, surfaces: { soil: 60, gravel: 30, stone: 10 }, vegetation: 'none' },
  jungle: { z0: 1.5, far: 0.75, surfaces: { leaves: 60, soil: 20, puddle: 15, water: 5 }, vegetation: 'deciduous' },
  farm: { z0: 0.05, far: 0.85, surfaces: { soil: 40, grass: 35, puddle: 15, metal: 10 }, vegetation: 'deciduous' },
};

const PRESET_ALIASES = {
  forest: 'deciduous_forest',
  woods: 'deciduous_forest',
  conifer: 'conifer_forest',
  pine_forest: 'conifer_forest',
  city: 'city_street',
  street: 'city_street',
  beach: 'coast',
  sea: 'coast',
  mountain: 'mountain_ridge',
  field: 'meadow',
};

export function placePreset(name) {
  const k = String(name || 'garden').toLowerCase().replace(/[\s-]+/g, '_');
  return PLACE_PRESETS[k] || PLACE_PRESETS[PRESET_ALIASES[k]] || PLACE_PRESETS.garden;
}

/** Log-law wind speed at height z (m) from the reported 10 m mean u10. */
export function windAtHeight(u10, z, z0 = 0.03) {
  const r = clamp(num(z0, 0.03), 0.0001, 3);
  const h = Math.max(num(z, 1.7), r * 1.5, 0.2);
  return Math.max(0, num(u10, 0)) * (Math.log(h / r) / Math.log(10 / r));
}

/** Longitudinal turbulence intensity ~ 1 / ln(z/z0). */
export function turbulenceIntensity(z, z0 = 0.03) {
  const r = clamp(num(z0, 0.03), 0.0001, 3);
  return 1 / Math.log(Math.max(num(z, 10), r * 2) / r);
}

/** 3-s gust factor from roughness: ~1.3 at sea, ~2 in forests and towns. */
export function gustFactorFromRoughness(z0 = 0.03) {
  return clamp(1 + 3 * turbulenceIntensity(10, z0), 1.15, 2.0);
}

/** Kaimal spectral peak period (s): n_peak = 0.045 U / z. */
export function kaimalPeakPeriod(U, z = 10) {
  return clamp(z / (0.045 * Math.max(0.5, num(U, 5))), 6, 150);
}

/**
 * Normalized Kaimal-like gust process: a weighted sum of Ornstein-Uhlenbeck processes
 * whose corner periods span the Kaimal peak down to the inertial range (amplitude per
 * octave ~ tau^(1/3), i.e. an f^-5/3 energy spectrum), normalized to unit variance and
 * scaled so the 3-s gust factor G is reached at ~2.5 sigma. Deterministic per seed.
 */
export class GustProcess {
  constructor(seed = 0, layer = 'gust') {
    this.seed = seed;
    this.layer = layer;
    this.reset();
  }

  reset() {
    this.rng = new CounterRng(this.seed, this.layer, 0);
    this.x = new Float64Array(5);
    for (let k = 0; k < 5; k++) this.x[k] = this.rng.normal();
    this.k = 0;
    this.value = 1;
  }

  /** Advance by dt seconds with mean speed U (m/s) and gust factor G; returns the multiplier u/U. */
  step(dt, U, gustFactor = 1.5) {
    const Tp = kaimalPeakPeriod(U, 10);
    const tau0 = (Tp / (2 * Math.PI)) * 4;
    let sum = 0;
    let norm = 0;
    for (let k = 0; k < 5; k++) {
      const tau = tau0 * Math.pow(3, -k);
      const a = Math.exp(-dt / tau);
      this.x[k] = this.x[k] * a + Math.sqrt(Math.max(0, 1 - a * a)) * this.rng.normal();
      const w = k === 0 ? 0.75 : Math.pow(tau / (tau0 / 3), 1 / 3);
      sum += w * this.x[k];
      norm += w * w;
    }
    const n = sum / Math.sqrt(norm);
    const sigma = Math.max(0, num(gustFactor, 1.5) - 1) / 2.5;
    this.k++;
    this.value = Math.max(0.08, 1 + sigma * n);
    return this.value;
  }
}

// ---------------------------------------------------------------------------
// Storm life cycle and shower structure
// ---------------------------------------------------------------------------

function bump(p, c, w) {
  const z = (p - c) / w;
  return Math.exp(-z * z);
}

/**
 * Storm-arc envelope: sparse big drops -> downpour (Dm falls while the drop count rises)
 * -> stratiform tail -> drips. Returns multipliers for rain rate, Dm, wind
 * and thunder rate, plus progress 0..1 through the arc.
 */
export function stormArcEnvelope(arc, tSec, arcMinutes = 20) {
  const T = Math.max(30, num(arcMinutes, 20) * 60);
  const p = clamp(num(tSec, 0) / T, 0, 1);
  switch (String(arc || 'steady')) {
    case 'arriving':
      return {
        rate: 0.03 + 0.97 * smoothstep(0.1, 0.85, p),
        dm: 1 + 0.35 * (1 - smoothstep(0.0, 0.55, p)),
        wind: 0.7 + 0.3 * smoothstep(0.3, 0.8, p) + 0.5 * bump(p, 0.72, 0.1),
        thunder: 0.25 + 0.75 * smoothstep(0.2, 0.9, p),
        progress: p,
      };
    case 'passing': {
      const core = bump(p, 0.45, 0.2);
      const tail = 0.18 * smoothstep(0.5, 0.65, p) * (1 - smoothstep(0.85, 1.0, p));
      return {
        rate: 0.03 + 0.97 * Math.max(core, tail),
        dm: p < 0.45 ? 1 + 0.3 * (1 - p / 0.45) : 1 - 0.18 * smoothstep(0.55, 0.9, p),
        wind: 0.8 + 0.6 * bump(p, 0.38, 0.1) - 0.3 * smoothstep(0.6, 1, p),
        thunder: bump(p, 0.4, 0.25),
        progress: p,
      };
    }
    case 'clearing':
      return {
        rate: Math.pow(1 - p, 1.5) * (1 - smoothstep(0.85, 1, p)),
        dm: 1 - 0.2 * p,
        wind: 1 - 0.4 * p,
        thunder: Math.max(0, 1 - 1.6 * p),
        progress: p,
      };
    default:
      return { rate: 1, dm: 1, wind: 1, thunder: 1, progress: p };
  }
}

/**
 * Rain-rate multiplier for the precipitation character, a pure function of time:
 * continuous (slow +-15 % 1/f wander), intermittent (on/off spells), showery (convective
 * cells with period `periodMin`).
 */
export function showerEnvelope(key, tSec, character = 'continuous', periodMin = 12) {
  const t = num(tSec, 0);
  const P = Math.max(60, num(periodMin, 12) * 60);
  const k = key >>> 0;
  switch (String(character || 'continuous')) {
    case 'intermittent': {
      const seg = P / 2;
      const x = t / seg;
      const i = Math.floor(x);
      const f = x - i;
      const val = (j) => {
        const h = mix32((k ^ mix32((j | 0) + 0x1234567)) >>> 0) / 4294967296;
        const h2 = mix32((k ^ mix32((j | 0) + 0x7654321)) >>> 0) / 4294967296;
        return h < 0.55 ? 0.6 + 0.7 * h2 : 0.0;
      };
      const v0 = val(i);
      const v1 = val(i + 1);
      const w = smoothstep(0.78, 1.0, f);
      return v0 + (v1 - v0) * w;
    }
    case 'showery': {
      const x = t / P;
      const i = Math.floor(x + 0.5);
      const phase = x - i;
      const peak = 0.7 + 1.1 * ((mix32((k ^ mix32((i | 0) + 0x2545f491)) >>> 0) / 4294967296));
      const width = 0.22 + 0.08 * fractalNoise1D(k ^ 0x51ed, x, 2);
      const e = Math.exp(-0.5 * (phase / width) * (phase / width));
      return 0.04 + peak * e * (1 + 0.15 * fractalNoise1D(k ^ 0xabc, t / 40, 3));
    }
    default:
      return clamp(1 + 0.15 * fractalNoise1D(k, t / 90, 3), 0.6, 1.4);
  }
}

// ---------------------------------------------------------------------------
// Wetness stores (dry -> film -> puddle)
// ---------------------------------------------------------------------------

/** Equilibrium film wetness 0..1 and puddle depth (mm) for a rain rate. */
export function wetnessEquilibrium(rate) {
  const r = Math.max(0, num(rate, 0));
  return {
    film: 1 - Math.exp(-r / 0.8),
    puddle: r > 0.2 ? Math.min(12, 2.2 * Math.pow(r, 0.55)) : 0,
  };
}

/** Advance {film, puddle} by dt seconds; fills fast with rain, drains with the aftermath constant. */
export function stepWetness(state, rate, dt, aftermathTauMin = 8) {
  const eq = wetnessEquilibrium(rate);
  const drain = Math.max(30, num(aftermathTauMin, 8) * 60);
  const tauF = eq.film > state.film ? 20 / (0.2 + Math.min(5, rate)) : drain * 0.5;
  const tauP = eq.puddle > state.puddle ? 150 / (0.3 + Math.min(5, Math.sqrt(rate))) : drain;
  state.film += (eq.film - state.film) * (1 - Math.exp(-dt / tauF));
  state.puddle += (eq.puddle - state.puddle) * (1 - Math.exp(-dt / tauP));
  return state;
}

// ---------------------------------------------------------------------------
// Materials: modal banks for the rain-surface worklet
// ---------------------------------------------------------------------------

/**
 * kind: solid | water | membrane | soft. model: plate (rectangular plate ratios + statistical
 * high modes), membrane (circular membrane Bessel ratios), random (log-uniform sparse modes).
 * fc: critical frequency, radiation efficiency ~ sqrt(f/fc) below it.
 * modal: energy level of the modal response; click: direct radiated pulse (dF/dt);
 * noise: per-impact noise burst (fabric rattle, leaf rustle, wet slap).
 */
export const MATERIALS = {
  tin_roof: { kind: 'solid', model: 'plate', f1: 90, asp: 1.4, fHi: 9000, n: 16, t60: 0.5, decay: 0.45, fc: 24000, refArea: 4, modal: 0.16, click: 0.9, noise: 0.03, noiseDecay: 0.002, splash: 0.6, wetDamp: 0.25, hardness: 1.0 },
  metal: { kind: 'solid', model: 'plate', f1: 240, asp: 1.2, fHi: 10000, n: 14, t60: 0.7, decay: 0.5, fc: 6000, refArea: 1, modal: 0.14, click: 1.0, noise: 0.02, noiseDecay: 0.002, splash: 0.6, wetDamp: 0.2, hardness: 1.0 },
  car_body: { kind: 'solid', model: 'plate', f1: 70, asp: 1.8, fHi: 6000, n: 14, t60: 0.25, decay: 0.45, fc: 12000, refArea: 2, modal: 0.15, click: 0.8, noise: 0.02, noiseDecay: 0.002, splash: 0.5, wetDamp: 0.25, hardness: 0.95 },
  glass: { kind: 'solid', model: 'plate', f1: 380, asp: 1.5, fHi: 9000, n: 12, t60: 0.14, decay: 0.4, fc: 3000, refArea: 1.5, modal: 0.1, click: 1.25, noise: 0.02, noiseDecay: 0.0015, splash: 0.5, wetDamp: 0.3, hardness: 0.9 },
  wood: { kind: 'solid', model: 'plate', f1: 160, asp: 1.3, fHi: 4000, n: 10, t60: 0.05, decay: 0.5, fc: 1400, refArea: 2, modal: 0.07, click: 0.6, noise: 0.05, noiseDecay: 0.002, splash: 0.35, wetDamp: 0.35, hardness: 0.5 },
  plastic: { kind: 'solid', model: 'plate', f1: 200, asp: 1.2, fHi: 6000, n: 8, t60: 0.06, decay: 0.5, fc: 9000, refArea: 0.5, modal: 0.09, click: 0.75, noise: 0.03, noiseDecay: 0.002, splash: 0.4, wetDamp: 0.3, hardness: 0.4 },
  shingle: { kind: 'solid', model: 'random', fLo: 400, fHi: 2500, n: 6, t60: 0.012, decay: 0.3, modal: 0.04, click: 0.6, noise: 0.05, noiseDecay: 0.002, splash: 0.4, wetDamp: 0.3, hardness: 0.5 },
  tile: { kind: 'solid', model: 'random', fLo: 1500, fHi: 7000, n: 6, t60: 0.012, decay: 0.3, modal: 0.04, click: 1.1, noise: 0.03, noiseDecay: 0.0015, splash: 0.85, puddleable: true, wetDamp: 0.2, hardness: 0.95 },
  concrete: { kind: 'solid', model: 'random', fLo: 900, fHi: 5000, n: 4, t60: 0.004, decay: 0.2, modal: 0.025, click: 1.0, noise: 0.04, noiseDecay: 0.0015, splash: 0.9, puddleable: true, wetDamp: 0.1, hardness: 1.0 },
  asphalt: { kind: 'solid', model: 'random', fLo: 600, fHi: 3500, n: 4, t60: 0.003, decay: 0.2, modal: 0.02, click: 0.85, noise: 0.04, noiseDecay: 0.0015, lpHz: 9000, splash: 0.8, puddleable: true, wetDamp: 0.1, hardness: 0.85 },
  stone: { kind: 'solid', model: 'random', fLo: 1400, fHi: 7000, n: 5, t60: 0.006, decay: 0.25, modal: 0.035, click: 1.1, noise: 0.03, noiseDecay: 0.0015, splash: 0.9, puddleable: true, wetDamp: 0.1, hardness: 1.0 },
  gravel: { kind: 'solid', model: 'random', fLo: 2500, fHi: 9500, n: 8, t60: 0.005, decay: 0.2, modal: 0.05, click: 0.8, noise: 0.18, noiseDecay: 0.004, splash: 0.4, puddleable: true, wetDamp: 0.15, hardness: 0.9 },
  soil: { kind: 'soft', model: 'random', fLo: 120, fHi: 600, n: 3, t60: 0.006, decay: 0.2, modal: 0.03, click: 0.25, noise: 0.06, noiseDecay: 0.002, lpHz: 3500, splash: 0.15, puddleable: true, wetDamp: 0.3, hardness: 0.1 },
  mud: { kind: 'soft', model: 'random', fLo: 100, fHi: 450, n: 3, t60: 0.005, decay: 0.2, modal: 0.025, click: 0.2, noise: 0.08, noiseDecay: 0.003, lpHz: 3000, splash: 0.1, puddleable: true, wetDamp: 0.4, hardness: 0.05, depthMm: 3 },
  grass: { kind: 'soft', model: 'random', fLo: 2200, fHi: 8000, n: 4, t60: 0.003, decay: 0.2, modal: 0.012, click: 0.12, noise: 0.32, noiseDecay: 0.003, lpHz: 7500, splash: 0.05, wetDamp: 0.3, hardness: 0.05 },
  leaves: { kind: 'soft', model: 'random', fLo: 1100, fHi: 6500, n: 8, t60: 0.022, decay: 0.3, modal: 0.07, click: 0.5, noise: 0.22, noiseDecay: 0.004, lpHz: 12000, splash: 0.2, wetDamp: 0.4, hardness: 0.15 },
  needles: { kind: 'soft', model: 'random', fLo: 2500, fHi: 9000, n: 5, t60: 0.007, decay: 0.25, modal: 0.03, click: 0.28, noise: 0.28, noiseDecay: 0.003, lpHz: 9000, splash: 0.1, wetDamp: 0.3, hardness: 0.1 },
  fabric: { kind: 'membrane', model: 'membrane', f1: 70, fHi: 5000, n: 14, t60: 0.08, decay: 0.4, refArea: 4, modal: 0.2, click: 0.35, noise: 0.3, noiseDecay: 0.005, lpHz: 10000, splash: 0.1, wetDamp: 0.3, hardness: 0.15 },
  umbrella: { kind: 'membrane', model: 'membrane', f1: 150, fHi: 7000, n: 12, t60: 0.06, decay: 0.4, refArea: 1, modal: 0.22, click: 0.55, noise: 0.28, noiseDecay: 0.004, lpHz: 12000, splash: 0.15, wetDamp: 0.25, hardness: 0.25 },
  snow: { kind: 'soft', model: 'random', fLo: 200, fHi: 800, n: 0, t60: 0.004, decay: 0.2, modal: 0, click: 0.03, noise: 0.05, noiseDecay: 0.002, lpHz: 2500, splash: 0, wetDamp: 0, hardness: 0.02 },
  puddle: { kind: 'water', depthMm: 12, waterClick: 1.0, bubble: 1.0 },
  water: { kind: 'water', depthMm: 1000, waterClick: 1.0, bubble: 1.0 },
};

export const MATERIAL_ALIASES = {
  puddles: 'puddle', tent_fabric: 'fabric', canvas: 'fabric', tent: 'fabric', tin: 'tin_roof', metal_roof: 'tin_roof', corrugated: 'tin_roof',
  steel: 'metal', iron: 'metal', aluminium: 'metal', aluminum: 'metal', car: 'car_body', vehicle: 'car_body', glass_roof: 'glass', window: 'glass',
  road: 'asphalt', tarmac: 'asphalt', pavement: 'concrete', sidewalk: 'concrete', paving: 'concrete', rock: 'stone', pebbles: 'gravel',
  forest_floor: 'leaves', leaf: 'leaves', moss: 'grass', lawn: 'grass', sand: 'soil', dirt: 'soil', earth: 'soil', snow_fresh: 'snow',
  snow_old: 'snow', roof_tiles: 'tile', tiles: 'tile', deck: 'wood', wood_deck: 'wood', timber: 'wood', pond: 'water', lake: 'water',
  sea: 'water', river: 'water', bucket: 'plastic', tarp: 'fabric', shingles: 'shingle',
  umbrella_fabric: 'umbrella', nylon: 'umbrella', slate: 'tile', brick: 'concrete', plaster: 'concrete', stone_wall: 'stone',
  leaf_litter: 'leaves', pine_needles: 'needles', conifer: 'needles', deciduous: 'leaves',
};

export function resolveMaterialName(name) {
  const k = String(name || '').toLowerCase().trim().replace(/[\s-]+/g, '_');
  if (MATERIALS[k]) return k;
  if (MATERIAL_ALIASES[k]) return MATERIAL_ALIASES[k];
  if (k.endsWith('s') && MATERIALS[k.slice(0, -1)]) return k.slice(0, -1);
  return 'concrete';
}

const MEMBRANE_RATIOS = [1, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.6, 3.652, 4.06, 4.154, 4.601, 4.832, 4.903, 5.131];

function plateRatios(asp, count) {
  const set = [];
  for (let m = 1; m <= 8; m++) for (let n = 1; n <= 8; n++) set.push(m * m + (n * n) / (asp * asp));
  set.sort((a, b) => a - b);
  const base = set[0];
  const out = [];
  for (const v of set) {
    const r = v / base;
    if (!out.length || r - out[out.length - 1] > 0.02) out.push(r);
    if (out.length >= count) break;
  }
  return out;
}

/**
 * Explicit material description for the worklet: modes as flat [f, t60, gain, ...],
 * deterministic per (seed, material, salt), scaled by panel area.
 */
export function buildMaterial(name, opts = {}) {
  const key = resolveMaterialName(name);
  const m = MATERIALS[key];
  const rng = new CounterRng(opts.seed ?? 1, `material:${key}:${opts.salt ?? ''}`, 0);
  const maxModes = clamp(Math.round(num(opts.maxModes, 16)), 0, 24);
  const out = {
    name: key,
    kind: m.kind,
    fraction: Math.max(0, num(opts.fraction, 1)),
    modes: [],
    modal: m.modal ?? 0,
    click: m.click ?? 0,
    noise: m.noise ?? 0,
    noiseDecay: m.noiseDecay ?? 0.002,
    splash: m.splash ?? 0,
    puddleable: m.puddleable ? 1 : 0,
    wetDamp: m.wetDamp ?? 0,
    hardness: m.hardness ?? 0.5,
    lpHz: m.lpHz ?? 0,
    depthMm: num(opts.depthMm, m.depthMm ?? 0),
    waterClick: m.waterClick ?? 1,
    bubble: m.bubble ?? 1,
    spatial: m.kind === 'membrane' ? 0.5 : 0.8,
  };
  if (m.kind === 'water') return out;
  const count = Math.min(maxModes, m.n || 0);
  if (count <= 0) return out;
  const area = Math.max(0.02, num(opts.area, m.refArea ?? 1));
  const scale = m.refArea ? clamp(Math.sqrt(m.refArea / area), 0.4, 2.5) : 1;
  let freqs = [];
  if (m.model === 'plate') {
    const nLow = Math.min(count, 5);
    const r = plateRatios(m.asp || 1.4, nLow);
    for (const x of r) freqs.push(m.f1 * scale * x * (1 + 0.02 * rng.normal()));
    const fStart = freqs[freqs.length - 1] * 1.3;
    const nHi = count - freqs.length;
    for (let i = 0; i < nHi; i++) {
      const u = (i + 0.5 + 0.35 * (rng.next() - 0.5)) / nHi;
      freqs.push(fStart * Math.pow(m.fHi / fStart, u));
    }
  } else if (m.model === 'membrane') {
    const nLow = Math.min(count, 10);
    for (let i = 0; i < nLow; i++) freqs.push(m.f1 * scale * MEMBRANE_RATIOS[i] * (1 + 0.015 * rng.normal()));
    const fStart = freqs[freqs.length - 1] * 1.4;
    const nHi = count - nLow;
    for (let i = 0; i < nHi; i++) {
      const u = (i + 0.5 + 0.35 * (rng.next() - 0.5)) / nHi;
      freqs.push(fStart * Math.pow(m.fHi / fStart, u));
    }
  } else {
    for (let i = 0; i < count; i++) {
      const u = (i + rng.next()) / count;
      freqs.push(m.fLo * Math.pow(m.fHi / m.fLo, u));
    }
  }
  const fRef = freqs[0];
  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    const t60 = Math.max(0.0015, m.t60 * Math.pow(fRef / f, m.decay ?? 0.3) * Math.exp(0.15 * rng.normal()));
    let g = Math.exp(0.3 * rng.normal()) / Math.sqrt(count);
    // Radiation efficiency below the critical frequency: sigma ~ sqrt(f/fc); amplitude ~ sqrt(sigma).
    if (m.fc) g *= 1.6 * Math.pow(Math.min(1, f / m.fc), 0.25);
    out.modes.push(+f.toFixed(2), +t60.toFixed(5), +g.toFixed(5));
  }
  return out;
}

/**
 * Split a {material: percent} surface mix into per-patch mixes (top 3 materials) with
 * seeded spatial heterogeneity: real ground is patchy (a puddle here, gravel there).
 */
export function patchMaterials(surfaces, seed, patchId, heterogeneity = 0.6) {
  const entries = Object.entries(surfaces || {})
    .map(([k, v]) => [resolveMaterialName(k), Math.max(0, num(v, 0))])
    .filter(([, v]) => v > 0);
  if (!entries.length) return [['grass', 1]];
  const merged = new Map();
  for (const [k, v] of entries) merged.set(k, (merged.get(k) || 0) + v);
  const rng = new CounterRng(seed, `patch:${patchId}`, 0);
  const list = [];
  for (const [k, v] of merged) list.push([k, v * Math.exp(heterogeneity * rng.normal())]);
  list.sort((a, b) => b[1] - a[1]);
  const top = list.slice(0, 3);
  const s = top.reduce((a, b) => a + b[1], 0) || 1;
  return top.map(([k, v]) => [k, v / s]);
}

// ---------------------------------------------------------------------------
// Geometry helpers (world: metres, x east, y north, z up; bearings deg true)
// ---------------------------------------------------------------------------

export const DEG = Math.PI / 180;

/** Unit vector (x east, y north) of a compass bearing. */
export function bearingVector(deg) {
  const r = num(deg, 0) * DEG;
  return [Math.sin(r), Math.cos(r)];
}

/** Compass bearing (deg true) from point a to point b. */
export function bearingTo(a, b) {
  const d = Math.atan2(b[0] - a[0], b[1] - a[1]) / DEG;
  return d < 0 ? d + 360 : d;
}

export function dist3(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = (a[2] || 0) - (b[2] || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Position along a polyline track after travelling `metres` (clamped to the end). */
export function alongTrack(track, metres) {
  if (!Array.isArray(track) || track.length === 0) return [0, 0];
  if (track.length === 1) return [num(track[0][0], 0), num(track[0][1], 0)];
  let left = Math.max(0, metres);
  for (let i = 0; i < track.length - 1; i++) {
    const a = track[i];
    const b = track[i + 1];
    const dx = num(b[0], 0) - num(a[0], 0);
    const dy = num(b[1], 0) - num(a[1], 0);
    const L = Math.hypot(dx, dy);
    if (left <= L || i === track.length - 2) {
      const u = L > 0 ? Math.min(1, left / L) : 0;
      return [num(a[0], 0) + dx * u, num(a[1], 0) + dy * u];
    }
    left -= L;
  }
  const last = track[track.length - 1];
  return [num(last[0], 0), num(last[1], 0)];
}

/**
 * True if world point p (x, y) lies inside an object's rotated footprint (size w x d).
 * rotation_deg is clockwise seen from above (the Composer's convention): the object's local
 * x axis points along (cos th, -sin th) and its local y axis along (sin th, cos th).
 */
export function insideFootprint(p, obj, margin = 0) {
  const pos = obj.position || [0, 0, 0];
  const size = obj.size || [1, 1, 0];
  const th = num(obj.rotation_deg, 0) * DEG;
  const dx = p[0] - num(pos[0], 0);
  const dy = p[1] - num(pos[1], 0);
  const lx = dx * Math.cos(th) - dy * Math.sin(th);
  const ly = dx * Math.sin(th) + dy * Math.cos(th);
  return Math.abs(lx) <= num(size[0], 1) / 2 + margin && Math.abs(ly) <= num(size[1], 1) / 2 + margin;
}

/** Object-local (lx along width, ly along depth, lz up) to world coordinates (clockwise rotation_deg). */
export function localToWorld(obj, lx, ly, lz = 0) {
  const pos = obj.position || [0, 0, 0];
  const th = num(obj.rotation_deg, 0) * DEG;
  const c = Math.cos(th);
  const s = Math.sin(th);
  return [num(pos[0], 0) + lx * c + ly * s, num(pos[1], 0) - lx * s + ly * c, num(pos[2], 0) + lz];
}

/** Atmospheric lapse rate (K/km) used for the refraction regime from hour and wind. */
export function lapseRateKPerKm(hour, windMps, cloudOkta = 8) {
  const h = num(hour, 12);
  const day = h > 8 && h < 18;
  const calm = num(windMps, 4) < 3;
  if (!day && calm && num(cloudOkta, 8) < 5) return -4; // clear calm night: inversion, sound carries
  if (day && num(cloudOkta, 8) < 5) return 9.8; // sunny afternoon: strong upward refraction
  return 6.5;
}
