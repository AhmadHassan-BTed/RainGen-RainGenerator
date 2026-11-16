/**
 * Lightning channels and thunder synthesis for the browser engine.
 * Design: "signal-based, physics-inspired" (Fineberg; Reiss, Tez & Selfridge, AES 150, 2021).
 *
 * - Channel: tortuous piecewise-linear random walk, mean direction change ~16 deg per
 *   15-45 m step (Few 1969; LeVine & Meneghini 1978). Cloud-to-ground flashes descend from
 *   a cloud base of 1.5-3 km to the ground with branches and keep an in-cloud part that
 *   climbs toward the 5-7 km charge region; intracloud flashes run largely horizontally.
 *   The same geometry is returned to the UI so the drawn bolt and the sound are locked.
 * - Source: every channel element emits an N-wave (Ribner & Roy 1982). Stroke energy
 *   E0 = 10-200 kJ/m sets the relaxation radius R0 = sqrt(E0 / (pi p0)) and the source
 *   frequency ~ c0 sqrt(p0 / E0) (~485 Hz at 50 kJ/m, Few); the N-wave then lengthens
 *   with distance (weak-shock propagation). CG flashes carry 3-4 return strokes ~60 ms
 *   apart (Rakov & Uman 2003) that re-excite the channel below the cloud base.
 * - Propagation: delay r / c(T), spherical spreading 1/r, broadside directivity of each
 *   element, ISO 9613-1 absorption applied per log-distance band (14 bands share one
 *   low-pass), the refraction shadow r = sqrt(2c/a)(sqrt(h_s) + sqrt(h_r))
 *   with a = dc/dz from the lapse rate, and wind refraction c_eff = c + u cos(theta).
 * - Rumble: the arrival-energy envelope, convolved with an exponential terrain/cloud echo
 *   tail, modulates decorrelated low-passed noise: the signal-based layer that carries the
 *   roll of distant thunder.
 * - Scheduling: counter-based thinning at a fixed ceiling rate, so flash times are a pure
 *   function of (seed, source, 10 s cell): identical in real time and in offline renders.
 */

import { CounterRng, clamp, smoothstep, speedOfSound, absorptionCutoffHz, fractalNoise1D, hashString, seedKey } from './physics.js';

export const FLASH_CELL_SEC = 10;
export const FLASH_CEILING_PER_MIN = 30;
const P0 = 101325;
const TWO_PI = Math.PI * 2;

const num = (x, d) => {
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
};

// ---------------------------------------------------------------------------
// Source physics
// ---------------------------------------------------------------------------

/** Stroke energy per unit channel length (J/m) for a 0..1 intensity: 10 kJ/m .. 200 kJ/m. */
export function strokeEnergy(intensity = 0.5) {
  return 1e4 * Math.pow(20, clamp(num(intensity, 0.5), 0, 1));
}

/** Few's relaxation radius R0 = sqrt(E0 / (pi p0)) in metres. */
export function relaxationRadius(E0, p0 = P0) {
  return Math.sqrt(Math.max(1, num(E0, 5e4)) / (Math.PI * p0));
}

/** Thunder source frequency ~ c0 sqrt(p0 / E0) (Hz): bigger strokes are deeper. */
export function sourceFrequency(E0, tC = 20, p0 = P0) {
  return speedOfSound(tC) * Math.sqrt(p0 / Math.max(1, num(E0, 5e4)));
}

/**
 * Acoustic shadow radius (m) for upward refraction: r = sqrt(2c/a) (sqrt(h_s) + sqrt(h_r)),
 * a = dc/dz = (c / 2T) * lapse. Infinite (no shadow) for an inversion.
 */
export function shadowRadiusM(hSourceM, hReceiverM, tC = 20, lapseKPerKm = 6.5) {
  const lapse = num(lapseKPerKm, 6.5);
  if (!(lapse > 0)) return Infinity;
  const c = speedOfSound(tC);
  const a = (c / (2 * (num(tC, 20) + 273.15))) * (lapse / 1000);
  return Math.sqrt((2 * c) / a) * (Math.sqrt(Math.max(0, num(hSourceM, 0))) + Math.sqrt(Math.max(0, num(hReceiverM, 1.7))));
}

/**
 * Playback peak for a roll whose nearest point is rNear metres away. Physically the level
 * falls ~1/r over a 30 dB range; the soundscape keeps the ordering but compresses the span
 * (the signal-based shaping layer), so distant thunder stays audible and near cracks stay safe.
 */
export function thunderGain(rNearM, intensity = 0.5) {
  const r = Math.max(50, num(rNearM, 5000));
  return clamp(0.9 * Math.pow(500 / r, 0.55), 0.035, 0.95) * (0.6 + 0.4 * clamp(num(intensity, 0.5), 0, 1));
}

// ---------------------------------------------------------------------------
// Deterministic flash schedule
// ---------------------------------------------------------------------------

/**
 * Candidate flashes in one 10 s cell of a thunder source, drawn at the fixed ceiling rate.
 * A candidate becomes a flash when u < rate(t) / FLASH_CEILING_PER_MIN (thinning), so the
 * schedule never depends on how the rate changed earlier.
 */
export function flashCandidates(seed, sourceId, cell) {
  const c = Math.max(0, Math.floor(num(cell, 0)));
  const rng = new CounterRng(seed, `flashcell:${sourceId}`, c);
  const n = rng.poisson((FLASH_CEILING_PER_MIN / 60) * FLASH_CELL_SEC);
  const out = [];
  for (let j = 0; j < n; j++) {
    out.push({ t: (c + rng.next()) * FLASH_CELL_SEC, u: rng.next(), key: `${sourceId}:${c}:${j}` });
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

// ---------------------------------------------------------------------------
// Fractal channel
// ---------------------------------------------------------------------------

function normalize(d) {
  const n = Math.hypot(d[0], d[1], d[2]) || 1;
  d[0] /= n;
  d[1] /= n;
  d[2] /= n;
  return d;
}

/** Tortuous random walk; returns the list of points (world metres). */
function walk(rng, start, dir, o) {
  const pts = [start.slice()];
  const p = start.slice();
  const d = normalize(dir.slice());
  for (let i = 0; i < o.steps; i++) {
    d[0] += o.sigma * rng.normal();
    d[1] += o.sigma * rng.normal();
    d[2] += o.sigma * rng.normal();
    if (o.bias) {
      const b = o.bias(p);
      d[0] += b[0];
      d[1] += b[1];
      d[2] += b[2];
    }
    normalize(d);
    if (o.maxDz !== undefined && d[2] > o.maxDz) {
      const h = Math.hypot(d[0], d[1]);
      const k = Math.sqrt(Math.max(0, 1 - o.maxDz * o.maxDz));
      if (h > 1e-9) {
        d[0] = (d[0] / h) * k;
        d[1] = (d[1] / h) * k;
      } else {
        const a = rng.next() * TWO_PI;
        d[0] = Math.sin(a) * k;
        d[1] = Math.cos(a) * k;
      }
      d[2] = o.maxDz;
    }
    const L = o.stepM * (0.55 + 0.9 * rng.next());
    const prev = pts[pts.length - 1];
    p[0] += d[0] * L;
    p[1] += d[1] * L;
    p[2] += d[2] * L;
    if (o.floorZ !== undefined && p[2] <= o.floorZ) {
      const f = (prev[2] - o.floorZ) / (prev[2] - p[2] || 1);
      pts.push([prev[0] + (p[0] - prev[0]) * f, prev[1] + (p[1] - prev[1]) * f, o.floorZ]);
      break;
    }
    pts.push(p.slice());
    if (o.onStep) o.onStep(p, d);
  }
  return pts;
}

function lineLength(pts) {
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
  return s;
}

function rounded(pts, maxPoints = 0) {
  let src = pts;
  if (maxPoints > 1 && pts.length > maxPoints) {
    src = [];
    const step = (pts.length - 1) / (maxPoints - 1);
    for (let i = 0; i < maxPoints; i++) src.push(pts[Math.round(i * step)]);
  }
  return src.map((p) => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10, Math.round(p[2] * 10) / 10]);
}

/**
 * Build a fractal 3-D lightning channel in world coordinates (x east, y north, z up).
 * opts: { seed, key, kind: 'cg'|'ic', strike: [x, y], intensity, cloudBaseM, topM }
 * Returns { kind, strike, cloudBase, top, channel, branches, cloud, lines, strokes, lengthM }:
 * `channel` is the visible bolt (cloud base to ground for CG), `lines` the acoustic model
 * ({ pts, w: relative source strength per metre, stroke: re-excited by return strokes }).
 */
export function buildChannel(opts = {}) {
  const kind = opts.kind === 'ic' ? 'ic' : 'cg';
  const key = String(opts.key ?? 'flash');
  const rng = new CounterRng(opts.seed ?? 0, `channel:${key}`, 0);
  const sx = num(opts.strike && opts.strike[0], 0);
  const sy = num(opts.strike && opts.strike[1], 0);
  const cloudBase = clamp(num(opts.cloudBaseM, 1500 + 1500 * rng.next()), 600, 4000);
  const top = clamp(num(opts.topM, 5000 + 2000 * rng.next()), cloudBase + 800, 12000);
  const lines = [];
  let channel = [];
  let branches = [];
  let cloud = [];
  let strokes = [{ t: 0, g: 1 }];
  let strike;

  if (kind === 'cg') {
    strike = [sx, sy, 0];
    const stepM = clamp(cloudBase / 75, 15, 40);
    const S0 = [sx + 300 * rng.normal(), sy + 300 * rng.normal(), cloudBase];
    const seeds = [];
    const main = walk(rng, S0, [0.12 * rng.normal(), 0.12 * rng.normal(), -1], {
      steps: 900,
      stepM,
      sigma: 0.28,
      floorZ: 0,
      maxDz: -0.12,
      bias: (p) => {
        const dx = sx - p[0];
        const dy = sy - p[1];
        const dz = -p[2];
        const n = Math.hypot(dx, dy, dz) || 1;
        return [(0.1 * dx) / n, (0.1 * dy) / n, (0.1 * dz) / n - 0.04];
      },
      onStep: (p, d) => {
        const frac = clamp(p[2] / cloudBase, 0, 1);
        if (seeds.length < 22 && rng.next() < 0.012 + 0.07 * frac * frac) seeds.push({ p: p.slice(), d: d.slice(), frac });
      },
    });
    if (main[main.length - 1][2] > 0) main.push([main[main.length - 1][0], main[main.length - 1][1], 0]);
    const branchLines = [];
    for (const b of seeds) {
      const dir = normalize([b.d[0] + 0.9 * rng.normal(), b.d[1] + 0.9 * rng.normal(), b.d[2] - 0.2]);
      const steps = Math.round(4 + rng.exponential(1) * 14 * (0.3 + b.frac));
      const line = walk(rng, b.p, dir, { steps, stepM: stepM * 0.8, sigma: 0.32, floorZ: 15, maxDz: -0.05 });
      if (line.length > 1) branchLines.push(line);
      if (line.length > 4 && rng.next() < 0.3) {
        const at = line[Math.floor(line.length / 2)];
        const sub = walk(rng, at, [rng.normal(), rng.normal(), -0.8], { steps: Math.max(2, Math.round(steps / 2)), stepM: stepM * 0.7, sigma: 0.35, floorZ: 15, maxDz: -0.05 });
        if (sub.length > 1) branchLines.push(sub);
      }
    }
    // Shear so the walk lands exactly on the strike point (a function of height only, so
    // branches stay attached to the main channel).
    const end = main[main.length - 1];
    const ex = sx - end[0];
    const ey = sy - end[1];
    const shear = (pts) => {
      for (const p of pts) {
        const f = clamp(1 - p[2] / cloudBase, 0, 1);
        p[0] += ex * f;
        p[1] += ey * f;
      }
    };
    shear(main);
    for (const l of branchLines) shear(l);

    // In-cloud part: climbs from the cloud base toward the charge region, mostly horizontal.
    const zc = cloudBase + (top - cloudBase) * (0.45 + 0.4 * rng.next());
    const az = rng.next() * TWO_PI;
    const horiz = 2000 + 4000 * rng.next();
    const cstep = 70;
    const cloudBias = (p) => [0, 0, clamp((zc - p[2]) / 3000, -0.25, 0.25)];
    const cloudMain = walk(rng, main[0], [Math.sin(az), Math.cos(az), 0.35], { steps: Math.round(horiz / cstep), stepM: cstep, sigma: 0.22, bias: cloudBias });
    const cloudLines = [cloudMain];
    const forks = 1 + Math.floor(rng.next() * 2);
    for (let f = 0; f < forks && cloudMain.length > 4; f++) {
      const at = cloudMain[1 + Math.floor(rng.next() * (cloudMain.length - 2))];
      const a2 = rng.next() * TWO_PI;
      const fork = walk(rng, at, [Math.sin(a2), Math.cos(a2), 0.1], { steps: 15 + Math.floor(rng.next() * 20), stepM: cstep, sigma: 0.25, bias: cloudBias });
      cloudLines.push(fork);
    }

    lines.push({ pts: main, w: 1, stroke: true });
    for (const l of branchLines) lines.push({ pts: l, w: 0.35, stroke: false });
    for (const l of cloudLines) lines.push({ pts: l, w: 0.55, stroke: false });

    // Return strokes: 1 + Poisson(2.2), capped at 6; intervals ~60 ms lognormal.
    const n = 1 + Math.min(5, rng.poisson(2.2));
    strokes = [{ t: 0, g: 1 }];
    let t = 0;
    for (let s = 1; s < n; s++) {
      t += clamp(0.06 * Math.exp(0.5 * rng.normal()), 0.015, 0.25);
      strokes.push({ t, g: 0.35 + 0.35 * rng.next() });
    }
    channel = rounded(main, 200);
    branches = branchLines.map((l) => rounded(l, 40));
    cloud = cloudLines.map((l) => rounded(l, 60));
  } else {
    const zc = 3500 + 3000 * rng.next();
    strike = [sx, sy, zc];
    const az = rng.next() * TWO_PI;
    const half = 2500 + 4000 * rng.next();
    const step = 60;
    const bias = (p) => [0, 0, clamp((zc - p[2]) / 2500, -0.2, 0.2)];
    const a = walk(rng, strike, [Math.sin(az), Math.cos(az), 0.05], { steps: Math.round(half / step), stepM: step, sigma: 0.22, bias });
    const b = walk(rng, strike, [-Math.sin(az), -Math.cos(az), -0.05], { steps: Math.round(half / step), stepM: step, sigma: 0.22, bias });
    const main = b.reverse().concat(a.slice(1));
    lines.push({ pts: main, w: 0.7, stroke: false });
    const forkLines = [];
    const nf = 2 + Math.floor(rng.next() * 3);
    for (let f = 0; f < nf; f++) {
      const at = main[Math.floor(rng.next() * main.length)];
      const a2 = rng.next() * TWO_PI;
      const fork = walk(rng, at, [Math.sin(a2), Math.cos(a2), 0.05 * rng.normal()], { steps: 12 + Math.floor(rng.next() * 25), stepM: step, sigma: 0.26, bias });
      forkLines.push(fork);
      lines.push({ pts: fork, w: 0.4, stroke: false });
    }
    channel = rounded(main, 200);
    branches = forkLines.map((l) => rounded(l, 40));
  }

  let lengthM = 0;
  for (const l of lines) lengthM += lineLength(l.pts);
  return { kind, strike, cloudBase, top, channel, branches, cloud, lines, strokes, lengthM };
}

/** Nearest distance (m) from a point to any vertex of the channel's acoustic lines. */
export function nearestDistance(chan, ear) {
  let best = Infinity;
  for (const l of (chan && chan.lines) || []) {
    for (const p of l.pts) {
      const d = Math.hypot(p[0] - ear[0], p[1] - ear[1], p[2] - ear[2]);
      if (d < best) best = d;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------

function lowpassCoefs(fc, sr, q = Math.SQRT1_2) {
  const w = (TWO_PI * clamp(fc, 5, 0.49 * sr)) / sr;
  const al = Math.sin(w) / (2 * q);
  const cw = Math.cos(w);
  const a0 = 1 + al;
  return [(1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0, (-2 * cw) / a0, (1 - al) / a0];
}

function highpassCoefs(fc, sr, q = Math.SQRT1_2) {
  const w = (TWO_PI * clamp(fc, 5, 0.49 * sr)) / sr;
  const al = Math.sin(w) / (2 * q);
  const cw = Math.cos(w);
  const a0 = 1 + al;
  return [(1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0, (-2 * cw) / a0, (1 - al) / a0];
}

function biquadInPlace(x, c) {
  let s1 = 0;
  let s2 = 0;
  const [b0, b1, b2, a1, a2] = c;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const y = b0 * v + s1;
    s1 = b1 * v - a1 * y + s2;
    s2 = b2 * v - a2 * y;
    x[i] = y;
  }
}

/**
 * Synthesize the thunder of one flash heard at `ear` (world [x, y, z], metres).
 * opts: { seed, key, tempC, humidityPct, intensity, lapseKPerKm, windMps, windFromDeg,
 *         sampleRate, maxRollSec, maxAudibleM }
 * Returns null when inaudible, else { sampleRate, direct (mono, peak 1), rumble [L, R],
 * offsetSec (buffer start after the flash), durationSec, nearestM, farthestM, centroid,
 * gain (recommended playback peak) }.
 */
export function synthesizeThunder(chan, ear, opts = {}) {
  if (!chan || !Array.isArray(chan.lines) || !chan.lines.length) return null;
  const tC = num(opts.tempC, 15);
  const rh = clamp(num(opts.humidityPct, 80), 5, 100);
  const intensity = clamp(num(opts.intensity, 0.5), 0, 1);
  const c = speedOfSound(tC);
  const rng = new CounterRng(opts.seed ?? 0, `thunder:${opts.key ?? ''}`, 0);
  const E = [num(ear && ear[0], 0), num(ear && ear[1], 0), num(ear && ear[2], 1.7)];
  const lapse = num(opts.lapseKPerKm, 6.5);
  const windU = Math.max(0, num(opts.windMps, 0));
  const windFrom = (num(opts.windFromDeg, 270) * Math.PI) / 180;
  const maxRoll = clamp(num(opts.maxRollSec, 38), 4, 60);
  const strokes = Array.isArray(chan.strokes) && chan.strokes.length ? chan.strokes : [{ t: 0, g: 1 }];
  const strokeSpan = strokes[strokes.length - 1].t;

  // 1) Channel elements: arrival distance, amplitude, return-stroke flag.
  const er = [];
  const ea = [];
  const es = [];
  let cx = 0;
  let cy = 0;
  let cz = 0;
  let cw = 0;
  for (const line of chan.lines) {
    const pts = line.pts || [];
    const w = num(line.w, 1);
    for (let i = 0; i + 1 < pts.length; i++) {
      const A = pts[i];
      const B = pts[i + 1];
      const dx = B[0] - A[0];
      const dy = B[1] - A[1];
      const dz = B[2] - A[2];
      const len = Math.hypot(dx, dy, dz);
      if (!(len > 0.01)) continue;
      const K = clamp(Math.ceil(len / 7), 1, 10);
      for (let j = 0; j < K; j++) {
        const f = (j + 0.5 + 0.7 * (rng.next() - 0.5)) / K;
        const px = A[0] + dx * f;
        const py = A[1] + dy * f;
        const pz = A[2] + dz * f;
        const vx = px - E[0];
        const vy = py - E[1];
        const vz = pz - E[2];
        const r = Math.max(5, Math.hypot(vx, vy, vz));
        const cosT = Math.abs(vx * dx + vy * dy + vz * dz) / (r * len);
        const dir = 0.3 + 0.7 * Math.sqrt(Math.max(0, 1 - cosT * cosT));
        const hd = Math.hypot(vx, vy);
        const rs = shadowRadiusM(pz, E[2], tC, lapse);
        const sh = hd > rs ? Math.max(0.02, Math.exp((-4 * (hd - rs)) / rs)) : 1;
        let wf = 1;
        if (windU > 0.5 && hd > 200) {
          // +1 when the source lies upwind, so its sound travels with the wind.
          const cwd = Math.cos(Math.atan2(vx, vy) - windFrom);
          const k = Math.min(1.5, windU / 10);
          wf = 1 + 0.25 * k * cwd;
          if (cwd < 0) wf *= 1 + 0.6 * cwd * Math.min(1, windU / 12) * smoothstep(1000, 6000, hd);
          wf = Math.max(0.05, wf);
        }
        const a = ((w * (len / K)) / r) * dir * sh * wf * Math.max(0.2, 1 + 0.3 * rng.normal());
        er.push(r);
        ea.push(a);
        es.push(line.stroke ? 1 : 0);
        const a2 = a * a;
        cx += px * a2;
        cy += py * a2;
        cz += pz * a2;
        cw += a2;
      }
    }
  }
  const n = er.length;
  if (!n || !(cw > 0)) return null;
  let rNear = Infinity;
  let rFar = 0;
  for (let i = 0; i < n; i++) {
    if (er[i] < rNear) rNear = er[i];
    if (er[i] > rFar) rFar = er[i];
  }
  if (rNear > num(opts.maxAudibleM, 35000)) return null;

  const sr = Math.round(clamp(num(opts.sampleRate, rNear < 1500 ? 48000 : rNear < 5000 ? 24000 : 16000), 8000, 192000));
  const pre = 0.03;
  const rEnd = Math.min(rFar, rNear + maxRoll * c);
  const tail = 0.8 + 3.2 * smoothstep(300, 12000, rNear);
  const durSec = pre + (rEnd - rNear) / c + strokeSpan + tail;
  const N = Math.ceil(durSec * sr);
  const direct = new Float32Array(N);
  const T0 = 1 / sourceFrequency(strokeEnergy(intensity), tC);
  const tN = (r) => T0 * Math.pow(1 + 2 * Math.log(1 + r / 250), 0.75);

  // 2) Distance bands sharing one absorption low-pass each.
  const NB = 14;
  const lr0 = Math.log(rNear);
  const lr1 = Math.log(Math.max(rEnd, rNear * 1.001));
  const lists = [];
  for (let b = 0; b < NB; b++) lists.push([]);
  for (let i = 0; i < n; i++) {
    if (er[i] > rEnd) continue;
    lists[clamp(Math.floor(((Math.log(er[i]) - lr0) / (lr1 - lr0)) * NB), 0, NB - 1)].push(i);
  }
  const dG = 0.01;
  const G = Math.ceil(durSec / dG) + 2;
  const grid = new Float64Array(G);
  let ns = (hashString(`noise:${opts.key ?? ''}`) ^ seedKey(opts.seed ?? 0) | 1) | 0;
  const white = () => {
    ns ^= ns << 13;
    ns ^= ns >>> 17;
    ns ^= ns << 5;
    return ns / 2147483648;
  };
  for (let b = 0; b < NB; b++) {
    const L = lists[b];
    if (!L.length) continue;
    const rLo = Math.exp(lr0 + (b / NB) * (lr1 - lr0));
    const rHi = Math.exp(lr0 + ((b + 1) / NB) * (lr1 - lr0));
    const i0 = Math.max(0, Math.floor(((rLo - rNear) / c + pre - 0.002) * sr));
    const wMax = Math.ceil(tN(rHi) * sr) + 2;
    const i1 = Math.min(N, Math.ceil(((rHi - rNear) / c + pre + strokeSpan) * sr) + wMax + Math.ceil(0.08 * sr));
    if (i1 <= i0) continue;
    const buf = new Float32Array(i1 - i0);
    for (const i of L) {
      const r = er[i];
      const a = ea[i];
      const W = Math.max(2, Math.round(tN(r) * sr));
      const inv = 2 / (W - 1);
      for (let s = 0; s < strokes.length; s++) {
        if (s > 0 && !es[i]) break;
        const amp = a * (s === 0 ? 1 : num(strokes[s].g, 0.5));
        const t = (r - rNear) / c + pre + num(strokes[s].t, 0);
        const k0 = Math.round(t * sr) - i0;
        for (let k = 0; k < W; k++) {
          const j = k0 + k;
          if (j >= 0 && j < buf.length) buf[j] += amp * (1 - k * inv);
        }
        const gi = Math.floor(t / dG);
        if (gi >= 0 && gi < G) grid[gi] += amp * amp * W;
        if (s === 0 && r < 1200) {
          // Near strikes: the tearing crack of many tiny, sharp shocks.
          const Lc = Math.max(4, Math.round((0.0003 + 0.0009 * rng.next()) * sr));
          const ca = amp * 0.8 * (1 - r / 1200);
          for (let k = 0; k < Lc; k++) {
            const j = k0 + k;
            if (j >= 0 && j < buf.length) buf[j] += ca * white() * (1 - k / Lc);
          }
        }
      }
    }
    const rMid = Math.sqrt(rLo * rHi);
    const fc = Math.min(0.45 * sr, 1.15 * absorptionCutoffHz(rMid, tC, rh, 6, 0.45 * sr));
    if (fc < 0.44 * sr) {
      const co = lowpassCoefs(fc, sr);
      biquadInPlace(buf, co);
      biquadInPlace(buf, co);
    }
    for (let k = 0; k < buf.length; k++) direct[i0 + k] += buf[k];
  }
  biquadInPlace(direct, highpassCoefs(18, sr));

  let peak = 0;
  let ss = 0;
  for (let i = 0; i < N; i++) {
    const v = Math.abs(direct[i]);
    if (v > peak) peak = v;
    ss += direct[i] * direct[i];
  }
  if (!(peak > 0)) return null;
  const inv = 1 / peak;
  for (let i = 0; i < N; i++) direct[i] *= inv;
  const directRms = Math.sqrt(ss / N) * inv;

  // 3) Rumble: smoothed arrival energy + exponential echo tail modulating low noise.
  const aS = Math.exp(-dG / 0.06);
  let acc = 0;
  for (let k = 0; k < G; k++) {
    acc = aS * acc + (1 - aS) * grid[k];
    grid[k] = acc;
  }
  acc = 0;
  for (let k = G - 1; k >= 0; k--) {
    acc = aS * acc + (1 - aS) * grid[k];
    grid[k] = acc;
  }
  const tauE = 0.5 + 2.4 * smoothstep(300, 12000, rNear);
  const aE = Math.exp(-dG / tauE);
  const env = new Float64Array(G);
  let tailE = 0;
  for (let k = 0; k < G; k++) {
    tailE = aE * tailE + grid[k];
    env[k] = Math.sqrt(grid[k] + 0.5 * tailE * (dG / tauE));
  }
  const fr = clamp(0.5 * absorptionCutoffHz(Math.max(rNear, 100), tC, rh, 6, 4000), 60, 160);
  const a1 = 1 - Math.exp((-TWO_PI * fr) / sr);
  const a2 = 1 - Math.exp((-TWO_PI * fr * 2) / sr);
  const rumble = [new Float32Array(N), new Float32Array(N)];
  const kBase = seedKey(opts.seed ?? 0) ^ hashString(`rumble:${opts.key ?? ''}`);
  for (let ch = 0; ch < 2; ch++) {
    const out = rumble[ch];
    const mk = (kBase + Math.imul(ch + 1, 0x9e3779b9)) >>> 0;
    let y1 = 0;
    let y2 = 0;
    let m0 = Math.exp(0.7 * fractalNoise1D(mk, 0, 3));
    let m1 = m0;
    for (let i = 0; i < N; i++) {
      const jb = i & 255;
      if (jb === 0) {
        m0 = m1;
        m1 = Math.exp(0.7 * fractalNoise1D(mk, ((i + 256) / sr) * 3, 3));
      }
      const mod = m0 + (m1 - m0) * (jb / 256);
      y1 += a1 * (white() - y1);
      y2 += a2 * (y1 - y2);
      const g = (i / sr) / dG;
      const k = Math.floor(g);
      const e = k + 1 < G ? env[k] + (env[k + 1] - env[k]) * (g - k) : env[G - 1];
      out[i] = y2 * e * mod;
    }
    biquadInPlace(out, highpassCoefs(20, sr));
  }
  let rs2 = 0;
  for (let ch = 0; ch < 2; ch++) for (let i = 0; i < N; i++) rs2 += rumble[ch][i] * rumble[ch][i];
  const rumbleRms = Math.sqrt(rs2 / (2 * N));
  const ratio = 0.45 + 1.1 * smoothstep(1000, 15000, rNear);
  if (rumbleRms > 0) {
    const g = (ratio * directRms) / rumbleRms;
    for (let ch = 0; ch < 2; ch++) for (let i = 0; i < N; i++) rumble[ch][i] *= g;
  }

  return {
    sampleRate: sr,
    direct,
    rumble,
    offsetSec: rNear / c - pre,
    durationSec: N / sr,
    nearestM: rNear,
    farthestM: rFar,
    centroid: [cx / cw, cy / cw, cz / cw],
    gain: thunderGain(rNear, intensity),
  };
}

function resampleLinear(x, srIn, srOut) {
  if (srIn === srOut) return x;
  const n = Math.max(1, Math.round((x.length * srOut) / srIn));
  const y = new Float32Array(n);
  const k = srIn / srOut;
  for (let i = 0; i < n; i++) {
    const p = i * k;
    const j = Math.floor(p);
    const f = p - j;
    y[i] = j + 1 < x.length ? x[j] + (x[j + 1] - x[j]) * f : x[x.length - 1];
  }
  return y;
}

function fillBuffer(buf, ch, data) {
  if (typeof buf.copyToChannel === 'function') buf.copyToChannel(data, ch, 0);
  else buf.getChannelData(ch).set(data);
}

/** AudioBuffers for a synthesized roll: { direct (1 ch), rumble (2 ch) }; resamples if the rate is refused. */
export function createThunderBuffers(ctx, syn) {
  let sr = syn.sampleRate;
  let d = syn.direct;
  let l = syn.rumble[0];
  let r = syn.rumble[1];
  let direct = null;
  let rumble = null;
  try {
    direct = ctx.createBuffer(1, d.length, sr);
    rumble = ctx.createBuffer(2, l.length, sr);
  } catch (e) {
    sr = ctx.sampleRate;
    d = resampleLinear(d, syn.sampleRate, sr);
    l = resampleLinear(l, syn.sampleRate, sr);
    r = resampleLinear(r, syn.sampleRate, sr);
    direct = ctx.createBuffer(1, d.length, sr);
    rumble = ctx.createBuffer(2, l.length, sr);
  }
  fillBuffer(direct, 0, d);
  fillBuffer(rumble, 0, l);
  fillBuffer(rumble, 1, r);
  return { direct, rumble };
}
