//! Thunder (port of raingen/synth/thunder.py): fractal lightning channels, Ribner & Roy N-wave
//! synthesis, ISO 9613-1 absorption, refraction, rumble, crack and terrain / building echoes.
//!
//! * Channel geometry: a multi-scale tortuous 3-D walk (mean turn 16 deg, Few 1969; Hill 1968).
//!   CG flashes start in the charge region (4.5-6.5 km), cross a 1-3 km cloud base and end on
//!   the ground, with in-cloud and air-terminated branches; IC flashes run horizontally at 5-8 km.
//! * Every straight segment radiates a weak-shock N-wave (Few's relaxation radius
//!   R0 = sqrt(E0 / (pi p0)), period ~ (r / R0)^(1/4), peak ~ r^-1 (R0 / r)^(1/4)), smeared over its
//!   Ribner & Roy arrival window l |cos psi| / c; segment kernels are summed on a 64-direction beam
//!   grid, re-encoded to 3rd-order SH, and the ISO 9613-1 absorption of the range r = c (t - t_flash)
//!   is applied per block by a 50 % sine-window STFT. Refraction (lapse + wind shear, upward
//!   refraction shadow) scales each segment. CG flashes carry 3-5 return strokes.
//! * Signal layer: a crack / tearing layer for near strikes, a rumbler (noise with the N-wave source
//!   spectrum, growl following the smoothed arrival power) and a deepener (sub-bass afterglow),
//!   both absorbed per 8192-sample frame at their arrival range; terrain / building echoes and a
//!   diffuse terrain tail (SH FDN room).
//! * Flashes follow the World's StormSystem schedule exactly (NumPy-compatible "storm.flash.<cell>"
//!   streams per block slot). Every flash emits a 'lightning' marker at t_flash and a 'thunder'
//!   marker at the first arrival t_flash + r_min / c, audible or not.
//!
//! Allocation happens only when a flash is spawned (channel geometry, segment table, buffers of
//! one event), on the render thread, never in the audio callback.

use super::common::*;
use super::npy::{self, NpRng};
use super::places::Room;
use crate::dsp::{Biquad, BiquadCoefs};
use crate::engine::{BlockCtx, BuildCfg, Voice};
use crate::rng::{combine, hash3, hash_str, Rng};
use crate::types::{bearing_to_azimuth, stability, Emission, Patch, Scene, Stem, Weather, N_AMBI};
use realfft::num_complex::Complex;
use std::f64::consts::PI;
use std::fmt::Write as _;

const MEAN_TURN_DEG: f64 = 16.0;
const MAX_EVENT_S: f64 = 40.0;
const N_WAVE_K: f64 = 4.5;
const RUMBLER_POWER: f64 = 0.8;
const DEEPENER_POWER: f64 = 0.45;
const BIN_S: f64 = 0.01;
const AFTERGLOW_TAU_S: f64 = 1.0;
/// Rumbler, deepener and a second, independent rumbler from the other side of the channel.
const RUMBLE_ROWS: usize = 3;
const RUMBLE_HOP: usize = 4096;
const CRACK_RANGE_M: f64 = 3500.0;
const TEAR_RANGE_M: f64 = 1500.0;
const N_BEAMS: usize = 64;
const MAX_EVENTS: usize = 16;
const P0: f64 = 101_325.0;
const ECHO_MAX_DELAY_S: f64 = 8.0;
const MAX_FLASHES_PER_SLOT: usize = 64;

/// Stroke intensity 0..1 -> energy per unit channel length E0 (1e4 .. 5e5 J/m, log scale).
pub fn stroke_energy_j_per_m(intensity: f64) -> f64 {
    1.0e4 * 50f64.powf(intensity.clamp(0.0, 1.0))
}

/// Few's characteristic thunder frequency f ~ c0 sqrt(p0 / E0).
pub fn source_frequency_hz(e0_j_per_m: f64, temp_c: f64) -> f64 {
    c_air(temp_c) * (P0 / e0_j_per_m.max(1.0)).sqrt()
}

#[inline]
fn c_air(temp_c: f64) -> f64 {
    331.3 * (1.0 + temp_c / 273.15).sqrt()
}

/// Distance to the upward-refraction shadow zone (9.8 K/km lapse).
fn shadow_distance_m(hs: f64, hr: f64, temp_c: f64) -> f64 {
    let t = temp_c + 273.15;
    let c = c_air(temp_c);
    let a = c / (2.0 * t) * (9.8 / 1000.0);
    (2.0 * c / a).sqrt() * (hs.max(0.0).sqrt() + hr.max(0.0).sqrt())
}

fn dtdz_k_per_m(stab: u8) -> f64 {
    match stab {
        stability::STABLE => 0.06,
        stability::UNSTABLE => -0.06,
        _ => -0.0098,
    }
}

/// raingen.physics.atmosphere.refraction_gain_db (vectorised form of thunder.refraction_gains_db).
fn refraction_db(r: f64, wcomp: f64, stab: u8, h_src: f64, h_rcv: f64, temp_c: f64) -> f64 {
    let hs = h_src.max(0.0);
    let hr = h_rcv.max(0.0);
    let c = c_air(temp_c);
    let tk = temp_c + 273.15;
    let a_t = c / (2.0 * tk) * dtdz_k_per_m(stab);
    let inv_ln = 1.0 / (10.0f64 / 0.03).ln();
    let z_m = (hs.max(0.5) * hr.max(0.5)).sqrt().max(2.0);
    let a = a_t + wcomp * inv_ln / z_m;
    if a < -1e-6 && r > 1.0 {
        let r_s = (2.0 * c / -a).sqrt() * (hs.max(0.05).sqrt() + hr.max(0.05).sqrt());
        let x = r / r_s;
        if x <= 0.7 {
            0.0
        } else {
            (-20.0 * (1.0 - (-2.0 * (x - 0.7)).exp())).max(-20.0)
        }
    } else if a > 1e-6 && r > 1.0 {
        let r_c = (8.0 * (c / a) * (0.5 * (hs + hr)).max(1.0)).sqrt();
        if r <= r_c {
            0.0
        } else {
            (10.0 * (r.max(1e-9) / r_c).log10()).min(10.0)
        }
    } else {
        0.0
    }
}

/// 2nd-order Butterworth low-pass at the frequency where the ISO 9613-1 loss over `distance_m`
/// reaches 3 dB (thunder.absorption_lowpass_sos); identity when it never does.
fn absorption_lowpass(distance_m: f64, temp_c: f32, rh_pct: f32, sr: f32) -> BiquadCoefs {
    let iso = Iso9613::new(temp_c, rh_pct);
    let f_hi = 0.45 * sr as f64;
    let (lo, hi) = (50f64.ln(), f_hi.ln());
    for i in 0..400 {
        let f = (lo + (hi - lo) * i as f64 / 399.0).exp();
        if iso.alpha(f) * distance_m.max(0.0) >= 3.0 {
            return lp2(f as f32, sr);
        }
    }
    BiquadCoefs::identity()
}

// --------------------------------------------------------------------------
// channel geometry
// --------------------------------------------------------------------------
type P3 = [f64; 3];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Main,
    Cloud,
    Branch,
    Ic,
}

#[inline]
fn norm3(x: f64, y: f64, z: f64) -> P3 {
    let r = (x * x + y * y + z * z).sqrt();
    if r <= 1e-12 {
        [0.0, 0.0, -1.0]
    } else {
        [x / r, y / r, z / r]
    }
}

#[inline]
fn len3(v: P3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn sub3(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Rodrigues rotation of unit d by `angle` about the axis `a` made perpendicular to d.
#[inline]
fn turn3(d: P3, angle: f64, a: P3) -> P3 {
    let p = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
    let a = norm3(a[0] - p * d[0], a[1] - p * d[1], a[2] - p * d[2]);
    let (s, c) = angle.sin_cos();
    norm3(
        d[0] * c + (a[1] * d[2] - a[2] * d[1]) * s,
        d[1] * c + (a[2] * d[0] - a[0] * d[2]) * s,
        d[2] * c + (a[0] * d[1] - a[1] * d[0]) * s,
    )
}

#[inline]
fn ri(rng: &mut Rng, lo: i64, hi: i64) -> i64 {
    if hi <= lo {
        lo
    } else {
        lo + rng.below((hi - lo) as u64) as i64
    }
}

#[inline]
fn turn_sigma() -> f64 {
    MEAN_TURN_DEG.to_radians() * (PI / 2.0).sqrt()
}

/// Tortuous walk from start to end: half-normal turns (mean 16 deg) steered to the target.
fn walk(rng: &mut Rng, start: P3, end: P3, step: f64, steer: f64, z_floor: f64) -> Vec<P3> {
    let sigma = turn_sigma();
    let mut p = start;
    let mut d = norm3(end[0] - p[0], end[1] - p[1], end[2] - p[2]);
    let mut pts = vec![p];
    let dist0 = len3(sub3(end, start));
    let iters = 5000usize.min((4.0 * dist0 / step) as usize + 16);
    for _ in 0..iters {
        let t = sub3(end, p);
        let dist = len3(t);
        if dist <= step {
            break;
        }
        let st = if dist > 3.0 * step { steer } else { steer.max(0.6) };
        d = norm3(
            (1.0 - st) * d[0] + st * t[0] / dist,
            (1.0 - st) * d[1] + st * t[1] / dist,
            (1.0 - st) * d[2] + st * t[2] / dist,
        );
        let ax = [rng.normal(), rng.normal(), rng.normal()];
        let g_turn = rng.normal();
        let g_len = rng.normal();
        d = turn3(d, (g_turn * sigma).abs(), ax);
        let l = (step * (0.4 * g_len).exp()).min(dist);
        p = [p[0] + d[0] * l, p[1] + d[1] * l, p[2] + d[2] * l];
        if p[2] < z_floor {
            p[2] = z_floor;
            d = norm3(d[0], d[1], d[2].abs());
        }
        pts.push(p);
    }
    pts.push(end);
    pts
}

/// Micro zig-zags (~35 m steps) between consecutive waypoints.
fn fill(rng: &mut Rng, way: &[P3], micro: f64, z_floor: f64) -> Vec<P3> {
    let mut pts = vec![way[0]];
    for w in way.windows(2) {
        let seg = walk(rng, w[0], w[1], micro, 0.18, z_floor);
        pts.extend_from_slice(&seg[1..]);
    }
    pts
}

/// Two-scale tortuosity: km-scale meander through waypoints, filled with ~35 m zig-zags.
fn tortuous(rng: &mut Rng, start: P3, end: P3, z_floor: f64) -> Vec<P3> {
    let way = walk(rng, start, end, 450.0, 0.3, z_floor);
    fill(rng, &way, 35.0, z_floor)
}

/// The channel below the cloud base: a descending stepped leader ending on the strike point.
fn leader(rng: &mut Rng, a: P3, b: P3, z_floor: f64) -> Vec<P3> {
    let macro_m = 450.0;
    let wander = 0.3;
    let levels = (((a[2] - b[2]).max(1.0) / macro_m).ceil() as usize).max(1);
    let mut cum = vec![[0.0f64; 2]; levels];
    let mut acc = [0.0f64; 2];
    for c in cum.iter_mut() {
        acc[0] += rng.normal() * wander * macro_m;
        acc[1] += rng.normal() * wander * macro_m;
        *c = acc;
    }
    let last = cum[levels - 1];
    let mut way = Vec::with_capacity(levels + 1);
    way.push(a);
    for (i, c) in cum.iter().enumerate() {
        let frac = (i + 1) as f64 / levels as f64;
        let mut p = [a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]), a[2] + frac * (b[2] - a[2])];
        p[0] += c[0] - frac * last[0];
        p[1] += c[1] - frac * last[1];
        way.push(p);
    }
    let l = way.len();
    way[l - 1] = b;
    fill(rng, &way, 35.0, z_floor)
}

#[allow(clippy::too_many_arguments)]
fn free_walk(
    rng: &mut Rng,
    start: P3,
    d0: P3,
    length: f64,
    step: f64,
    bias: P3,
    bias_w: f64,
    z_floor: f64,
    z_ceiling: f64,
) -> Vec<P3> {
    let sigma = turn_sigma();
    let mut p = start;
    let mut d = norm3(d0[0], d0[1], d0[2]);
    let b = norm3(bias[0], bias[1], bias[2]);
    let mut pts = vec![p];
    let mut total = 0.0;
    while total < length {
        d = norm3(
            (1.0 - bias_w) * d[0] + bias_w * b[0],
            (1.0 - bias_w) * d[1] + bias_w * b[1],
            (1.0 - bias_w) * d[2] + bias_w * b[2],
        );
        let ax = [rng.normal(), rng.normal(), rng.normal()];
        let g_turn = rng.normal();
        let g_len = rng.normal();
        d = turn3(d, (g_turn * sigma).abs(), ax);
        let l = step * (0.4 * g_len).exp();
        p = [p[0] + d[0] * l, p[1] + d[1] * l, p[2] + d[2] * l];
        total += l;
        if p[2] < z_floor {
            break;
        }
        if p[2] > z_ceiling {
            p[2] = z_ceiling;
            d = [d[0], d[1], -d[2].abs()];
        }
        pts.push(p);
    }
    pts
}

/// Polylines [(points, kind)] in the listener frame and the cloud-base height (listener z).
fn lightning_tree(rng: &mut Rng, sx: f64, sy: f64, cg: bool, zg: f64) -> (Vec<(Vec<P3>, Kind)>, f64) {
    let zb_km = rng.range(1.0, 3.0);
    let z_b = zg + 1000.0 * zb_km;
    let mut out: Vec<(Vec<P3>, Kind)> = Vec::new();
    if cg {
        let zt_km = rng.range(4.5, 6.5);
        let z_t = (zg + 1000.0 * zt_km).max(z_b + 1000.0);
        let ang = rng.range(0.0, 2.0 * PI);
        let off = rng.range(500.0, 3000.0);
        let o = [sx + off * ang.cos(), sy + off * ang.sin(), z_t];
        let bx = sx + rng.gauss(0.0, 250.0);
        let by = sy + rng.gauss(0.0, 250.0);
        let b = [bx, by, z_b];
        let s = [sx, sy, zg];
        let upper = tortuous(rng, o, b, z_b - 300.0);
        let lower = leader(rng, b, s, zg);
        let mut main = upper.clone();
        main.extend_from_slice(&lower[1..]);
        out.push((main, Kind::Main));
        let n_cloud = 1 + rng.poisson(1.5);
        for _ in 0..n_cloud {
            let i0 = ri(rng, 0, (upper.len() as i64 - 1).max(1)) as usize;
            let a = rng.range(0.0, 2.0 * PI);
            let h = [a.cos(), a.sin(), 0.0];
            let length = rng.range(1500.0, 6000.0);
            let br = free_walk(rng, upper[i0.min(upper.len() - 1)], [h[0], h[1], 0.1], length, 60.0, h, 0.12, z_b - 200.0, z_t + 1500.0);
            if br.len() > 2 {
                out.push((br, Kind::Cloud));
            }
        }
        let low_idx: Vec<usize> = (0..lower.len()).filter(|&i| lower[i][2] > zg + 300.0).collect();
        let n_down = if low_idx.len() > 2 { rng.poisson(3.0) } else { 0 };
        for _ in 0..n_down {
            let i0 = low_idx[ri(rng, 0, low_idx.len() as i64 - 1) as usize];
            let nxt = lower[(i0 + 1).min(lower.len() - 1)];
            let dm = sub3(nxt, lower[i0]);
            let d_main = norm3(dm[0], dm[1], dm[2]);
            let ang = rng.range(25.0, 50.0).to_radians();
            let ax = [rng.normal(), rng.normal(), rng.normal()];
            let d0 = turn3(d_main, ang, ax);
            let length = rng.range(100.0, 900.0);
            let br = free_walk(rng, lower[i0], d0, length, 30.0, [0.0, 0.0, -1.0], 0.08, zg + 20.0, f64::INFINITY);
            if br.len() > 2 {
                out.push((br, Kind::Branch));
            }
        }
    } else {
        let zc = zg + 1000.0 * rng.range(5.0, 8.0);
        let a = rng.range(0.0, 2.0 * PI);
        let half = 1000.0 * rng.range(2.5, 7.5);
        let h = [a.cos(), a.sin(), 0.0];
        let c = [sx, sy, zc];
        let e1z = rng.gauss(0.0, 400.0);
        let e2z = rng.gauss(0.0, 400.0);
        let e1 = [c[0] - half * h[0], c[1] - half * h[1], c[2] + e1z];
        let e2 = [c[0] + half * h[0], c[1] + half * h[1], c[2] + e2z];
        let main = tortuous(rng, e1, e2, zg + 3000.0);
        let n_br = 2 + rng.poisson(2.0);
        let mut branches = Vec::new();
        for _ in 0..n_br {
            let i0 = ri(rng, 0, (main.len() as i64 - 1).max(1)) as usize;
            let b = rng.range(0.0, 2.0 * PI);
            let hb = [b.cos(), b.sin(), 0.0];
            let length = rng.range(1000.0, 5000.0);
            let br = free_walk(rng, main[i0.min(main.len() - 1)], hb, length, 60.0, hb, 0.12, zg + 3000.0, zc + 2000.0);
            if br.len() > 2 {
                branches.push((br, Kind::Cloud));
            }
        }
        out.push((main, Kind::Ic));
        out.extend(branches);
    }
    (out, z_b)
}

fn strike_xy(distance_km: f32, bearing_deg: f32, heading_deg: f32) -> (f64, f64) {
    let az = bearing_to_azimuth(bearing_deg, heading_deg) as f64;
    let d = 1000.0 * (distance_km as f64).max(0.05);
    (d * az.cos(), d * az.sin())
}

// --------------------------------------------------------------------------
// N-wave kernels
// --------------------------------------------------------------------------
/// Running sum I of a unit N-wave of nT samples whose shock rises over nr samples (length
/// nT + nr - 1, I(end) = 0 exactly): thunder._nwave_integral, convolved in closed form where the
/// Hann rise window lies wholly on the linear ramp.
fn nwave_integral(nt: usize, nr: usize, win: &mut Vec<f64>, out: &mut Vec<f32>) {
    let nt = nt.max(2);
    let nr = nr.max(1);
    win.clear();
    let mut sw = 0.0;
    for j in 0..nr {
        let w = 0.5 - 0.5 * (2.0 * PI * (j + 1) as f64 / (nr + 1) as f64).cos();
        win.push(w);
        sw += w;
    }
    let mut mu = 0.0;
    for (j, w) in win.iter_mut().enumerate() {
        *w /= sw;
        mu += j as f64 * *w;
    }
    let l = nt + nr - 1;
    out.clear();
    out.reserve(l);
    let ntf = nt as f64;
    let mut cs = 0.0f64;
    let mut vals_last = 0.0;
    // first pass: cumulative sum (stored), residue removed in a second pass
    for t in 0..l {
        let y = if t + 1 >= nr && t < nt {
            1.0 - 2.0 * (t as f64 + 0.5 - mu) / ntf
        } else {
            let j0 = (t + 1).saturating_sub(nt);
            let j1 = (nr - 1).min(t);
            let mut s = 0.0;
            for j in j0..=j1 {
                s += win[j] * (1.0 - 2.0 * ((t - j) as f64 + 0.5) / ntf);
            }
            s
        };
        cs += y;
        out.push(cs as f32);
        vals_last = cs;
    }
    if l > 1 {
        let k = vals_last / (l - 1) as f64;
        for (i, v) in out.iter_mut().enumerate() {
            *v = (*v as f64 - k * i as f64) as f32;
        }
    }
}

/// Sum of squares of a segment's unit kernel (closed form of thunder._kernel_energy for the
/// parabolic I(x) = nT x (1 - x): sum I^2 = nT^3 / 30, autocorrelation nT^3 (1-s)^3 (1+3s+s^2) / 30).
/// Add one N-wave (shock up to +a, linear fall to -a, shock back to 0) of `len` samples at `at`.
fn add_nwave(y: &mut [f32], at: usize, len: usize, a: f32) {
    let len = len.max(2);
    let k = 2.0 / (len - 1) as f32;
    for i in 0..len {
        if let Some(v) = y.get_mut(at + i) {
            *v += a * (1.0 - k * i as f32);
        }
    }
}

/// A frame of Poisson-arriving N-waves (rate `lam` per s, periods lognormal about `t_mean`, mostly
/// compression first), normalised to unit variance per sample on average.
fn nwave_train(rng: &mut Rng, frame: &mut [f32], lam: f64, t_mean: f64, sr: f64) {
    frame.iter_mut().for_each(|v| *v = 0.0);
    let nf = frame.len();
    let count = rng.poisson(lam * nf as f64 / sr) as usize;
    // E[a^2] (lognormal sigma 0.6) * E[T] / 3 per N-wave
    let e_a2 = (2.0f64 * 0.36).exp();
    let var = lam / sr * e_a2 * t_mean * sr / 3.0;
    let g = (1.0 / var.max(1e-12)).sqrt();
    for _ in 0..count {
        let at = rng.below(nf as u64) as i64 - (0.5 * t_mean * sr) as i64;
        let len = ((t_mean * rng.lognormal(0.0, 0.3) * sr) as usize).max(4);
        let a = g * rng.lognormal(0.0, 0.6) * if rng.uniform() < 0.8 { 1.0 } else { -1.0 };
        let k = 2.0 / (len - 1) as f64;
        for i in 0..len {
            let t = at + i as i64;
            if t >= 0 && (t as usize) < nf {
                frame[t as usize] += (a * (1.0 - k * i as f64)) as f32;
            }
        }
    }
}

fn kernel_energy(nt: u32, m: u32) -> f64 {
    let ntf = nt as f64;
    if m <= 1 {
        return ntf / 3.0;
    }
    let mf = m as f64;
    let s0 = ntf.powi(3) / 30.0;
    let s = mf / ntf;
    let cross = if s < 1.0 { ntf.powi(3) * (1.0 - s).powi(3) * (1.0 + 3.0 * s + s * s) / 30.0 } else { 0.0 };
    ((2.0 * s0 - 2.0 * cross) / (mf * mf)).max(0.0)
}

fn hann_inner(l: usize) -> Vec<f64> {
    // np.hanning(l + 2)[1:-1]
    (1..=l).map(|k| 0.5 - 0.5 * (2.0 * PI * k as f64 / (l + 1) as f64).cos()).collect()
}

/// np.convolve(p, w, mode="same") for an odd-length window.
fn conv_same(p: &[f64], w: &[f64]) -> Vec<f64> {
    let nb = p.len();
    let lw = w.len();
    let h = (lw - 1) / 2;
    let mut out = vec![0.0; nb];
    for (i, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (k, wk) in w.iter().enumerate() {
            let idx = i as i64 + h as i64 - k as i64;
            if idx >= 0 && (idx as usize) < nb {
                s += wk * p[idx as usize];
            }
        }
        *o = s;
    }
    out
}

// --------------------------------------------------------------------------
// beam accumulator
// --------------------------------------------------------------------------
/// Overlap-add ring of mono signals on the 64-direction grid, in absolute samples.
struct BeamAcc {
    cap: usize,
    mask: usize,
    buf: Vec<f32>,
    base: i64,
    beg: [i64; N_BEAMS],
    end: [i64; N_BEAMS],
    row: Vec<f32>,
}

impl BeamAcc {
    fn new(capacity: usize, base: i64) -> BeamAcc {
        let cap = capacity.max(1024).next_power_of_two();
        BeamAcc {
            cap,
            mask: cap - 1,
            buf: vec![0.0; N_BEAMS * cap],
            base,
            beg: [i64::MAX; N_BEAMS],
            end: [base; N_BEAMS],
            row: Vec::with_capacity(4096),
        }
    }

    fn has_data(&self) -> bool {
        self.end.iter().any(|&e| e > self.base)
    }

    fn grow(&mut self, need: usize) {
        let cap = need.next_power_of_two();
        let mut nb = vec![0.0f32; N_BEAMS * cap];
        for r in 0..N_BEAMS {
            for i in 0..self.cap {
                let abs = self.base as usize + i;
                nb[r * cap + (abs & (cap - 1))] = self.buf[r * self.cap + (abs & self.mask)];
            }
        }
        self.buf = nb;
        self.cap = cap;
        self.mask = cap - 1;
    }

    fn add(&mut self, row: usize, start: i64, y: &[f32], g: f32) {
        let mut off = 0usize;
        let mut s = start;
        if s < self.base {
            off = (self.base - s) as usize;
            s = self.base;
        }
        if off >= y.len() {
            return;
        }
        let len = y.len() - off;
        let need = (s - self.base) as usize + len;
        if need > self.cap {
            self.grow(need);
        }
        let base = row * self.cap;
        for i in 0..len {
            let idx = (s as usize + i) & self.mask;
            self.buf[base + idx] += g * y[off + i];
        }
        let e = s + len as i64;
        if e > self.end[row] {
            self.end[row] = e;
        }
        if s < self.beg[row] {
            self.beg[row] = s;
        }
    }

    /// out (N_AMBI x n, zeroed by the caller) += Y @ next n columns; columns cleared, base advanced.
    fn pop_encode(&mut self, n: usize, beam_y: &[[f32; N_AMBI]; N_BEAMS], out: &mut [f32]) -> bool {
        let mut any = false;
        let base = self.base;
        let stop = base + n as i64;
        if self.row.len() < n {
            self.row.resize(n, 0.0);
        }
        for r in 0..N_BEAMS {
            if self.end[r] <= base || self.beg[r] >= stop {
                if self.end[r] <= stop {
                    self.beg[r] = i64::MAX;
                }
                continue;
            }
            let rb = r * self.cap;
            let i0 = (base as usize) & self.mask;
            let k1 = n.min(self.cap - i0);
            let row = &mut self.row[..n];
            row[..k1].copy_from_slice(&self.buf[rb + i0..rb + i0 + k1]);
            self.buf[rb + i0..rb + i0 + k1].iter_mut().for_each(|v| *v = 0.0);
            if k1 < n {
                row[k1..].copy_from_slice(&self.buf[rb..rb + n - k1]);
                self.buf[rb..rb + n - k1].iter_mut().for_each(|v| *v = 0.0);
            }
            any = true;
            for (ch, g) in beam_y[r].iter().enumerate() {
                let o = &mut out[ch * n..(ch + 1) * n];
                for (ov, x) in o.iter_mut().zip(row.iter()) {
                    *ov += g * x;
                }
            }
            if self.end[r] <= stop {
                self.beg[r] = i64::MAX;
            } else {
                self.beg[r] = self.beg[r].max(stop);
            }
        }
        self.base = stop;
        any
    }
}

// --------------------------------------------------------------------------
// shared tools of the voice (FFT plans and scratch)
// --------------------------------------------------------------------------
struct Tools {
    sr: f32,
    n: usize,
    beam_u: [[f64; 3]; N_BEAMS],
    beam_y: [[f32; N_AMBI]; N_BEAMS],
    // block STFT (2n)
    fft_blk: Fft,
    win_blk: Vec<f32>,
    freqs_blk: Vec<f32>,
    frame_blk: Vec<f32>,
    spec_blk: Vec<Complex<f32>>,
    gain_blk: Vec<f32>,
    // rumble frames (2 x hop)
    fft_r: Fft,
    freqs_r: Vec<f32>,
    wgt_r: Vec<f32>,
    scale_r: Vec<f32>,
    window_r: Vec<f32>,
    frame_r: Vec<f32>,
    spec_r: Vec<Complex<f32>>,
    gain_r: Vec<f32>,
    // crack
    fft_c: Fft,
    // kernels
    kern: Vec<f32>,
    win: Vec<f64>,
    // per block
    noise: Vec<f32>,
    growl: Vec<f32>,
    x: Vec<f32>,
    xb: Vec<f32>,
}

impl Tools {
    fn new(sr: f32, n: usize) -> Tools {
        let golden = PI * (3.0 - 5f64.sqrt());
        let mut beam_u = [[0.0; 3]; N_BEAMS];
        let mut beam_y = [[0.0f32; N_AMBI]; N_BEAMS];
        for i in 0..N_BEAMS {
            let z = 1.0 - (2.0 * i as f64 + 1.0) / N_BEAMS as f64;
            let az = (i as f64 * golden + PI).rem_euclid(2.0 * PI) - PI;
            let el = z.asin();
            beam_u[i] = [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()];
            beam_y[i] = crate::spatial::sh_real_f32(az as f32, el as f32);
        }
        let nb = 2 * n;
        let fft_blk = Fft::new(nb);
        let nbins = fft_blk.bins();
        let hop = RUMBLE_HOP;
        let nr = 2 * hop;
        let fft_r = Fft::new(nr);
        let fb = fft_r.bins();
        let mut scale_r = vec![(nr as f32 / 2.0).sqrt(); fb];
        scale_r[0] = (nr as f32).sqrt();
        scale_r[fb - 1] = (nr as f32).sqrt();
        let lc = (0.6 * sr) as usize;
        let fft_c = Fft::new((lc + 256).next_power_of_two());
        Tools {
            sr,
            n,
            beam_u,
            beam_y,
            win_blk: (0..nb).map(|i| (PI as f32 * (i as f32 + 0.5) / nb as f32).sin()).collect(),
            freqs_blk: rfft_freqs(nb, sr),
            frame_blk: vec![0.0; nb],
            spec_blk: vec![Complex::new(0.0, 0.0); nbins],
            gain_blk: vec![1.0; nbins],
            fft_blk,
            freqs_r: rfft_freqs(nr, sr),
            wgt_r: rfft_weights(nr),
            scale_r,
            window_r: (0..nr).map(|i| (PI as f32 * (i as f32 + 0.5) / nr as f32).sin()).collect(),
            frame_r: vec![0.0; nr],
            spec_r: vec![Complex::new(0.0, 0.0); fb],
            gain_r: vec![1.0; fb],
            fft_r,
            fft_c,
            kern: Vec::with_capacity(8192),
            win: Vec::with_capacity(256),
            noise: vec![0.0; RUMBLE_ROWS * n],
            growl: vec![0.0; n],
            x: vec![0.0; n],
            xb: vec![0.0; n],
        }
    }
}

// --------------------------------------------------------------------------
// one thunder event
// --------------------------------------------------------------------------
struct Rumble {
    start: i64,
    k: u64,
    src_mag: Vec<f32>,
    alpha: Vec<f32>,
    fifo: Vec<f32>,
    fcap: usize,
    have: usize,
    tail: Vec<f32>,
    growl: Biquad,
    growl_g: f32,
}

struct Event {
    key: u64,
    t_flash: f64,
    temp: f32,
    rh: f32,
    c: f64,
    sr: f64,
    // kept segments, time-ordered
    a: Vec<f32>,
    nt: Vec<u32>,
    nr: Vec<u32>,
    m: Vec<u32>,
    beam: Vec<u8>,
    // kernels (bumps), time-ordered
    b_start: Vec<i64>,
    b_seg: Vec<u32>,
    b_sign: Vec<i8>,
    b_ptr: usize,
    acc_cap: usize,
    acc: Option<BeamAcc>,
    x_cur: Vec<f32>,
    x_next: Vec<f32>,
    tail: Vec<f32>,
    x_started: bool,
    cur_nz: bool,
    tail_nz: bool,
    alpha_blk: Vec<f32>,
    // summary
    audible: bool,
    peak_pa: f64,
    r_min: f64,
    t_first: f64,
    near_dir: (f64, f64),
    audible_range_km: f64,
    strokes: Vec<f64>,
    end_sample: i64,
    // signal layer
    t_origin: f64,
    n_bins: usize,
    env_r: Vec<f32>,
    env_d: Vec<f32>,
    dir_az: Vec<f32>,
    dir_el: Vec<f32>,
    dir_sp: Vec<f32>,
    dir_g2: [f32; N_AMBI],
    env_end_sample: i64,
    f_nwave: f64,
    rumble: Option<Rumble>,
    dir_g: [f32; N_AMBI],
    // crack
    crack: Vec<f32>,
    crack_start: i64,
    crack_y: [f32; N_AMBI],
}

struct SegRow {
    mid: P3,
    seg_l: f64,
    r: f64,
    u: P3,
    fac: f64,
    low: bool,
    r_near: f64,
    r_far: f64,
    t_off: f64,
    e_mult: f64,
}

impl Event {
    #[allow(clippy::too_many_arguments)]
    fn new(
        key: u64,
        t_flash: f64,
        cg: bool,
        polys: &[(Vec<P3>, Kind)],
        z_cloud_base: f64,
        intensity: f64,
        w: &Weather,
        listener_h: f64,
        tl: &mut Tools,
        rng: &mut Rng,
    ) -> Event {
        let sr = tl.sr as f64;
        let temp = w.temperature_c as f64;
        let c = c_air(temp);
        let z_g = -listener_h;
        // ---- segment table ----
        let mut segs: Vec<SegRow> = Vec::new();
        for (pts, kind) in polys {
            if pts.len() < 2 {
                continue;
            }
            for p in pts.windows(2) {
                let (a, b) = (p[0], p[1]);
                let mid_z = 0.5 * (a[2] + b[2]);
                let fac = match kind {
                    Kind::Main => {
                        if mid_z > z_cloud_base {
                            0.7
                        } else {
                            1.0
                        }
                    }
                    Kind::Cloud => 0.55,
                    Kind::Branch => 0.3,
                    Kind::Ic => 0.5,
                };
                let low = *kind == Kind::Main && mid_z < z_cloud_base;
                let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), mid_z];
                let v = sub3(b, a);
                let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                let seg_l = vv.sqrt().max(1.0);
                let r = len3(mid).max(30.0);
                let u = [mid[0] / r, mid[1] / r, mid[2] / r];
                let tpar = (-(a[0] * v[0] + a[1] * v[1] + a[2] * v[2]) / vv.max(1e-9)).clamp(0.0, 1.0);
                let r_near = len3([a[0] + tpar * v[0], a[1] + tpar * v[1], a[2] + tpar * v[2]]).max(30.0);
                let r_far = len3(a).max(len3(b)).max(r_near);
                segs.push(SegRow { mid, seg_l, r, u, fac, low, r_near, r_far, t_off: 0.0, e_mult: 1.0 });
            }
        }
        // ---- return strokes along the lower main channel ----
        let mut strokes = vec![0.0f64];
        if cg {
            let n_str = (1 + rng.poisson(2.5)).min(8);
            let mut tau = 0.0;
            let mut extra = Vec::new();
            for _ in 1..n_str {
                tau += rng.lognormal(0.06f64.ln(), 0.5).clamp(0.015, 0.2);
                strokes.push(tau);
                extra.push((tau, rng.range(0.3, 0.7)));
            }
            let low_idx: Vec<usize> = (0..segs.len()).filter(|&i| segs[i].low).collect();
            if !low_idx.is_empty() && !extra.is_empty() {
                for (tau_s, e_s) in &extra {
                    for &i in &low_idx {
                        let s = &segs[i];
                        let row = SegRow { t_off: *tau_s, e_mult: *e_s, ..*s };
                        segs.push(row);
                    }
                }
            }
        }
        // ---- per-segment physics ----
        let e0_base = stroke_energy_j_per_m(intensity);
        let ns = segs.len();
        let mut amp = vec![0.0f64; ns];
        let mut per = vec![0.0f64; ns];
        let mut dtau = vec![0.0f64; ns];
        let mut az = vec![0.0f64; ns];
        let mut el = vec![0.0f64; ns];
        let mut t_start = vec![0.0f64; ns];
        let mut top_h = f64::NEG_INFINITY;
        let stab = w.stability;
        for (i, s) in segs.iter().enumerate() {
            let e0 = e0_base * s.fac * s.e_mult;
            let r0 = (e0 / (PI * P0)).sqrt();
            let t0 = (e0 / P0).sqrt() / c;
            // N-wave period: weak-shock stretching T ~ (r / R0)^(1/4) holds only while the shock is
            // strong (r < ~10 R0); beyond that the pulse propagates nearly acoustically and Few's
            // dominant frequency f_m = 0.63 c0 / R0 is almost independent of range (Few 1969, JGR 74,
            // 6926; Few 1982, Handbook of Atmospherics). Observed thunder spectra peak at 50-250 Hz
            // (Holmes et al. 1971, JGR 76, 2106). Only a slow spherical weak-shock growth
            // sqrt(1 + 0.04 ln(r / 10 R0)) remains (period ~ t0 = sqrt(E0 / p0) / c0 ~ 1.8 R0 / c0). The amplitude is rescaled by sqrt(T_old / T)
            // so each segment's acoustic energy (a^2 T) and hence the calibrated level is unchanged.
            let rc = 10.0 * r0;
            let per_old = (t0 * (s.r / r0).powf(0.25)).max(t0).min(0.08);
            per[i] = (t0 * (1.0 + 0.04 * (s.r / rc).max(1.0).ln()).sqrt())
                .max(t0)
                .min(0.08);
            let mut a = N_WAVE_K * e0.sqrt() * (s.seg_l / s.r) * (r0 / s.r).powf(0.25) * (per_old / per[i]).sqrt();
            dtau[i] = (s.r_far - s.r_near) / c;
            az[i] = s.u[1].atan2(s.u[0]);
            el[i] = s.u[2].clamp(-1.0, 1.0).asin();
            let r_h = s.mid[0].hypot(s.mid[1]);
            let h_src = (s.mid[2] - z_g).max(1.0);
            top_h = top_h.max(h_src);
            let wcomp = w.wind_u10_mps as f64 * (az[i] - w.wind_azimuth as f64).cos();
            a *= 10f64.powf(refraction_db(r_h, wcomp, stab, h_src, listener_h, temp) / 20.0);
            amp[i] = a;
            t_start[i] = t_flash + s.r_near / c + s.t_off;
        }
        let t_first_all = t_start.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut idx: Vec<usize> =
            (0..ns).filter(|&i| t_start[i] <= t_first_all + MAX_EVENT_S && amp[i] > 2e-6).collect();
        idx.sort_by(|&x, &y| t_start[x].partial_cmp(&t_start[y]).unwrap_or(std::cmp::Ordering::Equal));
        let k = idx.len();
        let mut a_k = Vec::with_capacity(k);
        let mut nt_k = Vec::with_capacity(k);
        let mut nr_k = Vec::with_capacity(k);
        let mut m_k = Vec::with_capacity(k);
        let mut beam_k = Vec::with_capacity(k);
        let mut s_start = Vec::with_capacity(k);
        let mut t_k = Vec::with_capacity(k);
        let mut dtau_k = Vec::with_capacity(k);
        let mut az_k = Vec::with_capacity(k);
        let mut el_k = Vec::with_capacity(k);
        let mut rnear_k = Vec::with_capacity(k);
        let mut max_len = 0usize;
        for &i in &idx {
            let s = &segs[i];
            a_k.push(amp[i]);
            let ss = (t_start[i] * sr).round() as i64;
            s_start.push(ss);
            let mut best = 0usize;
            let mut bd = f64::NEG_INFINITY;
            for (b, bu) in tl.beam_u.iter().enumerate() {
                let d = s.u[0] * bu[0] + s.u[1] * bu[1] + s.u[2] * bu[2];
                if d > bd {
                    bd = d;
                    best = b;
                }
            }
            beam_k.push(best as u8);
            let nt = ((per[i] * sr).ceil() as i64).max(2) as u32;
            let rise = (3.0 / sr).max(2e-4 * (s.r / 1000.0).sqrt());
            let nr = ((rise * sr).round() as i64).max(3) as u32;
            let m = ((dtau[i] * sr).round() as i64).max(1) as u32;
            max_len = max_len.max((nt + nr) as usize);
            nt_k.push(nt);
            nr_k.push(nr);
            m_k.push(m);
            t_k.push(per[i]);
            dtau_k.push(dtau[i]);
            az_k.push(az[i]);
            el_k.push(el[i]);
            rnear_k.push(s.r_near);
        }
        // kernels: +I/m at the first arrival and -I/m m samples later (dense N-wave when m <= 1)
        let mut bumps: Vec<(i64, u32, i8, usize)> = Vec::with_capacity(2 * k);
        for j in 0..k {
            let sign = if m_k[j] <= 1 { 0 } else { 1 };
            bumps.push((s_start[j], j as u32, sign, bumps.len()));
        }
        for j in 0..k {
            if m_k[j] > 1 {
                bumps.push((s_start[j] + m_k[j] as i64, j as u32, -1, bumps.len()));
            }
        }
        bumps.sort_by_key(|b| (b.0, b.3));
        let audible = k > 0 && a_k.iter().cloned().fold(0.0, f64::max) > 2e-5;
        let peak_pa = if k > 0 { a_k.iter().cloned().fold(0.0, f64::max) } else { 0.0 };
        let r_min_all = segs.iter().map(|s| s.r_near).fold(f64::INFINITY, f64::min);
        let r_min = if k > 0 { rnear_k.iter().cloned().fold(f64::INFINITY, f64::min) } else { r_min_all };
        let t_first = if k > 0 { t_flash + r_min / c } else { t_first_all };
        let near_dir = if k > 0 {
            let mut im = 0;
            for j in 1..k {
                if rnear_k[j] < rnear_k[im] {
                    im = j;
                }
            }
            (az_k[im], el_k[im])
        } else {
            (0.0, 0.0)
        };
        let audible_range_km = shadow_distance_m(top_h.max(0.0), listener_h, temp) / 1000.0;
        let mut end_sample = if k > 0 {
            (0..k).map(|j| s_start[j] + (nt_k[j] + nr_k[j] + m_k[j]) as i64).max().unwrap_or(0)
        } else {
            (t_first_all * sr).round() as i64
        };
        let n = tl.n;
        let mut ev = Event {
            key,
            t_flash,
            temp: w.temperature_c,
            rh: w.humidity_pct,
            c,
            sr,
            a: a_k.iter().map(|&v| v as f32).collect(),
            nt: nt_k,
            nr: nr_k,
            m: m_k,
            beam: beam_k,
            b_start: bumps.iter().map(|b| b.0).collect(),
            b_seg: bumps.iter().map(|b| b.1).collect(),
            b_sign: bumps.iter().map(|b| b.2).collect(),
            b_ptr: 0,
            acc_cap: max_len + 3 * n + 64,
            acc: None,
            x_cur: Vec::new(),
            x_next: Vec::new(),
            tail: Vec::new(),
            x_started: false,
            cur_nz: false,
            tail_nz: false,
            alpha_blk: Vec::new(),
            audible,
            peak_pa,
            r_min,
            t_first,
            near_dir,
            audible_range_km,
            strokes,
            end_sample: 0,
            t_origin: t_first_all,
            n_bins: 0,
            env_r: Vec::new(),
            env_d: Vec::new(),
            dir_az: Vec::new(),
            dir_el: Vec::new(),
            dir_sp: Vec::new(),
            dir_g2: [0.0; N_AMBI],
            env_end_sample: 0,
            f_nwave: 100.0,
            rumble: None,
            dir_g: sh_gains(near_dir.0 as f32, near_dir.1 as f32, 0.45),
            crack: Vec::new(),
            crack_start: 0,
            crack_y: [0.0; N_AMBI],
        };
        if k > 0 {
            ev.envelopes(&s_start, &t_k, &dtau_k, &az_k, &el_k, &mut end_sample);
            ev.end_sample = end_sample;
            ev.make_crack(rng, &rnear_k, tl);
        } else {
            ev.end_sample = end_sample;
        }
        ev
    }

    fn envelopes(&mut self, s_start: &[i64], per: &[f64], dtau: &[f64], az: &[f64], el: &[f64], end_sample: &mut i64) {
        let sr = self.sr;
        let k = s_start.len();
        let mut e = vec![0.0f64; k];
        let mut b0 = vec![0usize; k];
        let mut span = vec![0usize; k];
        let mut max_b = 0usize;
        for j in 0..k {
            let a = self.a[j] as f64;
            e[j] = a * a * kernel_energy(self.nt[j], self.m[j]) / sr;
            let t0 = s_start[j] as f64 / sr - self.t_origin;
            b0[j] = (t0 / BIN_S).floor().max(0.0) as usize;
            let dur = per[j].max(dtau[j]);
            span[j] = ((dur / BIN_S).ceil() as usize).max(1);
            max_b = max_b.max(b0[j] + span[j]);
        }
        // Rolling: the low-frequency afterglow lasts longer the farther the flash (range spread of
        // the channel plus refraction / turbulent-scattering multipath grow with range; rumble runs
        // ~5-20 s at a few km, Rakov & Uman 2003 ch. 11, Few 1975): tau = 0.8 + 0.35 r_km s.
        let glow_tau = (AFTERGLOW_TAU_S * (0.8 + 0.35 * self.r_min / 1000.0)).clamp(1.0, 4.0);
        // run the afterglow down to -40 dB (power ln(1e4) tau) before the envelope ends
        let tail_s = (9.2 * glow_tau).clamp(9.0, MAX_EVENT_S - 4.0);
        let tail_bins = (tail_s / BIN_S) as usize;
        let nb = max_b + tail_bins + 2;
        let mut diff = vec![0.0f64; nb + 1];
        let mut dx = vec![0.0f64; nb + 1];
        let mut dy = vec![0.0f64; nb + 1];
        let mut dz = vec![0.0f64; nb + 1];
        for j in 0..k {
            let rate = e[j] / (span[j] as f64 * BIN_S);
            diff[b0[j]] += rate;
            diff[b0[j] + span[j]] -= rate;
            let (ux, uy, uz) = (el[j].cos() * az[j].cos(), el[j].cos() * az[j].sin(), el[j].sin());
            dx[b0[j]] += rate * ux;
            dx[b0[j] + span[j]] -= rate * ux;
            dy[b0[j]] += rate * uy;
            dy[b0[j] + span[j]] -= rate * uy;
            dz[b0[j]] += rate * uz;
            dz[b0[j] + span[j]] -= rate * uz;
        }
        let cums = |d: &[f64]| -> Vec<f64> {
            let mut acc = 0.0;
            d[..nb].iter().map(|v| {
                acc += v;
                acc
            }).collect()
        };
        let p: Vec<f64> = cums(&diff).into_iter().map(|v| v.max(0.0)).collect();
        let mut wr = hann_inner(15);
        let sr_ = wr.iter().sum::<f64>();
        wr.iter_mut().for_each(|v| *v /= sr_);
        let pr = conv_same(&p, &wr);
        let mut wd = hann_inner(61);
        let sd = wd.iter().sum::<f64>();
        wd.iter_mut().for_each(|v| *v /= sd);
        let pd0 = conv_same(&p, &wd);
        // afterglow: peak hold decaying with AFTERGLOW_TAU_S (power)
        let decay = (-BIN_S / glow_tau).exp();
        let mut hold = 0.0f64;
        let mut pd = vec![0.0f64; nb];
        for i in 0..nb {
            hold = (hold * decay).max(pd0[i].max(0.0));
            pd[i] = hold;
        }
        self.env_r = pr.iter().map(|v| (RUMBLER_POWER * v.max(0.0)).sqrt() as f32).collect();
        // The low afterglow belongs to distant thunder: absorption (ISO 9613-1) and the longer,
        // more scattered paths leave mostly 30-100 Hz rumble beyond ~5 km, while near and mid-range
        // thunder peaks higher, ~80-250 Hz (Holmes et al. 1971; Few 1982).
        let r_km = self.r_min / 1000.0;
        let deep = DEEPENER_POWER * (0.12 + 0.88 * smoothstep(r_km as f32, 2.5, 8.0) as f64);
        self.env_d = pd.iter().map(|v| (deep * v).sqrt() as f32).collect();
        // energy-weighted direction track, held through silent stretches
        let wv = hann_inner(51);
        let vx = conv_same(&cums(&dx), &wv);
        let vy = conv_same(&cums(&dy), &wv);
        let vz = conv_same(&cums(&dz), &wv);
        let last = [
            self.near_dir.1.cos() * self.near_dir.0.cos(),
            self.near_dir.1.cos() * self.near_dir.0.sin(),
            self.near_dir.1.sin(),
        ];
        self.dir_az = vec![0.0; nb];
        self.dir_el = vec![0.0; nb];
        // angular extent of what arrives now: circular spread sqrt(2 (1 - R)) of the energy-weighted
        // resultant R (same window); the second rumbler sits that far to one side of the track
        let pw = conv_same(&p, &wv);
        self.dir_sp = vec![0.35; nb];
        let mut held = last;
        let mut held_sp = 0.35f64;
        for i in 0..nb {
            let nrm = (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]).sqrt();
            if nrm > 1e-18 {
                held = [vx[i] / nrm, vy[i] / nrm, vz[i] / nrm];
                if pw[i] > 1e-30 {
                    let r = (nrm / pw[i]).clamp(0.0, 1.0);
                    held_sp = (2.0 * (1.0 - r)).sqrt().clamp(0.35, 1.2);
                }
            }
            self.dir_az[i] = held[1].atan2(held[0]) as f32;
            self.dir_el[i] = held[2].clamp(-1.0, 1.0).asin() as f32;
            self.dir_sp[i] = held_sp as f32;
        }
        self.n_bins = nb;
        self.env_end_sample = ((self.t_origin + nb as f64 * BIN_S) * sr).round() as i64;
        *end_sample = (*end_sample + (0.5 * sr) as i64).max(self.env_end_sample);
        let se: f64 = e.iter().sum();
        let t_mean = per.iter().zip(&e).map(|(t, ee)| t * ee).sum::<f64>() / se.max(1e-30);
        self.f_nwave = (0.66 / t_mean.max(1e-4)).clamp(15.0, 400.0);
    }

    /// Near-strike crack and tearing (the 'strike' layer).
    fn make_crack(&mut self, rng: &mut Rng, r_near: &[f64], tl: &mut Tools) {
        let rmin = self.r_min;
        if rmin >= CRACK_RANGE_M {
            return;
        }
        let sr = tl.sr;
        let l = (0.6 * sr) as usize;
        let nfft = tl.fft_c.n;
        let mut burst = vec![0.0f32; nfft];
        // The crack is the first, nearly coherent arrivals from the channel segments closest to the
        // listener: a train of short N-waves with shock fronts (sub-ms rise, energy to several kHz;
        // Few 1969; Uman 1987 'The Lightning Discharge' ch. 11), each at its own segment's arrival
        // time and scaled by its own amplitude. The micro-tortuosity (1-10 m kinks) gives the short
        // 1-4 ms periods; the ISO 9613-1 path absorption is applied below.
        let mut near: Vec<(f64, f64)> = r_near
            .iter()
            .enumerate()
            .filter(|e| *e.1 - rmin < 0.35 * self.c)
            .map(|(j, &rn)| ((rn - rmin) / self.c, self.a[j] as f64))
            .collect();
        near.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        near.truncate(48);
        if near.is_empty() {
            near.push((0.0, 1.0));
        }
        let a_ref = near.iter().fold(0.0f64, |m, v| m.max(v.1)).max(1e-30);
        // the very first arrival opens the crack at full strength
        let first = near.iter().fold(f64::INFINITY, |m, v| m.min(v.0));
        near.push((first, a_ref));
        for &(dt, a) in &near {
            // 1-3 sub-pulses per segment from its kinks, each an N-wave of 1-4 ms
            let subs = ri(rng, 1, 4);
            for q in 0..subs {
                let t0 = dt + if q == 0 { 0.0 } else { rng.uniform() * 0.012 };
                let per = 0.001 + 0.003 * rng.uniform();
                let amp = (a / a_ref) * rng.lognormal(0.0, 0.35) * if q == 0 { 1.0 } else { 0.5 };
                let pol = if rng.uniform() < 0.85 { 1.0 } else { -1.0 };
                add_nwave(&mut burst[..l], (t0 * sr as f64) as usize, (per * sr as f64) as usize, (pol * amp) as f32);
            }
        }
        let tear = 3.0 * (1.0 - rmin / TEAR_RANGE_M).max(0.0).powf(1.5);
        if tear > 0.0 {
            let kc = ri(rng, 40, 150) as usize;
            let span = (0.12 * sr) as i64;
            let mut offs: Vec<usize> = (0..kc).map(|_| ri(rng, 0, span) as usize).collect();
            offs.sort_unstable();
            let mut imp = vec![0.0f32; l];
            for &o in &offs {
                let a = rng.lognormal(0.0, 0.6) * if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                imp[o.min(l - 1)] += a as f32;
            }
            let mut f1 = Biquad::new(hp2(1800.0, sr));
            let mut f2 = Biquad::new(lp2(7000.0, sr));
            f1.process_block(&mut imp);
            f2.process_block(&mut imp);
            for (b, v) in burst.iter_mut().zip(&imp) {
                *b += tear as f32 * v;
            }
        }
        // ISO 9613-1 magnitude over rmin, zero-phase by FFT
        let iso = Iso9613::new(self.temp, self.rh);
        let mut spec = vec![Complex::new(0.0f32, 0.0); tl.fft_c.bins()];
        tl.fft_c.forward(&mut burst, &mut spec);
        let df = sr as f64 / nfft as f64;
        for (b, s) in spec.iter_mut().enumerate() {
            let g = 10f64.powf(-iso.alpha(b as f64 * df) * rmin / 20.0) as f32;
            *s *= g / nfft as f32;
        }
        tl.fft_c.inverse(&mut spec, &mut burst);
        burst.truncate(l);
        let pk = burst.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if pk <= 0.0 {
            return;
        }
        let mut a_peak = 0.0f64;
        let mut found = false;
        for (j, &rn) in r_near.iter().enumerate() {
            if rn < 1.2 * rmin {
                a_peak = a_peak.max(self.a[j] as f64);
                found = true;
            }
        }
        if !found {
            a_peak = self.peak_pa;
        }
        // a close strike's crack is its loudest moment (peak levels well above the rumble; Uman 1987)
        let g = (1.6 * a_peak * (1.0 - rmin / CRACK_RANGE_M)) as f32 / pk;
        burst.iter_mut().for_each(|v| *v *= g);
        self.crack = burst;
        self.crack_start = ((self.t_flash + rmin / self.c) * self.sr).round() as i64;
        self.crack_y = sh_gains(self.near_dir.0 as f32, self.near_dir.1 as f32, 0.1);
        self.end_sample = self.end_sample.max(self.crack_start + l as i64);
    }

    fn schedule(&mut self, tl: &mut Tools, until: i64) {
        let acc = match self.acc.as_mut() {
            Some(a) => a,
            None => return,
        };
        while self.b_ptr < self.b_start.len() && self.b_start[self.b_ptr] < until {
            let k = self.b_ptr;
            let i = self.b_seg[k] as usize;
            nwave_integral(self.nt[i] as usize, self.nr[i] as usize, &mut tl.win, &mut tl.kern);
            let sg = self.b_sign[k];
            let a = self.a[i];
            if sg == 0 {
                // dense N-wave: diff(I, prepend 0)
                let kern = &mut tl.kern;
                for t in (1..kern.len()).rev() {
                    kern[t] -= kern[t - 1];
                }
                acc.add(self.beam[i] as usize, self.b_start[k], &tl.kern, a);
            } else {
                acc.add(self.beam[i] as usize, self.b_start[k], &tl.kern, sg as f32 * a / self.m[i] as f32);
            }
            self.b_ptr += 1;
        }
    }

    /// The absorbed N-wave sum for [start, start + n) added to `out` (one block of look-ahead).
    fn nwaves(&mut self, tl: &mut Tools, start: i64, n: usize, out: &mut [f32]) {
        if self.acc.is_none() {
            if !self.b_start.is_empty() && start + 2 * n as i64 <= self.b_start[0] {
                return; // still travelling: nothing has arrived yet
            }
            self.acc = Some(BeamAcc::new(self.acc_cap, start));
            self.x_cur = vec![0.0; N_AMBI * n];
            self.x_next = vec![0.0; N_AMBI * n];
            self.tail = vec![0.0; N_AMBI * n];
            let iso = Iso9613::new(self.temp, self.rh);
            self.alpha_blk = tl.freqs_blk.iter().map(|&f| iso.alpha(f as f64) as f32).collect();
        }
        if !self.x_started {
            {
                let acc = self.acc.as_mut().unwrap();
                if !acc.has_data() {
                    acc.base = acc.base.max(start);
                    acc.end.iter_mut().for_each(|e| *e = (*e).max(acc.base));
                }
            }
            self.schedule(tl, start + n as i64);
            self.x_cur.iter_mut().for_each(|v| *v = 0.0);
            self.cur_nz = self.acc.as_mut().unwrap().pop_encode(n, &tl.beam_y, &mut self.x_cur);
            self.tail.iter_mut().for_each(|v| *v = 0.0);
            self.tail_nz = false;
            self.x_started = true;
        }
        self.schedule(tl, start + 2 * n as i64);
        self.x_next.iter_mut().for_each(|v| *v = 0.0);
        let next_nz = self.acc.as_mut().unwrap().pop_encode(n, &tl.beam_y, &mut self.x_next);
        if !self.cur_nz && !next_nz && !self.tail_nz {
            std::mem::swap(&mut self.x_cur, &mut self.x_next);
            self.cur_nz = next_nz;
            return;
        }
        let r_now = (self.c * ((start + n as i64) as f64 / self.sr - self.t_flash)).max(1.0);
        let nf = 2 * n;
        let inv = 1.0 / nf as f32;
        for (g, a) in tl.gain_blk.iter_mut().zip(&self.alpha_blk) {
            *g = 10f32.powf(-a * r_now as f32 / 20.0) * inv;
        }
        let mut tail_nz = false;
        for ch in 0..N_AMBI {
            let xc = &self.x_cur[ch * n..(ch + 1) * n];
            let xn = &self.x_next[ch * n..(ch + 1) * n];
            let tt = &mut self.tail[ch * n..(ch + 1) * n];
            let live = xc.iter().any(|v| *v != 0.0) || xn.iter().any(|v| *v != 0.0);
            let o = &mut out[ch * n..(ch + 1) * n];
            if !live {
                if tt.iter().any(|v| *v != 0.0) {
                    for i in 0..n {
                        o[i] += tt[i];
                        tt[i] = 0.0;
                    }
                }
                continue;
            }
            let fr = &mut tl.frame_blk;
            for i in 0..n {
                fr[i] = xc[i] * tl.win_blk[i];
                fr[n + i] = xn[i] * tl.win_blk[n + i];
            }
            tl.fft_blk.forward(fr, &mut tl.spec_blk);
            for (s, g) in tl.spec_blk.iter_mut().zip(&tl.gain_blk) {
                *s *= *g;
            }
            tl.fft_blk.inverse(&mut tl.spec_blk, fr);
            for i in 0..n {
                o[i] += tt[i] + fr[i] * tl.win_blk[i];
                tt[i] = fr[n + i] * tl.win_blk[n + i];
            }
            if tt.iter().any(|v| v.abs() > 1e-20) {
                tail_nz = true;
            } else {
                tt.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        self.tail_nz = tail_nz;
        std::mem::swap(&mut self.x_cur, &mut self.x_next);
        self.cur_nz = next_nz;
    }

    fn setup_rumble(&mut self, tl: &Tools, start: i64) {
        let fb = tl.freqs_r.len();
        let mut src = vec![0.0f32; RUMBLE_ROWS * fb];
        for (b, &f) in tl.freqs_r.iter().enumerate() {
            let f = f.max(1e-3) as f64;
            // rumblers (rows 0, 2) are trains of N-waves (rumble_noise), so their spectrum is the
            // N-wave spectrum itself; here only the sub-sonic part is removed: thunder's infrasound
            // (< 20 Hz, Bohannon et al. 1977, JGR 82) is inaudible and only eats headroom.
            let x25 = (f / 25.0).powi(4);
            let b_r = x25 / (1.0 + x25);
            // deepener: the long low rumble of the distant channel, 30-100 Hz (Holmes et al. 1971;
            // the far-field rumble peak after ISO 9613-1 absorption), 4th-order edges.
            let x28 = (f / 28.0).powi(4);
            let b_d = x28 / (1.0 + x28) / (1.0 + (f / 95.0).powi(4));
            src[b] = b_r as f32;
            src[fb + b] = b_d as f32;
        }
        for v in 0..2 {
            let row = &mut src[v * fb..(v + 1) * fb];
            let p: f32 = row.iter().zip(&tl.wgt_r).map(|(a, w)| a * w).sum();
            let p = p.max(1e-30);
            row.iter_mut().for_each(|x| *x = (*x / p).sqrt());
        }
        src.copy_within(0..fb, 2 * fb);
        let iso = Iso9613::new(self.temp, self.rh);
        let alpha = tl.freqs_r.iter().map(|&f| iso.alpha(f as f64) as f32).collect();
        let gl = lp2(6.0, tl.sr);
        let fcap = RUMBLE_HOP + tl.n + 8;
        self.rumble = Some(Rumble {
            start,
            k: 0,
            src_mag: src,
            alpha,
            fifo: vec![0.0; RUMBLE_ROWS * fcap],
            fcap,
            have: 0,
            tail: vec![0.0; RUMBLE_ROWS * RUMBLE_HOP],
            growl: Biquad::new(gl),
            growl_g: noise_rms_gain(&[gl]),
        });
    }

    /// Next n samples of the rumbler (row 0) and deepener (row 1) into tl.noise (2 x n).
    fn rumble_noise(&mut self, tl: &mut Tools, key_rumble: u64, n: usize) {
        let ru = self.rumble.as_mut().unwrap();
        let hop = RUMBLE_HOP;
        let nf = 2 * hop;
        let fb = tl.freqs_r.len();
        while ru.have < n {
            let t_mid = (ru.start + ((ru.k + 1) as i64) * hop as i64) as f64 / self.sr;
            let r_m = (self.c * (t_mid - self.t_flash)).max(0.0) as f32;
            for (g, a) in tl.gain_r.iter_mut().zip(&ru.alpha) {
                *g = 10f32.powf(-a * r_m / 20.0);
            }
            let mut rng = Rng::from_key(combine(key_rumble, self.key), ru.k);
            // Rolling rumble as shaped impulse trains, not noise: every micro-segment of the tortuous
            // channel sends its own N-wave (Few 1969; Ribner & Roy 1982, JASA 72, 1911), arriving in
            // a Poisson stream whose density grows with range (more channel within each arrival
            // window, plus scattering), so close rumble crackles and distant rumble rolls smoothly.
            let r_km = (r_m as f64 / 1000.0).max(0.2);
            let lam = (25.0 + 55.0 * r_km).min(700.0);
            let t_mean = (0.66 / self.f_nwave.max(15.0)).clamp(0.002, 0.04);
            for v in 0..RUMBLE_ROWS {
                let mag = &ru.src_mag[v * fb..(v + 1) * fb];
                if v == 1 {
                    white_spectrum(&mut rng, &mut tl.spec_r);
                    for b in 0..fb {
                        tl.spec_r[b] *= mag[b] * tl.gain_r[b] * tl.scale_r[b];
                    }
                } else {
                    nwave_train(&mut rng, &mut tl.frame_r, lam, t_mean, self.sr);
                    tl.fft_r.forward(&mut tl.frame_r, &mut tl.spec_r);
                    for b in 0..fb {
                        tl.spec_r[b] *= mag[b] * tl.gain_r[b];
                    }
                }
                tl.fft_r.inverse(&mut tl.spec_r, &mut tl.frame_r);
                let k = 1.0 / nf as f32;
                let off = v * ru.fcap + ru.have;
                let tail = &mut ru.tail[v * hop..(v + 1) * hop];
                for i in 0..hop {
                    let x = tl.frame_r[i] * tl.window_r[i] * k;
                    ru.fifo[off + i] = x + tail[i];
                    tail[i] = tl.frame_r[hop + i] * tl.window_r[hop + i] * k;
                }
            }
            ru.have += hop;
            ru.k += 1;
        }
        for v in 0..RUMBLE_ROWS {
            let row = &mut ru.fifo[v * ru.fcap..(v + 1) * ru.fcap];
            tl.noise[v * n..(v + 1) * n].copy_from_slice(&row[..n]);
            row.copy_within(n..ru.have, 0);
        }
        ru.have -= n;
    }

    #[allow(clippy::too_many_arguments)]
    fn render(&mut self, tl: &mut Tools, ctx: &BlockCtx, keys: (u64, u64), start: i64, n: usize, out: &mut [f32]) {
        let end = start + n as i64;
        if !self.b_start.is_empty() && (self.b_ptr < self.b_start.len() || self.acc.is_some()) {
            self.nwaves(tl, start, n, out);
        }
        if !self.crack.is_empty() {
            let a = start.max(self.crack_start);
            let b = end.min(self.crack_start + self.crack.len() as i64);
            if b > a {
                for ch in 0..N_AMBI {
                    let g = self.crack_y[ch];
                    if g == 0.0 {
                        continue;
                    }
                    let o = &mut out[ch * n..(ch + 1) * n];
                    for t in a..b {
                        o[(t - start) as usize] += g * self.crack[(t - self.crack_start) as usize];
                    }
                }
            }
        }
        if self.n_bins > 0 && start < self.env_end_sample && end > (self.t_origin * self.sr) as i64 {
            if self.rumble.is_none() {
                self.setup_rumble(tl, start);
            }
            self.rumble_noise(tl, keys.0, n);
            let mut grng = Rng::from_key(combine(keys.1, self.key), ctx.index);
            fill_gauss(&mut grng, &mut tl.growl[..n]);
            {
                let ru = self.rumble.as_mut().unwrap();
                ru.growl.process_block(&mut tl.growl[..n]);
                let inv = 1.0 / ru.growl_g;
                tl.growl[..n].iter_mut().for_each(|v| *v *= inv);
            }
            let nb = self.n_bins;
            let interp = |env: &[f32], tb: f64| -> f32 {
                if tb < 0.0 || tb > (nb - 1) as f64 {
                    return 0.0;
                }
                let i = tb.floor() as usize;
                if i + 1 >= nb {
                    return env[nb - 1];
                }
                let fr = (tb - i as f64) as f32;
                env[i] + (env[i + 1] - env[i]) * fr
            };
            let mut tb_mid = 0.0;
            for i in 0..n {
                let tb = (((start + i as i64) as f64) / self.sr - self.t_origin) / BIN_S;
                if i == n / 2 {
                    tb_mid = tb;
                }
                let er = interp(&self.env_r, tb);
                let ed = interp(&self.env_d, tb);
                let gr = er * (1.0 + 0.6 * tl.growl[i]).max(0.0) * std::f32::consts::FRAC_1_SQRT_2;
                let deep = tl.noise[n + i] * ed;
                tl.x[i] = tl.noise[i] * gr + deep;
                tl.xb[i] = tl.noise[2 * n + i] * gr;
            }
            let kb = tb_mid.clamp(0.0, (nb - 1) as f64) as usize;
            let yt = sh_gains(self.dir_az[kb], self.dir_el[kb], 0.45);
            // the second rumbler: the far side of the channel's angular extent (side fixed per flash)
            let side = if self.key & 1 == 0 { 1.0 } else { -1.0 };
            let sp = self.dir_sp.get(kb).copied().unwrap_or(0.35);
            let yb = sh_gains(self.dir_az[kb] + side * sp, self.dir_el[kb] + 0.3 * sp, 0.45);
            if self.dir_g2[0] == 0.0 {
                self.dir_g2 = yb;
            }
            encode_ramped(out, &tl.x[..n], &self.dir_g, &yt);
            encode_ramped(out, &tl.xb[..n], &self.dir_g2, &yb);
            self.dir_g = yt;
            self.dir_g2 = yb;
        }
    }

    fn finished(&self, next_start: i64) -> bool {
        let pending = match &self.acc {
            Some(acc) => acc.has_data() || self.tail_nz || self.cur_nz,
            None => false,
        };
        self.b_ptr >= self.b_start.len() && !pending && next_start >= self.end_sample
    }
}

// --------------------------------------------------------------------------
// terrain and building echoes
// --------------------------------------------------------------------------
struct Tap {
    delay: usize,
    g: f32,
    y: [f32; N_AMBI],
    f: [Biquad; 2],
}

struct EchoField {
    sr: f32,
    taps: Vec<Tap>,
    ring: Vec<f32>,
    mask: usize,
    w: usize,
    quiet: usize,
    max_delay: usize,
    tail_wet: f32,
    tail_rt: f32,
    room: Room,
    x: Vec<f32>,
    bus: Vec<f32>,
}

impl EchoField {
    fn new(scene: &Scene, patches: &[Patch], cfg: &BuildCfg) -> EchoField {
        let sr = cfg.sr as f32;
        let n = cfg.block.max(1);
        let mut rng = cfg.rng.get("thunder.echo.cfg", 0);
        let preset = scene.place.preset.to_ascii_lowercase();
        let has = |keys: &[&str]| keys.iter().any(|k| preset.contains(k));
        let mut taps_raw: Vec<(f64, f64, f64, f64)> = Vec::new();
        let tail;
        if has(&["mountain", "ridge", "valley", "hill", "alp", "canyon", "fjord", "cliff"]) {
            for _ in 0..4 {
                let d = rng.range(400.0, 2500.0);
                let g = rng.range(0.35, 0.6) * 5000.0 / (5000.0 + 2.0 * d);
                let az = rng.range(-PI, PI);
                let el = rng.range(0.0, 0.15);
                taps_raw.push((d, g, az, el));
            }
            tail = (4.0, 0.25);
        } else if has(&["city", "street", "alley", "town", "downtown", "urban", "suburb", "village"]) {
            for _ in 0..4 {
                let d = rng.range(30.0, 200.0);
                let g = rng.range(0.3, 0.45);
                let az = rng.range(-PI, PI);
                taps_raw.push((d, g, az, 0.05));
            }
            tail = (2.5, 0.2);
        } else if has(&["forest", "wood", "jungle", "rainforest", "cabin", "camp"]) {
            for _ in 0..2 {
                let d = rng.range(60.0, 200.0);
                let az = rng.range(-PI, PI);
                taps_raw.push((d, 0.12, az, 0.1));
            }
            // trunks and canopy keep scattering the low rumble (forest RT60 1-2 s, Embleton 1963)
            tail = (2.2, 0.24);
        } else {
            let d = rng.range(800.0, 2500.0);
            let az = rng.range(-PI, PI);
            taps_raw.push((d, 0.12 * 5000.0 / (5000.0 + 2.0 * d), az, 0.03));
            // open ground: terrain and turbulent-atmosphere scattering still drag out the rumble
            // (Few 1975; Rakov & Uman 2003 ch. 11)
            tail = (2.0, 0.14);
        }
        for p in patches {
            if p.param_bool("reflector", false) && p.distance() < 80.0 {
                let g = if p.param_bool("facade", false) { 0.45 } else { 0.35 };
                taps_raw.push((p.distance() as f64, g, p.azimuth() as f64, p.elevation() as f64));
            }
        }
        let mut taps = Vec::new();
        for (d, g, az, el) in taps_raw {
            let delay = (2.0 * d / 343.0 * sr as f64).round() as i64;
            if delay < 1 || delay > (ECHO_MAX_DELAY_S * sr as f64) as i64 {
                continue;
            }
            taps.push(Tap {
                delay: delay as usize,
                g: g as f32,
                y: sh_gains(az as f32, el as f32, 0.3),
                f: [Biquad::new(absorption_lowpass(2.0 * d, 15.0, 70.0, sr)), Biquad::new(lp2(3500.0, sr))],
            });
        }
        let max_d = taps.iter().map(|t| t.delay).max().unwrap_or(0);
        let size = (max_d + 2 * n + 1).next_power_of_two();
        EchoField {
            sr,
            taps,
            ring: vec![0.0; size],
            mask: size - 1,
            w: 0,
            quiet: 1_000_000_000,
            max_delay: max_d,
            tail_wet: tail.1,
            tail_rt: tail.0,
            room: Room::new(tail.0, cfg.sr, 0.3, Some(25.0), 0.1),
            x: vec![0.0; n],
            bus: vec![0.0; N_AMBI * n],
        }
    }

    /// Adds the echoes of `bus` (N_AMBI x n) to it in place; false when long silent (skipped).
    fn process(&mut self, bus: &mut [f32], n: usize) -> bool {
        let loud = bus[..n].iter().any(|v| *v != 0.0);
        self.quiet = if loud { 0 } else { self.quiet.saturating_add(n) };
        if self.quiet > self.max_delay + ((1.5 * self.tail_rt + 0.5) * self.sr) as usize {
            return false;
        }
        if self.x.len() < n || self.ring.len() < self.max_delay + 2 * n + 1 {
            let size = (self.max_delay + 2 * n + 1).next_power_of_two();
            let mut nr = vec![0.0; size];
            for i in 0..self.ring.len().min(size) {
                let abs = self.w.wrapping_sub(i + 1);
                nr[abs & (size - 1)] = self.ring[abs & self.mask];
            }
            self.ring = nr;
            self.mask = size - 1;
            self.x = vec![0.0; n];
            self.bus = vec![0.0; N_AMBI * n];
        }
        for i in 0..n {
            self.ring[(self.w + i) & self.mask] = bus[i];
        }
        self.bus[..N_AMBI * n].copy_from_slice(&bus[..N_AMBI * n]);
        for tap in &mut self.taps {
            let x = &mut self.x[..n];
            for (i, v) in x.iter_mut().enumerate() {
                *v = self.ring[(self.w + i).wrapping_sub(tap.delay) & self.mask];
            }
            tap.f[0].process_block(x);
            tap.f[1].process_block(x);
            for ch in 0..N_AMBI {
                let g = tap.y[ch] * tap.g;
                if g == 0.0 {
                    continue;
                }
                let o = &mut bus[ch * n..(ch + 1) * n];
                for (ov, xv) in o.iter_mut().zip(x.iter()) {
                    *ov += g * xv;
                }
            }
        }
        self.w = self.w.wrapping_add(n);
        self.room.process_add(&self.bus[..N_AMBI * n], n, bus, self.tail_wet);
        true
    }

    fn reset(&mut self) {
        self.ring.iter_mut().for_each(|v| *v = 0.0);
        self.w = 0;
        self.quiet = 1_000_000_000;
        for t in &mut self.taps {
            t.f[0].reset();
            t.f[1].reset();
        }
        self.room.reset();
    }
}

// --------------------------------------------------------------------------
// the voice
// --------------------------------------------------------------------------
struct FlashCue {
    tf: f64,
    cell: usize,
    si: f64,
    k: u64,
    j: u64,
}

/// Thunder from the storm cells in WeatherState.storms (stem 'thunder', SH bus).
pub struct ThunderVoice {
    sr: f32,
    seed: u64,
    block_s: f64,
    heading: f32,
    listener_h: f64,
    events: Vec<Event>,
    pending: Vec<(f64, String)>,
    echo: EchoField,
    tools: Tools,
    flashes: Vec<FlashCue>,
    key_flash: u64,
    key_rumble: u64,
    key_growl: u64,
    /// 4th-order high-pass at 22 Hz per SH channel: thunder's sub-sonic content (the infrasonic
    /// pulse, < 20 Hz; Few 1969, Bohannon et al. 1977) is inaudible and attenuated.
    infra: Vec<Biquad>,
}

impl ThunderVoice {
    pub fn new(scene: &Scene, patches: &[Patch], cfg: &BuildCfg) -> ThunderVoice {
        npy::warm_up();
        let sr = cfg.sr as f32;
        ThunderVoice {
            sr,
            seed: cfg.seed,
            block_s: cfg.block.max(1) as f64 / cfg.sr as f64,
            heading: scene.listener.heading_deg,
            listener_h: scene.listener.height_m as f64,
            events: Vec::with_capacity(MAX_EVENTS + 1),
            pending: Vec::with_capacity(64),
            echo: EchoField::new(scene, patches, cfg),
            tools: Tools::new(sr, cfg.block.max(1)),
            flashes: Vec::with_capacity(MAX_FLASHES_PER_SLOT),
            key_flash: cfg.rng.key("thunder.flash"),
            key_rumble: cfg.rng.key("thunder.rumble"),
            key_growl: cfg.rng.key("thunder.growl"),
            infra: (0..2 * N_AMBI)
                .map(|i| Biquad::new(BiquadCoefs::highpass(sr, 22.0, if i % 2 == 0 { 0.541_196_1 } else { 1.306_563 })))
                .collect(),
        }
    }

    /// Flashes in [t0, t1) drawn exactly as StormSystem.flashes_between (NumPy streams per block slot).
    fn collect_flashes(&mut self, ctx: &BlockCtx) {
        self.flashes.clear();
        let w = ctx.weather;
        if w.storms.is_empty() {
            return;
        }
        let t0 = ctx.t0;
        let t1 = t0 + ctx.n as f64 / ctx.sr as f64;
        let bd = self.block_s;
        let k0 = (t0 / bd).floor().max(0.0) as u64;
        let k1 = ((t1 - 1e-12) / bd).floor().max(0.0) as u64;
        let mut times = [0.0f64; MAX_FLASHES_PER_SLOT];
        for (ci, cell) in w.storms.iter().enumerate() {
            let lam = (cell.flash_rate_per_min.max(0.0) as f64) / 60.0 * bd;
            if lam <= 0.0 {
                continue;
            }
            let key = npy::layer_key_parts(self.seed, &[b"storm.flash.", cell.id.as_bytes()]);
            for k in k0..=k1 {
                let mut rng = NpRng::for_block(key, k);
                let count = rng.poisson(lam) as usize;
                if count == 0 {
                    continue;
                }
                let cnt = count.min(MAX_FLASHES_PER_SLOT);
                for (i, t) in times.iter_mut().enumerate().take(count) {
                    let u = k as f64 * bd + rng.uniform(0.0, bd);
                    if i < cnt {
                        *t = u;
                    }
                }
                for (j, &tf) in times.iter().enumerate().take(cnt) {
                    let gk = rng.lognormal(0.0, 0.3);
                    if t0 <= tf && tf < t1 && self.flashes.len() < self.flashes.capacity() {
                        let si = (cell.stroke_intensity as f64 * gk).clamp(0.02, 1.0);
                        self.flashes.push(FlashCue { tf, cell: ci, si, k, j: j as u64 });
                    }
                }
            }
        }
        self.flashes.sort_by(|a, b| a.tf.partial_cmp(&b.tf).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn spawn(&mut self, ctx: &BlockCtx, i: usize) {
        let f = &self.flashes[i];
        let (tf, si, k, j) = (f.tf, f.si, f.k, f.j);
        let cell = &ctx.weather.storms[f.cell];
        let ekey = hash3(self.key_flash, hash_str(&cell.id), combine(k, j));
        let mut rng = Rng::from_key(ekey, k);
        let cg = rng.uniform() < cell.cg_fraction as f64;
        let (cx, cy) = strike_xy(cell.distance_km, cell.bearing_deg, self.heading);
        let sx = cx + rng.gauss(0.0, 1200.0);
        let sy = cy + rng.gauss(0.0, 1200.0);
        let eid = format!("{}.{}.{}", cell.id, k, j);
        let cell_id = cell.id.clone();
        self.launch(ctx, ekey, &eid, &cell_id, tf, sx, sy, cg, si, &mut rng);
    }

    /// A strike asked for with Engine.strike(): flash at the block start, the same lightning
    /// tree, N-wave synthesis and Ribner-Roy propagation as a storm-cell flash, so the thunder
    /// arrives at distance / c(T) from the nearest channel point.
    fn manual_strike(&mut self, ctx: &BlockCtx, st: &crate::engine::Strike) {
        let ekey = hash3(self.key_flash, hash_str("manual"), st.id);
        let mut rng = Rng::from_key(ekey, st.id);
        let d = if st.distance_km.is_finite() { st.distance_km.clamp(0.25, 40.0) } else { 3.0 };
        let b = if st.bearing_deg.is_finite() { st.bearing_deg } else { 0.0 };
        let (sx, sy) = strike_xy(d, b, self.heading);
        let si = if st.intensity.is_finite() { (st.intensity as f64).clamp(0.02, 1.0) } else { 0.7 };
        let eid = format!("strike.{}", st.id);
        self.launch(ctx, ekey, &eid, "manual", ctx.t0, sx, sy, true, si, &mut rng);
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &mut self,
        ctx: &BlockCtx,
        ekey: u64,
        eid: &str,
        cell_id: &str,
        tf: f64,
        mut sx: f64,
        mut sy: f64,
        cg: bool,
        si: f64,
        rng: &mut Rng,
    ) {
        let mut dh = sx.hypot(sy);
        if dh < 250.0 {
            if dh > 1e-6 {
                let s = 250.0 / dh;
                sx *= s;
                sy *= s;
            } else {
                sx = 250.0;
                sy = 0.0;
            }
            dh = 250.0;
        }
        let z_g = -self.listener_h;
        let (polys, z_b) = lightning_tree(rng, sx, sy, cg, z_g);
        let ev = Event::new(ekey, tf, cg, &polys, z_b, si, ctx.weather, self.listener_h, &mut self.tools, rng);
        let az = sy.atan2(sx);
        let bearing = (self.heading as f64 - az.to_degrees()).rem_euclid(360.0);
        let typ = if cg { "cg" } else { "ic" };
        // lightning marker (with the channel polylines)
        let mut s = String::with_capacity(4096);
        let _ = write!(
            s,
            "{{\"id\": {}, \"cell\": {}, \"type\": \"{}\", \"distance_km\": {}, \"bearing_deg\": {}, \"azimuth\": {}, \
             \"intensity\": {}, \"energy_j_per_m\": {}, \"strokes_s\": [",
            jstr(eid),
            jstr(cell_id),
            typ,
            jnum(dh / 1000.0, 3),
            jnum(bearing, 1),
            jnum(az, 4),
            jnum(si, 3),
            jnum(stroke_energy_j_per_m(si), 1)
        );
        for (i, st) in ev.strokes.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&jnum(*st, 4));
        }
        let _ = write!(
            s,
            "], \"cloud_base_m\": {}, \"flash_to_bang_s\": {}, \"audible_range_km\": {}, \"channel\": [",
            jnum(z_b - z_g, 1),
            jnum(ev.t_first - tf, 4),
            jnum(ev.audible_range_km, 2)
        );
        for (pi, (pts, kind)) in polys.iter().enumerate() {
            if pi > 0 {
                s.push_str(", ");
            }
            s.push('[');
            let step = if matches!(kind, Kind::Main | Kind::Ic) { 1 } else { 2 };
            for (qi, p) in pts.iter().step_by(step).enumerate() {
                if qi > 0 {
                    s.push_str(", ");
                }
                let _ = write!(s, "[{}, {}, {}]", jnum(p[0], 1), jnum(p[1], 1), jnum(p[2], 1));
            }
            s.push(']');
        }
        s.push_str("], \"frame\": \"listener: +x forward, +y left, +z up, metres from the ears\"}");
        ctx.marker(tf, "lightning", s);
        // thunder marker at the first arrival, audible or not
        let thunder = format!(
            "{{\"id\": {}, \"cell\": {}, \"flash_t\": {}, \"type\": \"{}\", \"distance_km\": {}, \"azimuth\": {}, \
             \"elevation\": {}, \"audible\": {}, \"peak_db_spl\": {}, \"duration_s\": {}}}",
            jstr(eid),
            jstr(cell_id),
            jnum(tf, 4),
            typ,
            jnum(ev.r_min / 1000.0, 4),
            jnum(ev.near_dir.0, 4),
            jnum(ev.near_dir.1, 4),
            ev.audible,
            jnum(20.0 * (ev.peak_pa.max(1e-9) / 20e-6).log10(), 1),
            jnum((ev.end_sample as f64 / self.sr as f64 - ev.t_first).max(0.0), 2)
        );
        let t_first = ev.t_first;
        let pos = self.pending.partition_point(|(t, _)| *t <= t_first);
        self.pending.insert(pos, (t_first, thunder));
        if !ev.audible {
            return;
        }
        self.events.push(ev);
        if self.events.len() > MAX_EVENTS {
            let mut q = 0;
            for (i, e) in self.events.iter().enumerate() {
                if e.peak_pa < self.events[q].peak_pa {
                    q = i;
                }
            }
            self.events.remove(q);
        }
    }
}

impl Voice for ThunderVoice {
    fn name(&self) -> &str {
        "thunder"
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        if n != self.tools.n {
            self.tools = Tools::new(self.sr, n);
            for ev in &mut self.events {
                ev.acc = None;
                ev.x_started = false;
                ev.rumble = None;
            }
        }
        let start = (ctx.t0 * ctx.sr as f64).round() as i64;
        self.collect_flashes(ctx);
        for i in 0..self.flashes.len() {
            self.spawn(ctx, i);
        }
        for st in ctx.strikes {
            self.manual_strike(ctx, st);
        }
        let t1 = ctx.t0 + n as f64 / ctx.sr as f64;
        let due = self.pending.partition_point(|(t, _)| *t < t1);
        if due > 0 {
            for (t, data) in self.pending.drain(..due) {
                ctx.marker(t, "thunder", data);
            }
        }
        let keys = (self.key_rumble, self.key_growl);
        let next = start + n as i64;
        for ev in self.events.iter_mut() {
            ev.render(&mut self.tools, ctx, keys, start, n, out);
        }
        self.events.retain(|e| !e.finished(next));
        let echoed = self.echo.process(out, n);
        if !echoed && self.events.is_empty() {
            return Emission::silent(Stem::Thunder);
        }
        for ch in 0..N_AMBI {
            let o = &mut out[ch * n..(ch + 1) * n];
            self.infra[2 * ch].process_block(o);
            self.infra[2 * ch + 1].process_block(o);
        }
        for v in out[..N_AMBI * n].iter_mut() {
            if !v.is_finite() || v.abs() < 1e-30 {
                *v = 0.0;
            }
        }
        Emission::diffuse(Stem::Thunder)
    }

    fn reset(&mut self) {
        self.events.clear();
        self.pending.clear();
        self.echo.reset();
        self.infra.iter_mut().for_each(|b| b.reset());
    }
}

#[allow(dead_code)]
fn _api(_: f64) -> f64 {
    source_frequency_hz(5.0e4, 15.0)
}

// --------------------------------------------------------------------------
// continuous cloud rumble (weather.thunder.rumble)
// --------------------------------------------------------------------------
/// The continuous low rumble of storms beyond the individual flashes: many distant, overlapping
/// and in-cloud discharges whose thunder arrives refracted and absorbed down to its lowest band
/// (energy 30-100 Hz; Holmes et al. 1971: distant thunder spectra peak 50-100 Hz; Few 1969). Slow
/// swells (OU processes, ~1 s and ~6 s) and rolling surges. Auto: louder the nearer and the more
/// active the storm cells; a fixed level (0..1) plays even without storms, still lifted by them.
pub struct CloudRumbleVoice {
    lvl: crate::env::common::AmbLevel,
    sr: f32,
    key: u64,
    heading: f32,
    filt: Vec<Biquad>,
    g: f32,
    ou_fast: f64,
    ou_slow: f64,
    surge: f64,
    storm: f64,
    az: f64,
    amp: Ramp,
}

impl CloudRumbleVoice {
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> CloudRumbleVoice {
        let sr = cfg.sr as f32;
        let secs = [
            BiquadCoefs::highpass(sr, 28.0, 0.541_196_1),
            BiquadCoefs::highpass(sr, 28.0, 1.306_563),
            BiquadCoefs::lowpass(sr, 95.0, 0.541_196_1),
            BiquadCoefs::lowpass(sr, 95.0, 1.306_563),
            peak_eq(55.0, 4.0, 0.8, sr),
        ];
        CloudRumbleVoice {
            lvl: crate::env::common::AmbLevel::new(crate::types::amb::RUMBLE),
            sr,
            key: cfg.rng.key("thunder.cloud_rumble"),
            heading: scene.listener.heading_deg,
            g: noise_rms_gain(&secs),
            filt: secs.iter().map(|c| Biquad::new(*c)).collect(),
            ou_fast: 0.0,
            ou_slow: 0.0,
            surge: 0.0,
            storm: -1.0,
            az: bearing_to_azimuth(scene.weather.thunder.bearing_deg, scene.listener.heading_deg) as f64,
            amp: Ramp::default(),
        }
    }

    /// (storm factor 0..~1.5, weighted azimuth) from the storm cells.
    fn storms(&self, w: &Weather) -> (f64, Option<f64>) {
        let (mut s, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
        for c in &w.storms {
            let d = (c.distance_km as f64).max(0.3);
            let near = 1.0 / (1.0 + (d / 10.0).powf(1.5));
            let active = 0.35 + 0.65 * smoothstep64(c.flash_rate_per_min as f64, 0.1, 6.0);
            let si = 0.5 + 0.5 * (c.stroke_intensity as f64).clamp(0.0, 1.0);
            let k = near * active * si;
            s += k;
            let a = bearing_to_azimuth(c.bearing_deg, self.heading) as f64;
            x += k * a.cos();
            y += k * a.sin();
        }
        let dir = if s > 1e-6 { Some(y.atan2(x)) } else { None };
        (s.min(1.5), dir)
    }
}

impl Voice for CloudRumbleVoice {
    fn name(&self) -> &str {
        "thunder.cloud_rumble"
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.sr as f64;
        let dt = n as f64 / sr;
        let (s_now, dir) = self.storms(ctx.weather);
        if self.storm < 0.0 {
            self.storm = s_now;
        }
        // storms build and fade over tens of seconds
        self.storm += (s_now - self.storm) * (1.0 - (-dt / 8.0).exp());
        if let Some(a) = dir {
            let d = (a - self.az + PI).rem_euclid(2.0 * PI) - PI;
            self.az += d * (1.0 - (-dt / 5.0).exp());
        }
        let st = self.storm.min(1.0);
        let auto = 0.75 * st;
        let raw = self.lvl.raw(ctx);
        let v = self.lvl.step(ctx, auto);
        let g = match raw {
            Some(_) => v * (0.55 + 0.45 * st),
            None => v,
        };
        let target = db_spl(68.0) * g as f32;
        if target <= 1e-9 && self.amp.idle() {
            return Emission::silent(Stem::Thunder);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let af = (-dt / 1.2).exp();
        let as_ = (-dt / 6.0).exp();
        self.ou_fast = af * self.ou_fast + (1.0 - af * af).sqrt() * rng.normal();
        self.ou_slow = as_ * self.ou_slow + (1.0 - as_ * as_).sqrt() * rng.normal();
        // rolling surges: a few per minute, more with an active storm
        if rng.uniform() < dt * (0.04 + 0.08 * st) {
            self.surge += rng.range(0.6, 1.4);
        }
        self.surge *= (-dt / 3.5).exp();
        let swell = (0.45 * self.ou_slow + 0.25 * self.ou_fast).exp() * (1.0 + 0.8 * self.surge.min(2.0));
        let (g0, g1) = self.amp.to(target * swell as f32);
        let o = &mut out[..n];
        fill_gauss(&mut rng, o);
        for f in self.filt.iter_mut() {
            f.process_block(o);
        }
        let inv = 1.0 / self.g;
        for (i, v) in o.iter_mut().enumerate() {
            *v *= ramp_at(g0, g1, i, n) * inv;
        }
        Emission::point(self.az as f32, 0.12, 3000.0, 0.75, Stem::Thunder)
    }

    fn reset(&mut self) {
        self.lvl.reset();
        for f in self.filt.iter_mut() {
            f.reset();
        }
        self.ou_fast = 0.0;
        self.ou_slow = 0.0;
        self.surge = 0.0;
        self.storm = -1.0;
        self.amp = Ramp::default();
    }
}
