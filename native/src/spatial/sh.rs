//! Real spherical harmonics, AmbiX convention (ACN order, SN3D, no Condon-Shortley phase),
//! the Rust twin of raingen/spatial/sh.py for the engine's order-3 bus.
//!
//!   Y_n^m(az, el) = N_n^|m| P_n^|m|(sin el) * { cos(m az)  m >= 0 ; sin(|m| az)  m < 0 }
//!   N_n^|m| = sqrt((2 - delta_m0) (n - |m|)! / (n + |m|)!),   ACN = n^2 + n + m

use crate::types::{AMBI_ORDER, N_AMBI};
use std::sync::OnceLock;

pub const ORDER: usize = AMBI_ORDER;

/// Degree n of each ACN channel.
pub const ACN_DEGREE: [usize; N_AMBI] = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3];

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |a, k| a * k as f64)
}

/// SN3D normalisation N_n^m for n, m <= ORDER (computed once).
fn sn3d_norm() -> &'static [[f64; ORDER + 1]; ORDER + 1] {
    static NORM: OnceLock<[[f64; ORDER + 1]; ORDER + 1]> = OnceLock::new();
    NORM.get_or_init(|| {
        let mut t = [[0.0; ORDER + 1]; ORDER + 1];
        for n in 0..=ORDER {
            for m in 0..=n {
                let d = if m == 0 { 1.0 } else { 0.0 };
                t[n][m] = ((2.0 - d) * factorial(n - m) / factorial(n + m)).sqrt();
            }
        }
        t
    })
}

/// Real SN3D spherical harmonics in ACN order (order 3, 16 values), f64.
pub fn sh_real(az: f64, el: f64) -> [f64; N_AMBI] {
    let (x, c) = el.sin_cos();
    let norm = sn3d_norm();
    let mut p = [[0.0f64; ORDER + 1]; ORDER + 1];
    let mut pmm = 1.0;
    for m in 0..=ORDER {
        if m > 0 {
            pmm *= (2 * m - 1) as f64 * c;
        }
        p[m][m] = pmm;
        if m < ORDER {
            p[m + 1][m] = x * (2 * m + 1) as f64 * pmm;
        }
        for l in (m + 2)..=ORDER {
            p[l][m] = ((2 * l - 1) as f64 * x * p[l - 1][m] - (l + m - 1) as f64 * p[l - 2][m]) / (l - m) as f64;
        }
    }
    // sin / cos of m az by the angle-addition recurrence (one sin_cos call)
    let mut sm = [0.0f64; ORDER + 1];
    let mut cm = [1.0f64; ORDER + 1];
    let (s1, c1) = az.sin_cos();
    for m in 1..=ORDER {
        sm[m] = sm[m - 1] * c1 + cm[m - 1] * s1;
        cm[m] = cm[m - 1] * c1 - sm[m - 1] * s1;
    }
    let mut out = [0.0f64; N_AMBI];
    for n in 0..=ORDER {
        let base = n * n + n;
        out[base] = norm[n][0] * p[n][0];
        for m in 1..=n {
            let nm = norm[n][m] * p[n][m];
            out[base + m] = nm * cm[m];
            out[base - m] = nm * sm[m];
        }
    }
    out
}

pub fn sh_real_f32(az: f32, el: f32) -> [f32; N_AMBI] {
    let y = sh_real(az as f64, el as f64);
    let mut o = [0.0f32; N_AMBI];
    for k in 0..N_AMBI {
        o[k] = y[k] as f32;
    }
    o
}

/// Legendre polynomials P_0..P_{ORDER+1} at x.
pub fn legendre(x: f64) -> [f64; ORDER + 2] {
    let mut out = [0.0f64; ORDER + 2];
    out[0] = 1.0;
    out[1] = x;
    for n in 2..ORDER + 2 {
        out[n] = ((2 * n - 1) as f64 * x * out[n - 1] - (n - 1) as f64 * out[n - 2]) / n as f64;
    }
    out
}

/// Per-degree SH coefficients of a uniform spherical cap of half-angle `a` (c_0 = 1).
pub fn cap_weights(half_angle: f64) -> [f64; ORDER + 1] {
    let mut out = [1.0f64; ORDER + 1];
    if half_angle <= 1e-6 {
        return out;
    }
    let x = half_angle.min(std::f64::consts::PI).cos();
    let p = legendre(x);
    let den = 1.0 - x;
    for n in 1..=ORDER {
        out[n] = (p[n - 1] - p[n + 1]) / ((2 * n + 1) as f64 * den);
    }
    out
}

/// max-rE per-degree weights g_n = P_n(r_E), r_E = largest root of P_{ORDER+1}.
pub fn maxre_weights() -> [f64; ORDER + 1] {
    // largest root of P_4: sqrt((3 + 2 sqrt(6/5)) / 7)
    let re = ((3.0 + 2.0 * (6.0f64 / 5.0).sqrt()) / 7.0).sqrt();
    let p = legendre(re);
    let mut out = [0.0; ORDER + 1];
    out.copy_from_slice(&p[..ORDER + 1]);
    out
}

/// Expand per-degree weights to the 16 ACN channels.
pub fn per_acn(w: &[f64; ORDER + 1]) -> [f64; N_AMBI] {
    let mut o = [0.0; N_AMBI];
    for k in 0..N_AMBI {
        o[k] = w[ACN_DEGREE[k]];
    }
    o
}

/// Target (coherent, diffuse) per-channel gains for a source of the given spread, exactly as
/// raingen.spatial.encoder.spread_gains: the coherent part is the cap expansion of half-angle
/// spread*pi (scaled by sqrt(1 - d) above W), the diffuse part (d = spread^2 of the energy) feeds
/// the decorrelators with the diffuse-field energy 1/(2n+1) per SN3D channel. W carries the
/// signal unchanged, so loudness never depends on spread.
pub fn spread_gains(az: f32, el: f32, spread: f32) -> ([f32; N_AMBI], [f32; N_AMBI]) {
    let s = if spread.is_finite() { spread.clamp(0.0, 1.0) as f64 } else { 0.0 };
    let y = sh_real(az as f64, el as f64);
    let mut g_coh = [0.0f32; N_AMBI];
    let mut g_dif = [0.0f32; N_AMBI];
    if s <= 0.0 {
        for k in 0..N_AMBI {
            g_coh[k] = y[k] as f32;
        }
        return (g_coh, g_dif);
    }
    let c = per_acn(&cap_weights(s * std::f64::consts::PI));
    let d = s * s;
    let k_coh = (1.0 - d).max(0.0).sqrt();
    for k in 0..N_AMBI {
        let v = y[k] * c[k];
        g_coh[k] = if k == 0 { v as f32 } else { (v * k_coh) as f32 };
        if d > 0.0 && k > 0 {
            let n = ACN_DEGREE[k] as f64;
            let e = (1.0 - c[k] * c[k]) / (2.0 * n + 1.0) + (c[k] * y[k]).powi(2);
            g_dif[k] = (d * e.max(0.0)).sqrt() as f32;
        }
    }
    (g_coh, g_dif)
}

/// SH-domain rotation about the vertical axis (yaw, rad, CCW): b' = R b. Row-major 16 x 16.
pub fn yaw_rotation(yaw: f32) -> [[f32; N_AMBI]; N_AMBI] {
    let mut r = [[0.0f32; N_AMBI]; N_AMBI];
    r[0][0] = 1.0;
    for n in 1..=ORDER {
        let base = n * n + n;
        r[base][base] = 1.0;
        for m in 1..=n {
            let (s, c) = ((m as f32) * yaw).sin_cos();
            // cos part (base + m) and sin part (base - m) rotate like a 2-D vector
            r[base + m][base + m] = c;
            r[base + m][base - m] = -s;
            r[base - m][base + m] = s;
            r[base - m][base - m] = c;
        }
    }
    r
}

/// Least-squares projector onto the order-3 SH span from a fixed Fibonacci point set:
/// (points, P = (A^T A)^-1 A^T as N_AMBI rows of Q values). Exact for any order-3 field.
fn rotation_projector() -> &'static (Vec<[f64; 3]>, Vec<[f64; N_AMBI]>, Vec<Vec<f64>>) {
    static P: OnceLock<(Vec<[f64; 3]>, Vec<[f64; N_AMBI]>, Vec<Vec<f64>>)> = OnceLock::new();
    P.get_or_init(|| {
        const Q: usize = 64;
        let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let mut pts = Vec::with_capacity(Q);
        let mut a = Vec::with_capacity(Q);
        for q in 0..Q {
            let z = 1.0 - 2.0 * (q as f64 + 0.5) / Q as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = ga * q as f64;
            let v = [r * phi.cos(), r * phi.sin(), z];
            pts.push(v);
            a.push(sh_real(v[1].atan2(v[0]), v[2].atan2(v[0].hypot(v[1]))));
        }
        // normal matrix and its inverse (Gauss-Jordan, partial pivoting)
        let mut m = [[0.0f64; 2 * N_AMBI]; N_AMBI];
        for i in 0..N_AMBI {
            for j in 0..N_AMBI {
                m[i][j] = a.iter().map(|row| row[i] * row[j]).sum();
            }
            m[i][N_AMBI + i] = 1.0;
        }
        for c in 0..N_AMBI {
            let piv = (c..N_AMBI).max_by(|x, y| m[*x][c].abs().partial_cmp(&m[*y][c].abs()).unwrap()).unwrap();
            m.swap(c, piv);
            let d = m[c][c];
            for v in m[c].iter_mut() {
                *v /= d;
            }
            for r in 0..N_AMBI {
                if r != c {
                    let f = m[r][c];
                    if f != 0.0 {
                        for k in 0..2 * N_AMBI {
                            m[r][k] -= f * m[c][k];
                        }
                    }
                }
            }
        }
        let p: Vec<Vec<f64>> = (0..N_AMBI)
            .map(|i| (0..Q).map(|q| (0..N_AMBI).map(|j| m[i][N_AMBI + j] * a[q][j]).sum()).collect())
            .collect();
        (pts, a, p)
    })
}

/// SH-domain rotation of the whole sound field for a listener who turned by `dheading`
/// (rad, compass sense: clockwise = turning right) and pitched up by `pitch` (rad): c = R b,
/// row-major. A source straight ahead moves to the left (+azimuth) when the listener turns right.
pub fn listener_rotation(dheading: f32, pitch: f32) -> [[f32; N_AMBI]; N_AMBI] {
    let mut r = [[0.0f32; N_AMBI]; N_AMBI];
    if dheading == 0.0 && pitch == 0.0 {
        for (k, row) in r.iter_mut().enumerate() {
            row[k] = 1.0;
        }
        return r;
    }
    let (pts, _, p) = rotation_projector();
    // H = Rz(-dheading) Ry(-pitch): head-frame direction -> reference-frame direction
    let (sy, cy) = (-(dheading as f64)).sin_cos();
    let (sp, cp) = (-(pitch as f64)).sin_cos();
    let b: Vec<[f64; N_AMBI]> = pts
        .iter()
        .map(|u| {
            let x1 = u[0] * cp + u[2] * sp;
            let z1 = -u[0] * sp + u[2] * cp;
            let y1 = u[1];
            let x2 = x1 * cy - y1 * sy;
            let y2 = x1 * sy + y1 * cy;
            sh_real(y2.atan2(x2), z1.atan2(x2.hypot(y2)))
        })
        .collect();
    for i in 0..N_AMBI {
        for j in 0..N_AMBI {
            let v: f64 = p[i].iter().zip(&b).map(|(pq, bq)| pq * bq[j]).sum();
            r[i][j] = if v.abs() < 1e-7 { 0.0 } else { v as f32 };
        }
    }
    r
}
