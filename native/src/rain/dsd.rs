//! Drop-size distribution pieces the precipitation voices need (port of raingen/physics/dsd.py):
//! terminal fall speed (Atlas 1973 / Stokes / bridge), sub-terminal impact speed after a fall,
//! Tate's law for drips, the acoustic class bounds and the flux-weighted per-class size table
//! with its Gauss-Legendre quadrature (`_ClassTable` of raingen/synth/rain.py).

use crate::types::{ln_gamma, Dsd, G, N_CLASSES, RHO_AIR, RHO_WATER, SIGMA_WATER};

const MU_AIR: f64 = 1.81e-5;
const D_STOKES_MM: f64 = 0.1;
const D_ATLAS_MM: f64 = 0.5;
const HARKINS_BROWN: f64 = 0.6;

#[inline]
fn stokes_k() -> f64 {
    G as f64 * 1e-6 * (RHO_WATER as f64 - RHO_AIR as f64) / (18.0 * MU_AIR)
}

#[inline]
fn atlas(d: f64) -> f64 {
    9.65 - 10.3 * (-0.6 * d).exp()
}

/// Terminal fall speed (m/s) of a raindrop of diameter `d_mm` at sea level.
#[inline]
pub fn fall_speed(d_mm: f32) -> f32 {
    let d = d_mm as f64;
    let v = if d >= D_ATLAS_MM {
        atlas(d).max(0.0)
    } else if d >= D_STOKES_MM {
        let vs = stokes_k() * D_STOKES_MM * D_STOKES_MM;
        let va = atlas(D_ATLAS_MM);
        vs + (va - vs) / (D_ATLAS_MM - D_STOKES_MM) * (d - D_STOKES_MM)
    } else if d > 0.0 {
        stokes_k() * d * d
    } else {
        0.0
    };
    v as f32
}

/// Impact speed (m/s) of a drop released at rest after a fall of `h_m` metres.
#[inline]
pub fn fall_speed_from_height(d_mm: f32, h_m: f32) -> f32 {
    let vt = fall_speed(d_mm) as f64;
    if vt <= 0.0 {
        return 0.0;
    }
    let h = (h_m as f64).max(0.0);
    let vs = vt.max(1e-6);
    (vt * (1.0 - (-2.0 * G as f64 * h / (vs * vs)).exp()).sqrt()) as f32
}

/// Detaching drop diameter (mm) from an edge of radius `edge_radius_mm` (Tate's law).
pub fn tate_drop_diameter_mm(edge_radius_mm: f32) -> f32 {
    let r = (edge_radius_mm as f64).max(0.05) * 1e-3;
    let vol = 2.0 * std::f64::consts::PI * r * SIGMA_WATER as f64 * HARKINS_BROWN / (RHO_WATER as f64 * G as f64);
    let d = (6.0 * vol / std::f64::consts::PI).cbrt() * 1e3;
    d.clamp(1.5, 7.0) as f32
}

/// Class edges of the two boundary schemes (Medwin et al. 1992 / Nystuen 2001).
pub fn class_edges(nystuen: bool) -> [f64; 4] {
    if nystuen {
        [0.8, 1.2, 2.0, 3.5]
    } else {
        [0.8, 1.1, 2.2, 3.5]
    }
}

/// (lo, hi) diameter bounds of the five classes.
pub fn class_bounds(nystuen: bool, d_max_mm: f64) -> [(f64, f64); N_CLASSES] {
    let e = class_edges(nystuen);
    let top = d_max_mm.max(e[3]);
    let pts = [0.0, e[0], e[1], e[2], e[3], top];
    let mut out = [(0.0, 0.0); N_CLASSES];
    for i in 0..N_CLASSES {
        out[i] = (pts[i], pts[i + 1]);
    }
    out
}

/// f(mu) of the normalized gamma DSD.
#[inline]
fn f_mu(mu: f64) -> f64 {
    6.0 / 256.0 * ((mu + 4.0) * (4.0 + mu).ln() - ln_gamma(mu + 4.0)).exp()
}

/// Unit-Nw number density shape N(D)/Nw (0 outside the cutoffs).
#[inline]
fn shape_density(dm: f64, mu: f64, fmu: f64, d_min: f64, d_max: f64, d: f64) -> f64 {
    if d < d_min || d > d_max || dm <= 0.0 {
        return 0.0;
    }
    let x = d.max(1e-9) / dm;
    fmu * (mu * x.ln() - (4.0 + mu) * x).exp()
}

/// Gauss-Legendre nodes / weights (8 points on [-1, 1], ascending as numpy's leggauss).
pub const QUAD: usize = 8;
const GL_X: [f64; QUAD] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329_0,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329_0,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL_W: [f64; QUAD] = [
    0.101_228_536_290_376_3,
    0.222_381_034_453_374_5,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362_0,
    0.362_683_783_378_362_0,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

const GRID: usize = 768;

/// Per-DSD class geometry: bounds, representative fall speeds, flux-weighted inverse CDF and the
/// per-class quadrature (diameters / weights in the flux quantile).
#[derive(Clone, Debug)]
pub struct ClassTable {
    pub valid: [bool; N_CLASSES],
    c_lo: [f64; N_CLASSES],
    c_hi: [f64; N_CLASSES],
    grid: Box<[f64; GRID]>,
    cdf: Box<[f64; GRID]>,
    /// Fall speed at each class's geometric-mean diameter (for the wind-angle flux factor).
    pub vt_c: [f32; N_CLASSES],
    /// Quadrature weights (sum 1).
    pub wq: [f32; QUAD],
    /// Quadrature diameters (mm) per class.
    pub dq: [[f32; QUAD]; N_CLASSES],
    /// Fall speeds at the quadrature diameters.
    pub vtq: [[f32; QUAD]; N_CLASSES],
    /// The DSD shape the table was built for: (Dm, mu, d_min, d_max).
    pub built: (f64, f64, f64, f64),
    pub nystuen: bool,
    pub ready: bool,
}

impl Default for ClassTable {
    fn default() -> Self {
        ClassTable {
            valid: [false; N_CLASSES],
            c_lo: [0.0; N_CLASSES],
            c_hi: [0.0; N_CLASSES],
            grid: Box::new([0.0; GRID]),
            cdf: Box::new([0.0; GRID]),
            vt_c: [0.0; N_CLASSES],
            wq: [0.0; QUAD],
            dq: [[0.0; QUAD]; N_CLASSES],
            vtq: [[0.0; QUAD]; N_CLASSES],
            built: (0.0, 0.0, 0.0, 0.0),
            nystuen: false,
            ready: false,
        }
    }
}

/// np.interp(x, xp, fp) for nondecreasing xp (flat runs pick the first knot of the run end).
#[inline]
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // first index with xp[i] > x
    let i = xp.partition_point(|&v| v <= x);
    let (x0, x1) = (xp[i - 1], xp[i]);
    let (y0, y1) = (fp[i - 1], fp[i]);
    if x1 > x0 {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    } else {
        y1
    }
}

impl ClassTable {
    /// True when the table in use still stands for `dsd` (Dm within 2 %, mu within 0.1, same
    /// cutoffs to 0.01 mm): the hysteresis of `_RainField._table`.
    pub fn matches(&self, dsd: &Dsd, nystuen: bool) -> bool {
        if !self.ready || self.nystuen != nystuen {
            return false;
        }
        let dm = (dsd.dm_mm as f64).max(1e-6);
        let r2 = |v: f64| (v * 100.0).round() / 100.0;
        (dm / self.built.0).ln().abs() <= 0.02
            && (dsd.mu as f64 - self.built.1).abs() <= 0.1
            && r2(dsd.d_min_mm as f64) == r2(self.built.2)
            && r2(dsd.d_max_mm as f64) == r2(self.built.3)
    }

    /// Rebuild for `dsd` (no allocation).
    pub fn build(&mut self, dsd: &Dsd, nystuen: bool) {
        let d_min = dsd.d_min_mm as f64;
        let d_max = dsd.d_max_mm as f64;
        let dm = (dsd.dm_mm as f64).max(1e-6);
        let mu = (dsd.mu as f64).clamp(-3.0, 20.0);
        let bounds = class_bounds(nystuen, d_max);
        let mut lo = [0.0; N_CLASSES];
        let mut hi = [0.0; N_CLASSES];
        for c in 0..N_CLASSES {
            lo[c] = bounds[c].0.max(d_min);
            hi[c] = bounds[c].1.min(d_max);
            self.valid[c] = hi[c] > lo[c];
        }
        let d0 = d_min;
        let d1 = d_max.max(d_min + 1e-3);
        let fmu = f_mu(mu);
        let mut prev_pdf = 0.0;
        for i in 0..GRID {
            let d = d0 + (d1 - d0) * i as f64 / (GRID - 1) as f64;
            self.grid[i] = d;
            let pdf = shape_density(dm, mu, fmu, d_min, d_max, d) * fall_speed(d as f32) as f64;
            if i == 0 {
                self.cdf[0] = 0.0;
            } else {
                let dx = d - self.grid[i - 1];
                self.cdf[i] = self.cdf[i - 1] + 0.5 * (pdf + prev_pdf) * dx;
            }
            prev_pdf = pdf;
        }
        for c in 0..N_CLASSES {
            self.c_lo[c] = interp(lo[c], &self.grid[..], &self.cdf[..]);
            self.c_hi[c] = interp(hi[c], &self.grid[..], &self.cdf[..]);
            self.valid[c] &= self.c_hi[c] > self.c_lo[c];
            let gm = if lo[c] > 0.0 { (lo[c] * hi[c]).max(1e-12).sqrt() } else { 0.5 * hi[c] };
            self.vt_c[c] = fall_speed(gm as f32);
            for q in 0..QUAD {
                let u = 0.5 * (GL_X[q] + 1.0);
                let d = if self.valid[c] { self.sample_f64(u, c) } else { gm };
                self.dq[c][q] = d as f32;
                self.vtq[c][q] = fall_speed(d as f32);
            }
        }
        for q in 0..QUAD {
            self.wq[q] = (0.5 * GL_W[q]) as f32;
        }
        self.built = (dm, mu, d_min, d_max);
        self.nystuen = nystuen;
        self.ready = true;
    }

    #[inline]
    fn sample_f64(&self, u: f64, c: usize) -> f64 {
        let (lo, hi) = (self.c_lo[c], self.c_hi[c]);
        interp(lo + u * (hi - lo), &self.cdf[..], &self.grid[..])
    }

    /// Flux-weighted diameter (mm) of class `c` for a uniform `u` (inverse CDF of N(D) V(D)).
    #[inline]
    pub fn sample(&self, u: f64, c: usize) -> f32 {
        self.sample_f64(u, c) as f32
    }
}
