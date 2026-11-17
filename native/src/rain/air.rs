//! ISO 9613-1 air absorption for the precipitation family (the native counterpart of
//! raingen.physics.atmosphere.air_absorption_sos / AirAbsorptionFilter).
//!
//! The loss alpha(f) d (exact ISO 9613-1 formula with temperature and humidity) is fitted by a
//! bank of RBJ high shelves at octave corners (gains solved by non-negative least squares on a
//! log-frequency grid, then refined on the true shelf responses): within ~1 dB up to 10 kHz for
//! the distances rain sources sit at. Redesigns (temperature > 1 C or humidity > 5 % away) are
//! warmed on the previous block and crossfaded over one block, so they never click.

use crate::dsp::BiquadCoefs;

pub const MAX_STAGES: usize = 8;

/// Pure-tone atmospheric absorption coefficient (dB/m), ISO 9613-1.
pub fn iso9613_alpha_db_per_m(f: f64, temp_c: f64, rh_pct: f64, pressure_hpa: f64) -> f64 {
    let t = temp_c + 273.15;
    let t0 = 293.15;
    let t01 = 273.16;
    let p_rel = pressure_hpa.max(1.0) / 1013.25;
    let c = -6.8346 * (t01 / t).powf(1.261) + 4.6151;
    let psat_rel = 10f64.powf(c);
    let h = rh_pct.clamp(0.0, 100.0) * psat_rel / p_rel;
    let fr_o = p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    let fr_n = p_rel * (t / t0).powf(-0.5) * (9.0 + 280.0 * h * (-4.170 * ((t / t0).powf(-1.0 / 3.0) - 1.0)).exp());
    let f2 = f * f;
    8.686
        * f2
        * (1.84e-11 / p_rel * (t / t0).sqrt()
            + (t / t0).powf(-2.5)
                * (0.01275 * (-2239.1 / t).exp() / (fr_o + f2 / fr_o) + 0.1068 * (-3352.0 / t).exp() / (fr_n + f2 / fr_n)))
}

const NF: usize = 40;
const MAX_SHELVES: usize = 7;
const Q_S1: f32 = std::f32::consts::FRAC_1_SQRT_2;

fn shelf_loss_db(fc: f32, gain_db: f32, f: f32, sr: f32) -> f32 {
    if gain_db <= 0.0 {
        return 0.0;
    }
    let c = BiquadCoefs::high_shelf(sr, fc, Q_S1, -gain_db);
    -20.0 * c.magnitude(f, sr).max(1e-12).log10()
}

/// Shelf cascade (and its stage count) matching alpha(f) * distance; passthrough under 1 m.
pub fn design(distance_m: f32, temp_c: f32, rh_pct: f32, sr: f32) -> ([BiquadCoefs; MAX_STAGES], usize) {
    let mut out = [BiquadCoefs::identity(); MAX_STAGES];
    if distance_m < 1.0 {
        return (out, 0);
    }
    let f_hi = (0.45 * sr).min(20000.0);
    let f_lo = 63.0f32;
    let mut fs = [0.0f32; NF];
    let mut target = [0.0f32; NF];
    let mut tmax = 0.0f32;
    for i in 0..NF {
        let f = f_lo * (f_hi / f_lo).powf(i as f32 / (NF - 1) as f32);
        fs[i] = f;
        let l = (iso9613_alpha_db_per_m(f as f64, temp_c as f64, rh_pct as f64, 1013.25) * distance_m as f64) as f32;
        target[i] = l.min(60.0);
        tmax = tmax.max(target[i]);
    }
    if tmax < 0.05 {
        return (out, 0);
    }
    // shelf corners: octaves from 500 Hz below 0.45 sr
    let mut fc = [0.0f32; MAX_SHELVES];
    let mut ns = 0;
    let mut c = 500.0f32;
    while ns < MAX_SHELVES && c < 0.45 * sr {
        fc[ns] = c;
        ns += 1;
        c *= 2.0;
    }
    // unit basis (dB loss per dB of shelf gain, linearised at 6 dB)
    let mut basis = [[0.0f32; NF]; MAX_SHELVES];
    for j in 0..ns {
        for i in 0..NF {
            basis[j][i] = shelf_loss_db(fc[j], 6.0, fs[i], sr) / 6.0;
        }
    }
    let nnls = |resid: &[f32; NF], g: &mut [f32; MAX_SHELVES], lo_bound: &[f32; MAX_SHELVES]| {
        // cyclic coordinate descent on 0.5 ||B dg - r||^2 with g + dg >= 0
        let mut dg = [0.0f32; MAX_SHELVES];
        let mut bb = [0.0f32; MAX_SHELVES];
        for j in 0..ns {
            bb[j] = basis[j].iter().map(|v| v * v).sum::<f32>().max(1e-9);
        }
        let mut r = *resid;
        for _ in 0..200 {
            for j in 0..ns {
                let num: f32 = (0..NF).map(|i| basis[j][i] * r[i]).sum();
                let mut step = num / bb[j];
                let newv = (dg[j] + step).max(lo_bound[j]);
                step = newv - dg[j];
                if step != 0.0 {
                    for i in 0..NF {
                        r[i] -= basis[j][i] * step;
                    }
                    dg[j] = newv;
                }
            }
        }
        for j in 0..ns {
            g[j] += dg[j];
        }
    };
    let mut g = [0.0f32; MAX_SHELVES];
    let lb0 = [0.0f32; MAX_SHELVES];
    nnls(&target, &mut g, &lb0);
    for _ in 0..3 {
        let mut resid = [0.0f32; NF];
        for i in 0..NF {
            let a: f32 = (0..ns).map(|j| shelf_loss_db(fc[j], g[j], fs[i], sr)).sum();
            resid[i] = target[i] - a;
        }
        let mut lb = [0.0f32; MAX_SHELVES];
        for j in 0..ns {
            lb[j] = -g[j];
        }
        nnls(&resid, &mut g, &lb);
        for gv in g.iter_mut() {
            *gv = gv.clamp(0.0, 48.0);
        }
    }
    let mut k = 0;
    for j in 0..ns {
        if g[j] > 0.02 && k < MAX_STAGES {
            out[k] = BiquadCoefs::high_shelf(sr, fc[j], Q_S1, -g[j]);
            k += 1;
        }
    }
    (out, k)
}

#[derive(Clone, Copy, Debug, Default)]
struct Stages {
    c: [BiquadCoefs; MAX_STAGES],
    z: [[f64; 2]; MAX_STAGES],
    n: usize,
}

impl Stages {
    fn run(&mut self, buf: &mut [f32]) {
        for s in 0..self.n {
            let c = self.c[s];
            let (mut z1, mut z2) = (self.z[s][0], self.z[s][1]);
            for v in buf.iter_mut() {
                let x = *v as f64;
                let y = c.b0 * x + z1;
                z1 = c.b1 * x - c.a1 * y + z2;
                z2 = c.b2 * x - c.a2 * y;
                *v = y as f32;
            }
            self.z[s] = [if z1.abs() < 1e-30 { 0.0 } else { z1 }, if z2.abs() < 1e-30 { 0.0 } else { z2 }];
        }
    }
    fn quiet(&self) -> bool {
        self.z[..self.n].iter().all(|z| z[0].abs() < 1e-25 && z[1].abs() < 1e-25)
    }
}

/// Stateful ISO 9613-1 absorption at a fixed distance, redesigned (with a warmed, crossfaded
/// filter) when temperature or humidity move.
#[derive(Clone, Debug)]
pub struct AirFilter {
    sr: f32,
    pub distance: f32,
    cur: Stages,
    params: Option<(f32, f32)>,
    last_in: Vec<f32>,
    tmp: Vec<f32>,
}

impl AirFilter {
    pub fn new(distance: f32, sr: f32, n_max: usize) -> AirFilter {
        AirFilter {
            sr,
            distance,
            cur: Stages::default(),
            params: None,
            last_in: vec![0.0; n_max],
            tmp: vec![0.0; n_max],
        }
    }

    pub fn reset(&mut self) {
        self.cur = Stages::default();
        self.params = None;
        self.last_in.fill(0.0);
    }

    pub fn quiet(&self) -> bool {
        self.cur.quiet()
    }

    fn changed(&self, t: f32, h: f32) -> bool {
        match self.params {
            None => true,
            Some((t0, h0)) => (t - t0).abs() > 1.0 || (h - h0).abs() > 0.05 * h0.max(1.0),
        }
    }

    /// Filter `buf` in place (one block).
    pub fn process(&mut self, buf: &mut [f32], temp_c: f32, rh_pct: f32) {
        let n = buf.len().min(self.tmp.len());
        let buf = &mut buf[..n];
        if self.params.is_some() && self.changed(temp_c, rh_pct) {
            let (c, k) = design(self.distance, temp_c, rh_pct, self.sr);
            let mut new = Stages { c, z: [[0.0; 2]; MAX_STAGES], n: k };
            // warm the new filter on the previous block's input
            let tmp = &mut self.tmp[..n];
            tmp.copy_from_slice(&self.last_in[..n]);
            new.run(tmp);
            tmp.copy_from_slice(buf);
            self.last_in[..n].copy_from_slice(buf);
            self.cur.run(buf);
            new.run(tmp);
            let den = (n.max(2) - 1) as f32;
            for (i, (o, nv)) in buf.iter_mut().zip(tmp.iter()).enumerate() {
                let f = i as f32 / den;
                *o = *o * (1.0 - f) + *nv * f;
            }
            self.cur = new;
            self.params = Some((temp_c, rh_pct));
            return;
        }
        if self.params.is_none() {
            let (c, k) = design(self.distance, temp_c, rh_pct, self.sr);
            self.cur = Stages { c, z: [[0.0; 2]; MAX_STAGES], n: k };
            self.params = Some((temp_c, rh_pct));
        }
        self.last_in[..n].copy_from_slice(buf);
        self.cur.run(buf);
    }
}
