//! The advected Kaimal gust field shared by every wind emitter (raingen.physics.wind.GustProcess
//! and raingen.synth.wind.GustField).
//!
//! Gust contract (raingen/physics/world.py): the World's listener gust is
//! `GustProcess(rngf, "wind.gust", integral_time_s=T0, ti=ti0, control_rate=sr/block)` stepped once
//! per block after `set_params(T_b, ti_b)`. This module re-runs that process on a NumPy-compatible
//! Philox stream (env::npy) with the same layer key, constructor arguments and per-block
//! parameters, so its listener value equals WeatherState.gust (to ~1e-11) and the gust heard in
//! upwind trees is the one that reaches the listener moments later (Taylor's hypothesis). If the
//! value ever disagrees with WeatherState.gust (a different seed, a hand-made weather stream), the
//! field is anchored to the delivered gust by a per-block ratio, so the listener's gust is always
//! WeatherState.gust.

use super::common::{kaimal_integral_time_s, scene_z0, stability_class, is_daytime, turbulence_intensity_for, wind_at_height};
use super::npy::{self, NpRng};
use crate::types::{precip, Scene, SceneWind, Weather};

const KAIMAL_SLOPE: f64 = 5.0 / 3.0;
const MAX_CORNERS: usize = 12;

/// Deterministic, advected, Kaimal-shaped gust multiplier at the control rate.
#[derive(Clone)]
pub struct GustProcess {
    key: [u64; 2],
    pub control_rate: f64,
    dt: f64,
    pub history_s: f64,
    pub lead_s: f64,
    lead: i64,
    n_hist: usize,
    buf: Vec<f64>,
    max_replay: i64,
    z: [f64; MAX_CORNERS],
    z_init: bool,
    nc: usize,
    next_k: i64,
    first_k: i64,
    block: i64,
    pub value: f64,
    pub integral_time_s: f64,
    pub ti: f64,
    corners: [f64; MAX_CORNERS],
    w: [f64; MAX_CORNERS],
    rho: [f64; MAX_CORNERS],
    innov: [f64; MAX_CORNERS],
    s: f64,
}

impl GustProcess {
    pub fn new(key: [u64; 2], integral_time_s: f64, ti: f64, control_rate: f64) -> GustProcess {
        npy::warm_up();
        let history_s = 30.0;
        let lead_s = (0.4f64 * history_s).min(12.0);
        let lead = (lead_s * control_rate).ceil() as i64;
        let n_hist = ((history_s + lead_s) * control_rate).ceil() as usize + 4;
        let mut g = GustProcess {
            key,
            control_rate,
            dt: 1.0 / control_rate,
            history_s,
            lead_s,
            lead,
            n_hist,
            buf: vec![1.0; n_hist],
            max_replay: (120.0 * control_rate) as i64,
            z: [0.0; MAX_CORNERS],
            z_init: false,
            nc: 0,
            next_k: 0,
            first_k: 0,
            block: -1,
            value: 1.0,
            integral_time_s: 0.0,
            ti: 0.0,
            corners: [0.0; MAX_CORNERS],
            w: [0.0; MAX_CORNERS],
            rho: [0.0; MAX_CORNERS],
            innov: [0.0; MAX_CORNERS],
            s: 0.0,
        };
        g.set_params(Some(integral_time_s), Some(ti));
        g
    }

    pub fn set_params(&mut self, integral_time_s: Option<f64>, ti: Option<f64>) {
        if let Some(t) = integral_time_s {
            let t = t.clamp(0.3, 120.0);
            if (t - self.integral_time_s).abs() > 1e-9 {
                self.integral_time_s = t;
                let f_c = 1.0 / (2.0 * std::f64::consts::PI * t);
                let f_max = (self.control_rate / 4.0).min(4.0);
                let mut c = [0f64; MAX_CORNERS];
                let mut nc = 1;
                c[0] = f_c;
                while c[nc - 1] * 2.0 <= f_max && nc < MAX_CORNERS {
                    c[nc] = c[nc - 1] * 2.0;
                    nc += 1;
                }
                while nc < 3 {
                    c[nc] = c[nc - 1] * 2.0;
                    nc += 1;
                }
                let mut var = [0f64; MAX_CORNERS];
                let mut tot = 0.0;
                for i in 0..nc {
                    var[i] = c[i].powf(1.0 - KAIMAL_SLOPE);
                    tot += var[i];
                }
                for i in 0..nc {
                    self.w[i] = (var[i] / tot).sqrt();
                    self.rho[i] = (-2.0 * std::f64::consts::PI * c[i] * self.dt).exp();
                    self.innov[i] = (1.0 - self.rho[i] * self.rho[i]).sqrt();
                }
                if self.z_init && nc != self.nc {
                    for i in self.nc.min(nc)..MAX_CORNERS {
                        self.z[i] = 0.0;
                    }
                }
                self.corners = c;
                self.nc = nc;
            }
        }
        if let Some(ti) = ti {
            self.ti = ti.clamp(0.0, 1.0);
        }
        self.s = (1.0 + self.ti * self.ti).ln().sqrt();
    }

    fn synth(&mut self, k: i64) {
        let mut rng = NpRng::for_block(self.key, k as u64);
        let mut eps = [0f64; MAX_CORNERS];
        for e in eps.iter_mut().take(self.nc) {
            *e = rng.standard_normal();
        }
        if !self.z_init {
            for i in 0..self.nc {
                self.z[i] = rng.standard_normal();
            }
            self.z_init = true;
        } else {
            for i in 0..self.nc {
                self.z[i] = self.rho[i] * self.z[i] + self.innov[i] * eps[i];
            }
        }
        let mut g = 0.0;
        for i in 0..self.nc {
            g += self.w[i] * self.z[i];
        }
        let s = self.s;
        let m = (s * g - 0.5 * s * s).exp();
        let idx = (k.rem_euclid(self.n_hist as i64)) as usize;
        self.buf[idx] = m.max(0.05);
    }

    fn generate_until(&mut self, k_last: i64) {
        while self.next_k <= k_last {
            let k = self.next_k;
            self.synth(k);
            self.next_k += 1;
        }
    }

    fn reset_at(&mut self, k_start: i64) {
        self.z_init = false;
        self.next_k = k_start.max(0);
        self.first_k = self.next_k;
    }

    /// Advance to `block_index` and return the gust multiplier at the listener.
    pub fn step(&mut self, b: i64) -> f64 {
        let back = self.max_replay.min((10.0 * self.integral_time_s * self.control_rate) as i64);
        if self.block < 0 || b < self.block {
            let start = if b <= self.max_replay { 0 } else { b - back };
            self.reset_at(start);
        } else if b - self.block > self.max_replay {
            self.reset_at(b - back);
        }
        self.generate_until(b + self.lead);
        self.block = b;
        self.value = self.buf[(b.rem_euclid(self.n_hist as i64)) as usize];
        self.value
    }

    fn available(&self) -> (i64, i64) {
        let lo = self.first_k.max(self.next_k - self.n_hist as i64 + 1);
        (lo, self.next_k - 1)
    }

    /// Multiplier at the listener `delay_s` seconds ago (negative = ahead, up to lead_s).
    pub fn at_delay(&self, delay_s: f64) -> f64 {
        if self.block < 0 {
            return 1.0;
        }
        let (lo, hi) = self.available();
        let pos = (self.block as f64 - delay_s * self.control_rate).clamp(lo as f64, hi as f64);
        let i0 = pos.floor() as i64;
        let frac = pos - i0 as f64;
        let i1 = (i0 + 1).min(hi);
        let n = self.n_hist as i64;
        let v0 = self.buf[i0.rem_euclid(n) as usize];
        let v1 = self.buf[i1.rem_euclid(n) as usize];
        v0 + (v1 - v0) * frac
    }

    /// Gust at a point `upwind_m` metres upwind of the listener (Taylor advection).
    pub fn at_position(&self, upwind_m: f64, wind_mps: f64) -> f64 {
        let u = wind_mps.max(0.5);
        self.at_delay(-upwind_m / u)
    }

    pub fn reset(&mut self) {
        self.block = -1;
        self.z_init = false;
        self.next_k = 0;
        self.first_k = 0;
        self.value = 1.0;
        self.buf.iter_mut().for_each(|v| *v = 1.0);
    }
}

/// Gust correlation time at the ears: user override, gust rate, or Kaimal at ear height.
pub fn gust_integral_time_s(wind: &SceneWind, u10: f32, z0: f32, listener_h: f32) -> f64 {
    if let Some(t) = wind.integral_time_s {
        if t != 0.0 {
            return t as f64;
        }
    }
    if let Some(r) = wind.gust_rate_per_min {
        if r != 0.0 {
            return 60.0 / (2.0 * std::f64::consts::PI * (r as f64).max(0.05));
        }
    }
    let z = listener_h.max(0.3);
    kaimal_integral_time_s(wind_at_height(u10, z, z0) as f64, z.max(1.0) as f64)
}

/// Turbulence intensity at the ears (a user TI / gust factor refers to 10 m); with automatic
/// turbulence the World's stability factor scales it.
pub fn ear_turbulence_intensity(wind: &SceneWind, z0: f32, listener_h: f32, stability_factor: f64) -> f64 {
    let ti10_auto = turbulence_intensity_for(z0, 10.0) as f64;
    let mut auto = false;
    let ti10 = if let Some(t) = wind.turbulence_intensity {
        t as f64
    } else if let Some(g) = wind.gust_factor {
        ((g as f64) - 1.0).max(0.0) / 3.0
    } else {
        auto = true;
        ti10_auto
    };
    let mut ti = turbulence_intensity_for(z0, listener_h.max(0.3)) as f64 * ti10 / ti10_auto.max(1e-6);
    if auto {
        ti *= stability_factor;
    }
    ti.clamp(0.0, 0.8)
}

/// The World's starting gust-variance stability factor for `scene`.
pub fn initial_stability_factor(scene: &Scene) -> f64 {
    let w0 = &scene.weather;
    let rate0 = if w0.precipitation.precip_type == precip::NONE { 0.0 } else { w0.precipitation.rate_mm_h.max(0.0) };
    let day0 = is_daytime(w0.hour, w0.latitude_deg, w0.season);
    let stab = stability_class(day0, w0.cloud_okta, w0.wind.speed_mps.max(0.0), rate0);
    match stab {
        crate::types::stability::UNSTABLE => std::f64::consts::SQRT_2,
        crate::types::stability::STABLE => 0.5f64.sqrt(),
        _ => 1.0,
    }
}

/// Advected along-wind gust field (one copy per voice: the process is a pure function of
/// (seed, params, block sequence), so every copy returns the same values without sharing state).
#[derive(Clone)]
pub struct GustField {
    proc: GustProcess,
    key: [u64; 2],
    t0: f64,
    ti0: f64,
    wind: SceneWind,
    z0: f32,
    listener_h: f32,
    sr: f64,
    block: usize,
    last: i64,
    pub value: f32,
    pub ti: f32,
    corr: f64,
}

pub const GUST_LAYER: &str = "wind.gust";

impl GustField {
    /// `seed` is the Python scene seed (RngFactory root seed of the World).
    pub fn new(scene: &Scene, seed: u64, sr: u32, block: usize) -> GustField {
        let key = npy::layer_key(seed, GUST_LAYER);
        let w = scene.weather.wind.clone();
        let z0 = scene_z0(scene);
        let h = scene.listener.height_m;
        let stab0 = initial_stability_factor(scene);
        let t0 = gust_integral_time_s(&w, w.speed_mps.max(0.0), z0, h);
        let ti0 = ear_turbulence_intensity(&w, z0, h, stab0);
        let block = block.max(1);
        let proc = GustProcess::new(key, t0, ti0, sr as f64 / block as f64);
        GustField {
            proc,
            key,
            t0,
            ti0,
            wind: w,
            z0,
            listener_h: h,
            sr: sr as f64,
            block,
            last: -1,
            value: 1.0,
            ti: ti0 as f32,
            corr: 1.0,
        }
    }

    /// Step to `block_index` with this block's weather (once per block; repeated calls are no-ops).
    pub fn advance(&mut self, block_index: u64, w: &Weather, n: usize) {
        let b = block_index as i64;
        if n != self.block && n > 0 {
            self.block = n;
            self.proc = GustProcess::new(self.key, self.t0, self.ti0, self.sr / n as f64);
            self.last = -1;
        }
        if self.last == b {
            return;
        }
        let t = gust_integral_time_s(&self.wind, w.wind_u10_mps.max(0.0), self.z0, self.listener_h);
        self.ti = w.turbulence_intensity;
        self.proc.set_params(Some(t), Some(w.turbulence_intensity as f64));
        let own = self.proc.step(b);
        let delivered = w.gust as f64;
        self.corr = if delivered > 0.0 && own > 0.0 {
            let c = delivered / own;
            if (c - 1.0).abs() < 1e-4 {
                1.0
            } else {
                c.clamp(0.2, 5.0)
            }
        } else {
            1.0
        };
        self.value = (own * self.corr) as f32;
        self.last = b;
    }

    /// Gust multiplier at a point `upwind_m` metres upwind (negative = downwind) of the listener.
    #[inline]
    pub fn at(&self, upwind_m: f32, u_mps: f32) -> f32 {
        (self.proc.at_position(upwind_m as f64, u_mps.max(0.5) as f64) * self.corr).max(0.05) as f32
    }

    pub fn reset(&mut self) {
        self.proc.reset();
        self.last = -1;
        self.value = 1.0;
        self.corr = 1.0;
    }
}
