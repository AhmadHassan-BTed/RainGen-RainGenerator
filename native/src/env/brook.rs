//! A brook near the listener (place.biophony.stream, 0..1, live).
//!
//! * Turbulent flow noise: a low 'body' band (shallow riffles, ~150-800 Hz) and a bright hiss
//!   whose low-pass corner rises with discharge, both slowly modulated (eddies; OU processes).
//! * Bubble micro-plinks: entrained bubbles ring at the Minnaert frequency f0 = 3.26 / a (a radius,
//!   m), damped by d = 0.13 f0 + 0.0072 f0^1.5 1/s, with the rising chirp f(t) = f0 (1 + xi d t)
//!   of a bubble nearing the surface (Minnaert 1933; van den Doel 2005, ACM TAP 2). Radii follow a
//!   power law (more small bubbles), 0.6-6 mm, i.e. ~0.5-5.4 kHz.
//! * Rain swells the stream: the catchment's response is an exponential average of the rain rate
//!   (rising over ~2 min, falling over ~10 min); discharge raises level, bubble rate and brightness.

use super::common::*;
use crate::dsp::{BiquadCoefs, Sos};
use crate::engine::{BlockCtx, BuildCfg, Voice};
use crate::rng::Rng;
use crate::types::{amb, precip, Emission, Scene, Stem};
use std::f64::consts::PI;

const MAX_BUBBLES: usize = 48;

#[derive(Clone, Copy, Default)]
struct Bubble {
    ph: f64,
    w: f64,
    dw: f64,
    a: f32,
    decay: f32,
    /// sample offset in the current block where it starts
    at: usize,
}

pub struct BrookVoice {
    lvl: AmbLevel,
    sr: f32,
    key: u64,
    az: f32,
    el: f32,
    dist: f32,
    level: f32,
    rain_ema: f64,
    body: Sos,
    body_g: f32,
    hiss: Sos,
    hiss_g: f32,
    lp_z: f32,
    ou1: f64,
    ou2: f64,
    amp: Ramp,
    bright: Ramp,
    bubbles: Vec<Bubble>,
    air: AirPath,
    tmp: Vec<f32>,
}

impl BrookVoice {
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> BrookVoice {
        let sr = cfg.sr as f32;
        let mut c: Rng = cfg.rng.get("brook.cfg", 0);
        let h = scene.listener.height_m;
        let az = c.range(-PI, PI);
        let d = c.range(9.0, 16.0);
        let pos = [(d * az.cos()) as f32, (d * az.sin()) as f32, 0.2 - h];
        let dist = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let body: Vec<BiquadCoefs> = vec![bp(380.0, 0.7, sr), hp2(110.0, sr)];
        let hiss: Vec<BiquadCoefs> = vec![hp2(900.0, sr)];
        BrookVoice {
            lvl: AmbLevel::new(amb::STREAM),
            sr,
            key: cfg.rng.key("brook"),
            az: pos[1].atan2(pos[0]),
            el: pos[2].atan2(pos[0].hypot(pos[1])),
            dist,
            // a babbling brook ~ 70 dB SPL at 1 m (50-55 dB at 10-15 m)
            level: db_spl(70.0),
            rain_ema: -1.0,
            body_g: noise_rms_gain(&body),
            body: Sos::new(&body),
            hiss_g: noise_rms_gain(&hiss),
            hiss: Sos::new(&hiss),
            lp_z: 0.0,
            ou1: 0.0,
            ou2: 0.0,
            amp: Ramp::default(),
            bright: Ramp::default(),
            bubbles: Vec::with_capacity(MAX_BUBBLES),
            air: AirPath::new(dist, sr, 8.0),
            tmp: vec![0.0; cfg.block],
        }
    }

    fn spawn(&mut self, rng: &mut Rng, at: usize, flow: f64) {
        if self.bubbles.len() >= MAX_BUBBLES {
            return;
        }
        let sr = self.sr as f64;
        // radius power law p(a) ~ a^-2.5 on [0.6, 6] mm (inverse CDF)
        let (lo, hi, k) = (0.6f64, 6.0f64, -1.5f64);
        let u = rng.uniform();
        let a_mm = (lo.powf(k) + u * (hi.powf(k) - lo.powf(k))).powf(1.0 / k);
        let f0 = 3.26 / (a_mm * 1e-3);
        let d = 0.13 * f0 + 0.0072 * f0.powf(1.5);
        let xi = rng.range(0.1, 0.35);
        let w = 2.0 * PI * f0 / sr;
        let dw = w * xi * d / sr;
        let amp = (rng.lognormal(0.0, 0.5) * (a_mm / 1.5).powf(0.7) * (0.7 + 0.3 * flow)) as f32;
        self.bubbles.push(Bubble { ph: rng.range(0.0, 0.3), w, dw, a: amp, decay: (-d / sr).exp() as f32, at });
    }
}

impl Voice for BrookVoice {
    fn name(&self) -> &str {
        "place.brook"
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.sr as f64;
        let dt = n as f64 / sr;
        let w = ctx.weather;
        let liquid = if w.precip_type == precip::NONE || w.precip_type == precip::SNOW { 0.0 } else { w.rain_rate_mm_h as f64 };
        if self.rain_ema < 0.0 {
            self.rain_ema = liquid;
        }
        let tau = if liquid > self.rain_ema { 120.0 } else { 600.0 };
        self.rain_ema += (liquid - self.rain_ema) * (1.0 - (-dt / tau).exp());
        let l = self.lvl.step(ctx, 0.0).clamp(0.0, 1.0);
        if (l <= 0.0 && self.amp.idle() && self.bubbles.is_empty()) || self.tmp.len() < n {
            return Emission::silent(Stem::Ambience);
        }
        // discharge factor: 1 at base flow, ~2.5 in a downpour
        let flow = 1.0 + 1.5 * (1.0 - (-self.rain_ema / 8.0).exp());
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let a1 = (-dt / 0.7).exp();
        let a2 = (-dt / 4.0).exp();
        self.ou1 = a1 * self.ou1 + (1.0 - a1 * a1).sqrt() * rng.normal();
        self.ou2 = a2 * self.ou2 + (1.0 - a2 * a2).sqrt() * rng.normal();
        let wobble = (0.22 * self.ou1 + 0.18 * self.ou2).exp();
        let g = self.level * (l as f32).sqrt() * (flow.powf(0.8) * wobble) as f32 / self.dist.max(1.0);
        let (g0, g1) = self.amp.to(g);
        let br = (flow - 1.0) / 1.5;
        let (b0, b1) = self.bright.to((0.35 + 0.5 * br) as f32);
        // hiss corner: ~2.5 kHz at base flow up to ~8 kHz in spate
        let fc = (2500.0 * flow.powf(1.25)).min(0.4 * sr);
        let lp_a = (-2.0 * PI * fc / sr).exp() as f32;

        let o = &mut out[..n];
        fill_gauss(&mut rng, o);
        self.body.process_block(o);
        let ib = 0.8 / self.body_g;
        let tmp = &mut self.tmp[..n];
        fill_gauss(&mut rng, tmp);
        self.hiss.process_block(tmp);
        let ih = 1.0 / self.hiss_g;
        let mut z = self.lp_z;
        for (i, (v, t)) in o.iter_mut().zip(tmp.iter()).enumerate() {
            z = lp_a * z + (1.0 - lp_a) * t * ih;
            *v = *v * ib + ramp_at(b0, b1, i, n) * z * 1.6;
        }
        self.lp_z = z;

        // bubbles: babble rate rises with discharge
        let rate = 140.0 * flow.powf(1.6) * (0.4 + 0.6 * l);
        let k = rng.poisson(rate * dt).min(64);
        for _ in 0..k {
            let at = rng.below(n as u64) as usize;
            self.spawn(&mut rng, at, flow);
        }
        let bub_g = 0.55f32;
        for b in self.bubbles.iter_mut() {
            let mut ph = b.ph;
            let mut wv = b.w;
            let mut a = b.a;
            for v in o.iter_mut().skip(b.at) {
                *v += bub_g * a * (ph as f32).sin();
                ph += wv;
                wv += b.dw;
                a *= b.decay;
            }
            b.ph = ph.rem_euclid(2.0 * PI);
            b.w = wv.min(PI * 0.9);
            b.a = a;
            b.at = 0;
        }
        self.bubbles.retain(|b| b.a > 1e-4);

        for (i, v) in o.iter_mut().enumerate() {
            *v *= ramp_at(g0, g1, i, n);
        }
        self.air.process(o, w.temperature_c, w.humidity_pct);
        Emission::point(self.az, self.el, self.dist, 0.5, Stem::Ambience)
    }

    fn reset(&mut self) {
        self.lvl.reset();
        self.rain_ema = -1.0;
        self.body.reset();
        self.hiss.reset();
        self.lp_z = 0.0;
        self.ou1 = 0.0;
        self.ou2 = 0.0;
        self.amp = Ramp::default();
        self.bright = Ramp::default();
        self.bubbles.clear();
        self.air.reset();
    }
}
