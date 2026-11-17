//! Gutters and downspouts (GutterGenerator of raingen/synth/rain.py; hypothesis model,
//! MASTER_PLAN 3.9).
//!
//! Driven by WeatherState.roof_flow_l_s (roof -> gutter routing of raingen.physics.wetness):
//! - trickle (q < ~5 ml/s): separate ~4 mm drops falling down the pipe / off the channel lip onto
//!   the elbow or the shell, each a force pulse through the metal's modes plus a water impact;
//! - bubbly flow: a cloud of entrained bubbles, radii ~ r^-2.6 (larger bubbles as the flow grows);
//! - slug flow (downspouts, q > ~0.15 l/s): big glugging bubbles;
//! - turbulent rush noise. Everything excites the air column: an open pipe with end corrections,
//!   f_k = k c / 2 (L + 0.6 d); an open half-round channel couples only weakly to its length modes.
//! - an open gutter overflows when the flow exceeds its design capacity (`overflow_rate_mm_h` over
//!   the catchment): a falling sheet that splashes on the ground below.

use super::air::AirFilter;
use super::dsd::fall_speed_from_height;
use super::events::{cox_offsets, thinned_offsets, RateModulator};
use super::kernels::{
    butter2_bandpass, db_to_pa, force_peak, loguniform, power_law, smoothstep, Cascade, KernelBank, ModalLanes, SolidVoice, NOISE_SIZE,
};
use super::rain::{bubble_pa_per_m, water_event, DirectBus, WaterParams, AIR_MIN_DIST_M};
use super::BuildCtx;
use crate::dsp::BiquadCoefs;
use crate::engine::{BlockCtx, Voice};
use crate::rng::Rng;
use crate::types::{speed_of_sound, Emission, MaterialKind, Patch, Stem};
use std::sync::Arc;

/// Bubbles inside a channel / pipe radiate less directly.
const BUBBLE_GAIN: f32 = 0.5;
const MAX_BUBBLES_PER_BLOCK: f64 = 500.0;

/// Unit-energy normalisation of a filter: 1 / sqrt(sum h^2) over 8192 samples.
pub fn filter_norm(c: &[BiquadCoefs]) -> f32 {
    let mut x = vec![0.0f64; 8192];
    x[0] = 1.0;
    Cascade::filter_f64(c, &mut x);
    let e: f64 = x.iter().map(|v| v * v).sum();
    (1.0 / e.max(1e-12).sqrt()) as f32
}

/// out[i] = Gaussian noise from the shared table starting at `pos` (wraps around).
pub fn noise_block(kb: &KernelBank, pos: u64, out: &mut [f32]) {
    let tab = &kb.noise;
    let mut p = (pos as usize) & (NOISE_SIZE - 1);
    let mut i = 0;
    while i < out.len() {
        let m = (out.len() - i).min(NOISE_SIZE - p);
        out[i..i + m].copy_from_slice(&tab[p..p + m]);
        i += m;
        p = 0;
    }
}

pub struct GutterVoice {
    name: String,
    kb: Arc<KernelBank>,
    n: usize,
    sr: f32,
    em: Emission,
    is_pipe: bool,
    share: f32,
    water_quality: f32,
    q_cap: f32,
    drop_fall_m: f32,
    pipe_mix: f32,
    distance: f32,
    pipe: ModalLanes,
    pipe_norm: f32,
    shell: SolidVoice,
    rush: Cascade,
    rush_norm: f32,
    sheet: Cascade,
    sheet_norm: f32,
    direct: DirectBus,
    air: Option<AirFilter>,
    modu: RateModulator,
    click_pa: f32,
    prev_rush: f32,
    prev_sheet: f32,
    overflowing: bool,
    key: u64,
    key_rush: u64,
    key_sheet: u64,
    offs: Vec<u32>,
    exc: Vec<f32>,
    tmp: Vec<f32>,
}

impl GutterVoice {
    pub fn new(bc: &mut BuildCtx, patch: &Patch, n_gutters: usize) -> GutterVoice {
        let sr = bc.sr as f32;
        let n = bc.block;
        let kb = bc.kb.clone();
        let is_pipe = super::is_downspout(patch);
        let share = patch.param_f32("share", 1.0 / n_gutters.max(1) as f32);
        let length = if is_pipe {
            patch.param_f32("pipe_length_m", patch.param_f32("length_m", 2.5)).max(0.3)
        } else {
            patch.param_f32("length_m", 6.0).max(0.3)
        };
        let diam = patch.param_f32("diameter_m", patch.param_f32("width_m", 0.08)).max(0.02);
        let mat_name = patch
            .param_str("material")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| if patch.material.is_empty() { "metal_gutter".into() } else { patch.material.clone() });
        let mat = bc.mats.get(&mat_name);
        let pipe_mat = if mat.kind == MaterialKind::Solid { mat } else { bc.mats.get("metal_gutter") };
        let temp = bc.scene.weather.temperature_c;
        let catch = patch.param_f32("catchment_m2", 25.0);
        let q_cap = match patch.params.get("overflow_rate_mm_h").and_then(|v| v.as_f64()) {
            Some(over) if !is_pipe => over as f32 * catch / 3600.0,
            _ => f32::INFINITY,
        };
        let drop_fall_m = if is_pipe { 0.8 * length } else { patch.param_f32("height_m", 2.5).max(0.3) };
        let distance = patch.distance().max(0.3);
        let spread = (2.0 * (0.25 * length / distance).atan() / std::f32::consts::PI).clamp(0.0, 1.0);
        // air column: open-open pipe with end corrections
        let c = speed_of_sound(temp);
        let f1 = c / (2.0 * (length + 0.6 * diam));
        let fmax = 5000.0f32.min(0.4 * sr);
        let modes: Vec<(f32, f32, f32)> = (1..64)
            .filter(|k| *k as f32 * f1 < fmax)
            .map(|k| {
                let kf = k as f32;
                (kf * f1, 0.35 * kf.powf(-0.4), kf.powf(-0.5))
            })
            .collect();
        let pipe = ModalLanes::new(&modes, sr, n);
        let pipe_norm = if pipe.n_modes() > 0 {
            let ir = ModalLanes::impulse_response(&modes, sr, (1.5 * sr) as usize);
            let e: f64 = ir.iter().map(|v| v * v).sum();
            (1.0 / e.max(1e-12).sqrt()) as f32
        } else {
            1.0
        };
        let ola_cap = n + (0.08 * sr) as usize + kb.max_kernel_len() + 64;
        let rush_c = butter2_bandpass(250.0, 5000.0f32.min(0.4 * sr), sr);
        let sheet_c = butter2_bandpass(200.0, 9000.0f32.min(0.4 * sr), sr);
        let name = format!("gutter/{}", patch.id);
        GutterVoice {
            em: Emission::point(patch.azimuth(), patch.elevation(), patch.distance(), spread, Stem::Aftermath),
            is_pipe,
            share,
            water_quality: bc.water_quality,
            q_cap,
            drop_fall_m,
            pipe_mix: if is_pipe { 0.8 } else { 0.3 },
            distance,
            pipe,
            pipe_norm,
            shell: SolidVoice::new(&pipe_mat.modes, &pipe_mat.rad, pipe_mat.cal, sr, n, ola_cap),
            rush: Cascade::new(&rush_c),
            rush_norm: filter_norm(&rush_c),
            sheet: Cascade::new(&sheet_c),
            sheet_norm: filter_norm(&sheet_c),
            direct: DirectBus::new(ola_cap, 4096, sr as f64),
            air: if distance >= AIR_MIN_DIST_M { Some(AirFilter::new(distance, sr, n)) } else { None },
            modu: RateModulator::new(&bc.rngf, &format!("gutter/{}/gurgle", patch.id), 0.45, 0.05, 4.0, sr as f64),
            click_pa: db_to_pa(bc.mats.level_of("puddle")),
            prev_rush: 0.0,
            prev_sheet: 0.0,
            overflowing: false,
            key: bc.rngf.key(&format!("gutter/{}", patch.id)),
            key_rush: bc.rngf.key(&format!("gutter/{}/rush", patch.id)),
            key_sheet: bc.rngf.key(&format!("gutter/{}/sheet", patch.id)),
            offs: Vec::with_capacity(1024),
            exc: vec![0.0; n],
            tmp: vec![0.0; n],
            name,
            kb,
            n,
            sr,
        }
    }
}

impl Voice for GutterVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n.min(self.n).min(out.len());
        if n == 0 {
            return Emission::silent(Stem::Aftermath);
        }
        let w = ctx.weather;
        let sr = self.sr as f64;
        let kb = self.kb.clone();
        let q = w.roof_flow_l_s.max(0.0) * self.share; // l/s through this channel / pipe
        let tails = self.direct.active() || self.shell.active() || !self.pipe.silent(1e-12);
        if q < 1e-5 && !tails && self.prev_rush == 0.0 && self.prev_sheet == 0.0 {
            return Emission::silent(Stem::Aftermath);
        }
        let m1 = self.modu.step(ctx.index, n);
        let m0 = self.modu.prev;
        let mean = (0.5 * (m0 + m1)) as f32;
        let mut rush = 0.0f32;
        let mut sheet = 0.0f32;
        if q > 0.0 {
            let mut rng = Rng::from_key(self.key, ctx.index);
            // trickle: separate ~4 mm drops (3.4e-5 l each) onto the elbow / channel bottom
            let drip_w = 1.0 - smoothstep(q, 0.002, 0.006);
            let mut offs = std::mem::take(&mut self.offs);
            cox_offsets(&mut rng, (drip_w * q / 3.4e-5) as f64, n, sr, m0, m1, &mut offs);
            if !offs.is_empty() {
                let wp = WaterParams { click_pa: self.click_pa, depth_mm: 10.0, wind_mps: 0.0, water_quality: self.water_quality, p_scale: 1.0 };
                let h = if self.is_pipe { self.drop_fall_m } else { 0.05 };
                for &off in offs.iter() {
                    let d = rng.range(3.5, 4.5) as f32;
                    let u = fall_speed_from_height(d, h).max(0.3);
                    self.shell.add_force(&kb, off as usize, 0.5 * force_peak(d, u), d as f64 * 1e-3 / u as f64, false);
                    water_event(&kb, &mut rng, &mut self.direct, off as usize, d, u, 4, 0.5, &wp);
                }
            }
            // bubbly flow
            let flow_w = smoothstep(q, 0.001, 0.01);
            let lam = flow_w * (900.0 * (q / 0.05).powf(0.8)).min(8000.0);
            let scb = thinned_offsets(&mut rng, lam as f64, n, sr, MAX_BUBBLES_PER_BLOCK, m0, m1, &mut offs);
            if !offs.is_empty() {
                let r_max = (1.5 + 6.0 * smoothstep(q, 0.02, 0.5)) as f64 * 1e-3;
                let base = bubble_pa_per_m() as f64 * scb * BUBBLE_GAIN as f64 * self.water_quality as f64;
                for &off in offs.iter() {
                    let radius = power_law(&mut rng, 0.4e-3, r_max, 2.6);
                    let eps = loguniform(&mut rng, 0.01, 0.06);
                    let xi = rng.range(0.0, 0.15);
                    self.direct.bubbles.spawn(off as usize, base * eps, radius, xi, 0.0);
                }
            }
            // slug-flow glugs (pipes only)
            if self.is_pipe {
                let glug_rate = smoothstep(q, 0.15, 0.6) * (0.7 + 2.5 * smoothstep(q, 0.3, 1.2));
                cox_offsets(&mut rng, glug_rate as f64, n, sr, 1.0, 1.0, &mut offs);
                for &off in offs.iter() {
                    let radius = loguniform(&mut rng, 4e-3, 10e-3);
                    let eps = rng.range(0.08, 0.18);
                    let xi = rng.range(0.0, 0.05);
                    self.direct.bubbles.spawn(off as usize, bubble_pa_per_m() as f64 * eps * BUBBLE_GAIN as f64, radius, xi, 0.0);
                }
            }
            self.offs = offs;
            rush = flow_w * db_to_pa(50.0 + 10.0 * (q.max(1e-6) / 0.1).log10()) * mean;
            if !self.is_pipe {
                rush *= 0.6;
            }
            // overflow sheet splashing on the ground below the gutter
            let over = q - self.q_cap;
            if over > 0.0 {
                sheet = db_to_pa(54.0 + 10.0 * (over.max(1e-6) / 0.05).log10()) * mean;
                if !self.overflowing {
                    ctx.marker(ctx.t0, "gutter_overflow", format!("{{\"source\": \"{}\", \"flow_l_s\": {}}}", self.name, q));
                }
                self.overflowing = true;
            } else {
                self.overflowing = false;
            }
        }

        // excitation: direct (drops, bubbles) + turbulent rush
        let exc = &mut self.exc[..n];
        exc.fill(0.0);
        self.direct.render(exc);
        if rush > 0.0 || self.prev_rush > 0.0 {
            let tmp = &mut self.tmp[..n];
            let pos = Rng::from_key(self.key_rush, ctx.index).below(1 << 30);
            noise_block(&kb, pos, tmp);
            self.rush.process(tmp);
            let (g0, g1) = (self.prev_rush, rush);
            let k = self.rush_norm;
            for (i, (e, t)) in exc.iter_mut().zip(tmp.iter()).enumerate() {
                let g = g0 + (g1 - g0) * i as f32 / n as f32;
                *e += *t * k * g;
            }
        }
        self.prev_rush = rush;
        let y = &mut out[..n];
        for (o, e) in y.iter_mut().zip(exc.iter()) {
            *o += 0.6 * *e;
        }
        if self.pipe.n_modes() > 0 {
            let tmp = &mut self.tmp[..n];
            tmp.fill(0.0);
            let any_exc = exc.iter().any(|v| *v != 0.0);
            if any_exc {
                self.pipe.process(Some(exc), tmp);
            } else if !self.pipe.silent(1e-12) {
                self.pipe.process(None, tmp);
            }
            let g = self.pipe_mix * self.pipe_norm;
            for (o, t) in y.iter_mut().zip(tmp.iter()) {
                *o += g * *t;
            }
        }
        if self.shell.active() {
            self.shell.render(y);
        }
        if sheet > 0.0 || self.prev_sheet > 0.0 {
            let tmp = &mut self.tmp[..n];
            let pos = Rng::from_key(self.key_sheet, ctx.index).below(1 << 30);
            noise_block(&kb, pos, tmp);
            self.sheet.process(tmp);
            let (g0, g1) = (self.prev_sheet, sheet);
            let k = self.sheet_norm;
            for (i, (o, t)) in y.iter_mut().zip(tmp.iter()).enumerate() {
                let g = g0 + (g1 - g0) * i as f32 / n as f32;
                *o += *t * k * g;
            }
        }
        self.prev_sheet = sheet;
        let live = y.iter().any(|v| *v != 0.0);
        if !live && !self.direct.active() {
            if let Some(a) = self.air.as_mut() {
                if !a.quiet() {
                    a.process(y, w.temperature_c, w.humidity_pct);
                    return self.em;
                }
            }
            return Emission::silent(Stem::Aftermath);
        }
        let g = 1.0 / self.distance.max(0.05);
        for v in y.iter_mut() {
            *v *= g;
        }
        if let Some(a) = self.air.as_mut() {
            a.process(y, w.temperature_c, w.humidity_pct);
        }
        self.em
    }

    fn reset(&mut self) {
        self.pipe.reset();
        self.shell.reset();
        self.rush.reset();
        self.sheet.reset();
        self.direct.clear();
        if let Some(a) = self.air.as_mut() {
            a.reset();
        }
        self.modu.reset();
        self.prev_rush = 0.0;
        self.prev_sheet = 0.0;
        self.overflowing = false;
    }
}
