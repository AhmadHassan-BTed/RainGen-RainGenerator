//! Aeolian tones, edge/gap whistles, wind chimes and flapping fabric (port of
//! raingen/synth/aeolian.py). One mono voice per emitter.
//!
//! * Vortex shedding (wires, guy lines, power lines, railing bars): f = St U_n / d, St ~ 0.2,
//!   U_n the wind normal to the axis; shedding stops below Re ~ 40. Level from Curle / Phillips:
//!   p_rms^2 = l_c St^2 rho^2 U^6 C_L'^2 / (16 c^2) (2 / r) atan(L / 2r). Lock-in onto string modes
//!   f_n = n/(2L) sqrt(T/mu) (aeolian harp, Blevins), narrowed by turbulence, +6..10 dB.
//!   Wet high-voltage lines add corona crackle and hum at twice the mains frequency.
//! * Edge / gap whistles: jet v_j = U sqrt(Cp), edge tone ~0.3 v_j / h locking onto a resonance,
//!   plus broadband howl (Howe).
//! * Wind chimes: free-free tube modes 1 : 2.757 : 5.404 : 8.933 struck by a wind-driven clapper.
//! * Flags / fabric: flutter at St ~0.2 on the chord, trailing-edge snaps once saturated, halyard
//!   slaps on the pole.
//! * Car body buffeting: gust pressure thumps felt inside the cabin.

use super::common::*;
use super::gust::GustField;
use crate::dsp::{Biquad, Sos};
use crate::engine::{BlockCtx, BoxedVoice, BuildCfg, Voice};
use crate::rng::Rng;
use crate::types::{speed_of_sound, Emission, Patch, Scene, Shelter, Stem, RHO_AIR};

const NU_AIR: f32 = 1.5e-5;
const ST_CYL: f32 = 0.2;
const CL_RMS: f32 = 0.25;
const CHIME_RATIOS: [f64; 4] = [1.0, 2.757, 5.404, 8.933];
const WIRE_KINDS: [&str; 5] = ["wire", "power_line", "cable", "guy_line", "rope"];
const BAR_KINDS: [&str; 6] = ["railing", "fence", "bars", "grille", "balustrade", "rail"];
const GAP_KINDS: [&str; 6] = ["gap", "vent", "slot", "edge", "chimney", "corner"];
const WOOD: [&str; 3] = ["wood", "bamboo", "timber"];

fn scale_ratios(name: &str) -> [f64; 9] {
    match name.to_ascii_lowercase().as_str() {
        "major" => [1.0, 9.0 / 8.0, 5.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0, 5.0 / 3.0, 15.0 / 8.0, 2.0, 9.0 / 4.0],
        "minor" => [1.0, 9.0 / 8.0, 6.0 / 5.0, 4.0 / 3.0, 3.0 / 2.0, 8.0 / 5.0, 9.0 / 5.0, 2.0, 9.0 / 4.0],
        "minor_pentatonic" => [1.0, 6.0 / 5.0, 4.0 / 3.0, 3.0 / 2.0, 9.0 / 5.0, 2.0, 12.0 / 5.0, 8.0 / 3.0, 3.0],
        "whole_tone" => {
            let mut r = [0f64; 9];
            for (k, v) in r.iter_mut().enumerate() {
                *v = 2f64.powf(k as f64 / 6.0);
            }
            r
        }
        _ => [1.0, 9.0 / 8.0, 5.0 / 4.0, 3.0 / 2.0, 5.0 / 3.0, 2.0, 9.0 / 4.0, 5.0 / 2.0, 3.0],
    }
}

#[inline]
fn spreading(dist: f32, ref_m: f32) -> f32 {
    ref_m / dist.max(0.3)
}

fn string_f1(length_m: f32, tension_n: f32, diameter_m: f32, density: f32) -> f32 {
    let mu = density * std::f32::consts::PI * diameter_m * diameter_m / 4.0;
    (tension_n.max(1.0) / mu.max(1e-6)).sqrt() / (2.0 * length_m.max(0.1))
}

/// Common geometry, local wind and propagation of one aeolian emitter.
struct Base {
    name: String,
    pos: [f32; 3],
    height: f32,
    stem: Stem,
    sr: f32,
    z0: f32,
    expo: f32,
    spread: f32,
    dist: f32,
    az: f32,
    el: f32,
    air: AirPath,
    field: GustField,
    key: u64,
    n: usize,
}

impl Base {
    #[allow(clippy::too_many_arguments)]
    fn new(id: &str, pos: [f32; 3], height: f32, stem: Stem, scene: &Scene, cfg: &BuildCfg, expo: f32, spread: f32) -> Base {
        let [x, y, z] = pos;
        let dist = (x * x + y * y + z * z).sqrt().max(0.3);
        Base {
            name: format!("aeolian.{id}"),
            pos,
            height: height.max(0.3),
            stem,
            sr: cfg.sr as f32,
            z0: scene_z0(scene),
            expo,
            spread,
            dist,
            az: y.atan2(x),
            el: z.atan2(x.hypot(y)),
            air: AirPath::new(dist, cfg.sr as f32, 3.0),
            field: GustField::new(scene, cfg.seed, cfg.sr, cfg.block),
            key: cfg.rng.key(&format!("aeolian.{id}")),
            n: cfg.block,
        }
    }

    /// (wind at the emitter's height incl. advected gust, 10 m-equivalent wind, gust multiplier)
    fn local_wind(&mut self, ctx: &BlockCtx) -> (f32, f32, f32) {
        let w = ctx.weather;
        self.field.advance(ctx.index, w, ctx.n);
        let u_h = wind_at_height(w.wind_u10_mps, self.height, self.z0);
        let g = self.field.at(upwind_distance(self.pos, w.wind_azimuth), u_h.max(0.5));
        (u_h * g * self.expo, w.wind_u10_mps * g * self.expo, g)
    }

    fn emit(&mut self, ctx: &BlockCtx, y: &mut [f32]) -> Emission {
        if self.stem != Stem::Interior {
            self.air.process(y, ctx.weather.temperature_c, ctx.weather.humidity_pct);
        }
        Emission::point(self.az, self.el, self.dist, self.spread, self.stem)
    }
}

// --------------------------------------------------------------------------
// cylinders: wires, guy lines, power lines, railing bars
// --------------------------------------------------------------------------
struct Element {
    d: f32,
    f1: f32,
    f: f32,
    phase: f64,
    psi: [f32; 2],
    jit: f32,
    lock: f32,
    mode: i64,
    am_lp: Biquad,
    band: Biquad,
    band_fc: f32,
    band_g: f32,
    amp: Ramp,
    bb_amp: Ramp,
}

pub struct WireVoice {
    b: Base,
    el: Vec<Element>,
    length: f32,
    lock_db: f32,
    axis_az: f32,
    gate: (f32, f32),
    power_line: bool,
    corona_db: f32,
    mains: f32,
    started: bool,
    am_norm: f32,
    corona_amp: Ramp,
    hum_phase: f64,
    crackle: Sos,
    key_corona: u64,
    y: Vec<f32>,
    nz: Vec<f32>,
    imp: Vec<f32>,
    offs: Vec<usize>,
}

pub struct WireCfg {
    pub diameter_m: f32,
    pub length_m: f32,
    pub n_elements: usize,
    pub f1_hz: f32,
    pub lock_db: f32,
    pub axis_az: Option<f32>,
    pub gate: (f32, f32),
    pub power_line: bool,
    pub corona_db: f32,
    pub mains_hz: f32,
    pub wind_exposure: f32,
}

impl WireVoice {
    #[allow(clippy::too_many_arguments)]
    fn new(id: &str, pos: [f32; 3], height: f32, stem: Stem, scene: &Scene, cfg: &BuildCfg, c: &mut Rng, wc: WireCfg) -> WireVoice {
        let mut b = Base::new(id, pos, height, stem, scene, cfg, wc.wind_exposure, 0.1);
        let sr = b.sr;
        let k = wc.n_elements.max(1);
        let length = wc.length_m.max(0.1);
        b.spread = if k == 1 {
            (0.7 * (2.0 / std::f32::consts::PI) * (length / (2.0 * b.dist)).atan()).min(0.9)
        } else {
            0.25
        };
        let jit: Vec<f32> = (0..k).map(|_| if k > 1 { c.range_f32(0.85, 1.15) } else { 1.0 }).collect();
        let f1j: Vec<f32> = (0..k).map(|_| if k > 1 { c.range_f32(0.97, 1.03) } else { 1.0 }).collect();
        let phases: Vec<f64> = (0..k).map(|_| c.range(0.0, std::f64::consts::TAU)).collect();
        let psi: Vec<[f32; 2]> = (0..k).map(|_| [c.range_f32(0.0, std::f32::consts::TAU), 0.0]).collect();
        let psi1: Vec<f32> = (0..k).map(|_| c.range_f32(0.0, std::f32::consts::TAU)).collect();
        let am = lp2(10.0, sr);
        let band0 = bp(200.0, 5.0, sr);
        let el = (0..k)
            .map(|i| Element {
                d: wc.diameter_m.max(5e-4) * jit[i],
                f1: wc.f1_hz * f1j[i],
                f: 200.0,
                phase: phases[i],
                psi: [psi[i][0], psi1[i]],
                jit: 0.0,
                lock: 0.0,
                mode: 0,
                am_lp: Biquad::new(am),
                band: Biquad::new(band0),
                band_fc: 200.0,
                band_g: bandpass_rms_gain(200.0, 5.0, sr, 1),
                amp: Ramp::default(),
                bb_amp: Ramp::default(),
            })
            .collect();
        let mut crackle_secs = vec![hp2(2500.0, sr), lp2(9000.0, sr)];
        let n = cfg.block;
        let axis_az = wc.axis_az.unwrap_or(b.az + 0.5 * std::f32::consts::PI);
        let key_corona = cfg.rng.key(&format!("aeolian.corona.{id}"));
        crackle_secs.shrink_to_fit();
        WireVoice {
            b,
            el,
            length,
            lock_db: wc.lock_db,
            axis_az,
            gate: wc.gate,
            power_line: wc.power_line,
            corona_db: wc.corona_db,
            mains: wc.mains_hz,
            started: false,
            am_norm: biquad_rms_gain(&am),
            corona_amp: Ramp::default(),
            hum_phase: 0.0,
            crackle: Sos::new(&crackle_secs),
            key_corona,
            y: vec![0.0; n],
            nz: vec![0.0; n],
            imp: vec![0.0; n],
            offs: Vec::with_capacity(512),
        }
    }

    fn corona(&mut self, ctx: &BlockCtx) {
        let w = ctx.weather;
        let n = ctx.n;
        let sr = self.b.sr;
        let wet = smoothstep(w.rain_rate_mm_h, 0.2, 6.0) + 0.35 * smoothstep(w.humidity_pct, 88.0, 100.0);
        let amp = db_spl(self.corona_db) * wet.min(1.2) * spreading(self.b.dist, 15.0);
        let (g0, g1) = self.corona_amp.to(amp);
        if g0.max(g1) < 1e-9 {
            return;
        }
        let mut rng = ctx.rng.stream(self.key_corona, ctx.index);
        let dph = std::f64::consts::TAU * 2.0 * self.mains as f64 / sr as f64;
        self.imp[..n].iter_mut().for_each(|v| *v = 0.0);
        let mut offs = std::mem::take(&mut self.offs);
        poisson_offsets(&mut rng, 60.0 + 400.0 * wet.min(1.0), n, sr, &mut offs);
        for &o in offs.iter() {
            self.imp[o] += rng.lognormal(0.0, 0.8) as f32 * if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
        }
        self.offs = offs;
        self.crackle.process_block(&mut self.imp[..n]);
        let mut ph = self.hum_phase;
        for i in 0..n {
            ph += dph;
            let p = ph as f32;
            let hum = 0.8 * p.sin() + 0.35 * (2.0 * p + 0.7).sin() + 0.2 * (3.0 * p + 1.9).sin();
            self.y[i] += ramp_at(g0, g1, i, n) * (0.6 * hum + 6.0 * self.imp[i]);
        }
        self.hum_phase = ph % std::f64::consts::TAU;
    }
}

impl Voice for WireVoice {
    fn name(&self) -> &str {
        &self.b.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n;
        let sr = self.b.sr;
        let dt = n as f32 / sr;
        let (u_loc, u_bft, _) = self.b.local_wind(ctx);
        if n != self.b.n {
            return Emission::silent(self.b.stem);
        }
        let normal = (w.wind_azimuth - self.axis_az).sin().abs().max(0.3);
        let un = u_loc * normal;
        let gate = smoothstep(u_bft, self.gate.0, self.gate.1);
        let idle = self.el.iter().all(|e| e.amp.v < 1e-9 && e.bb_amp.v < 1e-9);
        if gate <= 0.0 && idle && !self.power_line {
            return Emission::silent(self.b.stem);
        }
        let mut rng = ctx.rng.stream(self.b.key, ctx.index);
        let ti = w.turbulence_intensity.max(0.02);
        let a = (-dt / 0.25).exp();
        let bb = (1.0 - a * a).sqrt();
        let c = speed_of_sound(w.temperature_c);
        let r = self.b.dist;
        let k_el = self.el.len() as f32;
        let started = self.started;
        self.y[..n].iter_mut().for_each(|v| *v = 0.0);
        let mut any_bb = false;
        for e in self.el.iter_mut() {
            e.jit = a * e.jit + bb * rng.normal_f32();
            let fs = (ST_CYL * un.max(0.05) / e.d * (1.0 + 0.35 * ti * e.jit)).clamp(20.0, 0.4 * sr);
            let mut f_target = fs;
            if e.f1 > 0.0 {
                let f1 = e.f1.max(1e-3);
                let nmode = ((fs / f1).round() as i64).max(1);
                let fnm = nmode as f32 * f1;
                let det = (fs - fnm) / fnm;
                let mut bw = 0.07 * (1.0 - (ti / 0.5).min(0.8));
                if nmode == e.mode {
                    bw *= 1.6;
                }
                let locked = det.abs() < bw;
                let tgt = if locked { 1.0 } else { 0.0 };
                let rate = if tgt > e.lock { 1.0 - (-dt / 0.3).exp() } else { 1.0 - (-dt / 0.1).exp() };
                e.lock += (tgt - e.lock) * rate;
                if locked {
                    e.mode = nmode;
                }
                let ft = if locked { fnm * (1.0 + 0.1 * det) } else { fs };
                f_target = e.lock * ft + (1.0 - e.lock) * fs;
            }
            if !started {
                e.f = f_target;
            }
            let f_old = e.f;
            e.f = f_target;
            // turbulent amplitude flicker of the shedding (deep when unlocked)
            fill_white(&mut rng, &mut self.nz[..n]);
            e.am_lp.process_block(&mut self.nz[..n]);
            let depth = 0.55 * (1.0 - e.lock) + 0.12;
            // Phillips / Curle level along the cylinder
            let l_c = 4.0 * e.d;
            let geo = (l_c * (2.0 / r) * (self.length / (2.0 * r)).atan()).sqrt();
            let re = un * e.d / NU_AIR;
            let active = ((re - 40.0) / 110.0).clamp(0.0, 1.0);
            let mut p_rms = geo * ST_CYL * RHO_AIR * un.powi(3) * CL_RMS / (4.0 * c) * active;
            p_rms *= 10f32.powf(self.lock_db * e.lock / 20.0) * gate / k_el.sqrt();
            let (g0, g1) = e.amp.to(p_rms);
            let inv_am = 1.0 / self.am_norm;
            let mut ph = e.phase;
            let df = (f_target - f_old) as f64 / n as f64;
            let two_pi_sr = std::f64::consts::TAU / sr as f64;
            for i in 0..n {
                let fi = f_old as f64 + df * (i + 1) as f64;
                ph += two_pi_sr * fi;
                let p = ph as f32;
                let tone = SQRT_2 * (p.sin() + 0.12 * (2.0 * p + e.psi[0]).sin() + 0.07 * (3.0 * p + e.psi[1]).sin()) / 1.01;
                let am = (1.0 + depth * self.nz[i] * inv_am).max(0.0);
                self.y[i] += ramp_at(g0, g1, i, n) * tone * am;
            }
            e.phase = ph % std::f64::consts::TAU;
            // broadband shedding noise around the Strouhal frequency (unlocked, turbulent)
            let bb_t = p_rms * 0.6 * (1.0 - e.lock);
            if bb_t > 1e-9 || e.bb_amp.v > 1e-9 {
                any_bb = true;
                let (h0, h1) = e.bb_amp.to(bb_t);
                if (f_target - e.band_fc).abs() / e.band_fc > 0.03 {
                    e.band_fc = f_target;
                    e.band.set(bp(f_target, 5.0, sr));
                    e.band_g = bandpass_rms_gain(f_target, 5.0, sr, 1);
                }
                fill_white(&mut rng, &mut self.nz[..n]);
                let inv = 1.0 / e.band_g;
                for i in 0..n {
                    self.y[i] += ramp_at(h0, h1, i, n) * e.band.process(self.nz[i]) * inv;
                }
                e.band.flush_state();
            }
        }
        let _ = any_bb;
        self.started = true;
        if self.power_line && self.corona_db > 0.0 {
            self.corona(ctx);
        }
        let y = &mut self.y[..n];
        let em = self.b.emit(ctx, y);
        out[..n].copy_from_slice(y);
        em
    }

    fn reset(&mut self) {
        self.b.field.reset();
        self.b.air.reset();
        self.started = false;
        for e in &mut self.el {
            e.jit = 0.0;
            e.lock = 0.0;
            e.mode = 0;
            e.am_lp.reset();
            e.band.reset();
            e.amp = Ramp::default();
            e.bb_amp = Ramp::default();
        }
        self.corona_amp = Ramp::default();
        self.hum_phase = 0.0;
        self.crackle.reset();
    }
}

// --------------------------------------------------------------------------
// edge / gap / slot whistles
// --------------------------------------------------------------------------
pub struct GapVoice {
    b: Base,
    f_res: f32,
    gap: f32,
    cp: f32,
    onset: f32,
    level_db: f32,
    howl_db: f32,
    f: f32,
    phase: f64,
    jit: f32,
    env_lp: Biquad,
    env_norm: f32,
    howl: Sos,
    howl_g: f32,
    whistle_amp: Ramp,
    howl_amp: Ramp,
    y: Vec<f32>,
    env: Vec<f32>,
    nz: Vec<f32>,
}

pub struct GapCfg {
    pub f_res: f32,
    pub q: f32,
    pub gap_m: f32,
    pub cp: f32,
    pub onset_mps: f32,
    pub level_db: f32,
    pub wind_exposure: f32,
    pub spread: f32,
}

impl GapVoice {
    #[allow(clippy::too_many_arguments)]
    fn new(id: &str, pos: [f32; 3], height: f32, stem: Stem, scene: &Scene, cfg: &BuildCfg, c: &mut Rng, gc: GapCfg) -> GapVoice {
        let b = Base::new(id, pos, height, stem, scene, cfg, gc.wind_exposure, gc.spread);
        let sr = b.sr;
        let env_c = lp2(3.0, sr);
        let howl_secs = [bp(gc.f_res, 2.0, sr), bp(gc.f_res * 0.5, 0.8, sr)];
        let n = cfg.block;
        let _ = gc.q;
        GapVoice {
            f_res: gc.f_res,
            gap: gc.gap_m.max(5e-4),
            cp: gc.cp.max(0.05),
            onset: gc.onset_mps,
            level_db: gc.level_db,
            howl_db: gc.level_db - 8.0,
            f: gc.f_res,
            phase: c.range(0.0, std::f64::consts::TAU),
            jit: 0.0,
            env_lp: Biquad::new(env_c),
            env_norm: biquad_rms_gain(&env_c),
            howl: Sos::new(&howl_secs),
            howl_g: noise_rms_gain(&howl_secs),
            whistle_amp: Ramp::default(),
            howl_amp: Ramp::default(),
            y: vec![0.0; n],
            env: vec![0.0; n],
            nz: vec![0.0; n],
            b,
        }
    }
}

impl Voice for GapVoice {
    fn name(&self) -> &str {
        &self.b.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.b.sr;
        let dt = n as f32 / sr;
        let (u_loc, _, _) = self.b.local_wind(ctx);
        if n != self.b.n {
            return Emission::silent(self.b.stem);
        }
        let scp = self.cp.sqrt();
        let vj = u_loc * scp;
        let drive = smoothstep(vj, self.onset * scp, 1.6 * self.onset * scp);
        if drive <= 0.0 && self.whistle_amp.idle() && self.howl_amp.idle() {
            return Emission::silent(self.b.stem);
        }
        let mut rng = ctx.rng.stream(self.b.key, ctx.index);
        let a = (-dt / 0.4).exp();
        self.jit = a * self.jit + (1.0 - a * a).sqrt() * rng.normal_f32();
        let fe = (0.3 * vj.max(0.1) / self.gap).max(1.0);
        let lr = (fe / self.f_res).ln();
        let coupling = (-(lr / 0.35).powi(2)).exp();
        let f_new = self.f_res * (1.0 + 0.04 * (3.0 * lr).tanh() + 0.008 * self.jit);
        let f_old = self.f;
        self.f = f_new;
        fill_white(&mut rng, &mut self.env[..n]);
        self.env_lp.process_block(&mut self.env[..n]);
        let refv = 10.0 * scp;
        let wa = db_spl(self.level_db) * (vj / refv).powf(2.5) * coupling * drive * spreading(self.b.dist, 1.0);
        let ha = db_spl(self.howl_db) * (vj / refv).powi(3) * drive * spreading(self.b.dist, 1.0);
        let (w0, w1) = self.whistle_amp.to(wa);
        let (h0, h1) = self.howl_amp.to(ha);
        fill_white(&mut rng, &mut self.nz[..n]);
        self.howl.process_block(&mut self.nz[..n]);
        let inv_env = 1.0 / self.env_norm;
        let inv_howl = 1.0 / self.howl_g;
        let mut ph = self.phase;
        let df = (f_new - f_old) as f64 / n as f64;
        let k = std::f64::consts::TAU / sr as f64;
        for i in 0..n {
            ph += k * (f_old as f64 + df * (i + 1) as f64);
            let p = ph as f32;
            let inter = (0.55 + 0.45 * self.env[i] * inv_env).clamp(0.0, 1.5);
            let tone = SQRT_2 * (p.sin() + 0.15 * (2.0 * p + 1.1).sin());
            self.y[i] = ramp_at(w0, w1, i, n) * tone * inter + ramp_at(h0, h1, i, n) * self.nz[i] * inv_howl * inter.sqrt();
        }
        self.phase = ph % std::f64::consts::TAU;
        let y = &mut self.y[..n];
        let em = self.b.emit(ctx, y);
        out[..n].copy_from_slice(y);
        em
    }

    fn reset(&mut self) {
        self.b.field.reset();
        self.b.air.reset();
        self.f = self.f_res;
        self.jit = 0.0;
        self.env_lp.reset();
        self.howl.reset();
        self.whistle_amp = Ramp::default();
        self.howl_amp = Ramp::default();
    }
}

// --------------------------------------------------------------------------
// wind chimes
// --------------------------------------------------------------------------
pub struct ChimeVoice {
    b: Base,
    n_tubes: usize,
    bank: StrikeBank,
    mode_gain: Vec<f32>,
    tube_of_mode: Vec<usize>,
    level: f32,
    carry: Vec<(u64, usize, f32)>,
    strikes: Vec<(usize, usize)>,
    amps: Vec<f32>,
    offs: Vec<usize>,
    y: Vec<f32>,
}

impl ChimeVoice {
    #[allow(clippy::too_many_arguments)]
    fn new(id: &str, pos: [f32; 3], height: f32, scene: &Scene, cfg: &BuildCfg, c: &mut Rng, n_tubes: usize, base_hz: f32, scale: &str, material: &str, level_db: f32, expo: f32) -> ChimeVoice {
        let b = Base::new(id, pos, height, Stem::Wind, scene, cfg, expo, 0.05);
        let ratios = scale_ratios(scale);
        let nt = n_tubes.clamp(1, ratios.len());
        let wood = WOOD.contains(&material.to_ascii_lowercase().as_str());
        let mut freqs = Vec::new();
        let mut t60s = Vec::new();
        let mut gains_all = Vec::new();
        let mut tube_all = Vec::new();
        let f0: Vec<f64> = (0..nt).map(|i| ratios[i] * base_hz as f64 * c.range(0.995, 1.005)).collect();
        for (ti, &fi) in f0.iter().enumerate() {
            for (k, ratio) in CHIME_RATIOS.iter().enumerate() {
                let f = fi * ratio * c.range(0.997, 1.003);
                let t60 = if wood {
                    [0.35, 0.18, 0.1, 0.06][k] * (500.0 / fi).powf(0.3)
                } else {
                    6.0 * (523.0 / fi).powf(0.35) * (fi / f).powf(0.6)
                };
                freqs.push(f);
                t60s.push(t60);
                gains_all.push(if wood { [1.0f32, 0.35, 0.15, 0.06][k] } else { [1.0f32, 0.55, 0.35, 0.2][k] });
                tube_all.push(ti);
            }
        }
        let bank = StrikeBank::new(&freqs, &t60s, b.sr);
        let mode_gain = bank.keep.iter().map(|&i| gains_all[i]).collect();
        let tube_of_mode = bank.keep.iter().map(|&i| tube_all[i]).collect();
        let m = bank.m;
        let n = cfg.block;
        ChimeVoice {
            b,
            n_tubes: nt,
            bank,
            mode_gain,
            tube_of_mode,
            level: db_spl(level_db - if wood { 8.0 } else { 0.0 }),
            carry: Vec::with_capacity(32),
            strikes: Vec::with_capacity(64),
            amps: Vec::with_capacity(64 * m.max(1)),
            offs: Vec::with_capacity(64),
            y: vec![0.0; n],
        }
    }

    fn push_strike(&mut self, rng: &mut Rng, off: usize, tube: usize, amp: f32) {
        if self.strikes.len() >= self.strikes.capacity() {
            return;
        }
        let row = self.strikes.len();
        self.strikes.push((off, row));
        for md in 0..self.bank.m {
            let a = if self.tube_of_mode[md] == tube { amp * self.mode_gain[md] * rng.lognormal(0.0, 0.25) as f32 } else { 0.0 };
            self.amps.push(a);
        }
    }
}

impl Voice for ChimeVoice {
    fn name(&self) -> &str {
        &self.b.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.b.sr;
        let (u_loc, _, g) = self.b.local_wind(ctx);
        if n != self.b.n {
            return Emission::silent(Stem::Wind);
        }
        let start = (ctx.t0 * ctx.sr as f64).round() as u64;
        let rate = (3.5 * smoothstep(u_loc, 0.8, 3.5) * (u_loc.max(0.1) / 5.0).powf(0.6) * (1.0 + 1.5 * (g - 1.0).max(0.0))).min(10.0);
        if rate <= 0.0 && self.bank.silent() && self.carry.is_empty() {
            return Emission::silent(Stem::Wind);
        }
        let mut rng = ctx.rng.stream(self.b.key, ctx.index);
        self.strikes.clear();
        self.amps.clear();
        let end = start + n as u64;
        let mut i = 0;
        while i < self.carry.len() {
            if self.carry[i].0 < end {
                let (t, tube, amp) = self.carry.swap_remove(i);
                let off = t.saturating_sub(start) as usize;
                self.push_strike(&mut rng, off, tube, amp);
            } else {
                i += 1;
            }
        }
        let mut offs = std::mem::take(&mut self.offs);
        poisson_offsets(&mut rng, rate, n, sr, &mut offs);
        for &off in offs.iter() {
            let tube = rng.below(self.n_tubes as u64) as usize;
            let amp = rng.lognormal(0.0, 0.35) as f32 * (u_loc.max(0.1) / 6.0).min(1.0).powf(0.7);
            self.push_strike(&mut rng, off, tube, amp);
            if rng.uniform() < 0.3 {
                let off2 = off + (rng.range(0.015, 0.06) * sr as f64) as usize;
                let dir: i64 = if rng.bernoulli(0.5) { 1 } else { -1 };
                let tube2 = (tube as i64 + dir).clamp(0, self.n_tubes as i64 - 1) as usize;
                let amp2 = amp * rng.range_f32(0.3, 0.7);
                if off2 < n {
                    self.push_strike(&mut rng, off2, tube2, amp2);
                } else if self.carry.len() < self.carry.capacity() {
                    self.carry.push((start + off2 as u64, tube2, amp2));
                }
            }
        }
        self.offs = offs;
        self.strikes.sort_unstable();
        let y = &mut self.y[..n];
        y.iter_mut().for_each(|v| *v = 0.0);
        let gain = self.level * spreading(self.b.dist, 1.0);
        self.bank.process(&self.strikes, &self.amps, y, gain);
        let y = &mut self.y[..n];
        let em = self.b.emit(ctx, y);
        out[..n].copy_from_slice(y);
        em
    }

    fn reset(&mut self) {
        self.b.field.reset();
        self.b.air.reset();
        self.bank.reset();
        self.carry.clear();
    }
}

// --------------------------------------------------------------------------
// flags, fabric flutter, halyards
// --------------------------------------------------------------------------
pub struct FlagVoice {
    b: Base,
    chord: f32,
    st: f32,
    rustle_db: f32,
    snap_db: f32,
    snap_range: (f32, f32),
    onset: f32,
    phase: f64,
    f: f32,
    jit: f32,
    fc: f32,
    rustle: Biquad,
    rustle_g: f32,
    rustle_amp: Ramp,
    snaps: EventRing,
    pole: Option<StrikeBank>,
    pole_gain: Vec<f32>,
    y: Vec<f32>,
    nz: Vec<f32>,
    snap: Vec<f32>,
    wraps: Vec<usize>,
    strikes: Vec<(usize, usize)>,
    amps: Vec<f32>,
    offs: Vec<usize>,
}

pub struct FlagCfg {
    pub chord_m: f32,
    pub strouhal: f32,
    pub rustle_db: f32,
    pub snap_db: f32,
    pub snap_range: (f32, f32),
    pub onset_mps: f32,
    pub halyard: bool,
    pub wind_exposure: f32,
    pub spread: f32,
}

impl Default for FlagCfg {
    fn default() -> Self {
        FlagCfg { chord_m: 1.2, strouhal: 0.2, rustle_db: 52.0, snap_db: 66.0, snap_range: (6.0, 13.0), onset_mps: 2.0, halyard: false, wind_exposure: 1.0, spread: 0.2 }
    }
}

impl FlagVoice {
    #[allow(clippy::too_many_arguments)]
    fn new(id: &str, pos: [f32; 3], height: f32, stem: Stem, scene: &Scene, cfg: &BuildCfg, c: &mut Rng, fc_: FlagCfg) -> FlagVoice {
        let b = Base::new(id, pos, height, stem, scene, cfg, fc_.wind_exposure, fc_.spread);
        let sr = b.sr;
        let phase = c.uniform();
        let rc = bp(700.0, 0.8, sr);
        let (pole, pole_gain) = if fc_.halyard {
            let f_pole = c.range(700.0, 1300.0);
            let freqs: Vec<f64> = CHIME_RATIOS.iter().map(|r| f_pole * r * c.range(0.98, 1.02)).collect();
            let bank = StrikeBank::new(&freqs, &[1.6, 0.9, 0.5, 0.3], sr);
            let g = bank.keep.iter().map(|&i| [1.0f32, 0.6, 0.35, 0.2][i]).collect();
            (Some(bank), g)
        } else {
            (None, Vec::new())
        };
        let n = cfg.block;
        let snap_max = (0.016 * sr) as usize + 2;
        FlagVoice {
            chord: fc_.chord_m.clamp(0.2, 4.0),
            st: fc_.strouhal,
            rustle_db: fc_.rustle_db,
            snap_db: fc_.snap_db,
            snap_range: fc_.snap_range,
            onset: fc_.onset_mps,
            phase,
            f: 0.5,
            jit: 0.0,
            fc: 700.0,
            rustle: Biquad::new(rc),
            rustle_g: biquad_rms_gain(&rc),
            rustle_amp: Ramp::default(),
            snaps: EventRing::new(1, 2 * n + (0.1 * sr) as usize + snap_max),
            pole,
            pole_gain,
            y: vec![0.0; n],
            nz: vec![0.0; n],
            snap: vec![0.0; snap_max],
            wraps: Vec::with_capacity(64),
            strikes: Vec::with_capacity(32),
            amps: Vec::with_capacity(32 * 4),
            offs: Vec::with_capacity(32),
            b,
        }
    }

    fn make_snap(&mut self, rng: &mut Rng) -> usize {
        let sr = self.b.sr;
        let l = ((rng.range_f32(0.004, 0.016) * sr) as usize).clamp(4, self.snap.len());
        let tau = l as f32 / sr / 3.0;
        let fc = rng.range_f32(1200.0, 3200.0);
        let mut f = Sos::new(&[bp(fc, 0.9, sr), hp2(250.0, sr)]);
        for i in 0..l {
            let t = i as f32 / sr;
            self.snap[i] = f.process(rng.normal_f32() * (-t / tau).exp());
        }
        normalize_peak(&mut self.snap[..l]);
        l
    }
}

impl Voice for FlagVoice {
    fn name(&self) -> &str {
        &self.b.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.b.sr;
        let dt = n as f32 / sr;
        let (u, _, _) = self.b.local_wind(ctx);
        if n != self.b.n {
            return Emission::silent(self.b.stem);
        }
        let active = smoothstep(u, self.onset, self.onset * 2.0);
        let pole_idle = self.pole.as_ref().map(|p| p.silent()).unwrap_or(true);
        if active <= 0.0 && self.rustle_amp.idle() && !self.snaps.active() && pole_idle {
            return Emission::silent(self.b.stem);
        }
        let mut rng = ctx.rng.stream(self.b.key, ctx.index);
        let start = (ctx.t0 * ctx.sr as f64).round() as u64;
        let a = (-dt / 0.6).exp();
        self.jit = a * self.jit + (1.0 - a * a).sqrt() * rng.normal_f32();
        let f_new = (self.st * u / self.chord * (1.0 + 0.15 * self.jit)).max(0.05);
        let f_old = self.f;
        self.f = f_new;

        // flutter rustle, swelling twice per flap cycle
        let fc = 450.0 + 55.0 * u;
        if (fc - self.fc).abs() / self.fc > 0.04 {
            self.fc = fc;
            let c = bp(fc, 0.8, sr);
            self.rustle.set(c);
            self.rustle_g = biquad_rms_gain(&c);
        }
        let amp = db_spl(self.rustle_db) * (u.max(0.0) / 10.0).powi(2) * active * spreading(self.b.dist, 1.0);
        let (g0, g1) = self.rustle_amp.to(amp);
        fill_white(&mut rng, &mut self.nz[..n]);
        self.wraps.clear();
        let mut ph = self.phase;
        let df = (f_new - f_old) as f64 / n as f64;
        let inv = 1.0 / self.rustle_g;
        for i in 0..n {
            let prev = ph;
            ph += (f_old as f64 + df * (i + 1) as f64) / sr as f64;
            if ph.floor() > prev.floor() && self.wraps.len() < self.wraps.capacity() {
                self.wraps.push(i);
            }
            let s = (std::f64::consts::TAU * ph).sin() as f32;
            let motion = 0.35 + 0.65 * s * s;
            self.y[i] = ramp_at(g0, g1, i, n) * self.rustle.process(self.nz[i]) * inv * motion;
        }
        self.rustle.flush_state();
        self.phase = ph % 1.0;

        // trailing-edge snaps once the flapping saturates
        let p_snap = smoothstep(u, self.snap_range.0, self.snap_range.1);
        if p_snap > 0.0 {
            let s_amp = db_spl(self.snap_db) * (u / 10.0).powi(2) * spreading(self.b.dist, 1.0);
            let wraps = std::mem::take(&mut self.wraps);
            for &off in wraps.iter() {
                let r1 = rng.uniform() as f32;
                let r2 = rng.uniform() as f32;
                if r1 < p_snap * (0.6 + 0.4 * r2) {
                    let ln = rng.lognormal(0.0, 0.3) as f32;
                    let l = self.make_snap(&mut rng);
                    let sig = std::mem::take(&mut self.snap);
                    self.snaps.add(start + off as u64, 0, &sig[..l], s_amp * ln);
                    self.snap = sig;
                    if (rng.uniform() as f32) < 0.25 * p_snap {
                        let d2 = (rng.range(0.02, 0.08) * sr as f64) as u64;
                        let l = self.make_snap(&mut rng);
                        let sig = std::mem::take(&mut self.snap);
                        self.snaps.add(start + off as u64 + d2, 0, &sig[..l], 0.5 * s_amp);
                        self.snap = sig;
                    }
                }
            }
            self.wraps = wraps;
        }
        let y = &mut self.y[..n];
        if self.snaps.active() {
            self.snaps.take_add(start, n, y);
        }

        // halyard slapping the (metal) pole
        if let Some(pole) = &mut self.pole {
            let rate = 1.4 * smoothstep(u, 3.0, 9.0) * (1.0 + 0.5 * self.jit.abs());
            self.strikes.clear();
            self.amps.clear();
            let mut offs = std::mem::take(&mut self.offs);
            poisson_offsets(&mut rng, rate, n, sr, &mut offs);
            for &off in offs.iter() {
                if self.strikes.len() >= self.strikes.capacity() {
                    break;
                }
                let amp_s = rng.lognormal(0.0, 0.4) as f32 * (u / 8.0).min(1.5);
                let row = self.strikes.len();
                self.strikes.push((off, row));
                for md in 0..pole.m {
                    self.amps.push(amp_s * self.pole_gain[md] * rng.lognormal(0.0, 0.3) as f32);
                }
            }
            self.offs = offs;
            pole.process(&self.strikes, &self.amps, y, db_spl(64.0) * spreading(self.b.dist, 1.0));
        }
        let em = self.b.emit(ctx, y);
        out[..n].copy_from_slice(y);
        em
    }

    fn reset(&mut self) {
        self.b.field.reset();
        self.b.air.reset();
        self.f = 0.5;
        self.jit = 0.0;
        self.rustle.reset();
        self.rustle_amp = Ramp::default();
        self.snaps.reset();
        if let Some(p) = &mut self.pole {
            p.reset();
        }
    }
}

// --------------------------------------------------------------------------
// car body buffeting
// --------------------------------------------------------------------------
pub struct BuffetVoice {
    b: Base,
    level_db: f32,
    lp: Sos,
    lp_g: f32,
    amp: Ramp,
    g_prev: f32,
    y: Vec<f32>,
}

impl BuffetVoice {
    fn new(id: &str, pos: [f32; 3], height: f32, scene: &Scene, cfg: &BuildCfg, level_db: f32) -> BuffetVoice {
        let b = Base::new(id, pos, height, Stem::Interior, scene, cfg, 1.0, 0.9);
        let secs = [lp2(38.0, b.sr), hp2(8.0, b.sr)];
        let n = cfg.block;
        BuffetVoice { level_db, lp: Sos::new(&secs), lp_g: noise_rms_gain(&secs), amp: Ramp::default(), g_prev: 1.0, y: vec![0.0; n], b }
    }
}

impl Voice for BuffetVoice {
    fn name(&self) -> &str {
        &self.b.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let (u, _, g) = self.b.local_wind(ctx);
        let dgdt = (g - self.g_prev).abs() / (n as f32 / self.b.sr);
        self.g_prev = g;
        if (u < 4.0 && self.amp.idle()) || n != self.b.n {
            return Emission::silent(Stem::Interior);
        }
        let mut rng = ctx.rng.stream(self.b.key, ctx.index);
        let amp = db_spl(self.level_db) * (u / 15.0).powi(2) * smoothstep(u, 4.0, 9.0) * (0.5 + dgdt.min(2.0));
        let (g0, g1) = self.amp.to(amp / self.lp_g);
        let y = &mut self.y[..n];
        fill_white(&mut rng, y);
        self.lp.process_block(y);
        apply_ramp(y, g0, g1);
        out[..n].copy_from_slice(y);
        Emission::point(0.0, 0.0, 1.0, 0.9, Stem::Interior)
    }

    fn reset(&mut self) {
        self.b.field.reset();
        self.lp.reset();
        self.amp = Ramp::default();
        self.g_prev = 1.0;
    }
}

// --------------------------------------------------------------------------
// construction (raingen.synth.aeolian.AeolianGenerator)
// --------------------------------------------------------------------------
fn pnum(p: &Patch, key: &str, default: f32) -> f32 {
    p.param_f32(key, default)
}

fn patch_height(p: &Patch, h: f32) -> f32 {
    if p.params.get("height_m").is_some() {
        pnum(p, "height_m", 2.0)
    } else {
        (p.position[2] + h).max(0.3)
    }
}

fn mains_hz(scene: &Scene) -> f32 {
    match biophony_str(scene, "region") {
        Some(r) if r.to_ascii_lowercase() == "north_america" => 60.0,
        _ => 50.0,
    }
}

fn cfg_rng(cfg: &BuildCfg, tag: &str) -> Rng {
    Rng::from_key(cfg.rng.key(&format!("aeolian.cfg.{tag}")), 0)
}

fn from_patch(scene: &Scene, p: &Patch, cfg: &BuildCfg, h: f32, mains: f32) -> Option<BoxedVoice> {
    let kind = p.kind.name().to_ascii_lowercase();
    let ptype = p.param_str("type").map(|s| s.to_ascii_lowercase()).unwrap_or_else(|| kind.clone());
    let mut c = cfg_rng(cfg, &p.id);
    let height = patch_height(p, h);
    let expo = pnum(p, "wind_exposure", 1.0);
    let axis = p.params.get("axis_deg").map(|_| pnum(p, "axis_deg", 0.0).to_radians());
    let is = |list: &[&str]| list.contains(&kind.as_str()) || list.contains(&ptype.as_str());
    if is(&WIRE_KINDS) {
        let power = p.param_bool("power_line", ptype == "power_line");
        let d = pnum(p, "diameter_m", if power { 0.025 } else { 0.005 });
        let l = pnum(p, "length_m", 40.0);
        let f1 = if p.params.get("f1_hz").is_some() {
            pnum(p, "f1_hz", 0.0)
        } else {
            let tension = pnum(p, "tension_n", if power { 15000.0 } else { 900.0 });
            let density = if power { 3500.0 } else { 7850.0 };
            string_f1(l, tension, d, density)
        };
        let kv = pnum(p, "kv", if power { 132.0 } else { 0.0 });
        let corona = if power && kv > 1.0 { 36.0 + 12.0 * smoothstep(kv, 60.0, 220.0) } else { 0.0 };
        let wc = WireCfg {
            diameter_m: d,
            length_m: l,
            n_elements: 1,
            f1_hz: f1,
            lock_db: if l > 10.0 { 6.0 } else { 10.0 },
            axis_az: axis,
            gate: (7.5, 11.0),
            power_line: power,
            corona_db: corona,
            mains_hz: mains,
            wind_exposure: expo,
        };
        return Some(Box::new(WireVoice::new(&p.id, p.position, height, Stem::Wind, scene, cfg, &mut c, wc)));
    }
    if is(&BAR_KINDS) {
        let d = pnum(p, "diameter_m", if ptype == "fence" { 0.004 } else { 0.025 });
        let nb = pnum(p, "n_bars", 6.0).max(1.0) as usize;
        let wc = WireCfg {
            diameter_m: d,
            length_m: pnum(p, "bar_length_m", 1.0),
            n_elements: nb.min(64),
            f1_hz: 0.0,
            lock_db: 0.0,
            axis_az: Some(axis.unwrap_or(0.0)),
            gate: (7.0, 10.5),
            power_line: false,
            corona_db: 0.0,
            mains_hz: mains,
            wind_exposure: expo,
        };
        return Some(Box::new(WireVoice::new(&p.id, p.position, height, Stem::Wind, scene, cfg, &mut c, wc)));
    }
    if is(&GAP_KINDS) {
        let gap = pnum(p, "gap_mm", 4.0) / 1000.0;
        let default_f = 0.3 * 10.0 * 0.6f32.sqrt() / gap.max(1e-4);
        let f_res = if p.params.get("freq_hz").is_some() { pnum(p, "freq_hz", 600.0) } else { pnum(p, "f_res_hz", default_f) };
        let stem = if p.interior { Stem::Interior } else { Stem::Wind };
        let gc = GapCfg {
            f_res,
            q: pnum(p, "q", 10.0),
            gap_m: gap,
            cp: 0.6,
            onset_mps: pnum(p, "onset_mps", 6.0),
            level_db: pnum(p, "level_db", 50.0),
            wind_exposure: expo,
            spread: 0.15,
        };
        return Some(Box::new(GapVoice::new(&p.id, p.position, height, stem, scene, cfg, &mut c, gc)));
    }
    if kind == "chime" || ptype == "chime" {
        let nt = pnum(p, "n_tubes", 5.0).max(1.0) as usize;
        let scale = p.param_str("scale").unwrap_or("pentatonic").to_string();
        return Some(Box::new(ChimeVoice::new(
            &p.id,
            p.position,
            height,
            scene,
            cfg,
            &mut c,
            nt,
            pnum(p, "base_freq_hz", 523.25),
            &scale,
            &p.material,
            pnum(p, "level_db", 72.0),
            expo,
        )));
    }
    if kind == "flag" || ptype == "flag" {
        let area = pnum(p, "area_m2", p.area_m2);
        let pole_h = pnum(p, "pole_height_m", height);
        let fc = FlagCfg {
            chord_m: (area.max(0.05) * 1.5).sqrt(),
            halyard: p.param_bool("halyard", true),
            wind_exposure: expo,
            ..FlagCfg::default()
        };
        return Some(Box::new(FlagVoice::new(&p.id, p.position, pole_h.max(1.0), Stem::Wind, scene, cfg, &mut c, fc)));
    }
    None
}

/// Every wind-excited object of the layout plus the listener's shelter and street corners.
pub fn build(scene: &Scene, patches: &[Patch], cfg: &BuildCfg) -> Vec<BoxedVoice> {
    let h = scene.listener.height_m;
    let mains = mains_hz(scene);
    let mut out: Vec<BoxedVoice> = Vec::new();
    for p in patches {
        if let Some(v) = from_patch(scene, p, cfg, h, mains) {
            out.push(v);
        }
    }
    let shelter = scene.place.shelter;
    match shelter {
        Shelter::Tent => {
            let f1 = string_f1(1.8, 60.0, 0.003, 1150.0);
            for (k, (x, y)) in [(1.6f32, 1.3f32), (-1.6, -1.3)].iter().enumerate() {
                let mut c = cfg_rng(cfg, &format!("shelter.guy{k}"));
                let f1k = f1 * c.range_f32(0.9, 1.1);
                let wc = WireCfg {
                    diameter_m: 0.003,
                    length_m: 1.8,
                    n_elements: 1,
                    f1_hz: f1k,
                    lock_db: 10.0,
                    axis_az: None,
                    gate: (5.5, 9.0),
                    power_line: false,
                    corona_db: 0.0,
                    mains_hz: mains,
                    wind_exposure: 1.0,
                };
                out.push(Box::new(WireVoice::new(&format!("shelter.tent.guy{k}"), [*x, *y, 0.4 - h], 0.8, Stem::Wind, scene, cfg, &mut c, wc)));
            }
            let mut c = cfg_rng(cfg, "shelter.fly");
            let fc = FlagCfg {
                chord_m: 1.0,
                strouhal: 0.3,
                rustle_db: 58.0,
                snap_db: 70.0,
                snap_range: (6.0, 12.0),
                onset_mps: 2.5,
                halyard: false,
                wind_exposure: 0.8,
                spread: 0.7,
            };
            out.push(Box::new(FlagVoice::new("shelter.tent.fly", [0.0, 0.45, 0.35], (h + 0.35).max(1.0), Stem::Interior, scene, cfg, &mut c, fc)));
        }
        Shelter::Umbrella => {
            let mut c = cfg_rng(cfg, "shelter.umbrella");
            let fc = FlagCfg {
                chord_m: 0.6,
                strouhal: 0.35,
                rustle_db: 50.0,
                snap_db: 68.0,
                snap_range: (7.0, 14.0),
                onset_mps: 3.0,
                halyard: false,
                wind_exposure: 1.0,
                spread: 0.6,
            };
            out.push(Box::new(FlagVoice::new("shelter.umbrella.canopy", [0.0, 0.0, 0.35], h + 0.35, Stem::Wind, scene, cfg, &mut c, fc)));
        }
        Shelter::Car => {
            let mut c = cfg_rng(cfg, "shelter.seal");
            let f_res = c.range_f32(950.0, 1500.0);
            let gc = GapCfg { f_res, q: 18.0, gap_m: 0.0008, cp: 0.5, onset_mps: 9.0, level_db: 40.0, wind_exposure: 1.0, spread: 0.3 };
            out.push(Box::new(GapVoice::new("shelter.car.door_seal", [0.2, 0.6, 0.0], 1.2, Stem::Interior, scene, cfg, &mut c, gc)));
            out.push(Box::new(BuffetVoice::new("shelter.car.buffet", [0.0, 0.0, 0.0], 1.2, scene, cfg, 62.0)));
        }
        Shelter::RoomOpen | Shelter::RoomClosed => {
            let wo = scene.place.window_open;
            let mut c = cfg_rng(cfg, "shelter.window");
            if wo > 0.0 && wo <= 0.3 {
                let f_res = c.range_f32(350.0, 900.0);
                let gc = GapCfg { f_res, q: 12.0, gap_m: 0.002 + 0.05 * wo, cp: 0.7, onset_mps: 5.0, level_db: 52.0 - 20.0 * wo, wind_exposure: 1.0, spread: 0.2 };
                out.push(Box::new(GapVoice::new("shelter.room.window_gap", [2.0, 0.0, 0.0], h, Stem::Interior, scene, cfg, &mut c, gc)));
            } else {
                let f_res = c.range_f32(600.0, 1200.0);
                let gc = GapCfg { f_res, q: 20.0, gap_m: 0.0007, cp: 0.7, onset_mps: 11.0, level_db: 36.0, wind_exposure: 1.0, spread: 0.2 };
                out.push(Box::new(GapVoice::new("shelter.room.window_seal", [2.0, 0.0, 0.0], h, Stem::Interior, scene, cfg, &mut c, gc)));
            }
        }
        Shelter::Attic => {
            let mut c = cfg_rng(cfg, "shelter.vent");
            let f_res = c.range_f32(170.0, 320.0);
            let gc = GapCfg { f_res, q: 5.0, gap_m: 0.01, cp: 0.8, onset_mps: 7.0, level_db: 50.0, wind_exposure: 1.0, spread: 0.6 };
            out.push(Box::new(GapVoice::new("shelter.attic.vent", [0.0, 0.0, 1.6], h + 1.6, Stem::Interior, scene, cfg, &mut c, gc)));
        }
        _ => {}
    }
    let preset = scene.place.preset.to_ascii_lowercase();
    if ["city", "street", "alley", "downtown", "urban", "town"].iter().any(|k| preset.contains(k)) {
        let facade = patches
            .iter()
            .find(|p| p.param_bool("facade", false) && p.kind == crate::types::PatchKind::Surface)
            .map(|p| p.position)
            .unwrap_or([8.0, 6.0, 2.0]);
        let mut c = cfg_rng(cfg, "shelter.corner");
        let f_res = c.range_f32(220.0, 480.0);
        let gc = GapCfg { f_res, q: 3.0, gap_m: 0.05, cp: 0.9, onset_mps: 6.0, level_db: 58.0, wind_exposure: 1.0, spread: 0.5 };
        out.push(Box::new(GapVoice::new("scenery.corner_howl", facade, (facade[2] + h).max(3.0), Stem::Wind, scene, cfg, &mut c, gc)));
    }
    out
}
