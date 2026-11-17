//! Wind bed and microphone wind rumble (port of raingen/synth/wind.py).
//!
//! * The free-field air-movement bed: turbulence scattering around the head and near obstacles
//!   radiates as compact aerodynamic dipoles (Curle 1955): p ~ U^3 (56 dB SPL at a mean ear-height
//!   wind of 10 m/s) with a spectral corner rising with U; 8 directions on the SH bus, biased to the
//!   windward side, each hearing the advected gust a few metres away (gusts sweep across the bed).
//!   Levels are calibrated on the mean wind (E[(g e^{flutter})^6] divided out).
//! * Low-frequency pseudo-sound ~ U^2, omni.
//! * Loose debris from Beaufort 8 (17.2 m/s).
//! * Gust onsets exported as 'gust' markers (gust crosses 1 + 1.6 ti, hysteresis at 1 + 0.6 ti).
//! * Microphone wind noise (optional): -6.7 dB/decade below fc = V/(3D), -26.7 above (Strasberg
//!   1988, Wuttke 1992), level ~ U^2 minus the windscreen attenuation.

use super::common::*;
use super::gust::GustField;
use crate::dsp::{Biquad, BiquadCoefs, Sos};
use crate::engine::{BlockCtx, BuildCfg, Voice};
use crate::types::{Emission, Scene, Shelter, Stem, N_AMBI};

const BED_REF_DB: f32 = 56.0;
const BED_EXPONENT: f32 = 3.0;
const RUMBLE_REF_DB: f32 = 46.0;
const BED_ADVECT_M: f32 = 3.0;
const BED_FLUTTER: f32 = 0.15;
/// Small eddies at ear height: the Kaimal spectrum peaks near f z / U ~ 0.05 (Kaimal et al. 1972,
/// QJRMS 98), so at z ~ 1.6 m and U ~ 10 m/s much of the velocity variance sits at 0.1-2 Hz, faster
/// than the advected gusts. Each bed direction gets its own eddy process (log-velocity sigma
/// BED_EDDY * ti, time scale z / (2 pi 0.05 U)): the bed moves around the head at 0.5-5 Hz.
const BED_EDDY: f32 = 0.35;
const EDDY_Z_M: f32 = 1.6;
const N_DIR: usize = 8;
const DEBRIS_MAX_S: f32 = 0.64;

fn bed_amplitude(u: f32) -> f32 {
    if u <= 0.0 {
        return 0.0;
    }
    db_spl(BED_REF_DB) * (u / 10.0).powf(BED_EXPONENT) * smoothstep(u, 0.3, 1.5)
}

pub struct WindVoice {
    name: String,
    sr: f32,
    field: GustField,
    dir_az: [f32; N_DIR],
    y: [[f32; N_AMBI]; N_DIR],
    shape: Vec<Sos>,
    bright: DesignTable,
    fc: [f32; N_DIR],
    norm: [f32; N_DIR],
    rumble: Biquad,
    rumble_tab: DesignTable,
    rfc: f32,
    rnorm: f32,
    bed_ramp: [Ramp; N_DIR],
    rumble_ramp: Ramp,
    turb: [f32; N_DIR],
    eddy: [f32; N_DIR],
    key_eddy: u64,
    debris: EventRing,
    gusting: bool,
    last_gust_t: f64,
    key_bed: u64,
    key_debris: u64,
    noise: Vec<f32>,
    shaped: Vec<f32>,
    burst: Vec<f32>,
    tmp: Vec<f32>,
    offs: Vec<usize>,
}

impl WindVoice {
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> WindVoice {
        let sr = cfg.sr as f32;
        let n = cfg.block;
        let mut dir_az = [0f32; N_DIR];
        let mut y = [[0f32; N_AMBI]; N_DIR];
        for k in 0..N_DIR {
            dir_az[k] = 2.0 * std::f32::consts::PI * k as f32 / N_DIR as f32;
            let el = if k % 2 == 0 { 0.25 } else { -0.1 };
            y[k] = sh_gains(dir_az[k], el, 0.3);
        }
        let mut pink = slope_sections(-10.0, 20.0, 16000.0, sr);
        pink.push(hp2(18.0, sr));
        let bright = DesignTable::new(&pink, 100.0, 5200.0, 1.02, |f| lp2(f, sr));
        let (c0, norm) = bright.get(500.0);
        let mut chain = pink.clone();
        chain.push(c0);
        let shape = (0..N_DIR).map(|_| Sos::new(&chain)).collect();
        let rumble_tab = DesignTable::new(&[], 30.0, 2000.0, 1.03, |f| lp2(f, sr));
        let (rc, rnorm) = rumble_tab.get(60.0);
        let debris_cap = ((DEBRIS_MAX_S + 0.05) * sr) as usize + 2 * n;
        WindVoice {
            name: "wind".into(),
            sr,
            field: GustField::new(scene, cfg.seed, cfg.sr, cfg.block),
            dir_az,
            y,
            shape,
            bright,
            fc: [500.0; N_DIR],
            norm: [norm; N_DIR],
            rumble: Biquad::new(rc),
            rumble_tab,
            rfc: 60.0,
            rnorm,
            bed_ramp: [Ramp::default(); N_DIR],
            rumble_ramp: Ramp::default(),
            turb: [0.0; N_DIR],
            eddy: [0.0; N_DIR],
            key_eddy: cfg.rng.key("wind.bed.eddy"),
            debris: EventRing::new(N_AMBI, debris_cap),
            gusting: false,
            last_gust_t: -1e9,
            key_bed: cfg.rng.key("wind.bed"),
            key_debris: cfg.rng.key("wind.debris"),
            noise: vec![0.0; n],
            shaped: vec![0.0; N_DIR * n],
            burst: vec![0.0; ((DEBRIS_MAX_S + 0.04) * sr) as usize + 8],
            tmp: vec![0.0; ((DEBRIS_MAX_S + 0.04) * sr) as usize + 8],
            offs: Vec::with_capacity(64),
        }
    }

    /// Brightness of bed direction k. Each direction follows its own local gust speed: the dipole
    /// spectrum of turbulence scattered by the head and nearby obstacles peaks at f ~ St U / d
    /// (Curle 1955; Strouhal scaling), so a gust passing one side is a brighter whoosh there.
    fn set_brightness(&mut self, k: usize, fc: f32) {
        if self.fc[k] > 0.0 && (fc - self.fc[k]).abs() / self.fc[k] < 0.03 {
            return;
        }
        let (c, norm) = self.bright.get(fc);
        self.fc[k] = fc;
        self.norm[k] = norm;
        let s = &mut self.shape[k];
        let last = s.stages.len() - 1;
        s.stages[last].set(c);
    }

    fn gust_marker(&mut self, ctx: &BlockCtx, u: f32, gust: f32, ti: f32) {
        let on = 1.0 + 1.6 * ti;
        let off = 1.0 + 0.6 * ti;
        if !self.gusting && gust >= on && u >= 4.0 && ctx.t0 - self.last_gust_t >= 4.0 {
            let w = ctx.weather;
            let u10g = w.wind_u10_mps * gust;
            let data = format!(
                "{{\"wind_mps\": {}, \"gust\": {}, \"wind_10m_mps\": {}, \"beaufort\": {}, \"from_deg\": {}, \"azimuth\": {}}}",
                jnum(u as f64, 2),
                jnum(gust as f64, 3),
                jnum(u10g as f64, 2),
                jnum(mps_to_beaufort(u10g) as f64, 1),
                jnum(w.wind_from_deg as f64, 1),
                jnum(w.wind_azimuth as f64, 4)
            );
            ctx.marker(ctx.t0, "gust", data);
            self.gusting = true;
            self.last_gust_t = ctx.t0;
        } else if self.gusting && gust <= off {
            self.gusting = false;
        }
    }

    /// Skittering twigs / leaves / litter: a cluster of band-limited ticks plus a body knock
    /// (peak-normalised, into `self.burst`); returns its length.
    fn debris_burst(&mut self, rng: &mut crate::rng::Rng) -> usize {
        let sr = self.sr;
        let dur = rng.range_f32(0.08, 0.6);
        let n_body = (dur * sr) as usize;
        let l = (n_body + (0.03 * sr) as usize).min(self.burst.len());
        let imp = &mut self.tmp[..l];
        imp.iter_mut().for_each(|v| *v = 0.0);
        let k = rng.int_range(3, 27) as usize;
        for _ in 0..k {
            let off = rng.below(n_body.max(1) as u64) as usize;
            let a = rng.lognormal(0.0, 0.6) as f32 * if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
            if off < l {
                imp[off] += a;
            }
        }
        let mut crisp = Biquad::new(bp(rng.range_f32(1500.0, 5500.0), 2.0, sr));
        let mut body = Biquad::new(bp(rng.range_f32(250.0, 800.0), 4.0, sr));
        let out = &mut self.burst[..l];
        for i in 0..l {
            out[i] = crisp.process(imp[i]) + 0.6 * body.process(imp[i]);
        }
        normalize_peak(out);
        l
    }
}

impl Voice for WindVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n;
        let sr = self.sr;
        self.field.advance(ctx.index, w, n);
        let u = w.wind_listener_mps.max(0.0);
        let gust = w.gust.max(0.05);
        let u_mean = u / gust;
        let start = (ctx.t0 * ctx.sr as f64).round() as u64;
        let ti = w.turbulence_intensity.max(0.05);
        self.gust_marker(ctx, u, gust, ti);
        let bed_max = self.bed_ramp.iter().fold(0.0f32, |m, r| m.max(r.v));
        if u < 0.2 && !self.debris.active() && bed_max < 1e-7 {
            return Emission::silent(Stem::Wind);
        }
        if self.noise.len() < n {
            return Emission::silent(Stem::Wind);
        }
        let mut rng = ctx.rng.stream(self.key_bed, ctx.index);

        // Ornstein-Uhlenbeck turbulence per direction (control rate, tau ~ 2 s)
        let a = (-(n as f32 / sr) / 2.0).exp();
        let b = (1.0 - a * a).sqrt();
        for t in self.turb.iter_mut() {
            *t = a * *t + b * rng.normal_f32();
        }
        {
            let tau = (EDDY_Z_M / (2.0 * std::f32::consts::PI * 0.05 * u_mean.max(1.0))).clamp(0.12, 2.0);
            let ae = (-(n as f32 / sr) / tau).exp();
            let be = (1.0 - ae * ae).sqrt();
            let mut erng = ctx.rng.stream(self.key_eddy, ctx.index);
            for e in self.eddy.iter_mut() {
                *e = ae * *e + be * erng.normal_f32();
            }
        }
        let mut weight = [0f32; N_DIR];
        let mut rel = [0f32; N_DIR];
        let mut wsum = 0.0;
        for k in 0..N_DIR {
            rel[k] = (self.dir_az[k] - w.wind_azimuth).cos();
            let q = (1.0 + rel[k]) / 2.0;
            weight[k] = 0.35 + 0.65 * q * q;
            wsum += weight[k] * weight[k];
        }
        let wn = 1.0 / wsum.sqrt();
        let s2 = (ti * ti).ln_1p();
        let normc = (0.5 * BED_EXPONENT * (BED_EXPONENT - 1.0) * s2
            + 0.5 * (BED_FLUTTER * BED_EXPONENT * ti).powi(2)
            + 0.5 * (BED_EDDY * BED_EXPONENT * ti).powi(2))
        .exp();
        let base = bed_amplitude(u_mean) / normc.sqrt();
        for k in 0..N_DIR {
            let g_dir = self.field.at(BED_ADVECT_M * rel[k], u_mean)
                * (BED_FLUTTER * ti * self.turb[k] + BED_EDDY * ti * self.eddy[k]).exp();
            // local speed u_mean g_dir; the corner rises linearly with it (gust whoosh)
            self.set_brightness(k, (120.0 + 75.0 * u_mean * g_dir).clamp(100.0, 5000.0));
            let inv_norm = 1.0 / self.norm[k];
            let amp = base * g_dir.powf(BED_EXPONENT) * weight[k] * wn * inv_norm;
            let (g0, g1) = self.bed_ramp[k].to(amp);
            let sh = &mut self.shaped[k * n..(k + 1) * n];
            fill_white(&mut rng, sh);
            self.shape[k].process_block(sh);
            apply_ramp(sh, g0, g1);
        }
        // bus = Y @ shaped
        for k in 0..N_DIR {
            let sh = &self.shaped[k * n..(k + 1) * n];
            for c in 0..N_AMBI {
                let g = self.y[k][c];
                if g == 0.0 {
                    continue;
                }
                let o = &mut out[c * n..(c + 1) * n];
                for (ov, sv) in o.iter_mut().zip(sh) {
                    *ov += g * *sv;
                }
            }
        }

        // low broadband air movement, omni, ~U^2
        let rfc = 40.0 + 6.0 * u;
        if (rfc - self.rfc).abs() / self.rfc > 0.05 {
            self.rfc = rfc;
            let (c, nn) = self.rumble_tab.get(rfc);
            self.rumble.set(c);
            self.rnorm = nn;
        }
        let r_amp = db_spl(RUMBLE_REF_DB) * (u_mean / 10.0).powi(2) * smoothstep(u_mean, 0.5, 3.0) * gust * gust
            / gust_power_mean(4.0, ti).sqrt();
        let (r0, r1) = self.rumble_ramp.to(r_amp / self.rnorm);
        {
            let nb = &mut self.noise[..n];
            fill_white(&mut rng, nb);
            self.rumble.process_block(nb);
            let d = (r1 - r0) / n as f32;
            for i in 0..n {
                out[i] += nb[i] * (r0 + d * (i + 1) as f32);
            }
        }

        // debris from Beaufort 8
        if u > 15.0 {
            let mut drng = ctx.rng.stream(self.key_debris, ctx.index);
            let rate = 0.8 * smoothstep(u, 15.0, 30.0) * (u / 17.2);
            let mut offs = std::mem::take(&mut self.offs);
            poisson_offsets(&mut drng, rate, n, sr, &mut offs);
            for &off in offs.iter() {
                let l = self.debris_burst(&mut drng);
                let dist = drng.range_f32(2.0, 15.0);
                let az = w.wind_azimuth + std::f32::consts::PI + drng.normal_f32();
                let amp = db_spl(68.0) * (u / 20.0).powi(2) / dist;
                let y = sh_gains(az, -(1.6f32).atan2(dist), 0.2);
                self.debris.add_gains(start + off as u64, &y, &self.burst[..l], amp);
            }
            self.offs = offs;
        }
        if self.debris.active() {
            self.debris.take_add(start, n, out);
        } else {
            self.debris.take_add(start, 0, out);
        }
        Emission::diffuse(Stem::Wind)
    }

    fn reset(&mut self) {
        self.field.reset();
        for s in &mut self.shape {
            s.reset();
        }
        self.rumble.reset();
        self.bed_ramp = [Ramp::default(); N_DIR];
        self.rumble_ramp = Ramp::default();
        self.turb = [0.0; N_DIR];
        self.eddy = [0.0; N_DIR];
        self.debris.reset();
        self.gusting = false;
        self.last_gust_t = -1e9;
    }
}

// --------------------------------------------------------------------------
// microphone wind rumble
// --------------------------------------------------------------------------
/// (characteristic size D of the turbulent interaction region m, broadband attenuation dB)
fn windscreen(name: &str) -> (f32, f32) {
    match norm_name(name).as_str() {
        "none" => (0.021, 0.0),
        "foam" => (0.06, 12.0),
        "blimp" => (0.20, 22.0),
        "dead_cat" => (0.25, 32.0),
        "fur" => (0.08, 24.0),
        _ => (0.20, 22.0),
    }
}

/// How much wind reaches the capsule in this shelter (0 inside a car or a closed room).
pub fn mic_exposure(scene: &Scene) -> f32 {
    match scene.place.shelter {
        Shelter::Porch => 0.4,
        Shelter::Cave => 0.3,
        Shelter::Underpass => 0.7,
        Shelter::Umbrella => 0.9,
        Shelter::Car | Shelter::Tent | Shelter::RoomClosed | Shelter::Attic => 0.0,
        Shelter::RoomOpen => 0.05 * scene.place.window_open,
        _ => 1.0,
    }
}

pub struct MicRumbleVoice {
    name: String,
    sr: f32,
    d_m: f32,
    atten_db: f32,
    filt: [Sos; 3],
    tab: DesignTable,
    fc: f32,
    norm: f32,
    env_lp: Biquad,
    ramp: Ramp,
    pub exposure: f32,
    key: u64,
    x: Vec<f32>,
    env: Vec<f32>,
}

impl MicRumbleVoice {
    pub fn new(scene: &Scene, cfg: &BuildCfg) -> MicRumbleVoice {
        let sr = cfg.sr as f32;
        let (d_m, atten_db) = windscreen(&scene.weather.wind.windscreen);
        let tilt = slope_sections(-6.7, 5.0, 20000.0, sr);
        let tab = DesignTable::new(&tilt, 5.0, 2500.0, 1.03, |f| lp1(f, sr));
        let (c, norm) = tab.get(100.0);
        let mut chain = tilt.clone();
        chain.push(c);
        MicRumbleVoice {
            name: "mic_rumble".into(),
            sr,
            d_m,
            atten_db,
            filt: [Sos::new(&chain), Sos::new(&chain), Sos::new(&chain)],
            tab,
            fc: 0.0,
            norm,
            env_lp: Biquad::new(lp1(3.0, sr)),
            ramp: Ramp::default(),
            exposure: mic_exposure(scene),
            key: cfg.rng.key("wind.mic_rumble"),
            x: vec![0.0; 3 * cfg.block],
            env: vec![0.0; cfg.block],
        }
    }

    fn design(&mut self, fc: f32) {
        if self.fc > 0.0 && (fc - self.fc).abs() / self.fc < 0.05 {
            return;
        }
        let (c, norm) = self.tab.get(clip_f(fc, self.sr));
        self.fc = fc;
        self.norm = norm;
        for s in &mut self.filt {
            let last = s.stages.len() - 1;
            s.stages[last].set(c);
        }
    }
}

impl Voice for MicRumbleVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let u = ctx.weather.wind_listener_mps.max(0.0) * self.exposure;
        if (u < 0.3 && self.ramp.v < 1e-7) || self.env.len() < n {
            return Emission::silent(Stem::Wind);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        self.design(u.max(0.3) / (3.0 * self.d_m));
        let amp = db_spl(95.0 - self.atten_db) * (u / 10.0).powi(2) * smoothstep(u, 0.3, 1.5);
        let inv = 1.0 / self.norm;
        for c in 0..3 {
            let x = &mut self.x[c * n..(c + 1) * n];
            fill_white(&mut rng, x);
            self.filt[c].process_block(x);
        }
        let env = &mut self.env[..n];
        fill_white(&mut rng, env);
        self.env_lp.process_block(env);
        let k = (self.sr / (std::f32::consts::PI * 3.0)).sqrt();
        let (g0, g1) = self.ramp.to(amp);
        for i in 0..n {
            let e = (0.6 * (env[i] * k).clamp(-3.0, 3.0) - 0.18).exp();
            let g = ramp_at(g0, g1, i, n) * e * inv;
            out[i] += self.x[i] * g;
            out[n + i] += 0.8 * self.x[n + i] * g;
            out[3 * n + i] += 0.4 * self.x[2 * n + i] * g;
        }
        Emission::diffuse(Stem::Wind)
    }

    fn reset(&mut self) {
        for s in &mut self.filt {
            s.reset();
        }
        self.env_lp.reset();
        self.ramp = Ramp::default();
    }
}

#[allow(dead_code)]
fn _unused(_: BiquadCoefs) {}
