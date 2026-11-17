//! Wind in vegetation (port of raingen/synth/vegetation.py): species spectra, the Beaufort
//! ladder and advected gusts. One mono voice per plant / stand / hedge / lawn (the engine renders
//! them in parallel and encodes them with the encoder's spread model).
//!
//! * Level ~ U^chi (Fegeant 1999; Bolin 2006/2009), level_db = SPL 10 m from one crown at a
//!   crown-height mean wind of 5 m/s; levels follow the mean wind (E[g^(2 chi)] divided out).
//! * Crowns sit higher than the ears: the listener's gust is remapped to the crown's turbulence
//!   intensity with the same timing.
//! * Conifers: log-Gaussian needle whoosh at f = 0.2 U / d_needle. Broadleaf: granular leaf
//!   ticks (crisp band rising with U^0.3, darker when wet) plus foliage swish modulated by sway.
//!   Bare branches whistle at f = 0.2 U / d (twigs 3-10 mm). Trunks creak from Beaufort 8.
//! * ISO 9613-1 absorption folded into the spectra; gusts advected (Taylor).

use super::common::*;
use super::gust::GustField;
use crate::dsp::{Biquad, BiquadCoefs};
use crate::engine::{BlockCtx, BuildCfg, Voice};
use crate::rng::Rng;
use crate::types::{season, Emission, MaterialTable, Patch, PatchKind, Scene, Shelter, Stem, Weather};
use realfft::num_complex::Complex;

#[derive(Clone, Copy)]
struct Species {
    name: &'static str,
    chi: f32,
    level_db: f32,
    tick: f32,
    tick_rate: f32,
    swish: f32,
    needle: f32,
    needle_d_mm: f32,
    whistle: f32,
    height: f32,
    tree: bool,
}

const SPECIES: [Species; 6] = [
    Species { name: "deciduous", chi: 2.0, level_db: 45.0, tick: 1.0, tick_rate: 2500.0, swish: 0.55, needle: 0.0, needle_d_mm: 1.2, whistle: 0.12, height: 10.0, tree: true },
    Species { name: "conifer", chi: 1.5, level_db: 44.0, tick: 0.04, tick_rate: 600.0, swish: 0.3, needle: 1.0, needle_d_mm: 1.2, whistle: 0.18, height: 14.0, tree: true },
    Species { name: "bush", chi: 1.8, level_db: 37.0, tick: 0.8, tick_rate: 1800.0, swish: 0.6, needle: 0.0, needle_d_mm: 1.2, whistle: 0.05, height: 1.5, tree: false },
    Species { name: "grass", chi: 1.6, level_db: 32.0, tick: 0.3, tick_rate: 3000.0, swish: 1.0, needle: 0.0, needle_d_mm: 1.2, whistle: 0.0, height: 0.4, tree: false },
    Species { name: "palm", chi: 2.0, level_db: 44.0, tick: 0.6, tick_rate: 900.0, swish: 0.7, needle: 0.0, needle_d_mm: 1.2, whistle: 0.05, height: 9.0, tree: true },
    Species { name: "reeds", chi: 1.7, level_db: 34.0, tick: 0.5, tick_rate: 2000.0, swish: 0.9, needle: 0.4, needle_d_mm: 5.0, whistle: 0.0, height: 1.8, tree: false },
];

const ALIASES: &[(&str, &str)] = &[
    ("tree", "deciduous"), ("broadleaf", "deciduous"), ("oak", "deciduous"), ("beech", "deciduous"),
    ("birch", "deciduous"), ("maple", "deciduous"), ("poplar", "deciduous"), ("willow", "deciduous"),
    ("pine", "conifer"), ("spruce", "conifer"), ("fir", "conifer"), ("needles", "conifer"), ("cedar", "conifer"),
    ("hedge", "bush"), ("shrub", "bush"), ("leaves", "deciduous"), ("meadow", "grass"), ("bamboo", "reeds"),
];

const LEVEL_REF_M: f32 = 10.0;
const LEVEL_REF_MPS: f32 = 5.0;
const TICK_TAIL: usize = 512;

fn tick_m2() -> f32 {
    (2.0f32 * 0.7 * 0.7).exp()
}

fn species_index(name: &str) -> Option<usize> {
    SPECIES.iter().position(|s| s.name == name)
}

fn alias_of(name: &str) -> Option<&'static str> {
    ALIASES.iter().find(|(a, _)| *a == name).map(|(_, t)| *t)
}

pub fn species_of(patch: &Patch) -> &'static str {
    for key in ["species", "type", "kind"] {
        if let Some(v) = patch.params.get(key) {
            let sp = match v {
                serde_json::Value::String(s) => s.to_ascii_lowercase(),
                serde_json::Value::Null => String::new(),
                other => other.to_string().to_ascii_lowercase(),
            };
            if let Some(i) = species_index(&sp) {
                return SPECIES[i].name;
            }
            if let Some(a) = alias_of(&sp) {
                return a;
            }
        }
    }
    let mat = patch.material.to_ascii_lowercase();
    if let Some(i) = species_index(&mat) {
        return SPECIES[i].name;
    }
    alias_of(&mat).unwrap_or("deciduous")
}

fn leaf_factor(species: &str, s: u8) -> f32 {
    match species {
        "deciduous" => match s {
            season::SPRING => 0.8,
            season::SUMMER => 1.0,
            season::AUTUMN => 0.7,
            season::WINTER => 0.05,
            _ => 1.0,
        },
        "grass" | "reeds" => match s {
            season::WINTER => 0.5,
            season::AUTUMN => 0.85,
            _ => 1.0,
        },
        "bush" => match s {
            season::WINTER => 0.3,
            season::AUTUMN => 0.75,
            _ => 1.0,
        },
        _ => 1.0,
    }
}

fn wet_lookup(w: &Weather, name: &str) -> Option<f32> {
    w.wetness.binary_search_by(|(k, _)| k.as_str().cmp(name)).ok().map(|i| w.wetness[i].1)
}

/// Complex response of a biquad at z^-1 = e^{-j w}.
#[inline]
fn biquad_resp(c: &BiquadCoefs, f: f32, sr: f32) -> Complex<f32> {
    let w = 2.0 * std::f64::consts::PI * f as f64 / sr as f64;
    let (s1, c1) = w.sin_cos();
    let (s2, c2) = (2.0 * w).sin_cos();
    let num = Complex::new(c.b0 + c.b1 * c1 + c.b2 * c2, -(c.b1 * s1 + c.b2 * s2));
    let den = Complex::new(1.0 + c.a1 * c1 + c.a2 * c2, -(c.a1 * s1 + c.a2 * s2));
    let h = num / den;
    Complex::new(h.re as f32, h.im as f32)
}

pub struct VegVoice {
    name: String,
    sr: f32,
    n: usize,
    field: GustField,
    sp: Species,
    level: f32,
    tick_w: f32,
    tick_rate: f32,
    swish_w: f32,
    needle_w: f32,
    whistle_w: f32,
    grassy: bool,
    exposure: f32,
    level_scale: f32,
    count: f32,
    f_crisp0: f32,
    f_body: f32,
    sway_f: f32,
    sway_phase: f32,
    d_needle: f32,
    twig_d: [f32; 3],
    pos: [f32; 3],
    dist: f32,
    az: f32,
    el: f32,
    spread: f32,
    u_ratio: f32,
    ti_ratio: f32,
    material: String,
    key: u64,
    // state
    whistle_jit: [f32; 3],
    g_prev: f32,
    flutter: f32,
    air_key: (i32, i32),
    air_noise: Vec<f32>,
    air_tick: Vec<f32>,
    fc_crisp: f32,
    body: Vec<Complex<f32>>,
    kernel: Vec<Complex<f32>>,
    noise: ShapedNoise,
    mag2: Vec<f32>,
    b: Vec<f32>,
    y: Vec<f32>,
    n_tick: usize,
    f_tick: Vec<f32>,
    w_tick: Vec<f32>,
    tick_fft: Fft,
    tick_time: Vec<f32>,
    tick_spec: Vec<Complex<f32>>,
    carry: Vec<f32>,
    carry_active: bool,
    /// Leaf contacts already scheduled past this block (absolute sample, amplitude): a fluttering
    /// leaf knocks its neighbours several times in a row, so contacts come in clusters.
    pend: Vec<(u64, f32)>,
    creaks: EventRing,
    creak_imp: Vec<f32>,
    creak_y: Vec<f32>,
}

impl VegVoice {
    #[allow(clippy::too_many_arguments)]
    pub fn new(patch: &Patch, species: &str, scene: &Scene, cfg: &BuildCfg, level_scale: f32) -> VegVoice {
        let sr = cfg.sr as f32;
        let n = cfg.block;
        let sp = SPECIES[species_index(species).unwrap_or(0)];
        let mut crng = Rng::from_key(cfg.rng.key(&format!("veg.cfg.{}", patch.id)), 0);
        let count = patch.param_f32("count", 1.0).max(1.0).floor();
        let height = patch.param_f32("height_m", sp.height);
        let exposure = patch.param_f32("wind_exposure", 1.0);
        let level_scale = level_scale * patch.param_f32("level", 1.0);
        let s_code = scene.weather.season;
        let mut leaf = leaf_factor(sp.name, s_code);
        if patch.params.get("leaf_fraction").is_some() && sp.name != "grass" && sp.name != "reeds" {
            leaf = patch.param_f32("leaf_fraction", leaf).clamp(0.02, 1.0);
        }
        let autumn_crisp = if sp.name == "deciduous" && s_code == season::AUTUMN { 1.25 } else { 1.0 };
        let f_crisp0 = crng.range_f32(3200.0, 5200.0) * autumn_crisp;
        let f_body = crng.range_f32(1100.0, 1900.0);
        let sway_f = if sp.tree { crng.range_f32(0.18, 0.5) } else { crng.range_f32(0.6, 1.6) };
        let sway_phase = crng.range_f32(0.0, 2.0 * std::f32::consts::PI);
        let d_needle = patch.param_f32("needle_d_mm", sp.needle_d_mm) / 1000.0;
        let twig0 = patch.param_f32("twig_diameter_m", 0.004);
        let mut twig_d = [0.75 * twig0, 1.25 * twig0, 2.0 * twig0];
        for t in twig_d.iter_mut() {
            *t *= crng.range_f32(0.85, 1.2);
        }
        let area = patch.area_m2.max(1.0);
        let radius = patch.param_f32("crown_radius_m", patch.param_f32("canopy_radius_m", (area / std::f32::consts::PI).sqrt()));
        let dist = patch.distance();
        let spread = match patch.params.get("spread") {
            Some(v) if v.is_number() => patch.param_f32("spread", 0.5),
            _ => (radius / dist).clamp(0.05, 0.95),
        };
        let r_eff = dist.hypot(0.5 * radius);
        let level = db_spl(sp.level_db) * count.sqrt() * level_scale * LEVEL_REF_M / r_eff;
        let z0 = scene_z0(scene);
        let z_crown = (0.7 * height).max(0.5);
        let u_ratio = wind_at_height(1.0, z_crown, z0);
        let ti_ear = turbulence_intensity_for(z0, scene.listener.height_m.max(0.3));
        let ti_ratio = turbulence_intensity_for(z0, z_crown) / ti_ear.max(1e-6);
        let noise = ShapedNoise::new(n, sr);
        let bins = noise.bins();
        let n_tick = (n + TICK_TAIL).next_power_of_two();
        let tick_fft = Fft::new(n_tick);
        let tb = tick_fft.bins();
        let creak_len = ((1.4 + 0.25) * sr) as usize + 16;
        VegVoice {
            name: format!("vegetation.{}", patch.id),
            sr,
            n,
            field: GustField::new(scene, cfg.seed, cfg.sr, cfg.block),
            sp,
            level,
            tick_w: sp.tick * leaf,
            tick_rate: sp.tick_rate * leaf.max(0.02) * count.sqrt(),
            swish_w: sp.swish * (0.25 + 0.75 * leaf),
            needle_w: sp.needle,
            whistle_w: sp.whistle * (1.0 + 3.0 * (1.0 - leaf)),
            grassy: sp.name == "grass" || sp.name == "reeds",
            exposure,
            level_scale,
            count,
            f_crisp0,
            f_body,
            sway_f,
            sway_phase,
            d_needle: d_needle.max(1e-4),
            twig_d,
            pos: patch.position,
            dist,
            az: patch.azimuth(),
            el: patch.elevation(),
            spread,
            u_ratio,
            ti_ratio,
            material: patch.material.clone(),
            key: cfg.rng.key(&format!("veg.{}", patch.id)),
            whistle_jit: [0.0; 3],
            g_prev: 0.0,
            flutter: 0.0,
            air_key: (i32::MIN, i32::MIN),
            air_noise: vec![1.0; bins],
            air_tick: vec![1.0; tb],
            fc_crisp: 0.0,
            body: vec![Complex::new(0.0, 0.0); tb],
            kernel: vec![Complex::new(0.0, 0.0); tb],
            mag2: vec![0.0; bins],
            b: vec![0.0; bins],
            y: vec![0.0; n],
            n_tick,
            f_tick: rfft_freqs(n_tick, sr),
            w_tick: rfft_weights(n_tick),
            tick_time: vec![0.0; n_tick],
            tick_spec: vec![Complex::new(0.0, 0.0); tb],
            tick_fft,
            carry: vec![0.0; n_tick - n],
            carry_active: false,
            pend: Vec::with_capacity(8192),
            creaks: EventRing::new(1, creak_len + 2 * n),
            creak_imp: vec![0.0; creak_len],
            creak_y: vec![0.0; creak_len],
            noise,
        }
    }

    fn air(&mut self, temp_c: f32, rh: f32) {
        let key = (temp_c.round() as i32, (rh / 5.0).round() as i32);
        if key == self.air_key {
            return;
        }
        self.air_key = key;
        let iso = Iso9613::new(temp_c, rh);
        let d = if self.dist > 3.0 { self.dist as f64 } else { 0.0 };
        for (a, f) in self.air_noise.iter_mut().zip(&self.noise.freqs) {
            *a = 10f64.powf(-iso.alpha(*f as f64) * d / 20.0) as f32;
        }
        for (a, f) in self.air_tick.iter_mut().zip(&self.f_tick) {
            *a = 10f64.powf(-iso.alpha(*f as f64) * d / 20.0) as f32;
        }
        self.fc_crisp = 0.0;
    }

    fn tick_kernel(&mut self, f_crisp: f32) {
        let sr = self.sr;
        if self.fc_crisp == 0.0 {
            let cb = bp(self.f_body, 1.8, sr);
            for (b, f) in self.body.iter_mut().zip(&self.f_tick) {
                *b = biquad_resp(&cb, *f, sr) * 0.5;
            }
        } else if (f_crisp - self.fc_crisp).abs() <= 0.05 * self.fc_crisp.max(1.0) {
            return;
        }
        let cc = bp(f_crisp, 1.4, sr);
        let mut p = 0.0f32;
        for i in 0..self.kernel.len() {
            let k = biquad_resp(&cc, self.f_tick[i], sr) * 0.7 + self.body[i];
            self.kernel[i] = k;
            p += k.norm_sqr() * self.w_tick[i];
        }
        let g = 1.0 / p.max(1e-24).sqrt();
        for (k, a) in self.kernel.iter_mut().zip(&self.air_tick) {
            *k *= *a * g;
        }
        self.fc_crisp = f_crisp;
    }

    /// Trunk / branch creak: a stick-slip pulse train with gliding rate through wood modes
    /// (peak-normalised, into `creak_y`); returns its length.
    fn creak(&mut self, rng: &mut Rng) -> usize {
        let sr = self.sr;
        let dur = rng.range_f32(0.3, 1.4);
        let l = (dur * sr) as usize;
        let tail = (0.25 * sr) as usize;
        let tot = (l + tail).min(self.creak_imp.len());
        let (mut f0, mut f1) = (rng.range_f32(22.0, 60.0), rng.range_f32(40.0, 120.0));
        if rng.uniform() < 0.5 {
            std::mem::swap(&mut f0, &mut f1);
        }
        let ex = rng.range_f32(0.6, 1.6);
        let imp = &mut self.creak_imp[..tot];
        imp.iter_mut().for_each(|v| *v = 0.0);
        let mut phase = 0.0f64;
        let mut prev_floor = 0.0f64;
        for i in 0..l.min(tot) {
            let t = i as f32 / sr;
            let rate = f0 + (f1 - f0) * (t / dur).powf(ex);
            phase += rate as f64 / sr as f64;
            let fl = phase.floor();
            if i > 0 && fl > prev_floor {
                let env = (std::f32::consts::PI * i as f32 / l as f32).sin().max(0.0).powf(0.7);
                imp[i] = env * (1.0 + 0.3 * rng.normal_f32());
            }
            prev_floor = fl;
        }
        let base = rng.range_f32(160.0, 480.0);
        let y = &mut self.creak_y[..tot];
        y.iter_mut().for_each(|v| *v = 0.0);
        for (ratio, gain) in [(1.0f32, 1.0f32), (1.63, 0.6), (2.41, 0.35), (3.7, 0.2)] {
            let f = base * ratio * rng.range_f32(0.97, 1.03);
            let t60 = rng.range_f32(0.04, 0.12);
            let q = std::f32::consts::PI * f * t60 / 6.91;
            let mut bq = Biquad::new(bp(f, q, sr));
            bq.process_add(imp, y, gain);
        }
        normalize_peak(y);
        tot
    }
}

impl Voice for VegVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n;
        let sr = self.sr;
        self.field.advance(ctx.index, w, n);
        if n != self.n {
            return Emission::silent(Stem::Wind);
        }
        let u10 = w.wind_u10_mps.max(0.0);
        let u_h = u10 * self.u_ratio;
        let s_up = upwind_distance(self.pos, w.wind_azimuth);
        let ti_ear = w.turbulence_intensity.max(0.0);
        let g_ear = self.field.at(s_up, u_h.max(0.5));
        let ti_c = (ti_ear * self.ti_ratio).clamp(0.0, 0.8);
        let g = remap_gust(g_ear, ti_ear, ti_c);
        let u = u_h * g * self.exposure;
        let u_bft = u10 * g * self.exposure;
        let active = u >= 0.5 || u_bft >= 1.2;
        if !active && self.g_prev <= 1e-8 && !self.noise.tail_active && !self.carry_active && !self.creaks.active() {
            return Emission::silent(Stem::Wind);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let on_leaf = smoothstep(u_bft, 1.3, 3.0);
        let mut on_twig = smoothstep(u_bft, 3.2, 6.5);
        let on_whistle = smoothstep(u_bft, 7.0, 11.0);
        if self.grassy {
            on_twig = on_leaf;
        }
        let chi = self.sp.chi;
        let s2 = (ti_c * ti_c).ln_1p();
        let u_mean = u_h * self.exposure;
        let mut a = self.level * (u_mean.max(0.0) / LEVEL_REF_MPS).powf(chi) * g.powf(chi)
            * (-0.5 * chi * (2.0 * chi - 1.0) * s2).exp();
        a *= smoothstep(u, 0.8, 2.0) * if active { 1.0 } else { 0.0 };
        // Branch / foliage flutter: twigs and branches respond to the turbulence around their own
        // natural frequencies (~0.5-3 Hz, Moore & Maguire 2004; de Langre 2008), faster than the
        // advected gusts and independent per plant, so the rustle moves at 0.5-5 Hz. Log-velocity
        // OU (tau 0.3 s, sigma FLUTTER ti_c) through the U^chi law, mean power kept.
        {
            const FLUTTER: f32 = 0.6;
            let af = (-(n as f32 / sr) / 0.3).exp();
            let mut frng = ctx.rng.stream(crate::rng::combine(self.key, 0x0F1u64), ctx.index);
            self.flutter = af * self.flutter + (1.0 - af * af).sqrt() * frng.normal_f32();
            let sg = FLUTTER * ti_c * chi;
            a *= (sg * self.flutter - sg * sg).exp();
        }
        let wet = wet_lookup(w, &self.material).or_else(|| wet_lookup(w, "leaves")).unwrap_or(0.0);

        // crown sway (slow, linear over the block)
        let t0 = ctx.t0 as f32;
        let t1 = t0 + n as f32 / sr;
        let depth = 0.25 + 0.45 * on_twig;
        let tw = 2.0 * std::f32::consts::PI * self.sway_f;
        let sw0 = 1.0 + depth * (tw * t0 + self.sway_phase).sin();
        let sw1 = 1.0 + depth * (tw * t1 + self.sway_phase).sin();

        // ---- noise layers: swish, needle whoosh, bare-twig whistles in one spectrum ----
        self.air(w.temperature_c, w.humidity_pct);
        let swish_g = self.swish_w * on_twig * (1.0 + 0.3 * wet) * a;
        let needle_g = self.needle_w * on_leaf * a;
        let aj = (-(n as f32 / sr) / 0.7).exp();
        let bj = (1.0 - aj * aj).sqrt();
        for j in self.whistle_jit.iter_mut() {
            *j = aj * *j + bj * rng.normal_f32();
        }
        let wh_on = self.whistle_w * on_whistle;
        let mut wh_g = [0f32; 3];
        for j in 0..3 {
            wh_g[j] = wh_on / 3f32.sqrt() * (0.5 + 0.5 * self.whistle_jit[j].abs()) * a;
        }
        let mut have = false;
        self.mag2.iter_mut().for_each(|v| *v = 0.0);
        {
            let freqs = &self.noise.freqs;
            let add_term = |gain: f32, fc: f32, q: f32, squared: bool, b: &mut Vec<f32>, mag2: &mut Vec<f32>, wgt: &[f32]| {
                let mut p = 0.0f32;
                for k in 0..freqs.len() {
                    let mut v = bandpass_mag2(freqs[k], fc, q);
                    if squared {
                        v *= v;
                    }
                    b[k] = v;
                    p += v * wgt[k];
                }
                let s = gain * gain / p.max(1e-30);
                for k in 0..freqs.len() {
                    mag2[k] += s * b[k];
                }
            };
            let wgt: &[f32] = &self.noise.wgt;
            if swish_g > 0.0 {
                add_term(swish_g, 250.0 + 90.0 * u, 0.7, false, &mut self.b, &mut self.mag2, wgt);
                have = true;
            }
            if needle_g > 0.0 {
                let f_n = (0.2 * u / self.d_needle).clamp(150.0, 12000.0);
                add_term(needle_g, f_n, 1.0, true, &mut self.b, &mut self.mag2, wgt);
                have = true;
            }
            for j in 0..3 {
                if wh_g[j] > 0.0 {
                    let f_j = (0.2 * u / self.twig_d[j] * (1.0 + 0.03 * self.whistle_jit[j])).clamp(60.0, 9000.0);
                    add_term(wh_g[j], f_j, 15.0, false, &mut self.b, &mut self.mag2, wgt);
                    have = true;
                }
            }
        }
        let had_tail = self.noise.tail_active;
        if have {
            for (m, aa) in self.mag2.iter_mut().zip(&self.air_noise) {
                *m = m.sqrt() * *aa;
            }
            let mag = std::mem::take(&mut self.mag2);
            self.noise.process(&mut rng, Some(&mag), &mut self.y);
            self.mag2 = mag;
        } else {
            self.noise.process(&mut rng, None, &mut self.y);
        }
        if have || had_tail {
            let m0 = 1.0 + 0.75 * (sw0 - 1.0);
            let m1 = 1.0 + 0.75 * (sw1 - 1.0);
            apply_ramp(&mut self.y, m0, m1);
        }

        // ---- leaf grains: sparse signed impulses x sway, FFT convolution with carried tails ----
        let rate = (self.tick_rate * (u.max(0.1) / 5.0).powf(1.3)).min(15000.0) * on_leaf * if active { 1.0 } else { 0.0 };
        let tick_g = self.tick_w * on_leaf * (1.0 - 0.4 * wet) * a;
        // Leaf rustle is many discrete leaf-on-leaf contacts, not filtered noise. A fluttering leaf
        // (flutter 5-40 Hz for broad leaves; Vogel 1989, J. Exp. Bot. 40; Niklas 1992 'Plant
        // Biomechanics') strikes its neighbours on successive cycles: contacts form Neyman-Scott
        // clusters of ~1-7 knocks spaced 15-60 ms with decaying strength, and the cluster rate
        // follows the local gust (rate ~ U^1.3 above). Same mean contact rate as a Poisson stream.
        const M_BAR: f64 = 3.5;
        const DECAY: f32 = 0.72;
        let e_a2 = {
            // mean squared cluster-member weight for m ~ 1 + Poisson(M_BAR - 1)
            let mut acc = 0.0f64;
            let mut w = 1.0f64;
            for j in 0..12 {
                let p_reach = if j == 0 { 1.0 } else { 1.0 - poisson_cdf(M_BAR - 1.0, j - 1) };
                acc += p_reach * w;
                w *= (DECAY * DECAY) as f64;
            }
            (acc / M_BAR) as f32
        };
        let start_s = (ctx.t0 * ctx.sr as f64).round() as u64;
        let lam = rate as f64 / M_BAR * n as f64 / sr as f64;
        let n_clusters = if lam > 0.0 { rng.poisson(lam) as usize } else { 0 };
        let tick_norm = if rate > 0.0 { tick_g / (rate.max(1.0) / sr * tick_m2() * e_a2).sqrt() } else { 0.0 };
        for _ in 0..n_clusters.min(8000) {
            let p = rng.below(n as u64);
            let m = 1 + rng.poisson(M_BAR - 1.0) as usize;
            let mut t = start_s + p;
            let mut wgt = 1.0f32;
            let a0 = rng.lognormal(0.0, 0.6) as f32 * tick_norm;
            for _ in 0..m.min(10) {
                if self.pend.len() < self.pend.capacity() {
                    let sgn = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
                    self.pend.push((t, sgn * a0 * wgt * rng.lognormal(0.0, 0.25) as f32));
                }
                t += (rng.range(0.015, 0.06) * sr as f64) as u64;
                wgt *= DECAY;
            }
        }
        let end_s = start_s + n as u64;
        let count = self.pend.iter().filter(|e| e.0 < end_s).count();
        if count > 0 || self.carry_active {
            let f_crisp = self.f_crisp0 * (1.0 - 0.35 * wet) * (u.max(0.1) / 5.0).powf(0.3).clamp(0.6, 1.6);
            self.tick_kernel(f_crisp);
            let nt = self.n_tick;
            let tt = &mut self.tick_time;
            tt.iter_mut().for_each(|v| *v = 0.0);
            if count > 0 {
                let mut keep = 0usize;
                for i in 0..self.pend.len() {
                    let (t, amp) = self.pend[i];
                    if t < end_s {
                        let p = (t.saturating_sub(start_s) as usize).min(n - 1);
                        let sway_at = sw0 + (sw1 - sw0) * (p + 1) as f32 / n as f32;
                        tt[p] += amp * sway_at;
                    } else {
                        self.pend[keep] = (t, amp);
                        keep += 1;
                    }
                }
                self.pend.truncate(keep);
                self.tick_fft.forward(tt, &mut self.tick_spec);
                for (s, k) in self.tick_spec.iter_mut().zip(&self.kernel) {
                    *s *= *k;
                }
                self.tick_fft.inverse(&mut self.tick_spec, tt);
                let k = 1.0 / nt as f32;
                tt.iter_mut().for_each(|v| *v *= k);
            }
            if self.carry_active {
                for (t, c) in tt.iter_mut().zip(&self.carry) {
                    *t += *c;
                }
            }
            for (yv, t) in self.y.iter_mut().zip(tt.iter()) {
                *yv += *t;
            }
            let mut live = false;
            for (c, t) in self.carry.iter_mut().zip(&tt[n..]) {
                *c = *t;
                live |= t.abs() > 1e-12;
            }
            self.carry_active = live;
            if !live {
                self.carry.iter_mut().for_each(|v| *v = 0.0);
            }
        }

        // ---- creaks from Beaufort 8 (trees only) ----
        let start = (ctx.t0 * ctx.sr as f64).round() as u64;
        if self.sp.tree && u_bft > 13.5 {
            let c_rate = 0.05 * self.count * smoothstep(u_bft, 14.5, 23.0);
            let lam_c = c_rate as f64 * n as f64 / sr as f64;
            let k = if lam_c > 0.0 { rng.poisson(lam_c) as usize } else { 0 };
            for _ in 0..k.min(4) {
                let off = rng.below(n as u64);
                let amp_c = db_spl(72.0) * (u_bft / 20.0) * self.level_scale / self.dist.max(1.0);
                let l = self.creak(&mut rng);
                let sig = std::mem::take(&mut self.creak_y);
                self.creaks.add(start + off, 0, &sig[..l], amp_c);
                self.creak_y = sig;
            }
        }
        out[..n].copy_from_slice(&self.y);
        if self.creaks.active() {
            self.creaks.take_add(start, n, out);
        }
        self.g_prev = a;
        Emission::point(self.az, self.el, self.dist, self.spread, Stem::Wind)
    }

    fn reset(&mut self) {
        self.field.reset();
        self.whistle_jit = [0.0; 3];
        self.g_prev = 0.0;
        self.flutter = 0.0;
        self.noise.reset();
        self.carry.iter_mut().for_each(|v| *v = 0.0);
        self.carry_active = false;
        self.pend.clear();
        self.creaks.reset();
        self.fc_crisp = 0.0;
        self.air_key = (i32::MIN, i32::MIN);
    }
}

/// P(X <= k) for X ~ Poisson(lam).
fn poisson_cdf(lam: f64, k: usize) -> f64 {
    let mut term = (-lam).exp();
    let mut acc = term;
    for i in 1..=k {
        term *= lam / i as f64;
        acc += term;
    }
    acc
}

/// Every vegetation voice of the layout, plus a ground grass bed from the surface mix.
pub fn build(scene: &Scene, patches: &[Patch], _materials: &MaterialTable, cfg: &BuildCfg) -> Vec<VegVoice> {
    let mut out = Vec::new();
    let mut has_grass = false;
    for p in patches {
        if p.kind != PatchKind::Vegetation {
            continue;
        }
        let sp = species_of(p);
        has_grass |= sp == "grass";
        out.push(VegVoice::new(p, sp, scene, cfg, 1.0));
    }
    let mut total = 0.0f32;
    let mut grass = 0.0f32;
    if let serde_json::Value::Object(m) = &scene.place.surfaces {
        for (k, v) in m {
            let x = v.as_f64().unwrap_or(0.0) as f32;
            total += x;
            if k == "grass" {
                grass = x;
            }
        }
    }
    if total == 0.0 {
        total = 1.0;
    }
    let share = grass / total;
    let enclosed = matches!(scene.place.shelter, Shelter::RoomClosed | Shelter::Car | Shelter::Attic);
    if !has_grass && share >= 0.1 && !enclosed {
        let bed = Patch {
            id: "grass_bed".into(),
            kind: PatchKind::Vegetation,
            material: "grass".into(),
            position: [0.3, 0.0, -scene.listener.height_m + 0.2],
            area_m2: 60.0,
            params: serde_json::json!({"species": "grass", "spread": 1.0, "height_m": 0.4}),
            ..Patch::default()
        };
        out.push(VegVoice::new(&bed, "grass", scene, cfg, share.sqrt()));
    }
    out
}
