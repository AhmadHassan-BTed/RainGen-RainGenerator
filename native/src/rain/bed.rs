//! Far-field diffuse rain bed synthesized directly in the SH domain (RainBedGenerator of
//! raingen/synth/rain.py).
//!
//! 16 independent shaped noises (one per ACN channel, diffuse-field SN3D weights 1/sqrt(2n+1)),
//! with a slowly wandering low-order anisotropy (rain curtains, wind-driven rain, storm cells).
//! Power follows the class fluxes (so the DSD sets level and timbre) integrated over the annulus
//! r1..r2 beyond the near rings, the storm envelope and gusts; the spectrum is split into four
//! bands by drop class (impacts and bubbles) plus hail / ice pellets, each band tilted by ISO
//! 9613-1 air absorption at the bed's effective distance.

use super::air;
use super::events::RateModulator;
use super::hail::{ice_fall_speed, ice_mean_diameter_rel, ice_number_flux, ice_species};
use super::kernels::{bubble_damping, butter2_bandpass, db_to_pa, f_ref, force_peak, j_ref, smoothstep, Cascade, KernelBank, NoiseRow, Ola};
use super::rain::{bubble_pa_per_m, rain_gate};
use super::BuildCtx;
use crate::dsp::BiquadCoefs;
use crate::engine::{BlockCtx, Voice};
use crate::rng::Rng;
use crate::types::{bearing_to_azimuth, Emission, MaterialKind, PatchKind, Stem, Weather, N_AMBI, N_CLASSES};
use std::sync::Arc;

// The lowest band reaches down to 60 Hz: large drops (classes 3-4, 2.8-4.2 mm, ~9 m/s) carry
// 0.25-0.40 of their impact energy below 700 Hz, and on compliant ground (soil, litter, leaf
// plates with fundamental modes ~50-200 Hz) part of it radiates as the 60-120 Hz "thud" that
// gives a heavy downpour its body; with a 120 Hz edge the bed had none (-15 dB at 63 Hz).
const BANDS: [(f32, f32); 4] = [(60.0, 700.0), (700.0, 2500.0), (2500.0, 8000.0), (8000.0, 18000.0)];
const IMPACT_BANDS: [[f32; 4]; 5] = [
    [0.00, 0.10, 0.40, 0.50],
    [0.02, 0.15, 0.48, 0.35],
    [0.08, 0.32, 0.45, 0.15],
    [0.25, 0.40, 0.28, 0.07],
    [0.40, 0.35, 0.20, 0.05],
];
const BUBBLE_BANDS: [[f32; 4]; 5] = [
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.10, 0.90],
    [0.00, 0.50, 0.50, 0.00],
    [0.05, 0.45, 0.50, 0.00],
    [0.15, 0.55, 0.30, 0.00],
];
const HAIL_BANDS: [f32; 4] = [0.15, 0.35, 0.35, 0.15];
const T_EFF: f32 = 5e-4;
const REP_D: [f32; 5] = [0.6, 0.95, 1.6, 2.8, 4.2];

pub struct RainBedVoice {
    kb: Arc<KernelBank>,
    n: usize,
    sr: f32,
    heading: f32,
    area_factor: f32,
    r_air: f32,
    p_mix: f32,
    e_imp: [f32; 5],
    e_bub: [f32; 5],
    band: Vec<[BiquadCoefs; 2]>,
    band_e: [f32; 4],
    band_psd: Vec<f32>,
    band_key: Option<(i32, i32)>,
    ord_w: [f32; N_AMBI],
    v_prev: f32,
    var_prev: [f32; 4],
    aniso_prev: [f32; 3],
    rows: Vec<NoiseRow>,
    olas: Vec<Ola>,
    psd: Vec<f32>,
    wander: (RateModulator, RateModulator),
    cluster: RateModulator,
    key: u64,
}

impl RainBedVoice {
    pub fn new(bc: &mut BuildCtx) -> RainBedVoice {
        let scene = bc.scene;
        let sr = bc.sr as f32;
        let n = bc.block;
        let kb = bc.kb.clone();
        let rings: Vec<f32> = bc
            .patches
            .iter()
            .filter(|p| p.param_bool("ground", false) && p.params.get("r_outer").is_some())
            .map(|p| p.param_f32("r_outer", 0.0))
            .collect();
        let r1 = if !rings.is_empty() {
            rings.iter().cloned().fold(f32::MIN, f32::max)
        } else {
            let ground: Vec<f32> = bc
                .patches
                .iter()
                .filter(|p| {
                    matches!(p.kind, PatchKind::Surface | PatchKind::Water | PatchKind::Vegetation) && p.elevation() < 0.0 && !p.interior
                })
                .map(|p| p.distance())
                .collect();
            1.1 * if ground.is_empty() { 30.0 } else { ground.iter().cloned().fold(f32::MIN, f32::max) }
        };
        let r1 = r1.clamp(10.0, 80.0);
        let r2 = (6.0 * r1).max(300.0);
        let h = scene.listener.height_m;
        let preset = scene.place.preset.to_ascii_lowercase();
        let openness = if ["city", "street", "urban", "alley", "canyon", "town"].iter().any(|k| preset.contains(k)) {
            0.4
        } else if ["forest", "wood", "jungle"].iter().any(|k| preset.contains(k)) {
            0.5
        } else {
            0.7
        };
        let area_factor = std::f32::consts::PI * ((r2 * r2 + h * h) / (r1 * r1 + h * h)).ln() * openness;
        let r_air = 0.6 * (r1 * r2).sqrt();
        // surface mix of the place
        let mut surf: Vec<(String, f32)> = Vec::new();
        if let Some(m) = scene.place.surfaces.as_object() {
            for (k, v) in m {
                let x = v.as_f64().unwrap_or(0.0) as f32;
                if x > 0.0 {
                    surf.push((k.clone(), x));
                }
            }
        }
        surf.sort_by(|a, b| a.0.cmp(&b.0));
        if surf.is_empty() {
            surf.push(("grass".into(), 1.0));
        }
        let tot: f32 = surf.iter().map(|s| s.1).sum();
        let mut lin = 0.0f64;
        let mut wf = 0.0f32;
        for (name, pct) in &surf {
            let m = bc.mats.table().get(name);
            lin += (*pct / tot) as f64 * 10f64.powf(m.level_db as f64 / 10.0);
            if m.kind == MaterialKind::Water {
                wf += pct / tot;
            }
        }
        let p_mix = db_to_pa((10.0 * lin.max(1e-12).log10()) as f32);
        let wq = bc.water_quality;
        let mut e_imp = [0.0f32; 5];
        let mut e_bub = [0.0f32; 5];
        let a = [0.15e-3f32, 0.2e-3, 0.6e-3, 0.8e-3, 1.2e-3];
        let eps = [0.015f32, 0.045, 0.012, 0.06, 0.08];
        let p_ent = [0.0, 0.95, 0.03, 0.05 + 0.6 * smoothstep(2.8, 2.2, 4.0), 0.05 + 0.6 * smoothstep(4.2, 2.2, 4.0)];
        for c in 0..N_CLASSES {
            let vt = super::dsd::fall_speed(REP_D[c]);
            e_imp[c] = (force_peak(REP_D[c], vt) / f_ref()).powi(2) * p_mix * p_mix * T_EFF;
            let amp = bubble_pa_per_m() * eps[c] * a[c];
            e_bub[c] = p_ent[c] * amp * amp / (4.0 * bubble_damping(a[c] as f64) as f32) * wf * wq;
        }
        // band filters and their energies
        let mut band = Vec::new();
        let mut band_e = [0.0f32; 4];
        for (k, (lo, hi)) in BANDS.iter().enumerate() {
            let hi = hi.min(0.45 * sr);
            let lo = lo.min(0.5 * hi);
            let c = butter2_bandpass(lo, hi, sr);
            let mut imp = vec![0.0f64; 16384];
            imp[0] = 1.0;
            Cascade::filter_f64(&c, &mut imp);
            band_e[k] = imp.iter().map(|v| v * v).sum::<f64>() as f32;
            band.push(c);
        }
        let mut ord_w = [0.0f32; N_AMBI];
        for (k, w) in ord_w.iter_mut().enumerate() {
            let order = (k as f32).sqrt().floor();
            *w = 1.0 / (2.0 * order + 1.0).sqrt();
        }
        let nb = kb.half + 1;
        let ola_cap = n + kb.frame + 64;
        RainBedVoice {
            n,
            sr,
            heading: scene.listener.heading_deg,
            area_factor,
            r_air,
            p_mix,
            e_imp,
            e_bub,
            band,
            band_e,
            band_psd: vec![0.0; 4 * nb],
            band_key: None,
            ord_w,
            v_prev: 0.0,
            var_prev: [0.0; 4],
            aniso_prev: [0.0; 3],
            rows: (0..N_AMBI).map(|_| NoiseRow::new(&kb)).collect(),
            olas: (0..N_AMBI).map(|_| Ola::new(ola_cap)).collect(),
            psd: vec![0.0; nb],
            wander: (
                RateModulator::new(&bc.rngf, "rain/bed/wander_x", 0.6, 1.0, 30.0, sr as f64),
                RateModulator::new(&bc.rngf, "rain/bed/wander_y", 0.6, 1.0, 30.0, sr as f64),
            ),
            cluster: super::rain::cluster_modulator(bc),
            key: bc.rngf.key("rain/bed"),
            kb,
        }
    }

    /// Power spectra of the four band filters followed by ISO 9613-1 absorption at r_air.
    fn band_spectra(&mut self, temp_c: f32, rh_pct: f32) {
        let tq = temp_c.round() as i32;
        let hq = (rh_pct / 2.0).round() as i32 * 2;
        if self.band_key == Some((tq, hq)) {
            return;
        }
        self.band_key = Some((tq, hq));
        let (ac, an) = air::design(self.r_air, tq as f32, hq as f32, self.sr);
        let nb = self.kb.half + 1;
        let f_step = self.sr / self.kb.frame as f32;
        for k in 0..4 {
            for i in 0..nb {
                let f = (i as f32 * f_step).max(1e-3);
                let pb = Cascade::power_at(&self.band[k], f, self.sr);
                let pa = Cascade::power_at(&ac[..an], f, self.sr);
                self.band_psd[k * nb + i] = pb * pa;
            }
        }
    }

    fn aniso(&mut self, w: &Weather, block: u64, n: usize) -> [f32; 3] {
        let mut v = [0.0f32, 0.0, -0.2];
        v[0] += 0.12 * (self.wander.0.step(block, n) as f32).ln();
        v[1] += 0.12 * (self.wander.1.step(block, n) as f32).ln();
        let u10 = w.wind_u10_mps;
        if u10 > 0.0 {
            let k = 0.25 * (u10 / 12.0).min(1.0);
            v[0] += k * w.wind_azimuth.cos();
            v[1] += k * w.wind_azimuth.sin();
        }
        for cell in &w.storms {
            let az = bearing_to_azimuth(cell.bearing_deg, self.heading);
            let k = 0.3 * cell.rain_contribution.clamp(0.0, 1.0) / (1.0 + cell.distance_km.max(0.0) / 8.0);
            v[0] += k * az.cos();
            v[1] += k * az.sin();
        }
        let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if nrm > 0.5 {
            let s = 0.5 / nrm;
            [v[0] * s, v[1] * s, v[2] * s]
        } else {
            v
        }
    }

    fn band_power(&self, w: &Weather) -> [f32; 4] {
        let mut p = [0.0f32; 4];
        let gate = rain_gate(w.precip_type);
        if gate > 0.0 {
            let mut e_b = self.e_bub;
            e_b[1] *= 1.0 - smoothstep(w.wind_u10_mps * w.gust.max(0.0), 7.5, 10.0);
            for c in 0..N_CLASSES {
                let f = w.class_flux[c].max(0.0);
                if f <= 0.0 {
                    continue;
                }
                for b in 0..4 {
                    p[b] += gate * (f * self.e_imp[c] * IMPACT_BANDS[c][b] + f * e_b[c] * BUBBLE_BANDS[c][b]);
                }
            }
        }
        if let Some(sp) = ice_species(w) {
            if w.rain_rate_mm_h > 0.0 {
                let nflux = ice_number_flux(w.rain_rate_mm_h, &sp);
                let dm = 0.6 * sp.size * ice_mean_diameter_rel(sp.hi) / ice_mean_diameter_rel(1.6);
                let j = sp.rho * std::f32::consts::PI * (dm * 1e-3).powi(3) / 6.0 * ice_fall_speed(dm, sp.kind) * (1.0 + sp.rest);
                let e_hail = (self.p_mix * j / j_ref()).powi(2) * T_EFF * 0.3;
                for b in 0..4 {
                    p[b] += nflux * e_hail * HAIL_BANDS[b];
                }
            }
        }
        let mut env = 0.6 + 0.4 * w.intensity_envelope.clamp(0.0, 2.0);
        env *= (1.0 + 0.3 * (w.gust - 1.0)).clamp(0.5, 1.6);
        for v in p.iter_mut() {
            *v *= self.area_factor * env;
        }
        p
    }
}

impl Voice for RainBedVoice {
    fn name(&self) -> &str {
        "rain/bed"
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n.min(self.n);
        if n == 0 || out.len() < N_AMBI * n {
            return Emission::silent(Stem::RainFar);
        }
        let mut power = self.band_power(w);
        // the shared clustering, at half the log-depth: the bed averages over 10-300 m
        let cl = (self.cluster.step(ctx.index, n) as f32).sqrt();
        for p in power.iter_mut() {
            *p *= cl;
        }
        let v: f32 = power.iter().map(|x| x.max(0.0)).sum();
        // the wanderers advance every block so their state never depends on silence
        let vdir = self.aniso(w, ctx.index, n);
        if v <= 0.0 && self.v_prev <= 0.0 && !self.olas.iter().any(|o| o.active()) {
            self.aniso_prev = vdir;
            return Emission::silent(Stem::RainFar);
        }
        let mut var = [0.0f32; 4];
        let m1;
        if v <= 0.0 {
            var = self.var_prev;
            m1 = 0.0;
        } else {
            for b in 0..4 {
                var[b] = power[b].max(0.0) / self.band_e[b].max(1e-12);
            }
            m1 = 1.0;
            self.var_prev = var;
        }
        self.v_prev = v;
        if m1 > 0.0 {
            self.band_spectra(w.temperature_c, w.humidity_pct);
            let nb = self.kb.half + 1;
            self.psd.fill(0.0);
            for b in 0..4 {
                let src = &self.band_psd[b * nb..(b + 1) * nb];
                for (o, s) in self.psd.iter_mut().zip(src) {
                    *o += var[b] * m1 * *s;
                }
            }
            let mut rng = Rng::from_key(self.key, ctx.index);
            let base = rng.below(1 << 30);
            let stride = (self.kb.frame + 2) as u64;
            let kb = self.kb.clone();
            for ch in 0..N_AMBI {
                self.rows[ch].synth(&kb, &self.psd, base + ch as u64 * stride, &mut self.olas[ch]);
            }
        }
        for ch in 0..N_AMBI {
            let o = &mut out[ch * n..(ch + 1) * n];
            self.olas[ch].pop_into(o);
            let g = self.ord_w[ch];
            for v in o.iter_mut() {
                *v *= g;
            }
        }
        // low-order anisotropy, ramped across the block
        let a0 = self.aniso_prev;
        self.aniso_prev = vdir;
        let (w_ch, rest) = out.split_at_mut(n);
        for i in 0..n {
            let r = i as f32 / n as f32;
            let ws = w_ch[i];
            let vx = a0[0] + (vdir[0] - a0[0]) * r;
            let vy = a0[1] + (vdir[1] - a0[1]) * r;
            let vz = a0[2] + (vdir[2] - a0[2]) * r;
            rest[i] += ws * vy; // ACN 1 = Y (left)
            rest[n + i] += ws * vz; // ACN 2 = Z (up)
            rest[2 * n + i] += ws * vx; // ACN 3 = X (front)
        }
        Emission::diffuse(Stem::RainFar)
    }

    fn reset(&mut self) {
        for o in self.olas.iter_mut() {
            o.clear();
        }
        self.v_prev = 0.0;
        self.var_prev = [0.0; 4];
        self.aniso_prev = [0.0; 3];
        self.wander.0.reset();
        self.wander.1.reset();
        self.cluster.reset();
        self.band_key = None;
    }
}
