//! Life and towns that respond to the weather (port of raingen/synth/biophony.py): insects,
//! cicadas, birds (with owls), frogs, a distant insect chorus, and road traffic.
//!
//! * Crickets follow Dolbear's law N60 = 7.2 T_C - 32 chirps/min (field / tree crickets and
//!   carriers scale with temperature, Walker 1962); silent below ~12 C, in rain, in wind and by day.
//! * Cicadas sing by day in hot, dry weather (> ~22 C).
//! * Birds: dawn chorus around sunrise (Cooper declination), weaker dusk chorus, scattered day song,
//!   suppressed by rain and wind, rebounding after rain. Procedural, generic songs.
//! * Frogs chorus on warm, wet evenings and nights in spring and summer.
//! * Traffic: L ~ 10 log10(flow) + 30 log10(speed) (CNOSSOS-EU style) as a line source with a daily
//!   flow curve, wet-road tyre spray hiss, ground effect (Delany-Bazley impedance, spherical-wave
//!   reflection), ISO 9613-1 absorption, and individual Doppler pass-bys on nearby roads.
//!
//! Stem 'ambience'. Levels are peak / RMS dB SPL at 1 m (7.5 m for vehicles), spread spherically.
//! Every voice keeps its own copy of the activity state (a pure function of the weather sequence),
//! so all of them agree without sharing.

use super::common::*;
use crate::dsp::{Biquad, BiquadCoefs, Sos};
use crate::engine::{BlockCtx, BoxedVoice, BuildCfg, Voice};
use crate::rng::{combine, hash3, Rng};
use crate::types::{precip, season, Emission, Patch, PatchKind, Scene, Shelter, Stem, Weather, N_AMBI};
use realfft::num_complex::Complex64;
use std::f64::consts::PI;
use std::sync::OnceLock;

const AIR_MIN_M: f32 = 8.0;

/// Snowy tree cricket chirp rate (Dolbear 1897): N60 = 4 (T_F - 50) + 40.
pub fn dolbear_chirps_per_min(temp_c: f64) -> f64 {
    let t_f = 1.8 * temp_c + 32.0;
    (4.0 * (t_f - 50.0) + 40.0).max(0.0)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Auto,
    On,
    Off,
}

fn mode_of(scene: &Scene, key: &str) -> Mode {
    match biophony_str(scene, key) {
        None => Mode::Auto,
        Some(s) => match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" | "1" => Mode::On,
            "off" | "false" | "no" | "0" | "none" => Mode::Off,
            _ => Mode::Auto,
        },
    }
}

#[inline]
fn gauss_bump(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    if z.abs() < 4.0 {
        (-0.5 * z * z).exp()
    } else {
        0.0
    }
}

#[inline]
fn hour_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(24.0);
    if d > 12.0 {
        d - 24.0
    } else {
        d
    }
}

#[inline]
fn ss(x: f64, lo: f64, hi: f64) -> f64 {
    smoothstep64(x, lo, hi)
}

#[inline]
fn ri(rng: &mut Rng, lo: i64, hi: i64) -> i64 {
    if hi <= lo {
        lo
    } else {
        lo + rng.below((hi - lo) as u64) as i64
    }
}

/// ISO 9613-1 amplitude transmission at f over distance_m (1 inside AIR_MIN_M).
fn air_gain(f: f64, distance_m: f32, temp_c: f32, rh_pct: f32) -> f32 {
    if distance_m <= AIR_MIN_M {
        return 1.0;
    }
    10f64.powf(-iso9613_alpha(f, temp_c, rh_pct) * distance_m as f64 / 20.0) as f32
}

// --------------------------------------------------------------------------
// regions and species
// --------------------------------------------------------------------------
const REGIONS: [&str; 8] =
    ["temperate_europe", "north_america", "boreal", "mediterranean", "east_asia", "tropical", "arid", "oceania"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bird {
    Warbler,
    Flute,
    Trill,
    TwoNote,
    Chip,
    Coo,
    Caw,
    Owl,
}

impl Bird {
    fn name(self) -> &'static str {
        match self {
            Bird::Warbler => "warbler",
            Bird::Flute => "flute",
            Bird::Trill => "trill",
            Bird::TwoNote => "two_note",
            Bird::Chip => "chip",
            Bird::Coo => "coo",
            Bird::Caw => "caw",
            Bird::Owl => "owl",
        }
    }
    /// (peak dB SPL at 1 m, songs per minute while active, longest song s)
    fn spec(self) -> (f32, f32, f32) {
        match self {
            Bird::Warbler => (80.0, 6.0, 2.1),
            Bird::Flute => (82.0, 5.0, 2.9),
            Bird::Trill => (86.0, 5.0, 3.1),
            Bird::TwoNote => (82.0, 5.0, 3.3),
            Bird::Chip => (72.0, 9.0, 1.0),
            Bird::Coo => (74.0, 2.5, 3.1),
            Bird::Caw => (84.0, 1.5, 4.7),
            Bird::Owl => (78.0, 1.5, 5.7),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Frog {
    Common,
    Tree,
    ToadTrill,
    Peeper,
    Chorus,
    Bull,
    Coqui,
}

impl Frog {
    fn name(self) -> &'static str {
        match self {
            Frog::Common => "common_frog",
            Frog::Tree => "tree_frog",
            Frog::ToadTrill => "toad_trill",
            Frog::Peeper => "peeper",
            Frog::Chorus => "chorus_frog",
            Frog::Bull => "bullfrog",
            Frog::Coqui => "coqui",
        }
    }
    fn spec(self) -> (f32, f32, f32) {
        match self {
            Frog::Common => (66.0, 6.0, 1.3),
            Frog::Tree => (86.0, 5.0, 2.8),
            Frog::ToadTrill => (78.0, 2.0, 10.2),
            Frog::Peeper => (88.0, 50.0, 0.3),
            Frog::Chorus => (82.0, 12.0, 1.0),
            Frog::Bull => (84.0, 2.0, 7.7),
            Frog::Coqui => (86.0, 25.0, 0.4),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Insect {
    TreeCricket,
    FieldCricket,
    TrillCricket,
    BellCricket,
    Katydid,
}

impl Insect {
    /// chirps per second (None: long trills with gaps)
    fn rate(self, t: f64) -> Option<f64> {
        match self {
            Insect::TreeCricket => Some(dolbear_chirps_per_min(t) / 60.0),
            Insect::FieldCricket => Some(1.25 * dolbear_chirps_per_min(t) / 60.0),
            Insect::TrillCricket => None,
            Insect::BellCricket => Some(1.6 + 0.08 * (t - 20.0)),
            Insect::Katydid => Some(0.8 + 0.05 * (t - 20.0)),
        }
    }
    fn carrier(self, t: f64) -> f64 {
        match self {
            Insect::TreeCricket => 1450.0 + 75.0 * t,
            Insect::FieldCricket => 4600.0,
            Insect::TrillCricket => 2400.0 + 20.0 * t,
            Insect::BellCricket => 4400.0,
            Insect::Katydid => 10000.0,
        }
    }
    fn duty(self) -> f64 {
        match self {
            Insect::TreeCricket => 0.45,
            Insect::FieldCricket => 0.35,
            Insect::BellCricket | Insect::Katydid => 0.3,
            Insect::TrillCricket => 0.5,
        }
    }
    fn syll(self) -> f64 {
        match self {
            Insect::TreeCricket | Insect::Katydid => 3.0,
            Insect::FieldCricket => 4.0,
            Insect::BellCricket => 8.0,
            Insect::TrillCricket => 1.0,
        }
    }
    fn syll_rate(self, t: f64) -> f64 {
        30.0 + 1.5 * (t - 20.0)
    }
    fn level(self) -> f32 {
        match self {
            Insect::TreeCricket => 64.0,
            Insect::FieldCricket | Insect::Katydid => 70.0,
            Insect::TrillCricket | Insect::BellCricket => 62.0,
        }
    }
    fn noisy(self) -> bool {
        self == Insect::Katydid
    }
}

fn region_birds(r: &str) -> &'static [Bird] {
    use Bird::*;
    match r {
        "boreal" => &[Flute, Trill, Warbler, Chip, Caw],
        "mediterranean" => &[Warbler, TwoNote, Chip, Coo, Trill],
        "east_asia" => &[Warbler, TwoNote, Chip, Coo, Caw],
        "tropical" => &[Flute, TwoNote, Warbler, Chip, Coo, Trill],
        "arid" => &[Chip, Coo, Caw, TwoNote],
        "oceania" => &[Flute, Warbler, Chip, Caw],
        _ => &[Warbler, Flute, Trill, TwoNote, Chip, Coo, Caw],
    }
}

fn region_frogs(r: &str) -> &'static [Frog] {
    use Frog::*;
    match r {
        "north_america" => &[Peeper, Tree, Chorus, Peeper, Bull, ToadTrill],
        "boreal" => &[Common, Chorus],
        "mediterranean" => &[Tree, Common],
        "east_asia" => &[Tree, Chorus, Bull],
        "tropical" => &[Coqui, Peeper, Tree, Chorus],
        "arid" => &[ToadTrill],
        "oceania" => &[Tree, Chorus],
        _ => &[Common, Tree, ToadTrill],
    }
}

fn region_insects(r: &str) -> &'static [Insect] {
    use Insect::*;
    match r {
        "north_america" => &[TreeCricket, FieldCricket, Katydid],
        "boreal" | "arid" => &[FieldCricket],
        "east_asia" => &[BellCricket, FieldCricket],
        "tropical" => &[Katydid, FieldCricket, TrillCricket],
        "oceania" => &[FieldCricket, Katydid],
        _ => &[FieldCricket, TrillCricket],
    }
}

const CICADA_REGIONS: [&str; 6] = ["mediterranean", "east_asia", "tropical", "arid", "oceania", "north_america"];
const OWL_REGIONS: [&str; 5] = ["temperate_europe", "boreal", "north_america", "east_asia", "mediterranean"];

// --------------------------------------------------------------------------
// activity rules (shared logic, one copy per voice)
// --------------------------------------------------------------------------
const A_INSECTS: usize = 0;
const A_CICADAS: usize = 1;
const A_BIRDS: usize = 2;
const A_OWLS: usize = 3;
const A_FROGS: usize = 4;
const ACT_TAUS: [f64; 5] = [90.0, 60.0, 60.0, 60.0, 120.0];

#[derive(Clone, Debug)]
struct Activity {
    tropical: bool,
    lat: f32,
    m_birds: Mode,
    m_insects: Mode,
    m_frogs: Mode,
    has_water: bool,
    rain_ema: f64,
    wet_memory: f64,
    dry_s: f64,
    act: Option<[f64; 5]>,
}

impl Activity {
    fn targets(&mut self, w: &Weather, dt: f64) -> [f64; 5] {
        let r = w.rain_rate_mm_h as f64;
        let u = w.wind_listener_mps as f64;
        let t = w.temperature_c as f64;
        let sea = w.season;
        let hour = (w.hour as f64).rem_euclid(24.0);
        let tropical = self.tropical;
        if self.act.is_none() && self.rain_ema == 0.0 {
            self.rain_ema = r;
        }
        let a = 1.0 - (-dt / 600.0).exp();
        self.rain_ema += (r - self.rain_ema) * a;
        if r < 0.1 {
            self.dry_s += dt;
            self.wet_memory *= (-dt / 3600.0).exp();
        } else {
            self.dry_s = 0.0;
            self.wet_memory = self.wet_memory.max(self.rain_ema);
        }
        let (rise, sset) = sun_times(self.lat, sea);
        let (rise, sset) = (rise as f64, sset as f64);
        let night = !w.is_day;

        let rain_off = 1.0 - ss(r, 0.6, 2.0);
        let wind_off = 1.0 - ss(u, 5.0, 9.0);
        let seas_i = if tropical {
            1.0
        } else {
            match sea {
                season::SUMMER => 1.0,
                season::AUTUMN => 0.75,
                season::SPRING => 0.35,
                _ => 0.0,
            }
        };
        let dusk = gauss_bump(hour_dist(hour, sset), 0.5, 1.0);
        let time_i = if night { 1.0 } else { dusk };
        let mut insects = ss(t, 12.0, 16.0) * rain_off * wind_off * seas_i * time_i;
        match self.m_insects {
            Mode::On => insects = insects.max((1.0 - ss(r, 2.0, 6.0)) * wind_off),
            Mode::Off => insects = 0.0,
            Mode::Auto => {}
        }
        let mut cicadas = 0.0;
        if self.m_insects != Mode::Off {
            let seas_c = if tropical {
                1.0
            } else {
                match sea {
                    season::SUMMER => 1.0,
                    season::AUTUMN => 0.3,
                    _ => 0.0,
                }
            };
            cicadas = (if night { 0.0 } else { 1.0 }) * ss(t, 22.0, 27.0) * rain_off * wind_off * seas_c;
        }

        let dawn = gauss_bump(hour_dist(hour, rise + 0.3), 0.0, 0.75);
        let evening = 0.45 * gauss_bump(hour_dist(hour, sset - 0.3), 0.0, 0.7);
        let mut day = 0.0;
        if rise + 0.5 < hour && hour < sset - 0.5 {
            day = 0.3 * (1.0 - 0.35 * gauss_bump(hour, 13.5, 1.5));
        }
        let seas_b = if tropical {
            0.9
        } else {
            match sea {
                season::SPRING => 1.0,
                season::SUMMER => 0.7,
                season::AUTUMN => 0.35,
                _ => 0.2,
            }
        };
        let rain_b = 1.0 / (1.0 + (r / 0.8).powf(1.5));
        let wind_b = 1.0 / (1.0 + (u / 8.0).powi(3));
        let rebound = 1.0
            + 1.5 * (self.wet_memory / 2.0).min(1.0) * (-self.dry_s / 1200.0).exp() * ss(self.dry_s, 30.0, 240.0);
        let mut birds = dawn.max(evening).max(day) * seas_b * rain_b * wind_b * rebound;
        match self.m_birds {
            Mode::On => birds = birds.max(0.4 * rain_b * wind_b),
            Mode::Off => birds = 0.0,
            Mode::Auto => {}
        }
        let mut owls = 0.0;
        if self.m_birds != Mode::Off && night {
            let autumn_winter = sea == season::AUTUMN || sea == season::WINTER;
            owls = (1.0 / (1.0 + (r / 0.5).powi(2))) * wind_b * (0.6 + 0.4 * autumn_winter as u8 as f64);
        }

        let wet = ss(self.rain_ema, 0.1, 1.5)
            .max(ss(self.wet_memory, 0.2, 2.0) * (-self.dry_s / 10800.0).exp())
            .max(if self.has_water { 0.5 } else { 0.0 });
        let seas_f = if tropical {
            1.0
        } else {
            match sea {
                season::SPRING => 1.0,
                season::SUMMER => 0.85,
                season::AUTUMN => 0.25,
                _ => 0.0,
            }
        };
        let time_f = if night { 1.0 } else { gauss_bump(hour_dist(hour, sset), 0.3, 0.8).max(0.12) };
        let heavy = 1.0 - 0.4 * ss(r, 15.0, 40.0);
        let mut frogs = ss(t, 8.0, 14.0) * wet * seas_f * time_f * heavy * wind_off;
        match self.m_frogs {
            Mode::On => frogs = frogs.max(0.7 * heavy),
            Mode::Off => frogs = 0.0,
            Mode::Auto => {}
        }
        [insects, cicadas, birds, owls, frogs]
    }

    /// Smoothed activities for this block (call once per block).
    fn step(&mut self, w: &Weather, dt: f64) -> [f64; 5] {
        let tgt = self.targets(w, dt);
        match self.act.as_mut() {
            None => self.act = Some(tgt),
            Some(a) => {
                for k in 0..5 {
                    a[k] += (tgt[k] - a[k]) * (1.0 - (-dt / ACT_TAUS[k]).exp());
                }
            }
        }
        self.act.unwrap()
    }

    fn reset(&mut self) {
        self.rain_ema = 0.0;
        self.wet_memory = 0.0;
        self.dry_s = 3600.0;
        self.act = None;
    }
}

// --------------------------------------------------------------------------
// elementary sound shapes (appended to a buffer)
// --------------------------------------------------------------------------
#[derive(Clone, Copy)]
enum Shape {
    Lin,
    Exp,
    Chevron,
    Hook,
}

/// FM tone with a smooth envelope (bird notes, frog peeps, owl hoots).
#[allow(clippy::too_many_arguments)]
fn tone(
    out: &mut Vec<f32>,
    sr: f64,
    dur: f64,
    f0: f64,
    f1: f64,
    shape: Shape,
    vib_hz: f64,
    vib_depth: f64,
    harm: &[f64],
    attack: f64,
    release: f64,
) {
    let n = ((dur * sr) as usize).max(16);
    let a = ((attack * n as f64) as usize).max(1);
    let r = ((release * n as f64) as usize).max(1);
    let mut ph = 0.0f64;
    let fmax = 0.45 * sr;
    for i in 0..n {
        let x = i as f64 / n as f64;
        let mut f = match shape {
            Shape::Exp => f0 * (f1 / f0).powf(x),
            Shape::Chevron => f0 + (f1 - f0) * (PI * x).sin(),
            Shape::Hook => f0 + (f1 - f0) * x * x * x,
            Shape::Lin => f0 + (f1 - f0) * x,
        };
        if vib_hz > 0.0 && vib_depth > 0.0 {
            f *= 1.0 + vib_depth * (2.0 * PI * vib_hz * i as f64 / sr).sin();
        }
        let f = f.clamp(20.0, fmax);
        ph += 2.0 * PI * f / sr;
        if ph > 2.0 * PI {
            ph -= 2.0 * PI;
        }
        let p32 = ph as f32;
        let mut y = p32.sin();
        for (k, h) in harm.iter().enumerate() {
            let m = (k + 2) as f64;
            if *h > 0.0 && m * f < fmax {
                y += *h as f32 * ((k + 2) as f32 * p32).sin();
            }
        }
        let mut env = 1.0f32;
        if i < a {
            env = (std::f32::consts::FRAC_PI_2 * i as f32 / a as f32).sin().powi(2);
        }
        if i >= n.saturating_sub(r) {
            let j = i - n.saturating_sub(r);
            env = (std::f32::consts::FRAC_PI_2 * j as f32 / r as f32).cos().powi(2);
        }
        out.push(y * env);
    }
}

fn gap(out: &mut Vec<f32>, sr: f64, s: f64) {
    let n = (s * sr).max(0.0) as usize;
    out.extend(std::iter::repeat_n(0.0f32, n));
}

fn norm_from(out: &mut [f32], s0: usize) {
    if s0 < out.len() {
        normalize_peak(&mut out[s0..]);
    }
}

/// Carrier gated by a train of smooth pulses (frog trills, croaks, katydid buzz), peak-normalised.
#[allow(clippy::too_many_arguments)]
fn pulse_train(
    out: &mut Vec<f32>,
    rng: &mut Rng,
    sr: f64,
    dur: f64,
    rate: f64,
    pulse_frac: f64,
    carrier: f64,
    noise: bool,
    rate_end: Option<f64>,
    glide: f64,
) {
    let n = ((dur * sr) as usize).max(16);
    let r1 = rate_end.unwrap_or(rate);
    let s0 = out.len();
    let a = ((0.05 * n as f64) as usize).max(1);
    let lin = |i: usize, m: usize| if m > 1 { i as f64 / (m - 1) as f64 } else { 0.0 };
    let mut bpf = Biquad::new(bp(carrier as f32, 3.0, sr as f32));
    let phase0 = if noise { 0.0 } else { rng.range(0.0, 2.0 * PI) };
    let mut ph = 0.0f64;
    let mut cph = 0.0f64;
    for i in 0..n {
        let rr = rate + (r1 - rate) * lin(i, n);
        ph += rr / sr;
        let frac = ph.rem_euclid(1.0);
        let env = if frac < pulse_frac {
            (std::f32::consts::PI * (frac / pulse_frac.max(1e-3)) as f32).sin().powi(2) as f64
        } else {
            0.0
        };
        let c = if noise {
            bpf.process(rng.normal() as f32) as f64
        } else {
            let f = carrier * (1.0 + glide * lin(i, n));
            cph += f / sr;
            if cph >= 1.0 {
                cph -= 1.0;
            }
            ((2.0 * PI * cph + phase0) as f32).sin() as f64
        };
        let mut shape = 1.0;
        if i < a {
            shape = lin(i, a);
        }
        if i >= n - a {
            shape = 1.0 - lin(i - (n - a), a);
        }
        out.push((c * env * shape) as f32);
    }
    norm_from(out, s0);
}

/// One frog note as a train of damped vocal pulses: each pulse a fast (~1 ms) onset and an
/// exponential decay of a harmonic carrier, the laryngeal burst ringing in the vocal sac, not a
/// smooth AM envelope (Gerhardt & Huber 2002, 'Acoustic Communication in Insects and Anurans',
/// ch. 3; Ryan 1985). Notes swell over their first third; pulses jitter a little in period,
/// amplitude and pitch.
#[allow(clippy::too_many_arguments)]
fn frog_pulses(
    out: &mut Vec<f32>,
    rng: &mut Rng,
    sr: f64,
    n_pulses: usize,
    period0: f64,
    period1: f64,
    carrier: f64,
    glide: f64,
    decay_frac: f64,
    harm: &[f64],
    ramp_up: f64,
) {
    let s0 = out.len();
    let n_pulses = n_pulses.max(1);
    let mut times = Vec::with_capacity(n_pulses);
    let mut t = 0.0;
    for k in 0..n_pulses {
        times.push(t);
        let x = if n_pulses > 1 { k as f64 / (n_pulses - 1) as f64 } else { 0.0 };
        t += (period0 + (period1 - period0) * x) * rng.lognormal(0.0, 0.04);
    }
    let pmax = period0.max(period1);
    let len = ((t + 5.0 * decay_frac * pmax + 0.002) * sr) as usize + 16;
    out.extend(std::iter::repeat_n(0.0f32, len));
    let y = &mut out[s0..];
    let fmax = 0.45 * sr;
    for (k, &t0) in times.iter().enumerate() {
        let x = if n_pulses > 1 { k as f64 / (n_pulses - 1) as f64 } else { 0.0 };
        let amp = (ramp_up + (1.0 - ramp_up) * (x * 3.0).min(1.0)) * rng.lognormal(0.0, 0.12);
        let per = period0 + (period1 - period0) * x;
        let tau = (decay_frac * per).max(4e-4);
        let fc = carrier * (1.0 + glide * x) * rng.lognormal(0.0, 0.006);
        let i0 = (t0 * sr) as usize;
        let att = ((0.0012 * sr) as usize).max(1);
        let m = att + (5.0 * tau * sr) as usize;
        let ph0 = rng.range(0.0, 2.0 * PI);
        let w = 2.0 * PI * fc / sr;
        let kd = (-1.0 / (tau * sr)).exp();
        let mut env = 1.0f64;
        for j in 0..m {
            let idx = i0 + j;
            if idx >= y.len() {
                break;
            }
            let e = if j < att {
                (PI * 0.5 * j as f64 / att as f64).sin().powi(2)
            } else {
                env *= kd;
                env
            };
            let ph = ph0 + w * j as f64;
            let mut v = ph.sin();
            for (h, a) in harm.iter().enumerate() {
                if (h as f64 + 2.0) * fc < fmax {
                    v += a * ((h as f64 + 2.0) * ph).sin();
                }
            }
            y[idx] += (amp * e * v) as f32;
        }
    }
    norm_from(out, s0);
}

/// A bullfrog-like note: a glottal pulse train at f0 ~ 90-130 Hz (jittered) driving two vocal-
/// tract / sac resonances, ~200-300 Hz and ~1.2-1.5 kHz (Capranica 1965, 'The evoked vocal
/// response of the bullfrog'; Bee & Gerhardt 2001), a low harmonic drone with a soft rise and a
/// slow fall.
fn bull_note(out: &mut Vec<f32>, rng: &mut Rng, sr: f64, dur: f64, p: f64) {
    let s0 = out.len();
    let n = (dur * sr) as usize;
    let f0 = rng.range(90.0, 130.0) * p;
    let mut exc = vec![0.0f32; n];
    let mut t = 0.0f64;
    while (t * sr) < n as f64 {
        let i = (t * sr) as usize;
        if i < n {
            exc[i] += rng.lognormal(0.0, 0.08) as f32;
            if i + 1 < n {
                exc[i + 1] -= 0.6 * exc[i];
            }
        }
        t += rng.lognormal(0.0, 0.015) / f0;
    }
    let f1 = rng.range(200.0, 300.0) * p;
    let f2 = rng.range(1200.0, 1500.0) * p;
    let mut r1 = Biquad::new(bp(f1 as f32, 3.0, sr as f32));
    let mut r2 = Biquad::new(bp(f2 as f32, 5.0, sr as f32));
    let mut lp = Biquad::new(lp2(2500.0, sr as f32));
    let att = 0.08 * sr;
    let rel = 0.15 * sr;
    let am_f = rng.range(4.0, 7.0);
    for (i, e) in exc.iter().enumerate() {
        let a = r1.process(*e);
        let b = r2.process(*e);
        let x = lp.process(a + 0.35 * b);
        let fi = i as f64;
        let env = (fi / att).min(1.0).min(((n as f64 - fi) / rel).max(0.0)).powf(1.5)
            * (1.0 + 0.12 * (2.0 * PI * am_f * fi / sr).sin());
        out.push(x * env as f32);
    }
    norm_from(out, s0);
}

// --------------------------------------------------------------------------
// bird songs and frog calls
// --------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
enum Call {
    Bird(Bird),
    Frog(Frog),
}

fn song_bird(b: Bird, rng: &mut Rng, sr: f64, p: f64, out: &mut Vec<f32>) {
    let s0 = out.len();
    match b {
        Bird::Warbler => {
            let mut f = rng.range(2800.0, 4800.0) * p;
            for _ in 0..ri(rng, 6, 17) {
                f = (f * rng.lognormal(0.0, 0.18)).clamp(2200.0 * p, 7000.0 * p);
                let f1 = f * rng.lognormal(0.0, 0.2);
                let shape = [Shape::Lin, Shape::Exp, Shape::Chevron, Shape::Hook][ri(rng, 0, 4) as usize];
                let d = rng.range(0.025, 0.08);
                tone(out, sr, d, f, f1, shape, 0.0, 0.0, &[0.05], 0.15, 0.35);
                let g = rng.range(0.01, 0.04);
                gap(out, sr, g);
            }
        }
        Bird::Flute => {
            let base = rng.range(1600.0, 2800.0) * p;
            const RATIOS: [f64; 6] = [1.0, 9.0 / 8.0, 5.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0, 5.0 / 6.0];
            for _ in 0..ri(rng, 3, 7) {
                let f0 = base * RATIOS[ri(rng, 0, 6) as usize];
                let d = rng.range(0.12, 0.3);
                let f1 = f0 * rng.range(0.92, 1.08);
                let vh = rng.range(22.0, 38.0);
                let vd = rng.range(0.006, 0.02);
                tone(out, sr, d, f0, f1, Shape::Chevron, vh, vd, &[0.15, 0.04], 0.15, 0.35);
                let g = rng.range(0.03, 0.09);
                gap(out, sr, g);
            }
            if rng.uniform() < 0.6 {
                for _ in 0..ri(rng, 3, 8) {
                    let f0 = rng.range(4500.0, 7000.0) * p;
                    let d = rng.range(0.015, 0.04);
                    let f1 = f0 * rng.range(0.7, 1.3);
                    tone(out, sr, d, f0, f1, Shape::Lin, 0.0, 0.0, &[], 0.15, 0.35);
                    let g = rng.range(0.008, 0.02);
                    gap(out, sr, g);
                }
            }
        }
        Bird::Trill => {
            for _ in 0..ri(rng, 1, 3) {
                let rate = rng.range(14.0, 28.0);
                let f0 = rng.range(4000.0, 7000.0) * p;
                let f1 = f0 * rng.range(0.75, 1.25);
                let cnt = (rate * rng.range(0.6, 1.4)) as usize;
                for _ in 0..cnt {
                    tone(out, sr, 0.6 / rate, f0, f1, Shape::Lin, 0.0, 0.0, &[], 0.2, 0.4);
                    gap(out, sr, 0.4 / rate);
                }
                let g = rng.range(0.02, 0.08);
                gap(out, sr, g);
            }
        }
        Bird::TwoNote => {
            let hi = rng.range(4800.0, 6200.0) * p;
            let lo = hi * rng.range(0.62, 0.8);
            for _ in 0..ri(rng, 3, 9) {
                let d1 = rng.range(0.07, 0.12);
                tone(out, sr, d1, hi, hi * 0.97, Shape::Lin, 0.0, 0.0, &[], 0.15, 0.35);
                gap(out, sr, 0.03);
                let d2 = rng.range(0.07, 0.12);
                tone(out, sr, d2, lo, lo * 0.98, Shape::Lin, 0.0, 0.0, &[], 0.15, 0.35);
                let g = rng.range(0.06, 0.12);
                gap(out, sr, g);
            }
        }
        Bird::Chip => {
            for _ in 0..ri(rng, 1, 4) {
                let f0 = rng.range(3500.0, 7000.0) * p;
                let d = rng.range(0.015, 0.03);
                let f1 = f0 * rng.range(0.55, 0.8);
                tone(out, sr, d, f0, f1, Shape::Exp, 0.0, 0.0, &[], 0.1, 0.5);
                let g = rng.range(0.08, 0.25);
                gap(out, sr, g);
            }
        }
        Bird::Coo => {
            let f0 = rng.range(450.0, 620.0) * p;
            const P1: [(f64, f64); 3] = [(0.35, 1.0), (0.55, 1.08), (0.3, 0.98)];
            const P2: [(f64, f64); 5] = [(0.25, 1.0), (0.6, 1.1), (0.3, 1.0), (0.3, 0.97), (0.25, 0.95)];
            let pattern: &[(f64, f64)] = if rng.uniform() < 0.5 { &P1 } else { &P2 };
            for &(dur, r) in pattern {
                let d = dur * rng.range(0.9, 1.1);
                tone(out, sr, d, f0 * r, f0 * r * 0.93, Shape::Chevron, 0.0, 0.0, &[0.3, 0.08], 0.25, 0.4);
                let g = rng.range(0.08, 0.2);
                gap(out, sr, g);
            }
        }
        Bird::Caw => {
            for _ in 0..ri(rng, 2, 6) {
                let dur = rng.range(0.22, 0.4);
                let n = (dur * sr) as usize;
                let f0 = rng.range(420.0, 620.0) * p;
                let s1 = out.len();
                let sq = (n.max(1) as f64).sqrt();
                let mut walk = 0.0;
                let mut ph = 0.0;
                let mut lp = Biquad::new(lp2(80.0, sr as f32));
                for i in 0..n {
                    walk += rng.normal();
                    let ramp = 1.05 + (0.92 - 1.05) * if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
                    let f = f0 * (1.0 + 0.04 * walk / sq) * ramp;
                    ph += 2.0 * PI * f.clamp(50.0, 4000.0) / sr;
                    if ph > 2.0 * PI {
                        ph -= 2.0 * PI;
                    }
                    // sum_k sin(k ph) / k by the Chebyshev recurrence s_k = 2 cos(ph) s_{k-1} - s_{k-2}
                    let (s1, c1) = ph.sin_cos();
                    let (mut sm2, mut sm1) = (0.0f64, s1);
                    let mut y = 0.0;
                    for k in 1..13 {
                        if (k as f64) * f0 < 0.45 * sr {
                            y += sm1 / k as f64;
                        }
                        let nx = 2.0 * c1 * sm1 - sm2;
                        sm2 = sm1;
                        sm1 = nx;
                    }
                    out.push(y as f32);
                }
                for i in 0..n {
                    let m = lp.process(rng.normal() as f32);
                    out[s1 + i] *= 1.0 + 0.5 * m * 6.0;
                }
                let fc = rng.range(1100.0, 1700.0);
                let mut f1 = Biquad::new(bp(fc as f32, 1.4, sr as f32));
                let mut f2 = Biquad::new(hp2(300.0, sr as f32));
                f1.process_block(&mut out[s1..]);
                f2.process_block(&mut out[s1..]);
                for i in 0..n {
                    out[s1 + i] *= (PI * i as f64 / n as f64).sin().powf(0.6) as f32;
                }
                norm_from(out, s1);
                let g = rng.range(0.25, 0.5);
                gap(out, sr, g);
            }
        }
        Bird::Owl => {
            let f0 = rng.range(400.0, 650.0) * p;
            let d = rng.range(0.35, 0.55);
            tone(out, sr, d, f0, f0 * 0.97, Shape::Chevron, 0.0, 0.0, &[0.08], 0.2, 0.4);
            let g = rng.range(1.8, 3.2);
            gap(out, sr, g);
            for _ in 0..ri(rng, 1, 4) {
                tone(out, sr, 0.12, f0 * 0.95, f0, Shape::Lin, 0.0, 0.0, &[0.06], 0.15, 0.35);
                gap(out, sr, 0.12);
            }
            let d = rng.range(0.6, 1.0);
            let vh = rng.range(5.0, 8.0);
            tone(out, sr, d, f0, f0 * 0.96, Shape::Lin, vh, 0.03, &[0.08], 0.1, 0.45);
        }
    }
    norm_from(out, s0);
}

fn call_frog(fr: Frog, rng: &mut Rng, sr: f64, p: f64, out: &mut Vec<f32>) {
    let s0 = out.len();
    match fr {
        Frog::Common => {
            let d = rng.range(0.5, 1.2);
            let r = rng.range(22.0, 35.0);
            let c = rng.range(350.0, 550.0) * p;
            pulse_train(out, rng, sr, d, r, 0.5, c, true, None, 0.0);
        }
        Frog::Tree => {
            // tree frog 'creek' notes (Hyla arborea / Hyla-Pseudacris type): 10-15 pulses at
            // 150-200 pulses/s on a 1.5-2.5 kHz carrier, notes repeated in a series (Schneider 1967;
            // Friedl & Klump 2002)
            let car = rng.range(1700.0, 2400.0) * p;
            let rate = rng.range(150.0, 200.0);
            for _ in 0..ri(rng, 4, 13) {
                let np = ri(rng, 10, 16) as usize;
                frog_pulses(out, rng, sr, np, 1.0 / rate, 1.0 / rate, car, 0.03, 0.3, &[0.18, 0.05], 0.45);
                let g = rng.range(0.09, 0.14);
                gap(out, sr, g);
            }
        }
        Frog::ToadTrill => {
            let d = rng.range(3.0, 10.0);
            let r = rng.range(30.0, 40.0);
            let c = rng.range(1550.0, 1900.0) * p;
            pulse_train(out, rng, sr, d, r, 0.45, c, false, None, 0.0);
        }
        Frog::Peeper => {
            // spring peeper (Pseudacris crucifer): a single rising whistle, ~2.8-3 kHz, 0.1-0.2 s,
            // upsweep of a few hundred Hz (Wilczynski et al. 1984; Brenowitz et al. 1984)
            let f0 = rng.range(2700.0, 2900.0) * p;
            let d = rng.range(0.11, 0.18);
            let f1 = f0 * rng.range(1.05, 1.1);
            tone(out, sr, d, f0, f1, Shape::Hook, 0.0, 0.0, &[0.04], 0.12, 0.25);
        }
        Frog::Chorus => {
            // chorus frog 'creeek' (Pseudacris triseriata / maculata): 10-18 pulses, the pulse rate
            // rising through the call, ~2.5-3.2 kHz (like a thumb run along a comb)
            let r0 = rng.range(14.0, 20.0);
            let np = ri(rng, 10, 19) as usize;
            let c = rng.range(2500.0, 3200.0) * p;
            frog_pulses(out, rng, sr, np, 1.0 / r0, 1.0 / (r0 * 1.7), c, 0.08, 0.12, &[0.1], 0.6);
        }
        Frog::Bull => {
            for _ in 0..ri(rng, 2, 6) {
                let dur = rng.range(0.45, 0.8);
                bull_note(out, rng, sr, dur, p);
                let g = rng.range(0.3, 0.7);
                gap(out, sr, g);
            }
        }
        Frog::Coqui => {
            tone(out, sr, 0.1, 1150.0 * p, 1200.0 * p, Shape::Lin, 0.0, 0.0, &[], 0.2, 0.3);
            gap(out, sr, 0.05);
            tone(out, sr, 0.15, 1900.0 * p, 2250.0 * p, Shape::Exp, 0.0, 0.0, &[], 0.15, 0.35);
        }
    }
    norm_from(out, s0);
}

// --------------------------------------------------------------------------
// voices
// --------------------------------------------------------------------------
fn point_of(pos: [f32; 3]) -> (f32, f32, f32) {
    let (x, y, z) = (pos[0], pos[1], pos[2]);
    let dist = (x * x + y * y + z * z).sqrt().max(0.5);
    (y.atan2(x), z.atan2(x.hypot(y)), dist)
}

/// A bird, owl or frog: discrete songs / calls rendered whole, then played out block by block.
pub struct CallerVoice {
    name: String,
    act: Activity,
    act_idx: usize,
    call: Call,
    level: f32,
    per_min: f32,
    pitch: f64,
    az: f32,
    el: f32,
    dist: f32,
    sr: f32,
    key_song: u64,
    key_call: u64,
    busy_until: i64,
    song: Vec<f32>,
    song_start: i64,
    air: AirPath,
    /// A call being synthesized on the worker pool (whole songs take up to tens of ms; done
    /// off the block's critical path). It starts CALL_DELAY_BLOCKS after its trigger, a fixed
    /// delay, so the output stays deterministic however fast the job finishes.
    pending: bool,
    /// Chorus timing (period s, phase 0..1) for short-call frogs: neighbours call in each other's
    /// gaps (call alternation / call-and-response; Rosen & Lemon 1974; Gerhardt & Huber 2002 ch. 8),
    /// on a shared species clock with per-cycle timing jitter and skipped cycles.
    slot: Option<(f64, f64)>,
    /// Song type in use: a songbird repeats one song type several times before switching
    /// (eventual variety; Hartshorne 1956, Catchpole & Slater 2008 'Bird Song' ch. 8).
    song_type: u64,
    job_tx: crossbeam_channel::Sender<(Vec<f32>, i64)>,
    job_rx: crossbeam_channel::Receiver<(Vec<f32>, i64)>,
}

const CALL_DELAY_BLOCKS: i64 = 12;

impl CallerVoice {
    #[allow(clippy::too_many_arguments)]
    fn new(name: String, layer: &str, call: Call, pos: [f32; 3], act: &Activity, act_idx: usize, cfg_rng: &mut Rng, cfg: &BuildCfg) -> CallerVoice {
        let (lvl, per_min, max_s) = match call {
            Call::Bird(b) => b.spec(),
            Call::Frog(f) => f.spec(),
        };
        let level = db_spl(lvl) * cfg_rng.lognormal(0.0, 0.15) as f32;
        let per_min = per_min * cfg_rng.range(0.7, 1.3) as f32;
        let pitch = cfg_rng.range(0.92, 1.08);
        let (az, el, dist) = point_of(pos);
        let sr = cfg.sr as f32;
        let id = name.clone();
        let name = format!("bio.{id}");
        let (job_tx, job_rx) = crossbeam_channel::bounded(1);
        CallerVoice {
            act: act.clone(),
            act_idx,
            call,
            level,
            per_min,
            pitch,
            az,
            el,
            dist,
            sr,
            key_song: cfg.rng.key(&format!("{layer}.{id}")),
            key_call: cfg.rng.key(&format!("bio.calls.{id}")),
            busy_until: -1,
            song: Vec::with_capacity(((max_s + 0.2) * sr) as usize),
            song_start: 0,
            air: AirPath::new(dist, sr, AIR_MIN_M),
            name,
            pending: false,
            slot: None,
            song_type: 0,
            job_tx,
            job_rx,
        }
    }

    fn begin(&mut self, ctx: &BlockCtx, a: f64, start: i64, offset: usize) {
        let sr = self.sr as f64;
        let rng = ctx.rng.stream(self.key_song, ctx.index);
        let type_rng = Rng::from_key(combine(self.key_song, 0x5017_7E00 + self.song_type), 0);
        if matches!(self.call, Call::Bird(_)) && crate::rng::uniform_at(self.key_song, ctx.index, 0x7E) < 0.25 {
            self.song_type += 1;
        }
        let slotted = self.slot.is_some();
        let mut song = std::mem::take(&mut self.song);
        song.clear();
        let call = self.call;
        let pitch = self.pitch;
        let level = self.level;
        let dist = self.dist;
        let (temp, hum) = (ctx.weather.temperature_c, ctx.weather.humidity_pct);
        let mut air = self.air.clone();
        let tx = self.job_tx.clone();
        self.song_start = start + offset as i64 + CALL_DELAY_BLOCKS * ctx.n as i64;
        self.busy_until = i64::MAX;
        self.pending = true;
        crate::engine::background(move || {
            let job = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut rng = rng;
                match call {
                    Call::Bird(b) => {
                        let mut trng = type_rng;
                        song_bird(b, &mut trng, sr, pitch, &mut song)
                    }
                    Call::Frog(f) => call_frog(f, &mut rng, sr, pitch, &mut song),
                }
                let amp = level * rng.lognormal(0.0, 0.15) as f32 * (0.6 + 0.4 * a.min(1.0) as f32) / dist;
                song.iter_mut().for_each(|v| *v *= amp);
                if dist > AIR_MIN_M {
                    air.reset();
                    air.process(&mut song, temp, hum);
                }
                let rest = if slotted { 0 } else { (rng.range(0.5, 3.0) * sr) as i64 };
                (song, rest)
            }));
            let _ = tx.send(job.unwrap_or_else(|_| (Vec::new(), 0)));
        });
    }

    /// Collect the call being synthesized; `wait` blocks until it is done.
    fn collect(&mut self, wait: bool) {
        if !self.pending {
            return;
        }
        let got = if wait { self.job_rx.recv().ok() } else { self.job_rx.try_recv().ok() };
        if let Some((song, rest)) = got {
            self.song = song;
            self.busy_until = self.song_start + self.song.len() as i64 + rest;
            self.pending = false;
        } else if wait {
            // the job is gone (cannot happen: the sender is held here); drop the call
            self.pending = false;
            self.song.clear();
            self.busy_until = self.song_start;
        }
    }
}

impl Voice for CallerVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let dt = n as f64 / ctx.sr as f64;
        let act = self.act.step(ctx.weather, dt);
        let a = act[self.act_idx];
        let start = (ctx.t0 * ctx.sr as f64).round() as i64;
        if a > 1e-3 && start >= self.busy_until && self.slot.is_some() {
            let (per, phi) = self.slot.unwrap();
            let t0 = ctx.t0;
            let t1 = t0 + dt;
            let k0 = (t0 / per - phi).floor() as i64 - 1;
            for k in k0..k0 + 3 {
                let ku = k as u64;
                let jit = (crate::rng::uniform_at(self.key_call, ku, 1) - 0.5) * 0.08 * per;
                let ts = (k as f64 + phi) * per + jit;
                if ts >= t0 && ts < t1 {
                    let p_call = (a * self.per_min as f64 * per / 60.0).min(0.92);
                    if crate::rng::uniform_at(self.key_call, ku, 2) < p_call {
                        let off = (((ts - t0) * ctx.sr as f64) as usize).min(n - 1);
                        self.begin(ctx, a, start, off);
                    }
                    break;
                }
            }
        } else if a > 1e-3 && start >= self.busy_until {
            let mut crng = ctx.rng.stream(self.key_call, ctx.index);
            let lam = self.per_min as f64 / 60.0 * a * dt;
            let c = crng.poisson(lam);
            if c > 0 {
                let mut off = n;
                for _ in 0..c.min(64) {
                    off = off.min(crng.below(n as u64) as usize);
                }
                self.begin(ctx, a, start, off.min(n - 1));
            }
        }
        let end = start + n as i64;
        if self.pending {
            self.collect(self.song_start < end);
            if self.pending {
                return Emission::silent(Stem::Ambience);
            }
        }
        let s_end = self.song_start + self.song.len() as i64;
        if self.song.is_empty() || s_end <= start || self.song_start >= end {
            return Emission::silent(Stem::Ambience);
        }
        let lo = start.max(self.song_start);
        let hi = end.min(s_end);
        for t in lo..hi {
            out[(t - start) as usize] = self.song[(t - self.song_start) as usize];
        }
        Emission::point(self.az, self.el, self.dist, 0.05, Stem::Ambience)
    }

    fn reset(&mut self) {
        self.collect(true);
        self.act.reset();
        self.busy_until = -1;
        self.song_type = 0;
        self.song.clear();
        self.song_start = 0;
        self.air.reset();
    }
}

struct InsectState {
    sp: Insect,
    /// syllables (pulses) per chirp for this individual
    nsy: f64,
    jit: f64,
    cjit: f64,
    chirp_phase: f64,
    car_phase: f64,
    singing: bool,
    trill_len: f64,
    trill_gap: f64,
    level: f32,
    dist: f32,
    air: f32,
    amp: Ramp,
    noise_f: Biquad,
    g: [f32; N_AMBI],
}

impl InsectState {
    /// (chirps per second, duty, syllables per chirp, carrier Hz) at temperature temp_c.
    fn pattern(&self, temp_c: f64) -> (f64, f64, f64, f64) {
        let t = temp_c.max(12.0);
        match self.sp.rate(t) {
            None => {
                let period = self.trill_len + self.trill_gap;
                let n_syll = (self.sp.syll_rate(t) * self.trill_len).round().max(1.0);
                (1.0 / period, self.trill_len / period, n_syll, self.sp.carrier(t) * self.cjit)
            }
            Some(r) => (r.max(0.1) * self.jit, self.sp.duty(), self.nsy, self.sp.carrier(t) * self.cjit),
        }
    }
}

/// Every insect of the scene as one bank and one SH emission (self-encoded).
pub struct InsectBankVoice {
    act: Activity,
    ins: Vec<InsectState>,
    sr: f32,
    key: u64,
    air_key: (i32, i32),
    noise_g: f32,
    init: Vec<(f64, f64, bool)>,
    buf: Vec<f32>,
}

impl Voice for InsectBankVoice {
    fn name(&self) -> &str {
        "bio.insects"
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let w = ctx.weather;
        let sr = self.sr as f64;
        let dt = n as f64 / sr;
        let act = self.act.step(w, dt)[A_INSECTS];
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        for s in self.ins.iter_mut() {
            let u = rng.uniform();
            if s.singing && u < dt / 120.0 {
                s.singing = false;
            } else if !s.singing && u < dt / 20.0 * act.max(0.05) {
                s.singing = true;
            }
        }
        let temp = w.temperature_c as f64;
        let key = (w.temperature_c.round() as i32, (w.humidity_pct / 5.0).round() as i32);
        if key != self.air_key {
            self.air_key = key;
            for s in self.ins.iter_mut() {
                let (_, _, _, fc) = s.pattern(temp);
                s.air = air_gain(fc, s.dist, w.temperature_c, w.humidity_pct);
            }
        }
        let mut any = false;
        for s in &self.ins {
            let target = s.level * act as f32 * (s.singing as u8 as f32) / s.dist * s.air;
            if target > 0.0 || s.amp.v >= 1e-9 {
                any = true;
                break;
            }
        }
        if !any || self.buf.len() < n {
            for s in self.ins.iter_mut() {
                let (rate, _, _, fc) = s.pattern(temp);
                s.chirp_phase = (s.chirp_phase + rate * n as f64 / sr).rem_euclid(1.0);
                s.car_phase = (s.car_phase + 2.0 * PI * fc * n as f64 / sr).rem_euclid(2.0 * PI);
            }
            return Emission::silent(Stem::Ambience);
        }
        let inv_ng = 1.0 / self.noise_g;
        for s in self.ins.iter_mut() {
            let (rate, duty, n_syll, fc) = s.pattern(temp);
            let target = s.level * act as f32 * (s.singing as u8 as f32) / s.dist * s.air;
            let (g0, g1) = s.amp.to(target);
            if g0 < 1e-9 && g1 < 1e-9 {
                s.chirp_phase = (s.chirp_phase + rate * n as f64 / sr).rem_euclid(1.0);
                s.car_phase = (s.car_phase + 2.0 * PI * fc * n as f64 / sr).rem_euclid(2.0 * PI);
                continue;
            }
            let buf = &mut self.buf[..n];
            let mut cph = s.car_phase;
            let mut ph_last = s.chirp_phase;
            for (i, y) in buf.iter_mut().enumerate() {
                let ph = s.chirp_phase + rate * (i + 1) as f64 / sr;
                ph_last = ph;
                let frac = ph.rem_euclid(1.0);
                let sp_ = frac / duty * n_syll;
                let sf = sp_.rem_euclid(1.0);
                // Syllable (pulse) envelope of a stridulating cricket: the file-scraper escapement
                // drives the harp at its resonance, ~4.5-5 kHz in Gryllus, a nearly pure tone that
                // builds over the first ~third of the closing stroke and dies away, the pitch
                // sagging ~1-2 % as the wing slows (Bennet-Clark 1999, J. Exp. Biol. 202; Elliott &
                // Koch 1985). Pulses take ~0.55 of the syllable period (10-20 ms); the first
                // syllable of a chirp is weaker.
                const ON: f64 = 0.55;
                let env = if frac < duty && sf < ON {
                    let x = (sf / ON) as f32;
                    let e = if x < 0.3 {
                        (std::f32::consts::FRAC_PI_2 * x / 0.3).sin().powi(2)
                    } else {
                        (std::f32::consts::FRAC_PI_2 * (x - 0.3) / 0.7).cos().powi(2)
                    };
                    let si = sp_.floor() as usize;
                    e * [0.72f32, 1.0, 0.96, 0.88, 0.8, 0.75, 0.7, 0.66][si.min(7)]
                } else {
                    0.0
                };
                cph += 2.0 * PI / sr * fc * (1.0 - 0.018 * sf);
                let car = if s.sp.noisy() {
                    s.noise_f.process(rng.normal() as f32) * inv_ng
                } else {
                    let c = cph as f32;
                    c.sin() + 0.05 * (2.0 * c).sin()
                };
                *y = ramp_at(g0, g1, i, n) * env * car;
            }
            s.chirp_phase = ph_last.rem_euclid(1.0);
            s.car_phase = cph.rem_euclid(2.0 * PI);
            for (ch, g) in s.g.iter().enumerate() {
                if *g == 0.0 {
                    continue;
                }
                let o = &mut out[ch * n..(ch + 1) * n];
                for (ov, yv) in o.iter_mut().zip(buf.iter()) {
                    *ov += g * yv;
                }
            }
        }
        Emission::diffuse(Stem::Ambience)
    }

    fn reset(&mut self) {
        self.act.reset();
        self.air_key = (i32::MIN, i32::MIN);
        for (s, &(cp, carp, sing)) in self.ins.iter_mut().zip(&self.init) {
            s.chirp_phase = cp;
            s.car_phase = carp;
            s.singing = sing;
            s.amp = Ramp::default();
            s.noise_f.reset();
        }
    }
}

/// A cicada in a tree: pulsed noise buzz with slow swells (day, hot).
pub struct CicadaVoice {
    name: String,
    act: Activity,
    fc: f64,
    pulse: f64,
    swell_f: f64,
    phase0: f64,
    phase: f64,
    pphase: f64,
    filt: Sos,
    g: f32,
    amp: Ramp,
    level: f32,
    az: f32,
    el: f32,
    dist: f32,
    sr: f32,
    key: u64,
}

impl Voice for CicadaVoice {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.sr as f64;
        let a = self.act.step(ctx.weather, n as f64 / sr)[A_CICADAS];
        let target = self.level * a as f32 / self.dist;
        if target <= 0.0 && self.amp.v < 1e-9 {
            return Emission::silent(Stem::Ambience);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let w = ctx.weather;
        let air = air_gain(self.fc, self.dist, w.temperature_c, w.humidity_pct);
        let o = &mut out[..n];
        fill_gauss(&mut rng, o);
        self.filt.process_block(o);
        let (g0, g1) = self.amp.to(target);
        let inv = air / self.g;
        let mut sw = self.phase;
        let mut pp = self.pphase;
        for (i, v) in o.iter_mut().enumerate() {
            sw = self.phase + self.swell_f * (i + 1) as f64 / sr;
            pp = self.pphase + self.pulse * (i + 1) as f64 / sr;
            let swell = (1.4 * (2.0 * PI * sw).sin()).clamp(0.0, 1.0).powf(1.5);
            let pulses = 0.4 + 0.6 * (PI * pp.rem_euclid(1.0)).sin().powi(2);
            *v *= ramp_at(g0, g1, i, n) * (swell * pulses) as f32 * inv;
        }
        self.phase = sw.rem_euclid(1.0);
        self.pphase = pp.rem_euclid(1.0);
        Emission::point(self.az, self.el, self.dist, 0.15, Stem::Ambience)
    }

    fn reset(&mut self) {
        self.act.reset();
        self.phase = self.phase0;
        self.pphase = 0.0;
        self.filt.reset();
        self.amp = Ramp::default();
    }
}

/// The incoherent sum of many distant singers: band noise with a weak shared rhythm (diffuse).
/// The distant insect chorus: a field of singers all around, not one point. CHORUS_DIRS regions
/// (independent noise, filters and song phase each) on the horizon, so the chorus is diffuse and
/// decorrelated at the ears (binaural coherence ~ that of a diffuse field) instead of a mono
/// image inside the head.
const CHORUS_DIRS: usize = 6;

pub struct ChorusVoice {
    act: Activity,
    filts: Vec<Biquad>, // CHORUS_DIRS x carriers
    gains: Vec<f32>,
    rate_sp: Option<Insect>,
    level: f32,
    phase: f64,
    amp: Ramp,
    sr: f32,
    key: u64,
    tmp: Vec<f32>,
    acc: Vec<f32>,
    dir_y: [[f32; N_AMBI]; CHORUS_DIRS],
    ph_off: [f64; CHORUS_DIRS],
}

impl Voice for ChorusVoice {
    fn name(&self) -> &str {
        "bio.chorus"
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let n = ctx.n;
        let sr = self.sr as f64;
        let a = self.act.step(ctx.weather, n as f64 / sr)[A_INSECTS];
        let target = self.level * a as f32;
        if (target <= 0.0 && self.amp.v < 1e-9) || self.tmp.len() < n || self.acc.len() < n {
            return Emission::silent(Stem::Ambience);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let t = (ctx.weather.temperature_c as f64).max(12.0);
        let rate = match self.rate_sp.and_then(|s| s.rate(t)) {
            Some(r) => r,
            None => 0.3,
        }
        .max(0.2);
        let nc = self.gains.len().max(1);
        let k = 1.0 / ((nc * CHORUS_DIRS) as f32).sqrt();
        let (g0, g1) = self.amp.to(target);
        for d in 0..CHORUS_DIRS {
            let acc = &mut self.acc[..n];
            acc.iter_mut().for_each(|v| *v = 0.0);
            for (c, g) in self.gains.iter().enumerate() {
                let tmp = &mut self.tmp[..n];
                fill_gauss(&mut rng, tmp);
                self.filts[d * nc + c].process_block(tmp);
                let gg = k / g;
                for (ov, tv) in acc.iter_mut().zip(tmp.iter()) {
                    *ov += tv * gg;
                }
            }
            let p0 = self.phase + self.ph_off[d];
            for (i, v) in acc.iter_mut().enumerate() {
                let ph = p0 + rate * (i + 1) as f64 / sr;
                let am = 1.0 + 0.3 * (2.0 * PI * ph).sin();
                *v *= ramp_at(g0, g1, i, n) * am as f32;
            }
            for (ch, &g) in self.dir_y[d].iter().enumerate() {
                if g == 0.0 {
                    continue;
                }
                let o = &mut out[ch * n..(ch + 1) * n];
                for (ov, xv) in o.iter_mut().zip(acc.iter()) {
                    *ov += g * *xv;
                }
            }
        }
        self.phase = (self.phase + rate * n as f64 / sr).rem_euclid(1.0);
        Emission::diffuse(Stem::Ambience)
    }

    fn reset(&mut self) {
        self.act.reset();
        self.phase = 0.0;
        self.amp = Ramp::default();
        for f in &mut self.filts {
            f.reset();
        }
    }
}

fn perch(perches: &[&Patch], cfg: &mut Rng, h: f32) -> [f32; 3] {
    if !perches.is_empty() && cfg.uniform() < 0.8 {
        let p = perches[ri(cfg, 0, perches.len() as i64) as usize];
        let j = [cfg.gauss(0.0, 1.5), cfg.gauss(0.0, 1.5), cfg.gauss(0.0, 1.5)];
        return [p.position[0] + j[0] as f32, p.position[1] + j[1] as f32, p.position[2] + 0.5 * j[2] as f32];
    }
    let az = cfg.range(-PI, PI);
    let d = cfg.range(8.0, 60.0);
    let z = cfg.range(2.0, 12.0) as f32 - h;
    [(d * az.cos()) as f32, (d * az.sin()) as f32, z]
}

/// Crickets, cicadas, birds (incl. owls), frogs and the insect chorus of the scene
/// (place.biophony: region, birds / insects / frogs = auto | on | off).
pub fn build(scene: &Scene, patches: &[Patch], cfg: &BuildCfg) -> Vec<BoxedVoice> {
    let mut out: Vec<BoxedVoice> = Vec::new();
    let sr = cfg.sr as f32;
    let mut region = biophony_str(scene, "region").unwrap_or_default().trim().to_ascii_lowercase();
    if !REGIONS.contains(&region.as_str()) {
        region = "temperate_europe".into();
    }
    let region = region.as_str();
    let tropical = region == "tropical";
    let m_birds = mode_of(scene, "birds");
    let m_insects = mode_of(scene, "insects");
    let m_frogs = mode_of(scene, "frogs");
    let h = scene.listener.height_m;
    let mut rng = cfg.rng.get("bio.cfg", 0);
    let ponds: Vec<&Patch> = patches.iter().filter(|p| p.kind == PatchKind::Water || p.material == "puddle").collect();
    let has_water = m_frogs != Mode::Off && ponds.iter().any(|p| p.kind == PatchKind::Water);
    let act = Activity {
        tropical,
        lat: scene.weather.latitude_deg,
        m_birds,
        m_insects,
        m_frogs,
        has_water,
        rain_ema: 0.0,
        wet_memory: 0.0,
        dry_s: 3600.0,
        act: None,
    };

    let mut insects: Vec<InsectState> = Vec::new();
    let mut cicadas: Vec<CicadaVoice> = Vec::new();
    let mut chorus: Option<ChorusVoice> = None;
    if m_insects != Mode::Off {
        let species = region_insects(region);
        let n_ins = if tropical { ri(&mut rng, 10, 15) } else { ri(&mut rng, 6, 11) } as usize;
        for k in 0..n_ins {
            let sp = species[k % species.len()];
            let az = rng.range(-PI, PI);
            let d = rng.range(2.5, 25.0);
            let z = -h + if matches!(sp, Insect::TreeCricket | Insect::Katydid) { 1.0 } else { 0.1 };
            let pos = [(d * az.cos()) as f32, (d * az.sin()) as f32, z];
            let (paz, pel, dist) = point_of(pos);
            let jit = rng.range(0.95, 1.05);
            let cjit = rng.range(0.96, 1.04);
            let chirp_phase = rng.uniform();
            let car_phase = rng.range(0.0, 2.0 * PI);
            let singing = rng.uniform() < 0.7;
            let trill_len = rng.range(2.0, 6.0);
            let trill_gap = rng.range(0.6, 2.5);
            let level = db_spl(sp.level()) * rng.lognormal(0.0, 0.2) as f32;
            // field crickets (Gryllus) chirp 3-5 syllables (Alexander 1961; Bennet-Clark 1999)
            let nsy = if sp == Insect::FieldCricket { (3 + (k * 7 + 1) % 3) as f64 } else { sp.syll() };
            insects.push(InsectState {
                sp,
                nsy,
                jit,
                cjit,
                chirp_phase,
                car_phase,
                singing,
                trill_len,
                trill_gap,
                level,
                dist,
                air: 1.0,
                amp: Ramp::default(),
                noise_f: Biquad::new(bp(Insect::Katydid.carrier(20.0) as f32, 3.0, sr)),
                g: crate::spatial::spread_gains(paz, pel, 0.05).0,
            });
        }
        let carriers: Vec<f64> = species.iter().take(2).map(|s| s.carrier(20.0)).collect();
        let mut drng = cfg.rng.get("bio.chorus.dirs", 0);
        let az0 = drng.range(-PI, PI);
        let mut dir_y = [[0f32; N_AMBI]; CHORUS_DIRS];
        let mut ph_off = [0f64; CHORUS_DIRS];
        let mut filts: Vec<Biquad> = Vec::with_capacity(CHORUS_DIRS * carriers.len());
        for d in 0..CHORUS_DIRS {
            let az = az0 + 2.0 * PI * (d as f64 + drng.range(-0.25, 0.25)) / CHORUS_DIRS as f64;
            dir_y[d] = sh_gains(az as f32, drng.range(-0.05, 0.12) as f32, 0.3);
            ph_off[d] = drng.uniform();
            // each region's population sings a little sharp or flat (temperature / species spread)
            let detune = drng.range(0.97, 1.03);
            for &f in &carriers {
                filts.push(Biquad::new(bp((f * detune) as f32, 6.0, sr)));
            }
        }
        let gains: Vec<f32> = carriers.iter().map(|&f| noise_rms_gain(&[bp(f as f32, 6.0, sr)])).collect();
        chorus = Some(ChorusVoice {
            dir_y,
            ph_off,
            acc: vec![0.0; cfg.block],
            act: act.clone(),
            filts,
            gains,
            rate_sp: Some(species[0]),
            level: db_spl(if tropical { 46.0 } else { 36.0 }),
            phase: 0.0,
            amp: Ramp::default(),
            sr,
            key: cfg.rng.key("bio.chorus"),
            tmp: vec![0.0; cfg.block],
        });
        if CICADA_REGIONS.contains(&region) {
            for k in 0..2 {
                let az = rng.range(-PI, PI);
                let d = rng.range(10.0, 35.0);
                let pos = [(d * az.cos()) as f32, (d * az.sin()) as f32, 5.0 - h];
                let (paz, pel, dist) = point_of(pos);
                let fc = rng.range(4500.0, 6500.0);
                let pulse = rng.range(110.0, 180.0);
                let swell_f = 1.0 / rng.range(20.0, 40.0);
                let phase = rng.uniform();
                let secs = [bp(fc as f32, 2.5, sr), bp((fc * 1.1) as f32, 2.0, sr)];
                cicadas.push(CicadaVoice {
                    name: format!("bio.cicada{k}"),
                    act: act.clone(),
                    fc,
                    pulse,
                    swell_f,
                    phase0: phase,
                    phase,
                    pphase: 0.0,
                    filt: Sos::new(&secs),
                    g: noise_rms_gain(&secs),
                    amp: Ramp::default(),
                    level: db_spl(76.0),
                    az: paz,
                    el: pel,
                    dist,
                    sr,
                    key: cfg.rng.key(&format!("bio.cicada.cicada{k}")),
                });
            }
        }
    }

    let perches: Vec<&Patch> = patches
        .iter()
        .filter(|p| p.kind == PatchKind::Vegetation && p.param_str("species").unwrap_or("") != "grass")
        .collect();
    let mut birds: Vec<CallerVoice> = Vec::new();
    let mut owls: Vec<CallerVoice> = Vec::new();
    if m_birds != Mode::Off {
        let palette = region_birds(region);
        let n_b = match region {
            "arid" => 3,
            "boreal" => 5,
            "tropical" => 11,
            _ => 9,
        };
        for k in 0..n_b {
            let tpl = palette[k % palette.len()];
            let pos = perch(&perches, &mut rng, h);
            let name = format!("bird{k}.{}", tpl.name());
            birds.push(CallerVoice::new(name, "bio.bird", Call::Bird(tpl), pos, &act, A_BIRDS, &mut rng, cfg));
        }
        if OWL_REGIONS.contains(&region) {
            let az = rng.range(-PI, PI);
            let d = rng.range(40.0, 120.0);
            let pos = [(d * az.cos()) as f32, (d * az.sin()) as f32, 8.0 - h];
            owls.push(CallerVoice::new("owl0".into(), "bio.owl", Call::Bird(Bird::Owl), pos, &act, A_OWLS, &mut rng, cfg));
        }
    }

    let mut frogs: Vec<CallerVoice> = Vec::new();
    if m_frogs != Mode::Off {
        let species = region_frogs(region);
        let n_f = if tropical { ri(&mut rng, 12, 18) } else { ri(&mut rng, 6, 12) } as usize;
        for k in 0..n_f {
            let sp = species[k % species.len()];
            let pos = if !ponds.is_empty() {
                let src = ponds[ri(&mut rng, 0, ponds.len() as i64) as usize];
                let j0 = rng.gauss(0.0, 2.0) as f32;
                let j1 = rng.gauss(0.0, 2.0) as f32;
                [src.position[0] + j0, src.position[1] + j1, -h + 0.05]
            } else {
                let az = rng.range(-PI, PI);
                let d = rng.range(6.0, 40.0);
                [(d * az.cos()) as f32, (d * az.sin()) as f32, -h + 0.05]
            };
            let name = format!("frog{k}.{}", sp.name());
            frogs.push(CallerVoice::new(name, "bio.frog", Call::Frog(sp), pos, &act, A_FROGS, &mut rng, cfg));
        }
        // chorus slots: same-species short callers share a clock and spread their phases evenly, so
        // they alternate and answer each other instead of overlapping at random
        let mut srng = cfg.rng.get("bio.frog.slots", 0);
        for &sp in species {
            if !matches!(sp, Frog::Peeper | Frog::Tree | Frog::Chorus | Frog::Coqui | Frog::Common) {
                continue;
            }
            let idx: Vec<usize> =
                frogs.iter().enumerate().filter(|(_, f)| matches!(f.call, Call::Frog(x) if x == sp)).map(|(i, _)| i).collect();
            let m = idx.len();
            if m == 0 || frogs[idx[0]].slot.is_some() {
                continue;
            }
            let per = 60.0 / sp.spec().1 as f64 * srng.range(0.9, 1.1);
            for (j, &i) in idx.iter().enumerate() {
                let phi = ((j as f64 + 0.5 + srng.range(-0.15, 0.15)) / m as f64).rem_euclid(1.0);
                frogs[i].slot = Some((per, phi));
            }
        }
    }

    if !insects.is_empty() {
        let init = insects.iter().map(|s| (s.chirp_phase, s.car_phase, s.singing)).collect();
        let ng = noise_rms_gain(&[bp(Insect::Katydid.carrier(20.0) as f32, 3.0, sr)]);
        out.push(Box::new(InsectBankVoice {
            act: act.clone(),
            ins: insects,
            sr,
            key: cfg.rng.key("bio.insects"),
            air_key: (i32::MIN, i32::MIN),
            noise_g: ng,
            init,
            buf: vec![0.0; cfg.block],
        }));
    }
    for v in cicadas {
        out.push(Box::new(v));
    }
    for v in birds.into_iter().chain(owls).chain(frogs) {
        out.push(Box::new(v));
    }
    if let Some(c) = chorus {
        out.push(Box::new(c));
    }
    out
}

// --------------------------------------------------------------------------
// ground effect (raingen.physics.atmosphere.ground_effect_sos)
// --------------------------------------------------------------------------
fn flow_resistivity_kpa(ground: &str) -> f64 {
    match ground.trim().to_ascii_lowercase().as_str() {
        "snow_fresh" => 15.0,
        "snow_old" => 60.0,
        "snow" => 30.0,
        "forest_floor" | "leaves" => 40.0,
        "needles" => 30.0,
        "grass" | "lawn" => 200.0,
        "soil" => 500.0,
        "mud" => 800.0,
        "sand" | "gravel" => 400.0,
        "asphalt" => 20000.0,
        "concrete" | "tile" => 50000.0,
        "water" | "ice" | "puddle" => f64::INFINITY,
        _ => 200.0,
    }
}

fn delany_bazley_z(f: f64, sigma_kpa: f64) -> Complex64 {
    let x = (f / sigma_kpa).clamp(8.3, 833.0);
    Complex64::new(1.0 + 9.08 * x.powf(-0.75), 11.9 * x.powf(-0.73))
}

const WN: usize = 32;

/// Weideman (1994) rational approximation of the Faddeeva function, N = 32.
fn weideman() -> &'static ([f64; WN], f64) {
    static W: OnceLock<([f64; WN], f64)> = OnceLock::new();
    W.get_or_init(|| {
        let m = 2 * WN;
        let m2 = 2 * m;
        let l = (WN as f64 / 2f64.sqrt()).sqrt();
        let mut f = vec![0.0f64; m2];
        for (idx, k) in (-(m as i64) + 1..=(m as i64) - 1).enumerate() {
            let theta = k as f64 * PI / m as f64;
            let t = l * (theta / 2.0).tan();
            f[idx + 1] = (-t * t).exp() * (l * l + t * t);
        }
        let g: Vec<f64> = (0..m2).map(|i| f[(i + m2 / 2) % m2]).collect();
        let mut a = [0.0f64; WN];
        for j in 1..=WN {
            let mut s = 0.0;
            for (nn, gv) in g.iter().enumerate() {
                s += gv * (2.0 * PI * (j * nn) as f64 / m2 as f64).cos();
            }
            a[WN - j] = s / m2 as f64;
        }
        (a, l)
    })
}

fn wofz_upper(z: Complex64) -> Complex64 {
    let (a, l) = weideman();
    let i = Complex64::new(0.0, 1.0);
    let lz = Complex64::new(*l, 0.0) - i * z;
    let zz = (Complex64::new(*l, 0.0) + i * z) / lz;
    let mut p = Complex64::new(a[0], 0.0);
    for c in a.iter().skip(1) {
        p = p * zz + *c;
    }
    p * 2.0 / (lz * lz) + Complex64::new(1.0 / PI.sqrt(), 0.0) / lz
}

fn wofz(z: Complex64) -> Complex64 {
    if z.im >= 0.0 {
        wofz_upper(z)
    } else {
        (-(z * z)).exp() * 2.0 - wofz_upper(-z)
    }
}

/// Complex direct + reflected field relative to the direct path, 1 + Q (r1/r2) e^{ik(r2-r1)}.
fn ground_transfer(f: f64, d: f64, hs: f64, hr: f64, sigma: f64, temp_c: f64) -> Complex64 {
    let f = f.max(1.0);
    let d = d.max(0.1);
    let (hs, hr) = (hs.max(0.0), hr.max(0.0));
    let r1 = d.hypot(hs - hr);
    let r2 = d.hypot(hs + hr);
    let c = 331.3 * (1.0 + temp_c / 273.15).sqrt();
    let k = 2.0 * PI * f / c;
    let one = Complex64::new(1.0, 0.0);
    let q = if !sigma.is_finite() {
        one
    } else {
        let sin_psi = if r2 > 0.0 { (hs + hr) / r2 } else { 0.0 };
        let z = delany_bazley_z(f, sigma);
        let rp = (z * sin_psi - 1.0) / (z * sin_psi + 1.0);
        let beta = one / z;
        let w = Complex64::new(0.5, 0.5) * (k * r2).sqrt() * (beta + sin_psi);
        let mut fw = one + Complex64::new(0.0, PI.sqrt()) * w * wofz(w);
        if !(fw.re.is_finite() && fw.im.is_finite()) {
            fw = one;
        }
        let mut q = rp + (one - rp) * fw;
        let mag = q.norm();
        if mag > 1.2 {
            q = q / mag.max(1e-12) * 1.2;
        }
        q
    };
    one + q * (r1 / r2) * Complex64::new(0.0, k * (r2 - r1)).exp()
}

/// Low shelf + first interference dip fitted to the two-ray ground transfer (normalised to the
/// incoherent sum, so the response is flat above the first dip).
fn ground_effect_sections(d: f64, hs: f64, hr: f64, ground: &str, sr: f32) -> Vec<BiquadCoefs> {
    let temp_c = 15.0;
    if d < 1.0 || hs + hr <= 0.0 {
        return Vec::new();
    }
    let srf = sr as f64;
    let fmax = 20000f64.min(0.45 * srf);
    let nf = 240;
    let f: Vec<f64> = (0..nf).map(|i| (20f64.ln() + (fmax.ln() - 20f64.ln()) * i as f64 / (nf - 1) as f64).exp()).collect();
    let sigma = flow_resistivity_kpa(ground);
    let r1 = d.hypot(hs - hr);
    let r2 = d.hypot(hs + hr);
    let target: Vec<f64> = f
        .iter()
        .map(|&fv| {
            let h = ground_transfer(fv, d, hs, hr, sigma, temp_c);
            let q_mag = if sigma.is_finite() {
                let z = delany_bazley_z(fv, sigma);
                let sin_psi = (hs + hr) / r2;
                ((z * sin_psi - 1.0) / (z * sin_psi + 1.0)).norm()
            } else {
                1.0
            };
            let incoherent = 10.0 * (1.0 + (q_mag * r1 / r2).powi(2)).log10();
            20.0 * (h.norm() + 1e-12).log10() - incoherent
        })
        .collect();
    let band: Vec<bool> = f.iter().map(|&fv| (40.0..=4000f64.min(0.4 * srf)).contains(&fv)).collect();
    let mut rows: Vec<BiquadCoefs> = Vec::new();
    let mut f_dip: Option<f64> = None;
    if band.iter().any(|&b| b) {
        let first = (0..nf).find(|&i| band[i] && target[i] < -3.0);
        let i_min = match first {
            Some(mut i) => {
                while i + 1 < nf && band[i + 1] && target[i + 1] < target[i] {
                    i += 1;
                }
                i
            }
            None => {
                let mut best = usize::MAX;
                for i in 0..nf {
                    if band[i] && (best == usize::MAX || target[i] < target[best]) {
                        best = i;
                    }
                }
                best
            }
        };
        let depth = target[i_min].max(-20.0);
        if depth < -1.5 {
            let fd = f[i_min];
            f_dip = Some(fd);
            let half = 0.5 * depth;
            let mut lo = i_min;
            while lo > 0 && target[lo] < half {
                lo -= 1;
            }
            let mut hi = i_min;
            while hi < nf - 1 && target[hi] < half {
                hi += 1;
            }
            let bw = (f[hi] - f[lo]).max(1e-3);
            let q = (fd / bw).clamp(0.4, 3.0);
            rows.push(BiquadCoefs::peaking(sr, fd as f32, q as f32, depth as f32));
        }
    }
    let f_shelf = f_dip.map(|v| 0.6 * v).unwrap_or(150.0);
    let lim = (0.5 * f_dip.unwrap_or(200.0)).max(25.0);
    let sel: Vec<f64> = f.iter().zip(&target).filter(|(fv, _)| **fv <= lim).map(|(_, t)| *t).collect();
    if !sel.is_empty() {
        let lf_db = (sel.iter().sum::<f64>() / sel.len() as f64).clamp(-6.0, 6.0);
        if lf_db.abs() > 0.5 {
            rows.insert(0, BiquadCoefs::low_shelf(sr, f_shelf.max(30.0) as f32, std::f32::consts::FRAC_1_SQRT_2, lf_db as f32));
        }
    }
    rows
}

// --------------------------------------------------------------------------
// traffic
// --------------------------------------------------------------------------
const DAILY_FLOW: [f64; 24] = [
    0.08, 0.05, 0.04, 0.04, 0.06, 0.15, 0.45, 0.85, 1.0, 0.8, 0.7, 0.72, 0.75, 0.72, 0.72, 0.8, 0.95, 1.0, 0.85, 0.6,
    0.45, 0.35, 0.25, 0.15,
];

pub fn daily_flow_factor(hour: f64) -> f64 {
    let h = hour.rem_euclid(24.0);
    let i = h.floor() as usize % 24;
    let f = h - h.floor();
    DAILY_FLOW[i] * (1.0 - f) + DAILY_FLOW[(i + 1) % 24] * f
}

/// Road layout and wet-road film shared (by copy) by the bed and the pass-by voices.
#[derive(Clone, Debug)]
struct TrafficState {
    level: f64,
    road_d: f64,
    v_kmh: f64,
    q_max: f64,
    pass_bys: bool,
    axis: f64,
    side: f64,
    road_film: f64,
}

impl TrafficState {
    fn wetness(&mut self, w: &Weather, dt: f64) -> f64 {
        let liquid = if w.precip_type == precip::SNOW || w.precip_type == precip::NONE || w.precip_type == precip::GRAUPEL {
            0.0
        } else {
            w.rain_rate_mm_h as f64
        };
        let tau = 900.0 * (1.0 + (w.humidity_pct as f64 - 60.0).max(0.0) / 40.0);
        if self.road_film < 0.0 {
            self.road_film = liquid / 3600.0 * tau;
        }
        self.road_film += (liquid / 3600.0 - self.road_film / tau) * dt;
        let wet = ss(self.road_film, 0.002, 0.05);
        wet.max(w.wetness_named("asphalt") as f64).max(0.8 * w.wetness_named("concrete") as f64).min(1.0)
    }

    fn flow(&self, hour: f32) -> f64 {
        self.q_max * self.level * daily_flow_factor(hour as f64)
    }
}

/// The distant network bed (line source) with wet-tyre hiss, ground effect and air absorption.
pub struct TrafficBedVoice {
    st: TrafficState,
    st0: TrafficState,
    bed: Sos,
    bed_g: f32,
    hiss: Sos,
    hiss_g: f32,
    air: AirPath,
    bed_amp: Ramp,
    hiss_amp: Ramp,
    ou: f64,
    az: f32,
    el: f32,
    spread: f32,
    key: u64,
    tmp: Vec<f32>,
}

impl Voice for TrafficBedVoice {
    fn name(&self) -> &str {
        "traffic.bed"
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n;
        let dt = n as f64 / ctx.sr as f64;
        let wet = self.st.wetness(w, dt);
        let q = self.st.flow(w.hour);
        if (q < 0.5 && self.bed_amp.v < 1e-9) || self.tmp.len() < n {
            return Emission::silent(Stem::Ambience);
        }
        let mut rng = ctx.rng.stream(self.key, ctx.index);
        let st = &self.st;
        let l_bed = 10.0 * q.max(1.0).log10() + 33.0 + 30.0 * (st.v_kmh / 50.0).log10()
            - 10.0 * (st.road_d.max(1.0) / 10.0).log10()
            - if st.pass_bys { 8.0 } else { 0.0 };
        let a = (-dt / 15.0).exp();
        self.ou = a * self.ou + (1.0 - a * a).sqrt() * rng.normal();
        let depth = (2.0 / (q / 60.0 + 1.0).sqrt()).min(0.8);
        let qs = ss(q, 0.5, 5.0);
        let bed_t = db_spl(l_bed as f32) as f64 * qs * (1.0 + depth * 0.5 * self.ou).max(0.1);
        let hiss_t = db_spl((l_bed - 10.0 + 14.0 * wet) as f32) as f64 * ss(wet, 0.03, 0.4) * qs;
        let o = &mut out[..n];
        fill_gauss(&mut rng, o);
        self.bed.process_block(o);
        let (b0, b1) = self.bed_amp.to(bed_t as f32);
        let ib = 1.0 / self.bed_g;
        for (i, v) in o.iter_mut().enumerate() {
            *v *= ramp_at(b0, b1, i, n) * ib;
        }
        let tmp = &mut self.tmp[..n];
        fill_gauss(&mut rng, tmp);
        self.hiss.process_block(tmp);
        let (h0, h1) = self.hiss_amp.to(hiss_t as f32);
        let ih = 1.0 / self.hiss_g;
        for (i, (v, t)) in o.iter_mut().zip(tmp.iter()).enumerate() {
            *v += ramp_at(h0, h1, i, n) * t * ih;
        }
        self.air.process(o, w.temperature_c, w.humidity_pct);
        Emission::point(self.az, self.el, self.st.road_d as f32, self.spread, Stem::Ambience)
    }

    fn reset(&mut self) {
        self.st = self.st0.clone();
        self.bed.reset();
        self.hiss.reset();
        self.air.reset();
        self.bed_amp = Ramp::default();
        self.hiss_amp = Ramp::default();
        self.ou = 0.0;
    }
}

const MAX_CARS: usize = 16;

#[derive(Clone, Copy)]
enum VehicleKind {
    Car,
    Truck,
    Moto,
}

impl VehicleKind {
    /// (tyre dB, engine dB at 7.5 m and 50 km/h, firing order factor, rpm range)
    fn spec(self) -> (f64, f64, f64, (f64, f64)) {
        match self {
            VehicleKind::Car => (70.0, 62.0, 2.0, (1700.0, 2800.0)),
            VehicleKind::Truck => (76.0, 76.0, 3.0, (1100.0, 1700.0)),
            VehicleKind::Moto => (68.0, 74.0, 1.0, (3500.0, 6000.0)),
        }
    }
}

struct PassBy {
    active: bool,
    key: u64,
    v: f64,
    d_min: f64,
    dir: f64,
    t_half: f64,
    t_c: f64,
    end_sample: i64,
    pc: [f64; 2],
    e: [f64; 2],
    z: f64,
    tyre: f32,
    engine: f32,
    f_e: f64,
    e_phase: f64,
    tyre_f: Sos,
    hiss_f: Sos,
    g: [f32; N_AMBI],
    g_init: bool,
}

/// Individual pass-bys on nearby roads (tyre noise, wet spray hiss, Doppler-shifted engine),
/// self-encoded to SH so any number of concurrent vehicles share one emission.
pub struct TrafficCarsVoice {
    st: TrafficState,
    st0: TrafficState,
    cars: Vec<PassBy>,
    tyre_g: f32,
    hiss_g: f32,
    listener_h: f64,
    sr: f32,
    key_spawn: u64,
    key_car: u64,
    offs: Vec<usize>,
    y: Vec<f32>,
    n1: Vec<f32>,
    n2: Vec<f32>,
}

impl TrafficCarsVoice {
    fn spawn(&mut self, rng: &mut Rng, entry_sample: i64, block: u64, off: usize) {
        let slot = match self.cars.iter().position(|c| !c.active) {
            Some(s) => s,
            None => return,
        };
        let sr = self.sr as f64;
        let st = &self.st;
        let u = rng.uniform();
        let kind = if u < 0.86 {
            VehicleKind::Car
        } else if u < 0.96 {
            VehicleKind::Truck
        } else {
            VehicleKind::Moto
        };
        let (tyre_db, eng_db, order, rpm) = kind.spec();
        let v = (st.v_kmh * rng.lognormal(0.0, 0.12)).max(10.0) / 3.6;
        let d_min = (st.road_d + rng.range(-1.5, 1.5)).max(1.5);
        let dir = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        let half = 60f64.max(10.0 * d_min);
        let t_half = half / v;
        let t_c = entry_sample as f64 / sr + t_half;
        let az_c = st.axis + st.side * 0.5 * PI;
        let vk = v * 3.6;
        let f_e = rng.range(rpm.0, rpm.1) / 60.0 * order;
        let e_phase = rng.range(0.0, 2.0 * PI);
        let car = &mut self.cars[slot];
        car.active = true;
        car.key = hash3(self.key_car, block, off as u64);
        car.v = v;
        car.d_min = d_min;
        car.dir = dir;
        car.t_half = t_half;
        car.t_c = t_c;
        car.end_sample = ((t_c + t_half) * sr) as i64;
        car.pc = [d_min * az_c.cos(), d_min * az_c.sin()];
        car.e = [st.axis.cos(), st.axis.sin()];
        car.z = 0.5 - self.listener_h;
        car.tyre = (db_spl((tyre_db + 30.0 * (vk / 50.0).log10()) as f32) as f64 * 7.5) as f32;
        car.engine = (db_spl((eng_db + 10.0 * (vk / 50.0).log10()) as f32) as f64 * 7.5) as f32;
        car.f_e = f_e;
        car.e_phase = e_phase;
        car.tyre_f.reset();
        car.hiss_f.reset();
        car.g_init = false;
    }
}

impl Voice for TrafficCarsVoice {
    fn name(&self) -> &str {
        "traffic.cars"
    }

    fn channels(&self) -> usize {
        N_AMBI
    }

    fn render(&mut self, ctx: &BlockCtx, out: &mut [f32]) -> Emission {
        let w = ctx.weather;
        let n = ctx.n;
        let sr = self.sr as f64;
        let dt = n as f64 / sr;
        let wet = self.st.wetness(w, dt);
        let q = self.st.flow(w.hour);
        let any_active = self.cars.iter().any(|c| c.active);
        if (q < 0.5 && !any_active) || self.y.len() < n {
            return Emission::silent(Stem::Ambience);
        }
        let start = (ctx.t0 * sr).round() as i64;
        let mut crng = ctx.rng.stream(self.key_spawn, ctx.index);
        let mut offs = std::mem::take(&mut self.offs);
        poisson_offsets(&mut crng, (q / 3600.0) as f32, n, self.sr, &mut offs);
        for &off in &offs {
            self.spawn(&mut crng, start + off as i64, ctx.index, off);
        }
        self.offs = offs;
        let c = 331.3 * (1.0 + w.temperature_c as f64 / 273.15).sqrt();
        let hiss_g = 10f32.powf((-10.0 + 14.0 * wet as f32) / 20.0) * smoothstep(wet as f32, 0.03, 0.3);
        let mut emitted = false;
        for car in self.cars.iter_mut() {
            if !car.active {
                continue;
            }
            if start + n as i64 <= ((car.t_c - car.t_half) * sr) as i64 {
                continue;
            }
            let mut rng = Rng::from_key(combine(car.key, 0x7472_6166), ctx.index);
            let y = &mut self.y[..n];
            let n1 = &mut self.n1[..n];
            let n2 = &mut self.n2[..n];
            fill_gauss(&mut rng, n1);
            car.tyre_f.process_block(n1);
            fill_gauss(&mut rng, n2);
            car.hiss_f.process_block(n2);
            let it = 1.0 / self.tyre_g;
            let ih = hiss_g / self.hiss_g;
            let mut ph = car.e_phase;
            let mut x_mid = 0.0;
            let mut r_mid = 1.0;
            for i in 0..n {
                let t = (start + i as i64) as f64 / sr - car.t_c;
                let x = car.dir * car.v * t;
                let r = (car.d_min * car.d_min + x * x + car.z * car.z).sqrt();
                if i == n / 2 {
                    x_mid = x;
                    r_mid = r;
                }
                let live = if t.abs() <= car.t_half { 1.0 } else { 0.0 };
                let edge = (car.t_half - t.abs()).clamp(0.0, 1.0);
                let g = live * edge / r;
                let r_dot = x * car.dir * car.v / r;
                let doppler = c / (c + r_dot);
                ph += 2.0 * PI * car.f_e * doppler / sr;
                let eng = ph.sin() + 0.6 * (2.0 * ph + 0.3).sin() + 0.35 * (3.0 * ph + 1.1).sin() + 0.2 * (4.0 * ph + 2.0).sin();
                let tyre = n1[i] * it + n2[i] * ih;
                y[i] = (g * (car.tyre as f64 * tyre as f64 + car.engine as f64 * eng / 1.3)) as f32;
            }
            car.e_phase = ph.rem_euclid(2.0 * PI);
            let p = [car.pc[0] + x_mid * car.e[0], car.pc[1] + x_mid * car.e[1]];
            let az = p[1].atan2(p[0]) as f32;
            let el = car.z.atan2(p[0].hypot(p[1])) as f32;
            let _ = r_mid;
            let g1 = crate::spatial::spread_gains(az, el, 0.05).0;
            if !car.g_init {
                car.g = g1;
                car.g_init = true;
            }
            encode_ramped(out, y, &car.g, &g1);
            car.g = g1;
            emitted = true;
        }
        let next = start + n as i64;
        for car in self.cars.iter_mut() {
            if car.active && next >= car.end_sample {
                car.active = false;
            }
        }
        if emitted {
            Emission::diffuse(Stem::Ambience)
        } else {
            Emission::silent(Stem::Ambience)
        }
    }

    fn reset(&mut self) {
        self.st = self.st0.clone();
        for c in &mut self.cars {
            c.active = false;
            c.tyre_f.reset();
            c.hiss_f.reset();
        }
    }
}

/// Road traffic: a distant bed with a daily flow curve and wet-tyre hiss, plus individual pass-bys
/// on nearby roads (streets, suburbs, a parked car by the kerb). Nothing when place.biophony.traffic
/// is 0 (the default).
pub fn build_traffic(scene: &Scene, cfg: &BuildCfg) -> Vec<BoxedVoice> {
    let mut out: Vec<BoxedVoice> = Vec::new();
    let level = (biophony_num(scene, "traffic") as f64).clamp(0.0, 1.0);
    if !(level > 0.0) {
        return out;
    }
    let sr = cfg.sr as f32;
    let preset = scene.place.preset.to_ascii_lowercase();
    let has = |keys: &[&str]| keys.iter().any(|k| preset.contains(k));
    let mut rng = cfg.rng.get("traffic.cfg", 0);
    let h = scene.listener.height_m as f64;
    let street = has(&["city", "street", "downtown", "urban"]);
    let (road_d, v_kmh, q_max, pass_bys, axis) = if street {
        (rng.range(3.5, 6.0), 40.0, 900.0, true, 0.0)
    } else if preset.contains("alley") {
        (rng.range(15.0, 25.0), 35.0, 600.0, true, 0.5 * PI)
    } else if scene.place.shelter == Shelter::Car {
        (4.5, 50.0, 400.0, true, 0.0)
    } else if has(&["suburb", "village", "town", "residential", "garden", "backyard", "park"]) {
        let d = rng.range(20.0, 60.0);
        (d, 50.0, 300.0, true, rng.range(-PI, PI))
    } else {
        let d = rng.range(250.0, 900.0);
        (d, 100.0, 1800.0, false, rng.range(-PI, PI))
    };
    let side = if street { -1.0 } else if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
    let az_c = axis + side * 0.5 * PI;
    let ground = if scene.place.ground.trim().is_empty() { "grass".to_string() } else { scene.place.ground.clone() };
    let st = TrafficState { level, road_d, v_kmh, q_max, pass_bys, axis, side, road_film: -1.0 };

    let mut shape = slope_sections(-3.0, 40.0, 16000.0, sr);
    shape.push(peak_eq(900.0, 6.0, 0.8, sr));
    shape.push(peak_eq(90.0, 4.0, 1.0, sr));
    shape.push(hp2(25.0, sr));
    let ge = ground_effect_sections(road_d, 0.5, h, &ground, sr);
    let bed_g = noise_rms_gain(&shape);
    let mut bed_secs = shape.clone();
    bed_secs.extend_from_slice(&ge);
    let hiss_base = [hp2(1500.0, sr), lp2(8000.0, sr)];
    let hiss_g = noise_rms_gain(&hiss_base);
    let mut hiss_secs = hiss_base.to_vec();
    hiss_secs.extend_from_slice(&ge);
    out.push(Box::new(TrafficBedVoice {
        st: st.clone(),
        st0: st.clone(),
        bed: Sos::new(&bed_secs),
        bed_g,
        hiss: Sos::new(&hiss_secs),
        hiss_g,
        air: AirPath::new(road_d as f32, sr, 8.0),
        bed_amp: Ramp::default(),
        hiss_amp: Ramp::default(),
        ou: 0.0,
        az: az_c as f32,
        el: (0.5 - h).atan2(road_d) as f32,
        spread: if road_d < 100.0 { 0.75 } else { 0.5 },
        key: cfg.rng.key("traffic.bed"),
        tmp: vec![0.0; cfg.block],
    }));

    if pass_bys {
        let mut tyre = vec![peak_eq(1000.0, 8.0, 0.7, sr)];
        tyre.extend(slope_sections(-4.0, 100.0, 12000.0, sr));
        tyre.push(hp2(60.0, sr));
        let hiss = [hp2(1800.0, sr), lp2(9000.0, sr)];
        let cars = (0..MAX_CARS)
            .map(|_| PassBy {
                active: false,
                key: 0,
                v: 10.0,
                d_min: 5.0,
                dir: 1.0,
                t_half: 1.0,
                t_c: 0.0,
                end_sample: 0,
                pc: [0.0; 2],
                e: [1.0, 0.0],
                z: 0.0,
                tyre: 0.0,
                engine: 0.0,
                f_e: 50.0,
                e_phase: 0.0,
                tyre_f: Sos::new(&tyre),
                hiss_f: Sos::new(&hiss),
                g: [0.0; N_AMBI],
                g_init: false,
            })
            .collect();
        out.push(Box::new(TrafficCarsVoice {
            st: st.clone(),
            st0: st,
            cars,
            tyre_g: noise_rms_gain(&tyre),
            hiss_g: noise_rms_gain(&hiss),
            listener_h: h,
            sr,
            key_spawn: cfg.rng.key("traffic.spawn"),
            key_car: cfg.rng.key("traffic.car"),
            offs: Vec::with_capacity(32),
            y: vec![0.0; cfg.block],
            n1: vec![0.0; cfg.block],
            n2: vec![0.0; cfg.block],
        }));
    }
    out
}
