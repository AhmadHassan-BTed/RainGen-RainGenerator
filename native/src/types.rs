//! Shared data contracts, mirroring `raingen/core/types.py`, `raingen/scene/schema.py` and
//! `raingen/synth/materials.py`.
//!
//! Every struct deserializes straight from the JSON Python produces:
//!   * `Weather`   <- `json.dumps(dataclasses.asdict(world.step(...)))`
//!   * `Patch`     <- `json.dumps([dataclasses.asdict(p) for p in build_layout(scene)])`
//!   * `Scene`     <- `scene.to_json()`
//!   * `MaterialTable` <- `json.dumps({k: dataclasses.asdict(m) for k, m in MATERIALS.items()})`
//! Field names are the Python dataclass names. Enumerated strings (precip type, season,
//! stability, shelter, patch kind, material kind) are parsed once into small enums / u8 codes so
//! voices never compare strings per block. Unknown JSON keys are ignored everywhere.

use serde::de::{self, Deserializer};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fmt;

// ---------------------------------------------------------------------------
// engine constants
// ---------------------------------------------------------------------------
pub const SAMPLE_RATE: u32 = 48_000;
pub const BLOCK: usize = 1024;
pub const AMBI_ORDER: usize = 3;
pub const N_AMBI: usize = (AMBI_ORDER + 1) * (AMBI_ORDER + 1); // 16
pub const N_STEMS: usize = 7;
pub const N_CLASSES: usize = 5;

/// Reference pressure of dB SPL; digital 1.0 = 1 Pa (see raingen/core/constants.py).
pub const P_REF: f32 = 20e-6;
pub const RHO_WATER: f32 = 998.0;
pub const RHO_AIR: f32 = 1.204;
pub const P0: f32 = 101_325.0;
pub const GAMMA_AIR: f32 = 1.4;
pub const SIGMA_WATER: f32 = 0.0728;
pub const G: f32 = 9.81;

#[inline]
pub fn db_spl_to_pa(db: f32) -> f32 {
    P_REF * 10f32.powf(db / 20.0)
}

#[inline]
pub fn pa_to_db_spl(pa: f32) -> f32 {
    let pa = pa.abs();
    if pa > 0.0 {
        20.0 * (pa / P_REF).log10()
    } else {
        f32::NEG_INFINITY
    }
}

#[inline]
pub fn speed_of_sound(temperature_c: f32) -> f32 {
    331.3 * (1.0 + temperature_c / 273.15).sqrt()
}

/// Compass bearing (deg true, clockwise) -> listener azimuth (rad, CCW, 0 = front), in (-pi, pi].
pub fn bearing_to_azimuth(bearing_deg: f32, heading_deg: f32) -> f32 {
    let a = (-(bearing_deg - heading_deg)).to_radians() as f64;
    let tau = std::f64::consts::TAU;
    let mut r = a - tau * (a / tau).round();
    if r <= -std::f64::consts::PI {
        r += tau;
    }
    r as f32
}

// ---------------------------------------------------------------------------
// stems
// ---------------------------------------------------------------------------
/// Output stems, in raingen.core.constants.STEMS order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Stem {
    #[default]
    RainNear = 0,
    RainFar = 1,
    Wind = 2,
    Thunder = 3,
    Aftermath = 4,
    Ambience = 5,
    Interior = 6,
}

pub const STEM_NAMES: [&str; N_STEMS] =
    ["rain_near", "rain_far", "wind", "thunder", "aftermath", "ambience", "interior"];

impl Stem {
    pub const ALL: [Stem; N_STEMS] = [
        Stem::RainNear,
        Stem::RainFar,
        Stem::Wind,
        Stem::Thunder,
        Stem::Aftermath,
        Stem::Ambience,
        Stem::Interior,
    ];
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        STEM_NAMES[self as usize]
    }
    /// Unknown names route to `Ambience`, like BlockRenderer._stem_for.
    pub fn from_name(s: &str) -> Stem {
        match s {
            "rain_near" => Stem::RainNear,
            "rain_far" => Stem::RainFar,
            "wind" => Stem::Wind,
            "thunder" => Stem::Thunder,
            "aftermath" => Stem::Aftermath,
            "ambience" => Stem::Ambience,
            "interior" => Stem::Interior,
            _ => Stem::Ambience,
        }
    }
    pub fn from_index(i: usize) -> Stem {
        Stem::ALL[i.min(N_STEMS - 1)]
    }
}

// ---------------------------------------------------------------------------
// emissions and markers (Rust side only)
// ---------------------------------------------------------------------------
/// Spatial metadata of one voice's block, returned by `Voice::render`.
///
/// Level, distance attenuation, air absorption and shelter transmission are already in the
/// samples (as in Python); the encoder only adds direction and spread. For an ambisonic voice
/// (`Voice::channels() == N_AMBI`) only `stem` and `active` are read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emission {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
    /// 0 = point source, 1 = fully diffuse.
    pub spread: f32,
    pub stem: Stem,
    /// false = the voice wrote nothing this block; the encoder skips it (tails still flush).
    pub active: bool,
}

impl Default for Emission {
    fn default() -> Self {
        Emission::SILENT
    }
}

impl Emission {
    pub const SILENT: Emission = Emission {
        azimuth: 0.0,
        elevation: 0.0,
        distance: 1.0,
        spread: 0.0,
        stem: Stem::RainNear,
        active: false,
    };

    pub fn point(azimuth: f32, elevation: f32, distance: f32, spread: f32, stem: Stem) -> Emission {
        Emission { azimuth, elevation, distance, spread, stem, active: true }
    }

    /// An ambisonic (already encoded) emission or a fully diffuse bed.
    pub fn diffuse(stem: Stem) -> Emission {
        Emission { azimuth: 0.0, elevation: 0.0, distance: 1.0, spread: 1.0, stem, active: true }
    }

    pub fn silent(stem: Stem) -> Emission {
        Emission { stem, ..Emission::SILENT }
    }
}

/// A discrete event for captions / cue sheets / visual sync (lightning, thunder, gust, drip).
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub t: f64,
    pub kind: &'static str,
    /// JSON object text, e.g. `{"distance_km": 3.2}`.
    pub data: String,
}

// ---------------------------------------------------------------------------
// small enum codes
// ---------------------------------------------------------------------------
pub mod precip {
    pub const NONE: u8 = 0;
    pub const DRIZZLE: u8 = 1;
    pub const RAIN: u8 = 2;
    pub const SHOWERS: u8 = 3;
    pub const FREEZING_RAIN: u8 = 4;
    pub const SLEET: u8 = 5;
    pub const SNOW: u8 = 6;
    pub const GRAUPEL: u8 = 7;
    pub const HAIL: u8 = 8;
    pub const NAMES: [&str; 9] =
        ["none", "drizzle", "rain", "showers", "freezing_rain", "sleet", "snow", "graupel", "hail"];

    pub fn from_name(s: &str) -> u8 {
        let k = s.trim().to_ascii_lowercase();
        NAMES.iter().position(|n| *n == k).map(|i| i as u8).unwrap_or(match k.as_str() {
            "" | "dry" | "clear" => NONE,
            "freezing" | "freezing-rain" => FREEZING_RAIN,
            "ice_pellets" => SLEET,
            _ => RAIN,
        })
    }
    /// True for every liquid type (drizzle, rain, showers, freezing rain, and the rain part of sleet).
    pub fn is_liquid(code: u8) -> bool {
        matches!(code, DRIZZLE | RAIN | SHOWERS | FREEZING_RAIN | SLEET)
    }
}

pub mod season {
    pub const SPRING: u8 = 0;
    pub const SUMMER: u8 = 1;
    pub const AUTUMN: u8 = 2;
    pub const WINTER: u8 = 3;
    pub const NAMES: [&str; 4] = ["spring", "summer", "autumn", "winter"];
    pub fn from_name(s: &str) -> u8 {
        match s.trim().to_ascii_lowercase().as_str() {
            "spring" => SPRING,
            "summer" => SUMMER,
            "autumn" | "fall" => AUTUMN,
            "winter" => WINTER,
            _ => AUTUMN,
        }
    }
}

pub mod stability {
    pub const UNSTABLE: u8 = 0;
    pub const NEUTRAL: u8 = 1;
    pub const STABLE: u8 = 2;
    pub const NAMES: [&str; 3] = ["unstable", "neutral", "stable"];
    pub fn from_name(s: &str) -> u8 {
        match s.trim().to_ascii_lowercase().as_str() {
            "unstable" => UNSTABLE,
            "stable" => STABLE,
            _ => NEUTRAL,
        }
    }
}

/// Acoustic drop classes, in WeatherState.class_flux order.
pub const CLASS_NAMES: [&str; N_CLASSES] = ["minuscule", "small", "mid", "large", "very_large"];

/// Listener shelter (Place.shelter), aliases resolved like raingen.physics.layout._SHELTER_ALIASES.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Shelter {
    #[default]
    Open = 0,
    Tree,
    Umbrella,
    Tent,
    Porch,
    Car,
    RoomOpen,
    RoomClosed,
    Attic,
    Cave,
    Underpass,
}

impl Shelter {
    pub fn from_name(s: &str) -> Shelter {
        let k = s.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        match k.as_str() {
            "open" | "" | "none" | "outside" | "outdoors" => Shelter::Open,
            "tree" | "under_tree" | "forest" | "canopy" => Shelter::Tree,
            "umbrella" => Shelter::Umbrella,
            "tent" => Shelter::Tent,
            "porch" | "veranda" | "verandah" | "awning" => Shelter::Porch,
            "car" | "vehicle" | "in_car" => Shelter::Car,
            "room_open" | "window" | "room_window_open" => Shelter::RoomOpen,
            "room_closed" | "room" | "indoors" | "house" | "inside" => Shelter::RoomClosed,
            "attic" | "tin_roof" | "loft" | "barn" | "shed" => Shelter::Attic,
            "cave" | "cave_mouth" => Shelter::Cave,
            "underpass" | "bridge" | "tunnel" => Shelter::Underpass,
            _ => Shelter::Open,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Shelter::Open => "open",
            Shelter::Tree => "tree",
            Shelter::Umbrella => "umbrella",
            Shelter::Tent => "tent",
            Shelter::Porch => "porch",
            Shelter::Car => "car",
            Shelter::RoomOpen => "room_open",
            Shelter::RoomClosed => "room_closed",
            Shelter::Attic => "attic",
            Shelter::Cave => "cave",
            Shelter::Underpass => "underpass",
        }
    }
    /// The listener is inside an enclosure (interior stem, shelter transfer applies).
    pub fn is_enclosed(self) -> bool {
        matches!(self, Shelter::Tent | Shelter::Car | Shelter::RoomOpen | Shelter::RoomClosed | Shelter::Attic)
    }
}

impl<'de> Deserialize<'de> for Shelter {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = de_string_lossy(d)?;
        Ok(Shelter::from_name(&s))
    }
}

// ---------------------------------------------------------------------------
// tolerant field deserializers
// ---------------------------------------------------------------------------
fn code_from_value<E: de::Error>(v: &Value, from_name: fn(&str) -> u8, what: &str) -> Result<u8, E> {
    match v {
        Value::String(s) => Ok(from_name(s)),
        Value::Number(n) => n
            .as_u64()
            .map(|x| x.min(255) as u8)
            .ok_or_else(|| E::custom(format!("{what}: bad numeric code"))),
        Value::Null => Ok(from_name("")),
        _ => Err(E::custom(format!("{what}: expected string or integer"))),
    }
}

fn de_precip<'de, D: Deserializer<'de>>(d: D) -> Result<u8, D::Error> {
    let v = Value::deserialize(d)?;
    code_from_value(&v, precip::from_name, "precip_type")
}

fn de_season<'de, D: Deserializer<'de>>(d: D) -> Result<u8, D::Error> {
    let v = Value::deserialize(d)?;
    code_from_value(&v, season::from_name, "season")
}

fn de_stability<'de, D: Deserializer<'de>>(d: D) -> Result<u8, D::Error> {
    let v = Value::deserialize(d)?;
    code_from_value(&v, stability::from_name, "stability")
}

fn num_f32(v: &Value) -> f32 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
        Value::Bool(b) => *b as u8 as f32,
        Value::String(s) => s.trim().parse::<f32>().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// `{"minuscule": .., "small": .., ...}` or `[a, b, c, d, e]` -> [f32; 5].
fn de_class_flux<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; N_CLASSES], D::Error> {
    let v = Value::deserialize(d)?;
    let mut out = [0.0f32; N_CLASSES];
    match v {
        Value::Object(m) => {
            for (i, name) in CLASS_NAMES.iter().enumerate() {
                if let Some(x) = m.get(*name) {
                    out[i] = num_f32(x);
                }
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().take(N_CLASSES).enumerate() {
                out[i] = num_f32(x);
            }
        }
        Value::Null => {}
        _ => return Err(de::Error::custom("class_flux: expected object or array")),
    }
    Ok(out)
}

/// `{"name": value, ...}` or `[["name", value], ...]` -> Vec<(String, f32)> (sorted by name).
fn de_pairs<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<(String, f32)>, D::Error> {
    let v = Value::deserialize(d)?;
    let mut out: Vec<(String, f32)> = Vec::new();
    match v {
        Value::Object(m) => {
            for (k, x) in m.iter() {
                out.push((k.clone(), num_f32(x)));
            }
        }
        Value::Array(a) => {
            for it in a {
                if let Value::Array(p) = it {
                    if p.len() >= 2 {
                        if let Value::String(k) = &p[0] {
                            out.push((k.clone(), num_f32(&p[1])));
                        }
                    }
                }
            }
        }
        Value::Null => {}
        _ => return Err(de::Error::custom("expected a mapping of name -> number")),
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn de_seed<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    let v = Value::deserialize(d)?;
    match v {
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Ok(u)
            } else if let Some(i) = n.as_i64() {
                Ok(i as u64)
            } else {
                Ok(n.as_f64().unwrap_or(0.0) as u64)
            }
        }
        Value::String(s) => {
            let t = s.trim();
            if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                u64::from_str_radix(h, 16).map_err(de::Error::custom)
            } else {
                t.parse::<u64>().map_err(de::Error::custom)
            }
        }
        Value::Null => Ok(0),
        _ => Err(de::Error::custom("seed: expected integer")),
    }
}

fn de_vec3<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 3], D::Error> {
    let v = Value::deserialize(d)?;
    let mut out = [0.0f32; 3];
    if let Value::Array(a) = v {
        for (i, x) in a.iter().take(3).enumerate() {
            out[i] = num_f32(x);
        }
    }
    Ok(out)
}

fn de_string_lossy<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v = Value::deserialize(d)?;
    Ok(match v {
        Value::String(s) => s,
        Value::Null => String::new(),
        other => other.to_string(),
    })
}

fn de_opt_string<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v = Value::deserialize(d)?;
    Ok(match v {
        Value::String(s) if !s.is_empty() => Some(s),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// WeatherState
// ---------------------------------------------------------------------------
/// Normalized-gamma DSD: N(D) = Nw f(mu) (D/Dm)^mu exp(-(4+mu) D/Dm).
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Dsd {
    pub nw: f32,
    pub dm_mm: f32,
    pub mu: f32,
    pub d_min_mm: f32,
    pub d_max_mm: f32,
}

impl Default for Dsd {
    fn default() -> Self {
        Dsd { nw: 8000.0, dm_mm: 1.4, mu: 3.0, d_min_mm: 0.3, d_max_mm: 6.0 }
    }
}

impl Dsd {
    /// f(mu) = 6 (4 + mu)^(mu+4) / (4^4 Gamma(mu + 4)).
    pub fn f_mu(&self) -> f32 {
        let mu = self.mu as f64;
        let g = ln_gamma(mu + 4.0);
        (6.0 / 256.0 * ((mu + 4.0) * (4.0 + mu).ln() - g).exp()) as f32
    }
    /// N(D) in m^-3 mm^-1.
    pub fn n_of_d(&self, d_mm: f32) -> f32 {
        if d_mm <= 0.0 || self.dm_mm <= 0.0 {
            return 0.0;
        }
        let x = d_mm / self.dm_mm;
        self.nw * self.f_mu() * x.powf(self.mu) * (-(4.0 + self.mu) * x).exp()
    }
}

/// Lanczos ln Gamma (shared by DSD and RNG gamma helpers).
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Storm {
    #[serde(deserialize_with = "de_string_lossy")]
    pub id: String,
    pub distance_km: f32,
    pub bearing_deg: f32,
    pub flash_rate_per_min: f32,
    pub stroke_intensity: f32,
    pub cg_fraction: f32,
    pub rain_contribution: f32,
}

impl Default for Storm {
    fn default() -> Self {
        Storm {
            id: String::new(),
            distance_km: 8.0,
            bearing_deg: 270.0,
            flash_rate_per_min: 2.0,
            stroke_intensity: 0.5,
            cg_fraction: 0.4,
            rain_contribution: 0.0,
        }
    }
}

/// Physically resolved weather at the start of one control block (WeatherState).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Weather {
    pub t: f64,
    #[serde(deserialize_with = "de_precip")]
    pub precip_type: u8,
    pub rain_rate_mm_h: f32,
    pub dsd: Dsd,
    /// impacts / m^2 / s per acoustic class (minuscule, small, mid, large, very_large).
    #[serde(deserialize_with = "de_class_flux")]
    pub class_flux: [f32; N_CLASSES],
    pub hail_size_mm: f32,
    #[serde(alias = "wind_u10")]
    pub wind_u10_mps: f32,
    #[serde(alias = "wind_listener")]
    pub wind_listener_mps: f32,
    pub gust: f32,
    pub wind_from_deg: f32,
    /// Azimuth (rad, listener frame, CCW, 0 = front) the wind comes FROM.
    pub wind_azimuth: f32,
    #[serde(alias = "turbulence")]
    pub turbulence_intensity: f32,
    pub temperature_c: f32,
    pub humidity_pct: f32,
    pub pressure_hpa: f32,
    pub speed_of_sound: f32,
    pub hour: f32,
    #[serde(deserialize_with = "de_season")]
    pub season: u8,
    pub is_day: bool,
    pub storms: Vec<Storm>,
    /// material -> film wetness 0..1 (sorted by name).
    #[serde(deserialize_with = "de_pairs")]
    pub wetness: Vec<(String, f32)>,
    /// material -> puddle depth mm (sorted by name).
    #[serde(alias = "puddle_mm", deserialize_with = "de_pairs")]
    pub puddle_depth_mm: Vec<(String, f32)>,
    #[serde(alias = "canopy_mm")]
    pub canopy_storage_mm: f32,
    pub roof_flow_l_s: f32,
    #[serde(deserialize_with = "de_stability")]
    pub stability: u8,
    pub intensity_envelope: f32,
    /// Wetness per material index of the engine's MaterialTable (filled by `resolve`).
    #[serde(skip)]
    pub wet_by_mat: Vec<f32>,
    /// Puddle depth (mm) per material index (filled by `resolve`).
    #[serde(skip)]
    pub puddle_by_mat: Vec<f32>,
}

impl Default for Weather {
    fn default() -> Self {
        Weather {
            t: 0.0,
            precip_type: precip::NONE,
            rain_rate_mm_h: 0.0,
            dsd: Dsd::default(),
            class_flux: [0.0; N_CLASSES],
            hail_size_mm: 0.0,
            wind_u10_mps: 0.0,
            wind_listener_mps: 0.0,
            gust: 1.0,
            wind_from_deg: 270.0,
            wind_azimuth: 0.0,
            turbulence_intensity: 0.0,
            temperature_c: 12.0,
            humidity_pct: 90.0,
            pressure_hpa: 1013.0,
            speed_of_sound: speed_of_sound(12.0),
            hour: 22.0,
            season: season::AUTUMN,
            is_day: false,
            storms: Vec::new(),
            wetness: Vec::new(),
            puddle_depth_mm: Vec::new(),
            canopy_storage_mm: 0.0,
            roof_flow_l_s: 0.0,
            stability: stability::NEUTRAL,
            intensity_envelope: 1.0,
            wet_by_mat: Vec::new(),
            puddle_by_mat: Vec::new(),
        }
    }
}

impl Weather {
    pub fn from_json(text: &str) -> Result<Weather, String> {
        let mut w: Weather = serde_json::from_str(text).map_err(|e| format!("weather json: {e}"))?;
        if !(w.speed_of_sound > 1.0) {
            w.speed_of_sound = speed_of_sound(w.temperature_c);
        }
        Ok(w)
    }

    /// Fill the per-material-index lookups from the name maps.
    pub fn resolve(&mut self, materials: &MaterialTable) {
        let n = materials.len();
        self.wet_by_mat.clear();
        self.wet_by_mat.resize(n, 0.0);
        self.puddle_by_mat.clear();
        self.puddle_by_mat.resize(n, 0.0);
        for (name, v) in &self.wetness {
            if let Some(i) = materials.index_exact(name) {
                self.wet_by_mat[i] = *v;
            }
        }
        for (name, v) in &self.puddle_depth_mm {
            if let Some(i) = materials.index_exact(name) {
                self.puddle_by_mat[i] = *v;
            }
        }
    }

    /// Film wetness 0..1 of a material by table index (0 if unknown).
    #[inline]
    pub fn wetness_of(&self, material_index: usize) -> f32 {
        self.wet_by_mat.get(material_index).copied().unwrap_or(0.0)
    }

    /// Wetness by name (slower; for setup code).
    pub fn wetness_named(&self, name: &str) -> f32 {
        self.wetness.iter().find(|(k, _)| k == name).map(|(_, v)| *v).unwrap_or(0.0)
    }

    #[inline]
    pub fn puddle_mm_of(&self, material_index: usize) -> f32 {
        self.puddle_by_mat.get(material_index).copied().unwrap_or(0.0)
    }

    pub fn precip_name(&self) -> &'static str {
        precip::NAMES[(self.precip_type as usize).min(precip::NAMES.len() - 1)]
    }

    #[inline]
    pub fn total_flux(&self) -> f32 {
        self.class_flux.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// SourcePatch
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum PatchKind {
    #[default]
    Surface,
    Water,
    Membrane,
    Vegetation,
    Wire,
    Gutter,
    DripEdge,
    VehicleBody,
    Chime,
    Flag,
    Other(String),
}

impl PatchKind {
    pub fn from_name(s: &str) -> PatchKind {
        match s.trim().to_ascii_lowercase().as_str() {
            "surface" | "" => PatchKind::Surface,
            "water" => PatchKind::Water,
            "membrane" => PatchKind::Membrane,
            "vegetation" => PatchKind::Vegetation,
            "wire" => PatchKind::Wire,
            "gutter" => PatchKind::Gutter,
            "drip_edge" => PatchKind::DripEdge,
            "vehicle_body" => PatchKind::VehicleBody,
            "chime" => PatchKind::Chime,
            "flag" => PatchKind::Flag,
            other => PatchKind::Other(other.to_string()),
        }
    }
    pub fn name(&self) -> &str {
        match self {
            PatchKind::Surface => "surface",
            PatchKind::Water => "water",
            PatchKind::Membrane => "membrane",
            PatchKind::Vegetation => "vegetation",
            PatchKind::Wire => "wire",
            PatchKind::Gutter => "gutter",
            PatchKind::DripEdge => "drip_edge",
            PatchKind::VehicleBody => "vehicle_body",
            PatchKind::Chime => "chime",
            PatchKind::Flag => "flag",
            PatchKind::Other(s) => s.as_str(),
        }
    }
}

impl<'de> Deserialize<'de> for PatchKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = de_string_lossy(d)?;
        Ok(PatchKind::from_name(&s))
    }
}

/// A surface or object that rain / wind act upon, placed around the listener (SourcePatch).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Patch {
    #[serde(deserialize_with = "de_string_lossy")]
    pub id: String,
    pub kind: PatchKind,
    #[serde(deserialize_with = "de_string_lossy")]
    pub material: String,
    /// Listener frame, metres (+x forward, +y left, +z up, origin at the ears).
    #[serde(deserialize_with = "de_vec3")]
    pub position: [f32; 3],
    pub area_m2: f32,
    #[serde(deserialize_with = "de_vec3")]
    pub normal: [f32; 3],
    pub exposure: f32,
    #[serde(deserialize_with = "de_opt_string")]
    pub sheltered_by: Option<String>,
    pub interior: bool,
    pub params: Value,
}

impl Default for Patch {
    fn default() -> Self {
        Patch {
            id: String::new(),
            kind: PatchKind::Surface,
            material: "concrete".into(),
            position: [0.0; 3],
            area_m2: 1.0,
            normal: [0.0, 0.0, 1.0],
            exposure: 1.0,
            sheltered_by: None,
            interior: false,
            params: Value::Null,
        }
    }
}

impl Patch {
    pub fn list_from_json(text: &str) -> Result<Vec<Patch>, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("patches json: {e}"))?;
        let arr = match v {
            Value::Array(a) => a,
            Value::Object(mut m) => match m.remove("patches") {
                Some(Value::Array(a)) => a,
                _ => return Err("patches json: expected a list of SourcePatch dicts".into()),
            },
            Value::Null => Vec::new(),
            _ => return Err("patches json: expected a list of SourcePatch dicts".into()),
        };
        arr.into_iter()
            .enumerate()
            .map(|(i, x)| serde_json::from_value::<Patch>(x).map_err(|e| format!("patch {i}: {e}")))
            .collect()
    }

    #[inline]
    pub fn distance(&self) -> f32 {
        let [x, y, z] = self.position;
        (x * x + y * y + z * z).sqrt().max(0.05)
    }
    #[inline]
    pub fn azimuth(&self) -> f32 {
        self.position[1].atan2(self.position[0])
    }
    #[inline]
    pub fn elevation(&self) -> f32 {
        let [x, y, z] = self.position;
        z.atan2(x.hypot(y))
    }

    /// Numeric parameter from `params` (bools and numeric strings accepted).
    pub fn param_f32(&self, key: &str, default: f32) -> f32 {
        match self.params.get(key) {
            Some(Value::Number(n)) => n.as_f64().map(|x| x as f32).unwrap_or(default),
            Some(Value::Bool(b)) => *b as u8 as f32,
            Some(Value::String(s)) => s.trim().parse::<f32>().unwrap_or(default),
            _ => default,
        }
    }
    pub fn param_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(|v| v.as_str())
    }
    pub fn param_bool(&self, key: &str, default: bool) -> bool {
        match self.params.get(key) {
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64().map(|x| x != 0.0).unwrap_or(default),
            Some(Value::String(s)) => matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
            _ => default,
        }
    }
    /// Stable 64-bit key of this patch (for RNG streams): hash of its id.
    pub fn key(&self) -> u64 {
        crate::rng::hash_str(&self.id)
    }
}

// ---------------------------------------------------------------------------
// Material table
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MaterialKind {
    #[default]
    Solid,
    Water,
    Membrane,
    Porous,
    Vegetation,
    Snow,
}

impl MaterialKind {
    pub fn from_name(s: &str) -> MaterialKind {
        match s.trim().to_ascii_lowercase().as_str() {
            "water" => MaterialKind::Water,
            "membrane" => MaterialKind::Membrane,
            "porous" => MaterialKind::Porous,
            "vegetation" => MaterialKind::Vegetation,
            "snow" => MaterialKind::Snow,
            _ => MaterialKind::Solid,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            MaterialKind::Solid => "solid",
            MaterialKind::Water => "water",
            MaterialKind::Membrane => "membrane",
            MaterialKind::Porous => "porous",
            MaterialKind::Vegetation => "vegetation",
            MaterialKind::Snow => "snow",
        }
    }
}

impl<'de> Deserialize<'de> for MaterialKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = de_string_lossy(d)?;
        Ok(MaterialKind::from_name(&s))
    }
}

/// One mode of a material: (frequency Hz, T60 s, relative gain).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mode {
    pub freq_hz: f32,
    pub t60_s: f32,
    pub gain: f32,
}

fn de_modes<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Mode>, D::Error> {
    let v = Value::deserialize(d)?;
    let mut out = Vec::new();
    if let Value::Array(a) = v {
        for m in a {
            match m {
                Value::Array(t) if t.len() >= 3 => out.push(Mode {
                    freq_hz: num_f32(&t[0]),
                    t60_s: num_f32(&t[1]),
                    gain: num_f32(&t[2]),
                }),
                Value::Object(o) => out.push(Mode {
                    freq_hz: o.get("freq_hz").or_else(|| o.get("f")).map(num_f32).unwrap_or(0.0),
                    t60_s: o.get("t60_s").or_else(|| o.get("t60")).map(num_f32).unwrap_or(0.0),
                    gain: o.get("gain").or_else(|| o.get("g")).map(num_f32).unwrap_or(0.0),
                }),
                _ => return Err(de::Error::custom("material mode: expected [freq, t60, gain]")),
            }
        }
    }
    Ok(out)
}

/// Compact modal model of how a surface answers an impact (raingen.synth.materials.Material).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Material {
    #[serde(deserialize_with = "de_string_lossy")]
    pub name: String,
    pub kind: MaterialKind,
    #[serde(deserialize_with = "de_modes")]
    pub modes: Vec<Mode>,
    pub hardness: f32,
    pub splash: f32,
    pub radiation_hp_hz: f32,
    pub radiation_lp_hz: f32,
    /// Peak dB SPL @ 1 m of a 2 mm drop at 6.5 m/s on the dry surface.
    pub level_db: f32,
    #[serde(deserialize_with = "de_string_lossy")]
    pub ucs_cat: String,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: "concrete".into(),
            kind: MaterialKind::Solid,
            modes: Vec::new(),
            hardness: 0.5,
            splash: 0.5,
            radiation_hp_hz: 100.0,
            radiation_lp_hz: 16000.0,
            level_db: 45.0,
            ucs_cat: "RAINGen".into(),
        }
    }
}

impl Material {
    #[inline]
    pub fn is_water(&self) -> bool {
        self.kind == MaterialKind::Water
    }
    /// Hertz-contact Young's modulus: E = 10^(7 + 4.5 hardness) Pa.
    #[inline]
    pub fn youngs_modulus_pa(&self) -> f32 {
        10f32.powf(7.0 + 4.5 * self.hardness.clamp(0.0, 1.0))
    }
    pub fn max_t60(&self) -> f32 {
        self.modes.iter().map(|m| m.t60_s).fold(0.0, f32::max)
    }
}

/// Aliases of raingen.synth.materials._ALIASES (get_material resolution order is kept).
const MATERIAL_ALIASES: &[(&str, &str)] = &[
    ("metal", "metal_sheet"), ("steel", "metal_sheet"), ("aluminium", "metal_sheet"),
    ("aluminum", "metal_sheet"), ("sheet_metal", "metal_sheet"), ("shed", "metal_sheet"),
    ("tin", "tin_roof"), ("corrugated", "tin_roof"), ("roof_metal", "tin_roof"), ("car", "car_body"),
    ("vehicle", "car_body"), ("windshield", "glass"), ("windscreen", "glass"), ("pane", "glass"),
    ("glazing", "window"), ("timber", "wood"), ("deck", "wood"), ("plank", "wood"), ("fence", "wood"),
    ("polycarbonate", "plastic"), ("bin", "plastic"), ("stone", "concrete"), ("pavement", "concrete"),
    ("paving", "concrete"), ("sidewalk", "concrete"), ("brick", "brick_wall"), ("road", "asphalt"),
    ("tarmac", "asphalt"), ("roof", "tile"), ("roof_tile", "tile"), ("slate", "tile"),
    ("ceramic", "tile"), ("pebbles", "gravel"), ("mud", "soil"), ("dirt", "soil"), ("earth", "soil"),
    ("forest_floor", "leaves"), ("lawn", "grass"), ("meadow", "grass"), ("deciduous", "leaves"),
    ("broadleaf", "leaves"), ("foliage", "leaves"), ("canopy", "leaves"), ("conifer", "needles"),
    ("pine", "needles"), ("spruce", "needles"), ("cloth", "fabric"), ("awning", "canvas"),
    ("nylon", "umbrella"), ("hood", "fabric"), ("pond", "water"), ("lake", "water"),
    ("river", "water"), ("sea", "water"), ("pool", "water"), ("gutter", "metal_gutter"),
    ("downspout", "metal_gutter"), ("snow_fresh", "snow"), ("snow_old", "snow"),
];

/// The MATERIALS table, indexed. Python sends it once with `Engine.set_materials`.
#[derive(Clone, Debug, Default)]
pub struct MaterialTable {
    pub materials: Vec<Material>,
    index: HashMap<String, usize>,
    fallback: Material,
}

impl MaterialTable {
    /// Accepts `{name: Material-dict}` or `[Material-dict, ...]`.
    pub fn from_json(text: &str) -> Result<MaterialTable, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("materials json: {e}"))?;
        let mut mats: Vec<Material> = Vec::new();
        match v {
            Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                for k in keys {
                    let mut mat: Material = serde_json::from_value(m[k].clone())
                        .map_err(|e| format!("material {k}: {e}"))?;
                    if mat.name.is_empty() {
                        mat.name = k.clone();
                    }
                    mats.push(mat);
                }
            }
            Value::Array(a) => {
                for (i, x) in a.into_iter().enumerate() {
                    mats.push(serde_json::from_value(x).map_err(|e| format!("material {i}: {e}"))?);
                }
                mats.sort_by(|a, b| a.name.cmp(&b.name));
            }
            _ => return Err("materials json: expected an object or a list".into()),
        }
        Ok(MaterialTable::new(mats))
    }

    pub fn new(materials: Vec<Material>) -> MaterialTable {
        let mut index = HashMap::new();
        for (i, m) in materials.iter().enumerate() {
            index.insert(m.name.clone(), i);
        }
        MaterialTable { materials, index, fallback: Material::default() }
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    pub fn index_exact(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// Index by name or alias, resolved like `get_material` (exact, alias, longest contained
    /// material name, contained alias, then 'concrete'). None only if the table is empty.
    pub fn resolve(&self, name: &str) -> Option<usize> {
        if self.materials.is_empty() {
            return None;
        }
        let key = name.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        if let Some(i) = self.index_exact(&key) {
            return Some(i);
        }
        if let Some((_, t)) = MATERIAL_ALIASES.iter().find(|(a, _)| *a == key) {
            if let Some(i) = self.index_exact(t) {
                return Some(i);
            }
        }
        let mut names: Vec<(usize, &str)> =
            self.materials.iter().enumerate().map(|(i, m)| (i, m.name.as_str())).collect();
        names.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.1.cmp(b.1)));
        for (i, n) in names {
            if !n.is_empty() && key.contains(n) {
                return Some(i);
            }
        }
        for (a, t) in MATERIAL_ALIASES {
            if key.contains(a) {
                if let Some(i) = self.index_exact(t) {
                    return Some(i);
                }
            }
        }
        self.index_exact("concrete").or(Some(0))
    }

    /// Material by name or alias; a neutral default if the table is empty.
    pub fn get(&self, name: &str) -> &Material {
        match self.resolve(name) {
            Some(i) => &self.materials[i],
            None => &self.fallback,
        }
    }

    pub fn by_index(&self, i: usize) -> &Material {
        self.materials.get(i).unwrap_or(&self.fallback)
    }
}

// ---------------------------------------------------------------------------
// Scene (rainscene v1), the parts the native engine reads; everything is kept in `raw`.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Precipitation {
    #[serde(rename = "type", deserialize_with = "de_precip")]
    pub precip_type: u8,
    pub rate_mm_h: f32,
    pub drops: String,
    pub character: String,
    pub shower_period_min: f32,
    pub hail_size_mm: f32,
    pub dsd_model: String,
    pub mu: Option<f32>,
    pub dm_mm: Option<f32>,
    pub nw: Option<f32>,
    pub d_min_mm: f32,
    pub d_max_mm: f32,
    pub class_boundaries: String,
    pub water_quality: f32,
}

impl Default for Precipitation {
    fn default() -> Self {
        Precipitation {
            precip_type: precip::RAIN,
            rate_mm_h: 4.0,
            drops: "normal".into(),
            character: "continuous".into(),
            shower_period_min: 12.0,
            hail_size_mm: 13.0,
            dsd_model: "gamma".into(),
            mu: None,
            dm_mm: None,
            nw: None,
            d_min_mm: 0.3,
            d_max_mm: 6.0,
            class_boundaries: "medwin".into(),
            water_quality: 1.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct SceneWind {
    pub speed_mps: f32,
    pub from_deg: f32,
    pub gust_factor: Option<f32>,
    pub gust_rate_per_min: Option<f32>,
    pub turbulence_intensity: Option<f32>,
    pub spectrum: String,
    pub integral_time_s: Option<f32>,
    pub mic_rumble: bool,
    pub windscreen: String,
}

impl Default for SceneWind {
    fn default() -> Self {
        SceneWind {
            speed_mps: 4.4,
            from_deg: 270.0,
            gust_factor: None,
            gust_rate_per_min: None,
            turbulence_intensity: None,
            spectrum: "kaimal".into(),
            integral_time_s: None,
            mic_rumble: false,
            windscreen: "blimp".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct SceneThunder {
    pub enabled: bool,
    pub distance_km: f32,
    pub bearing_deg: f32,
    pub approach_kmh: f32,
    pub flashes_per_min: f32,
    pub stroke_intensity: f32,
    pub cg_fraction: f32,
}

impl Default for SceneThunder {
    fn default() -> Self {
        SceneThunder {
            enabled: false,
            distance_km: 8.0,
            bearing_deg: 270.0,
            approach_kmh: 0.0,
            flashes_per_min: 2.0,
            stroke_intensity: 0.5,
            cg_fraction: 0.4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct SceneWeather {
    pub precipitation: Precipitation,
    pub wind: SceneWind,
    pub thunder: SceneThunder,
    pub temperature_c: f32,
    pub humidity_pct: f32,
    pub pressure_hpa: f32,
    pub hour: f32,
    #[serde(deserialize_with = "de_season")]
    pub season: u8,
    pub latitude_deg: f32,
    pub storm_arc: String,
    pub arc_minutes: f32,
    pub cloud_okta: f32,
    pub visibility_km: f32,
}

impl Default for SceneWeather {
    fn default() -> Self {
        SceneWeather {
            precipitation: Precipitation::default(),
            wind: SceneWind::default(),
            thunder: SceneThunder::default(),
            temperature_c: 12.0,
            humidity_pct: 90.0,
            pressure_hpa: 1013.0,
            hour: 22.0,
            season: season::AUTUMN,
            latitude_deg: 50.0,
            storm_arc: "steady".into(),
            arc_minutes: 20.0,
            cloud_okta: 8.0,
            visibility_km: 5.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct SceneObject {
    #[serde(deserialize_with = "de_string_lossy")]
    pub id: String,
    #[serde(rename = "type", deserialize_with = "de_string_lossy")]
    pub obj_type: String,
    #[serde(deserialize_with = "de_string_lossy")]
    pub material: String,
    #[serde(deserialize_with = "de_vec3")]
    pub position: [f32; 3],
    #[serde(deserialize_with = "de_vec3")]
    pub size: [f32; 3],
    pub rotation_deg: f32,
    pub params: Value,
}

impl Default for SceneObject {
    fn default() -> Self {
        SceneObject {
            id: String::new(),
            obj_type: "surface".into(),
            material: String::new(),
            position: [0.0; 3],
            size: [1.0, 1.0, 0.0],
            rotation_deg: 0.0,
            params: Value::Null,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Place {
    pub preset: String,
    pub shelter: Shelter,
    /// material -> percent of ground.
    pub surfaces: Value,
    pub vegetation: Value,
    pub emitters: Value,
    pub objects: Vec<SceneObject>,
    pub z0_m: Option<f32>,
    pub ground: String,
    pub biophony: Value,
    pub aftermath_tau_min: f32,
    pub window_open: f32,
    pub wipers: String,
}

impl Default for Place {
    fn default() -> Self {
        Place {
            preset: "garden".into(),
            shelter: Shelter::Open,
            surfaces: Value::Null,
            vegetation: Value::Null,
            emitters: Value::Null,
            objects: Vec::new(),
            z0_m: None,
            ground: "grass".into(),
            biophony: Value::Null,
            aftermath_tau_min: 8.0,
            window_open: 0.0,
            wipers: "off".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Listener {
    #[serde(deserialize_with = "de_vec3")]
    pub position: [f32; 3],
    pub height_m: f32,
    pub heading_deg: f32,
    pub render: String,
    pub width: f32,
    pub hrtf: String,
}

impl Default for Listener {
    fn default() -> Self {
        Listener {
            position: [0.0; 3],
            height_m: 1.7,
            heading_deg: 0.0,
            render: "stereo".into(),
            width: 1.0,
            hrtf: "builtin".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct RenderSettings {
    pub sample_rate: u32,
    pub quality: String,
    pub ambi_order: u32,
    pub k_objects: u32,
    pub bit_depth: u32,
    pub destination: String,
    pub stems: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        RenderSettings {
            sample_rate: SAMPLE_RATE,
            quality: "studio".into(),
            ambi_order: 3,
            k_objects: 64,
            bit_depth: 24,
            destination: "raw".into(),
            stems: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct Scene {
    #[serde(alias = "rainscene")]
    pub version: u32,
    pub name: String,
    #[serde(deserialize_with = "de_seed")]
    pub seed: u64,
    pub weather: SceneWeather,
    pub place: Place,
    pub listener: Listener,
    pub mode: String,
    pub timeline: Value,
    pub render: RenderSettings,
    pub overrides: Value,
    pub vibe: Value,
    /// The whole scene document as sent by Python.
    #[serde(skip)]
    pub raw: Value,
}

impl Default for Scene {
    fn default() -> Self {
        Scene {
            version: 1,
            name: "Untitled weather".into(),
            seed: 0x5EED_CAFE,
            weather: SceneWeather::default(),
            place: Place::default(),
            listener: Listener::default(),
            mode: "none".into(),
            timeline: Value::Null,
            render: RenderSettings::default(),
            overrides: Value::Null,
            vibe: Value::Null,
            raw: Value::Null,
        }
    }
}

impl Scene {
    pub fn from_json(text: &str) -> Result<Scene, String> {
        let raw: Value = serde_json::from_str(text).map_err(|e| format!("scene json: {e}"))?;
        let mut sc: Scene = serde_json::from_value(raw.clone()).map_err(|e| format!("scene json: {e}"))?;
        sc.raw = raw;
        Ok(sc)
    }

    /// Dotted-path lookup into the raw document, e.g. `"place.biophony.birds"`.
    pub fn get(&self, path: &str) -> Option<&Value> {
        let mut v = &self.raw;
        for part in path.split('.') {
            v = match v {
                Value::Object(m) => m.get(part)?,
                Value::Array(a) => a.get(part.parse::<usize>().ok()?)?,
                _ => return None,
            };
        }
        Some(v)
    }

    pub fn get_f32(&self, path: &str, default: f32) -> f32 {
        match self.get(path) {
            Some(Value::Number(n)) => n.as_f64().map(|x| x as f32).unwrap_or(default),
            Some(Value::Bool(b)) => *b as u8 as f32,
            _ => default,
        }
    }

    pub fn shelter(&self) -> Shelter {
        self.place.shelter
    }
}

impl fmt::Display for Stem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Wrap an angle to (-pi, pi].
#[inline]
pub fn wrap_pi(a: f32) -> f32 {
    let tau = 2.0 * PI;
    let mut r = a - tau * (a / tau).round();
    if r <= -PI {
        r += tau;
    }
    r
}
