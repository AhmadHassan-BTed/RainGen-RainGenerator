//! Environment family (port of raingen/synth/wind.py, vegetation.py, aeolian.py, thunder.py,
//! biophony.py, places.py, environment.py): wind bed, vegetation, aeolian tones, mic rumble,
//! thunder, biophony / traffic, wipers, and the shelter transfer / room as a bus processor.
//!
//! Entry points used by the engine: `build_voices` and `build_bus_processors`.
//!
//! Graph order (raingen.synth.environment.make_environment_generators): wind bed, vegetation,
//! aeolian emitters, optional mic rumble, thunder (always: storm cells can appear mid-render),
//! biophony, traffic, car wipers. Every voice draws from its own RNG key; the gust field and the
//! storm-flash schedule use NumPy-compatible streams so they agree with the Python World.

pub mod aeolian;
pub mod biophony;
pub mod common;
pub mod gust;
pub mod npy;
pub mod places;
pub mod space;
pub mod thunder;
pub mod vegetation;
pub mod wind;

use crate::engine::{BoxedProcessor, BoxedVoice, BuildCfg};
use crate::types::{MaterialTable, Patch, Scene};

/// Every environment voice of the scene (after the rain voices in graph order).
pub fn build_voices(scene: &Scene, patches: &[Patch], materials: &MaterialTable, cfg: &BuildCfg) -> Vec<BoxedVoice> {
    let mut out: Vec<BoxedVoice> = Vec::new();
    out.push(Box::new(wind::WindVoice::new(scene, cfg)));
    for v in vegetation::build(scene, patches, materials, cfg) {
        out.push(Box::new(v));
    }
    out.extend(aeolian::build(scene, patches, cfg));
    if scene.weather.wind.mic_rumble {
        let mic = wind::MicRumbleVoice::new(scene, cfg);
        if mic.exposure > 0.0 {
            out.push(Box::new(mic));
        }
    }
    out.push(Box::new(thunder::ThunderVoice::new(scene, patches, cfg)));
    out.extend(biophony::build(scene, patches, cfg));
    out.extend(biophony::build_traffic(scene, cfg));
    if scene.place.shelter == crate::types::Shelter::Car {
        out.push(Box::new(places::CarWipersVoice::new(scene, patches, cfg)));
    }
    out
}

/// Processors applied to the per-stem SH buses after encoding (shelter transfer, room).
pub fn build_bus_processors(scene: &Scene, patches: &[Patch], materials: &MaterialTable, cfg: &BuildCfg) -> Vec<BoxedProcessor> {
    let _ = (patches, materials);
    let mut out: Vec<BoxedProcessor> = Vec::new();
    // place acoustics of the ambience stem first, so an enclosure transfer filters it too
    if let Some(sp) = space::PlaceSpace::new(scene, cfg) {
        out.push(Box::new(sp));
    }
    let st = places::ShelterProcessor::new(scene, cfg);
    if st.needed() {
        out.push(Box::new(st));
    }
    out
}
