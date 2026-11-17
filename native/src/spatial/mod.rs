//! Spatial core: order-3 SH (ACN/SN3D) encoding of voices into per-stem buses (gain ramps,
//! spread through orthogonal all-pass decorrelators), and the application of decoders designed
//! in Python (matrix + LFE, partitioned-FFT FIR binaural, raw SH passthrough).

pub mod allpass;
pub mod conv;
pub mod decoder;
pub mod encoder;
pub mod sh;
pub mod simd;
pub mod space;
pub mod stems;

pub use allpass::{allpass_bank, design_allpass_firs};
pub use conv::PartitionedConv;
pub use decoder::{Decoder, DecoderKind, Lfe};
pub use encoder::{design_decorrelators, Encoder, SourceState, DECOR_CH, DECORRELATOR_SEED};
pub use sh::{cap_weights, listener_rotation, maxre_weights, sh_real, sh_real_f32, spread_gains, yaw_rotation, ACN_DEGREE};
pub use stems::StemBuses;

use crate::types::N_AMBI;

/// Self-encoding helper for voices that render several sources into one ambisonic output
/// (`Voice::channels() == N_AMBI`): out (N_AMBI x n) += g_coh(az, el, spread) * x. No
/// decorrelation (use separate voices or the stem decorrelators for diffuse sources).
pub fn encode_point_into(out: &mut [f32], x: &[f32], az: f32, el: f32, spread: f32, gain: f32) {
    let n = x.len();
    let (g, _) = spread_gains(az, el, spread);
    for k in 0..N_AMBI {
        let gk = g[k] * gain;
        if gk == 0.0 {
            continue;
        }
        let o = &mut out[k * n..(k + 1) * n];
        for (ov, xv) in o.iter_mut().zip(x) {
            *ov += gk * *xv;
        }
    }
}
