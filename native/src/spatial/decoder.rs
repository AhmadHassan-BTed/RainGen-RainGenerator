//! Decoder application. Decoders are DESIGNED in Python (raingen/spatial/decoders.py) and
//! handed over as data; Rust only applies them per block:
//!   * Matrix:      out = D (C x 16) b, plus an optional W feed through a Linkwitz-Riley LR4
//!                  low-pass (2 x Butterworth-2, as Python's MatrixDecoder) into the LFE channel;
//!   * Fir:         binaural, out_e = sum_k h_{e,k} * b_k with FIRs (outputs x 16 x taps),
//!                  uniformly partitioned FFT convolution (one partition per block), tails
//!                  carried across blocks;
//!   * Passthrough: raw SH (foa = first 4 ACN channels, hoa2 = 9, hoa3 = all 16).

use super::conv::PartitionedConv;
use super::simd;
use crate::dsp::{Sos, DENORMAL_EPS};
use crate::types::N_AMBI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderKind {
    Matrix,
    Fir,
    Passthrough,
}

pub struct Lfe {
    pub index: usize,
    pub gain: f32,
    pub filter: Sos,
}

impl Lfe {
    /// W -> LFE feed: LR4 low-pass at `cutoff_hz`, `gain` linear.
    pub fn new(index: usize, gain: f32, cutoff_hz: f32, sr: u32) -> Lfe {
        Lfe { index, gain, filter: Sos::linkwitz_riley_lowpass(4, cutoff_hz, sr as f32) }
    }
}

pub struct Decoder {
    pub kind: DecoderKind,
    pub layout: String,
    channels: usize,
    n: usize,
    /// C x 16 row-major
    matrix: Vec<f32>,
    /// per output: the ACN channels with a non-zero gain
    nz: Vec<Vec<u8>>,
    fir: Option<PartitionedConv>,
    lfe: Option<Lfe>,
    latency: usize,
    scratch: Vec<f32>,
    bus_live: [bool; N_AMBI],
}

impl Decoder {
    /// Matrix decoder: `matrix` is C x 16 row-major (narrower inputs are zero-padded by caller).
    pub fn matrix(layout: &str, n: usize, channels: usize, matrix: Vec<f32>, lfe: Option<Lfe>) -> Decoder {
        let mut m = matrix;
        m.resize(channels * N_AMBI, 0.0);
        let nz = (0..channels)
            .map(|c| (0..N_AMBI).filter(|&k| m[c * N_AMBI + k] != 0.0).map(|k| k as u8).collect())
            .collect();
        Decoder {
            kind: DecoderKind::Matrix,
            layout: layout.to_string(),
            channels,
            n,
            matrix: m,
            nz,
            fir: None,
            lfe,
            latency: 0,
            scratch: vec![0.0; n],
            bus_live: [true; N_AMBI],
        }
    }

    /// FIR (binaural) decoder: `firs` is outputs x 16 x taps, row-major.
    pub fn fir(layout: &str, n: usize, outputs: usize, taps: usize, firs: &[f32], latency: usize) -> Decoder {
        let conv = PartitionedConv::new(n, outputs, N_AMBI, |e, k| {
            let o = (e * N_AMBI + k) * taps;
            if taps > 0 && o + taps <= firs.len() {
                &firs[o..o + taps]
            } else {
                &[]
            }
        });
        Decoder {
            kind: DecoderKind::Fir,
            layout: layout.to_string(),
            channels: outputs,
            n,
            matrix: Vec::new(),
            nz: Vec::new(),
            fir: Some(conv),
            lfe: None,
            latency,
            scratch: Vec::new(),
            bus_live: [true; N_AMBI],
        }
    }

    /// Raw SH output with `channels` = 1, 4, 9 or 16 ACN channels.
    pub fn passthrough(layout: &str, n: usize, channels: usize) -> Decoder {
        Decoder {
            kind: DecoderKind::Passthrough,
            layout: layout.to_string(),
            channels: channels.clamp(1, N_AMBI),
            n,
            matrix: Vec::new(),
            nz: Vec::new(),
            fir: None,
            lfe: None,
            latency: 0,
            scratch: Vec::new(),
            bus_live: [true; N_AMBI],
        }
    }

    /// Built-in fallback before Python sets an output: a pair of virtual cardioids at +-60 deg
    /// (SN3D first order), energy-normalised for a diffuse field.
    pub fn default_stereo(n: usize) -> Decoder {
        let a = 60f32.to_radians();
        let mut m = vec![0.0f32; 2 * N_AMBI];
        for (row, sgn) in [(0usize, 1.0f32), (1, -1.0)] {
            // cardioid 0.5 (W + cos(a) X + sin(a) Y); ACN 1 = Y, 3 = X
            m[row * N_AMBI] = 0.5;
            m[row * N_AMBI + 1] = 0.5 * sgn * a.sin();
            m[row * N_AMBI + 3] = 0.5 * a.cos();
        }
        // diffuse-field energy per output: 0.25 (1 + 1/3) -> normalise to 0.5
        let g = (0.5f32 / (0.25 * (1.0 + 1.0 / 3.0))).sqrt();
        m.iter_mut().for_each(|v| *v *= g);
        Decoder::matrix("stereo", n, 2, m, None)
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Declare the decoder's own latency (samples) so render() can trim it.
    pub fn with_latency(mut self, latency: usize) -> Decoder {
        self.latency = latency;
        self
    }

    pub fn latency(&self) -> usize {
        self.latency
    }

    pub fn block(&self) -> usize {
        self.n
    }

    /// The decoding matrix (C x 16) of a matrix decoder (empty otherwise).
    pub fn matrix_data(&self) -> &[f32] {
        &self.matrix
    }

    pub fn reset(&mut self) {
        if let Some(f) = &mut self.fir {
            f.reset();
        }
        if let Some(l) = &mut self.lfe {
            l.filter.reset();
        }
    }

    /// bus: N_AMBI x n (channel-major) -> out: channels x n (channel-major, overwritten).
    pub fn process(&mut self, bus: &[f32], out: &mut [f32]) {
        let n = self.n;
        match self.kind {
            DecoderKind::Matrix => {
                for k in 0..N_AMBI {
                    self.bus_live[k] = simd::max_abs(&bus[k * n..(k + 1) * n]) > DENORMAL_EPS;
                }
                for c in 0..self.channels {
                    let o = &mut out[c * n..(c + 1) * n];
                    let mut first = true;
                    for &k in &self.nz[c] {
                        let k = k as usize;
                        if !self.bus_live[k] {
                            continue;
                        }
                        let g = self.matrix[c * N_AMBI + k];
                        let b = &bus[k * n..(k + 1) * n];
                        if first {
                            simd::scale_into(o, g, b);
                            first = false;
                        } else {
                            simd::axpy(o, g, b);
                        }
                    }
                    if first {
                        o.iter_mut().for_each(|v| *v = 0.0);
                    }
                }
                if let Some(lfe) = &mut self.lfe {
                    if lfe.index < self.channels && lfe.gain > 0.0 {
                        self.scratch.copy_from_slice(&bus[..n]);
                        lfe.filter.process_block(&mut self.scratch);
                        let o = &mut out[lfe.index * n..(lfe.index + 1) * n];
                        simd::axpy(o, lfe.gain, &self.scratch);
                    }
                }
            }
            DecoderKind::Fir => {
                if let Some(f) = &mut self.fir {
                    f.process(bus, None, out, false);
                }
            }
            DecoderKind::Passthrough => {
                let m = self.channels * n;
                out[..m].copy_from_slice(&bus[..m]);
            }
        }
    }
}
