//! Per-stem SH buses: N_STEMS x N_AMBI x n, stem-major then channel-major.

use crate::types::{Stem, N_AMBI, N_STEMS};

#[derive(Clone, Debug)]
pub struct StemBuses {
    pub n: usize,
    pub data: Vec<f32>,
}

impl StemBuses {
    pub fn new(n: usize) -> StemBuses {
        StemBuses { n, data: vec![0.0; N_STEMS * N_AMBI * n] }
    }

    #[inline]
    pub fn stride(&self) -> usize {
        N_AMBI * self.n
    }

    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0.0);
    }

    /// One stem's bus (N_AMBI * n, channel-major).
    #[inline]
    pub fn bus(&self, stem: Stem) -> &[f32] {
        let s = self.stride();
        &self.data[stem.index() * s..(stem.index() + 1) * s]
    }

    #[inline]
    pub fn bus_mut(&mut self, stem: Stem) -> &mut [f32] {
        let s = self.stride();
        &mut self.data[stem.index() * s..(stem.index() + 1) * s]
    }

    #[inline]
    pub fn ch(&self, stem: Stem, ch: usize) -> &[f32] {
        let n = self.n;
        let o = stem.index() * self.stride() + ch * n;
        &self.data[o..o + n]
    }

    #[inline]
    pub fn ch_mut(&mut self, stem: Stem, ch: usize) -> &mut [f32] {
        let n = self.n;
        let o = stem.index() * self.stride() + ch * n;
        &mut self.data[o..o + n]
    }

    /// Mutable views of all stems at once (for parallel per-stem work).
    pub fn stems_mut(&mut self) -> Vec<&mut [f32]> {
        let s = self.stride();
        self.data.chunks_mut(s).collect()
    }

    pub fn scale_stem(&mut self, stem: Stem, g: f32) {
        if g != 1.0 {
            self.bus_mut(stem).iter_mut().for_each(|v| *v *= g);
        }
    }

    /// mix (N_AMBI * n) = sum over stems.
    pub fn sum_into(&self, mix: &mut [f32]) {
        let s = self.stride();
        mix[..s].iter_mut().for_each(|v| *v = 0.0);
        for st in 0..N_STEMS {
            let b = &self.data[st * s..(st + 1) * s];
            for (m, v) in mix[..s].iter_mut().zip(b) {
                *m += *v;
            }
        }
    }
}
