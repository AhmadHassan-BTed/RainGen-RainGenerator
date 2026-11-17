//! SH encoder (order 3, ACN/SN3D): mono voices -> per-stem buses, with per-voice gain ramps
//! and spread, mirroring raingen/spatial/encoder.py (encode_into + finish_block).
//!
//! * Coherent part: b += g_coh(az, el, spread) x, ramped linearly over the block whenever the
//!   voice's direction or spread changed (no zipper); static voices reuse their gains.
//! * Diffuse part (spread^2 of the energy): the gains are applied BEFORE the decorrelators,
//!   summed per stem, and every stem runs one 15-channel orthogonal all-pass FIR bank
//!   (partitioned FFT convolution, tails carried across blocks and flushed when the stem stops
//!   being diffuse). By linearity this equals decorrelating each voice separately. W is never
//!   decorrelated, so loudness never depends on spread.
//! * Ambisonic voices (16 channels) are added to their stem's bus unchanged.
//!
//! `encode_mono` only computes the gains and stages the block; `finish_block` does the
//! accumulation in parallel over the (stem, channel) rows of the buses, each row summing its
//! voices in graph order, so the result is bit-identical for any thread count.

use super::allpass::{allpass_bank, ALLPASS_ITERS};
use super::conv::PartitionedConv;
use super::sh::spread_gains;
use super::simd;
use super::stems::StemBuses;
use crate::types::{Emission, N_AMBI, N_STEMS};
use rayon::prelude::*;

pub const DECOR_CH: usize = N_AMBI - 1;
pub const DECORRELATOR_SEED: u64 = 0x5EED_DEC0_A11;
/// Decorrelator length (ms), as raingen.spatial.encoder.Encoder(decorrelator_ms=20).
pub const DECORRELATOR_MS: f32 = 20.0;
/// Multiply-adds per block below which the row accumulation runs on the calling thread.
const PAR_WORK: usize = 1 << 19;

/// Per-voice encoder state (lives in the voice's slot; created zeroed).
#[derive(Clone, Copy, Debug)]
pub struct SourceState {
    pub g_coh: [f32; N_AMBI],
    pub g_dif: [f32; N_AMBI],
    pub pos: [f32; 3],
    pub init: bool,
}

impl Default for SourceState {
    fn default() -> Self {
        SourceState { g_coh: [0.0; N_AMBI], g_dif: [0.0; N_AMBI], pos: [f32::NAN; 3], init: false }
    }
}

/// Orthogonal random-phase all-pass FIRs (channels x length), flat to ~0.05 dB at every
/// frequency, unit energy per channel (L-BFGS design, see `allpass`). Deterministic in `seed`
/// and cached process-wide; `iters` = 0 gives the plain magnitude-normalised start.
pub fn design_decorrelators(channels: usize, length: usize, seed: u64, iters: usize) -> Vec<Vec<f32>> {
    allpass_bank(channels, length, seed, iters).as_ref().clone()
}

/// One staged mono block: `x` (fade already applied) at `off`, and for a moving voice
/// `x * ramp` at `off_r`; row k receives a_k x + d_k x ramp.
#[derive(Clone, Copy)]
struct Item {
    stem: usize,
    off: usize,
    off_r: usize,
    /// gains at block start and their change over the block
    a_coh: [f32; N_AMBI],
    d_coh: [f32; N_AMBI],
    a_dif: [f32; N_AMBI],
    d_dif: [f32; N_AMBI],
    moving: bool,
}

pub struct Encoder {
    n: usize,
    sr: u32,
    /// diffuse inputs: N_STEMS x DECOR_CH x n
    din: Vec<f32>,
    din_row: [[bool; DECOR_CH]; N_STEMS],
    din_active: [bool; N_STEMS],
    coh_row: [[bool; N_AMBI]; N_STEMS],
    decor: Vec<PartitionedConv>,
    firs: Vec<Vec<f32>>,
    ramp: Vec<f32>,
    items: Vec<Item>,
    staging: Vec<f32>,
    staged: usize,
    /// item indices grouped by stem (stable: graph order inside a stem)
    order: Vec<u32>,
    stem_start: [usize; N_STEMS + 1],
}

impl Encoder {
    pub fn new(n: usize, sr: u32) -> Encoder {
        let len = ((DECORRELATOR_MS * 1e-3 * sr as f32).round() as usize).max(16);
        let firs = allpass_bank(DECOR_CH, len, DECORRELATOR_SEED, ALLPASS_ITERS);
        Encoder::with_decorrelators(n, sr, firs.as_ref().clone())
    }

    pub fn with_decorrelators(n: usize, sr: u32, firs: Vec<Vec<f32>>) -> Encoder {
        let mut firs = firs;
        firs.resize(DECOR_CH, Vec::new());
        let decor = (0..N_STEMS).map(|_| PartitionedConv::diagonal(n, &firs)).collect();
        const RESERVE_VOICES: usize = 64;
        Encoder {
            n,
            sr,
            din: vec![0.0; N_STEMS * DECOR_CH * n],
            din_row: [[false; DECOR_CH]; N_STEMS],
            din_active: [false; N_STEMS],
            coh_row: [[false; N_AMBI]; N_STEMS],
            decor,
            firs,
            ramp: (1..=n).map(|i| i as f32 / n as f32).collect(),
            items: Vec::with_capacity(RESERVE_VOICES),
            staging: vec![0.0; 2 * RESERVE_VOICES * n],
            staged: 0,
            order: Vec::with_capacity(RESERVE_VOICES),
            stem_start: [0; N_STEMS + 1],
        }
    }

    /// Replace the decorrelator FIRs (DECOR_CH x L), e.g. with Python's designed bank.
    pub fn set_decorrelators(&mut self, firs: Vec<Vec<f32>>) {
        let mut firs = firs;
        firs.resize(DECOR_CH, Vec::new());
        self.decor = (0..N_STEMS).map(|_| PartitionedConv::diagonal(self.n, &firs)).collect();
        self.firs = firs;
    }

    pub fn decorrelators(&self) -> &[Vec<f32>] {
        &self.firs
    }

    pub fn block(&self) -> usize {
        self.n
    }

    pub fn sample_rate(&self) -> u32 {
        self.sr
    }

    pub fn reset(&mut self) {
        for d in &mut self.decor {
            d.reset();
        }
        self.begin_block();
    }

    /// Start a block: forgets staged voices and the diffuse row flags.
    pub fn begin_block(&mut self) {
        self.items.clear();
        self.staged = 0;
        self.din_row = [[false; DECOR_CH]; N_STEMS];
        self.din_active = [false; N_STEMS];
        self.coh_row = [[false; N_AMBI]; N_STEMS];
    }

    /// Stage one mono block `x` (n samples) for the emission's stem with signal gain `g0`
    /// (block start) -> `g1` (block end) (crossfades; 1, 1 normally). The stem bus is complete
    /// after `finish_block`; `_bus` is accepted for API compatibility and not written here.
    pub fn encode_mono(
        &mut self,
        _bus: &mut [f32],
        x: &[f32],
        em: &Emission,
        st: &mut SourceState,
        g0: f32,
        g1: f32,
    ) {
        let n = self.n;
        let stem = em.stem.index().min(N_STEMS - 1);
        let pos = [em.azimuth, em.elevation, em.spread.clamp(0.0, 1.0)];
        if !pos.iter().all(|v| v.is_finite()) {
            return;
        }
        let (p_coh, p_dif, moving);
        if !st.init {
            let (c, d) = spread_gains(pos[0], pos[1], pos[2]);
            st.g_coh = c;
            st.g_dif = d;
            st.pos = pos;
            st.init = true;
            p_coh = c;
            p_dif = d;
            moving = false;
        } else if st.pos == pos {
            p_coh = st.g_coh;
            p_dif = st.g_dif;
            moving = false;
        } else {
            p_coh = st.g_coh;
            p_dif = st.g_dif;
            let (c, d) = spread_gains(pos[0], pos[1], pos[2]);
            st.g_coh = c;
            st.g_dif = d;
            st.pos = pos;
            moving = true;
        }
        let fading = g0 != 1.0 || g1 != 1.0;
        if g0 == 0.0 && g1 == 0.0 {
            return;
        }
        let mut it = Item {
            stem,
            off: 0,
            off_r: usize::MAX,
            a_coh: p_coh,
            d_coh: [0.0; N_AMBI],
            a_dif: p_dif,
            d_dif: [0.0; N_AMBI],
            moving,
        };
        let mut any = false;
        for k in 0..N_AMBI {
            if moving {
                it.d_coh[k] = st.g_coh[k] - p_coh[k];
                it.d_dif[k] = st.g_dif[k] - p_dif[k];
            }
            if p_coh[k] != 0.0 || st.g_coh[k] != 0.0 {
                self.coh_row[stem][k] = true;
                any = true;
            }
            if k > 0 && (p_dif[k] != 0.0 || st.g_dif[k] != 0.0) {
                self.din_row[stem][k - 1] = true;
                self.din_active[stem] = true;
                any = true;
            }
        }
        if !any {
            return;
        }
        let slots = if moving { 2 } else { 1 };
        let need = self.staged + slots * n;
        if self.staging.len() < need {
            // amortised growth (only while the graph grows)
            let cap = (need * 2).max(self.staging.len());
            self.staging.resize(cap, 0.0);
        }
        it.off = self.staged;
        let m = x.len().min(n);
        {
            let dst = &mut self.staging[it.off..it.off + n];
            if fading {
                simd::ramp_into(&mut dst[..m], &x[..m], g0, g1 - g0, &self.ramp[..m]);
            } else {
                dst[..m].copy_from_slice(&x[..m]);
            }
            dst[m..].iter_mut().for_each(|v| *v = 0.0);
        }
        if moving {
            it.off_r = it.off + n;
            let (xs, xr) = self.staging[it.off..it.off + 2 * n].split_at_mut(n);
            simd::mul_into(xr, xs, &self.ramp);
        }
        self.staged = need;
        self.items.push(it);
    }

    /// Add an already-encoded SH block (N_AMBI x n, channel-major) to `bus`, gain g0 -> g1.
    pub fn add_ambisonic(&self, bus: &mut [f32], x: &[f32], g0: f32, g1: f32) {
        let n = self.n;
        let m = x.len().min(bus.len()).min(N_AMBI * n);
        if g0 == 1.0 && g1 == 1.0 {
            for (o, v) in bus[..m].iter_mut().zip(&x[..m]) {
                *o += *v;
            }
        } else {
            for ch in 0..m / n {
                let o = &mut bus[ch * n..(ch + 1) * n];
                let s = &x[ch * n..(ch + 1) * n];
                for i in 0..n {
                    let gv = g0 + (g1 - g0) * self.ramp[i];
                    o[i] += gv * s[i];
                }
            }
        }
    }

    /// Accumulate the staged voices into the stem buses and the decorrelator inputs (parallel
    /// over rows), then run the per-stem decorrelators and add the wet signal to channels
    /// 1..N_AMBI (parallel over stems). Call inside the render pool.
    pub fn finish_block(&mut self, buses: &mut StemBuses) {
        let n = self.n;
        // group items by stem, keeping graph order
        let mut counts = [0usize; N_STEMS];
        for it in &self.items {
            counts[it.stem] += 1;
        }
        self.stem_start[0] = 0;
        for s in 0..N_STEMS {
            self.stem_start[s + 1] = self.stem_start[s] + counts[s];
        }
        self.order.clear();
        self.order.resize(self.items.len(), 0);
        let mut fill = self.stem_start;
        for (i, it) in self.items.iter().enumerate() {
            self.order[fill[it.stem]] = i as u32;
            fill[it.stem] += 1;
        }

        let items = &self.items;
        let order = &self.order;
        let starts = &self.stem_start;
        let staging = &self.staging;
        let coh_row = &self.coh_row;
        let din_row = &self.din_row;
        let stride = N_AMBI * n;

        // multiply-adds this block; below PAR_WORK the rayon fan-out costs more than it saves
        let mut rows_live = 0usize;
        for st in 0..N_STEMS {
            if counts[st] == 0 {
                continue;
            }
            let r = coh_row[st].iter().filter(|&&b| b).count() + din_row[st].iter().filter(|&&b| b).count();
            rows_live += r * counts[st];
        }
        let parallel = rows_live * n >= PAR_WORK;

        if !items.is_empty() {
            let bus_data = &mut buses.data[..N_STEMS * stride];
            let din = &mut self.din;
            let coh = |(row, out): (usize, &mut [f32])| {
                let stem = row / N_AMBI;
                let k = row % N_AMBI;
                if !coh_row[stem][k] {
                    return;
                }
                for &ii in &order[starts[stem]..starts[stem + 1]] {
                    let it = &items[ii as usize];
                    accumulate(out, staging, n, it, it.a_coh[k], it.d_coh[k]);
                }
            };
            let dif = |(row, out): (usize, &mut [f32])| {
                let stem = row / DECOR_CH;
                let k = row % DECOR_CH + 1;
                if !din_row[stem][k - 1] {
                    return;
                }
                out.iter_mut().for_each(|v| *v = 0.0);
                for &ii in &order[starts[stem]..starts[stem + 1]] {
                    let it = &items[ii as usize];
                    accumulate(out, staging, n, it, it.a_dif[k], it.d_dif[k]);
                }
            };
            if parallel {
                rayon::join(
                    || bus_data.par_chunks_mut(n).enumerate().for_each(coh),
                    || din.par_chunks_mut(n).enumerate().for_each(dif),
                );
            } else {
                bus_data.chunks_mut(n).enumerate().for_each(coh);
                din.chunks_mut(n).enumerate().for_each(dif);
            }
        }

        // decorrelators: one pass per stem with diffuse input or a tail still ringing
        let din_active = self.din_active;
        let s_din = DECOR_CH * n;
        let din = &self.din;
        let busy = self.decor.iter().enumerate().filter(|(st, d)| din_active[*st] || d.has_tail()).count();
        let run = |(st, (dec, bus)): (usize, (&mut PartitionedConv, &mut [f32]))| {
            if !(din_active[st] || dec.has_tail()) {
                return;
            }
            let rows = &din_row[st];
            dec.process(&din[st * s_din..(st + 1) * s_din], Some(rows), &mut bus[n..], true);
        };
        if busy > 1 {
            self.decor.par_iter_mut().zip(buses.data.par_chunks_mut(stride)).enumerate().for_each(run);
        } else if busy == 1 {
            self.decor.iter_mut().zip(buses.data.chunks_mut(stride)).enumerate().for_each(run);
        }
        self.items.clear();
        self.staged = 0;
    }
}

/// out += a x + d (x r): the staged block (fade applied) with its gain ramp.
#[inline]
fn accumulate(out: &mut [f32], staging: &[f32], n: usize, it: &Item, a: f32, d: f32) {
    let x = &staging[it.off..it.off + n];
    if it.moving && d != 0.0 {
        let xr = &staging[it.off_r..it.off_r + n];
        simd::axpy2(out, a, x, d, xr);
    } else if a != 0.0 {
        simd::axpy(out, a, x);
    }
}
