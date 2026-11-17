//! Uniformly partitioned overlap-save FFT convolution (UPOLS) of a multichannel FIR matrix:
//! out_e = sum_k h_{e,k} * in_k, streamed block by block with no added latency.
//!
//! Partition size = block size n, nfft = 2n. Every FIR is cut into P = ceil(L / n) partitions
//! whose spectra are precomputed (pre-scaled by 1/nfft, stored as split re/im arrays so the
//! complex multiply-accumulate vectorises). Each block costs one forward FFT per non-silent
//! input (of [previous block | this block]), P complex MACs per path and one inverse FFT per
//! output; the last n samples of the inverse are the exact linear-convolution output
//! (overlap-save). Inputs that are silent this and the previous block skip their FFT, zero
//! filter partitions and silent frequency-delay-line slots are skipped, and an output with no
//! live path is written as silence without an inverse FFT. Nothing allocates after `new`.

use super::simd;
use crate::dsp::DENORMAL_EPS;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

const NO_PATH: u32 = u32::MAX;

#[inline]
fn silent(x: &[f32]) -> bool {
    simd::max_abs(x) <= DENORMAL_EPS
}

pub struct PartitionedConv {
    n: usize,
    nfft: usize,
    bins: usize,
    parts: usize,
    taps: usize,
    inputs: usize,
    outputs: usize,
    /// e * inputs + k -> path index (NO_PATH = zero filter)
    path_of: Vec<u32>,
    /// path spectra, split complex: [path][partition][bin]
    h_re: Vec<f32>,
    h_im: Vec<f32>,
    /// [path][partition]: partition has non-zero taps
    h_live: Vec<bool>,
    /// previous input block per input (overlap-save history)
    prev: Vec<f32>,
    prev_live: Vec<bool>,
    /// frequency-domain delay line, split complex: [input][slot][bin]
    fdl_re: Vec<f32>,
    fdl_im: Vec<f32>,
    fdl_live: Vec<bool>,
    head: usize,
    acc_re: Vec<f32>,
    acc_im: Vec<f32>,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
}

impl PartitionedConv {
    /// `firs(e, k)` is the FIR from input k to output e (empty or all-zero = no path).
    pub fn new<'a, F>(n: usize, outputs: usize, inputs: usize, firs: F) -> PartitionedConv
    where
        F: Fn(usize, usize) -> &'a [f32],
    {
        let n = n.max(1);
        let nfft = 2 * n;
        let bins = n + 1;
        let mut taps = 1usize;
        for e in 0..outputs {
            for k in 0..inputs {
                let h = firs(e, k);
                // trailing zeros do not count
                let len = h.iter().rposition(|v| *v != 0.0).map(|i| i + 1).unwrap_or(0);
                taps = taps.max(len);
            }
        }
        let parts = taps.div_ceil(n).max(1);
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(nfft);
        let inv = planner.plan_fft_inverse(nfft);
        let mut time = fwd.make_input_vec();
        let mut spec = fwd.make_output_vec();
        let mut scratch_f = fwd.make_scratch_vec();
        let scratch_i = inv.make_scratch_vec();

        let mut path_of = vec![NO_PATH; outputs * inputs];
        let mut h_re = Vec::new();
        let mut h_im = Vec::new();
        let mut h_live = Vec::new();
        let norm = 1.0 / nfft as f32;
        let mut paths = 0u32;
        for e in 0..outputs {
            for k in 0..inputs {
                let h = firs(e, k);
                if h.iter().all(|v| *v == 0.0) {
                    continue;
                }
                path_of[e * inputs + k] = paths;
                paths += 1;
                for p in 0..parts {
                    let lo = (p * n).min(h.len());
                    let hi = ((p + 1) * n).min(h.len());
                    let seg = &h[lo..hi];
                    let live = seg.iter().any(|v| *v != 0.0);
                    h_live.push(live);
                    if !live {
                        h_re.extend(std::iter::repeat_n(0.0f32, bins));
                        h_im.extend(std::iter::repeat_n(0.0f32, bins));
                        continue;
                    }
                    time.iter_mut().for_each(|v| *v = 0.0);
                    for (t, v) in time.iter_mut().zip(seg) {
                        *t = *v * norm;
                    }
                    let _ = fwd.process_with_scratch(&mut time, &mut spec, &mut scratch_f);
                    h_re.extend(spec.iter().map(|c| c.re));
                    h_im.extend(spec.iter().map(|c| c.im));
                }
            }
        }
        PartitionedConv {
            n,
            nfft,
            bins,
            parts,
            taps,
            inputs,
            outputs,
            path_of,
            h_re,
            h_im,
            h_live,
            prev: vec![0.0; inputs * n],
            prev_live: vec![false; inputs],
            fdl_re: vec![0.0; inputs * parts * bins],
            fdl_im: vec![0.0; inputs * parts * bins],
            fdl_live: vec![false; inputs * parts],
            head: 0,
            acc_re: vec![0.0; bins],
            acc_im: vec![0.0; bins],
            time,
            spec,
            scratch_f,
            scratch_i,
            fwd,
            inv,
        }
    }

    /// k -> k filters only (decorrelator banks).
    pub fn diagonal(n: usize, firs: &[Vec<f32>]) -> PartitionedConv {
        let c = firs.len();
        PartitionedConv::new(n, c, c, |e, k| if e == k { firs[k].as_slice() } else { &[] })
    }

    pub fn block(&self) -> usize {
        self.n
    }
    pub fn taps(&self) -> usize {
        self.taps
    }
    pub fn partitions(&self) -> usize {
        self.parts
    }
    pub fn inputs(&self) -> usize {
        self.inputs
    }
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// True while any input still has signal inside the filters' memory (a tail to play out).
    pub fn has_tail(&self) -> bool {
        self.prev_live.iter().any(|&b| b) || self.fdl_live.iter().any(|&b| b)
    }

    pub fn reset(&mut self) {
        self.prev.iter_mut().for_each(|v| *v = 0.0);
        self.prev_live.iter_mut().for_each(|v| *v = false);
        self.fdl_live.iter_mut().for_each(|v| *v = false);
        self.head = 0;
    }

    /// One block. `input` is `inputs * n` channel-major, `out` is `outputs * n`; the result is
    /// ADDED into `out` when `add`, else written. `input_active[k] == false` declares input k
    /// silent this block (its samples are not read); None tests every input.
    pub fn process(&mut self, input: &[f32], input_active: Option<&[bool]>, out: &mut [f32], add: bool) {
        let n = self.n;
        let bins = self.bins;
        let parts = self.parts;
        self.head = (self.head + 1) % parts;
        let head = self.head;

        // forward transforms into the delay line
        for k in 0..self.inputs {
            let x = &input[k * n..(k + 1) * n];
            let act = input_active.map(|a| a.get(k).copied().unwrap_or(true)).unwrap_or(true) && !silent(x);
            let slot = k * parts + head;
            let prev = &mut self.prev[k * n..(k + 1) * n];
            if !act && !self.prev_live[k] {
                self.fdl_live[slot] = false;
                continue;
            }
            self.time[..n].copy_from_slice(prev);
            if act {
                self.time[n..].copy_from_slice(x);
                prev.copy_from_slice(x);
            } else {
                self.time[n..].iter_mut().for_each(|v| *v = 0.0);
                prev.iter_mut().for_each(|v| *v = 0.0);
            }
            self.prev_live[k] = act;
            let _ = self.fwd.process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_f);
            let o = slot * bins;
            for (b, c) in self.spec.iter().enumerate() {
                self.fdl_re[o + b] = c.re;
                self.fdl_im[o + b] = c.im;
            }
            self.fdl_live[slot] = true;
        }

        // per output: sum of path spectra x delayed input spectra, one inverse transform
        for e in 0..self.outputs {
            let mut any = false;
            for k in 0..self.inputs {
                let pi = self.path_of[e * self.inputs + k];
                if pi == NO_PATH {
                    continue;
                }
                let pi = pi as usize;
                for p in 0..parts {
                    if !self.h_live[pi * parts + p] {
                        continue;
                    }
                    let slot = (head + parts - p) % parts;
                    if !self.fdl_live[k * parts + slot] {
                        continue;
                    }
                    let ho = (pi * parts + p) * bins;
                    let xo = (k * parts + slot) * bins;
                    let hr = &self.h_re[ho..ho + bins];
                    let hi = &self.h_im[ho..ho + bins];
                    let xr = &self.fdl_re[xo..xo + bins];
                    let xi = &self.fdl_im[xo..xo + bins];
                    if !any {
                        simd::cmul(&mut self.acc_re[..bins], &mut self.acc_im[..bins], hr, hi, xr, xi);
                        any = true;
                    } else {
                        simd::cmac(&mut self.acc_re[..bins], &mut self.acc_im[..bins], hr, hi, xr, xi);
                    }
                }
            }
            let o = &mut out[e * n..(e + 1) * n];
            if !any {
                if !add {
                    o.iter_mut().for_each(|v| *v = 0.0);
                }
                continue;
            }
            for (b, c) in self.spec.iter_mut().enumerate() {
                *c = Complex::new(self.acc_re[b], self.acc_im[b]);
            }
            self.spec[0].im = 0.0;
            self.spec[bins - 1].im = 0.0;
            let _ = self.inv.process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_i);
            let y = &self.time[n..self.nfft];
            if add {
                simd::axpy(o, 1.0, y);
            } else {
                o.copy_from_slice(y);
            }
        }
    }
}
