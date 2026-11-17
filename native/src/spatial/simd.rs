//! Hot inner loops of the spatial path and the limiter, written so LLVM vectorises them, and
//! compiled twice on x86_64: a baseline (SSE2) version and an AVX2 version picked at run time.
//! Rust never contracts a * b + c into an FMA and every lane keeps the same summation order,
//! so both versions give bit-identical results (offline renders stay deterministic across
//! machines). No allocation, no locks.

macro_rules! multiversion {
    ($(#[$m:meta])* pub fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$m])*
        #[inline]
        pub fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            fn generic($($arg: $ty),*) $(-> $ret)? $body

            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? {
                    generic($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU supports AVX2 (checked above).
                    return unsafe { avx2($($arg),*) };
                }
            }
            generic($($arg),*)
        }
    };
}

multiversion! {
    /// out += a x
    pub fn axpy(out: &mut [f32], a: f32, x: &[f32]) {
        let m = out.len().min(x.len());
        for (o, v) in out[..m].iter_mut().zip(&x[..m]) {
            *o += a * *v;
        }
    }
}

multiversion! {
    /// out += a x + d xr
    pub fn axpy2(out: &mut [f32], a: f32, x: &[f32], d: f32, xr: &[f32]) {
        let m = out.len().min(x.len()).min(xr.len());
        for ((o, v), w) in out[..m].iter_mut().zip(&x[..m]).zip(&xr[..m]) {
            *o += a * *v + d * *w;
        }
    }
}

multiversion! {
    /// out = a x
    pub fn scale_into(out: &mut [f32], a: f32, x: &[f32]) {
        let m = out.len().min(x.len());
        for (o, v) in out[..m].iter_mut().zip(&x[..m]) {
            *o = a * *v;
        }
    }
}

multiversion! {
    /// out = x * (g0 + dg r), r the block ramp
    pub fn ramp_into(out: &mut [f32], x: &[f32], g0: f32, dg: f32, ramp: &[f32]) {
        let m = out.len().min(x.len()).min(ramp.len());
        for ((o, v), r) in out[..m].iter_mut().zip(&x[..m]).zip(&ramp[..m]) {
            *o = *v * (g0 + dg * *r);
        }
    }
}

multiversion! {
    /// out = x * r
    pub fn mul_into(out: &mut [f32], x: &[f32], r: &[f32]) {
        let m = out.len().min(x.len()).min(r.len());
        for ((o, v), w) in out[..m].iter_mut().zip(&x[..m]).zip(&r[..m]) {
            *o = *v * *w;
        }
    }
}

multiversion! {
    /// (acc_re, acc_im) = h x (split complex)
    pub fn cmul(acc_re: &mut [f32], acc_im: &mut [f32], hr: &[f32], hi: &[f32], xr: &[f32], xi: &[f32]) {
        let m = acc_re.len().min(acc_im.len()).min(hr.len()).min(hi.len()).min(xr.len()).min(xi.len());
        let (ar, ai) = (&mut acc_re[..m], &mut acc_im[..m]);
        let (hr, hi, xr, xi) = (&hr[..m], &hi[..m], &xr[..m], &xi[..m]);
        for b in 0..m {
            ar[b] = hr[b] * xr[b] - hi[b] * xi[b];
            ai[b] = hr[b] * xi[b] + hi[b] * xr[b];
        }
    }
}

multiversion! {
    /// (acc_re, acc_im) += h x (split complex)
    pub fn cmac(acc_re: &mut [f32], acc_im: &mut [f32], hr: &[f32], hi: &[f32], xr: &[f32], xi: &[f32]) {
        let m = acc_re.len().min(acc_im.len()).min(hr.len()).min(hi.len()).min(xr.len()).min(xi.len());
        let (ar, ai) = (&mut acc_re[..m], &mut acc_im[..m]);
        let (hr, hi, xr, xi) = (&hr[..m], &hi[..m], &xr[..m], &xi[..m]);
        for b in 0..m {
            ar[b] += hr[b] * xr[b] - hi[b] * xi[b];
            ai[b] += hr[b] * xi[b] + hi[b] * xr[b];
        }
    }
}

multiversion! {
    /// max |x| (0 for an empty slice)
    pub fn max_abs(x: &[f32]) -> f32 {
        let mut acc = [0.0f32; 8];
        let chunks = x.chunks_exact(8);
        let rest = chunks.remainder();
        for c in chunks {
            for l in 0..8 {
                acc[l] = acc[l].max(c[l].abs());
            }
        }
        let mut m = 0.0f32;
        for v in acc.iter().chain(rest.iter()) {
            m = m.max(v.abs());
        }
        m
    }
}

multiversion! {
    /// All 8 polyphase outputs at once: y[p] = sum_i kt[i][p] win[i] (kt: taps x 8, oldest first).
    pub fn poly8(kt: &[[f32; 8]], win: &[f32]) -> [f32; 8] {
        let mut acc = [0.0f32; 8];
        for (k, w) in kt.iter().zip(win) {
            for l in 0..8 {
                acc[l] += k[l] * *w;
            }
        }
        acc
    }
}

multiversion! {
    /// out[i] = sum_k m[k] x_k[i] over channel-major x (channels x n); first term overwrites.
    pub fn sum_rows_into(out: &mut [f32], x: &[f32], n: usize, gains: &[f32]) {
        let m = out.len().min(n);
        out[..m].iter_mut().for_each(|v| *v = 0.0);
        for (k, g) in gains.iter().enumerate() {
            if *g == 0.0 || (k + 1) * n > x.len() {
                continue;
            }
            let row = &x[k * n..k * n + m];
            for (o, v) in out[..m].iter_mut().zip(row) {
                *o += *g * *v;
            }
        }
    }
}
