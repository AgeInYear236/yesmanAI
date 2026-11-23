//! bf16 weight storage: exact copy of the checkpoint (Qwen2.5 ships bf16),
//! half the memory traffic of f32, converted to f32 inside the dot product.

use half::bf16;
use rayon::prelude::*;

pub struct Bf16Tensor {
    pub data: Vec<bf16>,
    pub rows: usize,
    pub cols: usize,
}

impl Bf16Tensor {
    pub fn from_raw(data: Vec<bf16>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let data = data.par_iter().map(|&v| bf16::from_f32(v)).collect();
        Self { data, rows, cols }
    }

    pub fn row(&self, r: usize) -> &[bf16] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn dequant_row(&self, r: usize) -> Vec<f32> {
        self.row(r).iter().map(|v| v.to_f32()).collect()
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[bf16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[bf16]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        s += *x * y.to_f32();
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[bf16]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    let mut out = unsafe {
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            // bf16 -> f32 is just a 16-bit left shift of the raw bits
            let v = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vf, acc);
            i += 8;
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
        _mm_cvtss_f32(s)
    };
    while i < n {
        out += a[i] * b[i].to_f32();
        i += 1;
    }
    out
}

/// `out[m,n] = x[m,k] @ w[n,k]^T`, weights in bf16.
///
/// For a single decode token (`m = 1`) there is only one input row, so we must
/// split the *output* dimension across rayon workers; otherwise one core does
/// the entire 152k-row lm_head and the other 19 idle.
pub fn matmul(out: &mut [f32], x: &[f32], w: &Bf16Tensor, m: usize, k: usize, n: usize) {
    assert_eq!(w.cols, k);
    assert!(out.len() >= m * n);
    let out = &mut out[..m * n];
    if m > 1 {
        out.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
            let xrow = &x[i * k..(i + 1) * k];
            for (j, o) in o.iter_mut().enumerate() {
                *o = dot(xrow, w.row(j));
            }
        });
    } else {
        let c = crate::ops::out_chunk(n);
        out.par_chunks_mut(c).enumerate().for_each(|(ci, o)| {
            let j0 = ci * c;
            for (j, o) in o.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += dot(&x[i * k..(i + 1) * k], w.row(j0 + j));
                }
                *o = s;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_matmul_matches_f32() {
        let k = 64;
        let w: Vec<f32> = (0..3 * k).map(|i| ((i % 11) as f32 - 5.0) / 5.0).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
        let bw = Bf16Tensor::from_f32(&w, 3, k);
        let mut got = vec![0.0; 3];
        matmul(&mut got, &x, &bw, 1, k, 3);
        for (j, g) in got.iter().enumerate() {
            let terms: Vec<f32> = x
                .iter()
                .zip(&w[j * k..(j + 1) * k])
                .map(|(a, b)| a * b)
                .collect();
            let want: f32 = terms.iter().sum();
            // bf16 rounds each weight to ~2^-9; with cancellation the absolute
            // error is bounded by the sum of magnitudes, not by |want|.
            let mag: f32 = terms.iter().map(|v| v.abs()).sum();
            let tol = 0.004 * mag + 1e-6;
            assert!((g - want).abs() < tol, "{g} vs {want} (tol {tol})");
        }
    }
}
