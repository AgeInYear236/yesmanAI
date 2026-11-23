//! Q4_0 block quantization (ggml-style): 32 4-bit values + one scale per block
//! -> 0.625 bytes/weight (~0.94 GB for the 1.5B model).
//!
//! Weights are Q4, activations stay Q8_0 (like llama.cpp): quantizing the
//! activations down to 4 bits as well costs too much accuracy for little gain.

use rayon::prelude::*;

use crate::Bf16Tensor;
use crate::quant::{BLOCK, BlockQ8};

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct BlockQ4 {
    pub d: f32,
    /// low nibble -> weights 0..15, high nibble -> weights 16..31
    pub q: [u8; BLOCK / 2],
}

/// Quantized matrix, row-major `[rows, cols]`, `cols` must be a multiple of 32.
pub struct Q4Tensor {
    pub blocks: Vec<BlockQ4>,
    pub rows: usize,
    pub cols: usize,
}

impl Q4Tensor {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        assert_eq!(cols % BLOCK, 0);
        let blocks = data
            .par_chunks(cols)
            .flat_map(|row| quantize_row(row))
            .collect();
        Self { blocks, rows, cols }
    }

    pub fn from_bf16(t: &Bf16Tensor) -> Self {
        assert_eq!(t.cols % BLOCK, 0);
        let blocks = t
            .data
            .par_chunks(t.cols)
            .flat_map(|row| {
                let f: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
                quantize_row(&f)
            })
            .collect();
        Self { blocks, rows: t.rows, cols: t.cols }
    }

    /// Construct from the raw bytes of a `Vec<BlockQ4>` (cache file), copying.
    pub fn from_bytes(rows: usize, cols: usize, bytes: &[u8]) -> Self {
        let size = std::mem::size_of::<BlockQ4>();
        assert_eq!(size, 20, "unexpected BlockQ4 layout");
        assert_eq!(bytes.len(), (rows * cols / BLOCK) * size);
        let blocks = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4, bytes.len() / size)
        }
        .to_vec();
        Self { blocks, rows, cols }
    }

    pub fn as_bytes(&self) -> &[u8] {
        let size = std::mem::size_of::<BlockQ4>();
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr() as *const u8,
                self.blocks.len() * size,
            )
        }
    }

    pub fn blocks_per_row(&self) -> usize {
        self.cols / BLOCK
    }

    pub fn row(&self, r: usize) -> &[BlockQ4] {
        let bpr = self.blocks_per_row();
        &self.blocks[r * bpr..(r + 1) * bpr]
    }

    pub fn dequantize_row(&self, r: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.cols);
        for b in self.row(r) {
            for i in 0..BLOCK / 2 {
                out.push((((b.q[i] & 0x0f) as i32 - 8) as f32) * b.d);
            }
            for i in 0..BLOCK / 2 {
                out.push(((((b.q[i] >> 4) as i32) - 8) as f32) * b.d);
            }
        }
        out
    }
}

fn quantize_row(x: &[f32]) -> Vec<BlockQ4> {
    assert_eq!(x.len() % BLOCK, 0);
    x.chunks(BLOCK)
        .map(|c| {
            // ggml's Q4_0: keep the sign of the largest-magnitude value in d.
            let mut max = 0.0f32;
            for &v in c {
                if v.abs() > max.abs() {
                    max = v;
                }
            }
            let d = -max / 8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut q = [0u8; BLOCK / 2];
            for i in 0..BLOCK / 2 {
                let lo = (c[i] * id + 8.0).round().clamp(0.0, 15.0) as u8;
                let hi = (c[i + BLOCK / 2] * id + 8.0).round().clamp(0.0, 15.0) as u8;
                q[i] = lo | (hi << 4);
            }
            BlockQ4 { d, q }
        })
        .collect()
}

#[inline]
fn block_dot(a: &BlockQ8, b: &BlockQ4) -> f32 {
    let mut s = 0i32;
    for i in 0..BLOCK / 2 {
        let bl = (b.q[i] & 0x0f) as i32 - 8;
        let bh = (b.q[i] >> 4) as i32 - 8;
        s += a.q[i] as i32 * bl + a.q[i + BLOCK / 2] as i32 * bh;
    }
    (a.d * b.d) * s as f32
}

#[inline]
fn dot_row(a: &[BlockQ8], b: &[BlockQ4]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_row_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| block_dot(x, y)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_row_avx2(a: &[BlockQ8], b: &[BlockQ4]) -> f32 {
    use std::arch::x86_64::*;
    let mut total = 0.0f32;
    let eight = _mm256_set1_epi8(8);
    for (ba, bb) in a.iter().zip(b.iter()) {
        let s = unsafe {
            let va = _mm256_loadu_si256(ba.q.as_ptr() as *const __m256i);
            let raw = _mm_loadu_si128(bb.q.as_ptr() as *const __m128i);
            let mask = _mm_set1_epi8(0x0f);
            let lo = _mm_and_si128(raw, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
            let nib = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
            let vb = _mm256_sub_epi8(nib, eight);

            let a0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let a1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let b0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let b1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            let p = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), _mm256_madd_epi16(a1, b1));
            let h = _mm256_hadd_epi32(p, p);
            let h = _mm256_hadd_epi32(h, h);
            _mm_cvtsi128_si32(_mm256_castsi256_si128(h))
                + _mm_cvtsi128_si32(_mm256_extracti128_si256(h, 1))
        };
        total += (ba.d * bb.d) * s as f32;
    }
    total
}

/// `out[m,n] = x[m,k] @ w[n,k]^T`; x quantized to Q8_0, weights Q4_0.
///
/// Same work splitting as the other matmuls: decode (`m = 1`) needs output
/// columns distributed across rayon workers or 19/20 cores would idle.
pub fn matmul_q8q4(
    out: &mut [f32],
    xq: &[BlockQ8],
    w: &Q4Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(w.cols, k);
    assert!(out.len() >= m * n);
    let bpr = k / BLOCK;
    let out = &mut out[..m * n];
    if m > 1 {
        out.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
            let xrow = &xq[i * bpr..(i + 1) * bpr];
            for (j, o) in o.iter_mut().enumerate() {
                *o = dot_row(xrow, w.row(j));
            }
        });
    } else {
        let c = 256.min(n).max(1);
        out.par_chunks_mut(c).enumerate().for_each(|(ci, o)| {
            let j0 = ci * c;
            let xrow = &xq[..bpr];
            for (j, o) in o.iter_mut().enumerate() {
                *o = dot_row(xrow, w.row(j0 + j));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_rows;

    #[test]
    fn quantize_roundtrip_within_step() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.37).collect();
        let blocks = quantize_row(&x);
        let t = Q4Tensor { blocks: blocks.clone(), rows: 2, cols: 32 };
        let got = t.dequantize_row(0);
        let amax = x[..32].iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let step = amax / 8.0;
        for (g, v) in got.iter().zip(&x[..32]) {
            assert!((g - v).abs() <= step + 1e-4, "{g} vs {v} (step {step})");
        }
    }

    /// 4-bit weights are inherently lossy; guard against a broken kernel,
    /// not against quantization noise: dequant RMSE ~6%, dot error < 1%.
    #[test]
    fn q4_kernel_accuracy() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let k = 1536;
        let w: Vec<f32> = (0..4 * k).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let q4 = Q4Tensor::from_f32(&w, 4, k);
        let mut de_err = 0.0f32;
        for j in 0..4 {
            let got = q4.dequantize_row(j);
            for (g, v) in got.iter().zip(&w[j * k..(j + 1) * k]) {
                de_err += (g - v).powi(2);
            }
        }
        let want_norm: f32 = w.iter().map(|v| v * v).sum();
        let rmse = (de_err / want_norm).sqrt();
        assert!(rmse < 0.10, "q4 dequant too noisy: {rmse}");

        use crate::quant::quantize_rows;
        let xq = quantize_rows(&x, k);
        let mut got = vec![0.0; 4];
        matmul_q8q4(&mut got, &xq, &q4, 1, k, 4);
        for (j, g) in got.iter().enumerate() {
            let want: f32 = x.iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
            assert!(
                (g - want).abs() < 0.01 * want.abs().max(1.0),
                "dot[{j}] {g} vs {want}"
            );
        }
    }

    #[test]
    fn q8_times_q4_matches_f32() {
        let k = 128;
        let w: Vec<f32> = (0..4 * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();
        let q4 = Q4Tensor::from_f32(&w, 4, k);
        let xq = quantize_rows(&x, k);
        let mut got = vec![0.0; 4];
        matmul_q8q4(&mut got, &xq, &q4, 1, k, 4);
        for (j, g) in got.iter().enumerate() {
            let want: f32 = x.iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
            assert!((g - want).abs() < 0.15 * want.abs().max(1.0), "{g} vs {want}");
        }
    }
}
