//! Q4_K-style block quantization.
//!
//! A super-block holds 256 weights and is split into 8 sub-blocks of 32. Each
//! sub-block is affine-quantized to 4 bits with its own scale and min-offset,
//! normalized by two f32 super-scales. Result: 152 bytes per 256 weights =
//! 0.594 byte/weight (less than Q4_0's 0.625) with markedly better accuracy,
//! because a per-32-value min/scale fits skewed weight distributions.
//!
//! Weight = (d * sc_j) * q - (dmin * m_j), q in 0..15.
//! Activations stay Q8_0 (32-value blocks), which aligns with the sub-blocks.
//!
//! Note this is a clean-room Q4_K *layout*, not ggml bit-compatible: the 16
//! scale/min values are stored as one byte each (ggml packs them into 6-bit
//! fields, which needs an RMSE search to stay accurate), the min-offset is
//! signed (bias 127) and the super-scales are f32 (f16 would be subnormal
//! at Qwen2's ~1e-5 weight scales). We load from safetensors, so GGUF
//! compatibility is not needed.

use rayon::prelude::*;

use crate::Bf16Tensor;
use crate::quant::{BLOCK, BlockQ8};

pub const SUB: usize = 32;
pub const NSUB: usize = 8;
pub const SUPER: usize = SUB * NSUB; // 256
const QBYTES: usize = 128; // 256 nibbles packed into bytes
const SCALES: usize = 16; // 8 scales + 8 signed min-offsets, one byte each

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4K {
    /// f32, not f16: `d = max_scale/63` is ~1e-5 for Qwen2 weights, i.e.
    /// subnormal in f16 where relative error explodes.
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; SCALES],
    pub qs: [u8; QBYTES],
}

/// Scale bytes 0..7, min-offset bytes 8..15 (bias 127).
#[inline]
fn set8(buf: &mut [u8; SCALES], idx: usize, v: u8) {
    buf[idx] = v;
}

#[inline]
fn get8(buf: &[u8; SCALES], idx: usize) -> u8 {
    buf[idx]
}

impl Default for BlockQ4K {
    fn default() -> Self {
        Self {
            d: 0.0,
            dmin: 0.0,
            scales: [127u8; SCALES],
            qs: [0u8; QBYTES],
        }
    }
}

impl BlockQ4K {
    fn quantize(c: &[f32]) -> Self {
        debug_assert_eq!(c.len(), SUPER);
        let mut scales = [0.0f32; NSUB];
        let mut offsets = [0.0f32; NSUB];
        let mut qs = [0u8; QBYTES];
        for j in 0..NSUB {
            let sub = &c[j * SUB..(j + 1) * SUB];
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in sub {
                min = min.min(v);
                max = max.max(v);
            }
            let scale = (max - min) / 15.0;
            let offset = -min;
            scales[j] = scale;
            offsets[j] = offset;
            let id = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for i in 0..SUB / 2 {
                let lo = ((sub[i] - min) * id).round().clamp(0.0, 15.0) as u8;
                let hi = ((sub[i + SUB / 2] - min) * id).round().clamp(0.0, 15.0) as u8;
                qs[j * (SUB / 2) + i] = lo | (hi << 4);
            }
        }
        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_offset = offsets.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = max_scale / 255.0;
        let dmin = max_offset / 127.0;
        let mut packed = [0u8; SCALES];
        for j in 0..NSUB {
            let sc = if d > 0.0 {
                (scales[j] / d).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
            // signed 8-bit min offset, bias 127
            let m = if dmin > 0.0 {
                ((offsets[j] / dmin).round() + 127.0).clamp(0.0, 255.0) as u8
            } else {
                127
            };
            set8(&mut packed, j, sc);
            set8(&mut packed, NSUB + j, m);
        }
        BlockQ4K { d, dmin, scales: packed, qs }
    }

    /// `(sc_d, min_d)` pairs for the 8 sub-blocks.
    #[inline]
    fn factors(&self) -> [(f32, f32); NSUB] {
        let d = self.d;
        let dmin = self.dmin;
        let mut out = [(0.0f32, 0.0f32); NSUB];
        for j in 0..NSUB {
            let sc = get8(&self.scales, j) as f32;
            let m = get8(&self.scales, NSUB + j) as f32 - 127.0;
            out[j] = (d * sc, dmin * m);
        }
        out
    }

    /// Decoded nibbles (0..15) of one 32-value sub-block.
    #[inline]
    fn nibbles(&self, j: usize, out: &mut [i8; SUB]) {
        let base = j * (SUB / 2);
        for i in 0..SUB / 2 {
            let b = self.qs[base + i];
            out[i] = (b & 0x0f) as i8;
            out[i + SUB / 2] = (b >> 4) as i8;
        }
    }
}

pub struct Q4KTensor {
    pub blocks: Vec<BlockQ4K>,
    pub rows: usize,
    pub cols: usize,
}

impl Q4KTensor {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        assert_eq!(cols % SUPER, 0, "cols must be a multiple of {SUPER}");
        let blocks = data
            .par_chunks(cols)
            .flat_map(|row| {
                row.chunks(SUPER)
                    .map(|c| BlockQ4K::quantize(c))
                    .collect::<Vec<_>>()
            })
            .collect();
        Self { blocks, rows, cols }
    }

    pub fn from_bf16(t: &Bf16Tensor) -> Self {
        assert_eq!(t.cols % SUPER, 0, "cols must be a multiple of {SUPER}");
        let blocks = t
            .data
            .par_chunks(t.cols)
            .flat_map(|row| {
                let f: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
                f.chunks(SUPER)
                    .map(|c| BlockQ4K::quantize(c))
                    .collect::<Vec<_>>()
            })
            .collect();
        Self { blocks, rows: t.rows, cols: t.cols }
    }

    /// Reconstruct from the raw bytes of a `Vec<BlockQ4K>` (cache file).
    pub fn from_bytes(rows: usize, cols: usize, bytes: &[u8]) -> Self {
        let size = std::mem::size_of::<BlockQ4K>();
        assert_eq!(size, 152, "unexpected BlockQ4K layout");
        assert_eq!(bytes.len(), (rows * cols / SUPER) * size);
        let n = bytes.len() / size;
        let mut blocks = vec![BlockQ4K::default(); n];
        const CHUNK: usize = 4096;
        blocks
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, dst)| {
                let src = unsafe {
                    std::slice::from_raw_parts(
                        bytes[i * CHUNK * size..].as_ptr() as *const BlockQ4K,
                        dst.len(),
                    )
                };
                dst.copy_from_slice(src);
            });
        Self { blocks, rows, cols }
    }

    pub fn as_bytes(&self) -> &[u8] {
        let size = std::mem::size_of::<BlockQ4K>();
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr() as *const u8,
                self.blocks.len() * size,
            )
        }
    }

    pub fn blocks_per_row(&self) -> usize {
        self.cols / SUPER
    }

    pub fn row(&self, r: usize) -> &[BlockQ4K] {
        let bpr = self.blocks_per_row();
        &self.blocks[r * bpr..(r + 1) * bpr]
    }

    pub fn dequantize_row(&self, r: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.cols);
        let mut nib = [0i8; SUB];
        for b in self.row(r) {
            let f = b.factors();
            for j in 0..NSUB {
                b.nibbles(j, &mut nib);
                let (sd, md) = f[j];
                for &q in nib.iter() {
                    out.push(sd * q as f32 - md);
                }
            }
        }
        out
    }
}

#[inline]
fn dot_row(a: &[BlockQ8], b: &[BlockQ4K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_row_avx2(a, b) };
        }
    }
    dot_row_scalar_impl(a, b)
}

/// Scalar reference: iterates super-block by super-block, each with 8 Q8
/// activation blocks. Also used as ground truth for the AVX2 kernel.
#[inline]
fn dot_row_scalar_impl(a: &[BlockQ8], b: &[BlockQ4K]) -> f32 {
    let mut total = 0.0f32;
    let mut nib = [0i8; SUB];
    for (bi, bb) in b.iter().enumerate() {
        let f = bb.factors();
        for j in 0..NSUB {
            let ba = &a[bi * NSUB + j];
            bb.nibbles(j, &mut nib);
            let (sd, md) = f[j];
            let mut sq = 0i32;
            let mut sa = 0i32;
            for i in 0..SUB {
                sq += ba.q[i] as i32 * nib[i] as i32;
                sa += ba.q[i] as i32;
            }
            total += ba.d * (sd * sq as f32 - md * sa as f32);
        }
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_row_avx2(a: &[BlockQ8], b: &[BlockQ4K]) -> f32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let mut total = 0.0f32;
    for (bi, bb) in b.iter().enumerate() {
        let f = bb.factors();
        for j in 0..NSUB {
            let ba = &a[bi * NSUB + j];
            let (sd, md) = f[j];
            let (sq, sa) = unsafe {
                let va = _mm256_loadu_si256(ba.q.as_ptr() as *const __m256i);
                let raw = _mm_loadu_si128(
                    bb.qs[j * (SUB / 2)..].as_ptr() as *const __m128i
                );
                let mask = _mm_set1_epi8(0x0f);
                let lo = _mm_and_si128(raw, mask);
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
                let nib = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

                let a0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
                let a1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
                let n0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(nib));
                let n1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(nib, 1));

                let p = _mm256_add_epi32(
                    _mm256_madd_epi16(a0, n0),
                    _mm256_madd_epi16(a1, n1),
                );
                let s = _mm256_add_epi32(
                    _mm256_madd_epi16(a0, ones),
                    _mm256_madd_epi16(a1, ones),
                );
                let hp = _mm256_hadd_epi32(p, p);
                let hp = _mm256_hadd_epi32(hp, hp);
                let hs = _mm256_hadd_epi32(s, s);
                let hs = _mm256_hadd_epi32(hs, hs);
                let sq = _mm_cvtsi128_si32(_mm256_castsi256_si128(hp))
                    + _mm_cvtsi128_si32(_mm256_extracti128_si256(hp, 1));
                let sa = _mm_cvtsi128_si32(_mm256_castsi256_si128(hs))
                    + _mm_cvtsi128_si32(_mm256_extracti128_si256(hs, 1));
                (sq, sa)
            };
            total += ba.d * (sd * sq as f32 - md * sa as f32);
        }
    }
    total
}

/// `out[m,n] = x[m,k] @ w[n,k]^T`; x quantized to Q8_0, weights Q4_K.
///
/// `k` must be a multiple of 256 (true for all Qwen2.5-1.5B matrices:
/// 1536, 8960 and 256 all are). Output columns are split across rayon
/// workers for single-token decode.
pub fn matmul_q8q4k(
    out: &mut [f32],
    xq: &[BlockQ8],
    w: &Q4KTensor,
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(w.cols, k);
    assert_eq!(k % SUPER, 0);
    assert!(out.len() >= m * n);
    let bpr = k / BLOCK; // Q8 activation blocks per row
    let out = &mut out[..m * n];
    if m > 1 {
        out.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
            let xrow = &xq[i * bpr..(i + 1) * bpr];
            for (j, o) in o.iter_mut().enumerate() {
                *o = dot_row(xrow, w.row(j));
            }
        });
    } else {
        let c = crate::ops::out_chunk(n);
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
    fn scale_packing_roundtrip() {
        let mut buf = [0u8; SCALES];
        let vals: Vec<u8> = (0..16).map(|i| (i * 17 % 256) as u8).collect();
        for (i, v) in vals.iter().enumerate() {
            set8(&mut buf, i, *v);
        }
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(get8(&buf, i), *v, "entry {i}");
        }
    }

    #[test]
    fn roundtrip_and_dot_accuracy() {
        let mut seed = 999u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let k = SUPER * 2;
        let w: Vec<f32> = (0..4 * k).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let q = Q4KTensor::from_f32(&w, 4, k);

        let mut de_err = 0.0f32;
        for j in 0..4 {
            let got = q.dequantize_row(j);
            for (g, v) in got.iter().zip(&w[j * k..(j + 1) * k]) {
                de_err += (g - v).powi(2);
            }
        }
        let norm: f32 = w.iter().map(|v| v * v).sum();
        let rmse = (de_err / norm).sqrt();
        println!("Q4_K dequant RMSE/||w|| = {rmse:.4}");
        assert!(rmse < 0.04, "too noisy: {rmse}");

        let xq = quantize_rows(&x, k);
        let mut got = vec![0.0; 4];
        matmul_q8q4k(&mut got, &xq, &q, 1, k, 4);
        for (j, g) in got.iter().enumerate() {
            let want: f32 = x.iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
            assert!(
                (g - want).abs() < 0.01 * want.abs().max(1.0),
                "dot[{j}] {g} vs {want}"
            );
        }
    }

    /// Diagnostic: compare Q4_K and Q4_0 dots against f32 on Gaussian weights
    /// (the distribution of a real checkpoint), 8-bit activations for both.
    #[test]
    fn compare_with_q4_0_on_gaussian() {
        let mut seed = 777u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let k = SUPER * 3;
        let w: Vec<f32> = (0..16 * k).map(|_| rnd() * 0.02).collect();
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let q4k = Q4KTensor::from_f32(&w, 16, k);
        let q4 = crate::q4::Q4Tensor::from_f32(&w, 16, k);
        let xq = quantize_rows(&x, k);
        let mut a = vec![0.0; 16];
        let mut b = vec![0.0; 16];
        matmul_q8q4k(&mut a, &xq, &q4k, 1, k, 16);
        crate::q4::matmul_q8q4(&mut b, &xq, &q4, 1, k, 16);
        let mut err_k = 0.0f32;
        let mut err_0 = 0.0f32;
        for j in 0..16 {
            let want: f32 = x.iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
            err_k += (a[j] - want).powi(2);
            err_0 += (b[j] - want).powi(2);
        }
        println!("Q4_K dot RMSE {:.6}, Q4_0 dot RMSE {:.6}", (err_k / 16.0).sqrt(), (err_0 / 16.0).sqrt());
        assert!(
            err_k < err_0 * 1.10,
            "Q4_K dot must be at least as accurate as Q4_0 on Gaussian weights"
        );
    }

    #[test]
    fn scalar_matches_avx2() {
        let mut seed = 42u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let k = SUPER * 3;
        let w: Vec<f32> = (0..5 * k).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let q = Q4KTensor::from_f32(&w, 5, k);
        let xq = quantize_rows(&x, k);
        for j in 0..5 {
            let s = dot_row_scalar_impl(&xq, q.row(j));
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                let v = unsafe { dot_row_avx2(&xq, q.row(j)) };
                assert!((s - v).abs() < 1e-2 * s.abs().max(1.0), "{s} vs {v}");
            }
        }
    }
}
