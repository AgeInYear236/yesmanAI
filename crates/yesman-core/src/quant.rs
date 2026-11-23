//! Q8_0 block quantization (ggml-compatible layout): 32 int8 values per block
//! plus one scale -> ~8.5 bits/weight.
//!
//! Activations are quantized on the fly, so the inner loop is int8 x int8 with
//! i32 accumulation: 4x less memory traffic than f32 and 2x the ALU width.

use rayon::prelude::*;

pub const BLOCK: usize = 32;

/// Scale is held as f32 (not ggml's f16) on purpose: it is read once per
/// 32 weights in the innermost dot loop, so avoiding two f16->f32 conversions
/// per block is worth the extra 2 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockQ8 {
    pub d: f32,
    pub q: [i8; BLOCK],
}

/// Quantized matrix, row-major `[rows, cols]`, `cols` must be a multiple of 32.
pub struct Q8Tensor {
    pub blocks: Vec<BlockQ8>,
    pub rows: usize,
    pub cols: usize,
}

impl Q8Tensor {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        assert_eq!(cols % BLOCK, 0, "cols must be a multiple of {BLOCK}");
        let blocks = data
            .par_chunks(cols)
            .flat_map(|row| quantize_row(row))
            .collect();
        Self { blocks, rows, cols }
    }

    /// Quantize straight from a bf16 tensor (no f32 materialization).
    pub fn from_bf16(t: &crate::Bf16Tensor) -> Self {
        assert_eq!(t.cols % BLOCK, 0);
        let blocks = t
            .data
            .par_chunks(t.cols)
            .flat_map(|row| quantize_row_bf16(row))
            .collect();
        Self { blocks, rows: t.rows, cols: t.cols }
    }

    /// Construct from the raw bytes of a `Vec<BlockQ8>` (cache file).
    ///
    /// The copy is parallel: a single-threaded 1.7 GB memcpy from the mmap is
    /// dominated by page faults and takes ~4 s, ~3x slower than this.
    pub fn from_bytes(rows: usize, cols: usize, bytes: &[u8]) -> Self {
        let size = std::mem::size_of::<BlockQ8>();
        assert_eq!(size, 36, "unexpected BlockQ8 layout");
        assert_eq!(bytes.len(), (rows * cols / BLOCK) * size);
        let n = bytes.len() / size;
        let mut blocks = vec![BlockQ8::default(); n];
        const CHUNK: usize = 4096;
        blocks
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, dst)| {
                let src = unsafe {
                    std::slice::from_raw_parts(
                        bytes[i * CHUNK * size..].as_ptr() as *const BlockQ8,
                        dst.len(),
                    )
                };
                dst.copy_from_slice(src);
            });
        Self { blocks, rows, cols }
    }

    /// View the blocks as raw bytes for the cache writer.
    pub fn as_bytes(&self) -> &[u8] {
        let size = std::mem::size_of::<BlockQ8>();
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

    pub fn row(&self, r: usize) -> &[BlockQ8] {
        let bpr = self.blocks_per_row();
        &self.blocks[r * bpr..(r + 1) * bpr]
    }

    pub fn dequantize_row(&self, r: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.cols);
        for b in self.row(r) {
            let d = b.d;
            out.extend(b.q.iter().map(|&v| v as f32 * d));
        }
        out
    }
}

pub fn quantize_row(x: &[f32]) -> Vec<BlockQ8> {
    assert_eq!(x.len() % BLOCK, 0);
    x.chunks(BLOCK)
        .map(|c| {
            let amax = c.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut q = [0i8; BLOCK];
            for (i, v) in c.iter().enumerate() {
                let v = *v;
                q[i] = (v * id).round().clamp(-127.0, 127.0) as i8;
            }
            BlockQ8 { d, q }
        })
        .collect()
}

fn quantize_row_bf16(x: &[half::bf16]) -> Vec<BlockQ8> {
    assert_eq!(x.len() % BLOCK, 0);
    x.chunks(BLOCK)
        .map(|c| {
            let amax = c.iter().fold(0.0f32, |a, v| a.max(v.to_f32().abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut q = [0i8; BLOCK];
            for (i, v) in c.iter().enumerate() {
                let v = v.to_f32();
                q[i] = (v * id).round().clamp(-127.0, 127.0) as i8;
            }
            BlockQ8 { d, q }
        })
        .collect()
}

/// Quantize every row of an `[m, k]` activation matrix.
///
/// During decode `m` is 1 and there is nothing to parallelize here; going
/// through rayon for such a tiny job costs far more than the work itself
/// (and it is invoked once per weight matrix, ~300 times per token).
pub fn quantize_rows(x: &[f32], k: usize) -> Vec<BlockQ8> {
    let m = x.len() / k;
    if m <= 2 {
        x.chunks(k).flat_map(|row| quantize_row(row)).collect()
    } else {
        x.par_chunks(k).flat_map(|row| quantize_row(row)).collect()
    }
}

#[inline]
fn dot_blocks(a: &[BlockQ8], b: &[BlockQ8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_blocks_avx2(a, b) };
        }
    }
    dot_blocks_scalar(a, b)
}

fn dot_blocks_scalar(a: &[BlockQ8], b: &[BlockQ8]) -> f32 {
    let mut sum = 0.0f32;
    for (ba, bb) in a.iter().zip(b.iter()) {
        let mut s = 0i32;
        for i in 0..BLOCK {
            s += ba.q[i] as i32 * bb.q[i] as i32;
        }
        sum += ba.d * bb.d * s as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_blocks_avx2(a: &[BlockQ8], b: &[BlockQ8]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = 0.0f32;
    for (ba, bb) in a.iter().zip(b.iter()) {
        let s = unsafe {
            let va = _mm256_loadu_si256(ba.q.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(bb.q.as_ptr() as *const __m256i);
            // widen i8 -> i16 (two halves), then i16*i16 -> i32 accumulate
            let a0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let a1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let b0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let b1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            let p = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), _mm256_madd_epi16(a1, b1));
            // horizontal add 8 x i32 -> 1: after two hadds, lane 0 of each
            // 128-bit half holds the sum of its 4 products.
            let h = _mm256_hadd_epi32(p, p);
            let h = _mm256_hadd_epi32(h, h);
            let lo = _mm256_castsi256_si128(h);
            let hi = _mm256_extracti128_si256(h, 1);
            _mm_cvtsi128_si32(lo) + _mm_cvtsi128_si32(hi)
        };
        sum += (ba.d * bb.d) * s as f32;
    }
    sum
}

/// `out[m,n] = x[m,k] @ w[n,k]^T` with both sides quantized.
///
/// Like [`crate::bf16::matmul`], the output dimension is split when `m` is
/// small so single-token decode still uses all cores on the huge lm_head.
pub fn matmul_q8(out: &mut [f32], xq: &[BlockQ8], w: &Q8Tensor, m: usize, k: usize, n: usize) {
    assert_eq!(w.cols, k);
    assert!(out.len() >= m * n);
    let bpr = k / BLOCK;
    let out = &mut out[..m * n];
    // In the single-token case split by output columns; with several input
    // rows split by rows (chunks of `c` would straddle row boundaries).
    if m > 1 {
        out.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
            let xrow = &xq[i * bpr..(i + 1) * bpr];
            for (j, o) in o.iter_mut().enumerate() {
                *o = dot_blocks(xrow, w.row(j));
            }
        });
    } else {
        let c = crate::ops::out_chunk(n);
        out.par_chunks_mut(c).enumerate().for_each(|(ci, o)| {
            let j0 = ci * c;
            for (j, o) in o.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += dot_blocks(&xq[i * bpr..(i + 1) * bpr], w.row(j0 + j));
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
    fn quantize_preserves_values_approximately() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.37).collect();
        let blocks = quantize_row(&x);
        // error is bounded by the block scale (amax/127), not by |v|: small
        // values inside a block with a large amax lose relative precision.
        let mut err = 0.0f32;
        for (b, chunk) in blocks.iter().zip(x.chunks(BLOCK)) {
            let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            for (i, v) in chunk.iter().enumerate() {
                let got = b.q[i] as f32 * b.d;
                err = err.max((got - v).abs() / amax.max(1e-6));
            }
        }
        assert!(err < 1.0 / 127.0 + 1e-3, "error larger than one quant step: {err}");
    }

    #[test]
    fn q8_matmul_close_to_f32() {
        let k = 128;
        let w: Vec<f32> = (0..4 * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();
        let qt = Q8Tensor::from_f32(&w, 4, k);
        let xq = quantize_rows(&x, k);
        let mut got = vec![0.0; 4];
        matmul_q8(&mut got, &xq, &qt, 1, k, 4);
        for (j, g) in got.iter().enumerate() {
            let want: f32 = x.iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
            assert!((g - want).abs() < 0.05 * want.abs().max(1.0), "{g} vs {want}");
        }
    }
}

