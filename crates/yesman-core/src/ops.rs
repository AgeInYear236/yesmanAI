//! Primitive ops for transformer inference. Everything is f32, row-major.
//! The kernels are deliberately explicit (no BLAS, no autograd) so that the
//! math stays auditable and portable.

use rayon::prelude::*;

/// Output columns per rayon task for single-token decode.
///
/// A fixed small chunk (e.g. 256 outputs) creates thousands of tiny tasks per
/// token; the dispatch overhead dominates the actual dot products. Aim for
/// ~2 tasks per worker so memory bandwidth is saturated with minimal
/// scheduling, while keeping enough slack for work stealing.
pub fn out_chunk(n: usize) -> usize {
    let workers = rayon::current_num_threads().max(1);
    n.div_ceil(2 * workers).max(64).min(n.max(1))
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut t0 = 0.0f32;
    let mut t1 = 0.0f32;
    let mut t2 = 0.0f32;
    let mut t3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        t0 += a[i] * b[i];
        t1 += a[i + 1] * b[i + 1];
        t2 += a[i + 2] * b[i + 2];
        t3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = (t0 + t1) + (t2 + t3);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    let mut out = unsafe {
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
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
        out += a[i] * b[i];
        i += 1;
    }
    out
}

/// `out[m,n] = x[m,k] @ w[n,k]^T`
///
/// Weights are stored as they come out of safetensors, i.e. `[out_features,
/// in_features]`, so the inner loop walks a contiguous weight row.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], m: usize, k: usize, n: usize) {
    assert!(out.len() >= m * n, "out too small");
    assert!(x.len() >= m * k, "x too small");
    assert!(w.len() >= n * k, "w too small");
    let out = &mut out[..m * n];

    // Tall input (prefill / batched): one rayon task per input row.
    // Short input (single-token decode): split the output dimension instead,
    // otherwise all cores but one would idle.
    if m > 1 {
        out.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
            matmul_rows(o, &x[i * k..(i + 1) * k], w, k);
        });
    } else {
        let c = out_chunk(n);
        out.par_chunks_mut(c).enumerate().for_each(|(ci, o)| {
            let j0 = ci * c;
            o.fill(0.0);
            for i in 0..m {
                let xrow = &x[i * k..(i + 1) * k];
                for (j, o) in o.iter_mut().enumerate() {
                    *o += dot(xrow, &w[(j0 + j) * k..(j0 + j + 1) * k]);
                }
            }
        });
    }
}

fn matmul_rows(o: &mut [f32], xrow: &[f32], w: &[f32], k: usize) {
    for (j, o) in o.iter_mut().enumerate() {
        *o = dot(xrow, &w[j * k..(j + 1) * k]);
    }
}

/// Same as [`matmul`] with a per-column bias added (Qwen2 has q/k/v biases).
pub fn matmul_bias(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    matmul(out, x, w, m, k, n);
    for row in out[..m * n].chunks_mut(n) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// RMSNorm over the last dimension: `x / sqrt(mean(x^2) + eps) * w`
///
/// A rayon task for a single 1536-element row costs more than the row itself;
/// only parallelize when there are several rows (prefill).
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let d = weight.len();
    let rows = x.len() / d;
    let row = |o: &mut [f32], xi: &[f32]| {
        let ss = dot(xi, xi) / d as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        for ((o, v), w) in o.iter_mut().zip(xi).zip(weight) {
            *o = *v * scale * w;
        }
    };
    if rows <= 2 {
        for (o, xi) in out.chunks_mut(d).zip(x.chunks(d)) {
            row(o, xi);
        }
    } else {
        out.par_chunks_mut(d)
            .zip(x.par_chunks(d))
            .for_each(|(o, xi)| row(o, xi));
    }
}

pub fn add_into(a: &mut [f32], b: &[f32]) {
    if a.len() <= 16384 {
        for (a, b) in a.iter_mut().zip(b) {
            *a += b;
        }
    } else {
        a.par_iter_mut()
            .zip(b.par_iter())
            .for_each(|(a, b)| *a += b);
    }
}

/// `out = silu(gate) * up` (SwiGLU feed-forward)
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    if out.len() <= 16384 {
        for (o, (g, u)) in out.iter_mut().zip(gate.iter().zip(up.iter())) {
            *o = (g / (1.0 + (-g).exp())) * u;
        }
    } else {
        out.par_iter_mut()
            .zip(gate.par_iter().zip(up.par_iter()))
            .for_each(|(o, (g, u))| *o = (g / (1.0 + (-g).exp())) * u);
    }
}

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Rotary embedding, HuggingFace/Qwen2 flavour: the vector is split in two
/// halves (`rotate_half`), not interleaved. `cos`/`sin` are `[max_pos, head_dim]`.
pub fn rope(x: &mut [f32], cos: &[f32], sin: &[f32], pos: usize, head_dim: usize) {
    let half = head_dim / 2;
    let c = &cos[pos * head_dim..pos * head_dim + half];
    let s = &sin[pos * head_dim..pos * head_dim + half];
    for head in x.chunks_mut(head_dim) {
        for d in 0..half {
            let a = head[d];
            let b = head[d + half];
            head[d] = a * c[d] - b * s[d];
            head[d + half] = b * c[d] + a * s[d];
        }
    }
}

/// Precompute the RoPE tables: `theta^(2i/head_dim)` for `i in 0..head_dim/2`,
/// duplicated across the two halves exactly like HF's `cat(freqs, freqs)`.
pub fn rope_tables(max_pos: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; max_pos * head_dim];
    let mut sin = vec![0.0f32; max_pos * head_dim];
    for pos in 0..max_pos {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let v = pos as f32 * freq;
            let (s, c) = v.sin_cos();
            cos[pos * head_dim + i] = c;
            cos[pos * head_dim + i + half] = c;
            sin[pos * head_dim + i] = s;
            sin[pos * head_dim + i + half] = s;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_naive() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // [2,2]
        let w = vec![1.0f32, 0.0, 0.0, 1.0]; // [2,2] identity
        let mut out = vec![0.0; 4];
        matmul(&mut out, &x, &w, 2, 2, 2);
        assert_eq!(out, x);
    }

    #[test]
    fn rmsnorm_unit() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0; 4];
        rmsnorm(&mut out, &x, &w, 1e-6);
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let expect: Vec<f32> = x.iter().map(|v| v / ss.sqrt()).collect();
        for (a, b) in out.iter().zip(expect) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let (cos, sin) = rope_tables(8, 8, 1e6);
        let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let before: f32 = x.iter().map(|v| v * v).sum();
        rope(&mut x, &cos, &sin, 3, 8);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-3);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![0.1f32, 0.5, -0.2, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
