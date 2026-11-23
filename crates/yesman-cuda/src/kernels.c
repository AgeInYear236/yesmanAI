// CUDA kernels for Qwen2.5 inference, compiled to PTX via Clang
// Target: nvptx64-nvidia-cuda sm_86 (Ampere / RTX 3060)

static inline float cuda_expf(float x) {
    float res;
    float x_log2e = x * 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x_log2e));
    return res;
}

static inline float cuda_rsqrtf(float x) {
    float res;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

static inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other;
        asm("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
            : "=f"(other) : "f"(val), "r"(offset));
        val += other;
    }
    return val;
}

// ----------------------------------------------------------------------------
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
// ----------------------------------------------------------------------------
__attribute__((noinline))
void rmsnorm_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int d,
    float eps
) {
    int row = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(row));
    int tid = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(tid));
    int bdim = 0;
    asm("mov.u32 %0, %%ntid.x;" : "=r"(bdim));

    float sum_sq = 0.0f;
    const float* x_row = x + row * d;
    for (int i = tid; i < d; i += bdim) {
        float val = x_row[i];
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    float scale = cuda_rsqrtf(sum_sq / (float)d + eps);
    asm("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=f"(scale) : "f"(scale));

    float* out_row = out + row * d;
    for (int i = tid; i < d; i += bdim) {
        out_row[i] = x_row[i] * scale * weight[i];
    }
}

// ----------------------------------------------------------------------------
// SwiGLU: act = (gate / (1 + exp(-gate))) * up
// ----------------------------------------------------------------------------
__attribute__((noinline))
void swiglu_kernel(
    float* __restrict__ act,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int tid = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(tid));
    int ctaid = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(ctaid));
    int bdim = 0;
    asm("mov.u32 %0, %%ntid.x;" : "=r"(bdim));

    int idx = ctaid * bdim + tid;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        float silu = g / (1.0f + cuda_expf(-g));
        act[idx] = silu * u;
    }
}

// ----------------------------------------------------------------------------
// Elementwise Residual Add: a[i] += b[i]
// ----------------------------------------------------------------------------
__attribute__((noinline))
void residual_add_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int tid = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(tid));
    int ctaid = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(ctaid));
    int bdim = 0;
    asm("mov.u32 %0, %%ntid.x;" : "=r"(bdim));

    int idx = ctaid * bdim + tid;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

// ----------------------------------------------------------------------------
// In-place RoPE (Rotary Position Embedding) for Q or K heads
// head_dim = 128 (half = 64)
// ----------------------------------------------------------------------------
__attribute__((noinline))
void rope_kernel(
    float* __restrict__ x,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int pos,
    int head_dim,
    int num_heads
) {
    int h = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(h)); // head index
    int lane = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(lane));

    if (h >= num_heads) return;

    int half = head_dim / 2;
    int base = h * head_dim;
    for (int tid = lane; tid < half; tid += 32) {
        float c = cos_tab[pos * head_dim + tid];
        float s = sin_tab[pos * head_dim + tid];
        float a = x[base + tid];
        float b = x[base + tid + half];
        x[base + tid] = a * c - b * s;
        x[base + tid + half] = b * c + a * s;
    }
}

// ----------------------------------------------------------------------------
// GEMV FP32: out[row] = sum(x[k] * w[row, k]) + (bias ? bias[row] : 0)
// One warp (32 threads) per output row
// ----------------------------------------------------------------------------
__attribute__((noinline))
void gemv_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    int k,
    int n
) {
    int row = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(row));
    int lane = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(lane));

    if (row >= n) return;

    float sum = 0.0f;
    const float* w_row = w + (long long)row * k;

    // 32 threads stripe across k
    for (int i = lane; i < k; i += 32) {
        sum += x[i] * w_row[i];
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        if (bias != 0) {
            sum += bias[row];
        }
        out[row] = sum;
    }
}

// ----------------------------------------------------------------------------
// GEMV Q8_0: out[row] = sum_{b=0}^{k/32-1} d_b * sum_{i=0}^{31} (x[b*32 + i] * q[i])
// Row size: (k / 32) * 36 bytes
// One warp (32 threads) per output row
// ----------------------------------------------------------------------------
__attribute__((noinline))
void gemv_q8_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const unsigned char* __restrict__ w_blocks,
    const float* __restrict__ bias,
    int k,
    int n
) {
    int row = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(row));
    int lane = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(lane));

    if (row >= n) return;

    int num_blocks = k / 32;
    const unsigned char* row_base = w_blocks + (long long)row * num_blocks * 36;

    float total = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        const unsigned char* block_ptr = row_base + b * 36;
        float d = *(const float*)block_ptr;
        signed char q = (signed char)block_ptr[4 + lane];
        float prod = x[b * 32 + lane] * (float)q;
        float sum_b = warp_reduce_sum(prod);
        if (lane == 0) {
            total += d * sum_b;
        }
    }

    if (lane == 0) {
        if (bias != 0) {
            total += bias[row];
        }
        out[row] = total;
    }
}

// ----------------------------------------------------------------------------
// GEMV Q4_K: out[row] = sum of 8 sub-blocks per super-block (256 weights)
// Super-block size: 152 bytes
// One warp (32 threads) per output row
// ----------------------------------------------------------------------------
__attribute__((noinline))
void gemv_q4k_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const unsigned char* __restrict__ w_blocks,
    const float* __restrict__ bias,
    int k,
    int n
) {
    int row = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(row));
    int lane = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(lane));

    if (row >= n) return;

    int num_super_blocks = k / 256;
    const unsigned char* row_base = w_blocks + (long long)row * num_super_blocks * 152;

    float total = 0.0f;

    for (int sb = 0; sb < num_super_blocks; sb++) {
        const unsigned char* sb_ptr = row_base + sb * 152;
        float d = *(const float*)(sb_ptr);
        float dmin = *(const float*)(sb_ptr + 4);
        const unsigned char* scales = sb_ptr + 8;
        const unsigned char* qs = sb_ptr + 24;

        const float* x_sb = x + sb * 256;

        for (int j = 0; j < 8; j++) {
            float sd = d * (float)scales[j];
            float md = dmin * ((float)scales[8 + j] - 127.0f);

            int byte_idx = (lane < 16) ? lane : (lane - 16);
            unsigned char raw_byte = qs[j * 16 + byte_idx];
            int nib = (lane < 16) ? (raw_byte & 0x0f) : (raw_byte >> 4);

            float x_val = x_sb[j * 32 + lane];
            float prod = x_val * (float)nib;

            float sum_prod = warp_reduce_sum(prod);
            float sum_x = warp_reduce_sum(x_val);

            if (lane == 0) {
                total += sd * sum_prod - md * sum_x;
            }
        }
    }

    if (lane == 0) {
        if (bias != 0) {
            total += bias[row];
        }
        out[row] = total;
    }
}

// ----------------------------------------------------------------------------
// Single-Token Decode Attention (GQA):
// 12 query heads, 2 KV heads (group = 6 query heads per KV head), head_dim = 128
// One warp per query head
// ----------------------------------------------------------------------------
__attribute__((noinline))
void decode_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ kv_k,
    const float* __restrict__ kv_v,
    int head_dim,
    int num_heads,
    int group,
    int kv_dim,
    int t_end,
    float scale
) {
    int h = 0;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(h)); // head index (0..11)
    int lane = 0;
    asm("mov.u32 %0, %%tid.x;" : "=r"(lane)); // lane index (0..31)

    if (h >= num_heads) return;

    int kv_h = h / group;
    const float* q_h = q + h * head_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    int num_dims = head_dim / 32;

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    for (int t = 0; t < t_end; t++) {
        const float* k_t = kv_k + (long long)t * kv_dim + kv_h * head_dim;

        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            dot += q_h[d] * k_t[d];
        }
        dot = warp_reduce_sum(dot) * scale;

        // Broadcast score to all lanes in the warp
        asm("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=f"(dot) : "f"(dot));

        // Online softmax update
        float p;
        if (dot > max_score) {
            float rescale = cuda_expf(max_score - dot);
            acc0 *= rescale;
            acc1 *= rescale;
            acc2 *= rescale;
            acc3 *= rescale;
            sum_exp = sum_exp * rescale + 1.0f;
            max_score = dot;
            p = 1.0f;
        } else {
            p = cuda_expf(dot - max_score);
            sum_exp += p;
        }

        const float* v_t = kv_v + (long long)t * kv_dim + kv_h * head_dim;
        if (num_dims >= 1) acc0 += p * v_t[lane];
        if (num_dims >= 2) acc1 += p * v_t[lane + 32];
        if (num_dims >= 3) acc2 += p * v_t[lane + 64];
        if (num_dims >= 4) acc3 += p * v_t[lane + 96];
    }

    float inv_sum = 1.0f / sum_exp;
    float* out_h = out + h * head_dim;
    if (num_dims >= 1) out_h[lane] = acc0 * inv_sum;
    if (num_dims >= 2) out_h[lane + 32] = acc1 * inv_sum;
    if (num_dims >= 3) out_h[lane + 64] = acc2 * inv_sum;
    if (num_dims >= 4) out_h[lane + 96] = acc3 * inv_sum;
}
