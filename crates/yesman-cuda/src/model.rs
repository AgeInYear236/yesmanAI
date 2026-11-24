use std::sync::Arc;

use anyhow::Result;
use yesman_model::{Config, GenerateOptions, Qwen2, Sampler, Tokenizer, chat};

use crate::{
    context::{CudaBuffer, CudaContext},
    kernels::CudaKernels,
};

pub const GPU_PREFILL_CHUNK: usize = 512;

pub struct CudaLayer {
    pub in_norm: CudaBuffer<f32>,
    pub post_norm: CudaBuffer<f32>,
    pub wq: CudaWeight,
    pub wk: CudaWeight,
    pub wv: CudaWeight,
    pub wo: CudaWeight,
    pub bq: CudaBuffer<f32>,
    pub bk: CudaBuffer<f32>,
    pub bv: CudaBuffer<f32>,
    pub w_gate: CudaWeight,
    pub w_up: CudaWeight,
    pub w_down: CudaWeight,
}

pub enum CudaWeight {
    F32 {
        buf: CudaBuffer<f32>,
        rows: usize,
        cols: usize,
    },
    Q8 {
        buf: CudaBuffer<u8>,
        rows: usize,
        cols: usize,
    },
    Q4K {
        buf: CudaBuffer<u8>,
        rows: usize,
        cols: usize,
    },
}

impl CudaWeight {
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } => *rows,
            Self::Q8 { rows, .. } => *rows,
            Self::Q4K { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } => *cols,
            Self::Q8 { cols, .. } => *cols,
            Self::Q4K { cols, .. } => *cols,
        }
    }

    pub fn gemv(
        &self,
        kernels: &CudaKernels,
        out: &CudaBuffer<f32>,
        x: &CudaBuffer<f32>,
        bias: Option<&CudaBuffer<f32>>,
    ) -> Result<()> {
        match self {
            Self::F32 { buf, rows, cols } => {
                kernels.gemv_f32(out, x, buf, bias, *cols, *rows)?;
            }
            Self::Q8 { buf, rows, cols } => {
                kernels.gemv_q8(out, x, buf, bias, *cols, *rows)?;
            }
            Self::Q4K { buf, rows, cols } => {
                kernels.gemv_q4k(out, x, buf, bias, *cols, *rows)?;
            }
        }
        Ok(())
    }

    pub fn gemm(
        &self,
        kernels: &CudaKernels,
        out: &CudaBuffer<f32>,
        x: &CudaBuffer<f32>,
        bias: Option<&CudaBuffer<f32>>,
        m: usize,
    ) -> Result<()> {
        match self {
            Self::F32 { buf, rows, cols } => {
                kernels.gemm_f32(out, x, buf, bias, *cols, *rows, m)?;
            }
            Self::Q8 { buf, rows, cols } => {
                kernels.gemm_q8(out, x, buf, bias, *cols, *rows, m)?;
            }
            Self::Q4K { buf, rows, cols } => {
                kernels.gemm_q4k(out, x, buf, bias, *cols, *rows, m)?;
            }
        }
        Ok(())
    }
}

/// Immutable GPU model weights, shared across sessions via `Arc<CudaQwen2Weights>`.
pub struct CudaQwen2Weights {
    pub cfg: Config,
    pub ctx: Arc<CudaContext>,
    pub kernels: CudaKernels,
    pub layers: Vec<CudaLayer>,
    pub embed: CudaWeight,
    pub final_norm: CudaBuffer<f32>,
    pub cos_tab: CudaBuffer<f32>,
    pub sin_tab: CudaBuffer<f32>,
}

unsafe impl Send for CudaQwen2Weights {}
unsafe impl Sync for CudaQwen2Weights {}

impl CudaQwen2Weights {
    pub fn from_cpu(model: &Qwen2, ctx: Arc<CudaContext>) -> Result<Arc<Self>> {
        ctx.set_current()?;
        let cfg = model.cfg.clone();
        let kernels = CudaKernels::new(Arc::clone(&ctx))?;
        let final_norm = ctx.from_slice(&model.weights.final_norm)?;
        let cos_tab = ctx.from_slice(&model.weights.cos)?;
        let sin_tab = ctx.from_slice(&model.weights.sin)?;

        let upload_weight = |w: &yesman_model::qwen2::Weight| -> Result<CudaWeight> {
            match w {
                yesman_model::qwen2::Weight::F32(t) => Ok(CudaWeight::F32 {
                    buf: ctx.from_slice(&t.data)?,
                    rows: t.rows(),
                    cols: t.cols(),
                }),
                yesman_model::qwen2::Weight::BF16(t) => {
                    let f32_data: Vec<f32> =
                        t.data.iter().map(|v: &half::bf16| v.to_f32()).collect();
                    Ok(CudaWeight::F32 {
                        buf: ctx.from_slice(&f32_data)?,
                        rows: t.rows,
                        cols: t.cols,
                    })
                }
                yesman_model::qwen2::Weight::Q8(t) => Ok(CudaWeight::Q8 {
                    buf: ctx.from_slice(t.as_bytes())?,
                    rows: t.rows,
                    cols: t.cols,
                }),
                yesman_model::qwen2::Weight::Q4(_) => {
                    anyhow::bail!("GPU backend supports Q8 and Q4_K weights");
                }
                yesman_model::qwen2::Weight::Q4K(t) => Ok(CudaWeight::Q4K {
                    buf: ctx.from_slice(t.as_bytes())?,
                    rows: t.rows,
                    cols: t.cols,
                }),
            }
        };

        let embed = upload_weight(&model.weights.embed)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in &model.weights.layers {
            layers.push(CudaLayer {
                in_norm: ctx.from_slice(&l.in_norm)?,
                post_norm: ctx.from_slice(&l.post_norm)?,
                wq: upload_weight(&l.wq)?,
                wk: upload_weight(&l.wk)?,
                wv: upload_weight(&l.wv)?,
                wo: upload_weight(&l.wo)?,
                bq: ctx.from_slice(&l.bq)?,
                bk: ctx.from_slice(&l.bk)?,
                bv: ctx.from_slice(&l.bv)?,
                w_gate: upload_weight(&l.w_gate)?,
                w_up: upload_weight(&l.w_up)?,
                w_down: upload_weight(&l.w_down)?,
            });
        }

        Ok(Arc::new(Self {
            cfg,
            ctx,
            kernels,
            layers,
            embed,
            final_norm,
            cos_tab,
            sin_tab,
        }))
    }

    pub fn create_session(self: &Arc<Self>, max_ctx: usize) -> Result<CudaQwen2> {
        self.ctx.set_current()?;
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let kv_dim = cfg.kv_dim();
        let vocab = cfg.vocab_size;

        let mut kv_k = Vec::with_capacity(cfg.num_hidden_layers);
        let mut kv_v = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            kv_k.push(self.ctx.alloc::<f32>(max_ctx * kv_dim)?);
            kv_v.push(self.ctx.alloc::<f32>(max_ctx * kv_dim)?);
        }

        // Single-token decode scratchpad
        let x = self.ctx.alloc::<f32>(h)?;
        let hb = self.ctx.alloc::<f32>(h)?;
        let q = self.ctx.alloc::<f32>(h)?;
        let k = self.ctx.alloc::<f32>(kv_dim)?;
        let v = self.ctx.alloc::<f32>(kv_dim)?;
        let attn_out = self.ctx.alloc::<f32>(h)?;
        let o = self.ctx.alloc::<f32>(h)?;
        let gate = self.ctx.alloc::<f32>(inter)?;
        let up = self.ctx.alloc::<f32>(inter)?;
        let act = self.ctx.alloc::<f32>(inter)?;
        let down = self.ctx.alloc::<f32>(h)?;
        let logits = self.ctx.alloc::<f32>(vocab)?;

        // Batched prefill scratchpad (up to GPU_PREFILL_CHUNK tokens in parallel)
        let pm = GPU_PREFILL_CHUNK;
        let pre_x = self.ctx.alloc::<f32>(pm * h)?;
        let pre_hb = self.ctx.alloc::<f32>(pm * h)?;
        let pre_q = self.ctx.alloc::<f32>(pm * h)?;
        let pre_k = self.ctx.alloc::<f32>(pm * kv_dim)?;
        let pre_v = self.ctx.alloc::<f32>(pm * kv_dim)?;
        let pre_attn = self.ctx.alloc::<f32>(pm * h)?;
        let pre_o = self.ctx.alloc::<f32>(pm * h)?;
        let pre_gate = self.ctx.alloc::<f32>(pm * inter)?;
        let pre_up = self.ctx.alloc::<f32>(pm * inter)?;
        let pre_act = self.ctx.alloc::<f32>(pm * inter)?;
        let pre_down = self.ctx.alloc::<f32>(pm * h)?;

        Ok(CudaQwen2 {
            weights: Arc::clone(self),
            kv_k,
            kv_v,
            max_ctx,
            kv_len: 0,
            x,
            hb,
            q,
            k,
            v,
            attn_out,
            o,
            gate,
            up,
            act,
            down,
            logits,
            pre_x,
            pre_hb,
            pre_q,
            pre_k,
            pre_v,
            pre_attn,
            pre_o,
            pre_gate,
            pre_up,
            pre_act,
            pre_down,
        })
    }
}

/// A lightweight GPU session owning its own KV cache and scratchpad,
/// while sharing model weights in `Arc<CudaQwen2Weights>`.
pub struct CudaQwen2 {
    pub weights: Arc<CudaQwen2Weights>,

    // Per-layer GPU KV cache: [pos, kv_dim]
    pub kv_k: Vec<CudaBuffer<f32>>,
    pub kv_v: Vec<CudaBuffer<f32>>,
    pub max_ctx: usize,
    pub kv_len: usize,

    // Reused single-token GPU scratchpad buffers
    pub x: CudaBuffer<f32>,
    pub hb: CudaBuffer<f32>,
    pub q: CudaBuffer<f32>,
    pub k: CudaBuffer<f32>,
    pub v: CudaBuffer<f32>,
    pub attn_out: CudaBuffer<f32>,
    pub o: CudaBuffer<f32>,
    pub gate: CudaBuffer<f32>,
    pub up: CudaBuffer<f32>,
    pub act: CudaBuffer<f32>,
    pub down: CudaBuffer<f32>,
    pub logits: CudaBuffer<f32>,

    // Reused batched prefill GPU scratchpad buffers
    pub pre_x: CudaBuffer<f32>,
    pub pre_hb: CudaBuffer<f32>,
    pub pre_q: CudaBuffer<f32>,
    pub pre_k: CudaBuffer<f32>,
    pub pre_v: CudaBuffer<f32>,
    pub pre_attn: CudaBuffer<f32>,
    pub pre_o: CudaBuffer<f32>,
    pub pre_gate: CudaBuffer<f32>,
    pub pre_up: CudaBuffer<f32>,
    pub pre_act: CudaBuffer<f32>,
    pub pre_down: CudaBuffer<f32>,
}

unsafe impl Send for CudaQwen2 {}
unsafe impl Sync for CudaQwen2 {}

impl CudaQwen2 {
    pub fn from_cpu(model: &Qwen2, ctx: Arc<CudaContext>, max_ctx: usize) -> Result<Self> {
        let weights = CudaQwen2Weights::from_cpu(model, ctx)?;
        weights.create_session(max_ctx)
    }

    pub fn clone_session(&self, max_ctx: usize) -> Result<Self> {
        self.weights.create_session(max_ctx)
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    pub fn cfg(&self) -> &Config {
        &self.weights.cfg
    }

    /// Single token decode step on the GPU:
    /// Runs embedding lookup, 28 transformer layers, and tied lm_head, returning logits.
    pub fn decode_step(&mut self, token: u32, pos: usize, cpu_embed: &Qwen2) -> Result<Vec<f32>> {
        self.weights.ctx.set_current()?;
        let cfg = &self.weights.cfg;
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let kv_dim = cfg.kv_dim();
        let group = cfg.gqa_group();
        let eps = cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // 1. Embedding lookup: copy token's embedding row into GPU x buffer
        let embed_row = cpu_embed.gather_embed(token as usize);
        self.x.copy_from_host(&embed_row)?;

        // 2. Transformer layers
        for (li, layer) in self.weights.layers.iter().enumerate() {
            // Attention pre-norm
            self.weights
                .kernels
                .rmsnorm(&self.hb, &self.x, &layer.in_norm, 1, h, eps)?;

            // Q, K, V projections with bias
            layer
                .wq
                .gemv(&self.weights.kernels, &self.q, &self.hb, Some(&layer.bq))?;
            layer
                .wk
                .gemv(&self.weights.kernels, &self.k, &self.hb, Some(&layer.bk))?;
            layer
                .wv
                .gemv(&self.weights.kernels, &self.v, &self.hb, Some(&layer.bv))?;

            // In-place RoPE
            self.weights.kernels.rope(
                &self.q,
                &self.weights.cos_tab,
                &self.weights.sin_tab,
                pos,
                hd,
                n_heads,
            )?;
            self.weights.kernels.rope(
                &self.k,
                &self.weights.cos_tab,
                &self.weights.sin_tab,
                pos,
                hd,
                cfg.num_key_value_heads,
            )?;

            // Store K and V into GPU KV cache at position `pos`
            let k_offset = (pos * kv_dim * std::mem::size_of::<f32>()) as u64;
            let v_offset = (pos * kv_dim * std::mem::size_of::<f32>()) as u64;
            let byte_count = kv_dim * std::mem::size_of::<f32>();
            unsafe {
                (self.weights.ctx.driver.cuMemcpyDtoD)(
                    self.kv_k[li].ptr + k_offset,
                    self.k.ptr,
                    byte_count,
                );
                (self.weights.ctx.driver.cuMemcpyDtoD)(
                    self.kv_v[li].ptr + v_offset,
                    self.v.ptr,
                    byte_count,
                );
            }

            // GQA Causal Attention on GPU
            self.weights.kernels.decode_attention(
                &self.attn_out,
                &self.q,
                &self.kv_k[li],
                &self.kv_v[li],
                hd,
                n_heads,
                group,
                kv_dim,
                pos + 1,
                scale,
            )?;

            // Output projection
            layer
                .wo
                .gemv(&self.weights.kernels, &self.o, &self.attn_out, None)?;
            self.weights.kernels.residual_add(&self.x, &self.o, h)?;

            // FFN post-norm
            self.weights
                .kernels
                .rmsnorm(&self.hb, &self.x, &layer.post_norm, 1, h, eps)?;

            // MLP gate and up projections
            layer
                .w_gate
                .gemv(&self.weights.kernels, &self.gate, &self.hb, None)?;
            layer
                .w_up
                .gemv(&self.weights.kernels, &self.up, &self.hb, None)?;

            // SwiGLU activation
            self.weights
                .kernels
                .swiglu(&self.act, &self.gate, &self.up, cfg.intermediate_size)?;

            // MLP down projection
            layer
                .w_down
                .gemv(&self.weights.kernels, &self.down, &self.act, None)?;
            self.weights.kernels.residual_add(&self.x, &self.down, h)?;
        }

        self.kv_len = pos + 1;

        // 3. Final layer norm
        self.weights
            .kernels
            .rmsnorm(&self.hb, &self.x, &self.weights.final_norm, 1, h, eps)?;

        // 4. LM Head projection on GPU
        self.weights
            .embed
            .gemv(&self.weights.kernels, &self.logits, &self.hb, None)?;

        self.weights.ctx.synchronize()?;
        self.logits.to_vec()
    }

    /// Batched prefill on GPU: runs all `tokens` in chunks up to `GPU_PREFILL_CHUNK`
    /// using parallel GEMM kernels and batched causal attention in a single pass.
    pub fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        cpu_embed: &Qwen2,
    ) -> Result<Vec<f32>> {
        let m = tokens.len();
        assert!(m > 0 && m <= GPU_PREFILL_CHUNK);
        self.weights.ctx.set_current()?;
        let cfg = &self.weights.cfg;
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let kv_dim = cfg.kv_dim();
        let group = cfg.gqa_group();
        let eps = cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // 1. Gather all m embedding rows on host and copy to GPU in ONE transfer
        let mut host_embed = Vec::with_capacity(m * h);
        for &t in tokens {
            let row = cpu_embed.gather_embed(t as usize);
            host_embed.extend_from_slice(&row);
        }
        self.pre_x.copy_from_host(&host_embed)?;

        // 2. Transformer layers in parallel for all m tokens
        for (li, layer) in self.weights.layers.iter().enumerate() {
            // Attention pre-norm (all m rows in parallel)
            self.weights
                .kernels
                .rmsnorm(&self.pre_hb, &self.pre_x, &layer.in_norm, m, h, eps)?;

            // Q, K, V GEMMs (shape [m, h] -> [m, h] and [m, kv_dim])
            layer.wq.gemm(
                &self.weights.kernels,
                &self.pre_q,
                &self.pre_hb,
                Some(&layer.bq),
                m,
            )?;
            layer.wk.gemm(
                &self.weights.kernels,
                &self.pre_k,
                &self.pre_hb,
                Some(&layer.bk),
                m,
            )?;
            layer.wv.gemm(
                &self.weights.kernels,
                &self.pre_v,
                &self.pre_hb,
                Some(&layer.bv),
                m,
            )?;

            // Batched RoPE across all m tokens
            self.weights.kernels.rope_batched(
                &self.pre_q,
                &self.weights.cos_tab,
                &self.weights.sin_tab,
                pos0,
                hd,
                n_heads,
                m,
            )?;
            self.weights.kernels.rope_batched(
                &self.pre_k,
                &self.weights.cos_tab,
                &self.weights.sin_tab,
                pos0,
                hd,
                cfg.num_key_value_heads,
                m,
            )?;

            // Write all m keys and values into KV cache starting at offset pos0
            let k_offset = (pos0 * kv_dim * std::mem::size_of::<f32>()) as u64;
            let v_offset = (pos0 * kv_dim * std::mem::size_of::<f32>()) as u64;
            let byte_count = m * kv_dim * std::mem::size_of::<f32>();
            unsafe {
                (self.weights.ctx.driver.cuMemcpyDtoD)(
                    self.kv_k[li].ptr + k_offset,
                    self.pre_k.ptr,
                    byte_count,
                );
                (self.weights.ctx.driver.cuMemcpyDtoD)(
                    self.kv_v[li].ptr + v_offset,
                    self.pre_v.ptr,
                    byte_count,
                );
            }

            // Batched Causal Attention on GPU
            self.weights.kernels.causal_attention_batched(
                &self.pre_attn,
                &self.pre_q,
                &self.kv_k[li],
                &self.kv_v[li],
                hd,
                n_heads,
                group,
                kv_dim,
                pos0,
                m,
                scale,
            )?;

            // Output projection GEMM
            layer
                .wo
                .gemm(&self.weights.kernels, &self.pre_o, &self.pre_attn, None, m)?;
            self.weights
                .kernels
                .residual_add(&self.pre_x, &self.pre_o, m * h)?;

            // FFN pre-norm
            self.weights
                .kernels
                .rmsnorm(&self.pre_hb, &self.pre_x, &layer.post_norm, m, h, eps)?;

            // MLP gate and up GEMMs
            layer
                .w_gate
                .gemm(&self.weights.kernels, &self.pre_gate, &self.pre_hb, None, m)?;
            layer
                .w_up
                .gemm(&self.weights.kernels, &self.pre_up, &self.pre_hb, None, m)?;

            // SwiGLU activation
            self.weights.kernels.swiglu(
                &self.pre_act,
                &self.pre_gate,
                &self.pre_up,
                m * cfg.intermediate_size,
            )?;

            // MLP down GEMM
            layer.w_down.gemm(
                &self.weights.kernels,
                &self.pre_down,
                &self.pre_act,
                None,
                m,
            )?;
            self.weights
                .kernels
                .residual_add(&self.pre_x, &self.pre_down, m * h)?;
        }

        self.kv_len = pos0 + m;

        // Extract logits for the last token (token m - 1)
        let last_offset = ((m - 1) * h * std::mem::size_of::<f32>()) as u64;
        unsafe {
            (self.weights.ctx.driver.cuMemcpyDtoD)(
                self.x.ptr,
                self.pre_x.ptr + last_offset,
                h * std::mem::size_of::<f32>(),
            );
        }

        // Final norm & LM head for the last token
        self.weights
            .kernels
            .rmsnorm(&self.hb, &self.x, &self.weights.final_norm, 1, h, eps)?;
        self.weights
            .embed
            .gemv(&self.weights.kernels, &self.logits, &self.hb, None)?;

        self.weights.ctx.synchronize()?;
        self.logits.to_vec()
    }

    pub fn generate(
        &mut self,
        tok: &Tokenizer,
        cpu_model: &Qwen2,
        prompt_ids: &[u32],
        opts: &GenerateOptions,
        sampler: &mut Sampler,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<(Vec<u32>, String)> {
        if prompt_ids.is_empty() {
            anyhow::bail!("empty prompt");
        }
        let prompt_slice = if prompt_ids.len() >= opts.max_ctx {
            let budget = opts.max_ctx.saturating_sub(16).max(1);
            let start = prompt_ids.len().saturating_sub(budget);
            &prompt_ids[start..]
        } else {
            prompt_ids
        };

        self.reset();
        let mut history = prompt_slice.to_vec();
        let mut logits = Vec::new();

        // Batched prefill on GPU (chunks of up to GPU_PREFILL_CHUNK tokens in parallel)
        let mut pos = 0usize;
        for chunk in prompt_slice.chunks(GPU_PREFILL_CHUNK) {
            logits = self.prefill_chunk(chunk, pos, cpu_model)?;
            pos += chunk.len();
        }

        let mut out_ids: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut pos = prompt_slice.len();

        for _ in 0..opts.max_new_tokens {
            let next = sampler.sample(&mut logits, &history);
            if opts.stop_ids.contains(&next) {
                break;
            }
            let piece = tok.decode(&[next])?;
            out_ids.push(next);
            text.push_str(&piece);
            on_token(&piece)?;
            history.push(next);
            pos += 1;
            if pos >= opts.max_ctx {
                break;
            }
            logits = self.decode_step(next, pos - 1, cpu_model)?;
        }

        Ok((out_ids, text))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_once(
        &mut self,
        tok: &Tokenizer,
        cpu_model: &Qwen2,
        cfg: &Config,
        messages: &[chat::Message],
        opts: &GenerateOptions,
        sampler: &mut Sampler,
        on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let prompt = chat::render(messages);
        let ids = tok.encode(&prompt)?;
        let mut stop = cfg.eos_ids();
        if let Some(id) = tok.id_of(chat::IM_END)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        if let Some(id) = tok.id_of("<|endoftext|>")
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        let opts = GenerateOptions {
            stop_ids: stop,
            ..opts.clone()
        };
        let (_, text) = self.generate(tok, cpu_model, &ids, &opts, sampler, on_token)?;
        Ok(text)
    }
}
