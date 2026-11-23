//! Qwen2ForCausalLM forward pass, written from scratch on top of `yesman-core`.
//!
//! Layout follows the released checkpoint exactly: weights are `[out, in]`
//! row-major, attention has q/k/v biases, RoPE is the HF "rotate half" variant
//! with theta = 1e6, and `lm_head` is tied to `embed_tokens`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use yesman_core::{
    Bf16Tensor, Q4KTensor, Q4Tensor, Q8Tensor, SafeFile, Tensor, bf16, cache, ops, q4, q4k,
    quant,
};

use crate::{config::Config, kv::KvCache};

/// How weights are held in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DType {
    /// bf16 as shipped in the checkpoint: exact, half the traffic of f32.
    #[default]
    Bf16,
    /// Widened to f32 (used by the numpy cross-check tests).
    F32,
    /// Q8_0 block quantization: ~1.6 GB.
    Q8,
    /// Q4_0 block quantization: ~0.94 GB, fastest on CPU.
    Q4,
    /// Q4_K super-block quantization: ~0.85 GB (Q4_K_S-like, all 4-bit).
    Q4K,
    /// Q4_K_M-like: Q4_K attention + Q8_0 FFN (quality close to Q8_0).
    Q4KM,
}

impl DType {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => DType::F32,
            "bf16" | "bfloat16" => DType::Bf16,
            "q8" | "q8_0" | "int8" => DType::Q8,
            "q4" | "q4_0" => DType::Q4,
            "q4k" | "q4_k" | "q4ks" | "q4_k_s" => DType::Q4K,
            "q4km" | "q4_k_m" => DType::Q4KM,
            other => return Err(anyhow!("unknown dtype `{other}` (f32|bf16|q8|q4|q4k|q4km)")),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Bf16 => "bf16",
            DType::Q8 => "Q8_0",
            DType::Q4 => "Q4_0",
            DType::Q4K => "Q4_K",
            DType::Q4KM => "Q4_K_M",
        }
    }
}

/// A weight matrix: f32, bf16, Q8_0 or Q4_0. Also the (tied) embedding.
enum Weight {
    F32(Tensor),
    BF16(Bf16Tensor),
    Q8(Q8Tensor),
    Q4(Q4Tensor),
    Q4K(Q4KTensor),
}

impl Weight {
    fn load(file: &SafeFile, name: &str, dt: DType) -> Result<Self> {
        Ok(match dt {
            DType::F32 => Weight::F32(file.get(name)?),
            DType::Bf16 => Weight::BF16(file.get_bf16(name)?),
            DType::Q8 => Weight::Q8(Q8Tensor::from_bf16(&file.get_bf16(name)?)),
            DType::Q4 => Weight::Q4(Q4Tensor::from_bf16(&file.get_bf16(name)?)),
            DType::Q4K | DType::Q4KM => {
                Weight::Q4K(Q4KTensor::from_bf16(&file.get_bf16(name)?))
            }
        })
    }

    /// `out[m,n] = x[m,k] @ self[n,k]^T`
    fn matmul(&self, out: &mut [f32], x: &[f32], m: usize, k: usize, n: usize) {
        match self {
            Weight::F32(t) => ops::matmul(out, x, &t.data, m, k, n),
            Weight::BF16(t) => bf16::matmul(out, x, t, m, k, n),
            Weight::Q8(q) => {
                let xq = quant::quantize_rows(x, k);
                quant::matmul_q8(out, &xq, q, m, k, n);
            }
            Weight::Q4(q) => {
                let xq = quant::quantize_rows(x, k);
                q4::matmul_q8q4(out, &xq, q, m, k, n);
            }
            Weight::Q4K(q) => {
                let xq = quant::quantize_rows(x, k);
                q4k::matmul_q8q4k(out, &xq, q, m, k, n);
            }
        }
    }

    fn gather(&self, row: usize) -> Vec<f32> {
        match self {
            Weight::F32(t) => t.row(row).to_vec(),
            Weight::BF16(t) => t.dequant_row(row),
            Weight::Q8(t) => t.dequantize_row(row),
            Weight::Q4(t) => t.dequantize_row(row),
            Weight::Q4K(t) => t.dequantize_row(row),
        }
    }

    fn from_cache(c: &cache::QuantCache, name: &str) -> Result<Self> {
        Ok(match c.entry(name)?.kind {
            cache::KIND_Q8 => Weight::Q8(c.q8(name)?),
            cache::KIND_Q4 => Weight::Q4(c.q4(name)?),
            cache::KIND_Q4K => Weight::Q4K(c.q4k(name)?),
            cache::KIND_F32 => {
                let e = c.entry(name)?;
                let data = c.f32_vec(name)?;
                Weight::F32(Tensor::new(data, vec![e.rows, e.cols])?)
            }
            other => return Err(anyhow!("tensor `{name}` has unknown kind {other}")),
        })
    }

    /// Borrow the raw quantized blocks for streaming into the cache.
    fn cache_bytes(&self) -> Option<(u8, usize, usize, &[u8])> {
        match self {
            Weight::Q8(t) => Some((cache::KIND_Q8, t.rows, t.cols, t.as_bytes())),
            Weight::Q4(t) => Some((cache::KIND_Q4, t.rows, t.cols, t.as_bytes())),
            Weight::Q4K(t) => Some((cache::KIND_Q4K, t.rows, t.cols, t.as_bytes())),
            _ => None,
        }
    }
}

fn f32_bytes(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * 4);
    for v in x {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    w_gate: Weight,
    w_up: Weight,
    w_down: Weight,
}

/// Scratch buffers reused across layers and across decode steps. Without this
/// a single token allocated ~11 vectors per layer (~300 malloc + 3 MB of
/// zeroing per token); now everything is resized once and sliced.
struct Workspace {
    x: Vec<f32>,
    hb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    o: Vec<f32>,
    hb2: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
    xf: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

impl Workspace {
    fn new(cfg: &Config) -> Self {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let kv_dim = cfg.kv_dim();
        Self {
            x: vec![0.0; h],
            hb: vec![0.0; h],
            q: vec![0.0; h],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn: vec![0.0; h],
            o: vec![0.0; h],
            hb2: vec![0.0; h],
            gate: vec![0.0; inter],
            up: vec![0.0; inter],
            act: vec![0.0; inter],
            down: vec![0.0; h],
            xf: vec![0.0; h],
            scores: vec![0.0; cfg.num_attention_heads],
            logits: vec![0.0; cfg.vocab_size],
        }
    }

    /// Grow (never shrink) all buffers for `m` tokens. Callers must slice to
    /// the exact length -- leftover elements from a larger prefill are stale.
    fn ensure(&mut self, m: usize, h: usize, inter: usize, kv_dim: usize) {
        let n_h = m * h;
        let n_i = m * inter;
        let n_kv = m * kv_dim;
        self.x.resize(n_h, 0.0);
        self.hb.resize(n_h, 0.0);
        self.q.resize(n_h, 0.0);
        self.k.resize(n_kv, 0.0);
        self.v.resize(n_kv, 0.0);
        self.attn.resize(n_h, 0.0);
        self.o.resize(n_h, 0.0);
        self.hb2.resize(n_h, 0.0);
        self.gate.resize(n_i, 0.0);
        self.up.resize(n_i, 0.0);
        self.act.resize(n_i, 0.0);
        self.down.resize(n_h, 0.0);
    }
}

pub struct Qwen2 {
    pub cfg: Config,
    embed: Weight,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    kv: KvCache,
    ws: Workspace,
}

impl Qwen2 {
    pub fn from_dir(dir: &Path, max_ctx: usize, dt: DType) -> Result<Self> {
        let cfg = Config::from_dir(dir)?;
        let weights = dir.join("model.safetensors");
        let file =
            SafeFile::open(&weights).with_context(|| format!("reading {}", weights.display()))?;

        let shape = file.shape("model.embed_tokens.weight")?;
        match shape.as_slice() {
            [v, h] if *v == cfg.vocab_size && *h == cfg.hidden_size => {}
            other => {
                return Err(anyhow!(
                    "embed shape {other:?} contradicts config {}x{}",
                    cfg.vocab_size,
                    cfg.hidden_size
                ))
            }
        }
        // Q4 noise on the 152k-row tied embedding/lm_head dominates the logit
        // error (llama.cpp keeps the output tensor in a higher-precision format
        // too). Keep it at Q8 unless the caller explicitly asked for Q4.
        let embed_dt = match dt {
            DType::Q4 | DType::Q4K | DType::Q4KM => DType::Q8,
            other => other,
        };
        let embed = Weight::load(&file, "model.embed_tokens.weight", embed_dt)?;
        let final_norm = file.get("model.norm.weight")?.data;

        // Q4_K_M keeps the large feed-forward matrices at 8 bits: they are
        // ~77% of the weights and dominate the quantization error.
        let mlp_dt = match dt {
            DType::Q4KM => DType::Q8,
            other => other,
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                in_norm: file.get(&format!("{p}.input_layernorm.weight"))?.data,
                post_norm: file.get(&format!("{p}.post_attention_layernorm.weight"))?.data,
                wq: Weight::load(&file, &format!("{p}.self_attn.q_proj.weight"), dt)?,
                wk: Weight::load(&file, &format!("{p}.self_attn.k_proj.weight"), dt)?,
                wv: Weight::load(&file, &format!("{p}.self_attn.v_proj.weight"), dt)?,
                wo: Weight::load(&file, &format!("{p}.self_attn.o_proj.weight"), dt)?,
                bq: file.get(&format!("{p}.self_attn.q_proj.bias"))?.data,
                bk: file.get(&format!("{p}.self_attn.k_proj.bias"))?.data,
                bv: file.get(&format!("{p}.self_attn.v_proj.bias"))?.data,
                w_gate: Weight::load(&file, &format!("{p}.mlp.gate_proj.weight"), mlp_dt)?,
                w_up: Weight::load(&file, &format!("{p}.mlp.up_proj.weight"), mlp_dt)?,
                w_down: Weight::load(&file, &format!("{p}.mlp.down_proj.weight"), mlp_dt)?,
            });
        }

        let (cos, sin) = ops::rope_tables(max_ctx, cfg.head_dim(), cfg.rope_theta);
        let kv = KvCache::new(cfg.num_hidden_layers, max_ctx, cfg.kv_dim());

        let ws = Workspace::new(&cfg);
        Ok(Self { cfg, embed, final_norm, layers, cos, sin, kv, ws })
    }

    /// Load from a pre-quantized cache produced by [`Qwen2::save_cache`].
    /// `dir` still supplies `config.json`; the cache supplies every tensor.
    pub fn from_cache(dir: &Path, max_ctx: usize, cache_path: &Path) -> Result<Self> {
        let cfg = Config::from_dir(dir)?;
        let c = cache::QuantCache::open(cache_path)?;

        let shape = cache_shape(&c, "model.embed_tokens.weight")?;
        match shape.as_slice() {
            [v, h] if *v == cfg.vocab_size && *h == cfg.hidden_size => {}
            other => {
                return Err(anyhow!(
                    "cached embed shape {other:?} contradicts config {}x{}",
                    cfg.vocab_size,
                    cfg.hidden_size
                ))
            }
        }
        let embed = Weight::from_cache(&c, "model.embed_tokens.weight")?;
        let final_norm = c.f32_vec("model.norm.weight")?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                in_norm: c.f32_vec(&format!("{p}.input_layernorm.weight"))?,
                post_norm: c.f32_vec(&format!("{p}.post_attention_layernorm.weight"))?,
                wq: Weight::from_cache(&c, &format!("{p}.self_attn.q_proj.weight"))?,
                wk: Weight::from_cache(&c, &format!("{p}.self_attn.k_proj.weight"))?,
                wv: Weight::from_cache(&c, &format!("{p}.self_attn.v_proj.weight"))?,
                wo: Weight::from_cache(&c, &format!("{p}.self_attn.o_proj.weight"))?,
                bq: c.f32_vec(&format!("{p}.self_attn.q_proj.bias"))?,
                bk: c.f32_vec(&format!("{p}.self_attn.k_proj.bias"))?,
                bv: c.f32_vec(&format!("{p}.self_attn.v_proj.bias"))?,
                w_gate: Weight::from_cache(&c, &format!("{p}.mlp.gate_proj.weight"))?,
                w_up: Weight::from_cache(&c, &format!("{p}.mlp.up_proj.weight"))?,
                w_down: Weight::from_cache(&c, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }

        let (cos, sin) = ops::rope_tables(max_ctx, cfg.head_dim(), cfg.rope_theta);
        let kv = KvCache::new(cfg.num_hidden_layers, max_ctx, cfg.kv_dim());
        let ws = Workspace::new(&cfg);
        Ok(Self { cfg, embed, final_norm, layers, cos, sin, kv, ws })
    }

    /// Write every tensor (weights + norms + biases) to a reusable cache.
    ///
    /// Only quantized weights can be cached; call with `--dtype q8|q4`.
    /// Weight blocks are streamed straight from the model (no 1.7 GB copy).
    pub fn save_cache(&self, path: &Path) -> Result<()> {
        struct Spec<'a> {
            name: String,
            kind: u8,
            rows: usize,
            cols: usize,
            small: Option<usize>,
            weight: Option<&'a Weight>,
        }

        fn push_small<'a>(
            specs: &mut Vec<Spec<'a>>,
            small: &mut Vec<Vec<u8>>,
            name: String,
            data: &[f32],
        ) {
            let idx = small.len();
            small.push(f32_bytes(data));
            specs.push(Spec {
                name,
                kind: cache::KIND_F32,
                rows: data.len(),
                cols: 1,
                small: Some(idx),
                weight: None,
            });
        }

        fn push_weight<'a>(
            specs: &mut Vec<Spec<'a>>,
            name: String,
            w: &'a Weight,
        ) -> Result<()> {
            let (kind, rows, cols, _) = w
                .cache_bytes()
                .ok_or_else(|| anyhow!("save_cache requires quantized weights (--dtype q8|q4|q4k)"))?;
            specs.push(Spec { name, kind, rows, cols, small: None, weight: Some(w) });
            Ok(())
        }

        let mut small: Vec<Vec<u8>> = Vec::new();
        let mut specs: Vec<Spec<'_>> = Vec::new();

        push_weight(&mut specs, "model.embed_tokens.weight".into(), &self.embed)?;
        push_small(&mut specs, &mut small, "model.norm.weight".into(), &self.final_norm);

        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            for (name, data) in [
                ("input_layernorm.weight", &l.in_norm),
                ("post_attention_layernorm.weight", &l.post_norm),
                ("self_attn.q_proj.bias", &l.bq),
                ("self_attn.k_proj.bias", &l.bk),
                ("self_attn.v_proj.bias", &l.bv),
            ] {
                push_small(&mut specs, &mut small, format!("{p}.{name}"), data);
            }
            for (suffix, w) in [
                ("self_attn.q_proj.weight", &l.wq),
                ("self_attn.k_proj.weight", &l.wk),
                ("self_attn.v_proj.weight", &l.wv),
                ("self_attn.o_proj.weight", &l.wo),
                ("mlp.gate_proj.weight", &l.w_gate),
                ("mlp.up_proj.weight", &l.w_up),
                ("mlp.down_proj.weight", &l.w_down),
            ] {
                push_weight(&mut specs, format!("{p}.{suffix}"), w)?;
            }
        }

        let entries: Vec<cache::CacheEntry<'_>> = specs
            .iter()
            .map(|s| {
                let data: &[u8] = match (s.small, s.weight) {
                    (Some(i), _) => small[i].as_slice(),
                    (None, Some(w)) => w.cache_bytes().unwrap().3,
                    _ => unreachable!(),
                };
                cache::CacheEntry {
                    name: s.name.as_str(),
                    kind: s.kind,
                    rows: s.rows,
                    cols: s.cols,
                    data,
                }
            })
            .collect();
        cache::write_cache(path, &entries)
    }

    pub fn kv_len(&self) -> usize {
        self.kv.len()
    }

    pub fn reset(&mut self) {
        self.kv.clear();
    }

    /// Run `tokens` starting at absolute position `pos0` and return the logits
    /// for the last token. K/V for these tokens is appended to the cache.
    pub fn forward(&mut self, tokens: &[u32], pos0: usize) -> Vec<f32> {
        let m = tokens.len();
        assert!(m > 0);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let hd = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let kv_dim = cfg.kv_dim();
        let group = cfg.gqa_group();
        let eps = cfg.rms_norm_eps;
        let scale = 1.0 / (hd as f32).sqrt();
        let vocab = cfg.vocab_size;
        let profiling = std::env::var_os("YESMAN_PROFILE").is_some();
        let mut prof = Profile::default();
        let now = || std::time::Instant::now();
        let mut mark = now();
        macro_rules! lap {
            ($dst:ident) => {
                if profiling {
                    prof.$dst += mark.elapsed().as_secs_f64();
                    mark = now();
                }
            };
        }

        // Split the borrows: the scratch space is mutable, everything else is
        // read-only during a forward pass (except the KV cache, reborrowed
        // below for each layer's writes).
        let ws = &mut self.ws;
        let layers = &self.layers;
        let kv = &mut self.kv;
        let cos = &self.cos;
        let sin = &self.sin;
        let embed = &self.embed;
        let final_norm = &self.final_norm;

        let t_max = pos0 + m;
        ws.ensure(m, h, inter, kv_dim);
        ws.scores.resize(m * n_heads * t_max, 0.0);
        let scores = &mut ws.scores[..m * n_heads * t_max];
        let x = &mut ws.x[..m * h];
        let hb = &mut ws.hb[..m * h];
        let q = &mut ws.q[..m * h];
        let k = &mut ws.k[..m * kv_dim];
        let v = &mut ws.v[..m * kv_dim];
        let attn = &mut ws.attn[..m * h];
        let o = &mut ws.o[..m * h];
        let hb2 = &mut ws.hb2[..m * h];
        let gate = &mut ws.gate[..m * inter];
        let up = &mut ws.up[..m * inter];
        let act = &mut ws.act[..m * inter];
        let down = &mut ws.down[..m * h];

        // --- embedding lookup -------------------------------------------------
        for (i, &t) in tokens.iter().enumerate() {
            let row = embed.gather(t as usize);
            x[i * h..(i + 1) * h].copy_from_slice(&row);
        }

        for (li, layer) in layers.iter().enumerate() {
            // --- attention ----------------------------------------------------
            mark = now();
            ops::rmsnorm(hb, x, &layer.in_norm, eps);

            layer.wq.matmul(q, hb, m, h, h);
            add_bias(q, &layer.bq);
            layer.wk.matmul(k, hb, m, h, kv_dim);
            add_bias(k, &layer.bk);
            layer.wv.matmul(v, hb, m, h, kv_dim);
            add_bias(v, &layer.bv);
            lap!(qkv);

            for i in 0..m {
                let pos = pos0 + i;
                ops::rope(&mut q[i * h..(i + 1) * h], cos, sin, pos, hd);
                ops::rope(&mut k[i * kv_dim..(i + 1) * kv_dim], cos, sin, pos, hd);
                kv.write(
                    li,
                    pos,
                    &k[i * kv_dim..(i + 1) * kv_dim],
                    &v[i * kv_dim..(i + 1) * kv_dim],
                );
            }
            lap!(rope);

            // Exact two-pass softmax (numerically identical to the numpy
            // reference) with a preallocated scores buffer: no per-head Vec
            // allocation, no second rayon dispatch. The streaming/online
            // variant is opt-in (YESMAN_ONLINE_ATTN=1); it saves one pass over
            // V but accumulates a small rounding drift on quantized weights.
            let kv_ref: &KvCache = kv;
            let online_attn = std::env::var_os("YESMAN_ONLINE_ATTN").is_some();
            attn.par_chunks_mut(hd)
                .zip(scores.par_chunks_mut(t_max))
                .enumerate()
                .for_each(|(idx, (out_head, sbuf))| {
                    for d in out_head.iter_mut() {
                        *d = 0.0;
                    }
                    let i = idx / n_heads;
                    let head = idx % n_heads;
                    let kv_head = head / group;
                    let qh = &q[i * h + head * hd..i * h + (head + 1) * hd];
                    let t_end = pos0 + i + 1;
                    let sbuf = &mut sbuf[..t_end];

                    if !online_attn {
                        for (t, slot) in sbuf.iter_mut().enumerate() {
                            let kh = &kv_ref.k(li, t)[kv_head * hd..(kv_head + 1) * hd];
                            *slot = ops::dot(qh, kh) * scale;
                        }
                        ops::softmax(sbuf);
                        for (t, &p) in sbuf.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let vh = &kv_ref.v(li, t)[kv_head * hd..(kv_head + 1) * hd];
                            for (a, &vv) in out_head.iter_mut().zip(vh) {
                                *a += p * vv;
                            }
                        }
                        return;
                    }

                    let mut max = f32::NEG_INFINITY;
                    let mut sum = 0.0f32;
                    for t in 0..t_end {
                        let kh = &kv_ref.k(li, t)[kv_head * hd..(kv_head + 1) * hd];
                        let s = ops::dot(qh, kh) * scale;
                        if s > max {
                            let rescale = (max - s).exp();
                            for a in out_head.iter_mut() {
                                *a *= rescale;
                            }
                            sum *= rescale;
                            max = s;
                        }
                        let p = (s - max).exp();
                        sum += p;
                        let vh = &kv_ref.v(li, t)[kv_head * hd..(kv_head + 1) * hd];
                        for (a, &vv) in out_head.iter_mut().zip(vh) {
                            *a += p * vv;
                        }
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for a in out_head.iter_mut() {
                            *a *= inv;
                        }
                    }
                });

            lap!(attn);
            layer.wo.matmul(o, attn, m, h, h);
            ops::add_into(x, o);
            lap!(o_proj);

            // --- SwiGLU feed-forward -----------------------------------------
            ops::rmsnorm(hb2, x, &layer.post_norm, eps);
            layer.w_gate.matmul(gate, hb2, m, h, inter);
            layer.w_up.matmul(up, hb2, m, h, inter);
            ops::swiglu(act, gate, up);
            layer.w_down.matmul(down, act, m, inter, h);
            ops::add_into(x, down);
            lap!(ffn);
        }

        kv.set_len(pos0 + m);

        // --- lm_head (tied embeddings) ---------------------------------------
        let xf = &mut ws.xf[..h];
        ops::rmsnorm(xf, &x[(m - 1) * h..m * h], final_norm, eps);
        let logits = &mut ws.logits[..vocab];
        embed.matmul(logits, xf, 1, h, vocab);
        lap!(head);
        if profiling {
            let _ = mark;
            prof.dump();
        }
        logits.to_vec()
    }

}

#[derive(Default)]
struct Profile {
    qkv: f64,
    rope: f64,
    attn: f64,
    o_proj: f64,
    ffn: f64,
    head: f64,
}

impl Profile {
    fn dump(&self) {
        eprintln!(
            "PROFILE ms/tok: qkv {:.1} rope {:.1} attn {:.1} o_proj {:.1} ffn {:.1} head {:.1} | sum {:.1}",
            self.qkv * 1e3,
            self.rope * 1e3,
            self.attn * 1e3,
            self.o_proj * 1e3,
            self.ffn * 1e3,
            self.head * 1e3,
            (self.qkv + self.rope + self.attn + self.o_proj + self.ffn + self.head) * 1e3,
        );
    }
}


fn cache_shape(c: &cache::QuantCache, name: &str) -> Result<Vec<usize>> {
    let e = c.entry(name)?;
    Ok(vec![e.rows, e.cols])
}

fn add_bias(out: &mut [f32], bias: &[f32]) {
    let n = bias.len();
    out.par_chunks_mut(n)
        .for_each(|row| for (v, b) in row.iter_mut().zip(bias) { *v += b; });
}
