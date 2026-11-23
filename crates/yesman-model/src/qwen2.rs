//! Qwen2ForCausalLM forward pass, written from scratch on top of `yesman-core`.
//!
//! Layout follows the released checkpoint exactly: weights are `[out, in]`
//! row-major, attention has q/k/v biases, RoPE is the HF "rotate half" variant
//! with theta = 1e6, and `lm_head` is tied to `embed_tokens`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use yesman_core::{
    Bf16Tensor, Q4Tensor, Q8Tensor, SafeFile, Tensor, bf16, cache, ops, q4, quant,
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
}

impl DType {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => DType::F32,
            "bf16" | "bfloat16" => DType::Bf16,
            "q8" | "q8_0" | "int8" => DType::Q8,
            "q4" | "q4_0" => DType::Q4,
            other => return Err(anyhow!("unknown dtype `{other}` (f32|bf16|q8|q4)")),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::Bf16 => "bf16",
            DType::Q8 => "Q8_0",
            DType::Q4 => "Q4_0",
        }
    }
}

/// A weight matrix: f32, bf16, Q8_0 or Q4_0. Also the (tied) embedding.
enum Weight {
    F32(Tensor),
    BF16(Bf16Tensor),
    Q8(Q8Tensor),
    Q4(Q4Tensor),
}

impl Weight {
    fn load(file: &SafeFile, name: &str, dt: DType) -> Result<Self> {
        Ok(match dt {
            DType::F32 => Weight::F32(file.get(name)?),
            DType::Bf16 => Weight::BF16(file.get_bf16(name)?),
            DType::Q8 => Weight::Q8(Q8Tensor::from_bf16(&file.get_bf16(name)?)),
            DType::Q4 => Weight::Q4(Q4Tensor::from_bf16(&file.get_bf16(name)?)),
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
        }
    }

    fn gather(&self, row: usize) -> Vec<f32> {
        match self {
            Weight::F32(t) => t.row(row).to_vec(),
            Weight::BF16(t) => t.dequant_row(row),
            Weight::Q8(t) => t.dequantize_row(row),
            Weight::Q4(t) => t.dequantize_row(row),
        }
    }

    fn from_cache(c: &cache::QuantCache, name: &str) -> Result<Self> {
        Ok(match c.entry(name)?.kind {
            cache::KIND_Q8 => Weight::Q8(c.q8(name)?),
            cache::KIND_Q4 => Weight::Q4(c.q4(name)?),
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

pub struct Qwen2 {
    pub cfg: Config,
    embed: Weight,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    kv: KvCache,
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
            DType::Q4 => DType::Q8,
            other => other,
        };
        let embed = Weight::load(&file, "model.embed_tokens.weight", embed_dt)?;
        let final_norm = file.get("model.norm.weight")?.data;

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
                w_gate: Weight::load(&file, &format!("{p}.mlp.gate_proj.weight"), dt)?,
                w_up: Weight::load(&file, &format!("{p}.mlp.up_proj.weight"), dt)?,
                w_down: Weight::load(&file, &format!("{p}.mlp.down_proj.weight"), dt)?,
            });
        }

        let (cos, sin) = ops::rope_tables(max_ctx, cfg.head_dim(), cfg.rope_theta);
        let kv = KvCache::new(cfg.num_hidden_layers, max_ctx, cfg.kv_dim());

        Ok(Self { cfg, embed, final_norm, layers, cos, sin, kv })
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
        Ok(Self { cfg, embed, final_norm, layers, cos, sin, kv })
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
                .ok_or_else(|| anyhow!("save_cache requires quantized weights (--dtype q8|q4)"))?;
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

        // --- embedding lookup -------------------------------------------------
        let mut x = vec![0.0f32; m * h];
        for (i, &t) in tokens.iter().enumerate() {
            let row = self.embed.gather(t as usize);
            x[i * h..(i + 1) * h].copy_from_slice(&row);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention ----------------------------------------------------
            mark = now();
            let mut hb = vec![0.0f32; m * h];
            ops::rmsnorm(&mut hb, &x, &layer.in_norm, eps);

            let mut q = vec![0.0f32; m * h];
            layer.wq.matmul(&mut q, &hb, m, h, h);
            add_bias(&mut q, &layer.bq);
            let mut k = vec![0.0f32; m * kv_dim];
            layer.wk.matmul(&mut k, &hb, m, h, kv_dim);
            add_bias(&mut k, &layer.bk);
            let mut v = vec![0.0f32; m * kv_dim];
            layer.wv.matmul(&mut v, &hb, m, h, kv_dim);
            add_bias(&mut v, &layer.bv);
            lap!(qkv);

            for i in 0..m {
                let pos = pos0 + i;
                ops::rope(&mut q[i * h..(i + 1) * h], &self.cos, &self.sin, pos, hd);
                ops::rope(
                    &mut k[i * kv_dim..(i + 1) * kv_dim],
                    &self.cos,
                    &self.sin,
                    pos,
                    hd,
                );
                self.kv.write(
                    li,
                    pos,
                    &k[i * kv_dim..(i + 1) * kv_dim],
                    &v[i * kv_dim..(i + 1) * kv_dim],
                );
            }
            lap!(rope);

            // One rayon task per (token, head): the flat [m, hidden] buffer is
            // exactly m*n_heads chunks of head_dim.
            let mut attn = vec![0.0f32; m * h];
            let kv_ref = &self.kv;
            attn.par_chunks_mut(hd).enumerate().for_each(|(idx, out_head)| {
                let i = idx / n_heads;
                let head = idx % n_heads;
                let kv_head = head / group;
                let qh = &q[i * h + head * hd..i * h + (head + 1) * hd];
                let t_end = pos0 + i + 1;

                let mut scores = vec![0.0f32; t_end];
                for t in 0..t_end {
                    let kh = &kv_ref.k(li, t)[kv_head * hd..(kv_head + 1) * hd];
                    scores[t] = ops::dot(qh, kh) * scale;
                }
                ops::softmax(&mut scores);
                for t in 0..t_end {
                    let p = scores[t];
                    if p == 0.0 {
                        continue;
                    }
                    let vh = &kv_ref.v(li, t)[kv_head * hd..(kv_head + 1) * hd];
                    for d in 0..hd {
                        out_head[d] += p * vh[d];
                    }
                }
            });

            lap!(attn);
            let mut o = vec![0.0f32; m * h];
            layer.wo.matmul(&mut o, &attn, m, h, h);
            ops::add_into(&mut x, &o);
            lap!(o_proj);

            // --- SwiGLU feed-forward -----------------------------------------
            let mut hb2 = vec![0.0f32; m * h];
            ops::rmsnorm(&mut hb2, &x, &layer.post_norm, eps);
            let mut gate = vec![0.0f32; m * inter];
            layer.w_gate.matmul(&mut gate, &hb2, m, h, inter);
            let mut up = vec![0.0f32; m * inter];
            layer.w_up.matmul(&mut up, &hb2, m, h, inter);
            let mut act = vec![0.0f32; m * inter];
            ops::swiglu(&mut act, &gate, &up);
            let mut down = vec![0.0f32; m * h];
            layer.w_down.matmul(&mut down, &act, m, inter, h);
            ops::add_into(&mut x, &down);
            lap!(ffn);
        }

        self.kv.set_len(pos0 + m);

        // --- lm_head (tied embeddings) ---------------------------------------
        let mut xf = vec![0.0f32; h];
        ops::rmsnorm(&mut xf, &x[(m - 1) * h..m * h], &self.final_norm, eps);
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.embed.matmul(&mut logits, &xf, 1, h, cfg.vocab_size);
        lap!(head);
        if profiling {
            let _ = mark;
            prof.dump();
        }
        logits
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
