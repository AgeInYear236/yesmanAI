# yesmanAI — Qwen2.5 rewritten in Rust

Inference for `Qwen2.5-1.5B-Instruct`, implemented from scratch in Rust.
No ML frameworks: no torch, no candle, no burn, no BLAS. Only `safetensors`
(container reader), `tokenizers` (BPE), `rayon`, `half`, `axum` for the server.

## Status

| Stage | State |
| --- | --- |
| safetensors reader (mmap, bf16/f16/f32 -> f32) | done |
| Kernels: matmul, RMSNorm, RoPE, softmax, SwiGLU (AVX2/FMA) | done |
| Qwen2 forward pass (GQA, qkv bias, tied lm_head) | done |
| KV cache + incremental decode | done |
| Tokenizer + ChatML template | done |
| Sampler (temperature / top-k / top-p / repetition penalty) | done |
| CLI chat | done |
| Verification vs numpy reference (tiny + real weights) | done, logits match to 3e-5 |
| bf16 weight storage (exact, 2.9 GB) | done |
| Q8_0 quantization (~1.7 GB) | done |
| Q4_0 quantization (~1.1 GB, 8-bit activations) | done, experimental |
| Persistent quant cache (startup ~1 s instead of ~6 s) | done |
| OpenAI-compatible HTTP API + SSE streaming | done |
| Q4_K / AWQ-style quantization | planned |
| CUDA backend (cudarc + custom kernels) | blocked: no nvcc/NVRTC/cuBLAS installed (driver 620.70.06 is present); needs the CUDA toolkit |

## Layout

```
crates/yesman-core     tensors, CPU kernels, safetensors reader, quantizers, cache
crates/yesman-model    Qwen2 architecture, KV cache, tokenizer, sampling, chat
crates/yesman-cli       `yesman` binary (one-shot or interactive chat)
crates/yesman-server    `yesman-server` (OpenAI-compatible HTTP API)
tools/qwen_ref.py      numpy reference forward pass (ground truth)
tools/make_tiny.py     generates a tiny random checkpoint + expected logits
tools/ref_logits.py    prints reference logits for the real checkpoint
tools/fetch_model.sh   parallel (multi-range) HF downloader
tools/finish_download.sh  resume + assemble + sha256 check
tools/verify.sh        compare Rust (bf16/q8) vs numpy + smoke generation
```

## Build & run

```bash
cargo build --release              # target/release/yesman, yesman-server

# download the checkpoint (one HF connection is throttled here, ~80 KB/s;
# the script fetches 40 ranges in parallel, ~3 MB/s)
bash tools/fetch_model.sh Qwen/Qwen2.5-1.5B-Instruct model.safetensors \
                          models/Qwen2.5-1.5B-Instruct 40

# build a reusable quant cache once, then start from the cache (~1 s)
./target/release/yesman --dtype q8 \
    --save-cache models/Qwen2.5-1.5B-Instruct/model.q8.ymq
./target/release/yesman --cache models/Qwen2.5-1.5B-Instruct/model.q8.ymq \
    --prompt "Why is the sky blue?" --temperature 0.2 --max-tokens 256

# interactive chat (no --prompt)
./target/release/yesman --cache models/Qwen2.5-1.5B-Instruct/model.q8.ymq

# --dtype: bf16 (default, exact, ~3 GB) | f32 (~6 GB) | q8 (~1.7 GB)
#          q4   (~1.1 GB, fastest, more noise -- see below)
```

### HTTP API

```bash
./target/release/yesman-server \
    --cache models/Qwen2.5-1.5B-Instruct/model.q8.ymq --port 8000

curl -s http://127.0.0.1:8000/v1/models
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-instruct","temperature":0,"max_tokens":64,
       "messages":[{"role":"user","content":"What is 2+2? One word."}]}'
# "stream": true returns OpenAI-style SSE chunks ending with data: [DONE]
```

Works with OpenAI-compatible clients (Open WebUI, `openai` SDK with
`base_url="http://127.0.0.1:8000/v1"`). Requests are serialized on a mutex —
one model, one KV cache.

## Checked on the real checkpoint

`Qwen/Qwen2.5-1.5B-Instruct`, sha256 of `model.safetensors` verified against
HF's `x-linked-etag`. Same token ids in — identical logits out:

```
rust bf16    785  27.158203   |  numpy f32  785  27.158226
rust q8_0    785  27.203852   |  identical when loaded from the cache
```

Generation is coherent (`Paris`, `Tokyo`, `4`, poems about Rust). Pure CPU
decode on 20 cores:

| weights | load from safetensors | load from cache | decode |
| --- | --- | --- | --- |
| bf16 | ~0.9 s | — | ~3.3 tok/s |
| Q8_0 | ~5.8 s | ~1.0 s | ~5 tok/s |
| Q4_0 | ~5.7 s | ~0.3 s | ~5 tok/s |

Q4_0 weights are 4-bit, activations stay Q8_0; the tied embedding/lm_head is
kept at Q8 because Q4 noise there dominates logit error (same trick llama.cpp
uses). Q4_0 is still noticeably lossier than Q8_0 on a 1.5B model — text stays
coherent, but top-8 logits shift; `--dtype q8` is the safe default.

Performance notes: for `m = 1` (decode) matmul must split the **output**
dimension — splitting by input rows leaves 19 of 20 cores idle. Also, routing
tiny activation quantization through rayon once per weight matrix (~300
dispatches/token) cost more than the quantization itself. `YESMAN_PROFILE=1`
prints a per-stage breakdown; `cargo run --release -p yesman-core --example
bench_matmul` microbenchmarks the kernels (all ~20 GB/s at lm_head size).

## Verification

Every kernel and the whole forward pass are checked against an independent
numpy implementation:

```bash
python3 tools/make_tiny.py                 # tiny fixture + expected logits
cargo test --release                       # 17 tests: forward, KV cache, quantizers
bash tools/verify.sh                      # real weights: Rust vs numpy + generation

# real checkpoint: same token ids in, compare logits out
./target/release/yesman --print-ids --prompt "Hello"             > ids.txt
python3 tools/ref_logits.py --model models/Qwen2.5-1.5B-Instruct \
        --ids "$(cat ids.txt)" --top 8
./target/release/yesman --top-logits 8 --prompt "Hello"
```

## Architecture notes

Details that are easy to get wrong and are covered by the tests:

- **q/k/v biases** — Qwen2 (unlike Llama) keeps biases on the attention
  projections; dropping them silently degrades output.
- **RoPE** — HF `rotate_half` (two halves), not interleaved NeoX-style, with
  `theta = 1e6` and duplicated `cat(freqs, freqs)` tables.
- **GQA** — 12 query heads share 2 KV heads (`kv_head = head / 6`), KV cache is
  `num_key_value_heads * head_dim` wide, not `hidden_size`.
- **Tied embeddings** — `lm_head` is absent from the checkpoint; logits are
  `embed_tokens @ x`.
- **head_dim** — derived as `hidden_size / num_attention_heads = 128`, it is not
  a config field.
