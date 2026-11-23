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
| Q4_K super-blocks: `q4k` (Q4_K_S-like) and `q4km` (mixed, Q4_K_M-like) | done |
| Persistent quant cache (startup ~0.3-2 s instead of ~6 s) | done |
| OpenAI-compatible HTTP API + SSE streaming | done |
| CUDA backend | dropped by user request (also blocked: no nvcc/NVRTC/cuBLAS) |
| Q4_K RMSE-optimized scale search / Q6_K | optional future work |

## Layout

```
crates/yesman-core     tensors, CPU kernels, safetensors reader, Q8/Q4_0/Q4_K, cache
  examples/diag_q4k.rs diagnostic: dequant RMSE of Q4_K vs Q4_0 on a fixture
crates/yesman-model    Qwen2 architecture, KV cache, tokenizer, sampling, chat
crates/yesman-cli       `yesman` binary (one-shot or interactive chat)
crates/yesman-server    `yesman-server` (OpenAI-compatible HTTP API)
tools/qwen_ref.py      numpy reference forward pass (ground truth)
tools/make_tiny.py     generates a tiny random checkpoint + expected logits
tools/ref_logits.py    prints reference logits for the real checkpoint
tools/fetch_model.sh   parallel (multi-range) HF downloader
tools/finish_download.sh  resume + assemble + sha256 check
tools/verify.sh        compare all formats vs numpy + caches + smoke generation
```

## Build & run

```bash
cargo build --release              # target/release/yesman, yesman-server

# download the checkpoint (one HF connection is throttled here, ~80 KB/s;
# the script fetches 40 ranges in parallel, ~3 MB/s)
bash tools/fetch_model.sh Qwen/Qwen2.5-1.5B-Instruct model.safetensors \
                          models/Qwen2.5-1.5B-Instruct 40

# build a reusable quant cache once, then start from the cache (~1 s)
# q4km = best quality/size; q4k = smallest; q8 = exact-ish baseline
./target/release/yesman --dtype q4km \
    --save-cache models/Qwen2.5-1.5B-Instruct/model.q4km.ymq
./target/release/yesman --cache models/Qwen2.5-1.5B-Instruct/model.q4km.ymq \
    --prompt "Why is the sky blue?" --temperature 0.2 --max-tokens 256

# interactive chat (no --prompt)
./target/release/yesman --cache models/Qwen2.5-1.5B-Instruct/model.q8.ymq

# --dtype: bf16 (default, exact, ~3 GB) | f32 (~6 GB) | q8 (~1.7 GB)
#          q4   (Q4_0, ~1.1 GB) | q4k (Q4_K all-4bit, ~0.97 GB)
#          q4km (Q4_K attention + Q8 FFN, ~1.6 GB, Q8-like quality)
```

## Q4_K

`q4k` and `q4km` use clean-room Q4_K super-blocks: 256 weights -> 8
sub-blocks of 32, each affine 4-bit (`w = d*sc_j*q - dmin*m_j`), with one-byte
scale/min per sub-block and f32 super-scales (f16 would be **subnormal** at
Qwen2's ~1e-5 weight scales). 152 bytes/256 weights = 0.594 byte/weight, less
than Q4_0. Activations stay Q8_0; the tied embedding/lm_head is Q8 in all 4-bit
modes because Q4 noise on 152k rows dominates logit error.

Two mixes (same trick llama.cpp uses, but with Q8 instead of Q6_K on the FFN):

| dtype | attention | FFN | embed | cache size | quality |
| --- | --- | --- | --- | --- | --- |
| `q4k`  | Q4_K | Q4_K | Q8 | 974 MB | Q4_K_S-like: coherent text, top-1 correct, some logits drift up to ~1.7 |
| `q4km` | Q4_K | Q8_0 | Q8 | 1.6 GB | Q4_K_M-like: logits within ~0.6 of f32 |

On the real checkpoint (`The capital of France is`, top logit):

```
format   top-1 (id / logit)   logit[785]
f32/bf16  785 / 27.158226     27.158226   (numpy reference)
q4km      785 / 27.265097     27.265097
q8        785 / 27.203852     27.203852
q4k       785 / 26.981298     26.981298
q4        59604 / 25.981861   24.418657   (top-1 shifts)
```

Kernel-level: Q4_K dequant RMSE 3.1% on Gaussian weights (Q4_0 ~6%),
dot RMSE 0.0047 vs Q4_0 0.0120, AVX2 kernel matches the scalar reference.
The format is *not* GGUF bit-compatible (we load safetensors); the difference
is intentional -- 8-bit scale/min avoids ggml's RMSE scale search while still
fitting in a smaller block than Q4_0. Quant caches carry a block-layout epoch:
a cache built with an older layout is rejected with a clear message instead of
being misread (rebuild it with `--save-cache`).

### HTTP API

```bash
./target/release/yesman-server \
    --cache models/Qwen2.5-1.5B-Instruct/model.q4km.ymq --port 8000

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
rust q8_0    785  27.203852   |  Q4_K_M  785  27.265097
all quantized modes: logits identical whether loaded from safetensors or cache
```

Generation is coherent (`Paris`, `Tokyo`, `4`, poems about Rust). Pure CPU
decode on 20 cores:

| weights | load from safetensors | load from cache | decode |
| --- | --- | --- | --- |
| bf16 | ~0.9 s | — | ~3.3 tok/s |
| Q8_0 | ~5.8 s | ~1.0 s | ~5 tok/s |
| Q4_0 | ~5.7 s | ~0.3 s | ~5 tok/s |
| Q4_K (`q4k`) | ~5.4 s | ~3.5 s | ~4 tok/s |
| Q4_K_M (`q4km`) | ~5.9 s | ~2.0 s | ~4.4 tok/s |

Q4_0 weights are 4-bit with activations in Q8_0. It is noticeably lossier than
Q8_0 on a 1.5B model — text stays coherent but top-8 logits shift.
`--dtype q4km` is the sweet spot; `q4k` is for when 974 MB matters.

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
cargo test --release                       # 23 tests: forward, KV cache, quantizers, cache roundtrip
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
