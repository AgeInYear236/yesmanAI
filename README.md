# yesmanAI — Qwen2.5 rewritten in Rust

Inference for `Qwen2.5-1.5B-Instruct`, implemented from scratch in Rust.
No ML frameworks: no torch, no candle, no burn, no BLAS. Only `safetensors`
(container reader), `tokenizers` (BPE), `rayon`, `half`, `axum` for the server.

> План развития, приоритеты и известные долги: **[ROADMAP.md](ROADMAP.md)**.

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
| Chunked prefill (8k–32k context without OOM) | done |
| Perplexity evaluation across all formats (WikiText-2) | done |
| CUDA backend (zero-dependency, dynamic libcuda.so.1 driver API) | done (~22 tok/s on RTX 3060) |
| Multi-session concurrent pool | done |

## Layout

```
crates/yesman-core     tensors, CPU kernels, safetensors reader, Q8/Q4_0/Q4_K, cache
  examples/diag_q4k.rs diagnostic: dequant RMSE of Q4_K vs Q4_0 on a fixture
crates/yesman-cuda     zero-dependency CUDA driver API, PTX kernels, GPU execution
  examples/bench_gpu.rs  standalone GPU benchmark
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
    --cache models/Qwen2.5-1.5B-Instruct/model.q4km.ymq --port 8000 --max-concurrency 4

# Chat completions (OpenAI format, supports streaming via "stream": true)
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-instruct","temperature":0.7,"max_tokens":64,
       "messages":[{"role":"user","content":"What is 2+2? One word."}]}'

# Legacy text completions
curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","temperature":0,"max_tokens":8}'

# Telemetry & Health
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/stats | jq .       # JSON runtime stats & tok/s
curl -s http://127.0.0.1:8000/metrics            # Prometheus exporter format
```

**Concurrent Session Pool:**
Multiple clients run concurrently without blocking each other. Model weights (~1.6 GB) are immutable and shared across all sessions via `Arc<Qwen2Weights>`, while each concurrent worker leases an independent `KvCache` and `Workspace` scratchpad.
Tested with 4 parallel clients generating simultaneously (`tools/bench_server.py`) at ~10 tok/s aggregate throughput with zero crosstalk.

**Deployment:**
- `Dockerfile`: minimal multi-stage image.
- `tools/yesman.service`: systemd service unit template with sandboxing.
- Graceful shutdown handles `SIGINT` / `SIGTERM` cleanly.

## Checked on the real checkpoint

`Qwen/Qwen2.5-1.5B-Instruct`, sha256 of `model.safetensors` verified against
HF's `x-linked-etag`. Same token ids in — identical logits out:

```
rust bf16    785  27.158203   |  numpy f32  785  27.158226
rust q8_0    785  27.203852   |  Q4_K_M  785  27.265097
all quantized modes: logits identical whether loaded from safetensors or cache
```

### Perplexity (WikiText-2, 2173 tokens)

Evaluated end-to-end on pure CPU via `yesman --perplexity <corpus>` or `bash tools/eval_perplexity.sh`:

| Format | Weight Storage / Cache | Perplexity | Degradation vs bf16 |
| --- | --- | --- | --- |
| `bf16` | 2.9 GB | **6.13** | baseline (exact) |
| `q8` | 1.7 GB | **6.15** | +0.4% |
| `q4km` | 1.6 GB | **6.28** | **+2.4%** |
| `q4k` | 993 MB | **6.39** | **+4.3%** |
| `q4` | 1.1 GB | **6.57** | +7.1% |

Tested up to 6.3k–32k context with **chunked prefill** without memory bloat.

Generation is coherent (`Paris`, `Tokyo`, `4`, poems about Rust). Pure CPU
decode on 20 cores:

| weights | load from safetensors | load from cache | decode |
| --- | --- | --- | --- |
| bf16 (CPU) | ~0.9 s | — | ~4 tok/s |
| Q8_0 (CPU) | ~5.8 s | ~2.0 s | ~11 tok/s |
| Q4_0 (CPU) | ~5.7 s | ~1.3 s | ~15 tok/s |
| Q4_K (`q4k`, CPU) | ~5.4 s | ~1.3 s | ~16 tok/s |
| Q4_K_M (`q4km`, CPU) | ~5.9 s | ~2.0 s | ~11 tok/s |
| **Q4_K_M on RTX 3060 (GPU)** | — | **~1.1 s** *(VRAM upload 0.28s)* | **~22 tok/s** |

Use `--device cuda` or `--device cpu` (default: `auto` selects CUDA if an NVIDIA GPU driver is available, otherwise CPU).

Q4_0 weights are 4-bit with activations in Q8_0. It is noticeably lossier than
Q8_0 on a 1.5B model — text stays coherent but top-8 logits shift.
`--dtype q4km` is the sweet spot; `q4k` is for when 974 MB matters.

### Performance notes

After three CPU passes this is now **memory-bandwidth bound** (~20 GB/s on this
machine): at Q8, `ffn` reads 1.15 G weights x 1.125 B = 1.3 GB and takes ~65 ms,
`head` reads 262 MB and takes ~12 ms -- both essentially the theoretical
minimum. What got us there from ~2.8 tok/s:

1. Decode matmul must split the **output** dimension, not the input rows (with
   `m = 1` that left 19 of 20 cores idle). `ops::out_chunk` now targets ~2
   rayon tasks per worker; a fixed 256-output chunk spawned ~19k tiny tasks
   per token and was pure dispatch overhead (2.8 -> 10.7 tok/s).
2. Tiny activation quantization must not go through rayon (~300 dispatches /
   token cost more than the quantization).
3. `Qwen2` owns a `Workspace` of reused buffers: decode allocates **zero**
   vectors (was ~11 per layer). Small `rmsnorm`/`swiglu`/`add_into` run
   sequentially; the attention `scores` buffer is preallocated.
4. A streaming (online) softmax is available with `YESMAN_ONLINE_ATTN=1` (one
   pass over V), but it accumulates ~0.15 logit of drift on quantized weights
   over 28 layers, so the default stays the exact two-pass softmax, which
   matches numpy to 3e-5.
5. **Chunked prefill**: long prompts are automatically partitioned into chunks
   of 512 tokens. Peak attention scratchpad scales with the chunk size rather than
   the total sequence length, preventing $O(t^2)$ memory explosion on 8k–32k context.

`YESMAN_PROFILE=1` prints a per-stage breakdown; `cargo run --release -p
yesman-core --example bench_matmul` microbenchmarks the kernels. Further CPU
gains now require fewer bytes per weight (q4k), not faster kernels.

## Verification

Every kernel and the whole forward pass are checked against an independent
numpy implementation:

```bash
python3 tools/make_tiny.py                 # tiny fixture + expected logits
cargo test --release                       # 25 tests: forward, KV cache, quantizers, cache roundtrip, chunked prefill, perplexity
bash tools/verify.sh                      # real weights: Rust vs numpy + generation
bash tools/eval_perplexity.sh             # perplexity across all formats on WikiText-2
```

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
