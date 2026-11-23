#!/usr/bin/env python3
"""Generate a tiny random Qwen2-shaped checkpoint + reference logits.

The result is used by `cargo test` (crates/yesman-model/tests/tiny.rs) so the
forward pass is verified against numpy without needing the 3 GB checkpoint.

usage: python3 tools/make_tiny.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_ref import forward, save_safetensors  # noqa: E402

CFG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 100,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "model_type": "qwen2",
    "bos_token_id": 0,
    "eos_token_id": 2,
}
IDS = [5, 7, 11, 13, 17, 19]

# Second fixture: all dimensions divisible by 256, needed by the Q4_K
# super-block format (the 64-wide one above can only exercise Q4_0).
CFG256 = dict(
    CFG,
    hidden_size=256,
    intermediate_size=256,
    num_hidden_layers=1,
    vocab_size=32,
    max_position_embeddings=128,
)
IDS256 = [5, 7, 11, 13]


def make(out: Path, cfg: dict, ids: list[int], seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    kv_dim = cfg["num_key_value_heads"] * (h // cfg["num_attention_heads"])

    def w(shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": w((cfg["vocab_size"], h)),
        "model.norm.weight": rng.standard_normal(h).astype(np.float32) * 0.5 + 1.0,
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        tensors.update({
            f"{p}.input_layernorm.weight": rng.standard_normal(h).astype(np.float32) * 0.5 + 1.0,
            f"{p}.post_attention_layernorm.weight": rng.standard_normal(h).astype(np.float32) * 0.5 + 1.0,
            f"{p}.self_attn.q_proj.weight": w((h, h)),
            f"{p}.self_attn.q_proj.bias": rng.standard_normal(h).astype(np.float32) * 0.1,
            f"{p}.self_attn.k_proj.weight": w((kv_dim, h)),
            f"{p}.self_attn.k_proj.bias": rng.standard_normal(kv_dim).astype(np.float32) * 0.1,
            f"{p}.self_attn.v_proj.weight": w((kv_dim, h)),
            f"{p}.self_attn.v_proj.bias": rng.standard_normal(kv_dim).astype(np.float32) * 0.1,
            f"{p}.self_attn.o_proj.weight": w((h, h)),
            f"{p}.mlp.gate_proj.weight": w((inter, h)),
            f"{p}.mlp.up_proj.weight": w((inter, h)),
            f"{p}.mlp.down_proj.weight": w((h, inter)),
        })

    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_safetensors(out / "model.safetensors", tensors, dtype="F32")
    logits = forward(tensors, cfg, ids)
    (out / "expected.json").write_text(
        json.dumps({"ids": ids, "logits": [float(v) for v in logits]}, indent=2)
    )
    print(f"wrote {out}: {len(tensors)} tensors, expected logits[0:3] = {logits[:3]}")


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("crates/yesman-model/tests/data")
    make(base / "tiny", CFG, IDS, 0)
    make(base / "tiny256", CFG256, IDS256, 1)


if __name__ == "__main__":
    main()
