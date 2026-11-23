#!/usr/bin/env python3
"""Export tiny256 model to GGUF format for testing GGUF loader."""

import json
import struct
from pathlib import Path
import numpy as np


def write_gguf_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_gguf_kv(f, key: str, val_type: int, val_bytes: bytes):
    write_gguf_string(f, key)
    f.write(struct.pack("<I", val_type))
    f.write(val_bytes)


def main():
    tiny_dir = Path("crates/yesman-model/tests/data/tiny256")
    cfg = json.loads((tiny_dir / "config.json").read_text())

    # Load safetensors using qwen_ref pure-Python loader
    import sys
    sys.path.insert(0, "tools")
    from qwen_ref import load_safetensors
    tensors = load_safetensors(tiny_dir / "model.safetensors")

    out_path = tiny_dir / "model.gguf"
    f = open(out_path, "wb")

    # 1. Header: magic GGUF, version 3, tensor_count, metadata_kv_count
    f.write(b"GGUF")
    f.write(struct.pack("<I", 3)) # v3

    # Mapping from safetensors name to GGUF name
    name_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        name_map.update({
            f"{p}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            f"{p}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            f"{p}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            f"{p}.self_attn.q_proj.bias": f"blk.{i}.attn_q.bias",
            f"{p}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            f"{p}.self_attn.k_proj.bias": f"blk.{i}.attn_k.bias",
            f"{p}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            f"{p}.self_attn.v_proj.bias": f"blk.{i}.attn_v.bias",
            f"{p}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            f"{p}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            f"{p}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            f"{p}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        })

    gguf_tensors = []
    for st_name, t in tensors.items():
        if st_name in name_map:
            gguf_tensors.append((name_map[st_name], t))

    metadata = [
        ("general.architecture", 8, struct.pack("<Q", 5) + b"qwen2"), # string = 8
        ("qwen2.context_length", 4, struct.pack("<I", cfg["max_position_embeddings"])), # uint32 = 4
        ("qwen2.embedding_length", 4, struct.pack("<I", cfg["hidden_size"])),
        ("qwen2.block_count", 4, struct.pack("<I", cfg["num_hidden_layers"])),
        ("qwen2.feed_forward_length", 4, struct.pack("<I", cfg["intermediate_size"])),
        ("qwen2.attention.head_count", 4, struct.pack("<I", cfg["num_attention_heads"])),
        ("qwen2.attention.head_count_kv", 4, struct.pack("<I", cfg["num_key_value_heads"])),
        ("qwen2.rope.freq_base", 6, struct.pack("<f", float(cfg["rope_theta"]))), # float32 = 6
        ("qwen2.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", float(cfg["rms_norm_eps"]))),
        ("general.alignment", 4, struct.pack("<I", 32)),
    ]

    f.write(struct.pack("<Q", len(gguf_tensors)))
    f.write(struct.pack("<Q", len(metadata)))

    # 2. Metadata
    for k, val_type, b in metadata:
        write_gguf_kv(f, k, val_type, b)

    # 3. Tensors info
    alignment = 32
    current_offset = 0
    tensor_payloads = []

    for name, t in gguf_tensors:
        write_gguf_string(f, name)
        dims = list(t.shape)
        dims.reverse() # GGUF stores dimensions in column-major order
        f.write(struct.pack("<I", len(dims)))
        for d in dims:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<I", 0)) # GGML_TYPE_F32 = 0
        f.write(struct.pack("<Q", current_offset))

        data = t.astype(np.float32).tobytes()
        tensor_payloads.append((current_offset, data))
        pad = (alignment - (len(data) % alignment)) % alignment
        current_offset += len(data) + pad

    # Pad header to alignment
    header_len = f.tell()
    pad = (alignment - (header_len % alignment)) % alignment
    f.write(b"\x00" * pad)

    # 4. Data payload
    for off, data in tensor_payloads:
        f.write(data)
        pad = (alignment - (len(data) % alignment)) % alignment
        if pad > 0:
            f.write(b"\x00" * pad)

    f.close()
    print(f"Exported {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
