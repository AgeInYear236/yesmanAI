#!/usr/bin/env python3
"""Shared numpy reference: safetensors read/write + Qwen2 forward pass.

Used by `ref_logits.py` (compare against Rust on the real checkpoint) and by
`make_tiny.py` (generate a small random model plus expected logits for tests).
"""
from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- io ---------
def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype, shape = meta["dtype"], meta["shape"]
        start, end = meta["data_offsets"]
        raw = np.frombuffer(mm, dtype=np.uint8, count=end - start, offset=8 + n + start)
        if dtype == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dtype == "F32":
            arr = raw.view(np.float32)
        else:
            raise ValueError(f"unsupported {dtype}")
        out[name] = arr.reshape(shape)
    return out


def save_safetensors(path: Path, tensors: dict[str, np.ndarray], dtype: str = "F32") -> None:
    """Minimal writer: JSON header (little-endian F32/BF16) + raw block."""
    header: dict[str, dict] = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32 if dtype == "F32" else np.float32)
        if dtype == "F32":
            blob = a.astype("<f4").tobytes()
        elif dtype == "BF16":
            u32 = a.view(np.uint32)
            blob = (u32 >> 16).astype(np.uint16).astype("<u2").tobytes()
        else:
            raise ValueError(dtype)
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    raw = json.dumps(header).encode("utf-8")
    pad = (-len(raw)) % 8
    raw += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


# -------------------------------------------------------------- math ---------
def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    ss = (x**2).mean(axis=-1, keepdims=True)
    return (x / np.sqrt(ss + eps)) * w.astype(np.float32)


def rope(x: np.ndarray, pos: int, theta: float) -> np.ndarray:
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    freq = pos * inv_freq
    cos = np.concatenate([np.cos(freq), np.cos(freq)]).astype(np.float32)
    sin = np.concatenate([np.sin(freq), np.sin(freq)]).astype(np.float32)
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + np.concatenate([-x2, x1], axis=-1) * sin


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def forward(W: dict[str, np.ndarray], cfg: dict, ids: list[int]) -> np.ndarray:
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = hidden // n_heads
    group = n_heads // n_kv
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    scale = head_dim**-0.5
    m = len(ids)

    x = W["model.embed_tokens.weight"][ids].astype(np.float32)
    for li in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{li}"
        h = rmsnorm(x, W[f"{p}.input_layernorm.weight"], eps)
        q = h @ W[f"{p}.self_attn.q_proj.weight"].T + W[f"{p}.self_attn.q_proj.bias"]
        k = h @ W[f"{p}.self_attn.k_proj.weight"].T + W[f"{p}.self_attn.k_proj.bias"]
        v = h @ W[f"{p}.self_attn.v_proj.weight"].T + W[f"{p}.self_attn.v_proj.bias"]

        q = q.reshape(m, n_heads, head_dim).copy()
        k = k.reshape(m, n_kv, head_dim).copy()
        v = v.reshape(m, n_kv, head_dim).copy()
        for t in range(m):
            q[t] = rope(q[t], t, theta)
            k[t] = rope(k[t], t, theta)

        attn = np.zeros((m, hidden), dtype=np.float32)
        for t in range(m):
            for head in range(n_heads):
                kv_head = head // group
                scores = (k[: t + 1, kv_head] @ q[t, head]) * scale
                attn[t, head * head_dim : (head + 1) * head_dim] = softmax(scores) @ v[: t + 1, kv_head]
        x = x + attn @ W[f"{p}.self_attn.o_proj.weight"].T

        h2 = rmsnorm(x, W[f"{p}.post_attention_layernorm.weight"], eps)
        g = h2 @ W[f"{p}.mlp.gate_proj.weight"].T
        u = h2 @ W[f"{p}.mlp.up_proj.weight"].T
        x = x + (silu(g) * u) @ W[f"{p}.mlp.down_proj.weight"].T

    xf = rmsnorm(x[-1], W["model.norm.weight"], eps)
    return W["model.embed_tokens.weight"].astype(np.float32) @ xf
