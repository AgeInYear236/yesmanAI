#!/usr/bin/env python3
"""Print reference logits for the real checkpoint (numpy, float32).

usage:
    python3 tools/ref_logits.py --model models/Qwen2.5-1.5B-Instruct \
        --ids 151644,198,8948 --top 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_ref import forward, load_safetensors  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads((args.model / "config.json").read_text())
    W = load_safetensors(args.model / "model.safetensors")
    ids = [int(t) for t in args.ids.split(",")]
    logits = forward(W, cfg, ids)
    for i in np.argsort(-logits)[: args.top]:
        print(f"{i}\t{logits[i]:.6f}")


if __name__ == "__main__":
    main()
