#!/usr/bin/env bash
# Compare Rust against the numpy reference and smoke-test generation across
# weight formats, plus check that the quant cache reproduces the same logits.
#
# usage: tools/verify.sh [model_dir] [prompt]
set -uo pipefail

MODEL=${1:-models/Qwen2.5-1.5B-Instruct}
PROMPT=${2:-"The capital of France is"}
BIN=./target/release/yesman

ids=$($BIN --model "$MODEL" --print-ids --prompt "$PROMPT")
echo "prompt: $PROMPT"
echo "ids: $ids"
echo
for dt in bf16 q8 q4; do
  echo "=== rust, $dt (top-8 logits) ==="
  $BIN --model "$MODEL" --dtype "$dt" --top-logits 8 --prompt "$PROMPT"
  echo
done
echo "=== numpy reference (f32) ==="
python3 tools/ref_logits.py --model "$MODEL" --ids "$ids" --top 8
echo
for dt in bf16 q8 q4; do
  echo "=== generation, $dt ==="
  $BIN --model "$MODEL" --dtype "$dt" --prompt "$PROMPT" --max-tokens 64 --temperature 0
  echo
done
for c in "$MODEL"/model.q8.ymq "$MODEL"/model.q4.ymq; do
  [ -f "$c" ] || continue
  echo "=== from cache: $c ==="
  $BIN --cache "$c" --top-logits 3 --prompt "$PROMPT"
done
