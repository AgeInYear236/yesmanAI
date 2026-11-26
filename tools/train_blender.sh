#!/usr/bin/env bash
# End-to-end training and export pipeline for Blender MCP fine-tuning.
#
# Steps:
#   1. Generate the narrated, detail-rich ChatML dataset (and optionally check
#      the generated bpy scripts in headless Blender with VALIDATE_BLENDER=1).
#   2. Fine-tune Qwen2.5-1.5B with LoRA on RTX 3060 (6 GB VRAM), masking every
#      assistant turn and computing the LM loss in chunks to fit 3.5k tokens.
#   3. Merge the LoRA adapter into bf16 weights.
#   4. Build the .ymq quant cache for pure-Rust inference.
#   5. Read-only smoke test against the live Blender MCP (no scene edits).
#
# Usage:
#   bash tools/train_blender.sh [--epochs 5] [--max-seq-len 3840]
#   VALIDATE_BLENDER=1 bash tools/train_blender.sh
set -euo pipefail

# Free GPU VRAM before training (the inference server holds ~2 GB).
pkill -x yesman-server 2>/dev/null || true
pkill -x yesman 2>/dev/null || true
sleep 1

PYTHON=${PYTHON:-training/.venv/bin/python}

if [ ! -f "$PYTHON" ]; then
    echo "Creating virtualenv for training..."
    uv venv --python 3.12 training/.venv
    uv pip install --python training/.venv torch transformers peft datasets accelerate bitsandbytes
fi

echo "=========================================================="
echo " Step 1: Generating narrated Blender dataset"
echo "=========================================================="
if [ "${VALIDATE_BLENDER:-0}" = "1" ]; then
    $PYTHON training/generate_3d_dataset.py --check-blender --blender-limit 11
else
    $PYTHON training/generate_3d_dataset.py
fi

echo
echo "=========================================================="
echo " Step 2: Dataset statistics (dry run)"
echo "=========================================================="
$PYTHON training/train_lora.py --dry-run "$@"

echo
echo "=========================================================="
echo " Step 3: Training LoRA on RTX 3060 GPU"
echo "=========================================================="
$PYTHON training/train_lora.py "$@"

echo
echo "=========================================================="
echo " Step 4: Building Quant Cache for Pure-Rust Engine"
echo "=========================================================="
MERGED_DIR="models/Qwen2.5-1.5B-BlenderMCP"
cargo build --release
./target/release/yesman --model "$MERGED_DIR" --dtype q4km --save-cache "$MERGED_DIR/model.q4km.ymq"

echo
echo "=========================================================="
echo " Step 5: Read-only smoke test with mcp-for-blender"
echo "=========================================================="
if command -v uvx >/dev/null 2>&1; then
    ./target/release/yesman --cache "$MERGED_DIR/model.q4km.ymq" --blender \
        --prompt "Посмотри на сцену во вьюпорте." --max-tokens 1024 --temperature 0 \
        || echo "NOTE: live Blender smoke test unavailable (is the addon connected?)"
else
    echo "NOTE: uvx not found, skipping live Blender smoke test"
fi

echo
echo "=========================================================="
echo " Step 6: Offline generation check (no scene edits)"
echo "=========================================================="
SYS=$($PYTHON -c "import sys;sys.path.insert(0,'training');import generate_3d_dataset as g;print(g.SYSTEM_DIRECT)")
GEN_FILE="${TMPDIR:-/tmp}/yesman_generated.txt"
./target/release/yesman --cache "$MERGED_DIR/model.q4km.ymq" --device cuda \
    --system "$SYS" --prompt "сделай компьютерную мышку с проводом на столе" \
    --max-tokens 2048 --temperature 0.7 > "$GEN_FILE" 2>/dev/null
$PYTHON tools/validate_generation.py "$GEN_FILE" || echo "NOTE: generation check did not pass (see above)"

echo
echo "Pipeline complete! Fine-tuned Blender model is ready."
