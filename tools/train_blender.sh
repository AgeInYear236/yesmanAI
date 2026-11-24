#!/usr/bin/env bash
# End-to-end training and export pipeline for Blender MCP fine-tuning.
#
# Steps:
#   1. Fetch open 3D/Blender datasets and format into ChatML tool-calling JSONL.
#   2. Fine-tune Qwen2.5-1.5B with LoRA on RTX 3060 (6 GB VRAM).
#   3. Merge LoRA adapter into base model weights.
#   4. Build .ymq quant cache for zero-overhead pure Rust inference.
#
# Usage:
#   bash tools/train_blender.sh [--epochs 2] [--batch-size 1]
set -euo pipefail

PYTHON=${PYTHON:-training/.venv/bin/python}

if [ ! -f "$PYTHON" ]; then
    echo "Creating virtualenv for training..."
    uv venv --python 3.12 training/.venv
    uv pip install --python training/.venv torch transformers peft datasets accelerate
fi

echo "=========================================================="
echo " Step 1: Preparing Blender MCP Tool-Calling Dataset"
echo "=========================================================="
$PYTHON training/prepare_dataset.py

echo
echo "=========================================================="
echo " Step 2: Training LoRA on RTX 3060 GPU"
echo "=========================================================="
$PYTHON training/train_lora.py "$@"

echo
echo "=========================================================="
echo " Step 3: Building Quant Cache for Pure-Rust Engine"
echo "=========================================================="
MERGED_DIR="models/Qwen2.5-1.5B-BlenderMCP"
if [ -d "$MERGED_DIR" ]; then
    ./target/release/yesman --model "$MERGED_DIR" --dtype q4km --save-cache "$MERGED_DIR/model.q4km.ymq"
    echo
    echo "=========================================================="
    echo " Step 4: Smoke Test with Blender MCP"
    echo "=========================================================="
    ./target/release/yesman --cache "$MERGED_DIR/model.q4km.ymq" --blender \
        --prompt "Посмотри сцену и создай золотой торус." --max-tokens 80 --temperature 0
fi

echo
echo "Pipeline complete! Fine-tuned Blender model is ready."
