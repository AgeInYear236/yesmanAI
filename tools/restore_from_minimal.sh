#!/usr/bin/env bash
# Восстановление «полного» чекаута после чистки диска.
#
# Минимальный набор (4.6 ГБ) уже умеет отвечать через
#   ./target/release/yesman --model models/Qwen2.5-1.5B-BlenderMCP \
#       --cache models/Qwen2.5-1.5B-BlenderMCP/model.q4km.ymq --blender
#
# Этот скрипт достраивает всё, что нужно для ПЕРЕОБУЧЕНИЯ:
#   1. release-бинарники (если исчезли после cargo clean);
#   2. training/.venv с torch/transformers/peft (~5.8 ГБ);
#   3. слитые model.safetensors из LoRA-адаптера + базовой модели (~2.9 ГБ);
#   4. .ymq кэш из слитых весов (~1.6 ГБ).
#
# Всё восстанавливается офлайн из того, что осталось: базовой модели,
# адаптера и исходников. Скачивать из HF ничего не нужно.
#
# Использование: bash tools/restore_from_minimal.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-training/.venv/bin/python}
BASE_DIR="models/Qwen2.5-1.5B-Instruct"
MERGED_DIR="models/Qwen2.5-1.5B-BlenderMCP"
CACHE="$MERGED_DIR/model.q4km.ymq"
ADAPTER="training/output/lora_adapter/adapter_model.safetensors"

echo "== 1/4 binaries =="
if [ ! -x target/release/yesman ] || [ ! -x target/release/yesman-server ]; then
    cargo build --release
else
    echo "  ok (уже есть)"
fi

echo "== 2/4 training venv =="
if [ ! -f "$PYTHON" ]; then
    command -v uv >/dev/null || { echo "ERROR: нужен uv (https://docs.astral.sh/uv/)"; exit 1; }
    uv venv --python 3.12 training/.venv
    uv pip install --python training/.venv torch transformers peft datasets accelerate bitsandbytes
else
    echo "  ok (уже есть)"
fi

echo "== 3/4 merged safetensors =="
if [ ! -f "$MERGED_DIR/model.safetensors" ]; then
    [ -f "$BASE_DIR/model.safetensors" ] || {
        echo "ERROR: нет базовой модели $BASE_DIR/model.safetensors — скачайте её tools/fetch_model.sh";
        exit 1;
    }
    [ -f "$ADAPTER" ] || { echo "ERROR: нет LoRA-адаптера $ADAPTER"; exit 1; }
    "$PYTHON" training/train_lora.py --merge-only
else
    echo "  ok (уже есть)"
fi

echo "== 4/4 quant cache =="
if [ ! -f "$CACHE" ]; then
    ./target/release/yesman --model "$MERGED_DIR" --dtype q4km --save-cache "$CACHE"
else
    echo "  ok (уже есть)"
fi

echo
echo "Готово. Проверка:"
echo "  python3 tools/validate_generation.py /tmp/gen.txt"
echo "  bash tools/train_blender.sh --epochs 8"
