#!/usr/bin/env bash
# Measure perplexity across all weight formats on a text corpus.
#
# usage: tools/eval_perplexity.sh [corpus_file] [model_dir]
set -euo pipefail

CORPUS=${1:-data/wikitext2_val_2k.txt}
MODEL=${2:-models/Qwen2.5-1.5B-Instruct}
BIN=./target/release/yesman

if [ ! -f "$CORPUS" ]; then
    echo "Corpus file '$CORPUS' not found. Fetching WikiText-2 valid split sample..."
    mkdir -p "$(dirname "$CORPUS")"
    curl -sSf https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt \
        | head -n 45 > "$CORPUS"
fi

echo "=========================================================="
echo " Perplexity Evaluation on $CORPUS"
echo "=========================================================="
printf "%-10s %-12s %-12s %-10s\n" "Format" "Cache/Size" "Perplexity" "Eval Speed"
echo "----------------------------------------------------------"

run_ppl() {
    local label=$1
    local extra_args=$2
    local out
    out=$($BIN --model "$MODEL" $extra_args --perplexity "$CORPUS" 2>&1)
    local ppl
    ppl=$(echo "$out" | grep "perplexity:" | awk '{print $2}')
    local speed
    speed=$(echo "$out" | grep "evaluated" | grep -oE "[0-9.]+ tok/s")
    printf "%-10s %-12s %-12s %-10s\n" "$label" "$3" "$ppl" "$speed"
}

run_ppl "bf16" "--dtype bf16" "2.9 GB"
if [ -f "$MODEL/model.q8.ymq" ]; then
    run_ppl "q8" "--cache $MODEL/model.q8.ymq" "1.7 GB"
else
    run_ppl "q8" "--dtype q8" "1.7 GB"
fi

if [ -f "$MODEL/model.q4km.ymq" ]; then
    run_ppl "q4km" "--cache $MODEL/model.q4km.ymq" "1.6 GB"
else
    run_ppl "q4km" "--dtype q4km" "1.6 GB"
fi

if [ -f "$MODEL/model.q4k.ymq" ]; then
    run_ppl "q4k" "--cache $MODEL/model.q4k.ymq" "993 MB"
else
    run_ppl "q4k" "--dtype q4k" "993 MB"
fi

if [ -f "$MODEL/model.q4.ymq" ]; then
    run_ppl "q4" "--cache $MODEL/model.q4.ymq" "1.1 GB"
else
    run_ppl "q4" "--dtype q4" "1.1 GB"
fi

echo "=========================================================="
