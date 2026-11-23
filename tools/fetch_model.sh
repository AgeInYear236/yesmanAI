#!/usr/bin/env bash
# Parallel (multi-range) downloader for HuggingFace checkpoints.
# A single connection to cdn-lfs is heavily throttled here (~80 KB/s), while
# N concurrent ranges scale almost linearly, so we fetch the file in slices.
#
# usage: tools/fetch_model.sh <repo> <filename> <dest> [parts]
#   e.g. tools/fetch_model.sh Qwen/Qwen2.5-1.5B-Instruct model.safetensors \
#                             models/Qwen2.5-1.5B-Instruct 32
set -uo pipefail

REPO=${1:?repo}
FILE=${2:?file}
DEST=${3:?dest}
PARTS=${4:-32}
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

mkdir -p "$DEST"
TMP="${DEST}/${FILE}.parts"
mkdir -p "$TMP"

SIZE=$(curl -sSIL "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
if [ -z "${SIZE:-}" ]; then
  echo "cannot determine size" >&2
  exit 1
fi
echo "size: $SIZE bytes, parts: $PARTS"

chunk=$(( (SIZE + PARTS - 1) / PARTS ))

fetch() {
  local i=$1
  local out="${TMP}/part.${i}"
  local start=$(( i * chunk ))
  local end=$(( start + chunk - 1 ))
  [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
  local want=$(( end - start + 1 ))
  for attempt in $(seq 1 40); do
    local have=0
    [ -f "$out" ] && have=$(stat -c%s "$out")
    if [ "$have" -ge "$want" ]; then return 0; fi
    curl -sS -L --http1.1 --retry 5 --retry-all-errors --retry-delay 1 \
         -r $(( start + have ))-$end -o "${out}.partial" "$URL" 2>/dev/null
    if [ -f "${out}.partial" ]; then
      cat "${out}.partial" >> "$out"
      rm -f "${out}.partial"
    fi
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && return 0
  done
  echo "part $i incomplete" >&2
  return 1
}

export -f fetch
export TMP SIZE chunk URL

for round in $(seq 1 12); do
  # only launch workers whose part is still incomplete
  todo=()
  for i in $(seq 0 $(( PARTS - 1 ))); do
    start=$(( i * chunk )); end=$(( start + chunk - 1 ))
    [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
    want=$(( end - start + 1 ))
    have=0; [ -f "${TMP}/part.${i}" ] && have=$(stat -c%s "${TMP}/part.${i}")
    [ "$have" -lt "$want" ] && todo+=("$i")
  done
  [ ${#todo[@]} -eq 0 ] && break
  echo "round $round: ${#todo[@]} parts remaining"
  printf '%s\n' "${todo[@]}" | xargs -P "$PARTS" -I{} bash -c 'fetch {}'
done

# assemble
cat $(for i in $(seq 0 $((PARTS-1))); do echo "${TMP}/part.${i}"; done) > "${DEST}/${FILE}"
GOT=$(stat -c%s "${DEST}/${FILE}")
echo "got: $GOT bytes"
if [ "$GOT" -eq "$SIZE" ]; then
  rm -rf "$TMP"
  echo "OK ${DEST}/${FILE}"
else
  echo "INCOMPLETE: $GOT != $SIZE (parts kept in $TMP)" >&2
  exit 1
fi
