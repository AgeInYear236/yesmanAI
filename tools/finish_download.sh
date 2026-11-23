#!/usr/bin/env bash
# Resume the tail of one unfinished part with parallel range requests, then
# assemble the final file and verify its sha256 against HF's x-linked-etag.
set -uo pipefail

PARTS_DIR=$1
OUT=$2
SHA=$3

SIZE=3087467144
CHUNK=$(( (SIZE + 40 - 1) / 40 ))
START=$(( 39 * CHUNK ))
END=$(( SIZE - 1 ))
WANT=$(( END - START + 1 ))
URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors"

HAVE=$(stat -c%s "$PARTS_DIR/part.39")
echo "resuming at $HAVE / $WANT"
REMAIN=$(( WANT - HAVE ))
N=12
SUB=$(( (REMAIN + N - 1) / N ))

fetch_sub() {
  local i=$1
  local s=$(( START + HAVE + i * SUB ))
  local e=$(( s + SUB - 1 ))
  [ "$e" -gt "$END" ] && e=$END
  [ "$s" -gt "$END" ] && return 0
  local out="$PARTS_DIR/sub.$i"
  for try in $(seq 1 30); do
    local got=0
    [ -f "$out" ] && got=$(stat -c%s "$out")
    local need=$(( e - s + 1 ))
    [ "$got" -ge "$need" ] && return 0
    curl -sS -L --http1.1 --max-time 180 --speed-time 20 --speed-limit 1024 \
         -r $(( s + got ))-$e -o "$out.partial" "$URL" 2>/dev/null
    [ -f "$out.partial" ] && { cat "$out.partial" >> "$out"; rm -f "$out.partial"; }
  done
  return 1
}
export -f fetch_sub
export PARTS_DIR START HAVE SUB END URL

for round in $(seq 1 8); do
  todo=()
  for i in $(seq 0 $(( N - 1 ))); do
    s=$(( START + HAVE + i * SUB )); e=$(( s + SUB - 1 ))
    [ "$e" -gt "$END" ] && e=$END
    [ "$s" -gt "$END" ] && continue
    need=$(( e - s + 1 )); got=0
    [ -f "$PARTS_DIR/sub.$i" ] && got=$(stat -c%s "$PARTS_DIR/sub.$i")
    [ "$got" -lt "$need" ] && todo+=("$i")
  done
  [ ${#todo[@]} -eq 0 ] && break
  echo "round $round: ${#todo[@]} sub-parts left"
  printf '%s\n' "${todo[@]}" | xargs -P "$N" -I{} bash -c 'fetch_sub {}'
done

for i in $(seq 0 $(( N - 1 ))); do
  [ -f "$PARTS_DIR/sub.$i" ] && cat "$PARTS_DIR/sub.$i" >> "$PARTS_DIR/part.39"
done
HAVE=$(stat -c%s "$PARTS_DIR/part.39")
echo "part.39 = $HAVE / $WANT"
[ "$HAVE" -eq "$WANT" ] || { echo "INCOMPLETE" >&2; exit 1; }

cat $(for i in $(seq 0 39); do echo "$PARTS_DIR/part.$i"; done) > "$OUT"
echo "assembled: $(stat -c%s "$OUT") bytes"
GOT=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "sha256: $GOT"
echo "expect: $SHA"
if [ "$GOT" = "$SHA" ]; then
  rm -rf "$PARTS_DIR"
  echo "OK $OUT — checksum matches"
else
  echo "CHECKSUM MISMATCH (parts kept)" >&2
  exit 1
fi
