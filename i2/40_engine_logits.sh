#!/usr/bin/env bash
# I2: our engine's OWN scores at a divergence point, in generate mode (--dump-logits writes one f32 row
# per stored position; --logits-stride stores position 0 and the FINAL INPUT position, which is the row
# that decides the next token).  The +1 row is the engine's answer to "what did you score there".
#
#   usage: bash i2/40_engine_logits.sh <tag> <prompt-ids-file>
set -o pipefail
R=/home/michael/strata-xpu
I2=$R/strata/i2
TAG=${1:?tag}
PF=${2:?prompt-ids-file}
DUMP=$I2/$TAG-logits.bin
OUT=$I2/$TAG-out.txt
ERR=$I2/$TAG-err.txt
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
IDS=$(cat "$PF")
cd "$R/strata" || exit 1
unset ZE_AFFINITY_MASK
{
  echo "==================== I2 engine logits dump $TAG   $(date -Is)"
  echo "prompt file $PF  ($(tr ',' '\n' < "$PF" | wc -l) tokens)"
  echo "CMD: ./build-sycl/strata --tokens \$ids --max-new 1 --dump-logits $DUMP --logits-stride 1000000 \\"
  echo "     --pack $PACK --native $SH1 --ple-gguf $SH2 --mtp $MTP --kv int8 --expert-cache auto \\"
  echo "     --expert-profile data/expert-profile.bin --mmap-experts --prefill 512 --spec 4 --spec-min-p 0.5 \\"
  echo "     --max-context 4096 --no-capture --stats --layer-split auto"
  echo "-- device check --"
  pgrep -a -f "build/bin/llama-server" || echo "   no llama-server"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py"
} > "$I2/$TAG.log" 2>&1
source /opt/intel/oneapi/setvars.sh >> "$I2/$TAG.log" 2>&1
START=$(date +%s)
timeout -s INT 3600 ./build-sycl/strata \
  --tokens "$IDS" --max-new 1 --dump-logits "$DUMP" --logits-stride 1000000 \
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP" \
  --kv int8 --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts \
  --prefill 512 --spec 4 --spec-min-p 0.5 --max-context 4096 --no-capture --stats --layer-split auto \
  > "$OUT" 2> "$ERR"
RC=$?
WALL=$(( $(date +%s) - START ))
{
  echo "== exit $RC wall ${WALL}s =="
  grep -E "sampling|logits dumped|tokens:|generated|tok/s" "$ERR" "$OUT" | head -20
  ls -la "$DUMP" 2>/dev/null
  tail -12 "$ERR"
} >> "$I2/$TAG.log" 2>&1
echo "=== $TAG exit $RC wall ${WALL}s; dump $(ls -la "$DUMP" 2>/dev/null | awk '{print $5}') bytes ==="
