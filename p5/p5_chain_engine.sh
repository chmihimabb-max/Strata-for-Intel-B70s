#!/usr/bin/env bash
# P5 (card t_bc967b65), stage 1: the int8 arms of the A/B that P2 found the 2.21x in.
#   shipped  = the prompt path (qsa_prompt_attn_batch, the portable v1 kernel)
#   batched  = STRATA_PROMPT_ATTN_OLD=1 -> the engine's own batched decode attention (qsa_decode_attn_batch)
# One engine at a time, both GPUs, config of record, 256 greedy tokens at every length.
#
#   usage: bash p5/p5_chain_engine.sh [KV:int8|fp16] [extra labels...]
R=/home/michael/strata-xpu/strata
KV=${1:-int8}
SUFFIX=""
[ "$KV" = fp16 ] && SUFFIX="-fp16"
LOG=$R/p5/chain-engine$SUFFIX.log
: > "$LOG"
{
  echo "=== P5 engine chain kv=$KV  $(date -Is) ==="
  for ctx in 4096 32768 131072; do
    case $ctx in 4096) c=4k ;; 32768) c=32k ;; 131072) c=128k ;; esac
    for arm in shipped batched; do
      TAG=p5-$c-$arm$SUFFIX
      echo "--- $TAG  $(date -Is) ---"
      bash $R/p5/p5_arm.sh "$TAG" "$ctx" "$arm" 256 "$KV"
      echo "--- $TAG exit $?  $(date -Is) ---"
    done
  done
  echo "=== P5 engine chain kv=$KV done $(date -Is) ==="
} >> "$LOG" 2>&1
