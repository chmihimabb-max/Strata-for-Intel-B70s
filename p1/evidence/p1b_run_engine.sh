#!/usr/bin/env bash
# P1b (card t_58d5592c): the engine arms this card needs, in one script.
#
# p1/p1_run_engine.sh is fixed to 4K + the stall hook.  This one parameterises what P1b has to measure:
#   * the same 4K stall arm (the failure path), with or without a SECOND ask after the failed window,
#   * the CLEAN path at 4K and 32K with the stall hook off (task 3: identical token ids),
#   * the target context, so the 32K arm uses the record's `--prefill auto --kv-resident` spelling.
#
# usage: bash p1b_run_engine.sh <TAG> <ENGINE_BINARY> [MAXNEW] [CTX] [STALL:0|1] [SECOND_ASK:0|1]
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; BIN=${2:?engine binary}; MAXNEW=${3:-16}; CTX=${4:-4096}; STALL=${5:-1}; SECOND=${6:-0}
if [ "$CTX" = 4096 ]; then PROMPT=$R/m6c/prompts/prompt-needle-ctx4096.txt; else PROMPT=$R/m6c/prompts/prompt-ctx${CTX}.txt; fi
[ -f "$PROMPT" ] || { echo "no prompt for ctx $CTX ($PROMPT)"; exit 2; }
D=$R/p1/runs/$TAG
mkdir -p "$D"
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
unset ZE_AFFINITY_MASK
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$R/sycl-cache/m6c
export STRATA_DECODE_TIMING=1 STRATA_VERIFY_RELEASE_DEBUG=1
export STRATA_TEST_VERIFY_STALL=$STALL
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1   # libsycl.so.9 (p1_run_engine.sh does this; this script must too)
PREFLAG=(--prefill 512); KVFLAG=()
if [ "$CTX" != 4096 ]; then PREFLAG=(--prefill auto); KVFLAG=(--kv-resident "$CTX"); fi
{
  echo "P1b engine arm $TAG   $(date -Is)"
  echo "binary: $BIN   ($(stat -c '%y %s bytes' "$BIN"))"
  echo "repo: $(cd $SRC && git log --oneline -1)"
  echo "env: STRATA_TEST_VERIFY_STALL=$STALL STRATA_RELEASE_DRAIN_MS=${STRATA_RELEASE_DRAIN_MS:-unset} STRATA_TEARDOWN_WAIT_MS=${STRATA_TEARDOWN_WAIT_MS:-unset} SECOND_ASK=$SECOND"
  echo "arm: max-new=$MAXNEW max-context=$CTX prompt=$PROMPT ($(wc -c < "$PROMPT") B of ids)"
} > "$D/log.txt" 2>&1
printf 'GEN %s %s\n' "$MAXNEW" "$(cat "$PROMPT")" > "$D/stdin.txt"
if [ "$SECOND" = 1 ]; then printf 'GEN %s %s\n' 8 "$(head -c 4000 "$PROMPT")" >> "$D/stdin.txt"; fi
printf 'QUIT\n' >> "$D/stdin.txt"
cd "$SRC" || exit 1
START=$(date +%s)
timeout -s INT 1800 /usr/bin/python3 "$SRC/m6c/m6c_drive.py" \
  --stdin "$D/stdin.txt" --out "$D/out.txt" --err "$D/err.txt" --timeline "$D/timeline.txt" --pidfile "$D/pid" -- \
  "$BIN" --serve --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP" \
  --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts \
  "${PREFLAG[@]}" --spec 4 --spec-min-p 0.5 --max-context "$CTX" --no-capture --stats \
  --layer-split auto --prompt-cache 0 --prompt-cache-every 0 "${KVFLAG[@]}" >> "$D/log.txt" 2>&1
RC=$?
{
  echo "== exit $RC, wall $(( $(date +%s) - START ))s =="
  grep -nE "verify release|teardown|memcpy failed|corrupted|timed out at layer|released the verify|no progress for|an earlier window" "$D/err.txt" | tail -25
  echo "-- generated T lines: $(grep -c '^T ' "$D/out.txt") --"
  grep -c '^T ' "$D/out.txt" > "$D/tcount.txt"
  grep '^T ' "$D/out.txt" > "$D/tokens.txt"
  md5sum "$D/tokens.txt" >> "$D/log.txt"
  grep -E "^PP|^DONE|^ERR|decode timing" "$D/out.txt" | tail -5
} >> "$D/log.txt" 2>&1
echo "=== $TAG done: exit $RC, wall $(( $(date +%s) - START ))s, tokens $(cat "$D/tcount.txt"), log $D/log.txt ==="
