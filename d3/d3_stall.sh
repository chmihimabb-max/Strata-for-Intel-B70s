#!/usr/bin/env bash
# D3 (card t_87aa2963): the #267 stall rig on the D3 binary, from p1/p1b_run_engine.sh's shape: the last layer's
# flag is withheld in window N (STRATA_TEST_VERIFY_STALL=N), so the GPU spins on a flag nobody raises and the
# engine's own bounded window wait + release + deterministic exit is what ends it.  This is the release path's
# guarantee, re-proved on the binary this card measures (and, if a change touches it, on the changed one).
#
#   bash d3/d3_stall.sh <TAG> <BIN> [MAXNEW] [CTX] [STALL_N] [SECOND_ASK]
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; BIN=${2:?engine binary}; MAXNEW=${3:-16}; CTX=${4:-4096}; STALL=${5:-1}; SECOND=${6:-0}
PROMPT=$R/m6c/prompts/prompt-ctx${CTX}.txt
[ -r "$PROMPT" ] || PROMPT=$R/m6c/prompts/prompt-needle-ctx${CTX}.txt
[ -r "$PROMPT" ] || { echo "no prompt for ctx $CTX"; exit 2; }
D=$SRC/d3/stall/$TAG
mkdir -p "$D"
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=$R/mtp/rt
unset ZE_AFFINITY_MASK
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$R/sycl-cache/m6c
export STRATA_DECODE_TIMING=1 STRATA_VERIFY_RELEASE_DEBUG=1 STRATA_SUBMIT_COUNT=1
export STRATA_TEST_VERIFY_STALL=$STALL STRATA_FINALIZER_WAIT_S=10
{
  echo "D3 stall arm $TAG   $(date -Is)"
  echo "binary: $BIN  ($(stat -c '%y %s bytes' "$BIN"))  md5 $(md5sum < "$BIN" | cut -c1-32)"
  echo "repo: $(cd $SRC && git log --oneline -1)"
  echo "env: STRATA_TEST_VERIFY_STALL=$STALL STRATA_RELEASE_DRAIN_MS=${STRATA_RELEASE_DRAIN_MS:-unset} SECOND_ASK=$SECOND"
  echo "arm: max-new=$MAXNEW max-context=$CTX prompt=$PROMPT one engine, both cards, layer-split auto, spec-min-p 0.7"
  pgrep -a -x strata || echo "   no engine of ours"
} > "$D/log.txt" 2>&1
printf 'GEN %s %s\n' "$MAXNEW" "$(cat "$PROMPT")" > "$D/stdin.txt"
if [ "$SECOND" = 1 ]; then printf 'GEN %s %s\n' 8 "$(head -c 4000 "$PROMPT")" >> "$D/stdin.txt"; fi
printf 'QUIT\n' >> "$D/stdin.txt"
source /opt/intel/oneapi/setvars.sh >> "$D/log.txt" 2>&1
cd "$SRC" || exit 1
START=$(date +%s)
timeout -s INT 1800 /usr/bin/python3 "$SRC/m6c/m6c_drive.py" \
  --stdin "$D/stdin.txt" --out "$D/out.txt" --err "$D/err.txt" --timeline "$D/timeline.txt" --pidfile "$D/pid" -- \
  "$BIN" --serve --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP" \
  --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts \
  --prefill auto --spec 4 --spec-min-p 0.7 --max-context "$CTX" --no-capture --stats \
  --layer-split auto --prompt-cache 0 --prompt-cache-every 0 --kv-resident 32768 >> "$D/log.txt" 2>&1
RC=$?
{
  echo "== exit $RC, wall $(( $(date +%s) - START ))s =="
  grep -nE "verify release|teardown|memcpy failed|corrupted|timed out at layer|released the verify|no progress for|an earlier window|the GPU rang" "$D/err.txt" | tail -25
  echo "-- generated T lines: $(grep -c '^T ' "$D/out.txt") --"
  grep '^T ' "$D/out.txt" > "$D/tokens.txt"
  md5sum "$D/tokens.txt"
  grep -E "^PP|^DONE|^ERR|decode timing" "$D/out.txt" | tail -5
  echo "== the engine's own end-of-run lines =="
  grep -E "strata verify:|does not finish|engine ends now|#267" "$D/err.txt" | tail -10
} >> "$D/log.txt" 2>&1
echo "=== $TAG done: exit $RC, wall $(( $(date +%s) - START ))s, tokens $(grep -c '^T ' "$D/out.txt"), log $D/log.txt ==="
