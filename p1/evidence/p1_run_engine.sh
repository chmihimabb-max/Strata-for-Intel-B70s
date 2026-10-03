#!/usr/bin/env bash
# P1 (card t_44a0ac61): run the config of record at 4K with an EXACT engine binary (for before/after A/B),
# with the #267 stall hook on, so the release path runs deterministically in both arms.
#
# usage: bash p1/p1_run_engine.sh <TAG> <ENGINE_BINARY> [MAXNEW] [PROMPT]
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; BIN=${2:?engine binary}; MAXNEW=${3:-256}
PROMPT=${4:-$R/m6c/prompts/prompt-needle-ctx4096.txt}
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
export STRATA_TEST_VERIFY_STALL=${STRATA_TEST_VERIFY_STALL:-1}
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
printf 'GEN %s %s\nQUIT\n' "$MAXNEW" "$(cat "$PROMPT")" > "$D/stdin.txt"
{
  echo "P1 engine arm $TAG   $(date -Is)"
  echo "binary: $BIN   ($(stat -c '%y %s bytes' "$BIN"))"
  echo "repo: $(cd $SRC && git log --oneline -1)"
  echo "env: STRATA_TEST_VERIFY_STALL=1 STRATA_VERIFY_RELEASE_DEBUG=1 STRATA_DECODE_TIMING=1"
  echo "cmd: $BIN --serve --pack ... --max-context 4096 --prefill 512 --layer-split auto (config of record, kv-resident off)"
} > "$D/log.txt" 2>&1
cd "$SRC" || exit 1
START=$(date +%s)
timeout -s INT 1800 /usr/bin/python3 "$SRC/m6c/m6c_drive.py" \
  --stdin "$D/stdin.txt" --out "$D/out.txt" --err "$D/err.txt" --timeline "$D/timeline.txt" --pidfile "$D/pid" -- \
  "$BIN" --serve --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP" \
  --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts \
  --prefill 512 --spec 4 --spec-min-p 0.5 --max-context 4096 --no-capture --stats --layer-split auto \
  --prompt-cache 0 --prompt-cache-every 0 >> "$D/log.txt" 2>&1
RC=$?
{
  echo "== exit $RC, wall $(( $(date +%s) - START ))s =="
  grep -nE "verify release|memcpy failed|corrupted|timed out at layer|released the verify|no progress for" "$D/err.txt" | tail -15
  echo "-- T lines: $(grep -c '^T ' "$D/out.txt") --"
} >> "$D/log.txt" 2>&1
echo "=== $TAG done: exit $RC, wall $(( $(date +%s) - START ))s, log $D/log.txt ==="
