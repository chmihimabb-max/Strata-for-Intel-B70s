#!/usr/bin/env bash
# P1b (card t_58d5592c): WHY does the engine die with `corrupted double-linked list` -> SIGABRT after a window that
# failed?  p1_gdb.sh cannot answer it: it exports STRATA_WATCHDOG_S=900, so the #267 release never runs and the host
# is still parked in the window's own tail sync when the harness interrupts gdb.  This harness keeps the default
# watchdog (60 s) so the release + teardown path runs, and it does NOT interrupt gdb: gdb's own `run` returns on
# SIGABRT and the batch commands then print every thread's backtrace.
#
# usage: bash p1b_gdb_abort.sh <TAG> [WAIT_S]
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; WAIT=${2:-300}
DIR=$R/p1/traces/$TAG
LOG=$R/p1/logs/$TAG.log
ERR=$R/p1/logs/$TAG-eng.log
OUT=$R/p1/logs/$TAG-out.log
PROMPT=$R/m6c/prompts/prompt-needle-ctx4096.txt
FIFO=$R/p1/logs/$TAG-stdin.fifo
mkdir -p "$DIR" "$R/p1/logs"; rm -f "$FIFO"; mkfifo "$FIFO"
: > "$OUT"; : > "$ERR"; : > "$LOG"
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
ARGS=(--serve --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP"
      --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts
      --prefill 512 --spec 4 --spec-min-p 0.5 --max-context 4096 --no-capture --stats --layer-split auto
      --prompt-cache 0 --prompt-cache-every 0)
unset ZE_AFFINITY_MASK
{
  echo "P1b gdb abort run $TAG   $(date -Is)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "binary: $SRC/build-sycl/strata ($(stat -c '%y %s bytes' $SRC/build-sycl/strata))"
  echo "env: STRATA_TEST_VERIFY_STALL=${STRATA_TEST_VERIFY_STALL:-1} STRATA_RELEASE_DRAIN_MS=${STRATA_RELEASE_DRAIN_MS:-unset}"
  echo "watchdog: STRATA_WATCHDOG_S=${STRATA_WATCHDOG_S:-default} (p1_gdb.sh forces 900; this one does not)"
  echo "gdb: parent; NOT interrupted - it returns on the signal itself (interrupt fallback ${WAIT}s)"
} >> "$LOG" 2>&1
source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
export STRATA_TEST_VERIFY_STALL=${STRATA_TEST_VERIFY_STALL:-1}
export STRATA_DECODE_TIMING=1 STRATA_VERIFY_RELEASE_DEBUG=1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$R/sycl-cache/m6c
GDB=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off"
     -ex "handle SIGINT stop print nopass" -ex run
     -ex "echo \n=== SIG %s STOPPED THE PROGRAM ===\n" -ex "bt"
     -ex "echo \n=== ALL THREADS ===\n" -ex "thread apply all bt"
     -ex "echo \n=== INFO THREADS ===\n" -ex "info threads" -ex "kill" -ex quit)
cd "$DIR" || exit 1
timeout -s INT 3600 "${GDB[@]}" --args "$SRC/build-sycl/strata" "${ARGS[@]}" < "$FIFO" > "$OUT" 2> "$ERR" &
HARNESS=$!
exec 9> "$FIFO"
for i in $(seq 1 1800); do
  grep -q "everything loaded" "$ERR" 2>/dev/null && break
  kill -0 "$HARNESS" 2>/dev/null || break
  sleep 1
done
echo "== loaded $(date -Is) ==" >> "$LOG"
printf 'GEN 16 %s\n' "$(cat "$PROMPT")" >&9
echo "== ask fed $(date -Is); waiting up to ${WAIT}s (the abort ends it earlier) ==" >> "$LOG"
for i in $(seq 1 "$WAIT"); do
  kill -0 "$HARNESS" 2>/dev/null || { echo "== gdb exited on its own after ${i}s ==" >> "$LOG"; break; }
  sleep 1
done
GPID=$(pgrep -f "gdb -q -batch" | head -1)
if [ -n "$GPID" ]; then
  echo "== the abort did NOT happen; interrupting gdb pid $GPID at $(date -Is) ==" >> "$LOG"
  kill -INT "$GPID"; sleep 30
fi
exec 9>&-
kill -9 "$HARNESS" 2>/dev/null
{
  echo
  echo "== the engine's own lines about the window =="
  grep -nE "verify release|teardown|#267|memcpy failed|corrupted|no progress|timed out at layer|stall report" "$ERR" | tail -30
  echo
  echo "== gdb: the fatal signal and the stacks =="
  sed -n '/=== SIG /,$p' "$OUT" | head -150
} >> "$LOG" 2>&1
echo "=== $TAG done: log $LOG ==="
