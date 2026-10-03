#!/usr/bin/env bash
# P1 (card t_44a0ac61): where is the host AFTER the (now non-blocking) release, under unitrace?
#
# Same trick as scripts/s3ut_gdb.sh: /proc/sys/kernel/yama/ptrace_scope=1, so gdb must be the engine's PARENT.
# The engine runs under gdb, gdb under unitrace --device-timing (S3's minimal stalling mode), and the harness
# SIGINTs gdb once the release has run and the engine has had time to spin - then `thread apply all bt` names
# the call site the host is parked in now.  S3UT's stack had it inside release_gpu_waits' blocking copy; the
# question this run answers is whether that is still where it is.
#
# usage: bash p1/p1_gdb.sh <TAG> [WAIT_S_AFTER_ASK] [UTRACE: 0|7]
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; WAIT=${2:-60}; UTRACE=${3:-7}
DIR=$R/p1/traces/$TAG
LOG=$R/p1/logs/$TAG.log
ERR=$R/p1/logs/$TAG-eng.log
OUT=$R/p1/logs/$TAG-out.log
PROMPT=$R/m6c/prompts/prompt-needle-ctx4096.txt
FIFO=$R/p1/logs/$TAG-stdin.fifo
SESSION="p1g${TAG//[^A-Za-z0-9]/}"
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
  echo "P1 gdb run $TAG   $(date -Is)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "unitrace: ${UTRACE} (7 = --start-paused --session X --device-timing)"
  echo "gdb: the engine's parent (ptrace_scope $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)), interrupt ${WAIT}s after the ask"
  echo "== devices =="; fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1
} >> "$LOG" 2>&1
source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
export STRATA_DECODE_TIMING=1 STRATA_VERIFY_DEBUG=1 STRATA_VERIFY_RELEASE_DEBUG=1
export STRATA_WATCHDOG_S=900
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$R/sycl-cache/m6c
UOPT=()
[ "$UTRACE" = 7 ] && UOPT=(--start-paused --session "$SESSION" --device-timing)
GDB=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex run
     -ex "echo \n=== ALL THREADS ===\n" -ex "thread apply all bt"
     -ex "echo \n=== INFO THREADS ===\n" -ex "info threads"
     -ex "kill" -ex "quit")
cd "$DIR" || exit 1
if [ "$UTRACE" = 7 ]; then
  timeout -s INT 3000 /home/michael/pti-gpu/tools/unitrace/build/unitrace "${UOPT[@]}" \
    "${GDB[@]}" --args "$SRC/build-sycl/strata" "${ARGS[@]}" < "$FIFO" > "$OUT" 2> "$ERR" &
else
  timeout -s INT 3000 "${GDB[@]}" --args "$SRC/build-sycl/strata" "${ARGS[@]}" < "$FIFO" > "$OUT" 2> "$ERR" &
fi
HARNESS=$!
exec 9> "$FIFO"
for i in $(seq 1 1800); do
  grep -q "everything loaded" "$ERR" 2>/dev/null && break
  kill -0 "$HARNESS" 2>/dev/null || break
  sleep 1
done
echo "== loaded $(date -Is) ==" >> "$LOG"
if [ "$UTRACE" = 7 ]; then
  /home/michael/pti-gpu/tools/unitrace/build/unitrace --resume "$SESSION" >> "$LOG" 2>&1
  sleep 1
fi
printf 'GEN 16 %s\n' "$(cat "$PROMPT")" >&9
echo "== ask fed $(date -Is); waiting ${WAIT}s for the read, the window stall, the release and the spin ==" >> "$LOG"
sleep "$WAIT"
GPID=$(pgrep -f "gdb -q -batch" | head -1)
echo "== interrupting gdb pid $GPID at $(date -Is) ==" >> "$LOG"
[ -n "$GPID" ] && kill -INT "$GPID"
sleep 30
exec 9>&-
printf 'QUIT\n' >&9 2>/dev/null
kill -9 "$HARNESS" 2>/dev/null
pkill -KILL -f "build-sycl/strata --serve" 2>/dev/null
{
  echo
  echo "== the engine's own lines about the window =="
  grep -n "verify release\|timed out at layer\|no progress\|stall report\|captured the\|verify dbg:" "$ERR" | tail -30
  echo
  echo "== gdb: the backtraces =="
  sed -n '/=== ALL THREADS ===/,$p' "$OUT" | head -120
} >> "$LOG" 2>&1
echo "=== $TAG done: log $LOG ==="
