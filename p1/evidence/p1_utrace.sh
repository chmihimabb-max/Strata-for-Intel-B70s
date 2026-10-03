#!/usr/bin/env bash
# P1 (card t_44a0ac61): the Level-Zero instrument re-test.
#
# Asks ONE question: with the release's flag publication no longer blocking, does a unitrace mode that
# instruments the DEVICE (the three modes S3 measured as "stalls the verify window": --device-timing,
# --chrome-kernel-logging, --chrome-device-logging) now run this engine to a written report?
#
# Difference from the S3/S3UT harness (scripts/s2_run.sh), deliberately: this script does NOT kill the engine
# when the watchdog reports a stall.  Killing it is what left unitrace's chrome JSON at 0 bytes; the question
# here is whether the engine can DRAIN and exit on its own, so the harness waits for it and then reports.
#
# usage: bash p1/p1_utrace.sh <TAG> <MODE: 0|1|7|9> [MAXNEW] [PROMPT]
#   MODE 0  no unitrace (the control)
#        1  --start-paused --session X --host-timing --device-timing --verbose + all three chrome logs
#        7  --start-paused --session X --device-timing                    (S3's minimal stalling mode)
#        9  --start-paused --session X --chrome-kernel-logging --chrome-device-logging
R=/home/michael/strata-xpu
SRC=$R/strata
trap '' PIPE
TAG=${1:?tag}; MODE=${2:?0|1|7|9}; MAXNEW=${3:-16}
PROMPT=${4:-$R/m6c/prompts/prompt-needle-ctx4096.txt}
CTX=4096
DIR=$R/p1/traces/$TAG
LOG=$R/p1/logs/$TAG.log
OUT=$R/p1/logs/$TAG-out.log
ERR=$R/p1/logs/$TAG-eng.log
FIFO=$R/p1/logs/$TAG-stdin.fifo
CACHEDIR=$R/sycl-cache/m6c
SESSION="p1${TAG//[^A-Za-z0-9]/}"
mkdir -p "$DIR" "$R/p1/logs"
rm -f "$FIFO"; mkfifo "$FIFO"
: > "$OUT"; : > "$ERR"; : > "$LOG"

PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
ARGS=(--serve --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP"
      --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts
      --prefill 512 --spec 4 --spec-min-p 0.5 --max-context "$CTX" --no-capture --stats
      --layer-split auto --prompt-cache 0 --prompt-cache-every 0)
unset ZE_AFFINITY_MASK

PRE=()
case "$MODE" in
  1) PRE=(/home/michael/pti-gpu/tools/unitrace/build/unitrace --start-paused --session "$SESSION"
          --host-timing --device-timing --verbose
          --chrome-call-logging --chrome-kernel-logging --chrome-device-logging
          --output "$DIR/$TAG.json") ;;
  7) PRE=(/home/michael/pti-gpu/tools/unitrace/build/unitrace --start-paused --session "$SESSION" --device-timing) ;;
  9) PRE=(/home/michael/pti-gpu/tools/unitrace/build/unitrace --start-paused --session "$SESSION"
          --chrome-kernel-logging --chrome-device-logging --output "$DIR/$TAG.json") ;;
esac

{
  echo "P1 unitrace arm $TAG   $(date -Is)"
  echo "mode=$MODE  max-new=$MAXNEW  prompt=$PROMPT ($(wc -c < "$PROMPT") B of ids, ctx=$CTX)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "engine binary: $(stat -c '%y  %s bytes' $SRC/build-sycl/strata)"
  echo "unitrace: /home/michael/pti-gpu/tools/unitrace/build/unitrace ($(stat -c '%y' /home/michael/pti-gpu/tools/unitrace/build/unitrace))"
  echo "unitrace cmd: ${PRE[*]:-<none: the control>}"
  echo "== the device check BEFORE the run =="
  echo "-- engines/servers of ours --"; pgrep -a -f "build-sycl/strata|serve/server.py" || echo "   none"
  echo "-- render-node holders --"; fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  echo "-- per-process VRAM right now --"; /usr/bin/python3 "$R/scripts/m6_occupancy.py"
  echo "-- load / memory --"; cat /proc/loadavg; free -g | head -2
  echo "== end of the device check =="
} >> "$LOG" 2>&1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHEDIR"
export STRATA_DECODE_TIMING=1 STRATA_VERIFY_RELEASE_DEBUG=1
{ echo "-- sycl-ls --"; sycl-ls 2>&1; } >> "$LOG" 2>&1

ASK=$(printf 'GEN %s %s' "$MAXNEW" "$(cat "$PROMPT")")
{
  echo
  echo "== the exact engine command line =="
  echo "cd $SRC && ZE_AFFINITY_MASK=<unset> SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$CACHEDIR \\"
  echo "  STRATA_DECODE_TIMING=1 STRATA_VERIFY_RELEASE_DEBUG=1 \\"
  echo "  ${PRE[0]:-/home/michael/strata-xpu/strata/build-sycl/strata} ./build-sycl/strata ${ARGS[*]}"
} >> "$LOG" 2>&1

cd "$SRC" || exit 1
if [ "$MODE" = 0 ]; then
  timeout -s INT 3600 /usr/bin/python3 "$SRC/m6c/m6c_drive.py" --stdin /dev/stdin --out "$OUT" --err "$ERR" \
      --timeline "$R/p1/logs/$TAG-timeline.txt" --pidfile "$R/p1/logs/$TAG.pid" -- \
      ./build-sycl/strata "${ARGS[@]}" < "$FIFO" >> "$LOG" 2>&1 &
else
  timeout -s INT 3600 "${PRE[@]}" ./build-sycl/strata "${ARGS[@]}" < "$FIFO" > "$OUT" 2> "$ERR" &
fi
EPID=$!
sleep 1
exec 9> "$FIFO"
E_START=$(date +%s)
echo "== engine pid $EPID; waiting for 'everything loaded' ==" >> "$LOG"
READY=0
for i in $(seq 1 900); do
  if grep -q "everything loaded" "$ERR" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$EPID" 2>/dev/null; then break; fi
  sleep 1
done
echo "== loaded=$READY after $(( $(date +%s) - E_START ))s ($(date -Is)) ==" >> "$LOG"

if [ "$READY" = 1 ]; then
  if [ "$MODE" != 0 ]; then
    /home/michael/pti-gpu/tools/unitrace/build/unitrace --resume "$SESSION" >> "$LOG" 2>&1
    echo "== unitrace session $SESSION resumed $(date -Is) ==" >> "$LOG"
    sleep 1
  fi
  T0=$(( $(date +%s%N) / 1000000 ))
  printf '%s\n' "$ASK" >&9
  echo "== the ask was fed at $(date -Is) (epoch_ms $T0): GEN $MAXNEW <$(wc -c < "$PROMPT") B of ids> ==" >> "$LOG"
  # NO kill on a stall: wait for the request's own line, or for the engine to go away, for up to 20 minutes.
  SAW=0
  SINCE_STALL=0
  for j in $(seq 1 6000); do
    if grep -q "strata serve: prompt " "$ERR" 2>/dev/null; then SAW=1; break; fi
    # the engine's own acknowledgement that the window failed and the release ran: the arm's question is
    # answered at that point, so stop waiting and let the harness ask the engine to quit (that is what gives
    # unitrace the process exit it needs to write its report).
    if grep -qE "verify release:|no progress for|its GPU waits were|the window's tail did not finish" "$ERR" 2>/dev/null; then
      if [ -z "$STALLED" ]; then
        STALLED=1
        echo "== the engine reported the failed window / ran the release at $(date -Is); NOT killing it (this arm asks whether it can stay alive and drain) ==" >> "$LOG"
      fi
    fi
    if [ -n "$STALLED" ]; then
      SINCE_STALL=$((SINCE_STALL + 1))
      if [ "$SINCE_STALL" -gt 900 ]; then
        echo "== 180 s past the release and the engine is still alive; stopping the arm and asking it to quit ==" >> "$LOG"
        break
      fi
    fi
    if ! kill -0 "$EPID" 2>/dev/null; then break; fi
    sleep 0.2
  done
  T1=$(( $(date +%s%N) / 1000000 ))
  echo "== ask answered=$SAW after $(( T1 - T0 )) ms wall ==" >> "$LOG"
  grep -E "strata serve: prompt |no progress for|verify release|strata verify:" "$ERR" | tail -12 >> "$LOG"
  if [ "$MODE" != 0 ] && [ "$MODE" != 9 ] && [ "$MODE" != 8 ]; then
    /home/michael/pti-gpu/tools/unitrace/build/unitrace --stop "$SESSION" >> "$LOG" 2>&1
    echo "== unitrace session $SESSION stopped $(date -Is) ==" >> "$LOG"
  fi
fi
printf 'QUIT\n' >&9 2>/dev/null || echo "== the QUIT write failed (the engine is gone) ==" >> "$LOG"
exec 9>&-
echo "== QUIT fed, waiting for the engine to exit ==" >> "$LOG"
for i in $(seq 1 300); do
  kill -0 "$EPID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$EPID" 2>/dev/null; then
  echo "== engine still alive 300 s after QUIT; killing it ==" >> "$LOG"
  kill -INT "$EPID" 2>/dev/null; sleep 10; kill -9 "$EPID" 2>/dev/null
fi
wait "$EPID" 2>/dev/null; RC=$?
echo "== engine exit $RC, wall $(( $(date +%s) - E_START )) s ==" >> "$LOG"
{
  echo
  echo "== the instrument's own verdict =="
  echo "-- trace dir --"; ls -la "$DIR"
  echo "-- S3's stall signature (a stall report with no finished request) --"
  grep -cE "no progress for" "$ERR"
  grep -E "verify release|strata verify:|released the verify window" "$ERR" | head -20
  echo "-- the request's numbers --"
  grep -E "strata serve: (prompt|decode timing)|decode timing" "$ERR" | tail -4
  grep -E "^DONE|^T " "$OUT" | head -3
  echo "T lines: $(grep -c '^T ' "$OUT" 2>/dev/null)"
  echo "-- the last 12 lines of the engine's stderr --"; tail -12 "$ERR"
} >> "$LOG" 2>&1
echo "=== $TAG done: exit $RC, wall $(( $(date +%s) - E_START ))s, log $LOG ==="
