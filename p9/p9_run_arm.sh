#!/usr/bin/env bash
# P9 (t_2403e6f6) arm runner: one engine arm at the config of record, optionally under unitrace, with the
# trace landing in the arm's OWN directory (unitrace writes the chrome timeline into the traced app's CWD).
#
#   bash p9/p9_run_arm.sh <TAG> <CTX> <MAXNEW> [--prompt F] [--prefill auto|512] [--kvres N] [--graph 0|1]
#                         [--trace 0|9|1|H] [--onecard N] [--profile 0|1] [--extra ARG]...
#
#   --trace 0  no unitrace (control)
#           9  --chrome-kernel-logging --chrome-device-logging            (device events, kernel names)
#           1  the full set: --host-timing --device-timing --verbose + all three chrome logs (+ API summary)
#           H  --host-timing --chrome-call-logging only      (the combination P1/S3 found does NOT stall)
#   --graph 1  exports STRATA_SYCL_GRAPH=1 (read once, at load)
#   --onecard N  exports ZE_AFFINITY_MASK=N and drops --layer-split
#   --profile 1  exports STRATA_VERIFY_PROFILE=1 (the engine's own per-stage GPU stamps)
#
# The tracer is started --start-paused and RESUMED after the engine's own "everything loaded" line, so the
# load (its JIT compiles and weight uploads) is inside the trace but the ask is fed only after the resume.
R=/home/michael/strata-xpu
SRC=$R/strata
trap '' PIPE
set +e

TAG=${1:?tag}; CTX=${2:?ctx}; MAXNEW=${3:?maxnew}; shift 3
PROMPT=""
PREFILL=auto
KVRES=32768
GRAPH=""
TRACE=0
ONECARD=""
PROFILE=0
EXTRA=()
EXTRAENV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT=$2; shift 2 ;;
    --prefill) PREFILL=$2; shift 2 ;;
    --kvres) KVRES=$2; shift 2 ;;
    --graph) GRAPH=$2; shift 2 ;;
    --trace) TRACE=$2; shift 2 ;;
    --onecard) ONECARD=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --env) EXTRAENV+=("$2"); shift 2 ;;
    --extra) EXTRA+=("$2"); shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
if [ -z "$PROMPT" ]; then
  PROMPT=$R/m6c/prompts/prompt-ctx$CTX.txt
  # there is no prompt-ctx4096.txt in the pack: 4K's prompt is the needle one (m6c_prompts.py writes
  # prompt-ctx<CTX> for CTX >= 32768 and prompt-needle-ctx<CTX> for 4096)
  [ -r "$PROMPT" ] || PROMPT=$R/m6c/prompts/prompt-needle-ctx$CTX.txt
fi
test -r "$PROMPT" || { echo "no prompt file for ctx=$CTX"; exit 2; }

D=$R/p9/runs/$TAG
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt; PIDF=$D/drive.pid
FIFO=$D/stdin.fifo
CACHEDIR=$R/sycl-cache/m6c
SESSION="p9${TAG//[^A-Za-z0-9]/}"
mkdir -p "$D"
rm -f "$FIFO"; mkfifo "$FIFO"
: > "$OUT"; : > "$ERR"; : > "$LOG"

PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=$R/mtp/rt
ENGINE=("$SRC/build-sycl/strata" --serve
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP"
  --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts
  --prefill "$PREFILL" --spec 4 --spec-min-p 0.5 --max-context "$CTX" --no-capture --stats
  --prompt-cache 0 --prompt-cache-every 0)
if [ -n "$ONECARD" ]; then export ZE_AFFINITY_MASK="$ONECARD"; else unset ZE_AFFINITY_MASK; ENGINE+=(--layer-split auto); fi
[ "$KVRES" != "0" ] && ENGINE+=(--kv-resident "$KVRES")
[ ${#EXTRA[@]} -gt 0 ] && ENGINE+=("${EXTRA[@]}")

PRE=()
case "$TRACE" in
  0) ;;
  9) PRE=(/home/michael/pti-gpu/tools/unitrace/build/unitrace --start-paused --session "$SESSION"
          --chrome-kernel-logging --chrome-device-logging --output "$D/utrace.json") ;;
  1) PRE=(/home/michael/pti-gpu/tools/unitrace/build/unitrace --start-paused --session "$SESSION"
          --host-timing --device-timing --verbose
          --chrome-call-logging --chrome-kernel-logging --chrome-device-logging --output "$D/utrace.json") ;;
  H) PRE=(/home/michael/pti-gpu/tools/unitrace/build/unitrace --start-paused --session "$SESSION"
          --host-timing --chrome-call-logging --output "$D/utrace.json") ;;
  *) echo "unknown --trace $TRACE"; exit 2 ;;
esac

# clear the A/B switches a previous arm may have exported (P2's lesson), then set this arm's own
unset STRATA_PROMPT_ATTN_OLD STRATA_PREFILL_TRACE STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_TRACE STRATA_DBG_NAN STRATA_DUMP_LADDER STRATA_VERIFY_TAIL_DEBUG STRATA_DUMP_LADDER
unset STRATA_SYCL_GRAPH STRATA_VERIFY_PROFILE
# the ablation switches an earlier arm may have exported (a leaked switch silently measures the other path)
unset STRATA_DEC_BATCH STRATA_HC_SPLIT STRATA_KV_ROT STRATA_GR_V3 STRATA_SEL_GFX12 STRATA_REFILL_BLOCKING
unset STRATA_RESIDENT_PIN STRATA_SPIN_PAUSE STRATA_PA_WMMA STRATA_COMMIT_SYNC STRATA_VERIFY_DEVICE_PLAN
unset STRATA_WINDOW_PLAIN_GR STRATA_FINALIZER_WAIT_S
for kv in "${EXTRAENV[@]}"; do export "$kv"; done
if [ -n "$GRAPH" ]; then export STRATA_SYCL_GRAPH="$GRAPH"; fi
if [ "$PROFILE" = 1 ]; then export STRATA_VERIFY_PROFILE=1; fi
export STRATA_SUBMIT_COUNT=1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHEDIR"
export STRATA_DECODE_TIMING=1
export STRATA_FINALIZER_WAIT_S=10

{
  echo "P9 arm $TAG   $(date -Is)"
  echo "ctx=$CTX max-new=$MAXNEW prompt=$PROMPT ($(wc -c < "$PROMPT") B) prefill=$PREFILL kvres=$KVRES"
  echo "trace=$TRACE graph=${GRAPH:-<unset>} onecard=${ONECARD:-<none>} profile=$PROFILE"
  echo "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>} SYCL_CACHE_DIR=$CACHEDIR"
  echo "mtp=$MTP"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "engine binary: $(stat -c '%y  %s bytes' $SRC/build-sycl/strata)  md5 $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  echo "unitrace: $(stat -c '%y' /home/michael/pti-gpu/tools/unitrace/build/unitrace)"
  echo "-- device check BEFORE the run --"
  pgrep -a -f "build-sycl/strata|serve/server.py" || echo "   no engine/server of ours"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
  cat /proc/loadavg; free -g | head -2; df -h "$R" | tail -1
} >> "$LOG" 2>&1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
cd "$D" || exit 1

echo "== the exact command ==" >> "$LOG"
echo "cd $D && ${PRE[*]} ${ENGINE[*]}   (stdin from $FIFO)" >> "$LOG"

ASKFILE=$D/ask.txt
printf 'GEN %s %s\n' "$MAXNEW" "$(cat "$PROMPT")" > "$ASKFILE"
test "$(wc -c < "$ASKFILE")" -gt 64 || { echo "the ask is malformed ($(wc -c < "$ASKFILE") B): $PROMPT"; tail -c 200 "$ASKFILE"; exit 3; }
echo "ask: $(wc -c < "$ASKFILE") B, $(awk '{print NF": "$1" "$2}' "$ASKFILE" | head -c 40), ids $(awk '{n=split($3,a,","); print n}' "$ASKFILE")" >> "$LOG"

/usr/bin/python3 "$SRC/p9/p9_drive.py" --fifo "$FIFO" --out "$OUT" --err "$ERR" --timeline "$TL" \
    --pidfile "$PIDF" -- "${PRE[@]}" "${ENGINE[@]}" >> "$LOG" 2>&1 &
DRV=$!
sleep 1
exec 9> "$FIFO"

E_START=$(date +%s)
echo "== drive pid $DRV; waiting for 'everything loaded' ==" >> "$LOG"
READY=0
for i in $(seq 1 1200); do
  if grep -q "everything loaded" "$ERR" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$DRV" 2>/dev/null; then break; fi
  sleep 1
done
echo "== loaded=$READY after $(( $(date +%s) - E_START ))s ($(date -Is)) ==" >> "$LOG"

if [ "$READY" = 1 ]; then
  if [ "$TRACE" != 0 ]; then
    /home/michael/pti-gpu/tools/unitrace/build/unitrace --resume "$SESSION" >> "$LOG" 2>&1
    echo "== tracer session $SESSION resumed at $(date -Is) ==" >> "$LOG"
    sleep 1
  fi
  T0=$(( $(date +%s%N) / 1000000 ))
  echo "== ask fed at $(date -Is) (epoch_ms $T0) ==" >> "$LOG"
  cat "$ASKFILE" >&9
fi

# wait for the request to be answered (a T line) or for the process to end; the under-tracer verify window
# may STALL, in which case the engine ends by itself (P1b's release path) - both are outcomes, not errors.
SAW=0
for j in $(seq 1 30000); do
  if grep -qE "^DONE|^ERR " "$OUT" 2>/dev/null; then SAW=1; break; fi
  if ! kill -0 "$DRV" 2>/dev/null; then break; fi
  sleep 0.2
done
T1=$(( $(date +%s%N) / 1000000 ))
echo "== the ask finished=$SAW after $(( T1 - T0 )) ms ($(date -Is)) ==" >> "$LOG"
grep -cE '^T ' "$OUT" >> "$LOG" 2>/dev/null

printf 'QUIT\n' >&9 2>/dev/null || echo "== QUIT write failed ==" >> "$LOG"
exec 9>&-
for i in $(seq 1 900); do
  kill -0 "$DRV" 2>/dev/null || break
  sleep 1
done
if kill -0 "$DRV" 2>/dev/null; then
  echo "== the driver is still alive 900 s after QUIT; killing it ==" >> "$LOG"
  kill -INT "$DRV" 2>/dev/null; sleep 10; kill -9 "$DRV" 2>/dev/null
fi
wait "$DRV" 2>/dev/null; RC=$?

{
  echo "== engine exit $RC, wall $(( $(date +%s) - E_START )) s =="
  echo "== the engine's numbers =="
  grep -E "strata decode timing|strata decode GPU stages|INFO context=" "$ERR" | tail -6
  echo "== submit lines =="
  grep -E "strata submit: " "$ERR" | tail -4
  echo "== the request =="
  grep -E "^DONE|^ERR " "$OUT" | tail -3
  echo "T lines: $(grep -c '^T ' "$OUT")  md5 $(grep '^T ' "$OUT" | md5sum | cut -c1-32)"
  echo "PP lines: $(grep -c '^PP ' "$OUT")  last: $(grep '^PP ' "$OUT" | tail -1)"
  echo "== the files this arm produced =="
  ls -la "$D"
  echo "-- chrome timelines anywhere under the arm dir --"
  find "$D" -name 'strata.*.json' -printf '%s\t%p\n' 2>/dev/null
} >> "$LOG" 2>&1
echo "=== P9 arm $TAG done: rc=$RC wall=$(( $(date +%s) - E_START ))s log=$LOG"
