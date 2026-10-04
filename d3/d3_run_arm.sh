#!/usr/bin/env bash
# D3 (card t_87aa2963) arm runner: D2's, retargeted at d3/runs and at the config of record AS D2 LANDED IT
# (--spec-min-p 0.7, not 0.5).  ONE engine, both B70s, ZE_AFFINITY_MASK unset.
#
#   bash d3/d3_run_arm.sh <TAG> <CTX> <MAXNEW> [--bin PATH] [--spec N] [--spec-min-p P] [--mtp-max-t N]
#                         [--prompt F] [--prefill auto|512] [--kvres N] [--graph 0|1] [--hist 0|1]
#                         [--histwindows N] [--onecard N] [--profile 0|1] [--extra ARG]... [--env VAR=VAL]...
#
#   --bin PATH   the engine binary this arm measures: copied into build-sycl/strata first (which is what the
#                config of record names), md5 recorded in the arm log.  Default: whatever is there.
#   --graph 0|1  exports STRATA_SYCL_GRAPH; omitted = the binary's own default (graph path ON since D1)
#   --hist 1     exports STRATA_LAUNCH_HIST=1 and STRATA_LAUNCH_HIST_FILE=$D/hist.txt (the launch-site census).
#                The census needs the CLOSURE path (--graph 0): a replayed graph gives no per-kernel timestamp.
#   --profile 1  exports STRATA_VERIFY_PROFILE=1 (P9's host-sampled stage stamps; +3.0% of the window)
#   --env K=V    one A/B or instrument switch (repeatable)
#
# Phases and every line of the engine's own output are timestamped in $D/timeline.txt; the ask is fed from a
# FIFO after the engine says "everything loaded", so the load is not part of any measured wall.
R=/home/michael/strata-xpu
SRC=$R/strata
set +e

TAG=${1:?tag}; CTX=${2:?ctx}; MAXNEW=${3:?maxnew}; shift 3
PROMPT=""
PREFILL=auto
KVRES=32768
BIN=""
SPEC=""
SPECMINP="0.7"
MTPMAXT=""
GRAPH=""
HIST=0
HISTW=2
ONECARD=""
PROFILE=0
EXTRA=()
EXTRAENV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT=$2; shift 2 ;;
    --prefill) PREFILL=$2; shift 2 ;;
    --kvres) KVRES=$2; shift 2 ;;
    --bin) BIN=$2; shift 2 ;;
    --spec) SPEC=$2; shift 2 ;;
    --spec-min-p) SPECMINP=$2; shift 2 ;;
    --mtp-max-t) MTPMAXT=$2; shift 2 ;;
    --graph) GRAPH=$2; shift 2 ;;
    --hist) HIST=$2; shift 2 ;;
    --histwindows) HISTW=$2; shift 2 ;;
    --onecard) ONECARD=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --env) EXTRAENV+=("$2"); shift 2 ;;
    --extra) EXTRA+=("$2"); shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
if [ -z "$PROMPT" ]; then
  PROMPT=$R/m6c/prompts/prompt-ctx$CTX.txt
  [ -r "$PROMPT" ] || PROMPT=$R/m6c/prompts/prompt-needle-ctx$CTX.txt
fi
test -r "$PROMPT" || { echo "no prompt file for ctx=$CTX"; exit 2; }
[ -n "$SPEC" ] || SPEC=4

if [ -n "$BIN" ]; then
  test -x "$BIN" || { echo "no binary at $BIN"; exit 2; }
  cp -f "$BIN" "$SRC/build-sycl/strata" || exit 2
fi

D=$R/strata/d3/runs/$TAG
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt; PIDF=$D/drive.pid
FIFO=$D/stdin.fifo; HISTF=$D/hist.txt
CACHEDIR=$R/sycl-cache/m6c
mkdir -p "$D"; rm -f "$FIFO"; mkfifo "$FIFO"
: > "$OUT"; : > "$ERR"; : > "$LOG"

PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=$R/mtp/rt
ENGINE=("$SRC/build-sycl/strata" --serve
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP"
  --kv int8 --expert-cache auto --expert-profile "$SRC/data/expert-profile.bin" --mmap-experts
  --prefill "$PREFILL" --spec "$SPEC" --spec-min-p "$SPECMINP" --max-context "$CTX" --no-capture --stats
  --prompt-cache 0 --prompt-cache-every 0)
[ -n "$MTPMAXT" ] && ENGINE+=(--mtp-max-t "$MTPMAXT")
if [ -n "$ONECARD" ]; then export ZE_AFFINITY_MASK="$ONECARD"; else unset ZE_AFFINITY_MASK; ENGINE+=(--layer-split auto); fi
[ "$KVRES" != "0" ] && ENGINE+=(--kv-resident "$KVRES")
[ ${#EXTRA[@]} -gt 0 ] && ENGINE+=("${EXTRA[@]}")

# clear every A/B switch a previous arm may have exported (a leaked switch silently measures the other path)
unset STRATA_PROMPT_ATTN_OLD STRATA_PREFILL_TRACE STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_TRACE STRATA_DBG_NAN STRATA_DUMP_LADDER STRATA_VERIFY_TAIL_DEBUG STRATA_DEC_BATCH STRATA_HC_SPLIT
unset STRATA_KV_ROT STRATA_GR_V3 STRATA_SEL_GFX12 STRATA_REFILL_BLOCKING STRATA_RESIDENT_PIN STRATA_SPIN_PAUSE
unset STRATA_PA_WMMA STRATA_COMMIT_SYNC STRATA_VERIFY_DEVICE_PLAN STRATA_WINDOW_PLAIN_GR STRATA_FINALIZER_WAIT_S
unset STRATA_VERIFY_PROFILE STRATA_SYCL_GRAPH STRATA_LAUNCH_HIST STRATA_LAUNCH_HIST_FILE STRATA_LAUNCH_HIST_WINDOWS
unset STRATA_MTP_BATCH STRATA_SPEC_COUPLED STRATA_TEST_VERIFY_STALL STRATA_RELEASE_DRAIN_MS STRATA_TEARDOWN_WAIT_MS
unset STRATA_SCORES_MULTI STRATA_TOPK_OLD STRATA_VERIFY_DEVICE_PLAN STRATA_MMVQ_MULTI_GENERIC
for kv in "${EXTRAENV[@]}"; do export "$kv"; done
if [ -n "$GRAPH" ]; then export STRATA_SYCL_GRAPH="$GRAPH"; fi
if [ "$HIST" = 1 ]; then
  export STRATA_LAUNCH_HIST=1 STRATA_LAUNCH_HIST_FILE="$HISTF" STRATA_LAUNCH_HIST_WINDOWS="$HISTW"
  : > "$HISTF"
fi
if [ "$PROFILE" = 1 ]; then export STRATA_VERIFY_PROFILE=1; fi
export STRATA_SUBMIT_COUNT=1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHEDIR"
export STRATA_DECODE_TIMING=1
export STRATA_FINALIZER_WAIT_S=10

{
  echo "D3 arm $TAG   $(date -Is)"
  echo "ctx=$CTX max-new=$MAXNEW prompt=$PROMPT ($(wc -c < "$PROMPT") B) prefill=$PREFILL kvres=$KVRES"
  echo "spec=$SPEC spec-min-p=$SPECMINP mtp-max-t=${MTPMAXT:-<none>} graph=${GRAPH:-<unset=default>} hist=$HIST/$HISTW profile=$PROFILE"
  echo "onecard=${ONECARD:-<none>} extra=${EXTRA[*]} extraenv=${EXTRAENV[*]}"
  echo "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>} SYCL_CACHE_DIR=$CACHEDIR  mtp=$MTP"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "engine binary: $(stat -c '%y  %s bytes' $SRC/build-sycl/strata)  md5 $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  echo "-- device check BEFORE the run --"
  pgrep -a -x strata || echo "   no engine of ours"
  pgrep -a -f "serve/server.py" || echo "   no server of ours"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  cat /proc/loadavg
} >> "$LOG" 2>&1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
cd "$D" || exit 1

echo "== the exact command ==" >> "$LOG"
echo "cd $D && ${ENGINE[*]}   (stdin from $FIFO)" >> "$LOG"

ASKFILE=$D/ask.txt
printf 'GEN %s %s\n' "$MAXNEW" "$(cat "$PROMPT")" > "$ASKFILE"
test "$(wc -c < "$ASKFILE")" -gt 64 || { echo "the ask is malformed ($(wc -c < "$ASKFILE") B): $PROMPT"; exit 3; }
echo "ask: $(wc -c < "$ASKFILE") B, ids $(awk '{n=split($3,a,","); print n}' "$ASKFILE")" >> "$LOG"

/usr/bin/python3 "$SRC/d3/d3_drive.py" --fifo "$FIFO" --out "$OUT" --err "$ERR" --timeline "$TL" \
    --pidfile "$PIDF" -- "${ENGINE[@]}" >> "$LOG" 2>&1 &
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

T0=""; T1=""
if [ "$READY" = 1 ]; then
  T0=$(( $(date +%s%N) / 1000000 ))
  echo "== ask fed at $(date -Is) (epoch_ms $T0) ==" >> "$LOG"
  cat "$ASKFILE" >&9
fi

SAW=0
for j in $(seq 1 60000); do
  if grep -qE "^DONE|^ERR " "$OUT" 2>/dev/null; then SAW=1; break; fi
  if ! kill -0 "$DRV" 2>/dev/null; then break; fi
  sleep 0.5
done
T1=$(( $(date +%s%N) / 1000000 ))
echo "== the ask finished=$SAW after $(( T1 - T0 )) ms ($(date -Is)) ==" >> "$LOG"

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
  echo "== engine exit $RC, wall $(( $(date +%s) - E_START )) s (ask wall $(( T1 - T0 )) ms) =="
  echo "== the graph path's own banner =="
  grep -E "strata/sycl: (the graph path|STRATA_SYCL_GRAPH)" "$ERR" | head -2
  echo "== the engine's numbers =="
  grep -E "strata decode timing|strata decode GPU stages|INFO context=|strata serve: prompt |strata serve: decode expert cache" "$ERR" | tail -8
  echo "== submit lines =="
  grep -E "strata submit: " "$ERR" | tail -4
  echo "== the request =="
  grep -E "^DONE|^ERR " "$OUT" | tail -3
  echo "PP lines: $(grep -c '^PP ' "$OUT")  last: $(grep '^PP ' "$OUT" | tail -1)"
  echo "T lines: $(grep -c '^T ' "$OUT")  md5 $(grep '^T ' "$OUT" | md5sum | cut -c1-32)"
  if [ "$HIST" = 1 ]; then
    echo "== the launch-site histogram lines written =="
    wc -l "$HISTF" 2>/dev/null
    head -3 "$HISTF" 2>/dev/null
  fi
} >> "$LOG" 2>&1
echo "=== D3 arm $TAG done: rc=$RC wall=$(( $(date +%s) - E_START ))s log=$LOG"
