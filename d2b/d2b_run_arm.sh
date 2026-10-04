#!/usr/bin/env bash
# D2b (card t_f93760a1) arm runner: ONE engine, the config of record's flags, at one context length.
#
#   bash d2b/d2b_run_arm.sh <TAG> <CTX> <MAXNEW> [options]
#
#   --bin PATH          the engine binary this arm measures: copied into build-sycl/strata first (which is what
#                       the config of record names), md5 recorded in the arm log.  Default: whatever is there.
#   --spec N            replaces the config of record's `--spec 4`
#   --spec-min-p P      the draft policy's min-p (0.7 = the config of record since D2; 0.5 = D2/D2a's lever value)
#   --sycl-cache DIR    SYCL_CACHE_DIR for this arm (default: the shared warm cache $R/sycl-cache/m6c)
#   --neo-cache DIR     NEO_CACHE_DIR (Intel NEO/IGC native-binary cache).  A fresh SYCL cache alone is NOT a
#                       cold compiler (d2a/spread-cold.txt): a genuinely cold first run needs BOTH fresh.
#                       Both directories are used as given and never modified: pointing them at fresh dirs is
#                       exactly equivalent to moving the entries aside, with nothing to restore afterwards.
#   --env VAR=VAL       extra environment for the engine (repeatable)
#   --extra ARG         extra engine argument (repeatable)
#   --graph 0|1         STRATA_SYCL_GRAPH (unset = the binary's own default)
#
# This is d2/d2_run_arm.sh (D1/D2/D2a's rig, unchanged in its phases and its ask) plus (a) the two cache
# directories as explicit knobs and (b) a before/after census of the cache entries, which is what turns "the
# build landed inside the ask" into a number.  The ask is fed from a FIFO after "everything loaded", so the
# load is not part of any measured wall.
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
GRAPH=""
SYCLCACHE=""
NEOCACHE=""
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
    --graph) GRAPH=$2; shift 2 ;;
    --sycl-cache) SYCLCACHE=$2; shift 2 ;;
    --neo-cache) NEOCACHE=$2; shift 2 ;;
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
  # --bin may point at build-sycl/strata itself (the arms of a chain that runs the binary just built)
  if [ "$(readlink -f "$BIN")" != "$(readlink -f "$SRC/build-sycl/strata")" ]; then
    cp -f "$BIN" "$SRC/build-sycl/strata" || exit 2
  fi
fi

D=$R/strata/d2/runs/$TAG
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt; PIDF=$D/drive.pid
FIFO=$D/stdin.fifo
CACHEDIR=${SYCLCACHE:-$R/sycl-cache/m6c}
NEOARG=()
mkdir -p "$D"; rm -f "$FIFO"; mkfifo "$FIFO"
: > "$OUT"; : > "$ERR"; : > "$LOG"
mkdir -p "$CACHEDIR"
if [ -n "$NEOCACHE" ]; then mkdir -p "$NEOCACHE"; NEOARG=(NEO_CACHE_PERSISTENT=1 "NEO_CACHE_DIR=$NEOCACHE"); fi

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
unset ZE_AFFINITY_MASK; ENGINE+=(--layer-split auto)
[ "$KVRES" != "0" ] && ENGINE+=(--kv-resident "$KVRES")
[ ${#EXTRA[@]} -gt 0 ] && ENGINE+=("${EXTRA[@]}")

# clear every A/B switch a previous arm may have exported (a leaked switch silently measures the other path)
unset STRATA_PROMPT_ATTN_OLD STRATA_PREFILL_TRACE STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_TRACE STRATA_DBG_NAN STRATA_DUMP_LADDER STRATA_VERIFY_TAIL_DEBUG STRATA_DEC_BATCH STRATA_HC_SPLIT
unset STRATA_KV_ROT STRATA_GR_V3 STRATA_SEL_GFX12 STRATA_REFILL_BLOCKING STRATA_RESIDENT_PIN STRATA_SPIN_PAUSE
unset STRATA_PA_WMMA STRATA_COMMIT_SYNC STRATA_VERIFY_DEVICE_PLAN STRATA_WINDOW_PLAIN_GR STRATA_FINALIZER_WAIT_S
unset STRATA_VERIFY_PROFILE STRATA_SYCL_GRAPH STRATA_LAUNCH_HIST STRATA_LAUNCH_HIST_FILE STRATA_LAUNCH_HIST_WINDOWS
unset STRATA_MTP_BATCH STRATA_SPEC_COUPLED STRATA_MMVQ_MULTI_GENERIC STRATA_MMVQ_WARMUP
for kv in "${EXTRAENV[@]}"; do export "$kv"; done
if [ -n "$GRAPH" ]; then export STRATA_SYCL_GRAPH="$GRAPH"; fi
export STRATA_SUBMIT_COUNT=1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHEDIR"
for kv in "${NEOARG[@]}"; do export "$kv"; done
export STRATA_DECODE_TIMING=1
export STRATA_FINALIZER_WAIT_S=10

count_src() {  # cache entries present in a directory (0 if it does not exist yet)
  find "$1" -name '0.src' 2>/dev/null | wc -l
}
NEO_ENTRIES_BEFORE="n/a"
[ -n "$NEOCACHE" ] && NEO_ENTRIES_BEFORE=$(find "$NEOCACHE" -type f 2>/dev/null | wc -l)
SYCL_BEFORE=$(count_src "$CACHEDIR")

{
  echo "D2b arm $TAG   $(date -Is)"
  echo "ctx=$CTX max-new=$MAXNEW prompt=$PROMPT ($(wc -c < "$PROMPT") B) prefill=$PREFILL kvres=$KVRES"
  echo "spec=$SPEC spec-min-p=$SPECMINP graph=${GRAPH:-<unset=default>}"
  echo "extra=${EXTRA[*]} extraenv=${EXTRAENV[*]}"
  echo "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>} SYCL_CACHE_DIR=$CACHEDIR ($SYCL_BEFORE entries) NEO_CACHE_DIR=${NEOCACHE:-<default ~/.cache/neo_compiler_cache>} ($NEO_ENTRIES_BEFORE files) mtp=$MTP"
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

/usr/bin/python3 "$SRC/d2/d2_drive.py" --fifo "$FIFO" --out "$OUT" --err "$ERR" --timeline "$TL" \
    --pidfile "$PIDF" -- "${ENGINE[@]}" >> "$LOG" 2>&1 &
DRV=$!
sleep 1
exec 9> "$FIFO"

E_START=$(date +%s)
echo "== drive pid $DRV; waiting for 'everything loaded' ==" >> "$LOG"
READY=0
for i in $(seq 1 2400); do
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

SYCL_AFTER=$(count_src "$CACHEDIR")
NEO_ENTRIES_AFTER="n/a"
NEWNEO="n/a"
if [ -n "$NEOCACHE" ]; then
  NEO_ENTRIES_AFTER=$(find "$NEOCACHE" -type f 2>/dev/null | wc -l)
  NEWNEO=$(( NEO_ENTRIES_AFTER - NEO_ENTRIES_BEFORE ))
fi

{
  echo "== engine exit $RC, wall $(( $(date +%s) - E_START )) s (ask wall $(( T1 - T0 )) ms) =="
  echo "== cache census: SYCL 0.src entries $SYCL_BEFORE -> $SYCL_AFTER (+$(( SYCL_AFTER - SYCL_BEFORE ))); NEO files $NEO_ENTRIES_BEFORE -> $NEO_ENTRIES_AFTER (+$NEWNEO) =="
  echo "== the warm-up's own line, if this binary has one =="
  grep -E "strata mmvq warmup" "$ERR" | tail -2
  echo "== the engine's numbers =="
  grep -E "strata decode timing|strata decode GPU stages|INFO context=|strata serve: prompt |strata serve: decode expert cache" "$ERR" | tail -8
  echo "== the request =="
  grep -E "^DONE|^ERR " "$OUT" | tail -3
  echo "PP lines: $(grep -c '^PP ' "$OUT")  last: $(grep '^PP ' "$OUT" | tail -1)"
  echo "T lines: $(grep -c '^T ' "$OUT")  md5 $(grep '^T ' "$OUT" | md5sum | cut -c1-32)"
} >> "$LOG" 2>&1
echo "=== D2b arm $TAG done: rc=$RC wall=$(( $(date +%s) - E_START ))s log=$LOG"
