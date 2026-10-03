#!/usr/bin/env bash
# P10 (t_1d052912) arm runner: ONE engine, the config of record, at one context length, with:
#   - the engine's own --stats (decode timing line: CPU experts / VRAM hits / PCIe; the serve prompt line:
#     prefill and decode tok/s; the prefill stats line: experts streamed/resident; the INFO context= line)
#   - per-thread CPU sampling (p10_threads.py) + raw `pidstat -t` so the CPU expert pool can be shown busy
#   - VRAM/RSS sampling from the driver's fdinfo (scripts/m6_monitor.py)
#
#   bash p10/p10_run_arm.sh <TAG> <CTX> <MAXNEW> [--onecard N] [--pool N] [--prefill auto|512]
#                           [--kvres N] [--prompt F] [--extra ARG]... [--env VAR=VAL]...
#
#   --onecard N  exports ZE_AFFINITY_MASK=N and does NOT pass --layer-split (correct for a single-card
#                instance: the mask is the only way to give one card to one engine; the two-GPU rule that
#                forbids it applies to a SPLIT run inside one engine, PLAN 11 U11)
#   --pool N     --pool-workers N (default: the engine's own default = every physical core but the first)
#
# Phases are marked in $D/markers.tsv: load_start, ready, ask_start, prompt_done, request_done, engine_exit.
R=/home/michael/strata-xpu
SRC=$R/strata
set +e

TAG=${1:?tag}; CTX=${2:?ctx}; MAXNEW=${3:?maxnew}; shift 3
PROMPT=""
PREFILL=auto
KVRES=32768
ONECARD=""
POOL=""
EXTRA=()
EXTRAENV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT=$2; shift 2 ;;
    --prefill) PREFILL=$2; shift 2 ;;
    --kvres) KVRES=$2; shift 2 ;;
    --onecard) ONECARD=$2; shift 2 ;;
    --pool) POOL=$2; shift 2 ;;
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

D=$R/p10/runs/$TAG
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt; PIDF=$D/drive.pid
FIFO=$D/stdin.fifo; MK=$D/markers.tsv; CSV=$D/threads.csv; MON=$D/monitor.csv
CACHEDIR=$R/sycl-cache/m6c
mkdir -p "$D"; rm -f "$FIFO"; mkfifo "$FIFO"
: > "$OUT"; : > "$ERR"; : > "$LOG"; : > "$MK"

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
[ -n "$POOL" ] && ENGINE+=(--pool-workers "$POOL")
[ ${#EXTRA[@]} -gt 0 ] && ENGINE+=("${EXTRA[@]}")

# clear the A/B switches a previous arm may have exported (a leaked switch silently measures the other path)
unset STRATA_PROMPT_ATTN_OLD STRATA_PREFILL_TRACE STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_TRACE STRATA_DBG_NAN STRATA_DUMP_LADDER STRATA_VERIFY_TAIL_DEBUG STRATA_DEC_BATCH STRATA_HC_SPLIT
unset STRATA_KV_ROT STRATA_GR_V3 STRATA_SEL_GFX12 STRATA_REFILL_BLOCKING STRATA_RESIDENT_PIN STRATA_SPIN_PAUSE
unset STRATA_PA_WMMA STRATA_COMMIT_SYNC STRATA_VERIFY_DEVICE_PLAN STRATA_WINDOW_PLAIN_GR STRATA_SYCL_GRAPH
unset STRATA_VERIFY_PROFILE
for kv in "${EXTRAENV[@]}"; do export "$kv"; done
export STRATA_SUBMIT_COUNT=1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHEDIR"
export STRATA_DECODE_TIMING=1
export STRATA_FINALIZER_WAIT_S=10

{
  echo "P10 arm $TAG   $(date -Is)"
  echo "ctx=$CTX max-new=$MAXNEW prompt=$PROMPT ($(wc -c < "$PROMPT") B) prefill=$PREFILL kvres=$KVRES pool=${POOL:-default}"
  echo "onecard=${ONECARD:-<none>}  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
  echo "extra=${EXTRA[*]}  extraenv=${EXTRAENV[*]}"
  echo "mtp=$MTP   SYCL_CACHE_DIR=$CACHEDIR"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "engine binary: $(stat -c '%y  %s bytes' $SRC/build-sycl/strata)  md5 $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  echo "-- device check BEFORE the run --"
  pgrep -a -f "build-sycl/strata|serve/server.py" || echo "   no engine/server of ours"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
  cat /proc/loadavg; free -g | head -2; df -h "$R" | tail -1
} >> "$LOG" 2>&1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
cd "$D" || exit 1

echo "== the exact command ==" >> "$LOG"
echo "cd $D && ${ENGINE[*]}   (stdin from $FIFO)" >> "$LOG"

ASKFILE=$D/ask.txt
printf 'GEN %s %s\n' "$MAXNEW" "$(cat "$PROMPT")" > "$ASKFILE"
echo "ask: $(wc -c < "$ASKFILE") B, ids $(awk '{n=split($3,a,","); print n}' "$ASKFILE")" >> "$LOG"

printf '%s\tload_start\n' "$(date +%s.%N)" >> "$MK"
/usr/bin/python3 "$SRC/p10/p10_drive.py" --fifo "$FIFO" --out "$OUT" --err "$ERR" --timeline "$TL" \
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
EPID=$(head -1 "$PIDF" 2>/dev/null)
printf '%s\tready\n' "$(date +%s.%N)" >> "$MK"

SAMP=0
if [ "$READY" = 1 ] && [ -n "$EPID" ]; then
  /usr/bin/python3 "$SRC/p10/p10_threads.py" "$EPID" "$CSV" 1.0 --stopfile "$D/threads.stop" >> "$LOG" 2>&1 &
  SAMP=$!
  /usr/bin/python3 "$R/scripts/m6_monitor.py" "$EPID" "$MON" 2.0 "$TAG" >> "$LOG" 2>&1 &
  MONP=$!
  ( pidstat -t -p "$EPID" 1 60 > "$D/pidstat.txt" 2>&1 ) &
  PSAMP=$!
  ( top -b -H -n 60 -d 1 -p "$EPID" > "$D/top-H.txt" 2>&1 ) &
  TSAMP=$!
  sleep 1
  T0=$(date +%s%N); T0MS=$(( T0 / 1000000 ))
  printf '%s\task_start\n' "$(date -Is)" >> "$LOG"
  printf '%s\task_start\n' "$(date +%s.%N)" >> "$MK"
  echo "== ask fed at $(date -Is) (epoch_ms $T0MS) ==" >> "$LOG"
  cat "$ASKFILE" >&9
fi

# wait for the request to be answered (a DONE line) or for the process to end
SAW=0
for j in $(seq 1 60000); do
  if grep -qE "^DONE|^ERR " "$OUT" 2>/dev/null; then SAW=1; break; fi
  if ! kill -0 "$DRV" 2>/dev/null; then break; fi
  sleep 0.5
done
T1=$(date +%s%N)
echo "== the ask finished=$SAW after $(( (T1 - T0) / 1000000 )) ms ($(date -Is)) ==" >> "$LOG"
printf '%s\trequest_done\n' "$(date +%s.%N)" >> "$MK"

# the prompt/decode split of the ask, straight from the engine's own DONE line:
#   DONE <n_gen> <prompt_tokens> <prompt_ms> <decode_ms> ...
# (rewritten here as well: the awk below can silently no-op on a quoting quirk, and the phase split of the
#  per-thread CPU evidence depends on it - p10_cpu_report.py also derives it from out.txt when it is absent)
DL=$(grep -E "^DONE" "$OUT" | tail -1)
PMS=$(printf '%s' "$DL" | awk '{print $4}')
ASK_EPOCH=$(awk -F'\t' '$2=="task_start"{print $1}' "$MK" | tail -1)
if [ -n "$ASK_EPOCH" ] && [ -n "$PMS" ]; then
  printf '%s\tprompt_done\n' "$(awk -v s="$ASK_EPOCH" -v ms="$PMS" 'BEGIN{printf "%.3f", s + ms/1000.0}')" >> "$MK"
else
  echo "== prompt_done marker not written: ASK_EPOCH='$ASK_EPOCH' PMS='$PMS' DL='$DL' ==" >> "$LOG"
fi

touch "$D/threads.stop" 2>/dev/null
sleep 3
kill "$SAMP" "$MONP" "$PSAMP" "$TSAMP" 2>/dev/null
wait "$SAMP" 2>/dev/null; wait "$MONP" 2>/dev/null; wait "$PSAMP" 2>/dev/null; wait "$TSAMP" 2>/dev/null

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
printf '%s\tengine_exit\n' "$(date +%s.%N)" >> "$MK"

{
  echo "== engine exit $RC, wall $(( $(date +%s) - E_START )) s =="
  echo "== RDMA/mmap tier: the engine's own load lines =="
  grep -E "expert cache auto|expert cache [0-9]+ slots|pre-filled|layer split auto|layer split across|of the experts resident|VRAM free with everything loaded|expert-pool workers|SSD is kept awake|experts via mmap" "$ERR"
  echo "== the engine's numbers =="
  grep -E "strata decode timing|strata decode GPU stages|INFO context=|strata serve: prompt |strata serve: decode expert cache|strata generate: prefill .* experts|strata serve: KV streaming|strata serve: layer split:" "$ERR" | tail -12
  echo "== submit lines =="
  grep -E "strata submit: " "$ERR" | tail -4
  echo "== the request =="
  grep -E "^DONE|^ERR " "$OUT" | tail -3
  echo "T lines: $(grep -c '^T ' "$OUT")  md5 $(grep '^T ' "$OUT" | md5sum | cut -c1-32)"
  echo "== monitor peak (driver fdinfo / RSS) =="
  tail -4 "$LOG" | grep monitor || true
  echo "== the files this arm produced =="
  ls -la "$D"
} >> "$LOG" 2>&1
echo "=== P10 arm $TAG done: rc=$RC wall=$(( $(date +%s) - E_START ))s log=$LOG"
