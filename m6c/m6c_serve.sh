#!/usr/bin/env bash
# M6c: the 256K measurement of record -- the Strata-recommended upstream IQ3_S GSQ-RCO GGUF, BOTH B70s,
# KV streaming on (--kv-resident 32768) unless --kvres 0 asks for the resident arm, >=256 usage-counted
# generated tokens at every length.
#
# The pair cannot run on the CLI path ("--layer-split ... needs --serve", exit 2), so this drives the
# engine's own serve protocol (the path serve/server.py uses), through m6c_drive.py so every engine line
# carries a timestamp:
#   stdin : GEN <max_new> <id,id,...>  then QUIT      (the prompt file is the comma-joined ids)
#   stdout: READY / RESUME n / PP pos total ms tok/s / T <id> per token / DONE ...
#   stderr: the split plan, the per-card expert caches, the KV-streaming lines, the request --stats
#
# ZE_AFFINITY_MASK is UNSET (PLAN 11 U11); the two-GPU config of record is strata-sycl-iq3s.json
# ("gpu": [0,1] -> serve/server.py appends --layer-split auto).
#
# usage: bash m6c/m6c_serve.sh <TAG> <CTX> [MAXNEW] [--drop] [--kvres N] [--prompt FILE] [--warmcache DIR] [EXTRA...]
R=/home/michael/strata-xpu
TAG=${1:?tag}; CTX=${2:?max-context}
shift 2
MAXNEW=256
DROP=0
KVRES=32768
TRACE=0
PREFILL=512
CACHEDIR=$R/sycl-cache/m6c
PROMPT=""
EXTRA=()
REQS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --drop) DROP=1; shift ;;
    --trace) TRACE=1; shift ;;
    --prefill-auto) PREFILL=auto; shift ;;
    --kvres) KVRES=$2; shift 2 ;;
    --prompt) PROMPT=$2; shift 2 ;;
    --req) REQS+=("$2"); shift 2 ;;
    --warmcache) CACHEDIR=$2; shift 2 ;;
    [0-9]*) MAXNEW=$1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[ -n "$PROMPT" ] || PROMPT=$R/m6c/prompts/prompt-ctx$CTX.txt

D=$R/m6c/runs/$TAG
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt
MON=$D/rss.csv; PIDF=$D/engine.pid; STDIN=$D/stdin.txt
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
mkdir -p "$D" "$CACHEDIR"
cd "$R/strata" || exit 1
if [ ${#REQS[@]} -eq 0 ]; then
  test -r "$PROMPT" || { echo "no prompt $PROMPT"; exit 2; }
fi
unset ZE_AFFINITY_MASK
if [ ${#REQS[@]} -gt 0 ]; then
  : > "$STDIN"
  for r in "${REQS[@]}"; do
    RN=${r%%:*}; RF=${r#*:}
    test -r "$RF" || { echo "no prompt $RF"; exit 2; }
    printf 'GEN %s %s\n' "$RN" "$(cat "$RF")" >> "$STDIN"
  done
  printf 'QUIT\n' >> "$STDIN"
else
  printf 'GEN %s %s\nQUIT\n' "$MAXNEW" "$(cat "$PROMPT")" > "$STDIN"
fi

{
  echo "=============================================================="
  echo "M6c serve run $TAG   $(date -Is)"
  echo "two GPUs: ZE_AFFINITY_MASK=<unset> (PLAN 11 U11), --layer-split auto (config of record strata-sycl-iq3s.json)"
  echo "model: IQ3_S GSQ-RCO snapshot ed59f92082b1e93c0e96d60a8b11aab089b52f09 (pack $PACK)"
  echo "prompt: $PROMPT  max-new=$MAXNEW  max-context=$CTX  kv-resident=$KVRES  page-cache drop=$DROP"
  echo "program cache: SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$CACHEDIR"
  echo "HEAD: $(git log --oneline -1)"
  echo "engine binary: $(stat -c '%y  %s bytes' build-sycl/strata)"
  echo
  echo "== device check BEFORE the run (PLAN 9 rule 5: no other process may hold either card) =="
  echo "-- engines/servers of ours --"
  pgrep -a -f "build-sycl/strata|serve/server.py" || echo "   none"
  echo "-- render-node holders --"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  echo "-- per-process VRAM right now (scripts/m6_occupancy.py) --"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py"
  echo "-- containers --"
  docker ps 2>&1 | head -2
  echo "-- load / disk / memory --"
  cat /proc/loadavg; df -h / | tail -1; free -g | head -2
  echo "-- biggest processes --"
  ps -eo pid,pcpu,pmem,rss,etime,comm --sort=-rss | head -6
  echo "== end of the device check =="
} > "$LOG" 2>&1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
{
  echo "-- sycl-ls (needs oneAPI setvars) --"; sycl-ls 2>&1
  echo "-- xpu-smi / intel_gpu_top --"
  command -v xpu-smi || echo "   xpu-smi NOT INSTALLED on this host"
  command -v intel_gpu_top || echo "   intel_gpu_top NOT INSTALLED on this host"
} >> "$LOG" 2>&1

export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="$CACHEDIR"
export STRATA_DECODE_TIMING=1
if [ "$TRACE" = 1 ]; then
  export STRATA_PREFILL_TRACE=1
fi

ENGINE=(./build-sycl/strata --serve
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP"
  --kv int8 --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts
  --prefill "$PREFILL" --spec 4 --spec-min-p 0.5 --max-context "$CTX" --no-capture --stats --layer-split auto
  --prompt-cache 0 --prompt-cache-every 0)
[ "$KVRES" != "0" ] && ENGINE+=(--kv-resident "$KVRES")
[ ${#EXTRA[@]} -gt 0 ] && ENGINE+=("${EXTRA[@]}")

{
  echo
  echo "== the exact engine command line =="
  echo "cd $R/strata && ZE_AFFINITY_MASK=<unset> SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$CACHEDIR \\"
  echo "  STRATA_DECODE_TIMING=1 STRATA_PREFILL_TRACE=1 \\"
  echo "  ./build-sycl/strata ${ENGINE[*]}"
  echo "  < 'GEN $MAXNEW <$(wc -c < "$PROMPT") B of ids>' + QUIT      (driven by m6c/m6c_drive.py)"
  echo "== checkpointing off: M6 measured the mid-prompt checkpoint OOMing at 32K on the split (318 MiB free) =="
  echo
} >> "$LOG" 2>&1

if [ "$DROP" = 1 ]; then
  {
    echo
    echo "== page-cache drop (POSIX_FADV_DONTNEED on both shards) =="
    grep -E "^Cached" /proc/meminfo
    /usr/bin/python3 "$R/i1/i1_io.py" drop "$SH1" "$SH2"
  } >> "$LOG" 2>&1
fi

{
  echo
  echo "== page cache just before the engine starts =="
  grep -E "^(MemTotal|MemFree|MemAvailable|Cached|Mlocked|Dirty|Unevictable)" /proc/meminfo
} >> "$LOG" 2>&1

START=$(date +%s)
timeout -s INT 14400 /usr/bin/python3 "$R/strata/m6c/m6c_drive.py" \
  --stdin "$STDIN" --out "$OUT" --err "$ERR" --timeline "$TL" --pidfile "$PIDF" -- \
  "${ENGINE[@]}" >> "$LOG" 2>&1 &
DPID=$!
sleep 3
/usr/bin/python3 "$R/strata/m6c/m6c_sample.py" "$DPID" "$MON" 14400 > "$D/rss.txt" 2>&1 &
SPID=$!
wait $DPID
RC=$?
kill $SPID 2>/dev/null
wait $SPID 2>/dev/null
WALL=$(( $(date +%s) - START ))

{
  echo "== engine exit $RC, wall ${WALL}s =="
  echo
  echo "== the split plan, the caches, the hand-off (engine stderr) =="
  grep -E "layer split|layers [0-9]+-|expert cache|resident expert|prompt path|window up to|VRAM free|device table|CUDA[01]|native pack|mmap-experts|PLE on|KV streaming" "$ERR" | head -40
  echo
  echo "== the request's own numbers (engine stderr: --stats + the serve summary) =="
  grep -E "strata serve: |^stats|^ *prefill|^ *decode|decode timing|CPU experts|expert cache hit|GPU [0-9]+ hits|since the start" "$ERR" | tail -40
  echo
  echo "== prompt chunks: first 3 and last 3 (stdout PP lines) =="
  grep -E "^PP " "$OUT" | head -3
  grep -E "^PP " "$OUT" | tail -3
  echo
  echo "== the raw protocol answer (stdout) =="
  echo "T lines: $(grep -c '^T ' "$OUT")"
  grep -E "^DONE|^ERR|^INFO|^READY|^RESUME" "$OUT" | head -8
  echo "-- first 12 and last 4 generated ids --"
  grep '^T ' "$OUT" | head -12 | tr '\n' ' '; echo
  grep '^T ' "$OUT" | tail -4 | tr '\n' ' '; echo
  echo
  echo "== the last 30 lines of the engine's stderr =="
  tail -30 "$ERR"
  echo
  echo "== RAM / SSD tier over the engine tree (m6c_sample.py) =="
  cat "$D/rss.txt"
  head -1 "$MON"
  echo "-- peak tree RSS --"
  sort -t, -k4 -n "$MON" | tail -1
  echo "-- first and last sample --"
  sed -n '2p' "$MON"; tail -1 "$MON"
} >> "$LOG" 2>&1
echo "=== $TAG done: exit $RC, wall ${WALL}s, log $LOG ==="
