#!/usr/bin/env bash
# I2: our engine on the same IQ3_S pack, same cards, given the oracle's ids (correctness prompts + the
# perf prompt) in ONE engine process.
#
#   usage: bash i2/21_engine_run_all.sh <tag> <spec> <spec_min_p> <maxnew> [ctx] [bench_maxnew]
set -o pipefail
R=/home/michael/strata-xpu
I2=$R/strata/i2
TAG=${1:?tag}
SPEC=${2:?spec}
MP=${3:?spec_min_p}
MAXNEW=${4:?maxnew}
CTX=${5:-4096}
BMAX=${6:-256}
KV=${7:-int8}
OUT=$I2/$TAG-out.txt
ERR=$I2/$TAG-err.txt
LOG=$I2/$TAG.log
MON=$I2/$TAG-monitor.csv
MONLOG=$I2/$TAG-monitor.txt
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
cd "$R/strata" || exit 1

/usr/bin/python3 "$I2/22_make_stdin.py" "$I2/llama-streams.json" "$MAXNEW" "$I2/llama-bench.json" "$BMAX" \
  > "$I2/$TAG-stdin.txt"

unset ZE_AFFINITY_MASK
{
  echo "=============================================================="
  echo "I2 engine run $TAG   $(date -Is)"
  echo "two GPUs: ZE_AFFINITY_MASK=<unset>, --layer-split auto (serve/server.py appends the same)"
  echo "HEAD: $(git log --oneline -1)"
  echo "spec=$SPEC spec_min_p=$MP max_new=$MAXNEW ctx=$CTX bench_max_new=$BMAX"
  echo "requests: $(grep -c '^GEN ' "$I2/$TAG-stdin.txt")"
  echo
  echo "== device check BEFORE the run (no other process may hold either card) =="
  pgrep -a -f "build-sycl/strata|serve/server.py|llama-server" || echo "   none"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  /usr/bin/python3 "$R/scripts/m6_occupancy.py"
  docker ps 2>&1 | head -2
  cat /proc/loadavg; free -g | head -2
  echo "== end of the device check =="
} > "$LOG" 2>&1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
{
  echo "-- sycl-ls --"
  sycl-ls 2>&1
} >> "$LOG" 2>&1

export STRATA_DECODE_TIMING=1

{
  echo
  echo "== the exact engine command line =="
  echo "cd $R/strata && ZE_AFFINITY_MASK=<unset> ./build-sycl/strata --serve \\"
  echo "  --pack $PACK --native $SH1 --ple-gguf $SH2 --mtp $MTP \\"
  echo "  --kv $KV --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts \\"
  echo "  --prefill 512 --spec $SPEC --spec-min-p $MP --max-context $CTX --no-capture --stats --layer-split auto \\"
  echo "  < '$TAG-stdin.txt'   (regenerated against llama-streams.json + llama-bench.json)"
  echo
} >> "$LOG" 2>&1

START=$(date +%s)
timeout -s INT 7200 ./build-sycl/strata --serve \
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP" \
  --kv "$KV" --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts \
  --prefill 512 --spec "$SPEC" --spec-min-p "$MP" --max-context "$CTX" --no-capture --stats --layer-split auto \
  < "$I2/$TAG-stdin.txt" > "$OUT" 2> "$ERR" &
EPID=$!
/usr/bin/python3 "$R/scripts/m6_monitor_tree.py" "$EPID" "$MON" 1 "$TAG" > "$MONLOG" 2>&1 &
MPID=$!
/usr/bin/python3 "$I2/23_io_watch.py" "$I2/$TAG-io.csv" 900 > "$I2/$TAG-io.txt" 2>&1 &
IPID=$!
wait $EPID
RC=$?
wait $MPID
kill "$IPID" 2>/dev/null || true
WALL=$(( $(date +%s) - START ))

{
  echo "== engine exit $RC, wall ${WALL}s (cold JIT compile of each stage's program unit is part of this wall) =="
  echo
  echo "== split plan / caches / start (stderr) =="
  grep -E "layer split|layers [0-9]+-|expert cache|resident expert|prompt path|window up to|VRAM free|strata serve:|device table|CUDA[01]|native pack|mmap" "$ERR" | head -40
  echo
  echo "== the requests' own numbers (stdout) =="
  grep -E "^READY|^DONE|^ERR" "$OUT"
  echo
  echo "== the served-path breakdown per request (stderr) =="
  grep -E "prompt .* tokens|expert cache hit rate|expert tiers" "$ERR" | head -20
  echo
  echo "== decode timing tail =="
  grep -E "decode timing|GPU stages|prefill" "$ERR" | tail -20
  echo
  echo "== peaks (m6_monitor_tree.py) =="
  cat "$MONLOG"
  echo
  echo "== disk reads during the run (23_io_watch.py: global device + this run's 'strata' processes) =="
  head -1 "$I2/$TAG-io.csv"
  tail -1 "$I2/$TAG-io.csv"
  echo "   (first -> last: subtract for the totals)"
} >> "$LOG" 2>&1
echo "=== $TAG done: exit $RC, wall ${WALL}s, log $LOG ==="
