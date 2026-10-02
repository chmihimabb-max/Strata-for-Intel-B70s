#!/usr/bin/env bash
# I2: wait for the oracle server to become healthy, then dump the load/placement evidence.
#   usage: bash i2/11_llama_wait.sh <tag> [timeout_s] [port]
set -o pipefail
R=/home/michael/strata-xpu
I2=$R/strata/i2
TAG=${1:?tag}
TMO=${2:-900}
PORT=${3:-58210}
LOG=$I2/$TAG-server.log
START=$(date +%s)
while :; do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
    echo "HEALTHY after $(( $(date +%s) - START ))s"
    break
  fi
  if ! kill -0 "$(cat "$I2/$TAG-server.pid" 2>/dev/null)" 2>/dev/null; then
    echo "SERVER EXITED before health (see $LOG)"
    break
  fi
  if [ $(( $(date +%s) - START )) -gt "$TMO" ]; then
    echo "TIMEOUT after ${TMO}s"
    break
  fi
  sleep 5
done
echo "== load/placement lines =="
grep -E "load_tensors|assigned to device|CPU buffer size|SYCL[0-9] model buffer|KV self size|compute buffer|CUDA[0-9]|SYCL[0-9]" "$LOG" | head -40
echo "== free VRAM/RAM as the loader reported it =="
grep -E "free|VRAM|host memory" "$LOG" | head -10
echo "== errors =="
grep -iE "error|abort|assert|out of resources|failed" "$LOG" | head -20 || true
tail -25 "$LOG"
