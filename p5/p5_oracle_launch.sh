#!/usr/bin/env bash
# P5: launch the independent oracle -- llama.cpp-SYCL (qwen4exp fork) on the SAME GSQ-RCO IQ3_S file the engine
# maps in place.  This is I2's harness (i2/10_llama_launch.sh) with the context passed in, because P5 needs the
# oracle at the SAME max-context as each engine arm (I2 only ran ctx 4096).
#
#   usage: bash p5/p5_oracle_launch.sh <tag> <ctx> [port] [ncmoe] [ts] [ub]
#   READ-ONLY use of ~/llama.cpp-qwen4-exp: nothing is written into that tree.
R=/home/michael/strata-xpu
D=$R/p5/oracle
TAG=${1:?tag}; CTX=${2:?ctx}
PORT=${3:-58210}; NCMOE=${4:-0}; TS=${5:-0.50,0.50}; UB=${6:-2048}
T=${T:-18}
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
BIN=$HOME/llama.cpp-qwen4-exp/build/bin/llama-server
LOG=$D/$TAG-server.log
mkdir -p "$D"

for q in $(pgrep -f "llama-server -m"); do kill -9 "$q" 2>/dev/null || true; done
for q in $(pgrep -f "build-sycl/strata"); do echo "p5_oracle_launch: REFUSING: a strata engine is running (pid $q)"; exit 3; done
sleep 2

source /opt/intel/oneapi/setvars.sh > "$D/$TAG-setvars.txt" 2>&1
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
export GGML_SCHED_DEBUG=0

{
  echo "P5 llama.cpp oracle  tag=$TAG  $(date -Is)"
  echo "tree (read-only): $BIN   ($(git -C "$HOME/llama.cpp-qwen4-exp" log --oneline -1 2>/dev/null))"
  echo "CMD: $BIN -m $SH1 -ngl 99 -ncmoe $NCMOE -ot per_layer_token_embd=CPU -c $CTX -fa on -t $T -ts $TS --parallel 1 -ub $UB --host 127.0.0.1 --port $PORT --jinja"
  echo "== device check BEFORE the oracle =="
  pgrep -a -f "build-sycl/strata|llama-server -m" || echo "   nothing of ours running"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -12
  echo "=============================================================="
} > "$LOG"

cd "$HOME/llama.cpp-qwen4-exp" || exit 1
setsid nohup "$BIN" \
  -m "$SH1" -ngl 99 -ncmoe "$NCMOE" -ot "per_layer_token_embd=CPU" \
  -c "$CTX" -fa on -t "$T" -ts "$TS" --parallel 1 -ub "$UB" \
  --host 127.0.0.1 --port "$PORT" --jinja \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$D/$TAG-server.pid"

START=$(date +%s)
while :; do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
    echo "HEALTHY after $(( $(date +%s) - START ))s"; READY=1; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "ORACLE EXITED before health -- see $LOG"; READY=0; break; fi
  if [ $(( $(date +%s) - START )) -gt 900 ]; then echo "TIMEOUT waiting for health"; READY=0; break; fi
  sleep 5
done
{
  echo "== placement / size lines =="
  grep -E "load_tensors|assigned to device|KV self size|compute buffer|SYCL[0-9] model buffer|CPU buffer size|n_ctx" "$LOG" | head -30
  echo "== errors =="
  grep -iE "error|abort|assert|out of resources|failed|not enough" "$LOG" | head -15 || true
} >> "$LOG"
echo "oracle tag=$TAG pid=$PID ctx=$CTX port=$PORT ready=$READY log=$LOG"
[ "$READY" = 1 ] || exit 4
