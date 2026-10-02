#!/usr/bin/env bash
# I2: launch the llama.cpp-SYCL (qwen4exp fork) oracle on the SAME IQ3_S file the Strata pack maps in place.
# Read-only use of ~/llama.cpp-qwen4-exp (sources untouched); nothing here writes into that tree.
#
#   usage: bash i2/10_llama_launch.sh <tag> [port] [ctx] [ncmoe] [ts] [ub]
#   defaults:                          58210  4096  0       0.50,0.50 2048
set -o pipefail
R=/home/michael/strata-xpu
I2=$R/strata/i2
TAG=${1:?tag}
PORT=${2:-58210}
CTX=${3:-4096}
NCMOE=${4:-0}
TS=${5:-0.50,0.50}
UB=${6:-2048}
T=${T:-18}
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
BIN=$HOME/llama.cpp-qwen4-exp/build/bin/llama-server
LOG=$I2/$TAG-server.log
mkdir -p "$I2"

for q in $(pgrep -f "llama-server -m"); do kill -9 "$q" 2>/dev/null || true; done
sleep 2

# oneAPI runtime (libiomp5.so / libsycl.so) - the build is not RPATH'd to it
source /opt/intel/oneapi/setvars.sh > "$I2/$TAG-setvars.txt" 2>&1
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
export GGML_SCHED_DEBUG=0
rm -f /tmp/ggml_sched_detail

{
  echo "I2 llama.cpp oracle  tag=$TAG  $(date -Is)"
  echo "tree (read-only): $BIN   ($(git -C "$HOME/llama.cpp-qwen4-exp" log --oneline -1 2>/dev/null))"
  echo "CMD: $BIN -m $SH1 -ngl 99 -ncmoe $NCMOE -ot per_layer_token_embd=CPU -c $CTX -fa on -t $T -ts $TS --parallel 1 -ub $UB --host 127.0.0.1 --port $PORT --jinja"
  echo "=============================================================="
} > "$LOG"

cd "$HOME/llama.cpp-qwen4-exp" || exit 1
setsid nohup "$BIN" \
  -m "$SH1" \
  -ngl 99 \
  -ncmoe "$NCMOE" \
  -ot "per_layer_token_embd=CPU" \
  -c "$CTX" \
  -fa on \
  -t "$T" \
  -ts "$TS" \
  --parallel 1 \
  -ub "$UB" \
  --host 127.0.0.1 --port "$PORT" \
  --jinja \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$I2/$TAG-server.pid"
echo "launched pid $PID port $PORT ctx $CTX ncmoe $NCMOE ts $TS ub $UB; log $LOG"
