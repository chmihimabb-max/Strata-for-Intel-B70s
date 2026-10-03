#!/usr/bin/env bash
# M6c: the oracle differential for the 256K needle.  I2's launch script (i2/10_llama_launch.sh) at
# --max-context 262144, with two changes and both of them forced:
#   1. a quantised KV cache crashes this fork at 256K -- `-ctk q8_0 -ctv q8_0` trips
#      `GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr) failed`
#      (src/models/qwen4exp.cpp:555, build_attn_qsa): the rotated-KV path is not implemented for QSA, so
#      the KV stays at the default (f16) exactly as I2 ran it;
#   2. f16 KV at 262,144 cells is ~6.35 GB beside 53.7 GiB of GPU experts (I2's 4K run left 5,745 MiB
#      free), so four MoE layers' experts stay on the CPU (-ncmoe 4).  I2's 4K row used -ncmoe 0; the
#      difference is stated in the write-up.
#
# usage: bash m6c/m6c_oracle_needle.sh <tag>
R=/home/michael/strata-xpu
TAG=${1:?tag}
PORT=${PORT:-58244}
CTX=${CTX:-262144}
NCMOE=${NCMOE:-4}
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
BIN=$HOME/llama.cpp-qwen4-exp/build/bin/llama-server
D=$R/m6c/oracle
mkdir -p "$D"
LOG=$D/$TAG-server.log

for q in $(pgrep -f "llama-server -m"); do kill -9 "$q" 2>/dev/null || true; done
sleep 2
source /opt/intel/oneapi/setvars.sh > "$D/$TAG-setvars.txt" 2>&1
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/umf/1.1/lib:${LD_LIBRARY_PATH:-}"
export GGML_SCHED_DEBUG=0

{
  echo "M6c oracle 256K needle  tag=$TAG  $(date -Is)"
  echo "tree (read-only): $BIN  ($(git -C "$HOME/llama.cpp-qwen4-exp" log --oneline -1 2>/dev/null))"
  echo "CMD: $BIN -m $SH1 -ngl 99 -ncmoe $NCMOE -ot per_layer_token_embd=CPU -c $CTX -fa on -t 18 -ts 0.50,0.50 --parallel 1 -ub 2048 --host 127.0.0.1 --port $PORT --jinja"
  echo "=============================================================="
} > "$LOG"
cd "$HOME/llama.cpp-qwen4-exp" || exit 1
setsid nohup "$BIN" -m "$SH1" -ngl 99 -ncmoe "$NCMOE" -ot "per_layer_token_embd=CPU" -c "$CTX" -fa on -t 18 \
  -ts 0.50,0.50 --parallel 1 -ub 2048 --host 127.0.0.1 --port "$PORT" --jinja \
  >> "$LOG" 2>&1 &
echo $! > "$D/$TAG-server.pid"
echo "launched pid $(cat "$D/$TAG-server.pid") port $PORT ctx $CTX (KV at its default f16, -ncmoe $NCMOE); log $LOG"
