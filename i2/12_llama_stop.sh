#!/usr/bin/env bash
# I2: stop the oracle server (by pid file, then by name) and prove both cards are free again.
#   usage: bash i2/12_llama_stop.sh <tag>
R=/home/michael/strata-xpu
I2=$R/strata/i2
TAG=${1:?tag}
PIDF=$I2/$TAG-server.pid
if [ -f "$PIDF" ]; then
  PID=$(cat "$PIDF")
  kill -INT "$PID" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$PID" 2>/dev/null || true
fi
for q in $(pgrep -f "build/bin/llama-server"); do kill -9 "$q" 2>/dev/null || true; done
sleep 3
echo "== after stop: who holds a card =="
pgrep -a -f "build/bin/llama-server|build-sycl/strata" || echo "   no engine process"
/usr/bin/python3 "$R/scripts/m6_occupancy.py"
