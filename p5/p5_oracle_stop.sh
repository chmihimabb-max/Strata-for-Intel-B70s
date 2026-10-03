#!/usr/bin/env bash
# P5: stop the oracle and re-prove both cards are free.
#   usage: bash p5/p5_oracle_stop.sh <tag>
R=/home/michael/strata-xpu
D=$R/p5/oracle
TAG=${1:?tag}
PIDF=$D/$TAG-server.pid
if [ -r "$PIDF" ]; then
  P=$(cat "$PIDF")
  kill -INT "$P" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
  kill -9 "$P" 2>/dev/null
fi
for q in $(pgrep -f "llama-server -m"); do kill -9 "$q" 2>/dev/null; done
sleep 2
echo "== after stop: who holds the cards =="
pgrep -a -f "llama-server -m|build-sycl/strata" || echo "   nothing of ours running"
/usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -6
