#!/usr/bin/env bash
# P6 (card t_5dfc11a3): start the resident Strata server + dashboard on port 8099.
#
#   bash /home/michael/strata-xpu/p6/p6_start.sh
#
# Both B70s ("gpu": [0,1] -> the server appends --layer-split auto), ZE_AFFINITY_MASK unset (PLAN 11 U11).
# Config of record: /home/michael/strata-xpu/strata/strata-sycl-iq3s.json (--max-context 262144, --kv-resident 32768).
# While this runs it holds BOTH GPUs: stop it (p6_stop.sh) before any other GPU work.
# NOTE: no `set -u` here on purpose - oneAPI's setvars.sh trips nounset and kills the sourcing shell (measured).
R=/home/michael/strata-xpu
SRC=$R/strata
PORT=8099
LOG=$R/logs/p6-server.log
ENGLOG=$R/logs/i1-iq3s-engine.log
PIDFILE=$SRC/p6/server.pid
cd "$SRC" || exit 1

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use - run p6_stop.sh first (or 'ss -ltnp | grep $PORT')"
  exit 1
fi
if pgrep -x strata >/dev/null; then
  echo "an engine (process 'strata') is already running - one engine at a time (PLAN 9 rule 1)"
  exit 1
fi

unset ZE_AFFINITY_MASK
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=$R/sycl-cache/m6c        # the warm program cache every measurement used

{
  echo "=============================================================="
  echo "P6 server start $(date -Is)   pid $$   port $PORT"
  echo "engine argv: $(/usr/bin/python3 $SRC/p6/p6_config_check.py | grep 'engine argv')"
} >> "$LOG"

nohup setsid /usr/bin/python3 -m serve.server --engine strata \
  --config strata-sycl-iq3s.json --port "$PORT" --api-monitor >> "$LOG" 2>&1 &
sleep 2
PID=$(pgrep -f '^/usr/bin/python3 -m serve\.server --engine strata' | head -1)
echo "${PID:-0}" > "$PIDFILE"
echo "launched: server pid ${PID:-unknown}, server log $LOG, engine log $ENGLOG"
echo "wait for the engine to load (2x ~83 GB of GGUF, ~48 GiB RSS): /v1/models answers when READY"
