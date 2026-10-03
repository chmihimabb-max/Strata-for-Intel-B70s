#!/usr/bin/env bash
# P6 (card t_5dfc11a3): stop the resident Strata server and its engine, and show that both cards are free again.
#
#   bash /home/michael/strata-xpu/p6/p6_stop.sh
#
# SIGTERM takes the server's Ctrl+C path: it sends QUIT to the engine and waits for it to free its VRAM/RAM.
# The patterns are ANCHORED and the engine is matched by process name: `pgrep -f <pattern>` also matches the
# command line of whatever shell runs this script (p6_start.sh's guard tripped on its own caller, measured).
R=/home/michael/strata-xpu
SRE='^/usr/bin/python3 -m serve\.server --engine strata'
PIDFILE=$R/strata/p6/server.pid

PIDS=$(pgrep -f "$SRE" || true)
if [ -z "$PIDS" ]; then
  echo "no server process matches '$SRE'"
else
  echo "stopping server pid(s): $PIDS"
  kill -TERM $PIDS          # the server's SIGTERM handler takes Ctrl+C's path and QUITs the engine
  for i in $(seq 1 30); do
    sleep 2
    pgrep -f "$SRE" >/dev/null || { echo "server gone after $((i*2))s"; break; }
  done
fi

if pgrep -x strata >/dev/null; then
  echo "the engine (process 'strata') survived the server; sending SIGINT, then SIGKILL if it is still there"
  pkill -INT -x strata 2>/dev/null || true
  sleep 10
  pgrep -x strata >/dev/null && pkill -KILL -x strata 2>/dev/null || true
fi

rm -f "$PIDFILE"
sleep 3
echo "--- what is left ---"
pgrep -a -x strata || echo "   no engine"
pgrep -f "$SRE" || echo "   no server"
ss -ltn 2>/dev/null | grep ":8099 " || echo "   port 8099 free"
/usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
