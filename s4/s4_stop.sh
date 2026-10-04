#!/usr/bin/env bash
# S4 (card t_30d9ccfb): stop the S4 server and its engine, and show that both cards are free again.
# Same path as p6_stop.sh: SIGTERM takes the server's Ctrl+C path and it QUITs the engine; the patterns are
# anchored so that `pgrep -f` cannot match the command line of the shell that runs this script.
#
#   bash s4/s4_stop.sh
R=/home/michael/strata-xpu
SRE='^/usr/bin/python3 -m serve\.server --engine strata'

PIDS=$(pgrep -f "$SRE" || true)
if [ -z "$PIDS" ]; then
  echo "no server process matches '$SRE'"
else
  echo "stopping server pid(s): $PIDS"
  kill -TERM $PIDS
  for i in $(seq 1 30); do
    sleep 2
    if ! pgrep -f "$SRE" >/dev/null; then echo "server gone after $((i * 2))s"; break; fi
  done
fi
if pgrep -f "$SRE" >/dev/null; then
  echo "the server survived SIGTERM; escalating to SIGKILL"
  pkill -KILL -f "$SRE" || true
fi
if pgrep -x strata >/dev/null; then
  echo "the engine (process 'strata') is still there; SIGINT, then SIGKILL"
  pkill -INT -x strata || true
  sleep 10
  if pgrep -x strata >/dev/null; then pkill -KILL -x strata || true; fi
fi
sleep 3
echo "--- what is left ---"
pgrep -a -x strata || echo "   no engine"
pgrep -f "$SRE" || echo "   no server"
ss -ltn 2>/dev/null | grep ":8099 " || echo "   port 8099 free"
/usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
