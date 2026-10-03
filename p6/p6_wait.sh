#!/usr/bin/env bash
# P6: wait until the resident server answers /v1/models, or give up. Prints the tail of both logs as it goes.
#   bash /home/michael/strata-xpu/p6/p6_wait.sh [seconds]
R=/home/michael/strata-xpu
PORT=8099
LIMIT=${1:-420}
START=$(date +%s)
while :; do
  NOW=$(date +%s); EL=$((NOW - START))
  if [ "$EL" -gt "$LIMIT" ]; then
    echo "=== gave up after ${EL}s: /v1/models did not answer ==="
    tail -5 "$R/logs/p6-server.log"
    exit 1
  fi
  BODY=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  if echo "$BODY" | grep -q '"object"'; then
    echo "=== READY after ${EL}s ==="
    echo "$BODY"
    exit 0
  fi
  echo "-- ${EL}s: $(tail -1 "$R/logs/p6-server.log" 2>/dev/null | cut -c1-140)"
  sleep 10
done
