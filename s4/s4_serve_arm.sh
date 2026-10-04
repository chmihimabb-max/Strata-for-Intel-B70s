#!/usr/bin/env bash
# S4 (card t_30d9ccfb): the served-path arms.  ONE resident serve.server (the config of record, or the config of
# record plus an --extra argument list), then a list of real /v1/chat/completions fired at it in the order given.
# The run stops at the first failed request (the engine dies on the one that fails, so the rest could not run).
#
#   bash s4/s4_serve_arm.sh <TAG> [REQ...] [--extra "ARG ARG"] [--ctx N]
#       REQ = <prompt_tokens>:<max_new>      e.g.  4096:16 16384:16 32768:16
#
# Everything the arm produced lands in s4/runs/<TAG>/ :
#   cfg.json           this arm's config (config of record verbatim but log/port/model_name)
#   log.txt            the run log: the server's effective engine argv, the timeline, the summary
#   prompt-ctx<N>.txt  the text prompt that went through the server (the server tokenizes it itself)
#   resp-ctx<N>.json   the raw response body (or .error)
#   resp-ctx<N>.json.timing.json
#   engine-ctx<N>.txt  the engine-log lines this request produced (its own stats/prompt lines, and any error)
#   server-ctx<N>.txt  the server-log lines this request produced (HTTP status, 400/503 bodies)
#   argv.txt           the server's engine argv + its sha256
R=/home/michael/strata-xpu
SRC=$R/strata
set +e
PORT=8099
SRE='^/usr/bin/python3 -m serve\.server --engine strata'

TAG=${1:?tag}; shift
REQS=(); EXTRA=""; CTX=0
while [ $# -gt 0 ]; do
  case "$1" in
    --extra) EXTRA=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    *) REQS+=("$1"); shift ;;
  esac
done
[ ${#REQS[@]} -gt 0 ] || { echo "no requests given"; exit 2; }

D=$SRC/s4/runs/$TAG
SL=$R/logs/s4-$TAG-server.log
mkdir -p "$D"; : > "$SL"
CACHEDIR=$R/sycl-cache/m6c

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT in use"; exit 1; fi
if pgrep -x strata >/dev/null; then echo "an engine is already running"; exit 1; fi

{
  echo "=== S4 arm $TAG  $(date -Is)"
  echo "HEAD        : $(cd "$SRC" && git log --oneline -1)"
  echo "engine      : $(stat -c '%y %s bytes' "$SRC/build-sycl/strata")  md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)"
  echo "requests    : ${REQS[*]}"
  echo "extra args  : ${EXTRA:-<none>}"
  echo "-- config --"
  /usr/bin/python3 "$SRC/s4/s4_cfg.py" "$TAG" ${EXTRA:+--extra "$EXTRA"} ${CTX:+--ctx "$CTX"} | tee "$D/argv.txt"
  echo "-- device check BEFORE --"
  pgrep -a -x strata || echo "   no engine"
  pgrep -f "$SRE" || echo "   no server"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
  cat /proc/loadavg; free -g | head -2
} >> "$D/log.txt" 2>&1

unset ZE_AFFINITY_MASK
source /opt/intel/oneapi/setvars.sh >> "$D/log.txt" 2>&1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$CACHEDIR
export STRATA_DECODE_TIMING=1
cd "$SRC" || exit 1

echo "== starting the server $(date -Is) ==" >> "$D/log.txt"
nohup setsid /usr/bin/python3 -m serve.server --engine strata --config "$D/cfg.json" \
    --port "$PORT" --api-monitor >> "$SL" 2>&1 &
sleep 3
SPID=$(pgrep -f "$SRE" | head -1)
ENGLOG=$(/usr/bin/python3 "$SRC/s4/s4_engine_log.py" "$D/cfg.json")
echo "server pid ${SPID:-unknown}; server log $SL; engine log $ENGLOG" >> "$D/log.txt"

E_START=$(date +%s)
READY=0
# ready = /v1/models answers with a model list AND the server's own "ready:" banner is in its log (the banner is
# printed when the engine reports loaded; /v1/models alone can answer earlier, as p6_wait.sh's plain '"object"'
# test would accept)
for i in $(seq 1 900); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"object"' && grep -q '^ready: http' "$SL"; then READY=1; break; fi
  pgrep -f "$SRE" >/dev/null || break
  sleep 1
done
ENGLOG=$(/usr/bin/python3 "$SRC/s4/s4_engine_log.py" "$D/cfg.json")
echo "== ready=$READY after $(( $(date +%s) - E_START ))s ($(date -Is)); engine log $ENGLOG ==" >> "$D/log.txt"

RESULT="not run"
for spec in "${REQS[@]}"; do
  N=${spec%%:*}; M=${spec##*:}
  P=$D/prompt-ctx$N.txt
  mkdir -p "$D/req"
  /usr/bin/python3 "$SRC/s4/s4_prompt.py" "$N" "$P" >> "$D/log.txt" 2>&1
  EL0=$(wc -l < "$ENGLOG" 2>/dev/null || echo 0); SL0=$(wc -l < "$SL")
  T0=$(date +%s%N)
  echo "-- request ctx=$N max_new=$M fired $(date -Is) --" >> "$D/log.txt"
  /usr/bin/python3 "$SRC/p10/p10_client.py" --port "$PORT" --prompt-file "$P" --max-new "$M" \
      --out "$D/resp-ctx$N.json" --tag "$TAG-ctx$N" --label s4 >> "$D/log.txt" 2>&1
  RC=$?
  T1=$(date +%s%N)
  tail -n +$((EL0 + 1)) "$ENGLOG" > "$D/engine-ctx$N.txt" 2>/dev/null
  tail -n +$((SL0 + 1)) "$SL" > "$D/server-ctx$N.txt" 2>/dev/null
  printf 'ctx=%s max_new=%s rc=%s wall=%sms %s\n' "$N" "$M" "$RC" "$(( (T1 - T0) / 1000000 ))" "$(date -Is)" >> "$D/markers.tsv"
  if [ "$RC" != 0 ]; then RESULT="FAILED at ctx=$N (rc=$RC)"; break; fi
  RESULT="last ok ctx=$N"
  pgrep -x strata >/dev/null || { RESULT="engine gone after ctx=$N"; break; }
done
echo "== requests done: $RESULT ($(date -Is)) ==" >> "$D/log.txt"

# stop the server the way p6_stop.sh does: SIGTERM takes the server's Ctrl+C path and it QUITs the engine
kill -TERM "$SPID" 2>/dev/null
for i in $(seq 1 45); do pgrep -f "$SRE" >/dev/null || break; sleep 2; done
if pgrep -f "$SRE" >/dev/null; then
  echo "== the server survived SIGTERM (a dead engine); escalating ==" >> "$D/log.txt"
  pkill -TERM -f "$SRE" 2>/dev/null; sleep 5; pkill -KILL -f "$SRE" 2>/dev/null
fi
if pgrep -x strata >/dev/null; then
  echo "== the engine survived the server; SIGINT then SIGKILL ==" >> "$D/log.txt"
  pkill -INT -x strata 2>/dev/null; sleep 10; pgrep -x strata >/dev/null && pkill -KILL -x strata 2>/dev/null
fi
sleep 3

{
  echo "== RESULT: $RESULT =="
  echo "== the request(s), client side =="
  for spec in "${REQS[@]}"; do
    N=${spec%%:*}
    echo "--- ctx=$N ---"
    cat "$D/resp-ctx$N.json.timing.json" 2>/dev/null | head -20
    echo "--- answer (first 300 chars) ---"
    /usr/bin/python3 "$SRC/s4/s4_show.py" "$D/resp-ctx$N.json" 2>/dev/null | head -12
  done
  echo "== the engine's own lines per request =="
  for spec in "${REQS[@]}"; do
    N=${spec%%:*}
    echo "--- ctx=$N ($(wc -l < "$D/engine-ctx$N.txt" 2>/dev/null || echo 0) lines) ---"
    grep -E "strata serve: prompt |strata decode timing|strata serve: expert tiers|strata serve: decode expert cache|strata serve: KV streaming|checkpoint|memcpy failed|OUT_OF_DEVICE|saving a|UR_RESULT|strata serve: layer split:|of the experts resident|expert cache auto|expert cache [0-9]+ slots|VRAM free with everything loaded" "$D/engine-ctx$N.txt" 2>/dev/null | head -40
  done
  echo "== the server's own lines per request =="
  for spec in "${REQS[@]}"; do
    N=${spec%%:*}
    echo "--- ctx=$N ---"
    grep -E "HTTP|400|503|error|Traceback|checkpoint|stopped unexpectedly|prompt" "$D/server-ctx$N.txt" 2>/dev/null | head -25
  done
  echo "== what is left =="
  pgrep -a -x strata || echo "   no engine"
  pgrep -f "$SRE" || echo "   no server"
  ss -ltn 2>/dev/null | grep ":$PORT " || echo "   port $PORT free"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
} >> "$D/log.txt" 2>&1

echo "=== S4 arm $TAG: $RESULT; log $D/log.txt ==="
