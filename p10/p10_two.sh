#!/usr/bin/env bash
# P10 (t_1d052912) concurrency rig: two single-card `serve.server` instances against one two-card instance.
#
#   bash p10/p10_two.sh <TAG> <2i|1x2> <POOL_A> <POOL_B> [CTX] [MAXNEW] [AFFINITY]
#
#   2i  two serve.server instances, ZE_AFFINITY_MASK=0 and =1, own config/log/domain, --pool-workers POOL_A/POOL_B
#   1x2 one serve.server over BOTH cards (mask unset, --layer-split auto), two requests fired at it concurrently
#       (serve/server.py: "one sequence at a time behind a FIFO" - so this arm measures the queue, not a split of
#       a request)
#
# Both clients are launched at the same instant and the wall clock of each request is recorded; the per-request
# prompt/decode numbers are the engines' own.  Per-thread CPU is sampled on each engine (p10_threads.py), VRAM/RSS
# from the driver's fdinfo (scripts/m6_monitor.py), and both instances' mmap reads come from the engines' own
# tier lines (the pack is a shared page cache now - the interference this arm exists to look for).
R=/home/michael/strata-xpu
SRC=$R/strata
set +e

TAG=${1:?tag}; MODE=${2:?mode}; PA=${3:-}; PB=${4:-}; CTX=${5:-32768}; MAXNEW=${6:-256}; AFF=${7:-}; PC=${8:-}
D=$R/p10/runs/$TAG
LOG=$D/log.txt; MK=$D/markers.tsv
mkdir -p "$D"; : > "$LOG"; : > "$MK"
PROMPTFILE=$R/p10/prompts/prompt-ctx$CTX-user.txt
test -r "$PROMPTFILE" || { echo "no text prompt $PROMPTFILE (run p10_prompt_text.py $CTX)"; exit 2; }

echo "=== P10 two-instance arm $TAG mode=$MODE pools=$PA/$PB ctx=$CTX maxnew=$MAXNEW affinity=${AFF:-default} pcache=${PC:-default} $(date -Is)" >> "$LOG"
echo "prompt: $PROMPTFILE ($(wc -c < "$PROMPTFILE") chars)" >> "$LOG"
/usr/bin/python3 "$SRC/p10/p10_two_configs.py" "$TAG" "$CTX" "$PA" "$PB" "$AFF" "$PC" >> "$LOG" 2>&1
pgrep -a -f "build-sycl/strata|serve/server.py" >> "$LOG" 2>&1 || echo "   no engine/server of ours" >> "$LOG"
/usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -2 >> "$LOG"

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1
cd "$SRC" || exit 1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$R/sycl-cache/m6c
export STRATA_DECODE_TIMING=1

SERVERS=()
PORTS=()
case "$MODE" in
  2i)
    for n in a b; do
      port=$([ "$n" = a ] && echo 8101 || echo 8102)
      log=$R/logs/p10-$TAG-$n-server.log
      nohup /usr/bin/python3 -m serve.server --engine strata --config "$D/cfg-$n.json" \
          --port "$port" --api-monitor >> "$log" 2>&1 &
      SERVERS+=($!); PORTS+=("$port")
      echo "started server $n pid $! port $port log $log" >> "$LOG"
    done ;;
  1x2)
    log=$R/logs/p10-$TAG-1x2-server.log
    nohup /usr/bin/python3 -m serve.server --engine strata --config "$D/cfg-1x2.json" \
        --port 8103 --api-monitor >> "$log" 2>&1 &
    SERVERS+=($!); PORTS+=("8103")
    echo "started server 1x2 pid $! port 8103 log $log" >> "$LOG" ;;
  *) echo "mode must be 2i or 1x2"; exit 2 ;;
esac
printf '%s\tload_start\n' "$(date +%s.%N)" >> "$MK"
# the SSD tier under this arm (the pack's device), for the interference question
/usr/bin/python3 "$SRC/p10/p10_disk.py" nvme0n1 "$D/disk.csv" 2 --stopfile "$D/threads.stop" >> "$LOG" 2>&1 &
DSK=$!

# readiness: /v1/models answers only when the engine has loaded
for i in $(seq 1 1800); do
  ok=1
  for p in "${PORTS[@]}"; do
    curl -s -m 5 "http://127.0.0.1:$p/v1/models" | grep -q '"object"' || ok=0
  done
  [ "$ok" = 1 ] && break
  sleep 1
done
echo "== ready after $i s ($(date -Is)) ==" >> "$LOG"
printf '%s\tready\n' "$(date +%s.%N)" >> "$MK"

# the engines: children of the servers, each with its own ZE_AFFINITY_MASK
EPIDS=()
for s in "${SERVERS[@]}"; do
  for t in $(seq 1 30); do
    pid=$(pgrep -P "$s" -x strata | head -1)
    [ -n "$pid" ] && break
    sleep 1
  done
  EPIDS+=("${pid:-0}")
done
for idx in "${!EPIDS[@]}"; do
  e=${EPIDS[$idx]}
  mask=$(tr '\0' '\n' < /proc/$e/environ 2>/dev/null | grep '^ZE_AFFINITY_MASK=' || echo "ZE_AFFINITY_MASK=<unset>")
  echo "engine $idx pid $e  $mask" >> "$LOG"
  /usr/bin/python3 "$SRC/p10/p10_threads.py" "$e" "$D/threads-$idx.csv" 1.0 --stopfile "$D/threads.stop" >> "$LOG" 2>&1 &
  eval "SAMP$idx=$!"
  /usr/bin/python3 "$R/scripts/m6_monitor.py" "$e" "$D/monitor-$idx.csv" 2.0 "$TAG-$idx" >> "$LOG" 2>&1 &
  eval "MON$idx=$!"
  ( pidstat -t -p "$e" 1 60 > "$D/pidstat-$idx.txt" 2>&1 ) &
  eval "PS$idx=$!"
  ( top -b -H -n 60 -d 1 -p "$e" > "$D/top-H-$idx.txt" 2>&1 ) &
  eval "TS$idx=$!"
done
sleep 2

# fire every request at the same instant.  In 1x2 mode BOTH clients go to the one two-card server (it serves
# one sequence at a time, so what is measured is the queue); in 2i mode one client per instance.
CL=()
case "$MODE" in
  2i)  CL=("0:8101" "1:8102") ;;
  1x2) CL=("0:8103" "1:8103") ;;
esac
printf '%s\ttask_start\n' "$(date +%s.%N)" >> "$MK"
CPIDS=()
for spec in "${CL[@]}"; do
  cidx=${spec%%:*}; cport=${spec##*:}
  /usr/bin/python3 "$SRC/p10/p10_client.py" --port "$cport" --prompt-file "$PROMPTFILE" \
      --max-new "$MAXNEW" --out "$D/resp-$cidx.json" --tag "$TAG-$cidx" --label concurrent >> "$LOG" 2>&1 &
  CPIDS+=($!)
done
echo "== fired ${#CPIDS[@]} concurrent requests at $(date -Is) ==" >> "$LOG"
for c in "${CPIDS[@]}"; do wait "$c"; done
printf '%s\trequest_done\n' "$(date +%s.%N)" >> "$MK"
echo "== all clients returned $(date -Is) ==" >> "$LOG"

sleep 3
touch "$D/threads.stop"
KPIDS=()
for v in "${SAMP0:-}" "${SAMP1:-}" "${MON0:-}" "${MON1:-}" "${PS0:-}" "${PS1:-}" "${TS0:-}" "${TS1:-}" "${DSK:-}"; do
  [ -n "$v" ] && [ "$v" != 0 ] && KPIDS+=("$v")
done
[ ${#KPIDS[@]} -gt 0 ] && kill "${KPIDS[@]}" 2>/dev/null
# NEVER a bare `wait` here: the servers are background jobs of THIS shell too, and a bare wait would block
# until they exit - which is exactly backwards (the servers are stopped below).  Measured: the 2i-32k-9 arm
# hung here for 4 minutes with both servers still alive.
for v in "${KPIDS[@]}"; do wait "$v" 2>/dev/null; done

# stop the servers the way p6_stop.sh does: SIGTERM (the server QUITs its engine), then reap the engines
for s in "${SERVERS[@]}"; do kill -TERM "$s" 2>/dev/null; done
for i in $(seq 1 60); do
  alive=0
  for s in "${SERVERS[@]}"; do kill -0 "$s" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  sleep 2
done
# the servers take the Ctrl+C path on SIGTERM and QUIT their engines; if one is still there (a stuck engine
# teardown), match it by its own config path and then the engines by name - both anchored, so this rig's own
# shell cannot match
if pgrep -f "serve\.server --engine strata --config $D/cfg" >/dev/null; then
  echo "== servers survived SIGTERM; escalating by config path ==" >> "$LOG"
  pkill -TERM -f "serve\.server --engine strata --config $D/cfg" 2>/dev/null
  sleep 10
  pkill -KILL -f "serve\.server --engine strata --config $D/cfg" 2>/dev/null
fi
pgrep -x strata >/dev/null && { pkill -INT -x strata; sleep 10; pgrep -x strata >/dev/null && pkill -KILL -x strata; }
sleep 2
printf '%s\tservers_stopped\n' "$(date +%s.%N)" >> "$MK"

{
  echo "== the engines' own numbers =="
  for n in "${!PORTS[@]}"; do
    case "$MODE" in
      2i) el=$R/logs/p10-$TAG-$([ "$n" = 0 ] && echo a || echo b)-engine.log ;;
      *)  el=$R/logs/p10-$TAG-1x2-engine.log ;;
    esac
    echo "--- instance $n (port ${PORTS[$n]}) engine log $el"
    grep -E "expert cache auto|expert cache [0-9]+ slots|pre-filled|layer split auto|of the experts resident|VRAM free with everything loaded|expert-pool workers|strata decode timing|INFO context=|strata serve: prompt |strata serve: expert tiers|strata serve: decode expert cache|strata serve: KV streaming" "$el" 2>/dev/null | tail -12
  done
  echo "== the clients =="
  for n in "${!PORTS[@]}"; do
    echo "--- client $n:"; cat "$D/resp-$n.json.timing.json" 2>/dev/null
  done
  echo "== this arm's report =="
  /usr/bin/python3 "$SRC/p10/p10_two_report.py" "$D" 2>&1 | tail -6
  for n in "${!PORTS[@]}"; do
    /usr/bin/python3 "$SRC/p10/p10_cpu_report.py" "$D" "$n" 2>&1 | head -7
    /usr/bin/python3 "$SRC/p10/p10_top_evidence.py" "$D" "$n" 2>&1 | head -4
  done
  grep -h "^\[disk\]" "$LOG" 2>/dev/null | tail -1
  echo "== what is left =="
  pgrep -a -x strata || echo "   no engine"
  pgrep -a -f "^/usr/bin/python3 -m serve\.server" || echo "   no server"
  ss -ltn 2>/dev/null | grep -E ":(8101|8102|8103) " || echo "   ports 8101/8102/8103 free"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -2
} >> "$LOG" 2>&1
echo "=== P10 arm $TAG ($MODE) done $(date -Is): $LOG"
