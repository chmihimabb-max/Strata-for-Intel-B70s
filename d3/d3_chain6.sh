#!/usr/bin/env bash
# D3 (card t_87aa2963): the two follow-ups the E-6 result needs.
#
#   bash d3/d3_chain6.sh
#
# 1) E-6 on the GRAPH path, uninstrumented (the arms chain5 measured E-6 with the census instruments on the
#    closure path; the served configuration runs the graph path, so the lever needs its own number there).
# 2) the #267 stall rig with E-6 ON: with the device planning its own group, the withheld last-layer flag is not
#    what the device is waiting for (`wait_flag_ge_or` returns on the device-written skip word), so this arm both
#    re-proves the release path under E-6 and shows the interaction.
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain6.log
mkdir -p "$RUN"
MAXCTX=262144

run_arm() {
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped $(date -Is)" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/d3/d3_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is)" >> "$LOG"
  sleep 5
}

echo "###### d3 chain6 started $(date -Is)" >> "$LOG"
run_arm "d3-devplan-noh-4096" 4096 256 --maxctx "$MAXCTX" --env STRATA_VERIFY_DEVICE_PLAN=1
run_arm "d3-devplan-noh-32768" 32768 256 --maxctx "$MAXCTX" --env STRATA_VERIFY_DEVICE_PLAN=1
for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
STRATA_VERIFY_DEVICE_PLAN=1 bash "$SRC/d3/d3_stall.sh" d3-stall-e6-4096 "$SRC/build-sycl/strata" 16 4096 1 1 >> "$LOG" 2>&1
echo "== the E-6 stall arm finished $(date -Is)" >> "$LOG"
echo "###### d3 chain6 done $(date -Is)" >> "$LOG"
tail -3 "$LOG"
