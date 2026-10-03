#!/usr/bin/env bash
# D1 group C: the kernel histogram.  Two arms per length:
#   * graph off (`--graph 0`): each window's kernels are real submissions, so the events carry the device
#     timestamps -> COUNT AND MICROSECONDS per launch site (the census P3 could only count);
#   * graph on (the default): the window that RECORDS gives the census of the window the engine actually runs
#     (counts only - a recorded node has no device timestamp until the graph executes).
#   bash d1/d1_hist.sh <4096|32768|131072|all>
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/d1/runs
LOG=$R/d1/chain.log
mkdir -p "$RUN"

run_arm() {
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped $(date -Is)" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/d1/d1_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is)" >> "$LOG"
  sleep 5
}

which=${1:?4096|32768|131072|all}
echo "###### d1 histogram $which started $(date -Is)" >> "$LOG"
for CTX in 4096 32768 131072; do
  [ "$which" = all ] || [ "$which" = "$CTX" ] || continue
  run_arm "d1-hist-${CTX}-closed" "$CTX" 150 --bin "$SRC/d1/strata-after" --graph 0 --hist 1
  run_arm "d1-hist-${CTX}-graph"  "$CTX" 150 --bin "$SRC/d1/strata-after" --hist 1
done
echo "###### d1 histogram $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
