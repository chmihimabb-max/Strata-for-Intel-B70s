#!/usr/bin/env bash
# D1 control: the SAME post-change binary with STRATA_SYCL_GRAPH=0, i.e. the closure path.  Two things this
# buys: (a) the off-switch really is the pre-change behaviour (the arms must read what the pre-change binary's
# arms read), and (b) it prices the new default honestly against a same-binary control, which is the comparison
# the tok/s delta is quoted from.
#   bash d1/d1_closure.sh [4096|32768|131072|all]
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

which=${1:-all}
echo "###### d1 closure control $which started $(date -Is)" >> "$LOG"
for CTX in 4096 32768 131072; do
  [ "$which" = all ] || [ "$which" = "$CTX" ] || continue
  run_arm "d1-guard-${CTX}-closure" "$CTX" 256 --bin "$SRC/d1/strata-after" --graph 0
done
echo "###### d1 closure control $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
