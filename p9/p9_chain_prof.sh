#!/usr/bin/env bash
# P9: the profiler chain - the arms that use the stage profiler this card repaired (STRATA_VERIFY_PROFILE=1).
# Same shape as p9_chain.sh (resumable, one engine at a time), run after that one has finished.
#
#   bash p9/p9_chain_prof.sh <group: prof4k|prof32k|prof128k|all>
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/p9/runs
LOG=$R/p9/chain-prof.log
mkdir -p "$RUN"

run_arm() {
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/p9/p9_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is)" >> "$LOG"
  sleep 5
}

group=${1:?group: prof4k|prof32k|prof128k|all}
echo "###### p9 profiler chain group $group started $(date -Is)" >> "$LOG"

if [ "$group" = prof4k ] || [ "$group" = all ]; then
  # the same arm on the NEW binary with the profiler OFF: the control for the new `tail` field and for what the
  # rebuilt binary does to the default path (it should be the baseline to within noise).
  run_arm p9-ctl2-4k 4096 150 --prefill auto --kvres 32768 --graph 0 --profile 0
  run_arm p9-p4k    4096 150 --prefill auto --kvres 32768 --graph 0 --profile 1
  # the two halves of the instrument's own cost, separated: the stamp publishes alone (=2), and the publishes
  # plus the sampler thread (=1).  p9-p4k above ran the FIRST form of the stamp (a __threadfence_system() per
  # stamp) and is kept as the measurement of what the fence costs.
  run_arm p9-p4k4   4096 150 --prefill auto --kvres 32768 --graph 0 --profile 0 --env STRATA_VERIFY_PROFILE=2
  run_arm p9-p4k3   4096 150 --prefill auto --kvres 32768 --graph 0 --profile 1
  # the shipped form: the sampler inside the layer spin loop (no thread), plus the chain-folded counters
  run_arm p9-ctl3-4k 4096 150 --prefill auto --kvres 32768 --graph 0 --profile 0
  run_arm p9-p4k5   4096 150 --prefill auto --kvres 32768 --graph 0 --profile 1
fi
if [ "$group" = prof32k ] || [ "$group" = all ]; then
  run_arm p9-p32k   32768 256 --prefill auto --kvres 32768 --graph 0 --profile 1
fi
if [ "$group" = prof128k ] || [ "$group" = all ]; then
  run_arm p9-p128k  131072 256 --prefill auto --kvres 32768 --graph 0 --profile 1
fi
echo "###### p9 profiler chain group $group done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
