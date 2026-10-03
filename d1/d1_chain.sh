#!/usr/bin/env bash
# D1 group A: the graph-path default's guard evidence.
#   For each length 4K/32K/128K: the PRE-CHANGE binary at its own default (the closure path, since
#   STRATA_SYCL_GRAPH was opt-in) against the POST-CHANGE binary at its default (the graph path).  Same
#   prompt, same max-new, same config of record otherwise -> the token ids must be byte-identical and the
#   window must be the cheaper one.
#   bash d1/d1_chain.sh guard
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

group=${1:?group: guard|spec|variants|hist|all}
echo "###### d1 chain group $group started $(date -Is)" >> "$LOG"

if [ "$group" = guard ] || [ "$group" = all ]; then
  run_arm d1-guard-4k-before   4096  256 --bin "$SRC/d1/strata-before"
  run_arm d1-guard-4k-after    4096  256 --bin "$SRC/d1/strata-after"
  run_arm d1-guard-32k-before  32768 256 --bin "$SRC/d1/strata-before"
  run_arm d1-guard-32k-after   32768 256 --bin "$SRC/d1/strata-after"
  run_arm d1-guard-128k-before 131072 256 --bin "$SRC/d1/strata-before"
  run_arm d1-guard-128k-after  131072 256 --bin "$SRC/d1/strata-after"
fi
echo "###### d1 chain group $group done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
