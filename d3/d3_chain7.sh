#!/usr/bin/env bash
# D3 (card t_87aa2963): the warm re-runs D2's JIT lesson demands, and the last census gap.
#
#   bash d3/d3_chain7.sh
#
# D2's rule: "a newly instantiated kernel's JIT lands INSIDE the first measured window ... run the arm TWICE and
# quote the warm one".  E-6 instantiates kernels this session had never run (resident_plan, wait_flag_ge_or,
# copy_i32_unless), and the graph-path E-6 arms are the first place they run with the graph path - so the two
# 4K/32K E-6 graph-path arms are re-run here and the warm pair is quoted.
# The third arm closes the one census gap: the top-k dispatch pair at 128K with the census instruments on the SAME
# max-context (maxctx 131072, STRATA_TOPK_OLD=1 = the memory kernel) against d3-hist-131072 (same maxctx, the
# register kernel).
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain7.log
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

echo "###### d3 chain7 started $(date -Is)" >> "$LOG"
run_arm "d3-devplan-noh2-4096" 4096 256 --maxctx "$MAXCTX" --env STRATA_VERIFY_DEVICE_PLAN=1
run_arm "d3-devplan-noh2-32768" 32768 256 --maxctx "$MAXCTX" --env STRATA_VERIFY_DEVICE_PLAN=1
run_arm "d3-oldtopkhistc-131072" 131072 256 --graph 0 --hist 1 --histwindows 8 --profile 1 --env STRATA_TOPK_OLD=1
echo "###### d3 chain7 done $(date -Is)" >> "$LOG"
tail -3 "$LOG"
