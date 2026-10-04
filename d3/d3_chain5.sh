#!/usr/bin/env bash
# D3 (card t_87aa2963): everything left after the record-config census, in priority order.  One engine at a time;
# each arm waits for the previous engine to leave.
#
#   bash d3/d3_chain5.sh        (no argument: the whole remaining sequence)
#
# Blocks, in order:
#   B1  4K  base (record config) and the old per-query scores grid (STRATA_SCORES_MULTI=0)       -> task 2 at 4K
#   B2  32K base / no-multi                                                                       -> task 2 at 32K
#   B3  128K base / no-multi                                                                      -> task 2 at 128K
#   D    the E-6 device plan with the SAME instruments as the census control, 4K then 32K         -> task 3 ceiling
#   S    the #267 stall rig on this binary (with a second ask: the session must die cleanly)      -> the guarantee
#   E1   the clean top-k pair at ONE max-context: maxctx 4096 + STRATA_TOPK_OLD=1 (memory, forced)
#   E2   the record's max-context 262144 against the arm's own CTX at 4K, uninstrumented
#   D128 / E3  the two 9-minute arms, last: E-6 at 128K, and the memory top-k at maxctx 131072 (the clean 128K pair)
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain5.log
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

echo "###### d3 chain5 started $(date -Is)" >> "$LOG"

for CTX in 4096 32768; do
  run_arm "d3-base-$CTX" "$CTX" 256 --maxctx "$MAXCTX"
  run_arm "d3-nomulti-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --env STRATA_SCORES_MULTI=0
done

run_arm "d3-devplan-4096" 4096 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1 --env STRATA_VERIFY_DEVICE_PLAN=1
run_arm "d3-devplan-32768" 32768 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1 --env STRATA_VERIFY_DEVICE_PLAN=1

echo "== the #267 stall rig $(date -Is)" >> "$LOG"
for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
bash "$SRC/d3/d3_stall.sh" d3-stall-4096 "$SRC/build-sycl/strata" 16 4096 1 1 >> "$LOG" 2>&1
echo "== the stall rig finished $(date -Is)" >> "$LOG"

run_arm "d3-oldtopkhist-4096" 4096 256 --graph 0 --hist 1 --histwindows 8 --profile 1 --env STRATA_TOPK_OLD=1
run_arm "d3-basearmctx-4096" 4096 256

for CTX in 131072; do
  run_arm "d3-base-$CTX" "$CTX" 256 --maxctx "$MAXCTX"
  run_arm "d3-nomulti-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --env STRATA_SCORES_MULTI=0
done
run_arm "d3-devplan-131072" 131072 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1 --env STRATA_VERIFY_DEVICE_PLAN=1
run_arm "d3-oldtopkhist-131072" 131072 256 --env STRATA_TOPK_OLD=1

echo "###### d3 chain5 done $(date -Is)" >> "$LOG"
tail -3 "$LOG"
