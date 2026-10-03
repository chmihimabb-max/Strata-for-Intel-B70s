#!/usr/bin/env bash
# P9: the arm chain.  One engine at a time, each arm through p9_run_arm.sh; a run whose log ends with the
# rig's own "done" line is not repeated, so the chain can be re-run to continue after an interruption.
#
#   bash p9/p9_chain.sh <group: abl4k|abl32k|abl128k|all>
#
# The arms answer "what is inside the verify window" by ABLATION with the engine's own switches (each of
# which is a measured A/B of one part of the window), because the chrome device trace cannot see the window
# at all (p9/STATUS-P9.md §1).  Every arm prints the same three things: the decode timing line (verify and
# its parts), the submission composition per stage, and the greedy token-id md5.
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/p9/runs
LOG=$R/p9/chain.log
mkdir -p "$RUN"

run_arm() {   # TAG CTX MAXNEW [extra args...]
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  # wait for the previous engine to be gone (never two engines on the two cards)
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/p9/p9_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is) rc=$?" >> "$LOG"
  sleep 5
}

group=${1:?group: abl4k|abl32k|abl128k|all}
echo "###### p9 chain group $group started $(date -Is)" >> "$LOG"

# ---- 4K: the switch ablations are cheap here, so most of them run at this length
if [ "$group" = abl4k ] || [ "$group" = all ]; then
  run_arm p9-4k-dec0    4096 150 --prefill auto --kvres 32768 --graph 0 --env STRATA_DEC_BATCH=0
  run_arm p9-4k-hcplain 4096 150 --prefill auto --kvres 32768 --graph 0 --env STRATA_HC_SPLIT=0
  run_arm p9-4k-graph   4096 150 --prefill auto --kvres 32768 --graph 1
fi

# ---- 32K: the baseline of record at this length + the two levers that scale with depth
if [ "$group" = abl32k ] || [ "$group" = all ]; then
  run_arm p9-32k-base   32768 256 --prefill auto --kvres 32768 --graph 0
  run_arm p9-32k-graph  32768 256 --prefill auto --kvres 32768 --graph 1
  run_arm p9-32k-dec0   32768 256 --prefill auto --kvres 32768 --graph 0 --env STRATA_DEC_BATCH=0
fi

# ---- 128K: the depth arms (each is ~9 min: the 129 024-token prefill is 372 s)
if [ "$group" = abl128k ] || [ "$group" = all ]; then
  run_arm p9-128k-base  131072 256 --prefill auto --kvres 32768 --graph 0
  run_arm p9-128k-kv0   131072 256 --prefill auto --kvres 0     --graph 0
  run_arm p9-128k-graph 131072 256 --prefill auto --kvres 32768 --graph 1
fi

echo "###### p9 chain group $group done $(date -Is)" >> "$LOG"
tail -3 "$LOG"
