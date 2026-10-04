#!/usr/bin/env bash
# D2 (card t_0416a0c0) tasks 1-3: the dispatch A/B.  Every arm is the config of record at one context length with
# ONE lever changed, and every arm's greedy token ids are compared against the baseline arm's (same prompt, same
# binary, same sessions).  Arms are appended after task 0's chain so that only one engine runs at a time.
#   bash d2/d2_chain1.sh
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2/runs
LOG=$SRC/d2/chain1.log
mkdir -p "$RUN"

run_arm() {
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped $(date -Is)" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/d2/d2_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is)" >> "$LOG"
  sleep 5
}

# one engine at a time: task 0's chain owns the cards until it says it is done
echo "###### d2 task1-3 dispatch chain queued $(date -Is)" >> "$LOG"
for i in $(seq 1 3000); do
  grep -q "d2 task0 spec-min-p chain all done" "$SRC/d2/chain0.log" 2>/dev/null && break
  sleep 10
done
echo "###### d2 task1-3 dispatch chain started $(date -Is)" >> "$LOG"

# 1. the decode GEMV's two reachable dispatch shapes: batched multi-column (default) vs per-token single-column
run_arm "d2-dec0-4096" 4096 256 --env STRATA_DEC_BATCH=0
# 2. the hyper-connection read: staged (the default on this card, on-card-checked) vs split vs the plain reference
run_arm "d2-hcplain-4096" 4096 256 --env STRATA_HC_SPLIT=0
run_arm "d2-hcsplit-4096" 4096 256 --env STRATA_HC_SPLIT=1
# 3. the mixer read: the fused multi read vs the per-token plain gr_read (a different kernel for the same work)
run_arm "d2-plaingr-4096" 4096 256 --env STRATA_WINDOW_PLAIN_GR=1
# 4. the census: the closure path, so every launch has a device timestamp, at 4K and 32K
run_arm "d2-hist-4096" 4096 256 --hist 1 --graph 0
run_arm "d2-hist-32768" 32768 256 --hist 1 --graph 0
# 5. the batched-vs-per-token arm at 32K, to see whether the 4K answer holds with depth
run_arm "d2-dec0-32768" 32768 256 --env STRATA_DEC_BATCH=0

echo "###### d2 task1-3 dispatch chain done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
