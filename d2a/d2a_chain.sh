#!/usr/bin/env bash
# D2a (card t_8306429a): engine-level repetition of the generic multi-column arm, now that the SYCL program
# cache holds its specializations.  Two chains, one engine at a time:
#
#   A. d2a-mg-4096-r1..r8   the D2 lever verbatim (STRATA_MMVQ_MULTI_GENERIC=1), 8 identical runs
#   B. d2a-mg0-4096-r1..r8  the same lever PLUS --suffix-draft 0: with the prompt-lookup drafter off the window
#                           size is no longer chosen by DraftPolicy, which is the only consumer of the round's
#                           measured time.  If the kernel is deterministic, these 8 runs must agree exactly.
#
# plus one unmodified control (d2a-rebase-4096) so every number here has a same-session baseline.
#   bash d2a/d2a_chain.sh
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2/runs
LOG=$SRC/d2a/chain.log
mkdir -p "$RUN"
cd "$SRC" || exit 1

run_arm() {
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped $(date -Is)" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  # one engine at a time
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/d2/d2_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is)" >> "$LOG"
  sleep 5
}

echo "###### d2a chain start $(date -Is) bin $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)" >> "$LOG"
# the same-session control: everything default
run_arm "d2a-rebase-4096" 4096 256
# A: the lever alone, eight times
for i in 1 2 3 4 5 6 7 8; do
  run_arm "d2a-mg-4096-r$i" 4096 256 --env STRATA_MMVQ_MULTI_GENERIC=1
done
echo "###### d2a chain A (the lever, 8x) done $(date -Is)" >> "$LOG"
# B: the lever with the timing-driven window choice removed
for i in 1 2 3 4 5 6 7 8; do
  run_arm "d2a-mg0-4096-r$i" 4096 256 --extra --suffix-draft --extra 0 --env STRATA_MMVQ_MULTI_GENERIC=1
done
echo "###### d2a chain B (the lever + --suffix-draft 0, 8x) done $(date -Is)" >> "$LOG"
tail -3 "$LOG"
