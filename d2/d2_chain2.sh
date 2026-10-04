#!/usr/bin/env bash
# D2 (card t_0416a0c0) task 2, the second lever: the routed experts' kernel VARIANT.  `iq_kernels.cu:1424` reads
# STRATA_OLD_IQ_MMVQ once and, when set, every expert gate+up / down launch takes `native_gu_kernel` /
# `native_down_kernel` (one entry at a time) instead of the shipped `native_gu_multi_kernel` /
# `native_down_multi_kernel` (GRP_NC = 4 entries per pass, iq_kernels.cu:994) -- the sibling pair the parity test
# `native_expert_parity` already A/Bs on the device.  Same weights, same contract, different kernel.
#   bash d2/d2_chain2.sh
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2/runs
LOG=$SRC/d2/chain2.log
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

echo "###### d2 task2 second lever queued $(date -Is)" >> "$LOG"
for i in $(seq 1 3000); do
  grep -q "d2 task1-3 dispatch chain done" "$SRC/d2/chain1.log" 2>/dev/null && break
  sleep 10
done
echo "###### d2 task2 second lever started $(date -Is)" >> "$LOG"
run_arm "d2-oldiq-4096" 4096 256 --env STRATA_OLD_IQ_MMVQ=1
run_arm "d2-oldiq-32768" 32768 256 --env STRATA_OLD_IQ_MMVQ=1
echo "###### d2 task2 second lever done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
