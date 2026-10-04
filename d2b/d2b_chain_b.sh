#!/usr/bin/env bash
# D2b (card t_f93760a1) chain B: the warm-up's own arms, on the rebuilt binary (md5 f49fe666...).
#
#   B1 d2b-wu-4096        warm caches, warm-up ON   - the residual must stay at the shipped ~3.5 ms and the ids
#                                                    must be the 4K canonical 66bf952d...
#   B2 d2b-wu-32768       warm caches, warm-up ON   - the 32K canonical d87373e8...
#   B3 d2b-ctloff-4096    warm caches, OFF          - the same-session rebuild control (new binary, nothing else)
#   B4 d2b-wu-cold-4096   BOTH caches fresh, ON     - a fresh install: the build must land in the load phase
#   B5 d2b-ctloff-cold-4096 BOTH caches fresh, OFF  - a fresh install without the warm-up (the A/B that says the
#                                                    warm-up is what removed the cost)
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2b/chain_b.log
NEW=$SRC/build-sycl/strata
{
  echo "###### D2b chain B started $(date -Is)"
  echo "## the rebuilt binary: $(md5sum < $NEW | cut -c1-32)  (was e841fd061fec873c2f24e785973a2ebe)"

  echo "###### B1 d2b-wu-4096 $(date -Is)"
  bash $SRC/d2b/d2b_run_arm.sh d2b-wu-4096 4096 256 --bin "$NEW" --spec-min-p 0.7

  echo "###### B3 d2b-ctloff-4096 $(date -Is)"
  bash $SRC/d2b/d2b_run_arm.sh d2b-ctloff-4096 4096 256 --bin "$NEW" --spec-min-p 0.7 \
       --env STRATA_MMVQ_WARMUP=0

  echo "###### B4 d2b-wu-cold-4096 $(date -Is)"
  rm -rf /tmp/d2b-cold2; mkdir -p /tmp/d2b-cold2/sycl /tmp/d2b-cold2/neo
  bash $SRC/d2b/d2b_run_arm.sh d2b-wu-cold-4096 4096 256 --bin "$NEW" --spec-min-p 0.7 \
       --sycl-cache /tmp/d2b-cold2/sycl --neo-cache /tmp/d2b-cold2/neo

  echo "###### B5 d2b-ctloff-cold-4096 $(date -Is)"
  rm -rf /tmp/d2b-cold3; mkdir -p /tmp/d2b-cold3/sycl /tmp/d2b-cold3/neo
  bash $SRC/d2b/d2b_run_arm.sh d2b-ctloff-cold-4096 4096 256 --bin "$NEW" --spec-min-p 0.7 \
       --env STRATA_MMVQ_WARMUP=0 --sycl-cache /tmp/d2b-cold3/sycl --neo-cache /tmp/d2b-cold3/neo

  echo "###### B2 d2b-wu-32768 (last: the longest arm) $(date -Is)"
  bash $SRC/d2b/d2b_run_arm.sh d2b-wu-32768 32768 256 --bin "$NEW" --spec-min-p 0.7

  echo "###### D2b chain B ALL DONE $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
