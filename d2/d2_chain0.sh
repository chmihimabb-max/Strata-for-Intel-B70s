#!/usr/bin/env bash
# D2 (card t_0416a0c0) Task 0: re-verify `--spec-min-p 0.5 -> 0.7` at 4K/32K/128K on the config of record.
# Two arms per length, one engine at a time, same binary for both (build-sycl/strata, i.e. the config of
# record's exe; md5 recorded in every arm log).  `--spec 4` in both arms, as the config of record carries.
#   bash d2/d2_chain0.sh <4096|32768|131072|all>
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2/runs
LOG=$SRC/d2/chain0.log
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

which=${1:?4096|32768|131072|all}
echo "###### d2 task0 spec-min-p chain $which started $(date -Is)" >> "$LOG"
for CTX in 4096 32768 131072; do
  [ "$which" = all ] || [ "$which" = "$CTX" ] || continue
  run_arm "d2-minp05-$CTX" "$CTX" 256 --spec 4 --spec-min-p 0.5
  run_arm "d2-minp07-$CTX" "$CTX" 256 --spec 4 --spec-min-p 0.7
done
echo "###### d2 task0 spec-min-p chain $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
