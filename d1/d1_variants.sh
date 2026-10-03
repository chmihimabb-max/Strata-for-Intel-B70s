#!/usr/bin/env bash
# D1 group D: variants at the best width (set SPECWIN below from the sweep, then run all three lengths).
#   bash d1/d1_variants.sh <4096|32768|131072|all>
# Each arm differs from the config of record in ONE thing, on top of the winning `--spec N`.
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/d1/runs
LOG=$R/d1/chain.log
SPECWIN=${SPECWIN:-6}          # <- the sweep's best width (D1/STATUS-D1.md records why)
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

which=${1:?4096|32768|131072|all}
echo "###### d1 variants (spec $SPECWIN) $which started $(date -Is)" >> "$LOG"
for CTX in 4096 32768 131072; do
  [ "$which" = all ] || [ "$which" = "$CTX" ] || continue
  run_arm "d1-v-minp-${CTX}" "$CTX" 256 --bin "$SRC/d1/strata-after" --spec "$SPECWIN" --spec-min-p 0.7
  run_arm "d1-v-maxt-${CTX}" "$CTX" 256 --bin "$SRC/d1/strata-after" --spec "$SPECWIN" --mtp-max-t 2
done
echo "###### d1 variants $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
