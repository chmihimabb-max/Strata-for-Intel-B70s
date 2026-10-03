#!/usr/bin/env bash
# D1 group B: the spec-width sweep.  `--spec N` at 4K/32K/128K, config of record otherwise (now including the
# graph path, which is the default after group A).  One arm per (width, length); the four numbers the card asks
# for (tokens/window, ms/window, acceptance, decode tok/s) all come out of that arm's own logs (d1_report.py).
#   bash d1/d1_spec.sh <4096|32768|131072|all>
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

which=${1:?4096|32768|131072|all}
echo "###### d1 spec sweep $which started $(date -Is)" >> "$LOG"
for CTX in 4096 32768 131072; do
  [ "$which" = all ] || [ "$which" = "$CTX" ] || continue
  for S in 2 4 6 8; do
    run_arm "d1-spec${S}-${CTX}" "$CTX" 256 --bin "$SRC/d1/strata-after" --spec "$S"
  done
done
echo "###### d1 spec sweep $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
