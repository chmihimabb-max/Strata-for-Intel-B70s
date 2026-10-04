#!/usr/bin/env bash
# D3 (card t_87aa2963): the last blocks - the sync ceiling (E-6) and the two 4K controls.
#
#   bash d3/d3_chain4.sh D    the coarsest sync the engine already has: STRATA_VERIFY_DEVICE_PLAN=1, with the SAME
#                             instruments as the record-config census control (d3-histc-*), so the device-side
#                             handshake numbers (wait_flag_ge vs wait_flag_ge_or) are directly comparable
#   bash d3/d3_chain4.sh E1   the top-k dispatch pair at ONE max-context: maxctx 4096 + STRATA_TOPK_OLD=1 (the
#                             memory kernel, by force) against d3-hist-4096 (maxctx 4096, default = register)
#   bash d3/d3_chain4.sh E2   the record's --max-context 262144 against the arm's own CTX at 4K, no instruments
#                             (the max-context effect alone, on the served decode window)
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain4.log
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

which=${1:?D|E1|E2}
echo "###### d3 chain4 $which started $(date -Is)" >> "$LOG"
case "$which" in
  D)
    run_arm "d3-devplan-4096" 4096 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1 \
            --env STRATA_VERIFY_DEVICE_PLAN=1
    run_arm "d3-devplan-32768" 32768 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1 \
            --env STRATA_VERIFY_DEVICE_PLAN=1
    run_arm "d3-devplan-131072" 131072 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1 \
            --env STRATA_VERIFY_DEVICE_PLAN=1
    ;;
  E1)
    run_arm "d3-oldtopkhist-4096" 4096 256 --graph 0 --hist 1 --histwindows 8 --profile 1 \
            --env STRATA_TOPK_OLD=1
    ;;
  E2)
    run_arm "d3-basearmctx-4096" 4096 256
    ;;
esac
echo "###### d3 chain4 $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
