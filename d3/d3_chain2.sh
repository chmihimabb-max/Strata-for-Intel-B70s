#!/usr/bin/env bash
# D3 (card t_87aa2963): the arm blocks that answer the card.  ONE ENGINE AT A TIME, both B70s, ZE_AFFINITY_MASK
# unset, the config of record verbatim (--spec 4 --spec-min-p 0.7 --kv int8 --expert-cache auto --mmap-experts
# --kv-resident 32768 --max-context 262144, serve + layer-split auto) except the one lever each arm names.
#
# WHY THE maxctx=262144 EXPLICITLY: the config of record serves --max-context 262144, and the QSA top-k's
# DISPATCH reads that capacity, not the arm's prompt length -- D1's and D2's rigs passed the ARM's CTX as
# --max-context (a stated deviation), so every published decode arm ran the register top-k kernel while the
# served configuration runs the memory-keyed one (qsa_select.cu:748-783: reach = max_blocks = max_cells/4 + 2 vs
# fit = TK_T*TK_PER).  D3 measures the record's own value and keeps one arm at the arm's CTX as the control.
#
#   bash d3/d3_chain2.sh A2   the selection's attribution at the CONFIG OF RECORD: census + P9 stamps, closure, 3 depths
#   bash d3/d3_chain2.sh B    task 2: the shipped window-level sharing vs the old per-query grid, 3 depths
#   bash d3/d3_chain2.sh C    the top-k dispatch: which kernel the record's max-context picks, and the arm-CTX control
#   bash d3/d3_chain2.sh D    task 3: the coarsest exact sync the engine has (E-6 device plan), 3 depths
#   bash d3/d3_chain2.sh E    the fixed path's arms (only if the top-k dispatch change is landed)
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain2.log
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

which=${1:?A2|B|C|D|E}
echo "###### d3 chain2 $which started $(date -Is)" >> "$LOG"
case "$which" in
  A2)
    # the census needs the closure path (STRATA_SYCL_GRAPH=0): a replayed graph carries no per-kernel timestamp.
    for CTX in 4096 32768 131072; do
      run_arm "d3-histc-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1
    done
    ;;
  B)
    for CTX in 4096 32768 131072; do
      run_arm "d3-base-$CTX" "$CTX" 256 --maxctx "$MAXCTX"
      run_arm "d3-nomulti-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --env STRATA_SCORES_MULTI=0
    done
    ;;
  C)
    # which top-k kernel the two capacities pick: the register one (launcher's own fit test) or the memory one
    run_arm "d3-oldtopk-4096" 4096 256 --maxctx "$MAXCTX" --env STRATA_TOPK_OLD=1
    run_arm "d3-armctx-4096" 4096 256 --env STRATA_TOPK_OLD=1
    ;;
  D)
    for CTX in 4096 32768 131072; do
      run_arm "d3-devplan-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --env STRATA_VERIFY_DEVICE_PLAN=1
    done
    ;;
  E)
    for CTX in 4096 32768 131072; do
      run_arm "d3-fix-$CTX" "$CTX" 256 --maxctx "$MAXCTX"
    done
    run_arm "d3-histfix-4096" 4096 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1
    ;;
esac
echo "###### d3 chain2 $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
