#!/usr/bin/env bash
# D3 (card t_87aa2963): the remaining arm blocks, in the order that answers the card first.  ONE ENGINE AT A TIME,
# both B70s, ZE_AFFINITY_MASK unset, the config of record verbatim (--spec 4 --spec-min-p 0.7 --kv int8
# --expert-cache auto --mmap-experts --kv-resident 32768 --max-context 262144, --layer-split auto) except the one
# lever each arm names.
#
# WHY --maxctx IS PASSED SEPARATELY FROM THE PROMPT LENGTH: the top-k's dispatch reads the CAPACITY
# (--max-context), not the depth a window actually reaches -- qsa_select.cu:748-783 takes the register kernel when
# reach = max_cells/4 + 2 <= fit = TK_T*TK_PER = 1024*33 = 33,792 blocks (135,168 cells).  The record's 262,144
# cells is 65,538 blocks, i.e. the memory-keyed kernel; D1's and D2's rigs passed the ARM's own CTX as
# --max-context, so every published decode arm ran the register one.  D3 measures the record's own value.
#
#   bash d3/d3_chain3.sh F    the record's 4K census + the memory top-k at the ARM's CTX (the dispatch's two sides)
#   bash d3/d3_chain3.sh A3   the record's census at 32K/128K
#   bash d3/d3_chain3.sh B    task 2: the shipped window-level sharing vs the old per-query grid, 3 depths
#   bash d3/d3_chain3.sh D    task 3: the coarsest exact sync the engine has (E-6 device plan), 3 depths
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain3.log
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

which=${1:?F|A3|B|D}
echo "###### d3 chain3 $which started $(date -Is)" >> "$LOG"
case "$which" in
  F)
    run_arm "d3-histc-4096" 4096 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1
    run_arm "d3-oldtopk-4096" 4096 256 --env STRATA_TOPK_OLD=1
    ;;
  A3)
    run_arm "d3-histc-32768" 32768 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1
    run_arm "d3-histc-131072" 131072 256 --maxctx "$MAXCTX" --graph 0 --hist 1 --histwindows 8 --profile 1
    ;;
  B)
    for CTX in 4096 32768 131072; do
      run_arm "d3-base-$CTX" "$CTX" 256 --maxctx "$MAXCTX"
      run_arm "d3-nomulti-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --env STRATA_SCORES_MULTI=0
    done
    ;;
  D)
    for CTX in 4096 32768; do
      run_arm "d3-devplan-$CTX" "$CTX" 256 --maxctx "$MAXCTX" --env STRATA_VERIFY_DEVICE_PLAN=1
    done
    ;;
esac
echo "###### d3 chain3 $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
