#!/usr/bin/env bash
# D3 (card t_87aa2963): the arm chain, one engine at a time.  Every arm is the config of record (--spec-min-p 0.7,
# --spec 4, --kv int8, --expert-cache auto, --mmap-experts, --kv-resident 32768, both B70s, layer-split auto) with
# exactly ONE lever/instrument changed, and every arm passes the two deviations D1's rig established (--prefill
# auto instead of the file's 512, and --prompt-cache 0 --prompt-cache-every 0 for S4's split OOM).
#
#   bash d3/d3_chain.sh A     the selection's attribution: census (hist) + P9 stage stamps, closure path, 3 depths
#   bash d3/d3_chain.sh B     task 2: the shipped window-level sharing (STRATA_SCORES_MULTI default) vs the old
#                             per-query grid (STRATA_SCORES_MULTI=0), 3 depths, graph path default
#   bash d3/d3_chain.sh C     task 3: the coarsest exact sync the engine has (STRATA_VERIFY_DEVICE_PLAN=1, E-6), 3 depths
#   bash d3/d3_chain.sh D     a 4K validation of the top-k dispatch arm's ids (see d3_chain_reg.sh)
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d3/runs
LOG=$SRC/d3/chain.log
mkdir -p "$RUN"

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

which=${1:?A|B|C}
echo "###### d3 chain $which started $(date -Is)" >> "$LOG"
case "$which" in
  A)
    # the census needs the closure path (STRATA_SYCL_GRAPH=0): a replayed graph has no per-kernel device timestamp.
    for CTX in 4096 32768 131072; do
      run_arm "d3-hist-$CTX" "$CTX" 256 --graph 0 --hist 1 --histwindows 8 --profile 1
    done
    ;;
  B)
    for CTX in 4096 32768 131072; do
      run_arm "d3-base-$CTX" "$CTX" 256
      run_arm "d3-nomulti-$CTX" "$CTX" 256 --env STRATA_SCORES_MULTI=0
    done
    ;;
  C)
    for CTX in 4096 32768 131072; do
      run_arm "d3-devplan-$CTX" "$CTX" 256 --env STRATA_VERIFY_DEVICE_PLAN=1
    done
    ;;
esac
echo "###### d3 chain $which done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
