#!/usr/bin/env bash
# D3 (card t_87aa2963): the remaining blocks in one background run, in the order that answers the card first.
# Each block's run_arm() waits for no engine of ours before it starts, so this can be launched while an arm is
# still finishing.  Blocks (see d3_chain3.sh for what each one contains):
#   F  the record's 4K census (memory top-k) + the memory top-k at the ARM's CTX
#   A3 the record's census at 32K and 128K
#   B  the shipped window-level sharing vs the old per-query grid, 3 depths
R=/home/michael/strata-xpu/strata
SRC=$R/strata
LOG=$SRC/d3/runall.log
echo "###### d3 runall started $(date -Is)" >> "$LOG"
for b in F A3 B; do
  echo "== runall: block $b $(date -Is)" >> "$LOG"
  bash "$SRC/d3/d3_chain3.sh" "$b" >> "$LOG" 2>&1
done
echo "###### d3 runall done $(date -Is)" >> "$LOG"
tail -3 "$LOG"
