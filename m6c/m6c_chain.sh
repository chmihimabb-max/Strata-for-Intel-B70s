#!/usr/bin/env bash
# M6c: the rest of the run order, chained so nothing overlaps (PLAN 9 rule 1: one engine at a time).
#   1. wait for m6c_block1.sh (the --prefill 512 arm) to finish;
#   2. block 1b: the --prefill auto arm of the short lengths;
#   3. block 2: the 256K arms.
R=/home/michael/strata-xpu
echo "M6c chain start $(date -Is)"
for i in $(seq 1 240); do
  if ! pgrep -f "m6c_[b]lock1.sh" > /dev/null; then break; fi
  sleep 20
done
echo "block 1 finished (or 80 min elapsed) $(date -Is)"
bash "$R/strata/m6c/m6c_block1b.sh" > /dev/null 2>&1
echo "block 1b returned $? $(date -Is)"
bash "$R/strata/m6c/m6c_block2.sh" > /dev/null 2>&1
echo "block 2 returned $? $(date -Is)"
echo "M6c chain end $(date -Is)"
