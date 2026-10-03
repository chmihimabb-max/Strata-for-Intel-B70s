#!/usr/bin/env bash
# P1b: recompute the P1 record's token-id md5s with the same recipe p1b_run_engine.sh uses (grep '^T '), so the
# before/after comparison in STATUS-P1B.md is over identical text.
R=/home/michael/strata-xpu
for a in p1-4k-before p1-4k-after p1-32k-before p1-32k-after; do
  f=$R/m6c/runs/$a/out.txt
  n=$(grep -c '^T ' "$f")
  m=$(grep '^T ' "$f" | md5sum | cut -d' ' -f1)
  echo "$a  T-lines=$n  md5-of-T-lines=$m"
done
