#!/bin/bash
# D2a: pair each mg arm's own timeline with the SYCL cache entries written while it ran.
SRC=/home/michael/strata-xpu/strata
for d in d2-mg-4096 d2-mg-4096-warm d2-mg-4096-warm2 d2-mg-32768 d2-rebase-4096 d2-minp05-4096; do
  echo "======== $d"
  grep -E "^D1 arm|loaded=1 after|ask fed at|the ask finished" "$SRC/d2/runs/$d/log.txt"
  echo "-- first/last engine epoch (timeline)"
  head -1 "$SRC/d2/runs/$d/timeline.txt"
  tail -1 "$SRC/d2/runs/$d/timeline.txt"
  echo "-- cache entries written inside that window"
  s=$(head -1 "$SRC/d2/runs/$d/timeline.txt" | cut -f1)
  e=$(tail -1 "$SRC/d2/runs/$d/timeline.txt" | cut -f1)
  awk -v s="$s" -v e="$e" '$2+0>=s && $2+0<=e {n++; sz=$3; c[sz]++} END{printf "   %d entries\n", n}' "$SRC/d2a/cache-entries.txt"
done
