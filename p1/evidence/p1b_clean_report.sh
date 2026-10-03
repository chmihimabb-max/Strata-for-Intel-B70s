#!/usr/bin/env bash
# P1b: the clean arms' token ids and timings, against P1's recorded HEAD arms.
R=/home/michael/strata-xpu
echo "record (P1, HEAD 3ccb530/b538baf):"
bash "$R/p1/p1b_md5.sh"
echo
for a in p1b-j-clean4k p1b-k-clean32k p1b-l-secondask; do
  d=$R/p1/runs/$a
  [ -f "$d/tokens.txt" ] || continue
  echo "== $a"
  echo "   T lines: $(grep -c '^T ' "$d/tokens.txt")  md5: $(md5sum "$d/tokens.txt" | cut -d' ' -f1)"
  grep -E "^PP 3831|^PP 3225|^DONE|^ERR|decode timing|prompt .* tokens" "$d/out.txt" "$d/err.txt" 2>/dev/null | tail -4
  grep -E "exit " "$d/log.txt" | tail -1
done
