#!/bin/bash
# D2a: what is inside a SYCL program-cache entry? (can the specialization be identified from it?)
C=/home/michael/strata-xpu/sycl-cache/m6c/13665086051514919768
D1=$C/10010679668444384200/6142509188972423790/9327767619666270553
echo "== files in one entry"
ls -la "$D1"
echo "== head of 0.src"
head -c 600 "$D1/0.src"
echo
echo "== any strata symbol in 0.src?"
grep -o "strata[a-zA-Z_:0-9<>]*" "$D1/0.src" | sort | uniq -c | sort -rn | head -5
echo "== the two entries written at 19:24:03 (the smallest class)"
find "$C" -name '0.src' -newermt '2026-10-03 19:24:02' ! -newermt '2026-10-03 19:24:05' -printf '%p\n' | while read -r f; do
  echo "--- $f ($(stat -c%s "$f") B)"
  grep -o "strata[a-zA-Z_:0-9<>]*" "$f" | sort | uniq -c | sort -rn | head -3
  grep -o "native_[a-z0-9_]*" "$f" | sort | uniq -c | head -5
done
