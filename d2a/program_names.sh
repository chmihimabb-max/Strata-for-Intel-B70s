#!/bin/bash
# D2a: name the SYCL programs the generic layout's first run built, from the mangled symbols in each entry.
C=/home/michael/strata-xpu/sycl-cache/m6c/13665086051514919768
OUT=/home/michael/strata-xpu/strata/d2a/programs.txt
: > "$OUT"
find "$C" -name '0.src' -printf '%T@ %h\n' | sort -n > /tmp/d2a_srcs2.txt
while read -r t d; do
  ts=$(date -d "@${t%.*}" '+%H:%M:%S')
  # the kernel specialization: the mangled lambda type inside strata::sycl_compat::launch
  sym=$(strings -a "$d/0.src" | grep -o 'native_mmvq[a-zA-Z0-9_]*E[JA-Za-z0-9_]*E\{0,3\}' | head -1)
  if [ -z "$sym" ]; then
    sym=$(strings -a "$d/0.src" | grep -o 'native_[a-z0-9_]*\(multi_\|\)kernel' | head -1)
  fi
  printf '%s %8s %s\n' "$ts" "$(stat -c%s "$d/0.src")" "$sym" >> "$OUT"
done < /tmp/d2a_srcs2.txt
echo "== the programs written while d2-mg-4096 decoded (19:24:03-19:24:11)"
awk '$1 >= "19:24:03" && $1 <= "19:24:11"' "$OUT"
echo "== the programs written while d2-mg-4096-warm decoded (19:30:55-19:31:00)"
awk '$1 >= "19:30:55" && $1 <= "19:31:00"' "$OUT"
echo "== distinct specialization names, all entries"
awk '{print $3}' "$OUT" | sort | uniq -c | sort -rn | head -20
