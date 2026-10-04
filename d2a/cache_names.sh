#!/bin/bash
# D2a: what the SYCL program cache holds, by mtime. Each entry has 0.src (the kernel source).
C=/home/michael/strata-xpu/sycl-cache/m6c/13665086051514919768
OUT=/home/michael/strata-xpu/strata/d2a/cache-entries.txt
: > "$OUT"
find "$C" -name '0.src' -printf '%T@ %h\n' | sort -n > /tmp/d2a_srcs.txt
echo "entries: $(wc -l < /tmp/d2a_srcs.txt)" | tee -a "$OUT"
while read -r t d; do
  ts=$(date -d "@${t%.*}" '+%Y-%m-%d %H:%M:%S')
  # the kernel name and its template arguments, from the generated 0.src
  k=$(grep -o 'native_[a-z0-9_]*kernel<[^>]*>' "$d/0.src" 2>/dev/null | head -1)
  if [ -z "$k" ]; then
    k=$(grep -o 'void kernel [a-zA-Z0-9_]*' "$d/0.src" 2>/dev/null | head -1)
  fi
  printf '%s %s %s\n' "$ts" "$(stat -c%s "$d/0.src")" "$k" >> "$OUT"
done < /tmp/d2a_srcs.txt
echo "written $OUT"
