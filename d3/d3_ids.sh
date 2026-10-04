#!/usr/bin/env bash
# D3 (t_87aa2963): one line per arm with the RIG's own token-id md5 (`grep '^T ' | md5sum`, trailing newline
# included) and the window numbers from the `strata decode timing` line - the guard table.  One method, so every
# id comparison in the write-up is like for like.  (D2's/P9's published ids were computed with their own readers:
# D2's reader joins the T lines without a trailing newline, the rig's md5sum includes it - the two are different
# hashes of the same stream, so a figure is only comparable within one method.)
R=/home/michael/strata-xpu/strata
cd "$R" || exit 1
printf '%-26s %-30s %-22s %7s %6s %5s %9s %8s %-34s\n' tag lever path maxctx win avgT ms/win tok/s "ids md5 (rig method)"
for d in "$R"/d3/runs/*/; do
  tag=$(basename "$d")
  [ -f "$d/log.txt" ] || continue
  lever=$(grep -o "extraenv=[^ ]*" "$d/log.txt" | head -1 | cut -d= -f2)
  path=$(grep -q "the graph path is ON" "$d/err.txt" 2>/dev/null && echo "graph(default)")
  [ -n "$path" ] || path=$(grep -oE "STRATA_SYCL_GRAPH=[01]" "$d/err.txt" 2>/dev/null | head -1)
  maxctx=$(grep -o "max-context [0-9]*" "$d/log.txt" | head -1 | awk '{print $2}')
  line=$(grep "strata decode timing" "$d/err.txt" 2>/dev/null | tail -1)
  win=$(printf '%s' "$line" | grep -oE "^strata decode timing: [0-9]+" | awk '{print $4}')
  avg=$(printf '%s' "$line" | grep -oE "avg T [0-9.]+" | awk '{print $3}')
  msw=$(printf '%s' "$line" | grep -oE "[0-9.]+ ms/window" | head -1 | awk '{print $1}')
  toks=$(grep -oE "[0-9]+ generated in [0-9.]+ ms \([0-9.]+ tok/s\)" "$d/err.txt" 2>/dev/null | tail -1 | grep -oE "[0-9.]+ tok/s" | awk '{print $1}')
  ids=$(grep '^T ' "$d/out.txt" 2>/dev/null | md5sum | cut -c1-32)
  n=$(grep -c '^T ' "$d/out.txt" 2>/dev/null)
  printf '%-26s %-30s %-22s %7s %6s %5s %9s %8s %-34s\n' "$tag" "${lever:--}" "${path:-closure(no banner)}" "${maxctx:--}" "$win" "$avg" "$msw" "$toks" "$ids (T lines $n)"
done
