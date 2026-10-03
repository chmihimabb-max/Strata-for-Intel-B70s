#!/usr/bin/env bash
# P7 (card t_6789d6df): pull the comparable numbers out of every arm's own logs.
R=/home/michael/strata-xpu
for d in "$R"/p7/runs/*/; do
  t=$(basename "$d")
  echo "================================================================"
  echo "ARM $t"
  grep -m1 "^INFO " "$d/out.txt" 2>/dev/null | sed 's/^/  /'
  grep -E "^mtp runtime|^mtp draft_vocab" "$d/log.txt" 2>/dev/null | sed 's/^/  /'
  echo "  -- DONE (serve) --"
  grep -E "^DONE " "$d/out.txt" 2>/dev/null | sed 's/^/  /'
  echo "  -- serve summary --"
  grep -E "strata serve: prompt |suffix drafts" "$d/err.txt" 2>/dev/null | head -3 | sed 's/^/  /'
  echo "  -- engine's own timing --"
  grep -E "^strata decode timing" "$d/err.txt" 2>/dev/null | sed 's/^/  /'
  echo "  -- drafter VRAM --"
  grep -E "draft layer loaded|draft head over|expert cache auto|expert cache [0-9]+ slots|slot 0 verified" "$d/err.txt" 2>/dev/null | head -6 | sed 's/^/  /'
  echo "  -- bench summary (if the arm ran the bench path) --"
  grep -E "^decode |^prefill |^speculation|^suffix drafts|^accepted per round" "$d/out.txt" 2>/dev/null | sed 's/^/  /'
  echo "  -- tokens --"
  [ -f "$d/tcount.txt" ] && echo "  T lines: $(cat "$d/tcount.txt")  md5 $(md5sum "$d/tokens.txt" | cut -d' ' -f1)"
  echo "  -- peak VRAM/RSS --"
  [ -s "$d/rss.csv" ] && sort -t, -k4 -n "$d/rss.csv" | tail -1 | sed 's/^/  /'
done
