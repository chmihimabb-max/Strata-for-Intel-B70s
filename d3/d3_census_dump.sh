#!/usr/bin/env bash
# D3 (t_87aa2963): dump the census of the named hist files into one file (the raw output the card's acceptance
# asks for).  Usage: bash d3/d3_census_dump.sh <out.txt> <hist.txt> [<hist.txt> ...]
OUT=${1:?out}; shift
R=/home/michael/strata-xpu/strata
cd "$R" || exit 1
{
  echo "D3 (card t_87aa2963) census dump   $(date -Is)"
  echo "repo $(git log --oneline -1)"
  echo "usage: d3/d3_select.py --brief <files>  then per file"
  echo
  /usr/bin/python3 d3/d3_select.py --brief "$@"
  echo
  for h in "$@"; do
    /usr/bin/python3 d3/d3_select.py "$h"
  done
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
