#!/usr/bin/env bash
# D1: the histogram tables for every histogram arm, into one file the write-up quotes from
# (d1/D1-HISTOGRAMS.txt) plus a CSV per arm (next to the arm's own hist.txt).
#   bash d1/d1_hist_all.sh [top]
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/d1/runs
TOP=${1:-20}
OUT=$SRC/d1/D1-HISTOGRAMS.txt
: > "$OUT"
for tag in d1-hist-4096-closed d1-hist-32768-closed d1-hist-131072-closed d1-hist-4096-graph d1-hist-32768-graph d1-hist-131072-graph; do
  f=$RUN/$tag/hist.txt
  [ -s "$f" ] || continue
  {
    echo "################################################################################"
    echo "### $tag   ($(grep -m1 '^ctx=' $RUN/$tag/log.txt))"
    echo "### engine line: $(grep 'strata decode timing' $RUN/$tag/err.txt | tail -1)"
    echo "### submit line: $(grep 'strata submit: ' $RUN/$tag/err.txt | tail -1)"
    echo "### ms/window from the arm: $(/usr/bin/python3 $SRC/d1/d1_report.py $tag | tail -1)"
    echo
    /usr/bin/python3 "$SRC/d1/d1_hist.py" "$f" --auto --top "$TOP" --sites --csv "$RUN/$tag/sites.csv"
    echo
  } >> "$OUT" 2>&1
done
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
