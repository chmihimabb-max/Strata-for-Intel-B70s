#!/usr/bin/env bash
# D2 (card t_0416a0c0): every raw line the card's acceptance asks for, into d2/D2-EVIDENCE.txt.
# Re-runnable: it only reads the arms' own logs.
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2/runs
O=$SRC/d2/D2-EVIDENCE.txt
{
  echo "# D2 (t_0416a0c0) raw evidence -- generated $(date -Is)"
  echo "## HEAD"
  git -C "$SRC" log --oneline -4
  echo "## the engine binary the arms ran (md5 per arm is in each arm's own log)"
  md5sum "$SRC/build-sycl/strata"
  echo
  echo "## 1. every arm, one row, read out of that arm's own logs (d2/d2_report.py)"
  /usr/bin/python3 "$SRC/d2/d2_report.py"
  echo
  echo "## 2. the raw engine lines, arm by arm (decode timing, the request's own line, the banners, ids)"
  for d in $(ls -d "$RUN"/*/ 2>/dev/null | sort); do
    tag=$(basename "$d")
    [ -f "$d/err.txt" ] || continue
    echo "=============================== $tag"
    grep -E "engine binary:|graph path|STRATA_SYCL_GRAPH=0|strata hc:|spec=|extraenv=" "$d/log.txt" 2>/dev/null | head -8
    grep -E "strata decode timing:" "$d/err.txt" | tail -1
    grep -E "strata serve: prompt " "$d/err.txt" | tail -1
    grep -E "0\.[0-9]+% of the experts|[0-9]+% of the experts resident|native projection matrices|canonical tensors skipped" "$d/err.txt" | head -4
    grep -E "^DONE" "$d/out.txt" 2>/dev/null | tail -1
    echo "   T lines: $(grep -c '^T ' "$d/out.txt" 2>/dev/null)  md5 $(grep '^T ' "$d/out.txt" 2>/dev/null | md5sum | cut -c1-32)"
    grep -E "wall [0-9]+ s" "$d/log.txt" | tail -1
  done
  echo
  echo "## 3. the window's census (the corrected attribution; closure-path arms carry device microseconds)"
  /usr/bin/python3 "$SRC/d2/d2_census.py" $(ls "$RUN"/*/hist.txt 2>/dev/null)
  echo
  echo "## 4. the launch sites of the 4K histogram arm, merged per kernel"
  if [ -f "$RUN/d2-hist-4096/hist.txt" ]; then
    /usr/bin/python3 "$SRC/d1/d1_hist.py" "$RUN/d2-hist-4096/hist.txt" --auto --top 25
  fi
  echo
  echo "## 5. the config of record, and the diff this card landed"
  if [ -f "$SRC/d2/config-before-d2.json" ]; then
    diff -u "$SRC/d2/config-before-d2.json" "$SRC/strata-sycl-iq3s.json" || true
  else
    echo "(no pre-D2 copy at $SRC/d2/config-before-d2.json)"
  fi
  echo "--- the landed file (config of record) ---"
  cat "$SRC/strata-sycl-iq3s.json"
} > "$O" 2>&1
echo "wrote $O ($(wc -l < "$O") lines)"
