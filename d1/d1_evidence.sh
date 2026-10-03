#!/usr/bin/env bash
# D1 (card t_d9ffcf38): collect every raw line the card's acceptance asks for out of the arms' own logs, into
# one file (d1/D1-EVIDENCE.txt) that is committed with the write-up.
#   bash d1/d1_evidence.sh
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/d1/runs
OUT=$SRC/d1/D1-EVIDENCE.txt
{
  echo "========================================"
  echo "D1 (card t_d9ffcf38) EVIDENCE  $(date -Is)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "host: $(uname -srm); $(nproc) cores; $(zgrep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  echo "two B70s, ZE_AFFINITY_MASK unset for every arm; config of record strata-sycl-iq3s.json (IQ3_S ed59f920)"
  echo "engine binaries: d1/strata-before md5 $(md5sum < $SRC/d1/strata-before | cut -c1-32)"
  echo "                 d1/strata-after  md5 $(md5sum < $SRC/d1/strata-after | cut -c1-32)"
  echo "                 build-sycl/strata md5 $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  echo
  for d in "$RUN"/*; do
    [ -d "$d" ] || continue
    tag=$(basename "$d")
    echo "################################################################################"
    echo "### ARM $tag"
    echo "###   $(date -r "$d" -Is)"
    echo
    echo "--- the arm's own setup line (binary, graph/hist switches, spec) ---"
    grep -E "^D1 arm |^ctx=|^spec=|^engine binary:|^HEAD:|^extra=|^onecard=" "$d/log.txt" 2>/dev/null | head -8
    echo
    echo "--- the graph path's banner ---"
    grep -E "strata/sycl: (the graph path|STRATA_SYCL_GRAPH)" "$d/err.txt" 2>/dev/null | head -2
    echo
    echo "--- the decode timing line (per window; the card's tokens/window + ms/window) ---"
    grep -E "strata decode timing" "$d/err.txt" 2>/dev/null | tail -2
    echo
    echo "--- the submit line (P3's counters: kernels/copies/barriers per window) ---"
    grep -E "strata submit: " "$d/err.txt" 2>/dev/null | tail -3
    echo
    echo "--- the request: DONE (gen/prompt/decode ms/finish/accepted/offered) and the token ids ---"
    grep -E "^DONE|^ERR " "$d/out.txt" 2>/dev/null | tail -2
    grep -E "strata serve: prompt " "$d/err.txt" 2>/dev/null | tail -2
    echo "    T lines: $(grep -c '^T ' "$d/out.txt" 2>/dev/null)  md5 $(grep '^T ' "$d/out.txt" 2>/dev/null | md5sum | cut -c1-32)"
    echo "    PP line: $(grep '^PP ' "$d/out.txt" 2>/dev/null | tail -1)"
    echo "    engine exit: $(grep -E '^== engine exit' "$d/log.txt" 2>/dev/null | tail -1)"
    if [ -s "$d/hist.txt" ]; then
      echo
      echo "--- the launch-site histogram (d1/runs/$tag/hist.txt): window headers ---"
      grep -E "^HW" "$d/hist.txt" | head -20
      echo "    (lines: $(wc -l < "$d/hist.txt"))"
    fi
    echo
  done
  echo "################################################################################"
  echo "### the engine's test suite (d1/runs/ctest)"
  for f in "$RUN"/ctest/*.log; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") ---"
    grep -E "tests passed|tests failed|Failed" "$f" | tail -8
  done
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
