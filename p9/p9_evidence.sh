#!/usr/bin/env bash
# P9 (t_2403e6f6): collect the card's raw output into one file, straight out of the arms' own logs.
#
#   bash p9/p9_evidence.sh            -> p9/P9-EVIDENCE.txt
#
# Everything the write-up quotes is printed here from the run directories, so a reader can re-derive it.
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/p9/runs
OUT=$SRC/p9/P9-EVIDENCE.txt

{
  echo "=============================================================================================="
  echo "P9 (card t_2403e6f6) - raw output.  Collected $(date -Is) on $(hostname)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "engine binary: md5 $(md5sum < $SRC/build-sycl/strata | cut -d' ' -f1)"
  echo "=============================================================================================="
  echo
  echo "## 0. the arm table (p9/p9_report.py: every arm, out of its own logs)"
  echo
  cd "$SRC" && /usr/bin/python3 p9/p9_report.py 2>&1

  echo
  echo "## 1. the per-stage attribution (p9/p9_stages.py: the engine's own GPU stage table)"
  echo
  cd "$SRC" && /usr/bin/python3 p9/p9_stages.py p9-p4k5 p9-p32k p9-p128k 2>&1

  echo
  echo "## 2. the decode-timing lines, verbatim (one per arm)"
  echo
  for d in $(ls -d $RUN/p9-* 2>/dev/null | sort); do
    t=$(basename "$d")
    line=$(grep -h "strata decode timing" "$d/err.txt" 2>/dev/null | tail -1)
    [ -n "$line" ] && echo "$t: $line"
  done

  echo
  echo "## 3. the stage lines, verbatim (profiled arms only)"
  echo
  for t in p9-p4k p9-p4k2 p9-p4k3 p9-p4k4 p9-p4k5 p9-p32k p9-p128k; do
    line=$(grep -ho "strata decode GPU stages.*" "$RUN/$t/err.txt" 2>/dev/null | tail -1)
    [ -n "$line" ] && { echo "$t:"; echo "  $line"; }
  done

  echo
  echo "## 4. the stalls, verbatim (the chrome-mode arms: the tracer's effect on the window)"
  echo
  for t in p9-ut9-4k-g0 p9-ut9-4k-g1; do
    echo "-- $t"
    grep -hE "PP |ERR verify|REUSED" "$RUN/$t/out.txt" 2>/dev/null | tail -4
    echo "   T lines: $(grep -c '^T ' "$RUN/$t/out.txt" 2>/dev/null)"
    echo "   trace: $(ls -la $RUN/$t/strata.*.json 2>/dev/null | awk '{print $5" B  "$9}')"
  done
  echo "-- P1b's own two arms (p1/logs), re-read for the same question"
  for t in p1b-x-ut9 p1b-z-ut1; do
    echo "   $t: $(grep -hE '^ERR verify' $R/p1/logs/$t-out.txt 2>/dev/null | tail -1)  T lines: $(grep -c '^T ' $R/p1/logs/$t-out.txt 2>/dev/null)"
  done

  echo
  echo "## 5. the device-clock probe (both cards)"
  echo
  $R/p9/p9_clock_probe 2>&1 | head -20

  echo
  echo "## 6. the engine's own profiler, before this card (the aspect crash)"
  echo
  grep -hE "PP 3831|terminate called|what\(\)|^ERR |T lines" "$RUN/p9-ctl-4k-16/out.txt" 2>/dev/null | tail -4
  tail -3 "$RUN/p9-ctl-4k-16/err.txt" 2>/dev/null

  echo
  echo "## 7. the trace census (mode 9, graph off): card 0's device time by kernel, and what follows the prefill"
  echo
  cd "$SRC" && /usr/bin/python3 p9/p9_summary.py $R/p9/dev/p9-ut9-4k-g0.c0.npz 10 2>&1
  cd "$SRC" && /usr/bin/python3 p9/p9_window.py $R/p9/dev/p9-ut9-4k-g0.c0.npz --big-ms 5 --list 5 2>&1 | tail -18

  echo
  echo "## 8. every arm's DONE line and token-id md5 (the correctness record)"
  echo
  for d in $(ls -d $RUN/p9-* 2>/dev/null | sort); do
    t=$(basename "$d")
    done_line=$(grep -h "^DONE" "$d/out.txt" 2>/dev/null | tail -1)
    md5=$(grep "^T " "$d/out.txt" 2>/dev/null | md5sum | cut -c1-32)
    [ -n "$done_line" ] && echo "$t: $done_line   md5(T lines)=$md5"
  done
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
