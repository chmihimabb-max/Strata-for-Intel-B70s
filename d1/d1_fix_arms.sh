#!/usr/bin/env bash
# D1: the histogram's fix (a mutex + "retain events only inside an open window") is an instrument-only change,
# so the arms it needs re-run are the instrument's own: the 128K graph-mode arm that crashed, and one arm that
# prices the dump (STRATA_LAUNCH_HIST_WINDOWS=1 against the default 8).  The default path is untouched by the
# fix (hist_take/hist_count are only reached under STRATA_LAUNCH_HIST), so the binaries differ only in the
# histogram: <before>/<after> md5s are recorded in each arm's own log.
#   bash d1/d1_fix_arms.sh
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$R/d1/runs
LOG=$R/d1/chain.log
set +e
if [ ! -f "$SRC/build-sycl/strata" ]; then echo "no engine"; exit 2; fi
# the arm that crashed, re-run with the fixed binary (same tag would be skipped by the chains, so it is named)
if [ -f "$RUN/d1-hist-131072-graph/log.txt" ]; then
  mv "$RUN/d1-hist-131072-graph" "$RUN/d1-hist-131072-graph-crash" 2>/dev/null
  echo "== kept the crashed arm as d1-hist-131072-graph-crash" >> "$LOG"
fi
bash "$SRC/d1/d1_run_arm.sh" d1-hist-131072-graph 131072 256 --bin "$SRC/d1/strata-after-histfix" --hist 1 >> "$LOG" 2>&1
echo "== d1 fix: 131072 graph re-run done $(date -Is)" >> "$LOG"
bash "$SRC/d1/d1_run_arm.sh" d1-hist-4096-w1 4096 256 --bin "$SRC/d1/strata-after-histfix" --graph 0 --hist 1 \
     --histwindows 1 >> "$LOG" 2>&1
echo "== d1 fix: 4K histogram-window-1 arm done $(date -Is)" >> "$LOG"
echo "###### d1 fix arms ALL DONE $(date -Is)" >> "$LOG"
