#!/usr/bin/env bash
# P1b, last check: the same failure and clean arms with the shipped binary (std::exit + bounded finalizer window).
R=/home/michael/strata-xpu
SRC=$R/strata
BIN=$SRC/build-sycl/strata
cd "$SRC" || exit 1
echo "=== P1b battery5 start $(date -Is) HEAD $(git log --oneline -1) binary $(md5sum $BIN | cut -d' ' -f1) ==="
bash "$R/p1/p1b_run_engine.sh" p1b-aa-stall   "$BIN" 16  4096  1 0
bash "$R/p1/p1b_run_engine.sh" p1b-ab-clean4k "$BIN" 150 4096  0 0
echo "=== P1b battery5 done $(date -Is) ==="
