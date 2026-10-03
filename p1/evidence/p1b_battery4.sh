#!/usr/bin/env bash
# P1b (card t_58d5592c), final: the same arms again with the binary that hard-exits when the released window's GPU
# work cannot be observed complete (std::_Exit(250) from ~Verifier instead of walking into a queue release that
# spins in queueFinish).  One engine at a time.
R=/home/michael/strata-xpu
SRC=$R/strata
BIN=$SRC/build-sycl/strata
cd "$SRC" || exit 1
echo "=== P1b battery4 start $(date -Is) HEAD $(git log --oneline -1) binary $(md5sum $BIN | cut -d' ' -f1) ==="
bash "$R/p1/p1_utrace.sh" p1b-r-ut7 7 16
bash "$R/p1/p1_utrace.sh" p1b-s-ut9 9 16
bash "$R/p1/p1b_run_engine.sh" p1b-t-stall   "$BIN" 16  4096  1 0
bash "$R/p1/p1b_run_engine.sh" p1b-u-clean4k "$BIN" 150 4096  0 0
echo "=== P1b battery4 done $(date -Is) ==="
