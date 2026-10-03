#!/usr/bin/env bash
# P1b (card t_58d5592c): the whole measurement battery, in the order the card needs it.
# One engine at a time (both B70s), so this is sequential by construction.
R=/home/michael/strata-xpu
SRC=$R/strata
BIN=$SRC/build-sycl/strata
cd "$SRC" || exit 1
echo "=== P1b battery start $(date -Is) HEAD $(git log --oneline -1) ==="
# 4. the instrument, S3UT's three stalling modes (the deliverable P2 waits for)
bash "$R/p1/p1_utrace.sh" p1b-g-ut7  7 16
bash "$R/p1/p1_utrace.sh" p1b-h-ut9  9 16
bash "$R/p1/p1_utrace.sh" p1b-i-ut1  1 16
# 2. a second ask after a failed window (the guard)
bash "$R/p1/p1b_run_engine.sh" p1b-l-secondask "$BIN" 16 4096 1 1
# 3. the clean path: same prompt, greedy, token ids vs P1's HEAD arms (4K 150 tokens, 32K 256 tokens)
bash "$R/p1/p1b_run_engine.sh" p1b-j-clean4k  "$BIN" 150 4096    0 0
bash "$R/p1/p1b_run_engine.sh" p1b-k-clean32k "$BIN" 256 32768   0 0
echo "=== P1b battery done $(date -Is) ==="
