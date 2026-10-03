#!/usr/bin/env bash
# P1b (card t_58d5592c), final binary (f21ba10 + the released-teardown skip): the instrument retry and the sanity arms.
R=/home/michael/strata-xpu
SRC=$R/strata
BIN=$SRC/build-sycl/strata
cd "$SRC" || exit 1
echo "=== P1b battery3 start $(date -Is) HEAD $(git log --oneline -1) binary $(stat -c '%y %s' $BIN) ==="
# mode 7 on a clean device (the first attempt died on 568 MiB of free VRAM at load: memcpy failed / OUT_OF_DEVICE_MEMORY)
bash "$R/p1/p1_utrace.sh" p1b-n-ut7 7 16
# the untraced failure path with the final binary
bash "$R/p1/p1b_run_engine.sh" p1b-o-stall  "$BIN" 16  4096  1 0
# the clean 4K arm with the final binary (token ids must still be P1's)
bash "$R/p1/p1b_run_engine.sh" p1b-p-clean4k "$BIN" 150 4096 0 0
# chrome logging with the final binary (it had no report with f21ba10)
bash "$R/p1/p1_utrace.sh" p1b-q-ut9 9 16
echo "=== P1b battery3 done $(date -Is) ==="
