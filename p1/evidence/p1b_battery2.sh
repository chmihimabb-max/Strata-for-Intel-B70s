#!/usr/bin/env bash
# P1b (card t_58d5592c), second half: the new frame under the instrument, then the arms the first battery missed.
#
# The first battery's last three arms died on `libsycl.so.9: cannot open shared object file` (p1b_run_engine.sh was
# missing the oneAPI setvars; fixed).  The instrument arms (mode 7 was an environment casualty too: 568 MiB of VRAM
# free on entry, `memcpy failed ... OUT_OF_DEVICE_MEMORY` during prefill, exit 1 in 33 s) still have no report in
# any mode, so this half also grabs the frame the host is in now, with gdb as the engine's parent under
# unitrace --device-timing.
R=/home/michael/strata-xpu
SRC=$R/strata
BIN=$SRC/build-sycl/strata
cd "$SRC" || exit 1
echo "=== P1b battery2 start $(date -Is) HEAD $(git log --oneline -1) ==="
bash "$R/p1/p1_gdb.sh" p1b-m-gdb7 200 7
bash "$R/p1/p1b_run_engine.sh" p1b-l-secondask "$BIN" 16  4096  1 1
bash "$R/p1/p1b_run_engine.sh" p1b-j-clean4k  "$BIN" 150 4096  0 0
bash "$R/p1/p1b_run_engine.sh" p1b-k-clean32k "$BIN" 256 32768 0 0
echo "=== P1b battery2 done $(date -Is) ==="
