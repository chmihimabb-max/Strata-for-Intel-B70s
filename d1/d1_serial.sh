#!/usr/bin/env bash
# D1: the serial driver for the rest of the card's arms, so the GPUs are never idle between chains.  It waits
# for the guard chain to close its own log line, then runs: the engine's test suite, the spec sweep, the
# histogram arms, and the closure control.  The variants chain is NOT here: its width comes out of the sweep,
# so it is a deliberate second step.
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$R/d1/chain.log
set +e
echo "###### d1 serial driver started $(date -Is)" >> "$LOG"
until grep -q "###### d1 chain group guard done" "$LOG" 2>/dev/null; do sleep 10; done
echo "###### d1 serial driver: guard group closed, ctest next $(date -Is)" >> "$LOG"
bash "$SRC/d1/d1_ctest.sh" >> "$R/d1/d1_ctest.out" 2>&1
echo "###### d1 serial driver: ctest done $(date -Is)" >> "$LOG"
bash "$SRC/d1/d1_spec.sh" all >> "$R/d1/d1_serial.out" 2>&1
echo "###### d1 serial driver: spec sweep done $(date -Is)" >> "$LOG"
bash "$SRC/d1/d1_hist_chain.sh" all >> "$R/d1/d1_serial.out" 2>&1
echo "###### d1 serial driver: histogram done $(date -Is)" >> "$LOG"
bash "$SRC/d1/d1_closure.sh" 4096 >> "$R/d1/d1_serial.out" 2>&1
bash "$SRC/d1/d1_closure.sh" 32768 >> "$R/d1/d1_serial.out" 2>&1
echo "###### d1 serial driver: ALL DONE $(date -Is)" >> "$LOG"
