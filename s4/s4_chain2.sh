#!/usr/bin/env bash
# S4 (card t_30d9ccfb): the second chain - the determinism question and the boundary re-proof, three server
# lifetimes one after the other (one engine at a time; each arm stops its own server).
#
#   1. s4-fix-32k-2   the config of record, 32,277 tokens, again on the FIXED binary: is the served path
#                     reproducible run-to-run with checkpoints on?
#   2. s4-pc0-32k-2   the same request with --prompt-cache 0 --prompt-cache-every 0, again: reproducible with
#                     checkpoints off?  (1 and 2 together say whether a text difference can be attributed to
#                     anything, and to what.)
#   3. s4-fix-16400   the request that FAILED before the fix (16,452 prompt tokens, the 16,384-token checkpoint),
#                     fresh server: it must now succeed on the config of record verbatim.
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$SRC/s4/runs/chain2.log
mkdir -p "$SRC/s4/runs"
: > "$LOG"
set +e
echo "###### S4 chain2 started $(date -Is)  HEAD $(cd "$SRC" && git log --oneline -1)" >> "$LOG"
echo "--- arm 1: s4-fix-32k-2" >> "$LOG"
bash "$SRC/s4/s4_serve_arm.sh" s4-fix-32k-2 32768:256 >> "$LOG" 2>&1
echo "--- arm 2: s4-pc0-32k-2" >> "$LOG"
bash "$SRC/s4/s4_serve_arm.sh" s4-pc0-32k-2 --extra "--prompt-cache 0 --prompt-cache-every 0" 32768:256 >> "$LOG" 2>&1
echo "--- arm 3: s4-fix-16400" >> "$LOG"
bash "$SRC/s4/s4_serve_arm.sh" s4-fix-16400 16400:64 >> "$LOG" 2>&1
echo "###### S4 chain2 ALL DONE $(date -Is)" >> "$LOG"
