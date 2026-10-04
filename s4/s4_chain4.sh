#!/usr/bin/env bash
# S4 (card t_30d9ccfb): the §7 dial - is the "checkpoints on vs off changes the served text" difference the GRAPH
# path (D1's new default) rather than the checkpoint path?  Same two 4,096-token arms as chain3, with
# STRATA_SYCL_GRAPH=0 (the closure path), exported so the server hands it to the engine (serve.server's child_env
# copies os.environ).  If the two texts match here but not with the graph path on, the interaction named by the
# follow-up card is the graph path, not the checkpoints.
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$SRC/s4/runs/chain4-4k-closure.log
: > "$LOG"
set +e
export STRATA_SYCL_GRAPH=0
echo "###### S4 chain4 (4K, STRATA_SYCL_GRAPH=0, ckpt on vs off) $(date -Is)" >> "$LOG"
bash "$SRC/s4/s4_serve_arm.sh" s4-closure-ckpt-4k 4096:64 >> "$LOG" 2>&1
bash "$SRC/s4/s4_serve_arm.sh" s4-closure-pc0-4k --extra "--prompt-cache 0 --prompt-cache-every 0" 4096:64 >> "$LOG" 2>&1
echo "###### S4 chain4 done $(date -Is)" >> "$LOG"
