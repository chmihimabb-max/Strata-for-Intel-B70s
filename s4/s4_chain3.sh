#!/usr/bin/env bash
# S4 (card t_30d9ccfb): the follow-up probe for §7 - is the "checkpoints on vs off" text difference length-dependent?
# Two fresh servers on the FIXED binary, one 4,096-token request each: the config of record (checkpoints ON, no
# mid-prompt checkpoint can fire at 4K) and the same + --prompt-cache 0 --prompt-cache-every 0.
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$SRC/s4/runs/chain3-4k.log
: > "$LOG"
set +e
echo "###### S4 chain3 (4K on/off) $(date -Is)" >> "$LOG"
bash "$SRC/s4/s4_serve_arm.sh" s4-ckpt-4k 4096:64 >> "$LOG" 2>&1
bash "$SRC/s4/s4_serve_arm.sh" s4-pc0-4k --extra "--prompt-cache 0 --prompt-cache-every 0" 4096:64 >> "$LOG" 2>&1
echo "###### S4 chain3 done $(date -Is)" >> "$LOG"
