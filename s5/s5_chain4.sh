#!/usr/bin/env bash
# S5 chain 4: the tail-boundary probes on the UNMODIFIED S4 prompt, all with the config of record's checkpoints ON.
#   s5-pc6-ttlast    --turn-token 198  = the id at n-1 itself -> the boundary IS the end of the read: ONE segment
#                    (like the checkpoints-OFF read) and a checkpoint IS still saved -> "is the save inert?"
#   s5-pc6-tt1tok    --turn-token 248068 (<think>, 1 occurrence, at n-2) -> a 1-token tail through the WINDOWS
#   s5-pc6-tt1tok-ns the same boundary, --short-read 0 -> the 1-token tail through the BATCHED path
set +e
SRC=/home/michael/strata-xpu/strata
GEN=$SRC/s5/prompts/gen-ctx4096.txt
echo "=== S5 chain4 start $(date -Is)"
bash "$SRC/s5/s5_drive.sh" s5-pc6-ttlast    "$GEN" --extra "--turn-token 198"
bash "$SRC/s5/s5_drive.sh" s5-pc6-tt1tok    "$GEN" --extra "--turn-token 248068"
bash "$SRC/s5/s5_drive.sh" s5-pc6-tt1tok-ns "$GEN" --extra "--turn-token 248068 --short-read 0"
echo "=== S5 chain4 done $(date -Is)"
