#!/usr/bin/env bash
# P10 (t_1d052912) chain 3: the like-for-like concurrency pair, with --prompt-cache 0.
#
# The first 1x2 arm (the config of record verbatim, checkpoints ON) died mid-request:
#   strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
#   strata serve: checkpoint save: conversation snapshot running-state copy: invalid argument
#   strata serve: saving a checkpoint part failed
# - a cross-card running-state copy for the checkpoint, which the split cannot do on this pair (S1 measured the
# same failure on a device0 -> device1 copy: no peer path).  The single-card instances save their checkpoints
# fine, so the two arms were also not like-for-like (the 2i arms each wrote 2 checkpoints inside prompt_ms).
# Both arms below pass --prompt-cache 0 --prompt-cache-every 0, exactly as every direct-serve arm of this card.
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$R/p10/chain3.log
: > "$LOG"

echo "=== P10 chain3 start $(date -Is) ===" >> "$LOG"
echo "--- one two-card instance, two concurrent requests, pc0 ---" >> "$LOG"
bash "$SRC/p10/p10_two.sh" 1x2-32k-pc0 1x2 19 "" 32768 256 "" pc0 >> "$LOG" 2>&1

echo "--- two single-card instances, 9/9 workers, pc0 ---" >> "$LOG"
bash "$SRC/p10/p10_two.sh" 2i-32k-9-pc0 2i 9 9 32768 256 "" pc0 >> "$LOG" 2>&1

echo "=== P10 chain3 done $(date -Is) ===" >> "$LOG"
