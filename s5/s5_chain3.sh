#!/usr/bin/env bash
# S5 chain 3: the one-card controls, the "no boundary found" control, the P1 (checkpoint-without-a-split) pair and
# the --spec 1 pair.
set +e
SRC=/home/michael/strata-xpu/strata
GEN=$SRC/s5/prompts/gen-ctx4096.txt
GEN1=$SRC/s5/prompts/gen-ctx4096-p1.txt
OFF="--prompt-cache 0 --prompt-cache-every 0"
echo "=== S5 chain3 start $(date -Is)"
# the same prompt, one card (lead: is the split involved at all?)
bash "$SRC/s5/s5_drive.sh" s5-pc0-1card "$GEN" --onecard 0 --extra "$OFF"
bash "$SRC/s5/s5_drive.sh" s5-pc6-1card "$GEN" --onecard 0
# the config of record's checkpoints ON but with NO turn boundary found (--turn-token -1): no split, no checkpoint
bash "$SRC/s5/s5_drive.sh" s5-pc6-ttnon "$GEN" --extra "--turn-token -1"
# P1: the S4 prompt + ONE unused token appended at the end -> the boundary is at n-1, so the read is ONE segment
# and a checkpoint is still saved (a checkpoint does not have to split the read)
/usr/bin/python3 "$SRC/s5/s5_p1.py" --gen "$GEN" --out "$GEN1" --report "$GEN1.report.json" >> "$SRC/s5/runs/chain3.log" 2>&1
RARE=$(grep -o '"rare_id": [0-9]*' "$GEN1.report.json" | awk '{print $2}')
bash "$SRC/s5/s5_drive.sh" s5-p1-pc0       "$GEN1" --extra "$OFF"
bash "$SRC/s5/s5_drive.sh" s5-p1-pc6-last  "$GEN1" --extra "--turn-token $RARE"
# the window-shape discriminator: no drafts at all (T = 1 every window)
bash "$SRC/s5/s5_drive.sh" s5-pc0-s1 "$GEN" --extra "$OFF --spec 1"
bash "$SRC/s5/s5_drive.sh" s5-pc6-s1 "$GEN" --extra "--spec 1"
echo "=== S5 chain3 done $(date -Is)"
