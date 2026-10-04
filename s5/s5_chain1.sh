#!/usr/bin/env bash
# S5 (card t_aae723be) chain 1: the id-level reproduction + the mechanism arms, one engine at a time.
#   s5-pc0        the config of record's argv + --prompt-cache 0 --prompt-cache-every 0   (checkpoints OFF)
#   s5-pc6        the config of record's argv verbatim (checkpoints ON, the served setting)
#   s5-pc6-ns     the same ON, but --short-read 0: every segment through the BATCHED prompt path
#                 (isolates the decode-window read of the turn header from the checkpoint save itself)
#   s5-pc6-ns-c   the same as -ns, a second run (repeatability)
set +e
SRC=/home/michael/strata-xpu/strata
GEN=$SRC/s5/prompts/gen-ctx4096.txt
OFF="--prompt-cache 0 --prompt-cache-every 0"

echo "=== S5 chain1 start $(date -Is)"
bash "$SRC/s5/s5_drive.sh" s5-pc0        "$GEN" --extra "$OFF"
bash "$SRC/s5/s5_drive.sh" s5-pc6        "$GEN"
bash "$SRC/s5/s5_drive.sh" s5-pc6-ns     "$GEN" --extra "--short-read 0"
bash "$SRC/s5/s5_drive.sh" s5-pc6-ns-c   "$GEN" --extra "--short-read 0"
echo "=== S5 chain1 done $(date -Is)"
