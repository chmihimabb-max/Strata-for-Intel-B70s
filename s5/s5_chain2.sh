#!/usr/bin/env bash
# S5 chain 2 (the rest of chain1): the ON arm and the two --short-read 0 probes.
set +e
SRC=/home/michael/strata-xpu/strata
GEN=$SRC/s5/prompts/gen-ctx4096.txt
echo "=== S5 chain2 start $(date -Is)"
bash "$SRC/s5/s5_drive.sh" s5-pc6      "$GEN"
bash "$SRC/s5/s5_drive.sh" s5-pc6-ns   "$GEN" --extra "--short-read 0"
bash "$SRC/s5/s5_drive.sh" s5-pc6-ns-c "$GEN" --extra "--short-read 0"
echo "=== S5 chain2 done $(date -Is)"
