#!/usr/bin/env bash
# S5 chain 5: the 32K id-level pair (the S4 length): the same prompt token ids the served arm used (32,277), 256
# greedy tokens, checkpoints off vs the config of record's own.
set +e
SRC=/home/michael/strata-xpu/strata
GEN=$SRC/s5/prompts/gen-ctx32768.txt
echo "=== S5 chain5 (32K id pair) start $(date -Is)"
bash "$SRC/s5/s5_drive.sh" s5-pc0-32k "$GEN" --extra "--prompt-cache 0 --prompt-cache-every 0" --timeout 2400
bash "$SRC/s5/s5_drive.sh" s5-pc6-32k "$GEN" --timeout 2400
echo "=== S5 chain5 done $(date -Is)"
