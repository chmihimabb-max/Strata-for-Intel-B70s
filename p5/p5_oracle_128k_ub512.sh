#!/usr/bin/env bash
# P5: one bounded attempt at the 128K oracle row.
#
# The ub-2048 arm dies twice (DEVICE_LOST at ~49,152 of 129,024 prompt tokens, once inside FLASH_ATTN_EXT and
# once in mul_mat_id) and the prefill rate is falling with depth (484 -> 174 tok/s at the knee), so the working
# hypothesis is a per-kernel duration the driver will not tolerate at depth.  -ub 512 shrinks each query batch
# (and therefore each kernel) by 4x; it is also a DIFFERENT oracle instantiation, which is why the -ub 512 probe
# at 4K/32K was run first and its own band measured (the oracle flips its own choice at 0.0120-0.1731 nats).
#
# Bounded on purpose: 1500 s of prompt processing, then stop and report the wall.
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=oracle-128k-ub512
LOG=$R/p5/chain-oracle-128k-ub512.log
: > "$LOG"
{
  echo "=== P5 oracle 128K attempt at -ub 512 (bounded)  $(date -Is) ==="
  bash $SRC/p5/p5_oracle_launch.sh "$TAG" 131072 58210 0 0.50,0.50 512
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "!!! $TAG: oracle did not come up (rc=$RC)"; bash $SRC/p5/p5_oracle_stop.sh "$TAG"; exit 4
  fi
  timeout -s INT 1500 /usr/bin/python3 $SRC/p5/p5_oracle_probe.py 58210 \
    $R/m6c/prompts/prompt-ctx131072.txt $R/p5/oracle/$TAG.json 256 5 "$TAG"
  echo "--- probe rc=$? (124 = the 1500 s bound) ---"
  bash $SRC/p5/p5_oracle_stop.sh "$TAG"
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
tail -30 "$LOG"
