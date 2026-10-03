#!/usr/bin/env bash
# P5: the oracle's own resolution probe -- the SAME prompts, the SAME oracle tree, at -ub 512 instead of -ub 2048.
#
# Why: I2 (i2/50_margins.py, I2-STATUS.md 5b) settled one divergence by showing the oracle itself emits a
# different token at that position when only the prompt-batch size changes.  That is the measurement of the
# oracle's OWN implementation band, and it is the threshold a divergence of ours has to exceed to be evidence.
# This runs it on P5's prompts, at the two lengths where an oracle stream exists.
#
#   usage: bash p5/p5_oracle_ub512_chain.sh
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$R/p5/chain-oracle-ub512.log
: > "$LOG"
{
  echo "=== P5 oracle ub-512 probe chain  $(date -Is) ==="
  for ctx in 4096 32768; do
    case $ctx in 4096) c=4k; P=$R/m6c/prompts/prompt-needle-ctx4096.txt ;;
                 32768) c=32k; P=$R/m6c/prompts/prompt-ctx32768.txt ;; esac
    TAG=oracle-$c-ub512
    echo "--- $TAG ctx=$ctx  $(date -Is) ---"
    bash $SRC/p5/p5_oracle_launch.sh "$TAG" "$ctx" 58210 0 0.50,0.50 512
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "!!! $TAG: oracle did not come up (rc=$RC)"
      bash $SRC/p5/p5_oracle_stop.sh "$TAG"
      continue
    fi
    /usr/bin/python3 $SRC/p5/p5_oracle_probe.py 58210 "$P" "$R/p5/oracle/$TAG.json" 256 5 "$TAG"
    bash $SRC/p5/p5_oracle_stop.sh "$TAG"
  done
  echo "=== P5 oracle ub-512 chain done $(date -Is) ==="
} >> "$LOG" 2>&1
