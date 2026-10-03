#!/usr/bin/env bash
# P5 (card t_bc967b65), stage 2: the independent oracle at the same three lengths.
# Writes p5/oracle/oracle-<ctx>.json (raw ids + the oracle's own per-position top-N logprobs).
#   usage: bash p5/p5_chain_oracle.sh
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$R/p5/chain-oracle.log
: > "$LOG"
{
  echo "=== P5 oracle chain  $(date -Is) ==="
  for ctx in 4096 32768 131072; do
    case $ctx in 4096) c=4k; P=$R/m6c/prompts/prompt-needle-ctx4096.txt ;;
                 32768) c=32k; P=$R/m6c/prompts/prompt-ctx32768.txt ;;
                 131072) c=128k; P=$R/m6c/prompts/prompt-ctx131072.txt ;; esac
    TAG=oracle-$c
    echo "--- $TAG ctx=$ctx  $(date -Is) ---"
    bash $SRC/p5/p5_oracle_launch.sh "$TAG" "$ctx" 58210 0 0.50,0.50 2048
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "!!! $TAG: oracle did not come up (rc=$RC) -- see $R/p5/oracle/$TAG-server.log"
      bash $SRC/p5/p5_oracle_stop.sh "$TAG"
      continue
    fi
    /usr/bin/python3 $SRC/p5/p5_oracle_probe.py 58210 "$P" "$R/p5/oracle/$TAG.json" 256 5 "$TAG"
    echo "--- probe rc=$? ---"
    bash $SRC/p5/p5_oracle_stop.sh "$TAG"
  done
  echo "=== P5 oracle chain done $(date -Is) ==="
} >> "$LOG" 2>&1
