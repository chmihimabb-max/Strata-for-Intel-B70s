#!/usr/bin/env bash
# P10 (t_1d052912) chain 2: the asymmetry control and the two-instance / concurrency arms.
#
#   1. 1c1-4k      a single-card instance on CARD 1 (the same arm that ran on card 0) - P8 measured card 1 as a
#                 half-bandwidth device, so "two instances, one per card" cannot be assumed to be 2x card 0.
#   2. 2i-32k-9     two serve.server instances, one per card, --pool-workers 9 each (38 workers would
#                 oversubscribe 20 cores)
#   3. 2i-32k-19    the same with the default 19 each (the oversubscribed control the card asks for)
#   4. 1x2-32k      ONE two-card serve.server, two requests fired at it concurrently (it serves one sequence at
#                 a time, so this measures the queue: the like-for-like aggregate for the two-card option)
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$R/p10/chain2.log
: > "$LOG"

echo "=== P10 chain2 start $(date -Is) ===" >> "$LOG"

echo "--- arm 1c1-4k (single card 1) ---" >> "$LOG"
bash "$SRC/p10/p10_run_arm.sh" 1c1-4k 4096 256 --onecard 1 >> "$LOG" 2>&1
/usr/bin/python3 "$SRC/p10/p10_summary.py" "$R/p10/runs/1c1-4k" >> "$LOG" 2>&1
/usr/bin/python3 "$SRC/p10/p10_cpu_report.py" "$R/p10/runs/1c1-4k" >> "$LOG" 2>&1

for spec in "2i-32k-9 9 9" "2i-32k-19 19 19"; do
  set -- $spec
  TAG=$1; PA=$2; PB=$3
  echo "--- two instances $TAG pools $PA/$PB ---" >> "$LOG"
  bash "$SRC/p10/p10_two.sh" "$TAG" 2i "$PA" "$PB" 32768 256 >> "$LOG" 2>&1
  for idx in 0 1; do
    /usr/bin/python3 "$SRC/p10/p10_cpu_report.py" "$R/p10/runs/$TAG" "$idx" >> "$LOG" 2>&1
  done
done

echo "--- one two-card instance, two concurrent requests ---" >> "$LOG"
bash "$SRC/p10/p10_two.sh" 1x2-32k 1x2 19 "" 32768 256 >> "$LOG" 2>&1
/usr/bin/python3 "$SRC/p10/p10_cpu_report.py" "$R/p10/runs/1x2-32k" 0 >> "$LOG" 2>&1

echo "=== P10 chain2 done $(date -Is) ===" >> "$LOG"
