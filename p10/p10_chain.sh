#!/usr/bin/env bash
# P10 (t_1d052912) chain 1: the six like-for-like arms.
#
#   single card (ZE_AFFINITY_MASK=0, no --layer-split) and the two-card split (mask unset, --layer-split auto),
#   each at 4096 / 32768 / 131072 prompt tokens, 256 generated tokens, the config of record otherwise.
#
# Ends by printing each arm's summary row (p10_summary.py) into the chain log.
R=/home/michael/strata-xpu
SRC=$R/strata
LOG=$R/p10/chain1.log
MAXNEW=256
: > "$LOG"

echo "=== P10 chain1 start $(date -Is) ===" >> "$LOG"
for spec in "1c-4k 4096 --onecard 0" "1c-32k 32768 --onecard 0" "2c-4k 4096" "2c-32k 32768" \
            "1c-128k 131072 --onecard 0" "2c-128k 131072"; do
  set -- $spec
  TAG=$1; CTX=$2; shift 2
  echo "--- arm $TAG ctx=$CTX $* ---" >> "$LOG"
  bash "$SRC/p10/p10_run_arm.sh" "$TAG" "$CTX" "$MAXNEW" "$@" >> "$LOG" 2>&1
  echo "--- arm $TAG rc=$? ---" >> "$LOG"
  /usr/bin/python3 "$SRC/p10/p10_summary.py" "$R/p10/runs/$TAG" >> "$LOG" 2>&1
  /usr/bin/python3 "$SRC/p10/p10_cpu_report.py" "$R/p10/runs/$TAG" >> "$LOG" 2>&1
done
echo "=== P10 chain1 done $(date -Is) ===" >> "$LOG"
