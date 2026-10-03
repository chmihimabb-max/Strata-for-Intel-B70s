#!/usr/bin/env bash
# P10 (t_1d052912): collect the raw output the card's ACCEPTANCE asks to be pasted, one block per arm.
#
#   bash p10/p10_evidence.sh > p10/P10-EVIDENCE.txt
#
# Every line under "the engine's own lines" is copied verbatim from the arm's err.txt/out.txt (the engine's own
# --stats output); the CPU blocks come from the two samplers; the VRAM/RSS numbers come from the driver's fdinfo.
R=/home/michael/strata-xpu
SRC=$R/strata
ARMS=${*:-"1c-4k 2c-4k 1c1-4k 1c-32k 2c-32k 1c-128k 2c-128k"}
for TAG in $ARMS; do
  D=$R/p10/runs/$TAG
  echo "======================================================================================"
  echo "ARM $TAG      dir $D"
  echo "======================================================================================"
  [ -d "$D" ] || { echo "  (no such arm)"; continue; }
  grep -hE "^P10 arm|^ctx=|^onecard=" "$D/log.txt" 2>/dev/null | head -3
  echo "-- the exact command --"
  grep -h "^cd $D &&" "$D/log.txt" 2>/dev/null
  echo
  echo "-- the engine's own lines --"
  grep -hE "strata generate: expert cache auto|strata generate: expert cache [0-9]+ slots|pre-filled|layer split auto: K=|layer split across|layer split: [0-9.]+% of the experts resident|token graph hit path|of the experts resident|expert-pool workers|VRAM free with everything loaded|experts via mmap|SSD is kept awake" \
      "$D/err.txt" 2>/dev/null
  grep -hE "^INFO context=" "$D/out.txt" "$D/err.txt" 2>/dev/null | head -1
  grep -hE "strata decode timing" "$D/err.txt" 2>/dev/null | tail -1
  grep -hE "strata serve: prompt [0-9]+ tokens" "$D/err.txt" 2>/dev/null | tail -1
  grep -hE "strata serve: decode expert cache hit rate" "$D/err.txt" 2>/dev/null | tail -1
  grep -hE "strata serve: expert tiers" "$D/err.txt" 2>/dev/null | tail -1
  grep -hE "strata serve: KV streaming" "$D/err.txt" 2>/dev/null | tail -1
  grep -hE "^DONE" "$D/out.txt" 2>/dev/null | tail -1
  grep -hE "^PP " "$D/out.txt" 2>/dev/null | tail -2
  echo
  echo "-- per-thread CPU: top -b -H, busiest sample --"
  /usr/bin/python3 "$SRC/p10/p10_top_evidence.py" "$D" 2>&1 | head -26
  echo
  echo "-- per-thread CPU by phase (p10_threads.py deltas, phase boundaries from the engine's own DONE line) --"
  /usr/bin/python3 "$SRC/p10/p10_cpu_report.py" "$D" 2>&1
  echo
  echo "-- VRAM / RSS peaks (driver fdinfo) --"
  grep -h "monitor" "$D/log.txt" 2>/dev/null || echo "  (no monitor lines)"
  /usr/bin/python3 "$SRC/p10/p10_summary.py" "$D" 2>/dev/null | grep -E "peak_|vram_free_loaded|req_file|req_ram|tiers_line"
  echo
done
