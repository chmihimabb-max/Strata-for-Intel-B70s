#!/usr/bin/env bash
# P1b: unitrace DID write chrome JSON for the device-instrumented arms - into the APP's cwd ($SRC), named
# strata.<app pid>.json, not into the harness's --output/trace dir.  Map the files to the arms and check them.
R=/home/michael/strata-xpu
SRC=$R/strata
echo "== the files"
ls -la "$SRC"/strata.*.json
echo
echo "== which arm each pid is (the harness logs record 'engine pid N')"
for l in "$R"/p1/logs/p1b-*-ut*.log; do
  pid=$(grep -o 'engine pid [0-9]*' "$l" | head -1 | cut -d' ' -f3)
  [ -n "$pid" ] || continue
  f="$SRC/strata.$pid.json"
  if [ -f "$f" ]; then sz=$(stat -c '%s' "$f"); else sz="MISSING"; fi
  echo "$(basename "$l")  engine pid=$pid  trace file bytes=$sz"
done
echo
echo "== is each file complete JSON, and does it carry the verify window's device kernels?"
for f in "$SRC"/strata.*.json; do
  echo "-- $(basename "$f") $(stat -c '%s bytes' "$f")"
  tail -c 60 "$f" | tr -d '\n'; echo
  echo "   wait_flag_ge mentions: $(grep -c -o 'wait_flag_ge' "$f" 2>/dev/null || echo 0)"
  echo "   device process keys:   $(grep -o '\"DEVICE[^\"]*' "$f" | sort -u | head -2 | tr '\n' ' ')"
  echo "   event count (ph\":\"X\"): $(grep -c -o '\"ph\": \"X\"' "$f" 2>/dev/null || echo 0)"
done
