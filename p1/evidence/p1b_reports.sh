#!/usr/bin/env bash
# P1b: where the instrument's report actually lands.
R=/home/michael/strata-xpu
SRC=$R/strata
echo "== mode 7 (--device-timing only): is there a timing summary in unitrace's own stdout?"
for t in p1b-f-ut7 p1b-n-ut7 p1b-r-ut7; do
  f=$R/p1/logs/$t-out.log
  echo "-- $t ($(wc -c < "$f") bytes)"
  grep -nE "Device Timing|device timing|Timing|Kernel|Summary|report|Total|Gpu|GPU" "$f" | head -8
done
echo
echo "== what is inside a chrome JSON: event names, sample"
f=$SRC/strata.848240.json
echo "file: $f ($(stat -c '%s' "$f") bytes)"
grep -o '"name": "[^"]*"' "$f" | sort | uniq -c | sort -rn | head -12
echo "-- a device (ph X) sample:"
grep -o '{"ph": "X"[^}]*}' "$f" | head -3
