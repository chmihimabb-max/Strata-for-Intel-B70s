#!/bin/bash
# I3 oracle assembly: the real write, plus the verification pass, in one detached run.
# Nothing here writes to / (the 103 GiB target is the external SSD that already holds the pack).
set -u
REPO=/home/michael/strata-xpu/strata
OUT=/run/media/michael/2208B12208B0F63F/strata-w4a16/i3-oracle/w4a16-q4_0-dense-ours-i3-oracle.gguf
cd "$REPO" || exit 3
echo "=== df before"; df -h / /run/media/michael/2208B12208B0F63F
echo "=== date"; date -Is
echo "=== write start"
/usr/bin/python3 i3/i3_assemble_oracle.py --write --out "$OUT"
echo "=== write rc=$?"
ls -la "$OUT"
echo "=== df after write"; df -h /
echo "=== verify start"
/usr/bin/python3 i3/i3_assemble_oracle.py --verify --out "$OUT" --sweep-sample 0
echo "=== verify rc=$?"
echo "=== df after verify"; df -h /
echo "=== date"; date -Is
echo "=== DONE"
