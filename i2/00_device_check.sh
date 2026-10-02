#!/usr/bin/env bash
# I2: pre-run device check (PLAN 9 rule 1/5, BRIEF 3b) - prove no other process holds either B70
# before every measurement.
#   usage: bash i2/00_device_check.sh <tag>
R=/home/michael/strata-xpu
I2=$R/strata/i2
TAG=${1:?tag}
OUT=$I2/$TAG-device-check.txt
{
  echo "=============================================================="
  echo "I2 device check  tag=$TAG   $(date -Is)"
  echo "-- engines that would hold a card --"
  pgrep -a -f 'llama-server|llama-cli|build-sycl/strata|serve/server.py|vllm' || echo "   none"
  echo "-- who holds the render nodes (fuser) --"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  echo "-- per-process VRAM held right now, driver DRM fdinfo (m6_occupancy.py) --"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1
  echo "-- containers / other engines --"
  docker ps 2>&1 | head -3
  ollama ps 2>&1 | head -2
  echo "-- load / memory / disk --"
  cat /proc/loadavg
  free -g | head -2
  df -h / | tail -1
  echo "== end of the device check =="
} > "$OUT" 2>&1
cat "$OUT"
