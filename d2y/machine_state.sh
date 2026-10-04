#!/usr/bin/env bash
# D2y: which cards were free and what the box looked like while the microbench ran.
set +e
SRC=/home/michael/strata-xpu/strata
{
  echo "###### d2y machine state at $(date -Is)"
  echo "### /dev/dri"
  ls -l /dev/dri 2>&1
  echo "### render nodes and their device ids"
  for c in /sys/class/drm/card*; do
    printf '%s vendor=%s device=%s\n' "$c" "$(cat $c/device/vendor 2>/dev/null)" "$(cat $c/device/device 2>/dev/null)"
  done
  echo "### /proc/driver/nvidia (absent on this box)"
  ls /proc/driver/nvidia 2>&1 | head -2
  echo "### load, memory"
  uptime
  free -g | head -2
  echo "### anything of mine still running (engine, server, bench, container)"
  pgrep -af "build-sycl/strata|serve.server|mmvq_xmx_price|docker" | head -10
  echo "### the bench binary this card measured with"
  ls -l "$SRC/build-sycl/mmvq_xmx_price" "$SRC/build-sycl/strata" 2>&1
  md5sum "$SRC/build-sycl/mmvq_xmx_price" 2>&1
} > "$SRC/d2y/machine.txt" 2>&1
cat "$SRC/d2y/machine.txt"
