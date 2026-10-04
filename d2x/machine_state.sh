#!/usr/bin/env bash
# D2x (card t_85e61269): the machine state before and after the measurement, into d2x/machine.txt.
# The card asks which cards were free and that anything started is stopped.
set +e
SRC=/home/michael/strata-xpu/strata
OUT=$SRC/d2x/machine.txt
source /opt/intel/oneapi/setvars.sh > /tmp/d2x-setvars2.txt 2>&1
{
  echo "== D2x machine state $(date -Is) =="
  echo "-- uptime / load"
  uptime
  echo "-- cards (sycl-ls)"
  sycl-ls
  echo "-- who holds the render nodes (fuser -v)"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1
  echo "-- containers"
  docker ps 2>&1 | head -3
  echo "-- engine processes (build-sycl/strata, serve.server)"
  pgrep -af "build-sycl/strata|serve.server" | grep -v pgrep || echo "   none"
  echo "-- port 8099"
  ss -ltnp 2>/dev/null | grep 8099 || echo "   8099 free"
  echo "-- this run's own binary"
  ls -la "$SRC/build-sycl/mmvq_xmx_price"
  md5sum "$SRC/build-sycl/mmvq_xmx_price"
} > "$OUT" 2>&1
cat "$OUT"
