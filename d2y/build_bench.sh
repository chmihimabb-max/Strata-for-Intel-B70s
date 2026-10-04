#!/usr/bin/env bash
# D2y (card t_ba006576): build the microbench.  native_mmvq.cu gained the exact-rows pricing knob, so
# strata_kernels relinks too (make handles the dependency).
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2y/build.log
source /opt/intel/oneapi/setvars.sh > /tmp/d2y-setvars.txt 2>&1
{
  echo "###### d2y bench build $(date -Is)"
  cmake -S "$SRC" -B "$SRC/build-sycl" >> "$LOG" 2>&1
  echo "###### cmake rc=$?"
  make -C "$SRC/build-sycl" mmvq_xmx_price -j8 2>&1 | tail -40
  echo "###### make rc=$? at $(date -Is)"
  ls -la "$SRC/build-sycl/mmvq_xmx_price"
} >> "$LOG" 2>&1
tail -25 "$LOG"
