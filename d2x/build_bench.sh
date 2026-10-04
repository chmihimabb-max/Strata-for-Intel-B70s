#!/usr/bin/env bash
# D2x (card t_85e61269): build the microbench.  Regenerates the build dir (the target is added to
# CMakeLists.txt) and builds only this target; strata_kernels is already built.
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2x/build.log
source /opt/intel/oneapi/setvars.sh > /tmp/d2x-setvars.txt 2>&1
{
  echo "###### d2x bench build $(date -Is)"
  cmake -S "$SRC" -B "$SRC/build-sycl" >> "$LOG" 2>&1
  echo "###### cmake rc=$?"
  make -C "$SRC/build-sycl" mmvq_xmx_price -j8 2>&1
  echo "###### make rc=$? at $(date -Is)"
  ls -la "$SRC/build-sycl/mmvq_xmx_price"
} >> "$LOG" 2>&1
tail -40 "$LOG"
