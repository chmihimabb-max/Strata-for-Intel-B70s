#!/bin/bash
# D2a: build the spread harness only (never the engine binary: build-sycl/strata must stay e841fd06...)
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2a/build.log
source /opt/intel/oneapi/setvars.sh > "$LOG" 2>&1
{
  echo "###### mmvq_multi_spread build $(date -Is)"
  echo "engine md5 before: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  cmake -S "$SRC" -B "$SRC/build-sycl" >> "$LOG" 2>&1
  echo "###### cmake rc=$?"
  make -C "$SRC/build-sycl" mmvq_multi_spread -j8
  echo "###### make rc=$? at $(date -Is)"
  echo "engine md5 after: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  ls -la "$SRC/build-sycl/mmvq_multi_spread" 2>&1
} >> "$LOG" 2>&1
tail -25 "$LOG"
