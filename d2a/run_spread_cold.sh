#!/bin/bash
# D2a: the build cost, reproduced on demand.  A fresh SYCL program cache alone is not a cold compiler: the
# Intel NEO/IGC native-binary cache (~/.cache/neo_compiler_cache) still holds the built kernels.  Redirect BOTH
# to fresh directories and the specializations have to be built, first launch by first launch - which is the
# engine's own condition on its first run of the layout.
set +e
SRC=/home/michael/strata-xpu/strata
rm -rf /tmp/d2a-cold2-cache /tmp/d2a-cold2-neo
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/d2a-cold2-cache
export NEO_CACHE_PERSISTENT=1 NEO_CACHE_DIR=/tmp/d2a-cold2-neo
echo "###### both caches fresh $(date -Is): SYCL_CACHE_DIR=$SYCL_CACHE_DIR NEO_CACHE_DIR=$NEO_CACHE_DIR"
"$SRC/build-sycl/mmvq_multi_spread" --reps 16 2>&1 | tee "$SRC/d2a/spread-cold-fully.txt"
echo "###### done $(date -Is)"
