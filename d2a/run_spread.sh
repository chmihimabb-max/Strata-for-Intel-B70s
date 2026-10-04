#!/bin/bash
# D2a: the kernel-level spread, twice - a fresh SYCL cache (every specialization's first launch carries the
# program build) and the same cache warm (first launch is steady state).  No engine, one card's worth of work.
set +e
SRC=/home/michael/strata-xpu/strata
BIN=$SRC/build-sycl/mmvq_multi_spread
CACHE=/tmp/d2a-cache
REPS=${1:-16}
rm -rf "$CACHE"
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHE"
echo "###### cold cache $(date -Is) reps=$REPS cache=$CACHE"
"$BIN" --reps "$REPS" 2>&1 | tee "$SRC/d2a/spread-cold.txt"
echo "###### warm cache $(date -Is)"
"$BIN" --reps "$REPS" 2>&1 | tee "$SRC/d2a/spread-warm.txt"
echo "###### done $(date -Is); cache entries: $(find $CACHE -name '0.src' | wc -l)"
