#!/usr/bin/env bash
# D1 (card t_d9ffcf38): the engine's own test suite on the shipped build, three ways.  ZE_AFFINITY_MASK=0
# (the M6/P3 convention: one card, since the suite's parity tests are single-device) and the oneAPI environment
# sourced first (without it the four KV parity tests fail instantly on libsycl.so.9 - a known env artefact).
#
#   1. the six capture/replay tests, with the NEW default (the graph path ON)
#   2. the full suite, with the NEW default
#   3. the full suite with STRATA_SYCL_GRAPH=0 (the closure path, i.e. what P3/P9 measured)
#
#   bash d1/d1_ctest.sh
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$R/d1/runs/ctest
mkdir -p "$OUT"
set +e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ZE_AFFINITY_MASK=0
cd "$SRC/build-sycl" || exit 1
{
  echo "== d1 ctest $(date -Is)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "engine binary: md5 $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
} > "$OUT/notes.txt"
echo "== 1. the six capture tests, graph path at its new default"
ctest -R "gr_parity|qsa_parity|ple_parity|sampler_parity|shared_expert_parity" --output-on-failure \
  > "$OUT/capture-graphdefault.log" 2>&1
tail -20 "$OUT/capture-graphdefault.log"
echo "== 2. the full suite, graph path at its new default"
ctest > "$OUT/full-graphdefault.log" 2>&1
grep -E "tests passed|tests failed" "$OUT/full-graphdefault.log" | tail -2
grep -E "Failed|\(Failed\)" "$OUT/full-graphdefault.log" | tail -10
echo "== 3. the full suite, closure path (STRATA_SYCL_GRAPH=0)"
STRATA_SYCL_GRAPH=0 ctest > "$OUT/full-closure.log" 2>&1
grep -E "tests passed|tests failed" "$OUT/full-closure.log" | tail -2
grep -E "Failed|\(Failed\)" "$OUT/full-closure.log" | tail -10
echo "== logs under $OUT =="
