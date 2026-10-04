#!/usr/bin/env bash
# D2b (card t_f93760a1): the engine's own tests on the warm-up build.
#
# The card asks for `ctest -R mmvq_multi_parity` (the layout contract the warm-up launches against) and the S4
# ctest set (s4/s4_ctest.sh's convention: oneAPI sourced first, ZE_AFFINITY_MASK=0, the whole suite, D1's
# "5 failed of 49" as the baseline).  The tests link the same strata_kernels the engine does, so they are
# rebuilt first: the warm-up is a new symbol in that library.
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$SRC/d2b/runs/ctest
mkdir -p "$OUT"
source /opt/intel/oneapi/setvars.sh > "$OUT/build.log" 2>&1
cd "$SRC/build-sycl" || exit 1
make -j8 >> "$OUT/build.log" 2>&1
echo "make rc=$? (tail below)"; tail -3 "$OUT/build.log"
{
  echo "== D2b ctest $(date -Is)"
  echo "HEAD: $(cd "$SRC" && git log --oneline -1)"
  echo "engine binary: md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)  $(stat -c '%y' "$SRC/build-sycl/strata")"
  echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
} > "$OUT/notes.txt" 2>&1
export ZE_AFFINITY_MASK=0
echo "== 1. ctest -R mmvq_multi_parity (the layout contract) =="
ctest -R mmvq_multi_parity --output-on-failure > "$OUT/mmvq_multi_parity.log" 2>&1
grep -E "tests passed|tests failed|Passed|Failed|Test *#" "$OUT/mmvq_multi_parity.log" | tail -6
echo "== 2. the ones the warm-up's own path touches =="
ctest -R "mmvq|iq_multi|draft_policy|coupled_draft|native" --output-on-failure > "$OUT/mmvq-family.log" 2>&1
grep -E "tests passed|tests failed|\(Failed\)" "$OUT/mmvq-family.log" | tail -8
echo "== 3. the S4 ctest set: the full suite, ZE_AFFINITY_MASK=0 =="
ctest > "$OUT/full.log" 2>&1
grep -E "tests passed|tests failed" "$OUT/full.log" | tail -2
grep -E "\(Failed\)" "$OUT/full.log" | tail -12
echo "== logs under $OUT =="
