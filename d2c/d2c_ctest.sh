#!/usr/bin/env bash
# D2c (card t_c7d8cd86): the engine's own tests on the warm-up build, the S4 convention (oneAPI sourced first,
# ZE_AFFINITY_MASK=0, the whole suite; D1/S4's "5 failed of 49" is the baseline).
#
# D2c adds a new symbol (`decode_warmup`) to strata_kernels and touches no kernel, so the tests are rebuilt first
# (they link the library) and the family set is the tests of the kernels this pass warms: bf16_gemv, elementwise,
# quantize_act, s2_expert_grouped, the mmvq/iq layouts, the sampler, the QSA/GDN/GR families, the router and the
# expert harnesses.
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$SRC/d2c/runs/ctest
mkdir -p "$OUT"
source /opt/intel/oneapi/setvars.sh > "$OUT/build.log" 2>&1
cd "$SRC/build-sycl" || exit 1
make -j8 >> "$OUT/build.log" 2>&1
echo "make rc=$? (tail below)"; tail -3 "$OUT/build.log"
{
  echo "== D2c ctest $(date -Is)"
  echo "HEAD: $(cd "$SRC" && git log --oneline -1)"
  echo "engine binary: md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)  $(stat -c '%y' "$SRC/build-sycl/strata")"
  echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
} > "$OUT/notes.txt" 2>&1
export ZE_AFFINITY_MASK=0
echo "== 1. ctest -R mmvq_multi_parity (the layout contract D2b's pass launches against) =="
ctest -R mmvq_multi_parity --output-on-failure > "$OUT/mmvq_multi_parity.log" 2>&1
grep -E "tests passed|tests failed|Passed|Failed|Test *#" "$OUT/mmvq_multi_parity.log" | tail -6
echo "== 2. the families whose kernels this pass warms =="
ctest -R "bf16_gemv|elementwise|quantize_act|s2_expert_grouped|mmvq|iq_multi|sampler|qsa_parity|gdn_parity|gr_parity|router_top10|expert" \
      --output-on-failure > "$OUT/warmed-families.log" 2>&1
grep -E "tests passed|tests failed|\(Failed\)" "$OUT/warmed-families.log" | tail -10
echo "== 3. the S4 ctest set: the full suite, ZE_AFFINITY_MASK=0 =="
ctest > "$OUT/full.log" 2>&1
grep -E "tests passed|tests failed" "$OUT/full.log" | tail -2
grep -E "\(Failed\)" "$OUT/full.log" | tail -12
echo "== logs under $OUT =="
