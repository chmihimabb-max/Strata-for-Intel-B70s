#!/usr/bin/env bash
# D2y: are quantize_act_parity and iq_multi_parity failing because of this card's engine edit, or were they
# already failing at HEAD?  Stash the two engine files, rebuild, run the same three tests, restore.
set +e
SRC=/home/michael/strata-xpu/strata
OUT=$SRC/d2y/D2Y-TESTS-HEAD.txt
source /opt/intel/oneapi/setvars.sh > /tmp/d2y-setvars.txt 2>&1
cd "$SRC"
{
  echo "###### d2y: the same three tests with the D2y engine edit STASHED ($(date -Is))"
  git stash push -- src/kernels/cuda/native_mmvq.cu include/strata/kernels/native_mmvq.hpp
  echo "###### git stash rc=$?  (diff should be empty below)"
  git diff --stat -- src/kernels/cuda/native_mmvq.cu include/strata/kernels/native_mmvq.hpp
  echo "###### rebuild"
  make -C build-sycl strata_kernels quantize_act_parity iq_multi_parity mmvq_multi_parity -j8 2>&1 | grep -E "error|Linking|Built target" | tail -8
  cd build-sycl
  ZE_AFFINITY_MASK=0 ctest -R "mmvq_multi_parity|quantize_act_parity|iq_multi_parity" --output-on-failure 2>&1 | tail -30
  cd "$SRC"
  echo "###### restoring the D2y edit"
  git stash pop
  echo "###### git stash pop rc=$?"
  make -C build-sycl strata_kernels mmvq_xmx_price quantize_act_parity iq_multi_parity -j8 2>&1 | grep -E "error|Built target" | tail -8
  echo "###### done at $(date -Is)"
} > "$OUT" 2>&1
grep -E "Failed|Passed|tests passed|restoring|pop rc|error" "$OUT" | head -20
