#!/usr/bin/env bash
# D2y: the engine's own parity tests for the kernels this card's knob touches (knob at its default 0).
set +e
SRC=/home/michael/strata-xpu/strata
OUT=$SRC/d2y/D2Y-TESTS.txt
source /opt/intel/oneapi/setvars.sh > /tmp/d2y-setvars.txt 2>&1
{
  echo "###### d2y engine tests $(date -Is)"
  echo "###### native_mmvq.cu gained one pricing knob (g_exact_rows, default 0 = the shipped rule); nothing in the"
  echo "###### engine calls its setter.  These are the parity tests for the kernels that path reaches."
  cd "$SRC/build-sycl"
  ZE_AFFINITY_MASK=0 ctest -R "mmvq_multi_parity|iq_multi_parity|quantize_act_parity|s2_gemv_q8_parity|bf16_gemv_parity" \
       --output-on-failure
  echo "###### ctest rc=$? at $(date -Is)"
} > "$OUT" 2>&1
tail -20 "$OUT"
