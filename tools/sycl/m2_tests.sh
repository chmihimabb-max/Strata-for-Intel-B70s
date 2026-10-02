#!/usr/bin/env bash
# tools/sycl/m2_tests.sh - the per-test pass/fail loop only (no full-suite run at the end).
#   tools/sycl/m2_tests.sh [m2|extra|all]
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${REPO}/build-sycl"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
cd "$REPO"
WHICH="${1:-all}"

M2="dequant_s2_parity s2_gemv_parity s2_gemv_q8_parity iq_parity iq_multi_parity mmvq_multi_parity
    s_gemv_parity s_gemv_q8k_parity router_top10_parity rope_parity bf16_gemv_parity
    elementwise_parity quantize_act_parity kv_q8_parity kv_q4_parity kv_stream_parity sampler_parity"
EXTRA="s2_expert_grouped_parity cvec_parity sampler_parity_one_block sampler_parity_old
       sycl_handoff_test sycl_mapped_alias_test"

case "$WHICH" in
  m2) LIST="$M2" ;;
  extra) LIST="$EXTRA" ;;
  *) LIST="$M2 $EXTRA" ;;
esac

for t in $LIST; do
  out="$(ctest --test-dir "$BUILD" -R "^${t}$" --output-on-failure --timeout 300 2>&1)"
  if printf '%s' "$out" | grep -q "100% tests passed"; then
    echo "PASS  $t"
  else
    echo "FAIL  $t"
    printf '%s\n' "$out" | grep -E "assert|FAIL|mismatch|Error|passed|differ|non-finite" | head -5
  fi
done
