#!/usr/bin/env bash
# tools/sycl/m2_acceptance.sh - the exact M2 acceptance loop (PLAN.md §5 M2) plus the extra tests the card
# names: per-test pass/fail with the failing assertion text, and the ctest count before/after.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${1:-${REPO}/build-sycl}"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
cd "$REPO"

echo "=== ctest -N | wc -l  (registered tests) ==="
ctest --test-dir "$BUILD" -N 2>/dev/null | grep -c "^  Test #"

echo
echo "=== the PLAN.md §5 M2 per-test loop ==="
for t in dequant_s2_parity s2_gemv_parity s2_gemv_q8_parity iq_parity iq_multi_parity mmvq_multi_parity \
         s_gemv_parity s_gemv_q8k_parity router_top10_parity rope_parity bf16_gemv_parity \
         elementwise_parity quantize_act_parity kv_q8_parity kv_q4_parity kv_stream_parity sampler_parity; do
  if out=$(ctest --test-dir "$BUILD" -R "^${t}$" --output-on-failure --timeout 300 2>&1); then
    echo "PASS  $t"
  else
    echo "FAIL  $t"
    echo "$out" | grep -E "assert|FAILED|mismatch|Error|Passed|Failed" | head -6
  fi
done

echo
echo "=== the rest of the M2 set (card scope: files #1,2,3 + Risk 8 + the M1 set) ==="
for t in s2_expert_grouped_parity cvec_parity sampler_parity_one_block sampler_parity_old \
         sycl_handoff_test sycl_mapped_alias_test sycl_device_selftest sycl_device_list \
         platform_memory_test pinned_shared_test; do
  if out=$(ctest --test-dir "$BUILD" -R "^${t}$" --output-on-failure --timeout 300 2>&1); then
    echo "PASS  $t"
  else
    echo "FAIL  $t"
    echo "$out" | grep -E "assert|FAILED|mismatch|Error|Passed|Failed" | head -6
  fi
done

echo
echo "=== full suite ==="
ctest --test-dir "$BUILD" --output-on-failure --timeout 300 2>&1 | tail -20
