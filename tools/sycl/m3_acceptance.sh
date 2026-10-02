#!/usr/bin/env bash
# tools/sycl/m3_acceptance.sh - the exact M3 acceptance loop (PLAN.md §5 M3, card t_a44aa58f), plus the three
# risk items that card names.  Every command is run from the repo root with oneAPI sourced, and the raw output
# is what goes into the card's completion comment.
#
#   tools/sycl/m3_acceptance.sh [build-dir]        -> stdout (tee it into plan-evidence/M3-acceptance.txt)
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${1:-${REPO}/build-sycl}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
cd "$REPO"

echo "=========== Risk 7 (hard gate): surviving asm tokens in every TU this build compiles ==========="
echo "-- generated TUs (build-sycl/sycl/*.cpp, produced by tools/sycl/syclify.py from the untouched .cu):"
grep -c -E '__asm__|\basm[[:space:]]*\(|asm[[:space:]]+volatile' "${BUILD}"/sycl/*.cpp 2>/dev/null | awk -F: '{s+=$2} END {print "   asm tokens: " s+0 "  (files: " NR ")"}'
echo "-- the 5 hand ports at src/kernels/sycl/ (PLAN.md §2.3):"
grep -c -E '__asm__|\basm[[:space:]]*\(|asm[[:space:]]+volatile' src/kernels/sycl/*.cpp 2>/dev/null
echo "-- (the .cu sources still carry their 16 sites by design: PLAN.md D1, they are CUDA/HIP sources)"
grep -rn -E '__asm__|\basm[[:space:]]*\(|asm[[:space:]]+volatile' src --include='*.cu' | wc -l

echo
echo "=========== Risk 1: tf32 / 16-row XMX shape verdict (probe21, exact driver text) ==========="
cat "${REPO}/../probe/probe21_shapes.txt" 2>/dev/null || echo "(run probe/probe21.sh first)"

echo
echo "=========== Risk 4: a work-group asking for >48 KiB of SLM runs on Xe (PLAN.md §6 Risk 4, M3 half) ==========="
if [ -x "${REPO}/../probe/probe22_lds" ]; then
  ZE_AFFINITY_MASK="$ZE_AFFINITY_MASK" "${REPO}/../probe/probe22_lds"
else
  icpx -fsycl -O2 -std=c++20 -o "${REPO}/../probe/probe22_lds" "${REPO}/../probe/probe22_lds.cpp" \
    && ZE_AFFINITY_MASK="$ZE_AFFINITY_MASK" "${REPO}/../probe/probe22_lds"
fi

echo
echo "=========== the PLAN.md §5 M3 ctest loop ==========="
echo "ctest -N | wc -l (registered tests before this loop):"
ctest --test-dir "$BUILD" -N 2>/dev/null | grep -c "^  Test #"
for t in qsa_parity gdn_parity gr_parity shared_expert_parity ple_parity ple_q5_parity ple_fp8_parity \
         kv_hybrid_parity cvec_parity dequant_bf16_test; do
  # `ctest -R` finds nothing and RETURNS 0 when a test is not registered, so "no match" must never be
  # reported as a pass: check the registration first and name why the test is absent.
  if ! ctest --test-dir "$BUILD" -N -R "^${t}$" 2>/dev/null | grep -q "Test #"; then
    echo "NOT REGISTERED  $t  (its REQUIRED fixture is absent on this box - see the configure output)"
    continue
  fi
  if out=$(ctest --test-dir "$BUILD" -R "^${t}$" --output-on-failure --timeout 900 2>&1); then
    echo "PASS  $t"
  else
    echo "FAIL  $t"
    echo "$out" | grep -E "assert|FAILED|mismatch|Error|Passed|Failed|refus" | head -8
  fi
done

echo
echo "=========== sycl_prompt_attn_parity --selftest, STRATA_SYCL_XMX=0 (portable) ==========="
STRATA_SYCL_XMX=0 "${BUILD}/sycl_prompt_attn_parity" --selftest 2>&1 | tee "${REPO}/../plan-evidence/M3-prompt-attn-xmx0.txt"
echo "exit=${PIPESTATUS[0]}"

echo
echo "=========== sycl_prompt_attn_parity --selftest, STRATA_SYCL_XMX=1 (gate armed) ==========="
STRATA_SYCL_XMX=1 "${BUILD}/sycl_prompt_attn_parity" --selftest 2>&1 | tee "${REPO}/../plan-evidence/M3-prompt-attn-xmx1.txt"
echo "exit=${PIPESTATUS[0]}"

echo
echo "=========== ctest, the M3 selection ==========="
ctest --test-dir "$BUILD" -R "qsa_parity|gdn_parity|gr_parity|shared_expert|ple_|kv_hybrid|cvec|dequant_bf16|prompt_attn|sycl_" --output-on-failure --timeout 900 2>&1 | tail -25
