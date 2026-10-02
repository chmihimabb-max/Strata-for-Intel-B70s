#!/usr/bin/env bash
# tools/sycl/handport_check.sh - build every committed hand port (src/kernels/sycl/*.cpp) with the same flags
# the cmake SYCL backend uses, printing OK/FAIL and the first errors.  This is the loop that proves the 5
# hand ports of PLAN.md §2.3 compile without the build having to be reconfigured.
#
#   tools/sycl/handport_check.sh            # all 5
#   tools/sycl/handport_check.sh fused_gr
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS="${REPO}/../syclhand/logs"
mkdir -p "$LOGS"

if [ "$#" -gt 0 ]; then NAMES="$*"; else NAMES="fused_gr qsa_prompt_attn qsa_select native_qsa_score verify_kernels"; fi

one() {
  local n="$1"
  if [ ! -f "${REPO}/src/kernels/sycl/${n}.cpp" ]; then echo "MISSING  ${n}"; return; fi
  if icpx -fsycl -fsycl-targets=spir64 -O1 -std=c++20 -x c++ \
        -I "${REPO}/include/strata/sycl_compat" -I "${REPO}/include" -I "${REPO}/third_party/ggml" \
        -include "${REPO}/include/strata/sycl_compat/cuda_runtime.h" \
        -DSTRATA_USE_SYCL=1 -fno-sycl-rdc -fsycl-default-sub-group-size=32 \
        -c "${REPO}/src/kernels/sycl/${n}.cpp" -o /dev/null > "${LOGS}/${n}.log" 2>&1; then
    echo "OK       ${n}"
  else
    echo "FAIL     ${n}"
  fi
}
export -f one
export REPO LOGS
echo "$NAMES" | tr ' ' '\n' | grep -v '^$' | xargs -P 5 -I{} bash -c 'one "$@"' _ {} | sort
echo "--- first errors ---"
for n in $NAMES; do
  if [ -f "${LOGS}/${n}.log" ] && grep -q "error:" "${LOGS}/${n}.log"; then
    echo "### ${n}  ($(grep -c 'error:' "${LOGS}/${n}.log") errors)"
    grep "error:" "${LOGS}/${n}.log" | head -5
  fi
done
