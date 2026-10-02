#!/usr/bin/env bash
# tools/sycl/m4_check.sh - fast generate+compile loop for the M4 prefill group (PLAN.md §5 M4, card t_086173b8).
# Same development tool as tools/sycl/m2_check.sh / m3_check.sh, pointed at tools/sycl/m4_files.txt.
# The prefill group is NOT all kernels: gemm.cu and prefill.cpp are host code (0 __global__), so the loop
# translates every file the same way syclify is used at build time and reports REFUSED separately from FAIL.
#
#   tools/sycl/m4_check.sh                 # the whole M4 group
#   tools/sycl/m4_check.sh gemm kernels    # by name
# /opt/intel/oneapi/compiler/latest/env/vars.sh:258 reads $OCL_ICD_FILENAMES without a default, so sourcing
# setvars.sh under `set -u` kills the shell ("unbound variable" -> non-interactive bash exits).
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GEN="${REPO}/../syclgen-m4-test"
LOGS="${GEN}/logs"
mkdir -p "$GEN" "$LOGS"

if [ "$#" -gt 0 ]; then
  NAMES="$*"
else
  NAMES="$(grep -oE '^[a-z0-9_]+' "${REPO}/tools/sycl/m4_files.txt" | tr '\n' ' ')"
fi

work_one() {
  local name="$1"
  local cu
  cu="$(ls "${REPO}"/src/prefill/"${name}".cu "${REPO}"/src/prefill/"${name}".cpp 2>/dev/null | head -1)"
  if [ -z "$cu" ]; then echo "MISSING  ${name}"; return; fi
  if ! python3 "${REPO}/tools/sycl/syclify.py" "$cu" "${GEN}/${name}.cpp" \
        --exceptions "${REPO}/tools/sycl/exceptions.txt" --repo-root "${REPO}" \
        >"${LOGS}/${name}.gen.log" 2>&1; then
    echo "REFUSED  ${name}"
    return
  fi
  if icpx -fsycl -fsycl-targets=spir64 -O1 -std=c++20 -x c++ \
        -I "${REPO}/include/strata/sycl_compat" -I "${REPO}/include" -I "${REPO}/src/prefill" \
        -include "${REPO}/include/strata/sycl_compat/cuda_runtime.h" \
        -DSTRATA_USE_SYCL=1 -fno-sycl-rdc -fsycl-default-sub-group-size=32 \
        -c "${GEN}/${name}.cpp" -o /dev/null >"${LOGS}/${name}.log" 2>&1; then
    echo "OK       ${name}"
  else
    echo "FAIL     ${name}"
  fi
}
export -f work_one
export REPO GEN LOGS

echo "$NAMES" | tr ' ' '\n' | grep -v '^$' | xargs -P 6 -I{} bash -c 'work_one "$@"' _ {} | sort

echo "--- first errors ---"
for n in $NAMES; do
  if [ -f "${LOGS}/${n}.log" ] && grep -q "error:" "${LOGS}/${n}.log" 2>/dev/null; then
    echo "### ${n}  ($(grep -c 'error:' "${LOGS}/${n}.log") errors)"
    grep "error:" "${LOGS}/${n}.log" | head -4
  elif [ -f "${LOGS}/${n}.gen.log" ] && grep -q "REFUSED" "${LOGS}/${n}.gen.log" 2>/dev/null; then
    echo "### ${n}  REFUSED"; cat "${LOGS}/${n}.gen.log"
  fi
done
