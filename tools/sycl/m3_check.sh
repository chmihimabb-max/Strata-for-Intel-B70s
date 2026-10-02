#!/usr/bin/env bash
# tools/sycl/m3_check.sh - fast generate+compile loop for the M3 group (PLAN.md §5 M3, card t_a44aa58f).
# Same development tool as tools/sycl/m2_check.sh, pointed at tools/sycl/m3_files.txt and its own
# generated-output directory so the two milestones can be compared side by side.
#
#   tools/sycl/m3_check.sh                      # the M3 group in tools/sycl/m3_files.txt
#   tools/sycl/m3_check.sh qsa native_flash_attn
# /opt/intel/oneapi/compiler/latest/env/vars.sh:258 reads $OCL_ICD_FILENAMES without a default, so sourcing
# setvars.sh under `set -u` kills the shell ("unbound variable" -> non-interactive bash exits).  Measured
# 2026-10-02 in probe_setvars2.sh; turn -u off for the source only.
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GEN="${REPO}/../syclgen-m3-test"
LOGS="${GEN}/logs"
mkdir -p "$GEN" "$LOGS"

if [ "$#" -gt 0 ]; then
  NAMES="$*"
else
  NAMES="$(grep -oE '^[a-z0-9_]+' "${REPO}/tools/sycl/m3_files.txt" | tr '\n' ' ')"
fi

work_one() {
  local name="$1"
  local cu
  cu="$(ls "${REPO}"/src/kernels/cuda/"${name}".cu "${REPO}"/src/core/"${name}".cu 2>/dev/null | head -1)"
  if [ -z "$cu" ]; then echo "MISSING  ${name}"; return; fi
  if ! python3 "${REPO}/tools/sycl/syclify.py" "$cu" "${GEN}/${name}.cpp" \
        --exceptions "${REPO}/tools/sycl/exceptions.txt" --repo-root "${REPO}" \
        >"${LOGS}/${name}.gen.log" 2>&1; then
    echo "REFUSED  ${name}"
    return
  fi
  if icpx -fsycl -fsycl-targets=spir64 -O1 -std=c++20 -x c++ \
        -I "${REPO}/include/strata/sycl_compat" -I "${REPO}/include" -I "${REPO}/third_party/ggml" \
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

echo "$NAMES" | tr ' ' '\n' | grep -v '^$' | xargs -P 10 -I{} bash -c 'work_one "$@"' _ {} | sort

echo "--- first errors ---"
for n in $NAMES; do
  if [ -f "${LOGS}/${n}.log" ] && grep -q "error:" "${LOGS}/${n}.log" 2>/dev/null; then
    echo "### ${n}  ($(grep -c 'error:' "${LOGS}/${n}.log") errors)"
    grep "error:" "${LOGS}/${n}.log" | head -4
  elif [ -f "${LOGS}/${n}.gen.log" ] && grep -q "REFUSED" "${LOGS}/${n}.gen.log" 2>/dev/null; then
    echo "### ${n}  REFUSED"; cat "${LOGS}/${n}.gen.log"
  fi
done
