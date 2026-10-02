#!/usr/bin/env bash
# tools/sycl/compile_survey.sh - compile every generated TLU with icpx and report which ones build.
#
# This is the M1 evidence for the transform's real coverage: which .cu files survive a full
# `icpx -fsycl -x c++` compile after the rewrite, and the first error for the ones that do not.
# It is a DEVELOPMENT tool, not part of the build: the build only compiles the kernels the backend
# currently claims, listed in cmake/sycl_backend.cmake.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GEN="${REPO}/../syclgen-test"
OUT="${REPO}/../syclgen-test/logs"
mkdir -p "$OUT"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export REPO OUT

compile_one() {
  local f="$1"
  local b
  b="$(basename "$f" .cpp)"
  if icpx -fsycl -fsycl-targets=spir64 -O1 -std=c++20 -x c++ \
        -I "${REPO}/include/strata/sycl_compat" -I "${REPO}/include" -I "${REPO}/third_party/ggml" \
        -include "${REPO}/include/strata/sycl_compat/cuda_runtime.h" \
        -DSTRATA_USE_SYCL=1 -fno-sycl-rdc \
        -c "$f" -o "/dev/null" >"$OUT/$b.log" 2>&1; then
    echo "OK       $b"
  else
    echo "FAIL     $b"
  fi
}
export -f compile_one

ls "$GEN"/*.cpp | sort | xargs -P 8 -I{} bash -c 'compile_one "$@"' _ {} | sort | tee "$OUT/survey.txt"
echo "--- summary: OK $(grep -c '^OK' "$OUT/survey.txt") / FAIL $(grep -c '^FAIL' "$OUT/survey.txt") ---"
