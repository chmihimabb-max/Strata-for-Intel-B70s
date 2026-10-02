#!/usr/bin/env bash
# tools/sycl/risk5_dp4a.sh - PLAN.md Risk 5: the dp4a emulation's cost and its instruction-level form.
#
# 1. builds and RUNS bench/micro/sycl_dp4a_cost.cpp (time per dot for the three spellings the plan names),
# 2. turns the same TU's device code into SPIR-V and counts the opcodes (tools/sycl/spirv_ops.py), which is
#    what answers "does the compiler contract it" - oneAPI ships no spirv-dis.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${REPO}/../plan-evidence"
WORK="${REPO}/../syclgen-test/risk5"
mkdir -p "$WORK" "$OUT"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
BIN=/opt/intel/oneapi/compiler/2026.1/bin/compiler

icpx -fsycl -fsycl-targets=spir64 -O3 -std=c++20 -o "$WORK/sycl_dp4a_cost" \
     "${REPO}/bench/micro/sycl_dp4a_cost.cpp" || exit 1
ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" "$WORK/sycl_dp4a_cost" | tee "$OUT/M2-risk5-dp4a-bench.txt"

"$BIN/clang" -fsycl -fsycl-device-only -O3 -std=c++20 -emit-llvm -c \
     "${REPO}/bench/micro/sycl_dp4a_cost.cpp" -o "$WORK/bench.bc" || exit 1
"$BIN/sycl-post-link" -O3 -device-globals -split=auto -o "$WORK/table.txt" "$WORK/bench.bc" || exit 1
"$BIN/file-table-tform" -extract=Code -drop_titles -o "$WORK/code.txt" "$WORK/table.txt" || exit 1
rm -f "$WORK"/*.spv
i=0
while read -r f; do
  [ -z "$f" ] && continue
  i=$((i + 1))
  "$BIN/llvm-spirv" -o "$WORK/code_${i}.spv" -spirv-max-version=1.5 "$f" || exit 1
done < "$WORK/code.txt"
{
  echo "--- SPIR-V opcode scan of the dp4a micro-benchmark's device image (all three spellings in one module) ---"
  for spv in "$WORK"/code_*.spv; do
    python3 "${REPO}/tools/sycl/spirv_ops.py" "$spv" --top 18
  done
} | tee "$OUT/M2-risk5-spirv-ops.txt"
