#!/usr/bin/env bash
# Find which per-kernel device image of ONE generated TU fails the SPIR-V translation.
# Development tool (M2): the "Invalid SPIR-V module" error names no file, so the pipeline is run by hand:
# device-only bitcode -> sycl-post-link (per-kernel split) -> llvm-spirv on each split image.
#   tools/sycl/spirv_bisect.sh build-sycl/sycl/kv_stream.cpp
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$1"
WORK="${REPO}/../syclgen-test/spirv_bisect"
mkdir -p "$WORK"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
BIN=/opt/intel/oneapi/compiler/2026.1/bin/compiler

icpx -fsycl -fsycl-device-only -O3 -std=c++20 \
     -I "${REPO}/include/strata/sycl_compat" -I "${REPO}/include" -I "${REPO}/third_party/ggml" \
     -include "${REPO}/include/strata/sycl_compat/cuda_runtime.h" -DSTRATA_USE_SYCL=1 \
     -emit-llvm -c "$SRC" -o "$WORK/dev.bc" || { echo "device compile failed"; exit 1; }

"$BIN/sycl-post-link" -O3 -device-globals -split=auto -o "$WORK/table.txt" "$WORK/dev.bc" || exit 1
"$BIN/file-table-tform" -extract=Code -drop_titles -o "$WORK/code.txt" "$WORK/table.txt" || exit 1

i=0
while read -r f; do
  [ -z "$f" ] && continue
  i=$((i + 1))
  if "$BIN/llvm-spirv" -o "$WORK/out_${i}.spv" -spirv-max-version=1.5 \
       -spirv-allow-unknown-intrinsics=llvm.genx. "$f" >"$WORK/conv_${i}.log" 2>&1; then
    echo "OK       image ${i}: $(basename "$f")"
  else
    echo "FAIL     image ${i}: $(basename "$f")"
    sed -n '1,3p' "$WORK/conv_${i}.log"
    echo "         source: $f"
  fi
done < "$WORK/code.txt"
