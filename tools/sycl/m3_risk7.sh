#!/usr/bin/env bash
# tools/sycl/m3_risk7.sh - Risk 7 evidence (PLAN.md §6 Risk 7, card t_a44aa58f), written to
# plan-evidence/M3-risk7-asm.txt.  The gate: zero surviving `asm(` tokens in every TU the SYCL build compiles.
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${REPO}/../plan-evidence/M3-risk7-asm.txt"
BUILD="${1:-${REPO}/build-sycl}"
PAT='__asm__|\basm[[:space:]]*\(|asm[[:space:]]+volatile'
cd "$REPO"
exec > "$OUT" 2>&1

echo "=== Risk 7 evidence (PLAN.md §6 Risk 7, card t_a44aa58f) ==="
date -Is
echo
echo "-- 1. THE HARD GATE: surviving asm tokens in every TU the SYCL build compiles --"
echo "   pattern: $PAT"
echo
echo "generated TUs (build-sycl/sycl/*.cpp), per-file counts (top 5 by count):"
grep -c -E "$PAT" "$BUILD"/sycl/*.cpp | sort -t: -k2 -rn | head -5 | sed 's/^/   /'
nfiles=$(ls "$BUILD"/sycl/*.cpp | wc -l)
ntok=$(grep -h -c -E "$PAT" "$BUILD"/sycl/*.cpp | awk '{s+=$1} END {print s+0}')
echo "   TOTAL: $nfiles files, $ntok surviving asm tokens"
echo
echo "the 5 hand ports (src/kernels/sycl/*.cpp), per-file counts:"
grep -c -E "$PAT" src/kernels/sycl/*.cpp | sed 's/^/   /'
echo "   TOTAL: $(grep -h -c -E "$PAT" src/kernels/sycl/*.cpp | awk '{s+=$1} END {print s+0}') surviving asm tokens"
echo
echo "-- 2. the refusal that keeps tools/sycl/exceptions.txt honest (syclify refuses a file with asm) --"
python3 tools/sycl/syclify.py src/kernels/cuda/qsa_prompt_attn.cu /dev/null \
        --exceptions tools/sycl/exceptions.txt --repo-root . 2>&1 | head -3
python3 tools/sycl/syclify.py src/kernels/cuda/fused_gr.cu /dev/null \
        --exceptions tools/sycl/exceptions.txt --repo-root . 2>&1 | head -3
echo
echo "-- 3. the .cu sources still carry their 16 sites: PLAN.md D1 keeps them untouched CUDA/HIP sources --"
grep -rn -E "$PAT" src --include='*.cu' | wc -l
echo
echo "-- 4. what a naive shim would have done, and where --"
echo "   '#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800' is TRUE when __CUDA_ARCH__ is undefined,"
echo "   i.e. exactly under SYCL, so the asm arm is the one that would be compiled:"
grep -rn -B1 -E '^\s*#(elif|if) !defined\(__CUDA_ARCH__\)' src/kernels/cuda/*.cu | grep -E '\.cu' | sed 's/^/   /'
