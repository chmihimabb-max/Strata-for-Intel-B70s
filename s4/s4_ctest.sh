#!/usr/bin/env bash
# S4 (card t_30d9ccfb): the engine's own test suite on the S4 build, the D1 convention (ZE_AFFINITY_MASK=0, the
# oneAPI environment sourced first - without it the four KV parity tests fail instantly on libsycl.so.9).
#
#   bash s4/s4_ctest.sh
#
# 1. the checkpoint/conversation tests by name (the code this card touches)
# 2. the full suite with the graph path at its default; the comparison baseline is D1's "5 failed of 49"
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$SRC/s4/runs/ctest
mkdir -p "$OUT"
set +e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ZE_AFFINITY_MASK=0
cd "$SRC/build-sycl" || exit 1
{
  echo "== S4 ctest $(date -Is)"
  echo "HEAD: $(cd "$SRC" && git log --oneline -1)"
  echo "engine binary: md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)  $(stat -c '%y' "$SRC/build-sycl/strata")"
  echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
  echo "== all tests whose name mentions the checkpoint path =="
  ctest -N 2>&1 | grep -iE "conv|snapshot|checkpoint|cache" || true
} > "$OUT/notes.txt" 2>&1
cat "$OUT/notes.txt"
echo "== 1. the checkpoint/conversation tests =="
ctest -R "conv|snapshot" --output-on-failure > "$OUT/checkpoint-tests.log" 2>&1
tail -12 "$OUT/checkpoint-tests.log"
echo "== 2. the full suite, graph path at its default =="
ctest > "$OUT/full.log" 2>&1
grep -E "tests passed|tests failed" "$OUT/full.log" | tail -2
grep -E "\(Failed\)" "$OUT/full.log" | tail -10
echo "== logs under $OUT =="
