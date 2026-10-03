#!/usr/bin/env bash
# P5: the two attention kernels' own accuracy measurement, on the engine's own shapes.
#
# `sycl_prompt_attn_parity <ctx> <queries> <reps>` runs BOTH kernels on one synthetic fixture and reports each
# one's max absolute error against an FP64 host reference, the output scale, and the two against each other --
# i.e. the kernels judged on their own axes, with no model and no generation in the loop.  P2 quoted it at
# ctx 32768; P5 needs the two contexts whose prompts produced divergences (3832 = the 4K prompt, 32768 = the
# 32K prompt) on the same binary and the same fixture.
#
#   usage: bash p5/p5_parity.sh
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$SRC/p5/evidence/parity-p5.txt
mkdir -p "$SRC/p5/evidence"
{
  echo "== qsa prompt attention parity, P5's shapes, card 0 (ZE_AFFINITY_MASK=0)"
  echo "== binary: $SRC/build-sycl/sycl_prompt_attn_parity  $(stat -c '%y %s bytes' $SRC/build-sycl/sycl_prompt_attn_parity)"
  echo "== HEAD: $(cd $SRC && git log --oneline -1)"
  echo
} > "$OUT"
source /opt/intel/oneapi/setvars.sh >> "$OUT" 2>&1
cd "$SRC" || exit 1
for ctx in 3832 32768; do
  echo "--- ./build-sycl/sycl_prompt_attn_parity $ctx 2048 3   (ZE_AFFINITY_MASK=0)" >> "$OUT"
  ZE_AFFINITY_MASK=0 ./build-sycl/sycl_prompt_attn_parity "$ctx" 2048 3 >> "$OUT" 2>&1
  echo "    exit $?" >> "$OUT"
done
cat "$OUT"
