#!/usr/bin/env bash
# D2y (card t_ba006576): the measurement arm.  One B70 (ZE_AFFINITY_MASK=0), nothing else on the card.
# Raw stdout+stderr -> d2y/D2Y-BENCH.txt (variants on) or d2y/D2Y-BASE.txt (shipped only, --no-variants).
set +e
SRC=/home/michael/strata-xpu/strata
OUT=${OUT:-$SRC/d2y/D2Y-BENCH.txt}
ARGS=${ARGS:---reps 20 --t 4}
source /opt/intel/oneapi/setvars.sh > /tmp/d2y-setvars.txt 2>&1
{
  echo "###### d2y bench $(date -Is)"
  echo "###### binary: $SRC/build-sycl/mmvq_xmx_price md5 $(md5sum < "$SRC/build-sycl/mmvq_xmx_price" | cut -c1-32)"
  echo "###### ZE_AFFINITY_MASK=0 (card 0 only; see d2y/machine.txt for which cards were free)"
  echo "###### command: ZE_AFFINITY_MASK=0 build-sycl/mmvq_xmx_price $ARGS"
  echo
  ZE_AFFINITY_MASK=0 stdbuf -o0 "$SRC/build-sycl/mmvq_xmx_price" $ARGS
  echo "###### exit=$? at $(date -Is)"
} > "$OUT" 2>&1
echo "wrote $OUT"
grep -c . "$OUT"
