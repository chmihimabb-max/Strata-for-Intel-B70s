#!/usr/bin/env bash
# D2x (card t_85e61269): the measurement arm.  One B70 (ZE_AFFINITY_MASK=0), nothing else on the card.
# Raw stdout+stderr -> d2x/D2X-BENCH.txt, engine-side lines of the probe included.
set +e
SRC=/home/michael/strata-xpu/strata
OUT=$SRC/d2x/D2X-BENCH.txt
source /opt/intel/oneapi/setvars.sh > /tmp/d2x-setvars.txt 2>&1
{
  echo "###### d2x bench $(date -Is)"
  echo "###### binary: $SRC/build-sycl/mmvq_xmx_price md5 $(md5sum < "$SRC/build-sycl/mmvq_xmx_price" | cut -c1-32)"
  echo "###### ZE_AFFINITY_MASK=0 (card 0 only; both cards were free - see machine.txt)"
  echo "###### command: ZE_AFFINITY_MASK=0 build-sycl/mmvq_xmx_price --reps ${REPS:-20} --t 4,6"
  echo
  ZE_AFFINITY_MASK=0 stdbuf -o0 "$SRC/build-sycl/mmvq_xmx_price" --reps "${REPS:-20}" --t 4,6
  echo "###### exit=$? at $(date -Is)"
} > "$OUT" 2>&1
echo "wrote $OUT"
grep -c . "$OUT"
