#!/usr/bin/env bash
# D2c (card t_c7d8cd86) chain A: THE COLD FIRST RUNS -- the number this card is for.
#
# Three arms, each with BOTH compiler caches fresh (a fresh install):
#   d2c-nooff-cold-4096   STRATA_MMVQ_WARMUP=0 STRATA_KERNEL_WARMUP=0   <- the shipped layout, no warm-up at all
#   d2c-mmoff-cold-4096   STRATA_KERNEL_WARMUP=0                        <- D2b's pass only (this card's "before")
#   d2c-wu-cold-4096      (both on, the default)                        <- D2b + D2c
# then the same-session warm controls on the shared warm cache.
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2b/d2b_run_arm.sh
TAG=${1:-d2c}
C1=/tmp/d2c-cold1-cache
C2=/tmp/d2c-cold2-cache
C3=/tmp/d2c-cold3-cache
rm -rf "$C1" "$C2" "$C3"

echo "=== A1 $TAG-nooff-cold-4096 (both warm-ups OFF, both caches fresh) $(date -Is)"
bash "$RUN" "$TAG-nooff-cold-4096" 4096 256 --sycl-cache "$C1/sycl" --neo-cache "$C1/neo" \
     --env STRATA_MMVQ_WARMUP=0 --env STRATA_KERNEL_WARMUP=0

echo "=== A2 $TAG-mmoff-cold-4096 (D2b's pass only, both caches fresh) $(date -Is)"
bash "$RUN" "$TAG-mmoff-cold-4096" 4096 256 --sycl-cache "$C2/sycl" --neo-cache "$C2/neo" \
     --env STRATA_KERNEL_WARMUP=0

echo "=== A3 $TAG-wu-cold-4096 (both passes, both caches fresh) $(date -Is)"
bash "$RUN" "$TAG-wu-cold-4096" 4096 256 --sycl-cache "$C3/sycl" --neo-cache "$C3/neo"

echo "=== A4 $TAG-wu-4096 warm control (shared warm cache, both passes) $(date -Is)"
bash "$RUN" "$TAG-wu-4096" 4096 256

echo "=== A5 $TAG-mmoff-4096 warm control (shared warm cache, D2b's pass only) $(date -Is)"
bash "$RUN" "$TAG-mmoff-4096" 4096 256 --env STRATA_KERNEL_WARMUP=0

echo "=== A6 $TAG-wu-32768 warm control at 32K (both passes) $(date -Is)"
bash "$RUN" "$TAG-wu-32768" 32768 256

echo "=== chain A done $(date -Is)"
/usr/bin/python3 "$SRC/d2c/analyze_c.py" --all-cold
/usr/bin/python3 "$SRC/d2c/analyze_c.py" --all-warm
/usr/bin/python3 "$SRC/d2b/ids_b.py" "$TAG-nooff-cold-4096" "$TAG-mmoff-cold-4096" "$TAG-wu-cold-4096" \
    "$TAG-wu-4096" "$TAG-mmoff-4096" "$TAG-wu-32768"
