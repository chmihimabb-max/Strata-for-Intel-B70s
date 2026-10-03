#!/usr/bin/env bash
# P7 (card t_6789d6df): the remaining arms, one after another.
#   1. old      serve, W4A16-derived rt (as served so far)          -> provenance comparison + its draft_vocab gap
#   2. new-nodv serve, canonical rt with draft_vocab.bin absent     -> isolates the vocab step from the provenance
#   3. bench-mtp     bench path, drafter                            -> the reference for the control
#   4. bench-headless bench path, NO --mtp (suffix/lookup only)     -> the measured cost of running without a drafter
set -u
P7=/home/michael/strata-xpu/strata/p7
CTX=${1:-4096}
MAXNEW=${2:-256}
bash "$P7/p7_arm.sh"   "p7-${CTX}-old"           "$CTX" old      "$MAXNEW" int8
bash "$P7/p7_arm.sh"   "p7-${CTX}-new-nodv"      "$CTX" new-nodv "$MAXNEW" int8
bash "$P7/p7_bench.sh" "p7-bench-${CTX}-mtp"     "$CTX" mtp      "$MAXNEW"
bash "$P7/p7_bench.sh" "p7-bench-${CTX}-headless" "$CTX" headless "$MAXNEW"
