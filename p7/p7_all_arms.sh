#!/usr/bin/env bash
# P7 (card t_6789d6df): the four arms, one after another (each arm holds both GPUs for its own run).
#
#   new      canonical rt (pinned checkpoint, upstream's tools) + draft_vocab.bin   <- the faithful config
#   old      the W4A16-derived rt-q2_0 as served so far (no draft_vocab.bin)
#   new-nodv the canonical rt with draft_vocab.bin absent  <- separates provenance from the vocab step
#   headless --spec 4 and NO --mtp: suffix/prompt-lookup drafting only (the control)
set -u
P7=/home/michael/strata-xpu/strata/p7
CTX=${1:-4096}
MAXNEW=${2:-256}
for arm in new old new-nodv headless; do
  bash "$P7/p7_arm.sh" "p7-${CTX}-${arm}" "$CTX" "$arm" "$MAXNEW" int8
done
