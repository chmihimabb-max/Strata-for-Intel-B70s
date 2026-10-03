#!/usr/bin/env bash
# P4 (card t_63cc226b) arm extractor: P3's extractor reads any tag under ~/strata-xpu/m6c/runs and prints the
# decode-timing line, the serve summary, the DONE line, the greedy token-id count + md5, and peak RSS/VRAM - which
# is exactly the before/after table P4 needs (token-id equality is the guard).  Wrapped here so the P4 evidence is
# reproducible from the repo.
#
# usage: bash p4/extract_arms.sh                 # the eight P4 arms
#        bash p4/extract_arms.sh p4-128k-default  # one
set -e
R=/home/michael/strata-xpu
TAGS=("$@")
if [ ${#TAGS[@]} -eq 0 ]; then
  TAGS=(p4-4k-default p4-4k-1x128 p4-32k-default p4-32k-1x128 p4-128k-default p4-128k-1x128 p4-128k-4x128
        p4-128k-16x128)
fi
/usr/bin/python3 "$R/p3/extract.py" "${TAGS[@]}"
