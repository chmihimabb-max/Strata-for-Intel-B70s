#!/usr/bin/env bash
# P4 (card t_63cc226b): the repeat-and-control arms.
#   default2 / 1x128b - a second run of the two extremes, to bound this harness's run-to-run spread (one run per
#                       arm cannot separate a 2% effect from a 2% drift, and the first arm of a session is slow);
#   32k-1x128 / 4k-1x128 - the CHANGED shape at the two lengths where `--kv-resident 32768` keeps every layer fully
#                       resident: the copy kernel never launches, so these are the token-id equality controls
#                       against the ids P2/P3 measured (32K md5 124a3cd39b33f7da31e2, 4K md5 ac9f16fceed82e5681e4).
#
# usage: bash p4/p4_chain3.sh
set -e
cd /home/michael/strata-xpu/strata
run() { bash p4/p4_arm.sh "$1" "$2" "$3" "$4" 2>>p4/arms.log; }
run p4-128k-default2 default 131072 256
run p4-128k-1x128b   1x128   131072 256
run p4-32k-1x128     1x128   32768  256
run p4-4k-1x128      1x128   4096   150
echo "repeat and control arms done"
