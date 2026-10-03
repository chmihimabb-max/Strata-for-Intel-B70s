#!/usr/bin/env bash
# P4 (card t_63cc226b): the end-to-end arms.  Same binary in every arm; only STRATA_KV_COPY_BLOCKS/THREADS differ.
#
# WHERE THE MECHANISM CAN BE FELT AT ALL: the copy kernel only runs for a block that MISSES in the residency map,
# and with `--kv-resident 32768` (the config of record) every layer is fully resident up to 32K cells - so at 4K and
# 32K the kernel never launches and the arms are a token-id equality control, not a speed comparison.  128K is the
# shortest length at which streaming is engaged on this box (32768 of 131072 cells resident).
#
# usage: bash p4/p4_chain.sh            # all eight arms
#        bash p4/p4_chain.sh 128k       # only the 128K arms
set -e
R=/home/michael/strata-xpu
cd "$R/strata"
WHICH=${1:-all}
run() { bash p4/p4_arm.sh "$1" "$2" "$3" "$4" 2>>p4/arms.log; }

if [ "$WHICH" = "all" ] || [ "$WHICH" = "short" ]; then
  run p4-4k-default  default 4096  150
  run p4-4k-1x128    1x128   4096  150
  run p4-32k-default default 32768 256
  run p4-32k-1x128   1x128   32768 256
fi
if [ "$WHICH" = "all" ] || [ "$WHICH" = "128k" ]; then
  run p4-128k-default default 131072 256
  run p4-128k-1x128   1x128   131072 256
  run p4-128k-4x128   4x128   131072 256
  run p4-128k-16x128  16x128  131072 256
fi
echo "all arms done"
