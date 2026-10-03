#!/usr/bin/env bash
# P4 (card t_63cc226b): the 4K equality control and the AMPLIFIED 128K pair.
#
#  * 4k-1x128: 4K, the changed shape, P3's own 4K arm shape (--prefill-auto 150, the needle prompt) so its greedy
#    ids are directly comparable to the md5 P2/P3 measured (ac9f16fceed82e5681e4, 147 ids).  At 4K every layer is
#    fully resident, so this is an equality control, not a speed comparison.
#  * 128k-amp-*: POSITIVE CONTROL for the end-to-end null.  At the config of record the copy moves 2.56 MiB per
#    window - 101 us of a 132 ms window - which is inside this harness's run-to-run spread.  `--kvres 20480` (the
#    engine's floor, `qsa_kv_resident_min`) shrinks the resident set 1.6x (20,480 of 131,072 cells) and multiplies
#    the miss volume, so the SAME two shapes should now separate by the ratio the bandwidth curve predicts
#    (1 block x 128 threads against 96 x 128 threads).  If they do, the null at the config of record is explained
#    by volume; if they do not, the port's copy is not on the critical path the way the code says it is.
#
# usage: bash p4/p4_chain4.sh
set -e
R=/home/michael/strata-xpu
cd "$R/strata"
export STRATA_KV_COPY_BLOCKS=1 STRATA_KV_COPY_THREADS=128
bash m6c/m6c_serve.sh p4-4k-1x128 4096 --prefill-auto 150 --prompt "$R/m6c/prompts/prompt-needle-ctx4096.txt" 2>>p4/arms.log
unset STRATA_KV_COPY_BLOCKS STRATA_KV_COPY_THREADS
bash m6c/m6c_serve.sh p4-128k-amp-default 131072 --prefill-auto 256 --kvres 20480 2>>p4/arms.log
export STRATA_KV_COPY_BLOCKS=1 STRATA_KV_COPY_THREADS=128
bash m6c/m6c_serve.sh p4-128k-amp-1x128 131072 --prefill-auto 256 --kvres 20480 2>>p4/arms.log
echo "amplified arms done"
