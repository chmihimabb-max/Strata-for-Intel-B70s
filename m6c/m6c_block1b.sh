#!/usr/bin/env bash
# M6c block 1b: the depth curve at --prefill auto, which is what the card names in item 1 and what
# upstream measured their table with ("auto = the largest chunk up to 8192 whose buffers the expert cache
# can lend", engine help).  Measured at 64K: auto picked an 8,192-token chunk and prefilled at 339.6 tok/s
# against 217.3 at --prefill 512 -- the same 256 generated tokens, decode unchanged (18.7 against 18.9).
# So the curve of record is the auto arm, and the 512 rows stay as a second, labelled arm.
R=/home/michael/strata-xpu
mkdir -p "$R/m6c"
{
  echo "M6c block 1b start $(date -Is)"
  bash "$R/strata/m6c/m6c_serve.sh" m6c-32k-auto  32768  256 --prefill-auto
  bash "$R/strata/m6c/m6c_serve.sh" m6c-128k-auto 131072 256 --prefill-auto
  bash "$R/strata/m6c/m6c_serve.sh" m6c-64k-res-auto 65536 256 --prefill-auto --kvres 0
  echo "M6c block 1b end $(date -Is)"
} > "$R/m6c/BLOCK1B-SUMMARY.txt" 2>&1
tail -4 "$R/m6c/BLOCK1B-SUMMARY.txt"
