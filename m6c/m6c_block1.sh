#!/usr/bin/env bash
# M6c block 1: the short end of the depth curve, one engine at a time (PLAN 9 rule 1).
#   32K  --max-context 32768   prompt 32,256   kv streaming NOT engaged (32,768 cells / 4 = 8,192 pages
#                              = the 8,192 resident slots, so the plan stays fully resident: expected
#                              kv_resident=0 in the INFO line -- reported, not hidden)
#   64K  --max-context 65536   prompt 64,512   16,384 pages > 8,192 slots -> kv_mode 1
#   128K --max-context 131072  prompt 129,024
#   64K streaming OFF (--kvres 0) -> the same-length residency/RAM comparison at a price we can afford
R=/home/michael/strata-xpu
mkdir -p "$R/m6c"
{
  echo "M6c block 1 start $(date -Is)"
  # the 4K needle control first: it is the cheap proof that the needle rows' prompt shape is answerable
  # (the plain-continuation form produced EOS on its first token, and 64 generated ended inside the
  # model's think block, so the control runs the same 256 max_new as the curve)
  bash "$R/strata/m6c/m6c_serve.sh" m6c-needle-4k 4096 256 --kvres 0 \
      --prompt "$R/m6c/prompts/prompt-needle-ctx4096.txt"
  bash "$R/strata/m6c/m6c_serve.sh" m6c-32k  32768  256
  bash "$R/strata/m6c/m6c_serve.sh" m6c-64k  65536  256
  # --prefill auto arm at 64K: upstream's table was measured with --prefill auto (the engine's own help:
  # "auto = the largest chunk up to 8192 whose buffers the expert cache can lend"), so this says whether
  # our --prefill 512 is part of the prefill gap rather than leaving it to speculation
  bash "$R/strata/m6c/m6c_serve.sh" m6c-64k-auto 65536 256 --prefill-auto
  bash "$R/strata/m6c/m6c_serve.sh" m6c-128k 131072 256
  bash "$R/strata/m6c/m6c_serve.sh" m6c-64k-res 65536 256 --kvres 0
  echo "M6c block 1 end $(date -Is)"
} > "$R/m6c/BLOCK1-SUMMARY.txt" 2>&1
tail -4 "$R/m6c/BLOCK1-SUMMARY.txt"
