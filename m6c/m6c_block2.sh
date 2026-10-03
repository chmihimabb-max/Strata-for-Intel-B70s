#!/usr/bin/env bash
# M6c block 2: the 256K arms -- the deliverable.
#   m6c-262k      --max-context 262144, prompt 259,943, --kv-resident 32768, and a SECOND request in the
#                 same process: the needle prompt (259,943 tokens, the answer at 50% depth, 64 greedy
#                 tokens) so correctness at depth is measured on a warm program cache.
#   m6c-262k-res  the same context and prompt with --kv-resident 0 (everything in VRAM): the
#                 streaming-on vs streaming-off residency / RAM / SSD comparison at 256K.
R=/home/michael/strata-xpu
P=$R/m6c/prompts
mkdir -p "$R/m6c"
{
  echo "M6c block 2 start $(date -Is)"
  bash "$R/strata/m6c/m6c_serve.sh" m6c-262k 262144 \
      --req "256:$P/prompt-ctx262144.txt" --req "256:$P/prompt-needle-ctx262144.txt"
  bash "$R/strata/m6c/m6c_serve.sh" m6c-262k-res 262144 256 --kvres 0
  echo "M6c block 2 end $(date -Is)"
} > "$R/m6c/BLOCK2-SUMMARY.txt" 2>&1
tail -4 "$R/m6c/BLOCK2-SUMMARY.txt"
