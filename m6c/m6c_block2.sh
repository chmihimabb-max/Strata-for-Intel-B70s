#!/usr/bin/env bash
# M6c block 2: the 256K arms -- the deliverable.  All three at --prefill auto, the chunk the card names
# and the one upstream's table used (measured 1.56x faster than --prefill 512 at 64K).
#   m6c-262k       --max-context 262144, prompt 259,943, --kv-resident 32768, and a SECOND request in the
#                  same process: the needle prompt (259,943 tokens, the answer at 50% depth, 256 greedy
#                  tokens) so correctness at depth is measured on a warm program cache.
#   m6c-262k-res   the same context and prompt with --kv-resident 0 (the whole K/V in VRAM): the
#                  streaming-on vs streaming-off residency / RAM comparison at 256K.
#   m6c-262k-cold  the same with the two GGUF shards dropped from the page cache (POSIX_FADV_DONTNEED,
#                  i1/i1_io.py): the SSD tier's own numbers at 262K -- bytes read and major faults -- on a
#                  machine whose 123 GB of RAM already holds the 83.62 GB model.
R=/home/michael/strata-xpu
P=$R/m6c/prompts
mkdir -p "$R/m6c"
{
  echo "M6c block 2 start $(date -Is)"
  bash "$R/strata/m6c/m6c_serve.sh" m6c-262k 262144 --prefill-auto \
      --req "256:$P/prompt-ctx262144.txt" --req "256:$P/prompt-needle-ctx262144.txt"
  bash "$R/strata/m6c/m6c_serve.sh" m6c-262k-res 262144 256 --prefill-auto --kvres 0
  bash "$R/strata/m6c/m6c_serve.sh" m6c-262k-cold 262144 256 --prefill-auto --drop --warmcache "$R/sycl-cache/m6c"
  echo "M6c block 2 end $(date -Is)"
} > "$R/m6c/BLOCK2-SUMMARY.txt" 2>&1
tail -4 "$R/m6c/BLOCK2-SUMMARY.txt"
