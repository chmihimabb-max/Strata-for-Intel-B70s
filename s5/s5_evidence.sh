#!/usr/bin/env bash
# S5 (card t_aae723be): assemble s5/S5-EVIDENCE.txt - the raw output behind every claim in s5/STATUS-S5.md, taken
# from the arms' own files (no retyping, no summaries).
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$SRC/s5/S5-EVIDENCE.txt
: > "$OUT"
say() { echo "$*" >> "$OUT"; }
hdr() { echo >> "$OUT"; echo "================================================================================" >> "$OUT"; echo "$*" >> "$OUT"; echo "================================================================================" >> "$OUT"; }
arm_ids() { for t in "$@"; do printf '  %-22s %-4s %-34s %s\n' "$t" "$(wc -l < "$SRC/s5/runs/$t/ids.txt" 2>/dev/null)" "$(cat "$SRC/s5/runs/$t/ids.md5" 2>/dev/null)" "$(grep -h 'strata serve: prompt ' "$SRC/s5/runs/$t/out.txt" "$SRC/s5/runs/$t/err.txt" 2>/dev/null | head -1)" >> "$OUT"; done; }

say "S5 (card t_aae723be) EVIDENCE - the raw output, assembled $(date -Is)"
say "repo HEAD      : $(cd "$SRC" && git log --oneline -1)"
say "engine binary  : $SRC/build-sycl/strata  md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)  ($(stat -c %y "$SRC/build-sycl/strata"))"
say "config of record: strata-sycl-iq3s.json  md5 $(md5sum < "$SRC/strata-sycl-iq3s.json" | cut -c1-32)  (unmodified by this card)"
say "assembled by   : s5/s5_evidence.sh"

hdr "1. THE ID-LEVEL PROMPT (what the SERVED request really sent)"
say "The served path tokenizes inside serve.server (Service.prepare, serve/server.py:1141): chat template over the"
say "request's messages, then the pack's tokenizer.  s5/s5_ids.py does those two calls itself and writes the engine's"
say "own line 'GEN <max_new> <ids>'.  The check is the engine's own count against the S4 arm's count."
for c in 4096 32768; do
  say
  say "--- ctx$c ---"
  cat "$SRC/s5/prompts/gen-ctx$c.report.json" >> "$OUT"
  say "ids in the GEN line : $(awk '{n=split($3,a,","); print n}' "$SRC/s5/prompts/gen-ctx$c.txt")"
  say "s4 arm it mirrors   : $(grep -h 'strata serve: prompt ' "$SRC"/s4/runs/*/engine-ctx$c.txt 2>/dev/null | head -2)"
done

hdr "2. THE SPLIT ARMS, 4K (4,148 prompt ids, 64 greedy) - ids md5 and the engine's own prompt line"
say "  all: config of record verbatim + --serve + --layer-split auto, ZE_AFFINITY_MASK unset, SYCL_CACHE_DIR=$R/sycl-cache/m6c"
say "  (the served path's own settings are the row with no extra flags)"
arm_ids s5-pc0 s5-pc6 s5-pc6-ns s5-pc6-ns-c s5-pc6-ttnon s5-pc6-ttlast s5-pc6-tt1tok s5-pc6-tt1tok-ns

hdr "3. THE ONE-CARD ARMS (ZE_AFFINITY_MASK=0, no --layer-split)"
arm_ids s5-pc0-1card s5-pc6-1card
say
say "  CPU experts per layer-window, from the arms' own 'strata decode timing' line:"
for t in s5-pc0-1card s5-pc6-1card; do say "    $t: $(grep -h 'strata decode timing' "$SRC/s5/runs/$t/err.txt" | sed 's/.*per layer-window: //')"; done

hdr "4. THE 32K ARMS (32,277 prompt ids, 256 greedy - the S4 length)"
arm_ids s5-pc0-32k s5-pc6-32k

hdr "5. THE COMPARISONS (s5/s5_cmp.py: md5 of the id list, first divergence)"
for t in s5-pc0 s5-pc6 s5-pc6-ns s5-pc6-ns-c s5-pc6-ttnon s5-pc6-ttlast s5-pc6-tt1tok s5-pc6-tt1tok-ns s5-pc0-1card s5-pc6-1card s5-pc0-32k s5-pc6-32k; do
  say "  $(printf '%-20s' "$t") $(/usr/bin/python3 "$SRC/s5/s5_cmp.py" table "$t" | tail -1)"
done
say
for p in "s5-pc0 s5-pc6" "s5-pc0 s5-pc6-ns" "s5-pc6 s5-pc6-ns" "s5-pc0 s5-pc6-ttnon" "s5-pc0 s5-pc6-ttlast" "s5-pc6 s5-pc6-tt1tok" "s5-pc0 s5-pc6-tt1tok-ns" "s5-pc0-1card s5-pc6-1card" "s5-pc0-32k s5-pc6-32k"; do
  say "--- div $p"
  /usr/bin/python3 "$SRC/s5/s5_cmp.py" div $p >> "$OUT" 2>&1
  say
done

hdr "6. THE ID RIG TIED TO THE SERVED TEXTS (the ids detokenized vs the server's own answer)"
for pair in "s5-pc0 s4/runs/s4-pc0-4k/resp-ctx4096.json" "s5-pc6 s4/runs/s4-ckpt-4k/resp-ctx4096.json"; do
  set -- $pair
  /usr/bin/python3 "$SRC/s5/s5_cmp.py" detok "$1" --served "$SRC/$2" >> "$OUT" 2>&1
  say
done

hdr "7. THE SERVED ARMS (serve.server, the config of record verbatim, one real /v1/chat/completions)"
for t in s5-serve-4k s5-serve-32k; do
  D=$SRC/s5/served/$t
  say "--- $t  (s4/runs/$t) ---"
  for f in "$D"/markers.tsv "$D"/argv.txt; do [ -r "$f" ] && { say "# $f"; cat "$f" >> "$OUT"; }; done
  say "# the engine's own lines for the request"
  grep -hE "strata serve: prompt |strata decode timing|checkpoint|HTTP|error" "$D"/engine-ctx*.txt "$D"/server-ctx*.txt 2>/dev/null | head -12 >> "$OUT"
  say "# the response body's usage + the answer (first 200 chars)"
  /usr/bin/python3 "$SRC/s4/s4_show.py" "$D"/resp-ctx*.json 2>/dev/null | head -8 >> "$OUT"
  say "# the response text file this card's ids were compared with"
  say "$(md5sum "$D"/response-text.txt 2>/dev/null) $(wc -c < "$D"/response-text.txt 2>/dev/null) bytes"
  say
done
say "--- the served texts against the driver ids of the same setting ---"
/usr/bin/python3 "$SRC/s5/s5_cmp.py" detok s5-pc6 --served "$SRC/s5/served/s5-serve-4k/resp-ctx4096.json" >> "$OUT" 2>&1
say
/usr/bin/python3 "$SRC/s5/s5_cmp.py" detok s5-pc6-32k --served "$SRC/s5/served/s5-serve-32k/resp-ctx32768.json" >> "$OUT" 2>&1

hdr "8. THE CODE (the operations the arms attribute the difference to)"
cd "$SRC" || exit 1
for spec in "408" "5111" "5119" "5310" "5319" "5326" "5334" "5363" "4394" "4448" "3794"; do
  say "# generate.cpp:$spec"
  sed -n "${spec}p" src/program/generate.cpp >> "$OUT"
done
say "# src/kernels/sycl/qsa_prompt_attn.cpp:1035"
sed -n '1035,1036p' src/kernels/sycl/qsa_prompt_attn.cpp >> "$OUT"
say "# docs/DETAILS.md (the paragraph this card added)"
git diff HEAD --stat -- docs/DETAILS.md >> "$OUT"
git diff HEAD -- docs/DETAILS.md >> "$OUT"

hdr "9. MACHINE STATE AT THE END OF THE EVIDENCE PASS"
say "engine  : $(pgrep -a -x strata || echo none)"
say "server  : $(pgrep -f '^/usr/bin/python3 -m serve\.server' || echo none)"
say "port 8099: $(ss -ltn 2>/dev/null | grep ':8099 ' || echo free)"
/usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3 >> "$OUT"
say "HEAD: $(cd "$SRC" && git log --oneline -1)"

echo "wrote $OUT ($(wc -l < "$OUT") lines)"
