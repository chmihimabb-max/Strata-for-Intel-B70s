#!/usr/bin/env bash
# S4 (card t_30d9ccfb): assemble the raw evidence for the write-up, straight from the arm directories - no
# transcription, every block is a file the arm wrote (or a git/diff command over the committed fix).
#
#   bash s4/s4_evidence.sh   ->  s4/S4-EVIDENCE.txt
R=/home/michael/strata-xpu
SRC=$R/strata
RUNS=$SRC/s4/runs
OUT=$SRC/s4/S4-EVIDENCE.txt
set +e

# the generated text of every completed request, in full, per arm (s4_text.py; a crashed arm has no body)
for tag in s4-prefix-sweep s4-prefix-32k s4-pc0-32k s4-fix-32k s4-fix-32k-2 s4-pc0-32k-2 s4-fix-16400 \
           s4-ckpt-root s4-oldckpt-32k s4-ckpt-4k s4-pc0-4k s4-closure-ckpt-4k s4-closure-pc0-4k; do
  d=$RUNS/$tag
  [ -d "$d" ] || continue
  : > "$d/response-text.txt"
  for f in "$d"/resp-ctx*.json; do
    [ -r "$f" ] && /usr/bin/python3 "$SRC/s4/s4_text.py" "$f" >> "$d/response-text.txt"
  done
done

hdr() { printf '\n\n================ %s ================\n' "$*"; }
show() {                       # show <label> <file> [lines]
  local label=$1 f=$2 n=${3:-0}
  hdr "$label   [$f]"
  if [ -r "$f" ]; then
    if [ "$n" -gt 0 ]; then tail -n "$n" "$f"; else cat "$f"; fi
  else
    echo "(missing: $f)"
  fi
}

{
  echo "S4 (t_30d9ccfb) - raw evidence"
  echo "assembled $(date -Is) from $RUNS"
  echo "HEAD: $(cd "$SRC" && git log --oneline -1)"
  echo "engine binary now: md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)"
  echo
  echo "THE ARMS"
  echo "  pre-fix  s4-prefix-sweep  4096:64 16000:64 16300:64 16400:64   config of record verbatim (checkpoints ON)"
  echo "  pre-fix  s4-prefix-32k    32768:256                            config of record verbatim (checkpoints ON)"
  echo "  control  s4-pc0-32k       32768:256                            + --prompt-cache 0 --prompt-cache-every 0"
  echo "  FIXED    s4-fix-32k       32768:256                            config of record verbatim (checkpoints ON)"
  echo "  FIXED    s4-fix-32k-2     32768:256                            the same arm again (run-to-run determinism)"
  echo "  control  s4-pc0-32k-2     32768:256                            the control again"
  echo "  FIXED    s4-fix-16400     16400:64                             the request class that used to fail"
  echo "  FIXED    s4-ckpt-root     32768:256  + --prompt-cache 6 --prompt-cache-every 0   (mid-prompt checkpoint off)"
  echo "  PRE-FIX  s4-oldckpt-32k   32768:256  + --prompt-cache 6 --prompt-cache-every 0   (the numerics control)"
  echo "  4K probe s4-ckpt-4k / s4-pc0-4k           4096:64, checkpoints on / off (graph path default)"
  echo "  4K probe s4-closure-ckpt-4k / -pc0-4k     4096:64, checkpoints on / off, STRATA_SYCL_GRAPH=0"

  hdr "1. PRE-FIX: the length sweep (one server lifetime, requests in order, stop at the first failure)"
  show "sweep markers" "$RUNS/s4-prefix-sweep/markers.tsv"
  show "sweep: the effective engine argv (the config of record verbatim, + the server's own --layer-split auto)" \
       "$RUNS/s4-prefix-sweep/argv.txt" 8
  show "sweep ctx=16400: the HTTP result" "$RUNS/s4-prefix-sweep/resp-ctx16400.json.timing.json" 20
  show "sweep ctx=16400: the engine's stderr (the failure)" "$RUNS/s4-prefix-sweep/engine-ctx16400.txt"
  show "sweep ctx=16400: the server's log (the prompt position it died at)" "$RUNS/s4-prefix-sweep/server-ctx16400.txt" 30
  for n in 4096 16000 16300; do
    show "sweep ctx=$n: the engine's own stats lines (this length PASSES)" \
         "$RUNS/s4-prefix-sweep/engine-ctx$n.txt" 60
    show "sweep ctx=$n: the client's timings" "$RUNS/s4-prefix-sweep/resp-ctx$n.json.timing.json" 20
  done

  hdr "2. PRE-FIX: the card's headline - a 32,277-token served request on the split"
  show "markers" "$RUNS/s4-prefix-32k/markers.tsv"
  show "the HTTP result" "$RUNS/s4-prefix-32k/resp-ctx32768.json.timing.json" 20
  show "the raw error body" "$RUNS/s4-prefix-32k/resp-ctx32768.json.error"
  show "the engine's stderr (last 12 lines)" "$RUNS/s4-prefix-32k/engine-ctx32768.txt" 12
  show "the server's log (last 10 lines)" "$RUNS/s4-prefix-32k/server-ctx32768.txt" 10

  hdr "3. THE CONTROL: the same request with --prompt-cache 0 --prompt-cache-every 0 (config of record + extras)"
  show "the effective engine argv / the extras" "$RUNS/s4-pc0-32k/argv.txt" 8
  show "the HTTP result" "$RUNS/s4-pc0-32k/resp-ctx32768.json.timing.json" 12
  show "the engine's own stats lines" "$RUNS/s4-pc0-32k/engine-ctx32768.txt" 60

  hdr "4. THE FIX (git)"
  (cd "$SRC" && git show --stat HEAD)
  (cd "$SRC" && git show HEAD -- src/program/generate.cpp)

  hdr "5. FIXED: the same 32,277-token request, config of record verbatim"
  show "the effective engine argv (config of record verbatim)" "$RUNS/s4-fix-32k/argv.txt" 8
  show "the HTTP result" "$RUNS/s4-fix-32k/resp-ctx32768.json.timing.json" 12
  show "the response text (s4_text.py, in full)" "$RUNS/s4-fix-32k/response-text.txt"
  show "the engine's own stats lines (residency, cache hit, tok/s, checkpoints)" \
       "$RUNS/s4-fix-32k/engine-ctx32768.txt" 60

  hdr "6. FIXED: the request class that used to fail (16,452 prompt tokens, the 16,384-token checkpoint)"
  show "markers" "$RUNS/s4-fix-16400/markers.tsv"
  show "the HTTP result" "$RUNS/s4-fix-16400/resp-ctx16400.json.timing.json" 12
  show "the engine's own stats lines" "$RUNS/s4-fix-16400/engine-ctx16400.txt" 60

  hdr "7. DETERMINISM: the same arm twice, per configuration, and the two configurations against each other"
  for tag in s4-fix-32k s4-fix-32k-2; do
    show "$tag: usage/timings" "$RUNS/$tag/resp-ctx32768.json.timing.json" 12
  done
  for tag in s4-pc0-32k s4-pc0-32k-2 s4-ckpt-root s4-oldckpt-32k; do
    show "$tag: usage/timings" "$RUNS/$tag/resp-ctx32768.json.timing.json" 12
    show "$tag: the engine's own stats lines" "$RUNS/$tag/engine-ctx32768.txt" 60
  done
  echo "--- the text comparisons (s4_cmp.py: md5 of each generated text field) ---"
  cmp2() { (cd "$SRC" && /usr/bin/python3 s4/s4_cmp.py "s4/runs/$1/resp-$3.json" "s4/runs/$2/resp-$3.json"); }
  cmp2 s4-fix-32k    s4-fix-32k-2    ctx32768
  cmp2 s4-pc0-32k    s4-pc0-32k-2    ctx32768
  cmp2 s4-pc0-32k-2  s4-fix-32k-2    ctx32768
  cmp2 s4-ckpt-root  s4-fix-32k-2    ctx32768
  cmp2 s4-oldckpt-32k s4-ckpt-root   ctx32768
  cmp2 s4-oldckpt-32k s4-fix-32k-2   ctx32768
  cmp2 s4-ckpt-4k    s4-pc0-4k       ctx4096
  cmp2 s4-closure-ckpt-4k s4-closure-pc0-4k ctx4096
  cmp2 s4-ckpt-4k    s4-closure-ckpt-4k  ctx4096

  hdr "8. THE SUITE (s4_ctest.sh: the fixed binary, ZE_AFFINITY_MASK=0)"
  show "notes (HEAD, binary md5, which tests mention the checkpoint path)" "$RUNS/ctest/notes.txt"
  show "the checkpoint-path test" "$RUNS/ctest/checkpoint-tests.log" 12
  (cd "$SRC" && grep -E "tests passed|tests failed|\(Failed\)" s4/runs/ctest/full.log | tail -8)
  echo "--- D1's pre-fix baseline, same file ---"
  (cd "$SRC" && grep -E "tests passed|tests failed|\(Failed\)" /home/michael/strata-xpu/d1/runs/ctest/full-graphdefault.log | tail -8)

  hdr "9. THE ARMS' OWN LOGS (head of each: HEAD, binary md5, argv, device state)"
  for tag in s4-prefix-sweep s4-prefix-32k s4-pc0-32k s4-fix-32k s4-fix-32k-2 s4-pc0-32k-2 s4-fix-16400 \
             s4-ckpt-root s4-oldckpt-32k s4-ckpt-4k s4-pc0-4k s4-closure-ckpt-4k s4-closure-pc0-4k; do
    show "$tag/log.txt (first 30 lines)" "$RUNS/$tag/log.txt" 0
  done

  hdr "10. MACHINE STATE AT THE END OF THE EVIDENCE RUN"
  date -Is
  pgrep -a -x strata || echo "no engine"
  pgrep -af '^/usr/bin/python3 -m serve\.server' || echo "no server"
  ss -ltn 2>/dev/null | grep ":8099 " || echo "port 8099 free"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
} > "$OUT" 2>&1
echo "wrote $OUT  ($(wc -l < "$OUT") lines)"
