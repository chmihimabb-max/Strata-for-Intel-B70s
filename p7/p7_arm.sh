#!/usr/bin/env bash
# P7 (card t_6789d6df): ONE engine arm, config of record, driven through the engine's own serve protocol.
#
#   bash p7/p7_arm.sh <TAG> <CTX> <new|old|headless> [MAXNEW] [KV]
#
#   new      --mtp /home/michael/strata-xpu/mtp/rt                     (canonical checkpoint, upstream's tools)
#   old      --mtp /run/media/.../strata-w4a16/mtp-bf16/rt-q2_0        (the W4A16-derived artifact being replaced)
#   headless no --mtp at all: suffix/prompt-lookup drafting only (the control)
#
# Everything else is the config of record (strata-sycl-iq3s.json), so the three arms differ in exactly one thing.
# ZE_AFFINITY_MASK is cleared here and only here; `--spec 4 --spec-min-p 0.5` is the config's window.
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; CTX=${2:?ctx}; ARM=${3:?new|old|headless}
MAXNEW=${4:-256}; KV=${5:-int8}
case "$ARM" in new|old|new-nodv|headless) ;; *) echo "p7_arm: arm must be new|old|new-nodv|headless"; exit 2 ;; esac
case "$CTX" in
  4096)   PROMPT=$R/m6c/prompts/prompt-needle-ctx4096.txt ; PREFILL=512  ; KVRES=32768 ;;
  32768)  PROMPT=$R/m6c/prompts/prompt-ctx32768.txt       ; PREFILL=auto ; KVRES=$CTX   ;;
  *) echo "p7_arm: unsupported ctx $CTX"; exit 2 ;;
esac
[ -r "$PROMPT" ] || { echo "p7_arm: no prompt $PROMPT"; exit 2; }

case "$ARM" in
  new)      MTP=$R/mtp/rt ;;
  new-nodv) MTP=$R/mtp/rt-noDV ;;
  old)      MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0 ;;
  headless) MTP="" ;;
esac
if [ -n "$MTP" ]; then
  [ -r "$MTP/dense.txt" ] || { echo "p7_arm: no MTP runtime at $MTP"; exit 2; }
  # draft_vocab.bin is an OPTIONAL step of upstream's recipe (docs/ORCA.md:39): the engine uses the head over that
  # token subset when the file is there, and the whole vocabulary when it is not.  The W4A16-derived artifact never
  # had it, so its absence is recorded, not fatal.
  if [ -r "$MTP/draft_vocab.bin" ]; then
    DV="present ($(stat -c %s "$MTP/draft_vocab.bin") B)"
  else
    DV="ABSENT (the engine drafts over the whole vocabulary)"
  fi
fi

D=$R/p7/runs/$TAG
mkdir -p "$D"
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt
MON=$D/rss.csv; PIDF=$D/engine.pid; STDIN=$D/stdin.txt
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf

printf 'GEN %s %s\nQUIT\n' "$MAXNEW" "$(cat "$PROMPT")" > "$STDIN"

{
  echo "=============================================================="
  echo "P7 engine arm $TAG   $(date -Is)"
  echo "arm=$ARM  ctx=$CTX  max-new=$MAXNEW  kv=$KV  prefill=$PREFILL  kv-resident=$KVRES"
  echo "mtp runtime: ${MTP:-<none: headless>}"
  echo "mtp draft_vocab: ${DV:-<n/a>}"
  echo "prompt=$PROMPT ($(wc -c < "$PROMPT") B of ids)"
  echo "binary: $(stat -c '%y %s bytes' $SRC/build-sycl/strata)   md5 $(md5sum $SRC/build-sycl/strata | cut -c1-32)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo
  echo "== device check BEFORE the run =="
  pgrep -a -f "build-sycl/strata|serve/server.py" || echo "   no engine running"
  cat /proc/loadavg; free -g | head -2
  echo "== end of the device check =="
} > "$LOG" 2>&1

unset ZE_AFFINITY_MASK STRATA_PROMPT_ATTN_OLD STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_PREFILL_TRACE STRATA_PREFILL_TRACE_SYNC STRATA_TEST_VERIFY_STALL STRATA_SYCL_GRAPH
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=$R/sycl-cache/m6c
export STRATA_DECODE_TIMING=1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1

ENGINE=(./build-sycl/strata --serve
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2"
  --kv "$KV" --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts
  --prefill "$PREFILL" --spec 4 --spec-min-p 0.5 --max-context "$CTX" --no-capture --stats
  --layer-split auto --prompt-cache 0 --prompt-cache-every 0 --kv-resident "$KVRES")
if [ -n "$MTP" ]; then ENGINE+=(--mtp "$MTP"); fi

{
  echo "cd $SRC && ZE_AFFINITY_MASK=<unset> \\"
  echo "  SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$SYCL_CACHE_DIR STRATA_DECODE_TIMING=1 \\"
  echo "  ${ENGINE[*]}"
  echo "  < 'GEN $MAXNEW <prompt ids>' + QUIT"
} >> "$LOG" 2>&1

cd "$SRC" || exit 1
START=$(date +%s)
timeout -s INT 7200 /usr/bin/python3 "$SRC/m6c/m6c_drive.py" \
  --stdin "$STDIN" --out "$OUT" --err "$ERR" --timeline "$TL" --pidfile "$PIDF" -- \
  "${ENGINE[@]}" >> "$LOG" 2>&1 &
DPID=$!
sleep 3
/usr/bin/python3 "$SRC/m6c/m6c_sample.py" "$DPID" "$MON" 7200 > "$D/rss.txt" 2>&1 &
SPID=$!
wait $DPID
RC=$?
kill $SPID 2>/dev/null
wait $SPID 2>/dev/null
WALL=$(( $(date +%s) - START ))

grep -c '^T ' "$OUT" > "$D/tcount.txt"
grep '^T ' "$OUT" > "$D/tokens.txt"
grep -E '^PP |^DONE|^ERR|^INFO' "$OUT" > "$D/protocol.txt"

# DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> ...
{
  echo
  echo "== engine exit $RC, wall ${WALL}s =="
  grep -E '^INFO ' "$OUT" | head -1
  echo
  echo "== the request's own numbers =="
  grep -E '^DONE|^ERR' "$OUT"
  grep -E "strata serve: prompt |drafts accepted" "$ERR" | tail -3
  grep -E "suffix drafts" "$ERR" | tail -2 || true
  echo
  echo "== decode timing (STRATA_DECODE_TIMING=1), last request =="
  grep -E "^strata decode timing" "$ERR" | tail -1
  echo
  echo "== generated tokens =="
  echo "T lines: $(cat "$D/tcount.txt")   md5 $(md5sum "$D/tokens.txt" | cut -d' ' -f1)"
  head -12 "$D/tokens.txt" | tr '\n' ' '; echo
  echo
  echo "== peak tree RSS (m6c_sample.py) =="
  head -1 "$MON"; sort -t, -k4 -n "$MON" | tail -1
} >> "$LOG" 2>&1
echo "=== p7 arm $TAG done: exit $RC, wall ${WALL}s, tokens $(cat "$D/tcount.txt"), arm=$ARM, mtp=${MTP:-<none>} ==="
