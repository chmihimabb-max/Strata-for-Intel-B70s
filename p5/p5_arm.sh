#!/usr/bin/env bash
# P5 (card t_bc967b65): ONE engine arm, config of record, driven through the engine's own serve protocol.
#
#   bash p5/p5_arm.sh <TAG> <CTX> <shipped|batched> [MAXNEW] [int8|fp16] [auto|512]
#
# The A/B switch is set HERE and only here (P2's pitfall: a STRATA_PROMPT_ATTN_OLD left exported by a previous arm
# makes the NEXT arm measure the old path and look like a 2.2x win), and the arm's signature is recorded from the
# log: the shipped path prints `qsa_prompt_attn_batch: SYCL backend -> portable v1 kernel ...` at load, the batched
# path does not (prefill.cpp:1621 short-circuits before the call).
#
# CTX decides the prompt file, exactly as P2/P1b's arms did:
#   4096   -> m6c/prompts/prompt-needle-ctx4096.txt   3831 ids   --prefill 512
#   32768  -> m6c/prompts/prompt-ctx32768.txt        32255 ids   --prefill auto (8192)
#   131072 -> m6c/prompts/prompt-ctx131072.txt      129023 ids   --prefill auto (8192)
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; CTX=${2:?ctx}; ARM=${3:?shipped|batched}
MAXNEW=${4:-256}; KV=${5:-int8}; PREFILL=${6:-}
case "$CTX" in
  4096)   PROMPT=$R/m6c/prompts/prompt-needle-ctx4096.txt ; PREFILL=${PREFILL:-512}  ; KVRES=32768 ;;
  32768)  PROMPT=$R/m6c/prompts/prompt-ctx32768.txt       ; PREFILL=${PREFILL:-auto} ; KVRES=$CTX ;;
  131072) PROMPT=$R/m6c/prompts/prompt-ctx131072.txt      ; PREFILL=${PREFILL:-auto} ; KVRES=$CTX ;;
  *) echo "p5_arm: unsupported ctx $CTX"; exit 2 ;;
esac
[ -r "$PROMPT" ] || { echo "p5_arm: no prompt $PROMPT"; exit 2; }
case "$ARM" in shipped|batched) ;; *) echo "p5_arm: arm must be shipped|batched"; exit 2 ;; esac

D=$R/p5/runs/$TAG
mkdir -p "$D"
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt; TL=$D/timeline.txt
MON=$D/rss.csv; PIDF=$D/engine.pid; STDIN=$D/stdin.txt
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MTP=/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0

printf 'GEN %s %s\nQUIT\n' "$MAXNEW" "$(cat "$PROMPT")" > "$STDIN"

{
  echo "=============================================================="
  echo "P5 engine arm $TAG   $(date -Is)"
  echo "arm=$ARM  ctx=$CTX  max-new=$MAXNEW  kv=$KV  prefill=$PREFILL  kv-resident=$KVRES"
  echo "prompt=$PROMPT ($(wc -c < "$PROMPT") B of ids, $(tr ',' '\n' < "$PROMPT" | grep -c '[0-9]') tokens)"
  echo "binary: $(stat -c '%y %s bytes' $SRC/build-sycl/strata)   md5 $(md5sum $SRC/build-sycl/strata | cut -c1-32)"
  echo "HEAD: $(cd $SRC && git log --oneline -1)"
  echo "SYCL_CACHE_DIR=$R/sycl-cache/m6c (warm, per P2's record)"
  echo
  echo "== device check BEFORE the run (PLAN 9 rule 5) =="
  pgrep -a -f "build-sycl/strata|serve/server.py|llama-server -m" || echo "   no engine/oracle running"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 || true
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -12
  cat /proc/loadavg; free -g | head -2
  echo "== end of the device check =="
} > "$LOG" 2>&1

# --- the A/B switch environment: cleared, then set for the batched arm only ---
unset ZE_AFFINITY_MASK STRATA_PROMPT_ATTN_OLD STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_PREFILL_TRACE STRATA_PREFILL_TRACE_SYNC STRATA_TEST_VERIFY_STALL STRATA_SYCL_GRAPH
if [ "$ARM" = batched ]; then export STRATA_PROMPT_ATTN_OLD=1; fi
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=$R/sycl-cache/m6c
export STRATA_DECODE_TIMING=1
export STRATA_PREFILL_TIMING=1

source /opt/intel/oneapi/setvars.sh >> "$LOG" 2>&1

ENGINE=(./build-sycl/strata --serve
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2" --mtp "$MTP"
  --kv "$KV" --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts
  --prefill "$PREFILL" --spec 4 --spec-min-p 0.5 --max-context "$CTX" --no-capture --stats
  --layer-split auto --prompt-cache 0 --prompt-cache-every 0 --kv-resident "$KVRES")

{
  echo
  echo "== the exact engine command line =="
  echo "cd $SRC && ZE_AFFINITY_MASK=<unset> STRATA_PROMPT_ATTN_OLD=${STRATA_PROMPT_ATTN_OLD:-<unset>} \\"
  echo "  SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=$SYCL_CACHE_DIR \\"
  echo "  STRATA_DECODE_TIMING=1 STRATA_PREFILL_TIMING=1 \\"
  echo "  ${ENGINE[*]}"
  echo "  < 'GEN $MAXNEW <$(( $(wc -c < "$PROMPT") / 1024 )) KiB of ids>' + QUIT"
  echo
  echo "== page cache just before the engine starts =="
  grep -E "^(MemTotal|MemFree|MemAvailable|Cached)" /proc/meminfo
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
grep -E '^PP |^DONE|^ERR' "$OUT" > "$D/protocol.txt"

{
  echo
  echo "== engine exit $RC, wall ${WALL}s =="
  echo "== WHICH PROMPT-ATTENTION PATH RAN (P2's signature; present on the shipped arm only) =="
  grep -n "qsa_prompt_attn_batch: SYCL backend" "$ERR" || echo "   (no 'portable v1 kernel' line: the batched/decode path ran, as STRATA_PROMPT_ATTN_OLD=1 asks)"
  grep -nE "^strata: STRATA" "$ERR" | head -10 || true
  echo
  echo "== the request's own numbers =="
  grep -E "^PP |^DONE|^ERR" "$OUT" | tail -6
  grep -E "strata serve: prompt|strata serve: |expert cache hit|CPU experts" "$ERR" | tail -8
  echo
  echo "== per-chunk prefill phase totals (STRATA_PREFILL_TIMING=1) =="
  grep -cE "^strata prefill timing: (\[chunk\] )?[0-9]+ tokens" "$ERR"
  grep -E "^strata prefill timing: (\[chunk\] )?[0-9]+ tokens" "$ERR" | sed -E 's/^.*(qsa attn [0-9]+ \([0-9.]+%\)).*$/\1/' | tr '\n' '|'; echo
  grep "^strata prefill timing: [0-9]" "$ERR" | tail -1
  grep "^strata prefill timing: host:" "$ERR" | tail -1
  echo
  echo "== generated tokens =="
  echo "T lines: $(cat "$D/tcount.txt")   md5 $(md5sum "$D/tokens.txt" | cut -d' ' -f1)"
  head -12 "$D/tokens.txt" | tr '\n' ' '; echo
  echo
  echo "== peak tree RSS / VRAM (m6c_sample.py) =="
  head -1 "$MON"; sort -t, -k4 -n "$MON" | tail -1
} >> "$LOG" 2>&1
echo "=== p5 arm $TAG done: exit $RC, wall ${WALL}s, tokens $(cat "$D/tcount.txt"), arm=$ARM, log $LOG ==="
