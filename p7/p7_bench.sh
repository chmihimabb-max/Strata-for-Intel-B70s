#!/usr/bin/env bash
# P7 (card t_6789d6df): the headless-vs-drafter CONTROL, on the engine's own bench path.
#
#   bash p7/p7_bench.sh <TAG> <CTX> <mtp|headless>
#
# `--serve` REFUSES to start without `--mtp` ("strata serve: needs --spec T, --mtp DIR and --prefill CHUNK"), so the
# headless arm cannot be measured through the resident server at all.  The bench path (`strata generate`) is where
# upstream makes `--mtp` optional, so both arms are measured there, with the same prompt, context, window, KV mode
# and expert-cache settings - the only difference is the drafter.
#
# The bench path prints what the card asks for: `speculation <rounds> rounds of <T>, drafts accepted A of B (p),
# X tokens per round`, `suffix drafts ...`, and `decode N tokens in M ms -> Z tok/s`.
R=/home/michael/strata-xpu
SRC=$R/strata
TAG=${1:?tag}; CTX=${2:?ctx}; ARM=${3:?mtp|headless}
MAXNEW=${4:-256}
case "$ARM" in mtp|headless) ;; *) echo "p7_bench: arm must be mtp|headless"; exit 2 ;; esac
case "$CTX" in
  4096)  PROMPT=$R/m6c/prompts/prompt-needle-ctx4096.txt ; PREFILL=512  ; KVRES=32768 ;;
  32768) PROMPT=$R/m6c/prompts/prompt-ctx32768.txt       ; PREFILL=auto ; KVRES=$CTX   ;;
  *) echo "p7_bench: unsupported ctx $CTX"; exit 2 ;;
esac
[ -r "$PROMPT" ] || { echo "p7_bench: no prompt $PROMPT"; exit 2; }
if [ "$ARM" = mtp ]; then MTP=$R/mtp/rt; else MTP=""; fi

D=$R/p7/runs/$TAG
mkdir -p "$D"
OUT=$D/out.txt; ERR=$D/err.txt; LOG=$D/log.txt
PACK=/run/media/michael/2208B12208B0F63F/strata-iq3s/pack
SNAP=$HOME/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
SH1=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SH2=$SNAP/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf

unset ZE_AFFINITY_MASK STRATA_PROMPT_ATTN_OLD STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_PREFILL_TRACE STRATA_PREFILL_TRACE_SYNC STRATA_TEST_VERIFY_STALL STRATA_SYCL_GRAPH
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=$R/sycl-cache/m6c
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

ENGINE=(./build-sycl/strata
  --pack "$PACK" --native "$SH1" --ple-gguf "$SH2"
  --tokens-file "$PROMPT" --max-new "$MAXNEW" --max-context "$CTX"
  --kv int8 --kv-resident "$KVRES"
  --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts
  --prefill "$PREFILL" --spec 4 --spec-min-p 0.5 --no-capture --stats)
if [ -n "$MTP" ]; then ENGINE+=(--mtp "$MTP"); fi

{
  echo "=============================================================="
  echo "P7 bench arm $TAG   $(date -Is)   (arm=$ARM)"
  echo "mtp runtime: ${MTP:-<none: headless>}"
  echo "prompt=$PROMPT  max-new=$MAXNEW  max-context=$CTX  kv=int8  kv-resident=$KVRES  prefill=$PREFILL"
  echo "binary md5 $(md5sum $SRC/build-sycl/strata | cut -c1-32)   HEAD $(cd $SRC && git log --oneline -1)"
  echo "cd $SRC && ZE_AFFINITY_MASK=<unset> SYCL_CACHE_DIR=$R/sycl-cache/m6c \\"
  echo "  ${ENGINE[*]}"
} > "$LOG" 2>&1

cd "$SRC" || exit 1
START=$(date +%s)
"${ENGINE[@]}" > "$OUT" 2> "$ERR"
RC=$?
WALL=$(( $(date +%s) - START ))

{
  echo "== engine exit $RC, wall ${WALL}s =="
  echo
  echo "== the bench path's own numbers =="
  grep -E "^decode |^prefill |^speculation|^suffix drafts|^accepted per round|^window sizes|^verify window" "$OUT"
  echo
  echo "== what the drafter cost in VRAM (engine's own line) =="
  grep -E "draft layer loaded|draft head over|expert cache auto|expert cache [0-9]+ slots" "$ERR"
  echo
  echo "== generated tokens =="
  grep -c "^T \|generated" "$OUT" >/dev/null 2>&1
  grep -E "^decode |^prefill " "$OUT"
} >> "$LOG" 2>&1
echo "=== p7 bench arm $TAG done: exit $RC, wall ${WALL}s, arm=$ARM, mtp=${MTP:-<none>} ==="
