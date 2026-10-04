#!/usr/bin/env bash
# S5 (card t_aae723be): the DRIVER arm - the engine binary on the config of record's own argv, ONE `GEN` line on
# stdin, the greedy `T` ids out.  No server, no monitoring, so an arm is ~70 s.
#
#   bash s5/s5_drive.sh <TAG> <GENLINE_FILE> [--extra "ARGS"] [--onecard N] [--env VAR=VAL]... [--timeout S]
#
# Everything lands in s5/runs/<TAG>/ : out.txt (stdout: READY/PP/T/stats), err.txt (stderr), ids.txt (the greedy T
# ids, one per line) + ids.md5, argv.txt (the exact argv, with the config of record's args byte-identical), meta.txt.
R=/home/michael/strata-xpu
SRC=$R/strata
set +e

TAG=${1:?tag}; GEN=${2:?genline}; shift 2
EXTRA=""; ONECARD=""; ENVS=(); TMO=1800
while [ $# -gt 0 ]; do
  case "$1" in
    --extra) EXTRA=$2; shift 2 ;;
    --onecard) ONECARD=$2; shift 2 ;;
    --env) ENVS+=("$2"); shift 2 ;;
    --timeout) TMO=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
[ -r "$GEN" ] || { echo "no GEN line file $GEN"; exit 2; }
GEN=$(readlink -f "$GEN")       # the run happens in $D, so the redirect needs an absolute path

D=$SRC/s5/runs/$TAG
CACHEDIR=$R/sycl-cache/m6c
mkdir -p "$D"
: > "$D/out.txt"; : > "$D/err.txt"

# the config of record's args, verbatim, one per line (read through JSON, not retyped)
mapfile -t BASE < <(/usr/bin/python3 "$SRC/s5/s5_argv.py")
ENGINE=("$SRC/build-sycl/strata" --serve "${BASE[@]}")
if [ -n "$ONECARD" ]; then
  export ZE_AFFINITY_MASK="$ONECARD"          # a single-card instance: the mask is the only way to give it one card
else
  unset ZE_AFFINITY_MASK                      # the two-GPU rule (PLAN 11 U11): the split needs it unset
  ENGINE+=(--layer-split auto)                # what serve/server.py appends for 'gpu': [0,1]
fi
[ -n "$EXTRA" ] && ENGINE+=($EXTRA)

# clear the A/B switches a previous arm may have exported (a leaked switch silently measures the other path)
unset STRATA_PROMPT_ATTN_OLD STRATA_PREFILL_TRACE STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_TRACE STRATA_DBG_NAN STRATA_DUMP_LADDER STRATA_VERIFY_TAIL_DEBUG STRATA_DEC_BATCH STRATA_HC_SPLIT
unset STRATA_KV_ROT STRATA_GR_V3 STRATA_SEL_GFX12 STRATA_REFILL_BLOCKING STRATA_RESIDENT_PIN STRATA_SPIN_PAUSE
unset STRATA_PA_WMMA STRATA_COMMIT_SYNC STRATA_VERIFY_DEVICE_PLAN STRATA_WINDOW_PLAIN_GR STRATA_SYCL_GRAPH
unset STRATA_VERIFY_PROFILE STRATA_CKPT_REREAD STRATA_SNAPSHOT_VERIFY
for kv in "${ENVS[@]}"; do export "$kv"; done
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="$CACHEDIR"
export STRATA_DECODE_TIMING=1 STRATA_FINALIZER_WAIT_S=10

source /opt/intel/oneapi/setvars.sh > "$D/setvars.log" 2>&1
cd "$D" || exit 1

{
  echo "=== S5 DRIVER arm $TAG   $(date -Is)"
  echo "HEAD        : $(cd "$SRC" && git log --oneline -1)"
  echo "engine      : $(stat -c '%y %s bytes' "$SRC/build-sycl/strata")  md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)"
  echo "GEN line    : $GEN  ($(wc -c < "$GEN") B, $(awk '{n=split($3,a,","); print n}' "$GEN") ids)"
  echo "extra       : ${EXTRA:-<none>}   env: ${ENVS[*]:-<none>}"
  echo "onecard     : ${ONECARD:-<none>}  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
  echo "engine argv : ${ENGINE[*]}"
  echo "argv sha256 : $(printf '%s\n' "${ENGINE[@]}" | sha256sum | cut -d' ' -f1)"
  echo "-- device check BEFORE --"
  pgrep -x strata || echo "   no engine"
  pgrep -f '^/usr/bin/python3 -m serve\.server' || echo "   no server"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
  cat /proc/loadavg; free -g | head -2
} > "$D/meta.txt" 2>&1
printf '%s\n' "${ENGINE[@]}" > "$D/argv.txt"

echo "== running (timeout ${TMO}s) $(date -Is) ==" >> "$D/meta.txt"
T0=$(date +%s%N)
timeout -k 20 "$TMO" "${ENGINE[@]}" < "$GEN" > "$D/out.txt" 2> "$D/err.txt"
RC=$?
T1=$(date +%s%N)

grep -E '^T -?[0-9]+$' "$D/out.txt" | awk '{print $2}' > "$D/ids.txt"
md5sum "$D/ids.txt" | cut -d' ' -f1 > "$D/ids.md5"
{
  echo "rc=$RC  wall=$(( (T1 - T0) / 1000000 ))ms  $(date -Is)"
  echo "T lines     : $(wc -l < "$D/ids.txt")   ids md5: $(cat "$D/ids.md5")"
  echo "unique ids  : $(sort -u "$D/ids.txt" | wc -l)"
  echo "-- the engine's own prompt/checkpoint line --"
  grep -hE "strata serve: prompt |checkpoints" "$D/out.txt" "$D/err.txt" | head -5
  echo "-- errors --"
  grep -hE "ERR |error|failed|abort" "$D/out.txt" "$D/err.txt" | head -10
  echo "-- what is left --"
  pgrep -x strata || echo "   no engine"
  /usr/bin/python3 "$R/scripts/m6_occupancy.py" 2>&1 | tail -3
} >> "$D/meta.txt"
echo "=== S5 arm $TAG: rc=$RC, $(wc -l < "$D/ids.txt") ids, md5 $(cat "$D/ids.md5"); $D ==="
