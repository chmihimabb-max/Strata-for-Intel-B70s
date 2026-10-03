#!/usr/bin/env bash
# P3 (t_d8afe53f) arm runner: one m6c serve arm at the config of record, from an EXPLICIT binary, with the
# submission counter on and the A/B switches a previous arm may have left in the shell cleared first.
#
#   bash p3/p3_run_arm.sh <TAG> <BIN> <CTX> [m6c_serve args...]
#     4K arm : bash p3/p3_run_arm.sh base-4k p3/strata-before-P3 4096 --prefill 512 150 --prompt <ids-file>
#     32K arm: bash p3/p3_run_arm.sh base-32k p3/strata-before-P3 32768 --prefill-auto 256
#
# The harness (strata/m6c/m6c_serve.sh) runs ./build-sycl/strata, so the arm's binary is swapped in there and
# verified by sha256 before and after; the previous binary is restored on exit.
R=/home/michael/strata-xpu
set +e
TAG=${1:?tag}; BIN=${2:?binary}; CTX=${3:?ctx}; shift 3
case "$BIN" in
  /*) ;;
  *) BIN="$R/strata/$BIN" ;;
esac
cd "$R/strata" || exit 1
test -x "$BIN" || { echo "no binary $BIN"; exit 2; }
# clear the A/B switches by hand (an exported switch survives into the next arm - measured in P2)
unset STRATA_PROMPT_ATTN_OLD STRATA_PREFILL_TRACE STRATA_PROMPT_ATTN_V1 STRATA_QSA_WARP STRATA_SYCL_XMX
unset STRATA_VERIFY_PROFILE STRATA_PREFILL_TIMING STRATA_TRACE STRATA_DBG_NAN STRATA_DUMP_LADDER
unset STRATA_VERIFY_TAIL_DEBUG
# the graph-replay arm: P3_GRAPH=1 sets STRATA_SYCL_GRAPH on the ENGINE (the switch is read once, at load)
if [ -n "${P3_GRAPH:-}" ]; then export STRATA_SYCL_GRAPH="$P3_GRAPH"; else unset STRATA_SYCL_GRAPH; fi
export STRATA_SUBMIT_COUNT=${STRATA_SUBMIT_COUNT:-1}
KEEP=$(mktemp /tmp/p3keep.XXXXXX)
cp build-sycl/strata "$KEEP"
cp "$BIN" build-sycl/strata
WANT=$(sha256sum < "$BIN" | cut -d' ' -f1)
GOT=$(sha256sum < build-sycl/strata | cut -d' ' -f1)
if [ "$WANT" != "$GOT" ]; then echo "binary swap failed"; cp "$KEEP" build-sycl/strata; exit 3; fi
echo "== P3 arm $TAG: bin=$BIN sha256=${WANT:0:16} ctx=$CTX STRATA_SUBMIT_COUNT=$STRATA_SUBMIT_COUNT args: $*" >&2
bash m6c/m6c_serve.sh "$TAG" "$CTX" "$@"
RC=$?
cp "$KEEP" build-sycl/strata
rm -f "$KEEP"
exit $RC
