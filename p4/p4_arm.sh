#!/usr/bin/env bash
# P4 (card t_63cc226b) engine arm: one m6c serve arm at the config of record with a CHOSEN copy-kernel launch
# shape (STRATA_KV_COPY_BLOCKS / STRATA_KV_COPY_THREADS, read once per process by kv_stream.cu).  The binary is
# always the current build-sycl/strata, so the two arms of an A/B differ by the environment alone.
#
#   bash p4/p4_arm.sh <TAG> default 131072 256     # the shipped 96 x 128
#   bash p4/p4_arm.sh <TAG> 1x128   131072 256     # the paper's confinement, at the port's thread count
#
# 128K is where KV streaming is engaged on this box: --kv-resident 32768 of 131072 cells (at 32K everything is
# resident and the copy kernel never runs - see qsa_stream_policy in src/core/layer.cpp).
R=/home/michael/strata-xpu
set -e
TAG=${1:?tag}; SHAPE=${2:?shape (default|BxT)}; CTX=${3:-131072}; MAXNEW=${4:-256}
cd "$R/strata"
if [ "$SHAPE" = "default" ]; then
  unset STRATA_KV_COPY_BLOCKS STRATA_KV_COPY_THREADS
else
  export STRATA_KV_COPY_BLOCKS=${SHAPE%x*}
  export STRATA_KV_COPY_THREADS=${SHAPE#*x}
fi
echo "== P4 arm $TAG: shape=$SHAPE blocks=${STRATA_KV_COPY_BLOCKS:-96} threads=${STRATA_KV_COPY_THREADS:-128} ctx=$CTX maxnew=$MAXNEW" >&2
echo "== binary $(sha256sum < build-sycl/strata | cut -c1-16) HEAD $(git log --oneline -1)" >&2
bash m6c/m6c_serve.sh "$TAG" "$CTX" --prefill-auto "$MAXNEW"
