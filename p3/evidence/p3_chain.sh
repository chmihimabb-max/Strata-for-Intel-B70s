#!/usr/bin/env bash
# P3 (t_d8afe53f) measurement chain: the same binary (p3/strata-graph-P3) with STRATA_SYCL_GRAPH off and on,
# at 4K / 32K / 128K, config of record (2 GPUs, --kv int8, --kv-resident 32768, MTP on, --prefill auto).
# One engine at a time; each arm's log is p3/<TAG>-run.log and the harness's own output is m6c/runs/<TAG>/.
set +e
R=/home/michael/strata-xpu
BIN=$R/p3/strata-graph-P3
P=$R/m6c/prompts
run() {   # run <TAG> <CTX> [maxnew] [--prompt FILE]
  TAG=$1; CTX=$2; shift 2
  echo "### $TAG  $(date -Is)" >> $R/p3/chain.log
  P3_GRAPH="$GRAPH" bash $R/p3/p3_run_arm.sh "$TAG" "$BIN" "$CTX" "$@" >> $R/p3/$TAG-run.log 2>&1
  echo "### $TAG exit=$? $(date -Is)" >> $R/p3/chain.log
}
GRAPH=  run p3-4k-closed  4096  --prefill-auto 150 --prompt $P/prompt-needle-ctx4096.txt
GRAPH=1 run p3-4k-graph2  4096  --prefill-auto 150 --prompt $P/prompt-needle-ctx4096.txt
GRAPH=  run p3-32k-closed 32768 --prefill-auto 256
GRAPH=1 run p3-32k-graph  32768 --prefill-auto 256
GRAPH=  run p3-128k-closed 131072 --prefill-auto 256
GRAPH=1 run p3-128k-graph  131072 --prefill-auto 256
echo "### chain done $(date -Is)" >> $R/p3/chain.log
