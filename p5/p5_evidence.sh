#!/usr/bin/env bash
# P5: regenerate every text artifact the report quotes, from the runs on disk.  Re-runnable; no GPU needed.
#   usage: bash p5/p5_evidence.sh
R=/home/michael/strata-xpu
SRC=$R/strata
E=$SRC/p5/evidence
mkdir -p "$E"
cd "$SRC" || exit 1
PY=/usr/bin/python3

{
  echo "== P5 the three-way tables (28_compare: oracle vs every arm)"
  echo "== oracle = llama.cpp-SYCL qwen4exp, the same IQ3_S file, f16 KV, -ub 2048 unless stated"
  echo "== 128K: the -ub 2048 oracle dies at 38% of the prompt, so that row's reference is the -ub 512 oracle"
  for ctx in 4096 32768 131072; do
    case $ctx in 4096) c=4k; O=oracle-4k.json ;; 32768) c=32k; O=oracle-32k.json ;;
                 131072) c=128k; O=oracle-128k-ub512.json ;; esac
    echo
    echo "########## ctx $ctx ##########"
    $PY p5/p5_compare.py "$ctx" "$R/p5/oracle/$O" \
      shipped:$R/p5/runs/p5-$c-shipped/out.txt \
      batched:$R/p5/runs/p5-$c-batched/out.txt \
      shipped-fp16:$R/p5/runs/p5-$c-shipped-fp16/out.txt \
      batched-fp16:$R/p5/runs/p5-$c-batched-fp16/out.txt 2>&1
  done
} > "$E/three-way-tables.txt" 2>&1

{
  echo "== the oracle's OWN resolution: the same prompts at -ub 512 (I2's method, I2-STATUS.md 5b)"
  echo "== a divergence of ours that the -ub 512 oracle also produces is inside the oracle's own band, not evidence"
  for ctx in 4096 32768; do
    case $ctx in 4096) c=4k ;; 32768) c=32k ;; esac
    echo
    echo "########## ctx $ctx: oracle-ub512 vs oracle-ub2048 ##########"
    $PY p5/p5_pairdiff.py "$ctx" "json:oracle-$c-ub512.json" "json:oracle-$c.json" 2>&1
    echo "-- the streams around the first divergence"
    case $ctx in 4096) $PY p5/p5_streams.py "$ctx" 74:82 ;; 32768) $PY p5/p5_streams.py "$ctx" 70:80 99:112 ;; esac
    echo "-- what -ub 512 does to the oracle's own stream length: ub2048 vs ub512"
    $PY p5/p5_oracle_ub_compare.py "$ctx" 2>&1
  done
} > "$E/oracle-band.txt" 2>&1

{
  echo "== engine-vs-engine: what the two attention paths do to each other, with no oracle in the loop"
  for ctx in 4096 32768 131072; do
    case $ctx in 4096) c=4k ;; 32768) c=32k ;; 131072) c=128k ;; esac
    echo
    echo "########## ctx $ctx ##########"
    echo "-- int8 KV (the config of record)"
    $PY p5/p5_pairdiff.py "$ctx" shipped batched 2>&1
    if [ -f "$R/p5/runs/p5-$c-shipped-fp16/out.txt" ]; then
      echo "-- fp16 KV"
      $PY p5/p5_pairdiff.py "$ctx" shipped-fp16 batched-fp16 2>&1
    fi
  done
} > "$E/arm-vs-arm.txt" 2>&1

{
  echo "== the 128K oracle: the wall, measured twice"
  echo "== (the ub-2048 arm crashed twice at ~49,152 of 129,024 prompt tokens; see the server logs)"
  for f in $R/p5/oracle/oracle-128k-server.log $R/p5/oracle/oracle-128k-r2-server.log; do
    echo
    echo "########## $f"
    grep -nE "CMD:|level_zero backend failed|ggml_sycl_error|ggml_abort|Error OP |prompt processing, n_tokens =  *4[0-9]{4}|print_timing" "$f" 2>/dev/null | tail -14
  done
  echo
  echo "-- the ub-512 attempt (bounded to 1500 s of prompt processing)"
  tail -12 $R/p5/chain-oracle-128k-ub512.log 2>/dev/null
  grep -E "prompt processing, n_tokens" $R/p5/oracle/oracle-128k-ub512-server.log 2>/dev/null | tail -4
  grep -E "level_zero backend failed|Error OP" $R/p5/oracle/oracle-128k-ub512-server.log 2>/dev/null | tail -3
} > "$E/oracle-128k-wall.txt" 2>&1

cp -f "$SRC/p5/chain-engine.log" "$SRC/p5/chain-engine-fp16.log" \
      "$R/p5/chain-oracle.log" "$R/p5/chain-oracle-ub512.log" "$R/p5/chain-oracle-128k-ub512.log" "$E/" 2>/dev/null
ls -la "$E"
