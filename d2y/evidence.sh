#!/usr/bin/env bash
# D2y: one file with every number this card quotes, and where each one comes from.
set +e
SRC=/home/michael/strata-xpu/strata
OUT=$SRC/d2y/D2Y-EVIDENCE.txt
{
  echo "D2y (card t_ba006576) -- every number in one file, with its source"
  echo "generated $(date -Is); the raw runs are d2y/D2Y-BENCH.txt, D2Y-BENCH-2.txt, D2Y-BASE.txt and the"
  echo "reader outputs D2Y-TABLE.txt, D2Y-TABLE-2.txt, D2Y-MARGINAL.txt; nothing here is retyped by hand."
  echo
  echo "======================================================================================================"
  echo "1. the machine, and the binary each run used (each raw file carries its own md5 in its header)"
  echo "======================================================================================================"
  cat "$SRC/d2y/machine.txt"
  echo
  echo "======================================================================================================"
  echo "2. the bench's own header, section 1 verdict and section 2 bandwidth (run 1)"
  echo "======================================================================================================"
  sed -n '1,12p' "$SRC/d2y/D2Y-BENCH.txt"
  sed -n '/VERDICT on this driver/,/^$/p' "$SRC/d2y/D2Y-BENCH.txt"
  sed -n '/== 2. the card/,/^== 3\./p' "$SRC/d2y/D2Y-BENCH.txt"
  echo
  echo "======================================================================================================"
  echo "3. the new section 6, run 1 and run 2 (the same binary, 4 minutes apart, same command)"
  echo "======================================================================================================"
  sed -n '/== 6. D2y/,/byte-table check/p' "$SRC/d2y/D2Y-BENCH.txt"
  echo "----------------------------------- run 2 -----------------------------------"
  sed -n '/== 6. D2y/,/byte-table check/p' "$SRC/d2y/D2Y-BENCH-2.txt"
  echo
  echo "======================================================================================================"
  echo "4. the extended reader on run 1: the per-shape/per-variant table, the window roll-up and the"
  echo "   inert control against the pre-edit binary (d2y/D2Y-BASE.txt)"
  echo "======================================================================================================"
  sed -n '/D2y section 6/,/the per-type roll-up/p' "$SRC/d2y/D2Y-TABLE.txt"
  sed -n '/the inert control/,/largest single-case/p' "$SRC/d2y/D2Y-TABLE.txt"
  echo
  echo "======================================================================================================"
  echo "5. the reader on run 2, and its inert control against run 1 (same binary: this is the rig's spread)"
  echo "======================================================================================================"
  sed -n '/D2y section 6/,/the per-type roll-up/p' "$SRC/d2y/D2Y-TABLE-2.txt"
  sed -n '/the inert control/,/largest single-case/p' "$SRC/d2y/D2Y-TABLE-2.txt"
  echo
  echo "======================================================================================================"
  echo "6. the per-column probe: the shipped kernel at ncols 2, 4 and 6 (d2y/D2Y-MARGINAL.txt)"
  echo "======================================================================================================"
  sed -n '/the shipped kernel against its own column count/,/^$/p' "$SRC/d2y/D2Y-MARGINAL.txt"
  echo
  echo "======================================================================================================"
  echo "7. the launch geometry (host arithmetic, d2y/attribute.py -> d2y/D2Y-ATTRIB.txt)"
  echo "======================================================================================================"
  sed -n '/the traits the shapes are made of/,$p' "$SRC/d2y/D2Y-ATTRIB.txt" | head -70
  echo
  echo "======================================================================================================"
  echo "8. the engine's own tests, with this card's knob and with the edit stashed at HEAD"
  echo "======================================================================================================"
  grep -E "Test #|tests passed|FAILED|quantize_act:|iq_multi_parity:" "$SRC/d2y/D2Y-TESTS.txt" | tail -12
  echo "--- with the D2y engine edit stashed (native_mmvq.cu/.hpp at HEAD) ---"
  grep -E "Test #|tests passed|FAILED" "$SRC/d2y/D2Y-TESTS-HEAD.txt" | tail -8
  echo
  echo "======================================================================================================"
  echo "9. the engine-side diff this card adds (price-only knob, default 0 = the shipped rule)"
  echo "======================================================================================================"
  cd "$SRC"
  git diff -- src/kernels/cuda/native_mmvq.cu include/strata/kernels/native_mmvq.hpp
} > "$OUT" 2>&1
echo "wrote $OUT"
grep -c . "$OUT"
