#!/usr/bin/env bash
# D2x (card t_85e61269): every number of this card in one file -> d2x/D2X-EVIDENCE.txt, plus the copy the
# board carries into ~/.hermes/kanban/attachments/t_85e61269/.
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
O=$SRC/d2x/D2X-EVIDENCE.txt
{
echo "############ D2x evidence -- an int8-XMX MMVQ priced against the shipped scalar MMVQ, $(date -Is)"
echo "###### repo: $(cd $SRC && git log --oneline -1)"
echo "###### files this card added/changed:"
echo "######   bench/micro/mmvq_xmx_price.cpp (new, the bench), CMakeLists.txt (+1 target block,"
echo "######     if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_SYCL), outside the NOT-SYCL block the two mmvq"
echo "######     harnesses live in), d2x/ (scripts, raw run, tables, write-up).  NO engine source, NO"
echo "######     config, NO test changed: nothing in this card's tree is read by the engine."
echo "###### binary measured: build-sycl/mmvq_xmx_price md5 $(md5sum < $SRC/build-sycl/mmvq_xmx_price | cut -c1-32)"
echo "###### the engine binary was NOT rebuilt this card: md5 $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
echo
echo "###### 1. THE RUN ITSELF (one B70, ZE_AFFINITY_MASK=0, 20 reps, ncols 4 and 6) - d2x/D2X-BENCH.txt"
echo
cat "$SRC/d2x/D2X-BENCH.txt"
echo
echo "###### 2. THE TABLES (d2x/xmx_price.py over the run above) - d2x/D2X-TABLE.txt"
echo
cat "$SRC/d2x/D2X-TABLE.txt"
echo
echo "###### 3. THE BYTE/FLOOR READING OFF THE PACK'S OWN GGUF - d2x/bytes-floor.txt"
echo
cat "$SRC/d2x/bytes-floor.txt"
echo
echo "###### 4. THE CARD'S PREMISE RE-MEASURED (the bench's section 1, driver's own words)"
grep -E "VERDICT|DRIVER|driver text|16x16x16|8x16x16|int8|f16 " "$SRC/d2x/D2X-BENCH.txt" | grep -v "^Build\|^error\|^in function" | head -20
echo
echo "###### 5. MACHINE STATE (d2x/machine.txt)"
cat "$SRC/d2x/machine.txt"
echo
echo "###### 6. WHAT THIS CARD DID NOT DO"
echo "  - no engine integration of the prototype (a microbench price, not an end-to-end decode arm)"
echo "  - the 3.7 s load-time dequant+repack and the +4.45 GB of f16 weights are NOT in the speedup"
echo "  - correctness of the prototype checked on one case (Q6_K 2560x10240), not all 25 geometries"
echo "  - the 16x16x16 joint_matrix build failure is measured, not diagnosed"
echo "  - no ctest run (no engine source, no test, no config changed)"
echo "  - nothing pushed (origin has no sycl-xpu branch)"
echo
echo "###### 7. GIT"
cd "$SRC" && git log --oneline -4 && git status --short | grep -v "^?? i3/\|^?? scripts/" | head
} > "$O" 2>&1
echo "wrote $O ($(wc -l < "$O") lines)"

DEST=$HOME/.hermes/kanban/attachments/t_85e61269
mkdir -p "$DEST"
cp -v "$SRC/d2x/D2X-EVIDENCE.txt" "$SRC/d2x/D2X-BENCH.txt" "$SRC/d2x/D2X-TABLE.txt" \
      "$SRC/d2x/STATUS-D2X.md" "$SRC/d2x/bytes-floor.txt" "$SRC/d2x/machine.txt" \
      "$SRC/bench/micro/mmvq_xmx_price.cpp" "$SRC/d2x/xmx_price.py" "$SRC/d2x/bytes_floor.py" \
      "$SRC/d2x/run_bench.sh" "$SRC/d2x/build_bench.sh" "$SRC/d2x/machine_state.sh" "$DEST/" 2>&1 | tail -14
ls -la "$DEST"
