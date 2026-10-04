#!/usr/bin/env bash
# D2a (card t_8306429a): every raw line the card's acceptance asks for, in one file.
R=/home/michael/strata-xpu
SRC=$R/strata
OUT=$SRC/d2a/D2A-EVIDENCE.txt

{
echo "############ D2a evidence -- the generic multi-column MMVQ layout, $(date -Is)"
echo "###### repo: $(cd $SRC && git log --oneline -1)"
echo "###### engine binary: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)  (D2's e841fd061fec873c2f24e785973a2ebe = unchanged)"
echo

echo "###### 1. THE MECHANISM: SYCL programs built while each arm decoded"
echo "### per arm: new programs (0.src entries) written between its ask and its finish, with the arm's own"
echo "### window time it cannot name (verify - wait - host - stage - tail)"
/usr/bin/python3 "$SRC/d2a/analyze.py"
echo
echo "### the programs by name and timestamp (mangled type name from each entry's 0.src)"
bash "$SRC/d2a/program_names.sh"
echo

echo "###### 2. THE KERNEL LEVEL: both layouts, the config's own shapes, 16 reps each, one input set"
echo "### 2a. BOTH caches fresh (SYCL_CACHE_DIR and NEO_CACHE_DIR): every specialization's first launch carries"
echo "### its own program build - this is the engine's condition on its first run of the layout"
cat "$SRC/d2a/spread-cold-fully.txt"
echo "### 2b. a fresh SYCL cache with the Intel NEO/IGC compiler cache (~/.cache/neo_compiler_cache) left in"
echo "### place: the native binaries are still there, so the first launches are cheap - a fresh SYCL_CACHE_DIR"
echo "### alone is NOT a cold compiler"
cat "$SRC/d2a/spread-cold.txt"
echo "### 2c. the same cache warm (steady state)"
cat "$SRC/d2a/spread-warm.txt"
echo "### the runner: $(cd $SRC && git log --oneline -1 -- src/kernels/mmvq_multi_spread.cpp CMakeLists.txt)"
echo

echo "###### 3. WHERE THE STREAMS PART"
echo "### the shipped layout's stream as the reference, token by token"
/usr/bin/python3 "$SRC/d2a/divergence.py" d2-rebase-4096 d2-mg-4096 d2-mg-4096-warm d2-mg-4096-warm2
echo
echo "### the window tables (index, first generated token it covers, T) side by side"
/usr/bin/python3 "$SRC/d2a/windowtable.py"
echo
echo "### 3b. eight warm runs of ONE lever: which window sequence gives which stream"
/usr/bin/python3 "$SRC/d2a/stream_vs_window.py"
echo

echo "###### 4. THE LAYOUT CONTRACT (unchanged): mmvq_multi_parity"
cd "$SRC/build-sycl" && SYCL_CACHE_DIR=$R/sycl-cache/m6c ./mmvq_multi_parity 2>&1 | tail -25
echo

echo "###### 5. THE RIG'S OWN CHECK: nothing but the harness/CMake was added (the engine binary is D2's)"
cd "$SRC" && git status --short | head -20
} > "$OUT" 2>&1
echo "written $OUT ($(wc -l < "$OUT") lines)"
