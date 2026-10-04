#!/usr/bin/env bash
# D2c (card t_c7d8cd86): every number of this card, in one file -> d2c/D2C-EVIDENCE.txt
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
O=$SRC/d2c/D2C-EVIDENCE.txt
{
echo "############ D2c evidence -- the other 61 decode-path programs (6 routed-expert gu/down + 55 one-off"
echo "############ kernel sites) are built in the load phase too, $(date -Is)"
echo "###### repo: $(cd $SRC && git log --oneline -1)"
echo "###### engine binaries:"
echo "######   before this card: f49fe6669704561bb8155f9c4c9acaed (D2b; kept at $R/d2c/strata-f49fe666)"
echo "######   after the second warm-up: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
echo "###### changed files: src/kernels/cuda/decode_warmup.cu (new), include/strata/kernels/decode_warmup.hpp (new),"
echo "######   src/program/generate.cpp (one include + one call), cmake/sycl_backend.cmake (one TU),"
echo "######   d2b/d2b_run_arm.sh (STRATA_KERNEL_WARMUP joins the unset list). No kernel source, no config change."
echo
echo "###### 1. THE COLD FIRST RUNS (both SYCL_CACHE_DIR and NEO_CACHE_DIR fresh = a fresh install)"
/usr/bin/python3 "$SRC/d2c/analyze_c.py" --all-cold
echo
echo "###### 2. THE WARM CONTROLS (same session, shared warm caches)"
/usr/bin/python3 "$SRC/d2c/analyze_c.py" --all-warm
echo
echo "###### 3. WHAT EACH COLD ARM BUILT INSIDE ITS OWN DECODE PHASE, BY LAUNCH SITE"
echo "######   before = D2b's pass only (STRATA_KERNEL_WARMUP=0); after = both passes."
echo "######   (the enumeration is from each entry's SPIR-V name section: d2c/enumerate_c.py)"
echo "== BEFORE  d2c-mmoff-cold-4096"
/usr/bin/python3 "$SRC/d2c/enumerate_c.py" d2c-mmoff-cold-4096 --phase decode
echo "== AFTER   d2c-wu-cold-4096"
/usr/bin/python3 "$SRC/d2c/enumerate_c.py" d2c-wu-cold-4096 --phase decode
echo "== BEFORE  d2c-nooff-cold-4096 (no warm-up at all: the shipped baseline)"
/usr/bin/python3 "$SRC/d2c/enumerate_c.py" d2c-nooff-cold-4096 --phase decode
echo
echo "###### 4. THE TWO WARM-UPS' OWN LINES, PER ARM"
for t in d2c-nooff-cold-4096 d2c-mmoff-cold-4096 d2c-wu-cold-4096 d2c-wu-4096 d2c-mmoff-4096 d2c-wu-32768; do
  echo "== $t"
  grep -E "strata mmvq warmup|strata kernel warmup" "$SRC/d2/runs/$t/err.txt" 2>/dev/null | tail -3
done
echo
echo "###### 5. THE LOAD WALL AND THE ASK, PER ARM"
for t in d2c-nooff-cold-4096 d2c-mmoff-cold-4096 d2c-wu-cold-4096 d2c-wu-4096 d2c-mmoff-4096 d2c-wu-32768; do
  printf '%-24s ' "$t"
  grep -E "loaded=1 after|the ask finished=1 after" "$SRC/d2/runs/$t/log.txt" 2>/dev/null | tr '\n' ' '
  echo
done
echo
echo "###### 6. THE IDS (d2b/ids_b.py's rule: md5 over the T lines joined by newline)"
/usr/bin/python3 "$SRC/d2b/ids_b.py" d2c-nooff-cold-4096 d2c-mmoff-cold-4096 d2c-wu-cold-4096 \
    d2c-wu-4096 d2c-mmoff-4096 d2c-wu-32768
echo "######   the 4K canonical is 66bf952d445e330c974c49e4f220b4b1 and the 32K one d87373e84417dab60732a95eb47666a6"
echo
echo "###### 7. THE ENGINES' OWN DECODE AND PROMPT LINES, PER ARM"
for t in d2c-nooff-cold-4096 d2c-mmoff-cold-4096 d2c-wu-cold-4096 d2c-wu-4096 d2c-mmoff-4096 d2c-wu-32768; do
  echo "== $t"
  grep -E "strata decode timing|strata serve: prompt " "$SRC/d2/runs/$t/err.txt" 2>/dev/null | tail -2
done
echo
echo "###### 8. THE CACHE CENSUS AND THE PROGRAMS WRITTEN IN THE ASK, PER ARM"
for t in d2c-nooff-cold-4096 d2c-mmoff-cold-4096 d2c-wu-cold-4096 d2c-wu-4096 d2c-mmoff-4096 d2c-wu-32768; do
  grep -H "cache census" "$SRC/d2/runs/$t/log.txt" 2>/dev/null | sed "s|$SRC/d2/runs/||"
done
echo
echo "###### 9. THE TESTS (ctest, ZE_AFFINITY_MASK=0)"
cat "$SRC/d2c/runs/ctest/mmvq_multi_parity.log" 2>/dev/null | grep -E "Test #|tests passed|tests failed" | tail -4
echo "-- the families whose kernels this pass warms:"
grep -E "tests passed|tests failed|\(Failed\)" "$SRC/d2c/runs/ctest/warmed-families.log" 2>/dev/null | tail -8
echo "-- the full suite (the S4 set) vs the recorded baseline (5 failed of 49, the same five names):"
grep -E "tests passed|tests failed" "$SRC/d2c/runs/ctest/full.log" 2>/dev/null | tail -2
grep -E "\(Failed\)" "$SRC/d2c/runs/ctest/full.log" 2>/dev/null | tail -8
echo "-- the baseline (s4/runs/ctest/full.log):"
grep -E "tests passed|tests failed" "$SRC/s4/runs/ctest/full.log" 2>/dev/null | tail -2
grep -E "\(Failed\)" "$SRC/s4/runs/ctest/full.log" 2>/dev/null | tail -8
echo
echo "###### 10. MACHINE STATE AT THE END OF THIS CARD"
pgrep -a -x strata || echo "   no engine of ours"
pgrep -a -f "serve/server.py" || echo "   no server of ours"
echo "   ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
cat /proc/loadavg
ss -ltn 2>/dev/null | grep -q ":8099" && echo "   port 8099 IN USE" || echo "   port 8099 free"
} > "$O" 2>&1
echo "wrote $O ($(wc -l < $O) lines)"
