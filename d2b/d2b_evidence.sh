#!/usr/bin/env bash
# D2b (card t_f93760a1): the full evidence dump -- every raw line this card's claims rest on.
set +e
SRC=/home/michael/strata-xpu/strata
OUT=$SRC/d2b/D2B-EVIDENCE.txt
R=/home/michael/strata-xpu
{
  echo "############ D2b evidence -- the decode path builds its (type x ncols) specializations inside the first"
  echo "############ decode windows; a load-phase warm-up moves them, 2026-10-03  $(date -Is)"
  echo "###### repo: $(cd $SRC && git log --oneline -1)"
  echo "###### engine binaries:"
  echo "######   before this card: e841fd061fec873c2f24e785973a2ebe  (kept at $R/d2b/strata-e841fd06)"
  echo "######   after the warm-up: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  echo
  echo "###### 1. THE USER-VISIBLE CASE: a cold first run of the SHIPPED layout"
  echo "######   (SYCL_CACHE_DIR and NEO_CACHE_DIR both empty = a fresh install; the shared warm caches were"
  echo "######    never moved: the cold arms point both variables at fresh /tmp dirs instead)"
  /usr/bin/python3 $SRC/d2b/analyze_b.py --all-cold 2>&1
  echo
  echo "###### 2. THE WARM CONTROLS (same session, shared warm caches)"
  /usr/bin/python3 $SRC/d2b/analyze_b.py d2b-rebase-4096 d2b-wu-4096 d2b-ctloff-4096 d2b-wu-32768 2>&1
  echo
  echo "###### 3. WHAT EACH COLD ARM BUILT INSIDE ITS OWN DECODE PHASE (post-prompt), by family"
  echo "######   the warm-up removes the dense mmvq family and nothing else: 93 -> 61 programs, 32 -> 0 mmvq"
  /usr/bin/python3 $SRC/d2b/classify.py d2b-cold-4096 d2b-ctloff-cold-4096 d2b-wu-cold-4096 2>&1
  echo
  echo "###### 3b. the dense mmvq specializations a cold shipped-layout decode still had to build (pre-fix),"
  echo "######     named from the mangled symbol in each entry's 0.src"
  /usr/bin/python3 $SRC/d2b/programs_b.py d2b-ctloff-cold-4096 --mmvq-only --limit 60 2>&1 | tail -45
  echo
  echo "###### 3c. and the 61 that remain WITH the warm-up: one program per decode-path kernel site, no mmvq"
  /usr/bin/python3 $SRC/d2b/name_other.py d2b-wu-cold-4096 2>&1 | grep -v "e.g." | head -22
  echo
  echo "###### 4. THE COST OF THE WARM-UP ITSELF (its own stderr line, and the load wall it lands in)"
  grep -h "strata mmvq warmup" $SRC/d2/runs/d2b-*/err.txt 2>/dev/null | sort | uniq -c
  echo "######   per arm: load wall (arm start -> 'everything loaded') and the ask"
  for a in d2b-cold-4096 d2b-ctloff-cold-4096 d2b-wu-cold-4096 d2b-rebase-4096 d2b-ctloff-4096 d2b-wu-4096 d2b-wu-32768; do
    printf '%-24s %s | %s\n' "$a" "$(grep -o 'loaded=[01] after [0-9]*s' $SRC/d2/runs/$a/log.txt)" \
      "$(grep -oE 'the ask finished=1 after [0-9]+ ms' $SRC/d2/runs/$a/log.txt)"
  done
  echo
  echo "###### 5. THE TESTS (ctest, ZE_AFFINITY_MASK=0)"
  echo "######   mmvq_multi_parity (the layout contract):"
  grep -E "Test #|tests passed|tests failed" $SRC/d2b/runs/ctest/mmvq_multi_parity.log 2>/dev/null | tail -3
  echo "######   the mmvq family:"
  grep -E "\(Failed\)|tests passed" $SRC/d2b/runs/ctest/mmvq-family.log 2>/dev/null | tail -4
  echo "######   the full suite (the S4 set) on this build vs the recorded baseline:"
  grep -E "tests passed" $SRC/d2b/runs/ctest/full.log | tail -1
  grep -E "\(Failed\)" $SRC/d2b/runs/ctest/full.log | tail -6
  echo "######   the baseline (s4/runs/ctest/full.log):"
  grep -E "tests passed" $R/strata/s4/runs/ctest/full.log | tail -1
  grep -E "\(Failed\)" $R/strata/s4/runs/ctest/full.log | tail -6
  echo
  echo "###### 6. THE IDS, as d2a/analyze.py and d2/d2_report.py compute them (the project's guard)"
  /usr/bin/python3 $SRC/d2b/ids_b.py d2b-cold-4096 d2b-ctloff-cold-4096 d2b-wu-cold-4096 d2b-rebase-4096 d2b-ctloff-4096 d2b-wu-4096 d2b-wu-32768 2>&1
  echo "######   (the 4K canonical is 66bf952d445e330c974c49e4f220b4b1 and the 32K one"
  echo "######    d87373e84417dab60732a95eb47666a6, per d2a/STATUS-D2A.md)"
  echo
  echo "###### 7. THE ENGINES' OWN DECODE AND PROMPT LINES, per arm"
  for a in d2b-cold-4096 d2b-ctloff-cold-4096 d2b-wu-cold-4096 d2b-rebase-4096 d2b-ctloff-4096 d2b-wu-4096 d2b-wu-32768; do
    echo "== $a"
    grep -hE "strata decode timing|strata serve: prompt [0-9]+ tokens" $SRC/d2/runs/$a/err.txt
  done
  echo
  echo "###### 8. HOW MANY PROGRAMS EACH ARM WROTE IN THE ASK (0.src entries) AND THE CENSUS LINE"
  grep -h "cache census" $SRC/d2/runs/d2b-*/log.txt 2>/dev/null | sort | uniq -c
  echo
  echo "###### 9. MACHINE STATE AT THE END OF THIS CARD"
  pgrep -a -x strata || echo "   no engine of ours"
  pgrep -a -f "serve/server.py" || echo "   no server of ours"
  echo "   ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
  cat /proc/loadavg
  ss -ltn 2>/dev/null | grep 8099 || echo "   port 8099 free"
} > "$OUT" 2>&1
wc -l "$OUT"; echo "written: $OUT"
