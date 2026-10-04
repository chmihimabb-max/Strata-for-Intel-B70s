#!/usr/bin/env bash
# D2 (card t_0416a0c0) task 2, the third lever: the multi-column MMVQ's LAYOUT.  The 300 dense native projections
# are 40.1% of a decode window's device time and they are the only family whose two shipped layouts had never been
# compared: `native_mmvq_set_multi_exact(true)` (the default, the ncols == 1 layout, bitwise equal to a
# single-column call) against `false` (llama.cpp's generic multi-column table, whose own header says "speed not yet
# measured").  The switch added to generate.cpp reads STRATA_MMVQ_MULTI_GENERIC once, before any capture.
#
# The build runs HERE, between chains, so that no compile overlaps a measured arm.
#   bash d2/d2_chain3.sh
R=/home/michael/strata-xpu
SRC=$R/strata
RUN=$SRC/d2/runs
LOG=$SRC/d2/chain3.log
mkdir -p "$RUN"

run_arm() {
  local tag=$1; shift
  if [ -f "$RUN/$tag/log.txt" ] && grep -q "== engine exit" "$RUN/$tag/log.txt" 2>/dev/null; then
    echo "== $tag: already complete, skipped $(date -Is)" >> "$LOG"; return 0
  fi
  echo "== $tag: starting $(date -Is)" >> "$LOG"
  for i in $(seq 1 300); do pgrep -x strata >/dev/null || break; sleep 2; done
  bash "$SRC/d2/d2_run_arm.sh" "$tag" "$@" >> "$LOG" 2>&1
  echo "== $tag: finished $(date -Is)" >> "$LOG"
  sleep 5
}

echo "###### d2 task2 third lever queued $(date -Is)" >> "$LOG"
for i in $(seq 1 3000); do
  grep -q "d2 task2 second lever done" "$SRC/d2/chain2.log" 2>/dev/null && break
  sleep 10
done
echo "###### d2 build (no engine running) $(date -Is)" >> "$LOG"
make -C "$SRC/build-sycl" strata -j8 >> "$LOG" 2>&1
echo "###### build rc=$? at $(date -Is); md5 $(md5sum < "$SRC/build-sycl/strata" | cut -c1-32)" >> "$LOG"
run_arm "d2-rebase-4096" 4096 256
run_arm "d2-mg-4096" 4096 256 --env STRATA_MMVQ_MULTI_GENERIC=1
run_arm "d2-mg-32768" 32768 256 --env STRATA_MMVQ_MULTI_GENERIC=1
echo "###### d2 task2 third lever done $(date -Is)" >> "$LOG"
tail -2 "$LOG"
