#!/usr/bin/env bash
# D2b (card t_f93760a1) chain A: the cold first run of the SHIPPED layout, and its warm control.
#
# The shipped binary (D2's e841fd06..., saved here so a later rebuild cannot change what these arms measure).
# "Cold" = SYCL_CACHE_DIR and NEO_CACHE_DIR both empty, which is a fresh install: everything the first request
# needs - including the decode path's (type x ncols) specializations - has to be built.  The shared warm cache
# the project has used all session ($R/sycl-cache/m6c and ~/.cache/neo_compiler_cache) is NOT touched: the cold
# arms point both variables at fresh /tmp directories, which is the same thing as moving the entries aside with
# nothing to restore afterwards.
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2b/chain_a.log
BIN=$SRC/build-sycl/strata          # whatever is in place; md5 is checked below
OLD=/home/michael/strata-xpu/d2b/strata-e841fd06
mkdir -p /home/michael/strata-xpu/d2b
{
  echo "###### D2b chain A started $(date -Is)"
  echo "## the shipped binary: $(md5sum < $BIN | cut -c1-32)"
  if [ ! -x "$OLD" ]; then cp -f "$BIN" "$OLD"; fi
  echo "## saved copy: $(md5sum < $OLD | cut -c1-32)  (must be e841fd061fec873c2f24e785973a2ebe)"
  test "$(md5sum < $OLD | cut -c1-32)" = "e841fd061fec873c2f24e785973a2ebe" || { echo "## WRONG BINARY - stop"; exit 1; }

  # A1: the cold first run of the shipped layout (both caches empty) at 4K
  rm -rf /tmp/d2b-cold1; mkdir -p /tmp/d2b-cold1/sycl /tmp/d2b-cold1/neo
  echo "###### A1 d2b-cold-4096 $(date -Is)"
  bash $SRC/d2b/d2b_run_arm.sh d2b-cold-4096 4096 256 --bin "$OLD" --spec-min-p 0.7 \
       --sycl-cache /tmp/d2b-cold1/sycl --neo-cache /tmp/d2b-cold1/neo

  # A2: the warm control, same binary, same config, same session
  echo "###### A2 d2b-rebase-4096 $(date -Is)"
  bash $SRC/d2b/d2b_run_arm.sh d2b-rebase-4096 4096 256 --bin "$OLD" --spec-min-p 0.7

  echo "###### D2b chain A ALL DONE $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
