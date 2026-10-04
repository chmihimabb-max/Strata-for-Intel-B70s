#!/usr/bin/env bash
# D2b (card t_f93760a1): rebuild build-sycl/strata with the warm-up.  The pre-change binary is kept at
# /home/michael/strata-xpu/d2b/strata-e841fd06 so every arm can name the binary it measured.
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2b/build.log
source /opt/intel/oneapi/setvars.sh > "$LOG" 2>&1
{
  echo "###### engine build $(date -Is)"
  echo "engine md5 before: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
  cmake -S "$SRC" -B "$SRC/build-sycl" >> "$LOG" 2>&1
  echo "###### cmake rc=$?"
  make -C "$SRC/build-sycl" strata -j8
  echo "###### make rc=$? at $(date -Is)"
  echo "engine md5 after: $(md5sum < $SRC/build-sycl/strata | cut -c1-32)"
} >> "$LOG" 2>&1
tail -25 "$LOG"
