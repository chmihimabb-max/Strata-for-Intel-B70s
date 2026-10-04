#!/usr/bin/env bash
# D2c (card t_c7d8cd86): rebuild build-sycl/strata with the decode-path warm-up.  The pre-change binary of this
# card is kept at /home/michael/strata-xpu/d2c/strata-f49fe666 (D2b's build) so every arm names its binary.
set +e
SRC=/home/michael/strata-xpu/strata
LOG=$SRC/d2c/build.log
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
grep -E "error|Error|rc=|md5 after" "$LOG" | tail -25
