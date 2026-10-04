#!/bin/bash
# D2a: where else does a compiled kernel live? (the SYCL program cache is not the only one)
echo "== likely driver/IGC caches"
ls -d "$HOME"/.cache/* 2>/dev/null
echo "== files modified in those caches in the last 30 minutes"
find "$HOME"/.cache -maxdepth 3 -type f -mmin -30 -printf '%TH:%TM %p\n' 2>/dev/null | head -20
echo "== /tmp caches"
ls -d /tmp/*cache* /tmp/*igc* 2>/dev/null
echo "== the harness's own cache"
find /tmp/d2a-cache -name '0.src' -printf '%TH:%TM:%TS %s %h\n' 2>/dev/null | sort | head -30
