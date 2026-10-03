#!/usr/bin/env bash
# P8: run the standalone H2D bench with the PCIe link-state watcher sampling the sysfs in parallel, so the link
# reading during a real transfer is in the same log as the bandwidth the transfer achieved.
#   bash scripts/p8_pcie_run.sh [MiB] [iters]
set -e
R=/home/michael/strata-xpu/strata
cd "$R"
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u
MIB=${1:-64}
ITERS=${2:-3}
LOG=/tmp/p8_linkwatch.log
rm -f "$LOG"
/usr/bin/python3 p8/p8_link_watch.py 60 0.01 "$LOG" &
WATCH=$!
sleep 1
echo "=== bench ($MIB MiB x $ITERS) ==="
./p8/p8_h2d_bench "$MIB" "$ITERS"
kill "$WATCH" 2>/dev/null || true
wait "$WATCH" 2>/dev/null || true
echo
echo "=== link states seen while the transfers ran (first 25, then the count) ==="
head -25 "$LOG"
echo "..."
echo "distinct $(sort -u "$LOG" | wc -l) states over $(wc -l < "$LOG") samples"
echo "=== any sample that is NOT 2.5 GT/s/x1 ==="
grep -v "2.5 GT/s PCIe/1" "$LOG" | head -10 || echo "   none - the driver reported 2.5 GT/s x1 for the whole transfer"
