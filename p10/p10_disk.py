#!/usr/bin/env python3
"""P10: the SSD tier under the arm - /proc/diskstats sectors read for one block device, per interval.

usage: p10_disk.py <device> <out.csv> [interval_s] [--stopfile F] [--startfile F]

The engine's own "MB read (the GGUF in place)" counter says how many bytes came out of its file tier (the mmap'd
GGUF), but not whether the page was in RAM.  When two instances hold ~95 GiB of RSS between them on a 123 GiB box,
the pack's pages cannot all stay cached - this sampler is what says whether the volume reached the device, which
is the interference the two-instance arm exists to look for.

Columns: epoch, t_s, sectors_read (512 B each), MiB_s over the interval, and the same for ms spent reading.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time


def diskstat(dev):
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if len(p) > 13 and p[2] == dev:
                return int(p[5]), int(p[6])          # sectors read, ms reading
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("device")
    ap.add_argument("out")
    ap.add_argument("interval", nargs="?", type=float, default=2.0)
    ap.add_argument("--stopfile", default=None)
    o = ap.parse_args()

    first = diskstat(o.device)
    if first is None:
        print(f"[disk] no such device {o.device} in /proc/diskstats")
        return 1
    prev, last_t = first, time.time()
    t0 = last_t
    rows = 0
    with open(o.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "t_s", "sectors_read", "mib_s", "sectors_delta", "ms_reading_delta"])
        while True:
            if o.stopfile and os.path.exists(o.stopfile):
                break
            time.sleep(o.interval)
            now = time.time()
            cur = diskstat(o.device)
            if cur is None:
                break
            ds, dm = cur[0] - prev[0], cur[1] - prev[1]
            dt = now - last_t
            w.writerow(["%.3f" % now, "%.2f" % (now - t0), cur[0], "%.1f" % (ds * 512.0 / 1048576.0 / dt),
                        ds, dm])
            f.flush()
            rows += 1
            prev, last_t = cur, now
    total = (prev[0] - first[0]) * 512.0 / 1073741824.0
    print(f"[disk] {o.device}: {rows} samples over {last_t-t0:.0f}s, {total:.2f} GiB read from the device "
          f"({prev[1]-first[1]} ms spent reading)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
