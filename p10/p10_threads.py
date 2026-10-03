#!/usr/bin/env python3
"""P10: per-THREAD CPU sampler for the engine process (the pool-workers-asleep question).

Every `interval` seconds, read /proc/<pid>/task/*/stat, take the utime+stime delta per thread and write one CSV
row per thread.  Per-thread deltas over a known interval are what the card asks for ("per-thread CPU sample of
the pool workers ... not 'since start'"): the pool's threads park between requests, so a since-start figure
reads 0.0% even while a request is draining experts on 19 cores.

usage: p10_threads.py <pid> <out.csv> [interval_s] [--stopfile F]

The sampler stops when the process dies or when the stop file appears.  A summary over the whole run is printed
at the end; p10_cpu_report.py splits it into the load / prefill / decode phases using the marker file the arm
runner writes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")


def read_thread(pid: int, tid: str):
    try:
        with open(f"/proc/{pid}/task/{tid}/stat") as f:
            s = f.read()
    except OSError:
        return None
    # field 2 is `comm` in parentheses and may contain spaces/parens: split on the LAST ')'
    i = s.rfind(")")
    if i < 0:
        return None
    comm = s[s.find("(") + 1:i]
    rest = s[i + 2:].split()
    if len(rest) < 13:
        return None
    try:
        utime, stime = int(rest[11]), int(rest[12])
    except ValueError:
        return None
    return comm, utime + stime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", type=int)
    ap.add_argument("out")
    ap.add_argument("interval", nargs="?", type=float, default=1.0)
    ap.add_argument("--stopfile", default=None)
    o = ap.parse_args()

    prev: dict[str, int] = {}
    first: dict[str, int] = {}
    t0 = time.time()
    last_t = t0
    peak: dict[str, float] = {}
    rows = 0
    with open(o.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "t_s", "tid", "comm", "ticks_delta", "cpu_pct", "ticks_total"])
        while True:
            if o.stopfile and os.path.exists(o.stopfile):
                break
            now = time.time()
            dt = now - last_t
            last_t = now
            live = 0
            for path in glob.glob(f"/proc/{o.pid}/task/*"):
                tid = os.path.basename(path)
                r = read_thread(o.pid, tid)
                if r is None:
                    continue
                live += 1
                comm, ticks = r
                first.setdefault(tid, ticks)
                d = ticks - prev.get(tid, ticks)
                prev[tid] = ticks
                pct = 100.0 * d / (dt * CLK) if dt > 0 else 0.0
                peak[tid] = max(peak.get(tid, 0.0), pct)
                w.writerow(["%.3f" % now, "%.2f" % (now - t0), tid, comm, d, "%.1f" % pct, ticks])
                rows += 1
            f.flush()
            if live == 0:
                break
            time.sleep(o.interval)
    thr = len(first)
    busy_ever = sum(1 for v in peak.values() if v >= 50.0)
    print(f"[threads] pid {o.pid}: {thr} threads seen, {rows} rows, {time.time()-t0:.0f}s")
    print(f"[threads] threads that ever reached >=50% of one core: {busy_ever}")
    top = sorted(peak.items(), key=lambda kv: -kv[1])[:25]
    print("[threads] top threads by peak cpu%%: " + ", ".join("%s=%.0f%%" % (t, p) for t, p in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
