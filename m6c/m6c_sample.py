#!/usr/bin/env python3
"""M6c: sample the engine's process TREE every second with epoch stamps.

One line per sample: epoch, t_s, tree RSS, tree read_bytes (device reads), tree majflt/minflt, the
largest single-process RSS (the engine), /proc/meminfo Cached and Mlocked, and the number of processes.

The epoch column is what lets the report line a sample up with a marker in the driver's timeline
(`READY`, `RESUME 0`, the first `T `, the last `T `) - `t_s` alone cannot be aligned with the engine's own
phase lines.  read_bytes/majflt are the SSD tier; RSS and Mlocked are the RAM tier.

usage: m6c_sample.py <root-pid> <csv> <seconds>
"""
from __future__ import annotations

import os
import sys
import time

COLS = ["rchar", "read_bytes", "syscr", "majflt", "minflt"]


def tree(root_pid: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % d) as f:
                st = f.read()
            _, _, rest = st.rpartition(")")           # comm may contain ')' - rpartition, not partition
            ppid = int(rest.split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, stack = [], [root_pid]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def meminfo() -> dict:
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0]) if v.split() else 0
    return d


def sample(root_pid: int, csv: str, secs: float) -> None:
    t0 = time.time()
    peak_rss = 0
    n = 0
    with open(csv, "w") as f:
        f.write("epoch,t_s,nprocs,rss_bytes,max_proc_rss,read_bytes,majflt,minflt,rchar,cached_kb,mlocked_kb\n")
        while time.time() - t0 < secs:
            d = {c: 0 for c in COLS}
            rss = 0
            maxp = 0
            procs = tree(root_pid)
            for pid in procs:
                try:
                    with open("/proc/%d/io" % pid) as g:
                        for line in g:
                            k, _, v = line.partition(":")
                            if k.strip() in COLS:
                                d[k.strip()] += int(v.strip())
                    with open("/proc/%d/stat" % pid) as g:
                        st = g.read().rsplit(")", 1)[1].split()
                    d["majflt"] += int(st[9])
                    d["minflt"] += int(st[7])
                    with open("/proc/%d/status" % pid) as g:
                        for line in g:
                            if line.startswith("VmRSS:"):
                                b = int(line.split()[1]) * 1024
                                rss += b
                                maxp = max(maxp, b)
                                break
                except (OSError, IndexError):
                    continue
            if not procs:
                break
            mi = meminfo()
            peak_rss = max(peak_rss, rss)
            n += 1
            f.write("%.3f,%.1f,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % (
                time.time(), time.time() - t0, len(procs), rss, maxp, d["read_bytes"], d["majflt"],
                d["minflt"], d["rchar"], mi.get("Cached", 0), mi.get("Mlocked", 0)))
            f.flush()
            time.sleep(1.0)
    print("sampled %d points over the tree of pid %d; peak tree RSS %.2f GiB" % (n, root_pid, peak_rss / 2 ** 30))


if __name__ == "__main__":
    sample(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]))
