#!/usr/bin/env python3
"""P10: the raw per-thread CPU evidence for one arm, out of the two samplers the card names.

usage: p10_top_evidence.py <arm_dir> [idx]

Reads <arm_dir>/top-H.txt (or top-H-<idx>.txt) and pidstat.txt and reports, for the busiest sample of each:
  * how many threads of the engine were at or above 50% of one core, and the top rows verbatim
  * pidstat's own AGGREGATE row (%CPU of the process, which the kernel computes from /proc/<pid>/stat) next to
    the maximum it reports for ANY of the process's threads in the same sample

The second line is the point: on this engine `pidstat -t` prints 0.00% for every thread while its own aggregate
row for the same second reads ~1694% and `top -b -H` shows 19 threads at 82-88%.  One of those two is a tool
artefact and it is not top: the process's /proc/<pid>/stat delta and the sum of its /proc/<pid>/task/<tid>/stat
deltas agree (p10_threadcheck.py), so the per-thread counters are real.
"""
from __future__ import annotations

import os
import re
import sys


def top_seconds(t):
    """top prints TIME+ as mm:ss.ss (or hh:mm:ss) - not a float."""
    parts = t.split(":")
    try:
        parts = [float(x) for x in parts]
    except ValueError:
        return float("nan")
    s = 0.0
    for x in parts:
        s = s * 60.0 + x
    return s


def top_evidence(path):
    if not os.path.exists(path):
        print(f"[top] {path}: missing")
        return
    text = open(path, errors="replace").read()
    # one sample = one "top - <time> up ..." header followed by its thread rows
    blocks = re.split(r"\ntop - ", "\n" + text)[1:]
    best = None
    for b in blocks:
        lines = b.splitlines()
        t = lines[0].split()[0] if lines else "?"
        rows = []
        for line in lines[1:]:
            p = line.split()
            if len(p) >= 12 and p[0].isdigit() and p[-1] == "strata":
                try:
                    rows.append((int(p[0]), float(p[8]), top_seconds(p[10])))
                except ValueError:
                    pass
        if not rows:
            continue
        hot = [r for r in rows if r[1] >= 50.0]
        if best is None or len(hot) > best[1]:
            best = (t, len(hot), rows)
    if not best:
        print(f"[top] {path}: no samples parsed")
        return
    t, nhot, rows = best
    rows.sort(key=lambda r: -r[1])
    print(f"[top] {os.path.basename(path)}: busiest sample {t}: {len(rows)} engine threads, "
          f"{nhot} at >=50% of one core, {len([r for r in rows if r[1] > 1.0])} above 1%")
    for r in rows[:22]:
        print(f"[top]    tid {r[0]:>8}  {r[1]:5.1f}% cpu  {r[2]:8.2f}s cpu-time")


def pidstat_evidence(path):
    if not os.path.exists(path):
        print(f"[pidstat] {path}: missing")
        return
    agg, thr = {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 12 or not p[0].count(":"):
                continue
            ts = p[0]
            try:
                cpu = float(p[9])
            except ValueError:
                continue
            if p[4] == "-":
                agg[ts] = max(agg.get(ts, 0.0), cpu)
            elif p[3] == "-" and p[4].isdigit():
                thr[ts] = max(thr.get(ts, 0.0), cpu)
    if not agg:
        print(f"[pidstat] {os.path.basename(path)}: no parseable samples")
        return
    ts = max(agg, key=lambda k: agg[k])
    print(f"[pidstat] {os.path.basename(path)}: busiest sample {ts}: process aggregate %CPU {agg[ts]:.0f} "
          f"({agg[ts]/100.0:.2f} cores) against the MAXIMUM of any single thread {thr.get(ts, 0.0):.2f}% "
          f"in the same sample")
    nz = sum(1 for k in thr if thr[k] > 1.0)
    print(f"[pidstat] samples where any thread read above 1%: {nz} of {len(thr)}")


def main() -> int:
    d = sys.argv[1].rstrip("/")
    idx = sys.argv[2] if len(sys.argv) > 2 else None
    suffix = f"-{idx}" if idx else ""
    top_evidence(os.path.join(d, f"top-H{suffix}.txt"))
    pidstat_evidence(os.path.join(d, f"pidstat{suffix}.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
