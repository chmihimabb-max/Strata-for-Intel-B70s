#!/usr/bin/env python3
"""P5: the biggest gaps between consecutive engine lines in an arm's timestamped timeline.

Used to separate a one-time cost (a JIT compile of a kernel instantiation the program cache has not seen, which
shows up as a multi-second silence before READY) from steady-state prefill.

    /usr/bin/python3 p5/p5_gaps.py <runs/p5-.../timeline.txt> [...]
"""
from __future__ import annotations

import sys

for path in sys.argv[1:]:
    rows = []
    for line in open(path, errors="replace"):
        if line.startswith("#"):
            continue
        p = line.split("\t", 3)
        if len(p) < 4:
            continue
        try:
            rows.append((float(p[0]), p[2], p[3].strip()[:70]))
        except ValueError:
            pass
    rows.sort()
    prev = None
    gaps = []
    for ts, tag, msg in rows:
        if prev is not None and ts - prev[0] > 5.0:
            gaps.append((round(ts - prev[0], 1), prev[1], prev[2], msg))
        prev = (ts, tag, msg)
    span = rows[-1][0] - rows[0][0] if rows else 0
    print("== %s  (%d lines, %.1f s span)" % (path, len(rows), span))
    for g in sorted(gaps, reverse=True)[:4]:
        print("   %8.1f s  after [%s] %-55s -> %s" % g)
