#!/usr/bin/env python3
"""D1 (card t_d9ffcf38): the launch-site histogram of ONE decode window.

Input: the file the engine wrote (STRATA_LAUNCH_HIST_FILE), whose lines are
  HW <tag> submissions <n> sites <k>
  H  <tag> <count> <us> <name>
where <tag> is `win=<i> [stage|single] T=<t> pos0=<p> layers <a>..<b> n=<layers> wall=<ms>ms`.

A window's slices (stage 0's and the nested stage 1's) share pos0, so they are summed here: the output is one
table per (window, sorted by total device microseconds and by count), restricted to ONE window (--window N, the
one whose census is complete: in graph mode that is the window that RECORDS, i.e. window 1).

usage: d1_hist.py <hist.txt> [--window 1] [--top 20] [--by count|us]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def parse(path: str):
    windows: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)
    subs = {}
    walls = {}
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if not f:
            continue
        if f[0] == "HW":
            tag = f[1]
            m = re.search(r"win=(\d+)", tag)
            if not m:
                continue
            w = int(m.group(1))
            n = None
            if len(f) > 2:
                mm = re.search(r"submissions (\d+)", f[2])
                if mm:
                    n = int(mm.group(1))
            subs[w] = (subs.get(w, 0) + n) if n is not None else subs.get(w)
        elif f[0] == "H" and len(f) >= 5:
            tag = f[1]
            m = re.search(r"win=(\d+)", tag)
            if not m:
                continue
            w = int(m.group(1))
            windows[w].append((int(f[2]), float(f[3]), f[4]))
            wm = re.search(r"wall=([\d.]+)ms", tag)
            if wm:
                walls.setdefault(w, []).append(float(wm.group(1)))
    return windows, subs, walls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--by", choices=["count", "us"], default="us")
    o = ap.parse_args()

    windows, subs, walls = parse(o.path)
    if not windows:
        print("no histogram lines in %s" % o.path)
        return 1
    print("windows present: " + ", ".join(
        "win%d(count=%d, subs=%s, wall=%s)" % (
            w, sum(c for c, _, _ in windows[w]), subs.get(w, "?"),
            ("%.2f" % (sum(walls[w]) / len(walls[w]))) if w in walls else "?")
        for w in sorted(windows)))
    w = o.window
    if w not in windows:
        print("window %d not in the file" % w)
        return 1
    agg: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
    for c, us, name in windows[w]:
        agg[name][0] += c
        agg[name][1] += us
    rows = [(name, int(v[0]), v[1]) for name, v in agg.items()]
    tot_c = sum(r[1] for r in rows)
    tot_us = sum(r[2] for r in rows)
    key = (lambda r: (-r[2], -r[1])) if o.by == "us" else (lambda r: (-r[1], -r[2]))
    rows.sort(key=key)
    print("window %d: %d launch sites, %d counted submissions, %.2f ms of summed per-submission device time"
          % (w, len(rows), tot_c, tot_us / 1000.0))
    print("%-6s %-10s %-8s %-8s %s" % ("rank", "count", "us", "share", "launch site"))
    for i, (name, c, us) in enumerate(rows[:o.top], 1):
        print("%-6d %-10d %-8.1f %-8s %s" % (i, c, us, "%.1f%%" % (100.0 * us / tot_us if tot_us else 0), name))
    print("\n-- top %d by COUNT --" % o.top)
    rows.sort(key=lambda r: (-r[1], -r[2]))
    print("%-6s %-10s %-8s %s" % ("rank", "count", "us", "launch site"))
    for i, (name, c, us) in enumerate(rows[:o.top], 1):
        print("%-6d %-10d %-8.1f %s" % (i, c, us, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
