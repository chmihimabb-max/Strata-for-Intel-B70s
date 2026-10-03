#!/usr/bin/env python3
"""D1 (card t_d9ffcf38): the launch-site / kernel histogram of ONE decode window.

Input: the file the engine wrote (STRATA_LAUNCH_HIST_FILE), lines
  HW <tag> submissions <n> sites <k>
  H  <tag> <count> <us> <name>
with <tag> = `win=<i> [stage|single] T=<t> pos0=<p> layers <a>..<b> n=<layers> wall=<ms>ms`.

A window's slices (stage 0's and the nested stage 1's) share pos0, so they are summed here.  The name is the
launch site: the enclosing host wrapper + `#<index>` of the launcher lambda inside it (the shim demangles it
from the lambda's own type; see the histogram note in include/strata/sycl_compat/cuda_runtime.h).  One kernel
can have several sites (a fast path and a fallback, a dynamic-shared-memory variant), so the tables below are
printed twice: merged per KERNEL (the `#n` dropped) and per SITE.

usage: d1_hist.py <hist.txt> [--window N] [--auto] [--top 20] [--sites] [--csv OUT]
  --auto   pick the window with the largest counted submission total (in graph mode that is the window that
           RECORDS: every later window at the same T is one replay and counts nothing)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def short(name: str) -> str:
    n = name
    for pre in ("strata::kernels::(anonymous namespace)::", "strata::kernels::", "strata::sycl_compat::",
                "(anonymous namespace)::"):
        n = n.replace(pre, "")
    depth = 0
    out = []
    for ch in n:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "(" and depth == 0:
            break
        out.append(ch)
    return "".join(out).strip()


def parse(path: str):
    windows: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)
    meta: Dict[int, dict] = defaultdict(lambda: {"subs": 0, "walls": [], "tags": []})
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        tag = f[1]
        m = re.search(r"win=(\d+)", tag)
        if not m:
            continue
        w = int(m.group(1))
        if f[0] == "HW":
            mm = re.search(r"submissions (\d+)", f[2])
            if mm:
                meta[w]["subs"] += int(mm.group(1))
            meta[w]["tags"].append(tag)
        elif f[0] == "H" and len(f) >= 5:
            windows[w].append((int(f[2]), float(f[3]), f[4]))
            wm = re.search(r"wall=([\d.]+)ms", tag)
            if wm:
                meta[w]["walls"].append(float(wm.group(1)))
    return windows, meta


def table(rows, top, title, unit):
    tot_c = sum(r[1] for r in rows)
    tot_us = sum(r[2] for r in rows)
    print(title)
    print("%-4s %-9s %-10s %-7s %s" % ("#", "count", "us", "share", unit))
    for i, (name, c, us) in enumerate(rows[:top], 1):
        print("%-4d %-9d %-10.1f %-7s %s" % (i, c, us, "%.1f%%" % (100.0 * us / tot_us if tot_us else 0.0), name))
    print("     %-9d %-10.1f  (%.2f ms of %.2f ms counted)" % (tot_c, tot_us, tot_us / 1000.0, tot_us / 1000.0))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--sites", action="store_true", help="also print the per-site table")
    ap.add_argument("--csv", default=None)
    o = ap.parse_args()

    windows, meta = parse(o.path)
    if not windows:
        print("no histogram lines in %s" % o.path)
        return 1
    print("windows: " + ", ".join(
        "win%d(count %d, subs %s, wall %s)" % (
            w, sum(c for c, _, _ in windows[w]), meta[w]["subs"],
            "%.2f/%.2f" % (min(meta[w]["walls"]), max(meta[w]["walls"])) if meta[w]["walls"] else "?")
        for w in sorted(windows)))
    w = max(windows, key=lambda x: sum(c for c, _, _ in windows[x])) if o.auto else o.window
    if w not in windows:
        print("window %d not in the file" % w)
        return 1
    print("\n== window %d ==" % w)
    for t in meta[w]["tags"]:
        print("   " + t)
    kern: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
    site: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
    for c, us, name in windows[w]:
        kern[short(name)][0] += c
        kern[short(name)][1] += us
        site[name][0] += c
        site[name][1] += us
    krows = sorted([(k, int(v[0]), v[1]) for k, v in kern.items()], key=lambda r: (-r[2], -r[1]))
    srows = sorted([(k, int(v[0]), v[1]) for k, v in site.items()], key=lambda r: (-r[2], -r[1]))
    table(krows, o.top, "-- top %d KERNELS by device microseconds --" % o.top, "kernel (sites merged)")
    kc = sorted(krows, key=lambda r: (-r[1], -r[2]))
    table(kc, o.top, "-- top %d KERNELS by count --" % o.top, "kernel (sites merged)")
    if o.sites:
        table(srows, o.top, "-- top %d SITES by device microseconds --" % o.top, "launch site")
        table(sorted(srows, key=lambda r: (-r[1], -r[2])), o.top,
              "-- top %d SITES by count --" % o.top, "launch site")
    if o.csv:
        with open(o.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["window", "count", "us", "kernel", "site"])
            for name, c, us in srows:
                wr.writerow([w, c, "%.3f" % us, short(name), name])
        print("wrote %s (%d sites)" % (o.csv, len(srows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
