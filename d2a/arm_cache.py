#!/usr/bin/env python3
"""D2a: which SYCL program-cache entries were written while each arm ran, and when.

For every d2 arm: the arm's own load/ask/finish timestamps (from its log.txt) and the
cache entries (mtime) that fall inside [arm start, arm end], with their size classes.
"""
import datetime as dt
import os
import re
import subprocess

SRC = "/home/michael/strata-xpu/strata"
CACHE = "/home/michael/strata-xpu/sycl-cache/m6c/13665086051514919768"

def entries():
    out = subprocess.run(["find", CACHE, "-name", "0.src", "-printf", "%T@\n"],
                         capture_output=True, text=True).stdout.split()
    return sorted(float(x) for x in out)

def parse_log(tag):
    p = os.path.join(SRC, "d2/runs", tag, "log.txt")
    txt = open(p, errors="replace").read()
    def t(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    # "D1 arm <tag>   2026-10-03T19:23:17-06:00"  and  "== loaded=1 after 31s (2026-10-03T19:23:50-06:00) =="
    def iso(s):
        return dt.datetime.fromisoformat(s).timestamp()
    start = None
    m = re.search(r"^D1 arm \S+\s+(\S+)", txt, re.M)
    if m:
        start = iso(m.group(1))
    loaded = None
    m = re.search(r"loaded=1 after \d+s \((\S+)\)", txt)
    if m:
        loaded = iso(m.group(1))
    fed = None
    m = re.search(r"ask fed at (\S+) \(epoch_ms (\d+)\)", txt)
    if m:
        fed = float(m.group(2)) / 1000.0
    fin = None
    m = re.search(r"the ask finished=1 after (\d+) ms \((\S+)\)", txt)
    if m:
        fin = iso(m.group(2))
    exit_ = None
    m = re.search(r"engine exited \d+ at epoch ([\d.]+)", txt)
    if m:
        exit_ = float(m.group(1))
    return start, loaded, fed, fin, exit_

def hm(x):
    return dt.datetime.fromtimestamp(x).strftime("%H:%M:%S") if x else "?"

E = entries()
tags = sorted(os.path.basename(os.path.dirname(p)) for p in
              __import__("glob").glob(os.path.join(SRC, "d2/runs/*/log.txt")))
for tag in tags:
    start, loaded, fed, fin, exit_ = parse_log(tag)
    if not (fed and fin):
        continue
    inside = [x for x in E if fed - 0.5 <= x <= fin + 0.5]
    dur = fin - fed
    line = (f"{tag:20s} start {hm(start)} load {hm(loaded)} ({loaded-start:.0f}s) "
            f"ask {hm(fed)}->{hm(fin)} ({dur:.1f}s)  ")
    if not inside:
        print(line + "no cache entries written")
        continue
    sizes = {}
    for x in inside:
        sizes[round(x - inside[0], 1)] = sizes.get(round(x - inside[0], 1), 0) + 1
    print(line + f"{len(inside)} cache entries during the ask, at +{inside[0]-fed:.1f}..+{inside[-1]-fed:.1f}s")
    # the 5-way size classes = the quant types; report per timestamp group
    groups = {}
    for x in inside:
        groups.setdefault(round(x - fed, 1), 0)
        groups[round(x - fed, 1)] += 1
    print("      entries per second after the ask: " +
          ", ".join(f"+{k}s:{v}" for k, v in sorted(groups.items())))
