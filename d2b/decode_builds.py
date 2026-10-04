#!/usr/bin/env python3
"""D2b: split an arm's in-ask program writes into the prompt phase and the decode phase, using the engine's own
"prompt ... read in N ms" line -- the decode phase is where a window's residual lives, so this says which
programs the warm-up still left behind.

usage: decode_builds.py <arm-tag> [--names]
"""
import datetime as dt
import os
import re
import subprocess
import sys

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")


def entries(cache):
    rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                          capture_output=True, text=True).stdout.split("\n")
    return sorted((float(l.split(" ", 1)[0]), l.split(" ", 1)[1]) for l in rows if l.strip())


def sym(d):
    blob = subprocess.run(["strings", "-a", os.path.join(d, "0.src")], capture_output=True, text=True).stdout
    for pat in (r"native_mmvq_multi_kernelI[^\s]*", r"native_[a-z0-9_]*mmvq[a-zA-Z0-9_]*kernel[^\s]*",
                r"native_[a-zA-Z0-9_]*kernel[^\s]*"):
        m = re.search(pat, blob)
        if m:
            return m.group(0)
    return "?"


def main():
    tag = sys.argv[1]
    names = "--names" in sys.argv
    lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
    er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
    cache = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg).group(1)
    fed = float(re.search(r"ask fed at .*epoch_ms (\d+)", lg).group(1)) / 1000.0
    fin = fed + float(re.search(r"the ask finished=1 after (\d+) ms", lg).group(1)) / 1000.0
    pm = re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er)
    pre = fed + float(pm.group(1)) / 1000.0 if pm else fed
    E = entries(cache)
    prompt = [(t, d) for t, d in E if fed - 0.5 <= t < pre]
    dec = [(t, d) for t, d in E if pre <= t <= fin + 0.5]
    shape = lambda x: max(0, int(x / 1000))
    print(f"{tag}: cache {cache}")
    print(f"  ask {dt.datetime.fromtimestamp(fed):%H:%M:%S} -> {dt.datetime.fromtimestamp(fin):%H:%M:%S}"
          f"   prompt phase {float(pm.group(1))/1000:.1f}s, decode phase {(fin-pre):.1f}s")
    print(f"  programs written in the PROMPT phase: {len(prompt)}")
    print(f"  programs written in the DECODE phase: {len(dec)}")
    if names and dec:
        for t, d in dec[:40]:
            print(f"     +{t - fed:6.2f}s  {sym(d)[:150]}")
    if dec:
        from collections import Counter
        c = Counter(sym(d) for _, d in dec)
        print("  decode-phase programs, by name:")
        for s, n in c.most_common():
            print(f"     {n:3d} x {s[:150]}")


if __name__ == "__main__":
    main()
