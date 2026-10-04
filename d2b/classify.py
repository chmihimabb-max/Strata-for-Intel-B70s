#!/usr/bin/env python3
"""D2b: classify the programs an arm built inside its decode phase -- the dense mmvq family (what the warm-up
covers), the routed experts' gu/down family, and everything else."""
import os
import re
import subprocess
import sys
from collections import Counter

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")


def classify(tag):
    lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
    er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
    cache = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg).group(1)
    fed = float(re.search(r"ask fed at .*epoch_ms (\d+)", lg).group(1)) / 1000.0
    fin = fed + float(re.search(r"the ask finished=1 after (\d+) ms", lg).group(1)) / 1000.0
    pm = re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er)
    pre = fed + float(pm.group(1)) / 1000.0
    rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                          capture_output=True, text=True).stdout.split("\n")
    cls = Counter()
    for line in rows:
        if not line.strip():
            continue
        t, d = line.split(" ", 1)
        t = float(t)
        if not (pre <= t <= fin + 0.5):
            continue
        blob = subprocess.run(["strings", "-a", os.path.join(d, "0.src")],
                              capture_output=True, text=True).stdout
        if "native_mmvq_multi_kernel" in blob or re.search(r"native_[a-z0-9_]*mmvq_kernel", blob) \
                or "native_quantize_q8_1" in blob:
            cls["dense mmvq (what the warm-up covers)"] += 1
        elif "native_gu_multi_kernel" in blob or "native_down_multi_kernel" in blob:
            cls["routed experts gu/down"] += 1
        else:
            m = re.search(r"native_[a-zA-Z0-9_]*kernel", blob)
            cls["other: " + (m.group(0) if m else "(unnamed)")] += 1
    print(f"{tag}: {sum(cls.values())} programs built in the decode phase")
    for k, n in cls.most_common(14):
        print(f"   {n:3d}  {k}")


for t in sys.argv[1:]:
    classify(t)
