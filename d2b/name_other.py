#!/usr/bin/env python3
"""D2b: name the programs the classifier could not name -- dump the interesting strings of each decode-phase
entry that has no native_*kernel symbol."""
import os
import re
import subprocess
import sys
from collections import Counter

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")
tag = sys.argv[1]
lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
cache = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg).group(1)
fed = float(re.search(r"ask fed at .*epoch_ms (\d+)", lg).group(1)) / 1000.0
fin = fed + float(re.search(r"the ask finished=1 after (\d+) ms", lg).group(1)) / 1000.0
pre = fed + float(re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er).group(1)) / 1000.0
rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                      capture_output=True, text=True).stdout.split("\n")
c = Counter()
examples = {}
for line in rows:
    if not line.strip():
        continue
    t, d = line.split(" ", 1)
    t = float(t)
    if not (pre <= t <= fin + 0.5):
        continue
    blob = subprocess.run(["strings", "-a", os.path.join(d, "0.src")], capture_output=True, text=True).stdout
    if re.search(r"native_[a-zA-Z0-9_]*kernel|native_gu_multi|native_down_multi", blob):
        continue
    cands = [s for s in blob.splitlines()
             if re.match(r"^(strata|[a-z_]+)::", s) and len(s) < 200 and " " not in s]
    if not cands:
        m = re.search(r"GLOBAL__N_1\d+([a-zA-Z0-9_]*kernel)[A-Za-z0-9_]*", blob)
        if m:
            cands = ["(anon ns) " + m.group(0)[:120]]
    key = cands[0] if cands else "(no symbol)"
    c[key] += 1
    examples.setdefault(key, d)
for k, n in c.most_common(20):
    print(f"{n:3d}  {k[:130]}")
    print(f"      e.g. {examples[k]}")
