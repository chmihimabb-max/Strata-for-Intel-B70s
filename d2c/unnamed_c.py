#!/usr/bin/env python3
"""D2c: dump the strings of the decode-phase 0.src entries that carry NO `_ZN6strata...` mangled name,
so they can be named from whatever the SPIR-V does contain (entry-point name, file path, op names)."""
import os
import re
import subprocess
import sys
from collections import Counter

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")
tag = sys.argv[1] if len(sys.argv) > 1 else "d2b-wu-cold-4096"
lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
cache = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg).group(1)
fed = float(re.search(r"ask fed at .*epoch_ms (\d+)", lg).group(1)) / 1000.0
fin = fed + float(re.search(r"the ask finished=1 after (\d+) ms", lg).group(1)) / 1000.0
pre = fed + float(re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er).group(1)) / 1000.0
rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                      capture_output=True, text=True).stdout.split("\n")
n = 0
seen = Counter()
for line in rows:
    if not line.strip():
        continue
    t, d = line.split(" ", 1)
    if not (pre <= float(t) <= fin + 0.5):
        continue
    raw = open(os.path.join(d, "0.src"), "rb").read()
    if re.search(rb"_ZN6strata", raw):
        continue
    n += 1
    blob = subprocess.run(["strings", "-a", os.path.join(d, "0.src")], capture_output=True, text=True).stdout
    # the OP names / entry point / any kernel-ish token
    cands = [s for s in blob.splitlines()
             if re.search(r"kernel|Kernel|_Z|spirv|OpE|\.cpp|\.cu|\.hpp", s) and len(s) < 160]
    key = " | ".join(cands[:4])[:200]
    seen[key] += 1
    if n <= 3:
        print(f"### entry {n}: {d}")
        for s in blob.splitlines()[:45]:
            print(f"    {s[:150]}")
        print()
print(f"# {n} entries with no _ZN6strata name; grouped by first interesting strings:")
for k, c in seen.most_common(40):
    print(f"  {c:3d}  {k}")
