#!/usr/bin/env python3
"""D2c: per-site cost ranking for the programs an arm built inside its decode phase.

Each 0.src is written when its JIT finishes, so the gap between consecutive writes is the build cost of the
later program (plus whatever else the host did in between).  Summing the gaps gives each site's share of the
phase, which is what a warm-up for that site would move out of the request.

usage: site_cost.py <arm-tag> [--top N]
"""
import os
import re
import subprocess
import sys

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")
SITE_RE = re.compile(r"sycl_compat::launch<strata::kernels::([A-Za-z0-9_]+)")
KERNEL_RE = re.compile(r"strata::kernels::(?:\(anonymous namespace\)::)?([A-Za-z0-9_]+)(<[^>]*>)?\(unsigned char\*")


def site(d):
    raw = open(os.path.join(d, "0.src"), "rb").read()
    toks = sorted({t.decode("ascii", "replace") for t in raw.split(b"\x00")
                   if t.startswith(b"_Z") and len(t) > 8})
    dem = subprocess.run(["c++filt"] + toks, capture_output=True, text=True).stdout.split("\n") if toks else []
    s = k = None
    for x in dem:
        m = SITE_RE.search(x)
        if m and not s:
            s = m.group(1)
        m = KERNEL_RE.search(x)
        if m and not k:
            k = m.group(1) + (m.group(2) or "")
    return (s or "(no site)") + (f" -> {k}" if k else "")


tag = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
cache = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg).group(1)
fed = float(re.search(r"ask fed at .*epoch_ms (\d+)", lg).group(1)) / 1000.0
pre = fed + float(re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er).group(1)) / 1000.0
fin = fed + float(re.search(r"the ask finished=1 after (\d+) ms", lg).group(1)) / 1000.0
rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                      capture_output=True, text=True).stdout.split("\n")
E = sorted((float(l.split(" ", 1)[0]), l.split(" ", 1)[1]) for l in rows if l.strip())
dec = [(t, d) for t, d in E if pre <= t <= fin + 0.5]
print(f"# {tag}: {len(dec)} decode-phase programs, phase {fin - pre:.2f}s"
      f" (first write {dec[0][0] - pre:.2f}s after the prompt line)")
tot = 0.0
per_site = {}
prev = pre
for t, d in dec:
    s = site(d)
    gap = (t - prev) * 1000.0
    per_site[s] = per_site.get(s, 0.0) + gap
    tot += gap
    prev = t
print(f"# summed gaps (first write included, tail of the phase not): {tot:.0f} ms")
for s, ms in sorted(per_site.items(), key=lambda x: -x[1])[:top]:
    print(f"  {ms:8.1f} ms  {s}")
print(f"# {len(per_site)} distinct sites")
