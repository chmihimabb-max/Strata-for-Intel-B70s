#!/usr/bin/env python3
"""D2b (card t_f93760a1): name the SYCL programs an arm built inside its own ask, from the mangled symbols in
each entry's 0.src -- the same rule as d2a/program_names.sh, but with the whole specialization name and the
arm's own cache directory, so a cold run's 153 programs can be grouped (which (type, NCOLS, NW, ROWS) pairs).

usage: programs_b.py <arm-tag> [--mmvq-only] [--limit N]
"""
import os
import re
import subprocess
import sys

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")


def fed_window(tag):
    lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
    fed = re.search(r"ask fed at .*epoch_ms (\d+)", lg)
    fin = re.search(r"the ask finished=1 after (\d+) ms", lg)
    cm = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg)
    if not (fed and fin and cm):
        return None, None, None
    a = float(fed.group(1)) / 1000.0
    return cm.group(1), a - 0.5, a + float(fin.group(1)) / 1000.0 + 0.5


def main():
    tag = sys.argv[1]
    only_mmvq = "--mmvq-only" in sys.argv
    limit = 200
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    cache, lo, hi = fed_window(tag)
    if not cache:
        print(f"{tag}: no ask window / cache line")
        return 1
    out = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                         capture_output=True, text=True).stdout.split("\n")
    rows = []
    for line in out:
        if not line.strip():
            continue
        t, d = line.split(" ", 1)
        if lo <= float(t) <= hi:
            rows.append((float(t), d))
    rows.sort()
    print(f"# {tag}: {len(rows)} SYCL programs written inside the ask (cache {cache})")
    syms = []
    for t, d in rows:
        blob = subprocess.run(["strings", "-a", os.path.join(d, "0.src")],
                              capture_output=True, text=True).stdout
        m = re.search(r"native_mmvq_multi_kernelI[^\s]*", blob)
        if not m:
            m = re.search(r"native_[a-z0-9_]*(?:mmvq|quantize)[a-zA-Z0-9_]*kernel[^\s]*", blob)
        if not m:
            m = re.search(r"native_[a-zA-Z0-9_]*(?:kernel|Traits)[^\s]*", blob)
        sym = m.group(0) if m else "?"
        if only_mmvq and "mmvq" not in sym:
            continue
        syms.append(sym)
        print(f"+{t - rows[0][0]:7.2f}s  {len(syms):3d}  {sym}")
    print(f"# distinct: {len(set(syms))} of {len(syms)}")
    for s, n in sorted(((s, syms.count(s)) for s in set(syms)), key=lambda x: -x[1]):
        print(f"   {n:3d} x {s[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
