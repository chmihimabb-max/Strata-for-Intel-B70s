#!/usr/bin/env python3
"""D2c (card t_c7d8cd86): name every SYCL program an arm built inside a window phase, by its LAUNCH SITE.

Every 0.src here is the SPIR-V module of one `strata::sycl_compat::launch(...)` call.  Its NUL-separated name
section carries mangled `_Z...` tokens; c++filt turns them into

  typeinfo name for strata::sycl_compat::launch<strata::kernels::<SITE>(args)::{lambda...}>::operator()<...>
  void strata::kernels::(anonymous namespace)::<KERNEL><template args>(unsigned char*, ..., launch_shape, ...)

The first gives the launch SITE (the helper function in src/kernels/cuda/*.cu that made the call); the second is
present only for a kernel written as a real function template (the templated ones) and gives its exact
instantiation.  D2b's classify.py/name_other.py saw neither for 35 of the 61 post-warm-up decode-phase programs
because they only looked for a `native_*kernel` symbol at the start of a strings line.

usage: enumerate_c.py <arm-tag> [--phase decode|prompt|ask] [--list] [--detail SUBSTR] [--json OUT]
"""
import json
import os
import re
import subprocess
import sys
from collections import Counter

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")

SITE_RE = re.compile(r"sycl_compat::launch<strata::kernels::([A-Za-z0-9_]+)")
KERNEL_RE = re.compile(r"strata::kernels::(?:\(anonymous namespace\)::)?([A-Za-z0-9_]+)(<[^>]*>)?\(unsigned char\*")


def arm_window(tag):
    lg = open(os.path.join(RUNS, tag, "log.txt"), errors="replace").read()
    er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
    cache = re.search(r"SYCL_CACHE_DIR=(\S+) \(", lg).group(1)
    fed = float(re.search(r"ask fed at .*epoch_ms (\d+)", lg).group(1)) / 1000.0
    fin = fed + float(re.search(r"the ask finished=1 after (\d+) ms", lg).group(1)) / 1000.0
    pm = re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er)
    pre = fed + float(pm.group(1)) / 1000.0 if pm else fed
    return cache, fed, pre, fin


def entries(cache, lo, hi):
    out = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                         capture_output=True, text=True).stdout.split("\n")
    rows = []
    for line in out:
        if line.strip():
            t, d = line.split(" ", 1)
            if lo <= float(t) <= hi:
                rows.append((float(t), d))
    return sorted(rows)


def demangle(d):
    raw = open(os.path.join(d, "0.src"), "rb").read()
    toks = sorted({t.decode("ascii", "replace") for t in raw.split(b"\x00")
                   if t.startswith(b"_Z") and len(t) > 8})
    if not toks:
        return []
    return [x for x in subprocess.run(["c++filt"] + toks, capture_output=True, text=True).stdout.split("\n") if x.strip()]


def site_of(dem):
    """(launch site, exact kernel instantiation or None)"""
    site, kernel = None, None
    for s in dem:
        m = SITE_RE.search(s)
        if m and not site:
            site = m.group(1)
        m = KERNEL_RE.search(s)
        if m and not kernel:
            kernel = m.group(1) + (m.group(2) or "")
    return site, kernel


def main():
    tag = sys.argv[1]
    phase = "decode"
    if "--phase" in sys.argv:
        phase = sys.argv[sys.argv.index("--phase") + 1]
    cache, fed, pre, fin = arm_window(tag)
    lo, hi = {"prompt": (fed - 0.5, pre), "decode": (pre, fin + 0.5), "ask": (fed - 0.5, fin + 0.5)}[phase]
    rows = entries(cache, lo, hi)
    sites, first, detail = Counter(), {}, {}
    for t, d in rows:
        dem = demangle(d)
        site, kernel = site_of(dem)
        key = f"{site or '(NO SITE)'}" + (f" -> {kernel}" if kernel else "")
        sites[key] += 1
        first.setdefault(key, t - rows[0][0])
        detail.setdefault(key, {"dir": d, "dem": dem[:3]})
    print(f"# {tag} phase={phase}: {len(rows)} programs, {len(sites)} distinct launch sites  (cache {cache})")
    for k, n in sites.most_common():
        print(f"  {n:3d} x {k}   (first +{first[k]:.1f}s)")
    if "--list" in sys.argv:
        print("\n# per entry, in build order")
        for t, d in rows:
            dem = demangle(d)
            site, kernel = site_of(dem)
            print(f"  +{t - rows[0][0]:7.2f}s  {site or '(NO SITE)'}" + (f" -> {kernel}" if kernel else ""))
    if "--detail" in sys.argv:
        sub = sys.argv[sys.argv.index("--detail") + 1]
        for k, v in detail.items():
            if sub in k:
                print(f"\n### {k}\n  dir {v['dir']}")
                for s in v["dem"]:
                    print(f"    {s[:300]}")
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump({k: {"n": n, "first_s": round(first[k], 2), "dir": detail[k]["dir"]}
                   for k, n in sites.items()}, open(out, "w"), indent=1)
        print(f"# wrote {out}")


if __name__ == "__main__":
    main()
