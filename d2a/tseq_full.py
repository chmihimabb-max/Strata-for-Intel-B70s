#!/usr/bin/env python3
"""D2a: the full T sequence of the three generic-layout runs against the shipped control."""
import os, re

RMS = "/home/michael/strata-xpu/strata/d2/runs"
pat = re.compile(r"strata submit: (?:stage|single) window T=(\d+) posN?0?=(\d+)")

def seq(tag):
    ts = []
    for line in open(os.path.join(RMS, tag, "err.txt"), errors="replace"):
        m = pat.search(line)
        if m:
            ts.append(int(m.group(1)))
    return ts[::2]

tags = ["d2-rebase-4096", "d2-mg-4096", "d2-mg-4096-warm", "d2-mg-4096-warm2"]
s = {t: seq(t) for t in tags}
w = 4
for t in tags:
    print(f"{t:20s} n={len(s[t])}  T={s[t]}")
print()
for a, b in [("d2-rebase-4096", "d2-mg-4096"), ("d2-mg-4096", "d2-mg-4096-warm"),
             ("d2-mg-4096-warm", "d2-mg-4096-warm2"), ("d2-mg-4096", "d2-mg-4096-warm2")]:
    x, y = s[a], s[b]
    n = min(len(x), len(y))
    d = next((i for i in range(n) if x[i] != y[i]), None)
    print(f"{a} vs {b}: first T difference at window {d}"
          + (f" ({x[d]} vs {y[d]})" if d is not None else " (common prefix; lengths differ)")
          + f"  [lens {len(x)}/{len(y)}]")
print()
# where are the T>4 windows (the only windows whose numerics differ between the layouts)?
for t in tags:
    wide = [(i, v) for i, v in enumerate(s[t]) if v > 4]
    print(f"{t:20s} T>4 windows (index,T): {wide}")
