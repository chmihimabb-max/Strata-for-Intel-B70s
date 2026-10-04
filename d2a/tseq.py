#!/usr/bin/env python3
"""D2a: the per-window T sequence of every d2 arm, from its own submit lines.

The submit line prints the stage's T, so the T sequence is directly readable per arm.
Prints, per arm: the sequence, its distinct counts, and whether it matches the shipped arms'.
"""
import glob, os, re, sys

RMS = "/home/michael/strata-xpu/strata/d2/runs"
pat = re.compile(r"strata submit: (?:stage|single) window T=(\d+) posN?0?=(\d+)")

def seq(tag):
    p = os.path.join(RMS, tag, "err.txt")
    if not os.path.exists(p):
        return None
    out = []
    for line in open(p, errors="replace"):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out

def compress(pairs):
    # one entry per window: a window is two stages (layers 0..22 then 23..47) with the same T
    ts = [t for t, _ in pairs]
    return ts[::2] if len(ts) % 2 == 0 else ts

tags = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RMS, "*/err.txt")))
seqs = {}
for t in tags:
    s = seq(t)
    if s is None:
        continue
    seqs[t] = compress(s)
    if not seqs[t]:
        continue
    from collections import Counter
    c = Counter(seqs[t])
    wide = sum(v for k, v in c.items() if k > 4)
    print(f"{t:22s} windows={len(seqs[t]):4d} T={dict(sorted(c.items()))} T>4 windows={wide} first20={seqs[t][:20]}")
print()
ref = "d2-minp05-4096"
for t in sorted(seqs):
    if t == ref or not seqs[t]:
        continue
    a, b = seqs[ref], seqs[t]
    n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), None)
    print(f"{t:22s} vs {ref}: len {len(a)} vs {len(b)}, first T difference at window {d}"
          + (f" ({a[d]} vs {b[d]})" if d is not None else "")
          + (", one is a prefix of the other" if d is None else ""))
