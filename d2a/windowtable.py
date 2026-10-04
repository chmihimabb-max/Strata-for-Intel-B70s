#!/usr/bin/env python3
"""D2a: the window table (index, first token it covers, T) of several arms side by side, to see where the
window sequences differ and what the windows around a given token were."""
import os
import re
import sys

RUNS = "/home/michael/strata-xpu/strata/d2/runs"
SUB = re.compile(r"strata submit: (?:stage|single) window T=(\d+) pos0=(\d+) ")


def windows(tag):
    if not os.path.exists(os.path.join(RUNS, tag, "err.txt")):
        return None
    w = []
    for l in open(os.path.join(RUNS, tag, "err.txt"), errors="replace"):
        m = SUB.search(l)
        if m:
            w.append((int(m.group(2)), int(m.group(1))))
    w = w[::2]
    base = w[0][0] if w else 0
    return [(i, p - base, t) for i, (p, t) in enumerate(w)]


def main():
    tags = sys.argv[1:] or ["d2-rebase-4096", "d2-mg-4096", "d2-mg-4096-warm", "d2-mg-4096-warm2"]
    tabs = {}
    for t in tags:
        w = windows(t)
        if w is None:
            print(f"{t}: no err.txt")
            continue
        tabs[t] = w
    print(f"{'win':>3s} " + " ".join(f"{t[-12:]:>14s}" for t in tabs))
    n = max(len(w) for w in tabs.values())
    for i in range(min(n, 34)):
        cells = []
        for t, w in tabs.items():
            cells.append(f"{w[i][1]}+{w[i][2]}" if i < len(w) else "-")
        print(f"{i:3d} " + " ".join(f"{c:>14s}" for c in cells))
    print()
    for t, w in tabs.items():
        print(f"{t:22s} T>4 windows: {[(i, tk) for i, _, tk in w if tk > 4]}")


if __name__ == "__main__":
    main()
