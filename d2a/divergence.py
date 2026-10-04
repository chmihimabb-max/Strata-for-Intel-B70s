#!/usr/bin/env python3
"""D2a: where exactly do the shipped-layout and generic-layout token streams part company?

Prints the first index at which two arms' generated token ids differ, and how many of the shipped stream's
tokens the generic stream agrees with up to that point.  Also prints, for each arm, the window boundaries
(pos0, T) so the first differing token can be mapped to the window that produced it.
"""
import os
import re
import sys

RUNS = "/home/michael/strata-xpu/strata/d2/runs"
SUB = re.compile(r"strata submit: (?:stage|single) window T=(\d+) pos0=(\d+) ")


def ids(tag):
    p = os.path.join(RUNS, tag, "out.txt")
    if not os.path.exists(p):
        return None
    return [int(l.split()[1]) for l in open(p, errors="replace") if l.startswith("T ")]


def windows(tag):
    p = os.path.join(RUNS, tag, "err.txt")
    out = []
    for l in open(p, errors="replace"):
        m = SUB.search(l)
        if m:
            out.append((int(m.group(2)), int(m.group(1))))
    return out[::2]


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "d2-rebase-4096"
    a = ids(ref)
    if a is None:
        print(f"reference {ref}: no output")
        return
    print(f"reference {ref}: {len(a)} tokens")
    for tag in sys.argv[2:] or ["d2-mg-4096", "d2-mg-4096-warm", "d2-mg-4096-warm2"]:
        b = ids(tag)
        if b is None:
            print(f"{tag:22s} (no output yet)")
            continue
        n = min(len(a), len(b))
        d = next((i for i in range(n) if a[i] != b[i]), None)
        w = windows(tag)
        # the window that committed token index d: the first window whose pos0 > d + (prompt len is 3832? the
        # T lines index from the start of generation; pos0 counts from 0 over the whole session, so map by
        # pos0 - pos0_of_first_window)
        base = w[0][0] if w else 0
        at = next(((i, p - base, t) for i, (p, t) in enumerate(w) if p - base > (d if d is not None else 0)), None)
        print(f"{tag:22s} {len(b)} tokens, first difference at index {d}"
              + (f" (ref {a[d]} vs {b[d]})" if d is not None else " (identical or a prefix)")
              + (f"; first window starting past it: #{at[0]} at token {at[1]}, T={at[2]}" if at else ""))
        print(f"    its window sizes T[0:20] = {[t for _, t in w][:20]}")


if __name__ == "__main__":
    main()
