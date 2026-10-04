#!/usr/bin/env python3
"""D2a: is the token stream a function of the window-size sequence?  Three warm runs of the SAME lever share
three streams; this pairs each stream with the window sequence that produced it and gives the first window at
which two sequences differ, next to the first token at which their streams differ."""
import hashlib
import os
import re

RUNS = "/home/michael/strata-xpu/strata/d2/runs"
SUB = re.compile(r"strata submit: (?:stage|single) window T=(\d+) pos0=(\d+) ")
TAGS = [f"d2a-mg-4096-r{i}" for i in (1, 3, 4, 8)] + [f"d2a-mg-4096-r{i}" for i in (2, 5, 6)] + ["d2a-mg-4096-r7"]


def load(tag):
    w = [(int(m.group(2)), int(m.group(1)))
         for m in (SUB.search(l) for l in open(os.path.join(RUNS, tag, "err.txt"), errors="replace")) if m][::2]
    base = w[0][0]
    ts = [t for _, t in w]
    ids = [int(l.split()[1]) for l in open(os.path.join(RUNS, tag, "out.txt"), errors="replace")
           if l.startswith("T ")]
    return ts, [(i, p - base, t) for i, (p, t) in enumerate(w)], ids


def main():
    data = {t: load(t) for t in TAGS}
    seen = {}
    for t in TAGS:
        ts, _, ids = data[t]
        h = hashlib.md5((",".join(map(str, ts))).encode()).hexdigest()[:8]
        ih = hashlib.md5(("\n".join("T " + str(x) for x in ids)).encode()).hexdigest()[:8]
        seen.setdefault((h, ih), []).append(t)
    print("window sequence -> token stream (all runs with the same pair grouped)")
    for (h, ih), group in seen.items():
        print(f"  Tseq {h} -> ids {ih}: {len(group)} runs  {group}")
    print()
    reps = [g[0] for g in seen.values()]
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            ta, wa, ia = data[a]
            tb, wb, ib = data[b]
            n = min(len(ta), len(tb))
            dw = next((k for k in range(n) if ta[k] != tb[k]), None)
            di = next((k for k in range(min(len(ia), len(ib))) if ia[k] != ib[k]), None)
            print(f"{a} vs {b}: first window-size difference at window {dw} "
                  f"({ta[dw]} vs {tb[dw]})" if dw is not None else f"{a} vs {b}: same window sizes")
            print(f"    .. first token difference at index {di} "
                  f"({ia[di] if di is not None else '-'} vs {ib[di] if di is not None else '-'}); "
                  f"the window covering that token: #{[k for k, p, _ in wa if p <= di][-1] if di is not None else '-'}"
                  f" T={[(t) for _, p, t in wa if p <= di][-1] if di is not None else '-'}")
            print(f"    .. their first T>4 windows: {[(k, t) for k, _, t in wa if t > 4][:3]} vs "
                  f"{[(k, t) for k, _, t in wb if t > 4][:3]}")


if __name__ == "__main__":
    main()
