#!/usr/bin/env python3
"""P9 / step 2: separate the HOST rows from the DEVICE rows of a unitrace chrome timeline.

The first scan showed the two pseudo-processes (HOST<host> and DEVICE<host>[card]) and that the HOST
rows carry the shim's OWN launch() template name (which embeds the kernel function name) while the
DEVICE rows carry `zeCommandListAppend*(...)[bytes]`-style names.  This splits them and prints, per
pseudo-process, the top names by count and by summed dur, so the question "does the DEVICE row name
the KERNEL?" is answered from the file instead of assumed.

usage: /usr/bin/python3 p9/p9_scan2.py <trace.json> [topN]
"""
import json
import sys
from collections import Counter, defaultdict


def main():
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    proc_names = {}
    per = defaultdict(Counter)          # pid -> name -> count
    perdur = defaultdict(lambda: defaultdict(float))
    durs = defaultdict(list)            # pid -> list of durs (sampled)
    cats = defaultdict(Counter)
    pids = set()

    with open(path, "r", errors="replace") as f:
        for line in f:
            if '"ph"' not in line:
                continue
            try:
                e = json.loads(line.rstrip().rstrip(","))
            except Exception:
                continue
            if e.get("ph") == "M":
                if e.get("name") == "process_name":
                    proc_names[e.get("pid")] = e.get("args", {}).get("name")
                continue
            if e.get("ph") != "X":
                continue
            p = e.get("pid")
            nm = e.get("name", "?")
            d = float(e.get("dur", 0.0))
            pids.add(p)
            per[p][nm] += 1
            perdur[p][nm] += d
            cats[p][e.get("cat")] += 1
            if len(durs[p]) < 400000:
                durs[p].append(d)

    for p in sorted(pids, key=lambda x: -sum(per[x].values())):
        tot = sum(per[p].values())
        print("=" * 100)
        print(f"pid {p}  name={proc_names.get(p, '?')}  events={tot:,}  cats={dict(cats[p])}")
        d = durs[p]
        if d:
            d2 = sorted(d)
            print(f"  dur µs: min {d2[0]:.3f}  p50 {d2[len(d2) // 2]:.3f}  p90 {d2[int(len(d2) * 0.9)]:.3f}  "
                  f"p99 {d2[int(len(d2) * 0.99)]:.3f}  max {d2[-1]:.1f}  sum {sum(d2) / 1e3:.1f} ms")
        print(f"  -- top {topn} by COUNT --")
        for nm, c in per[p].most_common(topn):
            print(f"    {c:>10,}  sum {perdur[p][nm] / 1e3:>11.2f} ms  {nm[:110]}")
        print(f"  -- top {topn} by TOTAL DUR --")
        for nm, dd in sorted(perdur[p].items(), key=lambda kv: -kv[1])[:topn]:
            print(f"    {dd / 1e3:>11.2f} ms  n={per[p][nm]:>10,}  {nm[:110]}")
        print(f"  distinct names: {len(per[p]):,}")


if __name__ == "__main__":
    main()
