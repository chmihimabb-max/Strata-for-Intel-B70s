#!/usr/bin/env python3
"""P9: per-tag device totals for one card's numpy cache (+ optional raw-line sanity check).

usage: /usr/bin/python3 p9/p9_summary.py <cache.npz> [topN] [--big] [--from T] [--to T]

--big   list the 20 biggest single events (t relative to the cache's first event)
--from/--to  restrict to a ts sub-range, given in SECONDS relative to the cache's first event
"""
import sys
from collections import defaultdict

import numpy as np


def main():
    p = sys.argv[1]
    topn = 25
    for a in sys.argv[2:]:
        if a.isdigit():
            topn = int(a)
    z = np.load(p, allow_pickle=True)
    ts, dur, tag = z["ts"], z["dur"], z["tag"]
    tags = [str(t) for t in z["tags"]]
    t0 = ts[0]
    if "--from" in sys.argv:
        m = ts >= t0 + float(sys.argv[sys.argv.index("--from") + 1]) * 1e6
        ts, dur, tag = ts[m], dur[m], tag[m]
    if "--to" in sys.argv:
        m = ts <= t0 + float(sys.argv[sys.argv.index("--to") + 1]) * 1e6
        ts, dur, tag = ts[m], dur[m], tag[m]
    print(f"{p}: {len(ts):,} events, busy {dur.sum() / 1e6:.1f} ms, span "
          f"{(ts[-1] - ts[0]) / 1e6:.3f} s (from {(ts[0] - t0) / 1e6:.3f} s into the cache)")
    print(f"dur us: min {dur.min():.3f} p50 {np.median(dur):.3f} p90 {np.percentile(dur, 90):.3f} "
          f"p99 {np.percentile(dur, 99):.3f} max {dur.max():.1f}")
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for i in range(len(dur)):
        nm = tags[tag[i]]
        tot[nm] += dur[i]
        cnt[nm] += 1
    print(f"\n{'total ms':>12} {'n':>9} {'avg us':>9}  tag")
    for nm, v in sorted(tot.items(), key=lambda kv: -kv[1])[:topn]:
        print(f"{v / 1e3:>12.3f} {cnt[nm]:>9,} {v / max(cnt[nm], 1):>9.3f}  {nm[:88]}")
    print(f"distinct tags: {len(tot)}   sum {sum(tot.values()) / 1e3:.1f} ms")
    if "--big" in sys.argv:
        print(f"\n== the 20 biggest single events ==")
        for j in np.argsort(-dur)[:20]:
            print(f"   t+{(ts[j] - t0) / 1e6:>9.4f} s  {dur[j] / 1e3:>9.3f} ms  {tags[tag[j]][:80]}")


if __name__ == "__main__":
    main()
