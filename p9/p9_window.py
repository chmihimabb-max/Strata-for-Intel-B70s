#!/usr/bin/env python3
"""P9: the phases of one card's device stream, and the first look inside the decode region.

A decode window's kernels are all short (P1b's 4K trace: device p99 = 236 us, max in decode ~1 ms) while
the PREFILL's QSA prompt attention is ~95 ms per launch.  So the last device event above a threshold is
the end of the prefill, and everything after it is decode.  This script locates that boundary and prints
what is on each side, plus the first N events of the decode region as a sequence.

usage: /usr/bin/python3 p9/p9_window.py <cache.npz> [--big-ms 5] [--list 300] [--tag NAME]
"""
import sys
from collections import defaultdict

import numpy as np


def main():
    p = sys.argv[1]
    big_ms = float(sys.argv[sys.argv.index("--big-ms") + 1]) if "--big-ms" in sys.argv else 5.0
    nlist = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 300
    want = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None

    z = np.load(p, allow_pickle=True)
    ts, dur, tag = z["ts"], z["dur"], z["tag"]
    tags = [str(t) for t in z["tags"]]
    t0 = ts[0]
    print(f"{p}: {len(ts):,} events, {len(tags)} tags, span {(ts[-1] - t0) / 1e6:.3f} s, "
          f"busy {dur.sum() / 1e6:.1f} ms")
    print(f"first event {t0 / 1e6:.3f} (epoch s)   last {ts[-1] / 1e6:.3f}")

    # ---- phases: locate the last 'big' event = the prefill's last attention launch
    big = np.nonzero(dur > big_ms * 1e3)[0]
    if len(big) == 0:
        print(f"NO device event above {big_ms} ms: cannot place the prefill/decode boundary this way")
        return
    b = big[-1]
    print(f"\n== 'big' events (> {big_ms} ms): {len(big)} ==")
    print(f"   last one at {(ts[b] - t0) / 1e6:.3f} s, dur {dur[b] / 1e3:.2f} ms: {tags[tag[b]][:90]}")
    for k in ["count", "dur", "tag"]:
        pass
    c = defaultdict(int)
    for i in big:
        c[tags[tag[i]]] += 1
    for nm, n in sorted(c.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   n={n:>6,}  {nm[:90]}")

    if want:
        want_id = [i for i, nm in enumerate(tags) if want in nm]
        sel = np.isin(tag, want_id)
        print(f"\n== every event whose tag contains {want!r}: {int(sel.sum())} ==")
        idx = np.nonzero(sel)[0]
        print(f"   dur ms: min {dur[idx].min() / 1e3:.3f} p50 {np.median(dur[idx]) / 1e3:.3f} "
              f"max {dur[idx].max() / 1e3:.3f}  total {dur[idx].sum() / 1e3:.1f}")
        for j in idx[:40]:
            print(f"   {(ts[j] - t0) / 1e6:>10.4f} s  {dur[j]:>12.2f} us  {tags[tag[j]][:70]}")
        return

    # ---- the two sides of the boundary
    for label, sl in (("PREFILL (up to and including the last big event)", slice(0, b + 1)),
                      ("after (decode + teardown)", slice(b + 1, None))):
        d = dur[sl]
        print(f"\n==== {label}: {len(d):,} events, busy {d.sum() / 1e6:.1f} ms, "
              f"wall {(ts[sl][-1] - ts[sl][0]) / 1e6:.3f} s, p50 {np.median(d):.1f} us, max {d.max() / 1e3:.2f} ms")

    print(f"\n== the first {nlist} events after the boundary ==")
    for j in range(b + 1, min(b + 1 + nlist, len(ts))):
        print(f"   t+{(ts[j] - ts[b]) / 1e3:>10.3f} ms  {dur[j]:>10.2f} us  {tags[tag[j]][:78]}")


if __name__ == "__main__":
    main()
