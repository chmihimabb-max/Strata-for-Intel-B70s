#!/usr/bin/env python3
"""P9: the coarse shape of a device-event TSV -- where are the load / prefill / decode phases?

Bins the device (gpu_op) rows of one card into BINms bins and prints busy ms + event count per bin,
plus the bin's dominant tag.  Used to locate the decode phase before any window is attributed.

usage: /usr/bin/python3 p9/p9_timeline.py <dev.tsv> [binms] [csvout]
"""
import sys
from collections import Counter, defaultdict

BIN = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
CSV = sys.argv[3] if len(sys.argv) > 3 else None


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 4)
            if len(p) < 5:
                continue
            rows.append((float(p[2]), float(p[3]), p[4]))
    rows.sort()
    t0 = rows[0][0]
    t1 = max(r[0] + r[1] for r in rows)
    print(f"file {path}")
    print(f"events {len(rows):,}  span {(t1 - t0) / 1e3:.3f} ms .. {(t1) / 1e3:.1f} "
          f"(wall {(t1 - t0) / 1e6:.2f} s)")
    print(f"busy total {sum(r[1] for r in rows) / 1e3:.1f} ms "
          f"= {sum(r[1] for r in rows) / 1e6 / ((t1 - t0) / 1e6) * 100:.1f}% of the span")

    nb = int((t1 - t0) / (BIN * 1000)) + 1
    busy = [0.0] * nb
    cnt = [0] * nb
    tagsum = defaultdict(lambda: defaultdict(float))
    for ts, dur, tag in rows:
        b = int((ts - t0) / (BIN * 1000))
        busy[b] += dur
        cnt[b] += 1
        tagsum[b][tag] += dur
    print(f"\nbin = {BIN} ms;  t = bin start relative to the first event (ms)")
    print(f"{'bin':>7} {'t_ms':>10} {'busy_ms':>9} {'events':>8}  dominant tag (ms in bin)")
    out = open(CSV, "w") if CSV else None
    if out:
        out.write("bin\tt_ms\tbusy_ms\tevents\tdominant\tdom_ms\n")
    for b in range(nb):
        if cnt[b] == 0:
            continue
        dom, dm = max(tagsum[b].items(), key=lambda kv: kv[1])
        print(f"{b:>7} {b * BIN:>10.1f} {busy[b] / 1e3:>9.2f} {cnt[b]:>8,}  {dom[:60]} ({dm / 1e3:.1f})")
        if out:
            out.write(f"{b}\t{b * BIN:.1f}\t{busy[b] / 1e3:.3f}\t{cnt[b]}\t{dom}\t{dm / 1e3:.3f}\n")
    if out:
        out.close()


if __name__ == "__main__":
    main()
