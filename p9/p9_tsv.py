#!/usr/bin/env python3
"""P9: load a device-event TSV into numpy arrays, one card at a time, cached as .npz.

usage: /usr/bin/python3 p9/p9_tsv.py <dev.tsv> [--pid N] [--from-ts T] [--to-ts T] [--out X.npz] [--summary]

Cache: <out> holds ts (float64, µs), dur (float64, µs), tag (int32 index into tags), tags (list of str),
pid.  Later steps (p9_window.py) read the cache instead of re-parsing 100-500 MB of TSV.
"""
import sys

import numpy as np


def load(path, pid=None, from_ts=None, to_ts=None):
    pid = None if pid is None else str(pid)
    ts = []
    dur = []
    tag = []
    tags = []
    tidx = {}
    with open(path, errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 4)
            if len(p) < 5:
                continue
            if pid is not None and p[0] != pid:
                continue
            t = float(p[2])
            if from_ts is not None and t < from_ts:
                continue
            if to_ts is not None and t > to_ts:
                continue
            i = tidx.get(p[4])
            if i is None:
                i = len(tags)
                tidx[p[4]] = i
                tags.append(p[4])
            ts.append(t)
            dur.append(float(p[3]))
            tag.append(i)
    ts = np.asarray(ts, dtype=np.float64)
    dur = np.asarray(dur, dtype=np.float64)
    tag = np.asarray(tag, dtype=np.int32)
    o = np.argsort(ts, kind="stable")
    return ts[o], dur[o], tag[o], tags


def main():
    path = sys.argv[1]
    pid = None
    if "--pid" in sys.argv:
        pid = sys.argv[sys.argv.index("--pid") + 1]
    f = t = None
    if "--from-ts" in sys.argv:
        f = float(sys.argv[sys.argv.index("--from-ts") + 1])
    if "--to-ts" in sys.argv:
        t = float(sys.argv[sys.argv.index("--to-ts") + 1])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else path + ".npz"
    ts, dur, tag, tags = load(path, pid, f, t)
    np.savez_compressed(out, ts=ts, dur=dur, tag=tag, tags=np.array(tags, dtype=object))
    span = (ts[-1] - ts[0]) / 1e6 if len(ts) else 0.0
    print(f"{path} pid={pid}: {len(ts):,} rows, {len(tags)} tags, span {span:.3f} s, "
          f"busy {dur.sum() / 1e6:.1f} ms -> {out}")
    if "--summary" in sys.argv:
        tot = {}
        cnt = {}
        for i, nm in enumerate(tags):
            m = tag == i
            tot[nm] = dur[m].sum()
            cnt[nm] = int(m.sum())
        for nm, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {tot[nm] / 1e3:>11.2f} ms n={cnt[nm]:>9,}  {nm[:100]}")


if __name__ == "__main__":
    main()
