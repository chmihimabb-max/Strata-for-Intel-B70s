#!/usr/bin/env python3
"""P9: one command that turns an arm's chrome timeline into per-card numpy caches.

  /usr/bin/python3 p9/p9_prep.py <arm-tag>

Reads the first chrome timeline found in /home/michael/strata-xpu/p9/runs/<tag>/ (strata.<pid>.json),
learns the DEVICE pseudo-process ids from the trace's own metadata rows (they are per card, so nothing is
hardcoded), and writes:
  ~/strata-xpu/p9/dev/<tag>.dev.tsv          all device rows (tag-canonicalised)
  ~/strata-xpu/p9/dev/<tag>.c<N>.npz         card N's rows, ts-sorted, numpy arrays
  ~/strata-xpu/p9/dev/<tag>.pids.txt         the pid -> device-name map it found
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p9_extract          # noqa: E402
import p9_tsv              # noqa: E402

R = "/home/michael/strata-xpu"
DEV = f"{R}/p9/dev"


def main():
    tag = sys.argv[1]
    d = f"{R}/p9/runs/{tag}"
    traces = sorted(glob.glob(f"{d}/strata.*.json"))
    if not traces:
        print(f"no chrome timeline under {d}")
        return 1
    trace = max(traces, key=os.path.getsize)
    print(f"trace: {trace}  ({os.path.getsize(trace):,} B)")

    # the DEVICE pseudo-processes, from the trace's own metadata (pid -> "DEVICE<host>[card] name")
    names = {}
    with open(trace, errors="replace") as f:
        for line in f:
            if '"process_name"' not in line:
                continue
            try:
                e = json.loads(line.rstrip().rstrip(","))
            except Exception:
                continue
            nm = e.get("args", {}).get("name", "")
            if nm.startswith("DEVICE"):
                names[e["pid"]] = nm
    with open(f"{DEV}/{tag}.pids.txt", "w") as g:
        for p, nm in sorted(names.items()):
            g.write(f"{p}\t{nm}\n")
            print(f"  device pid {p}: {nm}")
    if not names:
        print("no DEVICE pseudo-process in the trace: nothing to attribute")
        return 2

    tsv = f"{DEV}/{tag}.dev.tsv"
    os.makedirs(DEV, exist_ok=True)
    sys.argv = ["p9_extract.py", trace, tsv]
    p9_extract.main()

    card = 0
    for pid in sorted(names):
        out = f"{DEV}/{tag}.c{card}.npz"
        ts, dur, tg, tgs = p9_tsv.load(tsv, pid=pid)
        import numpy as np
        np.savez_compressed(out, ts=ts, dur=dur, tag=tg, tags=np.array(tgs, dtype=object))
        span = (ts[-1] - ts[0]) / 1e6 if len(ts) else 0
        print(f"  {out}: {len(ts):,} rows, span {span:.3f} s, busy {dur.sum() / 1e6:.1f} ms")
        card += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
