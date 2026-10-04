#!/usr/bin/env python3
"""D2b: the greedy ids md5 per arm, exactly as d2a/analyze.py and d2/d2_report.py compute it (md5 over the T
lines joined by newline), with the token count."""
import hashlib
import os
import sys

RUNS = "/home/michael/strata-xpu/strata/d2/runs"
CANON = {"4096": "66bf952d445e330c974c49e4f220b4b1", "32768": "d87373e84417dab60732a95eb47666a6"}
for tag in sys.argv[1:]:
    out = os.path.join(RUNS, tag, "out.txt")
    ts = [l for l in open(out, errors="replace").read().splitlines() if l.startswith("T ")]
    ids = hashlib.md5("\n".join(ts).encode()).hexdigest()[:32] if ts else "-"
    ctx = tag.rsplit("-", 1)[-1].replace("k", "000")
    want = CANON.get(ctx)
    mark = "  <-- canonical" if want and ids == want else ("  (no canonical for ctx %s)" % ctx if not want else "  <-- DIFFERS")
    print(f"  {tag:24s} {len(ts):4d} tokens  ids {ids}{mark}")
