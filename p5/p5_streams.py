#!/usr/bin/env python3
"""P5: side-by-side windows of the oracle and every engine arm at one length.

    /usr/bin/python3 p5/p5_streams.py <ctx> <lo> <hi> [<lo2:hi2> ...]
"""
from __future__ import annotations

import json
import pathlib
import sys

CTX_TO_NAME = {4096: "4k", 32768: "32k", 131072: "128k"}
R = pathlib.Path("/home/michael/strata-xpu")


def stream(tag_file: pathlib.Path):
    ids = []
    for line in tag_file.read_text(errors="replace").splitlines():
        if line.startswith("T "):
            ids.append(int(line[2:].split()[0]))
        elif line.startswith("DONE"):
            break
    return ids


ctx = int(sys.argv[1])
name = CTX_TO_NAME[ctx]
gold = json.loads((R / "p5/oracle" / ("oracle-%s.json" % name)).read_text())["gen_ids"] if \
    (R / "p5/oracle" / ("oracle-%s.json" % name)).exists() else []
labels = [("ora", gold)]
for lab in ("shipped", "batched", "shipped-fp16", "batched-fp16"):
    p = R / "p5/runs" / ("p5-%s-%s" % (name, lab)) / "out.txt"
    if p.exists():
        labels.append((lab, stream(p)))

for rng in sys.argv[2:]:
    lo, _, hi = rng.partition(":")
    lo, hi = int(lo), int(hi)
    print("== ctx %d positions %d..%d" % (ctx, lo, hi))
    print("   %-14s %s" % ("idx", " ".join("%6d" % i for i in range(lo, hi))))
    for lab, ids in labels:
        print("   %-14s %s" % (lab, " ".join("%6s" % (ids[i] if i < len(ids) else "-") for i in range(lo, hi))))
    print()
