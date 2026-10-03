#!/usr/bin/env python3
"""P5: print the three streams side by side around an index, and diff the two engine arms against EACH OTHER.

    /usr/bin/python3 p5/p5_window.py <ctx> <lo> <hi> [<hi2> ...]

The engine-vs-engine diff is the sharper instrument for the question "what does the batched kernel's arithmetic
change": both arms run the same stack, so anything that differs between them is the attention kernel's own doing
(plus its effect on the continuation).
"""
from __future__ import annotations

import difflib
import json
import pathlib
import sys

CTX_TO_NAME = {4096: "4k", 32768: "32k", 131072: "128k"}
R = pathlib.Path("/home/michael/strata-xpu")


def arm(label: str):
    ctx = int(sys.argv[1])
    p = R / "p5/runs" / ("p5-%s-%s" % (CTX_TO_NAME[ctx], label)) / "out.txt"
    ids = []
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("T "):
            ids.append(int(line[2:].split()[0]))
        elif line.startswith("DONE"):
            break
    return ids


ctx = int(sys.argv[1])
oracle_json = R / "p5/oracle" / ("oracle-%s.json" % CTX_TO_NAME[ctx])
gold = []
if oracle_json.exists():
    gold = json.loads(oracle_json.read_text())["gen_ids"]
else:
    print("(no oracle stream for ctx %d -- %s does not exist; engine-vs-engine only)" % (ctx, oracle_json))
sh, ba = arm("shipped"), arm("batched")

print("== aligned diff of the two ENGINE arms against each other (no oracle involved)")
sm = difflib.SequenceMatcher(None, sh, ba, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    print("   %-7s shipped[%d:%d] %s   vs batched[%d:%d] %s"
          % (tag, i1, i2, sh[i1:i2][:8], j1, j2, ba[j1:j2][:8]))
print("   matched %d of %d/%d" % (sum(i2 - i1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t == "equal"),
                                  len(sh), len(ba)))

for rng in sys.argv[2:]:
    lo, _, hi = rng.partition(":")
    lo, hi = int(lo), int(hi)
    print("\n== positions %d..%d" % (lo, hi))
    print("   %-4s %s" % ("idx", " ".join("%6d" % i for i in range(lo, hi))))
    print("   %-4s %s" % ("ora", " ".join("%6s" % (gold[i] if i < len(gold) else "-") for i in range(lo, hi))))
    print("   %-4s %s" % ("shp", " ".join("%6s" % (sh[i] if i < len(sh) else "-") for i in range(lo, hi))))
    print("   %-4s %s" % ("bat", " ".join("%6s" % (ba[i] if i < len(ba) else "-") for i in range(lo, hi))))
