#!/usr/bin/env python3
"""P5: aligned diff of any two arms (or an arm and the oracle) at one length.

    /usr/bin/python3 p5/p5_pairdiff.py <ctx> <labelA> <labelB>

label `ora` is the oracle's stream; anything else is an arm directory p5/runs/p5-<name>-<label>.
Each differing op is annotated with the ORACLE's own top1-top2 margin at the oracle positions it covers, so a
flip can be read as decisive or as a coin flip.
"""
from __future__ import annotations

import difflib
import json
import pathlib
import sys

CTX_TO_NAME = {4096: "4k", 32768: "32k", 131072: "128k"}
R = pathlib.Path("/home/michael/strata-xpu")


def load(ctx: int, label: str):
    name = CTX_TO_NAME[ctx]
    if label.startswith("json:"):
        p = pathlib.Path(label[5:])
        if not p.is_absolute():
            p = R / "p5/oracle" / p
        return json.loads(p.read_text())["gen_ids"]
    if label == "ora":
        p = R / "p5/oracle" / ("oracle-%s.json" % name)
        return json.loads(p.read_text())["gen_ids"] if p.exists() else []
    ids = []
    for line in (R / "p5/runs" / ("p5-%s-%s" % (name, label)) / "out.txt").read_text(errors="replace").splitlines():
        if line.startswith("T "):
            ids.append(int(line[2:].split()[0]))
        elif line.startswith("DONE"):
            break
    return ids


ctx = int(sys.argv[1])
la, lb = sys.argv[2], sys.argv[3]
orc_json = R / "p5/oracle" / ("oracle-%s.json" % CTX_TO_NAME[ctx])
margins = []
if orc_json.exists():
    cp = json.loads(orc_json.read_text())["completion_probabilities"] or []
    for m in cp:
        top = m.get("top_logprobs") or m.get("probs") or []
        margins.append(float(top[0]["logprob"]) - float(top[1]["logprob"]) if len(top) > 1 else None)
a, b = load(ctx, la), load(ctx, lb)
print("== ctx %d  %s (%d ids) vs %s (%d ids)" % (ctx, la, len(a), lb, len(b)))
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
first = None
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    if first is None:
        first = (tag, i1, i2, j1, j2)
    ms = [margins[i] for i in range(i1, min(i2, len(margins))) if margins[i] is not None]
    print("   %-7s %s[%d:%d]%s vs %s[%d:%d]%s  oracle-margin@start=%s max-in-block=%s"
          % (tag, la, i1, i2, a[i1:i2][:6], lb, j1, j2, b[j1:j2][:6],
             ("%.4f" % margins[i1]) if i1 < len(margins) and margins[i1] is not None else "n/a",
             ("%.4f" % max(ms)) if ms else "n/a"))
matched = sum(i2 - i1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t == "equal")
print("   matched %d/%d" % (matched, max(len(a), len(b))))
if first is not None:
    print("   FIRST difference: %s at %s[%d] vs %s[%d]" % (first[0], la, first[1], lb, first[3]))
