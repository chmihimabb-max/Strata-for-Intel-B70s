#!/usr/bin/env python3
"""P5: the oracle against ITSELF -- the same prompt, same tree, same everything but -ub (2048 vs 512).

This is the measurement of the oracle's own implementation band: if the oracle's -ub 512 arm emits the token our
engine emitted at the position where our engine diverged from the -ub 2048 arm, then that divergence is not
evidence about our arithmetic, it is the oracle's own indecision at that position.

    /usr/bin/python3 p5/p5_oracle_ub_compare.py <ctx>
"""
from __future__ import annotations

import difflib
import json
import pathlib
import sys

CTX_TO_NAME = {4096: "4k", 32768: "32k", 131072: "128k"}
R = pathlib.Path("/home/michael/strata-xpu")


def load(name: str):
    p = R / "p5/oracle" / name
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    cp = j.get("completion_probabilities") or []
    marg = []
    for m in cp:
        top = m.get("top_logprobs") or m.get("probs") or []
        marg.append(float(top[0]["logprob"]) - float(top[1]["logprob"]) if len(top) > 1 else None)
    return {"ids": j["gen_ids"], "margins": marg, "timings": j.get("timings") or {}}


ctx = int(sys.argv[1])
c = CTX_TO_NAME[ctx]
a = load("oracle-%s-ub512.json" % c)      # perturbed instantiation
b = load("oracle-%s.json" % c)            # the reference instantiation (n_probs margins come from here)
if a is None or b is None:
    print("ctx %d: need both oracle-%s.json and oracle-%s-ub512.json" % (ctx, c, c))
    raise SystemExit(0)

print("== ctx %d: oracle -ub 512 (%d ids, prefill %.1f tok/s) vs oracle -ub 2048 (%d ids, prefill %.1f tok/s)"
      % (ctx, len(a["ids"]), a["timings"].get("prompt_per_second") or 0, len(b["ids"]),
         b["timings"].get("prompt_per_second") or 0))
sm = difflib.SequenceMatcher(None, b["ids"], a["ids"], autojunk=False)
first = None
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    m = b["margins"][i1] if i1 < len(b["margins"]) else None
    print("   %-7s ub2048[%d:%d]%s vs ub512[%d:%d]%s   ub2048 margin at that position: %s nats"
          % (tag, i1, i2, b["ids"][i1:i2][:6], j1, j2, a["ids"][j1:j2][:6],
             ("%.4f" % m) if m is not None else "n/a"))
    if first is None:
        first = (i1, i2, j1, j2)
matched = sum(i2 - i1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t == "equal")
print("   matched %d/%d positions" % (matched, max(len(a["ids"]), len(b["ids"]))))
if first:
    i1, i2, j1, j2 = first
    print("   FIRST difference at ub-2048 position %d (margin %.4f nats): oracle chose %s, the -ub 512 oracle "
          "chose %s" % (i1, b["margins"][i1] if i1 < len(b["margins"]) else float("nan"),
                        b["ids"][i1:i1 + 3], a["ids"][j1:j1 + 3]))
