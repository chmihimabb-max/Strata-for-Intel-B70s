#!/usr/bin/env python3
"""P5: assemble the per-length three-way table (oracle vs each engine arm) out of the compare-*.json files, plus
the oracle-band cross-check.

    /usr/bin/python3 p5/p5_table.py [compare-4.json compare-32.json ...]

Writes p5/TABLE.md and prints the same table.  Columns: each arm's prefill/decode next to its agreement with the
oracle, the first divergent index, the nats margin there, whether that margin is above the oracle's OWN
implementation band (0.106 nats, I2's -ub measurement), and whether the oracle's -ub 512 instantiation emits our
token AT that position (the band check: the oracle agreeing with us under a different prompt batch size means the
position is one the oracle cannot decide).
"""
from __future__ import annotations

import difflib
import json
import pathlib
import sys

CTX_TO_NAME = {4096: "4k", 32768: "32k", 131072: "128k"}
R = pathlib.Path("/home/michael/strata-xpu")


def arm_ids(ctx: int, label: str):
    p = R / "p5/runs" / ("p5-%s-%s" % (CTX_TO_NAME[ctx], label)) / "out.txt"
    if not p.exists():
        return None
    ids = []
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("T "):
            ids.append(int(line[2:].split()[0]))
        elif line.startswith("DONE"):
            break
    return ids


def ub512_agrees(ctx: int, label: str, d: int):
    """Does the -ub 512 oracle emit our token at our index d (aligned)?  None if that arm/json is absent."""
    p = R / "p5/oracle" / ("oracle-%s-ub512.json" % CTX_TO_NAME[ctx])
    ours = arm_ids(ctx, label)
    if not p.exists() or ours is None or d is None:
        return None
    theirs = json.loads(p.read_text())["gen_ids"]
    sm = difflib.SequenceMatcher(None, ours, theirs, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= d < i2:
            return tag == "equal"
    return None


files = sys.argv[1:] or sorted(str(p) for p in pathlib.Path("p5").glob("compare-*.json"))
rows = [json.loads(pathlib.Path(f).read_text(encoding="utf-8")) for f in files]
rows.sort(key=lambda r: r["ctx"])

out = ["# P5 — the oracle-attributed A/B (per length)\n",
       "Prompts are P2's own arms' prompt files; oracle = llama.cpp-SYCL `qwen4exp` on the same IQ3_S file, f16 KV,",
       "-ub 2048 unless stated; both sides greedy, same prompt ids, 256 requested at every length.",
       "`margin` is the oracle's own top1-top2 log-prob gap at that position; `decided?` asks whether it clears the",
       "oracle's own -ub band (0.106 nats, I2); `ub-512 agrees?` asks whether the oracle's -ub 512 instantiation",
       "emits OUR token at that position.\n"]

for r in rows:
    o = r["oracle_timings"]
    ms = r["oracle_margin_stats"]
    out.append("\n## ctx %d  (prompt %d ids)\n" % (r["ctx"], r["oracle_prompt_tokens"]))
    out.append("oracle: %d generated, prefill %s tok/s, decode %s tok/s; its own top1-top2 margins: "
               "min %.4f / median %.4f / max %.4f nats; %s of %d positions below 0.15 nats\n"
               % (r["oracle_gen"], o.get("prompt_tok_s"), o.get("predicted_tok_s"),
                  ms["min"], ms["median"], ms["max"], ms["below"].get("0.15"), ms["n"]))
    out.append("| arm | prefill tok/s | decode tok/s | gen | first div | @div oracle vs engine | margin (nats) "
               "| decided? | ub-512 agrees? | prefix agreed | aligned matched |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, a in r["arms"].items():
        st = a["serve_stats"] or {}
        div = a["first_divergence"]
        ub = ub512_agrees(r["ctx"], label, div)
        if "ub512" in str(r.get("oracle", "")):
            ub = None   # this row's reference IS the -ub 512 instantiation; the cross-check would be circular
        # "decided" = the oracle's own margin at that position clears its own implementation band AND its other
        # instantiation does not itself emit our token there.  A position where the -ub 512 oracle emits our token
        # is one the oracle cannot decide, whatever its margin was.
        decided = None if div is None else (bool(a["first_div_decided"]) and ub is not True)
        out.append("| %s | %s | %s | %d | %s | %s vs %s | %s | %s | %s | %s/%s (%.1f%%) | %d/%d (%.1f%%) |"
                   % (label, ("%.1f" % st["prefill_tok_s"]) if st else "n/a",
                      ("%.2f" % st["decode_tok_s"]) if st else "n/a",
                      a["gen"], ("%d" % div) if div is not None else "**none**",
                      a["oracle_top1_at_div"] if div is not None else "-",
                      a["engine_id_at_div"] if div is not None else "-",
                      ("%.4f" % a["margin_at_div"]) if a["margin_at_div"] is not None else "n/a",
                      {True: "**YES**", False: "no", None: "-"}[decided],
                      {True: "YES", False: "no", None: "n/a"}[ub],
                      a["agreed_in_prefix"], a["compared_prefix"], a["prefix_agreement_pct"] or 0.0,
                      a["aligned_matched"], max(a["gen"], a["oracle_gen"]), a["aligned_agreement_pct"] or 0.0))
    out.append("")

pathlib.Path("p5/TABLE.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print("\nwrote p5/TABLE.md")
