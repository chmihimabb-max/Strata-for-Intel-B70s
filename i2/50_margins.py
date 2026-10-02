#!/usr/bin/env python3
"""I2: is the divergence only where the ORACLE's own choice was a coin flip?

    python3 i2/50_margins.py <oracle-probs.json> <engine-out.txt> [tag]

Only positions up to and including the first divergence are comparable (after it the two streams have
different contexts), so per prompt it reports:
  * how many leading tokens agreed,
  * the oracle's top1-top2 logprob margin AT the divergent index,
  * the smallest margin among the AGREED positions, i.e. was the divergence at the least decisive point.

A margin is log P(top1) - log P(top2) in nats; exp(margin) is the odds ratio between the oracle's first
and second choice.
"""
import json
import math
import pathlib
import sys


def parse_engine(path):
    streams, cur = [], []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("T "):
            cur.append(int(line[2:]))
        elif line.startswith("DONE"):
            streams.append(cur)
            cur = []
    return streams


def margin_at(cp, i):
    pr = cp[i]
    top = pr.get("top_logprobs") or pr.get("probs") or []
    if len(top) < 2:
        return None, None, None
    return (top[0].get("logprob", 0.0) - top[1].get("logprob", 0.0), top[0]["id"], top[1]["id"])


probs = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
eng_path = sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else pathlib.Path(eng_path).stem
eng = parse_engine(eng_path)
order = list(probs["prompts"].keys())

report = {"tag": tag, "oracle": sys.argv[1], "engine": eng_path, "prompts": {}, "total_compared": 0,
          "total_agreed": 0}
print("== per prompt (only the comparable prefix: up to the first divergence) ==")
for pi, name in enumerate(order):
    p = probs["prompts"][name]
    cp = p["completion_probabilities"]
    gold = p["gen_ids"] if p.get("gen_ids") else None
    e = eng[pi] if pi < len(eng) else []
    n = min(len(cp), len(e), len(gold) if gold else 10 ** 9)
    div = next((i for i in range(n) if gold[i] != e[i]), None)
    compared = (div + 1) if div is not None else n
    margins = [margin_at(cp, i)[0] for i in range(n)]
    agreed_margins = [m for i, m in enumerate(margins) if m is not None and i < compared - (1 if div is not None
                                                                                           else 0)]
    m_div, o1, o2 = (margin_at(cp, div) if div is not None else (None, None, None))
    # the margin of the position that diverged is the one to compare against the agreed ones
    agreed_min = min(agreed_margins) if agreed_margins else float("nan")
    entry = {"prompt": name, "compared": compared, "first_divergence": div,
             "oracle_top1_at_div": o1, "oracle_top2_at_div": o2, "margin_at_div": m_div,
             "min_margin_among_agreed": agreed_min,
             "median_margin_among_agreed": sorted(agreed_margins)[len(agreed_margins) // 2] if agreed_margins else None,
             "engine_token_at_div": (e[div] if div is not None and div < len(e) else None),
             "agreed_positions_with_margin_lt_div": sum(1 for m in agreed_margins
                                                        if m_div is not None and m < m_div)}
    report["prompts"][name] = entry
    report["total_compared"] += compared
    report["total_agreed"] += compared - (1 if div is not None else 0)
    if div is None:
        print("-- %-10s %3d/%3d agreed, no divergence; smallest oracle margin seen %.4f nats (median %.4f)"
              % (name, compared, compared, agreed_min,
                 sorted(agreed_margins)[len(agreed_margins) // 2] if agreed_margins else float("nan")))
    else:
        print("-- %-10s %3d agreed then DIVERGE at idx %d: oracle chose %s over %s by %.4f nats "
              "(odds %.2f:1) -> engine chose %s" % (name, div, div, o1, o2, m_div, math.exp(m_div),
                                                    entry["engine_token_at_div"]))
        print("            smallest oracle margin among the %d agreed positions: %.4f nats (median %.4f); "
              "agreed positions with a SMALLER margin than the divergence: %d"
              % (div, agreed_min, entry["median_margin_among_agreed"], entry["agreed_positions_with_margin_lt_div"]))
print()
print("== totals: %d/%d compared tokens agreed (%.2f%%), %d divergences ==" % (
    report["total_agreed"], report["total_compared"], 100.0 * report["total_agreed"] / report["total_compared"],
    len(order) - sum(1 for p in report["prompts"].values() if p["first_divergence"] is None)))
out = pathlib.Path(eng_path).with_name(pathlib.Path(eng_path).stem + "-margins.json")
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
print("wrote", out)
