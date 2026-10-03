#!/usr/bin/env python3
"""P5: the three-way comparison -- the independent oracle, and each engine arm, per generated position.

    /usr/bin/python3 p5/p5_compare.py <ctx> <oracle.json> <label>:<arm-out.txt> [<label>:<out.txt> ...]

For every arm this reports BOTH views, because they answer different questions:

* the **prefix view** (I2's): the tokens up to the first divergence, which is where a token stream is still the
  same comparison at all.  This is the number to quote for "how much of the stream does this arm get right".
* the **alignment view** (difflib): after an insertion the prefix view charges every later position as wrong even
  when the streams agree, so the aligned view counts the positions that actually *disagree arithmetically*.

Every divergence carries the oracle's own top1-top2 logprob margin at that position (nats).  A flip where the
oracle's own top-2 sits within its own decision noise is not the same evidence as a flip at 3 nats, so the
thresholded counts at 0.05 / 0.10 / 0.15 / 0.50 nats are reported next to the raw ones.

The engine's own prefill/decode rates are read from its --stats line, so speed and correctness come out of the
same run.
"""
from __future__ import annotations

import difflib
import json
import math
import pathlib
import re
import sys

THRESHOLDS = [0.05, 0.10, 0.15, 0.50]
SERVE_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), "
                      r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)")


def parse_arm(path: pathlib.Path):
    """T <id> lines, split per request (a DONE closes the stream), plus the last request's serve stats line."""
    streams, cur = [], []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("T "):
            cur.append(int(line[2:].split()[0]))
        elif line.startswith("DONE"):
            streams.append(cur)
            cur = []
    return streams


def serve_stats(err_path: pathlib.Path):
    if not err_path.exists():
        return None
    hits = list(SERVE_RE.finditer(err_path.read_text(errors="replace")))
    if not hits:
        return None
    g = hits[-1].groups()
    return {"prompt_tokens": int(g[0]), "reused": int(g[1]), "read_tokens": int(g[2]),
            "prefill_ms": float(g[3]), "prefill_tok_s": float(g[4]),
            "gen_tokens": int(g[5]), "decode_ms": float(g[6]), "decode_tok_s": float(g[7])}


def margin(m):
    """top1 - top2 logprob, and the two ids (None if the oracle reported < 2 candidates)."""
    if m is None:
        return None, None, None
    top = m.get("top_logprobs") or m.get("probs") or []
    if len(top) < 2:
        return None, None, None
    return (float(top[0].get("logprob", 0.0)) - float(top[1].get("logprob", 0.0)),
            top[0].get("id"), (top[1].get("id") if len(top) > 1 else None))


def main() -> int:
    ctx = int(sys.argv[1])
    oracle_path = pathlib.Path(sys.argv[2])
    arms = []
    for spec in sys.argv[3:]:
        label, _, out = spec.partition(":")
        arms.append((label, pathlib.Path(out)))

    orc = json.loads(oracle_path.read_text(encoding="utf-8"))
    gold = orc["gen_ids"]
    cp = orc["completion_probabilities"] or []
    tm = orc.get("timings") or {}
    margins = [margin(cp[i] if i < len(cp) else None)[0] for i in range(len(gold))]
    known = [m for m in margins if m is not None]

    report = {"ctx": ctx, "oracle": str(oracle_path), "oracle_prompt_file": orc["prompt_file"],
              "oracle_prompt_tokens": orc["prompt_tokens"], "oracle_gen": len(gold),
              "oracle_timings": {"prompt_n": tm.get("prompt_n"), "prompt_tok_s": tm.get("prompt_per_second"),
                                 "predicted_n": tm.get("predicted_n"), "predicted_tok_s": tm.get("predicted_per_second")},
              "oracle_margin_stats": {"n": len(known), "min": min(known), "median": sorted(known)[len(known) // 2],
                                      "max": max(known),
                                      "below": {str(t): sum(1 for m in known if m < t) for t in THRESHOLDS}},
              "arms": {}}

    print("=" * 118)
    print("P5 ctx %d  oracle=%s (%d prompt ids, %d generated, prefill %s tok/s, decode %s tok/s)"
          % (ctx, pathlib.Path(oracle_path).name, orc["prompt_tokens"], len(gold),
             tm.get("prompt_per_second"), tm.get("predicted_per_second")))
    print("  oracle's own decision margins over its %d positions: min %.4f / median %.4f / max %.4f nats; "
          "below %s" % (len(known), min(known), sorted(known)[len(known) // 2], max(known),
                        report["oracle_margin_stats"]["below"]))

    for label, out in arms:
        streams = parse_arm(out)
        eng = streams[0] if streams else []
        n = min(len(gold), len(eng))
        div = next((i for i in range(n) if gold[i] != eng[i]), None)
        compared = (div + 1) if div is not None else n
        agreed = compared - (1 if div is not None else 0)
        m_div, o1, o2 = margin(cp[div]) if (div is not None and div < len(cp)) else (None, None, None)

        sm = difflib.SequenceMatcher(None, gold, eng, autojunk=False)
        blocks = []
        matched = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                matched += i2 - i1
                continue
            bm = []
            for i in range(i1, min(i2, len(margins))):
                mv = margins[i]
                if mv is not None:
                    bm.append(mv)
            blocks.append({"tag": tag, "oracle_range": [i1, i2], "engine_range": [j1, j2],
                           "oracle_ids": gold[i1:i2][:6], "engine_ids": eng[j1:j2][:6],
                           "margin_at_block_start": (margins[i1] if i1 < len(margins) else None),
                           "max_margin_in_block": (max(bm) if bm else None),
                           "oracle_top1_top2_at_start": [o1, o2] if i1 == div else None})

        st = serve_stats(out.with_name("err.txt"))
        entry = {"label": label, "out_file": str(out), "gen": len(eng), "oracle_gen": len(gold),
                 "first_divergence": div, "compared_prefix": compared, "agreed_in_prefix": agreed,
                 "prefix_agreement_pct": (100.0 * agreed / compared) if compared else None,
                 "margin_at_div": m_div, "oracle_top1_at_div": o1 if div is not None else None,
                 "oracle_top2_at_div": o2 if div is not None else None,
                 "engine_id_at_div": (eng[div] if div is not None and div < len(eng) else None),
                 "aligned_matched": matched,
                 "aligned_agreement_pct": (100.0 * matched / max(len(gold), len(eng))) if eng else None,
                 "diff_blocks": blocks, "serve_stats": st}
        for t in THRESHOLDS:
            entry["blocks_with_margin_ge_%.2f" % t] = sum(
                1 for b in blocks if (b["max_margin_in_block"] or -1) >= t)
        report["arms"][label] = entry

        print("-" * 118)
        print("ARM %-22s generated %d" % (label, len(eng)))
        if st:
            print("   engine prefill %.1f tok/s (%d prompt tokens, %s read), decode %.1f tok/s"
                  % (st["prefill_tok_s"], st["prompt_tokens"], st["read_tokens"], st["decode_tok_s"]))
        if div is None:
            print("   PREFIX: %d/%d agreed, NO divergence in the first %d positions (oracle margins: min %.4f)"
                  % (agreed, compared, n, min(m for m in margins[:n] if m is not None)))
        else:
            odds = (math.exp(float(m_div)) if m_div is not None else float("nan"))
            print("   PREFIX: %d/%d agreed, first divergence idx %d (%.2f%%): oracle chose %s over %s by "
                  "%s nats (odds %s:1) -> engine chose %s"
                  % (agreed, compared, div, 100.0 * agreed / compared, o1, o2,
                     ("%.4f" % m_div) if m_div is not None else "?", odds,
                     entry["engine_id_at_div"]))
            print("           smallest oracle margin among the agreed positions: %.4f nats"
                  % min(m for m in margins[:div] if m is not None))
        print("   ALIGNED: %d/%d positions match (%.2f%%); %d differing block(s):"
              % (matched, max(len(gold), len(eng)), entry["aligned_agreement_pct"], len(blocks)))
        for b in blocks:
            print("      %-7s oracle[%d:%d] vs engine[%d:%d]  margin@start %s  max-in-block %s  oracle %s vs engine %s"
                  % (b["tag"], b["oracle_range"][0], b["oracle_range"][1], b["engine_range"][0],
                     b["engine_range"][1],
                     ("%.4f" % b["margin_at_block_start"]) if b["margin_at_block_start"] is not None else "n/a",
                     ("%.4f" % b["max_margin_in_block"]) if b["max_margin_in_block"] is not None else "n/a",
                     b["oracle_ids"], b["engine_ids"]))
        print("   blocks whose own oracle margin >= 0.05/0.10/0.15/0.50 nats: %s"
              % [entry["blocks_with_margin_ge_%.2f" % t] for t in THRESHOLDS])

    out_json = pathlib.Path("p5") / ("compare-%d.json" % ctx)
    out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("=" * 118)
    print("wrote %s" % out_json.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
