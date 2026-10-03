#!/usr/bin/env python3
"""D1 (card t_d9ffcf38): the spec-width sweep, with the fit that answers the card's hypothesis.

The card's hypothesis: if ~71% of a window is per-window fixed cost, banking more accepted tokens per window
divides that cost.  The counter-hypothesis: longer drafts lose acceptance faster than they add tokens, with the
ceiling 1/(1-p) at acceptance p.  Both are decidable from four numbers per arm, and this script reports them
and then fits ms/window = a + b * (tokens/window) per length: `a` is the fixed part the hypothesis is about and
`b` the marginal cost of one more verified token.  It also prints ms/token, which is what decode tok/s is.

usage: d1_sweep.py [--root /home/michael/strata-xpu/d1/runs] [--prefix d1-spec]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_report import one  # noqa: E402


def fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    # R^2
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/michael/strata-xpu/d1/runs")
    ap.add_argument("--prefix", default="d1-spec")
    ap.add_argument("--csv", default=None)
    o = ap.parse_args()
    tags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(o.root, o.prefix + "*")) if os.path.isdir(p))
    rows = []
    for t in tags:
        r = one(t, os.path.join(o.root, t))
        if r and r.get("tok_per_window"):
            rows.append(r)
    for ctx in sorted({r["ctx"] for r in rows}, key=lambda c: int(c)):
        print("== ctx %s ==" % ctx)
        print("%-18s %6s %9s %9s %9s %9s %10s %12s" % (
            "spec", "avgT", "tok/win", "ms/win", "ms/token", "dec tok/s", "acc rate", "vs best tok/s"))
        grp = [r for r in rows if r["ctx"] == ctx]
        best = max(r["decode_tok_s"] for r in grp)
        for r in sorted(grp, key=lambda r: float(r["spec"])):
            print("%-18s %6.2f %9.2f %9.2f %9.2f %9.2f %10.3f %11s" % (
                r["spec"], r["avg_T"], r["tok_per_window"], r["ms_window"],
                1000.0 / r["decode_tok_s"], r["decode_tok_s"], r["accept_rate"],
                "%+.1f%%" % (100.0 * (r["decode_tok_s"] / best - 1))))
        xs = [r["tok_per_window"] for r in grp]
        ys = [r["ms_window"] for r in grp]
        f = fit(xs, ys)
        if f:
            a, b, r2 = f
            print("    fit: ms/window = %.2f + %.2f * tokens/window   (R^2 %.4f)" % (a, b, r2))
            print("         -> the per-window FIXED part is %.2f ms; each verified token costs %.2f ms of window"
                  % (a, b))
            if a > 0 and b > 0:
                print("         -> at spec 4's %.2f tokens/window, fixed cost is %.1f%% of the window"
                      % (xs[len(xs) // 2], 100.0 * a / (a + b * xs[len(xs) // 2])))
            print("         -> the ceiling 1/(1-p): " + ", ".join(
                "p=%.3f -> %.2f tokens/window" % (r["accept_rate"], 1.0 / (1.0 - r["accept_rate"]))
                for r in sorted(grp, key=lambda r: float(r["spec"])) if r["accept_rate"] and r["accept_rate"] < 1))
        print()
    if o.csv:
        import csv as _csv
        with open(o.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["tag", "ctx", "spec", "spec_min_p", "mtp_max_t", "avg_T", "tok_per_window", "ms_window",
                        "decode_tok_s", "prefill_tok_s", "accepted", "offered", "accept_rate", "ids_md5"])
            for r in rows:
                w.writerow([r["tag"], r["ctx"], r["spec"], r["spec_min_p"], r["mtp_max_t"], r["avg_T"],
                            r["tok_per_window"], r["ms_window"], r["decode_tok_s"], r["prefill_tok_s"],
                            r["accepted"], r["offered"], r["accept_rate"], r["ids_md5"]])
        print("wrote %s" % o.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
