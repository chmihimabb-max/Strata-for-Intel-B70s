#!/usr/bin/env python3
"""S5: the prompt's tail ids - and, for each of them, where its LAST occurrence is (that is where `--turn-token`
puts the checkpoint boundary, generate.cpp:5310-5312).

usage: s5_tail.py [--gen GENFILE] [--n 8]
"""
from __future__ import annotations

import argparse
import pathlib
from collections import Counter

SRC = pathlib.Path("/home/michael/strata-xpu/strata")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=str(SRC / "s5" / "prompts" / "gen-ctx4096.txt"))
    ap.add_argument("--n", type=int, default=8)
    o = ap.parse_args()
    ids = [int(x) for x in pathlib.Path(o.gen).read_text().split()[2].split(",")]
    c = Counter(ids)
    print(f"n = {len(ids)}  (the read is [0, n-1); the read ends at n-1 = {len(ids) - 1})")
    print(f"{'index':>8} {'id':>8} {'count':>6} {'last index':>10}  -> turn_at if used as --turn-token")
    for i in range(len(ids) - o.n, len(ids)):
        last = max(j for j, x in enumerate(ids) if x == ids[i])
        print(f"{i:>8} {ids[i]:>8} {c[ids[i]]:>6} {last:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
