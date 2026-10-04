#!/usr/bin/env python3
"""S5: build the P1 probe prompt - the S4 4K prompt with ONE unused token appended at the end.

Why: `--turn-token` splits the prompt read at the LAST occurrence of that id (generate.cpp:5310-5312), so with a
token that appears exactly once, at position n-1, the read is a single segment AND a checkpoint is still saved at
n-1.  That separates "a checkpoint was saved" from "the read was split".

usage: s5_p1.py --gen GENFILE --out GENFILE_P1 [--report JSON]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

R = pathlib.Path("/home/michael/strata-xpu")
SRC = R / "strata"
TKDIR = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack/tokenizer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    o = ap.parse_args()

    parts = pathlib.Path(o.gen).read_text().split()
    assert parts[0] == "GEN", parts[0]
    max_new = parts[1]
    ids = [int(x) for x in parts[2].split(",")]
    vocab = json.loads((TKDIR / "vocab.json").read_text(encoding="utf-8"))
    present = set(ids)
    rare = next(i for i in range(200, len(vocab)) if i not in present and str(i) not in ("",))
    p1 = ids + [rare]
    pathlib.Path(o.out).write_text(f"GEN {max_new} " + ",".join(str(t) for t in p1) + "\n", encoding="utf-8")
    rep = {"gen": o.gen, "out": o.out, "n_base": len(ids), "n_p1": len(p1), "rare_id": rare,
           "rare_token": [t for t, i in vocab.items() if i == rare][:1], "turn_at": len(p1) - 1}
    if o.report:
        pathlib.Path(o.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
