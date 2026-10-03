#!/usr/bin/env python3
"""S4 (card t_30d9ccfb): build a text prompt of an EXACT token count for the served path.

usage: s4_prompt.py <target_tokens> <out.txt> [--base FILE] [--mode exact|chars]

The prompt has to go through the *server*, so it must be TEXT (the server tokenizes it itself) - the id-list
prompts the driver rigs use are not usable here.  The count is exact: the pack's own tokenizer is the pure-python
one the P10 rig used (`tools/strata_tokenizer.py`, loaded the same way p10_prompt_text.py loads it), and the
script takes the longest prefix of the base document whose token count is <= the target, reporting both numbers.
The authoritative server-side count is the response's `usage.prompt_tokens`, which is this count plus the chat
template the server adds (a handful of tokens; the arm log prints both).

The base text is the p10 32K depth-curve prompt's chat-ready form (the same text P10 fired at the split through
serve.server), so an S4 arm at 32768 is comparable with P10's failing arm.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

R = pathlib.Path("/home/michael/strata-xpu")
SRC = R / "strata"
BASE = R / "p10" / "prompts" / "prompt-ctx32768-user.txt"
PACK_TK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack/tokenizer")
CHARS_PER_TOKEN = 105544 / 32277.0


def load_tokenizer():
    sys.path.insert(0, str(SRC / "tools"))
    import strata_tokenizer as ST  # noqa: PLC0415

    ids = json.loads((PACK_TK / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(ids)
    for tok, i in ids.items():
        tokens[i] = tok
    merges = (PACK_TK / "merges.txt").read_text(encoding="utf-8").splitlines()
    types = json.loads((PACK_TK / "token_type.json").read_text(encoding="utf-8"))
    cfg = json.loads((PACK_TK / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, pre=cfg.get("pre", "qwen35"),
                        special_ids=cfg.get("special_ids"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=int)
    ap.add_argument("out")
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--mode", default="exact", choices=["exact", "chars"])
    o = ap.parse_args()

    base = pathlib.Path(o.base).read_text(encoding="utf-8")
    if o.mode == "chars":
        chars = max(256, int(round(o.target * CHARS_PER_TOKEN)))
        text = base[:chars] if chars <= len(base) else base
        pathlib.Path(o.out).write_text(text, encoding="utf-8")
        print(json.dumps({"out": o.out, "target": o.target, "mode": "chars", "chars": len(text), "tokens": None}))
        return 0

    tk = load_tokenizer()
    if len(tk.encode(base, parse_special=True)) <= o.target:
        text, n = base, len(tk.encode(base, parse_special=True))
    else:
        lo, hi = 1, len(base)
        while lo < hi:                                   # longest prefix with <= target tokens
            mid = (lo + hi + 1) // 2
            if len(tk.encode(base[:mid], parse_special=True)) <= o.target:
                lo = mid
            else:
                hi = mid - 1
        text = base[:lo]
        n = len(tk.encode(text, parse_special=True))
    pathlib.Path(o.out).write_text(text, encoding="utf-8")
    print(json.dumps({"out": o.out, "target": o.target, "mode": "exact", "chars": len(text), "tokens": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
