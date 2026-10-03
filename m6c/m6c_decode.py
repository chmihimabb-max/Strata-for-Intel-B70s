#!/usr/bin/env python3
"""M6c: decode a run's generated ids with the pack's own tokenizer and check the needle answer.

    /usr/bin/python3 m6c/m6c_decode.py RUN_DIR [...]

Writes <run>/answer.txt (one block per request: the raw T ids, the decoded text, and whether the buried
code appears) and prints the verdict.  The code is checked both exactly and loosely (the tokenizer can put
spaces or a stray newline inside it), because "it produced the code" and "it produced the code as one
token run" are different claims and only the loose one is about retrieval.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path("/home/michael/strata-xpu/strata")
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST  # noqa: E402

PACK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack")
ANSWER = "ZK-4471-QX"


def load_tok() -> ST.Tokenizer:
    t = PACK / "tokenizer"
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    merges = [m for m in (t / "merges.txt").read_text(encoding="utf-8").split("\n") if m.strip()]
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def requests_of(out: pathlib.Path) -> list[dict]:
    """Split the engine's stdout into requests: ids between a DONE and the previous DONE/start."""
    reqs: list[dict] = []
    cur: list[int] = []
    for line in out.read_text(errors="replace").splitlines():
        if line.startswith("T "):
            cur.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            p = line.split()
            reqs.append({"ids": cur, "generated": int(p[1]), "prompt_tokens": int(p[2]),
                         "prompt_ms": float(p[3]), "decode_ms": float(p[4]), "stop": p[5]})
            cur = []
    return reqs


def main() -> int:
    tok = load_tok()
    for arg in sys.argv[1:]:
        d = pathlib.Path(arg)
        reqs = requests_of(d / "out.txt")
        blocks, ok = [], 0
        for i, r in enumerate(reqs):
            text = tok.decode(r["ids"])
            loose = re.sub(r"[\s]+", "", text).upper()
            tgt = ANSWER.replace("-", "").upper()
            hit = ANSWER in text
            loose_hit = tgt in loose
            ok += 1 if loose_hit else 0
            blocks.append(f"request {i + 1}: {r['generated']} generated, {r['prompt_tokens']} prompt tokens, "
                          f"stop={r['stop']}\n  ids: {r['ids']}\n  text: {text!r}\n"
                          f"  exact {ANSWER}: {hit}   loose (case/space-insensitive): {loose_hit}\n")
        (d / "answer.txt").write_text("\n".join(blocks), encoding="utf-8")
        print("=" * 90)
        print(d.name, f"({len(reqs)} request(s))")
        for b in blocks:
            print(b)
        print(f"  -> needle retrieved in {ok} of {len(reqs)} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
