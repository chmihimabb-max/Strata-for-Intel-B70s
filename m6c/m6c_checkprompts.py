#!/usr/bin/env python3
"""M6c: print the head, the tail and the needle neighbourhood of every generated prompt, so the claim
"the answer is buried at 50% of a 256K context and the question is the last thing the model sees" is
inspectable rather than asserted.

usage: /usr/bin/python3 m6c/m6c_checkprompts.py
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path("/home/michael/strata-xpu/strata")
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST  # noqa: E402

PACK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack")
PDIR = pathlib.Path("/home/michael/strata-xpu/m6c/prompts")


def load_tok() -> ST.Tokenizer:
    t = PACK / "tokenizer"
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    merges = [m for m in (t / "merges.txt").read_text(encoding="utf-8").split("\n") if m.strip()]
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def main() -> int:
    tok = load_tok()
    man = json.loads((PDIR / "prompt-manifest.json").read_text())
    for p in man["prompts"]:
        ids = [int(x) for x in pathlib.Path(p["file"]).read_text().split(",")]
        assert len(ids) == p["prompt_tokens"], (p["file"], len(ids), p["prompt_tokens"])
        print("=" * 100)
        print(f"{pathlib.Path(p['file']).name}  {len(ids)} tokens  kind={p['kind']}")
        print(f"  head    : {tok.decode(ids[:12])!r}")
        print(f"  tail    : {tok.decode(ids[-30:])!r}")
        if p["kind"] == "needle":
            a = p["needle_at"]
            print(f"  needle @{a} ({100.0 * a / len(ids):.1f}%) : ...{tok.decode(ids[a - 20:a + 20])!r}...")
            print(f"  question: {tok.decode(p['question_ids'][:60])!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
