#!/usr/bin/env python3
"""P10: the id-list prompt as TEXT, for the arms that must go through serve.server (which takes OpenAI chat text).

The depth-curve prompts in ~/strata-xpu/m6c/prompts are comma-separated token id lists (the engine's own serve
protocol takes ids).  serve.server only accepts text, so for the two-instance / concurrency arms the same prompt
is decoded through the pack's own tokenizer and re-encoded, and the round trip is checked and printed - an arm
that silently tokenises to a different length would not be like-for-like.

usage: /usr/bin/python3 p10/p10_prompt_text.py <ctx> [out_dir]
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path("/home/michael/strata-xpu/strata")
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST  # noqa: E402

PACK_TK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack/tokenizer")
PROMPTS = pathlib.Path("/home/michael/strata-xpu/m6c/prompts")


def load_tokenizer() -> ST.Tokenizer:
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
    ctx = int(sys.argv[1])
    out_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("/home/michael/strata-xpu/p10/prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = PROMPTS / f"prompt-ctx{ctx}.txt"
    if not src.exists():
        src = PROMPTS / f"prompt-needle-ctx{ctx}.txt"
    ids = [int(x) for x in src.read_text().strip().rstrip(",").split(",") if x.strip()]
    tk = load_tokenizer()
    text = tk.decode(ids)
    back = tk.encode(text, parse_special=True)
    out = out_dir / f"prompt-ctx{ctx}-text.txt"
    out.write_text(text, encoding="utf-8")
    # A chat-ready form for the serve.server arms: the same document and the same review question without the
    # prompt's own <|im_start|>/<|im_end|> markup, which the server adds back through the model's chat template
    # (sending the marked-up text inside a user message would nest the markup and is not the prompt of record).
    doc = text
    for opener in ("<|im_start|>system\n", "<|im_start|>user\n"):
        if opener in doc:
            doc = doc.split(opener, 1)[1]
    for closer in ("<|im_end|>\n<|im_start|>assistant\n", "<|im_start|>assistant\n"):
        if closer in doc:
            doc = doc.rsplit(closer, 1)[0]
    doc = doc.replace("<|im_end|>", "").strip()
    docout = out_dir / f"prompt-ctx{ctx}-user.txt"
    docout.write_text(doc, encoding="utf-8")
    doc_ids = tk.encode(doc, parse_special=True)
    print(f"[prompt-text] chat-ready form: {len(doc)} chars -> {len(doc_ids)} ids (the server adds its template)")
    print(f"[prompt-text] head {doc[:80]!r}")
    print(f"[prompt-text] tail {doc[-120:]!r}")
    same = back == ids
    print(f"[prompt-text] {src.name}: {len(ids)} ids -> {len(text)} chars -> re-encode {len(back)} ids; "
          f"identical={same}")
    if not same:
        first = next((i for i, (a, b) in enumerate(zip(ids, back)) if a != b), None)
        print(f"[prompt-text] first difference at index {first}: {ids[first:first+6]} vs {back[first:first+6]}"
              if first is not None else f"[prompt-text] prefix equal; lengths {len(ids)} vs {len(back)}")
    print(f"[prompt-text] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
