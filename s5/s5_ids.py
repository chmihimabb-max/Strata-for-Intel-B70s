#!/usr/bin/env python3
"""S5 (card t_aae723be): build the DRIVER's GEN line for the very prompt an S4 served arm used.

The served path's ids are built by the SERVER, not by the tokenizer alone: `Service.prepare`
(serve/server.py:1141) renders the chat template over the request's messages and encodes THAT with the pack's
tokenizer.  This script does exactly those two steps, in the same order and with the same calls, so the GEN line
it writes carries the same ids the server put on the engine's stdin (the engine's own `prompt N tokens` line then
has to match the S4 arm's N - that is the check).

usage: s5_ids.py --prompt TEXTFILE --out GENFILE [--max-new 64] [--report JSON]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

R = pathlib.Path("/home/michael/strata-xpu")
SRC = R / "strata"
CFG = json.loads((SRC / "strata-sycl-iq3s.json").read_text(encoding="utf-8-sig"))
TKDIR = pathlib.Path(CFG["tokenizer"])

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402


def load_tokenizer():
    """The server's own load (serve/server.py:2475-2483) - no `pre`/`special_ids` extras."""
    vocab = json.loads((TKDIR / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TKDIR / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((TKDIR / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--report", default="")
    o = ap.parse_args()

    text = pathlib.Path(o.prompt).read_text(encoding="utf-8")
    # p10_client.py's body, verbatim: one user message, temperature 0, no tools, no chat_template_kwargs
    req = {"model": "strata", "messages": [{"role": "user", "content": text}],
           "max_tokens": o.max_new, "temperature": 0, "top_p": 1, "stream": False}
    messages, tools, kw = openai_to_messages(req)
    tpl = TKDIR / "chat_template.jinja"
    template = ChatTemplate(tpl if tpl.exists() else SRC / "serve" / "chat_template.jinja")
    prompt = template.render(messages, tools=tools, **kw)
    tok = load_tokenizer()
    ids = tok.encode(prompt, parse_special=True)
    # sampling_keys({temperature: 0, top_p: 1}) = "" (serve/server.py:343-387): greedy, no keys - so the engine
    # line is exactly `GEN <max_new> <ids>`
    line = f"GEN {o.max_new} " + ",".join(str(int(t)) for t in ids) + "\n"
    pathlib.Path(o.out).write_text(line, encoding="utf-8")
    rep = {"gen_line": o.out, "prompt_tokens": len(ids), "prompt_text_md5": hashlib.md5(prompt.encode()).hexdigest(),
           "max_new": o.max_new, "template": str(tpl if tpl.exists() else SRC / "serve" / "chat_template.jinja"),
           "rendered_tail": prompt[-200:], "rendered_head": prompt[:200], "kwargs": kw,
           "tokdir": str(TKDIR)}
    pathlib.Path(o.out + ".ids.txt").write_text(",".join(str(int(t)) for t in ids), encoding="utf-8")
    if o.report:
        pathlib.Path(o.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))


if __name__ == "__main__":
    sys.exit(main())
