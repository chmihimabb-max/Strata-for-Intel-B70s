#!/usr/bin/env python3
"""S5: compare the driver arms' greedy `T` ids - md5, first divergence, and (optionally) the ids detokenized
against a SERVED arm's own text, so the id-level rig is tied back to the server's response.

usage:
  s5_cmp.py table TAG...
  s5_cmp.py div TAG_A TAG_B
  s5_cmp.py detok TAG [--served JSONFILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

R = pathlib.Path("/home/michael/strata-xpu")
SRC = R / "strata"
sys.path.insert(0, str(SRC / "tools"))
import strata_tokenizer as ST  # noqa: E402

TKDIR = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack/tokenizer")


def load_tokenizer():
    vocab = json.loads((TKDIR / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TKDIR / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((TKDIR / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def run_dir(tag: str) -> pathlib.Path:
    return pathlib.Path(tag) if "/" in tag else SRC / "s5" / "runs" / tag


def ids_of(tag: str):
    p = run_dir(tag) / "ids.txt"
    if not p.exists():
        return None
    return [int(x) for x in p.read_text().split()]


def checkpoints_line(tag: str) -> str:
    d = run_dir(tag)
    for f in ("out.txt", "err.txt"):
        p = d / f
        if p.exists():
            for line in p.read_text(errors="replace").splitlines():
                if "strata serve: prompt " in line:
                    return line.strip()
    return "<no prompt line>"


def cmd_table(tags):
    print(f"{'tag':<26} {'n':>4} {'md5':<32} prompt line")
    for t in tags:
        ids = ids_of(t)
        if ids is None:
            print(f"{t:<26} {'--':>4} {'<missing>':<32}")
            continue
        md5 = hashlib.md5((",".join(map(str, ids)) + "\n").encode()).hexdigest()
        print(f"{t:<26} {len(ids):>4} {md5:<32} {checkpoints_line(t)}")


def cmd_div(a, b):
    ia, ib = ids_of(a), ids_of(b)
    if ia is None or ib is None:
        print("missing ids")
        return 1
    n = min(len(ia), len(ib))
    first = next((i for i in range(n) if ia[i] != ib[i]), None)
    print(f"A={a} ({len(ia)} ids)  B={b} ({len(ib)} ids)")
    if first is None and len(ia) == len(ib):
        print("IDENTICAL (all ids, same length)")
        return 0
    if first is None:
        print(f"same prefix, different length: {len(ia)} vs {len(ib)}")
        return 0
    tail = min(n, first + 8)
    print(f"first divergence at index {first} of {n}")
    print(f"  A[{first}:{tail}] = {ia[first:tail]}")
    print(f"  B[{first}:{tail}] = {ib[first:tail]}")
    print(f"  shared prefix before it: {first} ids")
    return 0


def cmd_detok(tag, served):
    ids = ids_of(tag)
    if ids is None:
        print("missing ids")
        return 1
    tok = load_tokenizer()
    text = tok.decode(ids)
    print(f"== {tag}: {len(ids)} ids -> {len(text)} chars ==")
    print(text)
    if served:
        body = json.loads(pathlib.Path(served).read_text())
        msg = (body.get("choices") or [{}])[0].get("message", {})
        ref = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        print(f"== served {served}: {len(ref)} chars ==")
        print(ref)
        print(f"== equal: {text == ref}  (prefix shared: "
              f"{next((i for i in range(min(len(text), len(ref))) if text[i] != ref[i]), min(len(text), len(ref)))} chars) ==")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "div", "detok"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--served", default="")
    o = ap.parse_args()
    if o.cmd == "table":
        cmd_table(o.args)
    elif o.cmd == "div":
        return cmd_div(o.args[0], o.args[1])
    else:
        return cmd_detok(o.args[0], o.served)
    return 0


if __name__ == "__main__":
    sys.exit(main())
