#!/usr/bin/env python3
"""S4: the numerics statement - is the generated text of two served arms identical?

usage: s4_cmp.py <resp-a.json> <resp-b.json>

The served path exposes TEXT, not token ids (the driver rigs' id protocol does not exist through serve.server),
so this is the honest form of the check: with temperature 0 the decode is greedy, so byte-identical text is the
statement available on this path.  Prints each field's length and md5, and says IDENTICAL or DIFFERENT per field.
"""
from __future__ import annotations

import hashlib
import json
import sys


def fields(path: str) -> dict[str, str]:
    ch = (json.load(open(path)).get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {k: v for k, v in msg.items() if isinstance(v, str)}


def main() -> int:
    a, b = fields(sys.argv[1]), fields(sys.argv[2])
    keys = sorted(set(a) | set(b))
    same = True
    for k in keys:
        x, y = a.get(k, ""), b.get(k, "")
        hx = hashlib.md5(x.encode()).hexdigest()
        hy = hashlib.md5(y.encode()).hexdigest()
        ok = hx == hy
        same = same and ok
        print(f"{k:18s} A={len(x):6d} chars {hx[:16]}   B={len(y):6d} chars {hy[:16]}   "
              f"{'IDENTICAL' if ok else 'DIFFERENT'}")
    print(f"VERDICT: {'the generated text is byte-identical' if same else 'THE TEXT DIFFERS'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
