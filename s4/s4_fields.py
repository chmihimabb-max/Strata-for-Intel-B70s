#!/usr/bin/env python3
"""S4: show the shape of a /v1/chat/completions body - which fields carry text - for the write-up.

usage: s4_fields.py <resp.json> [...]  ->  one line per body: the top-level keys, the choice's keys, and the
head of every string field longer than 0 chars.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    for path in sys.argv[1:]:
        o = json.load(open(path))
        print(f"--- {path}")
        print(f"    top keys    : {sorted(o.keys())}")
        ch = (o.get("choices") or [{}])[0]
        print(f"    choice keys : {sorted(ch.keys())}")
        msg = ch.get("message") or {}
        print(f"    message keys: {sorted(msg.keys())}")
        for k, v in msg.items():
            if isinstance(v, str) and v:
                print(f"    {k:16s}: {len(v)} chars :: {v[:160]!r}")
        print(f"    finish      : {ch.get('finish_reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
