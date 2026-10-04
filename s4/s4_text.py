#!/usr/bin/env python3
"""S4: print the generated text of a served response in full (every string field of the message).

usage: s4_text.py <resp.json> [...]
"""
from __future__ import annotations

import json
import sys

for path in sys.argv[1:]:
    ch = (json.load(open(path)).get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    print(f"--- {path}   finish={ch.get('finish_reason')}")
    for k, v in msg.items():
        if isinstance(v, str) and v:
            print(f"[{k}] {v}")
    print()
