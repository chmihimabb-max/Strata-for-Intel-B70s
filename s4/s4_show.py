#!/usr/bin/env python3
"""S4: print a /v1/chat/completions response body in the shape the card's evidence needs.

usage: s4_show.py <resp.json>   ->  the usage line, the timings line, then the answer text.

A failed request has no resp.json (p10_client.py writes <out>.error instead), which is what the caller prints.
"""
from __future__ import annotations

import json
import sys

o = json.load(open(sys.argv[1]))
print("usage    : " + json.dumps(o.get("usage")))
print("timings  : " + json.dumps(o.get("timings")))
ch = o.get("choices") or [{}]
print("finish   : " + str(ch[0].get("finish_reason")))
content = (ch[0].get("message") or {}).get("content") or ""
print("answer   : " + content[:1200].replace("\n", " "))
