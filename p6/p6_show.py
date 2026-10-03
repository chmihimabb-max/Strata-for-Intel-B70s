#!/usr/bin/env python3
"""Print a /v1/chat/completions response body compactly: the answer text, whether reasoning came back, usage, timings."""
import json
import sys

p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
ch = d["choices"][0]
m = ch["message"]
print("model       :", d.get("model"))
print("finish      :", ch.get("finish_reason"))
print("content     :", (m.get("content") or "").strip()[:600])
r = m.get("reasoning_content") or ""
print("reasoning   :", f"{len(r)} chars" + (" (present -> thinking ON)" if r else " (absent -> thinking OFF)"))
if r:
    print("  reasoning head:", r.strip()[:200].replace("\n", " "))
print("usage       :", json.dumps(d.get("usage")))
print("timings     :", json.dumps(d.get("timings")))
