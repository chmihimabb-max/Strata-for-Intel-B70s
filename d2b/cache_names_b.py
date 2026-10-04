#!/usr/bin/env python3
"""D2b: every entry in a cache dir with its mtime and its mmvq specialization name, sorted -- to see whether a
name built in the load phase (the warm-up) is built AGAIN inside the ask (which would mean the launch missed the
warm-up's program)."""
import os
import re
import subprocess
import sys
import datetime as dt

cache = sys.argv[1]
cut = float(sys.argv[2]) if len(sys.argv) > 2 else None
rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                      capture_output=True, text=True).stdout.split("\n")
seen = {}
out = []
for line in rows:
    if not line.strip():
        continue
    t, d = line.split(" ", 1)
    t = float(t)
    blob = subprocess.run(["strings", "-a", os.path.join(d, "0.src")], capture_output=True, text=True).stdout
    m = re.search(r"native_mmvq_multi_kernelI[^\s]*", blob)
    if not m:
        m = re.search(r"native_[a-z0-9_]*mmvq[a-zA-Z0-9_]*kernel[^\s]*", blob)
    sym = m.group(0) if m else "?"
    short = re.sub(r"^native_mmvq_multi_kernelINS1_", "", sym)
    out.append((t, short))
    seen.setdefault(short, []).append(t)
out.sort()
for t, s in out:
    print(f"{dt.datetime.fromtimestamp(t):%H:%M:%S}  {s[:120]}")
dups = {s: ts for s, ts in seen.items() if len(ts) > 1}
print(f"# entries {len(out)}, distinct names {len(seen)}")
for s, ts in sorted(dups.items()):
    print(f"# DUPLICATE x{len(ts)}: {[dt.datetime.fromtimestamp(x).strftime('%H:%M:%S') for x in ts]}  {s[:110]}")
