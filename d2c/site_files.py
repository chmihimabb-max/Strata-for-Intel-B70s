#!/usr/bin/env python3
"""D2c: map each decode-phase launch site to the lines that reference it (definition vs call)."""
import json
import os
import re
import subprocess
import sys

SRC = "/home/michael/strata-xpu/strata"
sites = json.load(open(os.path.join(SRC, sys.argv[1])))
names = []
for k in sites:
    s = k.split(" -> ")[0]
    if s == "(NO SITE)":
        s = k.split(" -> ")[-1].split("<")[0]
    names.append(s)
names = sorted(set(names))
dirs = ["src/kernels/cuda", "src/core", "src/program", "include/strata/kernels", "include/strata/core"]
out = subprocess.run(["grep", "-rn", "-E", "|".join(r"\b%s\b" % re.escape(n) for n in names)]
                     + [os.path.join(SRC, d) for d in dirs], capture_output=True, text=True).stdout
rows = []
for line in out.splitlines():
    m = re.match(r"([^:]+):(\d+):(.*)", line)
    if m:
        rows.append((os.path.relpath(m.group(1), SRC), int(m.group(2)), m.group(3).strip()))

print(f"# {len(names)} sites, {len(rows)} referencing lines")
for n in names:
    hits = [(f, l, t) for f, l, t in rows if re.search(r"\b%s\b" % re.escape(n), t)]
    defs = [(f, l, t) for f, l, t in hits if re.search(r"\b%s\s*\(" % re.escape(n), t)
            and not t.rstrip().endswith(";") and "=" not in t.split("(")[0]]
    print(f"\n## {n}   ({len(hits)} refs, {len(defs)} definition-shaped)")
    for f, l, t in (defs[:2] or hits[:2]):
        print(f"   {f}:{l}  {t[:120]}")
