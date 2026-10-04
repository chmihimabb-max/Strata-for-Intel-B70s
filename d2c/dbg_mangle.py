#!/usr/bin/env python3
import subprocess, sys
raw = open(sys.argv[1], "rb").read()
toks = sorted({t.decode() for t in raw.split(b"\x00") if t.startswith(b"_Z") and len(t) > 8})
print(f"# {len(toks)} _Z tokens")
out = subprocess.run(["c++filt"] + toks, capture_output=True, text=True).stdout.split("\n")
for t, d in zip(toks, out):
    print(f"--- {t[:60]}")
    print(f"    {d[:400]}")
