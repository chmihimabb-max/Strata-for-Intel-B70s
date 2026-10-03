#!/usr/bin/env python3
"""P8 debug: dump one /proc/<pid>/fdinfo file exactly as python sees it (repr), to check the key parsing.
Usage: p8_fdinfo_dump.py <pid> [fd]
"""
import os
import sys

pid = sys.argv[1]
fd = sys.argv[2] if len(sys.argv) > 2 else None
fds = sorted(os.listdir(f"/proc/{pid}/fdinfo"))
print(f"{len(fds)} fdinfo entries")
cands = [fd] if fd else fds
for f in cands:
    try:
        text = open(f"/proc/{pid}/fdinfo/{f}").read()
    except OSError as e:
        print(f"-- {f}: {e}")
        continue
    if "drm-pdev" not in text:
        continue
    print(f"-- fdinfo/{f} ({len(text)} bytes)")
    for line in text.splitlines():
        print("   " + repr(line))
