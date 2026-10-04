#!/usr/bin/env python3
"""Debug the NATIVE-DENSE parse: section counts vs the header's own tensor count."""
import re
import collections

fam = None
per = {}
tot = 0
tensors = []
for line in open("d2/D2-TYPES.txt"):
    m = re.match(r"## (NATIVE-DENSE|GLOBAL|OTHER) (\S+)\s+\((\d+) tensors?\)", line)
    if m:
        fam = m.group(2)
        per[m.group(2)] = [0, int(m.group(3))]
        continue
    if line.startswith("## "):
        fam = None
        continue
    if fam is None:
        continue
    mm = re.match(r"\s+(blk\.\d+\.|)(\S+)\s+(\S+)\s+shape=\[(\d+), (\d+)\]", line)
    if mm:
        per[fam][0] += 1
        tot += 1
        tensors.append((fam, mm.group(3), int(mm.group(4)), int(mm.group(5))))
    elif "shape=" in line:
        print("UNMATCHED:", line.rstrip())
for k, (got, exp) in per.items():
    print("%-40s parsed %3d  header %3d %s" % (k, got, exp, "" if got == exp else "  <-- MISMATCH"))
print("total", tot)
print()
c = collections.Counter((t[1], t[2], t[3]) for t in tensors)
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:30]:
    print("  %-8s %6dx%-6d %3d" % (k[0], k[1], k[2], v))
