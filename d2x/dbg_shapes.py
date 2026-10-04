#!/usr/bin/env python3
"""Per-section shapes of the NATIVE-DENSE set: which tensor carries which geometry, and which way round."""
import re
import collections

fam = None
sec = collections.defaultdict(collections.Counter)
sample = {}
for line in open("d2/D2-TYPES.txt"):
    m = re.match(r"## (NATIVE-DENSE|GLOBAL|OTHER) (\S+)\s+\((\d+) tensors?\)", line)
    if m:
        fam = m.group(2)
        continue
    if line.startswith("## "):
        fam = None
        continue
    if fam is None:
        continue
    mm = re.match(r"\s+(blk\.\d+\.|)(\S+)\s+(\S+)\s+shape=\[(\d+), (\d+)\]", line)
    if mm and fam:
        sec[fam][(int(mm.group(4)), int(mm.group(5)), mm.group(3))] += 1
        sample.setdefault((fam, int(mm.group(4)), int(mm.group(5))), mm.group(2))

# the raw GLOBAL lines, so the head's own shape spelling is visible
print("== raw GLOBAL block ==")
with open("d2/D2-TYPES.txt") as fh:
    for line in fh:
        if line.startswith("## GLOBAL "):
            print("   " + line.rstrip())
            for _ in range(3):
                nxt = next(fh, "")
                if nxt.startswith("## "):
                    break
                print("   " + nxt.rstrip())
print()
print("== NATIVE-DENSE sections: shape -> count (type) ==")
for k in sorted(sec):
    print(" %s" % k)
    for (a, b, t), n in sorted(sec[k].items(), key=lambda kv: -kv[1]):
        print("    %6d x %-6d %-7s %3d   e.g. %s" % (a, b, t, n, sample[(k, a, b)]))
