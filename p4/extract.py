#!/usr/bin/env python3
"""P4 (card t_63cc226b): turn the probe's raw `P4 ...` lines into the tables the status doc quotes.

    /usr/bin/python3 p4/extract.py p4/out-*.txt        # or p4/evidence/*.txt
"""
import sys
import re
from collections import defaultdict


def kv(line):
    out = {}
    for m in re.finditer(r"(\w+)=([^\s]+)", line):
        out[m.group(1)] = m.group(2)
    return out


def main(paths):
    rows = defaultdict(list)
    for p in paths:
        try:
            fh = open(p, encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in fh:
            if not line.startswith("P4 "):
                continue
            body = line[3:].strip()
            key = body.split(" ", 1)[0]
            rows[key].append(kv(body))
    for key in ("bw", "chunk", "layout", "inter", "dma", "hostcopy", "diag", "verify", "fixture", "work", "head"):
        if key not in rows:
            continue
        print(f"\n## P4 {key}  ({len(rows[key])} lines)")
        for r in rows[key]:
            print("   " + " ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["p4/out-all-run1.txt"])
