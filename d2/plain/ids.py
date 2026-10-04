#!/usr/bin/env python3
"""D2b: the greedy-id streams of the arms, hashed BOTH ways the cards use.

The rig's `grep '^T ' out.txt | md5sum` includes the trailing newline; D2's reader joins the ids with "\\n" and
hashes without it.  They are two different strings of the same stream - state the method with the number.
"""
import hashlib
import os
import sys

for d in sys.argv[1:]:
    out = os.path.join(d, "out.txt")
    if not os.path.exists(out):
        print("%-34s (no out.txt)" % d)
        continue
    ids = [l.split()[1] for l in open(out) if l.startswith("T ")]
    no_nl = hashlib.md5("\n".join(ids).encode()).hexdigest()[:32]
    with_nl = hashlib.md5("".join(l for l in open(out) if l.startswith("T ")).encode()).hexdigest()[:32]
    err = os.path.join(d, "err.txt")
    msw = ""
    if os.path.exists(err):
        for l in open(err):
            if "ms/window = verify" in l:
                msw = l.split("avg T")[0].strip().split(":")[-1].strip()
            if "strata serve: prompt" in l and "generated in" in l:
                msw += " | " + l.split("strata serve: prompt")[1].strip()
    print("%-34s n=%-4d D2-style %s  rig %s" % (os.path.basename(d), len(ids), no_nl, with_nl))
    print("      %s" % msw)
