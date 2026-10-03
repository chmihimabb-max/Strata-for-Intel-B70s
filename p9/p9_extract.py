#!/usr/bin/env python3
"""P9: pull the DEVICE (gpu_op) rows of a unitrace chrome timeline into a compact TSV.

These timelines are 0.6-10 GB; every analysis after this one reads the TSV instead.

usage: /usr/bin/python3 p9/p9_extract.py <trace.json> <out.tsv> [--host]

Columns: pid  tid  ts_us  dur_us  tag
`tag` is the raw event name with the shim's own wrappers stripped (`strata::sycl_compat::launch<`,
`strata::`, `(anonymous namespace)::`) and truncated to KEEP characters; the exact names the tags
stand for are written next to the TSV as <out>.tags (tag -> count + one full example), so nothing the
report quotes is a truncation artefact.
"""
import json
import sys
from collections import Counter

KEEP = 96


def canon(n):
    s = n.replace("strata::sycl_compat::launch<", "")
    s = s.replace("strata::kernels::(anonymous namespace)::", "anon::")
    s = s.replace("strata::kernels::", "")
    s = s.replace("(anonymous namespace)::", "anon::")
    s = s.replace("strata::prefill::", "prefill::")
    s = s.replace("strata::core::", "core::")
    s = s.replace("strata::sycl_compat::", "")
    s = s.replace("strata::", "")
    return s[:KEEP]


def main():
    path = sys.argv[1]
    out = sys.argv[2]
    want_host = "--host" in sys.argv
    n = 0
    kept = 0
    tags = Counter()
    example = {}
    with open(path, "r", errors="replace") as f, open(out, "w") as g:
        for line in f:
            if '"ph"' not in line:
                continue
            try:
                e = json.loads(line.rstrip().rstrip(","))
            except Exception:
                continue
            if e.get("ph") != "X":
                continue
            n += 1
            is_dev = e.get("cat") == "gpu_op"
            if is_dev == want_host:      # want_host selects the OTHER arm
                continue
            raw = e.get("name") or "?"
            tag = canon(raw) if is_dev else raw
            g.write("%s\t%s\t%.1f\t%.3f\t%s\n" % (e.get("pid"), e.get("tid"),
                                                  float(e.get("ts", 0.0)), float(e.get("dur", 0.0)), tag))
            kept += 1
            tags[tag] += 1
            if tag not in example:
                example[tag] = raw
            if n % 500000 == 0:
                sys.stderr.write(f"  rows={n:,} kept={kept:,}\n")
    with open(out + ".tags", "w") as t:
        for tag, c in tags.most_common():
            t.write(f"{c}\t{tag}\t{example[tag]}\n")
    print(f"rows={n:,} kept={kept:,} -> {out}  ({len(tags):,} distinct tags)")


if __name__ == "__main__":
    main()
