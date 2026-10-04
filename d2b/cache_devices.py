#!/usr/bin/env python3
"""D2b: which SYCL cache subdirectory (one per device) an entry belongs to, and when it was written."""
import os
import subprocess
import sys
from collections import Counter

cache = sys.argv[1]
rows = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                      capture_output=True, text=True).stdout.split("\n")
per = Counter()
per_time = {}
for line in rows:
    if not line.strip():
        continue
    t, d = line.split(" ", 1)
    dev = d[len(cache):].strip("/").split("/")[0]
    per[dev] += 1
    per_time.setdefault(dev, []).append(float(t))
print(f"cache {cache}")
for dev, n in per.most_common():
    ts = sorted(per_time[dev])
    import datetime as dt
    print(f"  device dir {dev[:16]}…  {n} entries   first {dt.datetime.fromtimestamp(ts[0]):%H:%M:%S}  last {dt.datetime.fromtimestamp(ts[-1]):%H:%M:%S}")
