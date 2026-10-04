import re
import sys
from collections import defaultdict

path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sites = defaultdict(lambda: [0, 0.0, set()])
for line in open(path, errors="replace"):
    f = line.rstrip("\n").split("\t")
    if len(f) < 5 or f[0] != "H":
        continue
    m = re.search(r"win=(\d+)", f[1])
    if not m or int(m.group(1)) != want:
        continue
    name = f[4]
    if "launch_multi_n" not in name and "launch_gu" not in name and "launch_down" not in name:
        continue
    sites[name][0] += int(f[2])
    sites[name][1] += float(f[3])
    sites[name][2].add(f[1].split(" ")[1])

for name, (c, us, slices) in sorted(sites.items(), key=lambda kv: -kv[1][1]):
    print("%6d %10.1fus  %-30s %s" % (c, us, "/".join(sorted(slices)), name[:200]))
print("total", sum(v[0] for v in sites.values()), "%.1f us" % sum(v[1] for v in sites.values()))
