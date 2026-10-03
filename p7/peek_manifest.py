"""P7 (card t_6789d6df): inspect the W4A16-derived MTP manifest (the artifact being replaced)."""
import json
import sys

p = sys.argv[1]
m = json.load(open(p))
print("file:", p)
print("entries:", len(m))
print("keys:", sorted(m[0].keys()))
print(json.dumps(m[0], indent=1)[:800])
for t in m:
    if "experts" in t["name"]:
        print(json.dumps(t, indent=1))
