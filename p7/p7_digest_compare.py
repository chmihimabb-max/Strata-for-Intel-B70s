"""P7 (card t_6789d6df): is the faithfully rebuilt drafter the same drafter?

Compares, tensor by tensor:
  A. the canonical fetch (tools/mtp_fetch.py --out ~/strata-xpu/mtp/canonical, pinned revision) -- digests
     RECOMPUTED here from the bytes on disk, not read from the manifest;
  B. the W4A16-derived artifact's input (scripts/w1b_mtp_pipeline.sh -> tools/mtp_w4a16_adapter.py output),
     also RECOMPUTED from its own bytes on disk, with its recorded digests checked for staleness;
and then the two runtime directories the engine actually loads (dense.bin / experts.bin / dense.txt), byte for byte.

    python3 p7/p7_digest_compare.py <canonical-dir> <w4a16-dir> <canonical-rt> <w4a16-rt>
"""
import hashlib
import json
import os
import sys

CAN = sys.argv[1] if len(sys.argv) > 1 else "/home/michael/strata-xpu/mtp/canonical"
W16 = sys.argv[2] if len(sys.argv) > 2 else "/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16"
RT_NEW = sys.argv[3] if len(sys.argv) > 3 else "/home/michael/strata-xpu/mtp/rt"
RT_OLD = sys.argv[4] if len(sys.argv) > 4 else "/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(d, name):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


print("=" * 100)
print("A. CANONICAL FETCH  (tools/mtp_fetch.py, pinned revision) : %s" % CAN)
print("B. W4A16-DERIVED    (tools/mtp_w4a16_adapter.py)          : %s" % W16)
print("=" * 100)

can = load(CAN, "mtp-manifest.json")
w16 = load(W16, "mtp-manifest.json")
inv = load(CAN, "mtp-inventory.json")
print("canonical source repo : %s" % inv["repo"])
print("canonical tensors     : %d, %.3f GB" % (len(can), sum(t["bytes"] for t in can) / 1e9))
print("W4A16 tensors         : %d, %.3f GB" % (len(w16), sum(t["bytes"] for t in w16) / 1e9))
print()

by_can = {t["name"]: t for t in can}
by_w16 = {t["name"]: t for t in w16}
names = sorted(set(by_can) | set(by_w16))
missing = sorted(set(by_can) ^ set(by_w16))
if missing:
    print("!! tensor name sets differ: %s" % missing)
print("%-58s %-6s %-6s %-8s %s" % ("tensor", "MB", "canon", "adapter", "recomputed-from-bytes"))
rows, same, diff, stale = [], 0, 0, 0
for n in names:
    a, b = by_can.get(n), by_w16.get(n)
    if a is None or b is None:
        print("%-58s %s" % (n, "only in one side"))
        diff += 1
        continue
    pa = os.path.join(CAN, a["file"])
    pb = os.path.join(W16, b["file"])
    ha, hb = sha256(pa), sha256(pb)
    a_ok = ha == a["sha256"]
    b_ok = hb == b["sha256"]
    match = ha == hb
    same += match
    diff += (not match)
    stale += (not a_ok) + (not b_ok)
    rows.append(dict(name=n, bytes=a["bytes"], canonical_sha256=ha, canonical_matches_manifest=a_ok,
                     w4a16_sha256=hb, w4a16_matches_manifest=b_ok, identical=match))
    print("%-58s %-6.1f %-6s %-8s %s%s" % (n, a["bytes"] / 1e6,
                                           "ok" if a_ok else "STALE",
                                           "ok" if b_ok else "STALE",
                                           "IDENTICAL" if match else "DIFFER h_can=%s h_w16=%s" % (ha[:16], hb[:16]),
                                           "" if match else ""))
print()
print("digest comparison: %d/%d tensors IDENTICAL, %d differ, %d recorded digests stale" % (same, len(names), diff, stale))

print()
print("=" * 100)
print("RUNTIME DIRS THE ENGINE LOADS")
print("  canonical -> %s" % RT_NEW)
print("  W4A16     -> %s" % RT_OLD)
print("=" * 100)
rt = {}
for f in ("dense.bin", "experts.bin", "dense.txt", "draft_vocab.bin"):
    pa, pb = os.path.join(RT_NEW, f), os.path.join(RT_OLD, f)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print("%-16s %-10s %-10s (missing on one side)" % (f, os.path.exists(pa), os.path.exists(pb)))
        continue
    ha, hb = sha256(pa), sha256(pb)
    rt[f] = dict(new=ha, old=hb, identical=ha == hb, bytes=os.path.getsize(pa))
    print("%-16s %12d B  %s\n    new %s\n    old %s" % (f, os.path.getsize(pa),
                                                        "IDENTICAL" if ha == hb else "DIFFER", ha, hb))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p7-digest-comparison.json")
with open(out, "w") as f:
    json.dump(dict(canonical_dir=CAN, w4a16_dir=W16, canonical_repo=inv["repo"], tensors=rows,
                   identical=same, differing=diff, stale_manifests=stale, runtime=rt), f, indent=1)
print()
print("wrote %s" % out)
