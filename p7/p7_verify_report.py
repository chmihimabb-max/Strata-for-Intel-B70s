"""P7 (card t_6789d6df): the acceptance-count report for tools/mtp_fetch.py verify.

Prints the revision actually read, the pinned revision the SHA256 set belongs to, and the verified count as
"N/31", so the card's acceptance evidence is one line rather than "verify printed nothing, exit 0".
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import mtp_fetch  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "/home/michael/strata-xpu/mtp/canonical"
env = os.environ.get("STRATA_MTP_REVISION")
print("STRATA_MTP_REVISION override : %s" % (env if env else "<unset - the pinned revision is used>"))
print("PINNED_REVISION              : %s" % mtp_fetch.PINNED_REVISION)
print("REVISION read this run       : %s" % mtp_fetch.REVISION)
print("repo read                    : %s" % mtp_fetch.REPO)
print("hashes apply (pinned)        : %s" % mtp_fetch.pinned())
print()
bad = mtp_fetch.verify(out)
for name in bad:
    print("MTP tensor missing or corrupt: %s" % name, file=sys.stderr)
total = len(mtp_fetch.SHA256)
print("verify: %d/%d tensors SHA256-verified against the pinned revision" % (total - len(bad), total))
if bad:
    print("verify: %d BAD: %s" % (len(bad), ", ".join(bad)))
man = os.path.join(out, "mtp-manifest.json")
if os.path.exists(man):
    m = json.load(open(man))
    print("mtp-manifest.json: %d tensors, %.3f GB of tensor bytes, source repo %s" % (
        len(m), sum(t["bytes"] for t in m) / 1e9,
        json.load(open(os.path.join(out, "mtp-inventory.json")))["repo"]))
sys.exit(mtp_fetch.BAD if bad else 0)
