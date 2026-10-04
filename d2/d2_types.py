"""D2 (t_0416a0c0): the tensor -> type census the dispatch map is built on.

Sources: the GGUF shard 1 the engine serves natively from (the engine reads THESE types for its `native_type`),
and the pack's own index/native_experts tables.  Prints, per tensor class (the suffix the dispatch is keyed on):
the type histogram, then the per-layer list for the classes that are not uniform.

    /usr/bin/python3 d2/d2_types.py > d2/D2-TYPES.txt
"""
import collections
import pathlib
import sys

sys.path.insert(0, "/home/michael/strata-xpu/strata/tools")
from gguf_reader import GGUFFile  # noqa: E402

SNAP = pathlib.Path(
    "/home/michael/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    "/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S")
SH1 = SNAP / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
PACK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack")

FLOAT = {"F32", "F16", "BF16"}

# The suffix the engine's dispatch keys on (native_dense.cpp:eligible), plus the ones the decode window reads.
NATIVE_SUFFIXES = [".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight",
                   ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
                   ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"]


def cls(name: str) -> str:
    for s in NATIVE_SUFFIXES:
        if name.endswith(s):
            return "NATIVE-DENSE " + s[1:]
    if name.endswith(".ple_key.weight"):
        return "ple_key.weight (RETAINED-IF-NATIVE-PLE)"
    if name.startswith("blk."):
        return "OTHER blk.* " + ".".join(name.rsplit(".", 2)[-2:]).replace("blk.", "")
    return "GLOBAL " + name


def main() -> int:
    g = GGUFFile(SH1)
    tensors = g.tensors
    print("# D2 tensor census -- %s" % SH1.name)
    print("# gguf tensors: %d" % len(tensors))
    per_class = collections.OrderedDict()
    for t in tensors:
        per_class.setdefault(cls(t.name), []).append(t)
    for c, ts in per_class.items():
        hist = collections.Counter(t.type_name for t in ts)
        print("\n## %s   (%d tensors)" % (c, len(ts)))
        print("   " + "  ".join("%s x%d" % (k, v) for k, v in sorted(hist.items())))
        if len(hist) > 1:
            for t in sorted(ts, key=lambda t: t.name):
                print("   %-52s %s  shape=%s" % (t.name, t.type_name, list(t.shape)))
    # the routed experts: gu/down types per layer, from the pack's own table
    print("\n## ROUTED EXPERTS (pack/native_experts.txt: layer gu_type d_type ...)")
    hist = collections.Counter()
    rows = []
    for line in (PACK / "native_experts.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        rows.append((f[0], f[1], f[2]))
        hist[(f[1], f[2])] += 1
    print("   gu_type x d_type: " + "  ".join("%s/%s x%d" % (a, b, n) for (a, b), n in sorted(hist.items())))
    print("   per layer: " + " ".join("%s:%s/%s" % r for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
