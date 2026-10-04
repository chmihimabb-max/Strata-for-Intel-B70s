#!/usr/bin/env python3
"""D2x step 1: what the 300 dense MMVQ launches actually move, and at what GB/s.

Sources, both in the tree:
  * the GGUF shard 1 the engine serves natively from, read with tools/gguf_reader.py -- so the type, the
    shape and the bytes of every dense weight matrix come from the pack's own declaration, not from a
    hand-rolled block table (the reader's BLOCK_GEOMETRY is the same contract the kernels share);
  * d2/D2-CENSUS.txt, arm d2-hist-4096 -- the corrected census's per-type launch count and device
    microseconds for one T=4 decode window (strata/d2/STATUS-D2.md section 4).

The dense set is the ten tensor classes src/kernels/cuda/native_dense.cpp's `eligible` dispatch keys on,
plus GLOBAL output.weight (the head, which the engine's own "300 native projection matrices" line counts
as the 301st launch).  A GEMV reads every weight byte exactly once, so bytes / achievable bandwidth is a
hard lower bound on the call's time; achieved/floor is what any "put the dot product on the matrix unit"
proposal has to beat.

    /usr/bin/python3 d2x/bytes_floor.py > d2x/bytes-floor.txt
"""
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
from gguf_reader import GGUFFile  # noqa: E402

SH1 = pathlib.Path(
    "/home/michael/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    "/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S"
    "/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")

DENSE = [".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
         ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
         ".ffn_down_shexp.weight"]
HEAD = "output.weight"

# the corrected census, arm d2-hist-4096 (d2/D2-CENSUS.txt): one T=4 decode window, both stages summed
CENSUS_US = {"Q6_K": 30718.5, "IQ4_NL": 6213.2, "Q5_K": 5131.1, "IQ4_XS": 4940.6, "Q4_K": 4171.1, "Q8_0": 92.8}
CENSUS_N = {"Q6_K": 129, "IQ4_NL": 47, "Q5_K": 35, "IQ4_XS": 42, "Q4_K": 47, "Q8_0": 1}


def dense(name):
    return any(name.endswith(s) for s in DENSE) or name == HEAD


def main():
    g = GGUFFile(SH1)
    per_type = collections.defaultdict(lambda: {"n": 0, "bytes": 0, "shapes": collections.Counter()})
    total = 0
    unknown = []
    for t in g.tensors:
        if not dense(t.name):
            continue
        nb = t.expected_bytes()
        if nb is None:
            unknown.append((t.name, t.type_name, t.shape))
            continue
        d = per_type[t.type_name]
        d["n"] += 1
        d["bytes"] += nb
        # ggml ne[0] is the contiguous reduction dim = n_in; ne[1] = the weight row count = n_out
        d["shapes"][(t.shape[0], t.shape[1])] += 1
        total += nb
    n_all = sum(d["n"] for d in per_type.values())

    print("== the dense native MMVQ set of the IQ3_S pack (%s) ==" % SH1.name)
    print("   gguf tensors: %d, of which the dense set carries %d weights (the 10 dispatched classes + the head)"
          % (len(g.tensors), n_all))
    print()
    print("%-8s %5s %14s %10s %10s %s" % ("type", "n", "bytes", "MiB", "census n", "note"))
    for typ in sorted(per_type, key=lambda t: -per_type[t]["bytes"]):
        d = per_type[typ]
        print("%-8s %5d %14d %10.1f %10s %s" % (typ, d["n"], d["bytes"], d["bytes"] / 2**20,
                                                 CENSUS_N.get(typ, "-"),
                                                 "" if CENSUS_N.get(typ) == d["n"] else "<-- differs from census"))
    print("%-8s %5d %14d %10.1f %10d" % ("TOTAL", n_all, total, total / 2**20, sum(CENSUS_N.values())))
    if unknown:
        print("\n   NO BLOCK GEOMETRY KNOWN FOR: %s" % (unknown,))

    print()
    print("== shapes (n_in x n_out : how many tensors) ==")
    for typ in sorted(per_type, key=lambda t: -per_type[t]["bytes"]):
        sh = ", ".join("%dx%d:%d" % (a, b, c) for (a, b), c in sorted(per_type[typ]["shapes"].items()))
        print("  %-8s %s" % (typ, sh))

    print()
    print("== the window's achieved bandwidth per type (census us for these launches) ==")
    print("%-8s %8s %14s %12s %9s %11s" % ("type", "launches", "bytes", "census us", "GB/s", "us/launch"))
    for typ in sorted(per_type, key=lambda t: -per_type[t]["bytes"]):
        b = per_type[typ]["bytes"]
        us = CENSUS_US.get(typ)
        if not us:
            continue
        print("%-8s %8d %14d %12.1f %9.1f %11.1f"
              % (typ, CENSUS_N[typ], b, us, b / (us * 1e-6) / 1e9, us / CENSUS_N[typ]))
    us_tot = sum(CENSUS_US.values())
    print("%-8s %8d %14d %12.1f %9.1f" % ("TOTAL", sum(CENSUS_N.values()), total, us_tot,
                                           total / (us_tot * 1e-6) / 1e9))

    print()
    print("== the floor: bytes the weights force, at a given peak bandwidth ==")
    print("   weight bytes per window: %d (%.1f MiB)" % (total, total / 2**20))
    for peak in (150e9, 200e9, 300e9, 350e9, 400e9, 450e9, 512e9):
        floor_ms = total / peak * 1e3
        print("   %6.0f GB/s -> floor %7.3f ms  (census %.3f ms = %6.2fx the floor, i.e. %.1f%% of the floor)"
              % (peak / 1e9, floor_ms, us_tot / 1e3, (us_tot / 1e3) / floor_ms, floor_ms / (us_tot / 1e3) * 100))

    print()
    print("== the same for what an XMX-usable weight representation would cost ==")
    # a variant that needs a different weight encoding has to read THAT encoding, every call, once per
    # window; the pack's own blocks are the baseline
    for label, per_elem in (("the pack's blocks (as shipped)", None),
                            ("int8, 1 byte/element", 1.0),
                            ("fp8 e4m3, 1 byte/element", 1.0),
                            ("bf16, 2 bytes/element", 2.0),
                            ("f16, 2 bytes/element", 2.0),
                            ("tf32/f32, 4 bytes/element", 4.0)):
        if per_elem is None:
            b = total
        else:
            b = 0
            for t in g.tensors:
                if dense(t.name):
                    b += t.elements * per_elem
        print("   %-28s %14d B  = %5.2fx the pack's bytes" % (label, b, b / total))
        for peak in (450e9,):
            print("        floor at %.0f GB/s: %7.3f ms (pack: %7.3f ms), +%+.3f ms per window"
                  % (peak / 1e9, b / peak * 1e3, total / peak * 1e3, (b - total) / peak * 1e3))


if __name__ == "__main__":
    raise SystemExit(main())
