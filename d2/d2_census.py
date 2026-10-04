#!/usr/bin/env python3
"""D2 (card t_0416a0c0): the decode window's census, with the attribution READ OFF THE SOURCE rather than guessed
from the symbol's spelling.

WHY THIS EXISTS.  D1's `d1_families.py` maps a kernel to a family by substring, and its first rule --
`("expert MMVQ", ("launch_multi_n",))` with the note "the native expert mat-vec, one launch per expert group" --
is WRONG, which D2 found by following the launcher to its definitions:

    `launch_multi_n<F, NCOLS>` is `native_mmvq.cu:1046`, the host launcher of `native_mmvq_multi_kernel`
    (`native_mmvq.cu:1001`), reachable only from `native_mmvq()`'s ncols > 1 path (`native_mmvq.cu:1064-1078`).
    It is the DENSE native projection MMVQ: one launch per (native weight matrix, window), and the window has
    exactly 300 of them -- the engine says so itself ("300 native projection matrices, 2018.88 MiB of weights"),
    and 300 = 36 GDN layers x (attn_qkv, attn_gate, ssm_out) + 12 QSA layers x (attn_q, attn_k, attn_v,
    attn_output) + 48 layers x (ffn_gate/up/down_shexp).
    The expert chain is `launch_gu<TG>` / `launch_down<TD>` (`iq_kernels.cu:1438/1447`), reached only through
    `native_expert_grouped` (`iq_kernels.cu:1590`), which the window calls once per (layer, group) -- 48 layers x
    2 groups = 96 calls, each with one gu and one down launch.

So the two families D1 merged into one 65.5% are two different things, and the split between them is the first
number this card has to get right.

Families (each one's defining symbol, and the file:line that reaches it):
  dense MMVQ        launch_multi_n<F,N> / launch_mmvq<T>        native_mmvq.cu:1046 / iq_kernels.cu:1426
  expert GU/DOWN    launch_gu<TG> / launch_down<TD>             iq_kernels.cu:1438/1447 via iq_kernels.cu:1603/1619
  expert epilogue   swiglu_entries_kernel / quantize_q8_1_kernel iq_kernels.cu:1611/1614
  expert indexing   native_expert_grouped                       iq_kernels.cu:1590
  bf16 MMA proj.    bf16_gemv_fp32_mmvf(_multi)                 native_bf16.cu:136/156 (router, indexer, shexp gate)
  mixer read        launch_multi (fused_gr.cpp)                 fused_gr.cpp:840 via verify.cpp:880
  flag handshake    wait_flag_ge / doorbell_*                   the per-layer host<->device protocol
  copies            <memcpy> / <memset> / *_from_mapped
  GDN / QSA         gdn_*, conv*, gr_* / qsa_*, kv_*, rope, norm, fwht
  routing/combine   router*, moe_*, native_moe_combine

usage: d2_census.py <hist.txt> [<hist.txt> ...] [--window N|--auto] [--top 30]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

TYPES = {2: "Q4_0", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
         42: "Q2_0", 29: "IQ1_M"}
TRAIT = {"Q4_0": "Q40", "Q5_0": "Q50", "Q8_0": "Q80", "Q3_K": "Q3K", "Q4_K": "Q4K", "Q5_K": "Q5K",
         "Q6_K": "Q6K", "IQ4_NL": "IQ4NLBlock", "IQ4_XS": "IQ4XS", "Q2_0": "Q20", "IQ3_XXS": "IQ3XXS",
         "IQ3_S": "IQ3S", "IQ2_S": "IQ2S", "IQ2_XXS": "IQ2XXS", "IQ2_XS": "IQ2XS"}


def short(name: str) -> str:
    """The same normalization d1_hist.py uses: strip the namespace prefixes and stop at the argument list."""
    n = name
    for pre in ("strata::kernels::(anonymous namespace)::", "strata::kernels::", "strata::sycl_compat::",
                "(anonymous namespace)::"):
        n = n.replace(pre, "")
    depth = 0
    out = []
    for ch in n:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "(" and depth == 0:
            break
        out.append(ch)
    return "".join(out).strip()


def traits_of(kernel: str) -> str:
    """The weight type a `launch_multi_n<Traits, N>` / `launch_gu<TG>` line is instantiated for.  Read off the
    template argument as it appears in the demangled type, so the table is per TYPE and not per site."""
    for name, trait in sorted(TRAIT.items(), key=lambda kv: -len(kv[1])):
        if (trait + "Traits") in kernel or trait + ">" in kernel or trait + "," in kernel \
                or ("SmallTraits<%s" % trait) in kernel:
            return name
    m = re.search(r"launch_(?:gu|down)<(\d+)>", kernel)
    if m:
        return TYPES.get(int(m.group(1)), "type%s" % m.group(1))
    return ""


def family(kernel: str) -> tuple:
    k = kernel.split("#")[0]
    if k.startswith("launch_multi_n<") or k.startswith("launch_mmvq<") or k.startswith("mmvq"):
        return "dense native MMVQ (one launch per weight matrix)", traits_of(kernel)
    if k.startswith("launch_multi"):
        return "mixer read (fused GR / hyper-connection)", ""
    if k.startswith("launch_gu") or k.startswith("native_gu"):
        return "routed experts: gate+up", traits_of(kernel)
    if k.startswith("launch_down") or k.startswith("native_down"):
        return "routed experts: down", traits_of(kernel)
    if k.startswith("native_expert_grouped"):
        return "routed experts: indexing", ""
    if k.startswith("swiglu_entries") or k.startswith("quantize_q8_1") or k.startswith("native_quantize_q8_1"):
        return "routed experts: epilogue (swiglu/quantize)", ""
    if k.startswith("bf16_gemv_fp32_mmvf"):
        return "bf16-weight projection (llama.cpp mmvf, FMA dot)", ""
    if k.startswith("wait_flag") or k.startswith("doorbell") or "flag" in k:
        return "flag handshake (device side)", ""
    if k.startswith("<memcpy") or k.startswith("<memset") or "from_mapped" in k or "to_mapped" in k:
        return "copies / mapped staging", ""
    if k.startswith("launch_multi"):
        return "mixer read (fused GR / hyper-connection)", ""
    if k.startswith(("qsa_", "kv_", "native_qsa", "native_rope", "rope", "rms_norm", "fwht", "attn")):
        return "QSA attention / indexer / norms", ""
    if k.startswith(("gdn_", "conv", "gr_", "fused_gr", "shared_expert")):
        return "GDN recurrence / shared expert", ""
    if k.startswith(("router", "moe_", "native_router", "native_moe")):
        return "routing / combine", ""
    return "other", ""


def parse(path: str):
    windows = defaultdict(list)
    meta = defaultdict(lambda: {"subs": 0, "walls": []})
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        m = re.search(r"win=(\d+)", f[1])
        if not m:
            continue
        w = int(m.group(1))
        if f[0] == "HW":
            mm = re.search(r"submissions (\d+)", f[2])
            if mm:
                meta[w]["subs"] += int(mm.group(1))
            wm = re.search(r"wall=([\d.]+)ms", f[1])
            if wm:
                meta[w]["walls"].append(float(wm.group(1)))
        elif f[0] == "H" and len(f) >= 5:
            windows[w].append((int(f[2]), float(f[3]), f[4]))
    return windows, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--auto", action="store_true", default=True)
    ap.add_argument("--top", type=int, default=30)
    o = ap.parse_args()

    for path in o.paths:
        windows, meta = parse(path)
        if not windows:
            print("%s: no histogram lines" % path)
            continue
        w = max(windows, key=lambda x: sum(c for c, _, _ in windows[x]))
        rows = windows[w]
        fam = defaultdict(lambda: [0, 0.0])
        sub = defaultdict(lambda: [0, 0.0])
        for c, us, name in rows:
            f, t = family(short(name))
            fam[f][0] += c
            fam[f][1] += us
            if t:
                sub[(f, t)][0] += c
                sub[(f, t)][1] += us
        tot_c = sum(v[0] for v in fam.values())
        tot_us = sum(v[1] for v in fam.values())
        print("== %s" % path)
        print("   window %d: %d counted submissions, %.1f ms of device time, wall %.1f ms (stage slices summed)"
              % (w, tot_c, tot_us / 1000.0, max(meta[w]["walls"]) if meta[w]["walls"] else -1))
        print("   %-46s %7s %11s %8s %8s" % ("family", "count", "us", "%count", "%time"))
        for name, v in sorted(fam.items(), key=lambda kv: -kv[1][1])[:o.top]:
            print("   %-46s %7d %11.1f %7.1f%% %7.1f%%" % (name, v[0], v[1], 100.0 * v[0] / tot_c,
                                                           100.0 * v[1] / tot_us))
        if sub:
            print("   -- the two projection families, by weight type --")
            for (f, t), v in sorted(sub.items(), key=lambda kv: -kv[1][1]):
                print("   %-46s %-10s %6d %10.1f %6.1f%%" % (f, t, v[0], v[1], 100.0 * v[1] / tot_us))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
