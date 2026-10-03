#!/usr/bin/env python3
"""D1 (card t_d9ffcf38): the window's launch sites rolled up into FAMILIES, so the histogram's answer is a
short list of targets rather than 60 rows.

Families are read off the kernel (the short name d1_hist.py writes in the `kernel` column):
  expert MMVQ   launch_multi_n<...>          the native expert matrix-vector product (one launch per expert group)
  expert GU     launch_gu<N>                 the native expert gate+up
  expert DOWN   launch_down<N>               the native expert down projection
  grouping      native_expert_grouped / native_quantize_q8_1 / native_router_top10 / moe_combine / moe_hit_add
  flag/handshake wait_flag_ge / doorbell_publish / raise_flag / publish_flag
  copies        <memcpy> / <memset> / *_from_mapped / copy+combine
  GDN           launch_multi (fused_gr: the GDN recurrence + its projections) / gdn_*
  QSA           qsa_* / kv_* / bf16_gemv*
  other         everything else

usage: d1_families.py <sites.csv> [...]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

FAMILIES = [
    ("expert MMVQ", ("launch_multi_n",)),
    ("expert GU", ("launch_gu",)),
    ("expert DOWN", ("launch_down",)),
    ("expert grouping/quantize", ("native_expert_grouped", "native_quantize_q8_1", "quantize_q8_1_rows",
                                  "native_router_top10", "native_moe_combine", "moe_hit_add",
                                  "native_expert_scatter", "native_expert_gather")),
    ("flag handshake (device)", ("wait_flag_ge", "doorbell_publish", "raise_flag", "publish_flag",
                                 "doorbell_wait", "wait_flag")),
    ("copies/mapped staging", ("<memcpy>", "<memset>", "from_mapped", "copy_to_mapped", "copy_rows")),
    ("GDN recurrence", ("launch_multi", "gdn_", "gr_", "fused_gr", "conv", "shared_expert")),
    ("QSA attention/indexer", ("qsa_", "kv_append", "kv_resolve", "bf16_gemv", "kv_", "native_qsa")),
]


def family(kernel: str) -> str:
    for name, pats in FAMILIES:
        for p in pats:
            if kernel.startswith(p) or p in kernel.split("#")[0]:
                return name
    return "other"


def main() -> int:
    for path in sys.argv[1:]:
        fam: dict = defaultdict(lambda: [0, 0.0])
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                f = family(row["kernel"])
                fam[f][0] += int(row["count"])
                fam[f][1] += float(row["us"])
        tot_c = sum(v[0] for v in fam.values())
        tot_us = sum(v[1] for v in fam.values())
        print("== %s: %d submissions, %.2f ms counted" % (path, tot_c, tot_us / 1000.0))
        print("%-26s %8s %11s %8s %8s" % ("family", "count", "us", "%count", "%time"))
        for name, v in sorted(fam.items(), key=lambda kv: -kv[1][1]):
            print("%-26s %8d %11.1f %7.1f%% %7.1f%%" % (name, v[0], v[1], 100.0 * v[0] / tot_c,
                                                        100.0 * v[1] / tot_us if tot_us else 0.0))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
