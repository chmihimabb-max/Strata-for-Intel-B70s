#!/usr/bin/env python3
"""P10: the comparison table, built from every arm's p10_summary.py --json output.

usage: p10_table.py [arm ...]        (default: the arms this card ran, in the order they were run)

Columns: what the card asks to be shown side by side - the mask/card count, the expert residency, the CPU
experts per layer-window, decode and prefill tok/s at each length, peak VRAM/RSS, and the file tier.
"""
from __future__ import annotations

import json
import subprocess
import sys

R = "/home/michael/strata-xpu"
SRC = f"{R}/strata"
DEFAULT = ["1c-4k", "2c-4k", "1c1-4k", "1c-32k", "2c-32k", "1c-128k", "2c-128k"]


def row(tag):
    out = subprocess.run(["/usr/bin/python3", f"{SRC}/p10/p10_summary.py", f"{R}/p10/runs/{tag}", "--json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def main() -> int:
    tags = sys.argv[1:] or DEFAULT
    cols = [("arm", "arm"), ("mask", "mask"), ("ctx", "engine_argv_context"),
            ("slots", "info_expert_slots"), ("cache MiB", "info_expert_cache_mib"),
            ("vram free", "info_vram_free_mib"), ("pool", "info_pool_workers"),
            ("CPU exp/lw", "cpu_experts_per_lw"), ("entries/lw", "cpu_entries_per_lw"),
            ("VRAM hits/lw", "vram_hits_per_lw"), ("window ms", "window_ms"),
            ("tok/win", "tokens_per_window"), ("prefill tok/s", "prefill_tok_s"),
            ("decode tok/s", "decode_tok_s"), ("gen tok", "gen_tokens"),
            ("prompt s", "prompt_ms"), ("card0 GiB", "peak_card0_gib"), ("card1 GiB", "peak_card1_gib"),
            ("RSS GiB", "peak_rss_gib")]
    rows = []
    for t in tags:
        r = row(t)
        if r is None:
            rows.append({"arm": t, "error": "no summary"})
            continue
        rows.append(r)
    head = [c[0] for c in cols]
    print("\t".join(head))
    for r in rows:
        cells = []
        for _, k in cols:
            v = r.get(k)
            if k == "prompt_ms" and v is not None:
                v = round(v / 1000.0, 1)
            cells.append("-" if v is None else str(v))
        print("\t".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
