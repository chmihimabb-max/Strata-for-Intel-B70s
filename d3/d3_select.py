#!/usr/bin/env python3
"""D3 (card t_87aa2963): the decode window's census, READ FOR THE SELECTION (and for the per-layer handshake).

The instrument is D1's launch-site histogram (`STRATA_LAUNCH_HIST=1`, closure path: every launch carries a device
timestamp).  D2 used it for the dispatch map; this card uses the same file to answer three questions D2's family
roll-up could not:

  * how much of a window is `qsa_block_scores` + `qsa_block_topk` (the QSA selection) at 4K/32K/128K,
  * which of the two top-k kernels the decode path is taking (the register kernel's launcher and the ref kernel's
    launcher are distinct symbols),
  * how many per-layer host<->device handshakes a window costs (`wait_flag_ge`/`doorbell_*` on the device,
    `<memcpy>` on the host side).

Line format written by the shim (see src/kernels/sycl/launch_hist or sycl_compat):
  H\t<tag with win=N ...>\t<count>\t<us>\t<symbol>#<site>
  HW\t<tag>\t<submissions N>\t<sites N>
so a row's `count` is how many times that site launched inside that window and `us` their total device time.

usage: d3_select.py <hist.txt> [...] [--window auto|N] [--all] [--top N]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict


def short(name: str) -> str:
    """D2's normalization: strip namespaces, stop at the argument list, keep template args."""
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


def base(name: str) -> str:
    """The symbol without its template arguments (a launcher's instantiations should not split a row)."""
    s = short(name)
    return s.split("<")[0]


def parse(path: str):
    wins: dict[int, dict] = defaultdict(lambda: {"rows": [], "subs": 0, "sites": 0, "walls": [], "tag": ""})
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        m = re.search(r"win=(\d+)", f[1])
        if not m:
            continue
        w = int(m.group(1))
        wins[w]["tag"] = f[1]
        if f[0] == "HW":
            mm = re.search(r"submissions (\d+)", f[2])
            if mm:
                wins[w]["subs"] += int(mm.group(1))
            mm = re.search(r"sites (\d+)", f[2])
            if mm:
                wins[w]["sites"] += int(mm.group(1))
            wm = re.search(r"wall=([\d.]+)ms", f[1])
            if wm:
                wins[w]["walls"].append(float(wm.group(1)))
        elif f[0] == "H" and len(f) >= 5:
            wins[w]["rows"].append((int(f[2]), float(f[3]), f[4]))
    return wins


def win_T(tag: str) -> str:
    m = re.search(r"\bT=(\d+)", tag)
    return m.group(1) if m else "?"


SELECTION = ("qsa_block_scores", "qsa_block_topk", "qsa_index_step", "topk_512", "qsa_kv_resolve",
             "qsa_decode_attn", "native_qsa_indexer", "native_qsa_score", "qsa_attention")
HANDSHAKE = ("wait_flag", "doorbell", "flag_raise", "flag_wait")
INSTR = ("gpu_stamp", "sampler")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--window", default="auto", help="auto (largest by counted submissions) or a number")
    ap.add_argument("--all", action="store_true", help="every window in the file, not just the chosen one")
    ap.add_argument("--brief", action="store_true",
                    help="one line per file, for the growth table: the chosen window's ms by family")
    ap.add_argument("--top", type=int, default=14)
    o = ap.parse_args()

    if o.brief:
        print("%-34s %3s %9s %9s %9s %9s %9s %9s %9s" %
              ("hist.txt", "T", "window_ms", "scores", "topk", "attn", "sel%", "waitflag", "stamp"))
        for path in o.paths:
            wins = parse(path)
            if not wins:
                print("%-34s  no histogram lines" % path)
                continue
            w = max(wins, key=lambda x: sum(c for c, _, _ in wins[x]["rows"]))
            d = wins[w]
            tot_us = sum(us for _, us, _ in d["rows"])
            g = lambda p: sum(us for _, us, n in d["rows"] if base(n).startswith(p))
            sc, tk, at = g("qsa_block_scores"), g("qsa_block_topk"), g("qsa_decode_attn")
            wf, st = g("wait_flag"), g("gpu_stamp")
            print("%-34s %3s %9.2f %9.3f %9.3f %9.3f %8.2f%% %9.3f %9.3f" %
                  (path, win_T(d["tag"]), tot_us / 1000.0, sc / 1000.0, tk / 1000.0, at / 1000.0,
                   100.0 * (sc + tk) / tot_us if tot_us else 0.0, wf / 1000.0, st / 1000.0))
        return 0

    for path in o.paths:
        wins = parse(path)
        if not wins:
            print("%s: no histogram lines" % path)
            continue
        print("== %s" % path)
        # per-window summary, every window: this is the raw output the card's attribution reads
        print("   %-4s %-3s %9s %9s %10s %12s %12s %12s" %
              ("win", "T", "subs", "sites", "total_ms", "scores_ms", "topk_ms", "share%"))
        for w in sorted(wins):
            d = wins[w]
            tot = sum(c for c, _, _ in d["rows"])            # sites
            tot_us = sum(us for _, us, _ in d["rows"])
            sc = sum(us for _, us, n in d["rows"] if base(n) == "qsa_block_scores")
            tk = sum(us for _, us, n in d["rows"] if base(n) == "qsa_block_topk")
            print("   %-4d %-3s %9d %9d %10.2f %12.3f %12.3f %11.2f%%" %
                  (w, win_T(d["tag"]), d["subs"], d["sites"], tot_us / 1000.0, sc / 1000.0, tk / 1000.0,
                   100.0 * (sc + tk) / tot_us if tot_us else 0.0))
        if o.window == "auto":
            w = max(wins, key=lambda x: sum(c for c, _, _ in wins[x]["rows"]))
        else:
            w = int(o.window)
            if w not in wins:
                print("   no window %d" % w)
                continue
        order = sorted(wins) if o.all else [w]
        for w in order:
            d = wins[w]
            printed = sum(c for c, _, _ in d["rows"])
            tot_us = sum(us for _, us, _ in d["rows"])
            print()
            print("   ---- window %d: %s" % (w, d["tag"]))
            print("        counted sites %d, device time %.3f ms, wall %s ms, submissions %d" %
                  (printed, tot_us / 1000.0, max(d["walls"]) if d["walls"] else -1, d["subs"]))
            for label, keys in (("the selection", SELECTION), ("the per-layer handshake", HANDSHAKE),
                               ("the instrument itself", INSTR)):
                rows = [(c, us, n) for c, us, n in d["rows"] if base(n).startswith(keys)]
                if not rows:
                    continue
                print("        -- %s --" % label)
                agg: dict[str, list] = defaultdict(lambda: [0, 0.0])
                for c, us, n in rows:
                    agg[base(n)][0] += c
                    agg[base(n)][1] += us
                for name, v in sorted(agg.items(), key=lambda kv: -kv[1][1]):
                    print("           %-42s %6d launches %10.3f ms %6.2f%%" %
                          (name, v[0], v[1] / 1000.0, 100.0 * v[1] / tot_us if tot_us else 0.0))
            print("        -- the window's top families by device time --")
            agg2: dict[str, list] = defaultdict(lambda: [0, 0.0])
            for c, us, n in d["rows"]:
                agg2[base(n)][0] += c
                agg2[base(n)][1] += us
            for name, v in sorted(agg2.items(), key=lambda kv: -kv[1][1])[:o.top]:
                print("           %-42s %6d launches %10.3f ms %6.2f%%" %
                      (name, v[0], v[1] / 1000.0, 100.0 * v[1] / tot_us if tot_us else 0.0))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
