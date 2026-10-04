#!/usr/bin/env python3
"""D3 (card t_87aa2963): the engine's own two lines per arm, side by side - the `strata decode timing` window
account and the P9 host-sampled `strata decode GPU stages` table (the one that prints the QSA selection as
`scores+topk`).  Every number is the engine's; this only lines them up.

usage: d3_stages.py <runs root> [tag ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def one(tag: str, d: str):
    err = os.path.join(d, "err.txt")
    if not os.path.exists(err):
        return None
    e = open(err, errors="replace").read()
    r = {"tag": tag}
    dt = None
    for line in e.splitlines():
        if line.startswith("strata decode timing:"):
            dt = line
    r["dt"] = dt or ""
    if dt:
        g = lambda p, c=float: (c(re.search(p, dt).group(1)) if re.search(p, dt) else None)
        r["windows"] = g(r"(\d+) windows", int)
        r["avgT"] = g(r"avg T ([\d.]+)")
        r["ms"] = g(r"([\d.]+) ms/window")
        r["verify"] = g(r"= verify ([\d.]+)")
        r["wait"] = g(r"GPU-reach wait ([\d.]+)")
        r["host"] = g(r"per-layer host ([\d.]+)")
        r["stage"] = g(r"\+ stage ([\d.]+)")
        r["tail"] = g(r"tail ([\d.]+)\)")
        r["commit"] = g(r"commit/emit ([\d.]+)")
        r["draft"] = g(r"\+ draft ([\d.]+)")
        r["vram"] = g(r"VRAM hits ([\d.]+)")
        r["cpu"] = g(r"CPU experts ([\d.]+)")
        r["entries"] = g(r"\(([\d.]+) entries\)")
        r["tokens"] = g(r"([\d.]+) tokens/window")
    st = None
    for line in e.splitlines():
        if line.startswith("strata decode GPU stages"):
            st = line
    r["stages"] = st or ""
    if st:
        m = re.search(r"QSA layers:(.*?)\|", st)
        qsa = m.group(1) if m else ""
        r["qsa_scores_topk"] = (re.search(r"scores\+topk ([\d.]+)", qsa) or [None, None])[1]
        r["qsa_attn"] = (re.search(r"attention ([\d.]+)", qsa) or [None, None])[1]
        r["qsa_qqidx"] = (re.search(r"q\+q-idx ([\d.]+)", qsa) or [None, None])[1]
        r["total"] = (re.search(r"total ([\d.]+) ms/window", st) or [None, None])[1]
    for k in ("qsa_scores_topk", "qsa_attn", "qsa_qqidx", "total"):
        r.setdefault(k, None)
    done = [l for l in open(os.path.join(d, "out.txt"), errors="replace").read().splitlines()
            if l.startswith("DONE")] if os.path.exists(os.path.join(d, "out.txt")) else []
    if done:
        f = done[-1].split()
        r["gen"], r["acc"], r["off"] = int(f[1]), int(f[6]), int(f[7])
        r["dec_ms"] = float(f[4])
        r["tok_s"] = 1000.0 * r["gen"] / r["dec_ms"] if r["dec_ms"] else None
    return r


def fmt(v, nd=2):
    return "-" if v is None else (("%." + str(nd) + "f") % v if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("tags", nargs="*")
    o = ap.parse_args()
    tags = o.tags or sorted(os.path.basename(p) for p in glob.glob(os.path.join(o.root, "*")) if os.path.isdir(p))
    print("%-20s %5s %5s %8s %8s %8s %8s %8s %8s %7s | %-24s %8s %8s %8s" %
          ("tag", "win", "avgT", "ms/win", "verify", "wait", "tail", "commit", "draft", "tok/s",
           "P9 stage: q+q-idx/scores+topk/attn", "total", "scoresTK", "attn"))
    for t in tags:
        r = one(t, os.path.join(o.root, t))
        if not r:
            continue
        print("%-20s %5s %5s %8s %8s %8s %8s %8s %8s %7s | %-24s %8s %8s %8s" %
              (r["tag"], fmt(r.get("windows"), 0), fmt(r.get("avgT")), fmt(r.get("ms")), fmt(r.get("verify")),
               fmt(r.get("wait")), fmt(r.get("tail")), fmt(r.get("commit")), fmt(r.get("draft")),
               fmt(r.get("tok_s")), r.get("qsa_qqidx") or "-", fmt(r.get("total")),
               fmt(r.get("qsa_scores_topk"), 3), fmt(r.get("qsa_attn"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
