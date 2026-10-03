#!/usr/bin/env python3
"""D1 (card t_d9ffcf38): one row per arm, read out of the arm's OWN logs.

Every number here is the engine's own, printed on its stderr:
  * `strata decode timing: ...`  -> windows, avg T, tokens/window, ms/window, verify and its parts, draft
  * `DONE <gen> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> ...`  -> decode tok/s,
    acceptance
  * `strata submit: ...`         -> the window's submissions and their kinds
  * `PP <...>`                   -> the prompt pass

usage: d1_report.py [--root /home/michael/strata-xpu/d1/runs] [tag ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def one(tag: str, d: str) -> dict | None:
    err = os.path.join(d, "err.txt")
    out = os.path.join(d, "out.txt")
    log = os.path.join(d, "log.txt")
    if not os.path.exists(err):
        return None
    e = open(err, errors="replace").read()
    o = open(out, errors="replace").read() if os.path.exists(out) else ""
    lg = open(log, errors="replace").read() if os.path.exists(log) else ""
    r: dict = {"tag": tag, "dir": d}

    m = re.search(r"engine binary: .*md5 ([0-9a-f]+)", lg)
    r["bin_md5"] = m.group(1) if m else "?"
    m = re.search(r"spec=(\d+) spec-min-p=([\d.]+) mtp-max-t=(\S+) graph=(\S+)", lg)
    if m:
        r["spec"], r["spec_min_p"], r["mtp_max_t"], r["graph"] = m.groups()
    else:   # an arm whose log predates the rig line: take it from the command echoed in the log
        m = re.search(r"--spec (\d+)", lg)
        r["spec"] = m.group(1) if m else "?"
        r["graph"] = "1" if "STRATA_SYCL_GRAPH=1" in lg else ("0" if "STRATA_SYCL_GRAPH=0" in lg else "default")
        r["spec_min_p"] = (re.search(r"--spec-min-p ([\d.]+)", lg) or [None, "?"])[1]
        r["mtp_max_t"] = (re.search(r"--mtp-max-t (\d+)", lg) or [None, "-"])[1]
    m = re.search(r"ctx=(\d+) max-new=(\d+)", lg)
    r["ctx"], r["maxnew"] = (m.group(1), m.group(2)) if m else ("?", "?")
    m = re.search(r"hist=(\d+)/(\d+)", lg)
    r["hist"] = m.group(1) if m else "0"

    # the graph banner (which path ran).  The PRE-CHANGE binary (md5 9eff0675) has no banner when the flag is
    # unset - STRATA_SYCL_GRAPH was opt-in, so that silent case IS the closure path.
    r["path"] = "?"
    if "the graph path is ON (default since D1" in e:
        r["path"] = "graph(default)"
    elif "STRATA_SYCL_GRAPH=0" in e:
        r["path"] = "closure(env=0)"
    elif "STRATA_SYCL_GRAPH=1" in e:
        r["path"] = "graph(env=1)"
    elif r["bin_md5"] == "9eff0675059cf7e8145ae7b4bca984a4":
        r["path"] = "closure(pre-change)"
    else:
        r["path"] = "closure(no banner)"

    # the decode timing line
    dt = None
    for line in e.splitlines():
        if line.startswith("strata decode timing:"):
            dt = line
    r["decode_line"] = dt or ""
    if dt:
        def g(pat, cast=float, default=None):
            mm = re.search(pat, dt)
            return cast(mm.group(1)) if mm else default
        r["windows"] = g(r"(\d+) windows", int, 0)
        r["avg_T"] = g(r"avg T ([\d.]+)")
        r["tok_per_window"] = g(r"([\d.]+) tokens/window")
        r["ms_window"] = g(r"([\d.]+) ms/window")
        r["verify_ms"] = g(r"= verify ([\d.]+)")
        r["wait_ms"] = g(r"GPU-reach wait ([\d.]+)")
        r["tail_ms"] = g(r"tail ([\d.]+)\)")
        r["commit_ms"] = g(r"commit/emit ([\d.]+)")
        r["draft_ms"] = g(r"\+ draft ([\d.]+)")
        r["vram_hits"] = g(r"VRAM hits ([\d.]+)")
        r["cpu_experts"] = g(r"CPU experts ([\d.]+)")
    else:
        for k in ("windows", "avg_T", "tok_per_window", "ms_window", "verify_ms", "wait_ms", "tail_ms",
                  "commit_ms", "draft_ms", "vram_hits", "cpu_experts"):
            r[k] = None

    # the submit line: the LAST window's, per stage
    subs = [l for l in e.splitlines() if l.startswith("strata submit:")]
    r["submit_lines"] = subs[-2:] if subs else []
    if subs:
        last = subs[-1]
        mm = re.search(r"submitted (\d+)", last)
        r["submitted_last"] = int(mm.group(1)) if mm else None
        mm = re.search(r"graph (\d+)\)", last)
        r["graph_subs"] = int(mm.group(1)) if mm else None
        mm = re.search(r"kernel (\d+) memset (\d+) memcpy (\d+)", last)
        r["kernels_last"], r["memsets_last"], r["memcpys_last"] = (map(int, mm.groups()) if mm else (None,) * 3)

    # the request's own DONE line
    done = [l for l in o.splitlines() if l.startswith("DONE")]
    r["done"] = done[-1] if done else ""
    if done:
        f = done[-1].split()
        r["gen_n"] = int(f[1])
        r["prompt_tokens"] = int(f[2])
        r["prompt_ms"] = float(f[3])
        r["decode_ms"] = float(f[4])
        r["finish"] = f[5]
        r["accepted"] = int(f[6])
        r["offered"] = int(f[7])
        r["decode_tok_s"] = (1000.0 * r["gen_n"] / r["decode_ms"]) if r["decode_ms"] > 0 else None
        r["prefill_tok_s"] = (1000.0 * r["prompt_tokens"] / r["prompt_ms"]) if r["prompt_ms"] > 0 else None
        r["accept_rate"] = (r["accepted"] / r["offered"]) if r["offered"] else None
    else:
        for k in ("gen_n", "prompt_tokens", "prompt_ms", "decode_ms", "finish", "accepted", "offered",
                  "decode_tok_s", "prefill_tok_s", "accept_rate"):
            r[k] = None
    ts = [l for l in o.splitlines() if l.startswith("T ")]
    r["t_lines"] = len(ts)
    import hashlib
    r["ids_md5"] = hashlib.md5("\n".join(ts).encode()).hexdigest()[:32] if ts else "-"
    return r


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/michael/strata-xpu/d1/runs")
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--full", action="store_true", help="also print every arm's decode-timing and submit lines")
    o = ap.parse_args()
    tags = o.tags or sorted(os.path.basename(p) for p in glob.glob(os.path.join(o.root, "*")) if os.path.isdir(p))
    rows = []
    for t in tags:
        r = one(t, os.path.join(o.root, t))
        if r:
            rows.append(r)
    hdr = ("tag", "spec", "gr", "path", "ctx", "win", "avgT", "tok/win", "ms/win", "dec tok/s", "pre tok/s",
           "acc/off", "rate", "ids md5")
    print("%-26s %4s %4s %-16s %7s %4s %5s %8s %7s %9s %8s %9s %5s %s" % hdr)
    for r in rows:
        acc = "%s/%s" % (fmt(r["accepted"]), fmt(r["offered"]))
        print("%-26s %4s %4s %-16s %7s %4s %5s %8s %7s %9s %8s %9s %5s %s" % (
            r["tag"], r["spec"], r["graph"], r["path"], r["ctx"], fmt(r["windows"], 0), fmt(r["avg_T"]),
            fmt(r["tok_per_window"]), fmt(r["ms_window"]), fmt(r["decode_tok_s"]), fmt(r["prefill_tok_s"]),
            acc, fmt(r["accept_rate"], 3), r["ids_md5"]))
    if o.full:
        print()
        for r in rows:
            print("== %s (bin %s, hist=%s)" % (r["tag"], r["bin_md5"], r["hist"]))
            print("   " + (r["decode_line"] or "<no decode timing line>"))
            for s in r["submit_lines"]:
                print("   " + s)
            print("   " + (r["done"] or "<no DONE line>"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
