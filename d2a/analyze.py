#!/usr/bin/env python3
"""D2a: the generic multi-column arm's own numbers, per run, with the part of the window the engine does not name.

    verify = wait + per-layer host + stage + tail + RESIDUAL

The residual is real time inside the window that no named counter owns; on the shipped layout it is ~3 ms, and
the D2 generic runs had 67.29 / 21.27 / 2.42 ms of it while the same three runs wrote 25 / 5 / 0 new SYCL program
files during their decode (d2a/arm_cache.py).  Prints the ids md5 exactly as d2/d2_report.py computes it, and the
window-size sequence (from the arm's own submit lines), so two runs can be compared window by window.
"""
import glob
import hashlib
import os
import re
import subprocess
import sys

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")
CACHE = "/home/michael/strata-xpu/sycl-cache/m6c/13665086051514919768"

TIMING = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = "
    r"verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[[^]]*\] \+ stage ([\d.]+) \+ "
    r"tail ([\d.]+)\)")
SUB = re.compile(r"strata submit: (?:stage|single) window T=(\d+) ")


def cache_times():
    out = subprocess.run(["find", CACHE, "-name", "0.src", "-printf", "%T@\n"],
                         capture_output=True, text=True).stdout.split()
    return sorted(float(x) for x in out)


def arm(tag, cache):
    log = os.path.join(RUNS, tag, "log.txt")
    err = os.path.join(RUNS, tag, "err.txt")
    if not os.path.exists(log):
        return None
    lg = open(log, errors="replace").read()
    m = TIMING.search(lg)
    if not m:
        return None
    win, avgT, tokwin, mswin, ver, wait, host, stage, tail = (float(x) for x in m.groups())
    residual = ver - (wait + host + stage + tail)
    ids = None
    ntok = -1
    out = os.path.join(RUNS, tag, "out.txt")
    if os.path.exists(out):
        ts_lines = [l for l in open(out, errors="replace").read().splitlines() if l.startswith("T ")]
        ntok = len(ts_lines)
        if ts_lines:
            ids = hashlib.md5("\n".join(ts_lines).encode()).hexdigest()[:32]
    ts = [int(x) for x in SUB.findall(open(err, errors="replace").read())][::2]
    fed = re.search(r"ask fed at .*epoch_ms (\d+)", lg)
    fin = re.search(r"the ask finished=1 after (\d+) ms", lg)
    inside = 0
    if fed and fin:
        a = float(fed.group(1)) / 1000.0
        b = a + float(fin.group(1)) / 1000.0
        inside = sum(1 for x in cache if a - 0.5 <= x <= b + 0.5)
    return dict(tag=tag, win=int(win), avgT=avgT, tokwin=tokwin, mswin=mswin, ver=ver, wait=wait, host=host,
                stage=stage, tail=tail, residual=residual, tokens=ntok,
                ids=ids if ids else "-",
                tseq=hashlib.md5((",".join(map(str, ts))).encode()).hexdigest()[:8],
                twide=sum(1 for x in ts if x > 4), caches=inside)


GROUPS = [
    ("D2 originals", ["d2-rebase-4096", "d2-mg-4096", "d2-mg-4096-warm", "d2-mg-4096-warm2"]),
    ("session control", ["d2a-rebase-4096"]),
    ("A: the lever, 8 repeats", [f"d2a-mg-4096-r{i}" for i in range(1, 9)]),
    ("B: the lever + --suffix-draft 0, 8 repeats", [f"d2a-mg0-4096-r{i}" for i in range(1, 9)]),
]


def main():
    cache = cache_times()
    rows = {}
    for _, tags in GROUPS:
        for t in tags:
            r = arm(t, cache)
            if r:
                rows[t] = r
    hdr = (f"{'tag':22s} {'win':>4s} {'ms/win':>8s} {'verify':>7s} {'wait':>7s} {'tail':>6s} {'RESID':>7s} "
           f"{'res*w':>7s} {'tokens':>6s} {'T>4':>3s} {'newprog':>7s} Tseq    ids md5")
    for name, tags in GROUPS:
        print(f"\n=== {name}")
        print(hdr)
        for t in tags:
            r = rows.get(t)
            if not r:
                print(f"{t:22s} (no timing line yet)")
                continue
            print(f"{r['tag']:22s} {r['win']:4d} {r['mswin']:8.2f} {r['ver']:7.2f} {r['wait']:7.2f} "
                  f"{r['tail']:6.2f} {r['residual']:7.2f} {r['residual']*r['win']:7.0f} {r['tokens']:6d} "
                  f"{r['twide']:3d} {r['caches']:7d} {r['tseq']:7s} {r['ids']}")
        got = [rows[t] for t in tags if t in rows]
        if len(got) > 1:
            ids = sorted({r["ids"] for r in got})
            tseq = sorted({r["tseq"] for r in got})
            ms = [r["mswin"] for r in got]
            res = [r["residual"] for r in got]
            print(f"  -> {len(got)} runs: ms/window {min(ms):.2f}-{max(ms):.2f} "
                  f"(spread {100*(max(ms)-min(ms))/min(ms):.1f}%), residual {min(res):.2f}-{max(res):.2f} ms, "
                  f"{len(ids)} distinct ids md5, {len(tseq)} distinct window sequences, "
                  f"new programs {sum(r['caches'] for r in got)}")


if __name__ == "__main__":
    sys.exit(main())
