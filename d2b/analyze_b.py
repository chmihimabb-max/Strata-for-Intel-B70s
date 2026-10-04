#!/usr/bin/env python3
"""D2b (card t_f93760a1): per arm, the part of the window no counter owns, the programs built during the ask,
and the greedy-ids md5 -- with the cache directories the arm actually used.

    verify = wait + per-layer host + stage + tail + RESIDUAL      (d2a/analyze.py's arithmetic, unchanged)

usage: analyze_b.py <arm-tag> [<arm-tag> ...]        # cache dir taken from each arm's own log.txt
       analyze_b.py --all-warm / --all-cold           # the two chains of this card

For every arm it prints: ms/window, residual ms/window and residual x windows (the run's own total), the number
of SYCL program files (0.src) written between the ask and the finish, the number of NEO/IGC cache files added,
the window-size sequence and the ids md5 exactly as d2/d2_report.py computes it, and the load wall
(arm start -> "everything loaded").  With --names it also names the programs built inside the ask, from the
mangled symbols in each entry's 0.src (d2a/program_names.sh's rule).
"""
import glob
import hashlib
import os
import re
import subprocess
import sys

SRC = "/home/michael/strata-xpu/strata"
RUNS = os.path.join(SRC, "d2/runs")

TIMING = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = "
    r"verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[[^]]*\] \+ stage ([\d.]+) \+ "
    r"tail ([\d.]+)\)")
SUB = re.compile(r"strata submit: (?:stage|single) window T=(\d+) ")
CACHE_RE = re.compile(r"SYCL_CACHE_DIR=(\S+) \((\d+) entries\)")
NEO_RE = re.compile(r"NEO_CACHE_DIR=(\S+) \((\d+|\S+) files\)")
WARMUP_RE = re.compile(r"strata mmvq warmup: (\d+) launches[^,]*,\s*([\d.]+) ms")


def entries(cache, name="0.src"):
    if not cache or not os.path.isdir(cache):
        return []
    out = subprocess.run(["find", cache, "-name", name, "-printf", "%T@ %h\\n"],
                         capture_output=True, text=True).stdout.split("\n")
    got = []
    for line in out:
        if line.strip():
            t, d = line.split(" ", 1)
            got.append((float(t), d))
    return sorted(got)


def neo_count(d):
    if not d or not os.path.isdir(d):
        return None
    return int(subprocess.run(["bash", "-c", f"find {d!r} -type f | wc -l"],
                              capture_output=True, text=True).stdout.strip() or 0)


def sym_of(d):
    """the specialization name: the mangled lambda type inside strata::sycl_compat::launch"""
    src = os.path.join(d, "0.src")
    try:
        blob = subprocess.run(["strings", "-a", src], capture_output=True, text=True).stdout
    except OSError:
        return "?"
    m = re.search(r"native_mmvq[a-zA-Z0-9_]*E[JA-Za-z0-9_]*E\{0,3\}", blob)
    if m:
        return m.group(0)
    m = re.search(r"native_[a-z0-9_]*(?:multi_)?kernel", blob)
    return m.group(0) if m else "?"


def arm(tag, names=False):
    log = os.path.join(RUNS, tag, "log.txt")
    if not os.path.exists(log):
        return None
    lg = open(log, errors="replace").read()
    m = TIMING.search(lg)
    if not m:
        return None
    win, avgT, tokwin, mswin, ver, wait, host, stage, tail = (float(x) for x in m.groups())
    residual = ver - (wait + host + stage + tail)
    cm = CACHE_RE.search(lg)
    cache = cm.group(1) if cm else ""
    sycl_before = int(cm.group(2)) if cm else 0
    nm = NEO_RE.search(lg)
    neo = nm.group(1) if nm else ""
    census = re.search(r"cache census: SYCL 0\.src entries (\d+) -> (\d+) \(\+(\d+)\); NEO files (\S+) -> (\S+) \(\+(\S+)\)", lg)
    out = os.path.join(RUNS, tag, "out.txt")
    tokens, ids = -1, "-"
    if os.path.exists(out):
        ts = [l for l in open(out, errors="replace").read().splitlines() if l.startswith("T ")]
        tokens = len(ts)
        if ts:
            ids = hashlib.md5("\n".join(ts).encode()).hexdigest()[:32]
    ws = [int(x) for x in SUB.findall(open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read())][::2]
    fed = re.search(r"ask fed at .*epoch_ms (\d+)", lg)
    fin = re.search(r"the ask finished=1 after (\d+) ms", lg)
    built = []
    if fed and fin:
        a = float(fed.group(1)) / 1000.0
        b = a + float(fin.group(1)) / 1000.0
        built = [(t, d) for t, d in entries(cache) if a - 0.5 <= t <= b + 0.5]
    load = re.search(r"loaded=1 after (\d+)s", lg)
    wu = WARMUP_RE.search(lg)
    return dict(tag=tag, win=int(win), avgT=avgT, tokwin=tokwin, mswin=mswin, ver=ver, wait=wait, host=host,
                stage=stage, tail=tail, residual=residual, tokens=tokens, ids=ids,
                tseq=hashlib.md5((",".join(map(str, ws))).encode()).hexdigest()[:8],
                twide=sum(1 for x in ws if x > 4), built=built, cache=cache,
                sycl_before=sycl_before, census=census.groups() if census else None, neo=neo,
                load=int(load.group(1)) if load else -1,
                warmup=(int(wu.group(1)), float(wu.group(2))) if wu else None,
                names=[sym_of(d) for _, d in built] if names else [])


def show(tag, names=False):
    r = arm(tag, names)
    if not r:
        print(f"{tag:24s} (no timing line yet)")
        return None
    built = len(r["built"])
    neo = r["census"][5] if r["census"] else "?"
    print(f"{r['tag']:24s} win {r['win']:3d}  ms/win {r['mswin']:7.2f}  verify {r['ver']:7.2f}  "
          f"RESID {r['residual']:7.2f} ({r['residual']*r['win']:6.0f} ms total)  tokens {r['tokens']:4d}  "
          f"T>4 {r['twide']:2d}  Tseq {r['tseq']}  ids {r['ids']}  load {r['load']:4d}s  "
          f"programs+{built} (neo {neo})")
    if r["warmup"]:
        print(f"{'':24s}   warm-up: {r['warmup'][0]} launches, {r['warmup'][1]:.0f} ms")
    if names and r["built"]:
        for (t, _), n in zip(r["built"], r["names"]):
            import datetime as dt
            print(f"{'':24s}   +{t - r['built'][0][0]:6.1f}s {n}")
    return r


GROUPS = {
    "warm": ["d2b-rebase-4096", "d2b-wu-4096", "d2b-ctloff-4096", "d2b-wu-32768"],
    "cold": ["d2b-cold-4096", "d2b-ctloff-cold-4096", "d2b-wu-cold-4096"],
}


def main():
    names = "--names" in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all-warm" in sys.argv:
        argv += GROUPS["warm"]
    if "--all-cold" in sys.argv:
        argv += GROUPS["cold"]
    if not argv:
        argv = GROUPS["cold"] + GROUPS["warm"]
    for tag in argv:
        show(tag, names)


if __name__ == "__main__":
    sys.exit(main())
