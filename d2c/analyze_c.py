#!/usr/bin/env python3
"""D2c (card t_c7d8cd86): per arm, the residual (the part of a window no counter owns), the programs the arm
built inside its own ask, the programs it built inside its DECODE phase, the two warm-up lines, the load wall and
the greedy-ids md5 -- D2b's analyze_b.py plus (a) the decode/prompt split of the in-ask builds, (b) the second
warm-up's own line, (c) the start -> first-answer total this card is judged on.

usage: analyze_c.py <arm-tag> [...] | --all-cold | --all-warm | --all
"""
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
NEO_RE = re.compile(r"NEO_CACHE_DIR=(\S+) \((\d+|n/a) files\)")
MMVQ_RE = re.compile(r"strata mmvq warmup: (?:(\d+) launches[^,]*,\s*([\d.]+) ms|off)")
KERNEL_RE = re.compile(r"strata kernel warmup: (?:(\d+) launches over (\d+) sites,\s*([\d.]+) ms|off)")


def entries(cache):
    if not cache or not os.path.isdir(cache):
        return []
    out = subprocess.run(["find", cache, "-name", "0.src", "-printf", "%T@ %h\\n"],
                         capture_output=True, text=True).stdout.split("\n")
    got = []
    for line in out:
        if line.strip():
            t, d = line.split(" ", 1)
            got.append((float(t), d))
    return sorted(got)


def arm(tag):
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
    er = open(os.path.join(RUNS, tag, "err.txt"), errors="replace").read()
    fed = re.search(r"ask fed at .*epoch_ms (\d+)", lg)
    fin = re.search(r"the ask finished=1 after (\d+) ms", lg)
    pre = None
    pm = re.search(r"prompt \d+ tokens = \d+ reused \+ \d+ read in (\d+) ms", er)
    built_all, built_dec, built_prompt = [], [], []
    if fed and fin:
        a = float(fed.group(1)) / 1000.0
        b = a + float(fin.group(1)) / 1000.0
        if pm:
            pre = a + float(pm.group(1)) / 1000.0
        for t, d in entries(cache):
            if a - 0.5 <= t <= b + 0.5:
                built_all.append((t, d))
                (built_dec if pre is not None and t >= pre else built_prompt).append((t, d))
    out = os.path.join(RUNS, tag, "out.txt")
    tokens, ids = -1, "-"
    if os.path.exists(out):
        ts = [l for l in open(out, errors="replace").read().splitlines() if l.startswith("T ")]
        tokens = len(ts)
        if ts:
            ids = hashlib.md5("\n".join(ts).encode()).hexdigest()[:32]
    load = re.search(r"loaded=1 after (\d+)s", lg)
    mmvq = MMVQ_RE.search(er)
    kern = KERNEL_RE.search(er)
    return dict(tag=tag, win=int(win), mswin=mswin, ver=ver, wait=wait, tail=tail, residual=residual,
                tokens=tokens, ids=ids, all_=len(built_all), dec=len(built_dec), prompt=len(built_prompt),
                dec_dirs=[d for _, d in built_dec], cache=cache,
                load=int(load.group(1)) if load else -1,
                ask=float(fin.group(1)) if fin else -1,
                mmvq=(float(mmvq.group(2)) if mmvq and mmvq.group(2) else ("off" if mmvq else "-")),
                kern=(float(kern.group(3)) if kern and kern.group(3) else ("off" if kern else "-")),
                klaunches=(int(kern.group(1)) if kern and kern.group(1) else 0),
                ksites=(int(kern.group(2)) if kern and kern.group(2) else 0))


def show(tag):
    r = arm(tag)
    if not r:
        print(f"{tag:26s} (no timing line yet)")
        return None
    first = r["load"] * 1000 + r["ask"] if r["load"] > 0 and r["ask"] > 0 else -1
    print(f"{r['tag']:26s} win {r['win']:3d}  ms/win {r['mswin']:7.2f}  verify {r['ver']:7.2f}  "
          f"RESID {r['residual']:7.2f} ({r['residual']*r['win']:6.0f} ms)  tokens {r['tokens']:4d}  "
          f"ids {r['ids']}")
    print(f"{'':26s}   programs in the ask {r['all_']:3d}  (prompt {r['prompt']:3d} + DECODE {r['dec']:3d})  "
          f"load {r['load']:3d}s  ask {r['ask']/1000:6.2f}s  start->answer {first/1000:6.2f}s")
    print(f"{'':26s}   mmvq warmup {r['mmvq']!s:>8}   kernel warmup {r['kern']!s:>8}"
          + (f"  ({r['klaunches']} launches / {r['ksites']} sites)" if r["klaunches"] else ""))
    return r


GROUPS = {
    "cold": ["d2c-nooff-cold-4096", "d2c-mmoff-cold-4096", "d2c-wu-cold-4096"],
    "warm": ["d2c-wu-4096", "d2c-mmoff-4096", "d2c-wu-32768"],
}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all-cold" in sys.argv:
        argv += GROUPS["cold"]
    if "--all-warm" in sys.argv:
        argv += GROUPS["warm"]
    if "--all" in sys.argv:
        argv += GROUPS["cold"] + GROUPS["warm"]
    if not argv:
        argv = GROUPS["cold"] + GROUPS["warm"]
    for tag in argv:
        show(tag)


if __name__ == "__main__":
    main()
