#!/usr/bin/env python3
"""P9: one row per arm, out of the arms' own logs/runs -- the table the report quotes.

usage: /usr/bin/python3 p9/p9_report.py <tag> [<tag> ...]      (default: every p9-* arm under ~/strata-xpu/p9/runs)

Per arm it reads
  runs/<tag>/out.txt   the request: PP lines, T lines (md5), DONE
  runs/<tag>/err.txt   the engine's decoded-window numbers: `strata decode timing`, `strata submit`
and prints one line per arm plus a per-arm block with the raw lines (so the report's table and its raw
output come from the same pass).
"""
import glob
import os
import re
import sys
from collections import defaultdict

R = "/home/michael/strata-xpu"
RUN = f"{R}/p9/runs"

DEC = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = "
    r"verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan ([\d.]+) actq ([\d.]+) "
    r"jobs ([\d.]+) CPU ([\d.]+)\] \+ stage ([\d.]+)(?: \+ tail ([\d.]+))?\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+); "
    r"per layer-window: CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)")
DEC_KEYS = ["windows", "avgT", "tokpw", "msgw", "verify", "wait", "host", "plan", "actq", "jobs", "cpu",
            "stage", "tail", "commit", "draft", "cpu_exp", "entries", "vram_hits", "pcie"]
SUB = re.compile(r"strata submit: (stage|single) window T=(\d+) pos0=(\d+) layers (\d+)\.\.(\d+) \((\d+)\): "
                 r"submitted (\d+) \(kernel (\d+) memset (\d+) memcpy (\d+) barrier (\d+) event (\d+) "
                 r"hostfn (\d+) graph (\d+)\) \+ recorded (\d+); ([\d.]+) per layer; GPU-reach wait ([\d.]+) ms")
DONE = re.compile(r"^DONE (\d+) (\d+) ([\d.]+) ([\d.]+) (\S+)", re.M)


def parse(tag):
    d = f"{RUN}/{tag}"
    out = open(f"{d}/out.txt", errors="replace").read() if os.path.exists(f"{d}/out.txt") else ""
    err = open(f"{d}/err.txt", errors="replace").read() if os.path.exists(f"{d}/err.txt") else ""
    log = open(f"{d}/log.txt", errors="replace").read() if os.path.exists(f"{d}/log.txt") else ""
    r = {"tag": tag}
    m = DEC.search(err)
    if m:
        for k, v in zip(DEC_KEYS, m.groups()):
            r[k] = (float(v) if "." in str(v) else int(v)) if v is not None else 0
    # the median window's submission composition: take the modal stage 0 / stage 1 line
    subs = defaultdict(list)
    for mm in SUB.finditer(err):
        st = mm.group(1) + mm.group(5)
        subs[st].append([int(x) for x in mm.groups()[6:15]] + [float(mm.group(16))])
    for st, rows in subs.items():
        rows.sort()
        med = rows[len(rows) // 2]
        r[f"sub_{st}"] = med
    tlines = re.findall(r"^T (\d+)", out, re.M)
    r["tokens"] = len(tlines)
    r["md5"] = (re.search(r"T lines: \d+\s+md5 (\w+)", log) or re.search(r"md5 (\w+)", log)).group(1)
    pp = re.findall(r"^PP (\d+) (\d+) ([\d.]+) ([\d.]+)", out, re.M)
    if pp:
        r["pp_ms"] = float(pp[-1][2])
        r["pp_toks"] = int(pp[-1][1])
    dm = DONE.search(out)
    if dm:
        r["done"] = [int(dm.group(1)), int(dm.group(2)), float(dm.group(3)), float(dm.group(4))]
        if float(dm.group(4)) > 0:
            r["dec_tok_s"] = int(dm.group(1)) / (float(dm.group(4)) / 1000.0)
        if float(dm.group(3)) > 0:
            r["pre_tok_s"] = int(dm.group(2)) / (float(dm.group(3)) / 1000.0)
    ln = re.search(r"ask finished=(\d) after (\d+) ms", log)
    r["answered"] = ln.group(1) == "1" if ln else None
    r["stalled"] = "timed out at layer" in err or "did not finish within 5 s" in err
    wm = re.search(r"engine exit (\S+), wall (\d+) s", log)
    r["wall"] = wm.group(2) if wm else ""
    return r


def main():
    tags = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(f"{RUN}/p9-*"))
    rows = [parse(t) for t in tags]
    hdr = f"{'arm':<17} {'win':>4} {'ms/win':>7} {'verify':>7} {'wait':>6} {'host':>5} {'stage':>6} {'tail':>6} " \
          f"{'commit':>6} {'draft':>6} {'vr_hit':>7} {'kern/w':>7} {'copy/w':>7} {'tok':>4} {'dec t/s':>7} " \
          f"{'pre t/s':>7} {'pp_s':>6} {'md5':<9} {'stall':>5}"
    print(hdr)
    for r in rows:
        s0 = r.get("sub_stage22")
        s1 = r.get("sub_stage47")
        kern = (s0[1] + s1[1]) if s0 and s1 else 0
        copy = (s0[3] + s1[3]) if s0 and s1 else 0
        print(f"{r['tag']:<17} {r.get('windows', 0):>4} {r.get('msgw', 0):>7.2f} {r.get('verify', 0):>7.2f} "
              f"{r.get('wait', 0):>6.2f} {r.get('host', 0):>5.2f} {r.get('stage', 0):>6.2f} {r.get('tail', 0):>6.2f} "
              f"{r.get('commit', 0):>6.2f} {r.get('draft', 0):>6.2f} {r.get('vram_hits', 0):>7.2f} "
              f"{kern:>7} {copy:>7} {r.get('tokens', 0):>4} {r.get('dec_tok_s', 0):>7.2f} {r.get('pre_tok_s', 0):>7.1f} "
              f"{r.get('pp_ms', 0) / 1000:>6.1f} "
              f"{r.get('md5', '')[:8]:<9} {str(r.get('stalled')):>5}")
    print()
    for r in rows:
        print(f"== {r['tag']}: answered={r.get('answered')} wall={r.get('wall')}s tokens={r.get('tokens')} "
              f"md5={r.get('md5')} pp={r.get('pp_ms')}ms/{r.get('pp_toks')}tok")
        if "sub_stage22" in r:
            print(f"   stage0 {r['sub_stage22']}   stage1 {r.get('sub_stage47')}")
        if "done" in r:
            print(f"   DONE {r['done']}")


if __name__ == "__main__":
    main()
