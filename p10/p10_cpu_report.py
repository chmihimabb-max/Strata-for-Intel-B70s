#!/usr/bin/env python3
"""P10: turn p10_threads.py's per-thread CSV into the per-phase "how many cores actually compute" table.

usage: p10_cpu_report.py <arm_dir>

Reads <arm_dir>/threads.csv and <arm_dir>/markers.tsv (load_start, ready, task_start/ask_start, request_done)
and <arm_dir>/out.txt for the engine's own DONE line, from which the prefill/decode boundary is taken
(DONE <n_gen> <prompt_tokens> <prompt_ms> <decode_ms>): prompt_done = task_start + prompt_ms.

Reported per phase: cores busy (= sum of every thread's cpu%% / 100, averaged over the sampled seconds),
how many threads averaged >=50%% and >=10%% of a core, and the busiest threads.
"""
from __future__ import annotations

import csv
import os
import sys


def load_markers(path):
    m = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split("\t")[:2]
            m[b] = float(a)
    return m


def main() -> int:
    d = sys.argv[1].rstrip("/")
    m = load_markers(os.path.join(d, "markers.tsv"))
    ask = m.get("ask_start", m.get("task_start"))
    done = m.get("request_done")
    if ask is None or done is None:
        print(f"[cpu] markers missing ask_start/request_done: have {sorted(m)}")
        return 1
    pms = None
    try:
        with open(os.path.join(d, "out.txt")) as f:
            for line in f:
                if line.startswith("DONE"):
                    p = line.split()
                    pms = float(p[3])
    except OSError:
        pass
    if pms is None:
        print("[cpu] no DONE line: cannot split prefill from decode")
        return 1
    prompt_done = ask + pms / 1000.0
    phases = [
        ("load", m.get("load_start", ask - 60), m.get("ready", ask)),
        ("prefill", ask, prompt_done),
        ("decode", prompt_done, done),
        ("whole request", ask, done),
    ]
    rows = []
    with open(os.path.join(d, "threads.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["epoch"]), r["tid"], r["comm"], float(r["cpu_pct"])))
    print(f"[cpu] {d}: {len(rows)} thread-ticks, {len({r[1] for r in rows})} distinct threads, "
          f"ask_start {ask:.1f}, prompt_done {prompt_done:.1f} (+{pms/1000.0:.1f}s), request_done {done:.1f}")
    for name, t0, t1 in phases:
        sel = [r for r in rows if t0 <= r[0] < t1]
        if not sel:
            print(f"{name:14s} no samples")
            continue
        ticks = len({round(r[0], 3) for r in sel})
        cores = sum(r[3] for r in sel) / 100.0 / max(ticks, 1)
        thr = {}
        for r in sel:
            thr.setdefault(r[1], []).append(r[3])
        means = {t: sum(v) / len(v) for t, v in thr.items()}
        hot = sorted(means.items(), key=lambda kv: -kv[1])
        n50 = sum(1 for v in means.values() if v >= 50.0)
        n10 = sum(1 for v in means.values() if v >= 10.0)
        print(f"{name:14s} {t1-t0:7.1f}s  cores busy {cores:5.2f}   threads>=50%: {n50:3d}  >=10%: {n10:3d}  "
              f"of {len(means)} alive")
        print("               busiest: " + ", ".join(f"{t}:{v:.0f}%" for t, v in hot[:22]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
