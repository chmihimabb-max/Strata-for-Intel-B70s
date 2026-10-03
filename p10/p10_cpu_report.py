#!/usr/bin/env python3
"""P10: turn p10_threads.py's per-thread CSV into the per-phase "how many cores actually compute" table.

usage: p10_cpu_report.py <arm_dir> [idx] [--prompt-ms MS] [--done-epoch E]

Reads <arm_dir>/threads.csv (or threads-<idx>.csv for the two-instance arms) and <arm_dir>/markers.tsv
(load_start, ready, task_start, request_done).  The prefill/decode boundary is the engine's own prompt time:
either from out.txt's DONE line (the direct serve arms: DONE <gen> <prompt> <prompt_ms> <decode_ms>) or from
--prompt-ms (the server arms, where the engine's prompt line is in its own log).

Reported per phase: cores busy (= sum of every thread's cpu%% / 100, averaged over the sampled seconds), how
many threads averaged >=50%% and >=10%% of a core, and the busiest threads.
"""
from __future__ import annotations

import csv
import json
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
    args = [a for a in sys.argv[1:]]
    prompt_ms = None
    if "--prompt-ms" in args:
        i = args.index("--prompt-ms")
        prompt_ms = float(args[i + 1])
        del args[i:i + 2]
    d = args[0].rstrip("/")
    idx = args[1] if len(args) > 1 else None
    csv_name = f"threads-{idx}.csv" if idx is not None else "threads.csv"
    m = load_markers(os.path.join(d, "markers.tsv"))
    ask = m.get("ask_start", m.get("task_start"))
    done = m.get("request_done")
    if ask is None or done is None:
        print(f"[cpu] {os.path.basename(d)}: markers missing task_start/request_done: have {sorted(m)}")
        return 1
    if prompt_ms is None:
        try:
            with open(os.path.join(d, "out.txt")) as f:
                for line in f:
                    if line.startswith("DONE"):
                        prompt_ms = float(line.split()[3])
        except OSError:
            pass
    if prompt_ms is None and idx is not None:
        # the server arms have no out.txt: the engine's own prompt time is echoed in the client's response
        try:
            t = json.loads(open(os.path.join(d, f"resp-{idx}.json.timing.json")).read())
            prompt_ms = (t.get("timings") or {}).get("prompt_ms")
        except (OSError, ValueError):
            pass
    if prompt_ms is None:
        print(f"[cpu] {os.path.basename(d)}: no prompt time (no DONE line and no --prompt-ms)")
        return 1
    prompt_done = ask + prompt_ms / 1000.0
    phases = [
        ("load", m.get("load_start", ask - 60), m.get("ready", ask)),
        ("prefill", ask, prompt_done),
        ("decode", prompt_done, done),
        ("whole request", ask, done),
    ]
    rows = []
    with open(os.path.join(d, csv_name)) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["epoch"]), r["tid"], r["comm"], float(r["cpu_pct"])))
    print(f"[cpu] {os.path.basename(d)}/{csv_name}: {len(rows)} thread-ticks, "
          f"{len({r[1] for r in rows})} distinct threads; prompt {prompt_ms/1000.0:.1f}s")
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
