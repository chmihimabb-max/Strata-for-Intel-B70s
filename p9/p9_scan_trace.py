#!/usr/bin/env python3
"""P9: what is actually inside a unitrace chrome timeline from this engine.

Streams a chrome-trace JSON (these are 0.6-1.3 GB, one event per line) and reports the shape of the
file before anything is attributed: the HOST vs DEVICE pseudo-processes, their threads, the event
categories, and -- the question -- whether the DEVICE rows carry a KERNEL name or only the
`zeCommandListAppend*` call name.

usage: /usr/bin/python3 p9/p9_scan_trace.py <trace.json> [topN]
"""
import json
import re
import sys
from collections import Counter, defaultdict

NAMES = ("name", "cat", "pid", "tid", "dur", "ts")


def main():
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    n_events = 0
    by_cat = Counter()
    by_pid_tid = Counter()
    proc_names = {}
    thread_names = {}
    dur_by_name = defaultdict(float)
    cnt_by_name = Counter()
    dur_by_cat = defaultdict(float)
    meta = 0
    first_ts = None
    last_ts = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            if '"ph"' not in line:
                continue
            try:
                e = json.loads(line.rstrip().rstrip(","))
            except Exception:
                continue
            ph = e.get("ph")
            if ph in ("M", "C"):
                meta += 1
                if ph == "M":
                    a = e.get("args", {})
                    if e.get("name") == "process_name":
                        proc_names[e.get("pid")] = a.get("name")
                    elif e.get("name") == "thread_name":
                        thread_names[(e.get("pid"), e.get("tid"))] = a.get("name")
                continue
            if ph != "X":
                continue
            n_events += 1
            cat = e.get("cat")
            name = e.get("name", "?")
            dur = float(e.get("dur", 0.0))
            ts = float(e.get("ts", 0.0))
            by_cat[cat] += 1
            dur_by_cat[cat] += dur
            by_pid_tid[(e.get("pid"), e.get("tid"))] += 1
            dur_by_name[name] += dur
            cnt_by_name[name] += 1
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts

    print(f"file: {path}")
    print(f"'X' events: {n_events:,}   metadata rows: {meta:,}")
    print(f"ts span: {first_ts} .. {last_ts}  = {((last_ts or 0) - (first_ts or 0)) / 1e6:.3f} s")
    print()
    print("== pseudo-processes (pid -> name) ==")
    for p, nm in sorted(proc_names.items(), key=lambda kv: str(kv[0])):
        print(f"  {p}  {nm}")
    print()
    print("== threads (pid, tid) -> name, events ==")
    for (p, t), c in sorted(by_pid_tid.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>12,}  pid={p} tid={t}  {thread_names.get((p, t), '?')}")
    print()
    print("== categories ==")
    for c, n in by_cat.most_common():
        print(f"  {n:>12,}  {c!r}   total {dur_by_cat[c] / 1e6:.1f} ms")
    print()
    print(f"== top {topn} distinct event names by total dur (ms) ==")
    for name, d in sorted(dur_by_name.items(), key=lambda kv: -kv[1])[:topn]:
        print(f"  {d / 1e6:>12.2f} ms  n={cnt_by_name[name]:>9,}  avg {d / max(cnt_by_name[name], 1) / 1e3:>9.2f} us  {name[:120]}")
    print()
    print(f"== distinct names: {len(dur_by_name):,} ==")


if __name__ == "__main__":
    main()
