#!/usr/bin/env python3
"""P10: is the pool really burning cores? Three independent instruments, one window.

The card wants per-thread CPU evidence, and the two obvious tools disagree with a third on this box:
`pidstat -t -p PID` and `top -b -H -p PID` both printed 0.00% for every engine thread during a request, while
summing /proc/<pid>/task/<tid>/stat deltas showed ~19 threads at ~85-89% of a core.  This measures the same
window three ways and, if they disagree, says which two agree:

  1. /proc/<pid>/stat          -> the process's own aggregate utime+stime delta (ground truth: it is the same
                                  counter the kernel charges, and thread deltas must sum to it)
  2. /proc/<pid>/task/*/stat   -> the sum of the per-thread deltas (what p10_threads.py reports)
  3. ps -L -o tid,pcpu,time,stat -p PID  -> an independent per-thread read

usage: p10_threadcheck.py <pid> [seconds] [interval]
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")


def proc_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
    except OSError:
        return None
    r = s[s.rfind(")") + 2:].split()
    return int(r[11]) + int(r[12])       # utime + stime (process aggregate)


def thread_ticks(pid):
    out = {}
    for p in glob.glob(f"/proc/{pid}/task/*"):
        tid = os.path.basename(p)
        try:
            with open(p + "/stat") as f:
                s = f.read()
        except OSError:
            continue
        r = s[s.rfind(")") + 2:].split()
        if len(r) < 13:
            continue
        out[tid] = int(r[11]) + int(r[12])
    return out


def main() -> int:
    pid = int(sys.argv[1])
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    iv = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    print(f"[check] pid {pid}, {secs}s, {iv}s intervals   CLK_TCK={CLK}")
    p0 = proc_ticks(pid)
    t0 = thread_ticks(pid)
    if p0 is None:
        print(f"[check] no such process {pid}")
        return 1
    print(f"[check] t=0 aggregate {p0} ticks, {len(t0)} threads, sum {sum(t0.values())}")
    n = 0
    t = time.time()
    while time.time() - t < secs:
        time.sleep(iv)
        n += 1
        p1, t1 = proc_ticks(pid), thread_ticks(pid)
        if p1 is None:
            print("[check] process gone")
            break
        dp = p1 - p0
        dt = sum(t1.get(k, 0) - t0.get(k, 0) for k in set(t0) | set(t1))
        cores_proc = dp / CLK / iv
        cores_thr = dt / CLK / iv
        nz = sum(1 for k in set(t0) | set(t1) if (t1.get(k, 0) - t0.get(k, 0)) / CLK / iv > 0.5)
        print(f"[check] {time.time():.0f} t={n*iv:5.1f}s  aggregate {cores_proc:5.2f} cores   sum(threads) {cores_thr:5.2f} cores   "
              f"threads >0.5 core: {nz:3d}   (dp={dp} dt={dt} ticks)")
        p0, t0 = p1, t1
    # instrument 3
    ps = subprocess.run(["/usr/bin/ps", "-L", "-o", "tid,pcpu,time,cputime,stat,comm", "-p", str(pid)],
                        capture_output=True, text=True)
    rows = [r for r in ps.stdout.splitlines()[1:] if r.strip()]
    rows.sort(key=lambda r: -float(r.split()[1]))
    print("[check] ps -L (top 25 by pcpu):")
    for r in rows[:25]:
        print("   " + r)
    alive = sum(1 for r in rows if float(r.split()[1]) > 0.5)
    print(f"[check] ps -L: {len(rows)} threads, {alive} above 0.5% cpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
