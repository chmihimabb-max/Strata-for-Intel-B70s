#!/usr/bin/env python3
"""P8 probe: run a known GPU workload (scripts/p8_h2d_bench) in a child and sample ITS xe fdinfo cycle counters,
so the busy derivation (delta(cycles<class>) / delta(total-cycles-<class>)) is validated against a workload whose
shape is known: a long single-thread kernel (compute, ccs) and a memcpy burst (copy engine, bcs).
Usage: p8_util_bench.py [MiB] [iters]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def counters(pid):
    out = {}
    for fd in os.listdir(f"/proc/{pid}/fdinfo"):
        try:
            text = open(f"/proc/{pid}/fdinfo/{fd}").read()
        except OSError:
            continue
        kv = {}
        for line in text.splitlines():
            if "\t" in line:
                k, _, v = line.partition("\t")
                kv[k.rstrip(":").strip()] = v.strip()
        if not kv.get("drm-pdev"):
            continue
        out[kv["drm-pdev"]] = kv
    return out


def main():
    mib = sys.argv[1] if len(sys.argv) > 1 else "256"
    iters = sys.argv[2] if len(sys.argv) > 2 else "300"
    p = subprocess.Popen([os.path.join(HERE, "p8_h2d_bench"), mib, iters],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    samples = []
    while p.poll() is None:
        c = counters(p.pid)
        if c:
            samples.append((time.time(), c))
        time.sleep(0.2)
    print(p.stdout.read().strip())
    print(f"\n{len(samples)} counter samples over the run; first/last deltas per card:")
    if len(samples) < 2:
        print("   not enough samples")
        return
    (t0, a), (t1, b) = samples[0], samples[-1]
    for pdev in sorted(set(a) & set(b)):
        print(f"--- {pdev}  (window {t1 - t0:.1f} s)")
        for cls in ("rcs", "ccs", "bcs", "vcs", "vecs"):
            k, tk = f"drm-cycles-{cls}", f"drm-total-cycles-{cls}"
            if k not in b[pdev]:
                continue
            d = int(b[pdev][k]) - int(a[pdev][k])
            tot = int(b[pdev][tk]) - int(a[pdev][tk])
            print(f"    {cls:5s} delta={d:>13d}  total_delta={tot:>13d}  busy={(100.0 * d / tot) if tot else 0:.2f}%"
                  f"   (cumulative {int(a[pdev][k]):,} -> {int(b[pdev][k]):,})")


main()
