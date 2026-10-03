#!/usr/bin/env python3
"""P8 probe: sample the xe fdinfo GPU-cycle counters of a pid twice, 1 s apart, and print the busy fraction per engine
class and per card - the same derivation intel_gpu_top makes (busy = delta(cycles<class>) / delta(total-cycles)).
Usage: p8_util_probe.py [pid] [interval]
"""
import os
import sys
import time

PID = sys.argv[1] if len(sys.argv) > 1 else "1021507"
IV = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def read(pid):
    """{pdev: {key: value}} for every render-node fd the pid holds."""
    out = {}
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError as e:
        print(f"cannot list /proc/{pid}/fdinfo: {e}")
        return out
    for fd in fds:
        try:
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                kv = {}
                for l in f.read().splitlines():
                    if "\t" in l:
                        k, _, v = l.partition("\t")
                        kv[k.rstrip(":").strip()] = v.strip()
        except OSError as e:
            print(f"  [skip fdinfo/{fd}: {e}]")
            continue
        pdev = kv.get("drm-pdev")
        if not pdev:
            continue
        out[pdev] = {k: int(v) for k, v in kv.items()
                     if k.startswith("drm-cycles-") or k.startswith("drm-total-cycles-")}
        out[pdev]["client"] = kv.get("drm-client-id", "").strip()
    return out


def main():
    a = read(PID)
    print(f"pid {PID}: cards {sorted(a)}")
    time.sleep(IV)
    b = read(PID)
    for pdev in sorted(set(a) & set(b)):
        print(f"--- {pdev} (client {b[pdev]['client']})")
        for k in sorted(k for k in b[pdev] if k.startswith("drm-cycles-")):
            cls = k[len("drm-cycles-"):]
            d = b[pdev][k] - a[pdev][k]
            tot = b[pdev].get("drm-total-cycles-" + cls, 0) - a[pdev].get("drm-total-cycles-" + cls, 0)
            pct = (100.0 * d / tot) if tot else None
            print(f"    {cls:5s} delta={d:>12d} total_delta={tot:>12d} busy={pct if pct is None else round(pct, 1)}%")


main()
