#!/usr/bin/env python3
"""P8: while a real H2D transfer is running in a child process, say WHERE the buffers landed (xe fdinfo: vram0 vs
system) and what the PCIe link state sysfs reports - so the achieved bandwidth and the link reading are compared
under the same transfer, and the transfer cannot be explained away as a host-RAM copy.
Usage: p8_pcie_account.py [MiB] [iters]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_SETVARS = "/opt/intel/oneapi/setvars.sh"


def link_state():
    out = []
    for c in ("card0", "card1"):
        d = f"/sys/class/drm/{c}/device"
        try:
            with open(d + "/current_link_speed") as f:
                sp = f.read().strip()
            with open(d + "/current_link_width") as f:
                w = f.read().strip()
        except OSError:
            sp = w = "?"
        out.append(f"{c}={sp}/x{w}")
    return " ".join(out)


def drm_fdinfo(pid):
    """Every drm-* line of that pid's fdinfo, grouped per drm-pdev (i.e. per card)."""
    rows = {}
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return rows
    for fd in fds:
        try:
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                kv = {}
                for l in f.read().splitlines():
                    if "\t" in l:
                        k, _, v = l.partition("\t")
                        kv[k.rstrip(":").strip()] = v.strip()
        except OSError:
            continue
        pdev = kv.get("drm-pdev")
        if not pdev:
            continue
        keep = {k: v.strip() for k, v in kv.items()
                if k.startswith(("drm-total-vram", "drm-resident-vram", "drm-total-system", "drm-resident-system"))}
        rows.setdefault(pdev, {}).update({f"fd{fd}:{k}": v for k, v in keep.items()})
    return rows


def main():
    mib = sys.argv[1] if len(sys.argv) > 1 else "1024"
    iters = sys.argv[2] if len(sys.argv) > 2 else "12"
    env = dict(os.environ)
    env.pop("ZE_AFFINITY_MASK", None)
    print(f"link before: {link_state()}", flush=True)
    p = subprocess.Popen([os.path.join(HERE, "p8_h2d_bench"), mib, iters], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    seen_links = set()
    seen_mem = {}
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < 120:
        seen_links.add(link_state())
        for pdev, kv in drm_fdinfo(p.pid).items():
            seen_mem.setdefault(pdev, set()).add(tuple(sorted(kv.items())))
        time.sleep(0.05)
    out = p.stdout.read() if p.stdout else ""
    print("--- bench output ---")
    print(out.strip())
    print("--- link states seen while the transfer ran ---")
    for s in sorted(seen_links):
        print("   " + s)
    print("--- fdinfo of the transferring process (per card) ---")
    for pdev, states in sorted(seen_mem.items()):
        last = dict(sorted(states)[-1])
        print(f"   {pdev}:")
        for k, v in sorted(last.items()):
            print(f"      {k} = {v}")
    print(f"link after: {link_state()}")


main()
