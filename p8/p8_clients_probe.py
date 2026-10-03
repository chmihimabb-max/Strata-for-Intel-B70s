#!/usr/bin/env python3
"""P8 probe: who the Intel backend's render-node clients are, and what each holds in VRAM - so the device-wide sum
(drm-total-vram0 over the clients) can be checked for double counting of shared buffers.
Usage: p8_clients_probe.py
"""
import os
import sys

sys.path.insert(0, "/home/michael/strata-xpu/strata")
from serve import telemetry  # noqa: E402


def comm(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return "?"


def main():
    clients = telemetry.render_clients() or {}
    for pdev, rows in sorted(clients.items()):
        print(f"=== {pdev}: {len(rows)} client fd(s)")
        by_pid = {}
        for kv in rows:
            # find the pid again: render_clients does not keep it, so re-scan is cheaper here
            pass
        total = shared = 0
        seen = []
        for kv in rows:
            t = telemetry._kib(kv.get("drm-total-vram0")) or 0
            sh = telemetry._kib(kv.get("drm-shared-vram0")) or 0
            total += t
            shared += sh
            seen.append((kv.get("drm-client-id"), t >> 20, sh >> 20))
        print(f"    sum(total-vram0) = {total >> 20} MiB   sum(shared-vram0) = {shared >> 20} MiB"
              f"   sum(total-shared) = {(total - shared) >> 20} MiB")
        for cid, t, sh in sorted(seen, key=lambda r: -r[1])[:12]:
            print(f"      client {cid}: vram0 {t} MiB (shared {sh} MiB)")


main()
