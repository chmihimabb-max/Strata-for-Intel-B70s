#!/usr/bin/env python3
"""P8: print the gpu_* fields of a saved /metrics payload, per card, as the report's table needs them.
Usage: p8_show.py p8/evidence/metrics-after-restart-idle.json
"""
import json
import sys

FIELDS = ("index", "util", "copy_busy", "mem_used", "mem_total", "mem_gtt", "mem_system", "clients", "temp",
          "temp_vram", "temp_vram_max", "temp_pcie", "fan_rpm", "power", "power_pkg", "power_limit", "power_crit",
          "freq_mhz", "freq_max_mhz", "gt_idle", "pcie_gen", "pcie_width", "pcie_rx_mb")


def main():
    d = json.load(open(sys.argv[1]))
    hw = d.get("hardware") or {}
    print("aggregate:")
    for k in sorted(k for k in hw if k.startswith("gpu_")):
        print(f"   {k} = {hw[k]!r}")
    for g in hw.get("gpus") or []:
        print(f"card index {g.get('index')}:")
        for k in FIELDS[1:]:
            print(f"   {k} = {g.get(k)!r}")


main()
