#!/usr/bin/env python3
"""P8 probe: instantiate the shipped Intel backend (serve/telemetry.py:_Intel / Telemetry) against the live machine and
dump every field it produces, per card, with the provenance strings it puts in the payload.
Usage: p8_backend_probe.py [samples] [interval]
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/michael/strata-xpu/strata")
from serve import telemetry  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    print("device dirs:", telemetry.intel_device_dirs())
    g = telemetry.gpu_reader(0, intel=True)
    print("ok:", g.ok(), "| name:", g.name(), "| pdev:", g.pdev, "| hwmon:", g.hwmon)
    print("labels:", json.dumps(g.labels, sort_keys=True))
    print("sources:", json.dumps(g.sources(), indent=1, sort_keys=True))
    print("unsupported:", json.dumps(g.unsupported(), indent=1, sort_keys=True))
    t = telemetry.Telemetry(gpu_index=0, gpu_indices=[0, 1], intel=True)
    for i in range(n):
        s = t.sample()
        print(f"--- sample {i + 1}")
        for k in sorted(s):
            if k.startswith("gpu"):
                print(f"    {k} = {s[k]}")
        for card in s.get("gpus", []):
            print(f"    gpus[{card['index']}] = " + json.dumps(card, sort_keys=True))
        time.sleep(iv)
    print("--- static")
    print(json.dumps(t.static, indent=1, sort_keys=True))


main()
