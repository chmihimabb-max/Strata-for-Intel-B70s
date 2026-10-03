#!/usr/bin/env python3
"""P8 probe: watch one /metrics reading over several samples to see it move (power, util, memory, temps).
The sampler thread is what fills the payload (and it is what consumes power's energy delta), so this reads
Telemetry.snapshot() - exactly what GET /metrics returns - rather than calling sample() a second time.
Usage: p8_watch.py [field] [samples] [interval]      e.g. p8_watch.py gpu_power 6 1.0
"""
import sys
import time

sys.path.insert(0, "/home/michael/strata-xpu/strata")
from serve import telemetry  # noqa: E402


def main():
    field = sys.argv[1] if len(sys.argv) > 1 else "gpu_power"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    iv = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    t = telemetry.Telemetry(gpu_index=0, gpu_indices=[0, 1], intel=True)
    key = field.replace("gpu_", "")
    for i in range(n):
        now = t.snapshot()["now"]
        per = "  ".join(f"card{g.get('index')}={g.get(key)}" for g in now.get("gpus", []))
        print(f"[{i}] {field} = {now.get(field)}      per-card: {per}", flush=True)
        time.sleep(iv)


main()
