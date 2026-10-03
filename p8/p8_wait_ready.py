#!/usr/bin/env python3
"""P8: wait until the freshly started server answers READY on /v1/models, printing the gpu_* fields it reports while
it waits (the engine loads ~2x 83 GB of GGUF, so this is minutes).  Usage: p8_wait_ready.py [seconds]
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8099"
LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.read().decode()


t0 = time.time()
while time.time() - t0 < LIMIT:
    try:
        m = json.loads(get("/metrics"))
        hw = m.get("hardware") or {}
        gpus = hw.get("gpus") or []
        print(f"[{time.time() - t0:6.1f}s] engine={m['engine']['engine'] if m.get('engine') else '?'} "
              f"state={m['live']['state']} gpu_name={(m.get('hardware_static') or {}).get('gpu_name')} "
              f"util={hw.get('gpu_util')} temp={hw.get('gpu_temp')} power={hw.get('gpu_power')} "
              f"mem_used={hw.get('gpu_mem_used')} mem_total={hw.get('gpu_mem_total')} cards={len(gpus)}", flush=True)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[{time.time() - t0:6.1f}s] /metrics not up yet: {e}", flush=True)
    try:
        models = json.loads(get("/v1/models"))
        if models.get("data"):
            print(f"[{time.time() - t0:6.1f}s] READY: {models['data'][0].get('id')}", flush=True)
            break
    except (urllib.error.URLError, OSError, ValueError):
        pass
    time.sleep(15)
