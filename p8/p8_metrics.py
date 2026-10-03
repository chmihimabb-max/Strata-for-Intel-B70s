#!/usr/bin/env python3
"""P8: print the /metrics payload of the live server, or the parts of it a given prefix/card selects.
Usage: p8_metrics.py [--save FILE] [--keys gpu_,hardware,engine,live,requests,totals] [--json]
Writes nothing to the server and does not disturb it - GET /metrics only.
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8099/metrics"


def main():
    save = None
    keys = None
    raw = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--save":
            save = args[i + 1]
            i += 2
        elif args[i] == "--keys":
            keys = args[i + 1].split(",")
            i += 2
        elif args[i] == "--json":
            raw = True
            i += 1
        else:
            i += 1
    with urllib.request.urlopen(URL, timeout=10) as r:
        text = r.read().decode()
    if save:
        with open(save, "w") as f:
            f.write(text)
        print(f"saved {len(text)} bytes to {save}")
    if raw:
        print(text)
        return
    d = json.loads(text)
    if keys:
        out = {k: v for k, v in d.items() if any(k.startswith(p) for p in keys)}
    else:
        out = {"engine": d.get("engine"), "live": d.get("live"),
               "hardware": {k: v for k, v in (d.get("hardware") or {}).items() if not k.startswith("gpu_")},
               "hardware_gpu": {k: v for k, v in (d.get("hardware") or {}).items() if k.startswith("gpu_")},
               "hardware_static": d.get("hardware_static")}
    print(json.dumps(out, indent=1, sort_keys=False))


main()
