#!/usr/bin/env python3
"""P8 probe: sample the two cards' PCIe link state (and temps/power) as fast as the sysfs allows, from a file log.
Usage: p8_link_watch.py <seconds> <interval_s> [outfile]
Prints the value of every distinct reading with the time it was seen, so a downshift/upshift shows up in the log.
"""
import os
import sys
import time

CARDS = ["card0", "card1"]
DEV = {c: f"/sys/class/drm/{c}/device" for c in CARDS}


def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    out = open(sys.argv[3], "a") if len(sys.argv) > 3 else None
    seen = {}
    n = 0
    t0 = time.time()
    while time.time() - t0 < secs:
        row = []
        for c in CARDS:
            d = DEV[c]
            row.append(f"{c}:{rd(d + '/current_link_speed')}/{rd(d + '/current_link_width')}")
        line = f"[{time.time() - t0:8.3f}s] " + "  ".join(row)
        n += 1
        key = line.split("] ", 1)[1]
        if seen.get(key) != 1:
            print(line, flush=True)
            seen[key] = 1
        if out:
            out.write(line + "\n")
            out.flush()
        time.sleep(iv)
    print(f"sampled {n} times in {time.time() - t0:.2f}s; distinct states: {len(seen)}")
    if out:
        out.close()


main()
