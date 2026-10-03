#!/usr/bin/env python3
"""P8 probe: sample the two Battlemage cards' sysfs telemetry (link, temps by label, energy counters, fan) for a while
and derive power from the energy counter deltas across the sampling interval.
Usage: p8_live_probe.py [seconds] [interval]
"""
import os
import sys
import time

CARDS = {"card0": "/sys/class/drm/card0/device", "card1": "/sys/class/drm/card1/device"}


def rd(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def hwmon(dev):
    h = os.path.join(dev, "hwmon")
    try:
        return os.path.join(h, sorted(os.listdir(h))[0])
    except OSError:
        return None


def labels(h):
    out = {}
    for f in sorted(os.listdir(h)):
        if f.startswith("temp") and f.endswith("_label"):
            out[f[:-6]] = rd(os.path.join(h, f))
    return out


def sample(prev):
    row = {}
    for c, dev in CARDS.items():
        h = hwmon(dev)
        row[c] = {
            "link_speed": rd(os.path.join(dev, "current_link_speed")),
            "link_width": rd(os.path.join(dev, "current_link_width")),
            "pkg_c": int(rd(os.path.join(h, "temp2_input"))) / 1000 if rd(os.path.join(h, "temp2_input")) else None,
            "vram_c": int(rd(os.path.join(h, "temp3_input"))) / 1000 if rd(os.path.join(h, "temp3_input")) else None,
            "pcie_c": int(rd(os.path.join(h, "temp5_input"))) / 1000 if rd(os.path.join(h, "temp5_input")) else None,
            "vramch_max_c": max(int(rd(os.path.join(h, f"temp{i}_input"))) for i in range(6, 22)) / 1000,
            "fan_rpm": int(rd(os.path.join(h, "fan1_input")) or 0),
            "energy_card_uj": int(rd(os.path.join(h, "energy1_input")) or 0),
            "energy_pkg_uj": int(rd(os.path.join(h, "energy2_input")) or 0),
            "cap_w": int(rd(os.path.join(h, "power1_cap")) or 0) / 1e6,
        }
    return row


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    print("labels card0:", labels(hwmon(CARDS["card0"])))
    prev = None
    t0 = time.time()
    while time.time() - t0 < secs:
        t = time.time()
        cur = sample(prev)
        line = []
        for c in CARDS:
            d = cur[c]
            p_card = p_pkg = None
            if prev:
                dt = t - prev["_t"]
                p_card = (d["energy_card_uj"] - prev[c]["energy_card_uj"]) / 1e6 / dt
                p_pkg = (d["energy_pkg_uj"] - prev[c]["energy_pkg_uj"]) / 1e6 / dt
            line.append(f"{c}: link={d['link_speed']}/{d['link_width']} pkg={d['pkg_c']}C vram={d['vram_c']}C "
                        f"pcie={d['pcie_c']}C vramch_max={d['vramch_max_c']}C fan={d['fan_rpm']} "
                        f"P_card={p_card if p_card is None else round(p_card,1)}W P_pkg={p_pkg if p_pkg is None else round(p_pkg,1)}W "
                        f"cap={d['cap_w']}W")
        print(f"[{time.time()-t0:6.1f}s] " + "\n                          ".join(line))
        cur["_t"] = t
        prev = cur
        time.sleep(max(0.0, iv - (time.time() - t)))
    print("done")


main()
