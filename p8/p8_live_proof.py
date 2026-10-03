#!/usr/bin/env python3
"""P8: prove the telemetry moves while a REAL request is in flight.

Sends one /v1/chat/completions and, while it runs, samples GET /metrics once a second (the payload the Monitor reads)
plus the raw sysfs numbers behind the Intel backend (hwmon temp/power inputs, energy counter, fdinfo cycles), so the
before / in-flight / after readings all come from the same request.  Writes p8/evidence/live-proof.json and prints a
table.  Usage: p8_live_proof.py [max_tokens] [prompt]
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8099"
HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = {"card0": "0000:03:00.0", "card1": "0000:08:00.0"}


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def sysfs():
    out = {}
    for card, bdf in CARDS.items():
        dev = f"/sys/bus/pci/devices/{bdf}"
        hw = os.path.join(dev, "hwmon")
        try:
            h = os.path.join(hw, sorted(os.listdir(hw))[0])
        except OSError:
            continue
        out[card] = {"pkg_c": rd(f"{h}/temp2_input"), "vram_c": rd(f"{h}/temp3_input"),
                     "pcie_c": rd(f"{h}/temp5_input"), "fan_rpm": rd(f"{h}/fan1_input"),
                     "energy1_uj": rd(f"{h}/energy1_input"), "power1_cap_uw": rd(f"{h}/power1_cap"),
                     "energy2_uj": rd(f"{h}/energy2_input"),
                     "link_speed": rd(f"{dev}/current_link_speed"), "link_width": rd(f"{dev}/current_link_width"),
                     "act_freq": rd(f"{dev}/tile0/gt0/freq0/act_freq"), "gt_idle": rd(f"{dev}/tile0/gt0/gtidle/idle_status")}
    return out


def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    prompt = sys.argv[2] if len(sys.argv) > 2 else ("Explain, in about 200 words, how a PCIe link trains and why a "
                                                    "link can renegotiate its width and speed. Be concrete.")
    samples = []
    result = {}

    def run():
        t0 = time.time()
        try:
            result["response"] = post("/v1/chat/completions", {
                "model": "qwen3.8-flash-next-iq3s",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens})
        except Exception as e:  # noqa: BLE001
            result["error"] = repr(e)
        result["wall_s"] = round(time.time() - t0, 1)

    before = get("/metrics")
    th = threading.Thread(target=run)
    th.start()
    t0 = time.time()
    while th.is_alive():
        m = get("/metrics")
        hw = m.get("hardware") or {}
        samples.append({"t": round(time.time() - t0, 1), "state": m["live"]["state"],
                        "tok_s": m["live"].get("tok_s"),
                        "gpu_util": hw.get("gpu_util"), "gpu_temp": hw.get("gpu_temp"),
                        "gpu_power": hw.get("gpu_power"), "gpu_mem_used": hw.get("gpu_mem_used"),
                        "gpu_copy_busy": hw.get("gpu_copy_busy"), "gpu_pcie_gen": hw.get("gpu_pcie_gen"),
                        "per_card": [{k: g.get(k) for k in ("index", "util", "temp", "power", "copy_busy",
                                                            "mem_used", "freq_mhz")} for g in hw.get("gpus", [])],
                        "sysfs": sysfs()})
        time.sleep(1.0)
    th.join()
    after = get("/metrics")
    with open(os.path.join(HERE, "evidence", "live-proof.json"), "w") as f:      # write before printing: never lose it
        json.dump({"prompt": prompt, "max_tokens": max_tokens, "request": {k: v for k, v in result.items()},
                   "before": before, "during": samples, "after": after}, f, indent=1)
    print(f"wrote {os.path.join(HERE, 'evidence', 'live-proof.json')}")

    print("=== the request")
    print(json.dumps({k: v for k, v in result.items() if k != "response"}, indent=1))
    msg = ((result.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
    txt = msg.get("content") or ""
    usage = (result.get("response") or {}).get("usage") or {}
    print(f"    message keys {sorted(msg)}; content {len(txt)} chars; usage {json.dumps(usage)}")
    print(f"    first 160 chars: {txt[:160]!r}")
    if not txt:
        print(f"    (no plain content - the raw message, trimmed: {json.dumps(msg)[:400]})")
    print("\n=== /metrics while it ran (the Monitor's own fields)")
    print(f"{'t':>5} {'state':>10} {'tok_s':>6} {'util':>5} {'temp':>5} {'power':>6} {'copy':>5} {'mem_used':>12} pcie_gen")
    for s in samples:
        print(f"{s['t']:>5} {s['state']:>10} {str(s['tok_s']):>6} {str(s['gpu_util']):>5} {str(s['gpu_temp']):>5} "
              f"{str(s['gpu_power']):>6} {str(s['gpu_copy_busy']):>5} {str(s['gpu_mem_used']):>12} {s['gpu_pcie_gen']}")
    print("\n=== raw sysfs behind it (card0 / card1)")
    for s in (samples[0], samples[len(samples) // 2], samples[-1]) if samples else []:
        for card in CARDS:
            d = s["sysfs"].get(card)
            if d:
                print(f"  t={s['t']:>5} {card}: {d['pkg_c']} mC pkg, {d['vram_c']} mC vram, {d['pcie_c']} mC pcie, "
                      f"fan {d['fan_rpm']} rpm, energy1 {d['energy1_uj']} µJ, cap {d['power1_cap_uw']} µW, "
                      f"link {d['link_speed']}/x{d['link_width']}, act_freq {d['act_freq']} MHz, gt {d['gt_idle']}")
    before_hw = before.get("hardware") or {}
    after_hw = after.get("hardware") or {}
    print("\n=== before / after the request (the same fields, at rest)")
    for k in ("gpu_util", "gpu_temp", "gpu_power", "gpu_mem_used", "gpu_copy_busy", "gpu_mem_gtt"):
        print(f"    {k:16} before={before_hw.get(k)!r:>14}  after={after_hw.get(k)!r}")


main()
