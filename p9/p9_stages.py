#!/usr/bin/env python3
"""P9: the engine's own per-stage table for the verify window, out of `strata decode GPU stages`.

The engine prints one line per request (STRATA_DECODE_TIMING=1 + STRATA_VERIFY_PROFILE=1):

  strata decode GPU stages (ms/window): GDN layers: hc-read0 1.2 ... | QSA layers: ... | total X ms/window over N windows

with the stage names it was written with (src/core/verify.cpp's names[kProfPer]).  This turns that line into a
table: stage -> ms/window -> share of the sum, per layer kind (GDN / QSA), and the sum itself to compare against
the window's own `verify` from the `strata decode timing` line.

usage: /usr/bin/python3 p9/p9_stages.py <tag> [<tag> ...]
"""
import re
import sys

R = "/home/michael/strata-xpu"
RUN = f"{R}/p9/runs"

# src/core/verify.cpp's names[kProfPer] (kProfPer = 33)
NAMES = ["-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx", "k/v+norm-rope",
         "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve", "attention", "gate", "", "out-proj",
         "hc-read1+router", "shared+quant", "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU",
         "copy+combine", "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""]
LINE = re.compile(r"strata decode GPU stages \(ms/window\): GDN layers:(.*?) \| QSA layers:(.*?) \| total "
                  r"([\d.]+) ms/window over (\d+) windows")
PAIR = re.compile(r"([^\s][^0-9]*?|  [a-z0-9 ]+?) ([\d.]+)")


def pairs(raw):
    """Walk a `name value name value ...` fragment.  A name may contain spaces (`q8+qkv/q-idx gemv`,
    `  hc0 norm`), so the name is every token since the last value."""
    out = {}
    words = raw.split()
    name = []
    for w in words:
        try:
            v = float(w)
        except ValueError:
            name.append(w)
            continue
        out[" ".join(name).strip()] = v
        name = []
    return out


def parse(tag):
    err = open(f"{RUN}/{tag}/err.txt", errors="replace").read()
    m = LINE.findall(err)
    if not m:
        return None
    gdn, qsa, total, wins = m[-1]
    return {"tag": tag, "total": float(total), "windows": int(wins),
            "gdn": pairs(gdn), "qsa": pairs(qsa)}


def main():
    tags = sys.argv[1:]
    for tag in tags:
        r = parse(tag)
        if r is None:
            print(f"== {tag}: no `strata decode GPU stages` line")
            continue
        tot = sum(r["gdn"].values()) + sum(r["qsa"].values())
        print(f"== {tag}: total {r['total']} ms/window over {r['windows']} windows; "
              f"the listed stages sum to {tot:.2f} ms/window = {tot / r['total'] * 100:.1f}% of the sum the "
              f"engine printed")
        print(f"{'stage':<22} {'GDN ms/w':>9} {'QSA ms/w':>9} {'sum ms/w':>9} {'% of sum':>9}")
        keys = []
        for nm in NAMES:
            k = nm.strip()
            if k and k not in keys and (k in r["gdn"] or k in r["qsa"]):
                keys.append(k)
        for k in sorted(r["gdn"]):
            if k and k not in keys:
                keys.append(k)
        for k in sorted(r["qsa"]):
            if k and k not in keys:
                keys.append(k)
        for k in keys:
            g = r["gdn"].get(k, 0.0)
            q = r["qsa"].get(k, 0.0)
            if g == 0 and q == 0:
                continue
            print(f"{k:<22} {g:>9.3f} {q:>9.3f} {g + q:>9.3f} {(g + q) / tot * 100:>8.1f}%")
        print()


if __name__ == "__main__":
    main()
