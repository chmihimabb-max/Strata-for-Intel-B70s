#!/usr/bin/env python3
"""Peaks from a m6c_sample.py CSV of the resident server's process tree (server + its engine child)."""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/michael/strata-xpu/strata/p6/p6-resource.csv"
rows = list(csv.DictReader(open(path)))
if not rows:
    print("no samples yet")
    raise SystemExit(1)
f = lambda r, k: float(r[k])
peak_rss = max(rows, key=lambda r: f(r, "rss_bytes"))
peak_v0 = max(rows, key=lambda r: f(r, "vram0_mib"))
peak_v1 = max(rows, key=lambda r: f(r, "vram1_mib"))
last = rows[-1]
print(f"samples            : {len(rows)}  ({f(rows[0], 't_s'):.0f}s..{f(last, 't_s'):.0f}s)")
print(f"processes in tree  : {last['nprocs']}")
print(f"peak tree RSS      : {f(peak_rss, 'rss_bytes') / 2**30:.2f} GiB  (t={f(peak_rss, 't_s'):.0f}s)")
print(f"      engine alone : {f(peak_rss, 'max_proc_rss') / 2**30:.2f} GiB")
print(f"peak VRAM card0    : {f(peak_v0, 'vram0_mib'):.1f} MiB  (t={f(peak_v0, 't_s'):.0f}s)")
print(f"peak VRAM card1    : {f(peak_v1, 'vram1_mib'):.1f} MiB  (t={f(peak_v1, 't_s'):.0f}s)")
print(f"last sample        : RSS {f(last, 'rss_bytes') / 2**30:.2f} GiB, VRAM {f(last, 'vram0_mib'):.1f} + "
      f"{f(last, 'vram1_mib'):.1f} MiB, mlocked {f(last, 'mlocked_kb') / 2**20:.2f} GiB, "
      f"read_bytes {f(last, 'read_bytes') / 2**30:.2f} GiB")
print("per-second VRAM (MiB) card0/card1 and tree RSS (GiB), every 10th sample:")
for r in rows[::10]:
    print(f"  t={f(r, 't_s'):5.0f}s  card0 {f(r, 'vram0_mib'):8.1f}  card1 {f(r, 'vram1_mib'):8.1f}  "
          f"rss {f(r, 'rss_bytes') / 2**30:6.2f}")
