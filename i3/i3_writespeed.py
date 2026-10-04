#!/usr/bin/env python3
"""i3_writespeed.py - measure sequential write throughput on the target filesystem.

    i3_writespeed.py <dir> [gib]

Writes <gib> GiB (default 4) of zeros in 64 MiB chunks, fsyncs, reports MiB/s, then TRUNCATES the file to 0
(no deletion anywhere: nothing in this run removes a file).

Why: the assembled oracle GGUF is ~103 GiB and the choice of volume (the external NTFS SSD that holds the
W4A16 pack, or / with 171 GiB free) has to be made on a measurement, not a guess.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

dirp = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/run/media/michael/2208B12208B0F63F/strata-w4a16")
gib = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
dirp.mkdir(parents=True, exist_ok=True)
p = dirp / "i3-writespeed.tmp"
chunk = b"\0" * (64 << 20)
n = int(gib * 1024 ** 3 / len(chunk))
t0 = time.perf_counter()
with open(p, "wb") as f:
    for _ in range(n):
        f.write(chunk)
    t_wb = time.perf_counter()
    f.flush()
    os.fsync(f.fileno())
t1 = time.perf_counter()
total = n * len(chunk)
print(f"path            {p}")
print(f"bytes written   {total} ({total/2**30:.2f} GiB) in {n} x 64 MiB chunks")
print(f"write+flush     {total/2**20/(t_wb-t0):.1f} MiB/s  ({t_wb-t0:.2f} s)")
print(f"write+fsync     {total/2**20/(t1-t0):.1f} MiB/s  ({t1-t0:.2f} s)")
with open(p, "r+b") as f:
    f.truncate(0)
print(f"truncated back to 0 bytes: {p.stat().st_size}")
