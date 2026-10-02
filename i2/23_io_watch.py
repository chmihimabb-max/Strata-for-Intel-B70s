#!/usr/bin/env python3
"""I2: sample what the engine actually pulled off the disk during a run, so "page-cache resident" is a
measurement and not an assumption.

    python3 i2/23_io_watch.py <csv> <seconds>

Two independent numbers per sample: the block device's own served bytes (all processes) and read_bytes
summed over every process whose comm contains "strata" (the engine, this run).
"""
import pathlib
import sys
import time

csv = pathlib.Path(sys.argv[1])
secs = int(sys.argv[2])
NVME = "nvme"


def dev_bytes():
    """bytes served by the nvme whole disks (their partition lines are the same accounting)"""
    total = 0
    for line in pathlib.Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) > 5 and f[2].startswith(NVME) and all(c.isdigit() or c == "n" for c in f[2][len(NVME):]):
            total += int(f[5]) * 512
    return total


def engine_bytes():
    total, procs = 0, []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / "comm").read_text().strip()
            if "strata" not in comm.lower():
                continue
            for line in (p / "io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    total += int(line.split()[1])
            procs.append(comm)
        except (OSError, ValueError):
            continue
    return total, procs


t0 = time.time()
with csv.open("w") as f:
    f.write("t,dev_bytes,strata_bytes,procs,maxrss_kb\n")
    while time.time() - t0 < secs:
        eb, procs = engine_bytes()
        rss = 0
        for p in pathlib.Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                comm = (p / "comm").read_text().strip()
                if "strata" not in comm.lower():
                    continue
                for line in (p / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss = max(rss, int(line.split()[1]))
            except (OSError, ValueError):
                continue
        f.write("%.1f,%d,%d,%s,%d\n" % (time.time() - t0, dev_bytes(), eb, "|".join(procs), rss))
        f.flush()
        time.sleep(1)
