#!/usr/bin/env python3
"""M6c: the RAM trajectory of a run, to see where the streamed KV shows up.

    /usr/bin/python3 m6c/m6c_rss_trace.py RUN_DIR [RUN_DIR...]

Prints RSS (and per-card VRAM, when the sampler has it) against the engine's own phase markers from the
timeline (`READY`, `RESUME`, `DONE`), so the difference between a streaming and a resident run at the same
context can be read off at the same instant instead of from a peak that both arms share.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "/home/michael/strata-xpu/strata/m6c")
from m6c_report import at, read_timeline, samples  # noqa: E402


def main() -> int:
    for arg in sys.argv[1:]:
        d = pathlib.Path(arg)
        tl = read_timeline(d / "timeline.txt")
        rows = samples(d / "rss.csv")
        if not rows:
            print("%s: no samples" % d.name)
            continue
        print("=" * 96)
        print("%s   epoch(RESUME)=%s" % (d.name, tl.get("resume")))
        keys = ["epoch", "t_s", "rss_bytes", "read_bytes", "majflt"]
        if "vram0_mib" in rows[0]:
            keys += ["vram0_mib", "vram1_mib"]
        for label, t in (("start", rows[0]["epoch"]), ("READY", tl.get("ready")), ("RESUME", tl.get("resume")),
                         ("first T", tl.get("first_t")), ("DONE", tl.get("done")), ("last", rows[-1]["epoch"])):
            if t is None:
                continue
            s = at(rows, t)
            print("  %-8s t=%+7.1f s  RSS %6.2f GiB  read_bytes %6.2f GiB  vram %s" % (
                label, t - tl.get("resume", t), s.get("rss_bytes", 0) / 2 ** 30,
                s.get("read_bytes", 0) / 2 ** 30,
                ("%.1f+%.1f MiB" % (s.get("vram0_mib", 0), s.get("vram1_mib", 0))) if "vram0_mib" in s else "-"))
        peak = max(rows, key=lambda r: r["rss_bytes"])
        print("  peak RSS %.2f GiB at t=%+.1f s" % (peak["rss_bytes"] / 2 ** 30, peak["epoch"] - tl.get("resume", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
