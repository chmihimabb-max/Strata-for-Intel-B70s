#!/usr/bin/env python3
"""P3 (t_d8afe53f) evidence extractor: one row per arm, straight out of the harness's own files.

usage: /usr/bin/python3 p3/extract.py <TAG> [<TAG> ...]
  <TAG> is a directory under ~/strata-xpu/m6c/runs/ (m6c_serve.sh writes err.txt/out.txt/rss.csv/log.txt).
Prints, per arm: the engine's decode-timing line (window composition), the serve summary line (prefill and
decode tok/s), the DONE line, the greedy token-id count and md5, the submission counts per window + stage
(from the STRATA_SUBMIT_COUNT lines: median of the T>=2 windows), and peak tree RSS + peak per-card VRAM.
"""
import hashlib
import re
import statistics
import sys
from pathlib import Path

RUNS = Path("/home/michael/strata-xpu/m6c/runs")


def peak_vram(mon: Path):
    """m6c_monitor_tree.py's csv: ...,vram0_mib,vram1_mib (last two columns)."""
    v0, v1 = [], []
    for i, line in enumerate(mon.read_text(errors="replace").splitlines()):
        if i == 0 or not line.strip():
            continue
        f = line.split(",")
        if len(f) < 2:
            continue
        try:
            v0.append(float(f[-2])); v1.append(float(f[-1]))
        except ValueError:
            pass
    return (max(v0) if v0 else 0.0, max(v1) if v1 else 0.0)


def main(tags):
    for tag in tags:
        d = RUNS / tag
        err = (d / "err.txt").read_text(errors="replace") if (d / "err.txt").exists() else ""
        out = (d / "out.txt").read_text(errors="replace") if (d / "out.txt").exists() else ""
        ids = [l.split()[1] for l in out.splitlines() if l.startswith("T ")]
        md5 = hashlib.md5(("\n".join("T " + i for i in ids) + "\n").encode()).hexdigest()[:20] if ids else "-"
        print("=" * 100)
        print("ARM %s" % tag)
        for pat in ("strata decode timing", "strata serve: prompt ", "strata serve: layer split",
                    "strata/sycl: STRATA_SYCL_GRAPH", "strata verify: window up to"):
            for line in err.splitlines():
                if line.startswith(pat):
                    print("  " + line)
                    break
        for line in out.splitlines():
            if line.startswith("DONE "):
                print("  %s   (tokens generated / prompt / prefill ms / decode ms / finish / "
                      "draft accepted / offered ...)" % line)
        print("  token ids: %d, md5 %s" % (len(ids), md5))

        # submission counts: the STRATA_SUBMIT_COUNT lines, T>=2 windows only (T=1 is the first window)
        subs = {}
        for line in err.splitlines():
            m = re.match(r"strata submit: (\S+) window T=(\d+) pos0=(\d+) layers (\d+)\.\.(\d+) \((\d+)\): "
                         r"submitted (\d+) \(kernel (\d+) memset (\d+) memcpy (\d+) barrier (\d+) event (\d+) "
                         r"hostfn (\d+) graph (\d+)\) \+ recorded (\d+).*GPU-reach wait ([\d.]+) ms", line)
            if not m:
                continue
            st, T, lb, le, n, tot = m.group(1), int(m.group(2)), int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))
            key = (lb, le)
            #                        T   tot   kernel  barrier  graph   wait
            subs.setdefault(key, []).append((T, tot, int(m.group(8)), int(m.group(11)), int(m.group(14)), float(m.group(16))))
        for (lb, le), rows in sorted(subs.items()):
            big = [r for r in rows if r[0] >= 2] or rows
            tot = statistics.median(r[1] for r in big)
            ker = statistics.median(r[2] for r in big)
            bar = statistics.median(r[3] for r in big)
            grf = statistics.median(r[4] for r in big)
            per = statistics.median(r[1] for r in big) / float(le - lb + 1)
            print("  submissions/window layers %d..%d (%d layers): %s total, %s kernels, %s barriers, %s graph "
                  "submissions -> %.1f per layer   [%d windows]" % (lb, le, le - lb + 1, tot, ker, bar, grf, per, len(big)))
        v0, v1 = peak_vram(d / "rss.csv")
        rss = 0.0
        for i, line in enumerate((d / "rss.csv").read_text(errors="replace").splitlines()):
            f = line.split(",")
            if i and len(f) > 3:
                try:
                    rss = max(rss, float(f[3]))
                except ValueError:
                    pass
        print("  peak tree RSS %.1f GiB, peak VRAM card0 %.1f GiB card1 %.1f GiB" % (rss / 2**30, v0 / 1024.0, v1 / 1024.0))
        for line in err.splitlines():
            if "expert cache hit rate" in line or "expert tiers" in line:
                print("  " + line.strip())


if __name__ == "__main__":
    main(sys.argv[1:] or ["p3-4k-closed", "p3-4k-graph2"])
