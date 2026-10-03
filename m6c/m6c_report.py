#!/usr/bin/env python3
"""M6c: turn one run directory into the numbers the card asks for.

    /usr/bin/python3 m6c/m6c_report.py RUN_DIR [RUN_DIR...]     # table on stdout
    /usr/bin/python3 m6c/m6c_report.py --json RUN_DIR[...]

TTFT is timed from the engine's own `RESUME 0` marker (the instant it starts reading the prompt) to the
first `T <id>` line on stdout, both taken from m6c_drive.py's timeline; the engine's own DONE summary is
reported next to it.  Peak RSS, read_bytes and major faults come from m6c_sample.py's per-second samples of
the engine process tree, and the sample nearest `RESUME 0` gives the RSS *before* the prompt was read --
the differential between the streaming and resident arms at the same context is the measured RAM cost of
the streamed KV.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

STREAM = re.compile(r"KV streaming: (\d+) of (\d+) cells per QSA layer in VRAM, the K/V in ([\d.]+) GiB of pinned RAM")
HITS = re.compile(r"strata serve: KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
DONE = re.compile(r"^DONE (\d+) (\d+) ([\d.]+) ([\d.]+) (\S+) (\d+) (\d+)")
INFO = re.compile(r"^INFO (.+)$")
PROMPT = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
PP = re.compile(r"^PP (\d+) (\d+) (\d+) ([\d.]+)")


def read_timeline(path: pathlib.Path) -> dict:
    out = []
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#"):
            continue
        p = line.split("\t", 3)
        if len(p) < 4:
            continue
        try:
            out.append((float(p[0]), p[2], p[3]))
        except ValueError:
            continue
    marks = {}
    for t, stream, text in out:
        if text.startswith("READY") and "ready" not in marks:
            marks["ready"] = t
        if text.startswith("RESUME") and "resume" not in marks:
            marks["resume"] = t
        if text.startswith("T ") and "first_t" not in marks:
            marks["first_t"] = t
        if text.startswith("T "):
            marks["last_t"] = t
        if text.startswith("DONE"):
            marks["done"] = t
    return marks


def samples(path: pathlib.Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        return rows
    hdr = lines[0].split(",")
    for line in lines[1:]:
        v = line.split(",")
        if len(v) != len(hdr):
            continue
        rows.append(dict(zip(hdr, [float(x) for x in v])))
    return rows


def at(rows: list[dict], t: float) -> dict:
    best = None
    for r in rows:
        if r["epoch"] <= t:
            best = r
        else:
            break
    return best or (rows[0] if rows else {})


def report(d: pathlib.Path) -> dict:
    r: dict = {"tag": d.name, "dir": str(d)}
    tl = read_timeline(d / "timeline.txt")
    if tl.get("resume") is not None and tl.get("first_t") is not None:
        r["ttft_s"] = round(tl["first_t"] - tl["resume"], 3)
    if tl.get("first_t") is not None and tl.get("last_t") is not None:
        r["decode_wall_s"] = round(tl["last_t"] - tl["first_t"], 3)
    if tl.get("resume") is not None and tl.get("done") is not None:
        r["request_wall_s"] = round(tl["done"] - tl["resume"], 3)

    out = (d / "out.txt").read_text(errors="replace") if (d / "out.txt").exists() else ""
    err = (d / "err.txt").read_text(errors="replace") if (d / "err.txt").exists() else ""
    r["requests"] = []
    for line in out.splitlines():
        m = DONE.match(line)
        if m:
            r["requests"].append({"generated": int(m.group(1)), "prompt_tokens": int(m.group(2)),
                                  "prompt_ms": float(m.group(3)), "decode_ms": float(m.group(4)),
                                  "stop": m.group(5), "drafts_accepted": int(m.group(6)),
                                  "drafts_offered": int(m.group(7)),
                                  "prefill_tok_s": round(int(m.group(2)) / (float(m.group(3)) / 1000.0), 1),
                                  "decode_tok_s": round(int(m.group(1)) / (float(m.group(4)) / 1000.0), 2)})
        m = INFO.match(line.strip())
        if m:
            r["info"] = dict(kv.split("=", 1) for kv in m.group(1).split(" ") if "=" in kv)
    if r["requests"]:
        r["done"] = r["requests"][0]
    pps = [PP.match(l).groups() for l in out.splitlines() if PP.match(l)]
    if pps:
        r["pp_first"] = pps[0]
        r["pp_last"] = pps[-1]
        if len(pps) > 1:
            a, b = pps[0], pps[-1]
            dt = (int(b[2]) - int(a[2])) / 1000.0
            dn = int(b[0]) - int(a[0])
            r["prefill_marginal_tok_s"] = round(dn / dt, 1) if dt > 0 else None
    r["hits_list"] = []
    r["summaries"] = []
    r["streaming_list"] = []
    for line in err.splitlines():
        m = STREAM.search(line)
        if m:
            s = {"resident_cells": int(m.group(1)), "context_cells": int(m.group(2)),
                 "pinned_gib": float(m.group(3)),
                 "pinned_kib_per_token": round(float(m.group(3)) * 1073741824.0 / int(m.group(2)) / 1024.0, 3)}
            r["streaming"] = s
            r["streaming_list"].append(s)
        m = HITS.search(line)
        if m:
            h = {"pct": float(m.group(1)), "lookups": int(m.group(2)), "ram_mib": float(m.group(3))}
            r["hits"] = h
            r["hits_list"].append(h)
        m = PROMPT.search(line)
        if m:
            r["serve_summary"] = line.strip()
            r["summaries"].append(line.strip())
    r["cache_lines"] = [l.strip() for l in err.splitlines()
                        if ("expert cache" in l or "resident slot" in l or "layer split" in l
                            or "KV streaming" in l or "prompt path" in l)]
    r["stats_lines"] = [l.strip() for l in err.splitlines()
                        if l.strip().startswith("strata decode timing") or "expert cache hit rate" in l
                        or "expert tiers" in l or "suffix drafts" in l]

    rows = samples(d / "rss.csv")
    if rows:
        r["peak_rss_gib"] = round(max(x["rss_bytes"] for x in rows) / 2 ** 30, 2)
        r["peak_max_proc_rss_gib"] = round(max(x["max_proc_rss"] for x in rows) / 2 ** 30, 2)
        r["read_bytes_gib"] = round((rows[-1]["read_bytes"] - rows[0]["read_bytes"]) / 2 ** 30, 2)
        r["majflt"] = int(rows[-1]["majflt"] - rows[0]["majflt"])
        r["minflt"] = int(rows[-1]["minflt"] - rows[0]["minflt"])
        r["cached_before_gib"] = round(rows[0]["cached_kb"] / 2 ** 20, 1)
        r["cached_after_gib"] = round(rows[-1]["cached_kb"] / 2 ** 20, 1)
        r["mlocked_peak_gib"] = round(max(x["mlocked_kb"] for x in rows) / 2 ** 20, 2)
        if tl.get("ready") is not None:
            s = at(rows, tl["ready"])
            r["rss_at_ready_gib"] = round(s.get("rss_bytes", 0) / 2 ** 30, 2)
            r["epoch_ready"] = tl["ready"]
            r["epoch_first_sample"] = rows[0]["epoch"]
    r["generated_ids"] = [int(l.split()[1]) for l in out.splitlines() if l.startswith("T ")]
    r["answer_text_file"] = str(d / "answer.txt")
    return r


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    reps = [report(pathlib.Path(a)) for a in args]
    if as_json:
        print(json.dumps(reps, indent=1))
        return 0
    for r in reps:
        print("=" * 100)
        print(r["tag"], r.get("dir"))
        if "done" in r:
            d = r["done"]
            print("  prompt %d tokens, prefill %.1f tok/s (%.0f ms), %d generated, decode %.2f tok/s (%.0f ms), "
                  "stop=%s, drafts %d/%d" % (d["prompt_tokens"], d["prefill_tok_s"], d["prompt_ms"], d["generated"],
                                             d["decode_tok_s"], d["decode_ms"], d["stop"], d["drafts_accepted"],
                                             d["drafts_offered"]))
        if "prefill_marginal_tok_s" in r:
            print("  prefill marginal (last-first PP chunk): %.1f tok/s   TTFT %.3f s   decode wall %.3f s   "
                  "request wall %.3f s" % (r["prefill_marginal_tok_s"], r.get("ttft_s", -1),
                                           r.get("decode_wall_s", -1), r.get("request_wall_s", -1)))
        if "streaming" in r:
            s = r["streaming"]
            print("  KV streaming: %d of %d cells per QSA layer in VRAM, %.2f GiB pinned RAM (%.3f KiB/context token)"
                  % (s["resident_cells"], s["context_cells"], s["pinned_gib"], s["pinned_kib_per_token"]))
        if "hits" in r:
            h = r["hits"]
            print("  KV streaming block reads: %.2f%% of %d hit VRAM, %.1f MiB read from RAM"
                  % (h["pct"], h["lookups"], h["ram_mib"]))
        if "info" in r:
            i = r["info"]
            print("  INFO: context=%s kv=%s kv_resident=%s expert_slots=%s expert_cache_mib=%s vram_free_mib=%s spec=%s"
                  % (i.get("context"), i.get("kv"), i.get("kv_resident"), i.get("expert_slots"),
                     i.get("expert_cache_mib"), i.get("vram_free_mib"), i.get("spec")))
        if "peak_rss_gib" in r:
            print("  RSS: peak %.2f GiB, at READY %.2f GiB, peak proc %.2f GiB, Mlocked peak %.2f GiB"
                  % (r["peak_rss_gib"], r.get("rss_at_ready_gib", -1), r["peak_max_proc_rss_gib"], r["mlocked_peak_gib"]))
            print("  SSD/RAM: read_bytes +%.2f GiB, majflt +%d, minflt +%d, Cached %.1f -> %.1f GiB"
                  % (r["read_bytes_gib"], r["majflt"], r["minflt"], r["cached_before_gib"], r["cached_after_gib"]))
        for l in r.get("cache_lines", [])[:12]:
            print("   |", l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
