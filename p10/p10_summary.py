#!/usr/bin/env python3
"""P10: one arm's numbers, pulled out of what the engine itself printed (no re-derivation).

usage: p10_summary.py <arm_dir> [--json]

Everything comes from files the arm left behind: err.txt / out.txt (the engine's own --stats lines), the driver
fdinfo monitor CSV, and the markers.  Fields the arm did not produce are reported as "-", never as 0.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys


def grab(pat, text, group=1, flags=0):
    m = re.search(pat, text, flags)
    return m.group(group).strip() if m else None


def num(v):
    if v is None:
        return None
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return v


def main() -> int:
    d = sys.argv[1].rstrip("/")
    want_json = "--json" in sys.argv
    err = open(os.path.join(d, "err.txt"), errors="replace").read()
    try:
        out = open(os.path.join(d, "out.txt"), errors="replace").read()
    except OSError:
        out = ""
    o: dict = {"arm": os.path.basename(d)}

    o["mask"] = grab(r"onecard=(\S+)", open(os.path.join(d, "log.txt"), errors="replace").read())
    o["engine_argv_context"] = grab(r"--max-context (\d+)", open(os.path.join(d, "log.txt"),
                                                                 errors="replace").read())
    o["auto_line"] = grab(r"(expert cache auto: .*)", err)
    o["cache_line"] = grab(r"(expert cache \d+ slots, .*)", err)
    o["split_line"] = grab(r"(layer split auto: K=\d+.*)", err)
    o["split_resident_line"] = grab(r"(layer split: [\d.]+% of the experts resident.*)", err)
    o["prefill_line"] = grab(r"(layer split auto: CUDA0.*)", err)
    info = grab(r"(INFO context=.*)", out) or grab(r"(INFO context=.*)", err)
    o["info_line"] = info
    if info:
        for k in ("context", "kv_resident", "expert_slots", "expert_cache_mib", "expert_slots_primary",
                  "vram_free_mib", "pool_workers", "pcie_frac"):
            v = grab(k + r"=(\S+)", info)
            o["info_" + k] = num(v)
    o["prefilled"] = grab(r"(pre-filled \d+ of \d+ slots.*)", err)
    o["parm_line"] = grab(r"(strata generate: layer split: CUDA1 runs layers.*|strata generate: layer split: CUDA0 runs layers.*)", err)
    dt = grab(r"(strata decode timing: .*)", err)
    o["decode_timing"] = dt
    if dt:
        o["windows"] = num(grab(r"strata decode timing: (\d+) windows", err))
        o["window_ms"] = num(grab(r"([\d.]+) ms/window", err))
        o["tokens_per_window"] = num(grab(r"([\d.]+) tokens/window", err))
        o["cpu_experts_per_lw"] = num(grab(r"per layer-window: CPU experts ([\d.]+)", err))
        o["cpu_entries_per_lw"] = num(grab(r"CPU experts [\d.]+ \(([\d.]+) entries\)", err))
        o["vram_hits_per_lw"] = num(grab(r"VRAM hits ([\d.]+)", err))
        o["pcie_per_lw"] = num(grab(r"PCIe ([\d.]+)\s*$", err, flags=re.M))
        o["host_per_layer_ms"] = num(grab(r"per-layer host ([\d.]+)", err))
        o["per_layer_jobs_ms"] = num(grab(r"jobs ([\d.]+)", err))
        o["per_layer_cpu_ms"] = num(grab(r"jobs [\d.]+ CPU ([\d.]+)", err))
        o["gpu_reach_wait_ms"] = num(grab(r"GPU-reach wait ([\d.]+)", err))
        o["draft_ms"] = num(grab(r"\+ draft ([\d.]+)", err))
    sp = grab(r"(strata serve: prompt \d+ tokens = .*)", err)
    o["serve_prompt_line"] = sp
    if sp:
        o["prompt_tokens"] = num(grab(r"prompt (\d+) tokens", sp))
        o["prompt_ms"] = num(grab(r"read in ([\d.]+) ms", sp))
        o["prefill_tok_s"] = num(grab(r"read in [\d.]+ ms \(([\d.]+) tok/s\)", sp))
        o["gen_tokens"] = num(grab(r"(\d+) generated", sp))
        o["decode_ms"] = num(grab(r"generated in ([\d.]+) ms", sp))
        o["decode_tok_s"] = num(grab(r"generated in [\d.]+ ms \(([\d.]+) tok/s\)", sp))
        o["drafts_accepted"] = grab(r"drafts accepted (\d+ of \d+)", sp)
    o["hit_rate_line"] = grab(r"(strata serve: decode expert cache hit rate: .*)", err)
    o["tiers_line"] = grab(r"(strata serve: expert tiers: .*)", err)
    o["resident_ram_line"] = grab(r"(strata serve: resident RAM: .*)", err)
    o["prefill_experts_line"] = grab(r"(strata generate: prefill .*experts .*)", err)
    o["ple_line"] = grab(r"(strata generate: prefill .*PLE.*)", err)
    o["vram_free_loaded"] = grab(r"strata serve: (\d+) MiB of VRAM free with everything loaded", err)
    o["kv_stream_line"] = grab(r"(strata serve: KV streaming: .*)", err)
    o["done"] = grab(r"(^DONE .*$)", out, flags=re.M)
    if o["done"]:
        f = o["done"].split()
        # DONE <gen> <prompt> <prompt ms> <decode ms> <finish> <acc> <off> <reused> [hits] [lookups]
        #      [RAM blobs] [file blobs] [file MB]
        if len(f) >= 13:
            o["req_gpu_hits"] = int(f[9])
            o["req_gpu_lookups"] = int(f[10])
            o["req_ram_blobs"] = int(f[11])
            o["req_file_blobs"] = int(f[12])
            o["req_file_mb"] = float(f[13]) if len(f) >= 14 else None
    o["t_lines"] = len(re.findall(r"^T ", out, flags=re.M))
    o["pp_last"] = grab(r"(^PP .*$)", out, flags=re.M | re.S)
    pp = re.findall(r"^PP .*$", out, flags=re.M)
    o["pp_last"] = pp[-1] if pp else None
    o["pp_lines"] = len(pp)
    o["answer_ids_md5"] = grab(r"T lines: \d+  md5 (\w+)", open(os.path.join(d, "log.txt"),
                                                               errors="replace").read())

    # the driver's own accounting, peaks over the arm
    peak = {}
    try:
        with open(os.path.join(d, "monitor.csv")) as f:
            rows = list(csv.DictReader(f))
        for k in ("vram0_kib", "resident0_kib", "gtt0_kib", "vram1_kib", "resident1_kib", "gtt1_kib",
                  "rss_kib", "hwm_kib"):
            peak[k] = max(int(r[k]) for r in rows) if rows else None
        o["peak_card0_gib"] = round(peak["vram0_kib"] / 1048576.0, 2)
        o["peak_card1_gib"] = round(peak["vram1_kib"] / 1048576.0, 2)
        o["peak_rss_gib"] = round(peak["hwm_kib"] / 1048576.0, 2)
        o["peak_gtt0_gib"] = round(peak["gtt0_kib"] / 1048576.0, 2)
        o["monitor_samples"] = len(rows)
    except (OSError, KeyError, ValueError) as ex:
        o["monitor_error"] = repr(ex)

    if want_json:
        print(json.dumps(o, indent=1))
        return 0
    keys = ["arm", "mask", "engine_argv_context", "auto_line", "cache_line", "split_line",
            "split_resident_line", "info_expert_slots", "info_expert_cache_mib", "info_vram_free_mib",
            "info_pool_workers", "prefilled", "cpu_experts_per_lw", "cpu_entries_per_lw", "vram_hits_per_lw",
            "pcie_per_lw", "host_per_layer_ms", "per_layer_jobs_ms", "per_layer_cpu_ms", "window_ms",
            "tokens_per_window", "prefill_tok_s", "decode_tok_s", "prompt_ms", "decode_ms", "drafts_accepted",
            "hit_rate_line", "tiers_line", "resident_ram_line", "prefill_experts_line", "vram_free_loaded",
            "kv_stream_line", "peak_card0_gib", "peak_card1_gib", "peak_rss_gib", "peak_gtt0_gib",
            "req_gpu_hits", "req_gpu_lookups", "req_ram_blobs", "req_file_blobs", "req_file_mb",
            "answer_ids_md5", "done"]
    print(f"[summary {o['arm']}]")
    for k in keys:
        print(f"  {k:24s} {o.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
