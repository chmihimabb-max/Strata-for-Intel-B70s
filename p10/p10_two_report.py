#!/usr/bin/env python3
"""P10: the two-instance / concurrency arm's numbers, per instance and aggregate.

usage: p10_two_report.py <arm_dir>

Reads the config each instance ran with (cfg-a.json / cfg-b.json / cfg-1x2.json) for its engine log path, then the
engine's OWN lines out of that log (its --stats), the client's timing JSON, the fdinfo monitor CSV and the
per-thread CSV.  Nothing is re-derived from wall clocks where the engine reported the number itself.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys


def grab(pat, text, flags=0):
    m = re.search(pat, text, flags)
    return m.group(1).strip() if m else None


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def instance(d, cfg_name, idx):
    o = {"idx": idx, "cfg": cfg_name}
    try:
        cfg = json.loads(open(os.path.join(d, cfg_name)).read())
    except OSError:
        return None
    o["gpu"] = cfg.get("gpu")
    o["port"] = cfg.get("port")
    args = cfg.get("args", [])
    o["pool_workers_arg"] = args[args.index("--pool-workers") + 1] if "--pool-workers" in args else "default"
    o["pool_affinity_arg"] = args[args.index("--pool-affinity") + 1] if "--pool-affinity" in args else "default"
    o["max_context"] = args[args.index("--max-context") + 1] if "--max-context" in args else None
    log = cfg.get("log")
    o["engine_log"] = log
    try:
        err = open(log, errors="replace").read()
    except OSError as ex:
        o["engine_log_error"] = repr(ex)
        return o
    o["built_line"] = grab(r"(strata generate: \d+ expert-pool workers.*)", err)
    o["auto_line"] = grab(r"(strata generate: expert cache auto: .*)", err)
    o["cache_line"] = grab(r"(strata generate: expert cache \d+ slots.*)", err)
    o["resident_line"] = grab(r"(strata generate: token graph hit path: .*)", err)
    o["split_resident"] = grab(r"(strata serve: layer split: [\d.]+% of the experts resident.*)", err)
    o["split_line"] = grab(r"(strata generate: layer split auto: K=.*)", err)
    o["decode_timing"] = grab(r"(strata decode timing: .*)", err)
    o["prompt_line"] = grab(r"(strata serve: prompt \d+ tokens = .*)", err)
    o["tiers_line"] = grab(r"(strata serve: expert tiers: .*)", err)
    o["hit_line"] = grab(r"(strata serve: decode expert cache hit rate: .*)", err)
    o["vram_free_loaded"] = grab(r"strata serve: (\d+) MiB of VRAM free with everything loaded", err)
    o["kv_stream"] = grab(r"(strata serve: KV streaming: .*)", err)
    pl = o["prompt_line"]
    if pl:
        o["prompt_tokens"] = fnum(grab(r"prompt (\d+) tokens", pl))
        o["prompt_ms"] = fnum(grab(r"read in ([\d.]+) ms", pl))
        o["prefill_tok_s"] = fnum(grab(r"read in [\d.]+ ms \(([\d.]+) tok/s\)", pl))
        o["gen_tokens"] = fnum(grab(r"(\d+) generated", pl))
        o["decode_ms"] = fnum(grab(r"generated in ([\d.]+) ms", pl))
        o["decode_tok_s"] = fnum(grab(r"generated in [\d.]+ ms \(([\d.]+) tok/s\)", pl))
    dt = o["decode_timing"]
    if dt:
        o["windows"] = fnum(grab(r"(\d+) windows", dt))
        o["window_ms"] = fnum(grab(r"([\d.]+) ms/window", dt))
        o["cpu_experts_per_lw"] = fnum(grab(r"per layer-window: CPU experts ([\d.]+)", dt))
        o["cpu_entries_per_lw"] = fnum(grab(r"CPU experts [\d.]+ \(([\d.]+) entries\)", dt))
        o["vram_hits_per_lw"] = fnum(grab(r"VRAM hits ([\d.]+)", dt))
        o["per_layer_cpu_ms"] = fnum(grab(r"jobs [\d.]+ CPU ([\d.]+)", dt))
        o["per_layer_jobs_ms"] = fnum(grab(r"jobs ([\d.]+)", dt))
    tl = o["tiers_line"]
    if tl:
        o["file_blobs"] = fnum(grab(r"files (\d+) blobs", tl))
        o["file_mb"] = fnum(grab(r"([\d.]+) MB read", tl))
        o["ram_blobs"] = fnum(grab(r"RAM (\d+) blobs", tl))
    # the client's side
    try:
        t = json.loads(open(os.path.join(d, f"resp-{idx}.json.timing.json")).read())
        o["wall_s"] = t.get("wall_s")
        o["usage"] = t.get("usage")
        o["finish_reason"] = t.get("finish_reason")
        o["answer_head"] = (t.get("answer_head") or "")[:80]
    except OSError:
        o["wall_s"] = None
    # fdinfo peaks
    try:
        with open(os.path.join(d, f"monitor-{idx}.csv")) as f:
            rows = list(csv.DictReader(f))
        o["peak_card0_gib"] = round(max(int(r["vram0_kib"]) for r in rows) / 1048576.0, 2)
        o["peak_card1_gib"] = round(max(int(r["vram1_kib"]) for r in rows) / 1048576.0, 2)
        o["peak_rss_gib"] = round(max(int(r["hwm_kib"]) for r in rows) / 1048576.0, 2)
        o["peak_gtt_gib"] = round(max(int(r["gtt0_kib"]) for r in rows) / 1048576.0, 2)
    except (OSError, KeyError, ValueError):
        pass
    return o


def main() -> int:
    d = sys.argv[1].rstrip("/")
    out = {"arm": os.path.basename(d), "instances": []}
    mode = "2i"
    try:
        head = open(os.path.join(d, "log.txt"), errors="replace").read(400)
        if "mode=1x2" in head:
            mode = "1x2"
    except OSError:
        pass
    cfgs = ("cfg-1x2.json",) if mode == "1x2" else ("cfg-a.json", "cfg-b.json")
    for idx, cfg in enumerate(cfgs):
        if not os.path.exists(os.path.join(d, cfg)):
            continue
        i = instance(d, cfg, idx)
        if i:
            out["instances"].append(i)
    out["mode"] = mode
    if mode == "1x2":
        # one engine, two clients: the requests are serialized by the server, so the per-request numbers come
        # from each client's own timing JSON and the engine's prompt lines (in the order they ran)
        reqs = []
        for i in (0, 1):
            try:
                t = json.loads(open(os.path.join(d, f"resp-{i}.json.timing.json")).read())
            except OSError:
                continue
            tm = t.get("timings") or {}
            reqs.append({"client": i, "wall_s": t.get("wall_s"), "prompt_tokens": tm.get("prompt_n"),
                         "prompt_ms": tm.get("prompt_ms"), "prompt_tok_s": tm.get("prompt_per_second"),
                         "generated": tm.get("predicted_n"), "decode_ms": tm.get("predicted_ms"),
                         "decode_tok_s": tm.get("predicted_per_second"), "error": t.get("error")})
        reqs.sort(key=lambda r: r.get("wall_s") or 0)
        out["requests"] = reqs
        tot = sum(r["generated"] or 0 for r in reqs)
        wall = max((r["wall_s"] or 0) for r in reqs) if reqs else 0
        out["aggregate"] = {"note": "serialized by the server: the two wall clocks are queue positions",
                            "generated_tokens": tot, "wall_of_last_request_s": wall,
                            "end_to_end_tok_s": round(tot / wall, 2) if wall else None,
                            "sum_of_decode_phases_tok_s": round(sum((r["decode_tok_s"] or 0) for r in reqs), 2)}
    # aggregate: the two requests ran concurrently, so the aggregate rate is the tokens they produced
    if mode != "1x2":
        tot_tok = sum(i.get("gen_tokens") or 0 for i in out["instances"])
        wall = max((i.get("wall_s") or 0) for i in out["instances"]) if out["instances"] else 0
        out["aggregate"] = {"generated_tokens": tot_tok, "wall_s": wall,
                            "aggregate_decode_tok_s": round(tot_tok / wall, 2) if wall else None,
                            "sum_per_request_decode_tok_s": round(sum((i.get("decode_tok_s") or 0)
                                                                      for i in out["instances"]), 2)}
    print(json.dumps(out, indent=1))
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[two] {out['aggregate']}")
    for i in out["instances"]:
        print(f"[two] instance {i['idx']} gpu={i['gpu']} pool={i['pool_workers_arg']} "
              f"wall={i.get('wall_s')}s prefill={i.get('prefill_tok_s')} decode={i.get('decode_tok_s')} tok/s "
              f"window={i.get('window_ms')}ms CPU/lw={i.get('cpu_experts_per_lw')} "
              f"rss={i.get('peak_rss_gib')}GiB card0={i.get('peak_card0_gib')} card1={i.get('peak_card1_gib')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
