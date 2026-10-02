#!/usr/bin/env python3
"""I2: the perf request, run on the ALREADY-RUNNING oracle server (round 2 of 2, per ~/flashnext-opt's
protocol: cold prefill via a nonce-prefixed prompt, >=256 generated tokens, page-cache-resident check).

    python3 i2/13_llama_bench.py <port> <out.json> [prompt_tokens] [gen_tokens]

Writes the token ids of the perf prompt too, so the Strata engine is asked the identical question.
"""
import json
import pathlib
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 2700
GEN = int(sys.argv[4]) if len(sys.argv) > 4 else 256
NONCE = "STRATA-I2-ORACLE-nonce-" + (sys.argv[5] if len(sys.argv) > 5 else "7e31ab")
DOCS = ["/home/michael/strata-xpu/strata/docs/AMD_HIP.md",
        "/home/michael/strata-xpu/strata/docs/MULTI_GPU.md",
        "/home/michael/strata-xpu/strata/docs/DETAILS.md"]
MNT = "/home/michael/.cache"


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def dev_read_bytes():
    """bytes the block device has served, summed over the nvme partitions of /"""
    total = 0
    for line in pathlib.Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) > 5 and f[2].startswith("nvme"):
            total += int(f[5]) * 512
    return total


def proc_read_bytes(pid):
    try:
        for line in pathlib.Path("/proc/%d/io" % pid).read_text().splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


text = NONCE + "\n" + "".join(pathlib.Path(p).read_text(encoding="utf-8") for p in DOCS
                              if pathlib.Path(p).exists())
ids = post("/tokenize", {"content": text})["tokens"]
head = post("/tokenize", {"content": NONCE + "\n"})["tokens"]
# truncate to the target size but keep the nonce at the front
if len(ids) > NP:
    ids = ids[:NP]
    if ids[:len(head)] != head:
        raise SystemExit("nonce lost by truncation")

import os
_pidfile = os.environ.get("STRATA_I2_PIDFILE") or str(pathlib.Path(sys.argv[2]).parent
                                                       / "llama-ctx4096-ncmoe0-server.pid")
try:
    pid = int(pathlib.Path(_pidfile).read_text().strip())
except (OSError, ValueError):
    pid = -1
d0, p0 = dev_read_bytes(), proc_read_bytes(pid)
t0 = time.time()
r = post("/completion", {
    "prompt": ids,
    "n_predict": GEN,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 1234,
    "return_tokens": True,
    "cache_prompt": False,
    "stream": False,
})
wall = time.time() - t0
d1, p1 = dev_read_bytes(), proc_read_bytes(pid)
t = r.get("timings", {})
res = {
    "protocol": "flashnext-opt of record: nonce-prefixed prompt (cold prefix), >=256 generated tokens, "
                "page-cache-resident check",
    "nonce": NONCE,
    "prompt_tokens": t.get("prompt_n"),
    "gen_tokens": t.get("predicted_n"),
    "prompt_per_second": t.get("prompt_per_second"),
    "predicted_per_second": t.get("predicted_per_second"),
    "prompt_ms": t.get("prompt_ms"),
    "predicted_ms": t.get("predicted_ms"),
    "wall_s": round(wall, 2),
    "cache_n": t.get("cache_n"),
    "device_read_bytes_during_request": d1 - d0,
    "proc_read_bytes_during_request": (None if p0 is None or p1 is None else p1 - p0),
    "prompt_ids": ids,
    "gen_ids": r.get("tokens", []),
    "gen_text": r.get("content", ""),
    "stop": r.get("stop_type"),
}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("prompt_ids", "gen_ids", "gen_text")}, indent=1))
print("gen_ids[:16]:", res["gen_ids"][:16])
print("text[:200]:", json.dumps(res["gen_text"][:200]))
print("wrote", OUT)
