#!/usr/bin/env python3
"""I2: the oracle's distribution at a chosen generated position of the PERF prompt (the bench divergence
is at index 100), so the late divergence gets the same treatment as the early one.

    python3 i2/15_llama_bench_probs.py <port> <out.json> <n_predict> [n_probs] [bench_json]
"""
import json
import pathlib
import sys
import urllib.request

PORT = int(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
NPRED = int(sys.argv[3])
NPROBS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
SRC = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else pathlib.Path(
    "/home/michael/strata-xpu/strata/i2/llama-bench.json")
ids = json.loads(SRC.read_text(encoding="utf-8"))["prompt_ids"]


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


r = post("/completion", {
    "prompt": ids, "n_predict": NPRED, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": 1234,
    "return_tokens": True, "cache_prompt": False, "stream": False, "n_probs": NPROBS,
})
cp = r.get("completion_probabilities") or []
res = {"prompt_tokens": len(ids), "gen_ids": r.get("tokens", []), "timings": r.get("timings"),
       "completion_probabilities": cp}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
for i, pr in enumerate(cp):
    top = pr.get("top_logprobs") or pr.get("probs") or []
    if i >= len(cp) - 6 or i in (0, 99, 100, 101):
        print("pos %3d chosen=%s logprob=%.4f top=%s" % (
            i, pr.get("id"), pr.get("logprob", float("nan")),
            [(t.get("id"), round(t.get("logprob", float("nan")), 4)) for t in top[:NPROBS]]))
print("gen_ids[:105]:", res["gen_ids"][:105])
print("wrote", OUT)
