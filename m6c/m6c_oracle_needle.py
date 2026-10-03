#!/usr/bin/env python3
"""M6c: put the SAME 259,943-token needle prompt through the independent oracle (llama.cpp-SYCL, the
qwen4exp fork) on the same box and the same IQ3_S shards, so the 256K needle result can be attributed:

    if the oracle also fails to retrieve at 256K, the failure is the model/quantisation at this length;
    if the oracle retrieves it, our engine's 256K path is the suspect.

The prompt is sent as raw ids (`prompt`: [ids]) so the tokenizer cannot differ, with the oracle's own
greedy settings (temperature 0, top_k 1, seed 1234, cache_prompt false) exactly as I2 used.

    /usr/bin/python3 m6c/m6c_oracle_needle.py <port> <prompt-file> <out.json> [n_predict]
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
PROMPT = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])
GEN = int(sys.argv[4]) if len(sys.argv) > 4 else 256
ANSWER = "ZK-4471-QX"


def post(path, payload, timeout=7200):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


ids = [int(x) for x in PROMPT.read_text().split(",")]
print("prompt %d ids from %s" % (len(ids), PROMPT))
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
t = r.get("timings", {})
text = r.get("content", "")
res = {"prompt_tokens": t.get("prompt_n"), "gen_tokens": t.get("predicted_n"),
       "prompt_per_second": t.get("prompt_per_second"), "predicted_per_second": t.get("predicted_per_second"),
       "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms"), "wall_s": round(wall, 2),
       "stop": r.get("stop_type"), "answer_present": ANSWER in text,
       "gen_ids": r.get("tokens", []), "gen_text": text}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("gen_ids", "gen_text")}, indent=1))
print("text tail:", json.dumps(text[-400:]))
print("wrote", OUT)
