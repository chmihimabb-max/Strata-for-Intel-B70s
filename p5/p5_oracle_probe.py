#!/usr/bin/env python3
"""P5: one greedy request to the oracle, with its own per-position top-N logprobs.

    /usr/bin/python3 p5/p5_oracle_probe.py <port> <prompt-file> <out.json> [n_predict] [n_probs] [tag]

* `<prompt-file>` is a comma-separated id list (the same file the engine arm is fed), so the model file, the
  quantisation, the tokenizer and the prompt are identical on both sides by construction.
* sampler: temperature 0, top_k 1, top_p 1, min_p 0, repeat_penalty 1 - i.e. what the engine's serve protocol
  documents for a GEN line with no sampling keys (greedy, no penalties).
* `n_probs` records the oracle's own top-N candidates at every generated position, which is what makes a
  divergence attributable: a flip at a 0.07-nat margin is not the same evidence as a flip at 3 nats.
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
NPRED = int(sys.argv[4]) if len(sys.argv) > 4 else 256
NPROBS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
TAG = sys.argv[6] if len(sys.argv) > 6 else PROMPT.stem

ids = [int(x) for x in PROMPT.read_text().strip().split(",")]
print("== oracle request tag=%s port=%d prompt=%s (%d ids) n_predict=%d n_probs=%d"
      % (TAG, PORT, PROMPT.name, len(ids), NPRED, NPROBS), flush=True)

payload = {
    "prompt": ids, "n_predict": NPRED, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": 1234,
    "return_tokens": True, "cache_prompt": False, "stream": False, "n_probs": NPROBS,
}
req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT,
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    res = json.loads(r.read().decode())
wall = time.time() - t0
tm = res.get("timings") or {}
entry = {
    "tag": TAG, "port": PORT, "prompt_file": str(PROMPT), "prompt_tokens": len(ids),
    "n_predict": NPRED, "n_probs": NPROBS, "wall_s": round(wall, 2),
    "gen_ids": res.get("tokens", []), "gen_text": res.get("content", ""),
    "timings": tm,
    "completion_probabilities": res.get("completion_probabilities"),
    "response_keys": sorted(res.keys()),
}
if entry["completion_probabilities"] is None:
    for k in res:
        if "prob" in k:
            entry["completion_probabilities"] = res[k]
            entry["probs_field"] = k
            break
OUT.write_text(json.dumps(entry, indent=1), encoding="utf-8")
print("== got %d ids in %.1f s wall (prompt %s tok, %s eval/s; predicted %s tok, %s tok/s)"
      % (len(entry["gen_ids"]), wall, tm.get("prompt_n"), tm.get("prompt_per_second"),
         tm.get("predicted_n"), tm.get("predicted_per_second")), flush=True)
cp = entry["completion_probabilities"] or []
print("== probs field=%s, %d positions" % (entry.get("probs_field", "completion_probabilities"), len(cp)), flush=True)
for i, pr in enumerate(cp[:4]):
    top = pr.get("top_logprobs") or pr.get("probs") or []
    print("   pos %2d chosen=%s logprob=%.4f top: %s"
          % (i, pr.get("id"), pr.get("logprob", float("nan")),
             [(t.get("id"), round(t.get("logprob", float("nan")), 4)) for t in top]), flush=True)
print("== wrote %s" % OUT, flush=True)
