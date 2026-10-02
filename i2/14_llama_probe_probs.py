#!/usr/bin/env python3
"""I2: the oracle's own token distribution at every generated position (n_probs), so a divergence can be
attributed instead of guessed.

    python3 i2/14_llama_probe_probs.py <port> <out.json> [n_predict] [n_probs]

Same prompts and sampler as 11_llama_probe.py, plus `n_probs`: the top-N candidates the oracle scored at
each step (pre-sampling logits), which is what an independent implementation must be compared against.
"""
import json
import pathlib
import sys
import urllib.request

PORT = int(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
NPRED = int(sys.argv[3]) if len(sys.argv) > 3 else 40
NPROBS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
STREAMS = pathlib.Path("/home/michael/strata-xpu/strata/i2/llama-streams.json")


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


res = {"port": PORT, "n_predict": NPRED, "n_probs": NPROBS, "prompts": {}}
src = json.loads(STREAMS.read_text(encoding="utf-8"))
for name, p in src["prompts"].items():
    ids = p["prompt_ids"]
    r = post("/completion", {
        "prompt": ids, "n_predict": NPRED, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": 1234,
        "return_tokens": True, "cache_prompt": False, "stream": False, "n_probs": NPROBS,
    })
    entry = {"prompt_ids": ids, "gen_ids": r.get("tokens", []), "gen_text": r.get("content", ""),
             "timings": r.get("timings"), "keys": sorted(r.keys())}
    cp = r.get("completion_probabilities")
    if cp is None:
        for k in r:
            if "prob" in k:
                cp = r[k]
                entry["probs_field"] = k
                break
    entry["completion_probabilities"] = cp
    res["prompts"][name] = entry
    print("== %s: %d gen; probs field=%s" % (name, len(entry["gen_ids"]), entry.get("probs_field", "completion_probabilities")))
    if cp:
        for i, pr in enumerate(cp[:6]):
            top = pr.get("top_logprobs") or pr.get("probs") or []
            print("   pos %2d chosen=%s logprob=%.4f  top: %s" % (
                i, pr.get("id"), pr.get("logprob", float("nan")),
                [(t.get("id"), round(t.get("logprob", float("nan")), 4)) for t in top[:NPROBS]]))
    print("   response keys:", entry["keys"])
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("wrote", OUT)
