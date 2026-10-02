#!/usr/bin/env python3
"""I2: the oracle's side of the comparison.

Renders/ tokenises every prompt THROUGH llama.cpp's own tokenizer (the GGUF's), then generates
greedy (temperature 0, top_k 1) from those exact token ids and records the raw sampled ids.

The prompts are turned into ids here and fed to BOTH engines as ids, so tokenisation is removed as a
source of disagreement: the only thing left that can differ is the arithmetic.

    python3 i2/11_llama_probe.py <port> <out.json> [n_predict] [max_prompt_tokens]
"""
import json
import pathlib
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
NPRED = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MAXP = int(sys.argv[4]) if len(sys.argv) > 4 else 0

DOC = pathlib.Path("/home/michael/strata-xpu/strata/docs/AMD_HIP.md").read_text(encoding="utf-8")


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return json.loads(r.read().decode())


def tokenize(text):
    return post("/tokenize", {"content": text})["tokens"]


def apply_template(messages):
    """the model's OWN chat template, rendered by llama.cpp (so both engines get one string)"""
    r = post("/apply-template", {"messages": messages})
    return r["prompt"] if isinstance(r, dict) and "prompt" in r else r


def gen(ids, npred):
    t0 = time.time()
    r = post("/completion", {
        "prompt": ids,
        "n_predict": npred,
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
    return r, wall


def build_prompt_set():
    """returns list of (name, prompt_text, note)"""
    out = []
    out.append(("capital", "The capital of France is", "the I1/M5 greedy prompt, raw completion"))
    tmpl = apply_template([{"role": "user",
                            "content": "Name the three largest moons of Jupiter, one per line, no explanation."}])
    out.append(("chat-turn", tmpl, "the model's own chat template, rendered by llama.cpp /apply-template"))
    tmpl2 = apply_template([{"role": "user",
                             "content": "Write a Python function that returns the nth Fibonacci number, "
                                        "iteratively. Only the code."}])
    out.append(("code", tmpl2, "code-ish prompt through the same chat template"))
    return out


res = {"port": PORT, "n_predict": NPRED, "model": get("/v1/models")["data"][0]["id"], "prompts": {}}
t0 = time.time()
for name, text, note in build_prompt_set():
    ids = tokenize(text)
    if MAXP and len(ids) > MAXP:
        ids = ids[:MAXP]
    entry = {"note": note, "prompt_text": text, "prompt_ids": ids, "prompt_tokens": len(ids)}
    print("== %s: %d prompt tokens" % (name, len(ids)))
    print("   ids:", ids[:40], "..." if len(ids) > 40 else "")
    try:
        r, wall = gen(ids, NPRED)
        entry.update({
            "gen_ids": r.get("tokens", []),
            "gen_text": r.get("content", ""),
            "stop": r.get("stop_type"),
            "stop_reason": r.get("stopping_word"),
            "timings": r.get("timings"),
            "wall_s": round(wall, 3),
            "usage": {"prompt": (r.get("timings") or {}).get("prompt_n"),
                      "gen": (r.get("timings") or {}).get("predicted_n")},
        })
        print("   gen ids:", entry["gen_ids"])
        print("   text:", json.dumps(entry["gen_text"][:200]))
        print("   timings:", entry["timings"])
    except Exception as e:  # noqa: BLE001
        entry["error"] = str(e)
        print("   FAILED:", e)
    res["prompts"][name] = entry
res["wall_s"] = round(time.time() - t0, 2)
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("wrote", OUT)
