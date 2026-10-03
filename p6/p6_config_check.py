#!/usr/bin/env python3
"""Print the effective engine command line serve/server.py will run from the config of record."""
import json
import sys

CFG = "/home/michael/strata-xpu/strata/strata-sycl-iq3s.json"
cfg = json.load(open(CFG, encoding="utf-8-sig"))
args = list(cfg["args"])
gpu = cfg.get("gpu") or []
if len(gpu) > 1 and "--layer-split" not in args:
    args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
print("config      :", CFG)
print("gpu         :", gpu)
print("port        :", cfg.get("port"))
print("log (engine):", cfg.get("log"))
print("model_name  :", cfg.get("model_name"))
print("tokenizer   :", cfg.get("tokenizer"))
print("engine argv :", cfg["exe"], " ".join(args))
for key in ("--max-context", "--kv-resident", "--prefill", "--kv", "--spec", "--mtp", "--expert-cache"):
    if key in args:
        print(f"  {key} = {args[args.index(key) + 1]}")
    else:
        print(f"  {key} = (not set)")
sys.exit(0)
