#!/usr/bin/env python3
"""Write the OPTIONAL snappy variant of the config of record: 32K context, KV fully in VRAM, --prefill auto.

This is NOT what the resident server runs - the resident server runs the config of record (262144 + KV
streaming).  This variant is the measured 32K row of m6c/TABLE.md ("m6c-32k-auto": 344.8 tok/s prefill,
21.71 tok/s decode, peak VRAM 29,438.2+31,970.2 MiB, peak RSS 48.31 GiB) for interactive use, where no
prompt needs the 256K depth.

usage: p6_make_snappy.py            -> writes strata/p6/config-32768-snappy.json
"""
import json
from pathlib import Path

SRC = Path("/home/michael/strata-xpu/strata")
src = SRC / "strata-sycl-iq3s.json"
dst = SRC / "p6" / "config-32768-snappy.json"

cfg = json.loads(src.read_text(encoding="utf-8-sig"))
args = list(cfg["args"])
for flag, value in (("--max-context", "32768"), ("--kv-resident", "0"), ("--prefill", "auto")):
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
cfg["args"] = args
cfg["_note"] = (
    "P6 (card t_5dfc11a3) OPTIONAL interactive variant of the config of record (strata-sycl-iq3s.json): "
    "--max-context 32768, --kv-resident 0 (all KV in VRAM, streaming off), --prefill auto - i.e. exactly the "
    "M6c 'm6c-32k-auto' row (m6c/TABLE.md): 344.8 tok/s prefill, 21.71 tok/s decode, peak VRAM "
    "29,438.2+31,970.2 MiB, peak RSS 48.31 GiB.  Same model, same drafter, same GPUs; only the context and "
    "the KV chunking differ.  The resident server does NOT use this file - start it with "
    "p6_start.sh (config of record) unless you want the snappy arm: "
    "python -m serve.server --engine strata --config p6/config-32768-snappy.json --port 8099 --api-monitor"
)
dst.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {dst}")
print("engine argv:", cfg["exe"], " ".join(a for a in args))
