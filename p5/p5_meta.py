#!/usr/bin/env python3
"""P5: what the GGUF itself says about context length / rope scaling (the oracle has to be launched at the same
max-context as the engine arm, so its own metadata decides whether 32K/128K are inside the model's trained
window at all).

    /usr/bin/python3 p5/p5_meta.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "/home/michael/strata-xpu/strata/tools")
from gguf_reader import GGUFFile  # noqa: E402

SNAP = (pathlib.Path.home() / ".cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        / "snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S")
KEYS = ("ctx", "context", "rope", "yarn", "attention", "head", "block_count", "expert", "general.name", "arch")
for shard in sorted(SNAP.glob("*.gguf")):
    g = GGUFFile(shard)
    print("== %s  v%d, %d tensors, %d metadata keys" % (shard.name, g.version, len(g.tensors), len(g.metadata)))
    for k, v in g.metadata.items():
        if any(t in k for t in KEYS):
            print("   %-52s %s" % (k, repr(v)[:160]))
    print("   tensor types:", g.by_type())
