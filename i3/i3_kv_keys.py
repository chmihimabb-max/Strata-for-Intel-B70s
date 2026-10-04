#!/usr/bin/env python3
"""i3_kv_keys.py - list the template GGUF's KV keys (names + type sizes) without printing big values."""
from __future__ import annotations

import pathlib
import struct
import sys

sys.path.insert(0, "/home/michael/strata-xpu/strata/tools")
from gguf_reader import GGUFFile  # noqa: E402

GOOD = (pathlib.Path.home() / ".cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        / "snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S"
        / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")

for p in sys.argv[1:] or [str(GOOD)]:
    g = GGUFFile(pathlib.Path(p))
    print(f"# {p}")
    print(f"# version={g.version} tensors={len(g.tensors)} kv={len(g.metadata)} alignment={g.alignment} data_start={g.data_start} size={pathlib.Path(p).stat().st_size}")
    for k, v in g.metadata.items():
        if isinstance(v, list):
            s = f"array[{len(v)}] first={v[0]!r}" if v else "array[0]"
        else:
            s = repr(v)
        if len(s) > 90:
            s = s[:90] + "..."
        print(f"  {k:55s} {s}")
