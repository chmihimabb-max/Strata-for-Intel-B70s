#!/usr/bin/env python3
"""I2: build the engine's stdin - one GEN line per prompt, using the ORACLE'S token ids verbatim.

    python3 i2/22_make_stdin.py <llama-streams.json> <max_new> [llama-bench.json <bench_max_new>] > <stdin>

No sampling keys: the engine documents absent keys as greedy, no penalties - the same thing the oracle
was asked for (temperature 0, top_k 1).  Order: correctness prompts first, then the perf prompt (so the
perf request is a later request of the same process, exactly the "round 2 of 2" the protocol wants).
"""
import json
import pathlib
import sys

d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
max_new = int(sys.argv[2])
lines = []
for name, p in d["prompts"].items():
    lines.append("GEN %d %s" % (max_new, ",".join(str(int(t)) for t in p["prompt_ids"])))
if len(sys.argv) > 4:
    b = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
    lines.append("GEN %d %s" % (int(sys.argv[4]), ",".join(str(int(t)) for t in b["prompt_ids"])))
lines.append("QUIT")
print("\n".join(lines))
