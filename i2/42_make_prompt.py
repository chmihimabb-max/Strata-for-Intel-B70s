#!/usr/bin/env python3
"""I2: build the exact prompt for an engine logits dump at a divergence point.

    python3 i2/42_make_prompt.py <out.txt> <base> <extra>
      base   : a key of llama-streams.json ("capital", "chat-turn", "code") or "bench" (llama-bench.json)
      extra  : "ids=1,2,3" (append these ids) or "prefix=N" (append the oracle's first N generated ids)

The rows a dump holds are the positions that PREDICT the next token, so appending the agreed prefix and
asking for one token gives the distribution AT the divergent index.
"""
import json
import pathlib
import sys

I2 = pathlib.Path("/home/michael/strata-xpu/strata/i2")
out = pathlib.Path(sys.argv[1])
base, extra = sys.argv[2], sys.argv[3]
if base == "bench":
    src = json.loads((I2 / "llama-bench.json").read_text(encoding="utf-8"))
    ids = list(src["prompt_ids"])
    gen = src["gen_ids"]
else:
    src = json.loads((I2 / "llama-streams.json").read_text(encoding="utf-8"))
    ids = list(src["prompts"][base]["prompt_ids"])
    gen = src["prompts"][base]["gen_ids"]
if extra.startswith("prefix="):
    ids += gen[:int(extra.split("=", 1)[1])]
elif extra.startswith("ids="):
    ids += [int(t) for t in extra.split("=", 1)[1].split(",") if t]
else:
    raise SystemExit("bad extra: %s" % extra)
out.write_text(",".join(str(int(t)) for t in ids), encoding="utf-8")
print("prompt tokens: %d -> %s" % (len(ids), out))
print("last 12: %s" % ids[-12:])
