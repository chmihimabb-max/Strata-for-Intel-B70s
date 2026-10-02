#!/usr/bin/env python3
"""I2: merge the oracle's per-position probabilities for the four requests into ONE file whose prompt
order matches the engine's request order (capital, chat-turn, code, bench), so the margin analysis can
walk both streams together.

    python3 i2/16_merge_probs.py <three-probs.json> <bench-probs.json> <out.json>
"""
import json
import pathlib
import sys

three = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
bench = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
out = {"prompts": {}}
for name, p in three["prompts"].items():
    out["prompts"][name] = {"prompt_ids": p["prompt_ids"], "gen_ids": p["gen_ids"],
                            "completion_probabilities": p["completion_probabilities"]}
out["prompts"]["bench"] = {"prompt_ids": bench["prompt_ids"] if "prompt_ids" in bench else None,
                          "gen_ids": bench["gen_ids"],
                          "completion_probabilities": bench["completion_probabilities"]}
pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote %s with %s" % (sys.argv[3], list(out["prompts"].keys())))
