#!/usr/bin/env python3
"""I2: split OUR engine's stdout into one stream per request, and diff it against the oracle.

    python3 i2/30_diff.py <llama-streams.json> [bench.json] <engine-tag-out.txt> [more out.txt ...]

Prints, per prompt, the two token-id streams and the first divergent index, and writes
i2/divergence.json.  Reads the engine's own DONE lines for the request's numbers.
"""
import json
import pathlib
import sys

root = pathlib.Path("/home/michael/strata-xpu/strata/i2")
streams = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
rest = sys.argv[2:]
bench = None
if rest and rest[0].endswith("llama-bench.json"):
    bench = json.loads(pathlib.Path(rest[0]).read_text(encoding="utf-8"))
    rest = rest[1:]

gold = {}
for name, p in streams["prompts"].items():
    gold[name] = {"prompt_ids": p["prompt_ids"], "gen_ids": p.get("gen_ids", []),
                  "prompt_text": p["prompt_text"], "timings": p.get("timings")}
if bench:
    gold["bench"] = {"prompt_ids": bench["prompt_ids"], "gen_ids": bench["gen_ids"],
                     "prompt_text": bench["nonce"], "timings": {
                         "prompt_n": bench["prompt_tokens"], "predicted_n": bench["gen_tokens"],
                         "prompt_per_second": bench["prompt_per_second"],
                         "predicted_per_second": bench["predicted_per_second"]}}
order = list(gold.keys())


def parse_engine(path):
    """the engine's stdout: T <id> lines between DONE lines; DONE carries the request's own numbers"""
    streams_out, dones, cur = [], [], []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("T "):
            cur.append(int(line[2:]))
        elif line.startswith("DONE"):
            streams_out.append(cur)
            dones.append(line)
            cur = []
    if cur:
        streams_out.append(cur)
    return streams_out, dones


report = {"oracle": {"file": str(sys.argv[1]), "n_predict": streams.get("n_predict")},
          "runs": [], "divergence": {}}
for path in rest:
    tag = pathlib.Path(path).stem
    eng, dones = parse_engine(path)
    run = {"tag": tag, "file": path, "done_lines": dones, "prompts": {}, "mismatches": 0, "matched": 0}
    print("=" * 100)
    print("RUN %s   (%d engine streams, %d oracle prompts)" % (tag, len(eng), len(order)))
    for i, name in enumerate(order):
        g = gold[name]["gen_ids"]
        e = eng[i] if i < len(eng) else []
        n = min(len(g), len(e))
        div = next((k for k in range(n) if g[k] != e[k]), None)
        if div is None and len(g) != len(e):
            div = n  # length-only divergence
        verdict = "IDENTICAL" if (div is None and len(g) == len(e)) else ("DIVERGE@%s" % div)
        if verdict == "IDENTICAL":
            run["matched"] += 1
        else:
            run["mismatches"] += 1
        run["prompts"][name] = {"oracle_ids": g, "engine_ids": e,
                                "first_divergence": div, "verdict": verdict}
        print("-- %-10s prompt %4d tok  %s" % (name, len(gold[name]["prompt_ids"]), verdict))
        print("   oracle: %s" % g[:48])
        print("   engine: %s" % e[:48])
        if div is not None:
            print("   first divergence at index %d: oracle=%s engine=%s (prefix length %d)"
                  % (div, g[div] if div < len(g) else None, e[div] if div < len(e) else None, n))
    report["runs"].append(run)

OUT = root / "divergence.json"
report["divergence"] = {r["tag"]: {k: v["first_divergence"] for k, v in r["prompts"].items()}
                        for r in report["runs"]}
OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
print("=" * 100)
print("divergence table:", json.dumps(report["divergence"], indent=1))
print("wrote", OUT)
