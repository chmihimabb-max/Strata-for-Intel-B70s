#!/usr/bin/env python3
"""I2: our engine's top candidates at a divergence point vs the oracle's, side by side.

    python3 i2/41_logits_top.py <dump.bin> <oracle-probs.json> <prompt-key> <pos> [vocab]

The dump is <int32 n_vocab, int32 n_rows> then f32 rows, one per stored position; --logits-stride
stores position 0 and the final input position, and the LAST row is the one that decides the next token.
"""
import json
import math
import pathlib
import struct
import sys

REPO = pathlib.Path("/home/michael/strata-xpu/strata")
sys.path.insert(0, str(REPO / "tools"))
PACKTOK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack/tokenizer")


def vocab():
    from strata_tokenizer import Tokenizer
    v = json.loads((PACKTOK / "vocab.json").read_text(encoding="utf-8"))
    merges = [m for m in (PACKTOK / "merges.txt").read_text(encoding="utf-8").split("\n") if m.strip()]
    types = json.loads((PACKTOK / "token_type.json").read_text(encoding="utf-8"))
    tokens = [None] * 248320
    for tok, i in v.items():
        if 0 <= i < len(tokens):
            tokens[i] = tok
    return Tokenizer(tokens, merges, types, pre="qwen35")


dump_path, oracle_path, key, pos = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
raw = pathlib.Path(dump_path).read_bytes()
n_vocab, n_rows = struct.unpack_from("<ii", raw, 0)
print("dump: n_vocab=%d n_rows=%d bytes=%d" % (n_vocab, n_rows, len(raw)))
expect = 8 + n_rows * n_vocab * 4
print("size check: %s (expected %d)" % ("OK" if len(raw) == expect else "MISMATCH", expect))
row = struct.unpack_from("<%df" % n_vocab, raw, 8 + (n_rows - 1) * n_vocab * 4)
order = sorted(range(n_vocab), key=lambda i: -row[i])
mx = max(row)
logz = mx + math.log(sum(math.exp(v - mx) for v in row))
tk = vocab()
print("== our engine, the row that decides token index %d (last stored row = final input position)" % pos)
print("   top-8 by our logits:")
for i in order[:8]:
    print("     %-8d %-14r logit=%8.4f  logprob=%8.4f" % (i, tk.decode([i]), row[i], row[i] - logz))
eng_top = order[:5]

o = json.loads(pathlib.Path(oracle_path).read_text(encoding="utf-8"))
cp = o["prompts"][key]["completion_probabilities"] if "prompts" in o else o["completion_probabilities"]
pr = cp[pos]
top = pr.get("top_logprobs") or pr.get("probs")
print("== the oracle at the same position (chosen=%s):" % pr.get("id"))
for t in top[:8]:
    print("     %-8d %-14r logprob=%8.4f" % (t["id"], tk.decode([t["id"]]), t.get("logprob", float("nan"))))
chosen = pr.get("id")
print("   oracle top-1 vs ours: oracle=%s ours=%s  -> %s" % (
    chosen, eng_top[0], "AGREE" if chosen == eng_top[0] else "DIFFER"))
gap_o = top[0].get("logprob", 0) - top[1].get("logprob", 0)
ours_scores = {i: row[i] for i in eng_top[:3]}
print("   oracle gap top1-top2 = %.4f nats (P ratio %.3f)" % (gap_o, math.exp(gap_o)))
if chosen in ours_scores and top[1]["id"] in ours_scores:
    print("   OUR engine's own scores at that pair: %s=%.4f  %s=%.4f  gap=%.4f nats" % (
        chosen, ours_scores[chosen], top[1]["id"], ours_scores[top[1]["id"]],
        abs(ours_scores[chosen] - ours_scores[top[1]["id"]])))
out = {"dump": dump_path, "n_rows": n_rows, "pos": pos,
       "engine_top": [(i, tk.decode([i]), row[i]) for i in order[:10]],
       "oracle_top": [(t["id"], tk.decode([t["id"]]), t.get("logprob")) for t in top],
       "engine_argmax": order[0], "oracle_choice": chosen}
pathlib.Path(dump_path).with_suffix(".top.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", pathlib.Path(dump_path).with_suffix(".top.json"))
