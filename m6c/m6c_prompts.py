#!/usr/bin/env python3
"""M6c: the 256K prompt set -- depth curve + needle, tokenised with the pack's OWN tokenizer.

Corpus: this repository's own text (docs/*.md, src/, include/, serve/, tools/) concatenated in a fixed
order, tokenised once.  That is a real long technical/code document, which is what upstream's code-agent
prompt is, and it is >5 MB so it covers the full 259,943-token prompt with no repetition.

Prompt shape (protocol of record, same as M6/M6b): a NONCE prefix of 8 fixed pseudo-random ids so the
prefill cannot be served from any prefix/conversation cache, then the corpus cut inside continuous text.
Each prompt in the depth curve is a PREFIX of the next one (nested), so the rows are comparable.

The curve rows are a real code-agent turn - a <|im_start|>system / user / assistant chat whose user turn
is the document and whose task is a code review of it, which is the shape upstream's table used ("a real
code-review task over long source files").  A plain continuation of the document is NOT usable: at 64K it
wrote the end-of-turn token after 62 tokens (DONE 62 64512 ... stop), which fails the card's ">=256
usage-counted generated tokens" per length.

Lengths: 32,256 / 64,512 / 129,024 / 259,943 prompt tokens (the last is upstream's 262K row length, so
262,144 - 259,943 = 2,201 cells stay free for the generated tokens: 256 of them here).

Needle rows (same total length as the matching curve row, the answer at 50% depth):
    needle at depth, then a question at the very end; greedy decode has to retrieve the code.

usage: /usr/bin/python3 m6c/m6c_prompts.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path("/home/michael/strata-xpu/strata")
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST  # noqa: E402

PACK = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-iq3s/pack")
OUT = pathlib.Path("/home/michael/strata-xpu/m6c/prompts")

# 8 fixed ids, plain-text range, distinct from the corpus start (same nonce as M6/M6b)
NONCE = [31337, 1001, 4243, 9871, 271, 12007, 3131, 42]

# ctx -> prompt tokens.  262,144 is the model's window; 259,943 is upstream's 262K prompt length.
LENGTHS = {32768: 32256, 65536: 64512, 131072: 129024, 262144: 259943}
# ctx -> prompt tokens for the needle rows (room for the question + 256 generated tokens; the engine
# refuses n + max_new + 8 > max-context, so the 4K control's prompt is 3,832 tokens: 3832+256+8 = 4096)
NEEDLE_LENGTHS = {4096: 3832, 262144: 259943}
NEEDLE_DEPTH = 0.50
# The 256K needle failed at 50% depth (the model answered "No such access code appears in the document"),
# so one more arm locates the failure: the same 259,943-token prompt with the needle near the START
# (5%).  If retrieval works there, the 256K path carries the content and the 50% failure is a position
# effect; if it fails there too, the 256K path itself is the suspect.
EXTRA_NEEDLE = [(262144, 259943, 0.05, "-d5")]

NEEDLE = ("\n\nThe fallback station log for the Diversity Antenna Project array records one access code, "
          "and that code is ZK-4471-QX. It is unique to that array's fallback station.\n\n")
# The 4K control with the question as a plain continuation of the document produced EOS on the first
# token (DONE 1 3968 17384 ... stop), so the needle rows are a real chat turn - the shape a code agent
# sends - with the question in the user turn and the answer taken from the assistant turn.
REVIEW = ("\n\nPlease review the document above, which is the source tree and the documentation of one C++/CUDA "
          "engine. List the most important correctness or performance risks you find in it, in order of "
          "severity, and for each one quote the evidence you saw and give a concrete fix. Take as long as "
          "you need.<|im_end|>\n<|im_start|>assistant\n")
HEADER = ("<|im_start|>system\nYou are a senior C++ and GPU engineer reviewing a large repository."
          "<|im_end|>\n<|im_start|>user\n")
NONCE_OPEN, NONCE_CLOSE = "\n[document revision marker: ", "]\n"
QUESTION = ("\n\nQuestion: In the document above, a project log records the access code for the Diversity "
            "Antenna Project array. What is that access code? Answer with the code only.<|im_end|>\n"
            "<|im_start|>assistant\n")

CORPUS_FILES = [
    "docs/DETAILS.md", "docs/HOW_IT_WORKS.md", "docs/QUANT.md", "docs/AMD_HIP.md", "docs/MULTI_GPU.md",
    "docs/INSTALL.md", "docs/AI_SETUP.md", "docs/MODELS.md", "docs/TROUBLESHOOTING.md",
    "docs/COMMUNITY_BENCHMARKS.md", "docs/AMD_HIP_PERFORMANCE.md", "docs/UNSLOTH_Q4.md", "docs/ORCA.md",
    "docs/ORCA_Q4_K_S.md", "docs/SECOND_GPU.md", "docs/MCP_SERVER.md",
]
CORPUS_GLOBS = [
    "src/**/*.cpp", "src/**/*.cu", "src/**/*.hpp",
    "include/**/*.hpp",
    "serve/**/*.py",
    "tools/*.py",
]


def load_tok() -> ST.Tokenizer:
    t = PACK / "tokenizer"
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    merges = [m for m in (t / "merges.txt").read_text(encoding="utf-8").split("\n") if m.strip()]
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def corpus_files() -> list[pathlib.Path]:
    seen: list[pathlib.Path] = []
    for rel in CORPUS_FILES:
        p = REPO / rel
        if p.is_file():
            seen.append(p)
    for g in CORPUS_GLOBS:
        for p in sorted(REPO.glob(g)):
            if p.is_file() and p not in seen:
                seen.append(p)
    return seen


def write_prompt(path: pathlib.Path, ids: list[int]) -> dict:
    path.write_text(",".join(str(i) for i in ids), encoding="utf-8")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": str(path), "prompt_tokens": len(ids), "bytes": path.stat().st_size, "sha256": h}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tok()
    files = corpus_files()
    parts = []
    for p in files:
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    text = "\n\n".join(parts)
    ids = tok.encode(text, parse_special=False)
    print(f"corpus: {len(files)} files, {len(text)} chars -> {len(ids)} tokens")
    man = {"nonce": NONCE, "corpus_files": [str(p.relative_to(REPO)) for p in files],
           "corpus_chars": len(text), "corpus_tokens": len(ids), "prompts": []}
    if len(ids) < max(LENGTHS.values()) + len(NONCE):
        print(f"FATAL: corpus has only {len(ids)} tokens, need {max(LENGTHS.values())}")
        return 2

    q_ids = tok.encode(QUESTION, parse_special=True)
    r_ids = tok.encode(REVIEW, parse_special=True)
    n_ids = tok.encode(NEEDLE, parse_special=False)
    head_ids = tok.encode(HEADER + NONCE_OPEN, parse_special=True) + NONCE + tok.encode(NONCE_CLOSE, parse_special=False)
    print(f"needle: {len(n_ids)} ids, review task: {len(r_ids)} ids, question+assistant turn: {len(q_ids)} ids, "
          f"header: {len(head_ids)} ids")

    for ctx, n in LENGTHS.items():
        body = n - len(head_ids) - len(r_ids)
        sel = head_ids + ids[:body] + r_ids
        assert len(sel) == n, (len(sel), n)
        row = {"kind": "curve", "ctx": ctx, "max_new": 256, "kv_total": n + 256,
               "task_text": REVIEW, "head_text": tok.decode(sel[:60]), "tail_text": tok.decode(sel[-90:])}
        row.update(write_prompt(OUT / f"prompt-ctx{ctx}.txt", sel))
        man["prompts"].append(row)
        print(f"  prompt-ctx{ctx}.txt {n} tokens (body {body})  tail={row['tail_text'][-60:]!r}")

    for (ctx, n, depth, suffix) in [(c, nn, NEEDLE_DEPTH, "") for c, nn in NEEDLE_LENGTHS.items()] + EXTRA_NEEDLE:
        body = n - len(head_ids) - len(q_ids)
        if body <= len(n_ids):
            print(f"FATAL: ctx {ctx} too short"); return 2
        d = int(body * depth)
        sel = head_ids + ids[:d] + n_ids + ids[d + len(n_ids):body] + q_ids
        assert len(sel) == n, (len(sel), n)
        row = {"kind": "needle", "ctx": ctx, "max_new": 256, "depth_frac": depth,
               "needle_ids": n_ids, "needle_at": len(head_ids) + d, "question_ids": q_ids,
               "question_text": QUESTION, "needle_text": NEEDLE, "answer": "ZK-4471-QX",
               "head_text": tok.decode(sel[:60]), "tail_text": tok.decode(sel[-90:])}
        row.update(write_prompt(OUT / f"prompt-needle-ctx{ctx}{suffix}.txt", sel))
        man["prompts"].append(row)
        print(f"  prompt-needle-ctx{ctx}{suffix}.txt {n} tokens  needle at {row['needle_at']} "
              f"({100.0 * row['needle_at'] / n:.1f}% depth)  {len(q_ids)} question ids")

    (OUT / "prompt-manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"manifest: {OUT / 'prompt-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
