"""tools/w4a16_serve_pack.py - the pack the W4A16 model is SERVED from (M5).

The W-track's pack (`t_e8373d23` + `t_1c729145`) is an inventory: `index.txt` carries all 1,079 dense
rows canonically, `experts.bin` the 67.9 GB of Q4_0 expert blobs, `native_experts.txt` the v3 expert
table.  The engine cannot serve it as it stands, and the reason is structural rather than a bug:

  * a pack with a `native_experts.txt` is a NATIVE pack (`generate.cpp:1699` = `expert_layout().native`),
    and a native pack runs its decode as verify windows, whose GDN/QSA/shared projections are read
    through `native_of()` - i.e. they must come from a GGUF (`verify.cpp:93-100`);
  * `NativeDense::load` refuses to attach a GGUF copy to a pack row that is not `quantized()`
    (`code_bits != 0`, `native_dense.cpp:150`), and every projection row W1b wrote is BF16.

`tools/iq_pack.py` already writes the form such a row has to take - a SHAPE-ONLY row that the GGUF
serves:  `name file 0 0 0 0 0 ne0 ne1 8 0 32 0 0 0 0 0 0 0` (see its `served` branch).  This tool writes
that index (everything else byte for byte), links the big binaries and builds the pack's `tokenizer/`
directory from the checkpoint, so the served pack is one directory:

    <out>/index.txt            rewritten: the GGUF-served rows carry no bytes
    <out>/dense.bin            -> the source pack's (the rows that are still canonical read here)
    <out>/experts.bin          -> the source pack's (67,947,724,800 B of Q4_0 experts)
    <out>/native_experts.txt   the expert table, copied
    <out>/tokenizer/           vocab.json / merges.txt / token_type.json / tokenizer.json / chat_template.jinja

    python tools/w4a16_serve_pack.py --pack <W-track pack> --out <serve pack> \
        --native-gguf <native-dense.gguf> --model <snapshot> [--copy] [--report FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_reader import GGUFFile  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from dense_w4a16_pack import safetensors_header  # noqa: E402

HEADER_COLS = ("name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems "
               "codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind").split()
# the shape-only row tools/iq_pack.py writes for a tensor the GGUF serves (its `served` branch)
SERVED_TAIL = ["8", "0", "32"] + ["0"] * 7
N_VOCAB = 248320


def served_row(fields: list[str]) -> str:
    return " ".join([fields[0], fields[1], "0", "0", "0", "0", "0", fields[7], fields[8]] + SERVED_TAIL)


def build_tokenizer(model: pathlib.Path, out: pathlib.Path, report: dict) -> None:
    """The pack's tokenizer/ directory, from the checkpoint's HF tokenizer.json.

    `tools/strata_tokenizer.py --gguf` is the tool of record, but the W4A16 checkpoint is safetensors
    with no GGUF anywhere in the pipeline, so the same three files are written here from the HF files.
    The one thing that is NOT a straight copy: the checkpoint's `model.vocab` covers ids 0..248,043 and
    its 33 added tokens cover 248,044..248,076 with a 12-id gap (248,058-248,069), while the model's own
    vocab is 248,320 (`output.weight`'s row count).  A shorter list would give `Tokenizer` duplicate
    `None` entries and it refuses to construct, so the hole and the tail are filled with reserved-token
    placeholders at token type 1 (normal: never matched as a literal) - the ids the model can generate
    and this build cannot name are counted in the report rather than hidden.
    """
    tj = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = dict(tj["model"]["vocab"])
    added = tj.get("added_tokens", [])
    for t in added:
        vocab.setdefault(t["content"], t["id"])
    types = [1] * N_VOCAB
    special_ids = {}
    for t in added:
        types[t["id"]] = 3 if t.get("special") else 1
        if t.get("special"):
            special_ids[t["content"]] = t["id"]
    reserved = []
    have = set(vocab.values())
    for i in range(N_VOCAB):
        if i not in have:
            name = "<|reserved_special_token_%d|>" % i
            vocab[name] = i
            reserved.append(i)
    if len(vocab) != N_VOCAB:
        sys.exit("tokenizer: %d ids covered, expected %d" % (len(vocab), N_VOCAB))
    tok_dir = out / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    merges = tj["model"]["merges"]
    # HF ships merges as ["a b", ...] in older files and as [["a", "b"], ...] in newer ones; the pack's
    # merges.txt is llama.cpp's "a b" per line either way (tools/strata_tokenizer.py:283).
    lines = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]
    for i, m in enumerate(lines):
        if len(m.split(" ")) != 2:
            sys.exit("tokenizer: merge %d is not a pair: %r" % (i, m))
    (tok_dir / "merges.txt").write_text("\n".join(lines), encoding="utf-8")
    (tok_dir / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    cfg = {"model": "gpt2", "pre": "qwen35", "vocab_size": N_VOCAB, "n_merges": len(merges),
           "special_ids": special_ids, "add_bos_token": False, "pre_pattern": ST.QWEN35_PATTERN,
           "pre_pattern_source": "tools/strata_tokenizer.py (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35)"}
    (tok_dir / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = model / "chat_template.jinja"
    if tpl.exists():
        shutil.copyfile(tpl, tok_dir / "chat_template.jinja")
    else:
        sys.exit("tokenizer: %s is missing; the server needs the model's chat template" % tpl)
    report["tokenizer"] = {"dir": str(tok_dir), "vocab": len(vocab), "merges": len(merges),
                           "specials": len(special_ids), "reserved_filled": len(reserved),
                           "reserved_ids": reserved}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="the W-track pack directory")
    ap.add_argument("--out", required=True, help="the serve pack directory to write")
    ap.add_argument("--native-gguf", required=True, help="the GGUF written by tools/w4a16_native_gguf.py")
    ap.add_argument("--model", default=None, help="the checkpoint snapshot (writes tokenizer/)")
    ap.add_argument("--copy", action="store_true", help="copy dense.bin/experts.bin instead of symlinking")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    src, out = pathlib.Path(a.pack), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = GGUFFile(pathlib.Path(a.native_gguf))
    gguf_names = {t.name for t in g.tensors}
    report = {"pack": str(src), "out": str(out), "native_gguf": str(a.native_gguf),
              "gguf_tensors": len(gguf_names)}

    lines = (src / "index.txt").read_text(encoding="utf-8").splitlines()
    kept, served, missing = [], [], []
    for line in lines:
        if not line or line.startswith("#"):
            kept.append(line)
            continue
        f = line.split()
        if len(f) != 19:
            sys.exit("index.txt: a row with %d fields: %s" % (len(f), line[:80]))
        if f[0] in gguf_names:
            served.append(f[0])
            kept.append(served_row(f))
        else:
            kept.append(line)
    if not served:
        sys.exit("no index row is served by %s: the pack's shapes and the GGUF do not meet" % a.native_gguf)
    # every GGUF tensor the engine will look for must have a row, or NativeDense has nothing to attach to
    rows = {l.split()[0] for l in lines if l and not l.startswith("#")}
    missing = sorted(gguf_names - rows)
    header = [l for l in lines if l.startswith("#")]
    (out / "index.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    report["rows_served"] = len(served)
    report["served"] = sorted(served)[:8] + (["... %d more" % (len(served) - 8)] if len(served) > 8 else [])
    report["gguf_tensors_without_a_row"] = missing
    report["header"] = header

    for name in ("dense.bin", "experts.bin"):
        dst = out / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if a.copy:
            shutil.copyfile(src / name, dst)
        else:
            os.symlink((src / name).resolve(), dst)
    shutil.copyfile(src / "native_experts.txt", out / "native_experts.txt")

    if a.model:
        build_tokenizer(pathlib.Path(a.model), out, report)

    print("serve pack %s" % out)
    for l in header:
        print("  header: %s" % l)
    print("  index rows: %d, of which GGUF-served: %d" % (len(kept) - len(header), len(served)))
    print("  served names: %s" % ", ".join(sorted(served)[:6]) + (" ..." if len(served) > 6 else ""))
    if missing:
        print("  WARNING: %d GGUF tensor(s) have no index row: %s" % (len(missing), ", ".join(missing[:5])))
    for name in ("index.txt", "dense.bin", "experts.bin", "native_experts.txt"):
        p = out / name
        print("  %-20s %s %d B" % (name, "->" if p.is_symlink() else "  ", p.stat().st_size))
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("  report: %s" % a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
