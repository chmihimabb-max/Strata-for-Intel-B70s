"""tools/w4a16_native_gguf.py - the native GGUF the W4A16 pack's *dense* half needs to be served.

WHY THIS EXISTS.  A pack with a `native_experts.txt` runs on the engine's native path: the decode is a
verify window, and `verify.cpp`'s GDN/QSA/shared-expert projections come from `NativeDense`, whose
`native_data` is only ever set from a **GGUF** (`native_dense.cpp:182`).  The W4A16 pack (W1b) carries
those tensors canonically in `dense.bin` (BF16), which is not what `native_of()` accepts
(`verify.cpp:93-100` -> "is not served natively (run with --native)").  So the pack needs a companion
GGUF holding the tensors `native_dense.cpp`'s `eligible()` names, and a pack index that marks those rows
as GGUF-served - exactly the shape `tools/iq_pack.py` writes for a Unsloth/ISTA pack
(`name file 0 0 0 0 0 ne0 ne1 8 0 32 0 0 0 0 0 0 0`).  `tools/w4a16_serve_pack.py` writes that index.

The names, their GGUF types and why:

    token_embd.weight      BF16 (type 30)   `NativeEmbed`: `embed_type_supported()` = is_iq() or 30;
                                            BF16 is the checkpoint's own type, so the copy is EXACT.
    output.weight          Q8_0 (type 8)    `NativeHead`, any `native_mmvq` type; Q8_0's error is
                                            <= d/2 per 32-value block (1/127 of the block's amax).
    blk.N.<proj>.weight    Q8_0 (type 8)    the ten GDN/QSA/shared-expert matrices `native_dense.cpp`
                                            accepts; Q8_0 is in `native_mmvq_supported` on both CUDA and
                                            the SYCL port (native_mmvq.cpp:1660-1665).

GGUF tensor order is `[ne0, ne1]` with ne0 the CONTIGUOUS axis, and `native_dense.cpp` checks
`shape[0] == ref.ne0 && shape[1] == ref.ne1` against the pack's own index row - so the shapes here are
read from the pack index rather than re-derived, and the payload is the checkpoint's row-major
`(ne1, ne0)` matrix written out row by row.  A mismatch is caught at load, but only for the tensors the
engine happens to need; this tool checks it up front against the checkpoint's own header.

    python tools/w4a16_native_gguf.py --model <snapshot dir> --index <pack>/index.txt \
        --out <dir>/native-dense.gguf [--layers N] [--embed-type bf16] [--report FILE]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dense_w4a16_pack as DP  # noqa: E402  (LAYER_MAP/MODEL_MAP/safetensors_header/bf16_f32)
from gguf_writer import GGUFWriter  # noqa: E402

H, NE, N_LAYER = 2560, 512, 48
VOCAB = 248320
# The ten matrices native_dense.cpp's eligible() accepts, by pack-name suffix.
PROJ_SUFFIXES = (".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
                 ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
                 ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight")
HF_LAYER_PREFIX = "model.language_model.layers."
HF_MODEL_PREFIX = "model.language_model."
ROLE = {"attn_qkv.weight": "gdn in_proj_qkv", "attn_gate.weight": "gdn in_proj_z",
        "ssm_out.weight": "gdn out_proj",
        "attn_q.weight": "qsa q_proj", "attn_k.weight": "qsa k_proj", "attn_v.weight": "qsa v_proj",
        "attn_output.weight": "qsa o_proj", "ffn_gate_shexp.weight": "shared gate",
        "ffn_up_shexp.weight": "shared up", "ffn_down_shexp.weight": "shared down"}


def index_rows(pack_dir: pathlib.Path):
    """(name -> {ne0, ne1}) from the pack's index.txt (19 columns, v3)."""
    cols = ("name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems "
            "codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind").split()
    out = {}
    for line in (pack_dir / "index.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 19:
            sys.exit("index.txt: a row with %d fields: %s" % (len(f), line[:80]))
        r = dict(zip(cols, f))
        out[r["name"]] = {"ne0": int(r["ne0"]), "ne1": int(r["ne1"]), "code_bits": int(r["code_bits"]),
                          "kind": int(r["kind"]), "dst_bytes": int(r["dst_bytes"])}
    return out


def pack_name_to_hf(pack: str) -> str:
    """`blk.7.attn_qkv.weight` -> `model.language_model.layers.7.linear_attn.in_proj_qkv.weight`."""
    if not pack.startswith("blk."):
        for hf, p in DP.MODEL_MAP.items():
            if p == pack:
                return hf
        sys.exit("no HF name for %s" % pack)
    parts = pack.split(".")
    layer, rest = int(parts[1]), ".".join(parts[2:])
    for hf, p in DP.LAYER_MAP.items():
        if p == rest:
            return "%s%d.%s" % (HF_LAYER_PREFIX, layer, hf)
    sys.exit("no HF name for %s (suffix %s)" % (pack, rest))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the checkpoint snapshot directory")
    ap.add_argument("--index", required=True, help="the pack directory holding index.txt")
    ap.add_argument("--out", required=True, help="the GGUF to write")
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (a smoke run)")
    ap.add_argument("--no-embed", action="store_true", help="skip token_embd + output.weight")
    ap.add_argument("--no-head", action="store_true", help="skip output.weight only")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    model = pathlib.Path(a.model)
    pack = pathlib.Path(a.index)
    rows = index_rows(pack)
    wmap = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    hdrs: dict[str, tuple] = {}

    def hdr_of(shard: str):
        if shard not in hdrs:
            hdrs[shard] = DP.safetensors_header(model / shard)
        return hdrs[shard]

    def raw_of(hf_name: str) -> bytes:
        shard = wmap[hf_name]
        h, base = hdr_of(shard)
        t = h[hf_name]
        if t["dtype"] != "BF16":
            sys.exit("%s is %s, not BF16: this tool's exactness claim does not hold" % (hf_name, t["dtype"]))
        with open(model / shard, "rb") as f:
            f.seek(base + t["data_offsets"][0])
            raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
        if len(raw) != t["data_offsets"][1] - t["data_offsets"][0]:
            sys.exit("%s: short read from %s" % (hf_name, shard))
        return raw

    # ---- the plan: what goes in, with the shape the pack's index demands
    jobs = []          # (pack name, kind 'embed'|'head'|'proj', hf name, ne0, ne1, type name)
    if not a.no_embed:
        for pname in ("token_embd.weight",):
            r = rows.get(pname)
            if r is None:
                sys.exit("%s is not in the pack index" % pname)
            jobs.append((pname, "embed", pack_name_to_hf(pname), r["ne0"], r["ne1"], "BF16"))
    if not a.no_head:
        r = rows.get("output.weight")
        if r is None:
            sys.exit("output.weight is not in the pack index")
        jobs.append(("output.weight", "head", pack_name_to_hf("output.weight"), r["ne0"], r["ne1"], "Q8_0"))
    n_layers = a.layers or N_LAYER
    for l in range(n_layers):
        for suffix in PROJ_SUFFIXES:
            pname = "blk.%d%s" % (l, suffix)
            r = rows.get(pname)
            if r is None:
                continue                     # a pruned/partial pack: say so in the report, do not invent it
            jobs.append((pname, "proj", pack_name_to_hf(pname), r["ne0"], r["ne1"], "Q8_0"))

    w = GGUFWriter()
    w.add("general.architecture", "qwen4exp")
    w.add("general.name", "Qwen3.8-Flash-Next W4A16 (Intel AutoRound) native dense half")
    w.add("qwen4exp.block_count", N_LAYER, "u32")
    w.add("qwen4exp.embedding_length", H, "u32")
    w.add("qwen4exp.expert_count", NE, "u32")           # presence contract (pruned models ship fewer)
    w.add("qwen4exp.expert_used_count", 10, "u32")
    w.add("qwen4exp.attention.head_count", 24, "u32")
    w.add("qwen4exp.attention.head_count_kv", 2, "u32")
    w.add("strata.m5.note", "native projections for the W4A16 pack; Q8_0 = ggml reference rule")
    w.add("strata.m5.index", str(pack / "index.txt"))

    report = {"model": str(model), "index": str(pack), "out": a.out, "tensors": [], "unmapped_rows": []}
    t0 = time.time()
    total = 0
    for pname, kind, hf, ne0, ne1, ty in jobs:
        raw = raw_of(hf)
        want = ne0 * ne1 * 2
        if len(raw) != want:
            sys.exit("%s: the checkpoint holds %d B (%d values) but the pack row says [%d, %d] = %d B"
                     % (pname, len(raw), len(raw) // 2, ne0, ne1, want))
        if ty == "BF16":
            # exact: the source IS bf16, so the payload is copied through and only the widening is undone
            arr = None
            w.add_bytes(pname, raw, [ne0, ne1], "BF16")
        else:
            arr = DP.bf16_f32(raw).reshape(ne1, ne0)
            w.add_q8_0(pname, arr, [ne0, ne1])
        total += ne0 * ne1
        suffix = pname.split(".", 2)[2] if pname.startswith("blk.") else pname
        report["tensors"].append({"pack": pname, "hf": hf, "role": ROLE.get(suffix, kind),
                                  "type": ty, "shape": [ne0, ne1], "values": ne0 * ne1})
        del arr, raw
    if not report["tensors"]:
        sys.exit("nothing to write: the index has none of the tensors this tool serves")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.write(out)
    report["bytes_written"] = out.stat().st_size
    report["values"] = total
    report["seconds"] = round(time.time() - t0, 1)
    print("wrote %s: %d tensors, %d values, %d B (%.3f GB) in %.1f s"
          % (out, len(report["tensors"]), total, report["bytes_written"],
             report["bytes_written"] / 1e9, report["seconds"]))
    by_type = {}
    for t in report["tensors"]:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1
    print("  by type: %s" % ", ".join("%s x%d" % (k, v) for k, v in sorted(by_type.items())))
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("  report: %s" % a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
