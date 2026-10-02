"""tools/dense_w4a16_pack.py - the dense/embed/head half of the W4A16 pack, from the safetensors checkpoint.

    python tools/dense_w4a16_pack.py --model <checkpoint dir> --out <pack dir> [--layers N] [--report J]

WHY THIS EXISTS.  `tools/strata_pack.py` and `tools/iq_pack.py` both build a pack from a GGUF: the first from
`canonical_xcheck`'s dequantising mappings, the second from a GGUF's own tensor directory.  Intel's
W4A16-AutoRound checkpoint ships **safetensors** (17 shards, HF names, no GGUF anywhere), and nothing in the tree
reads safetensors - `grep -rln safetensors src include` is empty.  So this tool is the missing adapter, and it is
an adapter and not a second packer:

  * the ENGINE FORM of every tensor comes from `tools/iq_pack.py`'s own `FORM`/`form_of` table,
  * the BYTES come from `tools/iq_pack.py`'s own `convert()` (kind 4 verbatim BF16, kind 2 exact BF16->F32
    widening, kind 3/5 F16 as FORM requires) - called, not copied,
  * the INDEX comes from `tools/iq_pack.py`'s own `write_index()`, in the 19-column v3 format
    `src/core/weights.cpp` parses.
  `tools/pack_index.py` is NOT used: its input is `manifest.json`, which only a GGUF-derived pack has (it needs
  `source_type`/`codes`/`scales` plane offsets that a safetensors header does not carry), and its convention
  promotes every BF16 tensor to F32 in the pack.  `iq_pack.py`'s index writer carries the same information for a
  float source directly, which is what the iq packs already ship.

NAME MAP.  The checkpoint addresses tensors in HF's layout (`model.language_model.layers.7.self_attn.q_proj`), the
engine by GGUF-style name (`blk.7.attn_q.weight`, `token_embd.weight`).  The manifest's shape is `[ne0, ne1]` with
ne0 CONTIGUOUS, which is HF's `[out, in]` with the axes swapped - so the pack shape is HF's shape REVERSED, and no
transpose is needed anywhere.  Two mappings are not a rename:

    self_attn.indexer.index_qk_proj.weight [640, 2560]  -> indexer.q_proj.weight [2560, 512]
                                                        +  indexer.k_proj.weight [2560, 128]
        (a ROW SPLIT: 512 query rows then 128 key rows, `layout.cpp:80-81`'s two counts)
    linear_attn.conv1d.weight [10240, 1, 4]             -> ssm_conv1d.weight [4, 10240]
    ple.conv1d.weight [10240, 1, 4]                     -> ple_conv1d.weight [4, 10240]
        (a RESHAPE that is the identity in memory: HF's [C, 1, K] stores c*K + k, and the engine's ne0=K fast
         axis stores the same c*K + k - `gdn.hpp:64` states the manifest's ne = [4, 10240].)

TWO TENSORS ARE QUANTISED HERE, because the engine requires an S-form and the checkpoint is all BF16:
`token_embd.weight` and `output.weight`.  `layer.cpp:1015`/`verify.cpp:433` refuse anything whose `code_bits` is
not 2/4/8, and `layer.cpp:1087` refuses a non-quantized `output.weight` outright.  They are written as **S8,
group 32, code_bias -128** - which is exactly the canonical form `tools/canonical_xcheck.py:571` maps a Q8_0
source to ("Q8_0 -> S8: codes + an fp16 scale per 32 elements, codebook the affine `code - 128`"), so the
convention is the repo's own rather than a reading of the kernel, and ggml's rounding (`d = amax/127`, codes
clipped to 1..255) is used because that is what that mapping decodes.  Measured on the checkpoint
(`scripts/w1b_embed_s8_measure.py`, 8,278 of 248,320 rows): row relative RMS **0.00596 (token_embd) / 0.00602
(output)**, against **0.09620 / 0.09713** for the S4-g32-bias-8 alternative, for +0.318 GB each
(715,161,600 B vs 397,312,000 B).
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
import iq_pack as IP  # noqa: E402  (FORM/form_of/convert/write_index - the pack's own contract)

ALIGN = IP.ALIGN
S_GROUP = 32          # the group of the two S-form tensors; 32 is Q8_0's own (canonical_xcheck.py:571)

# ---- HF name -> pack name, for the layer tensors.  `{L}` is the layer index.
LAYER_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "ple.key_proj.weight": "ple_key.weight",
    "ple.value_proj.weight": "ple_value.weight",
    "ple.norm_conv.weight": "ple_norm_conv.weight",
    "ple.norm_key.weight": "ple_norm_key.weight",
    "ple.norm_query.weight": "ple_norm_query.weight",
}
# 3-D in the checkpoint, [K, C] for the engine (a pure reshape; see the docstring)
LAYER_RESHAPE = {
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "ple.conv1d.weight": "ple_conv1d.weight",
}
MODEL_MAP = {
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": "output_hc_down.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": "output_hc_up.weight",
    "model.language_model.hyper_connection_mixer.hc_norm.weight": "output_hc_norm.weight",
}
S_QUANTISED = {  # pack name -> (code_bits, group, bias); the two the engine requires in an S-form
    "token_embd.weight": (8, S_GROUP, -128),
    "output.weight": (8, S_GROUP, -128),
}
SPLIT_INDEXER = "model.language_model.layers.{L}.self_attn.indexer.index_qk_proj.weight"
PLAN_PLE = "PLE: the 128 n-gram table shards = W4A16-PLAN DELIVERABLE 1 (this card's ple-*.gguf)"
PLAN_MTP = ("MTP block (1,536 unfused experts + 29 dense) = W4A16-PLAN DELIVERABLE 2 of this card: "
            "tools/mtp_w4a16_adapter.py -> tools/mtp_pack.py -> tools/mtp_rt.py")
PLAN_PLE_CONST = ("PLE hash constants (layer_multipliers / ngram_heads_offsets / "
                  "ngram_heads_vocab_sizes): already transcribed in "
                  "include/strata/kernels/ngram.hpp:36-66 (ple_artifact_consts)")
EXPERT_MARK = ".mlp.experts."
PLE_EMBED_MARK = ".ple.ple_embedding."
PLE_TABLE_MARK = ".ple.ple_embedding.ngram_embedding.shard_"


def safetensors_header(path: pathlib.Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def bf16_f32(raw: bytes, shape=None) -> np.ndarray:
    """The BF16 bytes as float32 values (the exact widening every conversion here starts from)."""
    u = np.frombuffer(raw, dtype="<u2")
    return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape if shape else -1)


def s4_quantise(values: np.ndarray, ne0: int, group: int, bias: int, bits: int = 4):
    """([rows, ne0] float32) -> (codes bytes LSB-first, f32 scales, scales, codes2d).

    ggml's own two conventions, which are also the canonical form's (`tools/canonical_xcheck.py:566,571`
    map Q4_0 -> S4 bias -8 and Q8_0 -> S8 bias -128):

        Q4_0: d = amax / 8,   code = clip(rint(v/d) + 8,   0,  15)     (2 codes per byte, LSB-first)
        Q8_0: d = amax / 127, code = clip(rint(v/d) + 128, 0, 255)     (one code per byte)

    so the S-form this writes is bit-compatible with what the repo's own mapping produces for a Q8_0/Q4_0
    source: unsigned codes, an affine `code + bias`, one f32 scale per `group` values, no offset plane.
    """
    v = values.reshape(-1, ne0).astype(np.float32)
    rows = v.shape[0]
    g = v.reshape(rows, ne0 // group, group)
    amax = np.abs(g).max(axis=2, keepdims=True)
    divisor = float(-bias) if bits == 4 else float(-bias - 1)
    d = amax / divisor
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = np.rint(g * inv).astype(np.int32) + (-bias)
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8)          # (rows, groups, group)
    scales = d.reshape(rows, ne0 // group).astype(np.float32)
    flat = q.reshape(rows, ne0)
    if bits == 4:
        if ne0 % 2:
            raise ValueError("an S4 row must be an even number of elements")
        codes = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)   # strata_pack.pack_codes(bits=4)
    else:
        codes = flat.astype(np.uint8)
    return codes.tobytes(), scales.tobytes(), scales, flat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the checkpoint directory (model.safetensors.index.json)")
    ap.add_argument("--out", required=True, help="the pack directory (dense.bin + index.txt written here)")
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (a smoke run)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--sample-rows", type=int, default=4096, help="rows per S-form tensor used for the error report")
    a = ap.parse_args()
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wmap = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    hdrs: dict[str, tuple] = {}

    def hdr_of(shard: str):
        if shard not in hdrs:
            hdrs[shard] = safetensors_header(model / shard)
        return hdrs[shard]

    # ---- plan: every tensor of the 17 shards, each either packed here or named with a reason
    plan, skipped, splits = [], {}, []
    for name in sorted(wmap):
        shard = wmap[name]
        if name.startswith("mtp."):
            skipped[PLAN_MTP] = skipped.get(PLAN_MTP, 0) + 1
            continue
        if EXPERT_MARK in name:
            skipped["expert triple (W1's experts pack, 3 tensors per expert)"] = \
                skipped.get("expert triple (W1's experts pack, 3 tensors per expert)", 0) + 1
            continue
        if PLE_TABLE_MARK in name and name.endswith(".weight"):
            skipped[PLAN_PLE] = skipped.get(PLAN_PLE, 0) + 1
            continue
        if PLE_EMBED_MARK in name:
            skipped[PLAN_PLE_CONST] = skipped.get(PLAN_PLE_CONST, 0) + 1
            continue
        if name.startswith("model.visual."):
            skipped["vision tower: no engine consumer (the engine is text-only)"] = \
                skipped.get("vision tower: no engine consumer (the engine is text-only)", 0) + 1
            continue
        h, base = hdr_of(shard)
        if name not in h:
            sys.exit("%s: not in %s's header" % (name, shard))
        t = h[name]
        # the indexer's single packed projection is the only tensor that becomes two
        if name.startswith("model.language_model.layers.") and \
                name.endswith(".self_attn.indexer.index_qk_proj.weight"):
            splits.append((name, shard, t, base))
            continue
        pack = None
        rest = None
        if name.startswith("model.language_model.layers."):
            parts = name.split(".")
            layer = int(parts[3])
            rest = ".".join(parts[4:])
            if rest in LAYER_MAP:
                pack = "blk.%d.%s" % (layer, LAYER_MAP[rest])
            elif rest in LAYER_RESHAPE:
                pack = "blk.%d.%s" % (layer, LAYER_RESHAPE[rest])
        if pack is None:
            pack = MODEL_MAP.get(name)
        if pack is None:
            sys.exit("no mapping for %s (layer-local suffix %r): either map it or name it as skipped" % (name, rest))
        if a.layers and name.startswith("model.language_model.layers.") and int(name.split(".")[3]) >= a.layers:
            skipped["beyond --layers %d (smoke run)" % a.layers] = \
                skipped.get("beyond --layers %d (smoke run)" % a.layers, 0) + 1
            continue
        plan.append((name, pack, shard, t, base))

    if a.layers:
        splits = [s for s in splits if int(s[0].split(".")[3]) < a.layers]

    print("checkpoint: %d tensors in %d shards" % (len(wmap), len(set(wmap.values()))))
    print("plan: %d tensors to pack, %d split into two, %d skipped"
          % (len(plan), len(splits), sum(skipped.values())), flush=True)

    rows, report, at = [], [], 0
    t0 = time.time()
    with open(out / "dense.bin", "wb") as fo:

        def emit(pack_name, raw, hf_name, hf_shape, note=""):
            """One tensor into dense.bin in the engine's form, via iq_pack's own convert()."""
            nonlocal at
            if len(hf_shape) == 3:
                # the two conv1d weights: HF is [C, 1, K] and the engine's manifest is ne = [K, C] with ne0
                # CONTIGUOUS - which is the SAME byte order (HF stores c*K + k).  `gdn.hpp:64` states it.
                if int(hf_shape[1]) != 1:
                    sys.exit("%s: expected [C, 1, K], got %s" % (hf_name, hf_shape))
                ne0, ne1 = int(hf_shape[2]), int(hf_shape[0])
            else:
                ne0 = int(hf_shape[-1])                   # ne0 CONTIGUOUS = HF's last axis
                ne1 = int(hf_shape[-2]) if len(hf_shape) > 1 else 0
            if ne0 * (ne1 if ne1 else 1) * 2 != len(raw):
                sys.exit("%s: %d elements do not fill %d bytes" % (hf_name, ne0 * (ne1 if ne1 else 1), len(raw)))
            if pack_name in S_QUANTISED:
                bits, group, bias = S_QUANTISED[pack_name]
                v = bf16_f32(raw, (ne1, ne0) if ne1 else (ne0,))
                codes, scales, sc, codes2d = s4_quantise(v, ne0, group, bias, bits)
                # the S4 decode the engine applies: (code + bias) * scale
                dec = ((codes2d.astype(np.float32) + bias) * np.repeat(sc, group, axis=1)).reshape(-1)
                ref = v.reshape(-1)
                rel = float(np.sqrt(((dec - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-30))
                step = max(1, (ne1 or 1) // a.sample_rows)
                sel = np.arange(0, ne1 or 1, step) * ne0
                ds = np.concatenate([np.arange(s, s + ne0) for s in sel])
                rr = float(np.sqrt(((dec[ds] - ref[ds]) ** 2).mean())
                           / (np.sqrt((ref[ds] ** 2).mean()) + 1e-30))
                cb, sb = len(codes), len(scales)
                blob = codes + scales
                rows.append([pack_name, "0", "0", str(at), str(len(blob)), "0",
                             str(cb + sb), str(ne0), str(ne1), str(bits), str(bias), str(group)] + ["0"] * 7)
                report.append({"pack": pack_name, "hf": hf_name, "form": "S%d-g%d-b%d" % (bits, group, bias),
                               "ne0": ne0, "ne1": ne1, "bytes": len(blob),
                               "relative_rms_all_rows": rel, "relative_rms_sampled": rr, "note": note})
            else:
                res = IP.convert(pack_name, "BF16", np.frombuffer(raw, dtype=np.uint8), True)
                if isinstance(res, str):
                    sys.exit("iq_pack.convert refused %s: %s" % (pack_name, res))
                kind, data, dst_bytes, rec = res
                blob = data
                rows.append([pack_name, "0", kind, str(at), str(len(data)), "0", str(dst_bytes),
                             str(ne0), str(ne1), "0", "0", "1"] + ["0"] * 7)
                form = IP.form_of(pack_name) or "as stored (BF16)"
                report.append({"pack": pack_name, "hf": hf_name, "form": form, "kind": kind,
                               "ne0": ne0, "ne1": ne1, "bytes": len(data), "dst_bytes": dst_bytes,
                               "exact": (rec or {}).get("exact", True),
                               "max_abs_err": (rec or {}).get("max_abs_err", 0.0),
                               "method": (rec or {}).get("method", "verbatim BF16 (kind 4)"), "note": note})
            fo.write(blob)
            at += len(blob)
            pad = (-at) % ALIGN
            fo.write(b"\0" * pad)
            at += pad

        for i, (name, pack, shard, t, base) in enumerate(plan):
            off0 = base + t["data_offsets"][0]
            nbytes = t["data_offsets"][1] - t["data_offsets"][0]
            with open(model / shard, "rb") as f:
                f.seek(off0)
                raw = f.read(nbytes)
            if len(raw) != nbytes:
                sys.exit("short read: " + name)
            emit(pack, raw, name, t["shape"])
            if (i + 1) % 100 == 0 or i + 1 == len(plan):
                print("  %4d/%d  %.2f GiB written  %.1f s" % (i + 1, len(plan), at / 2**30, time.time() - t0),
                      flush=True)

        for name, shard, t, base in splits:
            ne0 = int(t["shape"][-1])
            nbytes = t["data_offsets"][1] - t["data_offsets"][0]
            with open(model / shard, "rb") as f:
                f.seek(base + t["data_offsets"][0])
                raw = f.read(nbytes)
            layer = int(name.split(".")[3])
            q_rows, k_rows = 512, 128
            if tuple(t["shape"]) != (q_rows + k_rows, ne0):
                sys.exit("%s is %s, not [640, %d]" % (name, t["shape"], ne0))
            # emit() takes the tensor's BF16 BYTES: slice the source span, not a float32 view of it
            cut = q_rows * ne0 * 2
            if len(raw) != (q_rows + k_rows) * ne0 * 2:
                sys.exit("%s: %d B is not %d elements of BF16" % (name, len(raw), (q_rows + k_rows) * ne0))
            emit("blk.%d.indexer.q_proj.weight" % layer, raw[:cut], name, [q_rows, ne0],
                 "row split 0:512 of [640, 2560]")
            emit("blk.%d.indexer.k_proj.weight" % layer, raw[cut:], name, [k_rows, ne0],
                 "row split 512:640 of [640, 2560]")

    IP.write_index(out, [list(r) for r in rows], pathlib.Path(a.model), 0, 0, publish=True)
    rep = {"model": str(model), "out": str(out), "tensors_packed": len(rows), "dense_bytes": at,
           "dense_gib": at / 2**30, "seconds": time.time() - t0, "skipped": skipped,
           "split": [s[0] for s in splits], "tensors_in_checkpoint": len(wmap), "tensors": report}
    rp = pathlib.Path(a.report) if a.report else out / "dense-w4a16-report.json"
    rp.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("dense.bin: %.3f GiB, index.txt %d rows, %.1f s" % (at / 2**30, len(rows), time.time() - t0))
    print("report -> %s" % rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
