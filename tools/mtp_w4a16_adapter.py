"""tools/mtp_w4a16_adapter.py - the MTP block of Intel/Qwen3.8-Flash-Next-W4A16-AutoRound, in the layout
tools/mtp_pack.py reads.

    python tools/mtp_w4a16_adapter.py --extra <model_extra_tensors.safetensors> --out <dir>

tools/mtp_fetch.py fetches the 31 `mtp.*` tensors of the BF16 checkpoint over HTTP ranges and writes one raw
BF16 file per tensor plus mtp-manifest.json; tools/mtp_pack.py packs that directory and tools/mtp_rt.py turns the
GGUF into the runtime directory the engine's MtpDrafter loads.  The W4A16-AutoRound checkpoint carries the same
31-tensor MTP block, but inside ONE safetensors file (`model_extra_tensors.safetensors`, 1,565 tensors, all BF16)
and with the routed experts **unfused**:

    mtp.layers.0.mlp.experts.{0..511}.gate_proj.weight   [640, 2560] BF16
    mtp.layers.0.mlp.experts.{0..511}.up_proj.weight     [640, 2560] BF16
    mtp.layers.0.mlp.experts.{0..511}.down_proj.weight   [2560, 640] BF16

This tool is the adapter: it writes the 29 non-expert tensors byte-for-byte and FUSES the 1,536 expert tensors
into the two tensors mtp_pack.py expects,

    mtp.layers.0.mlp.experts.gate_up_proj                [512, 1280, 2560] BF16
    mtp.layers.0.mlp.experts.down_proj                   [512, 2560,  640] BF16

THE FUSION ORDER IS GATE FIRST, THEN UP.  It is not a guess about Qwen's exporter: it is the order the engine's
own relayout reads (`tools/mtp_rt.py:49-54`, "gate rows 0..639, up rows 640..1279") and the engine's expert blob
layout `moe_group_resident` consumes, so the pack, the relayout and the kernel agree by construction.  Nothing in
this file decodes a value - the bytes are copied, so the fusion cannot change a number.

The 1,565 tensors of the source are accounted for exactly: 29 non-expert + 1,536 unfused experts = 1,565, and any
name that is in neither set is a hard error, not a skip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

CHUNK = 16 << 20
FUSED = {
    "mtp.layers.0.mlp.experts.gate_up_proj": ("gate_proj.weight", "up_proj.weight"),
    "mtp.layers.0.mlp.experts.down_proj": ("down_proj.weight",),
}
EXPERT_FMT = "mtp.layers.0.mlp.experts.%d.%s"


def safetensors_header(path: pathlib.Path):
    with open(path, "rb") as f:
        import struct
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        return hdr, 8 + n


def dt_bytes(dtype: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}[dtype]


def known_mtp_names() -> dict:
    """tools/mtp_fetch.py's own 31-tensor set (name -> sha256 at its pinned revision), so 'the adapter wrote
    the right names AND the right bytes' is checkable rather than asserted."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import mtp_fetch  # noqa: E402  (constants only)
    return dict(mtp_fetch.SHA256)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", required=True, help="model_extra_tensors.safetensors")
    ap.add_argument("--out", required=True, help="the directory tools/mtp_pack.py --src wants")
    ap.add_argument("--experts", type=int, default=0, help="only the first N experts (a smoke run; the fused "
                                                           "shapes then differ and are reported)")
    a = ap.parse_args()
    src, out = pathlib.Path(a.extra), pathlib.Path(a.out)
    tdir = out / "tensors"
    tdir.mkdir(parents=True, exist_ok=True)
    hdr, base = safetensors_header(src)
    meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
    known = known_mtp_names()

    def span(name: str):
        t = meta[name]
        off = base + t["data_offsets"][0]
        n = t["data_offsets"][1] - t["data_offsets"][0]
        return t, off, n

    # ---- classify every tensor of the file, so nothing is dropped silently
    expert_names = set()
    for name in meta:
        if name in known:
            continue
        if ".mlp.experts." in name and name.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")):
            expert_names.add(name)
    unknown = [n for n in meta if n not in known and n not in expert_names]
    present = [n for n in known if n in meta]
    missing = [n for n in known if n not in meta]
    if unknown:
        sys.exit("names in neither mtp_fetch.py's set nor the expert pattern: %s" % unknown[:8])
    if sorted(missing) != sorted(FUSED):
        sys.exit("the non-expert set is not exactly mtp_fetch.py's 31 minus the two fused tensors: %s" % missing)
    print("%s: %d tensors = %d non-expert + %d expert (+2 fused written here)"
          % (src.name, len(meta), len(present), len(expert_names)), flush=True)

    manifest = []
    total = [0]

    def emit(name: str, shape, srcs, writer):
        """Write one tensor; `writer(f)` streams its bytes.  Returns the manifest row."""
        path = tdir / (name + ".bin")
        h = hashlib.sha256()
        with open(path, "wb") as f:
            n = writer(f, h)
        want = 1
        for d in shape:
            want *= int(d)
        if n != want * 2:
            sys.exit("%s: wrote %d B, expected %d" % (name, n, want * 2))
        total[0] += n
        row = {"name": name, "shape": list(shape), "dtype": "BF16", "file": str(path.relative_to(out)),
               "bytes": n, "sha256": h.hexdigest(), "source": src.name}
        if srcs:
            row["fused_from"] = list(srcs)
        if name in known:
            # the same 29 tensors that tools/mtp_fetch.py fetches from Qwen/Qwen3.8-Flash-Next at its pinned
            # revision: if these digests are the pinned ones, the W4A16 checkpoint's MTP block IS that block and
            # the MTP conversion is not a re-derivation.
            row["fetch_sha256"] = known[name]
            row["matches_fetch"] = (h.hexdigest() == known[name])
        manifest.append(row)
        print("  %-68s %-20s %10.3f MB  %s%s" % (name, "x".join(map(str, shape)), n / 1e6, h.hexdigest()[:12],
                                                 "" if name not in known else
                                                 ("  == mtp_fetch.py" if row["matches_fetch"] else
                                                  "  != mtp_fetch.py %s" % known[name][:12])), flush=True)
        return row

    # ---- the 29 non-expert tensors: raw copy
    for name in sorted(present):
        t, off, n = span(name)

        def copy(f, h, off=off, n=n):
            with open(src, "rb") as rf:
                rf.seek(off)
                left = n
                while left:
                    b = rf.read(min(CHUNK, left))
                    if not b:
                        sys.exit("short read in " + str(src))
                    f.write(b)
                    h.update(b)
                    left -= len(b)
            return n
        emit(name, t["shape"], None, copy)

    # ---- the two fused expert tensors: streamed one expert at a time
    nexp = a.experts or 512
    for fused, parts in FUSED.items():
        rows_per = None
        shape = None
        cols = None
        per_expert = []
        for e in range(nexp):
            srcs = []
            for p in parts:
                nm = EXPERT_FMT % (e, p)
                if nm not in meta:
                    sys.exit("missing expert tensor " + nm)
                srcs.append(nm)
            per_expert.append(srcs)
            t = meta[srcs[0]]
            if cols is None:
                cols = int(t["shape"][1])
            rows_per = sum(int(meta[s]["shape"][0]) for s in srcs)
            if sum(int(meta[s]["shape"][1]) != cols for s in srcs):
                sys.exit("the fused parts of %s disagree about the column count" % fused)
        shape = [nexp, rows_per, cols]
        # the interleaved file order is expert-major, and inside one expert the parts in `parts` order
        srcs_all = tuple(m for pair in per_expert for m in pair)

        def fuse(f, h, srcs_all=srcs_all):
            n = 0
            for nm in srcs_all:
                t, off, sz = span(nm)
                with open(src, "rb") as rf:
                    rf.seek(off)
                    left = sz
                    while left:
                        b = rf.read(min(CHUNK, left))
                        if not b:
                            sys.exit("short read in " + nm)
                        f.write(b)
                        h.update(b)
                        left -= len(b)
                        n += len(b)
            return n
        emit(fused, shape, ["mtp.layers.0.mlp.experts.{e}." + p for p in parts], fuse)

    (out / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    compared = [r for r in manifest if "matches_fetch" in r]
    matched = [r for r in compared if r["matches_fetch"]]
    report = {"source": str(src), "source_bytes": src.stat().st_size, "source_tensors": len(meta),
              "non_expert": len(present), "expert_unfused": len(expert_names), "experts": nexp,
              "fused_tensors": sorted(FUSED), "manifest_tensors": len(manifest), "bytes_written": total[0],
              "fetch_compared": len(compared), "fetch_matched": len(matched),
              "fetch_mismatched": [r["name"] for r in compared if not r["matches_fetch"]],
              "out": str(out)}
    (out / "mtp-adapter-report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\n%d manifest tensors, %.3f GB -> %s" % (len(manifest), total[0] / 1e9, out))
    print("bytes vs tools/mtp_fetch.py's pinned digests: %d/%d identical%s"
          % (len(matched), len(compared), "" if len(matched) == len(compared) else
             "  MISMATCHED: %s" % report["fetch_mismatched"]))
    print("census: %d = %d non-expert + %d unfused experts; %d written by this tool (2 fused)"
          % (len(meta), len(present), len(expert_names), len(manifest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
