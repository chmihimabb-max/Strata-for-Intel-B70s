"""tools/ple_bf16_pack.py - the PLE n-gram table from a BF16 checkpoint, as the GGUF the engine reads.

    python tools/ple_bf16_pack.py --model <checkpoint dir> --format iq4_nl --out <ple-iq4nl.gguf>
    python tools/ple_bf16_pack.py --model <checkpoint dir> --format f8_e4m3 --out <ple-fp8.gguf>

`tools/ple_fp8_pack.py` keeps Qwen's own F8_E4M3 shards by copying their bytes; a checkpoint that ships the
table as **BF16** (Intel/Qwen3.8-Flash-Next-W4A16-AutoRound: 128 shards `...ngram_embedding.shard_{k}.weight`,
[2500012, 160], 102.400 GB, 320,001,536 rows - which is exactly `PLE_TABLE_ROWS` in include/strata/kernels/
ngram.hpp) cannot be copied and must be QUANTISED.  This tool does that, to either form the engine accepts
(`src/kernels/ngram.cpp:196-211`):

    iq4_nl   90 bytes/row: 5 blocks of 32, each {fp16 d ; 16 bytes of split-half 4-bit codes} - the form every
             other Flash-Next pack uses.  28.80 GB.
    f8_e4m3  160 bytes/row, one F8_E4M3 byte per value times ONE global `strata.ple.scale` (F32), type I8, the
             encoding tools/ple_fp8_pack.py writes.  51.20 GB.

The row flattening is not a choice: `ngram.hpp:44-49` fixes it (ne0 = 160 is the fast axis, a row is 160
contiguous values = 5 blocks of 32 at 18 B) and `ngram.hpp:101-113` fixes the IQ4_NL nibble order (SPLIT
HALVES: `qs[j]` carries elements j and j+16, not 2j and 2j+1).

    --probe N   read N rows per shard, print the value statistics that decide the FP8 question: are the BF16
                values exactly E4M3 codes x a power of two (i.e. is the deleted FP8 source recoverable losslessly)?

The error report at the end is measured, not asserted: every emitted block is decoded again (the engine's own
rule) and compared against the BF16 source, per row.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np
import torch

GGML_TYPE_IQ4_NL = 20
GGML_TYPE_I8 = 24
GGML_TYPE_STRING, GGML_TYPE_F32 = 8, 6
ALIGN = 32
CHUNK_ROWS = 32768                 # rows per pass: 32768 x 160 = 5.2M values (IQ4_NL: 21 MB in, 2.9 MB out)

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                         dtype=np.int8)
# MSE scale search, as ggml's quantize_row_iq4_nl_impl does: the largest code is 113, so d >= amax/113 keeps the
# block's maximum inside the codebook, and larger d buys resolution for the rest of the block at the cost of the
# outlier.  1.0 is the no-clipping reference; beyond 1.12 the negative side starts to clip.
SCALE_GRID = (1.0, 1.05, 1.12, 1.25, 1.4, 1.6, 1.9, 2.4)


def safetensors_header(path: pathlib.Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, val: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGML_TYPE_STRING) + gguf_string(val)


def kv_f32(key: str, val: float) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGML_TYPE_F32) + struct.pack("<f", val)


def e4m3_table() -> np.ndarray:
    """The engine's own F8_E4M3 table (`ngram.cpp:117-130`): e4m3fn, 0x7F/0xFF are NaN and read as 0."""
    out = np.zeros(256, dtype=np.float32)
    for x in range(256):
        s, e, m = x >> 7, (x >> 3) & 15, x & 7
        f = 0.0 if (e == 15 and m == 7) else (np.ldexp(m / 8.0, -6) if e == 0 else np.ldexp(1.0 + m / 8.0, e - 7))
        if s:
            f = -f
        out[x] = f
    return out


FP8_TABLE = e4m3_table()


# ------------------------------------------------------------------------------------------------- quantisers
def iq4_nl_blocks(x: torch.Tensor) -> bytes:
    """(n, 160) float32 -> n x 90 bytes of IQ4_NL, split-half nibble order."""
    nb, d32 = x.shape[0] * (x.shape[1] // 32), 32
    b = x.reshape(-1, 32)
    amax = b.abs().amax(dim=1, keepdim=True)
    kv = torch.tensor(KVALUES_IQ4NL.astype(np.float32))
    thresh = (kv[:-1] + kv[1:]) / 2.0                     # 15 midpoints, ascending -> bucketize gives the code
    best_err = torch.full((b.shape[0], 1), float("inf"))
    best_d = torch.zeros_like(amax)
    for f in SCALE_GRID:
        d = amax / 113.0 * f
        inv = torch.where(d > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d))
        code = torch.bucketize(b * inv, thresh)           # 0..15
        err = ((kv[code] * d - b) ** 2).sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)
    d16 = best_d.to(torch.float16)
    d = d16.to(torch.float32)                             # quantise against the STORED (fp16) scale
    inv = torch.where(d > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d))
    code = torch.bucketize(b * inv, thresh).to(torch.uint8)
    codes = code.reshape(-1, 5, 32)                       # (rows, 5 blocks, 32 values)
    qs = (codes[:, :, :16] | (codes[:, :, 16:] << 4)).to(torch.uint8)      # byte j: value j (low), j+16 (high)
    out = torch.empty((b.shape[0], 18), dtype=torch.uint8)
    out[:, :2] = d16.view(torch.uint8).reshape(-1, 2)     # fp16 d first, as block_iq4_nl has it
    out[:, 2:] = qs.reshape(-1, 16)
    return out.numpy().reshape(-1).tobytes()


def iq4_nl_decode(raw: np.ndarray) -> np.ndarray:
    """The engine's decode (`ngram.cpp:101-114`) - an INDEPENDENT path from the encoder above."""
    b = raw.reshape(-1, 18)
    # `b[:, :2].copy().view(np.float16)` keeps shape (n, 1); `d[:, None]` then broadcast (n,16)*(n,1,1) into an
    # (n,n,16) allocation.  Flatten the fp16 scale to (n,) first - the W1b smoke run caught this as a 1.56 TiB
    # allocation attempt on the first real shard.
    d = np.ascontiguousarray(b[:, :2]).view(np.float16).reshape(-1).astype(np.float32)
    qs = b[:, 2:]
    lo = KVALUES_IQ4NL[(qs & 0x0F).astype(np.int64)].astype(np.float32)
    hi = KVALUES_IQ4NL[(qs >> 4).astype(np.int64)].astype(np.float32)
    return np.concatenate([(lo * d[:, None]).reshape(-1, 16), (hi * d[:, None]).reshape(-1, 16)],
                          axis=1).reshape(-1, 32)


def fp8_encode(x: torch.Tensor, scale: float) -> bytes:
    q = (x / scale).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).numpy().tobytes()


def fp8_decode(raw: np.ndarray, scale: float) -> np.ndarray:
    return FP8_TABLE[raw] * np.float32(scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the checkpoint directory (model.safetensors.index.json)")
    ap.add_argument("--format", choices=("iq4_nl", "f8_e4m3"), default="iq4_nl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", type=int, default=0, help="rows per shard: print statistics only, write nothing")
    ap.add_argument("--shards", type=int, default=0, help="limit to the first N shards (a smoke run)")
    ap.add_argument("--report", default=None, help="JSON error report (default: <out>.report.json)")
    ap.add_argument("--sample-rows", type=int, default=1024, help="rows per shard used for the error report")
    a = ap.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    wmap = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    shards = {}
    for name in wmap:
        if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
            shards[int(name.rsplit(".shard_", 1)[1].split(".")[0])] = name
    if not shards or sorted(shards) != list(range(len(shards))):
        sys.exit("the n-gram shards are not numbered 0..N without gaps in " + str(model))
    order = [shards[k] for k in sorted(shards)]
    if a.shards:
        order = order[:a.shards]

    parts = []
    for name in order:
        hdr, base = safetensors_header(model / wmap[name])
        t = hdr[name]
        if t["dtype"] != "BF16" or len(t["shape"]) != 2:
            sys.exit("%s is %s %s, not a 2-D BF16 tensor" % (name, t["dtype"], t["shape"]))
        parts.append((model / wmap[name], base + t["data_offsets"][0], t["shape"][0], t["shape"][1]))
    dim = parts[0][3]
    if any(p[3] != dim for p in parts):
        sys.exit("the shards disagree about the row width")
    rows = sum(p[2] for p in parts)
    print("%d shards, %d rows x %d values = %.3f GB of BF16, format %s"
          % (len(parts), rows, dim, rows * dim * 2 / 1e9, a.format), flush=True)

    # ---- the FP8 question, asked of the real bytes before anything is written
    if a.probe:
        return probe(parts, dim, a.probe, a.format)

    # ---- f8_e4m3 needs ONE scale for the whole table: pass 1 measures the global maximum.
    scale = 1.0
    amax = 0.0
    if a.format == "f8_e4m3":
        t0 = time.time()
        for path, off, nrows, dim in parts:
            mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(nrows, dim))
            for i in range(0, nrows, CHUNK_ROWS):
                x = torch.from_numpy(np.ascontiguousarray(mm[i:i + CHUNK_ROWS])).view(torch.bfloat16).float()
                amax = max(amax, float(x.abs().max()))
            mm._mmap.close()
        scale = amax / 448.0
        print("global max |value| = %.6g -> strata.ple.scale = %.9g  (pass 1: %.1f s)"
              % (amax, scale, time.time() - t0), flush=True)

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, " + ("F8_E4M3 from BF16" if a.format == "f8_e4m3"
                                                             else "IQ4_NL from BF16")),
           kv_string("strata.ple.format", "f8_e4m3" if a.format == "f8_e4m3" else "iq4_nl"),
           kv_string("strata.ple.source", model.name),
           kv_string("strata.ple.source_dtype", "BF16")]
    if a.format == "f8_e4m3":
        kvs.append(kv_f32("strata.ple.scale", scale))
        rb, gtype = dim, GGML_TYPE_I8
    else:
        rb, gtype = dim // 32 * 18, GGML_TYPE_IQ4_NL
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, rows)
    head += struct.pack("<I", gtype) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    written = 0
    err_rows, err_sum, err_worst, err_worst_row = 0, 0.0, 0.0, -1
    per_shard = []
    t0 = time.time()
    with open(tmp, "wb") as w:
        w.write(head)
        for si, (path, off, nrows, dim) in enumerate(parts):
            mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(nrows, dim))
            # a deterministic sample of rows for the error report: evenly spaced over the shard
            step = max(1, nrows // a.sample_rows)
            for i in range(0, nrows, CHUNK_ROWS):
                blk = mm[i:i + CHUNK_ROWS]
                x = torch.from_numpy(np.ascontiguousarray(blk)).view(torch.bfloat16).float()
                if a.format == "f8_e4m3":
                    raw = fp8_encode(x, scale)
                    got = fp8_decode(np.frombuffer(raw, dtype=np.uint8), scale).astype(np.float32)
                else:
                    raw = iq4_nl_blocks(x)
                    got = iq4_nl_decode(np.frombuffer(raw, dtype=np.uint8))
                w.write(raw)
                written += len(raw)
                # per-row relative RMS error against the BF16 source, on the sampled rows only
                sel = np.arange(0, x.shape[0], step)
                if sel.size:
                    ref = x.numpy()[sel].astype(np.float32)
                    rec = got.reshape(-1, dim)[sel]
                    rms = np.sqrt((rec - ref) ** 2).mean(axis=1)
                    denom = np.sqrt((ref ** 2).mean(axis=1)) + 1e-30
                    rel = rms / denom
                    worst = int(np.argmax(rel))
                    err_rows += rel.size
                    err_sum += float(rms.sum())
                    if rel[worst] > err_worst:
                        err_worst, err_worst_row = float(rel[worst]), i + int(sel[worst])
            mm._mmap.close()
            per_shard.append({"shard": si, "rows": nrows, "bytes": written})
            if si % 16 == 15 or si == len(parts) - 1:
                el = time.time() - t0
                print("  %3d/%d shards  %.2f GB written  %.1f s  (%.2f GB/s of source read)"
                      % (si + 1, len(parts), written / 1e9, el,
                         si + 1 and (sum(p[2] for p in parts[:si + 1]) * dim * 2 / 1e9) / max(el, 1e-9)),
                      flush=True)
    want = len(head) + rows * rb
    if tmp.stat().st_size != want:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, want))
    tmp.replace(out)
    rep = {"format": a.format, "rows": rows, "row_dim": dim, "row_bytes": rb, "header_bytes": len(head),
           "table_bytes": rows * rb, "file_bytes": out.stat().st_size, "scale": scale,
           "global_amax": amax, "source_bytes": rows * dim * 2,
           "error": {"rows_sampled": err_rows, "mean_row_rms": err_sum / max(err_rows, 1),
                     "worst_row_rel_rms": err_worst, "worst_row": err_worst_row},
           "seconds": time.time() - t0}
    rpath = pathlib.Path(a.report) if a.report else out.with_suffix(out.suffix + ".report.json")
    rpath.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("wrote %s: %d rows x %d B = %.3f GB (header %d B), %.1f s" % (out, rows, rb, rows * rb / 1e9, len(head),
                                                                        rep["seconds"]))
    print("  sampled %d rows: mean RMS %.6g, worst row relative RMS %.6g (row %d)"
          % (err_rows, rep["error"]["mean_row_rms"], err_worst, err_worst_row))
    print("  report -> %s" % rpath)
    return 0


def probe(parts, dim, nrows, fmt) -> int:
    """Are the BF16 values exactly (E4M3 code) x (power of two)?  If they are, the FP8 source is recoverable."""
    mant_ok = 0
    mant_tot = 0
    patterns = {}
    lo, hi = float("inf"), 0.0
    for path, off, n, _ in parts:
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(n, dim))
        sel = np.linspace(0, n - 1, nrows).astype(np.int64)
        u = np.asarray(mm[sel]).astype(np.uint16).view(np.uint16)
        x = (u.astype(np.uint32) << 16).view(np.float32)
        mant = (u & 0x007F).astype(np.uint8)             # bf16 mantissa = 7 bits
        keep = (x != 0)
        ok = (mant[keep] & 0x0F) == 0                    # low 4 mantissa bits zero -> <= 3 significant bits
        mant_ok += int(ok.sum())
        mant_tot += int(keep.sum())
        # the range was previously only computed for the FP8 arm, and `lo` started at 0.0 - so the min was
        # always 0.  Both fixed here: this is a property of the source, not of the target format.
        v = np.abs(x[keep].astype(np.float64))
        if v.size:
            nz = v[v != 0]
            if nz.size:
                lo = min(lo, float(nz.min()))
            hi = max(hi, float(v.max()))
        for m in np.unique(mant[keep])[:8]:
            patterns[int(m)] = patterns.get(int(m), 0) + 1
        mm._mmap.close()
    print("probe: %d values sampled, %d nonzero (format argument %s, unused for the statistic)" % (mant_tot, mant_tot, fmt))
    print("  of the nonzero values, %.6f%% have their low 4 BF16 mantissa bits zero"
          % (100.0 * mant_ok / max(mant_tot, 1)))
    print("  => %s" % ("consistent with a value that IS an E4M3 code x a power of two (<=3 significant mantissa "
                       "bits)" if mant_ok == mant_tot else
                       "NOT exactly E4M3 x a power of two: the deleted FP8 source is lossy to recover"))
    print("  |value| range among the sampled rows: %.6g .. %.6g" % (lo, hi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
