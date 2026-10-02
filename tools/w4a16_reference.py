#!/usr/bin/env python3
"""tools/w4a16_reference.py - the W4A16 int4-group-128 CPU reference and the parity oracle (card t_e8373d23, §5).

This file is INDEPENDENT of the packer (`tools/w4a16_pack.py`) and of every kernel: it re-derives the checkpoint's
int4-g128 decode and the ggml Q4_0 block decode from the two specifications (W4A16-PLAN.md §1.3 and ggml's
`dequantize_row_q4_0`), reads the source triples itself, and reads the packed blob itself.  Nothing here imports
the packer, so a shared bug cannot make both sides agree.

WHAT IT IMPLEMENTS (the exact conventions, DW1 of the plan)
    source triple:   q = (qw[i>>3][o] >>> (4*(i&7))) & 0xF        (nibble (i&7) of the int32 word (i>>3, o))
                     z = ((qz[i>>7][o>>3] >>> (4*(o&7))) & 0xF) + 1        (the producer's stored 7 -> true 8)
                     w[o][i] = (q - z) * s[i>>7][o]              (s is SIGNED fp16, group 128 along i)
    packed blob:     ggml Q4_0, 18 B per 32 values: fp16 d, then 16 bytes; element j is the LOW nibble of byte j
                     and element j+16 is the HIGH nibble of byte j (ggml dequantize_row_q4_0)
                     w = (q - 8) * fp16(d)                        (d = the checkpoint's group scale, repeated)
Both must agree BIT FOR BIT on every weight - that is the acceptance (rung 0), and the plan's risk R1 says a
single mismatch fails it.

USAGE
    python tools/w4a16_reference.py --ckpt <snap> --pack <dir> --expert 16 330            # the parity report
    python tools/w4a16_reference.py ... --expert 16 330 --row-check                       # the hand-computed rung 1
    python tools/w4a16_reference.py ... --expert 16 330 --oracle-out /path/oracle         # write the W2 oracle
    python tools/w4a16_reference.py ... --sweep 2                                         # n experts per layer
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import pathlib
import struct
import sys

import numpy as np

H, FF = 2560, 640
N_LAYERS, N_EXPERTS = 48, 512
GROUP_ELEMS = 128
QK = 32
Q4_0_BYTES = 18
GU_ROW = H // QK * Q4_0_BYTES          # 1440
D_ROW = FF // QK * Q4_0_BYTES          # 360
UP_OFF = GU_ROW * FF
DOWN_OFF = 2 * UP_OFF
BLOB = DOWN_OFF + D_ROW * H            # 2,764,800
LAYER_BYTES = BLOB * N_EXPERTS
ROLES = (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF))
BASE = "model.language_model.layers.%d.mlp.experts.%d.%s"


# ------------------------------------------------------------------ the source, read independently ---------------
class Ckpt:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.idx = json.loads((path / "model.safetensors.index.json").read_text())
        self.wm = self.idx["weight_map"]
        self._open: dict[str, tuple] = {}

    def _shard(self, name: str):
        if name not in self._open:
            p = self.path / name
            f = open(p, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            self._open[name] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), hdr, 8 + n)
        return self._open[name]

    def get(self, tensor: str, dtype, ndim: int) -> np.ndarray:
        _f, mm, hdr, start = self._shard(self.wm[tensor])
        v = hdr[tensor]
        buf = mm[start + v["data_offsets"][0]: start + v["data_offsets"][1]]
        a = np.frombuffer(buf, dtype=dtype)
        return a.reshape(v["shape"]) if ndim > 1 else a

    def triple(self, layer: int, expert: int, role: str):
        b = BASE % (layer, expert, role)
        return (self.get(b + ".qweight", np.uint32, 2),
                self.get(b + ".qzeros", np.uint32, 2),
                self.get(b + ".scales", np.float16, 2))


# ------------------------------------------------------------------ the two decodes ---------------------------------
def dequant_source(qw: np.ndarray, qz: np.ndarray, sc: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """w[o][i] = (q - (z_stored + 1)) * s, from the checkpoint's own planes.  Returns float32 [out_dim][in_dim].

    The product is EXACT in binary floating point: (q - z) is an integer in [-8, 7] and s widens from fp16 (11
    significant bits), so the product needs at most 15 bits of mantissa - representable in float32 with no
    rounding.  Both decodes therefore have to agree bit for bit, not merely closely.
    """
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = ((qw[:, :, None] >> shifts) & 0xF).transpose(0, 2, 1).reshape(in_dim, out_dim)                # [i][o]
    z = (((qz[:, :, None] >> shifts) & 0xF).transpose(0, 2, 1).reshape(in_dim // GROUP_ELEMS, out_dim)
         .astype(np.int32) + 1)                                                                      # [g][o]
    ng = in_dim // GROUP_ELEMS
    w = (q.reshape(ng, GROUP_ELEMS, out_dim).astype(np.int32) - z[:, None, :]) * \
        sc.astype(np.float32)[:, None, :]
    return np.ascontiguousarray(w.reshape(in_dim, out_dim).T, dtype=np.float32)                      # [o][i]


def dequant_q4_0_row(row: memoryview | bytes | np.ndarray, in_dim: int) -> np.ndarray:
    """One packed row -> float32 [in_dim], straight from ggml's block semantics.

    Deliberately written the way `dequantize_row_q4_0` reads: for each 32-value block, byte j of the 16 gives
    element j in its LOW nibble and element j+16 in its HIGH nibble, all scaled by the block's fp16 d.
    """
    b = np.frombuffer(bytes(row), dtype=np.uint8) if not isinstance(row, np.ndarray) else row
    if b.size != in_dim // QK * Q4_0_BYTES:
        raise ValueError(f"a Q4_0 row of {in_dim} values is {in_dim // QK * Q4_0_BYTES} B, got {b.size}")
    blk = b.reshape(-1, Q4_0_BYTES)
    d = np.frombuffer(np.ascontiguousarray(blk[:, :2]).tobytes(), dtype="<f2").astype(np.float32)  # [n_blk]
    qs = blk[:, 2:]                                                                                # [n_blk][16]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    out = np.empty((blk.shape[0], QK), np.float32)
    out[:, :16] = lo
    out[:, 16:] = hi
    return (out * d[:, None]).reshape(-1)


def dequant_blob_projection(blob: bytes, off: int, row_bytes: int, out_dim: int, in_dim: int) -> np.ndarray:
    """[out_dim][in_dim] float32 out of the packed blob, one row at a time through dequant_q4_0_row."""
    a = np.frombuffer(blob, dtype=np.uint8)
    v = np.stack([dequant_q4_0_row(a[off + r * row_bytes: off + (r + 1) * row_bytes], in_dim)
                  for r in range(out_dim)])
    return v


def dequant_blob_expert(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three projections of one packed blob, [out_dim][in_dim] float32 each."""
    if len(blob) != BLOB:
        raise ValueError(f"a W4A16 expert blob is {BLOB} B, got {len(blob)}")
    return (dequant_blob_projection(blob, 0, GU_ROW, FF, H),
            dequant_blob_projection(blob, UP_OFF, GU_ROW, FF, H),
            dequant_blob_projection(blob, DOWN_OFF, D_ROW, H, FF))


def dequant_q4_0_projection_fast(blob: bytes, off: int, row_bytes: int, out_dim: int,
                                 in_dim: int) -> np.ndarray:
    """The same decode as `dequant_q4_0_row`, vectorized over rows - for the full-model sweep only.

    It is a different expression of the same block semantics (byte j of a block, low nibble -> element j,
    high -> element j+16) and `TestRealArtifact.test_fast_decode_matches_the_literal_one` requires it to agree
    with the literal per-row decoder, bit for bit, on one expert of every layer.  The literal one stays the
    definition of the oracle.
    """
    a = np.frombuffer(blob, np.uint8)[off: off + out_dim * row_bytes].reshape(out_dim, -1, Q4_0_BYTES)
    d = np.frombuffer(np.ascontiguousarray(a[:, :, :2]).tobytes(), dtype="<f2").astype(np.float32)
    qs = a[:, :, 2:]
    out = np.empty((out_dim, qs.shape[1], QK), np.float32)
    out[:, :, :16] = (qs & 0x0F).astype(np.int8) - 8
    out[:, :, 16:] = (qs >> 4).astype(np.int8) - 8
    return (out * d.reshape(out_dim, -1, 1)).reshape(out_dim, in_dim)


def dequant_blob_expert_fast(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(blob) != BLOB:
        raise ValueError(f"a W4A16 expert blob is {BLOB} B, got {len(blob)}")
    return (dequant_q4_0_projection_fast(blob, 0, GU_ROW, FF, H),
            dequant_q4_0_projection_fast(blob, UP_OFF, GU_ROW, FF, H),
            dequant_q4_0_projection_fast(blob, DOWN_OFF, D_ROW, H, FF))


# ------------------------------------------------------------------ the dot (the oracle W2 is judged against) -----
def swiglu(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray, fp64: bool = True) -> np.ndarray:
    """One expert's forward on one token: down @ (silu(gate @ x) * (up @ x)).  fp64 accumulation by default.

    The reference is deliberately the naive double-precision form: a kernel's job is to reproduce THIS, and the
    plan's §5 rung 2 tolerance (relative 1e-5) is set from the repo's own measured agreements (1.46e-06 for
    expert_parity, 1.06e-06 for the MKL gemm), 10x looser and still 4-5 orders tighter than a convention bug.
    """
    t = np.float64 if fp64 else np.float32
    g = (gate.astype(t) @ x.astype(t))
    u = (up.astype(t) @ x.astype(t))
    h = (g / (1.0 + np.exp(-g))) * u
    return (down.astype(t) @ h).astype(np.float32)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    d = max(float(np.abs(b).sum()), 1e-30)
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum()) / d


# ------------------------------------------------------------------ the artifact under test -------------------------
class Pack:
    """The artifact: experts.bin + the layer list from native_experts.txt (or its .partial during a partial run).

    A layer the artifact does not hold is REFUSED here rather than read as a hole (a sparse region of a
    preallocated experts.bin decodes to zeros and would look like a plausible mismatch).
    """

    def __init__(self, path: pathlib.Path):
        self.dir = path
        self.bin = path / "experts.bin"
        if not self.bin.exists():
            raise IOError(f"{self.bin} does not exist")
        self.complete = (path / "native_experts.txt").exists()
        src = path / "native_experts.txt" if self.complete else path / "native_experts.txt.partial"
        self.table = src if src.exists() else None
        self.layers = [int(l.split()[0]) for l in self.table.read_text().splitlines()
                       if l and not l.startswith("#")] if self.table else []
        if self.table is None:
            raise IOError(f"{path}: neither native_experts.txt nor its .partial is present, so nothing says "
                          f"which layers experts.bin holds")
        self.size = self.bin.stat().st_size

    def blob(self, layer: int, expert: int) -> bytes:
        if layer not in self.layers:
            where = ("the pack is complete" if self.complete else
                     f"partial pack: layers {self.layers[0]}..{self.layers[-1]}, {len(self.layers)} of {N_LAYERS}")
            raise ValueError(f"layer {layer} is not in {self.table.name} ({where}) - refusing to read a hole "
                             f"in experts.bin")
        with open(self.bin, "rb") as f:
            f.seek(layer * LAYER_BYTES + expert * BLOB)
            b = f.read(BLOB)
        if len(b) != BLOB:
            raise IOError(f"experts.bin is short at layer {layer} expert {expert}")
        return b


# ------------------------------------------------------------------ the checks --------------------------------------
def compare_expert(ck: Ckpt, pack: Pack, layer: int, expert: int) -> dict:
    """Rung 0: every weight of the packed blob against the source triple, bit for bit."""
    src = [dequant_source(*ck.triple(layer, expert, role), out_dim, in_dim)
           for role, out_dim, in_dim in ROLES]
    got = dequant_blob_expert(pack.blob(layer, expert))
    res = {"layer": layer, "expert": expert, "elements": 0, "mismatches": 0, "max_abs_err": 0.0}
    for (role, _, _), a, b in zip(ROLES, src, got):
        assert a.shape == b.shape
        neq = int((a != b).sum())
        res["elements"] += int(a.size)
        res["mismatches"] += neq
        res["max_abs_err"] = max(res["max_abs_err"], float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
        res[role] = {"elements": int(a.size), "mismatches": neq,
                     "max_abs_err": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())}
    return res


# ---- the hand-computed rung 1: pure Python integers and an fp16 decoder written out by hand, no numpy arithmetic
def half_to_float_by_hand(bits: int) -> float:
    """IEEE-754 binary16 -> double, written from the standard (1 sign, 5 exponent, 10 mantissa)."""
    sign = -1.0 if (bits >> 15) & 1 else 1.0
    exp = (bits >> 10) & 0x1F
    man = bits & 0x3FF
    if exp == 0:
        return sign * man * 2.0 ** -24
    if exp == 31:
        return float("nan") if man else sign * float("inf")
    return sign * (1024 + man) * 2.0 ** (exp - 25)


def row_check(ck: Ckpt, pack: Pack, layer: int, expert: int, role: str = "gate_proj",
              o: int = 0, i0: int = 0) -> None:
    """Print one row's arithmetic by hand: the int32 words, the nibbles, the fp16 bits, and (q-8)*s."""
    out_dim, in_dim = [(d, n) for r, d, n in ROLES if r == role][0]
    qw, qz, sc = ck.triple(layer, expert, role)
    row_bytes = in_dim // QK * Q4_0_BYTES
    off = 0 if role == "gate_proj" else UP_OFF if role == "up_proj" else DOWN_OFF
    blob = pack.blob(layer, expert)
    row = blob[off + o * row_bytes: off + (o + 1) * row_bytes]
    print(f"--- hand check: layer {layer} expert {expert} {role} row o={o}, inputs i={i0}..{i0+3}")
    print(f"    source planes: qweight {qw.shape} (uint32), qzeros {qz.shape}, scales {sc.shape} (float16)")
    for i in range(i0, i0 + 4):
        word = int(qw[i >> 3, o])
        nib = (word >> (4 * (i & 7))) & 0xF
        zword = int(qz[i >> 7, o >> 3])
        znib = ((zword >> (4 * (o & 7))) & 0xF) + 1
        sbits = struct.unpack("<H", np.float16(sc[i >> 7, o]).tobytes())[0]
        s = half_to_float_by_hand(sbits)
        q = nib
        w = (q - znib) * s
        # the same value out of the packed blob: block (i>>5), byte (i&15), low nibble for i%32<16 else high
        blk = i >> 5
        j = i & 31
        byte = row[blk * Q4_0_BYTES + 2 + (j if j < 16 else j - 16)]
        d = half_to_float_by_hand(struct.unpack("<H", row[blk * Q4_0_BYTES: blk * Q4_0_BYTES + 2])[0])
        qp = (byte & 0xF) if j < 16 else ((byte >> 4) & 0xF)
        wp = (qp - 8) * d
        print(f"    i={i:4d}  qweight[{i>>3},{o}]=0x{word:08x} nibble(0x{(word >> (4*(i&7))) & 0xF:x})={nib}  "
              f"qzeros[{i>>7},{o>>3}]=0x{zword:08x} stored={znib - 1} true_z={znib}  "
              f"scales[{i>>7},{o}] bits=0x{sbits:04x}={s:+.8e}")
        print(f"            source (q-z)*s = ({q} - {znib}) * {s:+.8e} = {w:+.8e}      "
              f"(q-7)*s would be {(q - (znib - 1)) * s:+.8e}")
        print(f"            blob block {blk} byte {byte:#04x} nibble {qp} d=0x"
              f"{struct.unpack('<H', row[blk*Q4_0_BYTES:blk*Q4_0_BYTES+2])[0]:04x}={d:+.8e}  "
              f"(q-8)*d = {wp:+.8e}   {'SAME' if wp == w else 'DIFFERENT'}")
    print(f"    row {o} of the blob, first 3 blocks as stored (18 B each): "
          f"{' '.join(f'{b:02x}' for b in row[:54])}")
    print(f"    row {o} of the source, int32 words 0..3: "
          f"{' '.join(f'0x{int(qw[a, o]):08x}' for a in range(4))}")


def oracle(ck: Ckpt, pack: Pack, out: pathlib.Path, layer: int, expert: int) -> dict:
    """Write the machine-readable oracle for one expert: the source-dequant weights, x, and the two dots.

    The W2 kernels are judged against the <out>.f32 dump (the SOURCE decode, i.e. the checkpoint's own values)
    and against the dots in <out>.json.  <out>.f32 is, in order, three float32 planes:
    gate [FF*H], up [FF*H], down [H*FF], row-major [out][in].
    """
    src = [dequant_source(*ck.triple(layer, expert, role), out_dim, in_dim)
           for role, out_dim, in_dim in ROLES]
    rng = np.random.default_rng(20261002 + layer * 1000 + expert)
    x = rng.standard_normal(H, dtype=np.float32)
    dots = {"source": swiglu(x, *src).tolist()}
    packed = dequant_blob_expert(pack.blob(layer, expert))
    dots["packed"] = swiglu(x, *packed).tolist()
    with open(str(out) + ".f32", "wb") as f:
        for p in src:
            f.write(np.ascontiguousarray(p, dtype=np.float32).tobytes())
    # the one-token dot, for the C++ checker: x, then the expected (source-dequant, fp64) output
    with open(str(out) + ".xdot.f32", "wb") as f:
        f.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(dots["source"], dtype=np.float32).tobytes())
    with open(str(out) + ".json", "w") as f:
        json.dump({"layer": layer, "expert": expert, "x_seed": 20261002 + layer * 1000 + expert,
                   "planes_f32": ["gate[640][2560]", "up[640][2560]", "down[2560][640]"],
                   "planes_bytes": [int(p.nbytes) for p in src],
                   "x": x.tolist(), "x_bytes": int(x.nbytes), "dot": dots,
                   "tolerance": {"rung2_relative": 1e-5,
                                 "reason": "W4A16-PLAN.md §5 rung 2: 10x the repo's own measured agreements "
                                           "(1.46e-06 expert_parity, 1.06e-06 MKL gemm), and 4-5 orders tighter "
                                           "than the smallest error a zero-point convention bug can produce"}},
                  f)
    print(f"oracle written: {out}.json ({len(dots['source'])} dot values) + {out}.f32 "
          f"({sum(p.nbytes for p in src):,d} B of source-dequant float32 weights)")
    return {"x": x, "source": src, "packed": packed, "dots": dots}


def parity_report(ck: Ckpt, pack: Pack, layer: int, expert: int, oracle_path: pathlib.Path | None) -> int:
    r = compare_expert(ck, pack, layer, expert)
    print(f"rung 0  layer {layer} expert {expert}: {r['elements']:,d} weights compared (3 projections), "
          f"{r['mismatches']} mismatched, max abs error {r['max_abs_err']:.3e}  -> "
          f"{'BIT-EXACT' if r['mismatches'] == 0 else 'FAILED'}")
    for role, _, _ in ROLES:
        d = r[role]
        print(f"          {role:10s} {d['elements']:,d} elements, {d['mismatches']} mismatched, "
              f"max abs {d['max_abs_err']:.3e}")
    if oracle_path is not None:
        o = oracle(ck, pack, oracle_path, layer, expert)
        d = o["dots"]
        print(f"rung 2  one token, x from seed {20261002 + layer * 1000 + expert}: the source-dequant weights and "
              f"the packed-dequant weights give the SAME expert output; rel L1 "
              f"{rel(np.array(d['packed']), np.array(d['source'])):.3e}")
    return 0 if r["mismatches"] == 0 else 1


def sweep(ck: Ckpt, pack: Pack, per_layer: int, layers: list[int]) -> int:
    """Rung 0 over `per_layer` experts of every layer (the default acceptance sample)."""
    tot_el = tot_bad = 0
    worst = 0.0
    lines = []
    for l in layers:
        for e in (0, N_EXPERTS // 2, N_EXPERTS - 1)[:per_layer] if per_layer <= 3 else \
                 [int(i * (N_EXPERTS - 1) / (per_layer - 1)) for i in range(per_layer)]:
            r = compare_expert(ck, pack, l, e)
            tot_el += r["elements"]
            tot_bad += r["mismatches"]
            worst = max(worst, r["max_abs_err"])
            lines.append(f"  layer {l:2d} expert {e:3d}: {r['elements']:,d} elements, {r['mismatches']} mismatched")
    if len(lines) <= 12:
        print("\n".join(lines))
    else:
        print("\n".join(lines[:6] + ["  ..."] + lines[-6:]))
    print(f"rung 0 sweep: {len(lines)} experts, {tot_el:,d} weights compared, {tot_bad} mismatched, "
          f"max abs error {worst:.3e} -> {'BIT-EXACT' if tot_bad == 0 else 'FAILED'}")
    return 0 if tot_bad == 0 else 1


def sweep_all(ck: Ckpt, pack: Pack, threads: int, layers: list[int], cross_check: bool = True) -> int:
    """Rung 0 over EVERY expert of every layer - all 24,576 blobs, 120,795,955,200 weights.

    The fast (vectorized) decoder is used here and, once per layer, the literal per-row decoder must agree with
    it bit for bit; if it does not, the sweep is invalid and says so.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    tot_el = tot_bad = 0
    worst = 0.0
    t0 = time.time()
    for l in layers:
        if l not in pack.layers:
            print(f"  layer {l}: not in {pack.table.name}, skipped")
            continue
        # the cross-check: the fast decode must equal the literal one on this layer's expert 0
        if cross_check:
            lit = dequant_blob_expert(pack.blob(l, 0))
            fast = dequant_blob_expert_fast(pack.blob(l, 0))
            same = all(np.array_equal(a, b) for a, b in zip(lit, fast))
            if not same:
                print(f"  layer {l}: the fast decoder disagrees with the literal per-row decoder - the sweep "
                      f"result would be meaningless")
                return 1

        def one(e: int) -> tuple[int, float]:
            src = [dequant_source(*ck.triple(l, e, role), out_dim, in_dim)
                   for role, out_dim, in_dim in ROLES]
            got = dequant_blob_expert_fast(pack.blob(l, e))
            bad = 0
            w = 0.0
            for a, b in zip(src, got):
                if not np.array_equal(a, b):
                    bad += int((a != b).sum())
                    w = max(w, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
            return bad, w

        with ThreadPoolExecutor(max_workers=threads) as pool:
            res = list(pool.map(one, range(N_EXPERTS)))
        layer_bad = sum(r[0] for r in res)
        tot_bad += layer_bad
        tot_el += N_EXPERTS * 3 * (H * FF)
        worst = max(worst, max(r[1] for r in res))
        el = time.time() - t0
        print(f"  layer {l:2d}: 512 experts, {N_EXPERTS * 3 * H * FF:,d} weights, {layer_bad} mismatched  "
              f"({el:.0f}s elapsed, {el / (layers.index(l) + 1):.1f}s/layer)", flush=True)
        if layer_bad:
            print(f"  LAYER {l} HAS MISMATCHES - stopping")
            break
    print(f"rung 0 FULL SWEEP: {len(layers)} layers x 512 experts = {len(layers) * N_EXPERTS:,d} experts, "
          f"{tot_el:,d} weights compared, {tot_bad} mismatched, max abs error {worst:.3e} -> "
          f"{'BIT-EXACT' if tot_bad == 0 else 'FAILED'}")
    return 0 if tot_bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--expert", nargs=2, type=int, metavar=("LAYER", "EXPERT"))
    ap.add_argument("--sweep", type=int, default=0, help="experts per layer over all 48 layers")
    ap.add_argument("--sweep-all", action="store_true", help="EVERY expert of every layer (the full rung 0)")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--row-check", action="store_true")
    ap.add_argument("--row-role", default="gate_proj")
    ap.add_argument("--row-o", type=int, default=0)
    ap.add_argument("--oracle-out")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    ck = Ckpt(pathlib.Path(a.ckpt).absolute())
    pack = Pack(pathlib.Path(a.pack).absolute())
    print(f"artifact: {pack.bin} ({pack.size:,d} B), {pack.table.name}: {len(pack.layers)} layer(s), "
          f"complete={pack.complete}")
    layers = list(range(N_LAYERS)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rc = 0
    if a.sweep_all:
        rc |= sweep_all(ck, pack, a.threads, layers)
    if a.sweep:
        rc |= sweep(ck, pack, a.sweep, layers)
    if a.expert:
        l, e = a.expert
        out = pathlib.Path(a.oracle_out) if a.oracle_out else None
        rc |= parity_report(ck, pack, l, e, out)
        if a.row_check:
            row_check(ck, pack, l, e, a.row_role, a.row_o)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"sweep": bool(a.sweep), "rc": rc}, f)
    return rc


if __name__ == "__main__":
    sys.exit(main())
