#!/usr/bin/env python3
"""tools/w4a16_pack.py - the W4A16 int4-group-128 experts packer (W4A16-PLAN.md §2, decision DW2).

Intel/Qwen3.8-Flash-Next-W4A16-AutoRound keeps its 512 routed experts of every one of the 48 layers as
AutoGPTQ int4 group-128 symmetric triples - `.qweight` (I32), `.qzeros` (I32), `.scales` (F16) - nine separate
tensors per expert.  The engine reads an expert as ONE contiguous blob `[gate rows | up rows | down rows]` in the
ggml form named by `native_experts.txt`, so no in-place route exists: this tool writes `experts.bin`.

    python tools/w4a16_pack.py --ckpt <snapshot dir> --out <pack dir>            (the whole model, 67.95 GB)
    python tools/w4a16_pack.py --ckpt ... --out ... --layers 0 --dry-run         (the plan, nothing written)
    python tools/w4a16_pack.py --ckpt ... --out ... --verify                     (re-check a finished artifact)

FORMAT (DW2): ggml `Q4_0`, type id 2 - 32 values per block, 18 B: `ggml_half d` then 16 bytes of nibbles, with
element j in the LOW nibble of byte j and element j+16 in the HIGH nibble of the same byte (ggml's
`dequantize_row_q4_0`).  The checkpoint's own fp16 group scale is copied bit-for-bit and repeated into all four
32-blocks of its 128-group; the checkpoint's own nibbles are re-ordered into that layout; the zero-point plane is
DROPPED because every word of it is the constant 0x77777777 (stored 7, true zero point 8 - W4A16-PLAN.md §1.4)
and the `- 8` is folded into the decode `w = (q - 8) * d`.  No value is re-quantised: the decode of the blob is
bit-for-bit the decode of the source triple, which `tools/test_w4a16_pack.py` proves element by element.

BLOB (per expert, all three projections, contiguous):
    [ gate rows  FF x 1440 B | up rows FF x 1440 B | down rows H x 360 B ]  = 2,764,800 B
    1440 = 2560/32 * 18 (a gate/up row),  360 = 640/32 * 18 (a down row)

OUTPUT
    experts.bin            48 x 512 blobs, 67,947,724,800 B exactly (FileExpertSource refuses anything else)
    native_experts.txt     the engine's table; written LAST (temporary name + rename) - it is the completion
                           marker, so a stopped run leaves no half-pack the engine would read
    w4a16_experts.manifest.json   the measured account: per-layer sha256 of the bytes written, the zero-point
                           scan, the scale census, counts in vs out, timings
    w4a16_experts.progress.json   what is done, for --resume

RESUMABLE: experts.bin is preallocated to its exact final size; every layer whose bytes are flushed is recorded
in the progress file, and a re-run skips it.  A tensor that is missing, of the wrong shape/dtype, or whose
zero-point plane is not the expected constant is a hard error with the tensor's name - the pack never silently
skips anything, and it counts what it consumed against the index's own weight_map.

NEVER /tmp.  The pack is ~68 GB: stage it on a volume with the room (the W-track uses the NTFS volume).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import pathlib
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ---------------------------------------------------------------- geometry (W4A16-PLAN.md §1.2) ---------------
H, FF = 2560, 640                       # hidden size, expert intermediate size
N_LAYERS, N_EXPERTS = 48, 512           # 24,576 routed experts
GROUP_ELEMS = 128                       # the group scale runs along the contraction axis
QK = 32                                 # ggml Q4_0 block
Q4_0_BYTES = 18                         # ggml_half d + 16 bytes of nibbles
GU_ROW = H // QK * Q4_0_BYTES           # 1440 B, one gate/up row (2560 values)
D_ROW = FF // QK * Q4_0_BYTES           # 360 B, one down row (640 values)
UP_OFF = GU_ROW * FF                    # 921,600
DOWN_OFF = 2 * UP_OFF                   # 1,843,200
BLOB = DOWN_OFF + D_ROW * H             # 2,764,800
LAYER_BYTES = BLOB * N_EXPERTS          # 1,415,577,600
TOTAL_BYTES = LAYER_BYTES * N_LAYERS    # 67,947,724,800
ROLES = (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF))
PLANES = ("qweight", "qzeros", "scales")
Q4_0_TYPE_ID = 2                        # ggml GGML_TYPE_Q4_0
ZERO_WORD = 0x77777777                  # every qzeros int32 of this checkpoint (stored nibble 7)
TRUE_ZERO_POINT = 8                     # stored 7 + the producer's `zeros += 1`
NAME = "model.language_model.layers.%d.mlp.experts.%d.%s.%s"
BASE = "model.language_model.layers.%d.mlp.experts.%d.%s"


def triple(layer: int, expert: int) -> list[str]:
    """The nine tensor names of one expert, in the order the blob wants them."""
    return [NAME % (layer, expert, role, plane) for role, _, _ in ROLES for plane in PLANES]


# ---------------------------------------------------------------- the source -------------------------------------------------
class Shard:
    """One safetensors file, mmap'd read-only (no copy of a weight ever lands in RAM)."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        self.hdr = json.loads(self.fh.read(n))
        self.data_start = 8 + n
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)

    def info(self, name: str) -> dict:
        if name not in self.hdr:
            raise KeyError(f"{name} is not in {self.path.name}")
        return self.hdr[name]

    def raw(self, name: str) -> memoryview:
        v = self.info(name)
        return self.mm[self.data_start + v["data_offsets"][0]: self.data_start + v["data_offsets"][1]]

    def arr(self, name: str, dtype, ndim: int):
        a = np.frombuffer(self.raw(name), dtype=dtype)
        sh = self.info(name)["shape"]
        return a.reshape(sh) if len(sh) > 1 else a


class Source:
    """The checkpoint: the shards, the index, and a per-role open shard cache."""

    def __init__(self, ckpt: pathlib.Path):
        self.ckpt = ckpt
        idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self.index = idx
        self.shards: dict[str, Shard] = {}
        self.shard_names = sorted({v for v in self.weight_map.values()})

    def shard(self, name: str) -> Shard:
        p = self.ckpt / name
        if name not in self.shards:
            self.shards[name] = Shard(p)
        return self.shards[name]

    def of(self, tensor: str) -> Shard:
        try:
            return self.shard(self.weight_map[tensor])
        except KeyError:
            raise KeyError(f"{tensor} is not in the index's weight_map") from None

    def close(self) -> None:
        for s in self.shards.values():
            s.mm.close()
            s.fh.close()
        self.shards.clear()


# ---------------------------------------------------------------- the repack (the packer's own encoder) -------
def repack_projection(qw_u32: np.ndarray, sc_f16: np.ndarray, out_dim: int, in_dim: int) -> bytes:
    """One projection's int4-g128 triple -> ggml Q4_0 rows [out_dim][in_dim/32 * 18].

    qw_u32: [in/8, out] uint32 (nibble k of word (i>>3, o) is input i = (i>>3)*8 + k)
    sc_f16: [in/128, out] float16  (SIGNED; the group scale, not its magnitude)
    """
    # 1. unpack the nibbles: [in/8, out, 8] -> [in/8, 8, out] -> [in, out]
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = ((qw_u32[:, :, None] >> shifts) & 0xF).transpose(0, 2, 1).reshape(in_dim, out_dim)
    # 2. ggml's block: byte k holds element 32b+k in the low nibble and 32b+k+16 in the high one
    n_blk = in_dim // QK
    qr = q.reshape(n_blk // 4, 4, QK, out_dim)          # [group, block-in-group, element, out]
    qs = (qr[:, :, :16, :] | (qr[:, :, 16:, :] << 4)).astype(np.uint8)   # [group, 4, 16, out]
    qs = qs.reshape(n_blk, 16, out_dim)                 # [block, byte, out] - block = 4*group + b
    # 3. the scale: the checkpoint's fp16 group scale, repeated into the four blocks of its group, raw bits
    d = np.ascontiguousarray(np.repeat(sc_f16, 4, axis=0)).view(np.uint16)   # [block, out] raw fp16 bits
    row = np.empty((out_dim, n_blk, Q4_0_BYTES), np.uint8)
    row[:, :, 0] = (d & 0xFF).T
    row[:, :, 1] = (d >> 8).T
    row[:, :, 2:] = qs.transpose(2, 0, 1)
    return row.tobytes()


def repack_expert(src: Source, layer: int, expert: int) -> tuple[bytes, dict]:
    """One expert's nine tensors -> its 2,764,800-byte blob, plus the stats of what was read.

    Raises on anything unexpected (a missing tensor, the wrong shape, a zero-point plane that is not the
    constant) - this tool never silently skips a tensor.
    """
    parts = []
    st = {"tensors": 0, "qzeros_words": 0, "qzeros_min": 0xFFFFFFFF, "qzeros_max": 0, "scales": 0,
          "scales_negative": 0, "scales_zero": 0, "scale_min": float("inf"), "scale_max": float("-inf")}
    for role, out_dim, in_dim in ROLES:
        base = BASE % (layer, expert, role)
        sh = src.of(base + ".qweight")
        qw = sh.arr(base + ".qweight", np.uint32, 2)
        sc = sh.arr(base + ".scales", np.float16, 2)
        want_qw, want_sc = (in_dim // 8, out_dim), (in_dim // GROUP_ELEMS, out_dim)
        if qw.shape != want_qw:
            raise ValueError(f"{base}.qweight is {qw.shape}, expected {want_qw}")
        if sc.shape != want_sc:
            raise ValueError(f"{base}.scales is {sc.shape}, expected {want_sc}")
        # the zero-point plane: read it (a general GPTQ reader must), prove it is the constant, drop it
        zz = sh.arr(base + ".qzeros", np.uint32, 2)
        want_qz = (in_dim // GROUP_ELEMS, out_dim // 8)
        if zz.shape != want_qz:
            raise ValueError(f"{base}.qzeros is {zz.shape}, expected {want_qz}")
        lo, hi = int(zz.min()), int(zz.max())
        st["qzeros_words"] += int(zz.size)
        st["qzeros_min"] = min(st["qzeros_min"], lo)
        st["qzeros_max"] = max(st["qzeros_max"], hi)
        if lo != ZERO_WORD or hi != ZERO_WORD:
            raise ValueError(
                f"{base}.qzeros is not the constant 0x{ZERO_WORD:08x} (min 0x{lo:08x}, max 0x{hi:08x}): this "
                f"checkpoint's zero point is not the one W4A16-PLAN.md §1.4 measured, so the pack is refused "
                f"rather than mis-decoded")
        st["scales"] += int(sc.size)
        st["scales_negative"] += int((sc < 0).sum())
        st["scales_zero"] += int((sc == 0).sum())
        st["scale_min"] = min(st["scale_min"], float(sc.min()))
        st["scale_max"] = max(st["scale_max"], float(sc.max()))
        parts.append(repack_projection(qw, sc, out_dim, in_dim))
        st["tensors"] += 3
    blob = b"".join(parts)
    if len(blob) != BLOB:
        raise AssertionError(f"layer {layer} expert {expert}: blob is {len(blob)} B, expected {BLOB}")
    return blob, st


# ---------------------------------------------------------------- the pack ---------------------------------------------------
def layer_of(src: Source, layer: int, chunk: int, threads: int) -> tuple[bytes, dict]:
    """Every blob of one layer, in expert order, computed `chunk` experts at a time in `threads` threads."""
    out = bytearray()
    agg = {"tensors": 0, "qzeros_words": 0, "qzeros_min": 0xFFFFFFFF, "qzeros_max": 0, "scales": 0,
           "scales_negative": 0, "scales_zero": 0, "scale_min": float("inf"), "scale_max": float("-inf")}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for first in range(0, N_EXPERTS, chunk):
            last = min(first + chunk, N_EXPERTS)
            blobs = list(pool.map(lambda e: repack_expert(src, layer, e), range(first, last)))
            for b, _st in blobs:
                out += b
            # aggregated in the parent thread: the counters must be right, and a shared dict is not thread-safe
            for k in ("tensors", "qzeros_words", "scales", "scales_negative", "scales_zero"):
                agg[k] += sum(st[k] for _, st in blobs)
            agg["qzeros_min"] = min([agg["qzeros_min"]] + [st["qzeros_min"] for _, st in blobs])
            agg["qzeros_max"] = max([agg["qzeros_max"]] + [st["qzeros_max"] for _, st in blobs])
            agg["scale_min"] = min([agg["scale_min"]] + [st["scale_min"] for _, st in blobs])
            agg["scale_max"] = max([agg["scale_max"]] + [st["scale_max"] for _, st in blobs])
            del blobs
    if len(out) != LAYER_BYTES:
        raise AssertionError(f"layer {layer}: {len(out)} B, expected {LAYER_BYTES}")
    return bytes(out), agg


def read_json(p: pathlib.Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_json(p: pathlib.Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def native_experts_text(layers: list[int]) -> str:
    """The engine's table.  No GGUF: the experts are in experts.bin, so there is no gate/up/down offset column
    (the engine's reader takes the five-column line and leaves gguf_off empty - expert_layout.cpp:189-220)."""
    head = ("# strata native experts v3: layer gu_type d_type offset blob_bytes (n_expert %d, total %d; the "
            "experts are in experts.bin, so there is NO gate_off/up_off/down_off column - this pack has no GGUF)\n"
            % (N_EXPERTS, TOTAL_BYTES))
    body = "".join("%d %d %d %d %d\n" % (l, Q4_0_TYPE_ID, Q4_0_TYPE_ID, l * LAYER_BYTES, BLOB) for l in layers)
    return head + body


def cmd_pack(a) -> int:
    ckpt = pathlib.Path(a.ckpt).absolute()
    out = pathlib.Path(a.out).absolute()
    layers = parse_layers(a.layers)
    print(f"checkpoint: {ckpt}")
    print(f"pack:       {out}")
    print(f"format:     ggml Q4_0 (type {Q4_0_TYPE_ID}), blob {BLOB:,d} B/expert, "
          f"layer {LAYER_BYTES:,d} B, total {TOTAL_BYTES:,d} B ({TOTAL_BYTES/1e9:.3f} GB)")
    print(f"layers:     {len(layers)} ({layers[0]}..{layers[-1]})" if layers else "layers:     none")
    t0 = time.time()
    src = Source(ckpt)
    print(f"index:      {len(src.weight_map):,d} tensors in {len(src.shard_names)} shards")
    missing = []
    for l in layers:
        for e in range(0, N_EXPERTS, max(1, N_EXPERTS // 8)):        # probe 8 experts per layer
            for t in triple(l, e):
                if t not in src.weight_map:
                    missing.append(t)
    if missing:
        for t in missing[:10]:
            print(f"  MISSING {t}")
        print(f"the checkpoint does not have {len(missing)} of the tensors the plan is built on: refused")
        return 1
    if a.dry_run:
        print("dry run: nothing written.  The plan:")
        print(f"  tensors to consume: {len(layers) * N_EXPERTS * 9:,d} "
              f"({len(layers) * N_EXPERTS * 3:,d} triples)")
        print(f"  blobs to write:     {len(layers) * N_EXPERTS:,d}")
        print(f"  native_experts.txt: {len(layers)} lines")
        src.close()
        return 0

    out.mkdir(parents=True, exist_ok=True)
    binp = out / "experts.bin"
    prog = out / "w4a16_experts.progress.json"
    man = out / "w4a16_experts.manifest.json"
    complete = sorted(set(layers)) == list(range(N_LAYERS))
    if not complete:
        print(f"PARTIAL RUN: {len(layers)} of {N_LAYERS} layers.  experts.bin is still preallocated to the full "
              f"{TOTAL_BYTES:,d} B (the engine's FileExpertSource requires exactly that), but native_experts.txt "
              f"is written as .partial - a partial pack must not look finished.")
    fd = os.open(binp, os.O_RDWR | os.O_CREAT, 0o664)
    have = os.fstat(fd).st_size
    if have != TOTAL_BYTES:
        if a.resume and have not in (0, TOTAL_BYTES) and not a.truncate:
            print(f"{binp} is {have:,d} B, not 0 or {TOTAL_BYTES:,d}; --truncate to start it over")
            os.close(fd)
            return 1
        os.ftruncate(fd, TOTAL_BYTES)
        have = os.fstat(fd).st_size
    if have != TOTAL_BYTES:
        print(f"cannot preallocate {binp} to {TOTAL_BYTES:,d} B (it is {have:,d})")
        os.close(fd)
        return 1
    st = os.statvfs(out)
    free = st.f_bavail * st.f_frsize
    print(f"experts.bin: {TOTAL_BYTES:,d} B preallocated, {free/1e9:.1f} GB free on the volume")

    state = read_json(prog) or {}
    if a.truncate:
        state = {}
        prog.unlink(missing_ok=True)
    done = set(state.get("layers_done", [])) if a.resume else set()
    sha = dict(state.get("layer_sha256", {})) if a.resume else {}
    if done:
        print(f"resume: {len(done)} layer(s) already done: {sorted(done)[:8]}{' ...' if len(done) > 8 else ''}")

    census = {"tensors_consumed": 0, "qzeros_words": 0, "qzeros_min": 0xFFFFFFFF, "qzeros_max": 0,
              "scales": 0, "scales_negative": 0, "scales_zero": 0, "scale_min": float("inf"),
              "scale_max": float("-inf"), "blobs_written": 0, "seconds_compute": 0.0}
    todo = [l for l in layers if l not in done]
    print(f"packing {len(todo)} layer(s) with {a.threads} threads, {a.chunk} experts per batch", flush=True)
    tw = time.time()
    for n, l in enumerate(todo):
        t1 = time.time()
        buf, agg = layer_of(src, l, a.chunk, a.threads)
        census["seconds_compute"] += time.time() - t1
        t2 = time.time()
        os.pwrite(fd, buf, l * LAYER_BYTES)
        os.fsync(fd)
        census["blobs_written"] += N_EXPERTS
        census["tensors_consumed"] += agg["tensors"]
        for k in ("qzeros_words", "scales", "scales_negative", "scales_zero"):
            census[k] += agg[k]
        census["qzeros_min"] = min(census["qzeros_min"], agg["qzeros_min"])
        census["qzeros_max"] = max(census["qzeros_max"], agg["qzeros_max"])
        census["scale_min"] = min(census["scale_min"], agg["scale_min"])
        census["scale_max"] = max(census["scale_max"], agg["scale_max"])
        sha[str(l)] = hashlib.sha256(buf).hexdigest()
        done.add(l)
        del buf
        write_json(prog, {"schema": 1, "blob_bytes": BLOB, "layers_done": sorted(done), "layer_sha256": sha,
                          "tensors_consumed": census["tensors_consumed"],
                          "qzeros_words": census["qzeros_words"], "complete": sorted(done) == list(range(N_LAYERS))})
        el = time.time() - tw
        print(f"  layer {l:2d}  {census['blobs_written']:>6,d}/{len(layers)*N_EXPERTS:,d} blobs  "
              f"{census['tensors_consumed']:>7,d} tensors consumed  compute {time.time()-t1:.1f}s  "
              f"write {time.time()-t2:.1f}s  {el/(n+1):.1f}s/layer  {el:.0f}s elapsed", flush=True)
    os.close(fd)
    allsha = read_json(man) or {}
    prev = allsha.get("layer_sha256", {})
    prev.update({k: v for k, v in sha.items() if k not in prev})
    sha = prev
    elapsed = time.time() - t0
    manifest = {
        "schema": 1,
        "tool": "tools/w4a16_pack.py",
        "checkpoint": str(ckpt),
        "format": {"ggml_type": "Q4_0", "type_id": Q4_0_TYPE_ID, "block_bytes": Q4_0_BYTES, "block_values": QK,
                   "decode": "w = ((q & 0xF) - 8) * d, d = the checkpoint's fp16 group scale (signed), "
                             "repeated into the four 32-blocks of its 128-group",
                   "zero_point": {"stored_nibble": TRUE_ZERO_POINT - 1, "true": TRUE_ZERO_POINT,
                                  "source_qzeros_constant": f"0x{ZERO_WORD:08x}",
                                  "plane_dropped": True}},
        "geometry": {"H": H, "FF": FF, "layers": N_LAYERS, "experts_per_layer": N_EXPERTS,
                     "group_elems": GROUP_ELEMS, "gu_row_bytes": GU_ROW, "d_row_bytes": D_ROW,
                     "up_off": UP_OFF, "down_off": DOWN_OFF, "blob_bytes": BLOB, "layer_bytes": LAYER_BYTES,
                     "experts_bin_bytes": TOTAL_BYTES},
        "source": {"snapshot": str(ckpt),
                   "shards": [{"name": n, "size": (ckpt / n).stat().st_size} for n in src.shard_names],
                   "index_total_size": src.index.get("total_size"),
                   "index_total_parameters": src.index.get("total_parameters"),
                   "weight_map_entries": len(src.weight_map)},
        "counts": {"layers_written": len(done), "blobs_written": census["blobs_written"],
                   "expert_triples_expected": len(done) * N_EXPERTS,
                   "tensors_consumed": census["tensors_consumed"],
                   "tensors_expected": len(done) * N_EXPERTS * 9,
                   "planes_per_triple": 3, "zero_point_planes_dropped": len(done) * N_EXPERTS * 3},
        "zero_point_scan": {"int32_words": census["qzeros_words"], "min": f"0x{census['qzeros_min']:08x}",
                            "max": f"0x{census['qzeros_max']:08x}",
                            "all_equal_constant": census["qzeros_min"] == census["qzeros_max"] == ZERO_WORD},
        "scales": {"values": census["scales"], "negative": census["scales_negative"],
                   "zero": census["scales_zero"], "min": census["scale_min"], "max": census["scale_max"]},
        "timing": {"wall_seconds": elapsed, "compute_seconds_per_layer": census["seconds_compute"] / max(1, len(todo)),
                   "seconds_per_layer": elapsed / max(1, len(todo))},
        "layer_sha256": sha,
        "native_experts_sha256": hashlib.sha256(native_experts_text(layers).encode()).hexdigest(),
    }
    write_json(man, manifest)
    if complete and len(done) == N_LAYERS:
        tmp = out / "native_experts.txt.tmp"
        tmp.write_text(native_experts_text(list(range(N_LAYERS))))
        tmp.replace(out / "native_experts.txt")
        print(f"native_experts.txt written (the completion marker, {N_LAYERS} layers)")
    else:
        (out / "native_experts.txt.partial").write_text(native_experts_text(sorted(done)))
        print(f"partial: native_experts.txt.partial written; native_experts.txt is NOT (the pack is unfinished)")
    print(f"\nblobs {census['blobs_written']:,d}  tensors consumed {census['tensors_consumed']:,d}  "
          f"qzeros words scanned {census['qzeros_words']:,d} (all 0x{ZERO_WORD:08x}: "
          f"{census['qzeros_min'] == census['qzeros_max'] == ZERO_WORD})")
    print(f"scales {census['scales']:,d}  negative {census['scales_negative']:,d}  zero {census['scales_zero']:,d}  "
          f"[{census['scale_min']:.6g}, {census['scale_max']:.6g}]")
    print(f"wall {elapsed:.1f}s  ({elapsed/max(1,len(todo)):.1f}s/layer, compute "
          f"{census['seconds_compute']/max(1,len(todo)):.1f}s/layer)")
    src.close()
    return 0


def cmd_verify(a) -> int:
    out = pathlib.Path(a.out).absolute()
    man = read_json(out / "w4a16_experts.manifest.json")
    binp = out / "experts.bin"
    if man is None or not binp.exists():
        print(f"{out}: no manifest or no experts.bin")
        return 1
    size = binp.stat().st_size
    print(f"{binp}: {size:,d} B (the layout requires {TOTAL_BYTES:,d} B: {'OK' if size == TOTAL_BYTES else 'WRONG'})")
    lay = out / "native_experts.txt"
    print(f"native_experts.txt: {'present' if lay.exists() else 'ABSENT (the pack is not finished)'}")
    if lay.exists():
        txt = lay.read_text()
        lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
        print(f"  {len(lines)} layer lines; header: {txt.splitlines()[0][:120]}")
        bad = [ln for ln in lines if ln.split()[1:3] != ["2", "2"] or int(ln.split()[3]) % LAYER_BYTES or
               int(ln.split()[4]) != BLOB]
        print(f"  layers naming type 2/2, offset % layer_bytes == 0 and blob {BLOB}: "
              f"{len(lines) - len(bad)} of {len(lines)}")
    ok = True
    t0 = time.time()
    with open(binp, "rb") as f:
        complete = sorted(int(k) for k, _ in man["layer_sha256"].items())
        for l in complete:
            f.seek(l * LAYER_BYTES)
            h = hashlib.sha256()
            left = LAYER_BYTES
            while left:
                b = f.read(min(left, 16 << 20))
                if not b:
                    print(f"  layer {l}: short read")
                    ok = False
                    break
                h.update(b)
                left -= len(b)
            same = h.hexdigest() == man["layer_sha256"][str(l)]
            ok &= same
            print(f"  layer {l:2d}: sha256 {'MATCHES the manifest' if same else 'DIFFERS from the manifest'}")
    print(f"verify: {'the artifact is what the packer wrote' if ok else 'MISMATCH'} "
          f"({len(man['layer_sha256'])} layers in {time.time()-t0:.1f} s, {size/1e9/(time.time()-t0):.2f} GB/s)")
    print(f"counts in the manifest: {json.dumps(man['counts'])}")
    print(f"zero-point scan: {json.dumps(man['zero_point_scan'])}")
    return 0 if ok else 1


def parse_layers(spec: str) -> list[int]:
    if not spec or spec == "all":
        return list(range(N_LAYERS))
    out: list[int] = []
    for part in spec.split(","):
        if ":" in part or "-" in part:
            a, b = part.replace("-", ":").split(":")
            out += list(range(int(a), int(b)))
        else:
            out.append(int(part))
    return [l for l in out if 0 <= l < N_LAYERS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="the checkpoint's snapshot directory (with the 17 shards)")
    ap.add_argument("--out", required=True, help="the pack directory")
    ap.add_argument("--layers", default="all", help="all | 0-4 | 0,7,47 (default all)")
    ap.add_argument("--threads", type=int, default=min(20, os.cpu_count() or 8))
    ap.add_argument("--chunk", type=int, default=64, help="experts computed and written per batch")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip the layers the progress file records as done (default; no size change)")
    ap.add_argument("--truncate", action="store_true", help="truncate experts.bin and start over")
    ap.add_argument("--verify", action="store_true", help="re-check a finished artifact against its manifest")
    a = ap.parse_args()
    if a.verify:
        return cmd_verify(a)
    return cmd_pack(a)


if __name__ == "__main__":
    sys.exit(main())
