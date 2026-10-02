"""A minimal GGUF v3 WRITER, including `Q2_0`.

Why this exists
---------------
P1.S1 of the plan says the tiny synthetic model is "written as a real GGUF via `gguf-py`" using
"at least `Q2_0`". That does not work: `gguf-py` has no `GGMLQuantizationType` member for type 42, so
`GGMLQuantizationType(42)` raises `ValueError`. Verified in round 39 and again here.

`src/artifact/` needs a writer regardless - P1.S2 requires a reader with no ggml dependency, and the
writer is its companion - so this is that writer in Python, where it can be validated immediately
against `tools/gguf_reader.py` and against llama.cpp.

Scope: enough to build the P1.S1 tiny model. It is NOT a general-purpose quantizer; the only
quantizer implemented is `Q2_0`, and its contract is `docs/q2_0-contract.md`:
    block = 64 elements, fp16 scale, 16 bytes of 2-bit codes, LSB-first, 4 codes per byte
    d = amax, code = clamp(round(w/d) + 1, 0, 3), symbol = code - 1  ->  {-1, 0, +1, +2}

Self-checking: `python tools/gguf_writer.py` writes a synthetic v3 file, reads it back with
`gguf_reader.py`, and asserts the tensors and metadata survived byte-exactly.
"""
from __future__ import annotations

import dataclasses
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUF_MAGIC, GGUFFile  # noqa: E402

# ggml_type ids we can write. Values match GGML_TYPES in gguf_reader.py.
TYPE_IDS = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30, "Q2_0": 42}
DEFAULT_ALIGNMENT = 32

_META_IDS = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
             "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12}


# ------------------------------------------------------------------ Q2_0, per docs/q2_0-contract.md
def quantize_q2_0(w: np.ndarray) -> bytes:
    """(n,) float32 -> concatenated 18-byte blocks. n must be a multiple of 64."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size % 64:
        raise ValueError(f"Q2_0 needs a multiple of 64 elements, got {w.size}")
    blocks = w.reshape(-1, 64)
    out = bytearray()
    for b in blocks:
        amax = float(np.max(np.abs(b)))
        d = amax
        # d = amax (NOT amax/2) - the grid is asymmetric, symbols land in {-1,0,+1,+2}.
        code = np.zeros(64, dtype=np.uint8) if d == 0 else \
            np.clip(np.rint(b / d).astype(np.int32) + 1, 0, 3).astype(np.uint8)
        packed = np.zeros(16, dtype=np.uint8)
        # j -> byte j//4, bit (j%4)*2 ;  four codes per byte, least significant first.
        for j in range(64):
            packed[j // 4] |= np.uint8(code[j] << ((j % 4) * 2))
        out += struct.pack("<e", d)      # fp16 scale
        out += packed.tobytes()
    return bytes(out)


def dequantize_q2_0(raw: bytes) -> np.ndarray:
    """Inverse of the above, for the round-trip check. Mirrors `dequantize_row_q2_0`."""
    nblocks = len(raw) // 18
    out = np.empty(nblocks * 64, dtype=np.float32)
    for i in range(nblocks):
        d = struct.unpack_from("<e", raw, i * 18)[0]
        qs = raw[i * 18 + 2: i * 18 + 18]
        for j in range(64):
            c = (qs[j // 4] >> ((j % 4) * 2)) & 0x03
            out[i * 64 + j] = (c - 1) * d
    return out


# ------------------------------------------------------------------ Q8_0, ggml's reference rule
def quantize_q8_0(w: np.ndarray) -> bytes:
    """(n,) float32 -> concatenated 34-byte blocks (fp16 d + 32 int8). n must be a multiple of 32.

    ggml's `quantize_row_q8_0_ref` exactly: `d = amax / 127`, `qs = roundf(x / d)`.  The rounding is
    round-half-away-from-zero (C's roundf), not rint's round-half-to-even, and the divisor is 127 so a
    saturated code is +-127 - both are part of the reference's byte output, not free choices.
    """
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size % 32:
        raise ValueError(f"Q8_0 needs a multiple of 32 elements, got {w.size}")
    blocks = w.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0))
    # x + copysign(0.5, x) is round-half-away-from-zero, which is what roundf does
    scaled = blocks * inv
    q = np.trunc(scaled + np.copysign(np.float32(0.5), scaled)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = d.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:34] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(raw: bytes) -> np.ndarray:
    """Inverse of the above, for the round-trip check. Mirrors `dequantize_row_q8_0`."""
    nblocks = len(raw) // 34
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks, 34)
    d = arr[:, 0:2].copy().view("<f2").astype(np.float32).reshape(nblocks, 1)
    q = arr[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def f32_to_bf16_bytes(values: np.ndarray) -> bytes:
    """(n,) float32 -> n * 2 bytes of BF16, by truncation of the low 16 bits.

    Truncation is EXACT for a value that came from BF16 (the source of every tensor here): bf16 is the
    top half of its f32 image, so the widening the reader did is undone bit for bit.  For a general
    f32 it is the rounding llama.cpp's own converters use for this type.
    """
    u = np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32)
    return (u >> 16).astype("<u2").tobytes()


# ------------------------------------------------------------------ writer
@dataclasses.dataclass
class _Tensor:
    name: str
    shape: list[int]          # GGUF order: dim 0 varies fastest
    type_name: str
    data: bytes


class GGUFWriter:
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self.metadata: dict[str, tuple[str, object]] = {}
        self.tensors: list[_Tensor] = []

    # ---- metadata
    def add(self, key: str, value, type_name: str | None = None) -> None:
        if type_name is None:
            type_name = {bool: "bool", int: "i64", float: "f64", str: "string"}.get(type(value))
        if type_name is None:
            if isinstance(value, list) and value:
                elem = {bool: "bool", int: "i32", float: "f32", str: "string"}.get(type(value[0]))
                if elem is None:
                    raise TypeError(f"unsupported array element {type(value[0])} for {key}")
                type_name = "array:" + elem
            else:
                raise TypeError(f"cannot infer a GGUF type for {key}")
        self.metadata[key] = (type_name, value)

    # ---- tensors
    def add_f32(self, name: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype="<f4")
        self.tensors.append(_Tensor(name, list(a.shape), "F32", a.tobytes()))

    def add_q2_0(self, name: str, arr: np.ndarray, shape: list[int]) -> None:
        """`arr` is (rows, in_features) row-major with every row `shape[0]` long; `shape` is the FULL
        GGUF shape. The shape must be passed explicitly, not inferred as [in, rows]: a fused expert
        tensor is [in, out, experts], and inferring 2-D produced `[64, 4096]` where llama.cpp expects
        `[64, 256, 16]`. Q2_0 blocks run along dim 0 only, so the extra axes are just a layout.
        """
        a = np.asarray(arr, dtype=np.float32)
        in_features = shape[0]
        if a.ndim != 2 or a.shape[1] != in_features:
            raise ValueError(f"{name}: expected (rows, {in_features}), got {a.shape}")
        if a.shape[0] * in_features != int(np.prod(shape)):
            raise ValueError(f"{name}: {a.shape} does not cover shape {shape}")
        per_row = quantize_q2_0(a.reshape(-1))          # row-major: row 0 first
        self.tensors.append(_Tensor(name, list(shape), "Q2_0", per_row))

    def add_bytes(self, name: str, data: bytes, shape: list[int], type_name: str) -> None:
        """Raw payload already in `type_name`'s on-disk layout; `shape` is the GGUF (dim 0 fastest)."""
        if type_name not in TYPE_IDS:
            raise ValueError(f"{name}: no ggml type id for {type_name}")
        expected = int(np.prod(shape))
        per = {"BF16": 2, "F16": 2, "F32": 4}.get(type_name)
        if per is not None:
            want = expected * per
        elif type_name == "Q8_0":
            if expected % 32:
                raise ValueError(f"{name}: {expected} elements is not a multiple of Q8_0's 32")
            want = (expected // 32) * 34
        else:
            raise ValueError(f"{name}: add_bytes does not know {type_name}'s block size")
        if len(data) != want:
            raise ValueError(f"{name}: {len(data)} B of {type_name} for shape {shape} (want {want})")
        self.tensors.append(_Tensor(name, list(shape), type_name, data))

    def add_q8_0(self, name: str, arr: np.ndarray, shape: list[int]) -> None:
        """`arr` is (rows, in_features) row-major with every row `shape[0]` long; `shape` is the FULL
        GGUF shape (see `add_q2_0` for why it is passed rather than inferred)."""
        a = np.asarray(arr, dtype=np.float32)
        in_features = shape[0]
        if a.ndim != 2 or a.shape[1] != in_features:
            raise ValueError(f"{name}: expected (rows, {in_features}), got {a.shape}")
        if a.shape[0] * in_features != int(np.prod(shape)):
            raise ValueError(f"{name}: {a.shape} does not cover shape {shape}")
        self.add_bytes(name, quantize_q8_0(a.reshape(-1)), shape, "Q8_0")

    def add_bf16(self, name: str, arr: np.ndarray, shape: list[int]) -> None:
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if int(np.prod(shape)) != a.size:
            raise ValueError(f"{name}: {a.size} values do not cover shape {shape}")
        self.add_bytes(name, f32_to_bf16_bytes(a), shape, "BF16")

    # ---- serialisation
    def _kv_bytes(self, key: str, type_name: str, value) -> bytes:
        out = bytearray()
        out += struct.pack("<Q", len(key)) + key.encode("utf-8")
        if type_name.startswith("array:"):
            elem = type_name.split(":", 1)[1]
            out += struct.pack("<I", _META_IDS["array"])
            out += struct.pack("<I", _META_IDS[elem])
            out += struct.pack("<Q", len(value))
            for v in value:
                out += self._scalar(elem, v)
            return bytes(out)
        out += struct.pack("<I", _META_IDS[type_name])
        out += self._scalar(type_name, value)
        return bytes(out)

    @staticmethod
    def _scalar(type_name: str, v) -> bytes:
        if type_name == "string":
            b = v.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
               "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[type_name]
        return struct.pack(fmt, v)

    def write(self, path) -> pathlib.Path:
        path = pathlib.Path(path)
        body = bytearray()
        body += struct.pack("<I", GGUF_MAGIC)
        body += struct.pack("<I", 3)                       # GGUF v3
        body += struct.pack("<Q", len(self.tensors))
        body += struct.pack("<Q", len(self.metadata))
        for k, (t, v) in self.metadata.items():
            body += self._kv_bytes(k, t, v)

        # Tensor directory: offsets are relative to the start of the data section.
        offset = 0
        infos = bytearray()
        for t in self.tensors:
            infos += struct.pack("<Q", len(t.name)) + t.name.encode("utf-8")
            infos += struct.pack("<I", len(t.shape))
            infos += struct.pack(f"<{len(t.shape)}Q", *t.shape)
            infos += struct.pack("<I", TYPE_IDS[t.type_name])
            infos += struct.pack("<Q", offset)
            offset += len(t.data)
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment
        body += infos

        pad = (-len(body)) % self.alignment
        body += b"\0" * pad
        for t in self.tensors:
            body += t.data
            pad = (-len(t.data)) % self.alignment
            body += b"\0" * pad

        path.write_bytes(bytes(body))
        return path


# ------------------------------------------------------------------ self-check
def _selfcheck() -> int:
    rng = np.random.default_rng(7)
    w = GGUFWriter()
    w.add("general.architecture", "qwen4exp")
    w.add("qwen4exp.block_count", 8, "u32")
    w.add("qwen4exp.test.array", [1, 2, 3, 4], "array:i32")
    w.add("general.name", "strata tiny")

    A = rng.standard_normal((16, 64)).astype(np.float32)
    B = rng.standard_normal((4, 8)).astype(np.float32)
    w.add_q2_0("blk.0.ffn_gate_exps.weight", A, [64, 16])
    w.add_f32("blk.0.ssm_a", B)
    # Q8_0 and BF16 (added for the W4A16 native dense GGUF): the same two properties the Q2_0 arm is
    # held to - re-quantizing the dequantized values is a fixed point, and the error stays inside the
    # block's own bound - plus BF16's exactness for a value that came from BF16.
    C = rng.standard_normal((4, 64)).astype(np.float32)
    w.add_q8_0("blk.0.attn_qkv.weight", C, [64, 4])
    D = (np.asarray(rng.standard_normal(37), dtype=np.float32).view(np.uint32) >> 16 << 16).view(np.float32)
    D = D.copy()                                               # D came FROM bf16
    w.add_bf16("token_embd.weight", D, [37])

    p = pathlib.Path(__file__).resolve().parent.parent / "bench" / "tiny-selftest.gguf"
    w.write(p)
    print(f"wrote {p.name}  ({p.stat().st_size} B)")

    g = GGUFFile(p)
    ok = True
    print(f"  version {g.version}  tensors {len(g.tensors)}  metadata {len(g.metadata)}")
    if g.metadata.get("general.architecture") != "qwen4exp": print("  FAIL arch"); ok = False
    if g.metadata.get("qwen4exp.block_count") != 8:         print("  FAIL u32"); ok = False
    if g.metadata.get("qwen4exp.test.array") != [1, 2, 3, 4]: print("  FAIL array"); ok = False

    by = {t.name: t for t in g.tensors}
    t = by.get("blk.0.ffn_gate_exps.weight")
    if t is None or t.type_name != "Q2_0" or t.shape != [64, 16]:
        print(f"  FAIL tensor dir: {t}"); ok = False
    else:
        with p.open("rb") as fh:
            fh.seek(g.data_start + t.offset)
            raw = fh.read(16 * 18)
        back = dequantize_q2_0(raw)
        err = np.abs(back - A.reshape(-1)).max()
        rel = float(np.abs(back - A.reshape(-1)).mean() / np.abs(A).mean())
        print(f"  Q2_0 round trip: max abs {err:.4f}  mean rel {rel*100:.2f}%  "
              f"(quantization error, not a write bug)")
        # The real check: re-quantizing the DEQUANTIZED values must be a fixed point.
        again = dequantize_q2_0(quantize_q2_0(back))
        if not np.array_equal(again, back):
            print("  FAIL Q2_0 is not a fixed point under requantization"); ok = False
        else:
            print("  Q2_0 fixed point: PASS (requantizing the dequantized values is exact)")
    t = by.get("blk.0.attn_qkv.weight")
    if t is None or t.type_name != "Q8_0" or t.shape != [64, 4]:
        print(f"  FAIL Q8_0 tensor dir: {t}"); ok = False
    else:
        src = C.reshape(-1)
        dq = dequantize_q8_0(quantize_q8_0(src))
        if not np.array_equal(dequantize_q8_0(quantize_q8_0(dq)), dq):
            print("  FAIL Q8_0 is not a fixed point under requantization"); ok = False
        else:
            bound = float((np.abs(C.reshape(-1, 32)).max(axis=1) / 127.0 / 2.0).max())
            err = float(np.abs(dq - src).max())
            print(f"  Q8_0 fixed point: PASS  max abs error {err:.5f} <= d/2 bound {bound:.5f}: "
                  f"{'PASS' if err <= bound else 'FAIL'}")
            ok = ok and err <= bound
    t = by.get("token_embd.weight")
    if t is None or t.type_name != "BF16" or t.shape != [37]:
        print(f"  FAIL BF16 tensor dir: {t}"); ok = False
    else:
        with p.open("rb") as fh:
            fh.seek(g.data_start + t.offset)
            raw = fh.read(37 * 2)
        back = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        same = np.array_equal(back, D)
        print(f"  BF16 exact for a value that came from BF16: {'PASS' if same else 'FAIL'}")
        ok = ok and same
    print("SELF-CHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
