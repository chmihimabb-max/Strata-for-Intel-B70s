#!/usr/bin/env python3
"""i3_assemble_oracle.py - assemble ONE llama.cpp-loadable GGUF for the qwen4exp arch out of the Strata
W4A16 pack's own artifacts, so the same weights can be run through llama.cpp and through our engine.

    /usr/bin/python3 i3/i3_assemble_oracle.py --plan                 # checks + the layout, writes nothing
    /usr/bin/python3 i3/i3_assemble_oracle.py --write  --out <path>  # the real assembly
    /usr/bin/python3 i3/i3_assemble_oracle.py --verify --out <path>  # re-read + digest spot checks

WHAT GOES IN, AND WHY IT IS "SAME WEIGHTS"
  * KV metadata + the tensor NAME/SHAPE template: the upstream ISTA-DASLab GSQ-RCO IQ3_S shard 1, the one file
    this architecture is known to load in llama.cpp (I2).  Every key is copied byte-for-byte; the only edits are
    dropping split.no/split.count/split.tensors.count (this is one file) and rewording general.name.
  * the 144 routed-expert tensors (48 layers x gate/up/down, ne2 = 512): ggml Q4_0 (type 2), the bytes of the
    Strata W4A16 pack's experts.bin.  Our blob is EXPERT-MAJOR (expert e: gate|up|down back to back), ggml's
    3D tensor is ROLE-MAJOR (expert e's gate, then expert e+1's gate...), so the three role tensors are gathered
    out of each layer's 1,415,577,600-byte region - same bytes, re-laid out.
  * the 1079 dense tensors: whatever our engine itself reads for its W4A16 run of record, i.e. the serve-pack
    index: 302 of them are served from native-dense.gguf (M5's dense GGUF, Q8_0/BF16 = ggml types already) and
    777 from dense.bin (index kinds 2/4/5 = F32/BF16/F16, verbatim ggml row layouts).  The two tensors our pack
    quantises to its S-form (token_embd/output.weight) are among the 302, so no non-ggml form is needed here.
  * the PLE n-gram table: ple-iq4nl.gguf (IQ4_NL, type 20) - the fp8 variant the engine's record run uses is
    declared I8 with a strata.ple.format key, a type llama.cpp does not have, so both engines are pointed at the
    SAME IQ4_NL table instead (one shared piece, one representation).

NOTHING IS MODIFIED IN PLACE: every artifact is opened read-only; the output is a new file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

R = pathlib.Path("/home/michael/strata-xpu")
SSD = pathlib.Path("/run/media/michael/2208B12208B0F63F/strata-w4a16")
TEMPLATE = (pathlib.Path.home() / ".cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
            / "snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S"
            / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
SERVE_PACK = SSD / "serve-pack"
NATIVE_DENSE = SSD / "native" / "native-dense.gguf"
PLE_IQ4NL = SSD / "ple-iq4nl.gguf"
DENSE_BIN = SERVE_PACK / "dense.bin"
EXPERTS_BIN = SERVE_PACK / "experts.bin"
MANIFEST = SSD / "pack" / "w4a16_experts.manifest.json"   # W1's own record of the blob geometry (the anchor)
NATIVE_EXPERTS_TXT = SERVE_PACK / "native_experts.txt"    # the spans file the engine itself reads
DEFAULT_OUT = SSD / "i3-oracle" / "w4a16-q4_0-dense-ours-i3-oracle.gguf"

GGML = {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 8, "BF16": 30, "IQ4_NL": 20}
KIND_TO_TYPE = {2: "F32", 4: "BF16", 5: "F16"}
EXPERT_TYPE = "Q4_0"          # the pack's own experts.bin encoding (native_experts.txt: gu_type 2 d_type 2)
PLE_TYPE = "IQ4_NL"
H, FF, N_EXPERT = 2560, 640, 512
# ONE EXPERT'S ROLE BLOCK - the stride the gather must use.  The pack is EXPERT-MAJOR (expert e: gate|up|down
# back to back), so a per-expert block is 921,600 B, NOT the layer's whole role tensor; the two constants are
# easy to confuse and confusing them silently writes a permuted layout (the first version of this file did).
PER_EXPERT_GU = FF * (H // 32) * 18                                  # 921,600 B: one expert's gate (also: its up)
PER_EXPERT_DOWN = H * (FF // 32) * 18                                # 921,600 B: one expert's down
assert PER_EXPERT_GU == PER_EXPERT_DOWN == 921_600, (PER_EXPERT_GU, PER_EXPERT_DOWN)
PER_EXPERT_ROLE = PER_EXPERT_GU                                      # all three roles use the same stride
EXPERT_BLOB = 2 * PER_EXPERT_ROLE + PER_EXPERT_DOWN                  # 2,764,800 B: gate|up|down per expert
LAYER_BYTES = N_EXPERT * EXPERT_BLOB                                 # 1,415,577,600 B
ROLE_BYTES = N_EXPERT * PER_EXPERT_ROLE                              # 471,859,200 B per role tensor (all three equal)
ALIGN = 32


def log(msg: str) -> None:
    print(msg, flush=True)


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def elem_bytes(type_name: str, elements: int) -> int | None:
    geom = BLOCK_GEOMETRY.get(type_name)
    if geom is None:
        return None
    be, bb = geom
    if elements % be:
        return None
    return elements // be * bb


# --------------------------------------------------------------------------- template header
def read_template(path):
    """Return (kv_entries, tensors, data_start) with each KV entry kept as its raw bytes."""
    raw = pathlib.Path(path).read_bytes() if path.stat().st_size < (64 << 20) else None
    with open(path, "rb") as fh:
        magic = fh.read(4)
        assert magic == b"GGUF", magic
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        assert version == 3, version
        kv = []
        for _ in range(n_kv):
            start = fh.tell()
            (klen,) = struct.unpack("<Q", fh.read(8))
            key = fh.read(klen).decode("utf-8")
            vtype = struct.unpack("<I", fh.read(4))[0]
            _skip_value(fh, vtype)               # walk it to find the end
            kv.append((key, vtype, fh.tell() - start, start))
        tensors = []
        for _ in range(n_tensors):
            (nlen,) = struct.unpack("<Q", fh.read(8))
            name = fh.read(nlen).decode("utf-8")
            (nd,) = struct.unpack("<I", fh.read(4))
            dims = list(struct.unpack(f"<{nd}Q", fh.read(8 * nd)))
            tid, off = struct.unpack("<IQ", fh.read(12))
            tensors.append((name, dims, tid, off))
        pos = fh.tell()
        data_start = (pos + ALIGN - 1) // ALIGN * ALIGN
    return kv, tensors, data_start


def _skip_value(fh, vtype):
    """Advance the file position past one GGUF metadata value."""
    if vtype == 8:                                  # string
        (n,) = struct.unpack("<Q", fh.read(8))
        fh.seek(n, 1)
    elif vtype == 9:                                # array
        et, n = struct.unpack("<IQ", fh.read(12))
        if et == 8:
            for _ in range(n):
                (l,) = struct.unpack("<Q", fh.read(8))
                fh.seek(l, 1)
        else:
            fh.seek(_ESIZE[et] * n, 1)
    else:
        fh.seek(_ESIZE[vtype], 1)


_ESIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def encode_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


# --------------------------------------------------------------------------- source plans
def read_pack_index(path):
    rows = {}
    for ln in pathlib.Path(path).read_text().splitlines():
        if not ln or ln.startswith("#"):
            continue
        f = ln.split()
        if len(f) != 19:
            raise SystemExit(f"index row with {len(f)} fields: {ln[:80]}")
        rows[f[0]] = {"file": int(f[1]), "kind": int(f[2]), "src_off": int(f[3]), "src_bytes": int(f[4]),
                      "dst_off": int(f[5]), "dst_bytes": int(f[6]), "ne0": int(f[7]), "ne1": int(f[8]),
                      "code_bits": int(f[9]), "code_bias": int(f[10]), "group": int(f[11])}
    return rows


def build_plan():
    kv, tmpl_tensors, tmpl_data_start = read_template(TEMPLATE)
    tmpl = {n: (d, t, o) for n, d, t, o in tmpl_tensors}
    native = GGUFFile(NATIVE_DENSE)
    nat = {t.name: t for t in native.tensors}
    idx = read_pack_index(SERVE_PACK / "index.txt")

    ple = GGUFFile(PLE_IQ4NL)
    ple_t = ple.tensors[0]
    ple_name, ple_dims, ple_off_native = ple_t.name, list(ple_t.shape), ple_t.offset
    # PLE tensor type in the ggml table used by the fork
    assert ple_t.type_name == PLE_TYPE, ple_t.type_name

    tensors = []           # (name, dims, type_name, source dict)
    problems = []

    # 1. PLE
    if ple_name in tmpl:
        problems.append(f"the template already carries {ple_name}")
    tensors.append((ple_name, ple_dims, PLE_TYPE, {"kind": "ple", "off": ple_off_native}))

    # 2/3. the rest
    for name, (dims, tid, off) in tmpl.items():
        if "_exps." in name:
            role = "gate" if "ffn_gate_exps" in name else ("up" if "ffn_up_exps" in name else "down")
            layer = int(name.split(".")[1])
            want = {"gate": [H, FF, N_EXPERT], "up": [H, FF, N_EXPERT], "down": [FF, H, N_EXPERT]}[role]
            if dims != want:
                problems.append(f"{name}: template dims {dims} != expected {want}")
            tensors.append((name, dims, EXPERT_TYPE, {"kind": "expert", "layer": layer, "role": role}))
            continue
        if name in nat:
            nt = nat[name]
            if list(nt.shape) != list(dims):
                problems.append(f"{name}: native dims {nt.shape} != template {dims}")
            eb = elem_bytes(nt.type_name, nt.elements)
            if eb is None:
                problems.append(f"{name}: native type {nt.type_name} has no ggml geometry")
            tensors.append((name, dims, nt.type_name,
                            {"kind": "native", "off": nt.offset, "bytes": eb, "src_type": nt.type_name}))
            continue
        r = idx.get(name)
        if r is None:
            problems.append(f"{name}: in the template but in NEITHER source")
            continue
        if r["src_bytes"] == 0:
            problems.append(f"{name}: pack index says src_bytes 0 but it is not in native-dense.gguf")
            continue
        tn = KIND_TO_TYPE.get(r["kind"])
        if tn is None:
            problems.append(f"{name}: pack index kind {r['kind']} is not a ggml type (only 2/4/5 are usable)")
            continue
        n_elem = r["ne0"] * max(r["ne1"], 1)
        if r["src_bytes"] != r["dst_bytes"]:
            problems.append(f"{name}: src_bytes {r['src_bytes']} != dst_bytes {r['dst_bytes']}")
        eb = elem_bytes(tn, n_elem)
        if eb != r["src_bytes"]:
            problems.append(f"{name}: {tn} of {n_elem} elements is {eb} B, index says {r['src_bytes']} B")
        if r["ne0"] != dims[0] or (len(dims) > 1 and max(r["ne1"], 1) != dims[1]):
            problems.append(f"{name}: index ne {r['ne0']}x{r['ne1']} != template dims {dims}")
        tensors.append((name, dims, tn, {"kind": "densebin", "off": r["dst_off"], "bytes": r["src_bytes"]}))

    # every index/native name must be consumed
    used = {t[0] for t in tensors}
    for n in idx:
        if n not in used:
            problems.append(f"{n}: in the pack index but not in the template (llama.cpp would reject it)")
    for n in nat:
        if n not in used:
            problems.append(f"{n}: in native-dense.gguf but not in the template")

    # the blob geometry this file gathers with, against the packer's own record
    check_source_geometry(problems)

    # ---- layout of the data section (all offsets relative to data_start)
    layout = []
    at = 0
    dense_names = [t for t in tensors if t[3]["kind"] == "densebin"]
    native_names = [t for t in tensors if t[3]["kind"] == "native"]
    expert_names = [t for t in tensors if t[3]["kind"] == "expert"]
    ple_names = [t for t in tensors if t[3]["kind"] == "ple"]
    log(f"  sources: dense.bin {len(dense_names)}, native-dense.gguf {len(native_names)}, "
        f"experts.bin {len(expert_names)}, ple-iq4nl {len(ple_names)}  (total {len(tensors)})")

    base_dense = align_up(at, ALIGN)
    at = base_dense + DENSE_BIN.stat().st_size       # the whole pool, verbatim: keeps the pack's own offsets
    base_native = align_up(at, ALIGN)
    native_bytes = NATIVE_DENSE.stat().st_size - native.data_start
    at = base_native + native_bytes
    base_expert = align_up(at, ALIGN)
    at = base_expert + LAYER_BYTES * 48
    base_ple = align_up(at, ALIGN)
    ple_bytes = PLE_IQ4NL.stat().st_size - ple.data_start
    at = base_ple + ple_bytes
    data_bytes = at

    # every offset the plan hands to the writer is RELATIVE TO data_start
    offsets = {}
    for name, dims, tn, src in tensors:
        if src["kind"] == "densebin":
            offsets[name] = base_dense + src["off"]
        elif src["kind"] == "native":
            # we copy native-dense's bytes [data_start, EOF) to base_native, so its own relative offset survives
            offsets[name] = base_native + src["off"]
        elif src["kind"] == "ple":
            offsets[name] = base_ple + src["off"]
        else:
            offsets[name] = base_expert + src["layer"] * LAYER_BYTES + \
                {"gate": 0, "up": ROLE_BYTES, "down": 2 * ROLE_BYTES}[src["role"]]
    bad = {n: o for n, o in offsets.items() if o % ALIGN}
    if bad:
        problems.append(f"unaligned offsets: {list(bad.items())[:3]}")

    meta = {
        "template": str(TEMPLATE), "template_data_start": tmpl_data_start,
        "native_dense": str(NATIVE_DENSE), "native_data_start": native.data_start,
        "native_region_bytes": native_bytes, "serve_pack": str(SERVE_PACK),
        "dense_bin_bytes": DENSE_BIN.stat().st_size, "ple": str(PLE_IQ4NL),
        "ple_region_bytes": PLE_IQ4NL.stat().st_size - ple.data_start,
        "data_bytes": data_bytes,
        "bases": {"dense": base_dense, "native": base_native, "expert": base_expert, "ple": base_ple,
                  "expert_layer_bytes": LAYER_BYTES, "expert_role_bytes": ROLE_BYTES},
        "types": {},
        "counts": {"densebin": len(dense_names), "native": len(native_names),
                   "expert": len(expert_names), "ple": len(ple_names)},
    }
    for _, _, tn, _ in tensors:
        meta["types"][tn] = meta["types"].get(tn, 0) + 1
    return kv, tensors, offsets, meta, problems, data_bytes


# --------------------------------------------------------------------------- writer
def write(out_path, kv, tensors, offsets, data_bytes, drop_keys):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # header
    keep = [(k, t, n, s) for (k, t, n, s) in kv if k not in drop_keys]
    tmpl_raw = TEMPLATE.read_bytes()
    hdr = bytearray()
    hdr += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keep) + 1)
    for k, t, n, s in keep:
        if k == "general.name":
            hdr += encode_string(k) + struct.pack("<I", 8) + encode_string(NEW_NAME)
        else:
            hdr += tmpl_raw[s:s + n]
    hdr += encode_string("general.i3.oracle") + struct.pack("<I", 8) + encode_string(ORACLE_NOTE)
    for name, dims, tn, src in tensors:
        hdr += struct.pack("<Q", len(name)) + name.encode()
        hdr += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        hdr += struct.pack("<IQ", GGML[tn], offsets[name])
    pad = (-len(hdr)) % ALIGN
    hdr += b"\0" * pad
    assert len(hdr) % ALIGN == 0
    data_start = len(hdr)
    log(f"  header {data_start} B ({len(tensors)} tensors, {len(keep)+1} kv), data {data_bytes} B "
        f"({data_bytes/2**30:.2f} GiB), file {data_start + data_bytes} B")

    t0 = time.perf_counter()
    with open(out_path, "wb", buffering=0) as fo:
        fo.write(hdr)
        # region 1: dense.bin verbatim
        _copy_file(DENSE_BIN, fo, 0, DENSE_BIN.stat().st_size, "dense.bin")
        # region 2: native-dense data region verbatim
        nd = GGUFFile(NATIVE_DENSE)
        _copy_file(NATIVE_DENSE, fo, nd.data_start, NATIVE_DENSE.stat().st_size - nd.data_start, "native-dense")
        # region 3: the experts, re-laid out role-major
        _write_experts(fo, EXPERTS_BIN)
        # region 4: the PLE data region verbatim
        pl = GGUFFile(PLE_IQ4NL)
        _copy_file(PLE_IQ4NL, fo, pl.data_start, PLE_IQ4NL.stat().st_size - pl.data_start, "ple-iq4nl")
        fo.flush()
        os.fsync(fo.fileno())
    log(f"  written in {time.perf_counter()-t0:.1f} s")


def _copy_file(src, fo, off, nbytes, tag):
    t0 = time.perf_counter()
    chunk = 1 << 26
    done = 0
    with open(src, "rb", buffering=0) as fi:
        fi.seek(off)
        while done < nbytes:
            b = fi.read(min(chunk, nbytes - done))
            if not b:
                raise SystemExit(f"{src}: short read at {done}")
            fo.write(b)
            done += len(b)
    log(f"  {tag:13s} {nbytes} B copied in {time.perf_counter()-t0:.1f} s")


def _write_experts(fo, src):
    """experts.bin is expert-major; emit role-major: for each layer, the 512 gate blocks, then up, then down."""
    t0 = time.perf_counter()
    with open(src, "rb", buffering=0) as fi:
        for layer in range(48):
            buf = fi.read(LAYER_BYTES)
            if len(buf) != LAYER_BYTES:
                raise SystemExit(f"experts.bin: short layer {layer}")
            for role in (0, 1, 2):
                out = bytearray(ROLE_BYTES)
                ro = role * PER_EXPERT_ROLE
                for e in range(N_EXPERT):
                    s = e * EXPERT_BLOB + ro
                    out[e * PER_EXPERT_ROLE:(e + 1) * PER_EXPERT_ROLE] = buf[s:s + PER_EXPERT_ROLE]
                fo.write(out)
    log(f"  experts       {48*3*ROLE_BYTES} B gathered + written in {time.perf_counter()-t0:.1f} s")


NEW_NAME = "Qwen3.8-Flash-Next W4A16 (Intel AutoRound) experts Q4_0 + Strata dense, I3 oracle"
ORACLE_NOTE = ("assembled by strata/i3/i3_assemble_oracle.py from the Strata W4A16 pack "
               "(serve-pack/dense.bin + native-dense.gguf + experts.bin) and ple-iq4nl.gguf, "
               "using the ISTA-DASLab IQ3_S shard 1's KV and tensor table for this architecture")


# --------------------------------------------------------------------------- verify
def verify(out_path, tensors, offsets, meta):
    g = GGUFFile(pathlib.Path(out_path))
    log(f"  re-read: {len(g.tensors)} tensors, {len(g.metadata)} kv, alignment {g.alignment}, "
        f"data_start {g.data_start}, size {pathlib.Path(out_path).stat().st_size}")
    plan = {n: (d, t) for n, d, t, _ in tensors}
    ok = True
    for t in g.tensors:
        if t.name not in plan:
            log(f"  FAIL extra tensor {t.name}")
            ok = False
            continue
        dims, tn = plan[t.name]
        if list(t.shape) != list(dims) or t.type_name != tn:
            log(f"  FAIL {t.name}: written {t.shape}/{t.type_name} vs plan {dims}/{tn}")
            ok = False
        if t.offset != offsets[t.name]:
            log(f"  FAIL {t.name}: offset {t.offset} vs plan {offsets[t.name]}")
            ok = False
    log(f"  tensor table: {'OK' if ok else 'MISMATCH'}")

    # digest spot checks against the independent sources
    checks = []
    by_kind = {}
    for name, dims, tn, src in tensors:
        by_kind.setdefault(src["kind"], []).append((name, dims, tn, src))
    sample = {
        "densebin": _sample(by_kind["densebin"], 6),
        "native": _sample(by_kind["native"], 6),
        "expert": _sample(by_kind["expert"], 8),
        "ple": _sample(by_kind["ple"], 1),
    }
    with open(out_path, "rb") as f:
        for kind, items in sample.items():
            for name, dims, tn, src in items:
                nbytes = elem_bytes(tn, _elems(dims))
                f.seek(g.data_start + offsets[name])
                got = hashlib.sha256(f.read(nbytes)).hexdigest()
                want = _source_digest(kind, name, dims, tn, src, nbytes)
                same = got == want
                ok = ok and same
                checks.append({"tensor": name, "kind": kind, "bytes": nbytes, "sha256_out": got,
                               "sha256_src": want, "match": same})
                log(f"  {'OK  ' if same else 'FAIL'} {name:45s} {nbytes:>12} B  {got[:16]}")
    return ok, checks


def _elems(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def _sample(items, k):
    if not items:
        return []
    step = max(1, len(items) // k)
    return items[::step][:k]


def _source_digest(kind, name, dims, tn, src, nbytes):
    if kind == "densebin":
        with open(DENSE_BIN, "rb") as f:
            f.seek(src["off"])
            return hashlib.sha256(f.read(nbytes)).hexdigest()
    if kind == "native":
        nd = GGUFFile(NATIVE_DENSE)
        with open(NATIVE_DENSE, "rb") as f:
            f.seek(nd.data_start + src["off"])
            return hashlib.sha256(f.read(nbytes)).hexdigest()
    if kind == "ple":
        pl = GGUFFile(PLE_IQ4NL)
        with open(PLE_IQ4NL, "rb") as f:
            f.seek(pl.data_start + src["off"])
            return hashlib.sha256(f.read(nbytes)).hexdigest()
    # expert: hash the GATHERED order out of experts.bin (expert-major -> role-major)
    layer, role = src["layer"], src["role"]
    ro = {"gate": 0, "up": 1, "down": 2}[role] * PER_EXPERT_ROLE
    h = hashlib.sha256()
    with open(EXPERTS_BIN, "rb") as f:
        base = layer * LAYER_BYTES
        for e in range(N_EXPERT):
            f.seek(base + e * EXPERT_BLOB + ro)
            h.update(f.read(PER_EXPERT_ROLE))
    return h.hexdigest()


# --------------------------------------------------------------------------- source geometry / slot sweep
def check_source_geometry(problems):
    """Cross-check this file's constants against the packer's OWN record (the manifest) and the spans file the
    ENGINE reads.  Without this the gather's stride is an assumption; with it, every constant that decides the
    output layout is anchored to a written artifact of the W4A16 pack."""
    man = json.loads(MANIFEST.read_text())
    g = man["geometry"]
    for name, want, got in (("format.ggml_type", man["format"]["ggml_type"], EXPERT_TYPE),
                            ("geometry.experts_per_layer", g["experts_per_layer"], N_EXPERT),
                            ("geometry.layers", g["layers"], 48),
                            ("geometry.blob_bytes", g["blob_bytes"], EXPERT_BLOB),
                            ("geometry.up_off", g["up_off"], PER_EXPERT_ROLE),
                            ("geometry.down_off", g["down_off"], 2 * PER_EXPERT_ROLE),
                            ("geometry.layer_bytes", g["layer_bytes"], LAYER_BYTES),
                            ("geometry.experts_bin_bytes", g["experts_bin_bytes"], EXPERTS_BIN.stat().st_size)):
        if want != got:
            problems.append(f"manifest {name}: the packer recorded {want}, this file assumes {got}")
    rows = []
    for ln in NATIVE_EXPERTS_TXT.read_text().splitlines():
        if not ln or ln.startswith("#"):
            continue
        f = ln.split()
        rows.append({"layer": int(f[0]), "gu_type": int(f[1]), "d_type": int(f[2]),
                     "offset": int(f[3]), "blob": int(f[4])})
    if len(rows) != 48:
        problems.append(f"native_experts.txt: {len(rows)} layer rows, expected 48")
    for r in rows:
        if r["gu_type"] != GGML[EXPERT_TYPE] or r["d_type"] != GGML[EXPERT_TYPE]:
            problems.append(f"native_experts.txt layer {r['layer']}: types "
                            f"{r['gu_type']}/{r['d_type']} != {GGML[EXPERT_TYPE]} ({EXPERT_TYPE})")
        if r["offset"] != r["layer"] * LAYER_BYTES or r["blob"] != EXPERT_BLOB:
            problems.append(f"native_experts.txt layer {r['layer']}: offset/blob "
                            f"{r['offset']}/{r['blob']} != {r['layer'] * LAYER_BYTES}/{EXPERT_BLOB}")
    log(f"  source geometry: manifest + native_experts.txt agree with this file (per-expert role block "
        f"{PER_EXPERT_ROLE} B x 3 = blob {EXPERT_BLOB} B, layer {LAYER_BYTES} B, "
        f"{EXPERTS_BIN.stat().st_size} B total)")
    return man


def verify_source_layers(man, layers=(0, 1, 12, 24, 36, 47)):
    """The pack's own layer digests, recomputed here: anchors the SOURCE (experts.bin) to W1's record."""
    res = []
    with open(EXPERTS_BIN, "rb") as f:
        for lay in layers:
            f.seek(lay * LAYER_BYTES)
            h = hashlib.sha256()
            left = LAYER_BYTES
            while left > 0:
                b = f.read(min(1 << 26, left))
                h.update(b)
                left -= len(b)
            got = h.hexdigest()
            want = man["layer_sha256"][str(lay)]
            res.append({"layer": lay, "sha256": got, "manifest": want, "match": got == want})
            log(f"  experts.bin layer {lay:2d} sha256 {got[:24]} vs W1 manifest "
                f"{'OK' if got == want else 'MISMATCH'}")
    return res


def verify_expert_slots(out_path, n_sample=0):
    """THE id -> slot mapping, read back off the written file: slot `e` of blk.L.<role>_exps must BE blob `e`'s
    role block of layer L in the pack.  Both sides are read with plain byte arithmetic - the written side from
    the offsets the file's OWN header carries (tools/gguf_reader, not this writer's plan), the source side from
    the manifest's geometry - so a permutation (which is invisible to a whole-tensor digest) fails here.
    n_sample=0 sweeps every one of the 48 x 3 x 512 slots."""
    g = GGUFFile(pathlib.Path(out_path))
    tens = {t.name: t for t in g.tensors}
    step = 1 if not n_sample else max(1, N_EXPERT // n_sample)
    bad, n, nbytes = [], 0, 0
    roles = ((0, "ffn_gate_exps"), (1, "ffn_up_exps"), (2, "ffn_down_exps"))
    t0 = time.perf_counter()
    with open(EXPERTS_BIN, "rb") as src, open(out_path, "rb") as fo:
        for lay in range(48):
            for role, suffix in roles:
                name = f"blk.{lay}.{suffix}.weight"
                t = tens.get(name)
                if t is None:
                    bad.append(f"{name}: absent from the written file")
                    continue
                want_shape = [FF, H, N_EXPERT] if role == 2 else [H, FF, N_EXPERT]
                if t.type_name != EXPERT_TYPE or list(t.shape) != want_shape:
                    bad.append(f"{name}: {t.type_name} {t.shape} != {EXPERT_TYPE} {want_shape}")
                for e in range(0, N_EXPERT, step):
                    fo.seek(g.data_start + t.offset + e * PER_EXPERT_ROLE)
                    got = hashlib.sha256(fo.read(PER_EXPERT_ROLE)).hexdigest()
                    src.seek(lay * LAYER_BYTES + e * EXPERT_BLOB + role * PER_EXPERT_ROLE)
                    want = hashlib.sha256(src.read(PER_EXPERT_ROLE)).hexdigest()
                    n += 1
                    nbytes += PER_EXPERT_ROLE
                    if got != want:
                        bad.append(f"{name} slot {e}: {got[:16]} != blob {e} {want[:16]}")
            if lay % 8 == 7 or lay == 47:
                log(f"  expert slots: layer {lay:2d} done  ({n} slot blocks checked, {len(bad)} bad, "
                    f"{nbytes / 2**30:.1f} GiB verified, {time.perf_counter()-t0:.0f} s)")
    log(f"  expert slots: {n} of {48 * 3 * N_EXPERT} (layer, expert, role) blocks checked, {len(bad)} bad "
        f"({nbytes * 2 / 2**30:.1f} GiB read back on both sides, {time.perf_counter()-t0:.0f} s)")
    for b in bad[:12]:
        log(f"    BAD {b}")
    return {"checked": n, "bad": len(bad), "first_bad": bad[:12], "gib_per_side": nbytes / 2**30,
            "seconds": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sweep-sample", type=int, default=0,
                    help="0 = every (layer, expert, role) slot block is read back; N = N experts per tensor")
    a = ap.parse_args()
    out = pathlib.Path(a.out)

    log(f"template {TEMPLATE.name}")
    kv, tensors, offsets, meta, problems, data_bytes = build_plan()
    log(f"  template kv {len(kv)}, tensors {len(tensors)}")
    if problems:
        log("  PROBLEMS:")
        for p in problems:
            log(f"    - {p}")
    else:
        log("  all checks passed (types, shapes, offsets, coverage)")

    if a.plan and not problems:
        log("  layout:")
        for k, v in meta["bases"].items():
            log(f"    {k}: {v}")
        log(f"  types: {meta['types']}")
        st = os.statvfs(out.parent if out.parent.exists() else SSD)
        free = st.f_bavail * st.f_frsize
        log(f"  target {out.parent}: free {free/2**30:.1f} GiB, need {(data_bytes + 12*2**20)/2**30:.1f} GiB")
        if free < data_bytes * 1.02:
            log("  NOT ENOUGH SPACE")
    if problems and not (a.write or a.verify):
        return 1

    if a.write:
        st = os.statvfs(out.parent if out.parent.exists() else SSD)
        free = st.f_bavail * st.f_frsize
        if free < data_bytes * 1.02:
            raise SystemExit(f"free {free} B < need {data_bytes} B")
        log(f"writing {out}")
        write(out, kv, tensors, offsets, data_bytes, drop_keys={"split.no", "split.count", "split.tensors.count"})
        log(f"  file size {out.stat().st_size} B ({out.stat().st_size/2**30:.3f} GiB)")
        json.dump(meta, open(R / "strata" / "i3" / "assembly-plan.json", "w"), indent=1)
        # offsets of the plan are relative to data_start - keep them for verify
        g = GGUFFile(out)
        meta["data_start_written"] = g.data_start
        json.dump(meta, open(R / "strata" / "i3" / "assembly-plan.json", "w"), indent=1)

    if a.verify:
        g = GGUFFile(out)
        meta["data_start_written"] = g.data_start
        ok, checks = verify(out, tensors, offsets, meta)
        man = json.loads(MANIFEST.read_text())
        src = verify_source_layers(man)
        slots = verify_expert_slots(out, n_sample=a.sweep_sample)
        ok = ok and all(c["match"] for c in src) and slots["bad"] == 0
        json.dump({"checks": checks, "source_layers": src, "expert_slots": slots, "ok": ok},
                  open(R / "strata" / "i3" / "assembly-verify.json", "w"), indent=1)
        log(f"  digest spot checks / source layers / expert slots: {'ALL MATCH' if ok else 'MISMATCH(es) above'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
