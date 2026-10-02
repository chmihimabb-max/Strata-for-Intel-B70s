#!/usr/bin/env python3
"""tools/test_w4a16_pack.py - the parity test for the W4A16 experts pack (card t_e8373d23, W4A16-PLAN.md §5).

The oracle is the checkpoint itself (rung 0): the packer only RE-ORDERS the checkpoint's own nibbles and copies
its own fp16 group scales, so the decode of the artifact must equal the decode of the source triple BIT FOR BIT.
The decoder used here is `tools/w4a16_reference.py`, which does not import the packer.

    python tools/test_w4a16_pack.py                      # the hermetic tests (no checkpoint, no artifact)
    W4A16_CKPT=<snap> W4A16_PACK=<dir> python tools/test_w4a16_pack.py -v    # + the real-artifact tests

The real-artifact tests are SKIPPED (not silently passed) when the environment does not name them: this project's
rule is that a test whose required input is missing is not run at all, never registered and quietly green.
"""
from __future__ import annotations

import os
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import w4a16_pack as P          # noqa: E402  (the encoder under test)
import w4a16_reference as R     # noqa: E402  (the independent decoder/oracle)


class TestGeometry(unittest.TestCase):
    """The plan's numbers, from both files, with the arithmetic spelled out."""

    def test_blob_arithmetic(self):
        self.assertEqual(P.GU_ROW, 2560 // 32 * 18, "a gate/up row is 80 Q4_0 blocks of 18 B")
        self.assertEqual(P.D_ROW, 640 // 32 * 18, "a down row is 20 Q4_0 blocks of 18 B")
        self.assertEqual(P.GU_ROW, 1440)
        self.assertEqual(P.D_ROW, 360)
        self.assertEqual(P.BLOB, 2 * 640 * 1440 + 2560 * 360)
        self.assertEqual(P.BLOB, 2_764_800)
        self.assertEqual(P.TOTAL_BYTES, 48 * 512 * P.BLOB)
        self.assertEqual(P.TOTAL_BYTES, 67_947_724_800)
        # the two files must agree on every number, or the oracle is not the oracle
        for a, b in (("H", "H"), ("FF", "FF"), ("N_LAYERS", "N_LAYERS"), ("N_EXPERTS", "N_EXPERTS"),
                     ("GROUP_ELEMS", "GROUP_ELEMS"), ("QK", "QK"), ("Q4_0_BYTES", "Q4_0_BYTES"),
                     ("GU_ROW", "GU_ROW"), ("D_ROW", "D_ROW"), ("UP_OFF", "UP_OFF"), ("DOWN_OFF", "DOWN_OFF"),
                     ("BLOB", "BLOB"), ("LAYER_BYTES", "LAYER_BYTES")):
            self.assertEqual(getattr(P, a), getattr(R, b), f"packer.{a} != reference.{b}")

    def test_zero_point_convention(self):
        """The stored nibble is 7 and the true zero point is 8 (the producer's `zeros -= 1` / `zeros += 1`)."""
        self.assertEqual(P.ZERO_WORD, 0x77777777)
        self.assertEqual(P.TRUE_ZERO_POINT, 8)
        stored = (P.ZERO_WORD >> (4 * (7 % 8))) & 0xF      # the nibble the reader extracts
        self.assertEqual(stored, 7)
        self.assertEqual(stored + 1, P.TRUE_ZERO_POINT)


def make_triple(out_dim: int, in_dim: int, seed: int, zeros: np.uint32 = np.uint32(P.ZERO_WORD)):
    """A synthetic AutoGPTQ triple of the checkpoint's shape: [in/8, out] int32, [in/128, out/8] int32, f16."""
    rng = np.random.default_rng(seed)
    qw = rng.integers(0, 2**32, size=(in_dim // 8, out_dim), dtype=np.uint32)
    qz = np.full((in_dim // P.GROUP_ELEMS, out_dim // 8), zeros, dtype=np.uint32)
    # signed fp16 scales, as in the checkpoint (6,418 of 12,800 sampled were negative)
    mag = rng.uniform(1e-4, 9e-3, size=(in_dim // P.GROUP_ELEMS, out_dim)).astype(np.float16)
    sign = rng.choice(np.float16([-1.0, 1.0]), size=mag.shape).astype(np.float16)
    return qw, qz, (mag * sign).astype(np.float16)


class TestRepackBitExact(unittest.TestCase):
    """Rung 0 without the checkpoint: random nibbles, the reference's decoder as the judge."""

    def _check(self, out_dim, in_dim, seed):
        qw, qz, sc = make_triple(out_dim, in_dim, seed)
        src = R.dequant_source(qw, qz, sc, out_dim, in_dim)                     # what the checkpoint means
        row_bytes = in_dim // P.QK * P.Q4_0_BYTES
        blob = P.repack_projection(qw, sc, out_dim, in_dim)                     # what the packer wrote
        self.assertEqual(len(blob), out_dim * row_bytes)
        got = np.stack([R.dequant_q4_0_row(blob[r * row_bytes:(r + 1) * row_bytes], in_dim)
                        for r in range(out_dim)])
        bad = int((src != got).sum())
        self.assertEqual(bad, 0, f"{out_dim}x{in_dim} seed {seed}: {bad} of {src.size} weights differ")
        self.assertEqual(float(np.abs(src.astype(np.float64) - got.astype(np.float64)).max()), 0.0)

    def test_gate_shape(self):
        for seed in range(4):
            self._check(640, 2560, 100 + seed)

    def test_down_shape(self):
        for seed in range(4):
            self._check(2560, 640, 200 + seed)

    def test_awkward_group_boundaries(self):
        # the zero-point plane packs 8 output columns per int32, so out_dim is a multiple of 8; the group is
        # four 32-blocks, so in_dim is a multiple of 128.  Exercise the smallest legal ones and a mix.
        for out_dim, in_dim in ((8, 128), (16, 128), (8, 256), (24, 640)):
            self._check(out_dim, in_dim, 300 + in_dim)

    def test_nibble_order_is_not_accidentally_symmetric(self):
        """Distinct nibbles per input: byte k of a block must hold input k (low) and input k+16 (high)."""
        in_dim, out_dim = 128, 1
        nib = np.array([i % 16 for i in range(in_dim)], np.uint32)
        words = np.zeros((in_dim // 8, out_dim), np.uint32)
        for i in range(in_dim):
            words[i >> 3, 0] |= nib[i] << np.uint32(4 * (i & 7))
        sc = np.array([[np.float16(0.5)]], np.float16)
        blob = P.repack_projection(words, sc, out_dim, in_dim)
        got = R.dequant_q4_0_row(blob, in_dim)
        want = ((nib.astype(np.int32) - 8) * 0.5).astype(np.float32)
        self.assertTrue(np.array_equal(got, want), f"got {got.tolist()[:8]} want {want.tolist()[:8]}")
        # block 0's 16 qs bytes: byte k = nib(k) | nib(k+16) << 4 = k | (k << 4) = 17k
        for k in range(16):
            self.assertEqual(blob[2 + k], (k | (k << 4)) & 0xFF,
                             f"byte {k} of block 0: {blob[2+k]:#04x}")
        # and the block order along the row: block b covers inputs 32b..32b+31
        self.assertEqual(list(got[32:40]), list(want[32:40]))
        self.assertEqual(float(np.frombuffer(blob[18:20], dtype="<f2")[0]), 0.5)

    def test_scale_is_the_checkpoints_own_bits(self):
        """Every 32-block of a 128-group carries the group scale's fp16 bits, not a re-quantised scale."""
        in_dim, out_dim = 128, 4
        rng = np.random.default_rng(7)
        qw = rng.integers(0, 2**32, size=(in_dim // 8, out_dim), dtype=np.uint32)
        sc = rng.uniform(-9e-3, 9e-3, size=(1, out_dim)).astype(np.float16)
        blob = P.repack_projection(qw, sc, out_dim, in_dim)
        for o in range(out_dim):
            row = blob[o * (in_dim // 32) * 18: (o + 1) * (in_dim // 32) * 18]
            bits = [int(np.frombuffer(row[b * 18:b * 18 + 2], dtype="<u2")[0]) for b in range(in_dim // 32)]
            want = int(np.frombuffer(sc[0, o].tobytes(), dtype="<u2")[0])
            self.assertEqual(bits, [want] * 4, f"row {o}: the four blocks' d bits are the group scale's")


class TestRepackRefusals(unittest.TestCase):
    """The packer must refuse to guess, not skip."""

    class FakeSource:
        """The real three projections' shapes, from synthetic planes (so repack_expert runs for real)."""

        def __init__(self, poison_zero_point: bool):
            self.d = {}
            for role, out_dim, in_dim in P.ROLES:
                b = P.BASE % (0, 0, role)
                self.d[b + ".qweight"] = np.zeros((in_dim // 8, out_dim), np.uint32)
                self.d[b + ".scales"] = np.full((in_dim // P.GROUP_ELEMS, out_dim), np.float16(1e-3), np.float16)
                z = np.full((in_dim // P.GROUP_ELEMS, out_dim // 8), np.uint32(P.ZERO_WORD), np.uint32)
                if poison_zero_point:
                    z[0, 0] = np.uint32(0x88888888)
                self.d[b + ".qzeros"] = z

        def of(self, name):
            return self

        def arr(self, name, dtype, ndim):
            return self.d[name].view(dtype)

    def test_a_non_constant_zero_point_is_refused(self):
        blob, _st = P.repack_expert(self.FakeSource(False), 0, 0)
        self.assertEqual(len(blob), P.BLOB)
        with self.assertRaises(ValueError) as cm:
            P.repack_expert(self.FakeSource(True), 0, 0)
        self.assertIn("0x77777777", str(cm.exception))
        self.assertIn("refused", str(cm.exception))

    def test_a_missing_tensor_is_named(self):
        src = self.FakeSource(False)
        del src.d[P.BASE % (0, 0, "up_proj") + ".scales"]
        with self.assertRaises(KeyError) as cm:
            P.repack_expert(src, 0, 0)
        self.assertIn("up_proj.scales", str(cm.exception))


@unittest.skipUnless(os.environ.get("W4A16_CKPT") and os.environ.get("W4A16_PACK"),
                     "set W4A16_CKPT=<snapshot> and W4A16_PACK=<pack dir> for the real-artifact tests")
class TestRealArtifact(unittest.TestCase):
    """Rungs 0/1/2 against the real checkpoint and the real pack (the card's acceptance)."""

    @classmethod
    def setUpClass(cls):
        cls.ck = R.Ckpt(pathlib.Path(os.environ["W4A16_CKPT"]))
        pack_dir = pathlib.Path(os.environ["W4A16_PACK"])
        if not (pack_dir / "experts.bin").exists():
            raise unittest.SkipTest(f"{pack_dir}/experts.bin does not exist")
        cls.pack_dir = pack_dir
        cls.pack = R.Pack(pack_dir)

    def test_blob_size_and_completion_marker(self):
        binp = self.pack.bin
        self.assertEqual(binp.stat().st_size, R.LAYER_BYTES * R.N_LAYERS,
                         "FileExpertSource refuses any experts.bin that is not exactly 48x512x2,764,800 B")
        lay = self.pack_dir / "native_experts.txt"
        self.assertTrue(lay.exists(), "the pack has no native_experts.txt: it is not finished")
        lines = [l for l in lay.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(len(lines), R.N_LAYERS)
        for i, line in enumerate(lines):
            f = line.split()
            self.assertEqual(f[0], str(i), "the layers must be in order")
            self.assertEqual(f[1:3], ["2", "2"], "gu_type and d_type are ggml Q4_0 (2)")
            self.assertEqual(int(f[4]), R.BLOB)
            self.assertEqual(int(f[3]), i * R.LAYER_BYTES, "the layer offsets must be contiguous")

    def test_rung0_sample_every_layer(self):
        tot_el = tot_bad = 0
        worst = 0.0
        for l in range(R.N_LAYERS):
            for e in (0, R.N_EXPERTS // 2, R.N_EXPERTS - 1):
                r = R.compare_expert(self.ck, self.pack, l, e)
                tot_el += r["elements"]
                tot_bad += r["mismatches"]
                worst = max(worst, r["max_abs_err"])
                if r["mismatches"]:
                    print(f"  MISMATCH layer {l} expert {e}: {r['mismatches']} of {r['elements']} "
                          f"(max abs {r['max_abs_err']:.3e})")
        print(f"  rung 0 (every layer, 3 experts each): {tot_el:,d} weights compared, {tot_bad} mismatched, "
              f"max abs error {worst:.3e}")
        self.assertEqual(tot_bad, 0, "the repack is not value-preserving (W4A16-PLAN.md risk R1)")

    def test_rung1_hand_computed_spot_check(self):
        """Print the arithmetic by hand so the CONVENTION is pinned by inspection, not by error magnitude."""
        R.row_check(self.ck, self.pack, *self._spot())

    def test_fast_decode_matches_the_literal_one(self):
        """The sweep's vectorized decoder must equal the literal per-row decoder, bit for bit, every layer."""
        for l in range(R.N_LAYERS):
            lit = R.dequant_blob_expert(self.pack.blob(l, 0))
            fast = R.dequant_blob_expert_fast(self.pack.blob(l, 0))
            for a, b in zip(lit, fast):
                self.assertTrue(np.array_equal(a, b), f"layer {l}: the fast decoder differs from the literal one")

    def test_rung2_dot_oracle_agrees(self):
        l, e = self._spot()
        src = [R.dequant_source(*self.ck.triple(l, e, role), out_dim, in_dim)
               for role, out_dim, in_dim in R.ROLES]
        packed = R.dequant_blob_expert(self.pack.blob(l, e))
        rng = np.random.default_rng(20261002)
        x = rng.standard_normal(R.H, dtype=np.float32)
        a = R.swiglu(x, *src)
        b = R.swiglu(x, *packed)
        e_rel = R.rel(b, a)
        print(f"  rung 2 (layer {l} expert {e}): the source decode and the packed decode give the same expert "
              f"output; relative L1 error {e_rel:.3e} (tolerance 1e-5)")
        self.assertLess(e_rel, 1e-5)

    @staticmethod
    def _spot():
        return 16, 330          # the plan's own sample: layer 16, expert 330


if __name__ == "__main__":
    unittest.main(verbosity=2)
