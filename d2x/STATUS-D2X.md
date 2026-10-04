# D2x (board card t_85e61269, titled "D2c") — an int8-XMX MMVQ priced against the shipped scalar MMVQ

The card asked what the device's matrix unit could buy the decode path's **dense native projection MMVQ**
(`native_mmvq` → `launch_multi_n<Traits, 4>`, `src/kernels/cuda/native_mmvq.cu:1046`), which D2's corrected
census puts at **301 launches / 51.27 ms / 40.9% of a T=4 decode window**.  It named the shape to try:
"int8 XMX (`joint_matrix` int8 8x32x16 → int32)".  This is a measurement: **no engine source and no config
changed**, nothing in this card's tree is read by the engine.

Everything below is `d2x/D2X-BENCH.txt` (the raw run), `d2x/D2X-TABLE.txt` (the tables), the bench
`bench/micro/mmvq_xmx_price.cpp` (`build-sycl/mmvq_xmx_price`, md5 `42de31996d62c7e250a8e275f82866de`) and
the two host-side readings `d2x/bytes_floor.py` / `d2x/xmx_price.py`.  One B70, `ZE_AFFINITY_MASK=0`, both
cards free (no engine, no server, no container, load 0.8; `d2x/machine.txt`).

## 0. The answer, in one paragraph

**The shipped MMVQ is not at its bandwidth floor — it is 11.36x above it, at 8.8% of the card's measured
read rate — but the matrix unit is not the lever that closes that gap, and on this card the lever the card
named does not exist.**  Re-measured at HEAD: the driver **refuses the int8 8x32x16 / 16x32x16 dot outright**
(`joint_matrix with parameters matrix_type::sint32, use::accumulator, Rows=8, Cols=32 is not supported on
this device`), and the **f16/bf16 16x16x16 tiles that probe21 recorded as RUNNING no longer build at all** on
this toolchain (the backend has no implementation for the load the compiler emits).  The only tile that
builds, **f16 8x16x16**, was written as a real MMVQ prototype and priced: **2.94x faster on the window's 301
calls at ncols = 4** (52.42 → 17.84 ms) and 3.61x at ncols = 6, up to 21x on the worst shipped shape — but it
is **capped at 4.23x** by the 2 bytes/element it must read (its own floor is 12.41 ms against the pack's
4.62 ms), while **the same speedup on the pack's own bytes would buy 11.36x**.  So: a clean, useful
measurement, and the interesting statement is about the shipped kernel's 8.8%, not about XMX.

## 1. The device fact, re-measured here (not inherited)

probe21's table lives in a header comment (`include/strata/sycl_compat/sycl_xmx.hpp:7-24`).  The bench
re-runs every shape it names, at HEAD, on driver `1.15.37833+4`:

| tile | A / B layout | probe21's record | this run |
|---|---|---|---|
| f16 16x16x16 | row_major / row_major | RUNS | **DOES NOT BUILD** — `undefined reference to __builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_SG16_16x16_i16_8_global_v8i8_pi32_i32` |
| f16 16x16x16 | col_major / row_major | — | DOES NOT BUILD (the same builtin, `ColumnMajor`) |
| f16 16x16x16 | intel_packed / intel_packed | — | DOES NOT BUILD (the same builtin, unpacked name) |
| bf16 16x16x16 | row_major / row_major | RUNS | **DOES NOT BUILD** (same) |
| **f16 8x16x16** | row_major / row_major | RUNS | **RUNS** |
| f16 8x16x16 | col_major / row_major | — | RUNS |
| bf16 8x16x16 | row_major / row_major | — | RUNS |
| int8 8x32x16 | row_major / row_major, col_major, intel_packed | REFUSED | **REFUSED**, identical text |
| int8 16x32x16 | row_major / row_major | REFUSED | **REFUSED**, identical text |

The int8 refusal is a device verdict (`matrix_type::sint32, use::accumulator`); the 16x16x16 failures are a
toolchain verdict (the device never sees them — the fat binary does not link).  Two consequences worth
recording: **the card's premise ("this device HAS int8 XMX") is false for its own dot product**, and
`sycl_xmx.hpp`'s gate now takes its `REFUSED` branch for the engine's own f16 16x16x16 probe, i.e. the
engine's XMX gate is closed on this toolchain.  Neither is a regression this card caused; both are measured.

## 2. The bench, and the check that it is a proxy for the window

`bench/micro/mmvq_xmx_price.cpp` + target `mmvq_xmx_price` (CMakeLists, registered for SYCL **and** CUDA —
the two existing mmvq harnesses sit inside an `if(NOT STRATA_ENABLE_SYCL)` block and are therefore absent
from this build).

* **Data**: synthetic-but-valid — random bytes with finite fp16 scales, the parity harness's own generator
  (`src/kernels/mmvq_multi_spread.cpp:75-78`), passed through the tree's host dequantizers
  (`include/strata/artifact/dequant.hpp`), so every type's real block layout and dequant arithmetic are in
  play.  No model is loaded, no pack is touched.
* **Geometry**: all 25 distinct (type, n_in, n_out) of the 301 dense matrices (`d2/D2-TYPES.txt`, cross-
  checked against the pack's GGUF with `tools/gguf_reader.py` — the 10 dispatched classes + the head; the
  counts sum to the census's own per-type 129/35/47/42/47/1), at ncols 4 and 6.
* **Price per call**: `reps` launches submitted **back to back** between two events ÷ reps (the in-stream
  device cost, which is what the engine's own census measures), plus a one-call sync'd price for reference.
* **The check**: this table's shipped-layout column × the census's launch counts = **52 418.4 µs against the
  census's 51 267.3 µs = +2.2%**, and per type **0.86-1.14x** (Q6_K 1.01, IQ4_NL 1.01, IQ4_XS 1.04, Q4_K
  0.97, Q5_K 1.14, Q8_0 0.86).  The microbench IS the window's dense family, priced 301 times.

## 3. The floor: this card's read rate, and a trap in measuring it

A GEMV reads every weight byte once, so `weight_bytes / rate` bounds any MMVQ on any unit.  Measured here
(event- and wall-clock, agreeing to 0.2% at 1 GiB):

| what | rate |
|---|---|
| 1 GiB read-only, **memset-filled** buffer | 1352.2 GB/s |
| 1 GiB read-only, **hash-filled (incompressible)** buffer | **571.6 GB/s** |
| 1 GiB write-only memset (constant) | 1390 GB/s |
| 256 MiB device→device copy ×3 | 2724 GB/s of read+write |

The memset-filled read is **2.37x** the incompressible one: this driver **compresses uniform pages**, so a
constant-filled buffer is not a bandwidth measurement (and 1352 GB/s is not a number this class of card has).
The floor uses the hash-filled rate, which is what quantized weights are: **571.6 GB/s**.  Floors for one
window's 301 calls, by representation:

| representation | bytes | x pack | floor | vs the shipped layout |
|---|---|---|---|---|
| the pack's blocks (shipped) | 2 638 423 040 | 1.00 | **4.62 ms** | **11.36x** |
| int8 (1 B/element) — the shape this card named | 3 545 497 600 | 1.34 | 6.20 ms | 8.45x |
| f16 / bf16 (2 B/element) — the shape that builds | 7 090 995 200 | 2.69 | 12.41 ms | 4.23x |
| f32 (4 B/element) | 14 181 990 400 | 5.38 | 24.81 ms | 2.11x |

**The shipped family runs at 50.3 GB/s = 8.8% of this card's measured read rate.**  Nothing about that is a
property of the matrix unit.

## 4. The measurement table (ncols = 4; the full per-case table is `d2x/D2X-TABLE.txt`)

| variant | window (301 calls) | ms | % of the window | vs shipped | vs its own floor |
|---|---|---|---|---|---|
| shipped layout (`multi_exact`, the default) | 52 418.4 µs | 52.42 | 41.8% | 1.00x | **11.36x** |
| llama.cpp's generic layout (`multi_exact(false)`) | 47 223.7 µs | 47.22 | 37.7% | 1.11x | 10.23x |
| **f16 8x16x16 XMX prototype (as written)** | **17 841.6 µs** | **17.84** | **14.2%** | **2.94x** | 1.44x (69% of its floor) |
| the pack's own floor | 4 616.7 µs | 4.62 | 3.7% | 11.36x | 1.00x |
| the f16 prototype's own floor | 12 405.5 µs | 12.41 | 9.9% | 4.23x | 1.00x |
| *census, for the check* | *51 267.3 µs* | *51.27* | *40.9%* | | |

Per shape, the prototype's price against the shipped one (`f16/ex`) runs from **0.05** (IQ4_NL 640x2560:
6.5 µs against 134.0 µs, 20.5x) to **1.06** (Q5_K and Q4_K 6144x2560, the only two shapes where it is
marginally slower).  The shipped kernel's own achieved bandwidth ranges from **6.9 GB/s** (IQ4_NL 640x2560)
to **95.6 GB/s** (Q5_K 6144x2560) — a 14x spread across the same 301 launches, i.e. the family's cost is
shape-dominated, and the prototype's **131.7-565.7 GB/s** removes exactly that spread.

At ncols = 6 the same table reads shipped 65 124.7 µs / generic 59 645.4 / prototype 18 034.4 = **3.61x**.

## 5. What the prototype is, exactly, and what it costs in accuracy

Contract (also in the bench's header): weights **dequantized once and repacked into 8-row × 16-k f16 tiles**
(the tiles the unit consumes are not the pack's blocks — that repack is part of the price), activations f16,
`8x16x16` tiles into an fp32 accumulator, k split across 4 sub-groups and reduced in work-group local
memory, four 8-row tiles (32 output rows) per work-group so the activation tile is loaded once per k-step.
At ncols = 4, 12 of the 16 tile columns compute nothing a caller wants (the unit forces N = 16) — that is
arithmetic waste, not invented bytes.

Accuracy, against an fp64 dot of the same dequantized weights on the same activations, over three separated
row windows (0, 5120, 10176) of Q6_K 2560x10240, 192 × 4 outputs (identical in both runs):

| | max abs | rel_rms |
|---|---|---|
| (1) the f16 encoding alone, `f16(w)·x` vs `w·x` | 5.500e-01 | **2.081e-04** |
| (2) shipped kernel vs ref (Q8_1 activations — the shipped contract) | 1.579e+01 | 5.650e-03 |
| (3) f16 XMX kernel vs ref (f16 activations) | 1.073e+00 | 2.951e-04 |
| (4) f16 XMX kernel vs `f16(w)·f16(x)` — **its own arithmetic** | 5.166e-04 | **1.158e-07** |
| (5) f16 XMX kernel vs the shipped kernel | 1.551e+01 | 5.670e-03 |

Row (4) is the kernel's own error and it is fp32 accumulation noise (1.2e-07).  Rows (1)/(3) are the f16
representation's error (2.1-3.0e-04 rel_rms), i.e. ~19x smaller than the shipped path's Q8_1 activation
error (5.7e-03) on this input.  So the prototype is *more* accurate than the shipped kernel here; the f16
encoding is not the accuracy problem.

The price of that representation, which the speedup does not include: **+4 450 MB of weights in VRAM**
(7 090 995 200 against 2 638 423 040) and a one-time **dequantize + repack of 3 743 ms on this host** (the
bench's own per-case `dq ms` column, summed over the 301 matrices).

## 6. The ceiling, and the shape a variant would have to reach

* On this card, an int8 MMVQ — the one the card named — **cannot be built** (§1).  If it could, its floor of
  6.20 ms would cap it at **8.45x** against the shipped layout.
* The f16 MMA/MMVQ that *can* be built is capped at **4.23x** by its own 2 bytes/element; the prototype as
  written measures **2.94x** (69% of that cap).
* **The bar any variant must clear to beat the shipped layout** is its own representation's byte ratio times
  the shipped kernel's achieved rate **on the same shape**, and there the shipped kernel is only at
  **8.8% of the card's 571.6 GB/s** (50.3 GB/s on the window average; 6.9-95.6 GB/s per shape).  Per type the
  byte ratios are Q6_K 2.44x, Q5_K 2.91x, Q4_K 3.56x, IQ4_XS 3.76x, IQ4_NL 3.56x, Q8_0 1.88x (window
  average 2.69x, i.e. **135 GB/s** in one number).  **23 of the 25 geometries clear their own bar** (the
  prototype's rate against it ranges 1.12x to 20.5x); the two that do not are exactly the two whose
  `f16/ex` is above 1.00 — Q5_K 6144x2560 at 261.4 GB/s against a 278.1 GB/s bar and Q4_K 6144x2560 at
  261.7 against 277.0 (0.94x each).
* **The larger lever is not the matrix unit**: running *any* correct kernel at the card's measured rate over
  the pack's own bytes would take the family from 52.42 ms to 4.62 ms (**11.36x**, 40.9% of the window →
  3.7%).  The shipped kernel is 11.36x away from that, and this prototype demonstrates that a kernel of this
  class reaches 88-99% of the card's rate — so the 11.36x is an achievable-looking target, while the matrix
  unit's own 4.23x cap is not the interesting one.

## 7. Not validated, and what would settle it

1. **No engine integration was attempted.**  The prototype is a microbench; the numbers above are its price
   on the same shapes, not an end-to-end decode result.  An engine arm would also have to price the
   3.7 s load-time repack and the +4.45 GB of VRAM (which costs expert-cache slots on this box).
2. **The generic layout's −9.9% here does not contradict D2's refusal of it.**  This rig is
   deterministic-but-synthetic and measures one call at a time; D2 measured it *in the engine* at 4K with a
   60% run-to-run spread and three different greedy id streams (card t_8306429a).  In-stream stability in
   this rig is not stability in a window.  Nothing here proposes flipping that switch.
3. **The prototype covers `n_out % 32 == 0` shapes only** (every dense shape here is), at ncols ≤ 6, on
   synthetic values; its correctness is checked on Q6_K 2560x10240 only (192 × 4 outputs over three row
   windows), not on all 25 geometries and not against the llama.cpp oracle.
4. **The 16x16x16 build failure is not diagnosed.**  Which toolchain/backend change removed the
   `PackedA` load builtin is not established here, and `sycl_xmx.hpp`'s recorded table is now stale
   (worth a one-line correction there, not this card's business).
5. **The per-call price carries one method note**: the in-stream (batched) number is used everywhere,
   because the one-call sync'd price is 5-15% higher on the small shapes (the `ex 1call` column of the raw
   table).  The shipped row's agreement with the census is +2.2% in the measured run and +1.2% in an
   identical re-run of the same code, so this rig's run-to-run spread on that row is ~1% and the +2.2% is
   the rig, not the agent's arithmetic.
6. **No ctest run**: no engine source, no test and no config changed (only CMakeLists gained a target).
7. Nothing pushed (origin has no `sycl-xpu` branch), as in every sibling card.

## 8. State this card left the machine in

No engine, no `serve.server`, no container; both B70s free; the bench frees what it allocates and exits `0`
(the measured run took 11.7 s wall).  Raw output, tables, scripts and the binary are under `d2x/`; the
copies the board carries are in `~/.hermes/kanban/attachments/t_85e61269/`.
