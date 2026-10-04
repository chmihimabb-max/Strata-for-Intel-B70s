# D2y (board card t_ba006576) — where the dense MMVQ's 8.8% actually goes

The card asked for a kernel that "reads the pack's own bytes at the card's rate", on the evidence that the shipped
dense MMVQ is 11.36x above `weight_bytes / 571.6 GB/s` on the bytes it already reads.  This is that card's answer,
measured: **the bytes are not the problem.**  The same bytes, on the same grid, with the dot removed, take 2.9 ms
of a 301-call window that the shipped kernel spends 48.3-50.5 ms on, and on the one DRAM-sized matrix in the window
(497 MiB) the no-dot read runs at **589.4 GB/s against this card's independently measured 571.5 GB/s** while the
shipped kernel runs at 59.1 GB/s on it — **10.0x**, in both runs.  What the 8.8% is made of is the arithmetic on
those bytes: an int8 dot product that has no instruction on this device (`__dp4a` is four multiplies, `sycl_compat/
intrinsics.hpp:64-80`), 1.1-7.3 bytes of activation loads per weight byte, and a block-level reduction whose
`__shared__` partial arrays move **0.76x the weight bytes** through shared memory.  Every one of those costs is
*per output element*, which is why the shape-specialised variant the card asked for ("more rows per block") is
**monotonically worse** (r4 0.70-0.74x, r8 0.50-0.54x, r16 0.33-0.34x of the shipped layout) rather than better, and
why the only lever that moves this family is one that changes the arithmetic — D2x's f16 8x16x16 prototype, 2.93-3.04x
on the same calls, capped at 4.21x by its 2 bytes/element.

Everything below is `d2y/D2Y-BENCH.txt` and `d2y/D2Y-BENCH-2.txt` (two runs of the same binary,
`build-sycl/mmvq_xmx_price` md5 `159c3b1878590839de95f7938e14988f`, 23:44 and 23:48), `d2y/D2Y-BASE.txt` (the same
bench before this card's edit, T = 2/4/6, md5 `42de31996d62c7e250a8e275f82866de`), the tables the extended reader
prints (`d2y/D2Y-TABLE.txt`, `d2y/D2Y-TABLE-2.txt`, `d2y/D2Y-MARGINAL.txt`), and the host-side geometry arithmetic
`d2y/D2Y-ATTRIB.txt` (`d2y/attribute.py`).  One B70, `ZE_AFFINITY_MASK=0`; both B70s were free (`d2y/machine.txt`).

## 0. The answer, in one paragraph

The shipped dense MMVQ is at 52.3-54.7 GB/s on this card's 571.5 GB/s, and **it is not reading-limited**: its own
grid, reading its own bytes with the dot, the activations and the reduction removed, runs 10.0x faster on the one
matrix large enough to be a DRAM measurement (589.4 GB/s against the 571.5 GB/s the same bench measures for a plain
1 GiB incompressible read, on a 497 MiB case the shipped kernel spends 8.83 ms on and the bare read spends
0.885 ms), and 16.4-17.2x faster across the whole window (where the smaller cases are re-read from L2 by both
kernels).  So the 11.3x is inside the kernel's own instruction stream, and the card's target — "the pack's own bytes
at the card's rate" — is not a memory problem to solve: a kernel that is *only* the pack's bytes already reaches the
card's rate.  Three measured facts put the time where it actually is.  (i) The dot: `STRATA_DP4A` on this device is
emulated with four multiplies plus shifts and sign fixups per 4-byte lane (`include/strata/sycl_compat/
intrinsics.hpp:64-80`, "There is NO hardware int8 dot outside XMX/DPAS on this device"), and `q6_q8_dot_impl` issues
8 of them per 32 elements — money spent per element, per column, and it cannot be paid off by reading bytes faster.
(ii) The activation loads: 1.1-7.3 bytes of activation per weight byte, per column (`d2y/D2Y-ATTRIB.txt`), which is
why `ncols` costs 8.6 ms of window time per extra column (`d2y/D2Y-MARGINAL.txt`) — 62% of the family at ncols = 4
scales with the column count, and even the part that does not (the weight read at "zero columns", 146 GB/s) is 3.9x
above its own floor.  (iii) The reduction: the window's `__shared__` partial arrays total 2.00 GB against 2.64 GB of
weights.  Increasing the rows one work-group owns multiplies (ii) and (iii) without changing (i)'s per-byte cost, and
that is exactly what the measurement shows.  The honest summary: **the 11.3x is not reachable by any variant of this
kernel; it needs a different arithmetic (the f16 XMX prototype, +4.45 GB of VRAM and 3.7 s of repack, 2.93-3.04x,
capped at 4.21x) or a smaller representation of the weights — not a faster read of the same ones.**

## 1. The launch geometry of every layout the shipped code can take (step 1, arithmetic)

`d2y/attribute.py` computes this from source constants only (the traits at `native_mmvq.cu:793-994`, the shape rule
at `native_mmvq.cu:1059-1064`, the generic table at `native_mmvq.cu:1052-1056`), for all 25 geometries and the
window's 301 launches.  The numbers that decide between "not enough parallelism" and "too little work per block":

| shape class | launches | work-groups | threads with any work | kbx iterations per working thread | weight bytes per working thread | threads per weight byte |
|---|---|---|---|---|---|---|
| the 18 geometries on the ROWS = 1 path | 211 | 1 per row (512-248320) | 128/128 | 2.7 | **11.2-39.4** | 0.025-0.089 |
| the 7 on the ROWS = WARPS path (IQ4_XS, IQ4_NL, Q8_0: `n_in/DIV < BPI`) | 90 | 128-3072 | **40-80/128** | 1.0 | 18.0-68.0 | 0.024-0.089 |

and the two numbers that are not about parallelism at all: the window allocates **2 004 615 168 B of `__shared__`
partial arrays (0.76x the 2 638 423 040 B of weights)** and issues **1.1-7.3 bytes of activation load per byte of
weight** (`d2y/D2Y-ATTRIB.txt`, "the shipped rule, per geometry").

Read as a whole: a working thread in this kernel reads **16.4 bytes of weights in the entire call** (Q6_K/Q5_K/Q4_K
2560x10240: 2.5 kbx iterations of a 210-176-144 B block, 128 B/thread × 128 threads = 2.1-1.8 KB per work-group).
No kernel with 16 bytes per thread is bandwidth-limited, whatever it achieves.  The 7 small-`n_in` geometries are
worse: `IQ4_NL 640x2560` has **40 of 128 threads** with any work at all (`T = Qi/2 = 2` and 20 blocks per row), one
iteration each, and is the shape that runs at 6.9 GB/s.

The failed hypothesis is worth recording, because the card proposed it: making each work-group own **more** rows
(the "more rows per block" variant) is strictly worse, monotonically, on every geometry and in both runs.  The
`__shared__` partial array and the reduction grow with the rows while the work-groups shrink by the same factor, and
the per-output cost is what dominates — so the knob's r8/r16 columns buy nothing anywhere.  The card's other
suggestion ("a split-k scheme for the small shapes") is the *right direction* for `IQ4_NL 640x2560` — its 1280-block
ROWS = 2 form is 1.54-1.55x faster than the shipped 640-block ROWS = 4 form — and it is what the knob's `r2` column
measures; see section 4.

## 2. The measurement that decides it: the same bytes, no dot

The bench gained a section 6 (`--variants`) with, among the variants, two kernels that keep the shipped layout's grid
and row-to-group mapping — one work-group per ROWS rows exactly as the shipped rule picks them — and read every byte
of those rows once, with **no dot, no activation, no reduction**: `flat4` at 4 B per thread per pass and `flat16` at
16 B.  Both write one float per thread to a sink (an unconditional store, because a load the compiler can drop is how
a "read" kernel reports a rate the card does not have).

| case | weight B | shipped us | shipped GB/s | flat4 us | flat4 GB/s | shipped/flat4 |
|---|---|---|---|---|---|---|
| Q6_K 2560x248320-head (497 MiB, the only DRAM-sized one) | 521 472 000 | 8831.9 / 8829.1 | 59.0 / 59.1 | 884.7 / 884.7 | **589.5 / 589.4** | 10.0 / 10.0 |
| IQ4_NL 640x2560 (the pathological one, 47 launches) | 921 600 | 133.6 / 134.1 | 6.9 / 6.9 | 2.4 / 2.5 | 382 / 370 | **55.7 / 53.6** |
| Q6_K 2560x10240 | 21 504 000 | 412.6 / 311.3 | 52.1 / 69.1 | 21.7 / 21.7 | 989 / 990 | 19.0 / 14.3 |
| window (301 calls) | 2 638 423 040 | 50 493.6 / 48 250.8 | 52.3 / 54.7 | 2934.4 / 2935.0 | 899 / 899 | **17.2 / 16.4** |

Two things have to be read correctly here.  First, the DRAM-sized case: 589.4 GB/s is the same bench's section-2
read rate (571.5 GB/s over 1 GiB, incompressible) to +3%, measured on the *shipped kernel's grid*, with the shipped
kernel's own row-to-group mapping, reading every byte of every row exactly once — so the pack's own bytes can be had
at the card's rate, and the card's premise ("price a kernel that reads the pack's own bytes at the card's rate") has a
measured price: **884.7 us for the window's largest call against the shipped 8831.9 us**.  Second, the window row:
every case except the head is 0.7-26 MB and the rig re-reads it 20 times back to back, so `flat4` there is an L2 rate
(899 GB/s, 1.57x "over" the DRAM floor) and only the ratio is meaningful — the window's 16.4-17.2x is a
floor-on-the-floor.  Both readings agree on the direction and the order of magnitude, and neither is a DRAM claim for
the small cases (stated in the bench's own output).

The comparison separates *arithmetic* from *bytes*.  It does not compare access pattern with access pattern: the
shipped kernel's loads are coalesced too (a warp reads a contiguous 256 B run of a weight block) but it issues them
8-10 B at a time and follows each with ~24 integer operations; `flat4` issues a 4 B load per thread and does nothing
with it.  Narrow loads are therefore not the explanation either — `flat4` reaches the card's rate with exactly the
same 4 B per thread.

## 3. The variants, priced (step 2)

Full per-shape table: `d2y/D2Y-TABLE.txt` (run 1) and `d2y/D2Y-TABLE-2.txt` (run 2), both generated by the extended
`d2x/xmx_price.py --bench d2y/D2Y-BENCH.txt --variants --base …`.  Window roll-up, both runs:

| variant | window us (run 1 / run 2) | GB/s | vs shipped | of the 4617 us floor | bit for bit |
|---|---|---|---|---|---|
| (a) shipped layout (the rule in `n_in`) | 50 493.6 / 48 250.8 | 52.3 / 54.7 | 1.00x | 10.94x / 10.45x | the reference |
| (b) generic layout (`multi_exact(false)`) | 47 738.7 / 46 246.5 | 55.3 / 57.1 | 1.06x / 1.04x | 10.34x / 10.02x | **yes**, all 25 geometries |
| (c) exact layout, ROWS = 2 | 48 064.8 / 46 252.6 | 54.9 / 57.0 | 1.05x / 1.04x | 10.41x / 10.02x | **yes**, all 25 |
| (c) exact layout, ROWS = 4 | 72 301.4 / 65 134.8 | 36.5 / 40.5 | 0.70x / 0.74x | | yes |
| (c) exact layout, ROWS = 8 | 100 043.7 / 89 142.8 | 26.4 / 29.6 | 0.50x / 0.54x | | yes |
| (c) exact layout, ROWS = 16 | 147 208.3 / 144 918.9 | 17.9 / 18.2 | 0.34x / 0.33x | | yes |
| (d) f16 8x16x16 XMX prototype (D2x's, re-measured) | 17 953.6 / 17 841.3 | 295 / 296 | 2.81x / 2.70x | 3.89x / 3.86x | no (its own tolerance, section 4 of the bench) |
| the pack's bandwidth floor | 4617.3 / 4617.3 | 571.4 | 10.94x / 10.45x | 1.00x | not applicable |

*(The window `vs shipped` column of the variant rows is `shipped/variant`, so the f16 row's 2.81x/2.70x is against
this run's shipped row; D2x's own table, at its own shipped row, gave 2.94x.  Same kernel, same rig, the diff is the
shipped row's own spread.)*

**Which of them can ever be bit-equal** (the card asks this explicitly): every ROWS in (c), and (b) at ncols = 4, are
bitwise equal to the shipped kernel *by construction and by measurement* — a thread's kbx sequence, its accumulation
order, the warp butterfly and the cross-warp add order are untouched, and the bench memcmp's the full
`ncols x n_out` output against the shipped kernel's for all 25 geometries in both runs (`eq (bit for bit)` column:
`r2:Y r4:Y r8:Y r16:Y gen:Y` on every row).  (b) at ncols > 4 is not: above ncols = 4 the generic table drops to
NW = 2 warps, which changes the kbx partition and the reduction tree.  D2x's f16 prototype is not bit-equal and
states its own tolerance (`5.670e-03` rel_rms against the shipped kernel, and `1.158e-07` against its own encoding).
The reader's note that (b) is bit-equal at ncols = 4 here does not contradict D2's refusal of it (card t_8306429a):
D2 measured three *separate runs* of the engine producing three different token streams and a 60% window-cost
spread, which is a run-to-run stability question, not a "differs from the shipped kernel" one.  Nothing here proposes
flipping that switch.

The reader's window table prints (b) and (c)'s ROWS = 2 as 1.04-1.06x.  They are the **same kernel**
(`native_mmvq_multi_kernel<F, NCOLS, 4, 2>` with `blocks = ceil(n_out/2)`: the generic path *is* NW = 4, ROWS = 2
for ncols ≤ 4), so having both in the same case is a built-in drift meter, and it is the honest bound on this rig:
in run 2 they agree to 0.2-1.5% on every case, while in run 1 seven of the 25 geometries drift 16-40% between those
two adjacent measurements (e.g. `IQ4_XS 2560x10240`: r2 406.1 against gen 290.1).  Any verdict below ~2x on those
seven shapes is not measurable with this bench, which is why every number in this write-up is given for both runs.

### The one reproducible win, and why it is priced rather than wired

`IQ4_NL 640x2560` (47 of the 301 launches, 12.3% of the family) at ROWS = 2 instead of the shipped ROWS = 4:
**86.8 / 86.7 us against 133.6 / 134.1 us = 1.54x / 1.55x**, bitwise equal, in two runs; `Q8_0 640x2560` (1 launch)
79.7 -> 63.0 / 79.5 -> 63.3 = 1.27x / 1.26x.  Those two shapes are 86.9 us of the 1998-2245 us the r2 window saves,
i.e. essentially all of it; `IQ4_XS`'s five shapes swing 0.85-1.19x across the runs and are noise.  The mechanism is
consistent with section 1: those are the geometries where 88 of 128 threads do nothing but store partials, so
halving the rows a work-group owns halves the partial array (6 KB -> 3 KB) and the reduction per work-group while
doubling the number of work-groups.

It is not wired, and no engine default moved.  The window-level effect is 1.04-1.05x (≈1.8% of a T = 4 window), the
card's own bar for an engine arm is "below ~20 ms or better", D2a's lesson is that an in-stream microbench gain is
not a window gain, and a switch that fires only for `n_in/DIV < BPI && type in {IQ4_NL, Q8_0}` is shape-specific
enough that it needs its own ids-guarded arm rather than a ride on this card's.  The knob that measures it is in the
engine, OFF by default (`native_mmvq_set_exact_rows`, 0 = the shipped rule), and nothing in the engine calls it.

## 4. What the 8.8% is, in one line each

1. **The dot product, not the bytes.**  No int8 dot instruction exists outside XMX on this device; `__dp4a` is four
   multiplies plus shifts and sign fixups per lane, at 8 calls per 32 elements in `q6_q8_dot_impl`.  The same bytes
   with that work removed run 10.0x faster on the DRAM-sized case and 16.4-17.2x faster on the window.
2. **Per column.**  `ncols` = 2/4/6 costs 35.3/55.6/69.7 ms of window time (`d2y/D2Y-MARGINAL.txt`), a straight line
   of 8.6 ms per extra column against 18.1 ms at zero columns — 62% of the family at ncols = 4 scales with the
   column count (activation loads + the integer dot + the per-output reduction), and the residual 18.1 ms of
   pure weight reading is itself 3.9x above the same bytes' floor, i.e. not bandwidth either.
3. **The reduction.**  The window's `__shared__` partial arrays are 2.00 GB against 2.64 GB of weights (0.76x), and
   every one of the 128 threads stores `NCOLS x ROWS` floats to them whatever it computed.  More rows per work-group
   makes this worse per output, which is what r4/r8/r16 measure.
4. **The shape of the fix.**  Anything that keeps this arithmetic keeps this cost.  That is why the card's proposal
   (int8 XMX) cannot be built on this driver, why the f16 XMX prototype is the only measured win (2.93-3.04x, capped
   at 4.21x by 2 bytes/element), and why "read the pack's own bytes at the card's rate" is a real 11.3x *only* if the
   bytes can be consumed without paying for them — i.e. by a unit, not by the scalar path.

## 5. The engine side: one inert knob, and the tests

`src/kernels/cuda/native_mmvq.cu` gained `g_exact_rows` and `native_mmvq_set_exact_rows(int)` / `native_mmvq_exact_rows()`
(`include/strata/kernels/native_mmvq.hpp:35-40`): 0 — the default — is the shipped rule, and the only caller in the
tree is the microbench (`d2y` grep: the bench's section 6).  An engine process that never calls the setter runs the
shipped kernel, which is what makes section 3's shipped column and section 2's comparisons valid.  This is the first
card in the D2 family that changes engine source; the knob is a *price-only* arm, no default moved.

* Two controls that the edit is inert: (1) `d2y/D2Y-BASE.txt` was produced by the *previous* binary (md5
  `42de319…`, no knob in it at all) at the same reps — its shipped per-case rows agree with this card's to the rig's
  own spread (17 of 25 cases within ±2.9%, the eight others 8.5-90% — see section 6); (2)
  `mmvq_multi_parity` passes on both builds.
* The engine's own tests (`d2y/D2Y-TESTS.txt`, `d2y/D2Y-TESTS-HEAD.txt`): `mmvq_multi_parity` (the bitwise harness for
  exactly the multi-column kernels this knob touches) **passes**.  `quantize_act_parity` and `iq_multi_parity` fail —
  and fail identically with this card's edit **stashed** (same binary, HEAD, `d2y/D2Y-TESTS-HEAD.txt`), so they are
  pre-existing: `quantize_q8_K`'s f32 scale differs from its reference in the last byte or two, and the IQ2_XS/IQ3_XXS
  rows of `iq_multi_parity` report "0 outputs differ from the old kernel" while the *reference* comparison fails
  (rel 1.0e+00 to 3e+33).  Neither is a kernel this card touched; not attributed here.
* `build-sycl/strata` rebuilt at 23:52: md5 `3dbea776ac8bf94414c199bdae05877c` (was `cc6a41ad8ff3cf95fd10669d7600bf2e`
  as D2x recorded — the change is the knob plus the new template instantiations; no default behaviour differs).

## 6. The rig's own spread, which the reader now prints as the inert control

`d2x/xmx_price.py --base <other run>` prints the shipped layout's per-case rows from an earlier run against the
current one.  Two such comparisons, and they matter for anyone reading the per-shape table:

* this card's run 1 against run 2 (**same binary, 4 minutes apart, same command**): 25 rows, largest single-case
  delta **36.8%**; 17 of 25 within ±2.2%, and the eight that move 16.7-36.8% are the shapes with `n_out` = 10240 or
  12288 (Q6_K, Q5_K, Q4_K, IQ4_XS) plus `Q8_0 640x2560`.  Section 6's own shipped column, measured later in the same
  run, moves with them: `Q6_K 2560x10240` reads 412.6 us in run 1 and 311.3 us in run 2 while its section-3 row
  moves 2.2%.
* this card's run 1 against `D2Y-BASE` (**different binary**): largest single-case delta 90%.

So the per-shape price of this family on this rig has a bimodal band of up to ~2x on eight geometries, while the
*window* number (301 calls, which is what the census and this card quote) moves only 1.5-4.4% between runs: run 1
50 493.6 us, run 2 48 250.8 us, the census 51 267.3 us.  Every claim in this write-up is stated for both runs, and
every variant ratio is taken from measurements made **adjacently in one pass** (`ship, gen, r2, r4, r8, r16, flat4`)
— except the r2-vs-gen pair, which is deliberately the same kernel twice and is printed as the drift meter.

## 7. Not validated, and what would settle it

1. **No engine arm, no ids guard, no switch on by default.**  Nothing measured here clears the card's ~20 ms bar for
   the family (the shipped layout is 48.3-50.5 ms, the best variant of it 46.2-47.7 ms, the f16 prototype 17.8 ms),
   so no default was moved and there is no ids run to report.  The `IQ4_NL`/`Q8_0` ROWS = 2 result (1.26-1.55x on
   those shapes, 1.04-1.05x on the window, bitwise equal) is the one thing a follow-up card could carry to the
   engine, with the ids guard at 4K/32K and the per-type census (`STRATA_LAUNCH_HIST=1` + `d2/d2_census.py`); this
   card prices it and stops there, because its window effect is ~1.8% and D2a's counter-example (a −9.9% microbench
   win refused in-stream) is the standard this board holds such things to.
2. **The `flat4`/`flat16` window rows are L2 rates** for every case except the 497 MiB head; the bench prints that
   caveat, and the DRAM claim rests on the head alone (589.4 GB/s against the independently measured 571.5 GB/s).
   Nothing in the window row should be read as a DRAM bandwidth number.
3. **The per-column decomposition (section 4.2) is a two-point fit through noisy per-case data** (ncols 2 and 6, one
   run).  The *direction* is corroborated by D2x's own 4-vs-6 table (52.4 -> 65.1 ms), and the apportionment (62/38 at
   ncols = 4) is a model, not a measurement of the parts.
4. **The ROWS sweep is one knob, not the whole design space.**  Split-k across work-groups, a different warp count,
   wider per-thread weight loads staged through `__local` memory, or dequantize-then-dot with f16/f32 activations are
   all outside what was built here; of those, only the last changes the arithmetic, and D2x already priced the f16
   version of it.
5. **The 2x family spread is characterised, not explained.**  Seven geometries are bimodal between runs (their weight
   buffers are 10-26 MB); no experiment here isolates whether that is L2 residency across the reps burst, clock state,
   or allocation placement.  It is printed so the next card can see it rather than re-discover it.

## 8. State this card left the machine in

No engine run, no `serve.server`, no container; both B70s free (two `0xe223` devices, card0 and card1 — the card
measured on is card 0 via `ZE_AFFINITY_MASK=0`); no bench process left running (`d2y/machine.txt`, taken after the
runs).  The full target set was rebuilt so the tree is consistent: `build-sycl/strata` 23:52, `build-sycl/mmvq_xmx_price`
23:51 (the runs used `159c3b1878590839de95f7938e14988f`, recorded in each raw file's header).  Evidence, scripts and
raw stdout are under `d2y/`; the copies for the board are attached to the card.
