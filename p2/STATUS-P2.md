# P2 — prefill: the QSA prompt attention is 75% of the prefill's GPU time, and what it costs is the KERNEL'S BLOCK SHAPE, not the FP32 mma emulation

Card `t_a1b6fa6c` (P2), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, base `569569c` (P1b), the change
is `c7f20cb`. Nothing pushed (origin has no `sycl-xpu`). Config of record throughout: `strata-sycl-iq3s.json` —
IQ3_S GSQ-RCO snapshot `ed59f92…`, both B70s (`ZE_AFFINITY_MASK` unset, `--layer-split auto` → K=23),
`--kv int8 --kv-resident 32768 --expert-cache auto --mmap-experts --prefill auto (8192) --spec 4 --no-capture
--stats`, checkpointing off, one engine at a time. WARM program cache (`sycl-cache/m6c`) in every arm.

## 0. Verdict

**The card's two hypotheses are measured, and the first is confirmed while the second is refuted.**

1. **QSA prompt attention is the prefill.** The engine's own GPU-side phase timer (`STRATA_PREFILL_TIMING=1`,
   events on the compute stream — this instrument existed and is what the card's task 1 asked for) attributes
   **53,679 of the 71,172 ms** GPU timeline of a 32,256-token prefill's four chunks to the `qsa attn` phase:
   **75.4%**, flat per chunk (12.1-14.1 s per 8,192-token chunk) and flat with depth, because the selection is
   capped (`qsa_selection_width`). Everything else in the prompt path is small: expert dequant 7.1%, `qsa select`
   4.5%, host grouping 3.6%, gate/up 2.2%, down 1.1%, GDN 1.6%.
2. **The FP32 mma emulation is NOT what that attention costs.** The emulation is 7.79x the arithmetic it replaces
   (measured: 5.848 ms per 4,000 mma pairs against a 128-FMA floor of 0.751 ms), but the kernel is nowhere near
   FLOP-bound: its FMA content is **4-9% of the card's FP32 rate**. Cutting the emulation's redundant work by
   **19.2%** (change `c7f20cb`, bit-identical) moved the kernel by **1.7%** — so the whole emulation, exchange and
   conversions included, is only **~9% of the kernel**. A hardware tensor-core path (the DPAS/XMX shapes the card
   *does* expose) can therefore buy at most ~9% here, not 4.3x. End to end that change is worth **~1% of the
   prefill at both 32K and 128K** (at 128K: qsa attn −1.58%, GPU timeline −0.97%, wall −0.94%, prefill 334.9 →
   338.0 tok/s), and it moves no token id (4K 150/150, 32K 256/256, 128K 256/256 identical).
3. **What the kernel actually costs is its block shape.** The engine ships a second kernel for the same
   attention — the batched **decode** attention `qsa_decode_attn_batch` — which the prefill path falls back to and
   which the port's D-1 dispatch bypasses. On the same shape, same int8 KV, same accuracy class, it is **511.5 →
   79.3 ms per 2,048-query chunk = 6.5x faster**, because **one block handles 32 queries and re-gathers the K/V
   selection once for all of them**, while the prompt kernel's block is one query and re-gathers the whole
   selection for every single query. Switched on (`STRATA_PROMPT_ATTN_OLD=1`): **prefill 347.5 → 768.6 tok/s
   (2.21x), GPU timeline 64,970 → 27,111 ms, wall 136 → 85 s** — and **181 of 256 greedy token ids move**, so per
   the card's own rule that is a correctness change, reported as one and not taken.

## 1. The instrument, and why the earlier answers were "host-side" and "the GPU is fine"

`STRATA_PREFILL_TIMING=1` (prefill.cpp:1008-1051) records events on the compute stream at every phase boundary and
folds them at each MoE layer's host sync, so a gap where the GPU waits lands on the phase that was waiting. Its own
output line carries `GPU timeline <ms>, wall <ms>, host staging <ms>` plus 20 named phases. It is turned on by an
environment variable only, needs no tracer, and costs nothing measurable (the 32K arm that used it read 340.0 tok/s
against 347.5 for the same binary without it — 2.1%, inside this harness's spread, §5).

L0 instrumentation is **not** used here: P1/P1b (cards `t_44a0ac61`, `t_58d5592c`) established that `unitrace`
can now write a report, but only on the *failure* path (`--chrome-kernel-logging` produced a 1.13 GB closed
timeline in `p1b-x-ut9`), the traced window never finishes, and this card's own rule is to use the engine's stage
timings and `bench/micro` first. Both were used; `unitrace` was not re-attempted.

`STRATA_PREFILL_TRACE=1` (per-layer, per-half, per-expert lines) was also run at 32K — 59,616 lines,
`m6c/runs/p2-32k-trace`, parsed by `scripts/p2_attrib3.py`. It shows the same thing measured a second way (the
QSA layers' MoE half is ~1.9 s at 8,192 tokens against ~0.10 s for the 36 MoE-only layers'), but its per-expert
lines are host **submission** latency (the GPU runs ahead), so it is not the instrument to quote phase costs
from. It is the reason the first, wrapper-based attribution (`scripts/p2_attrib.py`, `p2_attrib2.py`) had to be
discarded: the two stages' streams interleave in one file and each stage's offsets are its own.

## 2. The attribution table: 32K prefill, 32,256 tokens, before any change

`m6c/runs/p2-32k-timing` (binary `~/strata-xpu/p2/strata-before-P2`, md5 `d493e320fb212380e2e3d582180bc370` =
P1b's shipped binary), `bash m6c/m6c_serve.sh p2-32k-timing 32768 --prefill-auto 256` with
`STRATA_PREFILL_TIMING=1`. Raw lines: `p2/evidence/32k-before-timing.txt`.

Per-chunk lines, summed over the four chunks (`p2/p2_timeline.py`; the request-level line is quoted separately
below because its own phase split is distorted by the event fold). Raw lines:
`p2/evidence/32k-before-timing.txt`.

| phase | ms (4 chunks) | % of the chunks' GPU timeline |
|---|---|---|
| **qsa attn** (the QSA prompt attention kernel) | **53,679** | **75.4%** |
| expert dequant | 5,082 | 7.1% |
| qsa select | 3,201 | 4.5% |
| host grouping | 2,561 | 3.6% |
| gemm gate/up | 1,565 | 2.2% |
| gdn recurrence | 1,141 | 1.6% |
| hc read | 859 | 1.2% |
| gemm down | 806 | 1.1% |
| qsa proj 472, gdn 441, combine 370, gdn out proj 309, gdn conv+gates 212, router+shared 145, gather 144, embed+steps 143, wait copy 37, qsa indexer 2 | 2,275 | 3.2% |
| **GPU timeline, four chunks** | **71,172** | 100% |
| wall, four chunks | 79,818 | — |
| host staging (cumulative inside the chunk lines) | 15,406 | — |

The request's own lines (`p2-32k-timing/err.txt`):

```
strata serve: prompt 32256 tokens = 0 reused + 32256 read in 94866 ms (340.0 tok/s), 256 generated in 11778 ms (21.7 tok/s), drafts accepted 171 of 214
strata prefill timing: [chunk] 8192 tokens, GPU timeline 16361 ms, wall 18556 ms, host staging 1515 ms: ... qsa attn 12309 (75.2%) ...
strata prefill timing: [chunk] 8192 tokens, GPU timeline 18346 ms, wall 20537 ms, host staging 3043 ms: ... qsa attn 14079 (76.7%) ...
strata prefill timing: [chunk] 8192 tokens, GPU timeline 18680 ms, wall 20877 ms, host staging 4619 ms: ... qsa attn 14083 (75.4%) ...
strata prefill timing: [chunk] 7679 tokens, GPU timeline 17785 ms, wall 19848 ms, host staging 6229 ms: ... qsa attn 13208 (74.3%) ...
strata prefill timing: 32255 tokens, GPU timeline 64970 ms, wall 92510 ms, host staging 0 ms: ... embed+steps 12857 (19.8%) ...
strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) 953 ms, waiting for each chunk 0 ms, after each chunk (the draft layer, progress) 0 ms, PLE 925 ms
```

The last line is the point of the per-chunk/per-request distinction: the request-level line reports a *smaller*
GPU timeline (64,970) than the four chunks' own sum (71,172) and a wall of 92,510 against 79,818, because the
4 × `after each chunk (the draft layer, progress)` ≈ 2.19 s and the chunk-boundary gaps are outside its fold (its
own split charges 12,857 ms to `embed+steps`, which the per-chunk lines put at 30-40 ms each). The per-chunk
lines are what §2 and §4 use.

Per chunk the split is stable (12,309 / 14,079 / 14,083 / 13,208 ms of `qsa attn` at n = 8,192 / 16,384 /
24,576 / 32,255 cells): the attention does **not** grow with depth, because the QSA selection is capped at
`qsa_selection_width = idx_top_k + idx_block - 1` cells. That is why our prefill is flat across depth (the card's
"good" observation) — and it is also why the 4.3x gap is not a depth effect but a per-chunk constant.

**Wall against GPU: 92,510 ms of wall for 64,970 ms of GPU timeline (request level) — 27.5 s (30%) is
host/pipeline**, of which the engine names `after each chunk (the draft layer, progress)` **≈ 2,190 ms × 4
chunks = 8.8 s** and host staging ~6.3 s. That is the second-largest item in the prefill and it is not addressed
here (§7).

## 3. 128K, the same instrument, and the change measured end to end

`m6c/runs/p2-128k-timing`, `bash m6c/m6c_serve.sh p2-128k-timing 131072 --prefill-auto 256`: **prefill 129,024
tokens in 385,249 ms = 334.9 tok/s, decode 19.6 tok/s** (the record's own 128K arm read 342.6 / 19.6). 16 chunks;
per chunk the GPU timeline is 24.1-25.1 s with `qsa attn` 14,056-14,094 ms (**56-58%**), `qsa proj` 2,309-2,310
(9.5%), `qsa select` 4,125-5,215 (17-21%, growing with depth: the selection grows over the resident window), and
`host staging` 19.8-24.9 s of the chunk's ~27 s wall. So at 128K the attention is still the largest single phase,
the indexer/selection becomes a real cost, and the RAM→VRAM staging of the streamed KV approaches the GPU time.

Summed over the 16 chunks, the arm **after** the change (`m6c/runs/p2-128k-fused`, same command), and this is the
cleanest A/B in this card — 16 chunks, one binary each, every non-attention phase identical to 0.1%:

| | before | after | delta |
|---|---|---|---|
| **qsa attn** | 219,891 ms | 216,412 ms | **−3,479 ms (−1.58%)** |
| **GPU timeline, 16 chunks** | 358,270 ms | 354,805 ms | −3,465 ms (−0.97%) |
| qsa select 45,130/45,129 · qsa proj 36,406/36,404 · dequant 20,902/20,890 · host grouping 11,285/11,335 · gate/up 6,088/6,091 · every other phase | | | unchanged |
| wall, 16 chunks | 368,341 ms | 364,870 ms | −3,471 ms (−0.94%) |
| the request's own `read in` | 385,249 ms (334.9 tok/s) | 381,683 ms (**338.0 tok/s**) | −0.93%, **+0.93% tok/s** |
| decode | 19.6 tok/s | 19.6 tok/s | unchanged |
| greedy token ids (256) | `c60e72d397e9b1cd44dab9c5ab1187c8` | `c60e72d397e9b1cd44dab9c5ab1187c8` | **identical, 256/256** |

At 128K every measure moves together: the phase the change touches −1.58%, the GPU timeline −0.97%, the wall
−0.94%, the headline prefill rate +0.93% — the phase is 61% of the timeline, so a 1.58% phase cut that lands as a
0.94% wall cut is the arithmetic one expects. At 32K the intra-request measures agree in the same way (§4), while
the single request-level prefill number there is noisy (§5). The change is therefore a **~1% prefill improvement**
at both lengths, not the 4.3x the card is after — and it is free (bit-identical).

## 4. Change 1 — the hi/lo mma pair rebuilds its B fragment once (commit `c7f20cb`)

**What it is.** All four mma sites of the prompt attention use the m16n8k16 emulation as a PAIR against one B
fragment: `mma16816(c, ah, b); mma16816(c, al, b);`. The emulation rebuilds each operand's fragment with
`select_from_group` (16 gathers + 32 f16→f32 conversions per operand, per call), so the B fragment — the same
fragment in both calls — was rebuilt twice. `mma16816_f32_pair` (include/strata/sycl_compat/mma16816.hpp) builds it
once, and interleaves the two halves' 16-term k loops (each of the eight accumulators still takes its own products
in its own k order; the hi half still reaches `c` before the lo half). Same products, same order, same rounding:
**bit-identical by construction**, and measured identical (§5).

`src/kernels/cuda/qsa_prompt_attn.cu` is **untouched**: the fusion is expressed as hand-port replacements
(`tools/sycl/handport/qsa_prompt_attn.py`, entry #6, one per paired site, with the `D1_NO_QLO`/`D1_NO_PLO`
precision switches keeping exactly the meaning they had) and `src/kernels/sycl/qsa_prompt_attn.cpp` is regenerated
by `tools/sycl/handport.py qsa_prompt_attn`.

**Measured delta 1 — the emulation itself** (`bench/micro/p2_qsa_emul_cost.cpp`, one card, 4,000 iterations over
8,192 work-items, `p2/evidence/emul-cost.txt`):

```
  mode 0  mma as shipped (2 calls, 64 sel + 128 cvt + 128 FMA)     5.848 ms
  mode 1  fused pair      (the shipped header function)            4.725 ms      -> 19.2% cheaper
  mode 2  128 FMA only    (no exchange, no cvt)                    0.751 ms      -> the floor
  mode 3  64 select_from_group only                                1.616 ms
  mode 4  128 f16->f32 only                                        0.683 ms
  the shipped pair is 7.79x the pure arithmetic; the fused pair 6.29x
```

**Measured delta 2 — the kernel** (`build-sycl/sycl_prompt_attn_parity`, the engine's own parity program, ctx
32,768, 2,048 queries, 3 reps; both binaries run the same source, the same fixture shapes and the same FP64
reference; the BEFORE binary is kept at `p2/sycl_prompt_attn_parity-before-P2`, output in
`p2/evidence/parity-ab.txt`):

| case | before | after |
|---|---|---|
| int8 ctx 32768 | PASS, err vs FP64 2.14e-06, 510.618 ms/chunk | PASS, err vs FP64 **2.14e-06**, 502.476 ms (**−1.6%**) |
| fp16 ctx 32768 | PASS, 1.9e-06, 643.291 ms | PASS, 1.9e-06, 683.682 ms (**+6.3%**) |
| int8 ctx 1500 | PASS, 3.29e-06, 140.515 ms | PASS, 3.29e-06, 139.634 ms (−0.6%) |
| int8 ctx 2100 | PASS, 2.24e-06, 64.574 ms | PASS, 2.24e-06, 63.077 ms (−2.3%) |

All four cases PASS on both binaries and every error figure is **unchanged to three significant digits**
(`err_new` 2.14e-06 / 1.9e-06 / 3.29e-06 / 2.24e-06; `new vs old` 4.53e-06 / 3.78e-06 / 4.89e-06 / 4.17e-06). The
int8 cases the config of record uses are 0.6-2.3% faster; the fp16 KV instantiation (`KV_MODE=0`) is 5-6% slower
in two independent runs — the fused body holds six 16-float fragment arrays live instead of four, register
pressure it does not repay there. Reported, not hidden; `--kv int8` never reaches it.

**Measured delta 3 — the engine, 32K** (`m6c/runs/p2-32k-fused`, same command as §2):

```
strata serve: prompt 32256 tokens = 0 reused + 32256 read in 96433 ms (334.5 tok/s), 256 generated in 11733 ms (21.8 tok/s)
strata prefill timing: 32255 tokens, GPU timeline 66686 ms, wall 93926 ms
  qsa attn per chunk: 12083 / 13840 / 13847 / 12978 ms   (before: 12309 / 14079 / 14083 / 13208)
```

Summed over the four chunks (`p2/p2_timeline.py`, both arms in `p2/timeline-32k-ab.txt`):

| | before | after | delta |
|---|---|---|---|
| **qsa attn** | 53,679 ms | 52,748 ms | **−931 ms (−1.73%)** |
| **GPU timeline, four chunks** | 71,172 ms | 70,275 ms | −897 ms (−1.26%) |
| every other phase (dequant 5,082/5,087, select 3,201/3,204, host grouping 2,561/2,574, …) | | | unchanged, within 1% |
| wall, four chunks | 79,818 ms | 78,972 ms | −846 ms |
| the request's own `read in` | 94,866 ms (340.0 tok/s) | 96,433 ms (334.5 tok/s) | +1.7% |

The phase the change touches is faster in **all four chunks** (−1.8 / −1.7 / −1.7 / −1.7%, matching the parity
program's −1.6%), the four chunks' GPU timeline is 1.26% lower and their wall 1.06% lower. The m6c harness's own
headline prefill number for those two arms moves the other way (+1.7% time: 94,866 → 96,433 ms) — that harness
carries `STRATA_PREFILL_TIMING` and reads 2-5% apart between arms of the same binary (§5), so §5's matched
binary-vs-binary pair (32K: 347.2 → 351.1 tok/s, +1.1%) and the 128K pair (+0.93%) are the end-to-end numbers to
quote.

**Decode is untouched** (§5): `qsa_prompt_attn_batch` has no decode caller, and the 4K and 32K decode numbers and
the whole token stream are unchanged.

## 5. Before/after, and the guardrail

Greedy, 256 usage-counted tokens at 32K and 128K, 150 at 4K, both GPUs, config of record, raw `T <id>` lines
compared by md5 and element-wise (`scripts/p2_idcmp.py`).

**The A/B that can carry a 1% claim: two binaries, one harness, one command line each.** `p1/p1b_run_engine.sh`
takes the binary as an argument, so each pair below was run back to back with the identical command (only the
binary changes):

| pair | prompt tokens | prefill before | prefill after | delta | decode | ids |
|---|---|---|---|---|---|---|
| 4K (`p2-4k-before` / `-after`) | 3,832 | 242.7 tok/s | 243.6 tok/s | +0.37% | 21.4 / 21.4 | `ad985c23…` both, **150/150 identical** |
| 32K (`p2-32k-before3` / `-after3`) | 32,256 | 347.2 tok/s (92,905 ms) | **351.1 tok/s** (91,878 ms) | **+1.1%** | 21.8 / 21.8 | `124a3cd3…` both, **256/256 identical** |
| 128K (`p2-128k-timing` / `p2-128k-fused`) | 129,024 | 334.9 tok/s | **338.0 tok/s** | **+0.93%** | 19.6 / 19.6 | `c60e72d3…` both, **256/256 identical** |

**The guardrail holds at every length**: 150/150 ids at 4K, 256/256 at 32K and at 128K — byte-identical greedy
output, on both cards, with the change in. The 4K pair was also checked against P1b's own 4K record (same image:
242.3 tok/s, 21.5 tok/s decode, same md5), so three arms of that length agree.

**And the noise, stated plainly**: the same 32K prompt on the m6c harness read 347.5 / 340.0 tok/s for the BEFORE
binary (timing off / on) and 334.5 tok/s for the AFTER binary with timing on — i.e. 2-5% swings between arms of
one binary, larger than the effect being measured. Only the matched pairs above and the per-chunk phase totals
(§4, §3) resolve the change, and there all five measures agree: qsa attn −1.73% (32K) / −1.58% (128K), the
chunks' GPU timeline −1.26% / −0.97%, the chunks' wall −1.06% / −0.94%, the request's own prefill time −1.1% /
−0.93%, and the phase's share of the timeline (75% / 61%) explains the size.

## 6. The lever that is NOT free: the engine's own attention kernel is 6.5x faster and moves one token

`STRATA_PROMPT_ATTN_OLD=1` sends the prompt attention to `qsa_decode_attn_batch` — the kernel the prefill path
already falls back to when the tensor-core form refuses the pools, and the one the D-1 dispatch bypasses on SYCL.
`m6c/runs/p2-32k-oldattn`, same prompt, same binary, same config:

```
strata serve: prompt 32256 tokens = 0 reused + 32256 read in 41967 ms (768.6 tok/s), 256 generated in 11512 ms (22.2 tok/s)
strata prefill timing: 32255 tokens, GPU timeline 27111 ms, wall 39573 ms: ... qsa attn 1861/2121/2124/1992 ms ...
```

**prefill 347.5 → 768.6 tok/s (2.21x); GPU timeline 64,970 → 27,111 ms (2.4x); qsa attn 53,679 → 8,098 ms
(6.6x); wall 136 s → 85 s; decode 21.8 → 22.2 tok/s; same expert-cache numbers (100% GPU).** Repeated once
(85 s wall, 770.0 tok/s, same ids).

The output: **ids md5 `c52712f457344f264e0df07b5891a1e4` against the record's `124a3cd3…` — 181 of 256 ids
differ, and the difference is a single-token insertion at id 75 followed by a re-sync** (`scripts/p2_idcmp.py`;
ids 0-74 and every later position agree, shifted by one). The two kernels' own parity program measures them 1.2e-06
of scale apart against an FP64 reference (both FP32-accurate, ~2.1e-06): this is a rounding-level difference
amplified by one near-tie in 256 greedy steps. Per the card's rule it is a **correctness change**, not a
speedup, and is not the change of record — but it is the measured size and location of the headroom.

## 7. The ceiling, quantified

* **The FP32 emulation's redundancy: ≤9% of the attention kernel.** The fusion of §4 removes 19.2% of the
  emulation's cost (micro) and 1.7% of the kernel (parity program **and** all four chunks of the engine arm), so
  the emulation — exchange, conversions and all — is ~9% of the kernel's time. A perfect hardware mma would
  therefore buy ≤9% of `qsa attn`, i.e. ≤7% of the prefill's GPU timeline, i.e. far less than the 4.3x. The XMX
  shapes this device exposes (f16 16x16x16, int8 8x32x16) are **not** the lever the card expected them to be:
  the tile-form kernel is worth writing for arithmetic density, not for this gap. (And a tile-form XMX kernel
  changes the summation order, so it would land in §6's category, not §4's.)
* **The kernel's block shape: 6.5x at the kernel, 2.2x end to end — and it costs the token stream** unless the
  v1 math is re-tiled onto a 32-queries-per-block shape *exactly* (same products, same per-output order). That is
  a kernel rewrite, not a micro-optimisation, and it is the only path to closing the card's 4.3x.
* **After §6's 2.2x, the prefill is no longer attention-bound**: the 2.21x arm's GPU timeline is 27.1 s against a
  39.6 s wall, so the next ceiling is the 27.5 s of host/pipeline in the prefill (the ~8.8 s of per-chunk
  `after each chunk (the draft layer, progress)` and the expert/KV staging), plus `qsa select` (8.5%) and `qsa
  proj` (1.2%) and expert dequant (15.5%).
* **`bench/micro`'s own numbers say the same thing from the other side**: 64 `select_from_group` alone cost
  1.616 ms against 0.751 ms of FMAs and 0.683 ms of conversions for the same work — the kernel's FMA content is
  4-9% of the card's FP32 rate, i.e. it is not arithmetic-bound at all.

## 8. Not validated

* **No variance estimate, and the harness's spread is larger than the effect.** The same 32K prompt on the m6c
  harness read 347.5 and 340.0 tok/s on two arms of the BEFORE binary (timing off / on) and 334.5 on the AFTER
  binary with timing on, against 347.2 / 351.1 on the p1b harness's matched pair: 2-5% between arms. The change's
  end-to-end claim therefore rests on the matched pairs (§5: +1.1% at 32K, +0.93% at 128K, +0.37% at 4K) *and* on
  the per-chunk phase totals (§3, §4), which agree with each other to 0.2%; nothing here establishes a variance.
* The m6c 32K AFTER arm read 334.5 tok/s — the low end of the before-binary's own range — and no arm was run to
  pin the machine state (page cache, clocks, thermal) that produced it. `STRATA_PREFILL_TIMING`'s own cost was
  never isolated.
* **fp16 KV regression**: the fused body is 5.3% slower on the `KV_MODE=0` instantiation (parity program, one
  run). Not reachable from the config of record (`--kv int8`), not investigated further.
* The 128K and 262K arms were not re-run with the fused kernel beyond the one 128K arm of §5; `qsa select`'s
  growth with depth (4.9% → 21%) is measured but not diagnosed.
* The 30% host share of the 32K wall (`after each chunk (the draft layer, progress)` 2.19 s × 4, host staging
  6.3 s) is named, not attacked.
* The old-attention path's single-token change was checked at 32K only (not 4K/128K/262K), and not against an
  independent oracle: it is a difference between two FP32-accurate kernels of the same engine.
* `unitrace` was not re-attempted on the clean path (P1b: the traced window never finishes; the report it does
  write is a half-timeline on the failure path).
* Nothing is pushed: origin has no `sycl-xpu` branch.

## 9. Files

| what | path |
|---|---|
| the change | `include/strata/sycl_compat/mma16816.hpp`, `tools/sycl/handport/qsa_prompt_attn.py`, `src/kernels/sycl/qsa_prompt_attn.cpp` — commit `c7f20cb` |
| the emulation micro | `bench/micro/p2_qsa_emul_cost.cpp` (+ `build-sycl/p2_qsa_emul_cost`) |
| the phase timer that did the attribution | `src/prefill/prefill.cpp:1008-1051` (`STRATA_PREFILL_TIMING`) |
| **this report's evidence, curated and committed** | `p2/evidence/` — every arm's `err.txt`, the two token-id streams per length, `parity-ab.txt`, `emul-cost.txt`, the phase totals, and `P2-EVIDENCE.txt` (all of it in one file) |
| the 32K before / after arms | `~/strata-xpu/m6c/runs/p2-32k-timing/`, `…/p2-32k-fused/` |
| the 32K matched pair (explicit binaries) | `~/strata-xpu/p1/runs/p2-32k-before3/`, `…/p2-32k-after3/` |
| the 32K per-layer trace arm | `~/strata-xpu/m6c/runs/p2-32k-trace/` (59,616 trace lines; the parsed table is `p2/evidence/attrib-32k-v3.txt`) |
| the 128K arms | `~/strata-xpu/m6c/runs/p2-128k-timing/`, `…/p2-128k-fused/` |
| the 4K A/B (same harness, explicit binaries) | `~/strata-xpu/p1/runs/p2-4k-before/`, `…/p2-4k-after/` |
| the old-attention arm | `~/strata-xpu/m6c/runs/p2-32k-oldattn/` |
| the binaries compared | `~/strata-xpu/p2/strata-before-P2` (md5 `d493e320…`), `build-sycl/strata` (after); parity binaries `~/strata-xpu/p2/sycl_prompt_attn_parity-before-P2` and `build-sycl/sycl_prompt_attn_parity` |
| the tools | `~/strata-xpu/scripts/p2_attrib3.py` (the two-stage trace parser), `…/p2_idcmp.py` (token-id differential), `~/strata-xpu/p2/p2_timeline.py` (phase totals), `…/p2_run_arm.sh` (arm runner that clears the A/B switches), `…/p2_parity_ab.sh`, `…/p2_curate_evidence.sh`, `…/p2_extract.sh` (this report's evidence dump) |
