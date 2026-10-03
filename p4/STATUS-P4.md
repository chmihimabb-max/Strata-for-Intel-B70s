# P4 — the GPU-assisted I/O mechanism, measured on this box (block quota, interference, transfer size, layout)

Card `t_63cc226b` (P4), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, base `8bb7aad`; the change is
`c744cb5`. Arm harness `p4/p4_arm.sh` + `p4/p4_chain*.sh`, arm extractor `p4/extract_arms.sh` (P3's `p3/extract.py`);
probe `p4/kv_io_probe.cu` (build `p4/build_probe.sh`, tables `p4/extract.py`); raw output `p4/evidence/P4-EVIDENCE.txt`.

Config of record in every engine arm: IQ3_S GSQ-RCO snapshot `ed59f920…`, **both B70s** (`ZE_AFFINITY_MASK`
unset, `--layer-split auto`), `--kv int8 --kv-resident 32768 --expert-cache auto --mmap-experts --prefill auto
(8192) --spec 4 --spec-min-p 0.5 --max-context 128K --no-capture --prompt-cache 0 --stats`, MTP drafter `rt-q2_0`
untouched, one engine at a time, warm program cache (`sycl-cache/m6c`), `STRATA_SYCL_XMX` off (the engine's own
default), 129,024-token prompt, 256 generated tokens. The 128K arms all ran 99 decode windows with the same
structure (avg T 3.08, 2.59 tokens/window) and the same 92.15% KV residency hit rate, so the arms are comparable.

The probe measures **the engine's own copy kernel** (`copy_kernel`, linked from `build-sycl/libstrata_kernels.a`) —
one B70 (`ZE_AFFINITY_MASK=0`) — not a re-implementation: `kv_stream_copy_probe` drives the shipping kernel with a
chosen direction, layout and launch shape.

## 0. Verdict

1. **Our mechanism is `copy_kernel`** (`src/kernels/cuda/kv_stream.cu:161-171`), launched from `kv_stream_resolve`
   (`:242-246`) once per QSA layer per streamed decode window, **96 blocks × 128 threads**, each thread moving one
   16 B `uint4`. Its host side is **pinned and device-mapped** (`cudaHostAlloc(Mapped)` → `sycl::malloc_host`,
   `src/core/layer.cpp:634-635`) — the mechanism's requirement, met. The prompt path does *not* use it: prefill
   stages with `cudaMemcpyAsync` (`kv_stage_from_host`, `kv_stream.cu:244`, called at `src/prefill/prefill.cpp:1460`).
2. **The shipping shape already saturates the link.** H2D 1/2/4/8/16/32/64/96/128 blocks × 1024 threads =
   1.07/2.15/4.36/8.91/17.00/21.90/24.65/25.64/25.77 GB/s; the shipped 96 × 128 = **25.55 GB/s** (96% of this
   box's 26.5 GB/s contiguous figure), and 4×128/16×128 reach 25.5-25.6. Bandwidth per block is ~1.07 GB/s at low
   counts: a 1024-thread block carries only **2 KB** in flight (128 lanes of a code run have one `uint4` each) at a
   measured ~1.9 µs effective latency — the paper's own `X = C·S/L`. Their 2 × 1024 threads reached ~50 of 64 GB/s
   on H200, i.e. their block carries far more concurrency than ours.
3. **The paper's quota would cost us 12x in bandwidth.** 2 blocks × 1024 threads = **2.12 GB/s H2D / 5.07-5.10 D2H**
   (8% / 19% of the link) against ~50 GB/s (78%) on theirs. And on this port the copy kernel is issued on the
   **same in-order stream** as the layer's compute, immediately before the attention that reads the slots
   (`src/core/verify.cpp:999-1002`), so there is no concurrent compute for a quota to protect and the kernel's whole
   duration is serial with the window. Measured consequence in §6: the confined shape (1 × 128) costs **3.4% of
   decode**, the shipped shape ~0.1%.
4. **Interference: nothing to trade.** Concurrent with the engine's own prompt attention (prefill-like), normalized
   throughput is **0.984-0.999 at every block count 1..128** — inside the paper's <5% bar even at one block.
   Concurrent with the engine's own Q5_K expert GEMV (decode-like) the compute reads 0.392-0.402, flat from 1 block
   to 128 — and the controls show that is **queue turn-taking, not contention**: two equal full-GPU workloads on two
   queues come out at 0.994/0.996 of sequential (no time-sharing at all), and in the order control the prefill-like
   compute keeps its speed (+3.2% worst) while the small-kernel decode-like workload pays up to +32% because each of
   its 1,200 launches queues behind an I/O kernel. No SM quota can change that (§3).
5. **Transfer size: our page sits in the flat part of the curve.** 512 B .. 1 MiB transfers at 32 × 1024 threads
   are a flat **26.7 GB/s H2D / 22.7 D2H**, 128 B holds **25.3 GB/s** (95% of peak), and the collapse is below
   128 B (§4). The paper's ~22% of PCIe 5.0 at page 32 is a *scattered, per-head-strided* page; our host copy is the
   identity layout (block-major, contiguous), so a 4-cell page moves at ~96% of peak, not 22%.
6. **Layout: a measured null.** A page-first host copy (a block's four runs adjacent in one allocation) at the same
   page size gives **24.15 against 25.58 GB/s** at the shipped shape (21.05 against 21.67 at 32 × 1024): no gain,
   marginally worse. Reason, measured: the four runs (2,048/2,048/64/64 B) are each already sequential in their own
   array, so the kernel reads four sequential streams; packing them interleaves two long runs with two very short
   ones and loses the per-array stream. The transform itself is free (one extra multiply per block). The layout
   change is **not adopted**; the paper's 4x disk win came from a layout our host copy never had (§5).
7. **Our granule's own shape**: the two code runs are 2,048 B and the two scale runs **64 B**, on which only 4 of
   128 threads work. It is the one place the page shape wastes threads and it costs nothing measurable
   (aggregate 25.5 GB/s), so it is reported, not fixed.
8. **End to end, the shape's effect is real but small and fully explained by §2.** 128K, identical window
   structure and identical token ids in every arm: **96 × 128 → 19.56 / 19.59 tok/s** (two runs; a third run of the
   same shape, the first arm of its session, measured 18.46 with a 140.08 ms window and a slow draft phase, and is
   the one outlier in the set) and **1 × 128 → 19.01 / 18.89 / 18.83** (three runs), i.e. the confined shape costs
   **~3.4% of decode**; 4 × 128 and 16 × 128 measure 19.49 / 19.59. The engine's own counters give the volume:
   **621 blocks = 2.50 MiB per window** (92.15% of 783,080 block reads resident, 247.8 MiB from RAM over 99
   windows). At the shipping 25.5 GB/s that volume is 103 µs of a 132 ms window (0.08%); at the confined shape's
   measured 1.07 GB/s it is 2.4 ms, and the arms measure ~4.5 ms. **Token ids are identical everywhere**
   (4K md5 `ac9f16fceed82e5681e4` 147/147, 32K `124a3cd39b33f7da31e2` 256/256, 128K `c60e72d397e9b1cd44da`
   256/256 — the same hashes P1/P2/P3 measured on the unchanged paths).

## 1. Task 1 — our I/O kernel, its shape, and its host memory

| what | where |
|---|---|
| the kernel | `copy_kernel`, `src/kernels/cuda/kv_stream.cu:161-171`: grid-stride over the miss list; per block, per run, `for (i = threadIdx.x; i < len/16; i += blockDim.x) dst[i] = src[i];` |
| the launch | `src/kernels/cuda/kv_stream.cu:242-246` in `kv_stream_resolve`; shipped **96 blocks × 128 threads** |
| who calls it | `src/core/layer.cpp:726` (`qsa_kv_resolve`) → `src/core/verify.cpp:999` (once per QSA layer per decode window) and `src/core/layer.cpp:948` |
| the driver kernel | `resolve_kernel` (`kv_stream.cu:82`), **1 block × 1024 threads** (`:245`): walks the selected cells, claims misses (`page_table -1 → -2`), clock-evicts one victim slot per miss |
| host side | **pinned, device-mapped** — `cudaHostAlloc(..., cudaHostAllocMapped \| cudaHostAllocPortable)` + `cudaHostGetDevicePointer` (`src/core/layer.cpp:634-635`); in the shim that is `sycl::malloc_host` and the UVA identity (`include/strata/sycl_compat/cuda_runtime.h:840-869`) |
| host layout | the identity layout of a fully resident pool: four arrays per layer, `[block][kv_head][page_size][head_dim]` per run (`include/strata/kernels/kv_stream.hpp:30-41`) |
| granule | one page = 4 cells = **4,224 B in int8**: 2,048 + 2,048 + 64 + 64 (k codes, v codes, k scales, v scales), run offsets 0 / 2048 / 4096 / 4160 |
| the other I/O path | `kv_stage_from_host` (`kv_stream.cu:244`, one `cudaMemcpyAsync` per run) stages the prompt's blocks into a resident pool — **the prompt path never runs the kernel** |
| the other direction | our append path is `kv_append_*` (a device kernel writing the pinned host copy); the probe's D2H arm is the same `copy_kernel` with source and destination exchanged |

So the mechanism is present, pinned, correct and fast — **but only on the decode miss path**, where the engine's own
counters put it at 621 blocks / 2.50 MiB per window at 128K (an earlier 128K run measured 3.20 MiB/window; same
order). `STRATA_KV_COPY_BLOCKS` / `STRATA_KV_COPY_THREADS` now override the launch shape once per process; unset
keeps 96 × 128 (previous behaviour, byte for byte). Regression evidence for the change: `kv_stream_parity` passes
(306 batches, 89.9% hit, bitwise-equal attention output) and the full suite serially is **5 failed out of 49 — the
same five the parent HEAD failed** (`platform_memory_test`, `elementwise_parity`, `quantize_act_parity`,
`iq_multi_parity`, `expert_multi_test`; 90.0 s against P3's 91.5 s). Under `ctest -j 4` a sixth,
`sycl_handoff_test`, fails in 1.6 s; it passes serially, which is the load-sensitivity P3's own suite run also
showed (P3 ran it serially too).

## 2. Task 2 — bandwidth vs concurrency (raw)

```
H2D  1 block x1024 = 1.07 GB/s    2 = 2.15    4 = 4.36    8 = 8.91    16 = 17.00
H2D 32 = 21.90   64 = 24.65   96 = 25.64   128 = 25.77      (each 132 MiB per pass, 2-28 passes)
H2D  1 block x 128 = 1.07        2 = 2.12    4 = 4.42    8 = 8.96    16 = 17.13   32 = 21.93   64 = 24.65
H2D 96 blocks x  128 threads = 25.55   256 = 25.55   512 = 25.49   1024 = 25.63
D2H  1 block x1024 = 2.72    2 = 5.07    4 = 9.10    8 = 16.99   16 = 19.34
D2H 32 = 21.21   64 = 22.56   96 = 21.98   128 = 22.23      (96 x 128 = 22.57; 256 = 22.59; 512 = 22.59)
D2H  1 block x 128 = 2.74    2 = 5.10    4 = 9.15    8 = 17.01   16 = 19.40   32 = 21.38   64 = 22.52
```

(All from `p4/out-all-run1.txt`; the earlier bw run reproduced this sweep to ≤1.5%, e.g. the shipped shape at
25.40 GB/s and 1 block at 1.05.)

* Linear in blocks up to ~16 (**≈1.07 GB/s per block**), then a knee, saturating at **~64 blocks**.
* **Threads per block do not matter** — not at 96 blocks (25.55 at 128 threads against 25.63 at 1024) and not at one
  block (1.07 against 1.07). The limit is the work available per KV block (264 `uint4`) and the serial grid-stride
  loop, not the thread count: only 128 lanes have a `uint4` on a code run and only 4 on a scale run.
* Little's Law on our numbers: 128 lanes × 16 B = **2 KB in flight per block**, 1.07 GB/s per block ⇒ an effective
  ~1.9 µs per round trip. That is why 2 blocks is 8% of our link and ~78% of theirs (see §0.3).
* The 26.5 GB/s reference: 128 blocks × 1024 threads = 25.77, dense 256 KiB transfers = 26.83 (§4), one contiguous
  128 MiB DMA = 25.59 H2D / 28.55 D2H (§4) — all consistent.
* The point the engine arms need is `1 × 128 = 1.07 GB/s`: 2.50 MiB of KV per window would take **2.4 ms** there
  against 103 µs at the shipped shape (§6).

## 3. Task 3 — interference vs concurrency, and the quota

### 3a. The curve (the paper's axes)

I/O job = N passes of 132 MiB pre-submitted on its own in-order queue, sized so the I/O alone would take ~1.5x the
compute's solo time; compute = the engine's own kernels on a second queue; two reps per point.

| blocks × threads | prefill-like (prompt attention) | decode-like (Q5_K expert GEMV) |
|---|---|---|
| 1 × 1024 | norm **0.990**, I/O 0.68 GB/s | norm **0.393**, I/O 1.06 GB/s |
| 2 × 1024 | 0.991, 1.31 | 0.393, 2.12 |
| 4 × 1024 | 0.988, 2.63 | 0.398, 4.29 |
| 8 × 1024 | 0.996, 5.21 | 0.397, 8.66 |
| 16 × 1024 | 0.992, 10.14 | 0.400, 15.98 |
| 32 × 1024 | 0.988, 13.05 | 0.398, 20.66 |
| 64 × 1024 | 0.992, 14.71 | 0.398, 23.12 |
| 96 × 1024 | 0.989, 15.27 | 0.397, 24.03 |
| 128 × 1024 | 0.994, 15.62 | 0.399, 24.71 |

`norm` = solo / concurrent compute time (1.000 = untouched). Compute solo: prefill 524.7 ms (8 calls of the engine's
prompt attention), decode 503.0 ms (2,500 calls of the expert GEMV). The concurrent work is the engine's own
`qsa_prompt_attn_batch` (the port's default portable FP32 path, the one the engine itself takes with
`STRATA_SYCL_XMX` off) and `native_q5_k_f32`.

### 3b. Why the decode column is not interference

* `P4 conc` — the **same** full-GPU workload on two queues: spans come out 0.994 (prefill) and 0.996
  (decode), i.e. two in-order queues do not time-share this device for saturating work.
* `P4 order` — same I/O job (12 passes × 32 blocks, 66-78 ms alone) against the same compute:
  * prefill-like: compute-first → compute 263.6 ms (solo 271.0, **-2.7%**) and I/O 339.2 ms (5x its solo, starved);
    I/O-first → compute 279.6 ms (**+3.2%**) and I/O 75.8 ms (its solo).
  * decode-like: compute-first → compute 317.0 ms (solo 240.4, **+32%**) and I/O 83.2 ms (its solo); I/O-first →
    compute 260.1 ms (**+8%**) and I/O 77.3 ms.
* So the queues take turns in submission order, and how much the compute loses depends on its **kernel
  granularity**, not on the I/O's block count: a workload of a few 66 ms kernels absorbs the I/O's short passes
  between them (+3% at worst), while a workload of 1,200 small kernels pays up to +32% because each arrival queues
  behind an I/O kernel. The inter arm's decode column (0.39) is that effect with a larger I/O backlog
  (concurrent 1,265 ms ≈ I/O 784 ms + compute 504 ms): turn-taking, not SM theft — the normalised value is flat at
  0.392-0.402 from 1 block to 128, which no contention model would produce.
* The **prefill column is the usable one**: 0.53 s of the engine's prompt attention, unaffected (0.984-0.999) by an
  I/O kernel running at 0.68-15.7 GB/s throughout, at every block count.

### 3c. Quota: no confinement, and the file:line where that is applied

**The shape stays 96 × 128** — `src/kernels/cuda/kv_stream.cu:242-246` (`kv_stream_copy_shape()`'s default; env
overridable, unset = previous behaviour). Justification, from the measured curves rather than their number:

* §2: bandwidth is still climbing at 16-32 blocks, and their 2 × 1024 quota delivers **2.12 GB/s = 8% of our link**
  (8% of ours against 78% of theirs).
* §3b: there is no concurrent compute to protect — our copy runs on the layer's own in-order stream
  (`src/core/verify.cpp:999-1002`), so a confined kernel's *entire* (slower) duration lands on the critical path.
  Buying interference margin we cannot use, at 12x less bandwidth, is a pure loss.
* §6: the end-to-end measurement agrees — 1 × 128 costs 3.4% of decode; 4 × 128, 16 × 128 and 96 × 128 are equal
  within the harness's spread.

If the port ever *did* overlap the copy (a separate stream, the paper's bubble-filling arrangement), the largest
block count that keeps prefill-like work inside the paper's <5% bar is **any count 1..128 as measured here**;
the decode-like column cannot be read that way on this backend at all (§3b), because two queues serialize. A quota
is not the instrument this port needs.

## 4. Task 4 — transfer size / fragmentation

Chunk j at `src + j*stride`, one warp per chunk, 32 blocks × 1024 threads, 128 MiB per shape; stride = chunk
(dense) or 2*chunk (a hole as large as the unit, "scattered").

| transfer size | H2D dense | H2D scattered | D2H dense | D2H scattered |
|---|---|---|---|---|
| 16 B | 20.79 | 11.77 | **0.11** | **0.10** |
| 32 B | 23.52 | 11.81 | **0.07** | **0.07** |
| 64 B | 23.61 | 23.62 | 22.71 | 21.86 |
| 128 B | 25.30 | 25.31 | 22.72 | 21.81 |
| 256 B | 26.72 | 26.69 | 22.72 | 20.98 |
| 512 B | 26.71 | 26.73 | 22.72 | 22.72 |
| 4 KiB | 26.72 | 26.72 | 22.72 | 22.72 |
| 64 KiB | 26.73 | 26.72 | 22.72 | 22.52 |
| 256 KiB | 26.72 | 26.83 | 22.72 | 15.39 (see limits) |
| 1 MiB | 26.67 | 26.75 | 22.71 | 18.24 (see limits) |

* **Reads (H2D) are flat at ~100% of the link from 512 B up, and 128 B still holds 25.3 GB/s (95%).** The paper's
  "1-2 MB transfers are needed to hit 75-80% of PCIe 5.0" is not true of this link and this access pattern. The
  read side degrades gently below that (64 B 23.6, 32 B 23.5, 16 B 20.8 dense) and by ~half when sub-64 B units
  are *scattered* (the sub-cache-line request stops covering a full DRAM burst).
* **Writes (D2H) have a cliff between 32 B and 64 B: 0.07-0.11 GB/s below it against 22.7 GB/s above — a ~250x
  step.** A device-side write to pinned host memory needs at least one 64 B cache line per request; below that the
  write path collapses. This is the sharpest mechanism limit found in this card, and the port's append path is well
  clear of it (it writes a cell's K/V — hundreds of bytes — per coalesced stripe).
* **Our 4-cell page (4,224 B in four runs) sits in the flat region** — measured through the engine's own kernel at
  25.5 GB/s H2D (§2). Its 64 B scale runs are at the write cliff's knee and below the read knee, but they are 3% of
  a block's bytes and the aggregate is unaffected.
* The paper's ~22%-of-peak at page 32 is about a paged page *scattered* across per-head arrays; ours is the
  identity layout, which is why we see ~100% at 4 KiB. The difference is the finding.

**The DMA road, for comparison** (`P4 dma`: one `cudaMemcpyAsync` per scattered chunk — "repeated small
cudaMemcpyAsync calls", the thing the paper replaces with a kernel). The host submission cost is a flat
**3.8 µs per call**, so the achieved rate collapses with the unit size:

| unit | 128 B | 512 B | 4 KiB | 64 KiB | 256 KiB | 1 MiB | one 128 MiB call |
|---|---|---|---|---|---|---|---|
| H2D GB/s | 0.03 | 0.13 | **1.05** | 10.33 | 19.67 | 24.53 | 25.59 |
| D2H GB/s | 0.03 | 0.13 | **1.07** | 12.65 | 21.95 | 26.43 | 28.55 |

At **our actual granule, 4 KiB, the DMA road delivers 1.05 GB/s against the kernel's 25.5 GB/s — a 24x
difference**, which is the paper's fragmentation lesson reproduced on our silicon with a like-for-like comparison
(and supersedes the historical 1.59 GB/s host-side scatter figure, which came from a different harness). It also
says why the port does not need the mechanism on the *prompt* path: `kv_stage_from_host` issues one memcpy per run
per layer (≈192 calls of MBs each), which sits at the top of this table.

## 5. Task 5 — the layout transformation

| layout (H2D at 96 × 128 / 32 × 1024) | GB/s | D2H at 96 × 128 / 32 × 1024 |
|---|---|---|
| identity (shipped) | **25.58** / 21.67 | 22.30 / 21.23 |
| page-first host (one allocation, a block's four runs adjacent) + identity device | 24.15 / 21.05 | 22.54 / 21.26 |
| page-first both sides (upper bound; the readers' device layout would change) | 24.15 / 21.05 | 22.47 / 21.25 |

**No gain, and 5.6% worse at the shipped shape** (24.15 against 25.58; the packed variants are equal to each other
to 3 digits, so the device-side half of the transformation is worth nothing at all). The four runs are already
sequential in their own arrays, so the kernel reads four sequential streams; packing them into one 4,224 B block
interleaves two long runs with two 64 B ones and loses that. The transformation cost is nil (one extra multiply per
block — the same `Runs` struct with `src_stride[a] = block_bytes`), so there is nothing to trade. **Not adopted**;
the paper's 4x disk-loading win came from a page-first-vs-page-strided change that does not describe our host copy.
The `P4 verify` cases confirm the packed kernel moves the right bytes in both directions (0 mismatches over
67,108,864 checked bytes each), so this is a real null, not a broken variant.

## 6. Task 6 — applied, and end to end

Two facts decide what is measurable end to end:

* **At 4K and 32K the copy kernel never runs.** `--kv-resident 32768` keeps every layer fully resident up to 32K
  cells (`qsa_stream_policy`, `src/core/layer.cpp:497-515`), `kv_mode` is 0, `kv_stream_resolve` is never called and
  the arms print no KV-streaming line at all. Those lengths are therefore an **equality control** (§6b).
* **At 128K streaming is engaged** (32,768 of 131,072 cells resident per QSA layer, 0.64 GiB pinned) and the copy
  runs ~621 times per window (2.50 MiB).

### 6a. The shape arms at 128K (both GPUs, config of record, 129,024-token prompt, 256 tokens)

| arm | shape | prefill ms (tok/s) | decode ms (tok/s) | window ms | verify | GPU-reach wait | draft | ids md5 |
|---|---|---|---|---|---|---|---|---|
| p4-128k-default | 96×128 | 375,653.6 (343.5) | 13,867.7 (**18.46**) | 140.08 | 117.79 | 42.21 | 20.11 | c60e72d3… |
| p4-128k-default2 | 96×128 | 372,699.0 (346.2) | 13,090.2 (**19.56**) | 132.22 | 110.80 | 42.72 | 19.46 | c60e72d3… |
| p4-128k-1x128 | 1×128 | 372,488.4 (346.4) | 13,469.0 (**19.01**) | 136.05 | 114.62 | 44.69 | 19.43 | c60e72d3… |
| p4-128k-1x128b | 1×128 | 372,716.3 (346.2) | 13,548.6 (**18.89**) | 136.85 | 115.37 | 44.99 | 19.42 | c60e72d3… |
| p4-128k-4x128 | 4×128 | 372,612.3 (346.3) | 13,134.5 (**19.49**) | 132.67 | 111.26 | 42.83 | 19.45 | c60e72d3… |
| p4-128k-16x128 | 16×128 | 372,654.1 (346.2) | 13,065.5 (**19.59**) | 131.98 | 110.55 | 42.44 | 19.46 | c60e72d3… |
| p4-128k-amp-default | 96×128, `--kvres 20480` | 372,413.2 (346.4) | 13,069.8 (**19.59**) | 132.02 | 110.58 | 42.36 | 19.47 | c60e72d3… |
| p4-128k-amp-1x128 | 1×128, `--kvres 20480` | 372,570.7 (346.4) | 13,597.6 (**18.83**) | 137.35 | 115.91 | 45.59 | 19.43 | c60e72d3… |

Readings:

* **Prefill is identical across arms** (372.4-372.7 s, spread 0.08% — and the one 375.65 s arm is the first arm of
  its session, +0.8%). Prefill does not touch the copy kernel, so this is the harness's own noise floor and the
  evidence that the machine state is the same in every arm.
* **Decode separates by shape, not by noise**: 96 × 128 = 19.56 / 19.59 across two sessions; 1 × 128 = 19.01 /
  18.89 / 18.83 across three. That is **-3.4% for the confined shape** (the first 96 × 128 arm, 18.46, is the one
  outlier: its whole window is 140.08 ms with a slower draft phase, 20.11 ms).
* 4 × 128 (19.49) and 16 × 128 (19.59) match the shipped shape within noise, i.e. the knee in §2 is real in the
  engine too: **4 blocks are enough, 1 is not.**
* The cost matches §2's volume arithmetic: 2.50 MiB/window at the shipped 25.4 GB/s = 103 µs (0.08%); at the
  measured `1 × 128` rate of 1.07 GB/s (§2) it is 2.4 ms, and the arms measure ~4.5 ms of extra window
  (136.3 against 132.1 ms, i.e. 3.4% of decode; 19.06 against 19.58 tok/s over the repeats). The two independent
  measurements agree to a factor of 1.9 — a confined block is 25-40x slower than the shipped shape, and that is what
  the extra time is.
* **The amplified pair** (`--kvres 20480`, the engine's floor) raises the miss volume only 1.24x (308.5 against
  247.8 MiB from RAM; 90.22% against 92.15% hit) because the selection's working set is recency-dominated — so the
  pair's separation is the same 4.0% (19.59 against 18.83), which is a useful negative result about where the
  streaming's cost does *not* scale.

### 6b. The 4K/32K controls, token-id equality

| arm | shape | context | prefill ms (tok/s) | decode ms (tok/s) | ids | md5 |
|---|---|---|---|---|---|---|
| p4-4k-1x128 | 1×128 | 4K, needle prompt, 150 max-new | 15,154.4 (252.8) | 6,815.2 (21.57) | 147 | `ac9f16fceed82e5681e4` |
| p4-32k-1x128 | 1×128 | 32K | 91,905.7 (350.9) | 11,725.3 (21.83) | 256 | `124a3cd39b33f7da31e2` |

Both md5s are **exactly the hashes P2/P3 measured at those lengths on the unchanged paths** (4K 147 ids, 32K 256
ids), and both are at lengths where no copy kernel runs — the changed shape reproduces the unchanged path's text.
At 4K/32K the arms are unchanged by construction (`kv_mode == 0`), which is stated rather than measured: the 32K
arm's window (137.94 ms, 85 windows) and 4K's (141.98 ms, 48 windows) sit on P3's same-config lines
(138.29 / 142.23 ms).

### 6c. What is applied

**Nothing in the engine's behaviour changed**: the shipped shape stays 96 × 128 and is now env-overridable
(`src/kernels/cuda/kv_stream.cu:242-246`), the layout is unchanged (§5), and the transfer-size and quota
conclusions above say the copy is not where this port's remaining 2.4x decode gap lives (P3 reached the same
conclusion from the submissions side). **The measured null/3.4% is the result.**

## 7. Raw evidence

* `p4/evidence/P4-EVIDENCE.txt` — the final probe run (`P4 ...` lines, all sections) plus every engine arm's
  summary lines (DONE, decode timing, KV streaming, expert cache), assembled by `p4/evidence.sh`.
* `p4/evidence/arms-table.txt` — P3's extractor run over all P4 arms (decode timing, DONE, token-id count + md5,
  peak RSS/VRAM).
* Earlier probe runs kept deliberately: `p4/evidence/probe-bw-earlier.txt` (the bw sweep before the 128-thread rows
  were added — reproduced the final numbers to ≤1.5%), `p4/evidence/probe-chunk-threadper-kernel-earlier.txt` (the
  transfer-size sweep with the discarded one-thread-per-chunk kernel, i.e. the 0.10 GB/s access-pattern finding),
  `p4/evidence/probe-inter-v1-discarded.txt` (the first interference harness, whose I/O loop synchronised per pass
  and therefore starved its own I/O — superseded by the §3 harness and kept as the record of why).
* `p4/chain-128k.log`, `p4/chain3.log`, `p4/chain4.log`, `p4/arms.log` — the arm runs, in order.
* `p4/evidence/ctest-p4-serial.txt` (5 failed of 49, the pre-existing five, 90.0 s), `p4/evidence/ctest-p4-parallel.txt`
  (`ctest -j 4`: the same five plus `sycl_handoff_test`, which passes serially), `p4/evidence/build-probe.txt`,
  `p4/evidence/arms-invocations.txt` (the arm lines: tag, shape, ctx, binary hash). The `p4/*.log` files those were
  copied from are ignored by the repo's `*.log` rule and live beside them locally.

## 8. Not validated / honest limits

* **One card at a time for the mechanism curves** (`ZE_AFFINITY_MASK=0`); the engine arms use both cards as the
  config of record requires. Nothing else was on either GPU.
* **One or two runs per arm.** The decode spread is what the arms show: same-shape repeats differ by 0.6% (1×128,
  two runs) and by 6% (96×128, two runs, one of which is the outlier above), so a 3.4% separation is real but a
  *single* pair could not have established it. The prefill column (0.08% spread) is what establishes the machine
  state is comparable.
* **The 3.4% is the confined shape**, i.e. one block at the port's 128 threads — not the paper's 2 × 1024, which §2
  puts at 2.12 GB/s (a predicted ~1.5 ms/window, ~1%).
* **`P4 conc`'s ratio cannot prove overlap for two full-GPU workloads** (the SM count is the limit); it is used here
  only to show that two saturating workloads on two queues give no speedup at all.
* **The inter arm's decode column is order-dominated** (§3b) and is reported with its controls, not as a quota input.
* **`P4 chunk` uses one warp per chunk.** A version with one *thread* per chunk measured 0.10 GB/s at every size
  (16 B stores scattered 32 ways) — that is the access-pattern component of the mechanism, and the production kernel
  is on the good side of it (one `uint4` per thread on a contiguous thread stripe).
* **`P4 hostcopy` is RAM→RAM, single-threaded** (8.9-17.6 GB/s; see evidence). It is the CPU's copy over the same
  pinned memory, not a PCIe transfer, and it does **not** reproduce the 1.59 GB/s host-side scatter figure — that
  came from scattered reads over a much larger arena. `P4 dma` is the comparable control.
* **Two D2H scattered points read low** (256 KiB 15.41, 1 MiB 18.23 against their neighbours' 22.7). Reported as
  measured and flagged unexplained rather than smoothed away.
* **The `--kvres 20480` amplification is only 1.24x**, so it does not turn the null into a large signal; the
  negative result (the streaming cost is not sensitive to the resident-set size in this range) is stated as such.
* **The engine's own copy time is not isolated by a counter**: what is measured is the window and the decode rate,
  and the copy's share is inferred from the measured volume and the §2 bandwidth curve.
* Nothing pushed: origin (`github.com/Niko1221/Strata`) has no `sycl-xpu` branch.
