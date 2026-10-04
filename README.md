# Strata for Intel Arc B70s — a SYCL / oneAPI port

A community port of **[Strata](https://github.com/Niko1221/Strata)** to **Intel Arc Pro B70 (Battlemage, `BMG-G31`)** via
**SYCL / oneAPI**, in place of the upstream CUDA and HIP backends.

Strata is the engine that runs **Qwen3.8-Flash-Next** — a 125-billion-parameter mixture-of-experts model, 24,576 experts,
~10 used per token — on a single PC by keeping the hot experts in VRAM, all of them in RAM, and computing the rest on the
CPU in place. This port makes that work on Intel GPUs.

**Status: working end to end on two Arc Pro B70s.** The Strata-recommended **IQ3_S** GGUF is served at the model's full
**262,144-token context with KV streaming**, and its output is validated token-for-token against an independent
llama.cpp-SYCL reference running the *same weights on the same cards*.

## Measured performance

Two Arc Pro B70 (31.9 GiB each), 123 GB RAM, `ZE_AFFINITY_MASK` unset, two-GPU layer split, `--kv int8`,
KV streaming above 64K, MTP speculative decoding on. Model: `IQ3_S` (see below).

| context | prompt processing | token generation |
|---:|---:|---:|
| 32K | 344.8 tok/s | 21.71 tok/s |
| 64K | 339.6 tok/s | 18.67 tok/s |
| 128K | 342.6 tok/s | 19.55 tok/s |
| 262K | 316.9 tok/s | 18.45 tok/s |

- **256K end to end**: a 259,943-token prompt read at 316.9 tok/s (TTFT 820 s), 256 usage-counted tokens generated.
- **Expert cache 100% GPU-resident** — 24,576 pairs across the two cards, 0.00 CPU expert entries per layer-window.
- **KV streaming** does the work of fitting the window: 32,768 of 262,144 cells resident per QSA layer, **90.99% of
  790,765 block reads served from VRAM**, 287 MiB from RAM, **12,672 B of pinned RAM per context token**.
- **Independently validated**: 182/184 comparable tokens identical to the llama.cpp-SYCL reference at `--kv int8`, 197/199
  at `--kv fp16`, every divergence attributed by measurement (one is the int8-KV trade, one a 0.07-nat coin flip).
- **Correctness units**: `ctest` 44/49 (the 5 failures are environmental on this box — `mlock` limits, missing AVX-512 —
  and unchanged since the recorded baseline); `kv_stream_parity` is **bitwise identical** streamed vs resident for
  int8 / fp16 / q4_0, and the decode-path parity tests (`mmvq_multi_parity`, `gr_parity`) are bit-exact to the shipped
  kernels.

## Known gaps (honest list)

- **Prompt processing is ~4.3x and token generation ~2.4x behind** Strata's published IQ3_S figures (measured on a 12 GB
  RTX 5070). The cause is attributed: the QSA prompt-attention kernel is **75.4% of the prefill's GPU timeline** and is
  bound by its **block shape**, not by arithmetic — and the engine's own 6.5x-faster batched variant **diverges from the
  reference** (0.17/0.14-nat margins) where the shipped kernel is byte-identical, so it is deliberately not enabled.
- On this card only **int8 8x32x16 → int32** (and bf16/f16 tile) XMX shapes exist; **4-bit cannot reach the tensor cores**
  at all.
- Token generation is **GPU-bound per layer**: the verify window is ~90% of a decode window, and the **GPU-reach wait
  is ~70% of the whole window** (94.81 ms of a 133.59 ms window at 4K — the 42–47 ms figure was one stage's half of
  it). The port's SYCL `command_graph` replay path is **on by default** (`STRATA_SYCL_GRAPH=0` turns it off): ~95%
  fewer submissions and **−3.0 / −3.9 / −4.7% of the window** at 128K / 32K / 4K, greedy tokens byte-identical.
- The web dashboard's **GPU telemetry is implemented for Arc** (`serve/telemetry.py`, an Intel backend on the `xe` driver:
  load from `drm-cycles-ccs` / `drm-total-cycles-ccs`, VRAM client-deduplicated from `drm-total-vram0` against the PCI BAR
  2 aperture, temperatures by hwmon label, power from the `energy1_input` counter across the sampler interval). PCIe link
  gen / width / throughput have **no sysfs source on this driver and are reported as `null` with the reason, never as 0**.
- Works on **this** hardware and configuration. This is not a general Intel support claim.

## Results by workstream

Beyond the end-to-end block above, each mechanism was measured on this box and written up per work item. The headline
of each, grouped and newest-first:

**Serving the model**
- **The two-GPU layer split** — *S1*. Layers 0–22 on card 0 and 23–47 + the head on card 1 give **exactly the one-card
  tokens**, with **2.09x the experts resident in VRAM** and CPU expert work down to a fifth, at ~1.9x the prefill cost.
  The two cards' host→device links are unequal (26.5 vs 13.4 GB/s) and the engine prints both.
- **First-request JIT, named and removed** — *S2*, *S3*. The cold-start cost was the DPC++ runtime JIT-compiling the
  prompt path, 12.9 s of it a **single program unit** (the QSA prompt-attention kernels). A persistent program cache
  takes a 165-token prompt from 16.0 s to 1.85 s (**8.3x TTFT**); `-fsycl-device-code-split=per_kernel` had to reach
  the *link* step for the device image to be split at all (that unit: 3,243 KiB / 12.88 s → 405 KiB / 1.64 s).
- **The served path** — *S4*, *S5*. A 32K request on the split returns `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` unless
  the mid-prompt checkpoint part is copied on the device that owns the state (fixed). And `--prompt-cache` **moves the
  served greedy ids** (4K and 32K, split and one card): the trigger is the *segmentation of the prompt read*, not the
  checkpoint — the no-boundary and last-token-boundary arms prove the save is inert, and a one-token tail is enough.
  Use `--prompt-cache 0` for byte-identical *and* comparable repeats.

**The decode window**
- **Graph replay as the default** — *D1*, *D2*. `sycl::command_graph` removes ~95% of a window's ~3,841 submissions and
  **−3.9 / −3.0 / −4.7%** of the window at 32K / 128K / 4K, greedy ids byte-identical. The spec width was swept, not
  assumed: `--spec 4` is best, and the drafter's confidence floor `--spec-min-p 0.7` is **+4.6% at 4K / +3.2% at 32K**
  (ids identical) and is now in the config of record.
- **What a window spends it on** — *D2*, *D2x*, *D2y*, *D3*. The dense per-matrix MMVQ is **40.9%** of a 4K window (301
  launches), running at **8.8%** of the card's measured read rate — **11.4x above its own bandwidth floor** — and the
  gap is the *arithmetic*, not the bytes (dot removed: 589 vs 59 GB/s, 10.0x), because this device has no int8 dot
  instruction. The int8 8×32×16 XMX dot is **refused by the driver**; the one tile that builds (f16 8×16×16) buys 2.94x
  but is capped at 4.23x by its 2 bytes/element, and "more rows per block" is monotonically worse. Cost grows with
  depth (the score sweep 7.0x from 4K to 32K) while the top-k is flat; the sharing and per-layer-sync arms are
  −1.0%/−0.2% and +2.9%/+3.3% (ids identical, not adopted).
- **Cold-start kernels moved out of the request** — *D2b*, *D2c*. The decode path was building its (quant-type × ncols)
  specializations *inside the first decode windows*; a load-phase warm-up drops a cold 4K first run from 93 programs to
  15 in the decode phase and the residual from 35.5 to 9.5 ms/window, ids reproduced.

**The verify window, attention and I/O**
- **The verify window, attributed** — *P9*. At 4K the window of record is **133.59 ms = verify 120.06 (89.9%) +
  commit/emit 2.03 + draft 11.49**, and the **GPU-reach wait is 94.81 ms — 71.0% of the whole window** (the 42–47 ms
  quoted earlier was one stage's half). The chrome device trace **cannot** see this window — device instrumentation
  breaks the host↔device flag ping-pong and it never completes under the tracer — so the engine's own stage profiler
  (dead three ways on this device) was fixed instead.
- **Release and resilience** — *P1*, *P1b*. The verify window's release was a SYCL-port artifact, not upstream (the CUDA
  original had no copy at all), and it was both *blocking and failing* the host. It is now device-published, and the
  engine comes back from a failed window.
- **Which attention path** — *P5*. At matching KV precision the shipped prompt-attention kernel reproduces an
  independent llama.cpp-SYCL reference **exactly** (148/148 at 4K, 256/256 at 128K); the engine's own faster batched
  variant (**2.21x** end to end, **6.5x** at the kernel) diverges once at each, so the fast path stays **refused**, on
  correctness.
- **GPU-assisted I/O** — *P4*. The port's `copy_kernel` (96 blocks × 128 threads, 16 B/thread, pinned device-mapped
  host memory) already runs at **25.55 GB/s = 96%** of the box's measured 26.5 GB/s; the paper's 2-block quota would
  cost **12x**. Interference with concurrent compute stays inside the paper's <5% bar at every block count, transfer
  size is flat to 512 B, and the paper's page-first host layout is a measured null here.
- **One card, and "why not two instances?"** — *P10*. On one B70, `--expert-cache auto` keeps **52–54%** of the experts
  in VRAM and the CPU pool computes the rest on **14.6–15.8 cores** — at **−4.9% / −15.5% decode** (32K / 128K) and
  **−25.8% / −40.3% prefill** against the split, with answers no longer bit-identical. Two one-card instances reach
  **24.5 vs 22.6 tok/s aggregate (+8%, not +100%)** — the shared RAM path eats the parallelism.

**Validation, tooling, and one alternative that was dropped**
- **Independent oracle** — *I2*. The same IQ3_S file on the same cards under llama.cpp-SYCL, greedy on the same token
  ids: 40/40 on the two prompts that matter, every divergence attributed to the oracle's own top-2 margin.
- **Drafter provenance** — *P7*. The resident server **cannot** run headless (`--serve` refuses without `--mtp`); the
  drafter is rebuilt upstream's way from the pinned canonical checkpoint (SHA256-verified per tensor, 31/31) and is
  byte-identical to the earlier artifact. Running without it costs **−39% decode** at 4K.
- **unitrace** — *S3UT*. Every device-instrumenting mode deadlocks the verify window; `--host-timing` alone measures it.
- **W4A16 (Q4-class): explored, then dropped** — *W1*, *W2*, *M5*. The Intel `W4A16-AutoRound` checkpoint was repacked
  to ggml Q4_0 and run on Arc with a bit-exact CPU parity oracle, but it emits degenerate text where IQ3_S emits
  coherent English at the same HEAD, prompt, path and cards — a defect in that pack/native path, not the SYCL backend.
  It was dropped by choice; **IQ3_S is the configuration of record.**

## Build

Requires **oneAPI 2026.1** (icpx + oneMKL) and CMake. The SYCL backend is mutually exclusive with the CUDA and HIP
backends, and `.cu` sources are compiled as C++ (the launch syntax has no C++ spelling — see "What the port changes").

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
cmake --build build-sycl -j
ctest --test-dir build-sycl -R 'kv|qsa|parity'      # source setvars first, or the KV tests fail on libsycl.so.9
```

## Model and run

1. **Weights** — the Strata-recommended quant, unchanged:
   `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, subdirectory `IQ3_S/` (83.62 GB): shard 1 is the model, **shard 2 is the
   PLE table** (the GSQ-RCO layout, not Orca's).
2. **Drafter** — the GSQ-RCO GGUF ships no MTP head, so build the drafter the way upstream does, from the model's own
   canonical checkpoint (pinned revision, SHA256-verified per tensor):
   ```sh
   python tools/mtp_fetch.py inventory --out mtp && python tools/mtp_fetch.py fetch --out mtp
   python tools/mtp_fetch.py verify --out mtp                 # 31/31 tensors
   python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
   python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
   cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
   ```
3. **Serve** (see `strata-sycl-iq3s.json`, the configuration of record):
   ```sh
   ./build-sycl/strata --serve \
     --pack <pack-dir> --native <…IQ3_S-00001-of-00002.gguf> --ple-gguf <…IQ3_S-00002-of-00002.gguf> \
     --mtp mtp/rt --kv int8 --expert-cache auto --mmap-experts \
     --spec 4 --spec-min-p 0.7 --kv-resident 32768 --max-context 262144 --no-capture --stats
   # or with the HTTP server + dashboard:
   python -m serve.server --engine strata --config strata-sycl-iq3s.json --port 8099 --api-monitor
   ```
   The app is at `/`, the API monitor at `/api-monitor`. While it runs the server holds **both GPUs**.

> Note: `--max-context 262144` with KV streaming is the configuration these numbers were measured in; a fresh long prompt
> costs minutes of prefill. Lower `--max-context` for interactive use.

## What the port changes

- **No fork of the engine sources.** The backend is a compat header directory (`include/strata/sycl_compat/`, like the
  existing `hip_compat/`) plus a **build-time source transform** (`tools/sycl/syclify.py`) that emits `.cpp` from the
  unmodified `.cu` files with `#line` mapping, because `icpx` has no spelling for `kernel<<<grid, block, smem>>>`.
- **Five kernels are hand-ported** with named SPIR-V replacements where SYCL has no spelling for the CUDA intrinsic
  (three `mma.sync` f16 sites, `ldmatrix`, `cp.async` groups, `%globaltimer`), including a `m16n8k16` MMA emulation.
- **Graph replay**: SYCL has no stream capture, so the port implements the engine's graph path with
  `sycl::ext::oneapi::experimental::command_graph` — now **on by default** (`STRATA_SYCL_GRAPH=0` turns it off).
- **The verify window's release had to be re-derived.** Upstream writes host-mapped words the spin kernels read; on this
  backend a poll over mapped host memory does not observe the host's stores, so the words are device memory published by a
  submitted copy on the verifier's own stream.
- **The served path's mid-prompt checkpoint had to be device-corrected.** Under the layer split, the save copied a card-1
  running state onto card 0's queue (one shared SYCL context, no peer path), so any prompt crossing 16,384 tokens returned
  `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`; the save now asserts the stage's own device.
- **Cold-start JIT was moved out of the request.** On a fresh program cache the first decode windows built the dense MMVQ
  specializations in-flight — measured as 127.93 ms/window of verify time that no counter owned, over 54 windows — so the
  serve path warms them before "everything loaded" (`native_mmvq_warmup` + `decode_warmup`: 192 + 54 launches, ~45 ms warm).
- **Two compile-side fixes.** The DPC++ persistent program cache keeps the first-request JIT out of every later session
  (`SYCL_CACHE_PERSISTENT=1`), and `-fsycl-device-code-split=per_kernel` had to be passed to the *link* step for the
  device image to be split at all (the QSA prompt-attention unit: 3,243 KiB / 12.88 s → 405 KiB / 1.64 s, and a cold
  16K prompt from 74.2 s to 61.8 s).
- Intrinsics, sub-group widths and memory limits are mapped per the CUDA-surface contract; the two-GPU split, per-layer
  expert-cache slabs and the pinned-arena strategy all differ from CUDA in mechanism while preserving the guarantee.

The engineering record lives in-tree — one directory per work item, each with the status report and the raw evidence for
it: `p1/` … `p10/`, the decode campaign's `d1/` … `d3/` (with `d2a/` … `d2y/`), `i2/`, `i3/`, `m6c/`, `s4/` and `s5/` —
plus the port's own probe scripts.

## Credits and license

- **Upstream engine**: [Niko1221/Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors —
  without it there is nothing to port. The upstream README is preserved as [`README-UPSTREAM.md`](README-UPSTREAM.md).
- **Research paper**: *Strata: Hierarchical Context Caching for Long Context Language Model Serving* (arXiv:2508.18572) —
  GPU-assisted I/O, decoupled GPU/CPU layouts, cache-aware scheduling. This port applies those ideas to Arc.
- **Model**: Qwen3.8-Flash-Next by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
- **Built on** parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).

**License: the same as upstream Strata — [MIT](LICENSE)** (`Copyright (c) 2026 Niko1221 and the Strata contributors`).
The Intel/SYCL port's contributions are released under the same MIT terms. A few parts keep their own licenses, exactly as
upstream states: `third_party/ggml` (MIT), the web app's font (SIL Open Font License 1.1), and
`data/experimental-speed-projection` (Qwen Community License 1.0). The model weights are not part of this repository and
their own licenses apply.
