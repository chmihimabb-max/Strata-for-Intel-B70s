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
- Token generation is **GPU-bound per layer** (the verify is the bulk of a decode window; the 42–47 ms "GPU-reach wait" is
  the GPU, not the host). The port's SYCL `command_graph` replay path is **on by default** (`STRATA_SYCL_GRAPH=0` turns it
  off): ~95% fewer submissions and **−3.0 / −3.9 / −4.7% of the window** at 128K / 32K / 4K, greedy tokens byte-identical.
- The web dashboard's **GPU telemetry is implemented for Arc** (`serve/telemetry.py`, an Intel backend on the `xe` driver:
  load from `drm-cycles-ccs` / `drm-total-cycles-ccs`, VRAM client-deduplicated from `drm-total-vram0` against the PCI BAR
  2 aperture, temperatures by hwmon label, power from the `energy1_input` counter across the sampler interval). PCIe link
  gen / width / throughput have **no sysfs source on this driver and are reported as `null` with the reason, never as 0**.
- Works on **this** hardware and configuration. This is not a general Intel support claim.

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
  `sycl::ext::oneapi::experimental::command_graph` behind `STRATA_SYCL_GRAPH=1`.
- **The verify window's release had to be re-derived.** Upstream writes host-mapped words the spin kernels read; on this
  backend a poll over mapped host memory does not observe the host's stores, so the words are device memory published by a
  submitted copy on the verifier's own stream.
- **The served path's mid-prompt checkpoint had to be device-corrected.** Under the layer split, the save copied a card-1
  running state onto card 0's queue (one shared SYCL context, no peer path), so any prompt crossing 16,384 tokens returned
  `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`; the save now asserts the stage's own device.
- **Cold-start JIT was moved out of the request.** On a fresh program cache the first decode windows built the dense MMVQ
  specializations in-flight — measured as 127.93 ms/window of verify time that no counter owned, over 54 windows — so the
  serve path warms them before "everything loaded" (`native_mmvq_warmup` + `decode_warmup`: 192 + 54 launches, ~45 ms warm).
- Intrinsics, sub-group widths and memory limits are mapped per the CUDA-surface contract; the two-GPU split, per-layer
  expert-cache slabs and the pinned-arena strategy all differ from CUDA in mechanism while preserving the guarantee.

The engineering record lives in-tree: `p1/` … `p10/`, `m6c/`, `i2/`, and the decode campaign's `d1/` … `d3/` with `s4/`,
`s5/` (status reports and evidence per work item), plus the port's own probe scripts.

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
