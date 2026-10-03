# P3 — decode: a decode window costs 3 841 submissions, and the graph-replay path the port lacked removes 95% of them (and 3-4% of the window)

Card `t_d8afe53f` (P3), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, base `fcbed4d`… the change is
`fcbed4d` (this card's own first commit). Nothing pushed (origin has no `sycl-xpu`). Config of record in every arm:
IQ3_S GSQ-RCO snapshot `ed59f920…`, **both B70s** (`ZE_AFFINITY_MASK` unset, `--layer-split auto` → layers 0-22 /
23-47), `--kv int8 --kv-resident 32768 --expert-cache auto --mmap-experts --prefill auto (8192) --spec 4
--spec-min-p 0.5 --no-capture --stats --prompt-cache 0`, MTP drafter `rt-q2_0` untouched, one engine at a time,
warm program cache (`sycl-cache/m6c`), counters on (`STRATA_SUBMIT_COUNT=1`) in **both** arms of every A/B.

Harness: `p3/p3_run_arm.sh <TAG> <BIN> <CTX> [m6c args]` (swaps the arm's binary into `build-sycl/strata`, so an
A/B of two binaries or of one binary with an env switch is the same command shape), `p3/p3_chain.sh` for the six
arms, `p3/extract.py` to pull the table out of `m6c/runs/<TAG>/{err,out,rss}.txt`. Raw output:
`p3/P3-EVIDENCE.txt` (committed as `strata/p3/evidence/P3-EVIDENCE.txt`).

## 0. Verdict

1. **A decode window costs 3 841.5 submissions** (4K and 32K, median of the T>=2 windows; 3 863 at 128K):
   stage 0 (23 layers) **1 769.5 = 1 197 kernels + 556 copies + 16 barriers**, stage 1 (25 layers) **2 072 = 1 307
   kernels + 750 copies + 15 barriers** — i.e. **76.9 and 82.9 submissions per layer per window**. That is the
   number the card's task 1 asked for, and it is depth-independent (4K, 32K and 128K agree to ~1%.
2. **The port now HAS the graph-replay equivalent** (task 3). oneAPI 2026.1 ships the stream-capture equivalent
   this backend was said to lack — the SYCL graph extension (`sycl::ext::oneapi::experimental::command_graph`),
   and the B70 reports `sycl::aspect::ext_oneapi_graph` — so a window's capture now records a **command_graph**
   and `cudaGraphLaunch` hands the whole window to the driver in **ONE submission**. Behind `STRATA_SYCL_GRAPH=1`,
   **off by default**, and the closure path is byte-for-byte the previous behaviour when it is unset.
3. **Measured, same binary, flag on against flag off: submissions per window 3 841.5 -> 184 (-95.2%), window
   142.23 -> 137.17 ms (-3.6%) at 4K, 138.29 -> 132.38 (-4.3%) at 32K, 132.38 -> 128.31 (-3.1%) at 128K;
   decode 21.5 -> 22.3, 21.8 -> 22.2, 19.5 -> 20.2 tok/s; and the greedy token ids are IDENTICAL at all three
   lengths (147/147, 256/256, 256/256)** — md5s `ac9f16fceed82e5681e4`, `124a3cd39b33f7da31e2`,
   `c60e72d397e9b1cd44da`, which are also the ids P1/P2 measured on the unchanged paths. This is a speedup, not a
   correctness change.
4. **The card's reading of the gap needs correcting, and the measurement says so.** "GPU-reach wait 42.71 ms —
   over a third of the window is the host waiting to reach the GPU" is true as a description of where the host
   is, but that wait is **not** submission overhead and cannot be bought back by cutting submissions: after
   removing **95.2%** of them, the GPU-reach wait is **unchanged or slightly higher** (4K 46.51 -> 47.30, 32K
   44.27 -> 45.26, 128K 42.41 -> 43.96 ms). The wait is the GPU's own per-layer time (the host is parked waiting
   for the doorbell; the doorbell is the GPU's router). What the graph removes is a separate, smaller term: the
   host's own submission work inside the window, measured at **4.1-5.9 ms per window** = **1.1-1.6 µs per
   submission**, exactly what the counter predicts.
5. **What is left after the graph: the window's own GPU timeline.** At 128K the window is 128.31 ms and
   `verify` is 106.83 ms of it (83%); the GPU-reach wait alone is 43.96 ms (34%). Decode on this port is GPU-bound
   per layer, not submission-bound at the window level. The remaining 184 submissions/window are **150 copies
   (memcpy)** plus **32 query barriers** plus the 2 graph submissions — the copies are the expert/hand-off
   copies submitted outside the recorded graph, in the pool and commit paths; at 1.4 µs each that is ~0.26
   ms/window, i.e. 0.2%: not worth chasing.

## 1. Task 1 — submissions per decode window, before any change

Raw, from the engine's stderr (`STRATA_SUBMIT_COUNT=1`); the counter lives in the shim
(`include/strata/sycl_compat/cuda_runtime.h`, `strata::sycl_compat::subs()`) and is printed per window **per
stage** in `src/core/verify.cpp` beside the engine's own GPU-reach wait. Every submission the engine makes
funnels through the shim (D2), so this is the host-side counterpart of the `zeCommandListAppend*` count a
unitrace report gives.

```
# 4K, T=4 window, arm p3-4k-closed (binary p3/strata-graph-P3, STRATA_SYCL_GRAPH unset)
strata submit: stage window T=4 pos0=3836 layers 0..22  (23): submitted 1764 (kernel 1197 memset 0 memcpy 556 barrier 11 event 0 hostfn 0 graph 0) + recorded 0; 76.7 per layer; GPU-reach wait 127.08 ms
strata submit: stage window T=4 pos0=3836 layers 23..47 (25): submitted 2066 (kernel 1307 memset 0 memcpy 750 barrier  9 event 0 hostfn 0 graph 0) + recorded 0; 82.6 per layer; GPU-reach wait 121.69 ms
# 32K, T=4 window, arm p3-32k-closed
strata submit: stage window T=4 pos0=32128 layers 0..22  (23): submitted 1764 (kernel 1197 memset 0 memcpy 556 barrier 11 event 0 hostfn 0 graph 0) + recorded 0; 76.7 per layer
strata submit: stage window T=4 pos0=32128 layers 23..47 (25): submitted 2066 (kernel 1307 memset 0 memcpy 750 barrier  9 event 0 hostfn 0 graph 0) + recorded 0; 82.6 per layer
```

Median over the T>=2 windows of each arm (`p3/extract.py`):

| arm | stage 0 (23 layers) | stage 1 (25 layers) | per layer | kernels | copies | barriers |
|---|---|---|---|---|---|---|
| 4K closed | 1 769.5 | 2 072.0 | 76.9 / 82.9 | 2 504 | 1 306 | 31.5 |
| 32K closed | 1 769 | 2 073 | 76.9 / 82.9 | 2 504 | 1 306 | 32 |
| 128K closed | 1 778 | 2 085 | 77.3 / 83.4 | 2 528 | 1 316 | 29 |
| 4K graph | 88.5 | 95.5 | 3.8 / 3.8 | 0 | 150 | 32 |
| 32K graph | 88 | 96 | 3.8 / 3.8 | 0 | 150 | 32 |
| 128K graph | 89 | 94 | 3.9 / 3.8 | 0 | 152 | 31 |

The window composition the card quotes (128K: `132.25 ms = verify 110.78 (GPU-reach wait 42.71 + per-layer host
0.46 + stage 1.37) + commit/emit 2.01 + draft 19.46`) **reproduces**: this card's own 128K closed arm reads
`132.38 ms = verify 110.94 (GPU-reach wait 42.41 + per-layer host 0.48 [plan 0.30 actq 0.21 jobs 0.23 CPU 0.00] +
stage 1.71) + commit/emit 1.98 + draft 19.46`.

## 2. Task 2 — what was cut, and the `zeCommandListCreateImmediate` share

### Change 1 (the one that worked): graph replay instead of 3 841 submissions — commit `fcbed4d`

What it is: the shim's capture contract had two halves, and only one was real. `--capture` records the window as
a **list of closures** (`capture_append`), and `cudaGraphLaunch` then walks that list and calls `q.submit()` once
per node — a faithful *capture*, but not a *replay*: a window cost exactly as many submissions as it had kernels.
That is the whole of task 2's "stop recreating command lists per launch": on this backend the per-submission cost
is the driver's immediate-append path, so the only way to stop paying it 3 841 times is to hand the driver the
window as ONE command list. The SYCL graph extension does that:

* `cudaStreamBeginCapture` creates a `command_graph<modifiable>` and calls `begin_recording(queue)`; the engine's
  body then runs **unchanged** — `launch()`, `cudaMemcpyAsync` and `cudaMemsetAsync` see recording is open and
  submit normally, which is what SYCL records (**nothing executes** while recording, as CUDA requires);
* `cudaStreamEndCapture` closes the recording and hands back a tagged graph handle;
* `cudaGraphInstantiate` calls `finalize()` — measured 5-7 ms per window shape, once per shape per process
  (the engine records one graph per T, 1..4, per stage, so <=8 one-time finalizes);
* `cudaGraphLaunch` submits the executable graph in **one** `q.submit(h.ext_oneapi_graph(exec))`;
* USM buffers are re-read at replay, so a later window with different data in the same buffers is correct, and
  the engine's `capture(T)` records each shape once and replays it for every window, exactly as before;
* a refusal throws and is reported as `cudaErrorStreamCaptureUnsupported` (with the driver's message on stderr)
  rather than silently falling back to a body that already ran.

Off by default: `graph_mode()` is false unless `STRATA_SYCL_GRAPH!=0`, and it also checks
`sycl::aspect::ext_oneapi_graph` before promising anything. With the flag unset the closure list is used exactly
as before, so the default path is unchanged (the full ctest suite below is the evidence).

Before a line of engine code was touched, the extension was spiked standalone on the B70
(`p3/spike_graph.cpp`, a kernel shape with a dynamic `local_accessor` because the shim always emits one):
recording 256 kernels costs **0.3 ms (1.3 µs/kernel)**, `finalize()` 5.0-7.0 ms, replay **210-217 µs/window against
681-695 µs/window** for the same 256 kernels submitted one by one = **3.2x on the submission cost**, bit-identical
(max abs difference 0.000e+00 over 4 096 elements), and a second replay after a `memset` of the output is
identical too — i.e. the buffers really are re-read.

Measured effect, same binary (`p3/strata-graph-P3`, sha256 `85ac6232…`), flag off against flag on:

| length | submissions/window | ms/window | decode tok/s | greedy token ids |
|---|---|---|---|---|
| 4K | 3 841.5 -> 184 | 142.23 -> 137.17 (**-3.6%**) | 21.5 -> 22.3 (**+3.7%**) | 147/147 identical, md5 `ac9f16fceed82e5681e4` |
| 32K | 3 842 -> 184 | 138.29 -> 132.38 (**-4.3%**) | 21.8 -> 22.2 (**+1.8%**) | 256/256 identical, md5 `124a3cd39b33f7da31e2` |
| 128K | 3 863 -> 183 | 132.38 -> 128.31 (**-3.1%**) | 19.5 -> 20.2 (**+3.2%**) | 256/256 identical, md5 `c60e72d397e9b1cd44da` |

The 4K/32K token ids are byte-for-byte the ids the same prompt produced on the pre-P3 binary too
(`p3-4k-base`: `ac9f16fceed82e5681e4`, 147/147), and the 128K md5 is the one P2 measured (`124a3cd3…` /
`c60e72d3…`), so the graph path also reproduces the P2 baseline's text.

At 32K the end-to-end delta (+1.8%) is smaller than the per-window delta (-4.3%) for a measurable reason and it
should be stated: the graph arm needed **87 windows for 256 tokens against the closed arm's 85** (2.94 against
3.01 tokens/window, 172 against 171 accepted drafts of 214, avg T 3.46 against 3.52). The draft policy picks T
from measured acceptance *and window cost*, so making the window cheaper changes its choice; the change under
test is the window, and the window is what moved.

Raw, both arms:
```
p3-4k-closed  strata decode timing: 48 windows, avg T 3.79, 3.06 tokens/window, 142.23 ms/window = verify 121.04 (GPU-reach wait 46.51 + per-layer host 0.52 [plan 0.31 actq 0.25 jobs 0.27 CPU 0.00] + stage 1.37) + commit/emit 2.07 + draft 19.13
p3-4k-graph2  strata decode timing: 48 windows, avg T 3.73, 3.06 tokens/window, 137.17 ms/window = verify 115.82 (GPU-reach wait 47.30 + per-layer host 0.49 [plan 0.30 actq 0.25 jobs 0.25 CPU 0.00] + stage 1.12) + commit/emit 1.94 + draft 19.41
p3-32k-closed strata decode timing: 85 windows, avg T 3.52, 3.01 tokens/window, 138.29 ms/window = verify 115.40 (GPU-reach wait 44.27 + per-layer host 0.50 [plan 0.30 actq 0.24 jobs 0.25 CPU 0.00] + stage 1.26) + commit/emit 2.04 + draft 20.85
p3-32k-graph  strata decode timing: 87 windows, avg T 3.46, 2.94 tokens/window, 132.38 ms/window = verify 109.52 (GPU-reach wait 45.26 + per-layer host 0.50 [plan 0.30 actq 0.23 jobs 0.22 CPU 0.00] + stage 1.20) + commit/emit 1.91 + draft 20.95
p3-128k-closed strata decode timing: 99 windows, avg T 3.08, 2.59 tokens/window, 132.38 ms/window = verify 110.94 (GPU-reach wait 42.41 + per-layer host 0.48 [plan 0.30 actq 0.21 jobs 0.23 CPU 0.00] + stage 1.71) + commit/emit 1.98 + draft 19.46
p3-128k-graph  strata decode timing: 99 windows, avg T 3.08, 2.59 tokens/window, 128.31 ms/window = verify 106.83 (GPU-reach wait 43.96 + per-layer host 0.45 [plan 0.29 actq 0.21 jobs 0.20 CPU 0.00] + stage 1.37) + commit/emit 1.86 + draft 19.61
```

### Change 2 (tried, did NOT help): `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` — the `zeCommandListCreateImmediate` share

The card asks for the immediate-list share S2/S3 named (7.70 ms in `zeCommandListCreateImmediate`). Tested two
ways, and the honest answer is that the switch cannot move it on this port:

* **The switch is applied and it does not take effect**: with `UR_L0_DEBUG=1`,
  `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` prints `NOTE: L0 Immediate CommandList Setting: 0`, and the adapter
  **still** creates an immediate command list (`create command list ordinal: 0, type: immediate, …, inOrder: 1`)
  — an in-order queue pins the adapter to the immediate path (the adapter's own message: *"L0 Synchronous
  Immediate Command List needed with In Order property"*). The shim's queues are in-order on purpose (PLAN §1.1).
* **Measured on the engine, it changes nothing**: 32K, same binary, flag off, `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`
  (and the legacy `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`) against the plain 32K closed arm:
  **138.13 against 138.29 ms/window (-0.1%, noise)**, decode 21.8 tok/s both, 11 741 against 11 755 ms.
* The reason is visible in a debug trace of 16 submissions (`p3/spike_graph 8 2 0` with `UR_L0_DEBUG=1`):
  `zeCommandListCreateImmediate` is called **5** times for 16 kernel submissions — it is created per queue and
  reused (`getAvailableCommandList`), not per submission. The per-submission driver cost is
  `zeCommandListAppendLaunchKernelWithArguments` (**16**, one per launch). So "stop recreating command lists per
  launch" was already true at the adapter level; what the port pays 3 841 times per window is the **append**, and
  the graph path is what removes it.

## 3. Task 3 — acceptance of the graph-replay equivalent

* **Off by default behind a flag**: `STRATA_SYCL_GRAPH=1`, read once at load; unset = the closure list, unchanged.
  The device is checked for `sycl::aspect::ext_oneapi_graph` before the mode is honoured, and a refusal
  (`begin_recording`/`end_recording`/`finalize`/launch throwing) is returned as
  `cudaErrorStreamCaptureUnsupported`/`cudaErrorUnknown` with the driver's message, never as a silent fallback.
* **Bit-identical in output to the recorded path**: greedy token ids identical at 4K (147/147), 32K (256/256),
  128K (256/256) against the same binary with the flag off, and the md5s equal the ids P1/P2 measured on the
  unchanged paths. Plus the six engine tests that assert capture/replay semantics (argument baking, buffers
  re-read on replay, ordering, cross-stream) **pass with the flag ON**:
  `ctest -R "gr_parity|qsa_parity|ple_parity|sampler_parity|shared_expert_parity"` -> **100% tests passed, 0
  failed out of 6** (`sampler_parity`, `sampler_parity_one_block`, `sampler_parity_old`, `qsa_parity`,
  `gr_parity`, `shared_expert_parity`; `ZE_AFFINITY_MASK=0`). `qsa_parity` is the strongest of them: it captures
  on a temporary stream, DESTROYS that stream and replays on the default one.
* **Measured, not assumed, to be faster**: -3.1% to -4.3% per window at 4K/32K/128K, table above, from two arms
  of the same binary with only the flag changed.
* **No regression in the default path**: the full suite with the flag off is **5 failed out of 49 — the same five
  the parent HEAD failed** (`platform_memory_test`, `elementwise_parity`, `quantize_act_parity`,
  `iq_multi_parity`, `expert_multi_test`; 2 skipped), 91.5 s.

## 4. Task 4 — the numbers after, and what did not help

Per-window composition, before -> after (all six arms are in `p3/P3-EVIDENCE.txt`):

| length | ms/window | verify | GPU-reach wait | per-layer host | commit/emit | draft | submissions/window |
|---|---|---|---|---|---|---|---|
| 4K | 142.23 -> 137.17 | 121.04 -> 115.82 | 46.51 -> 47.30 | 0.52 -> 0.49 | 2.07 -> 1.94 | 19.13 -> 19.41 | 3 841.5 -> 184 |
| 32K | 138.29 -> 132.38 | 115.40 -> 109.52 | 44.27 -> 45.26 | 0.50 -> 0.50 | 2.04 -> 1.91 | 20.85 -> 20.95 | 3 842 -> 184 |
| 128K | 132.38 -> 128.31 | 110.94 -> 106.83 | 42.41 -> 43.96 | 0.48 -> 0.45 | 1.98 -> 1.86 | 19.46 -> 19.61 | 3 863 -> 183 |

Prefill is untouched by the flag (the prompt path does not capture), which is the control that says the arms are
the same machine state: 4K 15 144 -> 15 159 ms (253.0 -> 252.8 tok/s), 32K 91 644 -> 91 780 ms (352.0 -> 351.4),
128K 372 455 -> 372 539 ms (346.4 -> 346.3).

Tiers, unchanged (peak tree RSS, peak VRAM card0 + card1, expert cache):

| length | arm | peak RSS | peak VRAM | decode expert-cache hits |
|---|---|---|---|---|
| 4K | closed / graph | 48.3 / 48.3 GiB | 28.6 + 30.9 / 28.6 + 30.9 GiB | 87 360 (100.0%) both |
| 32K | closed / graph | 48.3 / 48.4 GiB | 28.7 + 31.2 / 28.8 + 31.2 GiB | 143 520 / 144 480, 100.0% both |
| 128K | closed / graph | 48.4 / 48.4 GiB | 29.0 + 31.3 / 29.0 + 31.3 GiB | 146 400, 100.0% both |

**What did NOT help:** the `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` switch (§2, -0.1% = noise, and it cannot leave the
immediate path on an in-order queue). **And what the card's premise got wrong:** cutting submissions does not buy
back the GPU-reach wait, because that wait is the GPU, not the host (§0.4, measured: -95.2% submissions, wait
+0.8/+1.0/+1.6 ms). **What remains, with the number:** at 128K the window is 128.31 ms of which `verify` is
106.83 ms (83%) and the GPU-reach wait 43.96 ms (34%); the residual host submission (184/window = 152 copies +
31 barriers) is ~0.26 ms, 0.2%. So the ~2.4x decode gap to upstream IQ3_S is **per-layer GPU work**, not
submission count — the same conclusion P2 reached from the other side for prefill.

## 5. Not validated / honest limits

* **No variance estimate.** One run per arm; the m6c harness reads a few tenths of a percent apart between arms
  of one binary historically, but the 32K end-to-end delta (+1.8%) is the one number small enough for that to
  matter, and it has a named second-order cause (window count 85 vs 87). The per-window deltas (-3.1 to -4.3%)
  are well clear of it.
* **128K prefill is one arm each** (372 s per arm); the 128K "prefill unchanged" claim rests on one pair, though
  the flag cannot reach the prompt path by construction.
* **The graph is recorded once per (stage, T) and never re-recorded**: the correctness of replaying a graph at a
  later *position* rests on the engine's own design (the window's pointers come from the verifier's fixed arena
  and the position travels as DATA), which is exactly the assumption the closure path already depended on — the
  token-id equality at three lengths is what tests it, not an argument.
* **`UR_L0_DEBUG=1` tracing was used only on the standalone spike** (16 submissions), not on the engine: its
  per-call logging perturbs a measured window, and P1b's lesson on this box is that device-instrumenting tools
  change this engine's timing.
* **The finalize cost was measured in the spike, not isolated in the engine**: 5-7 ms per shape, so <=8 shapes per
  process is ~0.05 s of a 15-370 s request; it is not broken out of the numbers above.
* **Nothing pushed**: origin (`github.com/Niko1221/Strata`) has no `sycl-xpu` branch.
* The card's `zeCommandListAppendLaunchKernelWithArguments` figure (8.02 ms) was not re-measured: unitrace's
  host-timing is what produces it (S2/S3), and this card measured the shim's own submission count instead. The
  two agree in kind (the append is the per-submission cost) but not in units.
