# P9 — the verify window, attributed: the chrome device trace cannot see it, and what does

Card `t_2403e6f6` (P9), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`.  Config of record in every arm:
IQ3_S GSQ-RCO snapshot `ed59f920…`, **both B70s** (`ZE_AFFINITY_MASK` unset, `--layer-split auto`), `--kv int8
--kv-resident 32768 --expert-cache auto --mmap-experts --prefill auto --spec 4 --spec-min-p 0.5 --no-capture
--stats --prompt-cache 0 --prompt-cache-every 0`, MTP drafter `~/strata-xpu/mtp/rt` (P7's canonical `rt`, not
P1b's W4A16 one), one engine at a time, warm program cache (`sycl-cache/m6c`).  The resident server (P6/P8) was
**stopped before the first arm** (`p6_stop.sh`: server pid 1040401 gone in 2 s, no engine, port 8099 free) and
**left stopped** at the end of this card — see §8.

Rig: `p9/p9_run_arm.sh <TAG> <CTX> <MAXNEW> [--trace 0|9|1|H] [--graph 0|1] [--kvres N] [--env K=V]` (one engine,
the trace in the arm's own directory, a FIFO for the ask so a tracer can be resumed between the load and the
request), `p9/p9_chain.sh` (the arm chain), `p9/p9_drive.py` (timestamped tee + FIFO feed), `p9/p9_report.py`
(one row per arm out of the arms' own logs), `p9/p9_extract.py` + `p9/p9_tsv.py` + `p9/p9_scan_trace.py` +
`p9/p9_summary.py` + `p9/p9_window.py` (the chrome-timeline parser), `p9/p9_clock_probe.cpp` (the device-clock
probe).  Raw output: `p9/P9-EVIDENCE.txt`.

## 0. Verdict (the short version)

1. **The card's premise needs correcting, and it is the same class of correction P3 had to make once already.**
   P1b proved that the engine now *terminates* under unitrace and that the tracer *writes a complete timeline*
   (1.13 GB / 1.99 M events).  It did **not** prove that a verify window *completes* under the tracer — and it
   does not.  Measured here, config of record, 4K, mode 9: the prefill completes, the first verify window rings
   layer 0, stalls at layer 1, and the engine ends through its own release path with
   `ERR verify: timed out at layer 1; its GPU waits were released but the GPU did not finish within 5 s (#267)` —
   **0 decode tokens**, with the graph path OFF *and* ON.  The reason is structural, and §1 shows it in the trace:
   the verify window is a host↔device flag ping-pong (the host publishes a flag per layer with a 4-byte H2D copy
   and spins on the device's ring; the kernels spin on the host's flags), and device-level instrumentation breaks
   that protocol.  Cutting the submissions to ONE (graph replay) does not help, so the stall is not the
   submission count — it is the instrumentation of the window's own handshakes.
2. **Consequence for the file that IS written**: the timeline contains the load and the prefill and **none of the
   verify window's kernels** — not even the ones of layer 0, which did ring, because the tracer reports an op when
   its command list completes and the list never completes.  The per-prefill attribution the parser produces is
   complete and real (§1.4); a per-kernel attribution of the *verify* window from a chrome timeline is not
   obtainable on this port in this build.  This is a measured negative, not a workaround.
3. **The card's own answer had to come from a different instrument, so this card fixes the engine's.**  The
   window's stage profiler (`STRATA_VERIFY_PROFILE`) was dead three ways on this device: (a) its stamp kernel read
   a device clock via `sycl::ext::oneapi::experimental::clock`, whose `ext_oneapi_clock_device` aspect **this
   device does not report**, so the engine died with
   `terminate called after throwing an instance of 'sycl::__exception' what(): Required aspect
   ext_oneapi_clock_device is not supported on the device` at the first window (measured, exit -6, 0 tokens);
   (b) its buffer was device memory with a one-shot D2H copy at the end; (c) the accumulation was behind
   `if (prof_on_ && G == 1)`, and on the config of record (a split with T >= 2, so G == 2) that gate is never
   true, so even a working clock would have printed nothing.  All three are fixed in §2: the stamp is now a
   PUBLISH into a MAPPED buffer and a host sampler thread timestamps each stage code, which also gives the
   window's tail (where the host is parked in the driver) a timeline.
4. **A second, independent finding, and it is the largest single term on the decode side**: the engine's own
   breakdown of `verify` never summed to `verify`, for two reasons, both fixed here.  (a) The counters it printed
   were ONE stage's (stage 0's, layers 0-22): the rest of the model runs in stage 1's nested `run()`, so at 4K
   the line named 46.9 ms of a 118.8 ms `verify`.  (b) The window's tail sync was untimed.  Folded and timed, at
   4K the window of record is **133.59 ms = verify 120.06 (89.9%) + commit/emit 2.03 + draft 11.49**, and verify
   is **GPU-reach wait 94.81 ms (71.0% of the whole window, 79% of verify)** + the host's per-layer work 1.09 +
   the window's staging 1.40 + the tail sync 12.97 + 9.79 ms of post-loop host work (the head's sampling and the
   result copies).  **The card's "GPU-reach wait 43.96 ms (34%)" is one stage's half of it: the true figure is
   ~70% of the window at all three lengths** (§3).

## 1. What the chrome trace can and cannot see (measured, three ways)

### 1.1 The mode-9 arms: the window stalls, in both graph modes

`p9/p9_run_arm.sh p9-ut9-4k-g0 4096 16 --prefill auto --kvres 32768 --graph 0 --trace 9`, and the same with
`--graph 1`.  Both are the config of record at 4K with one change (the tracer):

```
# p9-ut9-4k-g0 (STRATA_SYCL_GRAPH unset = the closure path)
PP 3831 3832 14627 261.9
REUSED 0
ERR verify: timed out at layer 1; its GPU waits were released but the GPU did not finish within 5 s (#267)
verify teardown: the window on cs_ was released (#267) and has NOT finished in 5000.433 ms; the GPU work cannot be
  ended from here and the driver teardown is skipped - the engine ends now ...
[INFO] Timeline is stored in strata.1047887.json
# T lines: 0   -> 264,151,738 B of chrome timeline and ZERO generated tokens

# p9-ut9-4k-g1 (STRATA_SYCL_GRAPH=1: the whole window is ONE graph submission)
strata/sycl: STRATA_SYCL_GRAPH=1 - a window's capture records a SYCL command_graph and cudaGraphLaunch submits it
  in ONE submission
ERR verify: timed out at layer 1; ... GPU did not finish within 5 s (#267)
# T lines: 0   -> 250,385,887 B

# and P1b's own two arms, re-read from their logs (p1/logs/), stalled the same way:
#   p1b-x-ut9  (mode 9, 1,131,673,822 B): ERR verify: timed out at layer 1 ...  T lines: 0
#   p1b-z-ut1  (mode 1, 1,245,446,644 B + the 2.37 MB API Timing Summary): ERR verify: timed out at layer 1 ... T lines: 0
```

So the card's "the instrument works now" is true of the FILE and not of the WINDOW: the engine exits by itself
(P1b's fix, confirmed here) and the tracer writes a closed timeline, but the first verify window never finishes
under it, with 3 841 submissions **or** with 1 (graph).  The window's stall is not the submission count.

### 1.2 What is in the timeline, then: the load and the prefill, and none of the window

The parser (`p9/p9_prep.py`) learns the DEVICE pseudo-processes from the trace's own metadata — they are per card,
and both cards are there:

```
device pid 4261434910: DEVICE<michael-AI>[Intel(R) Graphics [0xe223]] 0:3:0:0 #0   (52,043 events)
device pid 4261434911: DEVICE<michael-AI>[Intel(R) Graphics [0xe223]] 0:8:0:0 #1   (87,340 events)
rows=445,239 kept=139,383        (the trace's X events -> the device rows)
card0: 5,071 ms of device time in a 16.41 s span;  card1: 8,011 ms
```

Card 0's device time, by kernel (`p9/p9_summary.py`), i.e. the load + prefill:

```
  3419.130 ms        5  683826.083 us  anon::launch<1>(float const*, QsaAttnPools const&, ... )   <- the prompt attention
   523.204 ms   10,168       51.456 us  iq_dequant_gu_f16(...)
   325.253 ms   10,168       31.988 us  iq_dequant_f16(...)
   145.640 ms       36     4045.544 us  prefill::gdn_recurrence(...)
    75.190 ms       74     1016.082 us  gemm_f16f16f32_nocopy_tn_64x40_4x8[SIMD16 {32; 1; 1} {64; 8; 1}]
    38.602 ms       44      877.310 us  prefill::gr_write_norm_rs(...)
    35.178 ms       46      764.744 us  prefill::gr_mix_r(...)
    32.614 ms      143      228.069 us  anon::launch<anon::H16>(...)
    31.945 ms       75      425.937 us  qsa_block_scores(...)
     5 events x 683 ms  = the whole prefill attention on this card's 5 QSA layers
```

and then, after the prefill's last >5 ms event (a 156,917,760 B D2H copy at t+5.999 s), the trace's remaining
device activity is **four** events, all in the release path:

```
== the first 60 events after the boundary ==
   t+ 10409.163 ms       12.24 us  zeCommandListAppendMemoryCopy(H2D)[98304]
   t+ 10410.788 ms        7.81 us  zeCommandListAppendMemoryCopy(H2D)[4]
   t+ 10410.815 ms        5.88 us  zeCommandListAppendMemoryCopy(H2D)[4]
   t+ 10410.840 ms        4.95 us  zeCommandListAppendMemoryCopy(H2D)[4]
```

Three 4-byte H2D copies: those are `publish_flag`'s flag publications, i.e. the release's, not the window's.
**The verify window contributes no completed device event at all — not even layer 0's, which did ring**
(`the GPU rang 1` in the diag line).  The reason is in how the tracer reports: a device event is written when
its command list completes, and the window's list never does.

### 1.3 Why: the window is a host↔device flag ping-pong, and device instrumentation breaks it

The verify window is not a plain kernel sequence.  Per layer it is:

* the host publishes a flag the KERNELS spin on, with a 4-byte H2D copy on the copy stream
  (`verify.cpp:1440-1446`: three `cudaMemcpyAsync(m_flag_, h_flag_, sizeof(uint32_t), H2D, copy_)` plus a
  `cudaStreamSynchronize(copy_)` **per window launch**), and republishes it per layer
  (`publish_flag`/`raise_flag_dev`, `verify.cpp:271-...`);
* the host then spins on a device-written counter — the "GPU-reach wait"
  (`verify.cpp:1471` `while (*seq < want) { _mm_pause(); ... }`, one ring per layer-step);
* and the kernels themselves block on the host's flags (`wait_flag_ge` and `doorbell_wait_kernel`, a device
  spin: `while (strata_flag_read(flag) != want) strata_spin_pause();`).

Under a device-instrumenting tracer, each of those 4-byte publications and ring reads goes through the tracing
layer; the chain that depends on them loses its handshake and the window dies at layer 1 — measured, twice
(graph off, graph on), and in P1b's two arms.  This is also why thinning the submissions cannot fix it: what the
tracer breaks is the latency of the handshakes, not the number of appends.

### 1.4 The instrument that still works for this file: the prefill

The parser produces a complete per-kernel table for the *prefill*, and the "big event" detector that finds the
prefill/decode boundary by kernel size works.  **Cross-check, trace against the engine's own line** — the one
comparison this file does support:

| reading | 4K prefill |
|---|---|
| the engine (`PP 3831 3832 14627 261.9`, its own wall clock for the read) | **14 627 ms** |
| the trace, card 0 (0:3:0:0) device busy, all kernels | 5 071 ms |
| the trace, card 1 (0:8:0:0) device busy, all kernels | 8 026 ms |
| **trace, both cards** | **13 098 ms = 89.5% of the engine's prefill wall** |

The two readings agree to 10%, and the residual is exactly what a device-busy sum cannot include (the staging
between kernels, the flag copies, the request's own host work).  Inside the prefill, on card 0: the prompt
attention is 3 419 ms of the 5 071 ms of device time (**67%**), with `iq_dequant_gu_f16` (10 168 launches at
51.5 us) and `iq_dequant_f16` (10 168 at 32.0 us) the next two.  `p9_window.py --tag <name>` prints the census for
any tag in a timeline; `p9_summary.py` prints the table.


### 1.5 And the engine's own GPU stage profiler was dead too

`STRATA_VERIFY_PROFILE=1` (the port's equivalent of the CUDA build's `gpu_stamp` stage table, with the 24 stage
sites already placed in the layer body) killed the engine at the first window:

```
# p9-ctl-4k-16 (config of record, 4K, STRATA_VERIFY_PROFILE=1, no tracer)
PP 3831 3832 13765 278.3
REUSED 0
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  Required aspect ext_oneapi_clock_device is not supported on the device
# T lines: 0, the request never completes
```

`p9/p9_clock_probe.cpp` asks the device directly (it is the reason the probe is committed): both B70s report
`aspect ext_oneapi_clock_device` as **no** while `ext_oneapi_graph` is YES, oneAPI 2026.1's `clock.hpp` offers no
other scope that avoids that aspect (device/work_group/sub_group all take it), and the OpenCL `clock()` builtin is
not reachable from SYCL device code (it resolves to the host's `time.h` declaration, which the compiler refuses:
`SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute`).  So this port has no readable
device clock, and the profiler's design — "one 1-thread kernel per stage writes a device-clock timestamp; the host
reads them at the end" — could not work as written.  `verify.cpp`'s beacon comment already said as much for the
tail beacons.

## 2. What this card fixed: the stage profiler, host-sampled (and the tail, named)

Files: `src/kernels/sycl/verify_kernels.cpp` (`gpu_stamp_kernel`), `include/strata/kernels/verify_kernels.hpp`,
`include/strata/core/verify.hpp`, `src/core/verify.cpp`, `src/program/generate.cpp`.  Nothing is enabled by
default; all of it is behind `STRATA_VERIFY_PROFILE=1` (already an opt-in), so the default path is unchanged.

1. **The stamp is a PUBLISH, not a timestamp** (`gpu_stamp_kernel`): it writes the marker word `buf[i] = 1` and
   then, after `__threadfence_system()`, the stage's own index into the buffer's progress slot
   (`kProfProgCode = 4095`, `verify_kernels.hpp`).  `volatile` + a system fence, because the reader is the CPU —
   the same combination `doorbell_ring_kernel` uses on this backend.
2. **The buffer is MAPPED** (`mapped()`), not device memory: the host has to read it *while* the window runs, and
   the one-shot D2H copy at the end is gone.
3. **A sampler thread** timestamps every stage code it sees, for the window's whole visible life — from the
   launch, through the layer loop, and through the tail sync, where the host is parked in the driver and could not
   sample anything.  It polls one mapped word (~a read of the host's own RAM: the device writes it over PCIe), so
   it does not sit on the host's critical path.  `SamplerStop` joins it on every early return.
4. **The per-window reset** is a host `memset` of the mapped buffer before the launch (a host→device write that is
   read once by the kernels, which this backend is measured to deliver — the flag note's caveat is about *poll
   loops*, not about one-shot reads).
5. **The `G == 1` gate is gone** (`if (prof_on_ && G == 1)` -> `if (prof_on_)`): on the config of record the window
   is a split with T >= 2, so G == 2, and the old gate meant the profiler printed nothing even with a working
   clock.  And **the tail sync is timed** (`ms_tail`, printed as `+ tail %.2f` in the `strata decode timing` line),
   which is what closes the engine's own account of `verify` (§0.4).

So after this card the engine can answer "where does a verify window's GPU time go" on a device with no device
clock, and its own summary line adds up.

## 3. The window's account, and what was missing from it

The decode-timing line's own arithmetic did not close: `verify` brackets the whole window call, but the counters
it printed were **one stage's** (on a split, stage 0's: layers 0-22), because the rest of the model runs in the
stage-1 verifier's nested `run()`.  Folded in (`verify.cpp`, the `le_ < g.n_layers` return path), with the
profiler ON (so the stamp set's +3% is included) and one control arm with it OFF:

```
# p9-ctl3-4k (no tracer, no profiler: the plain window of record on the binary this card ships)
strata decode timing: 49 windows, avg T 3.80, 3.00 tokens/window, 133.59 ms/window = verify 120.06
  (GPU-reach wait 94.81 + per-layer host 1.09 [plan 0.31 actq 0.25 jobs 0.27 CPU 0.00] + stage 1.40
   + tail 12.97) + commit/emit 2.03 + draft 11.49; ... VRAM hits 37.96, PCIe 0.00
# p9-p32k / p9-p128k (profiler ON, so ~3% of this is the instrument itself)
strata decode timing: 87 windows, ... 133.64 ms/window = verify 118.34 (wait 92.24 + host 1.04 + stage 1.33
  + tail 12.40) + commit/emit 2.06 + draft 13.24
strata decode timing: 99 windows, ... 128.61 ms/window = verify 114.32 (wait 89.34 + host 0.96 + stage 1.58
  + tail 11.32) + commit/emit 1.94 + draft 12.35
```

| length | window ms | verify | of which: GPU-reach wait | host/pool | staging | tail sync | verify minus the named parts | commit/emit | draft |
|---|---|---|---|---|---|---|---|---|---|
| 4K (control, no instrument) | 133.59 | 120.06 (89.9%) | **94.81 (71.0% of the window)** | 1.09 | 1.40 | 12.97 | 9.79 | 2.03 | 11.49 |
| 32K (profiled) | 133.64 | 118.34 | 92.24 (69.0%) | 1.04 | 1.33 | 12.40 | 11.33 | 2.06 | 13.24 |
| 128K (profiled) | 128.61 | 114.32 | 89.34 (69.5%) | 0.96 | 1.58 | 11.32 | 11.12 | 1.94 | 12.35 |

Two premises have to be corrected against this measurement:

* **the card's "GPU-reach wait 43.96 ms (34%)" is one stage's half of it.**  The whole chain's is 94.81 ms of
  133.59 ms at 4K, 92.24 of 133.64 at 32K and 89.34 of 128.61 at 128K.  P3's reading ("the window is GPU-bound
  per layer, and the wait is the GPU") is confirmed in direction and **understated ~2x in size: ~70% of the
  window is the host waiting for the GPU's layers**, at all three lengths.
* **the tail sync is not the big term** it looked like from the residual: 11-13 ms.  The ~69 ms this card first
  computed as unattributed at 4K was stage 1's counters (the nested `run()`), and what remains after the fold is
  9.8-11.3 ms of post-loop host work (the head's sampling and the result copies).  The engine's own line now
  adds up to within 8% of `verify`.


## 4. The per-stage attribution (the instrument this card repaired)

`p9/p9_stages.py` (and the raw line) for the 4K arm `p9-p4k5` (in-loop sampler; the stamp set costs +3.1% of the
window, see §6):

```
strata decode GPU stages (ms/window): GDN layers: q8+qkv/q-idx gemv 5.66 conv 0.10 ab 0.31 z 3.52 rec 0.45
  out-proj 2.98 hc-read1+router 3.17 shared+quant 3.35 waitA 0.16 VRAM hits 11.40 waitB 0.11 PCIe grp 0.53
  waitCPU 0.04 copy+combine 0.18 (gap) 0.03   hc0 norm 1.17   hc0 down 0.99   hc0 up 1.02 |
  QSA layers: q8+kv-idx 0.08 k/v+norm-rope 0.31 kv+idx append 0.12 q+q-idx 2.51 scores+topk 0.36 kv-resolve 0.01
  attention 0.83 gate 0.02 out-proj 0.90 hc-read1+router 0.98 shared+quant 1.09 waitA 0.04 VRAM hits 3.28
  waitB 0.03 PCIe grp 0.16 waitCPU 0.01 copy+combine 0.05 (gap) 0.01   hc0 norm 0.05   hc0 down 0.29
  hc0 up 0.30 | total 46.63 ms/window over 49 windows
```

| stage (the engine's own names) | 4K ms/w | 32K ms/w | 128K ms/w | 4K % of the sampled time |
|---|---|---|---|---|
| **VRAM hits — the resident experts' compute (10 experts x 48 layers)** | **14.680** | **14.000** | **12.470** | **31.5%** |
| q8+qkv/q-idx gemv (the q/k/v + indexer projections) | 5.660 | 5.420 | 5.050 | 12.1% |
| shared+quant (the shared expert + activation quantize) | 4.440 | 4.090 | 3.700 | 9.5% |
| hc-read1+router (the second hyper-connection read + the router) | 4.150 | 4.000 | 3.720 | 8.9% |
| out-proj | 3.880 | 3.700 | 3.470 | 8.3% |
| z (the GDN z projection) | 3.520 | 3.330 | 3.050 | 7.6% |
| hc0 norm + down + up (the first hyper-connection read, split) | 3.820 | 3.670 | 3.290 | 8.2% |
| q+q-idx (the QSA query + indexer projection) | 2.510 | 2.340 | 2.130 | 5.4% |
| scores+topk (the QSA index selection) | 0.360 | 0.830 | **2.340** | 0.8% |
| **attention (the QSA decode attention over the int8 KV)** | **0.830** | 0.780 | 0.710 | **1.8%** |
| kv+idx append | 0.120 | 0.110 | **0.710** | 0.3% |
| kv-resolve | 0.010 | 0.010 | **0.190** | 0.0% |
| k/v+norm-rope | 0.310 | 0.300 | 0.280 | 0.7% |
| PCIe grp, waitA/B/C, copy+combine, conv, ab, rec, gate, (gap) | 1.610 | 1.690 | 1.360 | 3.5% |
| **total sampled** | **46.63** | **44.78** | **43.24** | 100% |

**The answer to the card's question is in the first row and the attention row:** the verify window's GPU time is
dominated by the **resident experts' compute (29-32% of the sampled time) and the projections/mixers around them
(another ~35%: both GEMV groups, the z/out projections, the shared expert and the two hyper-connection reads)**,
while the **QSA decode attention over the int8 KV is 0.71-0.83 ms — 1.6-1.8%** at all three lengths.  The card's
list of candidates resolves to: experts ~31%, projections ~21%, hyper-connection reads ~17%, shared expert ~9.5%,
attention ~1.8%, copies/barriers ~0.5%, commit/emit 2 ms (1.5% of the window, measured separately).

**What DOES grow with depth is the QSA *selection*, not the attention**: `scores+topk` 0.36 -> 0.83 -> 2.34 ms and
`kv+idx append` 0.12 -> 0.11 -> 0.71 ms and `kv-resolve` 0.01 -> 0.19 ms, i.e. the KV-at-depth cost on this path
is the **index/score work over the paged KV (3.2 ms at 128K, 7.5% of the sampled time)** while the attention
kernel itself is flat (0.83 -> 0.71), which is the decode-side mirror of P2's finding that the *prompt* attention
is capped by `qsa_selection_width` and does not grow with depth.

**Shares of `verify`, as floors.**  Because the sampler's coverage is 33-39%, the honest way to read the table
against `verify` is as a floor per family: at 4K `verify` is 120.06 ms (123.17 in the profiled arm), so the
experts' 14.68 ms is **>=12.2% of verify (11.0% of the window)**, the two projection GEMV groups' 8.17 ms is
>=6.8% of verify, the hyper-connection reads' 7.97 ms is >=6.6%, the shared expert's 4.44 ms is >=3.7%, and the
attention's 0.83 ms is **>=0.7% of verify (0.6% of the window)**.  Every other number in the table is a floor the
same way.

**Honest coverage limit.**  The stage sum (43-47 ms) is **33-39% of the window and of `verify`**: the sampler sees
a stage code only if it is the latest one when the host samples, so stages the GPU raced through between two
samples contribute nothing (the engine's own D8 rule skips them, and this card's fix keeps that behaviour).
The SHARES above are therefore shares of the *sampled* GPU time, not of the whole window, and a stage that fires
in bursts is under-represented.  What the table establishes is the order of magnitude of each family — and the
two extremes (experts 14.7 ms against attention 0.83 ms at 4K) are 18x apart, far outside the sampling artefact.
The instrument's own cost is measured: the stamp set alone takes the window from 132.71 to 137.03 ms (+3.3%,
998 stamps at 4.3 us each), and with the in-loop sampler it is +3.0% (§6).


## 5. The ablation table (the second, independent reading of the same window)

Every arm is the config of record at its length with ONE switch changed, and every one of them produced
**byte-identical greedy token ids** (`grep '^T ' | md5sum`), so none of these is a correctness change:

| arm | 4K ms/win | 32K ms/win | 128K ms/win | decode tok/s (4K / 32K / 128K) | token ids (4K / 32K / 128K) |
|---|---|---|---|---|---|
| **baseline (of record)** | 132.38 | 129.16 | 124.06 | 22.66 / 22.78 / 20.84 | `ac9f16fc` / `124a3cd3` / `c60e72d3` |
| `STRATA_SYCL_GRAPH=1` (the window is ONE submission) | **125.55 (-5.2%)** | **123.91 (-4.1%)** | **120.85 (-2.6%)** | 22.96 / 23.75 / 21.40 | identical |
| `STRATA_DEC_BATCH=0` (the window's rows, token by token) | 140.49 (+6.1%) | 134.94 (+4.5%) | — | 21.35 / 21.81 | identical |
| `STRATA_HC_SPLIT=0` (the plain hyper-connection read) | 133.26 (+0.7%) | — | — | 22.51 | identical |
| `--kv-resident 0` (no KV streaming: K/V never resident in VRAM) | n/a (a no-op below 4N cells) | n/a | 122.84 (**-1.0%** decode, **-19.5% prefill**: 299.5 s against 370.0 s) | 21.05 (prefill 426.0 against 346.6 tok/s) | identical |

(the end-to-end tok/s is not exactly the window ratio — the MTP draft policy re-picks T from the measured window
cost, so the graph arm needed 51 windows for 147 tokens at 4K against the baseline's 49: P3's caveat, unchanged)


with the submission counts that explain the first two: the baseline window is **2 504 kernels + 1 306 copies +
31 barriers** per window (stage 0 1 197 kernels/556 copies, stage 1 1 307/750), `DEC_BATCH=0` adds **+1 188
kernels** (3 692), and the graph arm hands the driver **1 graph submission** per stage (150 copies remain
outside it).  So:

* **the batched window path already banks 8.11 ms/window at 4K and 5.78 at 32K** (the un-batched arm's delta),
  which is the marginal cost of 1 188 kernel launches: **6.8 us (4K) and 4.9 us (32K) per launch** — an
  independent confirmation of the stamp measurement (4.3 us/stamp) and of P3's 1.4 us host-side append figure
  being only part of a launch's cost;
* **the staged hyper-connection read banks 0.88 ms/window** at 4K (the plain read's delta) — it is already the
  default;
* **the graph path is worth 6.83/5.25/3.21 ms a window** (-5.2/-4.1/-2.6%) with identical ids, and its win is
  exactly the host's submission work: the GPU-reach wait does not move (±0.4 ms), while the window's
  host-visible part (verify minus the wait) drops by 7-8 ms — P3's reading, re-measured on a new binary and at
  three lengths;
* **KV streaming (`--kv-resident 32768`, the record's setting) costs 1.22 ms of decode at 128K and 70 s of
  prefill**, and it is what makes 256K fit at all (P6 measured that it buys almost no expert residency on 2x32
  GiB: 24 576 slots with it against 24 302 without).

## 6. The instrument's own cost, and the trap in it

Measured on the engine, same binary family, 4K, all with identical token ids:

```
p9-ctl2-4k  (the shipped binary, profiler OFF)              132.71 ms/window
p9-p4k4     STRATA_VERIFY_PROFILE=2 (stamps, NO sampler)    137.03 (+4.32, +3.3%)  <- 998 stamps at 4.3 us
p9-p4k3     STRATA_VERIFY_PROFILE=1 (stamps + sampler THREAD) 244.71 (+112.00, +84%)  <- NOT usable
p9-p4k5     STRATA_VERIFY_PROFILE=1 (stamps + in-loop sampler) 136.71 (+4.00, +3.0%)  <- shipped
p9-p4k      the first form: __threadfence_system() per stamp + the thread  246.80
```

**A sampler THREAD is a trap on this engine**: +107.7 ms of window (79%) — the layer loop is host-bound enough
that one more spinning thread on a 20-core box whose 19 expert-pool workers are already busy costs more than
everything the sampler measures.  The first form of this card's own fix had that thread, and the first table it
produced was garbage for a second, unrelated reason (§2.3's missing `prof_h_` fill: `waitA 7576 ms/window` for a
132 ms window).  Both are fixed and both are kept in the record as measurements: the sampler now runs inside the
layer spin loop, where the host is spinning anyway, for +3.0% of the window.

## 7. The shortlist, and what was measured

| candidate | measured | verdict |
|---|---|---|
| **the resident experts' compute** (31.5% of the sampled GPU time; 10 experts x 48 layers x 4 rows) | the biggest single item, and it is the same work at 4K/32K/128K | **the lever**, but nothing in this card's scope changes it: a faster expert GEMM (the IQ3 dequant + GEMV shapes) is a kernel project, not a switch.  Its shape is fixed by the model (top-10 of 256 experts per layer) |
| **the window's per-launch cost** (2 504 kernels + 1 306 copies per window, 6.8 us each) | `DEC_BATCH=0` measured 8.11/5.78 ms a window | already banked by the batched path (`1e4515c`); the remaining launches are the experts' own GEMMs, one per expert |
| **make the graph path the default** | -5.2/-4.1/-2.6% of the window, ids identical, at three lengths | **measured, not landed**: P3 deliberately left `STRATA_SYCL_GRAPH` opt-in and a refusal throws `cudaErrorStreamCaptureUnsupported` instead of falling back to the closure list, so flipping the default trades a measured 5% for a hard failure on a device that cannot record a graph.  That is a maintainer's call, not a rig's: the numbers are here, the switch is one env var |
| **KV streaming / page residency** | `--kv-resident 0` at 128K: -1.0% decode, -19.5% prefill, ids identical | **a config lever for this box, with a named cost**: it is what makes 256K fit (P6).  Not landed: the record's setting buys 256K |
| **the QSA decode attention at depth** | 1.8% of the sampled GPU time at 4K, and the row shrinks with depth | **no lever here** — the attention is not the cost; the card's "QSA decode attention over the int8 KV at depth" hypothesis is refuted by the table |
| **the copies/barriers** | 1 306 copies + 31 barriers a window; the graph arm's 150 remaining copies at 1.4 us = 0.2% | no lever (P3 reached the same number from the counters) |
| **the instrument itself** | the chrome device modes cannot see the window (§1); the stage profiler was dead three ways and now works (§2), at +3.0% of the window (§6) | landed, with its own cost measured |

## 8. Not validated, and the state left behind

**Not validated**

* **The chrome device modes cannot be made to work here, and this card did not find a way around it.**  Mode 9 was
  run twice by this card (graph path off and on) and P1b ran mode 1 and mode 9 (graph off): the window stalls at
  layer 1 in all four, with 3 841 submissions or with one graph submission.  What is *not* ruled out is a tracer
  build, or a unitrace mode that instruments less (the host-only `--chrome-call-logging` path is measured to work,
  and gives per-API-call times with no kernel names).
* **The stage table's coverage is 33-39%** of the window (§4).  A better sampler (e.g. one that reads the stage
  words of the current layer instead of only the progress word, or a per-stage ring) would raise it, and the
  shares would then be shares of the whole window rather than of the sampled time.
* **The stages the sampler misses are not uniformly random**: a stage that fires in bursts (the experts' per-expert
  GEMM chain is the obvious one) is under-represented, so the 31% is a floor for the expert path, not an estimate.
* **The stamps' own cost is measured but not subtracted**: the profiled arms are ~3% slower, and the deltas the
  table reports are deltas of a window that includes that cost.  Every profiled number in this card is labelled.
* **`STRATA_VERIFY_PROFILE` was not swept by stamp count.**  24 sites per layer is what the port already had; a
  6-site version would cost ~0.8% and might be the better instrument for a routine run.
* **The sampler thread's trap was measured at 4K only.**  The +107.7 ms (+79%) figure is one window shape; the
  in-loop sampler is what shipped, so the trap is a lesson rather than a number with a range.
* **No variance estimate**: one arm per configuration, as in P1b/P3.  The deltas quoted are 0.7-6.1% of the
  window, and the m6c harness's arms have historically read a few tenths of a percent apart, but the 0.7%
  (`HC_SPLIT`) and the 1.0% (128K `kv-resident 0`) are inside that band.
* **No VRAM/RSS tiers in this card's arms**: the rig samples neither, because the question is a timing one and
  the expert caches report themselves (`expert_slots=24576`, `vr_hits` per layer-window in every decode line);
  P3's tiers at the same config are 48.3-48.4 GiB RSS and 28.6-29.0 + 30.9-31.3 GiB of VRAM.
* **Nothing is pushed**: the origin (`github.com/Niko1221/Strata`) has no `sycl-xpu` branch.

**The state left behind**

* **The resident server (P6/P8) is STOPPED and was left stopped**: `p6_stop.sh` at the start of this card reported
  `server gone after 2s`, no engine, port 8099 free, both cards free; the p6 pid file it removes
  (`p6/server.pid`) is deleted in this card's commit.  Every arm in this card ran one engine at a time on both
  B70s with `ZE_AFFINITY_MASK` unset.
* `build-sycl/strata` is the binary this card measured (md5 `9eff0675…`); the arms' own md5s are in their logs.
* **The engine's test suite is unchanged by this card**: `ZE_AFFINITY_MASK=0 ctest` on the shipped build is
  `90% tests passed, 5 tests failed out of 49` in 96.95 s — the SAME five the parent HEAD failed in P3
  (`platform_memory_test`, `elementwise_parity`, `quantize_act_parity`, `iq_multi_parity`, `expert_multi_test`,
  with `iq_parity`/`iq_parity_fixtures` skipped), so the profiler's default-off path is not a regression
  (`/home/michael/strata-xpu/p9/ctest-p9.log`; the repo's `.gitignore` excludes `*.log`, so it stays beside the
  runs rather than in the commit).
* The engine changes are opt-in: **`STRATA_VERIFY_PROFILE` is the only switch, and the default path is unchanged**
  (`p9-ctl3-4k` against `p9-ctl-4k-150b`: 133.59 against 132.38 ms/window, 22.46 against 22.66 tok/s decode,
  identical token ids — i.e. the fold and the `tail` field cost nothing, and the 0.9% is the two binaries' noise).
* `/home/michael/strata-xpu/p9/` holds the runs, the traces and the analysis caches; the repo holds `p9/` (the rig,
  the parsers, the write-up) and `p9/P9-EVIDENCE.txt`.

## 9. Files

| what | path |
|---|---|
| the rig, one engine at a time, tracer optional | `p9/p9_run_arm.sh`, `p9/p9_chain.sh`, `p9/p9_chain_prof.sh`, `p9/p9_drive.py` |
| one row per arm out of the arms' logs | `p9/p9_report.py` |
| the chrome-timeline parser | `p9/p9_extract.py`, `p9/p9_tsv.py`, `p9/p9_prep.py`, `p9/p9_scan_trace.py`, `p9/p9_scan2.py`, `p9/p9_summary.py`, `p9/p9_timeline.py`, `p9/p9_window.py` |
| the stage table | `p9/p9_stages.py` |
| the device-clock probe (and why the fix is a publish) | `p9/p9_clock_probe.cpp` |
| the engine changes | `src/kernels/sycl/verify_kernels.cpp`, `include/strata/kernels/verify_kernels.hpp`, `include/strata/core/verify.hpp`, `src/core/verify.cpp`, `src/program/generate.cpp` |
| the arms' raw output | `p9/P9-EVIDENCE.txt` (the report's tables, the decode-timing lines, the stage lines, every arm's DONE line and token-id md5) |
| the traces | `/home/michael/strata-xpu/p9/runs/<TAG>/strata.<pid>.json` (the two mode-9 arms) |
| the analysis caches | `/home/michael/strata-xpu/p9/dev/*.tsv`, `*.npz` |



