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
   breakdown of `verify` never summed to `verify`.  At 4K `verify` is **118.84 ms** of the 132.38 ms window, and
   the line named only the GPU-reach wait (45.27), the host's per-layer work (0.51) and the staging (1.15) —
   **71.91 ms (60.5% of verify) was unattributed**.  It is one call: the window's tail `cudaStreamSynchronize(cs_)`
   after the layer loop, now timed (§2.5) and reported as `tail`.

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

The parser produces a complete per-kernel table for the *prefill* (a real, if partial, answer to "which kernel
dominates": on card 0 the prompt attention is 3,419 ms of the 5,071 ms of device time = 67%), and the "big event"
detector that finds the prefill/decode boundary by kernel size works.  Nothing else in the timeline is usable for
this card's question.  `p9_window.py --tag <name>` prints the census for any tag in a timeline.

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

