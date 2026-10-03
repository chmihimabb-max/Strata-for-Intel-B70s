# P1 — the verify window's GPU-wait release no longer blocks (and no longer fails) the host

Card `t_44a0ac61`. Repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, HEAD **b538baf** (3 commits added by this
card, not pushed: origin has no `sycl-xpu`). Config of record throughout: `strata-sycl-iq3s.json` — IQ3_S GSQ-RCO
snapshot `ed59f92…`, both B70s (`ZE_AFFINITY_MASK` unset, `--layer-split auto`), `--kv int8`, `--no-capture`,
`--mmap-experts`, `--spec 4`, one engine at a time. Evidence under `~/strata-xpu/p1/{logs,runs,traces}` and
`~/strata-xpu/m6c/runs/p1-*`; the copies committed to the repo are in `p1/evidence/`.

## 1. Verdict: the copy is a SYCL-port artifact, and it was blocking *and* failing

**The CUDA original had no copy at all.** At the port's own parent commit (`git show c2bbc75^:src/core/verify.cpp`,
release_gpu_waits at lines 136-155) the release wrote the MAPPED HOST words and stopped there; the spin kernels read
those same mapped host words, so there was no host→device copy and no sync/async question to answer.

**The copy was added by the port.** Commit `c2bbc75` (SYCL M5b, card `t_3ebd6083`) made the three words the KERNELS
read DEVICE memory, because on this backend a poll loop over mapped host memory never observes the host's stores
(M5b's own measurement: one transition to the last value ~318 ms after the burst, and a system-scope atomic load of
mapped host memory unreliable). With the kernels on a device word, *something* has to publish the release there —
that is the copy, and it is hand-written port code, not a transform artifact: `tools/sycl/syclify.py` only rewrites
`cudaMemcpyToSymbol` (`tools/sycl/syclify.py:621-628`), and the copy is a plain `cudaMemcpy` in the M5b diff.

**Its synchronousness is the shim's faithful CUDA emulation, chosen deliberately.** `cudaMemcpy` in the SYCL shim is
`memcpy_impl(dst, src, bytes, &default_queue(), /*wait=*/true)` (`include/strata/sycl_compat/cuda_runtime.h:636-640`),
which submits on the DEFAULT queue and then waits for the copy's event (`:596-604`; the USM-to-USM arm at `:601-602`,
the host-staging arm at `:605-608`). The port picked that spelling for a stated reason (the old comment at
`src/core/verify.cpp:191-194`: "SYNCHRONOUS on purpose … an async copy would be recorded as a node instead of
submitted" — `cudaMemcpyAsync` is capture-aware, `:648-651`).

**Measured at HEAD, that copy does not even land.** Instrumentation-only commit `3ccb530` (this card) put a timer in
the release; in the untraced `STRATA_TEST_VERIFY_STALL=1` rig *and* under `unitrace --device-timing`, all three copies
fail:

```
strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
verify release: flag publication 66.411 ms; drain budget 5000 ms
```

They run on the DEFAULT queue — the *current* device's queue, which on a two-stage layer split is not the verifier's
own device — so the 4-byte copy goes cross-device and the driver refuses it. The release therefore never raised the
device flag, i.e. the #267 guarantee it exists for was not being met on that path. S3UT (`t_7e1307a6`) is the other
half of the defect: the host was parked in that same wait for as long as the engine lived, which is why every
unitrace mode that instruments the device ended with a SIGKILL and a 0-byte report.

## 2. The change

`src/core/verify.cpp` (+ commit `e452dd0`):

* `publish_release_word()` (new, in the file's anonymous namespace): on SYCL it submits the 4-byte copy on the
  verifier's OWN copy stream through the shim's submission path —
  `sycl_compat::memcpy_impl(dev, host, 4, queue_for(copy_), /*wait=*/false)` — and does not wait. On CUDA/HIP it
  keeps the original blocking `cudaMemcpy` (the defect is measured on SYCL only and neither can be built here;
  `include/strata/kernels/flag.hpp` is the same backend split).
* `release_gpu_waits()` calls it for the three words, with the old call kept as the `copy_ == nullptr` fallback.
* `STRATA_VERIFY_RELEASE_DEBUG=1` (off by default, commit `3ccb530`) prints the publication and drain wall times.
  Commit `b538baf` records the read-back measurement below in the helper's own note.

**Why the ordering is safe.** `copy_` is the verifier's own copy stream — the stream `publish_flag` /
`raise_flag_dev` already use for every per-layer flag, on the verifier's own device, and the stream M5b measured a
4-byte copy through to a spinning kernel (10 of 10 rings, three runs). The value written is `UINT32_MAX`, past every
ring the kernels compare against (`wait_flag_ge_kernel`: `while (strata_flag_read(flag) < value)`), so sharing the
in-order stream with the window's other publications cannot satisfy a wait early. The drain loop that follows polls
exactly this stream, so "did the release get through" is still answered before the call returns. And the copy is
submitted, never recorded: that is what the old blocking call was chosen for and what `cudaMemcpyAsync` would lose
while a capture is open.

**Measured: it now lands.** With a temporary read-back probe on the same in-order stream (a 4-byte copy back to a
host word, ordered behind the publication; the probe is removed from the final commit), the three DEVICE words read
`UINT32_MAX` **1.05 ms** after the release, in both stages — against three copies that FAILED and cost 66.4 ms of
blocking at HEAD.

## 3. No behavioural change

Clean runs, greedy, config of record, both GPUs. **Raw token ids are byte-identical** (md5 of the `T <id>` lines):

| arm | prompt tokens | read | generated | `T`-line md5 | release calls |
|---|---|---|---|---|---|
| 4K before (`m6c/runs/p1-4k-before`) | 3832 | 15736 ms @ 243.5 tok/s | 150 @ 21.4 tok/s | `ad985c23cad68dcf1c462da621fc66e4` | 0 |
| 4K after (`m6c/runs/p1-4k-after`) | 3832 | 15812 ms @ 242.3 tok/s | 150 @ 21.5 tok/s | `ad985c23cad68dcf1c462da621fc66e4` | 0 |
| 32K before (`m6c/runs/p1-32k-before`) | 32256 | 92722 ms @ 347.9 tok/s | 256 @ 21.8 tok/s | `124a3cd39b33f7da31e225829625983a` | 0 |
| 32K after (`m6c/runs/p1-32k-after`) | 32256 | 92901 ms @ 347.2 tok/s | 256 @ 21.8 tok/s | `124a3cd39b33f7da31e225829625983a` | 0 |

The 32K arms reproduce M6c's published point (prefill 344.8 → 347.9/347.2 tok/s, decode 21.71 → 21.8/21.8), and the
4K arms reproduce M6c's needle arm (`DONE 150 3832`). Prefill differs by 0.2-0.5 % between the arms — run-to-run
noise, not a regression; the decode-side breakdown is unchanged (`STRATA_DECODE_TIMING`, 32K: 137.99 ms/window
before, 138.14 after; verify 115.12 → 115.28; GPU-reach wait 44.30 → 44.27).

**There is no decode win, and the card's "possible decode win" is measured as none**: `release_gpu_waits` is called
only from failure paths — `verify.cpp:1261` (the layer ring's 20 s bound), `:1307` (the tail's bounded wait),
`pool.cpp:486,509` (the CPU pool's stall watchdog), `generate.cpp:4705` (the serve no-progress watchdog) and
`verify.cpp:140` (every live verifier, before the engine ends) — and a clean 4K/32K run calls it exactly **0** times.
The value of this change is robustness and the instrument, not tok/s.

## 4. The instrument: still no report, but the blocker has moved off the release

`unitrace --start-paused --session X --device-timing` (S3UT's minimal stalling mode; S3 counts it as a "stall"
configuration), config of record, 4K, `GEN 16`:

| | before (`p1/logs/p1-ut7-before`, HEAD `3ccb530`) | after (`p1/logs/p1-ut7-after`, `b538baf`) |
|---|---|---|
| prompt read under the trace | completes (`PP 3831 3832 25402 150.8`) | completes (`PP 3831 3832 25473 150.4`) |
| the window | stalls: `no progress for 60 s … waiting for the GPU to reach layer 1`; host at layer step 1, GPU rang 1 | stalls at the same step: `ERR verify: timed out at layer 1` (`verify.cpp:1261`) |
| the release | publication **4.142 ms**, 3 × `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`; drain 1384.568 ms | publication **0.078 ms**, no failures; drain 5001.122 ms |
| the engine afterwards | no further output; harness killed at its 420 s cap | **never exits** (alive 6+ min: main thread `R` in user space at ~92 % of a core, 102 threads in `futex`, 1 in `clock_nanosleep`); harness stopped the session |
| unitrace report | trace dir 0 files | trace dir 0 files |

So: `--device-timing` still does not produce a report, and the reason is no longer the release. The release is
measured to publish in 0.078 ms with no error, and the read-back probe shows the device words receiving
`UINT32_MAX`; what remains is that the engine does not come back from the failed window at all, so the traced
process never exits and unitrace never flushes.

**The blocker is named** (`p1/logs/p1-gdb7-after-out.log`; engine under gdb — gdb is its parent because
`ptrace_scope=1` — with unitrace `--device-timing`, gdb `thread apply all bt` 60 s after the ask, i.e. after the
release and its 5 s drain). Its own trail stops at the same step the report names:

```
verify dbg: launched
verify dbg: layer 0 rang
verify dbg: layer 0 served                    <- nothing after this
verify release: flag publication 0.082 ms; drain budget 5000 ms
verify release: the GPU did NOT finish; drain 5000.095 ms, release 5000.197 ms
```

and the main thread (the only thread in user space; the other 104 are parked):

```
#0  __GI_sched_yield ()
#1  libze_intel_gpu.so.1
#2  libze_tracing_layer.so.1
#3  v2::ur_queue_immediate_in_order_t::queueFinish()
#4  ur::level_zero::urQueueFinish(ur_queue_handle_t_*)
#5  urQueueFinish
#6  sycl::_V1::detail::queue_impl::wait
#7  sycl::_V1::queue::wait_and_throw_proxy
#8  strata::sycl_compat::cudaStreamSynchronize(stream_t*)
#9  strata::core::Verifier::~Verifier()
#10 main
```

So the host is no longer in the release's copy (S3UT's frame was `release_gpu_waits`), it is in the **verifier's
destructor**: the failed window leaves work resident on the stream, `~Verifier` synchronizes that stream
(`src/core/verify.cpp:299`, `if (cs_) cudaStreamSynchronize(cs_);`), and `urQueueFinish` spins inside
`libze_intel_gpu`. **Next hypothesis, now with a frame**: that sync needs the same treatment the release got — a
bounded wait when the verifier is destroyed after a window that already failed — and, for the instrument
specifically, a trace that can be flushed without the process reaching a clean exit (unitrace writes nothing while
the app is alive). `perf` cannot help here (`perf_event_paranoid=4`); the gdb-as-parent trick is the tool.

## 5. Not validated

* **CUDA/HIP keep the blocking `cudaMemcpy`.** The defect is measured on SYCL only; neither backend can be built or
  run on this box, so their release path is deliberately unchanged.
* **The residual non-drain is not explained.** In `STRATA_TEST_VERIFY_STALL=1` (4K) and under `--device-timing` the
  window still does not finish within the 5 s drain bound, **before and after** the change (same rigs, same
  outcome). The release's own guarantee — "raise every flag past any ring" — is now measured to be delivered, so
  the non-finish is downstream of it. The instrument run names where the host then goes (`~Verifier` →
  `cudaStreamSynchronize` → `urQueueFinish` inside `libze_intel_gpu`, §4), but that step is **not** fixed here.
* **One instrument mode re-tested.** `--device-timing` only. S3UT's other stalling modes
  (`--chrome-kernel-logging`, `--chrome-device-logging`, and the full `UTRACE=1` configuration) were not re-run.
* **A pre-existing abort on the failure path, not diagnosed here.** Both the before and the after binary die with
  `corrupted double-linked list` → SIGABRT (`exit -6`, harness sees 250) in the `STRATA_TEST_VERIFY_STALL=1` rig.
  It is present with the HEAD-equivalent binary, so it is not from this change; it is worth its own card.
* One prompt and one engine per arm, no per-arm variance estimate, no 64K/256K arms (the card asked 4K and 32K), no
  MTP-off arm, no quality sweep.
* Nothing pushed: `origin` has no `sycl-xpu` branch.

## 6. Files

| what | path |
|---|---|
| the change (fix) | `src/core/verify.cpp` — commit `e452dd0` (+ notes `b538baf`) |
| the release instrumentation | `src/core/verify.cpp` — commit `3ccb530`, on by `STRATA_VERIFY_RELEASE_DEBUG=1` |
| the failure-path A/B rig (explicit engine binary, stall hook) | `~/strata-xpu/p1/p1_run_engine.sh` |
| the unitrace arm harness (does NOT kill the engine on a stall) | `~/strata-xpu/p1/p1_utrace.sh` |
| the gdb + unitrace stack grab | `~/strata-xpu/p1/p1_gdb.sh` |
| thread-state sampler for a still-alive engine | `~/strata-xpu/p1/p1_threadstate.sh` |
| clean-run arms (4K/32K, before/after) | `~/strata-xpu/m6c/runs/p1-{4k,32k}-{before,after}/{log,err,out}.txt` |
| failure-path arms (stall hook) | `~/strata-xpu/p1/runs/p1-4k-stall-beforebin/`, `~/strata-xpu/m6c/runs/p1-4k-stall-{before,after,diag,probe}/` |
| the traced arms | `~/strata-xpu/p1/logs/p1-ut7-{before,after}*`, traces in `~/strata-xpu/p1/traces/` |
| the binaries compared | `~/strata-xpu/p1/strata-before-timed` (3ccb530), `~/strata-xpu/p1/strata-after-P1` (b538baf) |
| copies committed to the repo | `p1/evidence/` |
