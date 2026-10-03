# P1b — the engine now comes back from a failed verify window (and the reason it did not was a false-negative wait)

Card `t_58d5592c` (P1b), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`.  Base **3d84db0** (P1's
write-up), the change is **f21ba10** and the commit that adds this file.  Nothing pushed (origin has no
`sycl-xpu`).  Config of record throughout: `strata-sycl-iq3s.json` — IQ3_S GSQ-RCO snapshot `ed59f92…`, both B70s
(`ZE_AFFINITY_MASK` unset, `--layer-split auto`), `--kv int8`, `--no-capture`, `--mmap-experts`, `--spec 4`, one
engine at a time.  Failure arms use the repo's own hook (`STRATA_TEST_VERIFY_STALL=1`, the N-th verify window keeps
its last layer's served flag withheld).  Binary under test: `build-sycl/strata`, md5 `14861c54671d005733d89ff53cb9dc52`,
kept at `~/strata-xpu/p1/strata-after-P1b`; the arms' own lines are quoted verbatim below.

## 0. Verdict

**Shape of the fix: (a), the teardown must not block on a stream whose window already failed — and the measurement
that chose it also exposed why the release's own drain never drained.**  P1 named `~Verifier`'s
`cudaStreamSynchronize(cs_)` (`src/core/verify.cpp:299` at 3d84db0) as the site where the host parked after a failed
window.  Bounding that sync is not enough on its own, and the reason is a second defect in the *measurement*:

* the release's drain (`release_gpu_waits`) polled the stream with `cudaStreamQuery`, and on this backend that call
  is not a "is the stream done" spelling: the shim submits a **fresh barrier on every call and then reports that
  barrier's status** (`include/strata/sycl_compat/cuda_runtime.h:806-813`).  A poll loop built on it re-arms its own
  obstacle once a millisecond, so it reported "the GPU did NOT finish" at **every** bound while the stream was in
  fact finishing.  Measured, one run, the same two streams: the drain reported `did NOT finish` for
  **30000.709 ms**, and the very next lines are the destructors' blocking syncs of those same streams returning in
  **0.003 ms** and **1.023 ms** (`~/strata-xpu/p1/runs/p1b-a-longdrain/err.txt:59-69`);
* so the fix is (a) **plus**: the drain asks with **one recorded event, polled** (`cudaEventQuery`, no re-arm), and
  the teardown's wait is bounded.  With the truthfully-measured drain the release now reports what really happens:
  untraced, the window's GPU work **does** finish — `the GPU finished; drain 3.180 ms` / `2.398 ms`
  (`p1b-e-fixed/err.txt:59,65`), and the two verifiers' teardown waits return in `1.056 ms` and `0.053 ms`;
* the process then ends instead of spinning: `exit 250`, wall 110 s (untraced) — against P1's instrument run where
  the only user-space thread sat in `urQueueFinish` for 6+ minutes and the process had to be killed;
* and when the GPU genuinely cannot be ended — which is what the tracer does — the engine no longer walks into the
  next queue release that spins in `queueFinish`: it ends at once, by design, with a line saying so (§4).

The glibc `corrupted double-linked list` → SIGABRT that P1 saw on this path is gone as well (§5): it was in the
SYCL/UR program-release path at process exit, with a release still in flight in another thread; with a truthful
drain the release is over in ~3 ms and the race does not happen.

## 1. The change

`src/core/verify.cpp` (+ the harnesses under `p1/`), commit **f21ba10**:

| site | before (3d84db0) | after |
|---|---|---|
| `~Verifier`: the window's stream | `verify.cpp:299` — `if (cs_) cudaStreamSynchronize(cs_);` (unbounded) | `verify.cpp:394-412` — the clean path keeps that call (`:396`); a **released** window (`released_`, the same state P1's release sets) gets `wait_stream_bounded(cs_, STRATA_TEARDOWN_WAIT_MS=5000)` |
| the same, when the GPU cannot be ended | (blocked there forever) | `verify.cpp:398-407` — the line is printed and `std::_Exit(250)` is called (`:407`): the driver teardown of this verifier is skipped, because the next driver call (`cudaStreamDestroy`) is itself a `queueFinish` spin (§4) |
| `release_gpu_waits`'s drain | `verify.cpp:261` — `while (cudaStreamQuery(s) == cudaErrorNotReady)` (false negative by construction, and it queues one barrier per millisecond on the stream it is asking about) | `verify.cpp:316` — `wait_stream_bounded(s, timeout_ms, &w)` |
| the primitive | — | `verify.cpp:88` `wait_stream_bounded()`: one `cudaEventRecord` + a `cudaEventQuery` poll with a budget |
| rig knobs (off by default) | — | `STRATA_RELEASE_DRAIN_MS` (a call site's drain bound), `STRATA_TEARDOWN_WAIT_MS` (the teardown's, 5000 ms) |

Instrumentation that stays (all on `STRATA_VERIFY_RELEASE_DEBUG=1`, which the failure rigs already set): the release
enters with the thread id and the budget, prints the publication/drain wall times, and on a timeout the host flag
words and the window's diag line; the teardown prints what it decided and how long it waited.

## 2. What the truncation of the wait does NOT change: every healthy path

Clean runs, greedy, config of record, both GPUs, `STRATA_TEST_VERIFY_STALL` unset.  **Raw token ids are
byte-identical to P1's HEAD record** — same recipe (`grep '^T ' out.txt | md5sum`), recomputed on P1's own arms to be
sure: `p1-4k-{before,after}` = `ad985c23cad68dcf1c462da621fc66e4`, `p1-32k-{before,after}` =
`124a3cd39b33f7da31e225829625983a`.

| arm | prompt tokens | generated | `T`-line md5 | prefill | decode |
|---|---|---|---|---|---|
| P1 record 4K (before / after) | 3832 | 150 | `ad985c23cad68dcf1c462da621fc66e4` | 243.5 / 242.3 tok/s | 21.4 / 21.5 tok/s |
| **P1b 4K**, `p1b-j-clean4k` (f21ba10) | 3832 | 150 | `ad985c23cad68dcf1c462da621fc66e4` | 241.8 | 21.4 |
| **P1b 4K**, `p1b-u-clean4k` (final binary) | 3832 | 150 | `ad985c23cad68dcf1c462da621fc66e4` | 243.3 | 21.5 |
| P1 record 32K (before / after) | 32256 | 256 | `124a3cd39b33f7da31e225829625983a` | 347.9 / 347.2 | 21.8 / 21.8 |
| **P1b 32K**, `p1b-k-clean32k` | 32256 | 256 | `124a3cd39b33f7da31e225829625983a` | 347.2 | 21.8 |

The decode breakdown is P1's to two decimals (`STRATA_DECODE_TIMING`, 4K: 142.62 ms/window = verify 121.35
[GPU-reach wait 46.87 + per-layer host 0.53] + commit/emit 2.06 + draft 19.21).  No behavioural change is expected
from the change on this path and none is measured: the fallback for `wait_stream_bounded` sits behind
`released_.load()`, which only three call sites can set — `generate.cpp:4705` (the serve watchdog),
`pool.cpp:486`, `pool.cpp:509` (the pool's stall watchdogs) — and a clean run calls none of them.

## 3. The engine comes back: the second ask (task 2)

`p1b-l-secondask` (4K, `STRATA_TEST_VERIFY_STALL=1`, `GEN 16` then a second `GEN 8`, then `QUIT`):

* the first ask's window fails as the rig intends and ends the request with
  `ERR verify: an earlier window never finished on the GPU (#267); restart the engine`
  (`p1b-l-secondask-out.txt`, the request's own answer);
* the second ask is **not served** — no `PP` line and no token line belongs to it, `T lines: 0`
  (`p1b-l-secondask-log.txt`) — because the engine has already been told to stop by issue #29's watchdog and ends;
* the engine **exits** instead of hanging: `exit 250, wall 110 s` (twice: `p1b-o-stall`, `p1b-t-stall`).

**Honest negative:** after a failed window this engine does not serve the next request — the serve layer's own
contract says so ("restart the engine", `verify.cpp:1365`), and the engine's end is now deterministic: 110 s, exit
250, no kill by the harness.  The release itself is now measured to do its job before that: `the GPU finished;
drain 3.508 ms` and `2.472 ms`, and the request's window unblocks (`p1b-t-stall/err.txt:57-66`).

## 4. The instrument (task 4): no mode writes a report — but the engine now ends by itself under every mode

`unitrace --start-paused --session X …`, config of record, 4K, `GEN 16`:

| mode | engine, **P1 binary** | engine, **final binary** | trace dir |
|---|---|---|---|
| `7` `--device-timing` (S3's minimal stalling mode) | never exits; harness stops the session (P1: main parked in `~Verifier → cudaStreamSynchronize → urQueueFinish` for 6+ min) | **exits by itself, `exit 250`, wall 88 s** (`p1b-r-ut7`) | **0 files** |
| `9` `--chrome-kernel-logging --chrome-device-logging` | (not run by P1) with the intermediate binary still alive 180 s past the release, SIGINT at 575 s | **exits by itself, `exit 250`, wall 91 s** (`p1b-s-ut9`) | **0 files** |
| `1` the full set (`--host-timing --device-timing --verbose` + all three chrome logs) | with the intermediate binary alive 300 s past QUIT, SIGINT at 574 s | **exits by itself, `exit 250`, wall 90 s** (`p1b-v-ut1`) | **0 files** |

So every configuration now yields **no** chrome JSON and no device-timing summary (the same 0-byte trace dir
P1 recorded), and the difference from P1 is that the app is no longer the reason: the engine terminates on its own
within ~90 s in all three modes instead of being SIGKILLed/SIGINTed with the tracer still holding its buffers.

**The frames, in order — this is the "new frame" the card asked for.**

1. P1's frame (before): `~Verifier → cudaStreamSynchronize(cs_) → urQueueFinish → libze_intel_gpu`
   (`~/strata-xpu/p1/logs/p1-gdb7-after-out.log`).
2. With the sync bounded (intermediate binary), gdb as the engine's parent under `unitrace --device-timing`
   (`p1b-m-gdb7-out.log`, 200 s after the ask) — the host moved one call down, to the queue's own release:

```
#0  libze_intel_gpu.so.1
#1  libze_tracing_layer.so.1
#2  v2::ur_queue_immediate_in_order_t::queueFinish()
#3  v2::ur_queue_immediate_in_order_t::~ur_queue_immediate_in_order_t()
#5  ur::level_zero::urQueueRelease
#6  urQueueRelease
#7  sycl::_V1::detail::queue_impl::~queue_impl()
#8  strata::core::Verifier::~Verifier()          <- cudaStreamDestroy(cs_)
#9  main
```

3. Final binary: **no frame at all** — the host is not parked anywhere.  Under the tracer the released window's
   stream is never observed complete (`the GPU did NOT finish; drain 5000.536 ms`, with the device flag words already
   `4294967295 4294967295 4294967295`), so `~Verifier` ends the engine instead of walking into (2):

```
verify teardown: the window on cs_ was released (#267) and has NOT finished in 5000.560 ms; the GPU work cannot be
ended from here and the driver teardown is skipped - the engine ends now (a queue release on a queue whose work
never finished spins in queueFinish)
```

**What this says about the instrument, and what it does not.**  The engine-side blocker the card exists for is gone:
a failed window no longer leaves a host thread spinning in `urQueueFinish`, and the process ends under every mode.
The tracer still produces no report, and the engine can no longer be the explanation (it exits); the leading
hypothesis, not yet tested, is on the instrument's side: `--device-timing` wedges the verify window's GPU work (the
release cannot end it — measured here, 5 s and 30 s bounds both, with the flags already past every ring), and an
engine that cannot end that work can only leave through a path that runs no app-side finalizers
(`std::_Exit`), which is also the path that skips the in-process tracing layer's own flush.  Deciding that is P2's
job; the evidence it needs is `p1b-r-ut7`, `p1b-s-ut9`, `p1b-v-ut1` and the frame above.

## 5. The pre-existing SIGABRT on the failure path: settled, with the mechanism

P1 left this undiagnosed: in the untraced stall rig the engine died with glibc's `corrupted double-linked list`
(`exit -6`, harness sees 250) with both the before and the after binary.  It is gone after this change (three
failure-path runs: `p1b-e-fixed`, `p1b-o-stall`, `p1b-t-stall` — no such line, and the engine ends through its own
path each time), and the mechanism is measured with gdb as the parent (`p1b-d-abort.log`):

```
#7  malloc_printerr ("corrupted double-linked list")     #8 unlink_chunk  #9 _int_free_merge_chunk
#11 libze_intel_gpu.so.1
#14 ur_program_handle_t_::ur_release_program_resources(bool)
#15 ur_program_handle_t_::~ur_program_handle_t_
#16 ur::level_zero::urProgramRelease
#18 sycl::_V1::detail::Managed<ur_program_handle_t_*>::~Managed()
#20 sycl::_V1::detail::KernelProgramCache::removeProgramByKey(...)
#22 sycl::_V1::detail::ProgramManager::removeImages(sycl_device_binaries_struct*)
#23 sycl.descriptor_unreg
#24 _dl_call_fini
#26 __run_exit_handlers      #27 __GI_exit     #28 __libc_start_call_main
```

i.e. it is the **SYCL runtime's own exit teardown** (`sycl.descriptor_unreg` → the L0 program release → the
driver's `free`) aborting the process, and it ran while the serve watchdog thread was still inside
`release_gpu_waits` — the log's own ordering is `release pass slot 1 → publication 0.035 ms → teardown of both
verifiers → corrupted double-linked list` (`p1b-a-longdrain-err.txt:63-70`).  Same root as the card's own subject:
the failure path ends the process while another thread is still in the driver.  With the release's drain truthful
and bounded (3 ms), main's exit no longer overlaps it, and the exit teardown is clean.  It is the same family, so it
was settled here rather than filed separately — but note the *fix* is the truthfulness of the drain, not a lock: no
mutual exclusion between the release and the exit was added, and that is on the not-validated list below.

## 6. Not validated

* **CUDA/HIP are unchanged and unmeasured.** `wait_stream_bounded` uses the shim's `cudaEventQuery` on both arms and
  the released-window branch is backend-neutral, but only the SYCL build runs on this box; the CUDA/HIP arms were
  not built.
* **`STRATA_TEARDOWN_WAIT_MS` was never swept.** Every arm used the 5000 ms default; the tracer case needs the whole
  budget and the untraced case needs 1 ms, but nothing between was measured.
* **No mutual exclusion between the release and the process exit.** The glibc abort no longer reproduces (3 runs),
  and it is explained by the measured ordering above, but the race is closed by timing, not by a lock.
* **The instrument's own reason for 0 files is a hypothesis.** `--device-timing` wedging the verify window's GPU
  work is measured from the engine's side (the release cannot end it, at 5 s and 30 s); that the tracer's in-process
  flush is what a `std::_Exit` skips was not tested.  No unitrace mode writes a report in any configuration tried.
* **The pool's release call sites were not exercised** (`pool.cpp:486,509`): they need the pool's own stall watchdog,
  which this rig does not reach (19 of 19 workers parked in every arm).
* One prompt, one engine per arm, no variance estimate; no 64K/256K arms; no MTP-off arm; no quality sweep.
* Nothing is pushed: origin has no `sycl-xpu` branch.
* Mode 7's first attempt (`p1b-g-ut7`) is an environment casualty, kept as evidence: the engine was given 568 MiB of
  free VRAM at load (the record's own runs see ~5.7 GiB) and died in the prefill with
  `memcpy failed … UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, exit 1 in 33 s.  It is the same *class* as the defect P1
  fixed (a copy that fails), on a device with no room, and was not reproduced afterwards.

## 7. Files

| what | path |
|---|---|
| the change | `src/core/verify.cpp` — commit `f21ba10` + the commit that adds this file |
| the same-file context it replaces | `src/core/verify.cpp:299` at `3d84db0` (P1's named site) |
| the failure rig (explicit binary, stall hook, knobs) | `~/strata-xpu/p1/p1b_run_engine.sh` |
| the batteries (instrument + clean arms, sequential) | `~/strata-xpu/p1/p1b_battery{2,3,4}.sh`, `p1b_battery.sh` |
| gdb-as-parent under unitrace, held for the abort | `~/strata-xpu/p1/p1b_gdb_abort.sh` |
| md5 / report helpers | `~/strata-xpu/p1/p1b_md5.sh`, `p1b_clean_report.sh`, `p1b_evidence.sh` |
| P1's harnesses (unchanged, reused) | `p1/evidence/{p1_run_engine.sh,p1_utrace.sh,p1_gdb.sh,p1_threadstate.sh}` |
| the failure path before / after | `~/strata-xpu/p1/runs/p1b-a-longdrain/`, `…/p1b-e-fixed/`, `…/p1b-o-stall/`, `…/p1b-t-stall/` |
| the second ask | `~/strata-xpu/p1/runs/p1b-l-secondask/` |
| the clean arms | `~/strata-xpu/p1/runs/p1b-{j,u}-clean4k/`, `…/p1b-k-clean32k/` |
| the frames | `~/strata-xpu/p1/logs/p1b-m-gdb7-out.log`, `~/strata-xpu/p1/logs/p1b-d-abort.log` |
| the instrument arms | `~/strata-xpu/p1/logs/p1b-{f,g,h,i,n,q,r,s,v}-ut*`, traces in `~/strata-xpu/p1/traces/` |
| the binaries compared | `~/strata-xpu/p1/strata-after-P1b` (md5 `14861c54671d005733d89ff53cb9dc52`), `~/strata-xpu/p1/strata-before-timed`, `~/strata-xpu/p1/strata-after-P1` (P1's) |
| curated copies committed with this file | `p1/evidence/p1b-*` |
