// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
// P1 (card t_44a0ac61): time the #267 release path (STRATA_VERIFY_RELEASE_DEBUG=1).
const bool g_release_dbg = std::getenv("STRATA_VERIFY_RELEASE_DEBUG") != nullptr;
// P1b (card t_58d5592c) RIG-ONLY KNOB: every call site of release_gpu_waits passes a literal 5000 ms bound.  This
// lets a run set that bound from the environment so "does the window EVER finish after the release?" can be
// answered without editing the call sites (STRATA_RELEASE_DRAIN_MS=<ms>; unset = the call site's own bound).
int64_t release_drain_ms(int64_t timeout_ms) {
    const char* e = std::getenv("STRATA_RELEASE_DRAIN_MS");
    if (e == nullptr) return timeout_ms;
    const long long v = std::atoll(e);
    return v > 0 ? (int64_t) v : timeout_ms;
}
// P1b (card t_58d5592c): the teardown's own bound, and the knob that sets it (STRATA_TEARDOWN_WAIT_MS).
int teardown_wait_ms() {
    const char* e = std::getenv("STRATA_TEARDOWN_WAIT_MS");
    if (e == nullptr) return 5000;
    const long long v = std::atoll(e);
    return v >= 0 ? (int) v : 5000;
}
// P1b: how long the process's own finalizers get, once the engine has decided to end on a window it cannot drain
// (STRATA_FINALIZER_WAIT_S, default 10 s).  See the comment at the exit in ~Verifier.
int finalizer_wait_s() {
    const char* e = std::getenv("STRATA_FINALIZER_WAIT_S");
    if (e == nullptr) return 10;
    const long long v = std::atoll(e);
    return v > 0 ? (int) v : 10;
}
/// P1b (card t_58d5592c): A BOUNDED, TRUTHFUL "has this stream finished?".  Returns true if the stream was
/// observed complete within `budget_ms`; false if the budget ran out.  ONE recorded event, polled with
/// cudaEventQuery - deliberately NOT cudaStreamQuery, which is not a "is the stream done" spelling on this
/// backend: the shim's cudaStreamQuery submits a FRESH BARRIER on every call and then reports THAT barrier's
/// status (include/strata/sycl_compat/cuda_runtime.h:806-813), so a poll loop built on it re-arms its own
/// obstacle on every iteration and can report not-ready for as long as it runs.  MEASURED, this rig, the same
/// stream in the same instant: 30000.709 ms of cudaStreamQuery not-ready in the release's drain, while the
/// destructor's blocking cudaStreamSynchronize of that stream returned in 0.003 ms and 1.023 ms (two verifiers,
/// ~/strata-xpu/p1/runs/p1b-a-longdrain/err.txt lines 59-69).  That false negative is what made the release
/// report "the GPU did NOT finish" at every bound, and it is also why the release's drain used to leave
/// thousands of useless barriers queued on the stream it was asking about.
bool wait_stream_bounded(cudaStream_t s, int budget_ms, double* waited_ms) {
    if (s == nullptr) { if (waited_ms != nullptr) *waited_ms = 0.0; return true; }
    cudaEvent_t ev = nullptr;
    if (cudaEventCreate(&ev) != cudaSuccess) { if (waited_ms != nullptr) *waited_ms = -1.0; return false; }
    cudaEventRecord(ev, s);
    const Clock::time_point t0 = Clock::now();
    bool done = false;
    for (;;) {
        if (cudaEventQuery(ev) == cudaSuccess) { done = true; break; }
        if (ms_since(t0) > (double) budget_ms) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cudaEventDestroy(ev);
    if (waited_ms != nullptr) *waited_ms = ms_since(t0);
    return done;
}
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

/// The FLAG words are the one staging buffer that is NOT mapped (Strata XPU M5b, card t_3ebd6083).
///
/// Every other buffer here is written by the host and read once by a kernel, and a mapped buffer is fine for
/// that: probe_m5b_payload.cpp measured a 16 KiB and a 1 MiB mapped payload rewritten per round and read once
/// to be read FRESH every time (12 of 12 rounds).  A flag is different - it is read by a POLL LOOP, and on this
/// backend a poll loop over mapped host memory never sees the host's stores: the device's cache answers from a
/// stale line, `_mm_clflush`/`_mm_sfence`/non-temporal stores do not change that, and a system-scope atomic
/// load of mapped host memory is unreliable too (measured: one transition to the last value after ~300 ms, and
/// a sparse subset for the atomic spellings - probes probe_m5b_observe.cpp / probe_m5b_flush.cpp,
/// plan-evidence/m5b-observe-run.log, m5b-flush-run.log).
///
/// So the host keeps a mapped word for its OWN reads (the diag line, the raise-only CAS) but the word the
/// KERNELS read is DEVICE memory, and the host publishes into it with a 4-byte copy (publish_flag /
/// raise_flag_dev).  That combination - device flag word, host publication by copy, system-scope atomic poll
/// (include/strata/kernels/flag.hpp) - is what measured 10 of 10 values in three separate runs, and what makes
/// tests/sycl/handoff.cpp complete (100 of 100 rounds; it used to hang at 300 s, M2 Risk 8).
bool flag_mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    *d = nullptr;
    if (cudaMalloc(d, bytes) != cudaSuccess) return false;
    return cudaMemset(*d, 0, bytes) == cudaSuccess;
}

/// P1 (card t_44a0ac61): raise the release flag in DEVICE memory on a stream, and return WITHOUT waiting for
/// the copy (SYCL) - or with the original blocking copy where the shim is not in the way (CUDA/HIP).
///
/// WHAT WAS WRONG WITH THE BLOCKING `cudaMemcpy` THE RELEASE USED.  On this build that call is the shim's
/// faithful emulation of CUDA's blocking copy: memcpy_impl(..., &default_queue(), wait = true)
/// (include/strata/sycl_compat/cuda_runtime.h:636-640), which submits on the DEFAULT queue and then waits for
/// the copy's event (cuda_runtime.h:596-604).  Two measured consequences, both on the path whose whole purpose
/// is to end a window that has already failed (~/strata-xpu/p1/logs, this card):
///
///   * it WAITS, and the wait can be the thing that never returns.  Under unitrace the host thread was parked
///     in exactly this wait for as long as the engine lived (S3UT, card t_7e1307a6, gdb stack:
///     release_gpu_waits -> memcpy_impl -> urEventWait -> libze_tracing_layer.so.1), and the engine's 60 s
///     watchdog then SIGKILLed the process, which is why unitrace never wrote a report.
///   * it runs on the DEFAULT queue, which is not the verifier's own stream.  Measured at HEAD (instrument-only
///     commit 3ccb530) in both the untraced STRATA_TEST_VERIFY_STALL=1 rig and under `unitrace --device-timing`,
///     all three of the release's copies failed:
///         strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
///     i.e. the release never reached the device flag word at all, and the #267 guarantee it exists for was not
///     being met on this path.
///
/// WHY SUBMIT ON `copy_` AND NOT WAIT.  `copy_` is the verifier's own copy stream - the one every other flag
/// publication in a window already uses (publish_flag / raise_flag_dev), on the verifier's own device, and the
/// stream M5b measured a 4-byte copy through to reach a spinning kernel (10 of 10 rings, three runs).  The
/// release value is UINT32_MAX, past every ring the kernels compare against, so putting it on the same
/// in-order stream as the windows' other publications cannot make a kernel read a ring it has not been served
/// (a later, larger value never satisfies an earlier wait it should not have).  And the drain loop that
/// follows polls exactly this stream, so "did the release get through" is still answered before this returns.
///
/// WHY NOT `cudaMemcpyAsync`.  That is the shim's submitting spelling, but it is capture-aware: with a capture
/// open it RECORDS a node instead of submitting (cuda_runtime.h:648-651).  A recorded release reaches the
/// spinning kernel only if the graph is launched later - and the window that just failed is the one that would
/// have launched it.  The release must be submitted even then, which is the guarantee the old blocking call was
/// picked for.  This helper keeps that guarantee and drops the wait.
///
/// MEASURED after the change, with a temporary read-back probe on this same in-order stream (a 4-byte copy back
/// to a host word, ordered behind the publication): the three DEVICE words read UINT32_MAX 1.05 ms after the
/// release, in both stages, against three copies that FAILED at HEAD and cost 66.4 ms of blocking.  So the part
/// of #267 that is "raise every flag past any ring" is now actually delivered on this path.
void publish_release_word(cudaStream_t s, void* dev, const void* host) {
#if defined(STRATA_USE_SYCL)
    (void) sycl_compat::memcpy_impl(dev, host, sizeof(uint32_t),
                                    sycl_compat::queue_for(reinterpret_cast<void*>(s)), /*wait=*/false);
#else
    // CUDA/HIP: unchanged.  The defect above is measured on the SYCL backend only, and this box cannot build
    // or run the other two, so their release stays the blocking cudaMemcpy it always was (see flag.hpp for the
    // same backend-split convention).
    cudaMemcpy(dev, host, sizeof(uint32_t), cudaMemcpyHostToDevice);
#endif
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    int slot_i = 0;
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            if (f != nullptr)
                std::fprintf(f, "strata: #267 release pass over live verifier slot %d (thread %llu)\n", slot_i,
                             (unsigned long long) std::hash<std::thread::id>{}(std::this_thread::get_id()));
            ++slot_i;
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace

/// Publish a flag the KERNELS are waiting on: the store goes to the host's own word (the diag line and the
/// engine's own reads use it) and then, as a 4-byte copy on the copy stream, into the DEVICE word the spin
/// kernels read.  See flag_mapped's note for why a mapped store alone is not enough on this backend.
void Verifier::publish_flag(uint32_t* host_word, uint32_t* dev_word, uint32_t value) {
    *(volatile uint32_t*) host_word = value;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (dev_word != nullptr) cudaMemcpyAsync(dev_word, host_word, sizeof(uint32_t), cudaMemcpyHostToDevice, copy_);
}

/// The same, for a flag that may only be RAISED (flag B): the CAS stays on the host word and the word itself is
/// what gets copied, so the device sees the CAS's result.
void Verifier::raise_flag_dev(uint32_t* host_word, uint32_t* dev_word, uint32_t value) {
    raise_flag(host_word, value);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (dev_word != nullptr) cudaMemcpyAsync(dev_word, host_word, sizeof(uint32_t), cudaMemcpyHostToDevice, copy_);
}

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    timeout_ms = (int) release_drain_ms(timeout_ms);
    if (g_release_dbg)
        std::fprintf(stderr, "verify release: enter (thread %llu, budget %d ms)\n",
                     (unsigned long long) std::hash<std::thread::id>{}(std::this_thread::get_id()), timeout_ms);
    const Clock::time_point r0 = Clock::now();
    // the host's own words: UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    // M5b: the words the spin kernels read are DEVICE memory, so the release has to be published there too -
    // and a plain mapped store would never be observed by a poll loop on this backend (see flag_mapped's note).
    // P1 (card t_44a0ac61): SUBMITTED on the verifier's own copy stream, not a blocking copy on the default
    // queue - see publish_release_word's note for what the old form cost (a wait that never returned under the
    // instrument, and three copies that failed with UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY at HEAD).
    uint32_t* const flag_host[3] = {h_flag_, h_flagA_, h_flagB_};
    uint32_t* const flag_dev[3] = {m_flag_, m_flagA_, m_flagB_};
    const Clock::time_point t_pub = Clock::now();
    for (int i = 0; i < 3; ++i) {
        if (flag_host[i] == nullptr || flag_dev[i] == nullptr) continue;
        if (copy_ != nullptr) publish_release_word(copy_, flag_dev[i], flag_host[i]);
        else cudaMemcpy(flag_dev[i], flag_host[i], sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    // P1 (card t_44a0ac61) instrumentation, off by default: this path only ever runs after something has
    // already failed, so without a line here its cost - the publication of the flag to the DEVICE words above,
    // and the drain below - is invisible in a log.  STRATA_VERIFY_RELEASE_DEBUG=1 prints both.
    if (g_release_dbg)
        std::fprintf(stderr, "verify release: flag publication %.3f ms; drain budget %d ms\n",
                     ms_since(t_pub), timeout_ms);
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    bool finished = true;
    double bad_wait = 0.0;
    for (cudaStream_t s : {cs_, copy_}) {
        if (s == nullptr) continue;
        double w = 0.0;
        if (!wait_stream_bounded(s, timeout_ms, &w)) { finished = false; bad_wait = w; break; }
    }
    if (!finished) {
        if (g_release_dbg) {
            std::fprintf(stderr, "verify release: the GPU did NOT finish; drain %.3f ms, release %.3f ms\n",
                         bad_wait, ms_since(r0));
            // P1b (card t_58d5592c): what the window looks like when the GPU has not finished.  HOST words only:
            // they are mapped and always readable, and this branch must not add a copy of its own to a stream it
            // has just failed to observe (the device-word read-back that proved the release lands is a separate,
            // explicitly-flagged probe; see p1/STATUS-P1B.md).
            const uint32_t* const hw[3] = {h_flag_, h_flagA_, h_flagB_};
            std::fprintf(stderr, "verify release: host flag words %u %u %u; window state:\n",
                         hw[0] ? *(const volatile uint32_t*) hw[0] : 0u,
                         hw[1] ? *(const volatile uint32_t*) hw[1] : 0u,
                         hw[2] ? *(const volatile uint32_t*) hw[2] : 0u);
            diag(stderr);
            std::fflush(stderr);
        }
        return false;
    }
    if (g_release_dbg)
        std::fprintf(stderr, "verify release: the GPU finished; drain %.3f ms, release %.3f ms (publication %.3f ms)\n",
                     ms_since(t0), ms_since(r0), ms_since(t_pub));
    return true;
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // #251: outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u; tail beacons",
                 current ? "" : " (last window, not the current stage)", last_t_, (long long) last_pos0_,
                 cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    // M5b: the tail beacons (0 layers done, 1 head's read done, 2 quantize done, 3 mmvq done, 4 argmax done,
    // 5 unused, 6 the last layer's gr_write done): the last non-zero word is how far the GPU got.
    for (int i = 0; i < 8; ++i) std::fprintf(f, " %d", rd(h_beacon_ + i));
    std::fprintf(f, "\n");
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    // ---- P1b (card t_58d5592c): THE TEARDOWN MUST NOT BLOCK UNBOUNDED ON A STREAM WHOSE WINDOW FAILED.
    // Up to P1 this was the bare `if (cs_) cudaStreamSynchronize(cs_);` and it is where the host parked: P1's
    // instrument run (`~/strata-xpu/p1/logs/p1-gdb7-after-out.log`) caught the only user-space thread in
    // ~Verifier -> cudaStreamSynchronize -> urQueueFinish inside libze_intel_gpu for 6+ minutes, on a stream whose
    // window had already failed and whose flags the release had already raised to UINT32_MAX.  Measured here, in
    // the STRATA_TEST_VERIFY_STALL=1 rig: once the release has run, that stream does not finish at all (the
    // release's drain did not end it in 30 s), and a process that is ending cannot wait for the GPU.  So the
    // blocking sync stays on the clean path - a healthy window's stream is idle by then, and this is the
    // measured-cheaper spelling - and a window that was released gets the bounded ONE-EVENT wait instead.
    // ---- P1b (card t_58d5592c): WHEN THE GPU CANNOT BE DRAINED, THE ENGINE ENDS WITHOUT TOUCHING THE DRIVER AGAIN.
    // Bounding the sync above is not enough on its own: the next call that touches the wedged queue is the queue's
    // own release, and THAT spins in queueFinish too.  Measured, gdb as the engine's parent under
    // `unitrace --device-timing` (p1b-m-gdb7-out.log), once the sync had been bounded:
    //
    //   #0 libze_intel_gpu.so.1
    //   #1 libze_tracing_layer.so.1
    //   #2 v2::ur_queue_immediate_in_order_t::queueFinish()
    //   #3 v2::ur_queue_immediate_in_order_t::~ur_queue_immediate_in_order_t()
    //   #5 ur::level_zero::urQueueRelease
    //   #6 urQueueRelease
    //   #7 sycl::_V1::detail::queue_impl::~queue_impl()
    //   #8 strata::core::Verifier::~Verifier()          <- cudaStreamDestroy(cs_)
    //   #9 main
    //
    // i.e. `cudaStreamDestroy` -> `~queue_impl` -> `urQueueRelease` -> the UR queue's own destructor calls
    // queueFinish() unconditionally.  So: if the window's stream is observed complete, the normal teardown runs
    // (the queue is idle, everything below it is a plain release); if it is NOT complete, nothing in this process
    // can ever finish that work, and every remaining driver call is a potential queueFinish spin - the engine ends
    // at once, with the maps and the host allocations already the kernel's problem, instead of parking here.
    const bool released_teardown = released_.load();
    if (cs_) {
        if (!released_teardown) {
            cudaStreamSynchronize(cs_);
        } else {
            double w = 0.0;
            const bool done = wait_stream_bounded(cs_, teardown_wait_ms(), &w);
            if (!done) {
                std::fprintf(stderr,
                             "verify teardown: the window on cs_ was released (#267) and has NOT finished in %.3f ms; "
                             "the GPU work cannot be ended from here and the driver teardown is skipped - the engine "
                             "ends now (a queue release on a queue whose work never finished spins in queueFinish)\n",
                             w);
                std::fflush(stderr);
                // P1b: `std::exit` rather than `std::_Exit` so the process's OWN finalizers run - they are what
                // flushes the injected tracing layer and completes its chrome timeline.  Measured, mode 9, same
                // workload: with the finalizers allowed to run the timeline is 1,131,673,822 bytes / 1,987,510
                // device events, against 612,845,511 bytes / 1,063,792 events when the process is ended hard - half
                // the trace is what a hard end costs.  Those finalizers can themselves wedge on the queue this
                // window left behind, so a detached thread gives them a bounded window and then ends the process
                // hard with the report already written.
                std::thread([budget = finalizer_wait_s()] {
                    std::this_thread::sleep_for(std::chrono::seconds(budget));
                    std::fprintf(stderr, "verify teardown: the process's own finalizers have not finished in %d s; "
                                         "ending hard (the traced report was already flushed)\n", budget);
                    std::fflush(stderr);
                    std::_Exit(250);
                }).detach();
                std::exit(250);
            }
            std::fprintf(stderr, "verify teardown: the window on cs_ was released (#267); it finished (%.3f ms of a %d ms budget)\n",
                         w, teardown_wait_ms());
            std::fflush(stderr);
        }
    }
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    if (cs_) cudaStreamDestroy(cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (arena_) cudaFree(arena_);
    if (lad_) cudaFree(lad_);
    if (ladb_) cudaFree(ladb_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_beacon_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
    // M5b: the flag words the kernels read are DEVICE allocations (see flag_mapped's note), so they are freed
    // with cudaFree - they are not in the host-pointer list above.
    for (uint32_t* d : {m_flag_, m_flagA_, m_flagB_})
        if (d) cudaFree(d);
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    strata::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it: lower --max-context, use --kv k8v4 or images on the CPU";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              flag_mapped(4, (void**) &h_flag_, (void**) &m_flag_) &&
              flag_mapped(4, (void**) &h_flagA_, (void**) &m_flagA_) &&
              flag_mapped(4, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(64, (void**) &h_beacon_, (void**) &m_beacon_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        hist_snap_ = b.take<float>(T * HS);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    // M5g: the per-layer residual ladder (STRATA_DUMP_LADDER).  DEVICE memory, filled by kernel launches inside
    // the capture: a D2H memcpy node enqueued inside a verify window stalls this backend's graph replay (measured:
    // the window rings layer 0 and never layer 1, `logs/m5g-ladder_dbg.log`), which is why this is a device buffer
    // written by scale_inplace/add_inplace and copied out once, after the window.
    if (std::getenv("STRATA_DUMP_LADDER") != nullptr) {
        const size_t lad_floats = (size_t) (g.n_layers + 2) * (size_t) T * (size_t) (g.hc * g.n_embd);
        if (cudaMalloc((void**) &lad_, lad_floats * sizeof(float)) != cudaSuccess) {
            err = "verify: the ladder buffer does not fit";
            return false;
        }
        cudaMemset(lad_, 0, lad_floats * sizeof(float));
        const size_t ladb_floats = (size_t) (g.n_layers + 2) * (size_t) T * (size_t) g.n_embd;
        if (cudaMalloc((void**) &ladb_, ladb_floats * sizeof(float)) != cudaSuccess) {
            err = "verify: the attention-output ladder buffer does not fit";
            return false;
        }
        cudaMemset(ladb_, 0, ladb_floats * sizeof(float));
    }
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        device_plan_ = v != nullptr && std::atoi(v) != 0;
    }
    // M5b: the tail wait is a bounded, self-diagnosing poll when asked for (see run()'s tail note).  The value is
    // the budget in ms (default 60 s).
    {
        const char* v = std::getenv("STRATA_VERIFY_TAIL_DEBUG");
        if (v != nullptr && std::atoi(v) != 0) {
            tail_debug_ = true;
            if (std::atoi(v) > 1) tail_wait_ms_ = std::atoi(v);
        }
    }
    if (device_plan_) {
        bool ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); device_plan_ = false; }
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    auto stamp = [&](int64_t l, int i, int grp) { if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs); };
    // M5b: the TAIL beacons.  `gpu_stamp` cannot be used for this (the oneAPI device clock extension is not
    // supported on this part: sycl::aspect::ext_oneapi_clock_device is absent, so the profiler path throws), and
    // the point is to see how far the GPU got when the window's tail never finishes.  A beacon is the engine's own
    // doorbell ring (one thread, a system fence, an increment of a MAPPED word), so the host can read it after a
    // timeout with no driver call.  Word i is non-zero once stage i has been REACHED in some window; one window per
    // probe run, so the map from "words set" to "the GPU got this far" is unambiguous.  Recorded only when
    // STRATA_VERIFY_TAIL_DEBUG is on, so a healthy window pays nothing.
    auto beacon = [&](int i) {
        if (tail_debug_ && m_beacon_ != nullptr) strata::kernels::doorbell_ring(m_beacon_ + i, cs);
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    groups_[T] = G;
    // M5g: the per-layer residual ladder, `STRATA_DUMP_LADDER` (armed by `init`).  Two device kernels (zero, then
    // add) are enqueued per entry INSIDE the capture, so they are nodes of the window's graph and replay with it:
    // the ladder is the window's own residual, exactly as the window computed it, and no arithmetic is touched.
    // (A D2H memcpy per entry - the C1 ladder's mechanism - stalls this backend's graph replay; that is why the
    // ladder is a device buffer, see the note in `init`.)
    auto lad_copy = [&](int64_t slot, int tb, int te) {
        if (lad_ == nullptr || windows != 0) return;
        for (int t = tb; t < te; ++t) {
            float* dst = lad_ + ((size_t) slot * (size_t) MT + (size_t) t) * (size_t) (HC * N);
            strata::kernels::scale_inplace(dst, (int64_t) (HC * N), 0.0f, cs);   // dst = 0
            strata::kernels::add_inplace(dst, Rt(t), (int64_t) (HC * N), cs);    // dst = R_t
        }
    };
    // M5g: the same mechanism for the second ladder, `STRATA_DUMP_LADDER`'s `.bo` companion: the ATTENTION
    // HALF'S output (`bo_`) per layer, captured before `post` overwrites it with the MoE combine.  On this window
    // that is the only place a dead GDN/QSA block (a collapsed attention, an indexer that selected nothing)
    // is visible next to a live one, since both write the same buffer at the same point of the layer.
    // CONVENTION: the slot IS the layer index - `lad_copy_bo(l - lb_)` - so `.bo` slot k holds layer (lb_ + k)'s
    // attention output, and only slots 0 .. n_layers - lb_ - 1 are ever written.  (The residual ladder above is
    // offset by one, because its entry 0 is the window's input; the two files must not be read against each
    // other's convention.)
    auto lad_copy_bo = [&](int64_t slot, int tb, int te) {
        if (ladb_ == nullptr || windows != 0) return;
        for (int t = tb; t < te; ++t) {
            float* dst = ladb_ + ((size_t) slot * (size_t) MT + (size_t) t) * (size_t) N;
            strata::kernels::scale_inplace(dst, (int64_t) N, 0.0f, cs);
            strata::kernels::add_inplace(dst, bo_ + (size_t) t * N, (int64_t) N, cs);
        }
    };

    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);
    if (ple_on) copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        for (int t = 0; t < T; ++t) {
            copy_from_mapped(Rt(t), hand_in_ + (size_t) t * HB, HC * N, cs);
            copy_from_mapped(bo_ + (size_t) t * N, hand_in_ + (size_t) t * HB + HC * N, N, cs);
            copy_from_mapped(inj2_ + (size_t) t * HC, hand_in_ + (size_t) t * HB + HC * N + N, HC, cs);
        }
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }
    lad_copy(0, 0, T);   // M5g: entry 0 = the window's initial R (the embedding broadcast, or a split stage's input)

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            for (int t = tb; t < te; ++t) {
                gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                try {
                    ple_block(ple_ + t * N, Rt(t), ss.ple.hist, ss.ple.w, po, ss.ple.scratch, cs);
                    ple_history_advance(ss.ple.hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, ss.ple.hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            // M5g: THE FUSED-VS-PLAIN A/B ARM.  `fused_gr_read_multi` is the SYCL port's own read and the only
            // read this path uses; its self-check compares its variants against EACH OTHER on the device, so a
            // systematic porting error in the read is invisible to it.  The plain `gr_read` (src/kernels/cuda/
            // gr.cu) is the kernel whose arithmetic `scripts/m5g_headmix.py` reproduces in numpy to 1e-6 on this
            // pack, so forcing it here is a reference, not a guess:
            //   ZE_AFFINITY_MASK=0 STRATA_WINDOW_PLAIN_GR=1 ./build-sycl/strata ...
            // `apply` is the write the fused read folds in (pending FFN write), done explicitly first.
            static const bool plain_gr = std::getenv("STRATA_WINDOW_PLAIN_GR") != nullptr;
            if (plain_gr) {
                for (int t = tb; t < te; ++t) {
                    if (apply) strata::kernels::gr_write(Rt(t), bo_ + t * N, inj_prev + t * HC, gs, Rt(t), cs);
                    strata::kernels::gr_read(Rt(t), (const float*) wn[half]->data, (const uint16_t*) wd[half]->data,
                                             (const uint16_t*) wu[half]->data, (const uint16_t*) wi[half]->data, EPS,
                                             gs, ss.block.gr, mixed_ + t * N, inj_out + t * HC, cs);
                }
                return;
            }
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        gr_read_group(0, pending, inj2_, inj_);
        stamp(l, 1, grp);
        // M5g: the fused read applies the previous layer's write, so R is materialised here - entry (l - lb_ + 1)
        // is the residual ENTERING layer l, i.e. the exact output of layer l - 1.
        lad_copy(l - lb_ + 1, tb, te);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                stamp(l, 2, grp);
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb);
                stamp(l, 3, grp);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                stamp(l, 5, grp);
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, nullptr, cs, tb);
                stamp(l, 6, grp);
                native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);
                native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && !st.kv_q4;
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                else for (int t = tb; t < te; ++t)
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                stamp(l, 7, grp);
                native_mmvq(wk->native_type, wk->native_data, xq_, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                native_mmvq(wv->native_type, wv->native_data, xq_, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                for (int t = tb; t < te; ++t) {
                    const int32_t* step_t = step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int t = tb; t < te; ++t)
                    native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              rope_scaling(), cs);
                stamp(l, 9, grp);
                native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD),
                            n, cs);
                if (qb) {
                    if (cudaMemcpy2DAsync(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                stamp(l, 13, grp);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        lad_copy_bo(l - lb_, tb, te);   // M5g: the attention half's output, before `post` overwrites bo_
        stamp(l, 16, grp);
        gr_read_group(1, true, inj_, inj2_);        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10) {
            try {
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                          (uint32_t) ((l - lb_) * G + grp + 1), cs);
        doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                         m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
        stamp(l, 17, grp);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = xq_;
            if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, cs);   // contiguous rows
            else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, cs);
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, cs);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        if (strata::kernels::cpu::expert_layout().native)
            quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
            wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
            copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
        } else {
            wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
            copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
        }
        stamp(l, 19, grp);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, hit_out, cs);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
            }
        };
        grouped(p_ptr, p_start, p_counts);
        stamp(l, 20, grp);
        if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs);
        else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
        if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
            const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
            uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
            fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
        }
        stamp(l, 21, grp);
        grouped(p_ptr2, p_start2, p_counts + 2);
        stamp(l, 22, grp);
        if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
            wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
            copy_or_zero_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                     skip_ + grp, ring, cs);
        } else {
            wait_flag_ge(m_flag_, ring, cs);               // the CPU's share is in the mapped rows
            stamp(l, 23, grp);
            if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                copy_rows_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                      p_dst, p_counts + 1, cs);
            else
                copy_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
        }
        moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, cap, N, cs);
        if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
            lad_copy(g.n_layers - lb_ + 1, tb, te);   // M5g: the final residual, after the head's read source is set
            if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
            beacon(6);   // M5b: the last layer's gr_write/cvec are done
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    beacon(0);   // M5b: the layer loop itself is done
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        for (int t = 0; t < T; ++t) {
            copy_from_mapped(hand_out_ + (size_t) t * HB, Rt(t), HC * N, cs);
            copy_from_mapped(hand_out_ + (size_t) t * HB + HC * N, bo_ + (size_t) t * N, N, cs);
            copy_from_mapped(hand_out_ + (size_t) t * HB + HC * N + N, inj2_ + (size_t) t * HC, HC, cs);
        }
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        for (int t = 0; t < T; ++t) {
            BlockBuffers bb = ss.block;
            bb.R = Rt(t);
            bb.mixed = head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        beacon(1);   // M5b: the head's per-token read/projection is done
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                beacon(2);   // M5b: the head's activation quantization is done
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                beacon(3);   // M5b: the head's native matvec is done
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        beacon(4);   // M5b: the window's last stage (the argmax) is done
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_USE_SYCL)   // a CUDA debug listing (node types, kernel names)
    // M4: SYCL skips it for the same reason HIP does - there is nothing to introspect.  The SYCL shim's graph is
    // the M3 capture EMULATION (a list of closures, PLAN.md §1.3(b)/D7), so it has no node TYPE, no kernel
    // parameter block and no cudaFuncGetName; asking for them would mean inventing an answer.  The listing is a
    // diagnostic behind STRATA_VERIFY_NODES, not part of what the window graph does.
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
                    const char* fn = nullptr;
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
    }
#endif
    const cudaError_t ie = cudaGraphInstantiate(&exec_[T], graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_[T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              rope_scaling(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    // P3 (card t_d8afe53f): the submissions this stage's window costs.  Snapshotted here and reported after the
    // tail sync, so the figure is "what the host submitted to get THIS stage's window onto the GPU", split into
    // the shim's submission kinds (a kernel launch, a memset, a copy, a query barrier, an event, a host fn).
    static const bool sub_on = std::getenv("STRATA_SUBMIT_COUNT") != nullptr;
    const strata::sycl_compat::submit_stats sub0 = strata::sycl_compat::subs();
    auto sub_delta = [](const strata::sycl_compat::submit_stats& a,
                        const strata::sycl_compat::submit_stats& b) {
        strata::sycl_compat::submit_stats d;
        d.kernel = a.kernel - b.kernel;
        d.memset_ = a.memset_ - b.memset_;
        d.memcpy_ = a.memcpy_ - b.memcpy_;
        d.barrier = a.barrier - b.barrier;
        d.event = a.event - b.event;
        d.host_fn = a.host_fn - b.host_fn;
        d.graph_launch = a.graph_launch - b.graph_launch;
        d.recorded = a.recorded - b.recorded;
        return d;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    if (ss.ple.ready() && ple_stage()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // M5b: the flags the kernels read are DEVICE words, so the reset has to reach them - and it has to reach them
    // BEFORE this window's kernels start polling, because a window's rings restart at 1 and a left-over value
    // from the previous window would satisfy a wait early (the layer would then read a payload the host has not
    // served yet).  The copies go on the copy stream; sync it, since the launch below is on cs_.
    cudaMemcpyAsync(m_flag_, h_flag_, sizeof(uint32_t), cudaMemcpyHostToDevice, copy_);
    cudaMemcpyAsync(m_flagA_, h_flagA_, sizeof(uint32_t), cudaMemcpyHostToDevice, copy_);
    cudaMemcpyAsync(m_flagB_, h_flagB_, sizeof(uint32_t), cudaMemcpyHostToDevice, copy_);
    // M5b: the tail beacons count the stages the GPU reached, so they restart at 0 with the window
    for (int i = 0; i < 8; ++i) h_beacon_[i] = 0;
    cudaStreamSynchronize(copy_);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    // (the flag word is no longer written through a pointer here: publish_flag copies it into the device word
    //  the kernels read - M5b, see flag_mapped's note)
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    const int64_t steps = (le_ - lb_) * G;
    const bool test_stall = g_test_stall > 0 && windows + 1 == g_test_stall;   // #267 test hook (off: false)
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                // #267: the caller ends the engine; no spin kernel may outlive it
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        VDBG("layer %lld served\n", (long long) l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            publish_flag(h_flagA_, m_flagA_, want);
            raise_flag_dev(h_flagB_, m_flagB_, want);
        }
        if (!(test_stall && k + 1 == steps)) publish_flag(h_flag_, m_flag_, want);
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267's TAIL COUNTERPART (M5b).  The layer loop above has its own bounded wait, but a window whose TAIL (the
    // last layer's combine and the head) never finishes used to sit in this blocking sync until the driver reset
    // the context - measured on the W4A16 pack as `strata/sycl: stream sync failed: level_zero backend failed
    // with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` with nothing said about where the GPU was.  With
    // STRATA_VERIFY_TAIL_DEBUG=1 the wait is a bounded poll that dumps the diag line (the same one the layer
    // watchdog dumps) and releases the window's GPU waits instead of blocking forever.  The blocking sync stays
    // the default: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver beside the
    // expert workers).
    if (tail_debug_) {
        const Clock::time_point ts = Clock::now();
        while (cudaStreamQuery(cs_) == cudaErrorNotReady) {
            const long long waited = (long long) ms_since(ts);
            if (waited > tail_wait_ms_) {
                diag(stderr);
                const bool released = release_gpu_waits(5000);
                std::fprintf(stderr,
                             "strata verify: the window's tail did not finish within %lld ms (#267 tail); "
                             "its GPU waits were %s\n",
                             waited, released ? "released and the GPU finished" : "released but the GPU did not finish within 5 s");
                err = "verify: the window's tail did not finish (the last layer's combine or the head)";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ms_tail += ms_since(ts);
        progress_at("verify window: waiting for the expert copies", (int64_t) T);
        cudaStreamSynchronize(copy_);
        // M5c: NOT a bare `cudaStreamQuery`.  Under this backend's compat layer a query SUBMITS a fresh barrier
        // and reports not-ready until that barrier executes, so the single check this line used to do failed the
        // window every time the loop above had just exited cleanly - measured on the M5c sweep: every window that
        // DID finish on the GPU died here with "the tail reported an error"
        // (logs/m5c-cache{2000,6000,8000,9000}.log), which is a false negative, not a diagnosis.  Give the
        // barrier a bounded moment instead, and if it still has not completed say the tail did not finish: a
        // query that is not ready is not an error.
        {
            const Clock::time_point tq = Clock::now();
            cudaError_t q = cudaErrorNotReady;
            for (;;) {
                q = cudaStreamQuery(cs_);
                if (q == cudaSuccess || ms_since(tq) >= 5000) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (q != cudaSuccess) {
                err = "verify: the window's tail did not finish (the last layer's combine or the head)";
                return false;
            }
        }
    } else {
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
        progress_at("verify window: waiting for the expert copies", (int64_t) T);
        cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
    }
    if (sub_on) {
        const strata::sycl_compat::submit_stats d = sub_delta(strata::sycl_compat::subs(), sub0);
        const int64_t nlayer = (le_ >= lb_ + g.n_layers) ? g.n_layers : (le_ - lb_);
        std::fprintf(stderr,
                     "strata submit: %s window T=%d pos0=%lld layers %lld..%lld (%lld): submitted %llu "
                     "(kernel %llu memset %llu memcpy %llu barrier %llu event %llu hostfn %llu graph %llu) "
                     "+ recorded %llu; %.1f per layer; GPU-reach wait %.2f ms\n",
                     (next_ != nullptr || lb_ != 0) ? "stage" : "single", T, (long long) pos0, (long long) lb_,
                     (long long) (le_ - 1), (long long) nlayer, d.submitted(), d.kernel, d.memset_, d.memcpy_,
                     d.barrier, d.event, d.host_fn, d.graph_launch, d.recorded,
                     nlayer > 0 ? (double) d.submitted() / (double) nlayer : 0.0, ms_wait);
    }
    if (prof_on_ && G == 1) {       // the window's GPU stage stamps
        cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
        const int64_t L = g.n_layers;
        auto at = [&](int64_t l, int i) { return prof_h_[(size_t) (l * kProfPer + i)]; };
        // D8: the derived columns below read stamps the hc-read kernels write themselves or the next
        // layer's first.  A slot no kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns -
        // which is why (gap)/head/hc0 printed ~1e14 ms per window.  A missing or out-of-order stamp now
        // contributes nothing.
        const auto gap = [](unsigned long long to, unsigned long long from) {
            return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
        };
        for (int64_t l = 0; l < L; ++l) {
            const int kind = is_qsa_layer(g, l) ? 1 : 0;
            unsigned long long prev = at(l, 0);
            for (int i = 1; i <= 24; ++i) {
                const unsigned long long x = at(l, i);
                if (x == 0 || x < prev) continue;
                prof_sum_[kind][i] += (double) (x - prev);
                prev = x;
            }
            if (l + 1 < L) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
            const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
            if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
                prof_sum_[kind][27] += dn;      // hc-read0: norm
                prof_sum_[kind][28] += dd;      //           down
                prof_sum_[kind][29] += du;      //           up (through both halves, as before)
                prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
            }
        }
        prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
        ++prof_windows_;
    }
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    // M5f: THE WINDOW'S OWN BUFFERS, WHICH NO OTHER DUMP ON THIS PATH CAN REACH.  `--dump-mixed` and
    // `--dump-residual` read `ss.block.mixed` / `ss.R` - the NON-native session's scratch, which a native pack
    // never fills, so both write all zeros (measured, ~/strata-xpu/M5F-STATUS.md §2).  `--dump-layers` (the C1
    // ladder) is written by `session_loop`, which a native pack never enters, and `--dump-logits` covers only
    // *generated* positions.  This writes the first window's head input, final residual and head logits, which
    // is what a head-vs-layers bisection needs: with the pack's own `output_hc_*`/`output.weight` (bit-exact
    // against the checkpoint) the head can be recomputed from the residual in numpy and compared with the
    // engine's own logits, so "the text is odd" becomes "the head is right and the layers are wrong" (or not).
    //
    //   int32 hdr[4] = { T, n_embd, hc, pos0 }
    //   T * n_embd            floats  the head's input, `head_mixed_ + t*n_embd`
    //   T * hc * n_embd       floats  the final residual, `R_ + t*hc*n_embd`
    //   T * n_vocab           floats  the head's logits, `head_logits_ + t*n_vocab`
    // M5g: THE PER-LAYER LADDER.  `STRATA_DUMP_LADDER=<path>` writes, for the FIRST window, the residual as it
    // entered every layer: entry 0 is the R the window started from (the embedding broadcast, or a split stage's
    // input), entry k (k >= 1) is the R ENTERING stage-local layer k-1 - i.e. the output of layer k-2 - and the
    // last entry is the final R.  `lad_copy` therefore fills slot `l - lb_ + 1`, and there are n_layers - lb_ + 2
    // entries, all of them written.  The `.bo` companion is indexed DIFFERENTLY: slot k is the attention-half
    // output of stage-local layer k (`lad_copy_bo(l - lb_)`), and it carries only the n_layers - lb_ rows that
    // exist - no padding.  Reading the two files against each other's convention is the defect that once made
    // the zero-initialised tail look like "layer 47 writes a bit-zero attention output": the tail is not data.
    // The values were staged by copy nodes inside the window's own graph (see `lad_copy`), so this is the
    // window's arithmetic, not a re-run.
    //   int32 hdr[5] = { n_entries, hc, n_embd, pos0, T }, then n_entries * T * hc * n_embd floats
    if (const char* lp = std::getenv("STRATA_DUMP_LADDER");
        lp != nullptr && lad_ != nullptr && windows == 0) {
        const int64_t n_entries = g.n_layers - lb_ + 2;
        std::FILE* f = std::fopen(lp, "wb");
        if (f == nullptr) { err = std::string("STRATA_DUMP_LADDER: cannot write ") + lp; return false; }
        const int32_t hdr[5] = {(int32_t) n_entries, (int32_t) g.hc, (int32_t) g.n_embd, (int32_t) pos0, (int32_t) T};
        const size_t per = (size_t) T * (size_t) (g.hc * g.n_embd);
        const size_t stride = (size_t) max_t_ * (size_t) (g.hc * g.n_embd);   // entries are strided by max_t_ columns
        std::vector<float> host((size_t) n_entries * per);
        bool ok = true;
        for (int64_t k = 0; ok && k < n_entries; ++k)
            ok = cudaMemcpy(host.data() + (size_t) k * per, lad_ + (size_t) k * stride, per * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
        ok = ok && std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
             std::fwrite(host.data(), sizeof(float), host.size(), f) == host.size();
        std::fclose(f);
        if (!ok) { err = "STRATA_DUMP_LADDER: a write failed"; return false; }
        std::fprintf(stderr, "strata dbg: ladder -> %s (%lld entries x T %d x hc %lld x n_embd %lld, pos0 %lld)\n",
                     lp, (long long) n_entries, T, (long long) g.hc, (long long) g.n_embd, (long long) pos0);
        // the attention half's output, same row layout, one row per layer - and only the rows that were filled:
        // slot k is layer (lb_ + k), there are n_layers - lb_ of them, and the buffer's zero-initialised tail is
        // NOT written out (it used to be, and read as a layer with a bit-zero attention output).
        const int64_t nb = g.n_layers - lb_;
        std::string bop = std::string(lp) + ".bo";
        if (std::FILE* fb = std::fopen(bop.c_str(), "wb")) {
            const size_t pb = (size_t) T * (size_t) g.n_embd;
            const size_t sb = (size_t) max_t_ * (size_t) g.n_embd;
            std::vector<float> hb((size_t) nb * pb);
            bool okb = true;
            for (int64_t k = 0; okb && k < nb; ++k)
                okb = cudaMemcpy(hb.data() + (size_t) k * pb, ladb_ + (size_t) k * sb, pb * sizeof(float),
                                 cudaMemcpyDeviceToHost) == cudaSuccess;
            const int32_t bhdr[5] = {(int32_t) nb, 1, (int32_t) g.n_embd, (int32_t) pos0, (int32_t) T};
            okb = okb && std::fwrite(bhdr, sizeof bhdr, 1, fb) == 1 &&
                  std::fwrite(hb.data(), sizeof(float), hb.size(), fb) == hb.size();
            std::fclose(fb);
            if (!okb) { err = "STRATA_DUMP_LADDER: the attention-output ladder write failed"; return false; }
            std::fprintf(stderr, "strata dbg: bo ladder -> %s (%lld rows, slot k == layer %lld + k, no padding)\n",
                         bop.c_str(), (long long) nb, (long long) lb_);
        }
    }
    if (const char* ws = std::getenv("STRATA_DUMP_WINDOW_STATE");
        ws != nullptr && windows == 0 && next_ == nullptr) {
        std::FILE* f = std::fopen(ws, "wb");
        if (f == nullptr) { err = std::string("STRATA_DUMP_WINDOW_STATE: cannot write ") + ws; return false; }
        const int32_t hdr[4] = {(int32_t) T, (int32_t) g.n_embd, (int32_t) g.hc, (int32_t) pos0};
        const size_t n_mixed = (size_t) T * (size_t) g.n_embd;
        const size_t n_res = n_mixed * (size_t) g.hc;
        const size_t n_log = (size_t) T * (size_t) n_vocab_;
        std::vector<float> mixed(n_mixed), residual(n_res), logits(n_log);
        bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                  cudaMemcpy(mixed.data(), head_mixed_, n_mixed * sizeof(float), cudaMemcpyDeviceToHost) ==
                      cudaSuccess &&
                  std::fwrite(mixed.data(), sizeof(float), n_mixed, f) == n_mixed &&
                  cudaMemcpy(residual.data(), R_, n_res * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                  std::fwrite(residual.data(), sizeof(float), n_res, f) == n_res &&
                  cudaMemcpy(logits.data(), head_logits_, n_log * sizeof(float), cudaMemcpyDeviceToHost) ==
                      cudaSuccess &&
                  std::fwrite(logits.data(), sizeof(float), n_log, f) == n_log;
        std::fclose(f);
        if (!ok) { err = "STRATA_DUMP_WINDOW_STATE: a read or write failed"; return false; }
        double s2 = 0.0, mag = 0.0;
        int64_t finite = 0, nonfinite = 0;
        double mx = 0.0;
        for (float v : mixed) {
            if (!std::isfinite(v)) { ++nonfinite; continue; }
            ++finite;
            s2 += (double) v * (double) v;
            mag += std::fabs((double) v);
            mx = std::max(mx, std::fabs((double) v));
        }
        int64_t r_nan = 0;
        for (float v : residual) r_nan += !std::isfinite(v);
        int64_t l_nan = 0;
        for (float v : logits) l_nan += !std::isfinite(v);
        std::fprintf(stderr,
                     "strata dbg: window state -> %s (T %d, n_embd %lld, hc %lld, pos0 %lld): head input rms %.6g, "
                     "mean|.| %.6g, max|.| %.6g, non-finite %lld of %lld; residual non-finite %lld of %lld; "
                     "logits non-finite %lld of %lld\n",
                     ws, T, (long long) g.n_embd, (long long) g.hc, (long long) pos0,
                     finite > 0 ? std::sqrt(s2 / (double) finite) : 0.0, finite > 0 ? mag / (double) finite : 0.0, mx,
                     (long long) nonfinite, (long long) mixed.size(), (long long) r_nan,
                     (long long) residual.size(), (long long) l_nan, (long long) logits.size());
    }
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; flag B - raised once the staged blobs have landed - is what the graph waits for before the
// PCIe groups run.
//
// M5b CHANGED THE RAISE, NOT THE ORDER.  It used to be `cudaLaunchHostFunc(copy_, raise_flag, …)`, which is what
// M4's note above describes; the flag is a DEVICE word now (flag_mapped's note: a mapped store is never observed
// by a poll loop on this backend), so the raise is a 4-byte copy into it - and because `copy_` is an IN-ORDER
// queue, enqueueing that copy here, right after the blobs, gives exactly the guarantee the host function was
// there for.  It also removes a recorded host function from the window, which the closure-based capture cannot
// replay anyway (cudaLaunchHostFunc submits to the queue instead of appending a node).
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) {
        v->raise_flag_dev(v->h_flagB_, v->m_flagB_, want);
        return;
    }
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    v->raise_flag_dev(v->h_flagB_, v->m_flagB_, want);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    v->publish_flag(v->h_flagA_, v->m_flagA_, v->cur_layer_ + 1);
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (cudaMemcpy(h.data(), head_logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        // STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a
        // top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md)
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::commit(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    // set_commit_async: no wait here - the next window runs on the same stream after it, and the drafter (its own
    // stream) reads only this window's final rows and its own K/V. h_commit_ is next written after the next window's
    // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit).
    if (!g_commit_async || next_ != nullptr) {
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    } else {
        const cudaError_t re = cudaEventRecord(commit_done_, cs_);
        if (re != cudaSuccess) { err = std::string("verify: commit event: ") + cudaGetErrorString(re); return false; }
        commit_pending_ = true;
    }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err);
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    return next_ == nullptr || next_->wait_commit(err);
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace strata::core
