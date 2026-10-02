// tests/sycl/handoff.cpp - the host<->device doorbell ring on the SYCL backend (PLAN.md Risk 8, M2).
//
// WHAT CHANGED, AND WHY (Strata XPU M5b, card t_3ebd6083).  This test was M2 Risk 8's "NOT SETTLED - measured
// negative": it printed its first line and then hung past the ctest timeout (300 s).  Two independent defects
// were behind that, and both are fixed here and in the engine:
//
// 1. THE TEST'S OWN DEADLOCK (fixed, and it is what the hang was).  Every launch in the round named the DEFAULT
//    stream (`nullptr`), and `sync_if_needed` (src/kernels/cuda/elementwise.cu:113) turns a launch on the
//    default stream into `cudaDeviceSynchronize()`.  That global sync sits BETWEEN the launch of
//    `doorbell_wait` - a kernel that waits for a flag the host has not written yet - and the host's store, so
//    it blocks on the very kernel the flag is supposed to release.  gdb on the hung process (M5b session):
//      #8 strata::kernels::sync_if_needed(...)   #9 main
//      #7 sycl::_V1::queue::wait_and_throw_proxy #6 queue_impl::wait #5 urQueueFinish #3
//      ur_queue_immediate_in_order_t::queueFinish ... libze_intel_gpu
//    The HIP original (tests/hip/handoff.cpp:20) creates a stream and launches on it, which is why the same
//    round works there.  This test now does the same, and the bounded wait records its event on that stream
//    too (an event on a stream with no work completes at once, which is not a bound at all).
//
// 2. THE HOST->GPU FLAG HAD NO WORKING SPELLING ON THIS BACKEND (fixed in the engine's mechanism).  With the
//    deadlock gone, the round failed honestly: rounds 1-3 passed and round 4 reported "the device did NOT
//    observe the host's flag within 500 ms".  Measured on card 0 (probes/probe_m5b_*.cpp,
//    plan-evidence/m5b-*.log): a poll loop over mapped HOST memory does NOT see the host's stores (device cache,
//    no snooping); `_mm_clflush`/`_mm_sfence`/non-temporal stores change nothing (so the device owns the stale
//    copy); a SYSTEM-scope atomic load of mapped host memory is unreliable too (a sparse subset of the values).
//    What works, reproducibly: the flag word in DEVICE memory, the host publishing it with a 4-byte H2D COPY on
//    a second stream, and the waiting kernel reading it with a SYSTEM-scope atomic load
//    (include/strata/kernels/flag.hpp: `strata_flag_read`).  That is the mechanism this test now exercises:
//    the flag and its ring live in device memory and the host publishes with `cudaMemcpyAsync`, exactly as the
//    engine's verify window does after the M5b fix.
//
// The reverse direction (device -> host: the GPU rings and the host must see it WITHOUT a driver call) is
// unchanged: on gfx1201 the HIP build needed a VOLATILE store for it (0 of 100 rings seen without a sync;
// volatile 100 of 100), and on this backend a volatile store gives 10 of 10 rings at 3-50 us.  The ring is
// therefore still MAPPED memory, which is also what the engine's `doorbell_ring` / `doorbell_publish` use.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

constexpr int kRounds = 100;
constexpr int64_t kN = 4096;   // enough elements for copy_from_mapped's float4 path

int failures = 0;

void check(const char* what, bool ok) {
    std::printf("  %-52s %s\n", what, ok ? "OK" : "FAILED");
    if (!ok) ++failures;
}

/// Wait for `s` to drain, but not forever: a device-side spin that never sees the host's publication would
/// otherwise hang the whole test (it did: 300 s).  Returns false if the budget expired.  The event is recorded
/// on the SAME stream the round's work is on - on any other stream it would complete at once.
bool wait_bounded(cudaEvent_t done, double budget_ms, cudaStream_t s) {
    cudaEventRecord(done, s);
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() < budget_ms) {
        if (cudaEventQuery(done) != cudaErrorNotReady) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return false;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: a hang must still show how far it got
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::printf("no SYCL device\n");
        return 77;   // ctest "skipped"
    }
    const size_t bytes = size_t(kN) * sizeof(float);
    float* host = nullptr;          // the mapped payload the host fills
    float* mapped = nullptr;        // its device view (the identity, D6)
    uint32_t* hstage = nullptr;     // the host's staging word: the value to publish
    uint32_t* dflag = nullptr;      // the FLAG word, DEVICE memory
    uint32_t* dseq = nullptr;       // the ring word the host polls, MAPPED memory (device -> host)
    float* device = nullptr;        // the round's device-side payload
    cudaStream_t stream = nullptr;  // the round's work
    cudaStream_t pub = nullptr;     // the publication stream (the engine's copy stream)

    if (cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped) != cudaSuccess) return 1;
    if (cudaHostAlloc(reinterpret_cast<void**>(&hstage), sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess) return 1;
    if (cudaHostAlloc(reinterpret_cast<void**>(&dseq), sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess) return 1;
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), host, 0) != cudaSuccess) return 1;
    if (cudaMalloc(reinterpret_cast<void**>(&dflag), sizeof(uint32_t)) != cudaSuccess) return 1;
    if (cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess) return 1;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    if (cudaStreamCreate(&pub) != cudaSuccess) return 1;

    std::printf("alias: host %p device %p -> %s; the flag word is DEVICE memory %p, the ring %p is mapped\n",
                static_cast<void*>(host), static_cast<void*>(mapped),
                mapped == host ? "the host pointer itself (USM identity, D6)" : "a distinct device address",
                static_cast<void*>(dflag), static_cast<void*>(dseq));
    uint32_t zero = 0;
    cudaMemcpy(dflag, &zero, sizeof(zero), cudaMemcpyHostToDevice);   // zero the flag before the rounds

    // ---- 1. host -> device: the kernel must see a flag the HOST publishes LATE --------------------------
    // The producer is deliberately late (50..210 us after the launch): a device-side poll that is served from
    // a stale cache line would either spin forever or pass the stale value, and the payload check below is
    // what catches the second case - the payload is written AFTER the flag is staged, so a missing publication
    // shows up as a stale payload, not only as a timeout.  The flag is published into DEVICE memory with a
    // 4-byte copy: that, plus the system-scope atomic read in the kernel, is the measured-working handshake.
    std::vector<float> got(size_t(kN), -1.0f);
    std::vector<float> want(size_t(kN), 0.0f);
    bool payload_ok = true;
    int rounds_completed = 0;
    bool device_saw_the_flag = true;
    double worst_publish_us = 0.0;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 1; r <= kRounds; ++r) {
        strata::kernels::wait_flag_ge(dflag, (uint32_t) r, stream);            // the layer's spin kernel
        strata::kernels::copy_from_mapped(device, mapped, kN, stream);         // then reads the mapped payload
        strata::kernels::scale_inplace(device, kN, 2.0f, stream);
        std::this_thread::sleep_for(std::chrono::microseconds(50 + (r % 17) * 10));
        for (int64_t i = 0; i < kN; ++i) {
            host[i] = float(r * 10000 + i);
            want[size_t(i)] = host[i] * 2.0f;
        }
        // The publication: stage the ring value, then have the host's copy stream write it into device memory.
        *hstage = (uint32_t) r;
        const auto p0 = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(dflag, hstage, sizeof(uint32_t), cudaMemcpyHostToDevice, pub) != cudaSuccess) {
            std::printf("  round %d: the flag publication failed\n", r);
            payload_ok = false;
            break;
        }
        worst_publish_us = std::max(worst_publish_us,
                                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0)
                                        .count());
        // The wait is bounded: if the device never observes the host's flag the queue stays busy forever, and
        // the measurement we want is exactly "it did not observe it", not a hung test.
        if (!wait_bounded(done, 500.0, stream)) {
            std::printf("  round %d: the device did NOT observe the host's flag within 500 ms\n", r);
            device_saw_the_flag = false;
            break;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::printf("  round %d: cudaStreamSynchronize failed\n", r);
            payload_ok = false;
            break;
        }
        if (cudaMemcpy(got.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            payload_ok = false;
            break;
        }
        ++rounds_completed;
        for (int64_t i = 0; i < kN; ++i) {
            if (got[size_t(i)] != want[size_t(i)]) {
                std::printf("  stale payload r=%d i=%lld got=%f want=%f\n", r, (long long) i, (double) got[size_t(i)],
                            (double) want[size_t(i)]);
                payload_ok = false;
                break;
            }
        }
        if (!payload_ok) break;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double per_round_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kRounds;
    check("host -> device flag (device memory + copy) + mapped payload, 100 late rounds",
          payload_ok && device_saw_the_flag && rounds_completed == kRounds);
    std::printf("  rounds completed: %d of %d; device saw the host's flag: %s; worst publication %.1f us\n",
                rounds_completed, kRounds, device_saw_the_flag ? "yes" : "NO", worst_publish_us);
    std::printf("  %.1f us per round (one device wait + copy + scale, producer late by 50-210 us)\n", per_round_us);
    cudaEventDestroy(done);

    // ---- 2. device -> host: the ring must be visible to the HOST WITHOUT ANY DRIVER CALL -------------------
    // This is the direction the HIP build needed a volatile store for, and the engine's `doorbell_ring` /
    // `doorbell_publish` use mapped memory for it.  The host polls the mapped word in a plain load loop; a
    // driver call (cudaStreamQuery/cudaDeviceSynchronize) here would flush the pipeline and hide a missing
    // fence, so none is made until the poll gives up.
    int seen_without_sync = 0;
    int seen_total = 0;
    for (int r = 0; r < kRounds; ++r) {
        const uint32_t before = *dseq;
        strata::kernels::doorbell_ring(dseq, stream);
        bool seen = false;
        const auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() < 2000.0) {
            if (*dseq != before) {
                seen = true;
                break;
            }
        }
        if (seen) ++seen_without_sync;
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        if (*dseq != before) ++seen_total;
    }
    check("device ring seen by the host with NO driver call, 100 rounds", seen_without_sync == kRounds);
    std::printf("  rings seen without a driver call: %d of %d; seen after a sync: %d of %d\n", seen_without_sync,
                kRounds, seen_total, kRounds);

    cudaStreamDestroy(pub);
    cudaStreamDestroy(stream);
    cudaFree(device);
    cudaFree(dflag);
    cudaFreeHost(host);
    cudaFreeHost(hstage);
    cudaFreeHost(dseq);
    std::printf("%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
