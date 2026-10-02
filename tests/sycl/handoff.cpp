// tests/sycl/handoff.cpp - the host<->device doorbell ring on the SYCL backend (PLAN.md Risk 8, M2).
//
// Port of tests/hip/handoff.cpp, minus its graph half: the HIP test wraps a whole round in a
// hipStreamBeginCapture/EndCapture pair, and stream capture has NO SYCL equivalent (PLAN.md §1.3(b), D7 -
// M1..M5 run with `--no-capture` and the interception layer is M6 work).  What is left is the part that
// decides whether the port works at all: the engine publishes a flag from the HOST and a kernel must see it,
// and a kernel publishes a ring the HOST must see without a driver call.  On gfx1201 the HIP build needed a
// VOLATILE store for the second direction (0 of 100 rings seen without a sync; volatile 100 of 100), so the
// same question is asked here with the same numbers.
//
// The protocol under test is the engine's own - strata::kernels::doorbell_wait / doorbell_ring /
// copy_from_mapped (src/kernels/cuda/elementwise.cu), reached through include/strata/kernels/elementwise.hpp.
// Both buffers are MAPPED PINNED memory: cudaHostAlloc(..., cudaHostAllocMapped) is sycl::malloc_host here
// and cudaHostGetDevicePointer is the identity, because a USM host pointer is device-usable (D6).
#include "strata/kernels/elementwise.hpp"

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

/// Wait for the queue to drain, but not forever: a device-side spin that never sees the host's publication
/// would otherwise hang the whole test (it did: 300 s).  Returns false if the budget expired.
bool wait_bounded(cudaEvent_t done, double budget_ms) {
    cudaEventRecord(done, nullptr);
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
    float* host = nullptr;
    float* mapped = nullptr;
    uint32_t* flag = nullptr;
    uint32_t* dflag = nullptr;
    uint32_t* seq = nullptr;
    uint32_t* dseq = nullptr;
    float* device = nullptr;

    if (cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped) != cudaSuccess) return 1;
    if (cudaHostAlloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess) return 1;
    if (cudaHostAlloc(reinterpret_cast<void**>(&seq), sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess) return 1;
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), host, 0) != cudaSuccess) return 1;
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&dflag), flag, 0) != cudaSuccess) return 1;
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&dseq), seq, 0) != cudaSuccess) return 1;
    if (cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess) return 1;

    std::printf("alias: host %p device %p -> %s\n", static_cast<void*>(host), static_cast<void*>(mapped),
                mapped == host ? "the host pointer itself (USM identity, D6)" : "a distinct device address");

    // ---- 1. host -> device: the kernel must see a flag the HOST writes, published LATE --------------------
    // The producer is deliberately late (50..210 us after the launch): a device-side poll that does not read
    // host-visible memory would either spin forever or pass the stale value, and the payload check below is
    // what catches the second case - the payload is written AFTER the flag in the HIP original, so a missing
    // fence shows up as a stale payload, not as a timeout.
    std::vector<float> got(size_t(kN), -1.0f);
    std::vector<float> want(size_t(kN), 0.0f);
    *flag = 0;
    *seq = 1;
    bool payload_ok = true;
    int rounds_completed = 0;
    bool device_saw_the_flag = true;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 1; r <= kRounds; ++r) {
        *seq = (uint32_t) r;
        strata::kernels::doorbell_wait(dflag, dseq, nullptr);              // device spins until *dflag == *dseq
        strata::kernels::copy_from_mapped(device, mapped, kN, nullptr);    // then reads the mapped payload
        strata::kernels::scale_inplace(device, kN, 2.0f, nullptr);
        std::this_thread::sleep_for(std::chrono::microseconds(50 + (r % 17) * 10));
        for (int64_t i = 0; i < kN; ++i) {
            host[i] = float(r * 10000 + i);
            want[size_t(i)] = host[i] * 2.0f;
        }
        __atomic_store_n(flag, (uint32_t) r, __ATOMIC_RELEASE);
        // The wait is bounded: if the device never observes the host's flag the queue stays busy forever, and
        // the measurement we want is exactly "it did not observe it", not a hung test.
        if (!wait_bounded(done, 500.0)) {
            std::printf("  round %d: the device did NOT observe the host's flag within 2000 ms\n", r);
            device_saw_the_flag = false;
            break;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("  round %d: cudaDeviceSynchronize failed\n", r);
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
    check("host -> device flag + mapped payload, 100 late rounds",
          payload_ok && device_saw_the_flag && rounds_completed == kRounds);
    std::printf("  rounds completed: %d of %d; device saw the host's flag: %s\n", rounds_completed, kRounds,
                device_saw_the_flag ? "yes" : "NO");
    std::printf("  %.1f us per round (one device wait + copy + scale, producer late by 50-210 us)\n", per_round_us);
    cudaEventDestroy(done);

    // ---- 2. device -> host: the ring must be visible to the HOST WITHOUT ANY DRIVER CALL -------------------
    // This is the direction the HIP build needed a volatile store for.  The host polls the mapped word in a
    // plain load loop; a driver call (cudaStreamQuery/cudaDeviceSynchronize) here would flush the pipeline and
    // hide a missing fence, so none is made until the poll gives up.
    int seen_without_sync = 0;
    int seen_total = 0;
    for (int r = 0; r < kRounds; ++r) {
        const uint32_t before = *seq;
        strata::kernels::doorbell_ring(dseq, nullptr);
        bool seen = false;
        const auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() < 2000.0) {
            if (*seq != before) {
                seen = true;
                break;
            }
        }
        if (seen) ++seen_without_sync;
        if (cudaDeviceSynchronize() != cudaSuccess) break;
        if (*seq != before) ++seen_total;
    }
    check("device ring seen by the host with NO driver call, 100 rounds", seen_without_sync == kRounds);
    std::printf("  rings seen without a driver call: %d of %d; seen after a sync: %d of %d\n", seen_without_sync,
                kRounds, seen_total, kRounds);

    cudaFree(device);
    cudaFreeHost(host);
    cudaFreeHost(flag);
    cudaFreeHost(seq);
    std::printf("%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
