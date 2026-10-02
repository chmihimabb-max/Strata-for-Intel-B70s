// include/strata/kernels/flag.hpp - how a kernel reads a doorbell/ring flag word.
//
// WHY THIS EXISTS (Strata XPU M5b, card t_3ebd6083).  The verify window's flow control is "the host raises a
// per-layer flag; the layer's spin kernel (wait_flag_ge / wait_flag_ge_or / doorbell_wait) waits for it".  On
// the CUDA and HIP backends the flag lives in mapped pinned host memory and the kernel reads it with a plain
// VOLATILE load, which the hardware keeps out of the caches for a mapped host access.
//
// On the SYCL/Level Zero backend that spelling does NOT work, and it is not a porting slip - it was measured on
// card 0 (Arc Pro B70, driver `xe`) with probes/probe_m5b_observe.cpp, probe_m5b_flush.cpp,
// probe_m5b_fix.cpp, probe_m5b_publish.cpp (plan-evidence/m5b-*.log):
//
//   * a volatile poll loop over mapped HOST memory does not observe the host's stores: with the host storing
//     1..10 (25 ms apart) the device's log showed ONE transition, straight to the last value, ~300-400 ms
//     later; with a plain host store the ring reached the CPU in 3-50 us, so it is only the host->device
//     direction that fails;
//   * `_mm_clflush` + `_mm_sfence` after every host store, and a non-temporal (`_mm_stream_si32`) store,
//     changed nothing -> the stale copy is in the DEVICE's cache, which does not snoop host stores to this
//     mapping; the host's own caches are not the problem;
//   * a SYSTEM-scope atomic load of mapped HOST memory is also unreliable (generic, global and seq_cst
//     spellings each observed only a sparse subset - 3 or 4 of 10 values - and cost ~900 ns per load);
//   * what DOES work, reproducibly (10 of 10 values, three separate runs, plus 36 rounds of the
//     flag-then-payload shape in probe_m5b_payload.cpp): the flag word in DEVICE memory (`cudaMalloc`), the
//     host publishing it with a 4-byte H2D COPY (cudaMemcpyAsync, on the engine's copy stream - a submission
//     costs 12-100 us; a KERNEL launch as the publisher is not usable, it was starved by the spinning kernel
//     for ~3 s), and the wait reading that device word with a SYSTEM-scope atomic load.
//
// So the split is: device-memory flags are read through `strata_flag_read` below, and the host must publish
// into them with a copy rather than a plain store.  Payloads can stay in mapped host memory: probe_m5b_payload
// measured that a 16 KiB and a 1 MiB mapped buffer, rewritten by the host and read once per round by a kernel
// that waits for the flag, is read FRESH every round (12 of 12; a 16 MiB payload showed a difference only
// because the probe's own float checksum loses precision past 2^24).  The device caches the line a poll loop
// hammers, not a buffer it reads once.
//
// The CUDA/HIP arm is the original volatile load, so nothing changes for those backends.  The SYCL arm is the
// system-scope atomic load, under the SYCL guard the tree already uses (STRATA_USE_SYCL, set by
// cmake/sycl_backend.cmake) so the .cu sources stay valid CUDA.
#pragma once

#include <cstdint>

#if defined(STRATA_USE_SYCL)
#include <sycl/sycl.hpp>
#endif

/// Read a flag word the HOST publishes (device memory - see the note above).  The `volatile` in the signature
/// is the CUDA spelling's requirement and is kept so both arms take the same argument type.
__device__ __forceinline__ uint32_t strata_flag_read(const volatile uint32_t* p) {
#if defined(STRATA_USE_SYCL)
    // system_scope: the value came from the host (through the copy engine), so it has to be read in a way the
    // device's own cache does not answer from a stale line.  Measured: this spelling sees every value.
    return sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system>(
               *const_cast<uint32_t*>(p))
        .load(sycl::memory_order::acquire);
#else
    return *(const volatile uint32_t*) p;
#endif
}
