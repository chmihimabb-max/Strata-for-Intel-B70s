# tools/sycl/handport/verify_kernels.py - the hand-port replacements for src/kernels/cuda/verify_kernels.cu.
#
# PLAN.md §2.3 site #3: `mov.u64 %0, %%globaltimer` - the verify window's stage profiler timestamp.  The
# replacement is the one PLAN.md names: SPIR-V's ReadClockKHR through
# sycl::ext::oneapi::experimental::clock<clock_scope::device>() (header
# sycl/ext/oneapi/experimental/clock.hpp; the AMD arm of the same function uses wall_clock64()).
#
# Everything else in this file is the same mechanical translation the other files get.  It has 25 launch sites
# and no tensor-core path, so the STRATA_SYCL_XMX gate is not consulted here.
ASM_SITES = 1

REPLACEMENTS = [
    # 1. the clock header
    (
        r"""#include <cuda_runtime.h>

#include <cstdio>
""",
        r"""#include <cuda_runtime.h>
#include <sycl/ext/oneapi/experimental/clock.hpp>   // ReadClockKHR: the %globaltimer replacement (PLAN.md §2.3)

#include <cstdio>
""",
    ),
    # 2. the timestamp itself
    (
        r"""// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
} }""",
        r"""// a GPU timestamp (ns) into buf[i] - the verify window's stage profiler.
// PLAN.md §2.3 site #3, hand-ported: `mov.u64 %0, %%globaltimer` has no SYCL spelling, and its CUDA guard
// (`#else` of __HIPCC__, i.e. always true under SYCL) would have compiled PTX into the device image
// (PLAN.md Risk 7).  The replacement is the SPIR-V device clock the plan names: ReadClockKHR via the oneAPI
// clock extension.  NOTE, and it is a real difference: %globaltimer on sm_80+ is a 1 ns COARSE counter, while
// the Level-Zero device clock's resolution is the device's own (probe/clock_probe is NOT in this milestone's
// evidence) - the timestamps are therefore comparable with each other within a run but not with a CUDA build's.
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#elif defined(STRATA_USE_SYCL)
    t = (unsigned long long) sycl::ext::oneapi::experimental::clock<
            sycl::ext::oneapi::experimental::clock_scope::device>();
#endif
    buf[i] = t;
} }""",
    ),
]
