#pragma once
// include/strata/sycl_compat/cuda_runtime.h - the CUDA runtime surface, for a SYCL (Intel XPU) build.
//
// This file plays the same role for SYCL that include/strata/hip_compat/cuda_runtime.h plays for HIP: it is
// force-included into every host and device source, and `<cuda_runtime.h>` from the kernels resolves to it.
// Unlike HIP, it is NOT a rename: there is no C++ spelling for `kernel<<<grid, block, smem, stream>>>(...)`,
// so the launch syntax is rewritten ahead of compilation by tools/sycl/syclify.py into a call to
// `strata::sycl_compat::launch<Tag>(cfg(grid, block, smem), stream, [=](sycl::nd_item<3> item, uint8_t* dyn){ KERNEL(args); })`.
// That exact shape was compiled AND run on the B70 with oneAPI 2026.1.1 by probe/probe_sycl_launch.cpp before
// this backend depended on it.
//
// Design rules, each forced by a measurement or by the SYCL spec:
//   * The four CUDA builtin containers (threadIdx, blockIdx, blockDim, gridDim) are FUNCTIONS of the current
//     work item, read through sycl::ext::oneapi::this_work_item::get_nd_item<3>() - measured working from a
//     plain function (probe5, probe_sycl_launch).  That is what lets a kernel body keep its text.
//   * The launch helper always calls its kernel lambda as fn(item, dyn): one uniform two-argument shape,
//     dyn == nullptr when the launch asked for no dynamic shared memory.  Two submission shapes means two
//     kernel-name types (kernel_name_static / kernel_name_dynamic), because DPC++ rejects one kernel name
//     used by two definitions ("definition with same mangled name as another definition").
//   * The sticky error cell (PLAN.md D5) is thread-local and only a FAILING shim entry point ever writes it:
//     CUDA's cudaGetLastError is sticky and 150 call sites in the tree depend on that.
//   * Every queue is IN-ORDER (PLAN.md §1.1): the engine assumes per-stream ordering, and an out-of-order
//     queue would silently break every "copy then read on the host" site.
//   * `cudaMemcpy` is SYNCHRONOUS, as CUDA's is, and a copy with an ordinary malloc'd host pointer on either
//     side is staged through sycl::malloc_host, because USM copy is only defined between USM pointers.
//
// Deliberately absent (PLAN.md §1.3): stream capture and cudaHostRegister have no SYCL equivalent.  Both
// return a specific, documented error so the engine's own degrade paths run instead of a silent no-op.
#include <sycl/sycl.hpp>
// P3 (card t_d8afe53f): the SYCL graph extension is what makes a REAL graph-replay equivalent possible on this
// backend - oneAPI 2026.1 ships sycl::ext::oneapi::experimental::command_graph for the Level Zero backend, and
// the B70 reports sycl::aspect::ext_oneapi_graph.  It is used only behind STRATA_SYCL_GRAPH=1 (off by default);
// the header is included unconditionally because a runtime flag cannot change a type.
#include <sycl/ext/oneapi/experimental/graph.hpp>


// CUDA's builtin vector types (float2/float4/int2/uint4/char4/... and the make_* constructors) live in
// their own header: cuda_fp16.h needs them too (__half22float2 returns a float2), and the kernels include
// <cuda_fp16.h> themselves (M2).  CUDA's <cuda_runtime.h> pulls <vector_types.h> in the same way.
#include "vector_types.h"

// <cuda_runtime.h> pulls CUDA's device math in; kernels in the tree call fmaf/expf/logf/sqrtf without
// including anything themselves (cvec.cu:66 is `fmaf` with only <stdexcept> beside it).  DPC++ ships
// sycl/stl_wrappers/cmath exactly so <cmath> stays usable from device code.
#include <cmath>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <functional>   // stream capture records each submission as a std::function<void()> (PLAN.md §1.3(b))

// ===========================================================================
// Device-visible surface.
// ===========================================================================
namespace strata::sycl_compat {

/// CUDA's dim3: an unsigned triple, constructible from one integer so `<<<blocks, threads>>>` keeps working.
struct dim3 {
    unsigned x = 1, y = 1, z = 1;
    dim3() = default;
    dim3(unsigned a) : x(a), y(1), z(1) {}                       // NOLINT(google-explicit-constructor)
    dim3(int a) : x((unsigned) a), y(1), z(1) {}                 // NOLINT
    dim3(unsigned long a) : x((unsigned) a), y(1), z(1) {}       // NOLINT
    dim3(unsigned long long a) : x((unsigned) a), y(1), z(1) {}  // NOLINT
    dim3(unsigned a, unsigned b) : x(a), y(b), z(1) {}
    dim3(unsigned a, unsigned b, unsigned c) : x(a), y(b), z(c) {}
};

/// The work item of the kernel this thread is inside.  Reachable from any function a kernel calls, which is
/// the property the whole port rests on.
///
/// A 1-D nd_range, deliberately.  M2 measured that with a MULTI-DIMENSIONAL nd_range the physical
/// sub-group's lanes do NOT correspond to the CUDA warp an engine kernel assumes: for a `{32, 4}` local range
/// `get_sub_group().get_local_linear_id()` returns `(y + 4x) mod 32`, i.e. repeated and non-monotonic values
/// (probe/probe16_shfl_2d.cpp), so every shuffle silently exchanged with the wrong work-item - fwht256 (2 of
/// its butterflies are shuffles) came out 248 of 256 elements wrong (probe/probe15_fwht_stages.cpp) and
/// mmvq/iq multi-token kernels produced non-finite output.  With a 1-D nd_range the sub-group IS 32
/// consecutive work-items, which is exactly CUDA's warp over the linearized thread index, and the CUDA
/// coordinates are reconstructed from `launch_shape` (see thread_idx below).
inline auto this_item() { return sycl::ext::oneapi::this_work_item::get_nd_item<1>(); }

/// The block/grid extents of the launch this work item belongs to.  The transform passes it as the second
/// kernel parameter (after the dynamic shared-memory base), evaluated at the HOST call site, and the
/// coordinate accessors below divide the 1-D local/group index by it.
struct launch_shape {
    unsigned gx = 1, gy = 1, gz = 1;
    unsigned bx = 1, by = 1, bz = 1;
};

inline launch_shape thread_shape(const launch_shape& s) { return s; }

/// threadIdx / blockIdx / blockDim / gridDim, from the 1-D work-item index and the launch's extents.
inline dim3 thread_idx(const launch_shape& s) {
    const unsigned lid = (unsigned) this_item().get_local_id(0);
    const unsigned bx = s.bx ? s.bx : 1u, by = s.by ? s.by : 1u;
    return dim3(lid % bx, (lid / bx) % by, lid / (bx * by));
}
inline dim3 block_idx(const launch_shape& s) {
    const unsigned gid = (unsigned) this_item().get_group(0);
    const unsigned gx = s.gx ? s.gx : 1u, gy = s.gy ? s.gy : 1u;
    return dim3(gid % gx, (gid / gx) % gy, gid / (gx * gy));
}
inline dim3 block_dim(const launch_shape& s) { return dim3(s.bx, s.by, s.bz); }
inline dim3 grid_dim(const launch_shape& s) { return dim3(s.gx, s.gy, s.gz); }

/// The linear local id: with a 1-D nd_range this is both the CUDA linearized thread index and the position
/// inside the sub-group, which is what makes the shuffles correct.
inline unsigned local_linear_id() { return (unsigned) this_item().get_local_id(0); }
inline unsigned group_linear_id() { return (unsigned) this_item().get_group(0); }

/// The physical lane inside the sub-group.  The B70 reports sub_group_sizes {16, 32} (PLAN.md §10), NOT
/// {8,16,32}: a kernel that needs a 32-wide logical warp has to say so - that is Risk 6, settled in M2.
inline unsigned lane_id() { return (unsigned) this_item().get_sub_group().get_local_linear_id(); }

/// CUDA's __syncthreads is a work-group barrier; __syncwarp is a sub-group one.
inline void syncthreads() { sycl::group_barrier(this_item().get_group()); }
inline void syncwarp() { sycl::group_barrier(this_item().get_sub_group()); }

struct launch_config {
    dim3 grid;
    dim3 block;
    unsigned long long smem = 0;
};

/// `cfg(grid, block, smem)` - emitted by syclify.py with the expressions it found inside `<<<>>>`.
template <class G, class B>
inline launch_config cfg(const G& g, const B& b, unsigned long long smem = 0) {
    return launch_config{dim3(g), dim3(b), smem};
}

/// A launch site's name, kept for the reader and for the M1 core TUs that already emit one.
///
/// **M2 measured that the launcher must NOT use it as the `parallel_for` kernel name.**  A launch inside a
/// function TEMPLATE (native_mmvq.cu:1052's `native_mmvq_multi_kernel<F, NCOLS, NW, 2>`, iq_kernels.cu's
/// `launch_mmvq<TY>`) is instantiated several times with different lambdas but the SAME (file, line), and a
/// launch inside a `#define` body (dequant_bf16.cu:206's `STRATA_DQ`) is expanded several times at ONE line:
/// naming the kernel after the source position therefore produces several different kernels under one name
/// ("definition with same mangled name as another definition").  `probe/probe12_unnamed_kernel.cpp` measured
/// that this DPC++ (2026.1) with `-fsycl-device-code-split=per_kernel` accepts and RUNS unnamed kernel
/// lambdas, including two instantiations of one template, so `launch()` below uses an unnamed kernel and the
/// whole collision class disappears.
template <unsigned long long File, unsigned Line, unsigned Ord = 0>
struct kernel_name {};

namespace detail {
template <class Tag>
struct kernel_name_static : Tag {};
template <class Tag>
struct kernel_name_dynamic : Tag {};
}  // namespace detail

// NOT guarded by `#ifndef __SYCL_DEVICE_ONLY__`, and that is deliberate.  The guard is the usual SYCL idiom
// for host-only code, but it makes the host API INVISIBLE during the device pass of the SAME translation
// unit - and DPC++ parses the whole TU in that pass, so the host half of a `__global__` wrapper (which calls
// cudaMalloc, not any kernel) fails with "unknown type name 'cudaError_t'".  Measured: that was the first
// error a real generated TU produced.  Only REACHABLE functions are device-compiled, so leaving the
// declarations visible costs nothing and is what lets one source file carry both halves.
// ---------------------------------------------------------------------------
// Host side: device table, queues, the sticky error cell, the launcher.
// ---------------------------------------------------------------------------

/// All host state behind one accessor, so the device pass compiles none of it.
struct backend_state {
    std::vector<sycl::device> devices;
    /// ONE context for every GPU, which is how this shim maps "one process, one USM namespace" (a queue built
    /// for device d with a shared context still allocates on device d - probe33_sharedctx measured 4 GiB on q0
    /// moving dev0's free memory and 4 GiB on q1 moving dev1's).  The layer split ran and returned the same
    /// tokens with ONE CONTEXT PER DEVICE as well (STRATA_SYCL_CONTEXT_PER_DEVICE=1, the A/B control), so this is
    /// hardening rather than the fix: a device pointer of one GPU is `usm::alloc::unknown` from another context
    /// (`is_usm_pointer`), and `cudaStreamWaitEvent`'s `ext_oneapi_submit_barrier({event})` throws for an event of
    /// another context.  What it does NOT buy: a copy between the two cards' memory still fails with
    /// UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY (probe33, no peer path on this pair).  nullptr when the driver
    /// refused the multi-device context, and `ctx_per_dev` stands in.
    sycl::context* ctx = nullptr;
    std::vector<sycl::context*> ctx_per_dev;
    /// ONE DEFAULT (CUDA legacy) STREAM PER DEVICE - `cudaSetDevice` has to move it, as it does on CUDA, where
    /// the legacy stream is per device.  Index = device.  (Measured on the two B70s, probe32_defaultq: with a
    /// single default queue created while device 0 was current, an allocation inside `cudaSetDevice(1)` landed
    /// on CARD 0 - 4 GiB: dev0 27.21 -> 23.21 GiB free, dev1 unchanged at 31.85 - which is what made the
    /// split's stage-1 cache, session and weights pile onto card 0 and die of UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY.)
    std::vector<sycl::queue*> streams;
    /// Every queue this process has, for `sync_all` (cudaDeviceSynchronize): the default queues and the
    /// explicit streams.  It used to hold the default queue alone, so a stage's work on its own stream was
    /// never waited on by a device sync.
    std::vector<sycl::queue*> all_queues;
    int current = 0;
    int driver_version = 0;
    int runtime_version = 0;
    bool initialized = false;
    size_t device_bytes = 0;   // live cudaMalloc device bytes: cudaMemGetInfo's accounting (Risk 10)
};

inline backend_state& backend() {
    static backend_state* b = new backend_state();   // never destroyed: shim calls run inside destructors
    return *b;
}

inline void init_backend();
inline sycl::queue& default_queue();
inline sycl::queue* queue_for(void* stream);
inline sycl::context& ctx_for_device(size_t d);
inline void note_queue(sycl::queue* q);
inline void drop_queue(sycl::queue* q);

struct stream_t;
using cudaStream_t = stream_t*;                 // nullptr is CUDA's default (legacy) stream
using cudaError_t = int;

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorNotReady = 34;
constexpr cudaError_t cudaErrorNotSupported = 801;
constexpr cudaError_t cudaErrorStreamCaptureUnsupported = 900;
constexpr cudaError_t cudaErrorStreamCaptureImplicit = 901;      // nested capture (CUDA: 901)
constexpr cudaError_t cudaErrorStreamCaptureInvalidated = 902;   // EndCapture without BeginCapture (CUDA: 902)
constexpr cudaError_t cudaErrorUnknown = 999;

// The sticky error cell (D5).  Only a failing shim entry point writes it, so a later success cannot erase an
// earlier failure - what the tree's 150 cudaGetLastError sites rely on.
inline cudaError_t& last_error_cell() {
    static thread_local cudaError_t e = cudaSuccess;
    return e;
}
inline void record(cudaError_t e) {
    if (e != cudaSuccess) last_error_cell() = e;
}
inline cudaError_t cudaGetLastError() {
    const cudaError_t e = last_error_cell();
    last_error_cell() = cudaSuccess;
    return e;
}
inline cudaError_t cudaPeekAtLastError() { return last_error_cell(); }
inline const char* cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorNotReady: return "device not ready";
        case cudaErrorNotSupported: return "operation not supported by the SYCL backend";
        case cudaErrorStreamCaptureUnsupported:
            return "stream capture has no SYCL equivalent (run with --no-capture; PLAN.md D7)";
        default: return "unknown error";
    }
}

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
};
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1 };
constexpr unsigned cudaStreamNonBlocking = 1u;

struct cudaDeviceProp {
    char name[256] = {0};
    int major = 0;
    int minor = 0;
    int multiProcessorCount = 0;
    int warpSize = 32;
    size_t sharedMemPerBlock = 0;
    size_t totalGlobalMem = 0;
    int l2CacheSize = 0;   // M2: s2_expert_grouped_parity's bench sizes its blob cycle from this
    char gcnArchName[64] = {0};
};

struct cudaFuncAttributes {
    size_t sharedSizeBytes = 0;
    size_t maxDynamicSharedSizeBytes = 0;
    int numRegs = 0;
};

enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
enum cudaDeviceAttr {
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrClockRate = 13,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
};

/// "bmg-g31" for an Arc Pro B70 (0xe223).  SYCL exposes no arch string, so it is derived from the device
/// name; anything unrecognised reports the name itself rather than a guessed architecture.
inline std::string device_arch_string(const sycl::device& d) {
    const std::string n = d.get_info<sycl::info::device::name>();
    if (n.find("e223") != std::string::npos) return "bmg-g31";
    return n;
}

/// Only Level-Zero GPU devices are enumerated: this shim stands in for CUDA, whose device list is the GPUs,
/// and sycl-ls also sees an OpenCL CPU device.  ZE_AFFINITY_MASK is applied by the driver before SYCL sees
/// anything, so `ZE_AFFINITY_MASK=1` makes the second B70 device 0 here - that is how the M1 acceptance
/// selects the second card.
inline void init_backend() {
    backend_state& b = backend();
    if (b.initialized) return;
    for (const sycl::platform& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const sycl::device& d : p.get_devices()) {
            if (d.is_gpu()) b.devices.push_back(d);
        }
    }
    b.initialized = true;
    if (!b.devices.empty()) {
        // Informational only (src/core/device.cu prints them): oneAPI 2026.1.1, reported the way CUDA reports
        // a driver/runtime version pair.
        b.driver_version = 20260101;
        b.runtime_version = 20260101;
        // ONE CONTEXT OVER EVERY GPU (see backend_state::ctx).  Measured on the two B70s: probe33_sharedctx
        // creates a 2-device context, allocates on both devices through it and reads one device's memory from
        // the other device's queue, which is what a layer split needs.  STRATA_SYCL_CONTEXT_PER_DEVICE=1 is the
        // A/B control (the pre-split behaviour).
        const char* per_dev = std::getenv("STRATA_SYCL_CONTEXT_PER_DEVICE");
        if (!(per_dev != nullptr && per_dev[0] == '1')) {
            try {
                b.ctx = new sycl::context(b.devices);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata/sycl: one context over %zu devices was refused (%s); one context per "
                                     "device, so a buffer of one GPU is not visible to another\n", b.devices.size(),
                             e.what());
                b.ctx = nullptr;
            }
        }
    }
}

/// The context a queue on device `d` must be built in: the shared one when the driver took it (the multi-GPU
/// case), otherwise one per device.
inline sycl::context& ctx_for_device(size_t d) {
    init_backend();   // the device table MUST exist first: this is reached from default_queue(), which used to get
                      // that from selected_device() - and argument evaluation order is unspecified, so a run that
                      // happened to evaluate the context first read an empty table and died in a SIGSEGV
    backend_state& b = backend();
    if (b.ctx != nullptr) return *b.ctx;
    if (b.ctx_per_dev.size() != b.devices.size()) b.ctx_per_dev.assign(b.devices.size(), nullptr);
    if (b.ctx_per_dev.empty()) {   // no Level-Zero GPU at all: the selector's device stands in
        static sycl::context* fallback = new sycl::context(sycl::device{sycl::gpu_selector_v});
        return *fallback;
    }
    if (d >= b.ctx_per_dev.size()) d = 0;
    if (b.ctx_per_dev[d] == nullptr) b.ctx_per_dev[d] = new sycl::context(b.devices[d]);
    return *b.ctx_per_dev[d];
}

/// Every queue is registered here: `sync_all` (cudaDeviceSynchronize) has to wait for a stage's own stream too,
/// not only for the default one (which is all it waited for before).
inline void note_queue(sycl::queue* q) { backend().all_queues.push_back(q); }
inline void drop_queue(sycl::queue* q) {
    std::vector<sycl::queue*>& v = backend().all_queues;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == q) { v[i] = v.back(); v.pop_back(); return; }
}

inline bool profiling_enabled() {
    const char* v = std::getenv("STRATA_SYCL_PROFILING");
    return v == nullptr || std::string(v) != "0";
}
inline sycl::property_list queue_properties() {
    // In-order always.  enable_profiling because cudaEventElapsedTime cannot answer without it; the M6
    // measurement block sets STRATA_SYCL_PROFILING=0 to price that choice.
    return profiling_enabled() ? sycl::property_list{sycl::property::queue::in_order{},
                                                     sycl::property::queue::enable_profiling{}}
                              : sycl::property_list{sycl::property::queue::in_order{}};
}

inline sycl::device selected_device() {
    init_backend();
    backend_state& b = backend();
    return b.devices.empty() ? sycl::device{sycl::gpu_selector_v} : b.devices[(size_t) b.current];
}

inline sycl::queue& default_queue() {
    init_backend();   // b.current, b.devices and b.streams are all read below
    backend_state& b = backend();
    const size_t d = (size_t) (b.current < 0 ? 0 : b.current);
    if (b.streams.size() <= d) b.streams.resize(b.devices.empty() ? d + 1 : b.devices.size(), nullptr);
    // A SINGLE queue used to be created here, on whichever device happened to be current at the first call, and
    // returned for every later call whatever `cudaSetDevice` did: CUDA's legacy stream is PER DEVICE.  That is
    // the defect probe32_defaultq measured (4 GiB allocated inside cudaSetDevice(1) landed on card 0), and with
    // a layer split it piles both stages' weights, sessions and caches onto card 0.
    const char* single = std::getenv("STRATA_SYCL_SINGLE_DEFAULT_QUEUE");
    const size_t slot = (single != nullptr && single[0] == '1') ? 0 : d;   // A/B control: the old single queue
    if (b.streams.size() <= slot) b.streams.resize(slot + 1, nullptr);
    if (b.streams[slot] == nullptr) {
        // `selected_device()` is the CURRENT device, which is the one this queue stands for (the A/B control
        // keeps the old rule: the device current at the first call - its context is that device's).
        sycl::queue* q = new sycl::queue(ctx_for_device(d), selected_device(), queue_properties());
        b.streams[slot] = q;
        note_queue(q);
    }
    return *b.streams[slot];
}

inline sycl::queue* queue_for(void* stream) {
    if (stream == nullptr) return &default_queue();
    return static_cast<sycl::queue*>(stream);
}

inline void sync_all() {
    init_backend();
    for (sycl::queue* q : backend().all_queues) q->wait_and_throw();
}

// ===========================================================================
// `__constant__` tables (PLAN.md §1.3(c), M2).
//
// `__constant__` has no SYCL spelling.  Two cases exist in the tree and they need different emulations:
//
//   * a table with a STATIC initializer (`__constant__ int8_t kv_iq4nl[16] = {...}`, dequant_bf16.cu:33,
//     s_gemv.cu:32) becomes `constexpr`, which the transform emits.  MEASURED (probe/probe10_const.cpp, card
//     0): DPC++ device code reads a constexpr global and gets the right bytes.
//   * a table filled at RUNTIME by cudaMemcpyToSymbol (`__constant__ float c_codes[256][4]`,
//     s2_gemv_fast.cu:44 with :59) cannot be constexpr - the bytes are not known until the host copies them.
//     It becomes a device USM block owned by `const_storage` on the host, and the kernels that read it get a
//     by-value handle (`const_ref1d`/`const_ref2d`) carrying the ARRAY'S OWN NAME, appended to their
//     parameter list by the transform - which is why the kernel body needs no rewriting at all.
//
// A `sycl::device_global` was measured for this and rejected: in oneAPI 2026.1 the host-side `.get()` throws
// ("get() is not supported on host device"), so a host-filled table cannot go through it.
// ===========================================================================
template <class T>
struct const_storage {
    T* dev = nullptr;
    size_t count = 0;
    /// cudaMemcpyToSymbol's semantics: allocate on first use, then copy H2D and WAIT (CUDA's is synchronous
    /// for the default stream).
    cudaError_t upload(const void* src, size_t bytes, size_t offset = 0) {
        init_backend();
        if (dev == nullptr) {
            dev = sycl::malloc_device<T>(bytes / sizeof(T) + (offset / sizeof(T)) + 1, default_queue());
            if (dev == nullptr) return cudaErrorMemoryAllocation;
            count = bytes / sizeof(T);
        }
        default_queue().memcpy(reinterpret_cast<unsigned char*>(dev) + offset, src, bytes).wait();
        return cudaSuccess;
    }
};

/// 1-D view of a const_storage block, usable as `t[i]` in device code.
template <class T>
struct const_ref1d {
    const T* p = nullptr;
    const T& operator[](long long i) const { return p[i]; }
};
template <class T, long long S1>
struct const_row {
    const T* p = nullptr;
    const T& operator[](long long j) const { return p[j]; }
};
/// 2-D view, usable as `t[i][j]` in device code.
template <class T, long long S1>
struct const_ref2d {
    const T* p = nullptr;
    const_row<T, S1> operator[](long long i) const { return const_row<T, S1>{p + i * S1}; }
};

// ---- stream capture: the capture contract, implemented (PLAN.md §1.3(b), D7) ------------------------
// WHAT THIS IS, and what it is not.  The engine funnels every kernel launch, copy and memset through this shim
// (D2), so a record mode IS the capture contract: while a capture is open the launcher appends a closure
// instead of submitting, EndCapture hands the list to a graph, and GraphLaunch replays it in order.  Argument
// semantics are CUDA's: the closure holds the kernel's arguments by value (baked at capture time) while the
// buffers are re-read on every replay - which is exactly what the graph tests in gr_parity/qsa_parity/
// shared_expert_parity assert (they change a host option between capture and replay and require the captured
// kernel choice, and they rewrite the buffers and require the new payloads).
//
// The STREAM a node replays on is the one cudaGraphLaunch is given, NOT the one the node was captured from.
// That is CUDA's rule and it is load-bearing here: qsa_parity captures on a temporary stream, DESTROYS that
// stream, and replays on the default one.  A closure that remembered the capture stream therefore called into
// a freed sycl::queue - MEASURED as a hang in that test's capture section (M3), which is why the recorded
// node takes its stream as an argument.
//
// WHAT IT IS NOT: no graph object reaches the driver, so there is no upload, no cross-node scheduling and the
// order is the recording order; a recorded operation must touch device memory only, as CUDA requires.  The
// engine's own graph path (src/core/graph.cpp, GraphRegistry) is NOT enabled by this - --no-capture stays the
// default for M1..M5 (D7) and M6 owns the engine-side interception layer.
inline bool& capture_flag() {
    static bool f = false;
    return f;
}
inline std::vector<std::function<void(cudaStream_t)>>& capture_nodes() {
    static std::vector<std::function<void(cudaStream_t)>> v;
    return v;
}
inline bool capture_active() { return capture_flag(); }
inline void capture_append(std::function<void(cudaStream_t)> fn) { capture_nodes().push_back(std::move(fn)); }

// ---- submission accounting (P3, card t_d8afe53f) ------------------------------------------------
// Every submission the engine makes funnels through this shim (D2), so THIS is the authoritative count
// of what the host asks Level Zero to do - and it is what `zeCommandListAppend*` in a unitrace report
// is the driver-side counterpart of.  One relaxed counter per submission: measured free beside a
// window's other host work, so it can stay on in a measured run.
struct submit_stats {
    unsigned long long kernel = 0;     // strata::sycl_compat::launch -> one q.submit(parallel_for)
    unsigned long long memset_ = 0;    // cudaMemsetAsync -> one q.memset
    unsigned long long memcpy_ = 0;    // memcpy_impl -> one q.memcpy
    unsigned long long barrier = 0;    // cudaStreamQuery -> one ext_oneapi_submit_barrier
    unsigned long long event = 0;      // cudaEventRecord / cudaStreamWaitEvent -> one barrier submit
    unsigned long long host_fn = 0;    // cudaLaunchHostFunc -> one q.submit(host_task)
    unsigned long long graph_launch = 0;   // cudaGraphLaunch in graph mode -> ONE q.submit(the whole window)
    unsigned long long recorded = 0;   // submissions RECORDED into a capture instead of submitted
    unsigned long long total() const { return kernel + memset_ + memcpy_ + barrier + event + host_fn; }
    /// What the host handed to the runtime: individual submissions plus whole-graph submissions.
    unsigned long long submitted() const { return total() + graph_launch; }
    unsigned long long all() const { return total() + recorded; }
};
inline submit_stats& subs() {
    static submit_stats s;
    return s;
}

// ---- stream capture and graphs: the handle, and the two implementations -----------------------------
// The graph handle owns the recorded node list; the exec handle owns a copy of it, so
// cudaGraphExecDestroy cannot invalidate a graph that is still referenced.
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaGraphNode_t = void*;
using cuda_graph_nodes = std::vector<std::function<void(cudaStream_t)>>;

// ---- P3: the graph-replay equivalent (STRATA_SYCL_GRAPH=1, OFF by default) --------------------------
// WHAT THE PORT WAS MISSING.  The engine's steady-state decode step is the same sequence of kernels every
// window; on CUDA the engine's own graph path replays it, and the cost the port pays instead is measured: the
// window's host side spends most of its time in zeCommandListAppendLaunchKernelWithArguments and
// zeCommandListCreateImmediate (S2/S3), and GPU-reach wait is 42.71 ms of a 132.25 ms window at 128K.
//
// The closure list below (M3) is a faithful CAPTURE contract but NOT a replay mechanism: cudaGraphLaunch walks
// the list and calls q.submit() once per node, so a window costs exactly as many submissions as it has kernels.
// This is the second, real implementation of the same contract, on the SYCL graph extension (oneAPI 2026.1,
// L0 backend): recording a window builds a command_graph and cudaGraphLaunch hands the whole thing to the
// driver in ONE submit.  Semantics kept identical to the closure path, deliberately:
//   * the engine's body runs UNCHANGED - it calls the same launch()/memcpy/memset entry points, which submit
//     normally while recording is open (SYCL records them; nothing executes);
//   * the graph is re-submitted on the stream cudaGraphLaunch names (queue_for), and USM buffers are read at
//     replay time, so a later window with different data in the same buffers is correct;
//   * anything the extension refuses throws and is reported as cudaErrorStreamCaptureUnsupported, loudly: a
//     failed recording must NOT silently fall back to "the body already ran".
namespace graph_detail {
using sycl_mod = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>;
using sycl_exec = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>;

struct rec_state {
    sycl_mod* mod = nullptr;
    sycl::queue* q = nullptr;
    unsigned long long nodes = 0;
    bool active = false;
    bool failed = false;
    std::string err;
};
inline rec_state& rec() {
    static rec_state s;
    return s;
}
/// Runtime switch, read once.  Off unless STRATA_SYCL_GRAPH=1, and off (with one line on stderr) on a device
/// that does not report sycl::aspect::ext_oneapi_graph.
inline bool graph_mode() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_SYCL_GRAPH");
        if (v == nullptr || std::atoi(v) == 0) return false;
        try {
            if (!default_queue().get_device().has(sycl::aspect::ext_oneapi_graph)) {
                std::fprintf(stderr,
                             "strata/sycl: STRATA_SYCL_GRAPH=1 but this device does not report "
                             "sycl::aspect::ext_oneapi_graph; the launch-closure replay is used instead\n");
                return false;
            }
        } catch (...) {
            return false;
        }
        std::fprintf(stderr, "strata/sycl: STRATA_SYCL_GRAPH=1 - a window's capture records a SYCL command_graph "
                             "and cudaGraphLaunch submits it in ONE submission\n");
        return true;
    }();
    return on;
}
inline bool graph_recording() { return rec().active; }
}  // namespace graph_detail

/// The handle both capture implementations share.  `kind` is the tag: 0 = the launch-closure list (M3's
/// contract, no driver object), 1 = a SYCL modifiable command_graph (STRATA_SYCL_GRAPH=1), 2 = its executable
/// form.  Keeping one tagged object means the engine's calls need no mode awareness.
struct cuda_graph_obj {
    int kind = 0;
    unsigned long long nodes = 0;
    cuda_graph_nodes closures;                            // kind 0
    graph_detail::sycl_mod* mod = nullptr;                // kind 1
    graph_detail::sycl_exec* exec = nullptr;              // kind 2
};

// ---- the launcher ---------------------------------------------------------
// Unnamed kernel lambdas on purpose - see the measurement note on `kernel_name` above.  ONE-DIMENSIONAL
// nd_range on purpose too - see the measurement note on `this_item()`: it is what makes the sub-group equal
// CUDA's warp.  The launcher flattens (grid, block) into a linear global/local size; the kernel reconstructs
// threadIdx/blockIdx/blockDim/gridDim from the `launch_shape` the transform passes it.
//
// THE CAPTURE HOOK (PLAN.md §1.3(b)): while a stream's capture is open, the launcher does not submit - it
// APPENDS the submission to the capture's node list as a closure.  That is what makes `cudaGraphLaunch` below
// a real replay with CUDA's semantics rather than a stub, and it is why every engine operation has to funnel
// through here (D2).  Argument semantics are CUDA's: the closure holds the kernel's arguments by value (baked
// at capture time) and the buffers it reads and writes are re-read on every replay.
template <class Fn>
inline void launch(const launch_config& c, cudaStream_t stream, Fn fn) {
    if (capture_active()) {
        ++subs().recorded;
        if (graph_detail::graph_recording()) {
            ++graph_detail::rec().nodes;   // recorded into the SYCL command graph: fall through and submit,
                                           // which is what SYCL records (nothing executes while recording)
        } else {
            capture_append([c, fn](cudaStream_t s) { launch(c, s, fn); });   // replayed on the LAUNCH's stream
            return;
        }
    }
    ++subs().kernel;
    sycl::queue& q = *queue_for(reinterpret_cast<void*>(stream));
    const unsigned gx = c.grid.x ? c.grid.x : 1u, gy = c.grid.y ? c.grid.y : 1u, gz = c.grid.z ? c.grid.z : 1u;
    const unsigned bx = c.block.x ? c.block.x : 1u, by = c.block.y ? c.block.y : 1u, bz = c.block.z ? c.block.z : 1u;
    const size_t groups = size_t(gx) * gy * gz;
    const size_t threads = size_t(bx) * by * bz;
    const sycl::range<1> global{groups * threads};
    const sycl::range<1> local{threads};
    if (c.smem != 0) {
        q.submit([=](sycl::handler& h) {
            sycl::local_accessor<uint8_t, 1> dyn{sycl::range<1>(c.smem), h};
            h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
                fn(item, const_cast<uint8_t*>(
                             dyn.get_multi_ptr<sycl::access::decorated::no>().get()));
            });
        });
    } else {
        q.submit([=](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>{global, local},
                           [=](sycl::nd_item<1> item) { fn(item, static_cast<uint8_t*>(nullptr)); });
        });
    }
}

/// The `launch_shape` value the transform passes at every launch site, evaluated on the HOST from the grid and
/// block expressions that stood between `<<<` and `>>>`.
template <class G, class B>
inline launch_shape shape_of(const G& g, const B& b) {
    const dim3 gd = dim3(g), bd = dim3(b);
    return launch_shape{gd.x, gd.y, gd.z, bd.x, bd.y, bd.z};
}

inline cudaStream_t default_stream() { return nullptr; }

// ---- memory ---------------------------------------------------------------
inline bool is_usm_pointer(const void* p) {
    if (p == nullptr) return false;
    return sycl::get_pointer_type(p, default_queue().get_context()) != sycl::usm::alloc::unknown;
}

inline cudaError_t cudaMalloc(void** p, size_t bytes) {
    if (p == nullptr || bytes == 0) return cudaErrorInvalidValue;
    try {
        *p = sycl::malloc_device(bytes, default_queue());
    } catch (...) {
        *p = nullptr;
    }
    if (*p == nullptr) {
        record(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }
    backend().device_bytes += bytes;
    return cudaSuccess;
}
template <class T>
inline cudaError_t cudaMalloc(T** p, size_t bytes) {
    return cudaMalloc(reinterpret_cast<void**>(p), bytes);
}

inline cudaError_t cudaFree(void* p) {
    if (p == nullptr) return cudaSuccess;
    sycl::free(p, default_queue());
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* p, int value, size_t bytes) {
    try {
        default_queue().memset(p, value, bytes).wait();
    } catch (...) {
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int value, size_t bytes, cudaStream_t s) {
    if (capture_active() && !graph_detail::graph_recording()) {   // recorded, not submitted: replayed by cudaGraphLaunch on ITS stream
        capture_append([=](cudaStream_t rs) { cudaMemsetAsync(p, value, bytes, rs); });
        return cudaSuccess;
    }
    if (graph_detail::graph_recording()) ++graph_detail::rec().nodes;
    queue_for(reinterpret_cast<void*>(s))->memset(p, value, bytes);
    ++subs().memset_;
    return cudaSuccess;
}

/// The one place a copy happens.  A copy with an ordinary malloc'd pointer on either side is staged through
/// sycl::malloc_host: `queue.memcpy` is only defined between USM pointers, and real call sites in the tree
/// (dequant_s2_parity's std::vector staging, the weights loader's pageable reads) pass ordinary host memory.
inline cudaError_t memcpy_impl(void* dst, const void* src, size_t bytes, sycl::queue* q, bool wait) {
    if (bytes == 0) return cudaSuccess;
    ++subs().memcpy_;
    try {
        const bool du = is_usm_pointer(dst), su = is_usm_pointer(src);
        if (du && su) {
            sycl::event e = q->memcpy(dst, src, bytes);
            if (wait) e.wait();
            return cudaSuccess;
        }
        if (du && !su) {   // plain host -> device
            void* tmp = sycl::malloc_host(bytes, *q);
            std::memcpy(tmp, src, bytes);
            q->memcpy(dst, tmp, bytes).wait();   // the staging buffer dies here, so this one is synchronous
            sycl::free(tmp, *q);
            return cudaSuccess;
        }
        if (!du && su) {   // device -> plain host
            void* tmp = sycl::malloc_host(bytes, *q);
            q->memcpy(tmp, src, bytes).wait();
            std::memcpy(dst, tmp, bytes);
            sycl::free(tmp, *q);
            return cudaSuccess;
        }
        std::memcpy(dst, src, bytes);   // neither side is USM
        return cudaSuccess;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata/sycl: memcpy failed: %s\n", e.what());
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
}

inline cudaMemcpyKind infer_kind(const void* dst, const void* src) {
    const bool du = is_usm_pointer(dst), su = is_usm_pointer(src);
    if (du && su) return cudaMemcpyDeviceToDevice;
    if (du && !su) return cudaMemcpyHostToDevice;
    if (!du && su) return cudaMemcpyDeviceToHost;
    return cudaMemcpyHostToHost;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    (void) kind;   // the direction follows from the pointers; the kind argument is accepted for source parity
    // SYNCHRONOUS, as CUDA's is (PLAN.md §1.1): the "copy then read on the host" sites depend on that.
    return memcpy_impl(dst, src, bytes, &default_queue(), /*wait=*/true);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind,
                                   cudaStream_t s = nullptr) {
    // M4: CUDA's own declaration gives the stream a default (`cudaMemcpyAsync(..., kind, stream = 0)`), and
    // src/program/generate.cpp:964,966 - the PCIe probe the engine runs before choosing its expert policy -
    // calls it with four arguments.  Without the default the SYCL build of `strata` does not compile
    // ("no matching function for call to 'cudaMemcpyAsync'"), which is what M4 measured before this line.
    (void) kind;
    if (capture_active() && !graph_detail::graph_recording()) {   // recorded, not submitted
        capture_append([=](cudaStream_t rs) { (void) cudaMemcpyAsync(dst, src, bytes, kind, rs); });
        return cudaSuccess;
    }
    if (graph_detail::graph_recording()) ++graph_detail::rec().nodes;
    return memcpy_impl(dst, src, bytes, queue_for(reinterpret_cast<void*>(s)), /*wait=*/false);
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                     size_t height, cudaMemcpyKind kind, cudaStream_t s) {
    // One 1-D copy per row (PLAN.md §1.1: measure before optimising; the row counts in the tree are small).
    for (size_t r = 0; r < height; ++r) {
        const cudaError_t e = cudaMemcpyAsync((uint8_t*) dst + r * dpitch, (const uint8_t*) src + r * spitch,
                                              width, kind, s);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

/// Risk 10, SETTLED for this backend (M5c, card t_d8f8eca1): where the DEVICE reports its own free memory, that
/// figure is the one the engine budgets from.  The accounting below cannot see the driver's own footprint.
///
/// MEASURED on card 0 (Arc Pro B70, `probe/probe_m5c_vram.cpp`, plan-evidence/m5c-vram.log): on an idle card the
/// accounting reads 31.89 GiB free while UR_DEVICE_INFO_GLOBAL_MEM_FREE - what
/// `sycl::ext::intel::info::device::free_memory` returns on this part, `info/ext_intel_device_traits.def:28` -
/// says 31.25 GiB.  The 0.65 GiB between them is the driver's own (the L0 context, the page tables for tens of
/// GiB of USM, the kernel modules, the command streams): no `cudaMalloc` in this process ever asked for it, and
/// the gap is CONSTANT as the footprint grows (measured at 5, 15, 25 and 30 GiB).
///
/// THE CONSEQUENCE, measured (`logs/m5c-cache{10000,9900,9800,9700,9500}.log`): `--expert-cache auto` sized the
/// cache from 26.55 GiB of claimed free space, so the process' footprint ended 0.6 GiB past what it believed was
/// free; the driver paged, and the verify window's tail - a 248,320-block mmvq launch, the biggest single launch
/// of the run - never finished (`logs/m5b-probe8.log`, UR_RESULT_ERROR_DEVICE_LOST).  One cache slot is
/// 2,764,800 B, the cliff is between 9,900 slots (finished) and 10,000 (hung), and the difference is 264 MiB of
/// TRUE headroom.  Two further things the accounting cannot do: it never goes back up (`cudaFree` does not
/// decrement `device_bytes`, so a long-lived process under-reports free space), and it cannot see another
/// process' allocations on the same card.
inline bool device_free_bytes(size_t& out) {
    init_backend();
    backend_state& b = backend();
    try {
        sycl::device& d = b.devices[(size_t) b.current];
        if (!d.has(sycl::aspect::ext_intel_free_memory)) return false;
        out = (size_t) d.get_info<sycl::ext::intel::info::device::free_memory>();
        return true;
    } catch (...) {
        return false;   // an extension query: a device without it keeps the accounting below
    }
}

inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    init_backend();
    backend_state& b = backend();
    if (b.devices.empty()) {
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    const size_t total = b.devices[(size_t) b.current].get_info<sycl::info::device::global_mem_size>();
    if (total_bytes) *total_bytes = total;
    if (free_bytes) {
        size_t driver_free = 0;
        *free_bytes = device_free_bytes(driver_free) ? driver_free
                                                     : (total > b.device_bytes ? total - b.device_bytes : 0);
    }
    return cudaSuccess;
}

// ---- host (pinned) memory -------------------------------------------------
constexpr unsigned cudaHostAllocDefault = 0;
constexpr unsigned cudaHostAllocPortable = 1;
constexpr unsigned cudaHostAllocMapped = 2;

/// Every sycl::malloc_host allocation is device-visible, so `Mapped` is a no-op and `Portable`
/// (cross-context) is meaningless: both are accepted and ignored, the faithful mapping (PLAN.md §1.1).
inline cudaError_t cudaHostAlloc(void** p, size_t bytes, unsigned /*flags*/) {
    if (p == nullptr || bytes == 0) return cudaErrorInvalidValue;
    try {
        *p = sycl::malloc_host(bytes, default_queue());
    } catch (...) {
        *p = nullptr;
    }
    if (*p == nullptr) {
        record(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}
template <class T>
inline cudaError_t cudaMallocHost(T** p, size_t bytes) {
    return cudaHostAlloc(reinterpret_cast<void**>(p), bytes, cudaHostAllocDefault);
}
inline cudaError_t cudaMallocHost(void** p, size_t bytes) { return cudaHostAlloc(p, bytes, 0); }
inline cudaError_t cudaFreeHost(void* p) {
    if (p == nullptr) return cudaSuccess;
    sycl::free(p, default_queue());
    return cudaSuccess;
}

/// USM's host pointer IS the device-usable address: the identity, which is exactly what the engine wants
/// (PLAN.md §1.1; tests/hip/mapped_alias.cpp documents the HIP/Windows failure mode this avoids).
inline cudaError_t cudaHostGetDevicePointer(void** dev, void* host, unsigned /*flags*/) {
    *dev = host;
    return cudaSuccess;
}

constexpr unsigned cudaHostRegisterDefault = 0;
constexpr unsigned cudaHostRegisterPortable = 1;
constexpr unsigned cudaHostRegisterMapped = 2;

/// NO faithful SYCL equivalent (PLAN.md §1.3(a)): no Level-Zero/SYCL API page-locks memory it did not
/// allocate.  A SPECIFIC error and a readable message, so pinned.cu's own degrade path
/// (`--mmap-experts` / `--resident-budget-gib`) runs and the reason is on stderr.
inline cudaError_t cudaHostRegister(void* /*p*/, size_t /*bytes*/, unsigned /*flags*/) {
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "strata/sycl: cudaHostRegister has no SYCL equivalent (pre-existing memory cannot be "
                     "page-locked by the driver); the engine's pageable expert path "
                     "(--mmap-experts / --resident-budget-gib) is the documented fallback.\n");
    }
    record(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}
inline cudaError_t cudaHostUnregister(void* /*p*/) { return cudaSuccess; }

// ---- streams and events ---------------------------------------------------
inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    // The shared context (see backend_state::ctx): an explicit stream and the default queue must be able to see
    // each other's USM, and a fuse with a barrier of a foreign context throws.
    sycl::queue* q = new sycl::queue(ctx_for_device((size_t) backend().current), selected_device(), queue_properties());
    note_queue(q);
    *s = reinterpret_cast<cudaStream_t>(q);
    return cudaSuccess;
}
/// cudaStreamNonBlocking is accepted and ignored: the engine assumes IN-ORDER per stream (§1.1), and an
/// out-of-order queue would break that assumption rather than implement the flag.
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned /*flags*/) {
    return cudaStreamCreate(s);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (s == nullptr) return cudaSuccess;
    sycl::queue* q = reinterpret_cast<sycl::queue*>(s);
    drop_queue(q);   // else cudaDeviceSynchronize would wait on a destroyed queue
    delete q;
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    try {
        default_queue();   // make sure a context exists
        queue_for(reinterpret_cast<void*>(s))->wait_and_throw();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata/sycl: stream sync failed: %s\n", e.what());
        record(cudaErrorUnknown);
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
/// The non-blocking poll the layer loop needs.  The WDDM flush that made this call necessary on Windows
/// (src/core/session.cpp:913) is a Windows-only workaround and is deliberately not emulated (PLAN.md §1.3d).
inline cudaError_t cudaStreamQuery(cudaStream_t s) {
    sycl::queue* q = queue_for(reinterpret_cast<void*>(s));
    const sycl::event e = q->ext_oneapi_submit_barrier();
    ++subs().barrier;
    return e.get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete
               ? cudaSuccess
               : cudaErrorNotReady;
}

struct shim_event {
    sycl::event e;
    bool recorded = false;
};
using cudaEvent_t = shim_event*;
constexpr unsigned cudaEventDisableTiming = 1;

inline cudaError_t cudaEventCreate(cudaEvent_t* ev) {
    *ev = new shim_event();
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* ev, unsigned /*flags*/) {
    return cudaEventCreate(ev);
}
inline cudaError_t cudaEventDestroy(cudaEvent_t ev) {
    delete ev;
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t ev, cudaStream_t s = nullptr) {
    ev->e = queue_for(reinterpret_cast<void*>(s))->ext_oneapi_submit_barrier();
    ev->recorded = true;
    ++subs().event;
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t ev, unsigned /*flags*/) {
    if (ev->recorded) queue_for(reinterpret_cast<void*>(s))->ext_oneapi_submit_barrier({ev->e});
    ++subs().event;
    return cudaSuccess;
}
/// A QUERY, never a blocking sync: a blocking wait here would serialize the layer loop (PLAN.md §1.1).
inline cudaError_t cudaEventQuery(cudaEvent_t ev) {
    if (!ev->recorded) return cudaSuccess;
    return ev->e.get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete
               ? cudaSuccess
               : cudaErrorNotReady;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t ev) {
    if (ev->recorded) ev->e.wait();
    return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    if (!a->recorded || !b->recorded) return cudaErrorInvalidValue;
    if (!profiling_enabled()) {
        record(cudaErrorNotSupported);
        return cudaErrorNotSupported;   // the queue was built without the profiling property
    }
    const auto t0 = a->e.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto t1 = b->e.get_profiling_info<sycl::info::event_profiling::command_end>();
    *ms = (float) ((double) (t1 - t0) / 1.0e6);
    return cudaSuccess;
}

// ---- device management ----------------------------------------------------
inline cudaError_t cudaGetDeviceCount(int* count) {
    init_backend();
    *count = (int) backend().devices.size();
    if (*count == 0) {
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;   // as HIP without a usable device does; device_count() maps it to 0
    }
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* d) {
    init_backend();
    *d = backend().current;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int d) {
    init_backend();
    if (d < 0 || d >= (int) backend().devices.size()) {
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    backend().current = d;
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int ordinal) {
    init_backend();
    backend_state& b = backend();
    if (ordinal < 0 || ordinal >= (int) b.devices.size()) {
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    const sycl::device& d = b.devices[(size_t) ordinal];
    std::snprintf(p->name, sizeof(p->name), "%s", d.get_info<sycl::info::device::name>().c_str());
    std::snprintf(p->gcnArchName, sizeof(p->gcnArchName), "%s", device_arch_string(d).c_str());
    p->multiProcessorCount = (int) d.get_info<sycl::info::device::max_compute_units>();
    p->sharedMemPerBlock = d.get_info<sycl::info::device::local_mem_size>();
    p->totalGlobalMem = d.get_info<sycl::info::device::global_mem_size>();
    p->l2CacheSize = (int) d.get_info<sycl::info::device::global_mem_cache_size>();
    p->warpSize = (int) d.get_info<sycl::info::device::sub_group_sizes>().back();   // 32 on BMG-G31
    // There is no compute capability on an Intel GPU, and 12.0 here is NOT a claim about the device: it is
    // the value that keeps the CUDA-only ">= 7.5 at RUN time" gate in src/core/device.cu from rejecting every
    // Intel GPU.  The SYCL build of device.cu takes its own branch and never reads these two fields.
    p->major = 12;
    p->minor = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int ordinal) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) return cudaErrorInvalidValue;
    switch (attr) {
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = (int) p.sharedMemPerBlock; break;
        // Xe has NO static/dynamic split: a work-group gets up to local_mem_size (128 KiB) either way, so the
        // non-opt-in CUDA limit (48 KiB) maps to the same number rather than to a smaller one - reporting less
        // than the device has would make the port slice tiles it does not have to.
        case cudaDevAttrMaxSharedMemoryPerBlock: *value = (int) p.sharedMemPerBlock; break;
        case cudaDevAttrMultiProcessorCount: *value = p.multiProcessorCount; break;
        case cudaDevAttrComputeCapabilityMajor: *value = p.major; break;
        case cudaDevAttrComputeCapabilityMinor: *value = p.minor; break;
        default: *value = 0; break;
    }
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    try {
        sync_all();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata/sycl: device synchronize failed: %s\n", e.what());
        record(cudaErrorUnknown);
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

inline cudaError_t cudaDriverGetVersion(int* v) {
    init_backend();
    *v = backend().driver_version;
    return cudaSuccess;
}
inline cudaError_t cudaRuntimeGetVersion(int* v) {
    init_backend();
    *v = backend().runtime_version;
    return cudaSuccess;
}

/// CUDA's opt-in for >48 KiB dynamic shared memory does not exist on Xe: the whole 128 KiB of SLM is
/// available, so this becomes a BOUNDS CHECK (PLAN.md §1.1, P4) - a kernel asking for more fails here rather
/// than launching with a silently truncated dynamic size.
template <class Kernel>
inline cudaError_t cudaFuncSetAttribute(Kernel /*kernel*/, cudaFuncAttribute /*attr*/, int value) {
    init_backend();
    backend_state& b = backend();
    const size_t lim = b.devices.empty()
                           ? 131072
                           : b.devices[(size_t) b.current].get_info<sycl::info::device::local_mem_size>();
    if ((size_t) value > lim) {
        std::fprintf(stderr,
                     "strata/sycl: kernel asked for %d B of dynamic local memory; this device has %zu B\n",
                     value, lim);
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
template <class Kernel>
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, Kernel /*kernel*/) {
    *a = cudaFuncAttributes{};
    return cudaSuccess;
}

// ---- the graph API: both implementations of the contract declared above -------------------------
inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode /*mode*/) {
    if (capture_flag()) {   // nested capture: CUDA refuses it, and so does this
        record(cudaErrorStreamCaptureImplicit);
        return cudaErrorStreamCaptureImplicit;
    }
    capture_nodes().clear();
    capture_flag() = true;
    graph_detail::rec_state& r = graph_detail::rec();
    r.nodes = 0;
    r.failed = false;
    r.err.clear();
    if (graph_detail::graph_mode()) {
        sycl::queue* q = queue_for(reinterpret_cast<void*>(s));
        try {
            r.q = q;
            r.mod = new graph_detail::sycl_mod(q->get_context(), q->get_device());
            r.mod->begin_recording(*q);
            r.active = true;
        } catch (const std::exception& e) {
            r.active = false;
            r.failed = true;
            r.err = e.what();
            delete r.mod;
            r.mod = nullptr;
            std::fprintf(stderr, "strata/sycl: SYCL graph recording is unavailable on this backend: %s\n", e.what());
            capture_flag() = false;
            record(cudaErrorStreamCaptureUnsupported);
            return cudaErrorStreamCaptureUnsupported;
        }
    }
    return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t /*s*/, cudaGraph_t* g) {
    if (!capture_flag()) {
        record(cudaErrorStreamCaptureInvalidated);
        return cudaErrorStreamCaptureInvalidated;
    }
    capture_flag() = false;
    graph_detail::rec_state& r = graph_detail::rec();
    if (r.active) {
        try {
            r.mod->end_recording();
        } catch (const std::exception& e) {
            r.active = false;
            r.failed = true;
            r.err = e.what();
            std::fprintf(stderr, "strata/sycl: SYCL graph recording failed to close: %s\n", e.what());
            delete r.mod;
            r.mod = nullptr;
            if (g) *g = nullptr;
            record(cudaErrorStreamCaptureUnsupported);
            return cudaErrorStreamCaptureUnsupported;
        }
        r.active = false;
        if (g) {
            auto* o = new cuda_graph_obj();
            o->kind = 1;
            o->nodes = r.nodes;
            o->mod = r.mod;
            *g = o;
        }
        return cudaSuccess;
    }
    if (g) {
        auto* o = new cuda_graph_obj();
        o->kind = 0;
        o->nodes = capture_nodes().size();
        o->closures = capture_nodes();
        *g = o;
    }
    capture_nodes().clear();
    return cudaSuccess;
}
inline cudaError_t cudaStreamIsCapturing(cudaStream_t /*s*/, cudaStreamCaptureStatus* st) {
    if (st) *st = capture_flag() ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
inline cudaError_t cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t* /*nodes*/, size_t* n) {
    const auto* o = static_cast<const cuda_graph_obj*>(g);
    if (n) *n = o ? (size_t) o->nodes : 0;
    return o ? cudaSuccess : cudaErrorInvalidValue;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, cudaGraphNode_t* /*err*/,
                                        char* /*log*/, size_t /*n*/) {
    if (e == nullptr) return cudaErrorInvalidValue;
    const auto* o = static_cast<const cuda_graph_obj*>(g);
    if (o == nullptr) return cudaErrorInvalidValue;
    auto* out = new cuda_graph_obj();
    out->nodes = o->nodes;
    if (o->kind == 0) {
        out->kind = 0;
        out->closures = o->closures;
    } else {
        try {
            out->kind = 2;
            out->exec = new graph_detail::sycl_exec(o->mod->finalize());   // MEASURED 5-7 ms per window shape
        } catch (const std::exception& ex) {
            delete out;
            std::fprintf(stderr, "strata/sycl: SYCL graph finalize failed: %s\n", ex.what());
            record(cudaErrorStreamCaptureUnsupported);
            return cudaErrorStreamCaptureUnsupported;
        }
    }
    *e = out;
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long /*flags*/) {
    return cudaGraphInstantiate(e, g, nullptr, nullptr, 0);
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t s) {
    auto* o = static_cast<cuda_graph_obj*>(e);
    if (o == nullptr) return cudaErrorInvalidValue;
    if (o->kind == 2) {   // ONE submission for the whole window: this is what the port was missing
        sycl::queue* q = queue_for(reinterpret_cast<void*>(s));
        try {
            graph_detail::sycl_exec* ex = o->exec;
            q->submit([ex](sycl::handler& h) { h.ext_oneapi_graph(*ex); });
            ++subs().graph_launch;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "strata/sycl: SYCL graph launch failed: %s\n", ex.what());
            record(cudaErrorUnknown);
            return cudaErrorUnknown;
        }
        return cudaSuccess;
    }
    for (auto& fn : o->closures) fn(s);   // CUDA: the nodes run on the stream this call names
    return cudaSuccess;
}
inline cudaError_t cudaGraphUpload(cudaGraphExec_t /*e*/, cudaStream_t /*s*/) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    auto* o = static_cast<cuda_graph_obj*>(g);
    if (o != nullptr && o->kind == 1) {   // the modifiable graph is the engine's cudaGraph_t in this mode
        delete o->mod;
        o->mod = nullptr;
    }
    delete o;
    return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) {
    auto* o = static_cast<cuda_graph_obj*>(e);
    if (o != nullptr && o->exec != nullptr) delete o->exec;
    delete o;
    return cudaSuccess;
}

// ---- the host callback (M4: src/core/verify.cpp:1252) ------------------------------------------------------
// `cudaLaunchHostFunc(stream, fn, arg)` runs `fn(arg)` on the host AFTER everything already enqueued on that
// stream has completed.  The shim's queue is in_order (see queue_properties), and SYCL's in-order queue runs a
// host_task submitted to it after the commands submitted before it - which is exactly the CUDA contract, so
// this is a real mapping rather than a no-op.  verify.cpp uses it to raise a doorbell flag only once the
// staged expert blobs have landed, so a no-op here would be a correctness bug, not a shortcut.
inline cudaError_t cudaLaunchHostFunc(cudaStream_t s, void (*fn)(void*), void* arg) {
    if (fn == nullptr) return cudaErrorInvalidValue;
    sycl::queue* q = queue_for(reinterpret_cast<void*>(s));
    ++subs().host_fn;
    q->submit([fn, arg](sycl::handler& h) { h.host_task([fn, arg] { fn(arg); }); });
    return cudaSuccess;
}

}  // namespace strata::sycl_compat

// ===========================================================================
// CUDA's names are in the GLOBAL namespace (cudaMalloc, not cuda::malloc), and every kernel and runtime
// source says so unqualified.  These using-declarations are what put the shim's names where the tree looks
// for them - the HIP backend gets the same effect with `#define cudaMalloc hipMalloc` in
// include/strata/hip_compat/cuda_runtime.h.  Overload sets and templates come through whole.
// ===========================================================================
using strata::sycl_compat::cudaDeviceAttr;
using strata::sycl_compat::cudaDeviceGetAttribute;
using strata::sycl_compat::cudaDeviceProp;
using strata::sycl_compat::cudaDeviceSynchronize;
using strata::sycl_compat::cudaDevAttrClockRate;
using strata::sycl_compat::cudaDevAttrComputeCapabilityMajor;
using strata::sycl_compat::cudaDevAttrComputeCapabilityMinor;
using strata::sycl_compat::cudaDevAttrMaxSharedMemoryPerBlock;
using strata::sycl_compat::cudaDevAttrMaxSharedMemoryPerBlockOptin;
using strata::sycl_compat::cudaDevAttrMultiProcessorCount;
using strata::sycl_compat::cudaDriverGetVersion;
using strata::sycl_compat::cudaErrorInvalidValue;
using strata::sycl_compat::cudaErrorMemoryAllocation;
using strata::sycl_compat::cudaErrorNotReady;
using strata::sycl_compat::cudaErrorNotSupported;
using strata::sycl_compat::cudaErrorStreamCaptureUnsupported;
using strata::sycl_compat::cudaError_t;
using strata::sycl_compat::cudaErrorUnknown;
using strata::sycl_compat::cudaEventCreate;
using strata::sycl_compat::cudaEventCreateWithFlags;
using strata::sycl_compat::cudaEventDestroy;
using strata::sycl_compat::cudaEventDisableTiming;
using strata::sycl_compat::cudaEventElapsedTime;
using strata::sycl_compat::cudaEventQuery;
using strata::sycl_compat::cudaEventRecord;
using strata::sycl_compat::cudaEventSynchronize;
using strata::sycl_compat::cudaEvent_t;
using strata::sycl_compat::cudaFree;
using strata::sycl_compat::cudaFreeHost;
using strata::sycl_compat::cudaFuncAttribute;
using strata::sycl_compat::cudaFuncAttributeMaxDynamicSharedMemorySize;
using strata::sycl_compat::cudaFuncAttributes;
using strata::sycl_compat::cudaFuncGetAttributes;
using strata::sycl_compat::cudaFuncSetAttribute;
using strata::sycl_compat::cudaGetDevice;
using strata::sycl_compat::cudaGetDeviceCount;
using strata::sycl_compat::cudaGetDeviceProperties;
using strata::sycl_compat::cudaGetErrorString;
using strata::sycl_compat::cudaGetLastError;
using strata::sycl_compat::cudaGraphDestroy;
using strata::sycl_compat::cudaGraphExecDestroy;
using strata::sycl_compat::cudaGraphExec_t;
using strata::sycl_compat::cudaGraphGetNodes;
using strata::sycl_compat::cudaGraphInstantiate;
using strata::sycl_compat::cudaGraphLaunch;
using strata::sycl_compat::cudaGraphNode_t;
using strata::sycl_compat::cudaGraph_t;
using strata::sycl_compat::cudaGraphUpload;
using strata::sycl_compat::cudaLaunchHostFunc;
using strata::sycl_compat::cudaHostAlloc;
using strata::sycl_compat::cudaHostAllocDefault;
using strata::sycl_compat::cudaHostAllocMapped;
using strata::sycl_compat::cudaHostAllocPortable;
using strata::sycl_compat::cudaHostGetDevicePointer;
using strata::sycl_compat::cudaHostRegister;
using strata::sycl_compat::cudaHostRegisterDefault;
using strata::sycl_compat::cudaHostRegisterMapped;
using strata::sycl_compat::cudaHostRegisterPortable;
using strata::sycl_compat::cudaHostUnregister;
using strata::sycl_compat::cudaMalloc;
using strata::sycl_compat::cudaMallocHost;
using strata::sycl_compat::cudaMemGetInfo;
using strata::sycl_compat::cudaMemcpy;
using strata::sycl_compat::cudaMemcpy2DAsync;
using strata::sycl_compat::cudaMemcpyAsync;
using strata::sycl_compat::cudaMemcpyDefault;
using strata::sycl_compat::cudaMemcpyDeviceToDevice;
using strata::sycl_compat::cudaMemcpyDeviceToHost;
using strata::sycl_compat::cudaMemcpyHostToDevice;
using strata::sycl_compat::cudaMemcpyHostToHost;
using strata::sycl_compat::cudaMemcpyKind;
using strata::sycl_compat::cudaMemset;
using strata::sycl_compat::cudaMemsetAsync;
using strata::sycl_compat::cudaPeekAtLastError;
using strata::sycl_compat::cudaRuntimeGetVersion;
using strata::sycl_compat::cudaSetDevice;
using strata::sycl_compat::cudaStreamBeginCapture;
using strata::sycl_compat::cudaStreamCaptureMode;
using strata::sycl_compat::cudaStreamCaptureModeGlobal;
using strata::sycl_compat::cudaStreamCaptureModeThreadLocal;
using strata::sycl_compat::cudaStreamCaptureStatus;
using strata::sycl_compat::cudaStreamCaptureStatusActive;
using strata::sycl_compat::cudaStreamCaptureStatusInvalidated;
using strata::sycl_compat::cudaStreamCaptureStatusNone;
using strata::sycl_compat::cudaStreamCreate;
using strata::sycl_compat::cudaStreamCreateWithFlags;
using strata::sycl_compat::cudaStreamDestroy;
using strata::sycl_compat::cudaStreamEndCapture;
using strata::sycl_compat::cudaStreamIsCapturing;
using strata::sycl_compat::cudaStreamNonBlocking;
using strata::sycl_compat::cudaStreamQuery;
using strata::sycl_compat::cudaStreamSynchronize;
using strata::sycl_compat::cudaStreamWaitEvent;
using strata::sycl_compat::cudaStream_t;
using strata::sycl_compat::cudaSuccess;
using strata::sycl_compat::dim3;

// Explicit overloads rather than a template: a `template <class T> min(T,T)` in the global namespace would
// also capture HOST calls in the same translation unit that meant std::min.  These are the CUDA overloads.
inline signed char min(signed char a, signed char b) { return a < b ? a : b; }
inline unsigned char min(unsigned char a, unsigned char b) { return a < b ? a : b; }
inline short min(short a, short b) { return a < b ? a : b; }
inline unsigned short min(unsigned short a, unsigned short b) { return a < b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline long min(long a, long b) { return a < b ? a : b; }
inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
inline float min(float a, float b) { return a < b ? a : b; }
inline double min(double a, double b) { return a < b ? a : b; }
inline signed char max(signed char a, signed char b) { return a > b ? a : b; }
inline unsigned char max(unsigned char a, unsigned char b) { return a > b ? a : b; }
inline short max(short a, short b) { return a > b ? a : b; }
inline unsigned short max(unsigned short a, unsigned short b) { return a > b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline long max(long a, long b) { return a > b ? a : b; }
inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline double max(double a, double b) { return a > b ? a : b; }

// ===========================================================================
// The macro surface the kernels are written against.
// ===========================================================================
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)

#define threadIdx (::strata::sycl_compat::thread_idx(_sycl_shape))
#define blockIdx (::strata::sycl_compat::block_idx(_sycl_shape))
#define blockDim (::strata::sycl_compat::block_dim(_sycl_shape))
#define gridDim (::strata::sycl_compat::grid_dim(_sycl_shape))
#define __syncthreads() (::strata::sycl_compat::syncthreads())
#define __syncwarp(...) (::strata::sycl_compat::syncwarp())

// cudaMemcpyToSymbol has no SYCL equivalent.  The transform rewrites its first argument to the const_storage
// handle the `__constant__` declaration became (PLAN.md §1.3(c)); this overload is that target.
template <class T>
inline cudaError_t cudaMemcpyToSymbol(::strata::sycl_compat::const_storage<T>& storage, const void* src,
                                      size_t bytes, size_t offset = 0) {
    return storage.upload(src, bytes, offset);
}

#include "intrinsics.hpp"

// ---------------------------------------------------------------------------
// CUDA's atomics are GLOBAL device functions and kernel text calls them unqualified:
// `atomicCAS(&slot, 0, 1)` (kv_stream.cu:102), `atomicOr(...)` (sampler.cu).  ADL cannot reach into
// strata::sycl_compat for a builtin argument type, so the names are introduced here - after intrinsics.hpp,
// which defines them.
// ---------------------------------------------------------------------------
using strata::sycl_compat::atomicAdd;
using strata::sycl_compat::atomicAnd;
using strata::sycl_compat::atomicCAS;
using strata::sycl_compat::atomicExch;
using strata::sycl_compat::atomicMax;
using strata::sycl_compat::atomicMin;
using strata::sycl_compat::atomicOr;
