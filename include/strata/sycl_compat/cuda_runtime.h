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
inline auto this_item() { return sycl::ext::oneapi::this_work_item::get_nd_item<3>(); }

inline dim3 thread_idx() {
    const auto it = this_item();
    return dim3(it.get_local_id(0), it.get_local_id(1), it.get_local_id(2));
}
inline dim3 block_idx() {
    const auto it = this_item();
    return dim3(it.get_group(0), it.get_group(1), it.get_group(2));
}
inline dim3 block_dim() {
    const auto it = this_item();
    return dim3(it.get_local_range(0), it.get_local_range(1), it.get_local_range(2));
}
inline dim3 grid_dim() {
    const auto it = this_item();
    return dim3(it.get_group_range(0), it.get_group_range(1), it.get_group_range(2));
}

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
    std::vector<sycl::queue*> streams;   // index 0 = the default (CUDA legacy) stream's queue
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

struct stream_t;
using cudaStream_t = stream_t*;                 // nullptr is CUDA's default (legacy) stream
using cudaError_t = int;

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorNotReady = 34;
constexpr cudaError_t cudaErrorNotSupported = 801;
constexpr cudaError_t cudaErrorStreamCaptureUnsupported = 900;
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
    }
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
    backend_state& b = backend();
    if (b.streams.empty()) {
        b.streams.push_back(new sycl::queue(selected_device(), queue_properties()));
    }
    return *b.streams[0];
}

inline sycl::queue* queue_for(void* stream) {
    if (stream == nullptr) return &default_queue();
    return static_cast<sycl::queue*>(stream);
}

inline void sync_all() {
    init_backend();
    for (sycl::queue* q : backend().streams) q->wait_and_throw();
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

// ---- the launcher ---------------------------------------------------------
// Unnamed kernel lambdas on purpose - see the measurement note on `kernel_name` above.  The tag parameter is
// gone: the transform no longer emits one, and the device compiler names the kernel after the lambda itself.
template <class Fn>
inline void launch(const launch_config& c, cudaStream_t stream, Fn fn) {
    sycl::queue& q = *queue_for(reinterpret_cast<void*>(stream));
    const sycl::range<3> global{c.grid.x * c.block.x, c.grid.y * c.block.y, c.grid.z * c.block.z};
    const sycl::range<3> local{c.block.x, c.block.y, c.block.z};
    if (c.smem != 0) {
        q.submit([=](sycl::handler& h) {
            sycl::local_accessor<uint8_t, 1> dyn{sycl::range<1>(c.smem), h};
            h.parallel_for(sycl::nd_range<3>{global, local}, [=](sycl::nd_item<3> item) {
                fn(item, const_cast<uint8_t*>(
                             dyn.get_multi_ptr<sycl::access::decorated::no>().get()));
            });
        });
    } else {
        q.submit([=](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<3>{global, local},
                           [=](sycl::nd_item<3> item) { fn(item, static_cast<uint8_t*>(nullptr)); });
        });
    }
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
    queue_for(reinterpret_cast<void*>(s))->memset(p, value, bytes);
    return cudaSuccess;
}

/// The one place a copy happens.  A copy with an ordinary malloc'd pointer on either side is staged through
/// sycl::malloc_host: `queue.memcpy` is only defined between USM pointers, and real call sites in the tree
/// (dequant_s2_parity's std::vector staging, the weights loader's pageable reads) pass ordinary host memory.
inline cudaError_t memcpy_impl(void* dst, const void* src, size_t bytes, sycl::queue* q, bool wait) {
    if (bytes == 0) return cudaSuccess;
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
                                   cudaStream_t s) {
    (void) kind;
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

/// SYCL has no driver-side free-memory query (Risk 10), so the shim accounts for what it handed out itself.
inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    init_backend();
    backend_state& b = backend();
    if (b.devices.empty()) {
        record(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    const size_t total = b.devices[(size_t) b.current].get_info<sycl::info::device::global_mem_size>();
    if (total_bytes) *total_bytes = total;
    if (free_bytes) *free_bytes = total > b.device_bytes ? total - b.device_bytes : 0;
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
    *s = reinterpret_cast<cudaStream_t>(new sycl::queue(selected_device(), queue_properties()));
    return cudaSuccess;
}
/// cudaStreamNonBlocking is accepted and ignored: the engine assumes IN-ORDER per stream (§1.1), and an
/// out-of-order queue would break that assumption rather than implement the flag.
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned /*flags*/) {
    return cudaStreamCreate(s);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (s == nullptr) return cudaSuccess;
    delete reinterpret_cast<sycl::queue*>(s);
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
inline cudaError_t cudaEventRecord(cudaEvent_t ev, cudaStream_t s) {
    ev->e = queue_for(reinterpret_cast<void*>(s))->ext_oneapi_submit_barrier();
    ev->recorded = true;
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t ev, unsigned /*flags*/) {
    if (ev->recorded) queue_for(reinterpret_cast<void*>(s))->ext_oneapi_submit_barrier({ev->e});
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

// ---- stream capture and graphs: no SYCL equivalent (PLAN.md §1.3(b), D7) ---
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaGraphNode_t = void*;

inline cudaError_t cudaStreamBeginCapture(cudaStream_t /*s*/, cudaStreamCaptureMode /*mode*/) {
    record(cudaErrorStreamCaptureUnsupported);
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t /*s*/, cudaGraph_t* g) {
    if (g) *g = nullptr;
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaStreamIsCapturing(cudaStream_t /*s*/, cudaStreamCaptureStatus* st) {
    *st = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
inline cudaError_t cudaGraphGetNodes(cudaGraph_t /*g*/, cudaGraphNode_t* /*nodes*/, size_t* n) {
    if (n) *n = 0;
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t /*g*/, cudaGraphNode_t* /*err*/,
                                        char* /*log*/, size_t /*n*/) {
    if (e) *e = nullptr;
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t /*g*/, unsigned long long /*flags*/) {
    if (e) *e = nullptr;
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t /*e*/, cudaStream_t /*s*/) {
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaGraphUpload(cudaGraphExec_t /*e*/, cudaStream_t /*s*/) {
    return cudaErrorStreamCaptureUnsupported;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t /*g*/) { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t /*e*/) { return cudaSuccess; }

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

#define threadIdx (::strata::sycl_compat::thread_idx())
#define blockIdx (::strata::sycl_compat::block_idx())
#define blockDim (::strata::sycl_compat::block_dim())
#define gridDim (::strata::sycl_compat::grid_dim())
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
