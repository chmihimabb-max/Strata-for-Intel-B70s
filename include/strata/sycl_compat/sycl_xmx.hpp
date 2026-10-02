// include/strata/sycl_compat/sycl_xmx.hpp - the STRATA_SYCL_XMX gate (PLAN.md §5 M3, §6 Risk 1).
//
// WHAT THE GATE IS.  `STRATA_SYCL_XMX=1` arms the XMX (Intel DPAS / joint_matrix) paths in the ported kernels.
// It is OFF by default because PLAN.md §5 M3 requires the portable FP32 fallbacks green BEFORE any XMX path is
// enabled, and because the answer to "does this device have the shape?" is a device fact, not a build flag.
//
// THE DEVICE FACT (measured, probe/probe21_xmx_shapes.cpp -> plan-evidence/M3-risk1-shapes.txt).  Every shape
// compiles on this toolchain - the unified joint_matrix API defers the shape to the driver - so the probe RUNS
// each tile and records the driver's exact refusal.  On Intel(R) Graphics [0xe223] (Arc Pro B70, BMG-G31):
//
//   f16  16x16x16  RUNS, max abs err 0 (B row_major and col_major)
//   f16   8x16x16  RUNS       bf16 16x16x16 RUNS       bf16 8x16x16 RUNS
//   tf32 16x16x8   REFUSED: "joint_matrix with parameters matrix_type::tf32, use::a, Rows=16, Cols=8 is not
//                            supported on this device"
//   tf32 16x8x8    REFUSED: "joint_matrix with parameters matrix_type::fp32, use::accumulator, Rows=16,
//                            Cols=8 is not supported on this device"
//   int8 16x32x16  REFUSED: "joint_matrix with parameters matrix_type::sint32, use::accumulator, Rows=16,
//                            Cols=32 is not supported on this device"
//   int8  8x32x16  REFUSED: same, Rows=8, Cols=32 - i.e. even the shape GATES.md G1 lists as supported, with
//                            B in row_major, col_major AND ext_intel_packed
//
// So of the three CUDA kernels that could use XMX, exactly one has a usable shape: qsa_prompt_attn.cu's
// mma.m16n8k16 f16.  qsa_select.cu and native_qsa_score.cu need tf32, which this device refuses outright - that
// is the Risk 1 verdict, and those two files keep their portable FP32 paths (the fallback PLAN.md Risk 1 names).
//
// xmx_f16_ok() therefore asks the device rather than trusting a table: it runs one f16 16x16x16 tile and catches
// the refusal.  It runs at most once per process and only when the gate is on.
#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(STRATA_USE_SYCL)
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "../sycl_compat/cuda_runtime.h"   // default_queue()
#endif

namespace strata::sycl_compat {

/// STRATA_SYCL_XMX: set and not "0" arms the XMX paths.
inline bool xmx_requested() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SYCL_XMX");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    return on;
}

#if defined(STRATA_USE_SYCL)
/// One f16 16x16x16 tile; 'y' compiled+ran, 'n' refused (the exception text is printed once), '?' never tried.
inline char xmx_f16_16x16x16_state() {
    static const char state = [] () -> char {
        if (!xmx_requested()) return '?';
        namespace mx = sycl::ext::oneapi::experimental::matrix;
        sycl::queue& q = default_queue();
        try {
            sycl::half* a = sycl::malloc_device<sycl::half>(256, q);
            sycl::half* b = sycl::malloc_device<sycl::half>(256, q);
            float* c = sycl::malloc_device<float>(256, q);
            q.memset(a, 0, 256 * sizeof(sycl::half));
            q.memset(b, 0, 256 * sizeof(sycl::half));
            q.parallel_for(sycl::nd_range<1>{32, 32}, [=](sycl::nd_item<1> it) {
                 auto sg = it.get_sub_group();
                 mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> A;
                 mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> B;
                 mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> C;
                 mx::joint_matrix_fill(sg, C, 0.0f);
                 mx::joint_matrix_load(sg, A,
                     sycl::address_space_cast<sycl::access::address_space::global_space,
                                              sycl::access::decorated::no>(a), 16);
                 mx::joint_matrix_load(sg, B,
                     sycl::address_space_cast<sycl::access::address_space::global_space,
                                              sycl::access::decorated::no>(b), 16);
                 mx::joint_matrix_mad(sg, C, A, B, C);
                 mx::joint_matrix_store(sg, C,
                     sycl::address_space_cast<sycl::access::address_space::global_space,
                                              sycl::access::decorated::no>(c), 16, mx::layout::row_major);
             }).wait_and_throw();
            sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
            std::fprintf(stderr, "STRATA_SYCL_XMX=1: the device ran an f16 16x16x16 joint_matrix tile "
                                 "(the only XMX shape the M3 kernels can use)\n");
            return 'y';
        } catch (const std::exception& e) {
            std::fprintf(stderr, "STRATA_SYCL_XMX=1: the device REFUSED the f16 16x16x16 tile: %s\n", e.what());
            return 'n';
        }
    }();
    return state;
}
inline bool xmx_f16_ok() { return xmx_f16_16x16x16_state() == 'y'; }
#else
inline char xmx_f16_16x16x16_state() { return '?'; }
inline bool xmx_f16_ok() { return false; }
#endif

/// The gate as a kernel may use it: the env switch AND the device agreeing.
inline bool xmx_enabled() { return xmx_requested() && xmx_f16_ok(); }

/// A one-line reason for a log line: exactly why the caller is (not) on an XMX path.
inline const char* xmx_reason() {
    if (!xmx_requested()) return "STRATA_SYCL_XMX is off: portable FP32 path (PLAN.md M3)";
    if (!xmx_f16_ok()) return "STRATA_SYCL_XMX is on but the device refused the f16 16x16x16 tile";
    return "STRATA_SYCL_XMX is on and the device ran an f16 16x16x16 tile; no tile-form XMX kernel exists in M3";
}

}  // namespace strata::sycl_compat
