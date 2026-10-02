# tools/sycl/handport/qsa_select.py - the hand-port replacements for src/kernels/cuda/qsa_select.cu.
#
# PLAN.md §2.3 sites #1-#2: `cvt.rna.tf32.f32` (the round-to-nearest tf32 conversion) and
# `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`.  This is the QSA block scorer the AMD backend
# accelerates with WMMA (PLAN.md §2.2) - i.e. the one M3 file where an XMX path would be worth having.
#
# THE XMX VERDICT FOR THIS FILE IS A MEASURED NEGATIVE (PLAN.md Risk 1, probe/probe21_xmx_shapes.cpp): the
# device refuses tf32 joint_matrix outright -
#   "joint_matrix with parameters matrix_type::tf32, use::a, Rows=16, Cols=8 is not supported on this device"
# (plan-evidence/M3-risk1-shapes.txt) - so there is no tf32 tile to build the tc kernel on.  The .cu's own
# portable path is the warp kernel it keeps for pre-sm_80 cards (block_scores_warp_kernel), which the host
# gate below now refuses through to.  The tf32_hi conversion keeps its portable definition, which the .cu
# already documents as what the pinned mma expects anyway ("Deliberately no cvt.rn.tf32: pinned mma.cuh passes
# raw F32 bits directly" - native_qsa_score.cu:48).
ASM_SITES = 2

REPLACEMENTS = [
    # 1. add the gate header
    (
        r"""#include <cuda_runtime.h>

#include <cfloat>""",
        r"""#include <cuda_runtime.h>
#include "strata/sycl_compat/sycl_xmx.hpp"   // STRATA_SYCL_XMX gate + the measured tf32 refusal

#include <cfloat>""",
    ),
    # 2. the sm80 guard (true under SYCL) -> 0, with the reason
    (
        r"""// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif""",
        r"""// SYCL hand port of the guard (PLAN.md §2.3 sites #1-#2, Risk 7).  The CUDA form's second arm,
//     #elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// is TRUE when __CUDA_ARCH__ is undefined, i.e. exactly under SYCL: left alone it would compile the tf32
// cvt/mma asm into the device image.  0 keeps the file's own pre-sm_80 semantics and the asm is deleted.
// Independent of the guard, the device was measured to REFUSE tf32 joint_matrix (probe21), so there is no XMX
// path to arm here even with STRATA_SYCL_XMX=1 - see qsa_block_scores_tc below.
#define STRATA_SEL_SM80 0""",
    ),
    # 3. the two asm functions -> their portable definitions
    (
        r"""__device__ __forceinline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_SEL_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}""",
        r"""// PLAN.md §2.3 sites #1-#2, hand-ported (tools/sycl/handport/qsa_select.py).
//
// tf32_hi keeps the portable definition the .cu already carries for pre-sm_80 cards, and which the pinned
// mma expects anyway: raw F32 bits, no cvt.rna (native_qsa_score.cu:48 says so in as many words).
__device__ __forceinline__ uint32_t tf32_hi(float x) { return __float_as_uint(x); }
// mma_tf32 has no SYCL form: this device refuses tf32 joint_matrix (measured; the exact driver text is in the
// SYCL_XMX gate header), so the tc kernel is unreachable and the host gate below refuses through to the warp
// kernel.  A trap keeps a mis-dispatch loud rather than silently returning wrong scores.
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    (void) c; (void) a; (void) b;
    __trap();
}""",
    ),
    # 4. the host gate: the shim reports cc 12.0, which would select the trapping tc kernel
    (
        r"""#else
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // STRATA_QSA_WARP=1|select (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
        }
        if (cc_major[dev] < 8) return false;
    }""",
        r"""#else
    {   // SYCL hand port of this gate.  Two independent reasons, and both are measured:
        //  (1) the shim answers the cc query with 12.0 (cuda_runtime.h:731-735 says why), so
        //      strata::cc_major_of(12) = 12 >= 8 would send the dispatch into the tc kernel - which on this
        //      backend is the __trap() body;
        //  (2) the tc kernel is the TF32 mma path, and the device refuses tf32 joint_matrix outright.  probe21,
        //      exact driver text (plan-evidence/M3-risk1-shapes.txt):
        //        "joint_matrix with parameters matrix_type::tf32, use::a, Rows=16, Cols=8 is not supported on
        //         this device"
        //      so STRATA_SYCL_XMX=1 does not change this either: there is no tf32 XMX path to arm on BMG-G31.
        // The warp kernel below is the portable FP32 scorer PLAN.md Risk 1 names as the fallback.
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr,
                         "qsa_block_scores: no TF32 tc path on this backend (the device refuses tf32 "
                         "joint_matrix); %s -> warp kernel\n", strata::sycl_compat::xmx_reason());
        }
        return false;
    }""",
    ),
]
