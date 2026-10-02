# tools/sycl/handport/qsa_prompt_attn.py - the hand-port replacements for src/kernels/cuda/qsa_prompt_attn.cu.
#
# PLAN.md §2.3 sites #4-#9: three mma.sync f16 (m16n8k8 x2 + a k16) and cp.async (cg.shared.global with the
# predicated 16/0 byte count, commit_group, wait_group 1).  6 sites.
#
# WHAT THE PORT DOES, and why each piece is the portable path rather than a guess:
#
#   * mma16816 -> the CUDA file's OWN pre-sm_80 branch.  The file documents (lines 30-32) that a pre-sm_75 build
#     "compiles the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old kernel
#     runs there", and that "Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead of
#     launch_i8".  This backend is in exactly that position - the v1 kernel (prompt_attn_kernel<KV_MODE>, pure
#     FP32 with shuffles, no asm at all) is the portable path PLAN.md Risk 1 names, and qsa_prompt_attn_batch
#     below selects it.  The i8 kernel stays in the file as the trap path, so a mis-dispatch would be loud.
#
#   * cp.async -> the plain predicated 16-byte copy.  THE ZERO-FILL FORM IS KEPT A ZERO FILL: the CUDA form
#     copies `valid ? 16 : 0` bytes, and the kernel reads all 16 bytes of a cell whether its pool row is
#     resident or not, so the non-valid case must leave zeros - not stale smem.
#
#   * the gate.  STRATA_SYCL_XMX (PLAN.md §5 M3) arms the XMX paths; the device probe behind
#     include/strata/sycl_compat/sycl_xmx.hpp measured that f16 16x16x16 is the ONE shape this device accepts
#     (plan-evidence/M3-risk1-shapes.txt).  This file needs it (m16n8k16 f16), so the gate is consulted and
#     logged here; no tile-form XMX kernel exists in M3, which the log line states in as many words rather
#     than implying speed that was never measured.
ASM_SITES = 6

REPLACEMENTS = [
    # 1. the gate header (host-side; the file's own includes stay where they are)
    (
        r"""#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
""",
        r"""#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "strata/sycl_compat/sycl_xmx.hpp"    // STRATA_SYCL_XMX gate + the measured f16 16x16x16 device probe
#include "strata/sycl_compat/mma16816.hpp"    // the FP32 emulation of mma.sync m16n8k16 f16 (PLAN.md §2.3)
""",
    ),
    # 2. the sm80 guard: TRUE when __CUDA_ARCH__ is undefined => it would select the asm body (PLAN.md Risk 7)
    (
        r"""// The MMA below needs sm_75 or newer (Turing runs it as two k=8 steps); cp.async needs sm_80. Builds for pre-sm_75
// cards compile the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old kernel runs
// there.  Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead of launch_i8.
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif""",
        r"""// SYCL hand port of the guard above (PLAN.md §2.3 sites #4-#9, Risk 7).  The CUDA form is
//     #elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// which is TRUE when __CUDA_ARCH__ is undefined - i.e. under SYCL it would pick the mma.sync/cp.async body and
// compile CUDA asm into the device image.  Forcing 0 keeps the file on the pre-sm_80 semantics its own comments
// describe ("compile the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old
// kernel runs there"), and the asm is deleted outright so no shim can select it.
#define STRATA_PA_SM80 0""",
    ),
    # 3. mma16816: the asm branches (m16n8k8 pair + m16n8k16) -> the trap-only body
    (
        r"""__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80 && (defined(__HIPCC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750)
    __trap();   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif !STRATA_PA_SM80
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}""",
        r"""// PLAN.md §2.3 sites #4-#6, hand-ported (tools/sycl/handport/qsa_prompt_attn.py).  SYCL/SPIR-V has no
// mma.sync and its XMX equivalent (joint_matrix f16 16x16x16, measured AVAILABLE on this device by probe21) is
// a TILE api, not the fragment-register one this kernel works in.  The portable arm must therefore be a real
// emulation, not a refusal: include/strata/sycl_compat/mma16816.hpp rebuilds the two operands with 32
// select_from_group ops and does the 16-term dot products in FP32 (products of f16 values are exact in FP32, so
// only the summation order differs from the hardware instruction).  Its fragment-layout convention is verified
// on the device against a scalar reference by probe/probe23_mma_emul.cpp (128 of 128 outputs, max abs err 0).
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    ::strata::sycl_compat::mma16816_f32(::strata::sycl_compat::this_item().get_sub_group(), c, a, b);
}""",
    ),
    # 4. cp.async: the predicated 16/0 byte copy, commit_group and wait_group 1
    (
        r"""__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    __trap();
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_PA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}""",
        r"""// PLAN.md §2.3 sites #7-#9, hand-ported.  cp.async has no SYCL spelling; the copy is synchronous here.  The
// predicate is NOT dropped: the CUDA form moves `valid ? 16 : 0` bytes, and the kernel reads all 16 bytes of a
// cell whether or not its pool row is resident, so a non-valid cell must be ZERO-FILLED, not left stale.
// The wait/commit pair is dropped because a synchronous copy is already complete when it returns; the ordering
// the async pipeline needed is provided by the __syncwarp() the call site keeps after cp_async_wait1()
// (qsa_prompt_attn.cu:504) and at the end of every chunk (:617), which is what makes the staged stage write
// visible to the lanes that read it.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(gmem);
    *reinterpret_cast<float4*>(smem) = v;
}
__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait1() {}""",
    ),
    # 5. the dispatcher: the shim's compute capability (12.0) would select the trapping i8 kernel
    (
        r"""    if (n_q <= 0) return true;
    bool turing = false;   // per call, from the CURRENT device (a layer split can mix Turing with newer cards)
    {   // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
        // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).  An older card
        // keeps the old kernel.
        // #371: the compute capability with its minor - sm_70 (V100) has no m16n8k8 (the kernels trap below sm_75)
        static int cc[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc[dev] == 0) {
            int major = 0, minor = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // STRATA_QSA_WARP=1|attn (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "attn")) ? 75
                      : 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);
        }
        if (cc[dev] < 75) return false;
        turing = cc[dev] < 80;
    }""",
        r"""    if (n_q <= 0) return true;
    // SYCL hand port of the device gate above.  The shim answers the compute-capability query with 12.0 (there is
    // no CUDA cc on an Intel GPU; cuda_runtime.h:731-735), so 10*cc_major_of(12)+cc_minor_of(0) = 120 would send
    // the dispatch to launch_i8 - the mma.sync/cp.async kernel, which on this backend is the __trap() body.
    // The v1 kernel is the portable path here for exactly the reason this file gives for Turing ("no cp.async,
    // so it runs the v1 kernel (launch<1>, same accuracy, another summation order)"), so take it.
    bool turing = true;
    {   // the XMX gate is still consulted, because this is the one M3 file whose CUDA kernel uses an XMX shape
        // this device accepts (f16 16x16x16, measured - probe21).  It does not change the path yet: the tile-form
        // kernel is not written, and the log line says so instead of implying a speedup.
        static bool reported = false;
        if (!reported) {
            reported = true;
            std::fprintf(stderr,
                         "qsa_prompt_attn_batch: SYCL backend -> portable v1 kernel (m16n8k16 has no SYCL "
                         "spelling); %s\n",
                         strata::sycl_compat::xmx_reason());
        }
    }""",
    ),
]
