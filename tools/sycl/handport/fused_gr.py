# tools/sycl/handport/fused_gr.py - the hand-port replacements for src/kernels/cuda/fused_gr.cu.
#
# PLAN.md §2.3 sites #13-#16: cp.async.cg.shared.global + cp.async.commit_group + wait_group 1/0 (4 sites).
# The .cu already carries its own portable branch for these (the `#else` arm of STRATA_GR_CP_ASYNC), and the
# guard that selects the asm arm is `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)`
# - FALSE under SYCL, so the behaviour is unchanged; what changes is that the asm text is GONE, which is what
# PLAN.md Risk 7 requires (0 surviving `asm(` in every TU this build compiles).
#
# MEASURED, not assumed, that the synchronous copy is correct here: gr_down_staged_kernel reads the buffer
# (fused_gr.cu:651-663), then `__syncthreads()` (:664), then stage_htile() refills that same buffer (:665).
# The barrier already provides the ordering cp.async's commit/wait pair exists to provide, so both waits are
# no-ops and the 16-byte copy is a plain float4 load/store.
#
# The XMX gate (STRATA_SYCL_XMX, PLAN.md §5 M3) is not consulted: this file has no tensor-core path at all -
# its CUDA form is `portable` in PLAN.md §2's XMX column.
ASM_SITES = 4

REPLACEMENTS = [
    # 1. the four asm wrappers -> the portable copies (PLAN.md §2.3 sites #13-#16)
    (
        r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}""",
        r"""// PLAN.md §2.3 sites #13-#16, hand-ported for SYCL (tools/sycl/handport/fused_gr.py).
//
// The CUDA form selects cp.async with `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)`
// - i.e. it does NOT require a shim to define __CUDA_ARCH__ >= 800; under SYCL __CUDA_ARCH__ is undefined, so
// the .cu's own portable arm is the one that would run.  This port makes that arm the only one and deletes the
// asm, so no shim change can compile PTX into the device image (PLAN.md Risk 7: 0 surviving asm tokens per TU).
//
// The synchronous copy is MEASURED safe rather than assumed: gr_down_staged_kernel reads the staged buffer
// (:651-663), hits __syncthreads() (:664) and only then refills that same buffer with tile h+2 (:665), so the
// ordering cp.async's commit/wait pair exists to provide is already provided by the barrier.  Both waits are
// therefore no-ops.  SITES #13-#16 = 4 (PLAN.md §2.3).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
}
__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait1() {}
__device__ __forceinline__ void cp_async_wait0() {}""",
    ),
]
