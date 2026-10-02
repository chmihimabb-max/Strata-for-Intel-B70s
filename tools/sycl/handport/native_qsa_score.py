# tools/sycl/handport/native_qsa_score.py - the hand-port replacements for src/kernels/cuda/native_qsa_score.cu.
#
# PLAN.md §2.3 sites #10-#12: ldmatrix.sync.aligned.m8n8.x4.b16, ldmatrix...x2.b16 and
# mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32.
#
# The file already carries the portable path PLAN.md Risk 1 names as the fallback: score_kernel's
# `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` arm is a plain FP32 FMA dot per indexer head, with the
# heads ReLU'd and added in the documented order (native_qsa_score.cu:64-79).  Under SYCL that condition is
# FALSE (__CUDA_ARCH__ is undefined, and the *other* guard - `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`
# - is TRUE, so the asm would be compiled: PLAN.md Risk 7 exactly).  The port flips both: the asm functions go,
# the FP32 arm becomes the compiled one.
#
# The XMX question for this file was measured, not assumed: the tc path needs tf32, and the device refuses tf32
# joint_matrix outright - "joint_matrix with parameters matrix_type::tf32, use::a, Rows=16, Cols=8 is not
# supported on this device" (probe21, plan-evidence/M3-risk1-shapes.txt).  So there is no XMX arm to gate here.
ASM_SITES = 3

REPLACEMENTS = [
    # 1. the three asm device helpers -> gone (the FP32 arm below does not use them)
    (
        r"""#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void load_a(TileA& a,const float* p) {
    const float* src=p+(threadIdx.x%16)*STRIDE+(threadIdx.x/16)*4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.x[0]),"=r"(a.x[1]),"=r"(a.x[2]),"=r"(a.x[3]):"l"(src));
}
__device__ __forceinline__ void load_b(TileB& b,const float* p) {
    const float* src=p+(threadIdx.x%8)*STRIDE+((threadIdx.x/8)*4)%8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
        : "=r"(b.x[0]),"=r"(b.x[1]):"l"(src));
}
__device__ __forceinline__ void mma(TileC& c,const TileA& a,const TileB& b) {
    // Deliberately no cvt.rn.tf32: pinned mma.cuh passes raw F32 bits directly.
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3])
        : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b.x[0]),"r"(b.x[1]));
}
#endif""",
        r"""// PLAN.md §2.3 sites #10-#12, hand-ported (tools/sycl/handport/native_qsa_score.py).  The three functions
// that carried them (ldmatrix x4, ldmatrix x2, mma.sync tf32) are deleted rather than re-spelled: SYCL has no
// ldmatrix, and tf32 joint_matrix is REFUSED by this device - probe21, exact driver text
// (plan-evidence/M3-risk1-shapes.txt):
//   "joint_matrix with parameters matrix_type::tf32, use::a, Rows=16, Cols=8 is not supported on this device"
// The CUDA guard here was `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`, TRUE when __CUDA_ARCH__ is
// undefined, i.e. TRUE under SYCL - which is why this file is hand-ported at all (PLAN.md Risk 7).
// score_kernel below therefore takes its FP32 FMA arm, the portable fallback PLAN.md Risk 1 names.""",
    ),
    # 2. the kernel's path selector: the pre-sm80 FP32 arm is the one this backend compiles
    (
        r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // Turing (STRATA_EXPERIMENTAL_SM75, a layer-split stage): no tf32 mma.  The same scores with FP32 FMAs, one row
    // per thread of the first warp - rounded differently from the tensor-core path (FP32 instead of TF32 inputs).""",
        r"""#if 1   // SYCL hand port: the tf32 ldmatrix/mma arm above is gone (device refuses tf32 joint_matrix, and
        // there is no SYCL ldmatrix).  This arm is the .cu's own portable one; see the note where the asm was.
    // Turing (STRATA_EXPERIMENTAL_SM75, a layer-split stage): no tf32 mma.  The same scores with FP32 FMAs, one row
    // per thread of the first warp - rounded differently from the tensor-core path (FP32 instead of TF32 inputs).""",
    ),
]
