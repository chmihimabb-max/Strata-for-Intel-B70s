#pragma once
// include/strata/sycl_compat/cublas_v2.h - cuBLAS's surface, for the SYCL backend.
//
// The engine's only BLAS use is one call shape in src/prefill/gemm.cu:388,407:
//
//   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, W, CUDA_R_16BF, K,
//                X, CUDA_R_16BF, K, &beta, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
//
// i.e. COLUMN-MAJOR, the weight matrix TRANSPOSED, alpha=1, and a caller-owned workspace from
// cublasSetWorkspace (PLAN.md §1.1).  The SYCL replacement is oneMKL's
// oneapi::mkl::blas::column_major::gemm on the engine's own queue, keeping the same transpose convention.
//
// STATUS: COMPLETE for the engine's call shape (M4, PLAN.md §5 M4 Route S1).  f32/f32, bf16/bf16 and f16/f16
// into an fp32 accumulator all go to oneMKL's column_major::gemm on the engine's queue with the same transpose
// convention cuBLAS gets; any other dtype combination is refused with a printed reason rather than guessed.
// The one thing that is NOT carried over is the caller-owned workspace - oneMKL has no scratch argument - and
// the reason is written out at cublasGemmEx below, together with the measurement that backs the dtype mapping.
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#if __has_include(<oneapi/mkl/blas.hpp>)
#define STRATA_SYCL_HAVE_ONEMKL 1
#include <oneapi/mkl/blas.hpp>
#else
#define STRATA_SYCL_HAVE_ONEMKL 0
#endif

// ---- enums -----------------------------------------------------------------
enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
};
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_16F = 64, CUBLAS_COMPUTE_32I = 70 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1 };
enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,
    CUDA_R_32F = 0,
    CUDA_R_8I = 3,
    CUDA_R_32I = 10,
};

/// The handle carries the engine's stream and workspace, exactly as cuBLAS's does.
struct cublasContext {
    sycl::queue* q = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
};
using cublasHandle_t = cublasContext*;

inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
    if (h == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    *h = new cublasContext();
    (*h)->q = &strata::sycl_compat::default_queue();
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t h) {
    delete h;
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s) {
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->q = strata::sycl_compat::queue_for(reinterpret_cast<void*>(s));
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t h, void* ws, size_t bytes) {
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->workspace = ws;
    h->workspace_bytes = bytes;
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m) {
    if (h == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    h->math = m;
    return CUBLAS_STATUS_SUCCESS;
}

#if STRATA_SYCL_HAVE_ONEMKL
namespace strata::sycl_compat::detail {
inline oneapi::mkl::transpose to_mkl(cublasOperation_t op) {
    return op == CUBLAS_OP_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
}
}  // namespace strata::sycl_compat::detail
#endif

/// The one call shape gemm.cu uses.  Column-major on both sides; A is (lda x m) with OP applied by oneMKL
/// through the same transpose enum cuBLAS gets, so the convention is preserved rather than reinterpreted.
///
/// M4 (Route S1) wires all four dtypes the engine passes:
///   cublasGemmEx(handle, OP_T, OP_N, N, T, K, &alpha, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y,
///                CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)      <- Gemm::bf16, gemm.cu:388
///   ...        CUDA_R_16F  both inputs                                          <- Gemm::f16,  gemm.cu:407
///   ...        CUDA_R_32F  both inputs and output                                <- the M1 arm
/// oneMKL 2026.1 has exactly these overloads with an fp32 accumulator
/// (oneapi/mkl/blas/usm_decls.hpp:38-47: `ONEMKL_DECLARE_GEMM(sycl::half, sycl::half, float, float)` and
/// `(bfloat16, bfloat16, float, float)`), so the mapping is a dtype change on the pointers and nothing else.
///
/// MEASURED (probe/probe27_mkl_gemm.cpp, M4, card 0, B70 driver 1.15.37833+4): all four arms agree with a
/// double-precision CPU reference at the engine's own shapes - worst relative error 1.0e-6 (bf16 T=4096
/// N=320 K=10240), i.e. fp32-accumulate noise, not a 16-bit-input-rounded result.  0 refused, 0 throws.
///
/// WHAT IS **NOT** PRESERVED, AND WHY IT CANNOT BE (PLAN.md §1.1, Risk 9): `cublasSetWorkspace` has no
/// oneMKL equivalent.  oneMKL BLAS has no user-supplied scratch argument at all in 2026.1 - only LAPACK does
/// (`oneapi/mkl/lapack/scratchpad.hpp`); the gemm overload set above takes dependencies, not a workspace.  So
/// the handle still RECORDS the engine's workspace (`cublasSetWorkspace`/`rebind` are unchanged, and the
/// engine's caller-owned-buffer contract still holds), but oneMKL allocates and manages its own internal
/// scratch and ignores ours.  The engine's 32 MiB workspace (prefill.cpp:436) is therefore reserved-but-unused
/// under this backend - a real deviation from PLAN §4.1.6, reported rather than papered over.
inline cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n,
                                   int k, const void* alpha, const void* A, cudaDataType_t typeA, int lda,
                                   const void* B, cudaDataType_t typeB, int ldb, const void* beta, void* C,
                                   cudaDataType_t typeC, int ldc, cublasComputeType_t compute,
                                   cublasGemmAlgo_t /*algo*/) {
    if (h == nullptr || h->q == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
#if STRATA_SYCL_HAVE_ONEMKL
    using namespace strata::sycl_compat::detail;
    try {
        if (typeC != CUDA_R_32F || compute != CUBLAS_COMPUTE_32F) {
            std::fprintf(stderr,
                         "strata/sycl: cublasGemmEx with C=%d compute=%d is not wired to oneMKL: the engine's "
                         "prefill GEMMs always accumulate in fp32 (gemm.cu:388,407).\n",
                         (int) typeC, (int) compute);
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        if (typeA == CUDA_R_32F && typeB == CUDA_R_32F) {
            oneapi::mkl::blas::column_major::gemm(*h->q, to_mkl(opA), to_mkl(opB), m, n, k,
                                                  *(const float*) alpha, (const float*) A, lda,
                                                  (const float*) B, ldb, *(const float*) beta, (float*) C, ldc);
            return CUBLAS_STATUS_SUCCESS;
        }
        if (typeA == CUDA_R_16BF && typeB == CUDA_R_16BF) {
            oneapi::mkl::blas::column_major::gemm(
                *h->q, to_mkl(opA), to_mkl(opB), m, n, k, *(const float*) alpha,
                reinterpret_cast<const oneapi::mkl::bfloat16*>(A), lda,
                reinterpret_cast<const oneapi::mkl::bfloat16*>(B), ldb, *(const float*) beta, (float*) C, ldc);
            return CUBLAS_STATUS_SUCCESS;
        }
        if (typeA == CUDA_R_16F && typeB == CUDA_R_16F) {
            oneapi::mkl::blas::column_major::gemm(
                *h->q, to_mkl(opA), to_mkl(opB), m, n, k, *(const float*) alpha,
                reinterpret_cast<const sycl::half*>(A), lda, reinterpret_cast<const sycl::half*>(B), ldb,
                *(const float*) beta, (float*) C, ldc);
            return CUBLAS_STATUS_SUCCESS;
        }
        std::fprintf(stderr,
                     "strata/sycl: cublasGemmEx with A=%d B=%d C=%d compute=%d has no oneMKL overload (M4 wired "
                     "f32/f32, bf16/bf16 and f16/f16 into fp32; anything else is refused rather than guessed).\n",
                     (int) typeA, (int) typeB, (int) typeC, (int) compute);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata/sycl: oneMKL gemm failed: %s\n", e.what());
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
#else
    (void) opA; (void) opB; (void) m; (void) n; (void) k; (void) alpha; (void) A; (void) typeA; (void) lda;
    (void) B; (void) typeB; (void) ldb; (void) beta; (void) C; (void) typeC; (void) ldc; (void) compute;
    std::fprintf(stderr, "strata/sycl: built without oneMKL headers; cublasGemmEx is unavailable\n");
#endif
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
