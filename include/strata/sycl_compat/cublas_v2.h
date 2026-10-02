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
// STATUS: the names, the enums and the handle are complete and the f32 case is a real oneMKL call; the
// bf16/f16 and CUBLAS_COMPUTE_32F cases are M4 work (PLAN.md M4, Route S1) and return
// CUBLAS_STATUS_NOT_SUPPORTED with a printed reason rather than a silently wrong number.  Nothing in M1
// compiles this header: strata_prefill is not part of the M1 target set.
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
inline cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB, int m, int n,
                                   int k, const void* alpha, const void* A, cudaDataType_t typeA, int lda,
                                   const void* B, cudaDataType_t typeB, int ldb, const void* beta, void* C,
                                   cudaDataType_t typeC, int ldc, cublasComputeType_t compute,
                                   cublasGemmAlgo_t /*algo*/) {
    if (h == nullptr || h->q == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
#if STRATA_SYCL_HAVE_ONEMKL
    using namespace strata::sycl_compat::detail;
    try {
        if (typeA == CUDA_R_32F && typeB == CUDA_R_32F && typeC == CUDA_R_32F && compute == CUBLAS_COMPUTE_32F) {
            oneapi::mkl::blas::column_major::gemm(*h->q, to_mkl(opA), to_mkl(opB), m, n, k,
                                                  *(const float*) alpha, (const float*) A, lda,
                                                  (const float*) B, ldb, *(const float*) beta, (float*) C, ldc);
            return CUBLAS_STATUS_SUCCESS;
        }
        std::fprintf(stderr,
                     "strata/sycl: cublasGemmEx with A=%d B=%d C=%d compute=%d is not wired to oneMKL yet "
                     "(M4, PLAN.md Route S1); the engine's own FP32/FP16 prefill kernels are the fallback.\n",
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
