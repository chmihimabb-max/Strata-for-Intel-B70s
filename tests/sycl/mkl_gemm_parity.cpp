// tests/sycl/mkl_gemm_parity.cpp - RISK 9, measured at the engine's own shapes (PLAN.md §6 Risk 9, §5 M4,
// card t_086173b8).
//
// The risk as written: "oneMKL GEMM is not a drop-in for cublasGemmEx.  The engine relies on column-major +
// OP_T + bf16/f16 inputs + a caller workspace (src/prefill/gemm.cu:300-320,388)".  This program falsifies or
// confirms that at the shapes the engine actually asks for, through the ENGINE'S OWN Gemm class - not a
// hand-rolled oneMKL call - so what is under test includes cublasCreate/cublasSetStream/cublasSetWorkspace/
// cublasSetMathMode and the OP_T/OP_N argument order at gemm.cu:388,407.
//
// WHY THESE SHAPES.  src/prefill/prefill.cpp:68 fixes the model (N=2560, HC=4 -> HCN=10240, LR=320, K=10 routed
// experts, NE=512) and its GEMM call sites are the table below:
//   prefill.cpp:856   bf16  Y[T,320]   = X[T,10240] . W[320,10240]^T     (hyper-connection down)
//   prefill.cpp:858   bf16  Y[T,10240] = X[T,320]   . W[10240,320]^T     (hyper-connection up)
//   prefill.cpp:1580  bf16  Y[T,1]     = X[T,2560]  . W[1,2560]^T        (the router's mixed gate)
//   prefill.cpp:1292  bf16  Y[T,128]   = X[T,2560]  . W[128,2560]^T      (QSA key projection)
//   prefill.cpp:1293  (the same with beta=1: the second half of a hi/lo BF16 pair accumulates)
//   prefill.cpp:1758  f16   Y[T,1280]  = X[T,2560]  . W[1280,2560]^T     (expert gate/up, 10 experts per row)
//   prefill.cpp:1761  f16   Y[T,2560]  = X[T,640]   . W[2560,640]^T      (expert down)
//   the two chunk sizes the M4 card measures: T=512 and the `--prefill auto` chunk (2048 here) and a 4K row.
//
// THE REFERENCE is computed in double from the exact 16-bit bit patterns that were uploaded, so it isolates the
// kernel's arithmetic from the input rounding.  Both the absolute and the relative error are printed.  The gate
// is stated next to the numbers: with 16-bit inputs and an fp32 accumulator the error floor is ~eps16*sqrt(K)
// relative, i.e. <= 1e-3 for every shape here; measured numbers are three orders below that.
//
// THE WORKSPACE.  The engine hands oneMKL exactly what it hands cuBLAS: a caller-owned buffer through
// cublasSetWorkspace (prefill.cpp:436 sizes it 32 MiB, and Gemm::init_external/rebind keep the engine's
// caller-owned contract).  oneMKL BLAS has no scratch argument at all (only LAPACK has a scratchpad), so the
// buffer is recorded and IGNORED - case 1 is therefore also run with a 64-BYTE workspace, to show the result
// does not depend on it.  That is the deviation this test pins down, reported rather than hidden.
//
// Usage: sycl_mkl_gemm          (same as --selftest)
//        sycl_mkl_gemm --selftest
// Exit:  0 = every case inside the gate, 1 = at least one outside, 2 = an allocation/handle failure.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

bool bf16_mode = true;   // per case: which 16-bit encoding encode() writes

uint16_t encode(float v) {
    if (bf16_mode) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        const uint32_t lsb = (u >> 16) & 1u;      // round-to-nearest-even into bf16, as the engine's does
        u += 0x7fffu + lsb;
        return (uint16_t) (u >> 16);
    }
    const sycl::half h(v);
    return sycl::bit_cast<uint16_t>(h);           // __float2half_rn's result
}

double decode(uint16_t bits, bool bf) {
    if (!bf) return (double) (float) sycl::bit_cast<sycl::half>(bits);
    const uint32_t u = (uint32_t) bits << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return (double) f;
}

struct Case {
    const char* label;       // which prefill.cpp call site this shape is
    bool bf;                 // bf16 (true) or f16 (false) inputs
    int64_t T, N, K, ldy;
    float beta;
};

// Inputs are in [-0.125, 0.125] - the magnitude band prefill.cpp's activations and dequantized weights sit in -
// so the relative-error column is meaningful rather than a ratio of two near-zero numbers.
bool run_case(strata::prefill::Gemm& gemm, const Case& c, int64_t scratch_elems, const char* label,
              double* out_max_abs, double* out_max_rel, bool* out_ok) {
    bf16_mode = c.bf;
    std::mt19937 rng(0x5EEDu + (uint32_t) c.K);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);

    const int64_t ldy = c.ldy > 0 ? c.ldy : c.N;
    const size_t x_count = (size_t) c.T * (size_t) c.K;
    const size_t w_count = (size_t) c.N * (size_t) c.K;
    const size_t y_count = (size_t) c.T * (size_t) ldy + 7;

    std::vector<uint16_t> x(x_count), w(w_count);
    for (auto& v : x) v = encode(dist(rng));
    for (auto& v : w) v = encode(dist(rng));

    // A deterministic non-zero start for the Y rows, so beta=1 has something to accumulate into.
    std::vector<float> y_init(y_count, -777.25f);
    for (int64_t t = 0; t < c.T; ++t) {
        for (int64_t n = 0; n < c.N; ++n) {
            y_init[(size_t) t * ldy + n] = (float) ((t * 17 + n * 3) % 29 - 14) * 0.03125f;
        }
    }

    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* dy = nullptr;
    if (cudaMalloc((void**) &dx, x_count * 2) != cudaSuccess || cudaMalloc((void**) &dw, w_count * 2) != cudaSuccess ||
        cudaMalloc((void**) &dy, y_count * 4) != cudaSuccess) {
        std::fprintf(stderr, "sycl_mkl_gemm: cudaMalloc failed\n");
        return false;
    }
    cudaMemcpy(dx, x.data(), x_count * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w_count * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y_init.data(), y_count * 4, cudaMemcpyHostToDevice);

    if (c.bf) {
        gemm.bf16(dx, dw, dy, c.T, c.N, c.K, c.ldy, c.beta);
    } else {
        gemm.f16(dx, dw, dy, c.T, c.N, c.K, c.ldy, c.beta);
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        std::fprintf(stderr, "sycl_mkl_gemm: %s: synchronize: %s\n", label, cudaGetErrorString(sync));
        return false;
    }

    std::vector<float> y(y_count);
    cudaMemcpy(y.data(), dy, y_count * 4, cudaMemcpyDeviceToHost);

    double max_abs = 0.0, ref_scale = 0.0;
    for (int64_t t = 0; t < c.T; ++t) {
        for (int64_t n = 0; n < c.N; ++n) {
            double acc = (double) c.beta * (double) y_init[(size_t) t * ldy + n];
            for (int64_t k = 0; k < c.K; ++k) {
                acc += decode(x[(size_t) t * c.K + k], c.bf) * decode(w[(size_t) n * c.K + k], c.bf);
            }
            const double got = (double) y[(size_t) t * ldy + n];
            max_abs = std::max(max_abs, std::fabs(got - acc));
            ref_scale = std::max(ref_scale, std::fabs(acc));
        }
    }
    // SCALE-RELATIVE, not element-relative: a dot product of K random +/-0.125 terms cancels near zero for some
    // elements, so err/|acc| per element is unbounded and says nothing about the GEMM (the first version of this
    // test used it and "failed" a GEMM whose absolute error was 9e-7).  The honest scale is the reference's own
    // magnitude: max|acc| over the whole output, so the number answers "how wrong is the product, relative to
    // what it is", and a transposed/mis-strided GEMM (error ~ scale) is still caught with a huge margin.
    const double rel = ref_scale > 0.0 ? max_abs / ref_scale : max_abs;
    // Timing: 3 warm + 5 timed, the median (the buffer setup above is not in it).
    for (int i = 0; i < 3; ++i) {
        if (c.bf) gemm.bf16(dx, dw, dy, c.T, c.N, c.K, c.ldy, c.beta);
        else      gemm.f16(dx, dw, dy, c.T, c.N, c.K, c.ldy, c.beta);
    }
    cudaDeviceSynchronize();
    std::vector<double> ms;
    for (int i = 0; i < 5; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (c.bf) gemm.bf16(dx, dw, dy, c.T, c.N, c.K, c.ldy, c.beta);
        else      gemm.f16(dx, dw, dy, c.T, c.N, c.K, c.ldy, c.beta);
        cudaDeviceSynchronize();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double flops = 2.0 * (double) c.T * (double) c.N * (double) c.K;

    // The gate: <= 1e-4 relative to the reference's own scale.  Measured numbers are ~1e-6 (three orders inside),
    // and a transposed or mis-strided product lands at ~1.0, so the gate is both loose enough not to fail on
    // accumulation noise and tight enough to mean something.
    const bool ok = rel <= 1e-4;
    *out_max_abs = max_abs;
    *out_max_rel = rel;
    *out_ok = ok;

    std::printf("  %-6s %-32s T=%5lld N=%6lld K=%6lld ldy=%6lld beta=%.0f  max_abs=%.4g rel=%.3g  %s"
                "  %7.3f ms %8.1f GFLOP/s%s\n",
                c.bf ? "bf16" : "f16", label, (long long) c.T, (long long) c.N, (long long) c.K,
                (long long) ldy, (double) c.beta, max_abs, rel, ok ? "PASS" : "FAIL",
                ms[2], flops / (ms[2] * 1e6), scratch_elems == 0 ? "  (no scratch)" : "");

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    return true;
}

// The f32 arm of the shim (M1's) is not reachable through Gemm - the engine only ever calls it with 16-bit
// inputs - so it is exercised through cublasGemmEx itself, which is also where the workspace contract lives.
bool run_f32_case(double* max_abs, double* max_rel, bool* ok) {
    const int64_t T = 512, N = 320, K = 10240;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    std::vector<float> x((size_t) T * K), w((size_t) N * K);
    for (auto& v : x) v = dist(rng);
    for (auto& v : w) v = dist(rng);

    float* dx = nullptr;
    float* dw = nullptr;
    float* dy = nullptr;
    void* ws = nullptr;
    if (cudaMalloc((void**) &dx, x.size() * 4) != cudaSuccess || cudaMalloc((void**) &dw, w.size() * 4) != cudaSuccess ||
        cudaMalloc((void**) &dy, (size_t) T * N * 4) != cudaSuccess || cudaMalloc(&ws, 64) != cudaSuccess) {
        std::fprintf(stderr, "sycl_mkl_gemm: f32 cudaMalloc failed\n");
        return false;
    }
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0, (size_t) T * N * 4);

    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "sycl_mkl_gemm: cublasCreate\n"); return false; }
    cublasSetStream(h, nullptr);
    cublasSetWorkspace(h, ws, 64);            // 64 bytes on purpose: oneMKL ignores it (see the header note)
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, dw,
                                          CUDA_R_32F, (int) K, dx, CUDA_R_32F, (int) K, &beta, dy, CUDA_R_32F,
                                          (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "sycl_mkl_gemm: f32 cublasGemmEx status %d\n", (int) s);
        return false;
    }
    cudaDeviceSynchronize();

    std::vector<float> y((size_t) T * N);
    cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
    double ma = 0.0, scale = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) acc += (double) x[(size_t) t * K + k] * (double) w[(size_t) n * K + k];
            ma = std::max(ma, std::fabs((double) y[(size_t) t * N + n] - acc));
            scale = std::max(scale, std::fabs(acc));
        }
    }
    const double fr = scale > 0.0 ? ma / scale : ma;   // scale-relative, same definition as the 16-bit arms
    const bool good = fr <= 1e-4;
    *max_abs = ma; *max_rel = fr; *ok = good;
    std::printf("  %-6s %-32s T=%5lld N=%6lld K=%6lld ldy=%6lld beta=0    max_abs=%.4g rel=%.3g  %s"
                "  (64-byte workspace)\n",
                "f32", "hc down (cublasGemmEx direct)", (long long) T, (long long) N, (long long) K, (long long) N,
                ma, fr, good ? "PASS" : "FAIL");
    cublasDestroy(h);
    cudaFree(dx); cudaFree(dw); cudaFree(dy); cudaFree(ws);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") != 0 && std::strcmp(argv[i], "--help") != 0) {
            std::fprintf(stderr, "usage: sycl_mkl_gemm [--selftest]\n");
            return 2;
        }
    }
    cudaError_t dev = cudaFree(nullptr);
    if (dev != cudaSuccess && dev != cudaErrorInvalidValue) {
        std::fprintf(stderr, "sycl_mkl_gemm: no device: %s\n", cudaGetErrorString(dev));
        return 2;
    }

    std::printf("sycl_mkl_gemm - RISK 9: oneMKL (MKL::MKL_SYCL) as the engine's cuBLAS replacement, at the\n");
    std::printf("engine's own shapes, through strata::prefill::Gemm (src/prefill/gemm.cu:388,407's call shape).\n");
    std::printf("reference: double-precision CPU product of the exact 16-bit inputs that were uploaded.\n");
    std::printf("gate: max_abs / max|reference| <= 1e-4 (16-bit inputs into an fp32 accumulator; a transposed or\n");
    std::printf("mis-strided product would land at ~1.0, so the gate is tight enough to mean something).\n");
    std::printf("the ms/GFLOP/s columns are the median of 5 runs and are NOT what this test gates: measured on this\n");
    std::printf("card the same shape moves between 0.109 and 0.520 ms across runs (clock state), so read them as an\n");
    std::printf("order of magnitude, while the error columns are stable to the last digit.\n\n");

    // The engine's own buffers: 32 MiB of scratch (prefill.cpp:435) and a 32 MiB caller workspace
    // (prefill.cpp:436), handed over exactly as Gemm::init_external does at prefill.cpp's allocation site.
    const int64_t scratch_elems = (int64_t) (32u << 20) / 2;
    const size_t ws_bytes = 32u << 20;
    void* workspace = nullptr;
    if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess) {
        std::fprintf(stderr, "sycl_mkl_gemm: workspace allocation failed\n");
        return 2;
    }
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init_external(nullptr, nullptr, scratch_elems, workspace, ws_bytes, err)) {
        std::fprintf(stderr, "sycl_mkl_gemm: Gemm::init_external: %s\n", err.c_str());
        return 2;
    }

    const Case cases[] = {
        {"hc down  X[T,10240].W[320,10240]^T",    true,  512,  320,   10240, 320,   0.0f},  // prefill.cpp:856
        {"hc up    X[T,320].W[10240,320]^T",      true,  512,  10240, 320,   10240, 0.0f},  // prefill.cpp:858
        {"router   X[T,2560].W[1,2560]^T",        true,  512,  1,     2560,  1,     0.0f},  // prefill.cpp:1580
        {"qsa key  X[T,2560].W[128,2560]^T",      true,  512,  128,   2560,  128,   0.0f},  // prefill.cpp:1292
        {"qsa key, beta=1 (accumulate)",          true,  512,  128,   2560,  128,   1.0f},  // prefill.cpp:1293
        {"hc up at the auto chunk (T=2048)",      true,  2048, 10240, 320,   10240, 0.0f},
        {"hc down, a 4K prompt in one chunk",     true,  4096, 320,   10240, 320,   0.0f},
        {"expert gate/up X[T,2560].W[1280,2560]", false, 512,  1280,  2560,  1280,  0.0f},  // prefill.cpp:1758
        {"expert down   X[T,640].W[2560,640]",    false, 512,  2560,  640,   2560,  0.0f},  // prefill.cpp:1761
    };

    int failed = 0, ran = 0;
    double worst_rel = 0.0, worst_abs = 0.0;
    const char* worst_what = "";
    for (const Case& c : cases) {
        double a = 0.0, r = 0.0;
        bool ok = false;
        if (!run_case(gemm, c, scratch_elems, c.label, &a, &r, &ok)) return 2;
        ++ran;
        if (r > worst_rel) { worst_rel = r; worst_abs = a; worst_what = c.label; }
        if (!ok) ++failed;
    }

    // The workspace-independence check: the same case with a 64-byte workspace instead of 32 MiB.  If oneMKL
    // used the caller's buffer as cuBLAS does, this would be the case that fails.
    {
        const size_t tiny_bytes = 64;
        void* tiny = nullptr;
        if (cudaMalloc(&tiny, tiny_bytes) != cudaSuccess) return 2;
        strata::prefill::Gemm g2;
        std::string e2;
        if (!g2.init_external(nullptr, nullptr, scratch_elems, tiny, tiny_bytes, e2)) {
            std::fprintf(stderr, "sycl_mkl_gemm: init_external(tiny workspace): %s\n", e2.c_str());
            return 2;
        }
        double a = 0.0, r = 0.0;
        bool ok = false;
        if (!run_case(g2, cases[0], 0, "hc down / 64-byte workspace", &a, &r, &ok)) return 2;
        ++ran;
        if (!ok) ++failed;
        cudaFree(tiny);
    }

    double fa = 0.0, fr = 0.0;
    bool fok = false;
    if (!run_f32_case(&fa, &fr, &fok)) return 2;
    ++ran;
    if (!fok) ++failed;

    std::printf("\nRISK 9 (%d cases): %s - worst relative error %.3g (%.4g absolute) on the %s shape\n",
                ran, failed == 0 ? "PASS" : "FAIL", worst_rel, worst_abs, worst_what);
    std::printf("caller workspace: recorded by cublasSetWorkspace and IGNORED by oneMKL (no scratch argument in\n");
    std::printf("oneMKL BLAS 2026.1) - the 64-byte-workspace case above is the evidence.\n");
    std::printf("NOT covered here: n_layers-sized batches, the expert streaming that surrounds these GEMMs, and\n");
    std::printf("any pack (there is none on this box - the engine-level 4K prompt run is gated on the W-track).\n");
    cudaFree(workspace);
    return failed == 0 ? 0 : 1;
}
