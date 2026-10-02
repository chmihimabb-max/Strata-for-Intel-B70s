// tests/sycl/prefill_chunk_bench.cpp - the M4 prompt-path measurement that does NOT need a model pack
// (PLAN.md §5 M4, card t_086173b8), plus the MMQ refusal check the card requires.
//
// WHY THIS EXISTS.  M4's acceptance names `./build-sycl/strata --pack ... --native ... --prefill 512 ...
// --stats`, and there is NO PACK on this box: the UD-Q4_K_XL GGUF P0-pack was going to pack was deleted by Mike
// (PLAN §11 U8), the W4A16 checkpoint needs the W-track's packer (W1/W2, not landed), and the Q2_0 GSQ-RCO
// shard the older parity tests use is absent.  So the engine-level prompt speed cannot be produced here, and
// this program measures the part that CAN be: the GEMM work one 4K prompt's chunks actually issue, through the
// engine's own strata::prefill::Gemm class at the shapes its call sites use, at chunk 512 and at the
// `--prefill auto` chunk.
//
// WHAT THIS NUMBER IS AND IS NOT.
//   IT IS: the sum of the per-chunk GEMM time for the prompt path's projection/expert GEMMs, i.e. an UPPER
//          BOUND on the prompt speed the oneMKL route can reach through these calls (nothing else can make it
//          faster), measured on card 0 at the engine's real shapes.
//   IT IS NOT: the engine's end-to-end prompt speed.  Excluded, and named here rather than buried: the expert
//          staging the prompt path does around these GEMMs (host->device blobs are the measured bottleneck on
//          this box - docs/UNSLOTH_Q4.md:159-161), the QSA attention/indexer, the GDN recurrence, the PLE
//          table, the KV writes, and the dequantization into the f16 scratch (the dequant kernel itself is
//          M2's ported dequant.cu and is not in these timings).  The one weight buffer per call shape is REUSED
//          for all 48 layers, where the engine streams the real per-layer weights; that is a timing stand-in,
//          and it is the reason this is a route measurement, not a model measurement.
//
// THE CALL LIST comes from src/prefill/prefill.cpp's own GEMM sites (N=2560, HCN=10240, LR=320, K=10 routed
// experts per token, 48 layers - prefill.cpp:68):
//   prefill.cpp:856   bf16  Y[T,320]       = X[T,10240] . W[320,10240]^T
//   prefill.cpp:858   bf16  Y[T,10240]     = X[T,320]   . W[10240,320]^T
//   prefill.cpp:1580  bf16  Y[T,1]         = X[T,2560]  . W[1,2560]^T
//   prefill.cpp:1292  bf16  Y[T,128]       = X[T,2560]  . W[128,2560]^T        (QSA key)
//   prefill.cpp:1299  bf16  Y[T,2560]      = X[T,2560]  . W[2560,2560]^T       (QSA value)
//   prefill.cpp:1758  f16   Y[T*10,1280]   = X[T*10,2560] . W[1280,2560]^T     (expert gate/up)
//   prefill.cpp:1761  f16   Y[T*10,2560]   = X[T*10,640]  . W[2560,640]^T      (expert down)
// Usage: sycl_prefill [--selftest]
// Exit:  0 = the MMQ refusal is clean and the measurement ran, 2 = an allocation/handle failure.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t N = 2560, HC = 4, HCN = N * HC, LR = 320, KEXP = 10, LAYERS = 48, PROMPT = 4096;

struct Mat {
    const char* what;
    bool bf;
    int64_t rows;          // output rows == N
    int64_t cols;          // K
    int64_t row_scale;     // output rows scale with T (experts: KEXP)
    int64_t ldy;
    uint16_t* w = nullptr;
    uint16_t* x = nullptr;
    float* y = nullptr;
};

std::mt19937 g_rng(0xC0FFEEu);
uint16_t encode(bool bf, float v) {
    if (bf) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        return (uint16_t) (u >> 16);
    }
    return sycl::bit_cast<uint16_t>(sycl::half(v));
}

bool alloc_mat(Mat& m, int64_t max_tokens) {
    const int64_t t = max_tokens * m.row_scale;
    if (cudaMalloc((void**) &m.w, (size_t) m.rows * m.cols * 2) != cudaSuccess) return false;
    if (cudaMalloc((void**) &m.x, (size_t) t * m.cols * 2) != cudaSuccess) return false;
    if (cudaMalloc((void**) &m.y, (size_t) t * m.ldy * 4) != cudaSuccess) return false;
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    std::vector<uint16_t> hw((size_t) m.rows * m.cols), hx((size_t) t * m.cols);
    for (auto& v : hw) v = encode(m.bf, dist(g_rng));
    for (auto& v : hx) v = encode(m.bf, dist(g_rng));
    cudaMemcpy(m.w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(m.x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    return true;
}

// One chunk of `T` tokens through all 48 layers: the same calls, in the same order, that prefill.cpp:856-858,
// :1292-1299 and :1758-1761 issue per layer.
void run_chunk(strata::prefill::Gemm& gemm, const Mat& hc_down, const Mat& hc_up, const Mat& router,
               const Mat& qsa_key, const Mat& qsa_val, const Mat& exp_gu, const Mat& exp_dn, int64_t T) {
    for (int64_t l = 0; l < LAYERS; ++l) {
        gemm.bf16(hc_down.x, hc_down.w, hc_down.y, T, hc_down.rows, hc_down.cols, hc_down.ldy);
        gemm.bf16(hc_up.x, hc_up.w, hc_up.y, T, hc_up.rows, hc_up.cols, hc_up.ldy);
        gemm.bf16(router.x, router.w, router.y, T, router.rows, router.cols, router.ldy);
        gemm.bf16(qsa_key.x, qsa_key.w, qsa_key.y, T, qsa_key.rows, qsa_key.cols, qsa_key.ldy);
        gemm.bf16(qsa_val.x, qsa_val.w, qsa_val.y, T, qsa_val.rows, qsa_val.cols, qsa_val.ldy);
        gemm.f16(exp_gu.x, exp_gu.w, exp_gu.y, T * KEXP, exp_gu.rows, exp_gu.cols, exp_gu.ldy);
        gemm.f16(exp_dn.x, exp_dn.w, exp_dn.y, T * KEXP, exp_dn.rows, exp_dn.cols, exp_dn.ldy);
    }
}

double median_ms(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") != 0 && std::strcmp(argv[i], "--help") != 0) {
            std::fprintf(stderr, "usage: sycl_prefill [--selftest]\n");
            return 2;
        }
    }

    // ---- the MMQ refusal (PLAN §4.2 Route S2; card: "must ... refuse cleanly if the MMQ path is off") ------
    std::printf("MMQ path (Route S2, the STRATA_SYCL_PREFILL_MMQ switch):\n");
    std::printf("  mmq::built() with no switch        = %s\n", strata::prefill::mmq::built() ? "true" : "false");
    setenv("STRATA_PREFILL_MMQ", "1", 1);
    const bool mmq_built = strata::prefill::mmq::built();
    std::printf("  mmq::built() with STRATA_PREFILL_MMQ=1 = %s (the refusal above is stderr, and it is the point)\n",
                mmq_built ? "true" : "false");
    if (mmq_built) {
        std::fprintf(stderr, "sycl_prefill: this build claims an MMQ path; the SYCL build must not\n");
        return 1;
    }
    unsetenv("STRATA_PREFILL_MMQ");

    // ---- the Route S1 measurement ------------------------------------------------------------------------
    const size_t ws_bytes = 32u << 20;
    void* workspace = nullptr;
    if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess) return 2;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init_external(nullptr, nullptr, (int64_t) (32u << 20) / 2, workspace, ws_bytes, err)) {
        std::fprintf(stderr, "sycl_prefill: Gemm::init_external: %s\n", err.c_str());
        return 2;
    }

    Mat hc_down{"hc down",  true,  320,   HCN,   1,   320};
    Mat hc_up{"hc up",      true,  HCN,   LR,    1,   HCN};
    Mat router{"router",    true,  1,     N,     1,   1};
    Mat qsa_key{"qsa key",  true,  128,   N,     1,   128};
    Mat qsa_val{"qsa val",  true,  N,     N,     1,   N};
    Mat exp_gu{"exp gate/up", false, 1280, N,    KEXP, 1280};
    Mat exp_dn{"exp down",    false, N,    640,  KEXP, N};
    const int64_t max_tokens = 8192;   // the largest chunk measured below: every buffer is sized for it
    for (Mat* m : {&hc_down, &hc_up, &router, &qsa_key, &qsa_val, &exp_gu, &exp_dn}) {
        if (!alloc_mat(*m, max_tokens)) {
            std::fprintf(stderr, "sycl_prefill: allocation for %s failed\n", m->what);
            return 2;
        }
    }

    std::printf("\nRoute S1 (oneMKL) GEMM route for a %lld-token prompt, card 0, weights reused across the %lld "
                "layers:\n", (long long) PROMPT, (long long) LAYERS);
    std::printf("  chunk   chunks   per-chunk min / median ms   prompt ms (min)   prompt tok/s (min / median)\n");
    for (int64_t chunk : {(int64_t) 512, (int64_t) 2048, (int64_t) 8192}) {
        // The engine never processes more tokens in a chunk than the prompt has: a 8192-token chunk on a 4096
        // token prompt is ONE chunk of 4096.  Sizing the work that way is what makes the three rows comparable.
        const int64_t t_chunk = chunk < PROMPT ? chunk : PROMPT;
        std::vector<double> per_chunk;
        for (int i = 0; i < 7; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            run_chunk(gemm, hc_down, hc_up, router, qsa_key, qsa_val, exp_gu, exp_dn, t_chunk);
            cudaDeviceSynchronize();
            per_chunk.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const double best = *std::min_element(per_chunk.begin(), per_chunk.end());
        const double ms = median_ms(per_chunk);
        const int64_t chunks = (PROMPT + chunk - 1) / chunk;
        // min is quoted as the headline because this card's clock state moves a burst of GEMMs by up to ~4x
        // between runs (measured: one shape at 0.109 and 0.520 ms in the same binary) - the least-throttled
        // repeat is the honest ceiling, and the median is printed next to it so the spread is visible.
        const double best_total = best * (double) chunks, med_total = ms * (double) chunks;
        std::printf("  %5lld   %6lld   %10.2f / %9.2f   %13.2f   %8.1f / %8.1f\n", (long long) chunk,
                    (long long) chunks, best, ms, best_total, (double) PROMPT / (best_total / 1000.0),
                    (double) PROMPT / (med_total / 1000.0));
    }
    std::printf("NOT the engine's end-to-end prompt speed: expert staging (the measured bottleneck on this box),\n");
    std::printf("QSA attention/indexer, GDN recurrence, PLE, KV writes and the dequant scratch are all excluded,\n");
    std::printf("and there is no pack on this box to run the real thing (PLAN §11 U8 / the W-track).  Chunk sizes:\n");
    std::printf("512 and 2048 are what the engine's --prefill names, and 8192 is its --prefill auto ceiling\n");
    std::printf("(prefill.cpp:332 prefill_auto_max); at 4096 tokens an 8192 chunk is the whole prompt in one shot,\n");
    std::printf("which is why its row is one chunk.  See this file's header for the full statement.\n");

    for (Mat* m : {&hc_down, &hc_up, &router, &qsa_key, &qsa_val, &exp_gu, &exp_dn}) {
        cudaFree(m->w); cudaFree(m->x); cudaFree(m->y);
    }
    cudaFree(workspace);
    return 0;
}
