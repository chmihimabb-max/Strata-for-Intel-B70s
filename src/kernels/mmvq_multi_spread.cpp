// src/kernels/mmvq_multi_spread.cpp - is the generic multi-column MMVQ layout racy, or only misplaced in time?
//
//     build-sycl/mmvq_multi_spread            (the default: 16 reps per shape/T/layout)
//     build-sycl/mmvq_multi_spread --reps 32
//
// WHY IT EXISTS (card t_8306429a, D2a).  D2 measured llama.cpp's generic multi-column layout for the 300 dense
// native projections (`native_mmvq_set_multi_exact(false)`, the `STRATA_MMVQ_MULTI_GENERIC=1` switch) and got
// three different greedy token streams and a 120.68-191.79 ms/window window cost from three identical runs of
// the same binary and prompt (125-126 ms/window and the same stream in all eight shipped-layout runs).  Two
// readings fit that: the generic kernel is racy (run-to-run nondeterminism on the device), or its cost is a
// function of something outside the device (first-launch program build, and the engine's own adaptivity).
//
// WHAT IT MEASURES, per (type, n_in, n_out) of this config and per column count T:
//   1. the FIRST launch of each specialization's own wall and its own event-measured device time - on a cold
//      SYCL cache that first launch carries the JIT program build, on a warm one it does not;
//   2. `--reps` further launches, event-timed, reported as min/median/max/spread - the kernel's own spread with
//      the program built;
//   3. a bitwise FNV-1a hash of every rep's whole output: whether identical inputs give identical bytes, rep
//      after rep, for each layout.  A race in the kernel shows up here; a different summation order does not
//      (it is stable and shows up instead as the generic-vs-exact difference count, column 5).
// THE DATA is the parity harness's (random bytes, finite fp16 scales) so this file adds no new assumption about
// the quant formats; the shapes are the config of record's own (n_in 2560, n_out 10240/6144; D2-TYPES.txt).
//
// HOW TO READ THE OUTPUT.  `first` is the launch that may carry a build; `min/med/max` are the reps after it.
// A spread of ~0 there plus `same` = yes for every rep means the kernel is deterministic in value and in time,
// and anything the engine saw beyond that came from outside the kernel.  `d(gen-exact)` counts the output words
// the two layouts disagree on, bit for bit (T <= 4: 0, they coincide; T > 4: nonzero, the reduction groups
// differently - `NW = NCOLS <= 4 ? 4 : 2` with `WARPS = 4`, native_mmvq.cu:1050).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

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

using Clock = std::chrono::steady_clock;

struct Case {
    const char* name;
    int type;          // stable GGML type id (native_mmvq.hpp)
    int n_in;
    int n_out;
    int block_elems;
    int block_bytes;
    int scale_at[2];   // byte offsets of the block's fp16 scales; -1 = none
};

// the dense native projection shapes of the config of record (d2/D2-TYPES.txt: [2560, 6144] attn/ssm, and the
// 10240-row shared-expert / ffn ones), one case per quant type the 300 dense matrices use
const Case CASES[] = {
    {"Q6_K 2560x10240",  14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 2560x6144",   14, 2560,  6144, 256, 210, {208, -1}},
    {"Q5_K 2560x6144",   13, 2560,  6144, 256, 176, {0, 2}},
    {"Q4_K 2560x6144",   12, 2560,  6144, 256, 144, {0, 2}},
    {"IQ4_XS 2560x6144", 23, 2560,  6144, 256, 136, {0, -1}},
    {"IQ4_NL 2560x10240", 20, 2560, 10240,  32,  18, {0, -1}},
};
// T = 4 is the widest window whose layout coincides with the exact one, T = 6 is one the engine really uses
// (all four D2 4K arms of this config had a T = 6 window; the generic runs' T = 5/6 windows are the first
// windows whose numbers can differ from the shipped layout's at all).
const int TS[] = {4, 6};

// a normal fp16 in +-[2^-10, 2^-5): random bytes in a scale field would put inf/NaN in the outputs
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

uint64_t fnv(const void* p, std::size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct Stat {
    double first_wall = 0, first_evt = 0;
    double mn = 0, med = 0, mx = 0;
    bool same = true, ran = false;
    uint32_t gen_exact_diff = 0;
};

// One layout, one case, one T: the first launch, `reps` timed launches, and the hash of every output.
bool run_layout(const Case& c, int T, bool exact, int reps, cudaStream_t s,
                const void* dw, const void* xq, float* dy, std::vector<uint32_t>& out, Stat& st) {
    st.ran = false;
    strata::kernels::native_mmvq_set_multi_exact(exact);
    const std::size_t ywords = (std::size_t) T * (std::size_t) c.n_out;

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (!ck(cudaEventCreate(&e0), "event") || !ck(cudaEventCreate(&e1), "event")) return false;

    // 1. the first launch of this specialization: host wall + event delta (a program build lands in both)
    if (cudaEventRecord(e0, s) != cudaSuccess) return false;
    const auto w0 = Clock::now();
    try {
        strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
    } catch (const std::exception& e) {
        std::printf("%s T=%d: %s\n", c.name, T, e.what());
        return false;
    }
    st.first_wall = std::chrono::duration<double, std::milli>(Clock::now() - w0).count();
    if (cudaEventRecord(e1, s) != cudaSuccess) return false;
    if (!ck(cudaStreamSynchronize(s), "first sync")) return false;
    float first_ms = 0.f;
    cudaEventElapsedTime(&first_ms, e0, e1);
    st.first_evt = first_ms;

    // 2. `reps` launches, each bracketed by its own event pair and synced individually, so every number is that
    // kernel execution's own device delta - the kernel's own spread with the program already built
    std::vector<float> ms((std::size_t) reps);
    for (int r = 0; r < reps; ++r) {
        if (cudaEventRecord(e0, s) != cudaSuccess) return false;
        strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
        if (cudaEventRecord(e1, s) != cudaSuccess) return false;
        if (!ck(cudaStreamSynchronize(s), "sync one rep")) return false;
        cudaEventElapsedTime(&ms[(std::size_t) r], e0, e1);
    }
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    st.mn = sorted.front();
    st.mx = sorted.back();
    st.med = (reps % 2) ? sorted[sorted.size() / 2]
                        : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);

    // 3. the bytes: one untimed launch per rep, copied back and hashed (bitwise, whole output)
    std::vector<uint32_t> host(ywords);
    uint64_t h0 = 0;
    for (int r = 0; r < reps; ++r) {
        strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
        if (!ck(cudaStreamSynchronize(s), "sync hash rep")) return false;
        if (!ck(cudaMemcpy(host.data(), dy, ywords * 4, cudaMemcpyDeviceToHost), "read y")) return false;
        const uint64_t h = fnv(host.data(), ywords * 4);
        if (r == 0) {
            h0 = h;
            out = host;
        } else if (h != h0) {
            st.same = false;
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    st.ran = true;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 16;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    if (reps < 1) reps = 1;

    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::printf("device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    const char* cache = std::getenv("SYCL_CACHE_DIR");
    std::printf("reps %d; SYCL_CACHE_DIR=%s\n", reps, cache ? cache : "<unset: a fresh build every run>");

    std::printf("\n%-18s %3s %8s %10s %8s %8s %8s %8s  %-9s %6s %10s %9s\n", "case", "T", "layout", "first wall",
                "first evt", "min", "med", "max", "reps same", "d(g-e)", "max|d|", "max|d|/rms");
    for (const Case& c : CASES) {
        if (!strata::kernels::native_mmvq_supported(c.type)) {
            std::printf("%-18s unsupported type %d\n", c.name, c.type);
            continue;
        }
        for (int T : TS) {
            const std::size_t wbytes = strata::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
            const std::size_t qall = strata::kernels::native_q8_1_bytes(c.n_in, T);
            std::mt19937 rng(1234u + (unsigned) c.type * 97u + (unsigned) T);
            std::vector<uint8_t> w(wbytes);
            for (auto& b : w) b = (uint8_t) (rng() & 0xff);
            const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
            for (std::size_t k = 0; k < n_blocks; ++k)
                for (int at : c.scale_at)
                    if (at >= 0) {
                        const uint16_t h = sane_half(rng);
                        std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
                    }
            std::vector<float> x((std::size_t) T * c.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : x) v = nd(rng);

            void* dw = nullptr;
            void* xq = nullptr;
            float* dx = nullptr;
            float* dy = nullptr;
            if (!ck(cudaMalloc(&dw, wbytes), "malloc w") || !ck(cudaMalloc(&xq, qall), "malloc xq") ||
                !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") || !ck(cudaMalloc(&dy, (std::size_t) T * c.n_out * 4), "malloc y"))
                return 1;
            if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
                !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
                return 1;
            strata::kernels::quantize_q8_1_rows(dx, T, c.n_in, xq, s);
            if (!ck(cudaStreamSynchronize(s), "quantize")) return 1;

            Stat se, sg;
            std::vector<uint32_t> ye, yg;
            if (!run_layout(c, T, true, reps, s, dw, xq, dy, ye, se)) return 1;
            if (!run_layout(c, T, false, reps, s, dw, xq, dy, yg, sg)) return 1;
            strata::kernels::native_mmvq_set_multi_exact(true);

            uint32_t diff = 0;
            double maxabs = 0.0, sumsq = 0.0;
            for (std::size_t i = 0; i < ye.size() && i < yg.size(); ++i) {
                float fa, fb;
                std::memcpy(&fa, &ye[i], 4);
                std::memcpy(&fb, &yg[i], 4);
                if (ye[i] != yg[i]) ++diff;
                const double d = std::fabs((double) fa - (double) fb);
                if (d > maxabs) maxabs = d;
                sumsq += (double) fb * (double) fb;
            }
            const double rms = std::sqrt(sumsq / (double) (ye.empty() ? 1 : ye.size()));
            se.gen_exact_diff = diff;
            sg.gen_exact_diff = diff;
            for (const Stat* st : {&se, &sg}) {
                std::printf("%-18s %3d %8s %9.2fms %8.3fms %7.3f %7.3f %7.3f %8s   %6u  %.3e %8.1e\n", c.name, T,
                            st == &se ? "exact" : "generic", st->first_wall, st->first_evt, st->mn, st->med, st->mx,
                            st->same ? "yes" : "NO", st == &sg ? diff : 0u, st == &sg ? maxabs : 0.0,
                            st == &sg && rms > 0 ? maxabs / rms : 0.0);
            }
            cudaFree(dw);
            cudaFree(xq);
            cudaFree(dx);
            cudaFree(dy);
        }
    }
    strata::kernels::native_mmvq_set_multi_exact(true);
    cudaStreamDestroy(s);
    std::printf("\nlegend: first wall/evt = the specialization's FIRST launch (cold SYCL cache: it carries the "
                "program build); min/med/max = the %d launches after it, event-measured; reps same = every "
                "rep's output hash equal; d(g-e) = output words the generic layout and the exact layout "
                "disagree on bit for bit; max|d| = the largest of those differences in absolute terms, and as a "
                "fraction of the exact layout's output RMS.\n", reps);
    return 0;
}
