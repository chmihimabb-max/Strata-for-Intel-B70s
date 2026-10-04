// bench/micro/mmvq_xmx_price.cpp - D2x (card t_85e61269): price an XMX ("matrix unit") MMVQ against the
// shipped scalar MMVQ on this config's own decode shapes.
//
// WHY IT EXISTS.  D2's corrected census (strata/d2/STATUS-D2.md section 4) puts the dense native projection
// MMVQ -- `native_mmvq` -> per-type `native_q{4,5,6}_k_mmvq` / `native_iq4_xs_mmvq` / `native_iq4_nl_mmvq`
// / `native_q8_0_mmvq` -> `launch_multi_n<Traits, 4>` (src/kernels/cuda/native_mmvq.cu:1046) -- at 301
// launches / 51.27 ms / 40.9% of a T=4 decode window's counted device time.  That kernel is a scalar FMA
// integer dot product: int8 Q8_1 activations against the pack's quantized blocks.  The card asked what the
// device's matrix unit (XMX / DPAS) could buy that family, and named it "int8 XMX (joint_matrix int8
// 8x32x16 -> int32)".  This program answers with measurements, in the order the questions must be asked:
//
//   1. DOES THE UNIT EXIST FOR THIS DOT PRODUCT?  include/strata/sycl_compat/sycl_xmx.hpp records probe21's
//      table (plan-evidence/M3-risk1-shapes.txt): on Intel(R) Graphics [0xe223] (Arc Pro B70, BMG-G31) the
//      f16/bf16 16x16x16 tiles RUN and the int8 8x32x16 / 16x32x16 tiles are REFUSED by the driver.  That
//      table lives in a header comment; section 1 re-runs the tiles at HEAD and prints the driver's own
//      words, so the premise is measured here rather than inherited.
//
//   2. HOW FAST IS THE SHIPPED KERNEL, PER SHAPE AND TYPE?  Section 3 times (a) the shipped default
//      multi-column layout and (b) llama.cpp's generic multi-column layout
//      (`native_mmvq_set_multi_exact(false)`) on all 25 distinct (type, n_in, n_out) geometries this pack's
//      301 dense matrices use, at ncols = 4 and 6.  Each row reports the call's own GB/s = weight bytes /
//      time, which is the number a "put it on the matrix unit" proposal has to beat.
//
//   3. WHAT IS THE FLOOR?  Section 2 measures the card's streaming read bandwidth.  A GEMV reads every
//      weight byte exactly once, so floor_us = weight_bytes / measured_bandwidth is a hard lower bound on
//      any MMVQ of any kind, on any unit.  The measurement needs the incompressible fill in section 2's last
//      arm: a memset-filled buffer reads at 2.37x the hash-filled one on this driver (1352 against 571 GB/s,
//      1 GiB), i.e. uniform pages are compressed and a constant-filled buffer is not a bandwidth measurement.
//      Quantized weights are incompressible, so the floor uses the hash-filled number.
//
//   4. WHAT DOES THE DEVICE'S ACTUAL MATRIX UNIT BUY?  The dot product has to be expressed in an encoding
//      the unit accepts, and section 1 measures which encodings this driver accepts: the int8 dot is refused
//      outright and the f16/bf16 16x16x16 tiles that probe21 recorded as running no longer build, so the only
//      tile left for a GEMV is f16 8x16x16 -> fp32.  Section 3 dequantizes each case's real blocks with the
//      tree's own host dequantizers (include/strata/artifact/dequant.hpp), so the prototype reads REAL weights
//      in the unit's own 2-byte encoding, and pays for the variant the way the engine would: bytes in, at the
//      card's rate.  The prototype's contract, exactly: weights dequantized and repacked into 8x16 tiles once
//      (the load-time cost this program measures per case), stored f16; activations f16; 8x16x16 tiles into an
//      fp32 accumulator; k split across NSG sub-groups, reduced in work-group local memory; four 8-row tiles
//      (32 output rows) per work-group, so the activation tile is loaded once per k-step for all four.  The
//      activation tile's 16 columns are not all real: at ncols = 4 only 4 of them carry a token (the rest
//      repeat column 0), i.e. the shape the unit forces is 4x wider than this config's window, and the tile
//      pays for that in MACs, never in invented bytes.  The store writes only the real columns.
//
//   5. WHAT DOES IT COST IN ACCURACY?  Section 4 checks both kernels against an fp64 dot of the same
//      dequantized weights on the same activations (the port's parity style: native_expert_parity,
//      s2_gemv_q8_parity), and separates the prototype's own arithmetic error from the f16 encoding error.
//
//   6. WHAT WOULD IT BE WORTH?  Section 5 extrapolates every variant to a whole window's 301 calls
//      (case count x per-call us) and compares against the census's own 51.267 ms.
//
// NO ENGINE DEFAULT CHANGES AND NOTHING HERE IS READ BY THE ENGINE.  The one engine switch this program
// touches is `native_mmvq_set_multi_exact`, and it restores the default (`true`) before returning.
//
// Build/run (from the repo root, after `source /opt/intel/oneapi/setvars.sh`):
//   cmake --build build-sycl --target mmvq_xmx_price
//   ZE_AFFINITY_MASK=0 build-sycl/mmvq_xmx_price [--reps N] [--t 4,6] [--case SUBSTR] [--quick]
//
// The weight data is synthetic-but-valid and this file's own assumption only: random bytes with finite fp16
// scales, exactly the parity harness's generator (src/kernels/mmvq_multi_spread.cpp:75-78), then passed
// through the tree's dequantizers -- so the encodings are real even though no model is loaded.  The
// geometries are d2/D2-TYPES.txt's and the launch counts d2/D2-CENSUS.txt's; the counts sum to that
// census's own per-type totals.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

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

namespace mx = sycl::ext::oneapi::experimental::matrix;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------- the case table
// Every distinct (type, n_in, n_out) of the 301 weight matrices the decode window's dense MMVQ family
// launches, with the launch count per window from the corrected census (d2/D2-CENSUS.txt, arm
// d2-hist-4096): the counts add up to that census's own per-type totals 129/35/47/42/47/1.
// n_in is the contiguous reduction dimension and n_out the row count (GGUF order: ne[0], ne[1]).
struct Case {
    const char* name;
    int type;
    int n_in;
    int n_out;
    int count;
};

const Case CASES[] = {
    {"Q6_K 2560x10240",            14, 2560, 10240, 22},
    {"Q6_K 2560x12288",            14, 2560, 12288, 5},
    {"Q6_K 2560x6144",             14, 2560, 6144, 14},
    {"Q6_K 2560x640",              14, 2560, 640, 26},
    {"Q6_K 2560x512",              14, 2560, 512, 21},
    {"Q6_K 6144x2560",             14, 6144, 2560, 40},
    {"Q6_K 2560x248320-head",      14, 2560, 248320, 1},
    {"Q5_K 2560x10240",            13, 2560, 10240, 10},
    {"Q5_K 2560x12288",            13, 2560, 12288, 3},
    {"Q5_K 2560x6144",             13, 2560, 6144, 4},
    {"Q5_K 2560x640",              13, 2560, 640, 12},
    {"Q5_K 2560x512",              13, 2560, 512, 2},
    {"Q5_K 6144x2560",             13, 6144, 2560, 4},
    {"Q4_K 2560x10240",            12, 2560, 10240, 2},
    {"Q4_K 2560x12288",            12, 2560, 12288, 1},
    {"Q4_K 2560x6144",             12, 2560, 6144, 10},
    {"Q4_K 2560x640",              12, 2560, 640, 30},
    {"Q4_K 6144x2560",             12, 6144, 2560, 4},
    {"IQ4_XS 2560x10240",          23, 2560, 10240, 2},
    {"IQ4_XS 2560x12288",          23, 2560, 12288, 3},
    {"IQ4_XS 2560x6144",           23, 2560, 6144, 8},
    {"IQ4_XS 2560x640",            23, 2560, 640, 28},
    {"IQ4_XS 2560x512",            23, 2560, 512, 1},
    {"IQ4_NL 640x2560",            20, 640, 2560, 47},
    {"Q8_0 640x2560",              8, 640, 2560, 1},
};

struct TypeInfo {
    int block_elems;
    int block_bytes;
    int scale_at[2];
    void (*dequant)(const uint8_t*, float*);
};
const TypeInfo* type_info(int type) {
    static const TypeInfo q6{256, 210, {208, -1}, strata::dequantize_q6_K};
    static const TypeInfo q5{256, 176, {0, 2}, strata::dequantize_q5_K};
    static const TypeInfo q4{256, 144, {0, 2}, strata::dequantize_q4_K};
    static const TypeInfo iq4xs{256, 136, {0, -1}, strata::dequantize_iq4_xs};
    static const TypeInfo iq4nl{32, 18, {0, -1}, strata::dequantize_iq4_nl};
    static const TypeInfo q8{32, 34, {0, -1}, strata::dequantize_q8_0};
    switch (type) {
        case 14: return &q6;
        case 13: return &q5;
        case 12: return &q4;
        case 23: return &iq4xs;
        case 20: return &iq4nl;
        case 8: return &q8;
        default: return nullptr;
    }
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

// a normal fp16 in +-[2^-10, 2^-5): random bytes in a scale field would put inf/NaN in the outputs
// (src/kernels/mmvq_multi_spread.cpp:75-78)
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t)(((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

void fill_blocks(std::vector<uint8_t>& w, const Case& c, const TypeInfo& ti, unsigned seed) {
    std::mt19937 rng(seed);
    for (auto& b : w) b = (uint8_t)(rng() & 0xff);
    const std::size_t per_row = (std::size_t)c.n_in / ti.block_elems;
    for (int row = 0; row < c.n_out; ++row) {
        for (std::size_t k = 0; k < per_row; ++k) {
            uint8_t* block = w.data() + ((std::size_t)row * per_row + k) * ti.block_bytes;
            for (int s = 0; s < 2; ++s) {
                if (ti.scale_at[s] < 0) continue;
                const uint16_t h = sane_half(rng);
                std::memcpy(block + ti.scale_at[s], &h, 2);
            }
        }
    }
}

// ---------------------------------------------------------------- section 1: the device's tile shapes
template <typename TA, typename TB, typename TC, int M, int K, int N, mx::layout LA, mx::layout LB>
bool probe_tile(sycl::queue& q, const char* label, std::string* refusal) {
    try {
        TA* a = sycl::malloc_device<TA>((std::size_t)M * K, q);
        TB* b = sycl::malloc_device<TB>((std::size_t)K * N, q);
        TC* c = sycl::malloc_device<TC>((std::size_t)M * N, q);
        q.memset(a, 0, sizeof(TA) * (std::size_t)M * K).wait();
        q.memset(b, 0, sizeof(TB) * (std::size_t)K * N).wait();
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) {
            sycl::sub_group sg = it.get_sub_group();
            auto gp = [](auto* p) {
                return sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(p);
            };
            mx::joint_matrix<sycl::sub_group, TA, mx::use::a, M, K, LA> A;
            mx::joint_matrix<sycl::sub_group, TB, mx::use::b, K, N, LB> B;
            mx::joint_matrix<sycl::sub_group, TC, mx::use::accumulator, M, N> C;
            mx::joint_matrix_fill(sg, C, TC(0));
            mx::joint_matrix_load(sg, A, gp(a), (std::size_t)K);
            mx::joint_matrix_load(sg, B, gp(b), (std::size_t)N);
            mx::joint_matrix_mad(sg, C, A, B, C);
            mx::joint_matrix_store(sg, C, gp(c), (std::size_t)N, mx::layout::row_major);
        }).wait_and_throw();
        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(c, q);
        std::printf("  %-54s RUNS\n", label);
        return true;
    } catch (const std::exception& e) {
        if (refusal != nullptr) *refusal = e.what();
        std::printf("  %-54s REFUSED: %s\n", label, e.what());
        return false;
    }
}

// ---------------------------------------------------------------- section 2: streaming read bandwidth
void read_only_kernel(sycl::queue& q, const float* src, float* sink, std::size_t n_floats) {
    const std::size_t per_thread = 16;                 // 4 x vec4, 4 independent chains
    const std::size_t groups = n_floats / per_thread;
    const std::size_t threads = 256;
    const std::size_t blocks = groups / threads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) {
            const std::size_t base = it.get_global_id(0) * per_thread;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int r = 0; r < 4; ++r) {
                const auto x = *reinterpret_cast<const sycl::vec<float, 4>*>(src + base + (std::size_t)r * 4);
                acc[r] += x.x() + x.y() + x.z() + x.w();
            }
            // an unconditional store of 4 B per 64 B read (6% extra traffic, and it is what keeps the loads
            // alive: a conditional store the compiler may fold away is how a "read" kernel reports 1310 GB/s)
            sink[it.get_global_id(0) & ((1u << 18) - 1)] = acc[0] + acc[1] + acc[2] + acc[3];
        });
    });
}

double time_read_only(sycl::queue& q, cudaStream_t s, const float* src, float* sink, std::size_t n_floats,
                      int reps) {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<double> ms;
    for (int r = -1; r < reps; ++r) {
        cudaEventRecord(e0, s);
        read_only_kernel(q, src, sink, n_floats);
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        float t = 0.f;
        cudaEventElapsedTime(&t, e0, e1);
        if (r >= 0) ms.push_back(t);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

// A cross-check on the event machinery itself for a kernel this long: submit the same kernel `passes` times
// back to back and time the whole burst with the wall clock, so a launch overhead and a mis-recorded event
// bracket are both visible.
double time_read_wall(sycl::queue& q, cudaStream_t s, const float* src, float* sink, std::size_t n_floats,
                      int passes) {
    read_only_kernel(q, src, sink, n_floats);
    cudaStreamSynchronize(s);
    const auto t0 = Clock::now();
    for (int p = 0; p < passes; ++p) read_only_kernel(q, src, sink, n_floats);
    cudaStreamSynchronize(s);
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / passes;
}

// ---------------------------------------------------------------- the prototype kernel
// The dot product on the matrix unit, in the only shape this toolchain actually builds for a GEMV-shaped
// kernel: f16 8x16x16 (M=8 output rows, K=16 contraction, N=16 columns) into an fp32 accumulator.  Section 1
// reports which shapes build and which the driver refuses, on this driver.
//
// Contract, exactly:
//   * weights: f16, repacked once at load time into 8-row x 16-k tiles --
//     wt[((row/8) * (n_in/16) + k/16) * 128 + m*16 + kk] = w[row][k]  (m < 8, kk < 16).
//     That repack is the price a real implementation pays: the tiles the unit consumes are not the pack's
//     blocks, and this program measures the repack + dequant per case.
//   * activations: f16, row-major [n_in][16]; only the first TCOLS columns carry a token (at ncols = 4 four
//     of the sixteen tile columns are real, the rest repeat column 0), so the shape's wasted lanes are
//     arithmetic, not invented bytes.
//   * RT = 4 row tiles (32 output rows) per work-group, one accumulated 8x16 fp32 tile each; NSG = 4
//     sub-groups split the k range of the same 32 rows and are reduced in work-group local memory; the store
//     writes the TCOLS real columns only.
//   * traffic: the weights are the stream (2 B/element, every call, no reuse), while the activation tile is
//     n_in x 16 x 2 B (at most 80 KiB here) and is re-read by every work-group, so it stays in cache: the
//     prototype's DRAM floor is (n_out * n_in * 2) / measured_bandwidth, i.e. 2.69x the pack's own byte
//     count, and that ratio is the whole reason its ceiling is below the shipped kernel's (section 5).
template <int NSG, int TCOLS>
void xmx_gemv_f16(sycl::queue& q, sycl::half* wt, sycl::half* x, float* y, int n_in, int n_out) {
    constexpr int RT = 4;
    const int nw = (n_out + 8 * RT - 1) / (8 * RT);
    const int kbt = n_in / 16;
    const int threads = 32 * NSG;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 3> part(sycl::range<3>((std::size_t)NSG, RT * 8, 16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((std::size_t)nw * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) {
            sycl::sub_group sg = it.get_sub_group();
            const int sgid = (int)sg.get_group_id();
            const int rowb = (int)it.get_group(0) * RT;    // first 8-row tile of this work-group
            auto gp = [](sycl::half* p) {
                return sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(p);
            };
            auto lp = [](float* p) {
                return sycl::address_space_cast<sycl::access::address_space::local_space,
                                                sycl::access::decorated::no>(p);
            };
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16, mx::layout::row_major> A;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> B;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> C0, C1, C2, C3;
            mx::joint_matrix_fill(sg, C0, 0.0f);
            mx::joint_matrix_fill(sg, C1, 0.0f);
            mx::joint_matrix_fill(sg, C2, 0.0f);
            mx::joint_matrix_fill(sg, C3, 0.0f);
            for (int kb = sgid; kb < kbt; kb += NSG) {
                joint_matrix_load(sg, B, gp(x + (std::size_t)kb * 256), (std::size_t)16);
                joint_matrix_load(sg, A, gp(wt + ((std::size_t)(rowb + 0) * kbt + kb) * 128), (std::size_t)16);
                mx::joint_matrix_mad(sg, C0, A, B, C0);
                joint_matrix_load(sg, A, gp(wt + ((std::size_t)(rowb + 1) * kbt + kb) * 128), (std::size_t)16);
                mx::joint_matrix_mad(sg, C1, A, B, C1);
                joint_matrix_load(sg, A, gp(wt + ((std::size_t)(rowb + 2) * kbt + kb) * 128), (std::size_t)16);
                mx::joint_matrix_mad(sg, C2, A, B, C2);
                joint_matrix_load(sg, A, gp(wt + ((std::size_t)(rowb + 3) * kbt + kb) * 128), (std::size_t)16);
                mx::joint_matrix_mad(sg, C3, A, B, C3);
            }
            mx::joint_matrix_store(sg, C0, lp(&part[sgid][0][0]), (std::size_t)16, mx::layout::row_major);
            mx::joint_matrix_store(sg, C1, lp(&part[sgid][8][0]), (std::size_t)16, mx::layout::row_major);
            mx::joint_matrix_store(sg, C2, lp(&part[sgid][16][0]), (std::size_t)16, mx::layout::row_major);
            mx::joint_matrix_store(sg, C3, lp(&part[sgid][24][0]), (std::size_t)16, mx::layout::row_major);
            sycl::group_barrier(it.get_group());
            const int lid = (int)it.get_local_id(0);
            if (lid < RT * 8 * TCOLS) {
                const int rt = lid / (8 * TCOLS);
                const int mj = lid % (8 * TCOLS);
                const int m = mj / TCOLS;
                const int j = mj % TCOLS;
                float acc = 0.f;
                for (int sg2 = 0; sg2 < NSG; ++sg2) acc += part[sg2][rt * 8 + m][j];
                const int row = (rowb + rt) * 8 + m;
                if (row < n_out) y[(std::size_t)j * n_out + row] = acc;
            }
        });
    });
}

// ---------------------------------------------------------------- timing one call
struct Stat {
    double min_us = 0, med_us = 0, batch_us = 0;
    bool ok = false;
};

// Two prices per call, because they answer different questions:
//   med_us   - one launch bracketed by its own event pair with a stream sync after it: what ONE call costs a
//              caller who waits for it (and it carries this rig's per-call sync overhead on the small shapes);
//   batch_us - `reps` launches submitted back to back between two events, divided by reps: the in-stream
//              device cost of a call, which is the quantity the engine's own census (STRATA_LAUNCH_HIST)
//              measures, so the window extrapolation uses this one.
template <typename Fn>
Stat time_calls(cudaStream_t s, int reps, Fn&& fn) {
    Stat st;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (!ck(cudaEventCreate(&e0), "event") || !ck(cudaEventCreate(&e1), "event")) return st;
    fn();                                              // warm-up (may carry a program build)
    if (!ck(cudaStreamSynchronize(s), "sync")) return st;
    std::vector<float> us;
    for (int r = 0; r < reps; ++r) {
        if (cudaEventRecord(e0, s) != cudaSuccess) return st;
        fn();
        if (cudaEventRecord(e1, s) != cudaSuccess) return st;
        if (!ck(cudaStreamSynchronize(s), "sync")) return st;
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        us.push_back(ms * 1e3f);
    }
    std::sort(us.begin(), us.end());
    st.min_us = us.front();
    st.med_us = us[us.size() / 2];
    if (cudaEventRecord(e0, s) != cudaSuccess) return st;
    for (int r = 0; r < reps; ++r) fn();
    if (cudaEventRecord(e1, s) != cudaSuccess) return st;
    if (!ck(cudaStreamSynchronize(s), "sync")) return st;
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    st.batch_us = ms * 1e3 / (double)reps;
    st.ok = true;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return st;
}

// ---------------------------------------------------------------- error metrics
struct Err {
    double max_abs = 0, rms_d = 0, rms_ref = 0, rel_rms = 0, max_over_rms = 0;
};
Err compare(const std::vector<double>& a, const std::vector<double>& ref) {
    Err e;
    double ss = 0, sr = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - ref[i]);
        if (d > e.max_abs) e.max_abs = d;
        ss += d * d;
        sr += ref[i] * ref[i];
    }
    e.rms_d = std::sqrt(ss / (double)a.size());
    e.rms_ref = std::sqrt(sr / (double)a.size());
    e.rel_rms = e.rms_ref > 0 ? e.rms_d / e.rms_ref : 0;
    e.max_over_rms = e.rms_ref > 0 ? e.max_abs / e.rms_ref : 0;
    return e;
}

// The bandwidth kernel above reads whatever is in the buffer.  A memset-filled buffer is a CONSTANT, and this
// driver compresses constant/uniform pages: the read figure comes out at 2-4x the DRAM rate.  So the floor is
// taken from a read of the SAME buffer after an incompressible fill (a hash pattern), which is what quantized
// weights are.  Both numbers are printed, because the difference is the point.
void fill_hash_kernel(sycl::queue& q, float* dst, std::size_t n_floats) {
    const std::size_t threads = 256;
    const std::size_t blocks = (n_floats + threads - 1) / threads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) {
            const std::size_t i = it.get_global_id(0);
            if (i >= n_floats) return;
            uint32_t h32 = (uint32_t)(i * 2654435761ull) ^ 0x9E3779B9u;
            h32 ^= h32 >> 15;
            h32 *= 0x85EBCA6Bu;
            reinterpret_cast<uint32_t*>(dst)[i] = h32;
        });
    });
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 20;
    int t_list[4] = {4, 6, 0, 0};
    int n_t = 2;
    const char* filter = nullptr;
    bool quick = false;
    bool skip_bw = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--t") == 0 && i + 1 < argc) {
            n_t = 0;
            for (char* tok = std::strtok(argv[++i], ","); tok != nullptr && n_t < 4; tok = std::strtok(nullptr, ","))
                t_list[n_t++] = std::atoi(tok);
        } else if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) filter = argv[++i];
        else if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--skip-bw") == 0) skip_bw = true;
    }
    if (reps < 1) reps = 1;
    if (n_t < 1) { n_t = 1; t_list[0] = 4; }

    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::printf("device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    sycl::queue& q = *strata::sycl_compat::queue_for(s);
    if (!skip_bw)
        std::printf("sycl device: %s, driver %s, max work-group %zu\n",
                    q.get_device().get_info<sycl::info::device::name>().c_str(),
                    q.get_device().get_info<sycl::info::device::driver_version>().c_str(),
                    q.get_device().get_info<sycl::info::device::max_work_group_size>());
    std::printf("reps %d, ncols", reps);
    for (int i = 0; i < n_t; ++i) std::printf(" %d", t_list[i]);
    std::printf("%s\n\n", quick ? ", quick" : "");

    // ---------------------------------------------------------------- 1. the tile shapes, re-measured here
    std::printf("== 1. the device's own joint_matrix shapes (probe21 re-run at this HEAD, not inherited) ==\n");
    bool f16_16_ok = false, f16_8_ok = false, int8_ok = false;
    std::string int8_msg;
    {
        std::string msg;
        f16_16_ok = probe_tile<sycl::half, sycl::half, float, 16, 16, 16, mx::layout::row_major,
                               mx::layout::row_major>(q, "f16  16x16x16  (A row_major, B row_major)", &msg);
        f16_16_ok |= probe_tile<sycl::half, sycl::half, float, 16, 16, 16, mx::layout::col_major,
                                mx::layout::row_major>(q, "f16  16x16x16  (A col_major, B row_major)", &msg);
        f16_16_ok |= probe_tile<sycl::half, sycl::half, float, 16, 16, 16, mx::layout::ext_intel_packed,
                                mx::layout::ext_intel_packed>(q, "f16  16x16x16  (A intel_packed, B intel_packed)",
                                                              &msg);
        f16_8_ok = probe_tile<sycl::half, sycl::half, float, 8, 16, 16, mx::layout::row_major,
                              mx::layout::row_major>(q, "f16   8x16x16  (A row_major, B row_major)", &msg);
        f16_8_ok |= probe_tile<sycl::half, sycl::half, float, 8, 16, 16, mx::layout::col_major,
                               mx::layout::row_major>(q, "f16   8x16x16  (A col_major, B row_major)", &msg);
        probe_tile<sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, float, 16, 16, 16,
                   mx::layout::row_major, mx::layout::row_major>(
            q, "bf16 16x16x16  (A row_major, B row_major)", &msg);
        probe_tile<sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, float, 8, 16, 16,
                   mx::layout::row_major, mx::layout::row_major>(
            q, "bf16  8x16x16  (A row_major, B row_major)", &msg);
        int8_ok |= probe_tile<int8_t, int8_t, int32_t, 8, 16, 32, mx::layout::row_major,
                              mx::layout::row_major>(q, "int8  8x32x16  (A row_major, B row_major)", &int8_msg);
        int8_ok |= probe_tile<int8_t, int8_t, int32_t, 8, 16, 32, mx::layout::row_major,
                              mx::layout::col_major>(q, "int8  8x32x16  (A row_major, B col_major)", &int8_msg);
        int8_ok |= probe_tile<int8_t, int8_t, int32_t, 8, 16, 32, mx::layout::row_major,
                              mx::layout::ext_intel_packed>(q, "int8  8x32x16  (A row_major, B intel_packed)",
                                                            &int8_msg);
        int8_ok |= probe_tile<int8_t, int8_t, int32_t, 16, 16, 32, mx::layout::row_major,
                              mx::layout::row_major>(q, "int8 16x32x16  (A row_major, B row_major)", &int8_msg);
        std::printf("\n  VERDICT on this driver (%s):\n"
                    "    f16 16x16x16 / bf16 16x16x16 (the shapes probe21 recorded as RUNNING): %s\n"
                    "    f16  8x16x16 (the only tile that builds for a GEMV-shaped kernel): %s\n"
                    "    the int8 8x32x16 / 16x32x16 dot (the shape this card was proposed for): %s\n",
                    q.get_device().get_info<sycl::info::device::driver_version>().c_str(),
                    f16_16_ok ? "RUNS" : "DOES NOT BUILD",
                    f16_8_ok ? "RUNS" : "DOES NOT BUILD",
                    int8_ok ? "RUNS in at least one layout" : "REFUSED by the driver in every layout tried");
        if (!int8_ok && !int8_msg.empty()) std::printf("  driver text: %s\n", int8_msg.c_str());
        std::printf("\n");
    }

    // ---------------------------------------------------------------- 2. the bandwidth the floor is made of
    double bw_gbs = 0, bw_compressible = 0;
    {
        float* src = sycl::malloc_device<float>((1u << 30) / 4, q);
        float* sink = sycl::malloc_device<float>(1u << 20, q);
        if (src == nullptr || sink == nullptr) {
            std::printf("== 2. the card's streaming rate, measured here ==\n   ALLOCATION FAILED\n\n");
            return 1;
        }
        std::printf("== 2. the card's streaming rate, measured here (event-timed on the same stream as the "
                    "kernels below) ==\n");
        for (std::size_t bytes : {std::size_t(8) << 20, std::size_t(256) << 20, std::size_t(1) << 30}) {
            const std::size_t n_floats = bytes / 4;
            q.memset(src, 0x3f, bytes).wait();
            const double ms = time_read_only(q, s, src, sink, n_floats, quick ? 3 : 7);
            const double gbs = (double)bytes / (ms * 1e-3) / 1e9;
            const double wms = time_read_wall(q, s, src, sink, n_floats, quick ? 3 : 5);
            const double wgbs = (double)bytes / (wms * 1e-3) / 1e9;
            if (bytes == (std::size_t(1) << 30)) bw_compressible = gbs;
            std::printf("   read-only over %6.0f MiB [memset-filled]: event %8.3f ms -> %7.1f GB/s | wall/burst "
                        "%8.3f ms -> %7.1f GB/s\n", bytes / 1048576.0, ms, gbs, wms, wgbs);
        }
        // the floor: the SAME read over the SAME buffer after an incompressible fill
        {
            const std::size_t bytes = (1u << 30);
            const std::size_t n_floats = bytes / 4;
            fill_hash_kernel(q, src, n_floats);
            cudaStreamSynchronize(s);
            const double ms = time_read_only(q, s, src, sink, n_floats, quick ? 3 : 7);
            bw_gbs = (double)bytes / (ms * 1e-3) / 1e9;
            std::printf("   read-only over %6.0f MiB [hash-filled, incompressible]: %.3f ms -> %7.1f GB/s   <- the "
                        "floor below uses this\n", bytes / 1048576.0, ms, bw_gbs);
            const double wms = time_read_wall(q, s, src, sink, n_floats, quick ? 3 : 5);
            std::printf("        (wall/burst cross-check: %.3f ms -> %.1f GB/s)\n", wms,
                        (double)bytes / (wms * 1e-3) / 1e9);
            std::printf("   the memset-filled read above is %.2fx the incompressible rate: the driver compresses "
                        "uniform pages, so a constant-filled\n   buffer is not a bandwidth measurement\n",
                        bw_compressible / bw_gbs);
        }
        // write-only: a memset cannot be optimised away by a compiler, but it IS compressible - printed for
        // completeness only
        {
            const std::size_t bytes = (1u << 30);
            q.memset(src, 0x3f, bytes).wait();
            cudaEvent_t e0 = nullptr, e1 = nullptr;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, s);
            q.memset(src, 0x00, bytes);
            cudaEventRecord(e1, s);
            cudaStreamSynchronize(s);
            float t = 0.f;
            cudaEventElapsedTime(&t, e0, e1);
            std::printf("   write-only memset of %6.0f MiB (constant, compressible): event %8.3f ms -> %7.1f GB/s\n",
                        bytes / 1048576.0, t, (double)bytes / (t * 1e-3) / 1e9);
        }
        // a device->device copy (read + write) for scale: this is the traffic a two-pass variant would pay
        float* dst = sycl::malloc_device<float>((256u << 20) / 4, q);
        const std::size_t cbytes = (256u << 20);
        auto copy3 = [&] {
            for (int pass = 0; pass < 3; ++pass)
                q.submit([&](sycl::handler& h) { h.memcpy(dst, src, cbytes); });
        };
        copy3();
        cudaStreamSynchronize(s);
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, s);
        copy3();
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        float cms = 0.f;
        cudaEventElapsedTime(&cms, e0, e1);
        if (cms > 0) std::printf("   device->device copy of %6.0f MiB x3: %.3f ms -> %.1f GB/s of traffic "
                                 "(read+write)\n", cbytes / 1048576.0, cms,
                                 2.0 * (double)cbytes * 3 / ((double)cms * 1e-3) / 1e9);
        std::printf("   (the floor below uses the read-only number: a GEMV reads the weights and writes n_out "
                    "outputs)\n\n");
        sycl::free(dst, q);
        sycl::free(src, q);
        sycl::free(sink, q);
    }

    // ---------------------------------------------------------------- 3. per-case table
    std::printf("== 3. per (type, shape, ncols): the shipped kernel, the generic layout, the f16 XMX prototype\n");
    std::printf("   (floor = this card's measured %.1f GB/s on the pack's own weight bytes; prototype GB/s is on f16 "
                "bytes;\n    us/call is the IN-STREAM price - `reps` launches back to back / reps - which is what the "
                "census measures)\n", bw_gbs);
    std::printf("%-22s %2s %5s %13s %9s %8s %8s %8s %8s %8s %8s %7s %8s\n", "case", "T", "cnt", "weight B",
                "floor us", "ex us", "ex GB/s", "gen us", "gen GB/s", "f16 us", "f16 GB/s", "dq ms", "ex 1call");
    double w_exact[4] = {0, 0, 0, 0}, w_gen[4] = {0, 0, 0, 0}, w_f16[4] = {0, 0, 0, 0}, w_floor[4] = {0, 0, 0, 0};
    double w_f16floor[4] = {0, 0, 0, 0};
    double w_bytes = 0;
    std::vector<std::string> byte_check;
    std::vector<std::string> parity_report;

    for (const Case& c : CASES) {
        if (filter != nullptr && std::strstr(c.name, filter) == nullptr) continue;
        const TypeInfo* ti = type_info(c.type);
        if (ti == nullptr || !strata::kernels::native_mmvq_supported(c.type)) {
            std::printf("%-22s unsupported\n", c.name);
            continue;
        }
        const std::size_t wbytes = (std::size_t)c.n_out * (c.n_in / ti->block_elems) * ti->block_bytes;
        const std::size_t engine_bytes = strata::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
        if (wbytes != engine_bytes)
            byte_check.push_back(std::string(c.name) + ": table " + std::to_string(wbytes) + " vs engine " +
                                 std::to_string(engine_bytes));

        std::vector<uint8_t> hw(wbytes);
        fill_blocks(hw, c, *ti, 1234u + (unsigned)c.type * 97u + (unsigned)c.n_in + (unsigned)c.n_out);
        // The dequantized weights.  w32: the fp64 reference (rows_real rows).  wt: the SAME values in the
        // unit's own f16 encoding, in 16x16 tile order (one pass, no separate repack buffer).
        // For a case whose blocks exceed 64 MiB only the first 4096 rows are dequantized and then tiled - the
        // heavy one is the 521 MiB head, whose kernel has no data-dependent branch (stated in the write-up).
        const int rows_real = (wbytes > (64u << 20)) ? 4096 : c.n_out;
        const int kbt = c.n_in / 16;
        std::vector<float> w32((std::size_t)c.n_in * rows_real);
        std::vector<sycl::half> wt((std::size_t)c.n_out * c.n_in);
        const double dq0 = now_ms();
        {
            const std::size_t per_row = (std::size_t)c.n_in / ti->block_elems;
            for (int row = 0; row < rows_real; ++row)
                for (std::size_t k = 0; k < per_row; ++k)
                    ti->dequant(hw.data() + ((std::size_t)row * per_row + k) * ti->block_bytes,
                                w32.data() + ((std::size_t)row * c.n_in + k * ti->block_elems));
            for (int row = 0; row < c.n_out; ++row) {
                const int src = row < rows_real ? row : 0;
                const int m = row % 8, rb = row / 8;
                const float* s = w32.data() + (std::size_t)src * c.n_in;
                for (int kb = 0; kb < kbt; ++kb)
                    for (int kk = 0; kk < 16; ++kk)
                        wt[((std::size_t)rb * kbt + kb) * 128 + m * 16 + kk] = sycl::half(s[kb * 16 + kk]);
            }
        }
        const double dq_ms = now_ms() - dq0;

        void* dw = nullptr;
        if (!ck(cudaMalloc(&dw, wbytes), "malloc w")) return 1;
        if (!ck(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "copy w")) return 1;
        sycl::half* w16d = sycl::malloc_device<sycl::half>(wt.size(), q);
        q.memcpy(w16d, wt.data(), wt.size() * sizeof(sycl::half)).wait();

        const int maxT = *std::max_element(t_list, t_list + n_t);
        std::vector<float> hx((std::size_t)maxT * c.n_in);
        std::mt19937 rng(77u + (unsigned)c.type);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hx) v = nd(rng);
        float* xf = nullptr;
        void* xq = nullptr;
        float* dy = nullptr;
        if (!ck(cudaMalloc(&xf, hx.size() * 4), "malloc x")) return 1;
        if (!ck(cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(c.n_in, maxT)), "malloc xq")) return 1;
        if (!ck(cudaMalloc(&dy, (std::size_t)std::max(maxT, 16) * c.n_out * 4), "malloc y")) return 1;
        if (!ck(cudaMemcpy(xf, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "copy x")) return 1;
        const int tile_cols = 16;
        std::vector<sycl::half> hxh((std::size_t)c.n_in * tile_cols);
        for (int k = 0; k < c.n_in; ++k)
            for (int j = 0; j < tile_cols; ++j)
                hxh[(std::size_t)k * tile_cols + j] = sycl::half(hx[(std::size_t)(j % maxT) * c.n_in + k]);
        sycl::half* xh = sycl::malloc_device<sycl::half>(hxh.size(), q);
        q.memcpy(xh, hxh.data(), hxh.size() * sizeof(sycl::half)).wait();

        for (int it_t = 0; it_t < n_t; ++it_t) {
            const int T = t_list[it_t];
            strata::kernels::native_quantize_q8_1(xf, xq, c.n_in, T, s);
            if (!ck(cudaStreamSynchronize(s), "quantize")) return 1;
            const double floor = (double)wbytes / (bw_gbs * 1e9) * 1e6;
            strata::kernels::native_mmvq_set_multi_exact(true);
            const Stat ex = time_calls(s, reps, [&] {
                strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
            });
            strata::kernels::native_mmvq_set_multi_exact(false);
            const Stat ge = time_calls(s, reps, [&] {
                strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
            });
            strata::kernels::native_mmvq_set_multi_exact(true);
            double f16_us = 0;
            if (T <= 8) {
                const Stat pr = time_calls(s, reps, [&] {
                    if (T == 4) xmx_gemv_f16<4, 4>(q, w16d, xh, dy, c.n_in, c.n_out);
                    else if (T == 6) xmx_gemv_f16<4, 6>(q, w16d, xh, dy, c.n_in, c.n_out);
                    else xmx_gemv_f16<4, 2>(q, w16d, xh, dy, c.n_in, c.n_out);
                });
                f16_us = pr.batch_us;
            }
            const double egb = wbytes / (ex.batch_us * 1e-6) / 1e9;
            const double ggb = wbytes / (ge.batch_us * 1e-6) / 1e9;
            const double pgb = f16_us > 0 ? (double)wt.size() * 2 / (f16_us * 1e-6) / 1e9 : 0;
            std::printf("%-22s %2d %5d %13zu %9.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %7.1f %8.1f\n", c.name,
                        T, c.count, wbytes, floor, ex.batch_us, egb, ge.batch_us, ggb, f16_us, pgb, dq_ms,
                        ex.med_us);
            if (it_t == 0) {
                w_exact[it_t] += c.count * ex.batch_us;
                w_gen[it_t] += c.count * ge.batch_us;
                w_f16[it_t] += c.count * f16_us;
                w_floor[it_t] += c.count * floor;
                // the prototype's own floor: its weights are f16 (2 B/element, no reuse), the activation tile
                // is cache-resident
                w_f16floor[it_t] += c.count * (double)wt.size() * 2 / (bw_gbs * 1e9) * 1e6;
            }
        }
        w_bytes += (double)c.count * wbytes;

        // ---------------------------------------------------------------- 4. parity on the biggest Q6_K case
        if (std::strstr(c.name, "Q6_K 2560x10240") != nullptr && rows_real == c.n_out) {
            const int T = 4;
            const int PROWS = quick ? 32 : 64;
            // three separated row windows, so a wrong tile index or a missed work-group shows up:
            // the first, the middle and the last rows of the matrix
            const int r0[3] = {0, c.n_out / 2, c.n_out - PROWS};
            std::vector<int> rows;
            for (int k = 0; k < 3; ++k)
                for (int m = 0; m < PROWS; ++m) rows.push_back(r0[k] + m);
            const int NR = (int)rows.size();
            std::vector<double> ref((std::size_t)NR * T), ref16((std::size_t)NR * T), ref16x((std::size_t)NR * T);
            std::vector<double> ship((std::size_t)NR * T), proto((std::size_t)NR * T);
            for (int i = 0; i < NR; ++i) {
                const int row = rows[i];
                for (int j = 0; j < T; ++j) {
                    double a = 0, b = 0, cc = 0;
                    for (int k = 0; k < c.n_in; ++k) {
                        const double wv = w32[(std::size_t)row * c.n_in + k];
                        const double xv = hx[(std::size_t)j * c.n_in + k];
                        const double xh16 = (float)sycl::half((float)xv);
                        const double wh16 = (float)sycl::half((float)wv);
                        a += wv * xv;              // dequantized f32 weights . f32 activations
                        b += wh16 * xv;            // the f16 encoding of the weights only
                        cc += wh16 * xh16;         // the f16 encoding of BOTH: what the kernel can at best match
                    }
                    ref[(std::size_t)j * NR + i] = a;
                    ref16[(std::size_t)j * NR + i] = b;
                    ref16x[(std::size_t)j * NR + i] = cc;
                }
            }
            strata::kernels::native_mmvq_set_multi_exact(true);
            strata::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
            cudaStreamSynchronize(s);
            std::vector<float> hy((std::size_t)T * c.n_out);
            cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
            std::vector<float> py((std::size_t)T * c.n_out);
            xmx_gemv_f16<4, 4>(q, w16d, xh, dy, c.n_in, c.n_out);
            q.wait();
            cudaMemcpy(py.data(), dy, py.size() * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < T; ++j) {
                    ship[(std::size_t)j * NR + i] = hy[(std::size_t)j * c.n_out + rows[i]];
                    proto[(std::size_t)j * NR + i] = py[(std::size_t)j * c.n_out + rows[i]];
                }
            const Err es = compare(ship, ref);
            const Err ep = compare(proto, ref);
            const Err ep16 = compare(proto, ref16x);
            const Err e16 = compare(ref16, ref);
            const Err esp = compare(proto, ship);
            char buf[1400];
            std::snprintf(buf, sizeof buf,
                          "  parity, %s ncols=%d, rows {0, %d, %d} + %d (%d x %d outputs), fp64 reference = the same "
                          "dequantized weights . the same f32 activations:\n"
                          "    (1) the f16 encoding alone      (f16(w).x  vs w.x)     : max %.3e  rel_rms %.3e\n"
                          "    (2) shipped kernel vs ref       (Q8_1 activations)    : max %.3e  rel_rms %.3e\n"
                          "    (3) f16 XMX kernel vs ref       (f16 activations)     : max %.3e  rel_rms %.3e\n"
                          "    (4) f16 XMX kernel vs f16(w).f16(x), its own encoding : max %.3e  rel_rms %.3e   <- the "
                          "kernel's own arithmetic\n"
                          "    (5) f16 XMX kernel vs shipped kernel                  : max %.3e  rel_rms %.3e\n",
                          c.name, T, r0[1], r0[2], PROWS, NR, T, e16.max_abs, e16.rel_rms, es.max_abs, es.rel_rms,
                          ep.max_abs, ep.rel_rms, ep16.max_abs, ep16.rel_rms, esp.max_abs, esp.rel_rms);
            parity_report.push_back(buf);
        }

        cudaFree(dw);
        cudaFree(xf);
        cudaFree(xq);
        cudaFree(dy);
        sycl::free(w16d, q);
        sycl::free(xh, q);
    }
    strata::kernels::native_mmvq_set_multi_exact(true);

    std::printf("\n== 4. accuracy (any variant that changes the numerics states its own tolerance) ==\n");
    for (const std::string& p : parity_report) std::printf("%s", p.c_str());

    std::printf("\n== 5. what a whole T=%d window's 301 dense calls would cost ==\n", t_list[0]);
    std::printf("   the census (d2-hist-4096) measures this family at 301 launches / 51267.3 us of device time"
                " / 40.9%% of the window\n");
    std::printf("   weight bytes per window: %.0f (%.1f MiB); card read-only rate %.1f GB/s\n", w_bytes,
                w_bytes / 1048576.0, bw_gbs);
    const char* names[5] = {"shipped layout (multi_exact)", "generic layout (multi_exact off)",
                            "f16 XMX prototype (as written)", "the pack's own bandwidth floor",
                            "the prototype's own bandwidth floor"};
    const double totals[5] = {w_exact[0], w_gen[0], w_f16[0], w_floor[0], w_f16floor[0]};
    for (int i = 0; i < 5; ++i) {
        if (totals[i] <= 0) continue;
        std::printf("   %-34s %9.1f us  %6.2f ms  %5.1f%% of the window  vs shipped %+7.1f%%  vs the pack's floor"
                    " %5.2fx\n", names[i], totals[i], totals[i] / 1e3, totals[i] / 51267.3 * 40.9,
                    (totals[i] / totals[0] - 1.0) * 100.0, totals[i] / w_floor[0]);
    }
    std::printf("   the census's own 51267.3 us is the check on the shipped row above: this table's per-call\n"
                "   times x the census's own launch counts (the geometries' counts sum to 301)\n");

    std::printf("\nbyte-table check (this file's own block sizes vs native_mmvq_weight_bytes): %s\n",
                byte_check.empty() ? "every case MATCHES" : "MISMATCHES:");
    for (const std::string& b : byte_check) std::printf("   %s\n", b.c_str());
    cudaStreamDestroy(s);
    return 0;
}
