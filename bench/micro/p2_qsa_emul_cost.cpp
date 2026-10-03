// bench/micro/p2_qsa_emul_cost.cpp - card t_a1b6fa6c (P2): what the FP32 m16n8k16 emulation costs on BMG-G31.
//
// The QSA prompt attention kernel (src/kernels/sycl/qsa_prompt_attn.cpp) is one block per (query, KV head) and
// spends its inner loop on include/strata/sycl_compat/mma16816.hpp: each emulated mma is 32 select_from_group
// gathers, 64 f16->f32 conversions and 64 FMAs - i.e. the hardware tensor-core instruction is replaced by a
// fragment exchange the Xe cores have to issue lane by lane.  This micro prices the three parts separately so
// the attribution of the kernel's time is measured and not argued:
//
//   mode 0  mma as shipped      (32 selects + 64 cvt + 64 FMA) x2, the kernel's `mma(ah,b); mma(al,b)` pair
//   mode 1  the fused pair      (24 selects + 48 cvt + 64 FMA) x2: the B fragment is built ONCE for both calls
//   mode 2  FMAs only           128 FMA, no exchange at all (the arithmetic the math actually needs)
//   mode 3  the exchange only   64 select_from_group, nothing else
//   mode 4  conversions only    128 f16->f32
//
// The FMA count is the same in modes 0 and 1 (bit-identical arithmetic: same values, same 16-term order), so
// 0 -> 1 is the only difference a change may make; 2 is the floor, and 3 tells whether the exchange or the
// FMAs dominate the issue.
//
// Build (no CMake target: a bench, not a test):
//   icpx -fsycl -fsycl-targets=spir64 -fsycl-default-sub-group-size=32 -I include \
//        bench/micro/p2_qsa_emul_cost.cpp -o build-sycl/p2_qsa_emul_cost
// Run: ZE_AFFINITY_MASK=0 ./build-sycl/p2_qsa_emul_cost [iters] [blocks]
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/sycl_compat/mma16816.hpp"

namespace sc = strata::sycl_compat;

namespace {

inline float f16_lo(uint32_t x) { return (float) sycl::bit_cast<sycl::half>((uint16_t) (x & 0xffffu)); }
inline float f16_hi(uint32_t x) { return (float) sycl::bit_cast<sycl::half>((uint16_t) (x >> 16)); }

// ---- mode 1: the fused pair.  Identical arithmetic to two mma16816_f32 calls with the same b (the 16-term
// FMA order per output is untouched); the B fragment's 16 gathers and 32 conversions are done once.
// t_a1b6fa6c: this is the shipped implementation (include/strata/sycl_compat/mma16816.hpp), not a copy of it.
inline void mma_pair_fused(const sycl::sub_group& sg, float* c, const uint32_t* ahi, const uint32_t* alo,
                          const uint32_t* b) {
    sc::mma16816_f32_pair(sg, c, ahi, alo, b);
}

template <int MODE>
void one_iter(const sycl::sub_group& sg, float* c, const uint32_t* regs, int i) {
    uint32_t a[4] = {regs[0] + (uint32_t) i, regs[1], regs[2], regs[3]};
    uint32_t al[4] = {regs[6], regs[7], regs[8], regs[9]};
    uint32_t b[2] = {regs[4] + (uint32_t) i, regs[5]};
    if constexpr (MODE == 0) {
        sc::mma16816_f32(sg, c, a, b);
        sc::mma16816_f32(sg, c, al, b);
    } else if constexpr (MODE == 1) {
        mma_pair_fused(sg, c, a, al, b);
    } else if constexpr (MODE == 2) {   // 128 FMA, no exchange: the floor the arithmetic needs
        float x0 = a[0], x1 = a[1], x2 = a[2], x3 = a[3], y0 = b[0], y1 = b[1];
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            c[0] = sycl::fma(x0, y0, c[0]);
            c[1] = sycl::fma(x1, y1, c[1]);
            c[2] = sycl::fma(x2, y0, c[2]);
            c[3] = sycl::fma(x3, y1, c[3]);
        }
    } else if constexpr (MODE == 3) {   // 64 gathers, nothing else
        const int lane = (int) sg.get_local_linear_id();
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int q = (lane & ~3) | t;
            c[0] += (float) sycl::select_from_group(sg, a[0], (uint32_t) q);
            c[1] += (float) sycl::select_from_group(sg, a[1], (uint32_t) q);
            c[2] += (float) sycl::select_from_group(sg, a[2], (uint32_t) q);
            c[3] += (float) sycl::select_from_group(sg, a[3], (uint32_t) q);
            c[0] += (float) sycl::select_from_group(sg, b[0], (uint32_t) q);
            c[1] += (float) sycl::select_from_group(sg, b[1], (uint32_t) q);
            c[2] += (float) sycl::select_from_group(sg, a[0], (uint32_t) q);
            c[3] += (float) sycl::select_from_group(sg, b[1], (uint32_t) q);
            c[0] += (float) sycl::select_from_group(sg, b[0], (uint32_t) q);
            c[1] += (float) sycl::select_from_group(sg, a[2], (uint32_t) q);
            c[2] += (float) sycl::select_from_group(sg, a[3], (uint32_t) q);
            c[3] += (float) sycl::select_from_group(sg, b[1], (uint32_t) q);
            c[0] += (float) sycl::select_from_group(sg, a[1], (uint32_t) q);
            c[1] += (float) sycl::select_from_group(sg, b[0], (uint32_t) q);
            c[2] += (float) sycl::select_from_group(sg, a[1], (uint32_t) q);
            c[3] += (float) sycl::select_from_group(sg, b[0], (uint32_t) q);
        }
    } else {   // MODE 4: 128 f16 -> f32, no exchange
        c[0] += f16_lo(a[0]) + f16_hi(a[0]) + f16_lo(a[1]) + f16_hi(a[1]);
        c[1] += f16_lo(a[2]) + f16_hi(a[2]) + f16_lo(a[3]) + f16_hi(a[3]);
        c[2] += f16_lo(b[0]) + f16_hi(b[0]) + f16_lo(b[1]) + f16_hi(b[1]);
        c[3] += f16_lo(a[0]) + f16_hi(a[1]) + f16_lo(b[0]) + f16_hi(b[1]);
#pragma unroll
        for (int k = 0; k < 28; ++k) {
            c[0] += f16_lo(a[k & 3]) + f16_hi(b[k & 1]);
            c[1] += f16_lo(b[k & 1]) + f16_hi(a[(k + 1) & 3]);
            c[2] += f16_lo(a[(k + 2) & 3]);
            c[3] += f16_hi(b[(k + 1) & 1]);
        }
    }
}

template <int MODE>
class Cost {
public:
    Cost(uint32_t* regs, float* out, int iters) : regs_(regs), out_(out), iters_(iters) {}
    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t* r = regs_ + (size_t) (it.get_global_linear_id() % 16) * 10;
        for (int i = 0; i < iters_; ++i) one_iter<MODE>(sg, c, r, i);
        float* o = out_ + (size_t) it.get_global_linear_id() * 4;
        o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = c[3];
    }
    Cost(const Cost&) = default;
private:
    uint32_t* regs_; float* out_; int iters_;
};

double bench(sycl::queue& q, int mode, int iters, int blocks) {
    std::vector<uint32_t> hregs(16 * 10);
    uint32_t s = 12345u;
    for (auto& x : hregs) { s = s * 1664525u + 1013904223u; x = s; }
    uint32_t* regs = sycl::malloc_device<uint32_t>(hregs.size(), q);
    q.memcpy(regs, hregs.data(), hregs.size() * 4).wait();
    float* out = sycl::malloc_device<float>((size_t) blocks * 128 * 4, q);
    auto run = [&](int m) {
        switch (m) {
            case 0: q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t) blocks * 128, 128), Cost<0>(regs, out, iters)); }); break;
            case 1: q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t) blocks * 128, 128), Cost<1>(regs, out, iters)); }); break;
            case 2: q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t) blocks * 128, 128), Cost<2>(regs, out, iters)); }); break;
            case 3: q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t) blocks * 128, 128), Cost<3>(regs, out, iters)); }); break;
            default: q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t) blocks * 128, 128), Cost<4>(regs, out, iters)); }); break;
        }
    };
    run(mode);   // warmup
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    run(mode); run(mode); run(mode);
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 3.0;
    sycl::free(regs, q);
    sycl::free(out, q);
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 4000;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 64;
    sycl::queue q{sycl::property::queue::in_order{}};
    const auto d = q.get_device();
    std::printf("device %s  (%s), iters %d, blocks %d, %d work-items\n",
                d.get_info<sycl::info::device::name>().c_str(),
                d.get_info<sycl::info::device::driver_version>().c_str(), iters, blocks, blocks * 128);
    const double per_iter_wl = (double) iters * blocks * 128;
    const char* names[5] = {"mma as shipped (2 calls, 64 sel + 128 cvt + 128 FMA)",
                            "fused pair      (24+24 sel + 96 cvt + 128 FMA)",
                            "128 FMA only    (no exchange, no cvt)",
                            "64 select_from_group only",
                            "128 f16->f32 only"};
    double ms[5];
    for (int m = 0; m < 5; ++m) {
        ms[m] = bench(q, m, iters, blocks);
        std::printf("  mode %d  %-52s %9.3f ms   %7.3f ns per work-item-iteration   %7.2f M iter/s\n", m,
                    names[m], ms[m], ms[m] * 1e6 / per_iter_wl, per_iter_wl / (ms[m] * 1e3));
    }
    std::printf("\n  shippped pair %.3f ms vs FMA floor %.3f ms = %.2fx the pure arithmetic\n", ms[0], ms[2],
                ms[0] / ms[2]);
    std::printf("  fused pair    %.3f ms vs FMA floor %.3f ms = %.2fx ; fusion saves %.1f%% of the shipped pair\n",
                ms[1], ms[2], ms[1] / ms[2], 100.0 * (ms[0] - ms[1]) / ms[0]);
    std::printf("  the exchange alone (mode 3) is %.3f ms against %.3f ms of FMAs (mode 2)\n", ms[3], ms[2]);
    return 0;
}
