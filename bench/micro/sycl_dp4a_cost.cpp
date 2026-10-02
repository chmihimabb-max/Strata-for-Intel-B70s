// bench/micro/sycl_dp4a_cost.cpp - PLAN.md Risk 5: what does the dp4a EMULATION cost in the decode path?
//
// There is no DP4A instruction on this target and no `sycl::ext::oneapi::dot_product` spelling that is
// anything but multiplies (`plan-evidence/env_probe.txt`), so "with and without the emulation" cannot mean
// "native vs emulated" - on an Intel GPU both sides ARE an emulation.  What can be measured, and what
// decides the inner-loop shape, is the cost of the three spellings the plan's falsifying experiment names:
//
//   EMU    the shim's `dp4a` (include/strata/sycl_compat/intrinsics.hpp): 4 signed byte extracts + 4 mul-adds
//          - what every ported kernel uses today
//   VECDOT `sycl::ext::oneapi::dot_acc` on `sycl::vec<int8_t,4>` - oneAPI's own spelling of the same op
//          (its header is 4 multiplies too, see dot_product.hpp:57), which the backend may or may not lower
//          differently from ours
//   WIDEN  the same 4 multiplies on widened int32 lanes (the "keep the word, avoid the packing" variant)
//
// Each variant is the same dependent chain at the decode inner loop's shape (one dot per iteration, the
// second operand updated every iteration), run kIters times per thread over kBlocks x kThreads threads, so
// one launch is ~137M dots and the launch overhead is noise.  The instruction-level question - whether the
// backend CONTRACTS either spelling into a dot-product opcode - is answered separately by
// tools/sycl/spirv_ops.py over the device image this file's kernels produce.
//
// Build/run (from ~/strata-xpu with oneAPI sourced):
//   icpx -fsycl -fsycl-targets=spir64 -O3 -std=c++20 -I strata/include/strata/sycl_compat \
//        -o bench/micro/sycl_dp4a_cost bench/micro/sycl_dp4a_cost.cpp && ZE_AFFINITY_MASK=0 ./bench/micro/sycl_dp4a_cost
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>   // sycl::ext::oneapi::dot_acc (the VECDOT spelling)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kBlocks = 512;
constexpr int kThreads = 256;
constexpr int kIters = 1024;   // dots per thread per launch
constexpr int kRep = 5;

inline int emu_dp4a(int a, int b, int acc) {
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    int r = acc;
    for (int k = 0; k < 4; ++k) r += (int) pa[k] * (int) pb[k];
    return r;
}

inline int vecdot_dp4a(int a, int b, int acc) {
    // sycl::ext::oneapi::dot_acc - oneAPI's OWN spelling.  Its header (dot_product.hpp:57-74) is 4 multiplies
    // too: `a.s0() * b.s0() + a.s1() * b.s1() + ...`, which is the plan's point about the emulation being
    // unavoidable on this target.  Measured against ours because a different formulation can still lower
    // differently (the operand order and the vector type are the knobs the backend sees).
    const auto va = sycl::bit_cast<sycl::vec<int8_t, 4>>(a);
    const auto vb = sycl::bit_cast<sycl::vec<int8_t, 4>>(b);
    return sycl::ext::oneapi::dot_acc(va, vb, acc);
}

inline int widen_dp4a(int a, int b, int acc) {
    const int a0 = (int) (int8_t) (a & 0xFF), a1 = (int) (int8_t) ((a >> 8) & 0xFF);
    const int a2 = (int) (int8_t) ((a >> 16) & 0xFF), a3 = (int) (int8_t) ((a >> 24) & 0xFF);
    const int b0 = (int) (int8_t) (b & 0xFF), b1 = (int) (int8_t) ((b >> 8) & 0xFF);
    const int b2 = (int) (int8_t) ((b >> 16) & 0xFF), b3 = (int) (int8_t) ((b >> 24) & 0xFF);
    return acc + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
}

template <int Which>
void submit(sycl::queue& q, float* out, const int* w) {
    const sycl::nd_range<1> r{sycl::range<1>(size_t(kBlocks) * kThreads), sycl::range<1>(kThreads)};
    q.parallel_for(r, [=](sycl::nd_item<1> it) {
        const int gid = (int) it.get_global_id(0);
        int acc = 0;
        const int a = w[gid & 255];
        int b = w[(gid + 1) & 255];
        for (int i = 0; i < kIters; ++i) {
            b ^= (b << 13) | (b >> 19);            // keeps the second operand live and dependent
            if constexpr (Which == 0) {
                acc = emu_dp4a(a, b, acc);
            } else if constexpr (Which == 1) {
                acc = vecdot_dp4a(a, b, acc);
            } else {
                acc = widen_dp4a(a, b, acc);
            }
        }
        out[gid] = (float) acc;
    });
    q.wait();
}

double time_one(sycl::queue& q, float* out, const int* w, int which) {
    if (which == 0) {
        submit<0>(q, out, w);
    } else if (which == 1) {
        submit<1>(q, out, w);
    } else {
        submit<2>(q, out, w);
    }
    const auto t0 = std::chrono::steady_clock::now();   // the call above was the warm-up (JIT + caches)
    for (int r = 0; r < kRep; ++r) {
        if (which == 0) {
            submit<0>(q, out, w);
        } else if (which == 1) {
            submit<1>(q, out, w);
        } else {
            submit<2>(q, out, w);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / kRep;
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    float* out = sycl::malloc_device<float>(size_t(kBlocks) * kThreads, q);
    int* w = sycl::malloc_device<int>(256, q);
    std::vector<int> host_w(256);
    for (int i = 0; i < 256; ++i) host_w[i] = (int) ((i * 2654435761u) >> 3);
    q.memcpy(w, host_w.data(), sizeof(int) * 256).wait();

    const double total_dots = double(kBlocks) * double(kThreads) * double(kIters);
    std::printf("%d threads x %d dots per launch = %.1f M dots per launch, %.0f ns resolution\n\n", kBlocks * kThreads,
                kIters, total_dots / 1e6, 0.0);
    const char* names[3] = {"EMU   (shim dp4a: 4 byte extracts + 4 mul-adds)",
                            "VECDOT(sycl::dot on vec<int8_t,4>)        ",
                            "WIDEN (4 multiplies on widened int32 lanes)"};
    double ns[3] = {0, 0, 0};
    for (int which = 0; which < 3; ++which) {
        ns[which] = time_one(q, out, w, which);
        std::printf("  %s %9.3f ms/launch %8.3f ns/dot %7.2f Gdot/s\n", names[which], ns[which] / 1e6,
                    ns[which] / total_dots, total_dots / ns[which]);
    }
    std::printf("\n  VECDOT / EMU = %.3f x      WIDEN / EMU = %.3f x\n", ns[1] / ns[0], ns[2] / ns[0]);
    std::printf("  (1.000 x on either ratio means the spelling does not matter at this shape)\n");
    sycl::free(out, q);
    sycl::free(w, q);
    return 0;
}
