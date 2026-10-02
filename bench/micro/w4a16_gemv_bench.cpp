// bench/micro/w4a16_gemv_bench.cpp - card t_7e74bb89 (Strata XPU W2): the decode-path cost of the W4A16 int4
// expert kernel, on card 0, at the real MoE node shapes.  See W4A16-PLAN.md §3.3 and §6 (W2 step 4): this is
// the command whose number decides DW3 (whether a first-class S4-g128 format is ever worth building).
//
//     ZE_AFFINITY_MASK=0 build-sycl/w4a16_gemv_bench --pack <pack dir> [--layer 16] [--experts 10]
//                                                    [--iters 200] [--warmup 20]
//
// WHAT IS MEASURED, at H = 2560, FF = 640, ONE TOKEN, TEN EXPERTS (the shapes QUANT.md §5.2 measured for the
// Q4_K anchor, so the numbers are comparable):
//
//   1. the submission floor      N `cudaMemsetAsync` of 4 bytes on the same stream - what a call costs when
//                                the kernel body is nothing.  QUANT.md measured 12.30 us for a trivial-op graph;
//                                the decode kernel is only ~5.6 us above it, which is why this is measured
//                                first and printed next to every kernel number.
//   2. the grouped Q4_0 kernel   `native_expert_grouped`, ten groups of one entry - one call per LAYER, which
//                                is what the resident-expert hit path does (it groups the routed experts).
//   3. the per-column Q4_0 path  `native_q4_0_mmvq` gate, up and down per expert - the shape the kernel had
//                                before grouping, and the only Q4_0 expert kernel that existed until this card.
//   4. the CPU AVX2 path         `cpu::native_gu_rows`/`native_down_rows` (ggml-cpu's Q4_0 vec_dot), one
//                                thread, one expert - the miss path the GPU hits are compared against.
//   5. Q4_0 vs Q4_K at one shape `native_mmvq`'s Q4_K and Q4_0 single-column GEMVs on 2560x640 synthetic rows
//                                of both formats - the gap QUANT.md §3.3 named as UNKNOWN (Q4_0 was not in its
//                                table).
//
// Reported per arm: per call, per expert (bytes/expert = 2,764,800, so effective GB/s too) and the implied
// share of a token: the model routes 10 experts per layer over 48 layers = 480 expert instances per token.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

namespace {

constexpr int64_t H = 2560, FF = 640;
constexpr size_t kBlob = 2764800;
constexpr int64_t kLayers = 48, kExpertsPerToken = 10, kMaxExperts = 512;

double now_us() {
    using clk = std::chrono::steady_clock;
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count() * 1e-3;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

bool read_blob(const std::string& pack, int layer, int expert, uint64_t layer_off, std::vector<uint8_t>& out) {
    std::ifstream f(pack + "/experts.bin", std::ios::binary);
    if (!f) return false;
    out.resize(kBlob);
    f.seekg((std::streamoff) (layer_off + (uint64_t) expert * kBlob), std::ios::beg);
    f.read((char*) out.data(), (std::streamoff) kBlob);
    return (bool) f && f.gcount() == (std::streamoff) kBlob;
}

void print_usage() {
    std::printf("usage: ZE_AFFINITY_MASK=0 w4a16_gemv_bench --pack <dir> [--layer 16] [--experts 10]\n"
                "                                        [--iters 200] [--warmup 20]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string pack;
    int layer = 16, n_experts = 10, iters = 200, warmup = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) n_experts = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else { print_usage(); return 2; }
    }
    if (pack.empty() || n_experts < 1 || n_experts > kMaxExperts) { print_usage(); return 2; }

    // the pack's table: the layer's base offset and the blob size it declares
    uint64_t layer_off = 0, blob_bytes = 0;
    int gu_type = -1, d_type = -1;
    {
        std::ifstream lines(pack + "/native_experts.txt");
        if (!lines) { std::printf("--pack %s: native_experts.txt is not readable\n", pack.c_str()); return 1; }
        std::string l;
        while (std::getline(lines, l)) {
            if (l.empty() || l[0] == '#') continue;
            long long ly;
            int gt, dt;
            unsigned long long off, bb;
            if (std::sscanf(l.c_str(), "%lld %d %d %llu %llu", &ly, &gt, &dt, &off, &bb) != 5) continue;
            if (ly == layer) { layer_off = off; blob_bytes = bb; gu_type = gt; d_type = dt; }
        }
    }
    if (blob_bytes != kBlob || gu_type != 2) {
        std::printf("--pack %s layer %d: expected a Q4_0 (type 2) layer of %zu B, the table says type %d / %llu B\n",
                    pack.c_str(), layer, kBlob, gu_type, (unsigned long long) blob_bytes);
        return 1;
    }
    const K::NativeExpertLayout L = K::native_expert_layout(2, 2, H, FF);
    const K::NativeExpertLayout Lp = K::native_expert_layout(K::kQ4_0PinnedForm, K::kQ4_0PinnedForm, H, FF);

    std::printf("w4a16_gemv_bench: %d experts of layer %d, H %lld FF %lld, one token, %d iterations (+%d warmup)\n",
                n_experts, layer, (long long) H, (long long) FF, iters, warmup);
    std::printf("  bytes per expert %zu (gate %zu + up %zu + down %zu);  %d experts = %zu MiB resident\n", L.bytes,
                (size_t) FF * L.gu_row, (size_t) FF * L.gu_row, (size_t) H * L.d_row, n_experts,
                (size_t) n_experts * L.bytes >> 20);

    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");

    // ---- the weights: the REAL blobs of this layer, one per expert
    std::vector<uint8_t> host((size_t) n_experts * kBlob);
    for (int e = 0; e < n_experts; ++e) {
        std::vector<uint8_t> b;
        if (!read_blob(pack, layer, e, layer_off, b)) { std::printf("experts.bin is short at expert %d\n", e); return 1; }
        std::memcpy(host.data() + (size_t) e * kBlob, b.data(), kBlob);
    }
    uint8_t* d_w = nullptr;
    check(cudaMalloc((void**) &d_w, host.size()), "malloc weights");
    check(cudaMemcpy(d_w, host.data(), host.size(), cudaMemcpyHostToDevice), "copy weights");

    // the routed-expert metadata the hit path builds: one group per distinct expert, one entry each
    std::vector<unsigned long long> grp((size_t) n_experts);
    for (int e = 0; e < n_experts; ++e) grp[(size_t) e] = (unsigned long long) (d_w + (size_t) e * kBlob);
    std::vector<int32_t> start((size_t) n_experts + 1), dst((size_t) n_experts), tok((size_t) n_experts, 0);
    for (int e = 0; e <= n_experts; ++e) start[(size_t) e] = e;
    for (int e = 0; e < n_experts; ++e) dst[(size_t) e] = e;
    const int32_t count = n_experts;
    unsigned long long* d_grp = nullptr;
    int32_t *d_start = nullptr, *d_count = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    check(cudaMalloc((void**) &d_grp, grp.size() * 8), "malloc grp");
    check(cudaMalloc((void**) &d_start, start.size() * 4), "malloc start");
    check(cudaMalloc((void**) &d_count, 4), "malloc count");
    check(cudaMalloc((void**) &d_dst, dst.size() * 4), "malloc dst");
    check(cudaMalloc((void**) &d_tok, tok.size() * 4), "malloc tok");
    check(cudaMemcpy(d_grp, grp.data(), grp.size() * 8, cudaMemcpyHostToDevice), "copy grp");
    check(cudaMemcpy(d_start, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "copy start");
    check(cudaMemcpy(d_count, &count, 4, cudaMemcpyHostToDevice), "copy count");
    check(cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "copy dst");
    check(cudaMemcpy(d_tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "copy tok");

    float* d_x = nullptr;
    void* d_q8 = nullptr;
    void* d_q8s = nullptr;
    void* d_scratch = nullptr;
    float* d_out = nullptr;
    check(cudaMalloc((void**) &d_x, (size_t) H * 4), "malloc x");
    check(cudaMalloc(&d_q8, (size_t) H / 32 * 36), "malloc q8");
    check(cudaMalloc(&d_q8s, (size_t) H / 32 * 36), "malloc q8s");
    check(cudaMalloc(&d_scratch, K::native_expert_scratch_bytes(n_experts, FF)), "malloc scratch");
    check(cudaMalloc((void**) &d_out, (size_t) n_experts * H * 4), "malloc out");
    {
        std::vector<float> x((size_t) H);
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] = std::sin(0.01f * (float) i);   // finite, non-trivial
        check(cudaMemcpy(d_x, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice), "copy x");
    }

    // ---- the GPU arms
    struct Arm {
        const char* name;
        const K::NativeExpertLayout* L;
        const void* act;      // the q8_1 activation this arm reads (the sumq arm needs its own)
        bool hq_sumq;
        double us;
    };
    const K::NativeExpertLayout Ls = K::native_expert_layout(K::kQ4_0SumqForm, K::kQ4_0SumqForm, H, FF);
    Arm grouped[3] = {{"grouped  type 2   (exact, 2 dp4a)   ", &L, d_q8, false, 0.0},
                      {"grouped  type 102 (pinned, llama.cpp)", &Lp, d_q8, false, 0.0},
                      {"grouped  type 103 (exact, sum(q))    ", &Ls, d_q8s, true, 0.0}};

    // warmup + timed: iters calls on one stream, then one sync (a decode layer submits once and waits late)
    auto run_grouped = [&](Arm& arm) {
        for (int i = 0; i < warmup; ++i)
            K::native_expert_grouped(*arm.L, d_grp, d_start, d_count, d_dst, d_tok, n_experts, n_experts, arm.act,
                                     d_scratch, d_out, s, arm.hq_sumq);
        check(cudaStreamSynchronize(s), "sync warmup");
        const double t0 = now_us();
        for (int i = 0; i < iters; ++i)
            K::native_expert_grouped(*arm.L, d_grp, d_start, d_count, d_dst, d_tok, n_experts, n_experts, arm.act,
                                     d_scratch, d_out, s, arm.hq_sumq);
        check(cudaStreamSynchronize(s), "sync timed");
        arm.us = (now_us() - t0) / iters;
    };
    // the quantized activation is an input to the hit kernel: quantize it once, as the engine does per layer
    K::quantize_q8_1_rows((const float*) d_x, 1, H, d_q8, s);
    K::quantize_q8_1_rows_sumq((const float*) d_x, 1, H, d_q8s, s);
    check(cudaStreamSynchronize(s), "sync q8");
    for (int i = 0; i < 3; ++i) run_grouped(grouped[i]);
    // The three arms: type 2 and type 103 are BOTH meant to be the exact form (type 102 is the pinned
    // convention and is expected to differ).  type 103 gets its correction from sum(q) stored in the q8_1
    // block's fp16 `ds.y`, and fp16 holds integers exactly only to 2048 - so the deviation measured here is
    // reported WITH the largest |sum(q)| this activation produces, which is what decides whether the sum(q)
    // form is usable at all:
    {
        std::vector<float> out[3];
        for (int i = 0; i < 3; ++i) {
            K::native_expert_grouped(*grouped[i].L, d_grp, d_start, d_count, d_dst, d_tok, n_experts, n_experts,
                                     grouped[i].act, d_scratch, d_out, s, grouped[i].hq_sumq);
            check(cudaStreamSynchronize(s), "sync check");
            out[i].resize((size_t) n_experts * H);
            check(cudaMemcpy(out[i].data(), d_out, out[i].size() * 4, cudaMemcpyDeviceToHost), "read check");
        }
        auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double n = 0, d = 0;
            for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - (double) b[i]); d += std::fabs((double) b[i]); }
            return n / (d + 1e-30);
        };
        // the largest |sum(q)| per 32-value block of this activation (the codes are identical in both q8_1
        // buffers, only ds.y differs), i.e. the quantity fp16 must hold exactly
        int max_sumq = 0;
        {
            std::vector<int8_t> codes((size_t) H);
            std::vector<uint8_t> blk((size_t) H / 32 * 36);
            check(cudaMemcpy(blk.data(), d_q8, blk.size(), cudaMemcpyDeviceToHost), "read codes");
            for (int64_t k = 0; k < H / 32; ++k) {
                int s = 0;
                for (int j = 0; j < 32; ++j) s += ((const int8_t*) blk.data())[k * 36 + 4 + j];
                max_sumq = (std::max)(max_sumq, s < 0 ? -s : s);
            }
        }
        std::printf("  correctness of the timed arms: type 2 vs type 103 %.3e (both meant to be exact), "
                    "type 2 vs type 102 %.3e (different convention)\n", rel(out[0], out[2]), rel(out[0], out[1]));
        std::printf("  the activation's largest |sum(q)| per 32-value block: %d "
                    "(fp16 holds integers exactly to 2048)\n", max_sumq);
        if (!(rel(out[0], out[1]) > 1e-4)) {
            std::printf("  FAIL: the pinned convention is not distinguishable from the exact one - no power\n");
            return 1;
        }
    }

    // the per-column path: gate, up and down per expert, one call each
    double percol_us = 0;
    {
        std::vector<float> y((size_t) H);
        float* d_y = nullptr;
        check(cudaMalloc((void**) &d_y, (size_t) H * 4), "malloc y");
        void* d_hq = nullptr;
        check(cudaMalloc(&d_hq, (size_t) FF / 32 * 36), "malloc hq");
        std::vector<float> h((size_t) FF);
        for (int64_t i = 0; i < FF; ++i) h[(size_t) i] = 0.1f * (float) i;
        float* d_h = nullptr;
        check(cudaMalloc((void**) &d_h, (size_t) FF * 4), "malloc h");
        check(cudaMemcpy(d_h, h.data(), (size_t) FF * 4, cudaMemcpyHostToDevice), "copy h");
        K::native_quantize_q8_1(d_h, d_hq, (int) FF, 1, s);
        auto run = [&]() {
            for (int e = 0; e < n_experts; ++e) {
                const uint8_t* b = d_w + (size_t) e * kBlob;
                K::native_q4_0_mmvq(b, d_q8, d_y, (int) H, (int) FF, 1, s);
                K::native_q4_0_mmvq(b + L.up_off, d_q8, d_y, (int) H, (int) FF, 1, s);
                K::native_q4_0_mmvq(b + L.down_off, d_hq, d_y, (int) FF, (int) H, 1, s);
            }
        };
        for (int i = 0; i < warmup; ++i) run();
        check(cudaStreamSynchronize(s), "sync warmup percol");
        const double t0 = now_us();
        for (int i = 0; i < iters; ++i) run();
        check(cudaStreamSynchronize(s), "sync timed percol");
        percol_us = (now_us() - t0) / iters;
        cudaFree(d_y); cudaFree(d_h); cudaFree(d_hq);
    }

    // the submission floors: one API call per iteration with no work (memset), and one REAL kernel launch per
    // iteration with trivial work (a single 32-value q8_1 quantize block).  QUANT.md §5.4 measured 12.30 us for
    // its own trivial-op control; neither of these is that number, and both are printed so the kernel numbers
    // below can be read as "above the floor" rather than as pure kernel time.
    double floor_us = 0, kernel_floor_us = 0;
    {
        for (int i = 0; i < warmup; ++i) check(cudaMemsetAsync(d_out, 0, 4, s), "memset warmup");
        check(cudaStreamSynchronize(s), "sync warmup floor");
        const double t0 = now_us();
        for (int i = 0; i < iters; ++i) check(cudaMemsetAsync(d_out, 0, 4, s), "memset timed");
        check(cudaStreamSynchronize(s), "sync timed floor");
        floor_us = (now_us() - t0) / iters;

        for (int i = 0; i < warmup; ++i) K::quantize_q8_1_rows((const float*) d_x, 1, 32, d_q8, s);
        check(cudaStreamSynchronize(s), "sync warmup kfloor");
        const double t1 = now_us();
        for (int i = 0; i < iters; ++i) K::quantize_q8_1_rows((const float*) d_x, 1, 32, d_q8, s);
        check(cudaStreamSynchronize(s), "sync timed kfloor");
        kernel_floor_us = (now_us() - t1) / iters;
    }

    // ---- Q4_0 vs Q4_K, one 2560x640 GEMV each, synthetic rows of both formats (ggml's own quantizers)
    double q40_1col_us = 0, q4k_1col_us = 0;
    {
        std::vector<float> w((size_t) H * FF);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(0.001f * (float) i) * 0.05f;
        std::vector<uint8_t> q40((size_t) FF * (H / 32) * 18), q4k((size_t) FF * (H / 256) * 144);
        ggml_get_type_traits_cpu((ggml_type) 2)->from_float(w.data(), q40.data(), (int64_t) w.size());
        ggml_get_type_traits_cpu((ggml_type) 12)->from_float(w.data(), q4k.data(), (int64_t) w.size());
        uint8_t* d40 = nullptr;
        uint8_t* d4k = nullptr;
        float* d_y = nullptr;
        check(cudaMalloc((void**) &d40, q40.size()), "malloc q40");
        check(cudaMalloc((void**) &d4k, q4k.size()), "malloc q4k");
        check(cudaMalloc((void**) &d_y, (size_t) FF * 4), "malloc y");
        check(cudaMemcpy(d40, q40.data(), q40.size(), cudaMemcpyHostToDevice), "copy q40");
        check(cudaMemcpy(d4k, q4k.data(), q4k.size(), cudaMemcpyHostToDevice), "copy q4k");
        auto timeit = [&](auto fn) {
            for (int i = 0; i < warmup; ++i) fn();
            check(cudaStreamSynchronize(s), "sync warmup 1col");
            const double t0 = now_us();
            for (int i = 0; i < iters; ++i) fn();
            check(cudaStreamSynchronize(s), "sync timed 1col");
            return (now_us() - t0) / iters;
        };
        q40_1col_us = timeit([&] { K::native_q4_0_mmvq(d40, d_q8, d_y, (int) H, (int) FF, 1, s); });
        q4k_1col_us = timeit([&] { K::native_mmvq(12, d4k, d_q8, d_y, (int) H, (int) FF, 1, s); });
        std::printf("  control one 2560x640 GEMV, one column, synthetic ggml rows: Q4_0 %.2f us (%.0f GB/s), "
                    "Q4_K %.2f us (%.0f GB/s)\n", q40_1col_us, (double) q40.size() / q40_1col_us / 1e3,
                    q4k_1col_us, (double) q4k.size() / q4k_1col_us / 1e3);
        cudaFree(d40); cudaFree(d4k); cudaFree(d_y);
    }

    // ---- the CPU paths: the engine's OWN pool (row-split over physical cores) and one thread
    double cpu_node_pool_us = 0, cpu_node_1_us = 0, cpu_expert_us = 0;
    {
        cpu::NativeFmt fmt;
        std::string err;
        if (!cpu::native_fmt(2, 2, H, FF, fmt, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 1; }
        std::vector<float> x((size_t) H);
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] = std::sin(0.01f * (float) i);
        std::vector<uint8_t> act(fmt.act_bytes);
        cpu::native_quant_act(fmt, x.data(), act.data());
        std::vector<float> out((size_t) n_experts * H);
        std::vector<cpu::ExpertJobMulti> jobs((size_t) n_experts);
        for (int e = 0; e < n_experts; ++e) {
            jobs[(size_t) e].blob = host.data() + (size_t) e * kBlob;
            jobs[(size_t) e].nt = 1;
            jobs[(size_t) e].nact[0] = act.data();
            jobs[(size_t) e].out[0] = out.data() + (size_t) e * H;
        }
        auto bench_pool = [&](int workers, const char* what, int reps) {
            cpu::ExpertPool pool(workers, true, false);
            for (int i = 0; i < 2; ++i) pool.run_split_multi_native(fmt, jobs.data(), n_experts);
            const double t0 = now_us();
            for (int i = 0; i < reps; ++i) pool.run_split_multi_native(fmt, jobs.data(), n_experts);
            const double us = (now_us() - t0) / reps;
            std::printf("  cpu     %s: node %8.2f us  |  per expert %7.2f us  |  %4.0f GB/s\n", what, us,
                        us / n_experts, (double) kBlob / (us / n_experts) / 1e3);
            return us;
        };
        {
            const auto topo = cpu::detect_cpu_topology(true);
            std::printf("  cpu     topology: %zu worker cores (hybrid %d, p_cores %d, e_cores %d)\n",
                        topo.worker_cores.size(), (int) topo.is_hybrid, topo.p_cores, topo.e_cores);
        }
        cpu_node_pool_us = bench_pool(0, "the engine's pool (row-split, pinned)  ", 20);
        cpu_node_1_us = bench_pool(1, "one worker                          ", 10);
        cpu_expert_us = cpu_node_1_us / n_experts;
        // the single-expert, single-thread number the card asks for, with the timing loop around one expert
        std::vector<float> ff((size_t) FF);
        std::vector<uint8_t> hq(fmt.h_bytes);
        auto run_one = [&]() {
            const void* a[1] = {act.data()};
            float* f[1] = {ff.data()};
            cpu::native_gu_rows(fmt, host.data(), a, 1, f, 0, (int) FF);
            cpu::native_quant_h(fmt, ff.data(), hq.data());
            const void* h[1] = {hq.data()};
            float* o[1] = {out.data()};
            cpu::native_down_rows(fmt, host.data(), h, 1, o, 0, (int) H);
        };
        for (int i = 0; i < 3; ++i) run_one();
        const double t0 = now_us();
        for (int i = 0; i < 20; ++i) run_one();
        const double one = (now_us() - t0) / 20;
        std::printf("  cpu     one expert, one thread, AVX2 ggml-cpu vec_dot: %.1f us (%.2f GB/s), %.2f us per row "
                    "(%lld rows)\n", one, (double) kBlob / one / 1e3, one / (double) (2 * FF + H),
                    (long long) (2 * FF + H));
    }

    // ---- the report
    const double per_expert_grouped[3] = {grouped[0].us / n_experts, grouped[1].us / n_experts,
                                          grouped[2].us / n_experts};
    std::printf("\n  == one layer's routed experts: %d experts, one token (the QUANT.md §5.2 node shape) ==\n",
                n_experts);
    std::printf("  submission floor: memsetAsync %.2f us, one real kernel launch (1 quantize block) %.2f us, "
                "both per call on this stream\n", floor_us, kernel_floor_us);
    for (int i = 0; i < 3; ++i)
        std::printf("  %s node %8.2f us  |  per expert %7.2f us  |  %4.0f GB/s  |  %5.2f us above the kernel floor\n",
                    grouped[i].name, grouped[i].us, per_expert_grouped[i],
                    (double) kBlob / per_expert_grouped[i] / 1e3, per_expert_grouped[i] - kernel_floor_us);
    std::printf("  per-column Q4_0 path (3 mmvq calls per expert) node %8.2f us  |  per expert %7.2f us  |  "
                "%4.0f GB/s\n", percol_us, percol_us / n_experts, (double) kBlob / (percol_us / n_experts) / 1e3);

    // type 2 is the exact form and is the one a pack can name; type 103 is the same arithmetic clocked at the
    // pinned form's dp4a count, and it is only exact while every block's |sum(q)| stays <= 2048 (see the
    // diagnostic above) - so the headline number is type 2's, and 103's is printed next to it, labelled.
    const double gpu_exact = per_expert_grouped[0];
    std::printf("\n  == which is faster per expert ==\n");
    std::printf("  gpu hit  (grouped Q4_0, exact)  : %8.2f us/expert  ->  the engine's %lld experts/token over %lld "
                "layers (a fully resident cache) = %.1f ms/token = %.0f tok/s ceiling from the expert GEMVs alone\n",
                gpu_exact, (long long) kExpertsPerToken, (long long) kLayers,
                gpu_exact * (double) (kExpertsPerToken * kLayers) * 1e-3,
                1e6 / (gpu_exact * (double) (kExpertsPerToken * kLayers)));
    std::printf("  gpu hit  (type 103, sum(q) in fp16, exact only while |sum(q)| <= 2048): %8.2f us/expert\n",
                per_expert_grouped[2]);
    std::printf("  cpu miss (1 thread)             : %8.2f us/expert  ->  same 480 instances on ONE core = "
                "%.1f ms/token = %.0f tok/s ceiling\n", cpu_expert_us,
                cpu_expert_us * (double) (kExpertsPerToken * kLayers) * 1e-3,
                1e6 / (cpu_expert_us * (double) (kExpertsPerToken * kLayers)));
    std::printf("  cpu miss (the engine's pool)    : %8.2f us/expert  ->  same 480 instances on the pool = "
                "%.1f ms/token = %.0f tok/s ceiling\n", cpu_node_pool_us / n_experts,
                cpu_node_pool_us / n_experts * (double) (kExpertsPerToken * kLayers) * 1e-3,
                1e6 / (cpu_node_pool_us / n_experts * (double) (kExpertsPerToken * kLayers)));
    std::printf("  -> per expert, %s is faster: the Arc is %.1fx one core and %.2fx the engine's pool at this "
                "shape and format, on the exact (CPU-parity) arithmetic\n",
                gpu_exact < cpu_node_pool_us / n_experts ? "the GPU" : "the CPU pool", cpu_expert_us / gpu_exact,
                (cpu_node_pool_us / n_experts) / gpu_exact);

    // ---- the expert-cache arithmetic (what the format change costs the cache, W4A16-PLAN.md §2.4)
    {
        const double q40 = (double) kBlob, q20 = 1382400.0, s4 = 2534400.0;
        std::printf("\n  == expert cache: slots are fixed-size blobs, so the format sets the slot count ==\n");
        std::printf("  %-26s %12s %10s %14s %14s\n", "format", "B/expert", "GiB model", "slots @12 GiB", "@24 GiB");
        const char* names[3] = {"canonical Q2_0 (lossy)", "W4A16 Q4_0 (this pack)", "S4 g128 (DW3, exact)"};
        const double bytes[3] = {q20, q40, s4};
        for (int i = 0; i < 3; ++i)
            std::printf("  %-26s %12.0f %10.2f %14.0f %14.0f\n", names[i], bytes[i], bytes[i] * 24576.0 / 1073741824.0,
                        floor(12.0 * 1073741824.0 / bytes[i]), floor(24.0 * 1073741824.0 / bytes[i]));
        std::printf("  -> Q4_0 vs the 2,457,600 B/expert int4 source: +8.27%% bytes, -425 slots at 12 GiB (-7.6%%);\n"
                    "     S4 g128 would save 8.33%% and win back 424 slots.  Both figures are arithmetic on the\n"
                    "     measured blob sizes, not measurements of a cache.\n");
    }

    cudaFree(d_w); cudaFree(d_grp); cudaFree(d_start); cudaFree(d_count); cudaFree(d_dst); cudaFree(d_tok);
    cudaFree(d_x); cudaFree(d_q8); cudaFree(d_q8s); cudaFree(d_scratch); cudaFree(d_out);
    cudaStreamDestroy(s);
    return 0;
}
