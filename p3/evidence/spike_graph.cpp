// p3/spike_graph.cpp - does oneAPI 2026.1's SYCL graph extension (command_graph) actually work on the
// Arc Pro B70, with the kernel shapes this shim submits (1-D nd_range, unnamed lambda, dynamic
// local_accessor, USM pointers), and does a replay of a many-kernel graph cost less than submitting the
// same kernels one by one?
//
// This is a SPIKE: it decides whether the engine's graph-replay equivalent can be built on top of the
// SYCL graph extension.  Nothing here ships.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace g = sycl::ext::oneapi::experimental;

static double now_ms() {
    using C = std::chrono::steady_clock;
    static const C::time_point t0 = C::now();
    return std::chrono::duration<double, std::milli>(C::now() - t0).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    const int NK = argc > 1 ? std::atoi(argv[1]) : 256;      // kernels per "window"
    const int REP = argc > 2 ? std::atoi(argv[2]) : 200;     // replays
    const bool use_smem = argc > 3 ? std::atoi(argv[3]) != 0 : true;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("backend: %s\n", q.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "other");
    std::printf("graph recording support: %d\n",
                (int) q.get_device().has(sycl::aspect::ext_oneapi_graph));

    constexpr size_t N = 4096;
    const size_t bytes = N * sizeof(float);
    float* a = sycl::malloc_device<float>(N, q);
    float* b = sycl::malloc_device<float>(N, q);
    float* c = sycl::malloc_device<float>(N, q);
    std::vector<float> h(N), hb(N);
    for (size_t i = 0; i < N; ++i) hb[i] = (float) (i % 17) * 0.25f;
    q.memcpy(a, h.data(), bytes);
    q.memcpy(b, hb.data(), bytes);
    q.wait();

    // one "window" = NK kernels, each a 1-D nd_range launch like the shim emits
    const size_t threads = 256;
    const size_t groups = (N + threads - 1) / threads;
    auto submit_one = [&](sycl::queue& qq, float* dst, float* s1, float* s2, int k) {
        if (use_smem) {
            qq.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 1> dyn{sycl::range<1>(threads), h};
                h.parallel_for(sycl::nd_range<1>{groups * threads, threads}, [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    float* s = dyn.get_multi_ptr<sycl::access::decorated::no>().get();
                    s[item.get_local_id(0)] = s1[i];
                    sycl::group_barrier(item.get_group());
                    dst[i] = s[(item.get_local_id(0) + (size_t) k) % threads] * 1.0f + s2[i] * 0.0f;
                });
            });
        } else {
            qq.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>{groups * threads, threads},
                               [=](sycl::nd_item<1> item) { dst[item.get_global_id(0)] = s1[item.get_global_id(0)]; });
            });
        }
    };

    // ---- arm A: plain submission, NK kernels per window, REP windows
    double t0 = now_ms();
    for (int r = 0; r < REP; ++r)
        for (int k = 0; k < NK; ++k) submit_one(q, c, a, b, k);
    q.wait();
    const double t_plain = now_ms() - t0;
    q.memcpy(h.data(), c, bytes).wait();
    std::vector<float> h_plain = h;

    // ---- arm B: record the same NK kernels into a command_graph, then replay it REP times
    g::command_graph<g::graph_state::modifiable> graph(q.get_context(), q.get_device());
    const double t_rec0 = now_ms();
    try {
        graph.begin_recording(q);
        q.memset(c, 0, bytes);            // the shim submits memsets inside the window capture too
        q.memcpy(c, b, bytes);            // and copies
        for (int k = 0; k < NK; ++k) submit_one(q, c, a, b, k);
        graph.end_recording(q);
    } catch (const std::exception& e) {
        std::printf("RECORD FAILED: %s\n", e.what());
        return 1;
    }
    const double t_record = now_ms() - t_rec0;
    const double t_inst0 = now_ms();
    g::command_graph<g::graph_state::executable> exec = graph.finalize();
    const double t_inst = now_ms() - t_inst0;
    std::printf("recording %d kernels: %.1f ms (%.1f us/kernel)\n", NK, t_record, 1000.0 * t_record / NK);
    std::printf("finalize: %.1f ms\n", t_inst);

    t0 = now_ms();
    for (int r = 0; r < REP; ++r) q.submit([&](sycl::handler& h) { h.ext_oneapi_graph(exec); });
    q.wait();
    const double t_graph = now_ms() - t0;
    q.memcpy(h.data(), c, bytes).wait();

    // ---- compare
    double maxabs = 0.0;
    size_t diff = 0;
    for (size_t i = 0; i < N; ++i) {
        const double d = std::fabs((double) h[i] - (double) h_plain[i]);
        if (d > maxabs) maxabs = d;
        if (d != 0.0) ++diff;
    }
    std::printf("plain: %.1f ms total, %.3f us/kernel-submit-window (%.1f us/window)\n", t_plain,
                1000.0 * t_plain / (NK * REP), 1000.0 * t_plain / REP);
    std::printf("graph: %.1f ms total, %.3f us/window\n", t_graph, 1000.0 * t_graph / REP);
    std::printf("speedup: %.2fx  (smem=%d)\n", t_plain / t_graph, (int) use_smem);
    std::printf("token-id-style equality: maxabs=%.3e, differing elements=%zu of %zu\n", maxabs, diff, N);

    // ---- a second replay must also be correct (the steady-state decode claim)
    q.memset(c, 0, bytes);
    q.submit([&](sycl::handler& h) { h.ext_oneapi_graph(exec); });
    q.wait();
    q.memcpy(h.data(), c, bytes).wait();
    maxabs = 0.0;
    for (size_t i = 0; i < N; ++i) maxabs = std::max(maxabs, std::fabs((double) h[i] - (double) h_plain[i]));
    std::printf("second replay after a memset: maxabs=%.3e\n", maxabs);

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return 0;
}
