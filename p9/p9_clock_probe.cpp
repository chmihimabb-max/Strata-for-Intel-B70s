// P9: which device clock can a kernel read on this Arc B70 under SYCL/DPC++?
//
// STRATA_VERIFY_PROFILE's stamp kernel uses sycl::ext::oneapi::experimental::clock<clock_scope::device>(),
// and the engine dies on it with "Required aspect ext_oneapi_clock_device is not supported on the device".
// This probe asks the device directly: which aspects it reports, and which of three clock spellings
// actually run.  Variant 4 measures the tick rate against a known busy loop.
//
// build: source /opt/intel/oneapi/setvars.sh && icpx -fsycl -O2 p9_clock_probe.cpp -o p9_clock_probe
// run:   ./p9_clock_probe            (both cards: ZE_AFFINITY_MASK unset)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <sycl/sycl.hpp>

namespace exp_oneapi = sycl::ext::oneapi::experimental;

// 1 = the oneAPI extension clock (what the engine uses today)
static uint64_t clk_oneapi() {
    return (uint64_t) exp_oneapi::clock<exp_oneapi::clock_scope::device>();
}

// 2 = the OpenCL C builtin clock()
static uint64_t clk_builtin() { return (uint64_t) clock(); }

int main() {
    for (const auto& d : sycl::device::get_devices()) {
        std::printf("device: %s\n", d.get_info<sycl::info::device::name>().c_str());
        for (const char* a : {"ext_oneapi_clock_device", "ext_oneapi_graph", "fp64", "usm_device_allocations"}) {
            bool has = false;
            std::string s = a;
            if (s == "ext_oneapi_clock_device") has = d.has(sycl::aspect::ext_oneapi_clock_device);
            else if (s == "ext_oneapi_graph") has = d.has(sycl::aspect::ext_oneapi_graph);
            else if (s == "fp64") has = d.has(sycl::aspect::fp64);
            else if (s == "usm_device_allocations") has = d.has(sycl::aspect::usm_device_allocations);
            std::printf("  aspect %-26s %s\n", a, has ? "YES" : "no");
        }
        std::printf("  max_clock_frequency (MHz): %u\n",
                    d.get_info<sycl::info::device::max_clock_frequency>());
    }

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("queue device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    uint64_t* buf = sycl::malloc_device<uint64_t>(8, q);
    q.memset(buf, 0, 8 * sizeof(uint64_t));

    // the OpenCL builtin spelling, three launches, so a difference can be read
    try {
        q.submit([&](sycl::handler& h) { h.single_task([=]() { buf[0] = clk_builtin(); }); }).wait();
        // a busy loop between two reads: gives the tick rate of whatever counter this is
        q.submit([&](sycl::handler& h) {
            h.single_task([=]() {
                const uint64_t a = clk_builtin();
                float x = 1.0f;
                for (int i = 0; i < 20000000; ++i) x = x * 1.0000001f + 1e-7f;
                const uint64_t b = clk_builtin();
                buf[1] = a; buf[2] = b; buf[3] = (uint64_t) (x != 0.0f);
            });
        }).wait();
        uint64_t h[4] = {0, 0, 0, 0};
        q.memcpy(h, buf, sizeof h).wait();
        std::printf("clock() builtin: OK   first %llu  a %llu  b %llu  delta %llu ticks for ~20M FMA\n",
                    (unsigned long long) h[0], (unsigned long long) h[1], (unsigned long long) h[2],
                    (unsigned long long) (h[2] - h[1]));
    } catch (const sycl::exception& e) {
        std::printf("clock() builtin: THREW: %s\n", e.what());
    }

    // the oneAPI extension spelling the engine uses
    try {
        q.submit([&](sycl::handler& h) { h.single_task([=]() { buf[4] = clk_oneapi(); }); }).wait();
        uint64_t v = 0;
        q.memcpy(&v, buf + 4, 8).wait();
        std::printf("ext oneapi clock<device>: OK  %llu\n", (unsigned long long) v);
    } catch (const sycl::exception& e) {
        std::printf("ext oneapi clock<device>: THREW: %s\n", e.what());
    }
    return 0;
}
