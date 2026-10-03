// P8 probe: the host->device bandwidth of each Battlemage card, measured two ways -
//   1. queue.memcpy() from pinned host (USM malloc_host) to device memory: the copy-engine path the engine's own
//      cudaMemcpyAsync staging uses;
//   2. a device kernel reading the same pinned host buffer (USM device-mapped host memory): the path the engine's
//      copy_kernel uses for the expert arena.  This one forces the SMs to read system memory, i.e. PCIe.
// Both are real transfers; the number either agrees with a gen1 x1 link (~0.25 GB/s) or it does not.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sycl;

static const char* kind_name(void* p, queue& q) {
    switch (get_pointer_type(p, q.get_context())) {
        case usm::alloc::host: return "host";
        case usm::alloc::device: return "device";
        case usm::alloc::shared: return "shared";
        default: return "unknown";
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // a pipe is block-buffered; a crash must not eat the numbers
    const size_t bytes = (argc > 1 ? (size_t) atoll(argv[1]) : (size_t) 256) << 20;
    const int iters = argc > 2 ? atoi(argv[2]) : 4;
    std::vector<device> devs = device::get_devices(info::device_type::gpu);
    printf("devices: %zu\n", devs.size());
    for (size_t i = 0; i < devs.size(); ++i)
        printf("  [%zu] %s | %s | max_work_group=%zu, global_mem=%zu MiB\n", i,
               devs[i].get_info<info::device::name>().c_str(),
               devs[i].get_info<info::device::driver_version>().c_str(),
               devs[i].get_info<info::device::max_work_group_size>(),
               devs[i].get_info<info::device::global_mem_size>() >> 20);
    if (devs.empty()) return 1;

    for (size_t di = 0; di < devs.size(); ++di) {
        queue q(devs[di]);
        printf("\n=== device %zu: %s\n", di, q.get_device().get_info<info::device::name>().c_str());
        void* h = malloc_host(bytes, q);
        void* d = malloc_device(bytes, q);
        if (!h || !d) { printf("  allocation failed (host=%p device=%p) - skipped\n", h, d); continue; }
        memset(h, 1, bytes);

        // warmup
        q.memcpy(d, h, bytes).wait();
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) q.memcpy(d, h, bytes);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("  memcpy  H2D %zu MiB x%d in %.3f s -> %.2f GB/s  (host ptr type: %s)\n",
               bytes >> 20, iters, s, (double) iters * bytes / s / 1e9, kind_name(h, q));

        // kernel path: the SMs read the pinned host buffer (a fixed small slice: this test is a latency-bound lower
        // bound, not a bandwidth number)
        const size_t kbytes = bytes < (16ull << 20) ? bytes : (16ull << 20);
        int* out = (int*) malloc_device(sizeof(unsigned long long), q);
        auto t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            q.submit([&](handler& hd) {
                const char* p = (const char*) h;
                unsigned long long* o = (unsigned long long*) out;
                hd.parallel_for(range<1>(1), [=](id<1>) {
                    unsigned long long acc = 0;
                    for (size_t k = 0; k < kbytes; ++k) acc += (unsigned char) p[k];
                    *o = acc;
                });
            });
        }
        q.wait();
        auto t3 = std::chrono::steady_clock::now();
        s = std::chrono::duration<double>(t3 - t2).count();
        printf("  kernel  reads host %zu MiB x%d in %.3f s -> %.2f GB/s (1 thread, latency-bound; the point is that "
               "a device read of host memory over this link is not ~0.25 GB/s)\n",
               kbytes >> 20, iters, s, (double) iters * kbytes / s / 1e9);
        free(h, q);
        free(d, q);
        free(out, q);
    }
    return 0;
}
