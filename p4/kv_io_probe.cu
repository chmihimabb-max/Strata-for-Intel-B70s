// p4/kv_io_probe.cu - P4 (card t_63cc226b): the port's GPU-assisted I/O mechanism measured on this box.
//
// The paper's GPU-assisted I/O is a kernel whose threads each move a small chunk between GPU memory and REGISTERED
// PINNED host memory, with a block quota that confines it to a few SMs (2 blocks H2D / 1 D2H) to bound its
// interference with concurrent prefill and decode.  The port's mechanism is
// `copy_kernel` (src/kernels/cuda/kv_stream.cu:161), launched from `kv_stream_resolve` into the decode window.
//
// This probe measures THE SAME KERNEL (linked from strata_kernels, not a re-implementation) on the paper's axes:
//   1 bw     - sustained H2D and D2H GB/s against block count, 1..128 blocks x 1024 threads/block;
//   2 chunk  - transfer size 128 B .. 1 MiB on the scattered path, against the port's real block shape;
//   3 layout - the identity host layout vs a page-first (packed) host layout at the same page size;
//   4 inter  - the I/O kernel concurrent with the engine's own prompt-attention (prefill-like) and quantized
//              expert GEMV (decode-like) kernels, normalized throughput vs block count;
//   5 host   - the host-side scattered copy, the trap the mechanism exists to defeat.
//
// Nothing here ships to the engine: `kv_stream_copy_probe` is a probe entry point beside the engine's own path.
// Sizes and directions are command-line selectable; every measurement prints one raw `P4 ...` line.
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace k = strata::kernels;

namespace {

double now_s() {
    using C = std::chrono::steady_clock;
    static const C::time_point t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "P4 probe: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

const char* msg() { return cudaGetErrorString(cudaGetLastError()); }

// ---- the two kinds of memory the mechanism moves between ------------------------------------------------
// pinned, device-mapped host memory (what the engine's cudaHostAlloc(Mapped) resolves to: sycl::malloc_host)
uint8_t* pinned(size_t nbytes) {
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, nbytes + 4096, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, nbytes + 4096);
    ck(cudaHostGetDevicePointer(&d, h, 0), "hostgetdevptr");
    return (uint8_t*) d;
}
uint8_t* device(size_t nbytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, nbytes + 4096), "malloc");
    ck(cudaMemset(p, 0, nbytes + 4096), "memset");
    return (uint8_t*) p;
}

// ---- the probe's own kernel: the transfer-size instrument --------------------------------------------------
// One WARP per scattered chunk, so the accesses inside a chunk are contiguous (the engine's copy kernel has each
// thread write ONE uint4 at a thread-stripe offset, i.e. a warp's 32 stores are 512 contiguous bytes).  A
// thread-per-chunk version was measured first and is kept below: with 16 B stores scattered 32 ways, this link
// delivers 0.10 GB/s (see `chunk_threadper_kernel`), so the access pattern inside a block, not just the transfer
// size, decides the rate.
__global__ void chunk_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, long long n_chunks,
                             int chunk, long long stride) {
    const int lanes = 32;
    const long long tid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long nwarps = ((long long) gridDim.x * blockDim.x) / lanes;
    const int lane = (int) (tid % lanes);
    for (long long j = tid / lanes; j < n_chunks; j += nwarps) {
        const uint4* s = reinterpret_cast<const uint4*>(src + j * stride);
        uint4* d = reinterpret_cast<uint4*>(dst + j * stride);
        for (int i = lane; i < chunk / 16; i += lanes) d[i] = s[i];
    }
}

// The same sweep with ONE THREAD per chunk (each thread walks its own chunk): the access pattern a naive
// thread-per-chunk gather produces.  Reference point, not the mechanism's shape.
__global__ void chunk_threadper_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                       long long n_chunks, int chunk, long long stride) {
    for (long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x; j < n_chunks;
         j += (long long) gridDim.x * blockDim.x) {
        const uint4* s = reinterpret_cast<const uint4*>(src + j * stride);
        uint4* d = reinterpret_cast<uint4*>(dst + j * stride);
        for (int i = 0; i < chunk / 16; ++i) d[i] = s[i];
    }
}

// ---- the fixture: a streamed QSA layer's KV, real geometry, int8 ----------------------------------------
struct Fixture {
    k::QsaShapes s = k::qsa_real_shapes();
    int fmt = k::kKvInt8;
    int64_t pages = 0;      // blocks in the host copy and in the device slot pool
    int64_t block_bytes = 0;
    int64_t run_bytes[4] = {0, 0, 0, 0};
    int64_t run_off[4] = {0, 0, 0, 0};
    int n_runs = 4;

    k::KvHostPools host{};        // pinned device-mapped pointer bundle (identity layout)
    k::QsaAttnPools slots{};      // device slot pool (identity layout)
    k::KvStreamMap map{};
    int32_t* ident = nullptr;     // a page table for the readers in the interference arm

    void init(int64_t n_pages) {
        pages = n_pages;
        k::kv_block_run_offsets(s, fmt, run_off, &block_bytes);
        const int64_t rows = s.n_head_kv * s.page_size;
        run_bytes[0] = run_bytes[1] = rows * s.head_dim;
        run_bytes[2] = run_bytes[3] = rows * (s.head_dim / k::KV_Q8_GROUP) * 2;
        n_runs = 4;
        const size_t codes = (size_t) pages * run_bytes[0];
        const size_t scales = (size_t) pages * run_bytes[2];
        host.k_q = (int8_t*) pinned(codes); host.v_q = (int8_t*) pinned(codes);
        host.k_scale = (uint16_t*) pinned(scales); host.v_scale = (uint16_t*) pinned(scales);
        slots.k_q = (const int8_t*) device(codes); slots.v_q = (const int8_t*) device(codes);
        slots.k_scale = (const uint16_t*) device(scales); slots.v_scale = (const uint16_t*) device(scales);
        // a non-zero pattern in the host copy: the copy must be visible, not just fast
        for (size_t i = 0; i < codes; ++i) ((int8_t*) host.k_q)[i] = (int8_t) (i * 7 + 3);
        for (size_t i = 0; i < codes; ++i) ((int8_t*) host.v_q)[i] = (int8_t) (i * 11 + 5);
        for (size_t i = 0; i < scales / 2; ++i) ((uint16_t*) host.k_scale)[i] = (uint16_t) (i * 13 + 7);
        for (size_t i = 0; i < scales / 2; ++i) ((uint16_t*) host.v_scale)[i] = (uint16_t) (i * 17 + 9);
        map.page_table = (int32_t*) device((size_t) pages * 4);
        map.slot_block = (int32_t*) device((size_t) pages * 4);
        map.slot_stamp = (int32_t*) device((size_t) pages * 4);
        map.slot_ref = (int32_t*) device((size_t) pages * 4);
        map.miss_block = (int32_t*) device((size_t) pages * 4);
        map.miss_slot = (int32_t*) device((size_t) pages * 4);
        map.ctl = (int32_t*) device((size_t) k::kKvCtlInts * 4);
        map.n_blocks = pages;
        map.n_slots = pages;
        ident = (int32_t*) device((size_t) pages * 4);
        std::vector<int32_t> t((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) t[(size_t) i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), t.size() * 4, cudaMemcpyHostToDevice), "ident");
        slots.page_table = ident;
        // the miss list: every block, into its own slot
        std::vector<int32_t> mv((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) mv[(size_t) i] = (int32_t) i;
        ck(cudaMemcpy(map.miss_block, mv.data(), mv.size() * 4, cudaMemcpyHostToDevice), "miss_block");
        ck(cudaMemcpy(map.miss_slot, mv.data(), mv.size() * 4, cudaMemcpyHostToDevice), "miss_slot");
        // the copy kernel reads its work list size from ctl[2]
        const int32_t need = (int32_t) pages;
        ck(cudaMemcpy((int32_t*) map.ctl + 2, &need, 4, cudaMemcpyHostToDevice), "ctl[2]");
    }
    int64_t bytes_per_pass() const { return pages * block_bytes; }
};

// one timed pass group: `passes` launches of the real copy kernel, adaptively grown until `floor_s` is reached
struct Rate {
    double gbs = 0, seconds = 0;
    long long bytes = 0;
    int passes = 0;
};

Rate run_copy(Fixture& f, int dir, int packed, int blocks, int threads, double floor_s, int max_passes = 64) {
    Rate r;
    cudaStream_t st = nullptr;
    const double t0 = now_s();
    while (r.passes < max_passes) {
        k::kv_stream_copy_probe(f.map, f.slots, f.host, f.fmt, f.s, dir, packed, blocks, threads, st);
        ++r.passes;
        ck(cudaStreamSynchronize(st), "copy sync");
        r.seconds = now_s() - t0;
        if (r.seconds >= floor_s) break;
    }
    r.bytes = (long long) f.bytes_per_pass() * r.passes;
    r.gbs = (double) r.bytes / r.seconds / 1.0e9;
    return r;
}

// ---- 1: bandwidth vs concurrency ------------------------------------------------------------------------
void section_bw(Fixture& f) {
    const int sweep[] = {1, 2, 4, 8, 16, 32, 64, 96, 128};
    for (int which = 0; which < 2; ++which) {
        const int dir = which;
        const char* dn = dir ? "D2H" : "H2D";
        for (int b : sweep) {
            const Rate r = run_copy(f, dir, 0, b, 1024, 0.15);
            std::printf("P4 bw dir=%s blocks=%d threads=1024 passes=%d bytes=%lld ms=%.3f GBs=%.2f\n", dn, b,
                        r.passes, r.bytes, r.seconds * 1000.0, r.gbs);
            std::fflush(stdout);
        }
        for (int th : {128, 256, 512, 1024}) {   // fix the block count at the shipped 96, move the threads
            const Rate r = run_copy(f, dir, 0, 96, th, 0.15);
            std::printf("P4 bw dir=%s blocks=96 threads=%d passes=%d bytes=%lld ms=%.3f GBs=%.2f\n", dn, th,
                        r.passes, r.bytes, r.seconds * 1000.0, r.gbs);
            std::fflush(stdout);
        }
    }
}

// ---- 2: transfer size -----------------------------------------------------------------------------------
void section_chunk() {
    const int64_t total = 128LL << 20;   // 128 MiB moved per shape
    for (int dir = 0; dir < 2; ++dir) {
        for (int chunk : {128, 512, 4096, 65536, 262144, 1048576}) {
            const int64_t n_chunks = total / chunk;
            const int64_t span = n_chunks * (int64_t) chunk * 2;   // room for a hole as large as the chunk
            uint8_t* ds = device((size_t) span);
            uint8_t* ps = pinned((size_t) span);
            uint8_t* pd = pinned((size_t) span);
            const uint8_t* src = dir == 0 ? ps : ds;   // 0 = H2D (the fill direction), 1 = D2H (the append path)
            uint8_t* dst = dir == 0 ? ds : pd;
            for (int stride_k = 1; stride_k <= 2; ++stride_k) {     // 1 = dense, 2 = scattered (chunk + hole)
                const long long stride = (long long) chunk * stride_k;
                const int blocks = 32, threads = 1024;
                // WARM-UP: a shape's first launch pays the device code load, which is ~1.2 s here and would otherwise
                // be the whole measurement (measured: every size read 1.28 s before this was added).
                chunk_kernel<<<blocks, threads, 0, nullptr>>>(src, dst, n_chunks, chunk, stride);
                ck(cudaDeviceSynchronize(), "chunk warmup");
                double best = 1e30, sum = 0;
                int passes = 0;
                while (passes < 12) {
                    const double t0 = now_s();
                    chunk_kernel<<<blocks, threads, 0, nullptr>>>(src, dst, n_chunks, chunk, stride);
                    ck(cudaDeviceSynchronize(), "chunk");
                    const double el = now_s() - t0;
                    ++passes;
                    sum += el;
                    best = std::min(best, el);
                    if (sum >= 0.10 && passes >= 5) break;
                }
                const double bytes = (double) n_chunks * chunk;   // per pass
                std::printf("P4 chunk dir=%s bytes=%d stride=%lld blocks=%d threads=%d passes=%d"
                            " moved_per_pass=%lld best_ms=%.3f GBs_best=%.2f mean_ms=%.3f GBs_mean=%.2f\n",
                            dir ? "D2H" : "H2D", chunk, stride, blocks, threads, passes, (long long) bytes,
                            best * 1000.0, bytes / best / 1.0e9, sum / passes * 1000.0,
                            bytes / (sum / passes) / 1.0e9);
                std::fflush(stdout);
            }
            cudaFreeHost(pd);
            cudaFreeHost(ps);
            cudaFree(ds);
        }
    }
}

// ---- diagnostic: which side is slow (the transfer-size sweep hit a constant ~0.1 GB/s ceiling) ----------------
void section_diag(Fixture& f) {
    const int64_t mb = 1 << 20;
    for (int chunk : {4096, 1048576}) {
        const int64_t n_chunks = (64 * mb) / chunk;
        const int64_t span = n_chunks * (int64_t) chunk * 2;
        uint8_t* ds = device((size_t) span);
        uint8_t* ps = pinned((size_t) span);
        uint8_t* pd = pinned((size_t) span);
        struct V { const char* name; uint8_t* src; uint8_t* dst; };
        const V vs[] = {{"dev_to_pinned", ds, pd}, {"dev_to_dev", ds, (uint8_t*) device((size_t) span)},
                        {"pinned_to_dev", ps, ds}, {"pinned_to_pinned", ps, pd}};
        for (const V& v : vs) {
            for (int blocks : {32, 160}) {
                chunk_kernel<<<blocks, 1024, 0, nullptr>>>(v.src, v.dst, n_chunks, chunk, chunk);
                ck(cudaDeviceSynchronize(), "diag warm");
                const double t0 = now_s();
                chunk_kernel<<<blocks, 1024, 0, nullptr>>>(v.src, v.dst, n_chunks, chunk, chunk);
                ck(cudaDeviceSynchronize(), "diag");
                const double el = now_s() - t0;
                std::printf("P4 diag case=%s chunk=%d blocks=%d moved_MiB=64 ms=%.3f GBs=%.2f\n", v.name, chunk,
                            blocks, el * 1000.0, (double) n_chunks * chunk / el / 1.0e9);
                std::fflush(stdout);
            }
        }
        // the same ends with one THREAD per chunk (the access pattern the warp form avoids)
        for (int blocks : {32, 160}) {
            chunk_threadper_kernel<<<blocks, 1024, 0, nullptr>>>(ds, pd, n_chunks, chunk, chunk);
            ck(cudaDeviceSynchronize(), "diag warm");
            const double t0 = now_s();
            chunk_threadper_kernel<<<blocks, 1024, 0, nullptr>>>(ds, pd, n_chunks, chunk, chunk);
            ck(cudaDeviceSynchronize(), "diag");
            const double el = now_s() - t0;
            std::printf("P4 diag case=dev_to_pinned_threadper chunk=%d blocks=%d moved_MiB=64 ms=%.3f GBs=%.2f\n",
                        chunk, blocks, el * 1000.0, (double) n_chunks * chunk / el / 1.0e9);
            std::fflush(stdout);
        }
        // and the engine's own kernel on the same two ends, for the control
        const Rate rh = run_copy(f, 0, 0, 96, 128, 0.10);
        const Rate rd = run_copy(f, 1, 0, 96, 128, 0.10);
        std::printf("P4 diag engine_copy H2D GBs=%.2f D2H GBs=%.2f (96x128)\n", rh.gbs, rd.gbs);
        std::fflush(stdout);
        cudaFreeHost(ps);
        cudaFreeHost(pd);
        cudaFree(ds);
    }
}
void section_layout(Fixture& f) {
    uint8_t* packed_host = pinned((size_t) f.pages * f.block_bytes);
    uint8_t* packed_dev = device((size_t) f.pages * f.block_bytes);
    // fill the packed host image from the identity host copy's own bytes, one block at a time
    {
        const uint8_t* base[4] = {(const uint8_t*) f.host.k_q, (const uint8_t*) f.host.v_q,
                                  (const uint8_t*) f.host.k_scale, (const uint8_t*) f.host.v_scale};
        for (int64_t b = 0; b < f.pages; ++b)
            for (int a = 0; a < 4; ++a)
                std::memcpy(packed_host + b * f.block_bytes + f.run_off[a], base[a] + b * f.run_bytes[a],
                            (size_t) f.run_bytes[a]);
        ck(cudaMemcpy(packed_dev, packed_host, (size_t) f.pages * f.block_bytes, cudaMemcpyHostToDevice), "packed h2d");
    }
    struct Arm {
        const char* name;
        int packed;
        k::KvHostPools host;
        k::QsaAttnPools slots;
    };
    std::vector<Arm> arms;
    arms.push_back({"identity", 0, f.host, f.slots});
    {   // page-first host, identity device: the layout change the engine COULD adopt
        k::KvHostPools h = f.host;
        h.k_q = (int8_t*) packed_host;
        arms.push_back({"packed_host", 1, h, f.slots});
    }
    {   // page-first both sides: the pure page-first copy (upper bound), device layout NOT read by the engine
        k::KvHostPools h = f.host;
        h.k_q = (int8_t*) packed_host;
        k::QsaAttnPools p = f.slots;
        p.k_q = (const int8_t*) packed_dev;
        arms.push_back({"packed_both", 3, h, p});
    }
    for (int dir = 0; dir < 2; ++dir) {
        for (const Arm& a : arms) {
            for (auto shape : {std::pair<int, int>{32, 1024}, std::pair<int, int>{96, 128}}) {
                Rate r;
                cudaStream_t st = nullptr;
                const double t0 = now_s();
                int passes = 0;
                while (passes < 64) {
                    k::kv_stream_copy_probe(f.map, a.slots, a.host, f.fmt, f.s, dir, a.packed, shape.first,
                                            shape.second, st);
                    ++passes;
                    ck(cudaStreamSynchronize(st), "layout sync");
                    r.seconds = now_s() - t0;
                    if (r.seconds >= 0.15) break;
                }
                r.passes = passes;
                r.bytes = (long long) f.bytes_per_pass() * passes;
                r.gbs = (double) r.bytes / r.seconds / 1.0e9;
                std::printf("P4 layout tag=%s dir=%s blocks=%d threads=%d passes=%d bytes=%lld ms=%.3f GBs=%.2f\n",
                            a.name, dir ? "D2H" : "H2D", shape.first, shape.second, r.passes, r.bytes,
                            r.seconds * 1000.0, r.gbs);
                std::fflush(stdout);
            }
        }
    }
    cudaFreeHost(packed_host);
    cudaFree(packed_dev);
}

// ---- correctness: the probe's copy really moves the bytes, in both layouts and both directions -----------
// Without this the curves could be measuring a copy that does nothing.  Each case poisons the DESTINATION, runs the
// probe's own copy kernel, and compares the destination against the expected image - de-packing the packed host
// block with the same offsets the kernel used.
void section_verify(Fixture& f) {
    const size_t codes = (size_t) f.pages * f.run_bytes[0];
    const uint8_t* idbase[4] = {(const uint8_t*) f.host.k_q, (const uint8_t*) f.host.v_q,
                                (const uint8_t*) f.host.k_scale, (const uint8_t*) f.host.v_scale};
    uint8_t* packed_host = pinned((size_t) f.pages * f.block_bytes);
    std::vector<int8_t> p1(codes), got(codes);
    for (size_t i = 0; i < codes; ++i) p1[i] = (int8_t) (i * 7 + 3);
    // the expected K image is p1 for H2D; the D2H cases first put a DIFFERENT pattern in the device slots
    std::vector<int8_t> p2(codes);
    for (size_t i = 0; i < codes; ++i) p2[i] = (int8_t) (i * 5 + 1);

    struct Case { const char* name; int dir; int packed; const std::vector<int8_t>* expect; };
    const Case cases[] = {{"identity_h2d", 0, 0, &p1}, {"packed_h2d", 0, 1, &p1},
                          {"identity_d2h", 1, 0, &p2}, {"packed_d2h", 1, 1, &p2}};
    for (const Case& c : cases) {
        // the host side starts as p1 (in both layouts)
        ck(cudaMemcpy((void*) f.host.k_q, p1.data(), codes, cudaMemcpyHostToDevice), "host p1");
        for (int64_t b = 0; b < f.pages; ++b)
            for (int a = 0; a < 4; ++a)
                std::memcpy(packed_host + (size_t) b * f.block_bytes + f.run_off[a],
                            idbase[a] + (size_t) b * f.run_bytes[a], (size_t) f.run_bytes[a]);
        // and the device side as the pattern this direction must move (p1 for H2D, p2 for D2H)
        ck(cudaMemcpy((void*) f.slots.k_q, c.dir == 0 ? (const void*) p1.data() : (const void*) p2.data(), codes,
                      cudaMemcpyHostToDevice), "slots pattern");
        void* poison = c.dir == 0 ? (void*) f.slots.k_q : (c.packed ? (void*) packed_host : (void*) f.host.k_q);
        ck(cudaMemset(poison, 0xAA, codes), "poison");
        k::KvHostPools h = f.host;
        if (c.packed) h.k_q = (int8_t*) packed_host;
        k::kv_stream_copy_probe(f.map, f.slots, h, f.fmt, f.s, c.dir, c.packed, 32, 1024, nullptr);
        ck(cudaDeviceSynchronize(), "verify sync");
        if (c.dir == 0) {
            ck(cudaMemcpy(got.data(), (const void*) f.slots.k_q, codes, cudaMemcpyDeviceToHost), "readback slots");
        } else if (!c.packed) {
            std::memcpy(got.data(), f.host.k_q, codes);
        } else {   // de-pack the host block image through the same offsets the kernel used
            for (int64_t b = 0; b < f.pages; ++b)
                std::memcpy(got.data() + (size_t) b * f.run_bytes[0],
                            packed_host + (size_t) b * f.block_bytes + f.run_off[0], (size_t) f.run_bytes[0]);
        }
        int bad = 0;
        for (size_t i = 0; i < codes && bad < 4; ++i)
            if (got[i] != (*c.expect)[i]) ++bad;
        std::printf("P4 verify case=%s checked=%zu mismatches=%d expect=%s\n", c.name, codes, bad,
                    c.expect == &p1 ? "p1" : "p2");
        std::fflush(stdout);
    }
    cudaFreeHost(packed_host);
    // the fixture's host copy must be p1 again for the timing sections
    ck(cudaMemcpy((void*) f.host.k_q, p1.data(), codes, cudaMemcpyHostToDevice), "restore p1");
}

// ---- 5: the host-side copy paths ------------------------------------------------------------------------
// (a) the CPU's own copy of the same sizes over the same pinned memory (RAM -> RAM, one thread): the control the
//     paper's "host-side scattered copy" would be if it were done by memcpy;
// (b) the DMA path: ONE cudaMemcpyAsync PER SCATTERED CHUNK - "repeated small cudaMemcpyAsync calls", the thing the
//     paper replaces with a kernel.  This is the port's own prompt-path staging shape (kv_stage_from_host is one
//     memcpy per run per layer) at small transfer sizes.
void section_hostcopy() {
    const int64_t total = 64LL << 20;
    for (int chunk : {128, 512, 4096, 65536, 262144, 1048576}) {
        const int64_t n_chunks = total / chunk;
        const int64_t span = n_chunks * (int64_t) chunk * 2;
        std::vector<uint8_t> src((size_t) span, 1), dst((size_t) span, 0);
        for (int stride_k = 1; stride_k <= 2; ++stride_k) {
            const long long stride = (long long) chunk * stride_k;
            double el = 0;
            int passes = 0;
            const double t0 = now_s();
            while (passes < 8) {
                for (int64_t j = 0; j < n_chunks; ++j)
                    std::memcpy(dst.data() + j * stride, src.data() + j * stride, (size_t) chunk);
                ++passes;
                el = now_s() - t0;
                if (el >= 0.15) break;
            }
            std::printf("P4 hostcopy bytes=%d stride=%lld passes=%d moved=%lld ms=%.3f GBs=%.2f\n", chunk, stride,
                        passes, (long long) n_chunks * chunk * passes, el * 1000.0,
                        (double) (n_chunks * chunk) * passes / el / 1.0e9);
            std::fflush(stdout);
        }
    }
}

// the DMA road: one cudaMemcpyAsync per scattered chunk, in both directions
void section_dma() {
    const int64_t total = 32LL << 20;
    for (int dir = 0; dir < 2; ++dir) {
        for (int chunk : {128, 512, 4096, 65536, 262144, 1048576}) {
            const int64_t n_chunks = total / chunk;
            const int64_t span = n_chunks * (int64_t) chunk * 2;
            uint8_t* ds = device((size_t) span);
            uint8_t* ps = pinned((size_t) span);
            const void* src = dir == 0 ? (const void*) ps : (const void*) ds;
            void* dst = dir == 0 ? (void*) ds : (void*) ps;
            const double t0 = now_s();
            for (int64_t j = 0; j < n_chunks; ++j) {
                ck(cudaMemcpyAsync((uint8_t*) dst + j * chunk, (const uint8_t*) src + j * chunk, (size_t) chunk,
                                   cudaMemcpyDefault, nullptr),
                   "dma");
            }
            ck(cudaStreamSynchronize(nullptr), "dma sync");
            const double el = now_s() - t0;
            std::printf("P4 dma dir=%s bytes=%d calls=%lld moved=%lld ms=%.3f GBs=%.2f us_per_call=%.3f\n",
                        dir ? "D2H" : "H2D", chunk, (long long) n_chunks, (long long) n_chunks * chunk,
                        el * 1000.0, (double) n_chunks * chunk / el / 1.0e9, el * 1.0e6 / (double) n_chunks);
            std::fflush(stdout);
            cudaFreeHost(ps);
            cudaFree(ds);
        }
    }
    // one big contiguous transfer per direction over the same ends, the ceiling a DMA staging path can reach
    const int64_t bytes = 128LL << 20;
    uint8_t* ds = device((size_t) bytes);
    uint8_t* ps = pinned((size_t) bytes);
    for (int dir = 0; dir < 2; ++dir) {
        const double t0 = now_s();
        ck(cudaMemcpyAsync(dir == 0 ? (void*) ds : (void*) ps, dir == 0 ? (const void*) ps : (const void*) ds,
                           (size_t) bytes, cudaMemcpyDefault, nullptr), "dma big");
        ck(cudaStreamSynchronize(nullptr), "dma big sync");
        const double el = now_s() - t0;
        std::printf("P4 dma dir=%s bytes=%lld calls=1 moved=%lld ms=%.3f GBs=%.2f us_per_call=%.0f\n",
                    dir ? "D2H" : "H2D", (long long) bytes, (long long) bytes, el * 1000.0,
                    (double) bytes / el / 1.0e9, el * 1.0e6);
        std::fflush(stdout);
    }
    cudaFreeHost(ps);
    cudaFree(ds);
}

// The concurrent work is the engine's OWN kernels: `qsa_prompt_attn_batch` (the tensor-core prompt attention,
// P2 measured 75.4% of the 32K prefill's GPU timeline) as prefill-like, and `native_q5_k_f32` (the quantized
// expert GEMV that dominates a decode window) as decode-like.  The I/O kernel runs on a second queue.
struct Workloads {
    // prefill-like
    k::QsaShapes s = k::qsa_real_shapes();
    int64_t cap = 0, n_q = 0, pages = 0;
    float* q = nullptr;
    float* attn = nullptr;
    int32_t* ids = nullptr;
    int32_t* steps = nullptr;
    k::QsaAttnPools pools{};
    int32_t* table = nullptr;
    int prefill_calls = 0;
    // decode-like
    void* w = nullptr;
    float* x = nullptr;
    float* y = nullptr;
    void* xq8 = nullptr;
    int n_in = 0, n_out = 0;
    int decode_calls = 0;
    double prefill_solo = 0, decode_solo = 0;

    void init(int64_t pages_, int64_t n_q_, int prefill_calls_, int decode_calls_) {
        cap = k::qsa_selection_width(k::kTopkMaxCells, s);
        n_q = n_q_;
        pages = pages_;
        prefill_calls = prefill_calls_;
        decode_calls = decode_calls_;
        const int64_t rows = pages * s.n_head_kv * s.page_size;
        pools.k_q = (const int8_t*) device((size_t) rows * s.head_dim);
        pools.v_q = (const int8_t*) device((size_t) rows * s.head_dim);
        pools.k_scale = (const uint16_t*) device((size_t) rows * (s.head_dim / k::KV_Q8_GROUP) * 2);
        pools.v_scale = (const uint16_t*) device((size_t) rows * (s.head_dim / k::KV_Q8_GROUP) * 2);
        table = (int32_t*) device((size_t) pages * 4);
        std::vector<int32_t> t((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) t[(size_t) i] = (int32_t) i;
        ck(cudaMemcpy(table, t.data(), t.size() * 4, cudaMemcpyHostToDevice), "table");
        pools.page_table = table;
        ids = (int32_t*) device((size_t) n_q * cap * 4);
        steps = (int32_t*) device((size_t) n_q * k::kStepCount * 4);
        q = (float*) device((size_t) n_q * s.n_head * s.head_dim * 4);
        attn = (float*) device((size_t) n_q * s.n_head * s.head_dim * 4);
        std::vector<int32_t> hids((size_t) n_q * cap), hst((size_t) n_q * k::kStepCount, 0);
        for (int64_t i = 0; i < cap; ++i) hids[(size_t) i] = (int32_t) i;   // query 0: cells 0..cap-1, ascending
        for (int64_t t2 = 1; t2 < n_q; ++t2)
            for (int64_t i = 0; i < cap; ++i) hids[(size_t) (t2 * cap + i)] = (int32_t) i;
        for (int64_t t2 = 0; t2 < n_q; ++t2) {
            hst[(size_t) (t2 * k::kStepCount + k::kStepPos)] = (int32_t) (cap - 1);
            hst[(size_t) (t2 * k::kStepCount + k::kStepNKv)] = (int32_t) cap;
            hst[(size_t) (t2 * k::kStepCount + k::kStepNBid)] = (int32_t) (cap / 4);
            hst[(size_t) (t2 * k::kStepCount + k::kStepWidth)] = (int32_t) cap;
        }
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        // decode-like: the engine's Q5_K GEMV, weights = unmodified GGUF Q5_K blocks (176 B per 256 elements)
        n_in = 4096;
        n_out = 16384;
        const size_t wb = (size_t) n_out * (n_in / 256) * 176;
        w = device(wb);
        x = (float*) device((size_t) n_in * 4);
        y = (float*) device((size_t) n_out * 4);
        xq8 = device(k::native_q8_1_bytes(n_in, 1));
        std::vector<uint8_t> hw(wb, 0);
        for (size_t i = 0; i < wb; i += 176) hw[i] = 0x00, hw[i + 1] = 0x3c;   // fp16 1.0 as the block scale
        ck(cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice), "weights");
        // the prompt attention must return true, or the prefill arm is measuring nothing
        const bool ok = k::qsa_prompt_attn_batch(q, pools, ids, steps, cap, s, attn, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "prompt probe");
        std::printf("P4 work prefill_supported=%d cap=%lld n_q=%lld pages=%lld decode n_in=%d n_out=%d weights_MiB=%.1f\n",
                    (int) ok, (long long) cap, (long long) n_q, (long long) pages, n_in, n_out,
                    (double) wb / 1048576.0);
        std::fflush(stdout);
        if (!ok) { std::fprintf(stderr, "P4 probe: prompt attention refused the shape\n"); std::exit(3); }
    }

    void prefill(cudaStream_t st) {
        for (int i = 0; i < prefill_calls; ++i)
            k::qsa_prompt_attn_batch(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    void decode(cudaStream_t st) {
        for (int i = 0; i < decode_calls; ++i)
            k::native_q5_k_f32(w, x, xq8, y, n_in, n_out, 1, st);
    }
};

void section_inter(Fixture& f) {
    const int64_t pages = 512 + 8;   // the selection's 2051 cells = 513 blocks; one page spare
    Workloads w;
    w.init(pages, 256, 8, 2500);

    for (int which = 0; which < 2; ++which) {
        cudaStream_t a = nullptr;
        ck(cudaStreamCreate(&a), "stream a");
        double solo = 1e30;
        for (int rep = 0; rep < 3; ++rep) {   // best of three: the first call of a workload is slower
            const double t0 = now_s();
            if (which == 0) w.prefill(a);
            else w.decode(a);
            ck(cudaStreamSynchronize(a), "solo");
            solo = std::min(solo, now_s() - t0);
        }
        std::printf("P4 inter work=%s blocks=0 threads=0 solo_ms=%.2f calls=%d io_GBs=0.00 io_ms=0.00 norm=1.000\n",
                    which == 0 ? "prefill" : "decode", solo * 1000.0, which == 0 ? w.prefill_calls : w.decode_calls);
        std::fflush(stdout);

        // The I/O JOB is fixed work, pre-submitted on its own queue: `passes` passes of 132 MiB, sized so that the
        // I/O alone would take ~1.5x the compute's solo time (so the I/O is contending for the whole compute span).
        for (int rep = 0; rep < 2; ++rep) {
            for (int b : {1, 2, 4, 8, 16, 32, 64, 96, 128}) {
                cudaStream_t bq = nullptr;
                ck(cudaStreamCreate(&bq), "stream b");
                // one timed pass on B alone, for this block count
                const double p0 = now_s();
                k::kv_stream_copy_probe(f.map, f.slots, f.host, f.fmt, f.s, 0, 0, b, 1024, bq);
                ck(cudaStreamSynchronize(bq), "io calib");
                const double pass_s = now_s() - p0;
                int passes = (int) std::ceil(1.5 * solo / pass_s);
                passes = std::max(2, std::min(passes, 2048));
                const long long bytes = (long long) f.bytes_per_pass() * passes;
                double io0 = 0, io1 = 0, ta = 0;
                std::atomic<bool> go{false};
                std::thread iothr([&] {
                    while (!go) std::this_thread::sleep_for(std::chrono::microseconds(200));
                    io0 = now_s();
                    for (int i = 0; i < passes; ++i)
                        k::kv_stream_copy_probe(f.map, f.slots, f.host, f.fmt, f.s, 0, 0, b, 1024, bq);
                    ck(cudaStreamSynchronize(bq), "io job");
                    io1 = now_s();
                });
                go = true;
                const double t0 = now_s();
                if (which == 0) w.prefill(a);
                else w.decode(a);
                ck(cudaStreamSynchronize(a), "conc compute");
                ta = now_s() - t0;
                iothr.join();
                const double io_s = io1 - io0;
                std::printf("P4 inter work=%s rep=%d blocks=%d threads=1024 solo_ms=%.2f conc_ms=%.2f norm=%.3f "
                            "io_passes=%d io_MiB=%lld io_ms=%.2f io_GBs=%.2f io_pass_solo_ms=%.3f\n",
                            which == 0 ? "prefill" : "decode", rep + 1, b, solo * 1000.0, ta * 1000.0,
                            solo / ta, passes, (long long) (bytes >> 20), io_s * 1000.0,
                            (double) bytes / io_s / 1.0e9, pass_s * 1000.0);
                std::fflush(stdout);
                ck(cudaStreamDestroy(bq), "destroy b");
            }
        }
        ck(cudaStreamDestroy(a), "destroy a");
    }
}

// ---- concurrency of two in-order queues: do two kernels from different queues time-share this device? ---------
// This decides how to read the interference arm: if the two queues serialize, then "concurrent work" is really
// "sequential work" and the I/O's own duration is added on top (the port's arrangement), while a real overlap would
// show overlap ~ 2.0.  W is the SAME workload on both queues, so the ratio is unambiguous.
void section_conc(Fixture& f) {
    (void) f;
    Workloads w;
    w.init(520, 256, 4, 1200);
    for (int which = 0; which < 2; ++which) {
        cudaStream_t a = nullptr, b = nullptr;
        ck(cudaStreamCreate(&a), "stream a");
        ck(cudaStreamCreate(&b), "stream b");
        auto run = [&](cudaStream_t s) {
            if (which == 0) w.prefill(s);
            else w.decode(s);
        };
        double soloA = 1e30, soloB = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            double t0 = now_s();
            run(a);
            ck(cudaStreamSynchronize(a), "solo a");
            soloA = std::min(soloA, now_s() - t0);
            t0 = now_s();
            run(b);
            ck(cudaStreamSynchronize(b), "solo b");
            soloB = std::min(soloB, now_s() - t0);
        }
        double both = 0, ta = 0, tb = 0;
        for (int rep = 0; rep < 2; ++rep) {
            std::atomic<bool> go{false};
            std::thread thr([&] {
                while (!go) std::this_thread::sleep_for(std::chrono::microseconds(200));
                const double s0 = now_s();
                run(a);
                ck(cudaStreamSynchronize(a), "conc a");
                ta = now_s() - s0;
            });
            go = true;
            const double t0 = now_s();
            run(b);
            ck(cudaStreamSynchronize(b), "conc b");
            tb = now_s() - t0;
            both = std::max(ta, tb);
            thr.join();
        }
        std::printf("P4 conc work=%s same_workload both_queues soloA_ms=%.2f soloB_ms=%.2f "
                    "concurrent_A_ms=%.2f concurrent_B_ms=%.2f wall_ms=%.2f overlap=%.3f (2.0 = full, 1.0 = serial)\n",
                    which == 0 ? "prefill" : "decode", soloA * 1000.0, soloB * 1000.0, ta * 1000.0, tb * 1000.0,
                    both * 1000.0, (soloA + soloB) / both);
        std::fflush(stdout);
        ck(cudaStreamDestroy(b), "destroy b");
        ck(cudaStreamDestroy(a), "destroy a");
    }
}
// ---- ordering: is the "interference" real contention, or the two queues taking turns in submission order? -----
// Two full-GPU workloads cannot overlap whatever the driver does (the SM count is the limit), so the inter arm's
// decode case needs a control: the SAME I/O job and the SAME compute, submitted in the two possible orders.  If
// io-first makes the compute wait for almost the whole I/O span, the cost is QUEUE ORDERING, not contention.
void section_order(Fixture& f) {
    Workloads w;
    w.init(520, 256, 4, 1200);
    for (int which = 0; which < 2; ++which) {
        cudaStream_t a = nullptr, b = nullptr;
        ck(cudaStreamCreate(&a), "stream a");
        ck(cudaStreamCreate(&b), "stream b");
        auto run = [&](cudaStream_t s) {
            if (which == 0) w.prefill(s);
            else w.decode(s);
        };
        double solo = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            const double t0 = now_s();
            run(a);
            ck(cudaStreamSynchronize(a), "solo");
            solo = std::min(solo, now_s() - t0);
        }
        const int passes = 12;
        for (int io_first = 0; io_first < 2; ++io_first) {
            std::atomic<bool> go{false};
            double io0 = 0, io1 = 0, ta = 0, t0 = 0;
            std::thread iothr([&] {
                while (!go) std::this_thread::sleep_for(std::chrono::microseconds(200));
                io0 = now_s();
                for (int i = 0; i < passes; ++i)
                    k::kv_stream_copy_probe(f.map, f.slots, f.host, f.fmt, f.s, 0, 0, 32, 1024, b);
                ck(cudaStreamSynchronize(b), "io job");
                io1 = now_s();
            });
            go = true;
            if (io_first) std::this_thread::sleep_for(std::chrono::milliseconds(60));   // let the I/O queue fill
            t0 = now_s();
            run(a);
            ck(cudaStreamSynchronize(a), "ordered compute");
            ta = now_s() - t0;
            iothr.join();
            std::printf("P4 order work=%s io_first=%d passes=%d blocks=32 solo_ms=%.2f compute_ms=%.2f io_ms=%.2f "
                        "serial_expect_ms=%.2f\n",
                        which == 0 ? "prefill" : "decode", io_first, passes, solo * 1000.0, ta * 1000.0,
                        (io1 - io0) * 1000.0, solo * 1000.0 + (io1 - io0) * 1000.0);
            std::fflush(stdout);
        }
        ck(cudaStreamDestroy(b), "destroy b");
        ck(cudaStreamDestroy(a), "destroy a");
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::string want = argc > 1 ? argv[1] : "all";
    int dev_count = 0;
    ck(cudaGetDeviceCount(&dev_count), "device count");
    const k::KvCopyShape shipped = k::kv_stream_copy_shape();
    std::printf("P4 head devices=%d shipped_shape=%dx%d\n", dev_count, shipped.blocks, shipped.threads);
    std::fflush(stdout);

    Fixture f;
    f.init(32768);   // 32768 blocks x 4224 B = 132 MiB per pass
    int64_t off[4] = {0, 0, 0, 0}, bb = 0;
    k::kv_block_run_offsets(f.s, f.fmt, off, &bb);
    std::printf("P4 fixture pages=%lld block_bytes=%lld run_bytes=%lld,%lld,%lld,%lld run_off=%lld,%lld,%lld,%lld "
                "n_head_kv=%lld page_size=%lld head_dim=%lld\n",
                (long long) f.pages, (long long) f.block_bytes, (long long) f.run_bytes[0], (long long) f.run_bytes[1],
                (long long) f.run_bytes[2], (long long) f.run_bytes[3], (long long) off[0], (long long) off[1],
                (long long) off[2], (long long) off[3], (long long) f.s.n_head_kv, (long long) f.s.page_size,
                (long long) f.s.head_dim);
    std::fflush(stdout);

    if (want == "verify" || want == "all") section_verify(f);
    if (want == "diag" || want == "all") section_diag(f);
    if (want == "bw" || want == "all") section_bw(f);
    if (want == "chunk" || want == "all") section_chunk();
    if (want == "layout" || want == "all") section_layout(f);
    if (want == "hostcopy" || want == "all") section_hostcopy();
    if (want == "dma" || want == "all") section_dma();
    if (want == "inter" || want == "all") section_inter(f);
    if (want == "conc" || want == "all") section_conc(f);
    if (want == "order" || want == "all") section_order(f);
    std::printf("P4 done\n");
    return 0;
}
