// D2c (card t_c7d8cd86): build the decode path's REMAINING first-launch SYCL programs in the load phase.
//
// D2b (card t_f93760a1) moved the dense (quant type x ncols) MMVQ family out of the first decode windows with
// `native_mmvq_warmup()` (native_mmvq.cu).  This file is the same treatment for the rest of the decode path:
// on a cold first run the decode phase still built 61 programs - 6 routed-expert `native_gu_multi_kernel<T>` /
// `native_down_multi_kernel<T>` (the pack's gate/up and down types) plus 55 one-off kernels, one SYCL program per
// decode-path kernel SITE (`d2c/ENUMERATION.txt`, built by `d2c/enumerate_c.py` from the SPIR-V name section of
// each cold-run cache entry).
//
// WHY IT IS SAFE, AND WHY IT CANNOT CHANGE A NUMBER
// A SYCL program here is one `strata::sycl_compat::launch(...)` call site (the device code is split per kernel,
// `-fsycl-device-code-split=per_kernel` at compile AND link time, cmake/sycl_backend.cmake).  The program is
// built by the RUNTIME at the first launch of that site and the specialization is a function of the call site
// and of the compile-time template arguments it is instantiated with - never of the operand values.  So this
// pass calls each site ONCE, on the caller's stream, over zeroed scratch of its own; every count it steers (the
// routed-expert group counts, the commit length, the blob count) is a zeroed device word, which makes the
// kernels return from their first guard.  Nothing it computes is read by anything, so the ids guard (§ the
// card's checks) is the evidence, not the argument.
//
// WHERE IT RUNS
// From the serve path, immediately after `native_mmvq_warmup`, i.e. still before the "everything loaded" line -
// after every weight, the drafter and the prompt paths are in place, before the ask can arrive (the FIFO is fed
// after that line) and before any window or graph capture has run.
// `STRATA_KERNEL_WARMUP=0` is its A/B control arm, separate from D2b's `STRATA_MMVQ_WARMUP=0`, so the two passes
// can be measured apart.
//
// WHAT IT DOES NOT COVER, AND WHY (the card's "bound the rest")
//  * `qsa_decode_attn_batch` (`attn_chunk_kernel<1>`, `attn_merge_kernel`), `qsa_block_scores`
//    (`block_scores_multi_kernel`), `native_qsa_indexer_append` (`append<false>`): these read the QSA attention
//    POOLS and the indexer's pooled/tail/dead buffers through a per-page table whose size and contents a request
//    builds (`QsaAttnPools`, `QsaIndexerBuffers`, `QsaShapes`).  A dummy pool is not a smaller pool here: the
//    page table indexes it, so a zeroed table and a zeroed pool are only safe if their arithmetic agrees with
//    the real geometry, and the geometry is built after the load phase.  Left to the request.
//  * `shared_expert` / `shared_expert_multi` and `native_ple_postops`: these dispatch on the pack's OWN weight
//    types and forms (`NativeSharedWeights::gate_type/up_type/down_type`, `SForm`, `PleWeights`), which come
//    from the loaded pack's shared-expert/PLE metadata rather than from a compile-time table.  Warming them with
//    a guessed type would build a program the request does not use, which is strictly worse than not warming
//    (D2b's over-warm measurement): they need the pack's type fields plumbed in, and are left to the request.
//  * The dense (type x ncols) MMVQ family is D2b's pass, not repeated here.
//
// The scratch arena is ~8 MiB, allocated once and freed at the end (D2b's pass is ~100 KiB; the engine prints
// its free-VRAM line right after this, so this pass stays small on purpose - a page-in while a verify graph
// spins on a host flag stalls a request for good).
//
// This file is part of the SYCL kernel TU list only (cmake/sycl_backend.cmake STRATA_SYCL_KERNELS): it carries
// no `__global__` and no launch syntax of its own, so `syclify` emits it unchanged and every error names this
// file's own line.

#include "strata/kernels/decode_warmup.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace strata::kernels {
namespace {

/// One allocation per element type, all zeroed once: the pass is a sequence of launches that must only build
/// their program, so a single zeroed arena is enough for every site and there is nothing to keep apart except
/// the buffers a kernel validates for overlap (ple_history_advance).
struct Arena {
    void* f32 = nullptr;   // 1.3M floats
    void* u16 = nullptr;   // 512K
    void* u8 = nullptr;    // 1M
    void* i32 = nullptr;   // 16K
    void* u64 = nullptr;   // 1K
    void* u32 = nullptr;   // 1K
    size_t f32_bytes = 0, u16_bytes = 0, u8_bytes = 0, i32_bytes = 0, u64_bytes = 0, u32_bytes = 0;

    float* F(size_t off) const { return static_cast<float*>(f32) + off; }
    uint16_t* U16(size_t off) const { return static_cast<uint16_t*>(u16) + off; }
    uint8_t* U8(size_t off) const { return static_cast<uint8_t*>(u8) + off; }
    int32_t* I32(size_t off) const { return static_cast<int32_t*>(i32) + off; }
    unsigned long long* U64(size_t off) const { return static_cast<unsigned long long*>(u64) + off; }
    uint32_t* U32(size_t off) const { return static_cast<uint32_t*>(u32) + off; }
};

// f32 arena offsets, in floats
constexpr size_t F_ACT = 0;          // activations / x / R_src       (64K floats)
constexpr size_t F_W = 64 * 1024;    // y / dst                       (64K floats)
constexpr size_t F_Y = 128 * 1024;   // out                           (64K floats)
constexpr size_t F_LOGITS = 192 * 1024;   // 32768 floats for the sampler / row_top_prob
constexpr size_t F_PLE_HIST = 256 * 1024;    // NG_HIST * NG_HC_DIM = 92160 floats
constexpr size_t F_PLE_NORM = 384 * 1024;    // NG_HC_DIM = 10240 floats
constexpr size_t F_BIG = 448 * 1024;         // the grouped/attention kernels' generous tail
constexpr size_t F_TOTAL = 1216 * 1024;      // 4.75 MiB

// i32 arena slots, in ints.  EVERY word a kernel READS as a count/index/length has its own slot and is zero at
// the start of the pass, because a slot a later site WRITES (a grp_start, a dst row, a table entry) would
// otherwise be read back as that count by a later site's kernel - the first version of this pass aliased them
// and the device faulted inside `fetch_blobs` on a pointer the group-resident kernel had just written
// (measured: engine exit -11, then -6 with the fault named at the next sync).
constexpr size_t I_NGROUPS = 0;    // read by native_expert_grouped / moe_grouped_s2 - must stay 0
constexpr size_t I_NCOUNT = 4;     // read by fetch_blobs / rebase_ptrs - must stay 0
constexpr size_t I_NKEEP = 8;      // read by gdn_conv_commit - must stay 0 (0 = leave the history as it is)
constexpr size_t I_INDEX = 12;     // read by copy_indexed - must stay 0 (negative returns early)
constexpr size_t I_IDS = 16;       // read by row_top_prob / doorbell_publish / mtp_select - must stay 0
constexpr size_t I_ROWDEV = 20;    // read by mtp_select - must stay 0
constexpr size_t I_TABLE = 32;     // read by map_ids (table[ids[i]]) - must stay 0
constexpr size_t I_HITROWS = 64;   // read by copy_rows_from_mapped - must stay 0
constexpr size_t I_HITCOUNT = 68;  // read by copy_rows_from_mapped / moe_hit_add - must stay 0
constexpr size_t I_GRPSTART = 160; // written by moe_group_resident
constexpr size_t I_COUNTS = 164;   // written by moe_group_resident
constexpr size_t I_ENTDST = 168;   // written by moe_group_resident
constexpr size_t I_ENTTOK = 176;   // written by moe_group_resident
constexpr size_t I_DSTROWS = 192;  // written by copy_i32_from_mapped / mtp_select
constexpr size_t I_MAPIDS = 224;   // written by map_ids
constexpr size_t I_SAMPLEOUT = 240;  // written by sample_tokens

bool alloc_arena(Arena& a, cudaStream_t s) {
    a.f32_bytes = F_TOTAL * sizeof(float);
    a.u16_bytes = 512 * 1024;
    a.u8_bytes = 1024 * 1024;
    a.i32_bytes = 64 * 1024;
    a.u64_bytes = 8 * 1024;
    a.u32_bytes = 4 * 1024;
    bool ok = cudaMalloc(&a.f32, a.f32_bytes) == cudaSuccess &&
              cudaMalloc(&a.u16, a.u16_bytes) == cudaSuccess &&
              cudaMalloc(&a.u8, a.u8_bytes) == cudaSuccess &&
              cudaMalloc(&a.i32, a.i32_bytes) == cudaSuccess &&
              cudaMalloc(&a.u64, a.u64_bytes) == cudaSuccess &&
              cudaMalloc(&a.u32, a.u32_bytes) == cudaSuccess;
    if (!ok) return false;
    cudaMemsetAsync(a.f32, 0, a.f32_bytes, s);
    cudaMemsetAsync(a.u16, 0, a.u16_bytes, s);
    cudaMemsetAsync(a.u8, 0, a.u8_bytes, s);
    cudaMemsetAsync(a.i32, 0, a.i32_bytes, s);
    cudaMemsetAsync(a.u64, 0, a.u64_bytes, s);
    cudaMemsetAsync(a.u32, 0, a.u32_bytes, s);
    return true;
}

void free_arena(Arena& a) {
    cudaFree(a.f32);
    cudaFree(a.u16);
    cudaFree(a.u8);
    cudaFree(a.i32);
    cudaFree(a.u64);
    cudaFree(a.u32);
    cudaGetLastError();
}

/// The pack's dense expert gate/up and down types (`pack/native_experts.txt`: gu_type/d_type per layer).  The
/// kernel instantiation is per TYPE, so one call per distinct type pair builds every
/// `native_gu_multi_kernel<TG>` / `native_down_multi_kernel<TD>` the model can reach.
struct ExpertPair { int gu; int down; };

}  // namespace

void decode_warmup(void* stream) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (s == nullptr) return;
    const char* off = std::getenv("STRATA_KERNEL_WARMUP");
    if (off != nullptr && off[0] == '0') {
        std::fprintf(stderr, "strata kernel warmup: off (STRATA_KERNEL_WARMUP=0)\n");
        return;
    }

    Arena a;
    if (!alloc_arena(a, s)) {
        std::fprintf(stderr, "strata kernel warmup: scratch allocation failed (%s); the first windows will build\n",
                     cudaGetErrorString(cudaGetLastError()));
        free_arena(a);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    int launches = 0, refused = 0, sites = 0;
    const bool trace_sites = std::getenv("STRATA_KERNEL_WARMUP_TRACE") != nullptr;
    int faulted = 0;
    auto site = [&](const char* what, auto&& fn) {
        ++sites;
        if (trace_sites) {
            std::fprintf(stderr, "strata kernel warmup: site %d %s\n", sites, what);
            std::fflush(stderr);
        }
        try {
            fn();
            ++launches;
        } catch (const std::exception& e) {
            if (refused++ == 0) std::fprintf(stderr, "strata kernel warmup: %s: %s\n", what, e.what());
        }
        if (trace_sites) {
            // one sync per site: the first one that comes back with an error names the launch whose PROGRAM the
            // device refused (a device-side fault only surfaces at the next sync otherwise)
            const cudaError_t rc = cudaStreamSynchronize(s);
            if (rc != cudaSuccess) {
                std::fprintf(stderr, "strata kernel warmup: FAULT %s: %s\n", what, cudaGetErrorString(rc));
                ++faulted;
            }
            cudaGetLastError();
            std::fflush(stderr);
        }
    };

    // ---- the BF16 MMVF family: the block size is a HOST function of n_in (mmvf_block_size) and the column
    // count a host branch of n_tok, so the two n_in below cover `bf16_f32_mmvf_kernel<160>` and `<256>` and the
    // two n_tok cover `bf16_f32_mmvf_multi_kernel<256,4>` and `<256,8>` whatever the window sequence reaches.
    const int64_t bf16_n_out = 8;
    for (const int64_t n_in : {int64_t(640), int64_t(2560), int64_t(4096)}) {
        site("bf16_gemv_fp32_mmvf", [&] {
            bf16_gemv_fp32_mmvf(a.F(F_ACT), a.U16(0), a.F(F_Y), n_in, bf16_n_out, s);
        });
    }
    for (const int n_tok : {2, 4, 8}) {
        site("bf16_gemv_fp32_mmvf_multi", [&] {
            bf16_gemv_fp32_mmvf_multi(a.F(F_ACT), 4096, a.U16(0), a.F(F_Y), bf16_n_out, 4096, bf16_n_out, n_tok, s);
        });
    }

    // ---- the routed experts: `native_expert_grouped` is the single public dispatch, so one call per (gu, down)
    // type pair the pack carries builds the whole family.  n_groups is a zeroed DEVICE word: the gu and down
    // kernels return from `g >= *n_groups` before they touch grp_ptr, so the blob is never read, and grp_ptr
    // stays a zeroed DEVICE array (writing a pointer into it from the HOST is an invalid access to device memory
    // and segfaults the engine - measured, this pass's first version).
    const int64_t e_embd = 2560, e_ff = 1536;   // the model's dims; only the TYPE is part of the program
    void* e_scratch = a.U8(512 * 1024);         // handed to the kernels as scratch, never read back by the host
    void* e_act = a.U8(0);                      // the activations, disjoint from the scratch
    const ExpertPair pairs[] = {{18, 20}, {18, 42}, {21, 20}, {21, 42}, {22, 20}, {22, 42}, {23, 20}};
    for (const ExpertPair& p : pairs) {
        site("native_expert_grouped", [&] {
            const NativeExpertLayout L = native_expert_layout(p.gu, p.down, e_embd, e_ff);
            native_expert_grouped(L, a.U64(0), a.I32(I_GRPSTART), a.I32(I_NGROUPS), a.I32(I_DSTROWS),
                                  a.I32(I_ENTTOK), 1, 1, e_act, e_scratch, a.F(F_Y), s, false);
        });
    }

    // ---- the drafter's S2 grouped experts (the MTP layer's 512 experts, H = 2560 / FF = 640 by construction).
    // The new kernels are selected on pointer alignment (`new_grouped`), so the arena gives them the fast path -
    // the same one the decode windows take.
    site("moe_grouped_s2", [&] {
        moe_grouped_s2(a.U64(0), a.I32(I_GRPSTART), a.I32(I_NGROUPS), a.I32(I_DSTROWS), a.I32(I_ENTTOK), 1, 1,
                       a.U8(0), a.F(F_ACT), e_scratch, a.F(F_Y), s);
    });
    site("moe_group_resident", [&] {
        moe_group_resident(a.I32(I_IDS), 1, 1, a.U8(0), 1 << 20, a.U64(0), a.I32(I_GRPSTART), a.I32(I_COUNTS),
                           a.I32(I_ENTDST), a.I32(I_ENTTOK), s);
    });
    site("moe_hit_add", [&] {
        moe_hit_add(a.F(F_Y), a.F(F_W), a.I32(I_ENTDST), a.I32(I_HITCOUNT), 1, 2560, s);
    });

    // ---- the GDN (linear-attention mixer) family.  n_keep is a zeroed device word, which is the kernels' own
    // "read, do not write" path, and n_tok = 1 vs 4 spans the column branches.
    for (const int n_tok : {1, 4}) {
        site("gdn_conv_l2_multi", [&] {
            gdn_conv_l2_multi(a.F(F_ACT), a.F(F_W), a.F(F_Y), a.F(F_BIG), 512, 4, 1e-6f, n_tok, s, 0);
        });
        site("gdn_ab_multi", [&] {
            gdn_ab_multi(a.F(F_ACT), a.U16(0), a.U16(64 * 1024), a.F(F_W), a.F(F_Y), a.F(F_BIG), a.F(F_BIG + 4096),
                         512, 16, n_tok, s);
        });
        site("gdn_step_norm_multi", [&] {
            gdn_step_norm_multi(a.F(F_BIG), a.F(F_ACT), 512, a.F(F_W), a.F(F_Y), a.F(F_BIG + 8192),
                                a.F(F_BIG + 12288), 1e-6f, a.F(F_BIG + 16384), 4, 16, n_tok, nullptr, s, 0);
        });
    }
    site("gdn_conv_commit", [&] {
        gdn_conv_commit(a.F(F_ACT), a.F(F_W), 512, a.I32(I_NKEEP), s);
    });

    // ---- the MTP / sampler tail
    site("mtp_select", [&] {
        mtp_select(a.F(F_ACT), 4096, a.I32(I_IDS), a.I32(I_ROWDEV), a.F(F_Y), a.I32(I_DSTROWS),
                   a.I32(I_DSTROWS + 4), 1, s, nullptr, nullptr);
    });
    for (const int n_tok : {1, 8}) {
        site("native_router_top10_multi", [&] {
            native_router_top10_multi(a.F(F_LOGITS), a.I32(I_DSTROWS), a.F(F_Y), n_tok, s);
        });
    }
    site("native_router_top10", [&] {
        native_router_top10(a.F(F_LOGITS), a.I32(I_DSTROWS), a.F(F_Y), s);
    });
    for (const float temp : {0.0f, 1.0f}) {
        site("sample_tokens", [&] {
            SamplerParams p;
            p.temperature = temp;
            sample_tokens(a.F(F_LOGITS), 1, 32768, a.I32(I_TABLE), 0, p, a.I32(I_SAMPLEOUT), s);
        });
    }
    site("row_top_prob", [&] {
        row_top_prob(a.F(F_LOGITS), 1, 32768, a.I32(I_IDS), a.F(F_Y), s);
    });
    site("map_ids", [&] {
        map_ids(a.I32(I_MAPIDS), a.I32(I_TABLE), 4, s);
    });

    // ---- the MIXER / MoE combination
    for (const int n_tok : {1, 4}) {
        site("native_moe_combine_multi", [&] {
            native_moe_combine_multi(a.F(F_ACT), a.F(F_W), a.F(F_Y), a.F(F_BIG), 512, 10, n_tok, s);
        });
    }
    site("native_moe_combine", [&] {
        native_moe_combine(a.F(F_ACT), a.F(F_W), a.F(F_Y), a.F(F_BIG), 512, 10, s);
    });
    for (const bool fused : {false, true}) {
        site("native_gr_pre_gated", [&] {
            native_gr_pre_gated(a.F(F_ACT), a.F(F_W), a.F(F_Y), 512, 4, fused, s);
        });
    }
    site("native_gr_post", [&] {
        native_gr_post(a.F(F_ACT), a.F(F_W), a.F(F_Y), a.F(F_BIG), 512, 4, s);
    });
    site("native_gr_down_silu", [&] {
        native_gr_down_silu(a.F(F_BIG), 512, 4, s);
    });
    for (const int rows : {1, 4}) {
        site("native_gr_rms_norm_weighted", [&] {
            native_gr_rms_norm_weighted(a.F(F_ACT), a.F(F_W), a.F(F_Y), 512, rows, 1e-6f, s);
        });
    }
    site("native_qsa_gate_apply", [&] {
        native_qsa_gate_apply(a.F(F_ACT), a.F(F_W), a.F(F_Y), 4, 64, s);
    });

    // ---- the PLE history advance
    site("ple_history_advance", [&] {
        ple_history_advance(a.F(F_PLE_HIST), a.F(F_PLE_NORM), s);
    });

    // ---- the verify window's one-off helpers (the fixed-size copies / flags / quantizers)
    site("copy_from_mapped", [&] { copy_from_mapped(a.F(F_Y), a.F(F_ACT), 256, s); });
    site("copy_i32_from_mapped", [&] { copy_i32_from_mapped(a.I32(I_DSTROWS), a.I32(I_TABLE), 16, s); });
    site("copy_rows_from_mapped", [&] {
        copy_rows_from_mapped(a.F(F_Y), a.F(F_ACT), 2, 64, a.I32(I_HITROWS), a.I32(I_HITCOUNT), s);
    });
    site("f32_to_bf16_bulk", [&] { f32_to_bf16_bulk(a.F(F_ACT), a.U16(0), 256, s); });
    site("broadcast_streams", [&] { broadcast_streams(a.F(F_ACT), a.F(F_Y), 64, 4, 1, s); });
    site("copy_indexed", [&] { copy_indexed(a.F(F_Y), a.F(F_ACT), 64, a.I32(I_INDEX), 64, s); });
    site("doorbell_publish", [&] {
        doorbell_publish(a.F(F_ACT), a.I32(I_IDS), a.F(F_W), 64, 1, a.F(F_Y), a.I32(I_MAPIDS), a.F(F_BIG),
                         a.U32(0), s);
    });
    site("fetch_blobs", [&] { fetch_blobs(a.U64(0), a.I32(I_NCOUNT), a.U8(0), 4096, 1, s); });
    site("rebase_ptrs", [&] { rebase_ptrs(a.U64(0), a.I32(I_NCOUNT), a.U8(0), 4096, s); });
    site("wait_flag_ge", [&] { wait_flag_ge(a.U32(0), 0u, s); });
    site("quantize_q8_1_rows", [&] { quantize_q8_1_rows(a.F(F_ACT), 1, 256, a.U8(0), s); });
    site("quantize_q8_0_scaled", [&] {
        quantize_q8_0_scaled(a.F(F_ACT), a.U8(0), a.F(F_Y), 256, s);
    });

    const bool synced = cudaStreamSynchronize(s) == cudaSuccess;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "strata kernel warmup: %d launches over %d sites, %.0f ms%s\n", launches, sites, ms,
                 (refused || faulted) ? " (SOME LAUNCHES REFUSED OR FAULTED)" : (synced ? "" : " (STREAM SYNC FAILED)"));
    free_arena(a);
}

}  // namespace strata::kernels
