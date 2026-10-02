// include/strata/sycl_compat/mma16816.hpp - the m16n8k16 f16 -> f32 MMA (PTX mma.sync.aligned.m16n8k16.row.col)
// emulated in FP32 for the SYCL backend.
//
// WHY THIS EXISTS.  PLAN.md §2.3 sites #4-#6 (three mma.sync f16 sites in qsa_prompt_attn.cu) have no SYCL
// spelling: SPIR-V has no mma.sync, and the Intel equivalent (joint_matrix / DPAS) is a TILE API - it consumes
// shared-memory or global tiles, not the per-lane fragment registers the ported kernel works in.  The file's
// own portable arm (`D1_NO_QLO`/`D1_NO_PLO`) is a compile-time PRECISION switch, not a fallback: it drops the
// low half of q and p, which is a different algorithm.  So the portable path needs a correct emulation, and
// this is it.
//
// THE FRAGMENT LAYOUT (PTX ISA, mma.m16n8k16 with f16 A and B, f32 C).  With gid = laneid >> 2 and
// tig = laneid & 3, one lane holds
//   A (m16 x k16, row major), 4 x .b32 = 8 f16:
//       a0 = { A[gid][2*tig],     A[gid][2*tig + 1]     }
//       a1 = { A[gid + 8][2*tig],  A[gid + 8][2*tig + 1] }
//       a2 = { A[gid][2*tig + 8],  A[gid][2*tig + 9]     }
//       a3 = { A[gid + 8][2*tig + 8], A[gid + 8][2*tig + 9] }
//   B (k16 x n8, "col": k is the row index), 2 x .b32 = 4 f16:
//       b0 = { B[2*tig][gid],     B[2*tig + 1][gid]     }
//       b1 = { B[2*tig + 8][gid], B[2*tig + 9][gid]     }
//   C (m16 x n8), 4 x .f32:
//       c0 = C[gid][2*tig]  c1 = C[gid][2*tig + 1]  c2 = C[gid + 8][2*tig]  c3 = C[gid + 8][2*tig + 1]
//
// THE EXCHANGE.  Every lane needs the full k=0..15 rows of A for its two rows, and the full k=0..15 columns of
// B for the two columns it owns.  Rebuilding them costs 32 `select_from_group` ops: 16 for A (four quad mates
// x four registers) and 16 for B (columns 2*tig and 2*tig+1 x four registers), then 64 FMAs.  Note the B
// column for column n = 2*tig lives in the lane (4*n + tig'): the source lane is NOT a quad mate, which is why
// the CUDA `__shfl_xor` form could not be used here.
//
// ACCURACY.  Every f16 input converts to f32 exactly and every product of two f16 values is exact in f32, so
// this emulation differs from the hardware mma in ONE respect: the summation order of the 16 terms.  That is
// why the parity gates for the paths that use it are tolerances (the FP64-reference gate of
// src/kernels/qsa_prompt_attn_parity.cpp), not bit equality - the same situation the .cu documents for its
// own Turing two-step k=8 form ("the sum now rounds twice, so the two paths do not agree bit for bit").
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::sycl_compat {

/// The low 16 bits of a fragment word as f32 (f16 -> f32 is exact).
inline float mma_f16_lo(uint32_t x) {
    return (float) sycl::bit_cast<sycl::half>((uint16_t) (x & 0xffffu));
}
/// The high 16 bits of a fragment word as f32.
inline float mma_f16_hi(uint32_t x) {
    return (float) sycl::bit_cast<sycl::half>((uint16_t) (x >> 16));
}

#if defined(__SYCL_DEVICE_ONLY__)
/// c[0..3] += A(16x16 f16) * B(16x8 f16), the PTX m16n8k16.row.col contract, in FP32.
inline void mma16816_f32(const sycl::sub_group& sg, float* c, const uint32_t* a, const uint32_t* b) {
    const int lane = (int) sg.get_local_linear_id();
    const int gid = lane >> 2, tig = lane & 3;
    auto sel = [&](uint32_t v, int src) { return sycl::select_from_group(sg, v, (uint32_t) src); };

    float A0[16], A1[16];   // rows gid and gid+8
    float B0[16], B1[16];   // columns 2*tig and 2*tig+1
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int q = (lane & ~3) | t;          // the quad mate that holds k columns 2t, 2t+1, 2t+8, 2t+9
        const uint32_t a0 = sel(a[0], q), a1 = sel(a[1], q);
        const uint32_t a2 = sel(a[2], q), a3 = sel(a[3], q);
        A0[2 * t] = mma_f16_lo(a0);      A0[2 * t + 1] = mma_f16_hi(a0);
        A1[2 * t] = mma_f16_lo(a1);      A1[2 * t + 1] = mma_f16_hi(a1);
        A0[2 * t + 8] = mma_f16_lo(a2);  A0[2 * t + 9] = mma_f16_hi(a2);
        A1[2 * t + 8] = mma_f16_lo(a3);  A1[2 * t + 9] = mma_f16_hi(a3);
        const int s0 = 8 * tig + t, s1 = 8 * tig + 4 + t;   // lanes whose gid is 2*tig and 2*tig+1
        const uint32_t p0 = sel(b[0], s0), p1 = sel(b[0], s1);
        const uint32_t r0 = sel(b[1], s0), r1 = sel(b[1], s1);
        B0[2 * t] = mma_f16_lo(p0);      B0[2 * t + 1] = mma_f16_hi(p0);
        B1[2 * t] = mma_f16_lo(p1);      B1[2 * t + 1] = mma_f16_hi(p1);
        B0[2 * t + 8] = mma_f16_lo(r0);  B0[2 * t + 9] = mma_f16_hi(r0);
        B1[2 * t + 8] = mma_f16_lo(r1);  B1[2 * t + 9] = mma_f16_hi(r1);
    }
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        r0 = sycl::fma(A0[k], B0[k], r0);
        r1 = sycl::fma(A0[k], B1[k], r1);
        r2 = sycl::fma(A1[k], B0[k], r2);
        r3 = sycl::fma(A1[k], B1[k], r3);
    }
    c[0] += r0; c[1] += r1; c[2] += r2; c[3] += r3;
}
#else
inline void mma16816_f32(const sycl::sub_group&, float*, const uint32_t*, const uint32_t*) {
    // host pass: a kernel is the only caller, and this is never reached on the host
}
#endif

}  // namespace strata::sycl_compat
