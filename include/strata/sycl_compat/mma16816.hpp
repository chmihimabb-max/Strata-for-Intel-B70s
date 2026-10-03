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

#if defined(__SYCL_DEVICE_ONLY__)
/// The kernel's TWO-CALL form: A's hi half and A's lo half against the SAME B fragment, i.e.
///     mma16816(c, ah, b);  mma16816(c, al, b);
/// which is how every q.k and p.v site in qsa_prompt_attn.cu uses it.  B is the same operand in both calls, so
/// its rebuild - 16 `select_from_group` gathers and 32 f16 -> f32 conversions - is done ONCE instead of twice.
///
/// NUMERICALLY IDENTICAL to the two calls, and checkable by eye: each half runs its own 16-term k loop in the
/// same order into its own accumulators (same A[k] * B[k] products, same order, same rounding), and the two
/// halves then reach c in the same order the two calls did (the hi half first, then the lo half).  Nothing
/// about the products, their order or the accumulator widths changes; only the exchange's redundancy is gone.
///
/// MEASURED, this device (BMG-G31), bench/micro/p2_qsa_emul_cost.cpp: the shipped pair 5.848 ms against the
/// fused pair 4.755 ms for 4000 iterations over 8192 work-items = 18.7% cheaper, against a 128-FMA floor of
/// 0.792 ms - i.e. the emulation is ~6x the arithmetic it replaces even after this.
inline void mma16816_f32_pair(const sycl::sub_group& sg, float* c, const uint32_t* ahi, const uint32_t* alo,
                              const uint32_t* b) {
    const int lane = (int) sg.get_local_linear_id();
    const int tig = lane & 3;
    auto sel = [&](uint32_t v, int src) { return sycl::select_from_group(sg, v, (uint32_t) src); };

    float AH0[16], AH1[16], AL0[16], AL1[16], B0[16], B1[16];   // rows gid/gid+8, columns 2*tig / 2*tig+1
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int q = (lane & ~3) | t;          // the quad mate that holds k columns 2t, 2t+1, 2t+8, 2t+9
        const uint32_t h0 = sel(ahi[0], q), h1 = sel(ahi[1], q), h2 = sel(ahi[2], q), h3 = sel(ahi[3], q);
        const uint32_t l0 = sel(alo[0], q), l1 = sel(alo[1], q), l2 = sel(alo[2], q), l3 = sel(alo[3], q);
        AH0[2 * t] = mma_f16_lo(h0);      AH0[2 * t + 1] = mma_f16_hi(h0);
        AH1[2 * t] = mma_f16_lo(h1);      AH1[2 * t + 1] = mma_f16_hi(h1);
        AH0[2 * t + 8] = mma_f16_lo(h2);  AH0[2 * t + 9] = mma_f16_hi(h2);
        AH1[2 * t + 8] = mma_f16_lo(h3);  AH1[2 * t + 9] = mma_f16_hi(h3);
        AL0[2 * t] = mma_f16_lo(l0);      AL0[2 * t + 1] = mma_f16_hi(l0);
        AL1[2 * t] = mma_f16_lo(l1);      AL1[2 * t + 1] = mma_f16_hi(l1);
        AL0[2 * t + 8] = mma_f16_lo(l2);  AL0[2 * t + 9] = mma_f16_hi(l2);
        AL1[2 * t + 8] = mma_f16_lo(l3);  AL1[2 * t + 9] = mma_f16_hi(l3);
        const int s0 = 8 * tig + t, s1 = 8 * tig + 4 + t;   // lanes whose gid is 2*tig and 2*tig+1
        const uint32_t p0 = sel(b[0], s0), p1 = sel(b[0], s1);
        const uint32_t r0 = sel(b[1], s0), r1 = sel(b[1], s1);
        B0[2 * t] = mma_f16_lo(p0);      B0[2 * t + 1] = mma_f16_hi(p0);
        B1[2 * t] = mma_f16_lo(p1);      B1[2 * t + 1] = mma_f16_hi(p1);
        B0[2 * t + 8] = mma_f16_lo(r0);  B0[2 * t + 9] = mma_f16_hi(r0);
        B1[2 * t + 8] = mma_f16_lo(r1);  B1[2 * t + 9] = mma_f16_hi(r1);
    }
    float rh0 = 0.0f, rh1 = 0.0f, rh2 = 0.0f, rh3 = 0.0f;   // the hi call's accumulators
    float rl0 = 0.0f, rl1 = 0.0f, rl2 = 0.0f, rl3 = 0.0f;   // the lo call's
    // The two halves' k loops are INTERLEAVED, which is free: each accumulator still takes its own 16 products
    // in its own k order (nothing about a product, its order or its rounding changes), there are simply eight
    // independent chains instead of four, so the scheduler can fill the gaps.  Measured: this is where the win
    // is, not in the shared B rebuild (which alone measures as a wash).
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        rh0 = sycl::fma(AH0[k], B0[k], rh0);
        rh1 = sycl::fma(AH0[k], B1[k], rh1);
        rh2 = sycl::fma(AH1[k], B0[k], rh2);
        rh3 = sycl::fma(AH1[k], B1[k], rh3);
        rl0 = sycl::fma(AL0[k], B0[k], rl0);
        rl1 = sycl::fma(AL0[k], B1[k], rl1);
        rl2 = sycl::fma(AL1[k], B0[k], rl2);
        rl3 = sycl::fma(AL1[k], B1[k], rl3);
    }
    // ... and the two halves reach c in the same order the two calls did: the hi half, then the lo half.
    c[0] += rh0; c[1] += rh1; c[2] += rh2; c[3] += rh3;
    c[0] += rl0; c[1] += rl1; c[2] += rl2; c[3] += rl3;
}
#else
inline void mma16816_f32_pair(const sycl::sub_group&, float*, const uint32_t*, const uint32_t*, const uint32_t*) {
    // host pass, as above
}
#endif

}  // namespace strata::sycl_compat
