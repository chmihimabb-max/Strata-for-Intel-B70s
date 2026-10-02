#pragma once
// include/strata/sycl_compat/intrinsics.hpp - the CUDA device intrinsics the Strata kernels use, for SYCL.
//
// Included only from the SYCL compat shim (which is force-included into every source of the backend), exactly
// like include/strata/hip_compat/intrinsics.hpp.  The mapping and the semantics each one must preserve are
// fixed by PLAN.md §1.2; where an intrinsic has no faithful SYCL equivalent the documented emulation is here
// rather than in kernel code.
//
// Two rules that are easy to get wrong and are therefore called out:
//   * Sub-group shuffles.  The device reports sub_group_sizes {16, 32} (PLAN.md §10) - NOT {8,16,32}.  CUDA's
//     `width` argument partitions the warp into segments of `width` lanes and shuffles WITHIN a segment, so
//     every shuffle here computes the logical segment base itself instead of assuming a full 32-lane warp.
//     A partial lane MASK is refused loudly, as the HIP shim does: silently dropping mask semantics is how a
//     race becomes a wrong number.
//   * atomicAdd/atomicCAS are DEVICE-scope relaxed in CUDA.  Weakening the scope would lose the doorbell
//     ring's publication (PLAN.md §3.1); keep memory_scope::device.
#include <cstdint>

namespace strata::sycl_compat {

// ---------------------------------------------------------------------------
// bit casts (CUDA's __int_as_float & friends)
// ---------------------------------------------------------------------------
inline float int_as_float(int x) { return sycl::bit_cast<float>(x); }
inline int float_as_int(float x) { return sycl::bit_cast<int>(x); }
inline float uint_as_float(unsigned x) { return sycl::bit_cast<float>(x); }
inline unsigned float_as_uint(float x) { return sycl::bit_cast<unsigned>(x); }
inline long long double_as_longlong(double x) { return sycl::bit_cast<long long>(x); }
inline double longlong_as_double(long long x) { return sycl::bit_cast<double>(x); }

// ---------------------------------------------------------------------------
// integer bit operations
// ---------------------------------------------------------------------------
inline int popc(unsigned x) { return (int) sycl::popcount(x); }
inline int popcll(unsigned long long x) { return (int) sycl::popcount(x); }
inline int clz_int(unsigned x) { return x == 0u ? 32 : (int) sycl::clz(x); }
inline int clz_ll(unsigned long long x) { return x == 0ull ? 64 : (int) sycl::clz(x); }
inline int ffs_int(int x) { return x == 0 ? 0 : (int) sycl::ctz((unsigned) x) + 1; }

/// CUDA's __brev: reverse the 32 bits.  Pure SWAR, so it is the same value on every target.
inline unsigned brev(unsigned x) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    return (x << 16) | (x >> 16);
}
inline unsigned long long brevll(unsigned long long x) {
    return ((unsigned long long) brev((unsigned) x) << 32) | (unsigned long long) brev((unsigned) (x >> 32));
}

/// CUDA's __funnelshift_l/_r: (hi:lo) shifted by shift & 31, taking the low 32 bits of the 64-bit funnel.
inline unsigned funnel_shift_left(unsigned lo, unsigned hi, unsigned shift) {
    const unsigned s = shift & 31u;
    if (s == 0u) return lo;
    return (unsigned) ((((unsigned long long) hi << 32 | lo) << s) >> 32);
}
inline unsigned funnel_shift_right(unsigned lo, unsigned hi, unsigned shift) {
    const unsigned s = shift & 31u;
    if (s == 0u) return hi;
    return (unsigned) (((unsigned long long) hi << 32 | lo) >> s);
}

// ---------------------------------------------------------------------------
// __dp4a: four signed byte products, accumulated modulo 2^32.
//
// There is NO hardware int8 dot outside XMX/DPAS on this device (PLAN.md §8.1.2), so every available spelling
// is four multiplies (sycl::ext::oneapi::dot_product.hpp, ggml-sycl's dpct::dp4a, this one).  Risk 5 measures
// what that costs in the GEMV hot path at M2; the semantics here are the contract, not the speed.
// ---------------------------------------------------------------------------
inline int dp4a(int a, int b, int c) {
    const uint32_t ua = (uint32_t) a, ub = (uint32_t) b;
    uint32_t sum = (uint32_t) c;
    for (int lane = 0; lane < 4; ++lane) {
        const uint32_t sa = (ua >> (lane * 8)) & 0xffu;
        const uint32_t sb = (ub >> (lane * 8)) & 0xffu;
        sum += (uint32_t) ((int) sa < 0x80 ? (int) sa : (int) sa - 0x100) *
               (uint32_t) ((int) sb < 0x80 ? (int) sb : (int) sb - 0x100);
    }
    return (int) sum;
}

// ---------------------------------------------------------------------------
// __byte_perm (default mode): result byte i is byte s.nibble[i] & 7 of the pair {y:x}, x the low word.
// The HIP backend measured +15% decode from getting this into one instruction (docs/AMD_HIP.md:212); SYCL's
// sycl::permute is the same byte selection, so the nibble selector is spread to bytes exactly as HIP does.
// ---------------------------------------------------------------------------
inline uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    // PLAN.md D4 maps this to `sycl::permute`.  MEASURED: there is no `sycl::permute` in oneAPI 2026.1 - the
    // only `permute*` in the SYCL headers is sycl::permute_group_by_xor, a sub-group shuffle.  So this is the
    // portable form: four byte extracts out of the {y:x} pair.  It costs more instructions than the HIP
    // backend's single v_perm_b32 (which measured +15% decode, docs/AMD_HIP.md:212-214); pricing that on Xe
    // is M2 work (Risk 5).  The SEMANTICS here are exact.
    const uint64_t pair = ((uint64_t) y << 32) | (uint64_t) x;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const int idx = (int) ((s >> (i * 4)) & 0x7u);
        r |= (uint32_t) ((pair >> (idx * 8)) & 0xffull) << (i * 8);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Packed-byte SWAR, ported verbatim from include/strata/hip_compat/intrinsics.hpp:55-79 (pure integer math,
// no vendor builtin - gfx11/gfx12 have no packed 8-bit subtract and neither does Xe).
// ---------------------------------------------------------------------------
constexpr uint32_t kHigh = 0x80808080u;

inline int vsub4(int a, int b) {
    const uint32_t ua = (uint32_t) a, ub = (uint32_t) b;
    return (int) (((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}
inline int vsubss4(int a, int b) {
    const uint32_t ua = (uint32_t) a, ub = (uint32_t) b;
    const uint32_t d = (uint32_t) vsub4(a, b);
    const uint32_t overflow = (ua ^ ub) & (ua ^ d) & kHigh;
    const uint32_t mask = (overflow >> 7) * 0xffu;
    const uint32_t bound = 0x7f7f7f7fu + ((ua & kHigh) >> 7);
    return (int) ((d & ~mask) | (bound & mask));
}
inline int vcmpne4(int a, int b) {
    const uint32_t t = (uint32_t) a ^ (uint32_t) b;
    const uint32_t nonzero = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
    return (int) ((nonzero >> 7) * 0xffu);
}

// ---------------------------------------------------------------------------
// Sub-group shuffles.  The logical-segment rule (Risk 6): CUDA partitions the warp into segments of `width`
// lanes and shuffles within a segment, so the source lane is computed relative to this lane's segment base.
// ---------------------------------------------------------------------------
inline void require_full_mask(uint32_t mask) {
    // CUDA's full-warp mask.  A partial mask means a kernel relies on divergence semantics we do not emulate;
    // the HIP shim traps here for the same reason (hip_compat/intrinsics.hpp:83-85).
    if (mask != 0xffffffffu && mask != 0xffffu) __builtin_trap();
}

inline int segment_base(int lane, int width) {
    const int n = (int) this_item().get_sub_group().get_local_range().get(0);
    const int w = width <= 0 ? n : (width < n ? width : n);
    return (lane / w) * w;
}

/// Shuffles whose value type is narrower than 32 bits go through a uint32_t.
///
/// MEASURED (M2): `sycl::select_from_group` on a `uint8_t` returns 0 on this backend, so
/// `__shfl_down_sync(mask, qc, 16)` in kv_q4.cu's Q4_0 quantizer produced `byte = qc | (0 << 4)` - the high
/// nibble of every packed byte was zero and 15584 of the Q4_0 blocks differed from the host quantizer
/// (kv_q4_parity), with the same shape of failure in the streamed q4_0 attention.  Widening the value to a
/// 32-bit word for the exchange keeps the caller's type and the CUDA semantics.
template <typename T>
inline T shfl_xor_sync(uint32_t mask, T value, int lane_mask, int width = 32) {
    require_full_mask(mask);
    const auto sg = this_item().get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int n = (int) sg.get_local_range().get(0);
    const int w = width <= 0 ? n : (width < n ? width : n);
    const int base = segment_base(lane, w);
    const int src = base + (((lane - base) ^ lane_mask) & (w - 1));
    if constexpr (sizeof(T) < sizeof(uint32_t)) {
        const uint32_t got = sycl::select_from_group(sg, (uint32_t) value, (uint32_t) (src % n));
        return (T) got;
    }
    return sycl::select_from_group(sg, value, (uint32_t) (src % n));
}
template <typename T>
inline T shfl_down_sync(uint32_t mask, T value, unsigned delta, int width = 32) {
    require_full_mask(mask);
    const auto sg = this_item().get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int n = (int) sg.get_local_range().get(0);
    const int w = width <= 0 ? n : (width < n ? width : n);
    const int base = segment_base(lane, w);
    const int src = (lane - base) + (int) delta;
    if (src >= w) return value;   // CUDA returns the caller's own value when the source is out of range
    if constexpr (sizeof(T) < sizeof(uint32_t)) {
        const uint32_t got = sycl::select_from_group(sg, (uint32_t) value, (uint32_t) ((base + src) % n));
        return (T) got;
    }
    return sycl::select_from_group(sg, value, (uint32_t) ((base + src) % n));
}
template <typename T>
inline T shfl_up_sync(uint32_t mask, T value, unsigned delta, int width = 32) {
    require_full_mask(mask);
    const auto sg = this_item().get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int n = (int) sg.get_local_range().get(0);
    const int w = width <= 0 ? n : (width < n ? width : n);
    const int base = segment_base(lane, w);
    const int src = (lane - base) - (int) delta;
    if (src < 0) return value;
    if constexpr (sizeof(T) < sizeof(uint32_t)) {
        const uint32_t got = sycl::select_from_group(sg, (uint32_t) value, (uint32_t) ((base + src) % n));
        return (T) got;
    }
    return sycl::select_from_group(sg, value, (uint32_t) ((base + src) % n));
}
template <typename T>
inline T shfl_sync(uint32_t mask, T value, int source_lane, int width = 32) {
    require_full_mask(mask);
    const auto sg = this_item().get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int n = (int) sg.get_local_range().get(0);
    const int w = width <= 0 ? n : (width < n ? width : n);
    const int base = segment_base(lane, w);
    const int src = base + ((source_lane % w) + w) % w;
    if constexpr (sizeof(T) < sizeof(uint32_t)) {
        const uint32_t got = sycl::select_from_group(sg, (uint32_t) value, (uint32_t) (src % n));
        return (T) got;
    }
    return sycl::select_from_group(sg, value, (uint32_t) (src % n));
}

/// __ballot_sync: a 32-bit mask, one bit per lane, oldest lane in bit 0.
inline unsigned ballot_sync(uint32_t mask, int predicate) {
    require_full_mask(mask);
    const auto sg = this_item().get_sub_group();
    sycl::ext::oneapi::sub_group_mask m = sycl::ext::oneapi::group_ballot(sg, predicate != 0);
    unsigned bits = 0;
    m.extract_bits(bits);   // the extension's conversion for a 32-lane mask (sub_group_mask.hpp:199)
    return bits;
}
inline unsigned activemask() {
    const auto sg = this_item().get_sub_group();
    sycl::ext::oneapi::sub_group_mask m = sycl::ext::oneapi::group_ballot(sg, true);
    unsigned bits = 0;
    m.extract_bits(bits);
    return bits;
}

// ---------------------------------------------------------------------------
// Atomics: DEVICE scope, relaxed, as CUDA's are.  See the file comment.
//
// NO EXPLICIT ADDRESS SPACE, i.e. sycl::atomic_ref's default `generic_space`.  That is not a style choice:
// the kernels apply these to GLOBAL memory (kv_stream's page table, sampler's hit masks) AND to SHARED memory
// (`atomicAdd(&s_nmiss, 1)` in kv_stream.cu:103/107, where `s_nmiss` is a static __shared__ int that the
// transform turns into a local-address-space object).  `address_space::global_space` forced an
// `addrspacecast local -> global`, which llvm-spirv refuses outright ("Invalid SPIR-V module: Casts from
// private/local/global address space are allowed only to generic") - measured on kv_stream_parity, whose
// link failed until this changed.  A generic-space atomic_ref is legal for both.
// ---------------------------------------------------------------------------
template <typename T>
inline T atomicAdd(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    return a.fetch_add(v);
}
template <typename T>
inline T atomicExch(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    return a.exchange(v);
}
template <typename T>
inline T atomicMax(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    return a.fetch_max(v);
}
inline int atomicCAS(int* p, int compare, int val) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    a.compare_exchange_strong(compare, val);   // compare is updated to the value actually read
    return compare;                            // CUDA's atomicCAS returns the old value
}
inline unsigned atomicCAS(unsigned* p, unsigned compare, unsigned val) {
    sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    a.compare_exchange_strong(compare, val);
    return compare;
}
template <typename T>
inline T atomicOr(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    return a.fetch_or(v);
}
template <typename T>
inline T atomicAnd(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    return a.fetch_and(v);
}
template <typename T>
inline T atomicMin(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*p);
    return a.fetch_min(v);
}

// ---------------------------------------------------------------------------
// Cache hints and backoff.
//   __ldg: a read-only load.  SYCL has no cache-hint form for USM; a plain load is correctness-identical and
//          the performance difference is unmeasured (PLAN.md §1.2).
//   __nanosleep: NO-OP.  Documented deviation, matching the HIP shim's constant hint; it is only ever used as
//          a spin backoff (PLAN.md §1.2/§1.3d).
//   __trap: the caller asked for a hard stop.
// ---------------------------------------------------------------------------
template <typename T>
inline T __ldg_impl(const T* p) {
    return *p;
}

// ---------------------------------------------------------------------------
// Device math CUDA provides as intrinsics.  The fast variants map onto SYCL's native (lower-accuracy) set,
// which is the same intent; SYCL has no exact bit-for-bit __expf and none of these is used on a path whose
// result is compared bit-exactly (that is what the parity tests are for).
// ---------------------------------------------------------------------------
inline float expf_fast(float x) { return sycl::native::exp(x); }
inline float logf_fast(float x) { return sycl::native::log(x); }
inline float sinf_fast(float x) { return sycl::native::sin(x); }
inline float cosf_fast(float x) { return sycl::native::cos(x); }
inline float powf_fast(float x, float y) { return sycl::native::powr(x, y); }
inline float rsqrtf_fast(float x) { return sycl::rsqrt(x); }
inline float rcpf_fast(float x) { return 1.0f / x; }
inline float fdividef_fast(float x, float y) { return x / y; }
inline float saturatef(float x) { return sycl::clamp(x, 0.0f, 1.0f); }
inline float fmaf_rn(float a, float b, float c) { return sycl::fma(a, b, c); }
inline float fminf_(float a, float b) { return sycl::fmin(a, b); }
inline float fmaxf_(float a, float b) { return sycl::fmax(a, b); }
inline float fabsf_(float a) { return sycl::fabs(a); }
inline float __hadd_scalar(float a, float b) { return a + b; }

// ---------------------------------------------------------------------------
// The remaining device math CUDA declares in <device_functions.h>/<cuda_runtime.h>.  Each is the IEEE
// operation CUDA documents for its name; SYCL's non-native spellings are the same roundings (M2 measured the
// parity tests that depend on them - elementwise/quantize_act compare bitwise against a CPU reference).
// ---------------------------------------------------------------------------
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fsqrt_rn(float a) { return sycl::sqrt(a); }
inline float __fmaf_rz(float a, float b, float c) { return sycl::fma(a, b, c); }
inline int __float2int_rn(float a) { return (int) sycl::rint(a); }
inline int __float2int_rz(float a) { return (int) sycl::trunc(a); }
inline int __float2int_rd(float a) { return (int) sycl::floor(a); }
inline int __float2int_ru(float a) { return (int) sycl::ceil(a); }
inline unsigned __float2uint_rn(float a) { return (unsigned) sycl::rint(a); }
inline unsigned __float2uint_rz(float a) { return (unsigned) sycl::trunc(a); }
inline unsigned __float2uint_rd(float a) { return (unsigned) sycl::floor(a); }
inline unsigned __float2uint_ru(float a) { return (unsigned) sycl::ceil(a); }
inline float __int2float_rn(int a) { return (float) a; }
inline float __int2float_rz(int a) { return (float) a; }
inline float __uint2float_rn(unsigned a) { return (float) a; }
inline float __uint2float_rz(unsigned a) { return (float) a; }
inline double __double2int_rn(double a) { return sycl::rint(a); }
inline float __double2float_rn(double a) { return (float) a; }
inline float __exp10f(float x) { return sycl::native::exp10(x); }
inline float __log2f(float x) { return sycl::native::log2(x); }
inline float __log10f(float x) { return sycl::native::log10(x); }
inline float __tanf(float x) { return sycl::tan(x); }
inline float __asinf(float x) { return sycl::asin(x); }
inline float __acosf(float x) { return sycl::acos(x); }
inline float __atan2f(float y, float x) { return sycl::atan2(y, x); }
inline float __fpowf(float x, float y) { return sycl::pow(x, y); }
inline float __fsqrtf_rz(float x) { return sycl::sqrt(x); }

// The double-precision spellings (__dadd_rn/__dmul_rn/__ddiv_rn/__dsqrt_rn are used by the sampler's
// probability math, which is computed in double and compared bitwise in sampler_parity).
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __dsub_rn(double a, double b) { return a - b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __ddiv_rn(double a, double b) { return a / b; }
inline double __dsqrt_rn(double a) { return sycl::sqrt(a); }
inline double __drcp_rn(double a) { return 1.0 / a; }
inline double __fma_rn(double a, double b, double c) { return sycl::fma(a, b, c); }

/// The CUDA device spelling of these two is what the kernels call, and neither has a DPC++ device wrapper:
/// `rsqrtf` is a GNU extension glibc declares for the host only, and `__isnanf` comes from glibc's
/// bits/mathcalls.h.  The transform renames the call sites to these (RENAMES in tools/sycl/syclify.py).
inline float isnanf_dev(float x) { return sycl::isnan(x) ? 1 : 0; }
inline int isinff_dev(float x) { return sycl::isinf(x) ? 1 : 0; }

/// CUDA's PLAIN `isnan` (not `__isnanf`) has the same problem, and src/prefill/kernels.cu is the one file that
/// calls it from inside a kernel (kernels.cu:43 hf_sat, kernels.cu:472 the router's softmax guard).  Same
/// treatment: a device-callable wrapper plus a call-site RENAME, so the .cu stays as it is.
inline bool isnan_dev(float x) { return sycl::isnan(x); }

/// The three fences.  CUDA's __threadfence() is a DEVICE-scope release/acquire fence, __threadfence_block()
/// a work-group one, and __threadfence_system() a SYSTEM-scope one - which is the one the host<->device
/// doorbell ring needs (PLAN.md Risk 8, measured by tests/sycl/handoff.cpp).
inline void threadfence() { sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device); }
inline void threadfence_block() { sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group); }
inline void threadfence_system() { sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system); }

/// The B70's sub-group is 32 lanes, so "wave" reads as the sub-group here.
inline unsigned warp_size_lanes() { return (unsigned) this_item().get_sub_group().get_local_range().get(0); }

}  // namespace strata::sycl_compat

// ---------------------------------------------------------------------------
// The CUDA spelling of each intrinsic, as macros, so kernel text is unchanged.
// ---------------------------------------------------------------------------
#define __dp4a(a, b, c) (::strata::sycl_compat::dp4a((a), (b), (c)))
#define __byte_perm(x, y, s) (::strata::sycl_compat::byte_perm((x), (y), (s)))
#define __vsub4(a, b) (::strata::sycl_compat::vsub4((a), (b)))
#define __vsubss4(a, b) (::strata::sycl_compat::vsubss4((a), (b)))
#define __vcmpne4(a, b) (::strata::sycl_compat::vcmpne4((a), (b)))
#define __shfl_xor_sync(...) (::strata::sycl_compat::shfl_xor_sync(__VA_ARGS__))
#define __shfl_down_sync(...) (::strata::sycl_compat::shfl_down_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (::strata::sycl_compat::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (::strata::sycl_compat::shfl_sync(__VA_ARGS__))
#define __ballot_sync(mask, predicate) (::strata::sycl_compat::ballot_sync((mask), (predicate)))
#define __activemask() (::strata::sycl_compat::activemask())
#define __popc(x) (::strata::sycl_compat::popc((unsigned) (x)))
#define __popcll(x) (::strata::sycl_compat::popcll((unsigned long long) (x)))
#define __clz(x) (::strata::sycl_compat::clz_int((unsigned) (x)))
#define __clzll(x) (::strata::sycl_compat::clz_ll((unsigned long long) (x)))
#define __ffs(x) (::strata::sycl_compat::ffs_int((int) (x)))
#define __brev(x) (::strata::sycl_compat::brev((unsigned) (x)))
#define __brevll(x) (::strata::sycl_compat::brevll((unsigned long long) (x)))
#define __funnelshift_l(lo, hi, shift) (::strata::sycl_compat::funnel_shift_left((lo), (hi), (shift)))
#define __funnelshift_r(lo, hi, shift) (::strata::sycl_compat::funnel_shift_right((lo), (hi), (shift)))
#define __int_as_float(x) (::strata::sycl_compat::int_as_float((x)))
#define __float_as_int(x) (::strata::sycl_compat::float_as_int((x)))
#define __uint_as_float(x) (::strata::sycl_compat::uint_as_float((x)))
#define __float_as_uint(x) (::strata::sycl_compat::float_as_uint((x)))
#define __double_as_longlong(x) (::strata::sycl_compat::double_as_longlong((x)))
#define __longlong_as_double(x) (::strata::sycl_compat::longlong_as_double((x)))
#define __ldg(p) (::strata::sycl_compat::__ldg_impl((p)))
#define __ldca(p) (::strata::sycl_compat::__ldg_impl((p)))
#define __stcg(p, v) (*(p) = (v))
#define __trap() __builtin_trap()
#define __brkpt() __builtin_trap()
#define __nanosleep(cycles) ((void) 0)   // documented no-op, PLAN.md §1.2/§1.3(d)
// __expf / __logf / __sinf / __cosf / __powf are DELIBERATELY NOT macros here.  glibc's <math.h> declares
// `extern float __cosf(float)` and its four siblings (bits/mathcalls.h:62,64,117,126,177 - the __*f builtin
// aliases), so a macro of the same name expands INSIDE the system header and breaks it ("no member named
// 'strata' in the global namespace").  MEASURED with probe/macro_collide.py: exactly these five collide, the
// other 38 do not.  Their CUDA spelling is rewritten at the call site instead, by the RENAMES table in
// tools/sycl/syclify.py - the same mechanism, one layer earlier.
#define __frsqrt_rn(x) (::strata::sycl_compat::rsqrtf_fast((x)))
#define __frcp_rn(x) (::strata::sycl_compat::rcpf_fast((x)))
#define __fdividef(x, y) (::strata::sycl_compat::fdividef_fast((x), (y)))
#define __saturatef(x) (::strata::sycl_compat::saturatef((x)))
#define __fmaf_rn(a, b, c) (::strata::sycl_compat::fmaf_rn((a), (b), (c)))
#define __laneid() (::strata::sycl_compat::lane_id())

// M2 additions: the round-to-nearest/literal-rounding float ops, the float<->int conversions, and the three
// fences.  None of these names is declared by glibc's <math.h> (unlike __expf/__logf/__sinf/__cosf/__powf,
// which are renamed at the call site instead - see the RENAMES table in tools/sycl/syclify.py), so a macro is
// safe and keeps the kernel text unchanged.
#define __fadd_rn(a, b) (::strata::sycl_compat::__fadd_rn((a), (b)))
#define __fsub_rn(a, b) (::strata::sycl_compat::__fsub_rn((a), (b)))
#define __fmul_rn(a, b) (::strata::sycl_compat::__fmul_rn((a), (b)))
#define __fdiv_rn(a, b) (::strata::sycl_compat::__fdiv_rn((a), (b)))
#define __fsqrt_rn(a) (::strata::sycl_compat::__fsqrt_rn((a)))
#define __fmaf_rz(a, b, c) (::strata::sycl_compat::__fmaf_rz((a), (b), (c)))
#define __float2int_rn(a) (::strata::sycl_compat::__float2int_rn((a)))
#define __float2int_rz(a) (::strata::sycl_compat::__float2int_rz((a)))
#define __float2int_rd(a) (::strata::sycl_compat::__float2int_rd((a)))
#define __float2int_ru(a) (::strata::sycl_compat::__float2int_ru((a)))
#define __float2uint_rn(a) (::strata::sycl_compat::__float2uint_rn((a)))
#define __float2uint_rz(a) (::strata::sycl_compat::__float2uint_rz((a)))
#define __float2uint_rd(a) (::strata::sycl_compat::__float2uint_rd((a)))
#define __float2uint_ru(a) (::strata::sycl_compat::__float2uint_ru((a)))
#define __int2float_rn(a) (::strata::sycl_compat::__int2float_rn((a)))
#define __int2float_rz(a) (::strata::sycl_compat::__int2float_rz((a)))
#define __uint2float_rn(a) (::strata::sycl_compat::__uint2float_rn((a)))
#define __uint2float_rz(a) (::strata::sycl_compat::__uint2float_rz((a)))
#define __double2float_rn(a) (::strata::sycl_compat::__double2float_rn((a)))
// the double-precision spellings (qsa.cu's attention math)
#define __dadd_rn(a, b) (::strata::sycl_compat::__dadd_rn((a), (b)))
#define __dsub_rn(a, b) (::strata::sycl_compat::__dsub_rn((a), (b)))
#define __dmul_rn(a, b) (::strata::sycl_compat::__dmul_rn((a), (b)))
#define __ddiv_rn(a, b) (::strata::sycl_compat::__ddiv_rn((a), (b)))
#define __dsqrt_rn(a) (::strata::sycl_compat::__dsqrt_rn((a)))
#define __drcp_rn(a) (::strata::sycl_compat::__drcp_rn((a)))
#define __fma_rn(a, b, c) (::strata::sycl_compat::__fma_rn((a), (b), (c)))
#define __threadfence() (::strata::sycl_compat::threadfence())
#define __threadfence_block() (::strata::sycl_compat::threadfence_block())
#define __threadfence_system() (::strata::sycl_compat::threadfence_system())
#define __umulhi(a, b) ((unsigned) (((unsigned long long) (unsigned) (a) * (unsigned long long) (unsigned) (b)) >> 32))
