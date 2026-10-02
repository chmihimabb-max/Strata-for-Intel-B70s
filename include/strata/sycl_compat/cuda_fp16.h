#pragma once
// include/strata/sycl_compat/cuda_fp16.h - CUDA's half-precision header, for the SYCL backend.
//
// The engine uses __half, __half2 and the conversion helpers below (61 __half2float, 27 __ushort_as_half,
// 27 __half2, 20 __half22float2, 12 __float2half_rn, ...).  SYCL's sycl::half is the same IEEE-754 binary16
// type, so __half is an alias.  sycl::half2 is sycl::vec<half,2>, whose components are ACCESSORS (.x()) and
// not members, so __half2 is a two-member struct here rather than an alias: kernel text writes `v.x`.
#include <sycl/sycl.hpp>

#include <cstdint>

using __half = sycl::half;

struct float2 {
    float x = 0.0f, y = 0.0f;
};

/// CUDA's __half2: two halves, .x and .y as members (sycl::vec spells them .x()/.y()).
struct __half2 {
    sycl::half x;
    sycl::half y;
    __half2() = default;
    __half2(sycl::half a, sycl::half b) : x(a), y(b) {}
};

inline __half __float2half(float f) { return sycl::half(f); }
inline __half __float2half_rn(float f) { return sycl::half(f); }   // SYCL's software conversion is RN
inline __half __float2half_rz(float f) { return sycl::half(f); }
inline float __half2float(__half h) { return (float) h; }
inline float __low2float(__half2 h) { return (float) h.x; }
inline float __high2float(__half2 h) { return (float) h.y; }

inline __half2 __floats2half2_rn(float a, float b) { return __half2{sycl::half(a), sycl::half(b)}; }
inline __half2 __halves2half2(__half a, __half b) { return __half2{a, b}; }
inline __half __low2half(__half2 h) { return h.x; }
inline __half __high2half(__half2 h) { return h.y; }

/// CUDA orders the two halves of a __half2 with .x in the LOW 16 bits.
inline unsigned short __half_as_ushort(__half h) { return (unsigned short) sycl::bit_cast<uint16_t>(h); }
inline __half __ushort_as_half(unsigned short u) { return sycl::bit_cast<sycl::half>(u); }

inline float2 __half22float2(__half2 h) { return float2{(float) h.x, (float) h.y}; }
inline float2 __half2float2(__half2 h) { return __half22float2(h); }

inline __half2 __hmul2(__half2 a, __half2 b) {
    return __half2{sycl::half((float) a.x * (float) b.x), sycl::half((float) a.y * (float) b.y)};
}
inline __half2 __hadd2(__half2 a, __half2 b) {
    return __half2{sycl::half((float) a.x + (float) b.x), sycl::half((float) a.y + (float) b.y)};
}
inline __half __hmul(__half a, __half b) { return sycl::half((float) a * (float) b); }
inline __half __hadd(__half a, __half b) { return sycl::half((float) a + (float) b); }

/// The sycl::half2 view of a __half2, for code that wants a SYCL vector.
inline sycl::half2 as_sycl_half2(__half2 h) { return sycl::half2{h.x, h.y}; }
inline __half2 from_sycl_half2(sycl::half2 h) { return __half2{h.x(), h.y()}; }
