#pragma once
// include/strata/sycl_compat/vector_types.h - CUDA's builtin vector types, for the SYCL backend.
//
// CUDA's <vector_types.h> gives every arithmetic type a 1/2/3/4-component struct whose components are
// MEMBERS (`v.x`, `v.y`, `v.z`, `v.w`) with CUDA's alignment (1/2/4 components of 4-byte scalars are
// 4/8/16-aligned; `float4`/`int4`/`uint4` are 16-aligned, which is what makes
// `*reinterpret_cast<const float4*>(p)` a 16-byte load in both languages).
//
// They cannot be aliases of `sycl::vec`: sycl::vec spells its components `v.x()` / `v.y()`, and 600+ sites in
// this tree write `v.x`.  A struct is therefore the faithful mapping, and the `make_*` constructors keep the
// CUDA spelling too.
//
// Included from both cuda_runtime.h (force-included into every source) and cuda_fp16.h (which the kernels
// include themselves for __half/half2).
#include <cstdint>

// ---------------------------------------------------------------------------
// 1/2/4-component vectors.  3-component vectors follow the same pattern (added
// below) - CUDA has them and mrope/rope code uses float3 for positions.
// ---------------------------------------------------------------------------
#define STRATA_SYCL_VEC1(T, N)   \
    struct alignas(sizeof(T)) N { \
        T x;                      \
    }
#define STRATA_SYCL_VEC2(T, N)        \
    struct alignas(2 * sizeof(T)) N { \
        T x, y;                       \
    }
#define STRATA_SYCL_VEC3(T, N)        \
    struct alignas(4 * sizeof(T)) N { \
        T x, y, z;                    \
    }
#define STRATA_SYCL_VEC4(T, N)        \
    struct alignas(4 * sizeof(T)) N { \
        T x, y, z, w;                 \
    }

// char/uchar have no vector form in CUDA below 4 components; char4 is a real CUDA type (used by kv_q8.cu).
STRATA_SYCL_VEC1(char, char1);
STRATA_SYCL_VEC2(char, char2);
STRATA_SYCL_VEC3(char, char3);
STRATA_SYCL_VEC4(char, char4);
STRATA_SYCL_VEC1(unsigned char, uchar1);
STRATA_SYCL_VEC2(unsigned char, uchar2);
STRATA_SYCL_VEC3(unsigned char, uchar3);
STRATA_SYCL_VEC4(unsigned char, uchar4);
STRATA_SYCL_VEC1(short, short1);
STRATA_SYCL_VEC2(short, short2);
STRATA_SYCL_VEC3(short, short3);
STRATA_SYCL_VEC4(short, short4);
STRATA_SYCL_VEC1(unsigned short, ushort1);
STRATA_SYCL_VEC2(unsigned short, ushort2);
STRATA_SYCL_VEC3(unsigned short, ushort3);
STRATA_SYCL_VEC4(unsigned short, ushort4);
STRATA_SYCL_VEC1(int, int1);
STRATA_SYCL_VEC2(int, int2);
STRATA_SYCL_VEC3(int, int3);
STRATA_SYCL_VEC4(int, int4);
STRATA_SYCL_VEC1(unsigned int, uint1);
STRATA_SYCL_VEC2(unsigned int, uint2);
STRATA_SYCL_VEC3(unsigned int, uint3);
STRATA_SYCL_VEC4(unsigned int, uint4);
STRATA_SYCL_VEC1(long long, longlong1);
STRATA_SYCL_VEC2(long long, longlong2);
STRATA_SYCL_VEC3(long long, longlong3);
STRATA_SYCL_VEC4(long long, longlong4);
STRATA_SYCL_VEC1(unsigned long long, ulonglong1);
STRATA_SYCL_VEC2(unsigned long long, ulonglong2);
STRATA_SYCL_VEC3(unsigned long long, ulonglong3);
STRATA_SYCL_VEC4(unsigned long long, ulonglong4);
STRATA_SYCL_VEC1(float, float1);
STRATA_SYCL_VEC2(float, float2);
STRATA_SYCL_VEC3(float, float3);
STRATA_SYCL_VEC4(float, float4);
STRATA_SYCL_VEC2(double, double2);

#undef STRATA_SYCL_VEC1
#undef STRATA_SYCL_VEC2
#undef STRATA_SYCL_VEC3
#undef STRATA_SYCL_VEC4

// ---------------------------------------------------------------------------
// The make_* constructors (CUDA's, same argument order).
// ---------------------------------------------------------------------------
// CUDA's make_* constructors take exactly as many arguments as the type has components
// (make_char1(x) / make_char2(x,y) / make_char3(x,y,z) / make_char4(x,y,z,w)).
#define STRATA_SYCL_MAKE_TYPES(T, N) \
    inline N make_##N(T x) { return N{x}; }
#define STRATA_SYCL_MAKE_TYPES2(T, N) \
    inline N make_##N(T x, T y) { return N{x, y}; }
#define STRATA_SYCL_MAKE_TYPES3(T, N) \
    inline N make_##N(T x, T y, T z) { return N{x, y, z}; }
#define STRATA_SYCL_MAKE_TYPES4(T, N) \
    inline N make_##N(T x, T y, T z, T w) { return N{x, y, z, w}; }

STRATA_SYCL_MAKE_TYPES(char, char1)
STRATA_SYCL_MAKE_TYPES2(char, char2)
STRATA_SYCL_MAKE_TYPES3(char, char3)
STRATA_SYCL_MAKE_TYPES4(char, char4)
STRATA_SYCL_MAKE_TYPES(unsigned char, uchar1)
STRATA_SYCL_MAKE_TYPES2(unsigned char, uchar2)
STRATA_SYCL_MAKE_TYPES3(unsigned char, uchar3)
STRATA_SYCL_MAKE_TYPES4(unsigned char, uchar4)
STRATA_SYCL_MAKE_TYPES(short, short1)
STRATA_SYCL_MAKE_TYPES2(short, short2)
STRATA_SYCL_MAKE_TYPES3(short, short3)
STRATA_SYCL_MAKE_TYPES4(short, short4)
STRATA_SYCL_MAKE_TYPES(unsigned short, ushort1)
STRATA_SYCL_MAKE_TYPES2(unsigned short, ushort2)
STRATA_SYCL_MAKE_TYPES3(unsigned short, ushort3)
STRATA_SYCL_MAKE_TYPES4(unsigned short, ushort4)
STRATA_SYCL_MAKE_TYPES(int, int1)
STRATA_SYCL_MAKE_TYPES2(int, int2)
STRATA_SYCL_MAKE_TYPES3(int, int3)
STRATA_SYCL_MAKE_TYPES4(int, int4)
STRATA_SYCL_MAKE_TYPES(unsigned int, uint1)
STRATA_SYCL_MAKE_TYPES2(unsigned int, uint2)
STRATA_SYCL_MAKE_TYPES3(unsigned int, uint3)
STRATA_SYCL_MAKE_TYPES4(unsigned int, uint4)
STRATA_SYCL_MAKE_TYPES(long long, longlong1)
STRATA_SYCL_MAKE_TYPES2(long long, longlong2)
STRATA_SYCL_MAKE_TYPES3(long long, longlong3)
STRATA_SYCL_MAKE_TYPES4(long long, longlong4)
STRATA_SYCL_MAKE_TYPES(unsigned long long, ulonglong1)
STRATA_SYCL_MAKE_TYPES2(unsigned long long, ulonglong2)
STRATA_SYCL_MAKE_TYPES3(unsigned long long, ulonglong3)
STRATA_SYCL_MAKE_TYPES4(unsigned long long, ulonglong4)
STRATA_SYCL_MAKE_TYPES(float, float1)
STRATA_SYCL_MAKE_TYPES2(float, float2)
STRATA_SYCL_MAKE_TYPES3(float, float3)
STRATA_SYCL_MAKE_TYPES4(float, float4)
STRATA_SYCL_MAKE_TYPES2(double, double2)

#undef STRATA_SYCL_MAKE_TYPES
#undef STRATA_SYCL_MAKE_TYPES2
#undef STRATA_SYCL_MAKE_TYPES3
#undef STRATA_SYCL_MAKE_TYPES4
