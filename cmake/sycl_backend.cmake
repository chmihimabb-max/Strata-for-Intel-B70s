# Opt-in SYCL (Intel XPU / oneAPI) configuration - the third Strata backend, alongside CUDA and HIP.
#
# Contract: ~/strata-xpu/PLAN.md §4.1.  The shape mirrors cmake/hip_backend.cmake, with one structural
# difference that PLAN.md D1 fixes: HIP works through a pure rename because hipcc *is* a CUDA-like front end.
# icpx is not - `kernel<<<grid, block, smem, stream>>>(args)` has no C++ spelling, and the tree has 284-321
# such sites - so every engine `.cu` is TRANSLATED first (tools/sycl/syclify.py) and the generated C++ is what
# gets compiled, as C++ with `-x c++` (measured: `icpx -fsycl -c x.cu` fails looking for libdevice).
#
# Configure with:  source /opt/intel/oneapi/setvars.sh && cmake -S . -B build-sycl \
#                    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#
# WHAT THIS BACKEND BUILDS TODAY (M1, card t_b1028685): the device/memory layer (`strata_core`),
# `strata-device`, and the kernels the one GPU-free parity harness needs.  The rest of the 50 `.cu` files are
# M2..M5 work and are listed per-file in PLAN.md §2; a `.cu` that is NOT in STRATA_SYCL_KERNELS below is
# simply not compiled, which is deliberate - the alternative (compiling files the port does not yet support)
# would produce a build whose failures say nothing about what works.

# ---- compiler ---------------------------------------------------------------------------------------
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  message(FATAL_ERROR
    "STRATA_ENABLE_SYCL needs the Intel DPC++ compiler: configure with -DCMAKE_CXX_COMPILER=icpx after "
    "`source /opt/intel/oneapi/setvars.sh` (CMAKE_CXX_COMPILER_ID is '${CMAKE_CXX_COMPILER_ID}').")
endif()

set(STRATA_SYCL_ARCHS "bmg-g31" CACHE STRING
    "Strata SYCL target architecture(s): bmg-g31 = Intel Arc Pro B70 (BMG-G31, 8086:e223)")
set(STRATA_SYCL_TARGETS "spir64" CACHE STRING "SYCL device targets passed to -fsycl-targets")

set(STRATA_SYCL_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/sycl_compat")

# IntelSYCL gives IntelSYCL::SYCL_CXX (the SYCL include dirs and the link flags); oneMKL gives
# MKL::MKL_SYCL (MKLConfig.cmake:1044), which is `_strata_gpu_blas_target` for this backend.
find_package(IntelSYCL REQUIRED)
if(NOT TARGET MKL::MKL_SYCL)
  find_package(MKL CONFIG QUIET)
endif()
if(TARGET MKL::MKL_SYCL)
  set(STRATA_SYCL_MKL ON)
else()
  set(STRATA_SYCL_MKL OFF)
  message(STATUS "Strata SYCL: oneMKL (MKL::MKL_SYCL) not found; the prefill GEMM path (M4) is unavailable")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---- the runtime interface target -------------------------------------------------------------------
add_library(strata_sycl_runtime INTERFACE)
target_include_directories(strata_sycl_runtime BEFORE INTERFACE
  "${STRATA_SYCL_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_sycl_runtime INTERFACE
  STRATA_USE_SYCL=1 "STRATA_SYCL_ARCHS=\"${STRATA_SYCL_ARCHS}\"")
target_compile_options(strata_sycl_runtime INTERFACE
  -fsycl
  -fsycl-targets=${STRATA_SYCL_TARGETS}
  -fno-sycl-rdc                              # no device function pointers across TUs; shorter link
  -fsycl-device-code-split=per_kernel)       # the engine loads every kernel at startup
# The compat cuda_runtime.h is force-included into every host and device source, exactly as the HIP target
# force-includes its own (hip_backend.cmake:75-83).  A source that does not mention cudaMalloc still gets the
# shim for the types it passes around.
target_compile_options(strata_sycl_runtime INTERFACE
  "-include" "${STRATA_SYCL_COMPAT_INCLUDE_DIR}/cuda_runtime.h")
target_link_libraries(strata_sycl_runtime INTERFACE IntelSYCL::SYCL_CXX)
if(STRATA_SYCL_MKL)
  target_link_libraries(strata_sycl_runtime INTERFACE MKL::MKL_SYCL)
endif()

set(STRATA_SYCL_RUNTIME_TARGET strata_sycl_runtime)
set(STRATA_SYCL_BLAS_TARGET MKL::MKL_SYCL)
set(STRATA_SYCL_INCLUDE_DIRECTORIES "${STRATA_SYCL_COMPAT_INCLUDE_DIR}")

# ---- the .cu -> .cpp translation --------------------------------------------------------------------
set(STRATA_SYCL_GEN_DIR "${CMAKE_BINARY_DIR}/sycl")
set(STRATA_SYCL_SYCLIFY "${CMAKE_CURRENT_SOURCE_DIR}/tools/sycl/syclify.py")
set(STRATA_SYCL_EXCEPTIONS "${CMAKE_CURRENT_SOURCE_DIR}/tools/sycl/exceptions.txt")
if(NOT EXISTS "${STRATA_SYCL_SYCLIFY}")
  message(FATAL_ERROR "STRATA_ENABLE_SYCL needs ${STRATA_SYCL_SYCLIFY}")
endif()

# strata_sycl_generate(<source.cu> <out_var>): add the translation step and return the generated path.
# The generated files carry their own #line markers, so a compile error names the ORIGINAL .cu and line.
function(strata_sycl_generate cu out_var)
  get_filename_component(_name "${cu}" NAME_WE)
  set(_out "${STRATA_SYCL_GEN_DIR}/${_name}.cpp")
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${Python3_EXECUTABLE}" "${STRATA_SYCL_SYCLIFY}" "${cu}" "${_out}"
            --exceptions "${STRATA_SYCL_EXCEPTIONS}" --repo-root "${CMAKE_SOURCE_DIR}"
    DEPENDS "${cu}" "${STRATA_SYCL_SYCLIFY}" "${STRATA_SYCL_EXCEPTIONS}"
    COMMENT "syclify ${_name}")
  set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

# ---- the M1 target set ------------------------------------------------------------------------------
# src/core/device.cu + src/core/pinned.cu are REUSED UNCHANGED (PLAN.md §4.1.5): the single source of truth
# for the device and the pinned expert arena, translated like every other engine source.
set(STRATA_SYCL_CORE_CU
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/device.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/pinned.cu")
set(_strata_sycl_core_tus "")
foreach(_cu IN LISTS STRATA_SYCL_CORE_CU)
  strata_sycl_generate("${_cu}" _gen)
  list(APPEND _strata_sycl_core_tus "${_gen}")
endforeach()
set(STRATA_SYCL_CORE_TUS "${_strata_sycl_core_tus}")

# The kernels this backend compiles TODAY.  Every one of them passes
#   icpx -fsycl -x c++ -c  (tools/sycl/compile_survey.sh, evidence in M1's completion comment;
#   tools/sycl/m2_check.sh for the same loop over one milestone's file list)
# and the .cu is unmodified - only the generated TU is built.
#
# M1 set:   dequant_s2.cu (the GPU-free harness M1's acceptance names).
# M2 set:   the decode hot path (PLAN.md §2 M2, card t_6f4e9d6c) - the expert/quant/router/rope/KV/
#           sampler group, every file of which now survives the full compile: see
#           plan-evidence/M2-check-r7.txt (28 OK / 0 FAIL / 0 refused).
set(STRATA_SYCL_KERNELS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/dequant_s2.cu"          # 50
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/s2_gemv.cu"            # 49
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/s2_gemv_quads.cu"       # 47
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/s2_gemv_q8.cu"          # 46
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/s2_gemv_fast.cu"        # 35
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/s_gemv.cu"              # 10
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/iq_kernels.cu"          # 2
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_mmvq.cu"         # 1
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/s2_expert_grouped.cu"   # 3
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/router_top10.cu"        # 15
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_router.cu"       # 42 (router_top10_parity)
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/rope.cu"                # 38
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_rope.cu"         # 36
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_bf16.cu"         # 32
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/bf16_gemv.cu"           # 37
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/elementwise.cu"         # 18
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/quantize_act.cu"        # 19
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/kv_q8.cu"               # 40
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/kv_q4.cu"               # 26
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/kv_stream.cu"           # 22
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/sampler.cu"             # 6
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/dequant_bf16.cu"        # 23
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/gdn.cu"                 # 24
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/fused_gdn.cu"           # 34
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_gdn.cu"          # 43
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_gr_norm.cu"      # 45
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/cvec.cu"                # 33
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_moe.cu"            # 48
    # qsa.cu is M3's file, but it carries the two KV entry points kv_q8_parity links against
    # (kv_append_step/kv_gather_step, qsa.cu:595/641) and it now translates and compiles cleanly.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/qsa.cu"                  # 8 (M3; kv_q8_parity needs it)
    # qsa_decode_attn.cu is the other M3 file kv_stream_parity links against (qsa_decode_attn_batch,
    # qsa_decode_attn.cu), which is the whole point of that test: streamed vs resident attention.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/qsa_decode_attn.cu")     # 20 (M3; kv_stream_parity needs it)

set(_strata_sycl_kernel_tus "")
foreach(_cu IN LISTS STRATA_SYCL_KERNELS)
  strata_sycl_generate("${_cu}" _gen)
  list(APPEND _strata_sycl_kernel_tus "${_gen}")
endforeach()
set(STRATA_SYCL_KERNEL_TUS "${_strata_sycl_kernel_tus}" CACHE INTERNAL "")

# The arithmetic contract of PLAN.md §4.3, on the GENERATED translation units.  The same two files get
# `-ffp-contract=off` on CUDA and HIP (CMakeLists.txt:307-312, :350-359), because Q8_K's separately-rounded
# multiply and the magic-bias rounding change tie results when they are contracted into an FMA:
# quantize_act_parity fails without it (the Q8_K `d` scale came out one ulp off: ref -0.0232852157 vs
# -0.0232852139, 303338 differing bytes).  It has to be set HERE and not next to the CUDA/HIP branch, because
# under SYCL the file that is compiled is ${CMAKE_BINARY_DIR}/sycl/<name>.cpp - a property on the .cu would be
# a silent no-op.
set_source_files_properties("${STRATA_SYCL_GEN_DIR}/elementwise.cpp" "${STRATA_SYCL_GEN_DIR}/quantize_act.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

message(STATUS "Strata: SYCL enabled, arch ${STRATA_SYCL_ARCHS}, targets ${STRATA_SYCL_TARGETS} "
               "(oneMKL: ${STRATA_SYCL_MKL}); ${_strata_sycl_kernel_tus}"
               "kernel translation unit(s) - the rest of PLAN.md §2 is M2..M5 work")
