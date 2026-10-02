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
# THE SUB-GROUP SIZE IS PINNED TO 32, and this is a CORRECTNESS requirement, not a performance one.
# MEASURED (M3, probe/probe24_pa_fp16.cpp + the hand-port canary): the B70 reports sub_group_sizes {16, 32}
# (M2, PLAN.md §10) and DPC++ picks per kernel.  A kernel under register pressure got an EFFECTIVE 16-wide
# sub-group while `get_sub_group().get_local_range()` still reported 32, and then
#   sycl::select_from_group(sg, mine, 20)  returned the caller's OWN value instead of lane 20's
# (probe24's dbg xchk line: lane 0, mine=1, asked for lane 20 and 31, got 1 for both).  Every CUDA `__shfl_*`
# with a source lane >= 16 therefore silently exchanged with the wrong work-item, which is the same class of
# defect M2 fixed for the 2-D nd_range (PROBE16) - the 1-D launch is necessary but NOT sufficient.  With this
# flag the sub-group is 32 for every kernel in the port; the parity tests are what prove it.
target_compile_options(strata_sycl_runtime INTERFACE -fsycl-default-sub-group-size=32)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/qsa_decode_attn.cu"     # 20 (M3; kv_stream_parity needs it)
    # M3 set: attention & mixers (PLAN.md §5 M3, card t_a44aa58f) - the QSA family, the mixers'
    # remaining files, the PLE/shared-expert paths and flash attention.  All 15 M3 .cu files translate
    # and compile: plan-evidence/M3-check-r1.txt (15 OK / 0 FAIL / 0 refused).
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_qsa.cu"          # 41 qsa_parity
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_qsa_indexer.cu"  # 21 qsa_parity
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_gdn_preprocess.cu" # 31
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/gr.cu"                  # 13 gr_parity
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_gr_postops.cu"   # 44 gr_parity
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/ple.cu"                 # 16 ple_parity, ple_q5_parity, ple_fp8_parity
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_ple_postops.cu"  # 28
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/shared_expert.cu"       # 17 shared_expert_parity
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/native_flash_attn.cu")  # 30

set(_strata_sycl_kernel_tus "")
foreach(_cu IN LISTS STRATA_SYCL_KERNELS)
  strata_sycl_generate("${_cu}" _gen)
  list(APPEND _strata_sycl_kernel_tus "${_gen}")
endforeach()
set(STRATA_SYCL_KERNEL_TUS "${_strata_sycl_kernel_tus}" CACHE INTERNAL "")

# ---- the 5 hand ports (PLAN.md §2.3, Risk 7) --------------------------------------------------------
# These are the files that carry the 16 inline-PTX sites.  tools/sycl/syclify.py REFUSES any file containing
# `asm` (that refusal is what keeps this list honest), so each one has a hand-ported SYCL source committed at
# src/kernels/sycl/<name>.cpp, built here as an ordinary source file.  Nothing is generated at build time for
# them: what is reviewed is what is compiled.  Rebuild one with
#   tools/sycl/handport.py <name>          # from the UNMODIFIED .cu + tools/sycl/handport/<name>.py
# and compile the whole set with tools/sycl/handport_check.sh.
#
# Risk 7 is the reason they are not "translated then patched": the guard the CUDA sources use,
# `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`, is TRUE when __CUDA_ARCH__ is undefined - i.e. under
# SYCL - so a transform that keeps the asm text can silently compile it.  Measured: 0 surviving asm tokens in
# every generated TU and in every hand port (plan-evidence/M3-risk7-asm.txt).
set(STRATA_SYCL_HAND_PORTS
    fused_gr            # 4  cp.async.cg.shared.global + commit_group + wait_group 1/0
    qsa_prompt_attn     # 6  mma.sync f16 x3, cp.async x3
    qsa_select          # 2  cvt.rna.tf32.f32, mma.sync tf32
    verify_kernels      # 1  mov.u64 %0, %%globaltimer
    native_qsa_score)   # 3  ldmatrix x2, mma.sync tf32
set(_strata_sycl_hand_tus "")
foreach(_hp IN LISTS STRATA_SYCL_HAND_PORTS)
  list(APPEND _strata_sycl_hand_tus "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sycl/${_hp}.cpp")
endforeach()
set(STRATA_SYCL_HAND_TUS "${_strata_sycl_hand_tus}" CACHE INTERNAL "")

# The arithmetic contract of PLAN.md §4.3, on the GENERATED translation units.  The same two files get
# `-ffp-contract=off` on CUDA and HIP (CMakeLists.txt:307-312, :350-359), because Q8_K's separately-rounded
# multiply and the magic-bias rounding change tie results when they are contracted into an FMA:
# quantize_act_parity fails without it (the Q8_K `d` scale came out one ulp off: ref -0.0232852157 vs
# -0.0232852139, 303338 differing bytes).  It has to be set HERE and not next to the CUDA/HIP branch, because
# under SYCL the file that is compiled is ${CMAKE_BINARY_DIR}/sycl/<name>.cpp - a property on the .cu would be
# a silent no-op.
set_source_files_properties("${STRATA_SYCL_GEN_DIR}/elementwise.cpp" "${STRATA_SYCL_GEN_DIR}/quantize_act.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# ---- M4: the prefill group (PLAN.md §4.2 Route S1, §5 M4, card t_086173b8) ---------------------------
# The prompt path's five sources.  Three of them are Route S1 itself - kernels.cu (the dequant/activation
# kernels the chunked path calls), gemm.cu (the cuBLAS surface, whose ONE call shape the compat cublas_v2.h now
# maps onto oneMKL's column_major::gemm for f32/bf16/f16) and prefill.cpp (the host driver, 0 __global__).
# The other two are the MMQ files: on this backend they compile to their documented REFUSAL form (see the guard
# at the top of each: ggml-CUDA's common.cuh/mmq.cuh/quantize.cuh are not part of a SYCL build, and PLAN §4.2
# Route S2 is ggml-SYCL's mmq.cpp against the PINNED commit, not these).  Building them here is what makes
# "compiles and refuses cleanly when the MMQ path is off" a build-verified statement rather than a claim:
# `syclify` translates all five, and a 6th source would have to be added deliberately.
# gemm.cu and prefill.cpp carry 0 launches, so their generated TUs are the sources unchanged; they go through
# the same generation step anyway, so the M4 group has ONE rule and every error names the original .cu/.cpp line.
set(STRATA_SYCL_PREFILL_CU
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/kernels.cu"        # 842  dequant/rms/activation kernels
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/gemm.cu"           # 432  0 __global__: the oneMKL target (Route S1)
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/prefill.cpp"       # 1977 0 __global__: the chunked prompt driver
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_mmq.cu"        # 231  MMQ (Route S2) - refusal form here
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/ggml_cuda_host.cu")# 134  MMQ host shims - refusal form here
set(_strata_sycl_prefill_tus "")
foreach(_cu IN LISTS STRATA_SYCL_PREFILL_CU)
  strata_sycl_generate("${_cu}" _gen)
  list(APPEND _strata_sycl_prefill_tus "${_gen}")
endforeach()
set(STRATA_SYCL_PREFILL_TUS "${_strata_sycl_prefill_tus}" CACHE INTERNAL "")

# Route S2 (PLAN §4.2): the ggml-SYCL MMQ path is NOT wired on this backend - it needs the pinned tree
# (`git archive <pin> | tar -x -C ~/strata-xpu/ggml-pin`, passed as -DSTRATA_GGML_DIR=...; never a
# worktree/checkout inside ~/llama.cpp-qwen4-exp, BRIEF §9), and it is a different source set (ggml-SYCL's
# mmq.cpp + common.cpp), not the two files above.  The switch exists so that asking for it REFUSES cleanly
# instead of silently doing nothing.
option(STRATA_SYCL_PREFILL_MMQ "Build the ggml-SYCL MMQ prefill path (PLAN §4.2 Route S2, NOT WIRED)" OFF)
if(STRATA_SYCL_PREFILL_MMQ)
  message(FATAL_ERROR
    "STRATA_SYCL_PREFILL_MMQ is not wired: PLAN.md §4.2 Route S2 needs the PINNED ggml tree "
    "(git archive 3cf03257f219afbe7334045ff7c6a06ac68c627d | tar -x -C ~/strata-xpu/ggml-pin) and ggml-SYCL's "
    "mmq.cpp/common.cpp built against it - the ggml-CUDA MMQ sources the HIP build uses cannot run on this "
    "backend. Leave the switch OFF: the prompt path then runs Route S1 (oneMKL) and REFUSES MMQ out loud "
    "(src/prefill/prefill.cpp's `#ifndef STRATA_PREFILL_MMQ` stubs print the reason when STRATA_PREFILL_MMQ=1 "
    "reaches the engine at runtime).")
endif()

message(STATUS "Strata: SYCL enabled, arch ${STRATA_SYCL_ARCHS}, targets ${STRATA_SYCL_TARGETS} "
               "(oneMKL: ${STRATA_SYCL_MKL}); ${_strata_sycl_kernel_tus}"
               "kernel translation unit(s) + ${_strata_sycl_prefill_tus} prefill translation unit(s) "
               "- the rest of PLAN.md §2 is M2..M5 work")
