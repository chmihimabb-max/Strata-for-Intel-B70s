#!/usr/bin/env bash
# P4 (card t_63cc226b) probe build.  The probe is a .cu because the launch syntax needs tools/sycl/syclify.py, and it
# links the ENGINE's own kernel library, so the copy kernel under test is the shipping one (kv_stream.cu), not a
# re-implementation.  Mirrors the flags CMake uses for the SYCL engine sources.
set -e
R=/home/michael/strata-xpu/strata
cd "$R"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
mkdir -p p4/gen
python3 tools/sycl/syclify.py p4/kv_io_probe.cu p4/gen/kv_io_probe.cpp --repo-root "$R"
ONE=/opt/intel/oneapi
icpx -O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64 -fno-sycl-rdc \
  -fsycl-device-code-split=per_kernel -fsycl-default-sub-group-size=32 -DMKL_ILP64 \
  -DSTRATA_SYCL_ARCHS=\"bmg-g31\" -DSTRATA_USE_SYCL=1 -DSTRATA_VERSION=\"0.1.34\" \
  -I "$R/third_party/ggml" -I "$R/include" -I "$R/include/strata/sycl_compat" \
  -isystem "$ONE/compiler/2026.1/include" -isystem "$ONE/mkl/2026.1/include" \
  -include "$R/include/strata/sycl_compat/cuda_runtime.h" \
  p4/gen/kv_io_probe.cpp -o p4/kv_io_probe \
  build-sycl/libstrata_kernels.a build-sycl/libstrata_core.a \
  "$ONE/compiler/2026.1/lib/libsycl.so" \
  -fsycl-allow-device-image-dependencies \
  -Wl,-rpath="$ONE/mkl/2026.1/lib" \
  "$ONE/mkl/2026.1/lib/libmkl_sycl_blas.so" "$ONE/mkl/2026.1/lib/libmkl_sycl_lapack.so" \
  "$ONE/mkl/2026.1/lib/libmkl_sycl_dft.so" "$ONE/mkl/2026.1/lib/libmkl_sycl_sparse.so" \
  "$ONE/mkl/2026.1/lib/libmkl_sycl_data_fitting.so" "$ONE/mkl/2026.1/lib/libmkl_sycl_rng.so" \
  "$ONE/mkl/2026.1/lib/libmkl_sycl_stats.so" "$ONE/mkl/2026.1/lib/libmkl_sycl_vm.so" \
  "$ONE/mkl/2026.1/lib/libmkl_intel_ilp64.so" "$ONE/mkl/2026.1/lib/libmkl_tbb_thread.so" \
  "$ONE/mkl/2026.1/lib/libmkl_core.so" \
  -Wl,-rpath="$ONE/tbb/2023.1/lib/intel64/gcc4.8" -L"$ONE/tbb/2023.1/lib/intel64/gcc4.8" \
  -ltbb -lm -ldl -lpthread
echo "built p4/kv_io_probe"
