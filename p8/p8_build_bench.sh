#!/usr/bin/env bash
# P8 probe build: a standalone SYCL program (no engine sources), so it is a second opinion on the engine's own
# probe_pcie_h2d_gbps().  Same compiler and SYCL target as build-sycl.
set -e
R=/home/michael/strata-xpu/strata
cd "$R"
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u
icpx -O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64 \
  p8/p8_h2d_bench.cpp -o p8/p8_h2d_bench
echo "built p8/p8_h2d_bench"
