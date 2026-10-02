#!/usr/bin/env bash
# Collect the raw M1 acceptance output into plan-evidence/.  Run from ~/strata-xpu/strata after a build.
set -u
cd "$(dirname "$0")/../.." || exit 1
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
OUT=/home/michael/strata-xpu/plan-evidence/M1-acceptance.txt
{
  echo "### M1 acceptance raw output - $(date -Is)"
  echo
  echo "## git"; git log --oneline -1; git status --short | head -20
  echo
  echo "## GPU exclusivity check (no vLLM container, no ollama model)"
  echo "\$ docker ps"; docker ps 2>&1 | head -5
  echo "\$ ollama ps"; ollama ps 2>&1 | head -5
  echo
  echo "## compiler"; icpx --version | head -1
  echo
  echo "## ./build-sycl/strata-device --list-devices"
  ./build-sycl/strata-device --list-devices
  echo
  echo "## ZE_AFFINITY_MASK=0 ./build-sycl/strata-device --selftest"
  ZE_AFFINITY_MASK=0 ./build-sycl/strata-device --selftest; echo "exit=$?"
  echo
  echo "## ZE_AFFINITY_MASK=1 ./build-sycl/strata-device --selftest"
  ZE_AFFINITY_MASK=1 ./build-sycl/strata-device --selftest; echo "exit=$?"
  echo
  echo "## ZE_AFFINITY_MASK=0 ./build-sycl/strata-device --list-devices (the mask really selects one card)"
  ZE_AFFINITY_MASK=0 ./build-sycl/strata-device --list-devices
  echo "## ZE_AFFINITY_MASK=1 ./build-sycl/strata-device --list-devices"
  ZE_AFFINITY_MASK=1 ./build-sycl/strata-device --list-devices
  echo
  echo "## ./build-sycl/dequant_s2_parity --selftest"
  ./build-sycl/dequant_s2_parity --selftest; echo "exit=$?"
  echo
  echo "## ctest --test-dir build-sycl"
  ctest --test-dir build-sycl --timeout 300 2>&1 | tail -8
} > "$OUT" 2>&1
echo "wrote $OUT"
