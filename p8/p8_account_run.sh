#!/usr/bin/env bash
# P8: the H2D bench under the account probe (which the buffer landed in, VRAM or system memory, and what the link
# registers said while it ran).  setvars is sourced here: oneAPI's runtime library path is what the bench needs.
#   bash p8/p8_account_run.sh [MiB] [iters]
set -e
R=/home/michael/strata-xpu/strata
cd "$R"
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u
exec /usr/bin/python3 p8/p8_pcie_account.py "${1:-256}" "${2:-20}"
