#!/usr/bin/env bash
# D2c: rebuild (the two-line warning fix) and then run the cold chain.
set +e
SRC=/home/michael/strata-xpu/strata
bash "$SRC/d2c/build_engine.sh"
echo "######## run chain A"
bash "$SRC/d2c/d2c_chain_a.sh"
echo "######## ALL DONE $(date -Is)"
