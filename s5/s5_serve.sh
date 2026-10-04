#!/usr/bin/env bash
# S5: the served-path proof - a real /v1/chat/completions through serve.server with the config of record verbatim
# (checkpoints ON, the setting this card is about), using the S4 rig as-is, and the run copied under s5/ so this
# card carries its own raw evidence.
#
#   bash s5/s5_serve.sh <TAG> <ctx>:<max_new> [...]
set +e
SRC=/home/michael/strata-xpu/strata
TAG=${1:?tag}; shift
bash "$SRC/s4/s4_serve_arm.sh" "$TAG" "$@" > "$SRC/s5/runs/serve-$TAG.out" 2>&1
RC=$?
mkdir -p "$SRC/s5/served"
if [ -e "$SRC/s5/served/$TAG" ]; then
  echo "s5/served/$TAG already exists - leaving it alone (and the S4 copy is where the arm wrote it)"
else
  cp -r "$SRC/s4/runs/$TAG" "$SRC/s5/served/$TAG"
fi
echo "rc=$RC; s4/runs/$TAG -> s5/served/$TAG"
tail -4 "$SRC/s5/runs/serve-$TAG.out"
