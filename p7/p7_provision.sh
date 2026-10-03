#!/usr/bin/env bash
# P7 (card t_6789d6df): provision the MTP drafter the way upstream does, from the model's OWN canonical checkpoint.
#
#   inventory -> fetch -> verify -> mtp_pack.py --experts q2_0 -> mtp_rt.py -> cp data/draft_vocab.bin
#
# No W4A16 checkpoint, no tools/mtp_w4a16_adapter.py: the 31 `mtp.*` tensors come straight from
# Qwen/Qwen3.8-Flash-Next at tools/mtp_fetch.py's PINNED revision, SHA256-verified per tensor.
set -u
S=/home/michael/strata-xpu
SRC=$S/strata
MT=$S/mtp
CAN=$MT/canonical
PY=/usr/bin/python3
LOG=$S/logs/p7_provision.log
export STRATA_GGUF_PY=$SRC/build-sycl/_deps/strata_llamacpp-src/gguf-py
cd "$SRC" || exit 1
mkdir -p "$MT"
{
  echo "================================================================"
  echo "== P7 canonical MTP provisioning, started $(date -Is)"
  echo "== revision: STRATA_MTP_REVISION=${STRATA_MTP_REVISION:-<unset>}"
  grep -n "^PINNED_REVISION" tools/mtp_fetch.py
  echo "== gguf-py: $STRATA_GGUF_PY"
  echo
  echo "===================== 1a. inventory (headers only) ====================="
  "$PY" tools/mtp_fetch.py inventory --out "$CAN"
  echo "  inventory exit=$?"
  echo
  echo "===================== 1b. fetch (the 31 tensors, range reads) ====================="
  "$PY" tools/mtp_fetch.py fetch --out "$CAN"
  echo "  fetch exit=$?"
  echo
  echo "===================== 1c. verify (SHA256 of the pinned revision) ====================="
  "$PY" p7/p7_verify_report.py "$CAN"
  echo "  verify exit=$?"
  echo
  echo "===================== 2. mtp_pack.py --experts q2_0 ====================="
  "$PY" tools/mtp_pack.py --src "$CAN" --experts q2_0 --out "$MT/mtp-q2_0.gguf"
  echo "  mtp_pack exit=$?"
  echo
  echo "===================== 3. mtp_rt.py ====================="
  "$PY" tools/mtp_rt.py --gguf "$MT/mtp-q2_0.gguf" --out "$MT/rt"
  echo "  mtp_rt exit=$?"
  echo
  echo "===================== 4. draft_vocab.bin ====================="
  cp "$SRC/data/draft_vocab.bin" "$MT/rt/draft_vocab.bin"
  echo "  cp exit=$?  ($(stat -c %s "$MT/rt/draft_vocab.bin") B)"
  echo
  echo "-- artifacts --"
  ls -l "$MT"/*.gguf "$MT"/*.json 2>/dev/null
  ls -l "$MT/rt"
  echo
  echo "-- sha256 of the runtime the engine loads --"
  sha256sum "$MT/rt/dense.bin" "$MT/rt/experts.bin"
  echo
  echo "-- disk --"
  df -h /home/michael
  du -sh "$CAN" "$MT"
  echo "== done $(date -Is)"
} >> "$LOG" 2>&1
echo "P7_PROVISION_DONE $(date -Is)" >> "$LOG"
