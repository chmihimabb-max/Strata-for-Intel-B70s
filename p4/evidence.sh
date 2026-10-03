#!/usr/bin/env bash
# P4 (card t_63cc226b): assemble the raw evidence file from the probe output and the engine arms.
# usage: bash p4/evidence.sh [probe-output-file]
R=/home/michael/strata-xpu
cd "$R/strata"
PROBE=${1:-p4/out-all-run1.txt}
OUT=p4/evidence/P4-EVIDENCE.txt
mkdir -p p4/evidence
{
  echo "P4 (t_63cc226b) raw evidence   $(date -Is)"
  echo "host: $(uname -sr)  $(nproc) cores  $(free -g | awk '/Mem:/{print $2" GiB RAM"}')"
  echo "binary in every engine arm: build-sycl/strata sha256 $(sha256sum < build-sycl/strata | cut -d' ' -f1)"
  echo "git HEAD: $(git log --oneline -1)"
  echo
  echo "==============================================================="
  echo "1. THE PROBE (one B70, ZE_AFFINITY_MASK=0): $PROBE"
  echo "==============================================================="
  cat "$PROBE"
  echo
  echo "==============================================================="
  echo "2. THE ENGINE ARMS (both B70s, config of record, m6c harness)"
  echo "==============================================================="
  for d in "$R"/m6c/runs/p4-*; do
    [ -d "$d" ] || continue
    t=$(basename "$d")
    echo
    echo "---- $t"
    echo "-- arm line: $(grep -hF "arm $t:" p4/arms.log 2>/dev/null | tail -1)"
    echo "-- summary (P3's extractor: decode timing, DONE, token-id count + md5, peak RSS/VRAM):"
    /usr/bin/python3 "$R/p3/extract.py" "$t" 2>/dev/null | grep -vE "^={10}|^ARM |^  strata serve: layer split|^  strata generate: (expert|layer)" | head -12
    echo "-- KV volume line (err.txt):"
    grep -hE "serve: KV streaming" "$d/err.txt" 2>/dev/null | tail -1
  done
} > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
