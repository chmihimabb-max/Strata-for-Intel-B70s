#!/usr/bin/env bash
# D3 (card t_87aa2963): every raw line the card's acceptance asks for, into d3/D3-EVIDENCE.txt.
#   bash d3/d3_evidence.sh
R=/home/michael/strata-xpu/strata
cd "$R" || exit 1
OUT=$R/d3/D3-EVIDENCE.txt
{
  echo "############ D3 (t_87aa2963) raw evidence   $(date -Is)"
  echo "repo: $(git log --oneline -1)   branch $(git branch --show-current)"
  echo
  echo "############ 1. the arm table (d3/d3_report.py)"
  /usr/bin/python3 d3/d3_report.py --root "$R/d3/runs" --full
  echo
  echo "############ 2. the census per arm (d3/d3_select.py: per-window device time, the selection, the handshake)"
  for h in "$R"/d3/runs/*/hist.txt; do
    [ -s "$h" ] || continue
    /usr/bin/python3 d3/d3_select.py "$h"
  done
  echo
  echo "############ 3. per-arm raw lines (the engine's own)"
  for d in "$R"/d3/runs/*/; do
    tag=$(basename "$d")
    [ -f "$d/log.txt" ] || continue
    echo "---- $tag"
    grep -E "^(ctx=|spec=|onecard=|ZE_AFFINITY|engine binary|HEAD:)" "$d/log.txt"
    grep -E "== the ask finished|engine exit" "$d/log.txt"
    grep -E "strata decode timing|strata decode GPU stages" "$d/err.txt" | tail -2
    grep -E "strata submit: " "$d/err.txt" | tail -2
    grep -E "^DONE|^ERR " "$d/out.txt" | tail -1
    echo "   T lines $(grep -c '^T ' "$d/out.txt")   md5 $(grep '^T ' "$d/out.txt" | md5sum | cut -c1-32)"
    grep -E "STRATA_SCORES_MULTI|STRATA_TOPK_OLD|STRATA_VERIFY_DEVICE_PLAN|the graph path|STRATA_SYCL_GRAPH" "$d/err.txt" | head -3
    echo
  done
  echo "############ 4. the #267 stall arms (the release path's guarantee)"
  for d in "$R"/d3/stall/*/; do
    tag=$(basename "$d")
    [ -f "$d/log.txt" ] || continue
    echo "---- $tag"
    grep -E "^binary:|^env:|^arm:|exit " "$d/log.txt" | head -5
    grep -E "timed out at layer|verify release|teardown|released|engine ends now|an earlier window" "$d/err.txt" | tail -12
    echo "   T lines $(grep -c '^T ' "$d/out.txt")   md5 $(grep '^T ' "$d/out.txt" | md5sum | cut -c1-32)"
    echo
  done
  echo "############ 5. the state this card left the machine in"
  pgrep -a -x strata || echo "no engine running"
  pgrep -a -f "serve/server.py" || echo "no server running"
  ss -ltnp 2>/dev/null | grep 8099 || echo "port 8099 free"
  fuser -v /dev/dri/renderD128 /dev/dri/renderD129 2>&1 | tail -6
  cat /proc/loadavg
  echo "build-sycl/strata md5 $(md5sum < "$R/build-sycl/strata" | cut -c1-32)"
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
