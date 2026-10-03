#!/usr/bin/env bash
# M6c: the 256K needle through the independent oracle, end to end -- launch llama.cpp-SYCL, wait for it,
# ask the identical 259,943-token prompt, stop it, keep the answers.
# usage: bash m6c/m6c_oracle_needle_run.sh <tag>
R=/home/michael/strata-xpu
TAG=${1:?tag}
PORT=${PORT:-58244}
D=$R/m6c/oracle
mkdir -p "$D"
export PORT
bash "$R/strata/m6c/m6c_oracle_needle.sh" "$TAG"
for i in $(seq 1 120); do
  if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
    echo "oracle ready after ${i}0 s-ish"; break
  fi
  sleep 10
done
curl -s --max-time 5 "http://127.0.0.1:$PORT/health" || echo "no health"
/usr/bin/python3 "$R/strata/m6c/m6c_oracle_needle.py" "$PORT" \
  "$R/m6c/prompts/prompt-needle-ctx262144.txt" "$D/$TAG-needle.json" 256 2>&1 | tail -25
for q in $(pgrep -f "llama-server -m"); do kill -9 "$q" 2>/dev/null || true; done
echo "oracle stopped $(date -Is)"
