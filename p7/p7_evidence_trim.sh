#!/usr/bin/env bash
# Keep only the summary lines of each bench arm's stdout in the evidence bundle (the raw out.txt is ~143 KB of
# per-window/per-position lines; the numbers the card quotes are the summary block).
set -u
EV=/home/michael/strata-xpu/strata/p7/evidence
R=/home/michael/strata-xpu/p7/runs
for pair in "bench-4k-drafter:p7-bench-4096-mtp" "bench-4k-headless:p7-bench-4096-headless" \
            "bench-32k-drafter:p7-bench-32768-mtp" "bench-32k-headless:p7-bench-32768-headless"; do
  out=${pair%%:*}
  dir=${pair##*:}
  grep -E "^speculation|^window sizes|^suffix drafts|^accepted per round|^decode |^prefill |^verify window" \
       "$R/$dir/out.txt" | cut -c1-200 > "$EV/$out.summary.txt"
done
ls -la "$EV"
wc -l "$EV"/*.summary.txt
