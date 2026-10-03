#!/usr/bin/env bash
# P1b: copy the final binary aside and curate the evidence the card's write-up cites (the repo copy under p1/evidence).
R=/home/michael/strata-xpu
SRC=$R/strata
EV=$SRC/p1/evidence
mkdir -p "$EV" "$R/p1/runs"
cp "$SRC/build-sycl/strata" "$R/p1/strata-after-P1b"
{
  echo "binary md5s ($(date -Is))"
  md5sum "$SRC/build-sycl/strata" "$R/p1/strata-after-P1b" "$R/p1/strata-after-P1" "$R/p1/strata-before-timed" 2>/dev/null
  echo "repo: $(cd $SRC && git log --oneline -1)"
} | tee "$EV/binaries-p1b.txt"

# the failure path, before (P1's binary + P1b's rig knob only) and after
cp "$R/p1/runs/p1b-a-longdrain/err.txt"        "$EV/p1b-a-longdrain-err.txt"
cp "$R/p1/runs/p1b-e-fixed/err.txt"            "$EV/p1b-e-fixed-err.txt"
cp "$R/p1/runs/p1b-t-stall/err.txt"            "$EV/p1b-t-stall-err.txt"
cp "$R/p1/runs/p1b-l-secondask/err.txt"        "$EV/p1b-l-secondask-err.txt"
cp "$R/p1/runs/p1b-l-secondask/out.txt"        "$EV/p1b-l-secondask-out.txt"
cp "$R/p1/runs/p1b-l-secondask/timeline.txt"   "$EV/p1b-l-secondask-timeline.txt"
# the clean path (token ids + timings)
for a in p1b-j-clean4k p1b-k-clean32k p1b-u-clean4k; do
  cp "$R/p1/runs/$a/tokens.txt" "$EV/$a-tokens.txt"
  cp "$R/p1/runs/$a/out.txt"    "$EV/$a-out.txt"
  cp "$R/p1/runs/$a/log.txt"    "$EV/$a-log.txt"
done
cp "$R/p1/runs/p1b-k-clean32k/err.txt"         "$EV/p1b-k-clean32k-err.txt"
cp "$R/p1/runs/p1b-u-clean4k/err.txt"          "$EV/p1b-u-clean4k-err.txt"
# the frames
sed -n '/=== ALL THREADS ===/,$p' "$R/p1/logs/p1b-m-gdb7-out.log" | head -60 > "$EV/p1b-m-gdb7-gdb-threads.txt"
cp "$R/p1/logs/p1b-d-abort.log"                "$EV/p1b-d-abort-log.txt"
# the instrument arms, verdict sections
for t in p1b-f-ut7 p1b-h-ut9 p1b-i-ut1 p1b-n-ut7 p1b-q-ut9 p1b-r-ut7; do
  sed -n '/instrument.s own verdict/,$p' "$R/p1/logs/$t.log" > "$EV/$t-verdict.txt"
  cp "$R/p1/logs/$t-eng.log" "$EV/$t-eng.txt"
done
sed -n '/instrument.s own verdict/,$p' "$R/p1/logs/p1b-v-ut1.log" > "$EV/p1b-v-ut1-verdict.txt"
cp "$R/p1/logs/p1b-v-ut1-eng.log" "$EV/p1b-v-ut1-eng.txt"
cp "$R/p1/logs/p1b-battery.log"  "$EV/p1b-battery.log"
cp "$R/p1/logs/p1b-battery2.log" "$EV/p1b-battery2.log"
cp "$R/p1/logs/p1b-battery3.log" "$EV/p1b-battery3.log"
cp "$R/p1/logs/p1b-battery4.log" "$EV/p1b-battery4.log"
ls -la "$EV" | tail -30
