#!/usr/bin/env bash
# P1b: put this card's harnesses in the repo copy and stage the commit's files.
R=/home/michael/strata-xpu
SRC=$R/strata
for f in p1b_run_engine.sh p1b_battery.sh p1b_battery2.sh p1b_battery3.sh p1b_battery4.sh p1b_gdb_abort.sh p1b_md5.sh p1b_clean_report.sh p1b_evidence.sh; do
  cp "$R/p1/$f" "$SRC/p1/evidence/$f"
done
cp "$SRC/p1/STATUS-P1B.md" "$R/P1B-STATUS.md"
ls -la "$SRC/p1/evidence/" | grep p1b_ 
echo "---- repo status:"
cd "$SRC" && git status --short | head -40
