#!/usr/bin/env bash
# the report lives outside the board, so keep a copy in this profile's kanban attachments dir
set +e
R=/home/michael/strata-xpu/strata
A=/home/michael/.hermes/kanban/attachments/t_2b6b6797
mkdir -p "$A"
cp -f "$R/d2/plain/STATUS-D2B-PLAINGR.md" "$A/"
cp -f "$R/d2/plain/ladder-1c.txt" "$A/"
cp -f "$R/d2/plain/ladder.txt" "$A/"
cp -f "$R/d2/plain/grparity.txt" "$A/"
cp -f "$R/d2/plain/m5g.txt" "$A/"
cp -f "$R/d2/STATUS-D2.md" "$A/STATUS-D2.md"
cp -f "$R/d2/plain/d2b_ladder.py" "$A/"
ls -la "$A"
