#!/usr/bin/env bash
# D2b (t_2b6b6797): file the evidence.  Raw dumps go into the arm's own run dir (D2's convention), the scripts
# and the derived tables into strata/d2/plain/, and my leftovers leave the other card's strata/d2b/ alone.
set +e
R=/home/michael/strata-xpu/strata
S=$R/d2b
D=$R/d2/plain
mkdir -p "$D"

cp -f "$S/ladder-fused.bin.lb23"    "$R/d2/runs/d2b-fused-4096/ladder.bin.lb23"
cp -f "$S/ladder-fused.bin.lb23.bo" "$R/d2/runs/d2b-fused-4096/ladder.bin.lb23.bo"
cp -f "$S/state-fused.bin"          "$R/d2/runs/d2b-fused-4096/window-state.bin"
cp -f "$S/ladder-plain.bin.lb23"    "$R/d2/runs/d2b-plain-4096/ladder.bin.lb23"
cp -f "$S/ladder-plain.bin.lb23.bo" "$R/d2/runs/d2b-plain-4096/ladder.bin.lb23.bo"
cp -f "$S/state-plain.bin"          "$R/d2/runs/d2b-plain-4096/window-state.bin"
cp -f "$S/lad1c-fused.bin.lb0"      "$R/d2/runs/d2p-1c-fused-4096/ladder.bin.lb0"
cp -f "$S/lad1c-fused.bin.lb0.bo"   "$R/d2/runs/d2p-1c-fused-4096/ladder.bin.lb0.bo"
cp -f "$S/state1c-fused.bin"        "$R/d2/runs/d2p-1c-fused-4096/window-state.bin"
cp -f "$S/lad1c-plain.bin.lb0"      "$R/d2/runs/d2p-1c-plain-4096/ladder.bin.lb0"
cp -f "$S/lad1c-plain.bin.lb0.bo"   "$R/d2/runs/d2p-1c-plain-4096/ladder.bin.lb0.bo"
cp -f "$S/state1c-plain.bin"        "$R/d2/runs/d2p-1c-plain-4096/window-state.bin"

mv -f "$S/d2b_ladder.py"   "$D/d2b_ladder.py"
mv -f "$S/m5g.txt"         "$D/m5g.txt"
mv -f "$S/ladder.txt"      "$D/ladder-2c.txt"
mv -f "$S/ladder-1c.txt"   "$D/ladder-1c.txt"
mv -f "$S/grparity-d2b.txt" "$D/grparity-1st.txt"
mv -f "$S/grparity-d2b.err" "$D/grparity-1st.err"
mv -f "$S/arm-fused.out"    "$D/arm-fused.out"
mv -f "$S/arm-plain.out"    "$D/arm-plain.out"
mv -f "$S/arm-1c-fused.out" "$D/arm-1c-fused.out"
mv -f "$S/arm-1c-plain.out" "$D/arm-1c-plain.out"

mv -f "$S/ladder-fused.bin.lb23"    "$D/raw-ladder-fused-2c.bin"
mv -f "$S/ladder-fused.bin.lb23.bo" "$D/raw-ladder-fused-2c.bin.bo"
mv -f "$S/state-fused.bin"          "$D/raw-state-fused-2c.bin"
mv -f "$S/ladder-plain.bin.lb23"    "$D/raw-ladder-plain-2c.bin"
mv -f "$S/ladder-plain.bin.lb23.bo" "$D/raw-ladder-plain-2c.bin.bo"
mv -f "$S/state-plain.bin"          "$D/raw-state-plain-2c.bin"
mv -f "$S/lad1c-fused.bin.lb0"      "$D/raw-ladder-fused-1c.bin"
mv -f "$S/lad1c-fused.bin.lb0.bo"   "$D/raw-ladder-fused-1c.bin.bo"
mv -f "$S/state1c-fused.bin"        "$D/raw-state-fused-1c.bin"
mv -f "$S/lad1c-plain.bin.lb0"      "$D/raw-ladder-plain-1c.bin"
mv -f "$S/lad1c-plain.bin.lb0.bo"   "$D/raw-ladder-plain-1c.bin.bo"
mv -f "$S/state1c-plain.bin"        "$D/raw-state-plain-1c.bin"

echo "== d2b/ (the other card's dir) =="
ls -la "$S"
echo "== d2/plain/ =="
ls -la "$D"
