#!/bin/bash
# I3a: the real assembly write, into the external SSD that already holds the pack.
# / has 171 GB free; the file needs 103 GiB = 110.6 GB, so the SSD (1046 GiB free) is the target.
# Nothing is deleted; every source is opened read-only.
cd /home/michael/strata-xpu/strata || exit 3
OUT=/run/media/michael/2208B12208B0F63F/strata-w4a16/i3-oracle/w4a16-q4_0-dense-ours-i3-oracle.gguf
echo "=== df before"
df -h / /run/media/michael/2208B12208B0F63F | tee i3/00_df_before.txt
echo "=== date"; date -Is
echo "=== plan"
/usr/bin/python3 tools/w4a16_gguf_assemble.py --plan 2>&1 | tee i3/09_plan.log
echo "plan rc=${PIPESTATUS[0]}"
echo "=== write start"
/usr/bin/python3 tools/w4a16_gguf_assemble.py --write --out "$OUT" > i3/10_write.log 2>&1
echo "write rc=$?"
ls -la "$OUT" 2>&1 | tee i3/10_write_ls.txt
echo "=== df after"
df -h / /run/media/michael/2208B12208B0F63F | tee i3/00_df_after.txt
echo "=== date"; date -Is
echo "=== DONE"
