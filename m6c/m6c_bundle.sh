#!/usr/bin/env bash
# M6c: copy the evidence for card t_1037480d where the board can keep it, because the report lives
# outside the repository.  Only real files (find -type f): a cp of a FIFO would block forever.
R=/home/michael/strata-xpu
D=$HOME/.hermes/kanban/attachments/t_1037480d/m6c
rm -rf "$D"
mkdir -p "$D/runs" "$D/prompts"
cp "$R/m6c/TABLE.md" "$R/m6c/WRITEUP-M6C.md" "$R/m6c/notes-prompt-shape.txt" "$R/m6c/CHAIN.log" "$D/" 2>/dev/null
cp "$R/m6c/BLOCK1-SUMMARY.txt" "$R/m6c/BLOCK1B-SUMMARY.txt" "$R/m6c/BLOCK2-SUMMARY.txt" "$D/" 2>/dev/null
cp "$R/m6c/prompts/prompt-manifest.json" "$R/m6c/prompts/check.log" "$R/m6c/prompts/build.log" "$D/prompts/" 2>/dev/null
for d in $(find "$R/m6c/runs" -maxdepth 1 -mindepth 1 -type d); do
  t=$(basename "$d")
  mkdir -p "$D/runs/$t"
  for f in log.txt err.txt out.txt answer.txt rss.csv timeline.txt; do
    [ -f "$d/$f" ] && cp "$d/$f" "$D/runs/$t/$f"
  done
done
du -sh "$D"; find "$D" -type f | wc -l
