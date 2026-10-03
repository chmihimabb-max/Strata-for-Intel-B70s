#!/usr/bin/env bash
# P1b: the tracer's chrome timelines live in the app's cwd; keep the two CLOSED ones as evidence, inventory the rest,
# and drop the unclosed dumps (they are 600 MB each and carry no closing bracket).
R=/home/michael/strata-xpu
SRC=$R/strata
INV=$R/p1/logs/p1b-trace-files.txt
{
  echo "unitrace chrome timelines written into the app's cwd ($SRC) during P1b - $(date -Is)"
  echo "name  bytes  closed?(last 8 bytes)  mtime"
  for f in "$SRC"/strata.*.json; do
    [ -f "$f" ] || continue
    printf '%s  %s  [%s]  %s\n' "$(basename "$f")" "$(stat -c '%s' "$f")" "$(tail -c 8 "$f" | tr -d '\n')" "$(stat -c '%y' "$f")"
  done
} > "$INV"
cat "$INV"
mkdir -p "$R/p1/traces/p1b-x-ut9" "$R/p1/traces/p1b-z-ut1"
mv "$SRC/strata.862316.json" "$R/p1/traces/p1b-x-ut9/chrome-timeline-strata.862316.json"
mv "$SRC/strata.866275.json" "$R/p1/traces/p1b-z-ut1/chrome-timeline-strata.866275.json"
rm -f "$SRC"/strata.816501.json "$SRC"/strata.822900.json "$SRC"/strata.839564.json \
      "$SRC"/strata.848240.json "$SRC"/strata.851199.json "$SRC"/strata.855615.json
echo "---- kept:"
ls -la "$R/p1/traces/p1b-x-ut9" "$R/p1/traces/p1b-z-ut1"
echo "---- repo dir now:"
ls "$SRC"/strata.*.json 2>/dev/null || echo "   (no strata.*.json left in the repo working dir)"
