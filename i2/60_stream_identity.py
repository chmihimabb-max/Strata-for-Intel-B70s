#!/usr/bin/env python3
"""I2: are the engine's own runs bit-identical? (spec 2 vs spec 4, cold vs warm page cache, int8 vs fp16 KV)

    python3 i2/60_stream_identity.py <out.txt> [more out.txt ...]

Prints each file's token count and, for every pair, whether the streams are identical.
"""
import pathlib
import sys


def parse(path):
    cur, streams = [], []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith("T "):
            cur.append(int(line[2:]))
        elif line.startswith("DONE"):
            streams.append(cur)
            cur = []
    return streams


files = sys.argv[1:]
data = {f: parse(f) for f in files}
for f in files:
    per = ["%d" % len(s) for s in data[f]]
    print("%-34s %d requests, tokens per request: %s" % (pathlib.Path(f).name, len(data[f]), " ".join(per)))
print()
for i, a in enumerate(files):
    for b in files[i + 1:]:
        same = data[a] == data[b]
        print("%-32s vs %-32s : %s" % (pathlib.Path(a).name, pathlib.Path(b).name,
                                       "IDENTICAL token ids" if same else "different"))
        if not same:
            for k, (sa, sb) in enumerate(zip(data[a], data[b])):
                if sa != sb:
                    d = next((j for j in range(min(len(sa), len(sb))) if sa[j] != sb[j]), min(len(sa), len(sb)))
                    print("     request %d: %d/%d tokens, first difference at index %d (%s vs %s)"
                          % (k, len(sa), len(sb), d, sa[d] if d < len(sa) else None, sb[d] if d < len(sb) else None))
