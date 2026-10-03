#!/usr/bin/env python3
"""M6c: put the 256K block into the file the card names (`~/strata-xpu/WRITEUP.md`), keeping M6's part
intact: a short M6c pointer after the opening of the document, and the whole Part II at the end.

    /usr/bin/python3 m6c/m6c_assemble_writeup.py            # dry run (prints the plan)
    /usr/bin/python3 m6c/m6c_assemble_writeup.py --write
"""
from __future__ import annotations

import pathlib
import sys

W = pathlib.Path("/home/michael/strata-xpu/WRITEUP.md")
PART = pathlib.Path("/home/michael/strata-xpu/m6c/WRITEUP-M6C.md")
MARK = "## M6c: the 256K block (Part II of this file)"
ANCHOR = "## 1. What was run, and the device check"
PART_HEAD = "# Part II -- M6c: IQ3_S at 256K context with KV streaming on two Arc Pro B70"

POINTER = f"""{MARK}

**Everything below §"Part II" is the M6c block (card `t_1037480d`): IQ3_S served at the model's full
**262,144-token window** on both B70s with **KV streaming** on, measured 2026-10-02 at the same box, same
pack and same protocol as the 16K/32K block in Part I above.** In one paragraph: a **259,943-token prompt**
was read at **316.9 tok/s** and **256 usage-counted tokens** were generated at **18.45 tok/s** (TTFT
820.5 s); the curve at `--prefill auto` is **344.8 / 339.6 / 342.6 / 316.9 tok/s** prefill and
**21.71 / 18.67 / 19.55 / 18.45 tok/s** decode at 32K / 64K / 128K / 262K; streaming was verified engaged
(32,768 of 262,144 cells per QSA layer resident, 90.99% of 790,765 block reads served from VRAM, 287.1 MiB
from RAM, 12,672 B of RAM per context token against the docs' 13.7 KB) and on this box it **does not buy
expert residency** (24,576 slots against 24,302) while costing 18.9% of the prompt read; the needle is
answered exactly at 4K and at 5% depth of 256K and is **not** retrieved at 50% depth of 256K, which is a
position effect and is reported as such. Part I's numbers are a **different pack** (W4A16 Q4_0) and are not
a baseline for these rows.

"""


def main() -> int:
    write = "--write" in sys.argv
    doc = W.read_text(encoding="utf-8")
    part = PART.read_text(encoding="utf-8")
    # idempotent: strip a previous Part II and a previous pointer block, then re-insert both
    if PART_HEAD in doc:
        doc = doc[:doc.index(PART_HEAD)].rstrip() + "\n"
    if MARK in doc and ANCHOR in doc:
        i, j = doc.index(MARK), doc.index(ANCHOR)
        doc = doc[:i] + doc[j:]
    if ANCHOR not in doc:
        print("FATAL: anchor not found:", ANCHOR)
        return 2
    before = len(doc)
    doc = doc.replace(ANCHOR, POINTER + ANCHOR, 1)
    doc = doc.rstrip() + "\n\n" + part.rstrip() + "\n"
    print("base %d chars -> %d chars; pointer inserted before %r; Part II appended (%d chars)"
          % (before, len(doc), ANCHOR, len(part)))
    if write:
        W.write_text(doc, encoding="utf-8")
        print("wrote", W)
    else:
        print("dry run (pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
