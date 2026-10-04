#!/usr/bin/env python3
"""S5: print the config of record's engine args, one per line, so the arm scripts use them verbatim.

usage: s5_argv.py [--config FILE] [EXTRA ARG...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

SRC = pathlib.Path("/home/michael/strata-xpu/strata")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(SRC / "strata-sycl-iq3s.json"))
    ap.add_argument("extra", nargs="*")
    o = ap.parse_args()
    cfg = json.loads(pathlib.Path(o.config).read_text(encoding="utf-8-sig"))
    for a in list(cfg["args"]) + list(o.extra):
        print(str(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
