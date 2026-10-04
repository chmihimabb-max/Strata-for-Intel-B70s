#!/usr/bin/env python3
"""Why does the bench row not match?  Print the exact bytes of one data row."""
import re

for line in open("d2x/D2X-BENCH.txt"):
    if line.startswith("Q6_K 2560x10240 "):
        print(repr(line))
        break
