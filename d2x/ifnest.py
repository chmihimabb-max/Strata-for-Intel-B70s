#!/usr/bin/env python3
"""Which `if()` blocks enclose a given line of CMakeLists.txt?  Depth-counts if/endif pairs."""
import re
import sys

path = "CMakeLists.txt"
target = int(sys.argv[1]) if len(sys.argv) > 1 else 617
depth = 0
stack = []
with open(path) as fh:
    for n, line in enumerate(fh, 1):
        code = line.split("#")[0].strip()
        if re.match(r"^if\(|^if ?\(", code):
            depth += 1
            stack.append((depth, n, code))
        elif re.match(r"^endif\(", code):
            if stack:
                stack.pop()
            depth -= 1
        if n == target:
            print("line %d: %s" % (n, line.rstrip()))
            print("enclosed by, outermost first:")
            for d, ln, code in stack:
                print("  depth %d  line %5d  %s" % (d, ln, code[:90]))
            break
