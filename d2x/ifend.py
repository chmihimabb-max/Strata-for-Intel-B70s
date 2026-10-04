#!/usr/bin/env python3
"""Print each if/endif at depth 1/2 with its line, to locate the end of the if(NOT STRATA_ENABLE_SYCL) block."""
import re

depth = 0
with open("CMakeLists.txt") as fh:
    for n, line in enumerate(fh, 1):
        code = line.split("#")[0].strip()
        if re.match(r"^if\(", code):
            depth += 1
            if depth <= 2 and n > 200:
                print("%6d  depth->%d  %s" % (n, depth, code[:80]))
        elif re.match(r"^endif\(", code):
            if depth <= 2 and n > 380:
                print("%6d  depth<=%d  endif" % (n, depth))
            depth -= 1
