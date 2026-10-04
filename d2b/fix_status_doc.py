#!/usr/bin/env python3
"""One-off: replace the garbled paragraph in d2b/STATUS-D2B.md §3 with a clean statement."""
import io

P = "/home/michael/strata-xpu/strata/d2b/STATUS-D2B.md"
txt = io.open(P, encoding="utf-8").read()
start = txt.index("window size is chosen from measured round times")
end = txt.index("Timestamps of the cold run's programs")
old = txt[start:end]
new = ("window size is chosen from measured round times through `DraftPolicy`), which is what makes the\n"
       "pre-fix cost data-dependent: *which* (type, NCOLS) pairs a run needs is a function of its own window\n"
       "sequence.\n\n"
       "The cold run's residual (6908 ms) is the sum over all 93 first launches inside its decode phase; the 32\n"
       "dense mmvq builds are the named part of it, and the only part the warm-up removes. That part is worth\n"
       "127.93 - 36.27 = **91.66 ms per window x 54 windows = 4949 ms**, i.e. about 155 ms per dense build - the\n"
       "same order as D2a's kernel-level first-launch costs (63.8-519.7 ms, mean 169 ms for 20 specializations).\n\n")
assert "no -" not in old or True
io.open(P, "w", encoding="utf-8").write(txt[:start] + new + txt[end:])
print("replaced %d chars with %d" % (len(old), len(new)))
print(new)
