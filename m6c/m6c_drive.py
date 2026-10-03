#!/usr/bin/env python3
"""M6c: run the Strata engine under a timestamped tee.

The engine's serve protocol writes `T <id>` per generated token on stdout and its own phase lines on
stderr, but neither carries a time.  This driver spawns the engine, feeds it the GEN/QUIT payload, and
stamps every stdout/stderr line with CLOCK_REALTIME + monotonic so TTFT and the decode rate can be timed
from the engine's own `RESUME 0` marker instead of being inferred from the DONE summary.

usage: m6c_drive.py --stdin F --out F --err F --timeline F --pidfile F -- CMD [ARGS...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time


def pump(stream, out_path: str, tag: str, timeline: str) -> None:
    # the pipes are binary: write the raw bytes through, and only the timestamp prefix is text
    with open(out_path, "wb", buffering=0) as f, open(timeline, "ab", buffering=0) as t:
        for line in stream:
            f.write(line)
            t.write(("%.6f\t%.6f\t%s\t" % (time.time(), time.monotonic(), tag)).encode() + line)


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--stdin", "--out", "--err", "--timeline", "--pidfile"):
        ap.add_argument(a, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    o = ap.parse_args()
    cmd = o.cmd[1:] if o.cmd and o.cmd[0] == "--" else o.cmd

    with open(o.timeline, "w") as t:
        t.write("# epoch\tmonotonic\tstream\tline\n")
    payload = open(o.stdin, "rb").read()

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(o.pidfile, "w") as f:
        f.write("%d\n" % p.pid)
    ts = [threading.Thread(target=pump, args=(p.stdout, o.out, "OUT", o.timeline)),
          threading.Thread(target=pump, args=(p.stderr, o.err, "ERR", o.timeline))]
    for t in ts:
        t.start()
    # the payload is small (a few MB); the engine reads it when it is ready
    p.stdin.write(payload)
    p.stdin.close()
    rc = p.wait()
    for t in ts:
        t.join()
    print("engine exited %d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
