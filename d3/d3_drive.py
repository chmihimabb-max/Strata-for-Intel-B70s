#!/usr/bin/env python3
"""D3 (card t_87aa2963): D2's driver, verbatim in shape.  Spawn a command, timestamp every stdout/stderr line,
feed stdin from a FIFO with O_RDWR (so the open never blocks and the rig decides WHEN to feed).

usage: d3_drive.py --fifo F --out O --err E --timeline T --pidfile P -- CMD [ARGS...]

timeline lines: <epoch>\t<monotonic>\t<OUT|ERR>\t<line>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time


def pump(stream, out_path, tag, timeline, tl_lock):
    with open(out_path, "wb", buffering=0) as f, open(timeline, "ab", buffering=0) as t:
        for line in stream:
            f.write(line)
            with tl_lock:
                t.write(("%.6f\t%.6f\t%s\t" % (time.time(), time.monotonic(), tag)).encode() + line)


def feeder(fifo, stdin, errlog):
    try:
        with os.fdopen(os.open(fifo, os.O_RDWR), "rb", buffering=0) as fr:
            while True:
                line = fr.readline()
                if not line:
                    continue
                stdin.write(line)
                stdin.flush()
                with open(errlog, "a") as e:
                    e.write("== d3_drive fed: %s" % line.decode(errors="replace")[:60])
    except Exception as ex:                                   # noqa: BLE001
        with open(errlog, "a") as e:
            e.write("d3_drive feeder ended: %r\n" % (ex,))


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--fifo", "--out", "--err", "--timeline", "--pidfile"):
        ap.add_argument(a, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    o = ap.parse_args()
    cmd = o.cmd[1:] if o.cmd and o.cmd[0] == "--" else o.cmd

    with open(o.timeline, "w") as t:
        t.write("# epoch\tmonotonic\tstream\tline\n")

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(o.pidfile, "w") as f:
        f.write("%d\n" % p.pid)
    lock = threading.Lock()
    ts = [threading.Thread(target=pump, args=(p.stdout, o.out, "OUT", o.timeline, lock), daemon=True),
          threading.Thread(target=pump, args=(p.stderr, o.err, "ERR", o.timeline, lock), daemon=True),
          threading.Thread(target=feeder, args=(o.fifo, p.stdin, o.err), daemon=True)]
    for t in ts:
        t.start()
    rc = p.wait()
    exit_epoch = time.time()
    for t in ts:
        t.join(timeout=5)
    print("engine exited %d at epoch %.6f" % (rc, exit_epoch))
    with open(o.timeline, "a") as t:
        t.write("%.6f\t%.6f\tEXIT\t%s\n" % (exit_epoch, time.monotonic(), rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
