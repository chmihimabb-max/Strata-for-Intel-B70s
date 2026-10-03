#!/usr/bin/env python3
"""P10 sanity check for the per-thread CPU evidence tools: 4 busy threads, 12 s.

pidstat -t reported 0.00% for every thread of the engine during a request (measured, smoke-1c-4k), while
/proc/<pid>/task/<tid>/stat deltas showed ~36% on 19 threads. This settles which tool to trust on this box.
"""
import subprocess
import sys
import threading
import time

stop = False


def burn():
    x = 0
    while not stop:
        x += 1


ts = [threading.Thread(target=burn, daemon=True) for _ in range(4)]
for t in ts:
    t.start()

pid = __import__("os").getpid()
print("pid", pid, flush=True)
time.sleep(1)
p = subprocess.Popen(["/usr/bin/pidstat", "-t", "-p", str(pid), "1", "6"], stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True)
top = subprocess.Popen(["/usr/bin/top", "-b", "-H", "-n", "3", "-d", "1", "-p", str(pid)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out_p, _ = p.communicate()
out_t, _ = top.communicate()
stop = True
print("===== pidstat -t =====")
print(out_p[-2500:])
print("===== top -b -H =====")
print(out_t[-2500:])
sys.stdout.flush()
