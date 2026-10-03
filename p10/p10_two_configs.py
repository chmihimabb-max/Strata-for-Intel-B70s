#!/usr/bin/env python3
"""P10: write the serve.server configs for the concurrency arms (one per card, or one over both cards).

usage: p10_two_configs.py <tag> <ctx> <pool_a> <pool_b> [affinity]

- arm A: "gpu": 0 -> the server exports ZE_AFFINITY_MASK=0 for its engine (serve/server.py:653-669)
- arm B: "gpu": 1 -> ZE_AFFINITY_MASK=1
- single: "gpu": [0, 1] -> the server appends --layer-split auto and leaves the mask unset (PLAN 11 U11)

Everything else is the config of record (strata-sycl-iq3s.json): same pack, same native shard, same PLE shard,
the faithful drafter, --kv int8 --expert-cache auto --mmap-experts --spec 4, --max-context <ctx>.
"""
from __future__ import annotations

import json
import pathlib
import sys

SRC = pathlib.Path("/home/michael/strata-xpu/strata")
R = pathlib.Path("/home/michael/strata-xpu")


def make(base: dict, tag: str, ctx: int, gpu, port: int, log: str, pool: str, affinity: str,
         pc0: bool = False) -> dict:
    cfg = dict(base)
    args = list(base["args"])
    # context length of this arm
    if "--max-context" in args:
        args[args.index("--max-context") + 1] = str(ctx)
    # the pool levers (operator note): worker count and affinity
    if pool:
        if "--pool-workers" in args:
            args[args.index("--pool-workers") + 1] = pool
        else:
            args += ["--pool-workers", pool]
    if affinity:
        if "--pool-affinity" in args:
            args[args.index("--pool-affinity") + 1] = affinity
        else:
            args += ["--pool-affinity", affinity]
    if pc0:
        # no conversation checkpoints.  Not an optimisation: with the checkpoint path ON the two-card split
        # FAILS mid-request on this pair (a cross-card running-state copy, measured in the first 1x2 arm -
        # "memcpy failed: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY" then "saving a checkpoint part failed"), and
        # every like-for-like arm of this card (the direct-serve rigs) passes --prompt-cache 0 as well.
        for k, v in (("--prompt-cache", "0"), ("--prompt-cache-every", "0")):
            if k in args:
                args[args.index(k) + 1] = v
            else:
                args += [k, v]
    cfg["args"] = args
    cfg["gpu"] = gpu
    cfg["log"] = log
    cfg["port"] = port
    cfg["model_name"] = f"qwen3.8-flash-next-iq3s-{tag}"
    return cfg


def main() -> int:
    tag, ctx, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    affinity = sys.argv[5] if len(sys.argv) > 5 else ""
    pc0 = len(sys.argv) > 6 and sys.argv[6] == "pc0"
    base = json.loads((SRC / "strata-sycl-iq3s.json").read_text())
    d = R / "p10" / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    specs = [("a", 0, 8101, pa), ("b", 1, 8102, pb)]
    for name, gpu, port, pool in specs:
        log = str(R / "logs" / f"p10-{tag}-{name}-engine.log")
        cfg = make(base, tag, int(ctx), gpu, port, log, pool, affinity, pc0)
        p = d / f"cfg-{name}.json"
        p.write_text(json.dumps(cfg, indent=1))
        print(f"[cfg] {p}  gpu={gpu} port={port} pool={pool or 'default'} affinity={affinity or 'default'} "
              f"pc0={pc0}")
        print(f"[cfg]   log {log}")
    # the one-instance-over-both-cards control, in the same directory
    log = str(R / "logs" / f"p10-{tag}-1x2-engine.log")
    cfg = make(base, tag + "-1x2", int(ctx), [0, 1], 8103, log, pa, affinity, pc0)
    cfg["layer_split"] = "auto"
    p = d / "cfg-1x2.json"
    p.write_text(json.dumps(cfg, indent=1))
    print(f"[cfg] {p}  gpu=[0,1] port=8103 pool={pa or 'default'} (the ZE_AFFINITY_MASK must stay unset)")
    print(f"[cfg]   log {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
