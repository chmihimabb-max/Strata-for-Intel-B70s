#!/usr/bin/env python3
"""S4 (card t_30d9ccfb): write one arm's serve.server config from the config of record.

usage: s4_cfg.py <tag> [--extra "ARG ARG ..."] [--no-extra] [--ctx N]

The config of record is strata-sycl-iq3s.json, taken VERBATIM except for three fields that only name this arm's
own outputs (`log`, `port`, `model_name` so arms do not collide).  Nothing in `args` is rewritten unless the
caller asks for it with --extra: the whole point of this card is that the *unmodified* config of record is what
fails, so an arm's config must be diffable against it.

Prints the diff against the config of record's own args, the server's effective engine argv (imported from
serve.server, so it is what the server really runs - including the `--layer-split auto` it appends for a
multi-GPU config), and the argv's SHA256 so an arm's log names the exact invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

SRC = pathlib.Path("/home/michael/strata-xpu/strata")
R = pathlib.Path("/home/michael/strata-xpu")
RECORD = SRC / "strata-sycl-iq3s.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--extra", default="")
    ap.add_argument("--ctx", type=int, default=0)
    o = ap.parse_args()

    base = json.loads(RECORD.read_text())
    cfg = dict(base)
    args = list(base["args"])
    if o.ctx:
        args[args.index("--max-context") + 1] = str(o.ctx)
    extra = o.extra.split() if o.extra else []
    args += extra
    cfg["args"] = args
    d = SRC / "s4" / "runs" / o.tag
    d.mkdir(parents=True, exist_ok=True)
    cfg["log"] = str(R / "logs" / f"s4-{o.tag}-engine.log")
    cfg["port"] = int(base.get("port", 8099))
    cfg["model_name"] = f"qwen3.8-flash-next-iq3s-s4-{o.tag}"
    (d / "cfg.json").write_text(json.dumps(cfg, indent=1) + "\n")

    sys.path.insert(0, str(SRC))
    from serve.server import engine_args  # noqa: PLC0415
    argv = engine_args(cfg)

    print(f"[cfg] {d / 'cfg.json'}")
    print(f"[cfg] config of record args : {' '.join(base['args'])}")
    print(f"[cfg] this arm's args       : {' '.join(args)}")
    print(f"[cfg] --extra added         : {' '.join(extra) if extra else '<none> (verbatim config of record)'}")
    print(f"[cfg] server gpu field      : {cfg.get('gpu')} -> ZE_AFFINITY_MASK={','.join(str(g) for g in cfg['gpu'])}")
    print(f"[cfg] engine argv           : {' '.join(argv)}")
    print(f"[cfg] engine argv sha256    : {hashlib.sha256(' '.join(argv).encode()).hexdigest()}")
    print(f"[cfg] engine log            : {cfg['log']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
