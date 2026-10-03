#!/usr/bin/env python3
"""P10: one OpenAI-compatible request, timed, for the concurrency arms.

usage: p10_client.py --port 8101 --prompt-file F --max-new 256 --out resp.json [--tag a] [--label solo]

Writes: <out> (the response body or the error), <out>.timing.json with the wall clock, the reported usage and
the first-token time when the server streams.  Non-streaming: the timings come from the server's own
prompt/decode report and from the engine's log, not from this client.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default="strata")
    o = ap.parse_args()

    text = open(o.prompt_file, encoding="utf-8").read()
    body = json.dumps({
        "model": o.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": o.max_new,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{o.port}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    err = None
    resp = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as ex:
        err = "HTTP %s: %s" % (ex.code, ex.read()[:500])
    except Exception as ex:                                   # noqa: BLE001
        err = repr(ex)
    t1 = time.time()
    timing = {"tag": o.tag, "label": o.label, "port": o.port, "wall_s": round(t1 - t0, 3),
              "started": t0, "ended": t1, "error": err}
    if resp is not None:
        timing["usage"] = resp.get("usage")
        timing["finish_reason"] = (resp.get("choices") or [{}])[0].get("finish_reason")
        timing["timings"] = resp.get("timings")
        open(o.out, "w").write(json.dumps(resp))
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        timing["answer_chars"] = len(content)
        timing["answer_head"] = content[:200]
    else:
        open(o.out + ".error", "w").write(str(err))
    open(o.out + ".timing.json", "w").write(json.dumps(timing, indent=1))
    print(json.dumps(timing))
    return 0 if resp is not None else 1


if __name__ == "__main__":
    sys.exit(main())
