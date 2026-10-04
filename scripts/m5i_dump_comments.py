import json
p = "/home/michael/.hermes/profiles/dev-gpu1/cache/spillover/call_00_OhNzTPrZ4IF83F4CriKV2720.txt"
d = json.load(open(p))
cs = d.get("comments") or []
for i, c in enumerate(cs):
    print("=" * 78)
    print("COMMENT index", i, "author", c.get("author") or c.get("created_by"), "at", c.get("created_at"))
    print(c.get("body"))
