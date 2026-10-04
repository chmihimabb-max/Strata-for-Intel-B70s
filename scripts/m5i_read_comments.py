import json, sys
p = "/home/michael/.hermes/profiles/dev-gpu1/cache/spillover/call_00_OhNzTPrZ4IF83F4CriKV2720.txt"
d = json.load(open(p))
print("top keys:", list(d.keys()))
cs = d.get("comments") or []
print("n comments:", len(cs))
for c in cs:
    body = c.get("body", "") or ""
    print(c.get("id"), "|", c.get("author") or c.get("created_by"), "|", c.get("created_at"), "|", len(body))
want = [str(x) for x in sys.argv[1:]] or ["69", "70", "71"]
for c in cs:
    if str(c.get("id")) in want:
        print("=" * 70)
        print("COMMENT", c.get("id"))
        print(c.get("body"))
