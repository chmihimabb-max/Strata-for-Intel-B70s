#!/usr/bin/env bash
# P6 probe (card t_5dfc11a3): what the server prints when it is offered to the LAN.
#
# --lazy so no engine process is spawned: this probe touches NO GPU and can run beside the resident server
# on 8099.  --host 0.0.0.0 is what makes serve/server.py:2589-2598 print the "from other devices:" line,
# and AGENTS.md requires an API key whenever the server leaves 127.0.0.1 (this key is a throwaway).
exec /usr/bin/python3 -m serve.server --engine strata \
  --config strata-sycl-iq3s.json --port 8098 --api-monitor --lazy \
  --host 0.0.0.0 --api-key "probe-only-not-a-real-key"
