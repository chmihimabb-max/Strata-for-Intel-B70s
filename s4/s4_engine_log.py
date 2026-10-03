#!/usr/bin/env python3
"""S4: print the engine log path a serve.server config names (one line, nothing else)."""
from __future__ import annotations

import json
import sys

print(json.load(open(sys.argv[1]))["log"])
