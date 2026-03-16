#!/usr/bin/env python3
import sys

print("boom", file=sys.stderr, flush=True)
raise SystemExit(1)
