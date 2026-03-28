#!/usr/bin/env python3
import sys

# Accept the real daemon CLI contract even though this fixture always fails.
_ = sys.argv

print("boom", file=sys.stderr, flush=True)
raise SystemExit(1)
