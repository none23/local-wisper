#!/usr/bin/env python3
"""Backward-compatible entry point for Local Wisper."""

from __future__ import annotations

import sys

from local_wisper import cli as _cli


if __name__ == "__main__":
    raise SystemExit(_cli.main())

# Preserve the historical module API as well as the executable path. In
# particular, callers that patch or import helpers from ``wisper_cli`` should
# interact with the implementation module directly.
sys.modules[__name__] = _cli
