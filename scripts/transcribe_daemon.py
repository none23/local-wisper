#!/usr/bin/env python3
"""Backward-compatible launcher for the transcription daemon."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_wisper.daemon import main


if __name__ == "__main__":
    raise SystemExit(main())
