#!/usr/bin/env python3
"""Transcribe an existing WAV file using wisper_cli model helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wisper_cli import AppError, load_model, transcribe_with_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe a WAV file and print text.")
    parser.add_argument("audio_path", type=Path, help="Path to WAV audio file.")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--compute-type", default="int8")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.audio_path.exists() or args.audio_path.stat().st_size < 2048:
        print("Audio file missing or too short.", file=sys.stderr)
        return 1

    try:
        model = load_model(args.model, args.compute_type, verbose=False)
        text = transcribe_with_model(args.audio_path, model, verbose=False, show_banner=False)
    except AppError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if text:
        print(text)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
