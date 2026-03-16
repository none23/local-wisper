#!/usr/bin/env python3
"""Persistent transcription worker for lw.nvim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wisper_cli import AppError, load_model, transcribe_with_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent WAV transcription worker.")
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = build_parser().parse_args()

    try:
        model = load_model(args.model, args.compute_type, args.device, verbose=False)
    except AppError as exc:
        emit({"type": "fatal", "error": str(exc)})
        return 1
    except Exception as exc:
        emit({"type": "fatal", "error": f"{exc.__class__.__name__}: {exc}"})
        return 1

    emit({"type": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        req_id = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            audio_path = Path(request["audio_path"])

            if not audio_path.exists() or audio_path.stat().st_size < 2048:
                emit({"type": "no_speech", "id": req_id})
                continue

            text = transcribe_with_model(
                audio_path,
                model,
                verbose=False,
                show_banner=False,
                vad_filter=args.vad_filter,
            )
            if text:
                emit({"type": "result", "id": req_id, "text": text})
            else:
                emit({"type": "no_speech", "id": req_id})
        except AppError as exc:
            emit({"type": "error", "id": req_id, "error": str(exc)})
        except Exception as exc:
            emit({"type": "error", "id": req_id, "error": f"{exc.__class__.__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
