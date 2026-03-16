#!/usr/bin/env python3
"""Persistent socket-based transcription daemon for lw.nvim."""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wisper_cli import AppError, load_model, transcribe_with_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent transcription daemon.")
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--socket", required=True)
    return parser


def emit(conn_file, payload: dict) -> None:
    conn_file.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
    conn_file.flush()


def handle_request(req: dict, model, vad_filter: bool) -> dict:
    msg_type = req.get("type")
    req_id = req.get("id")

    if msg_type == "ping":
        return {"type": "ready", "id": req_id}

    if msg_type != "transcribe":
        return {"type": "error", "id": req_id, "error": "unsupported request"}

    try:
        audio_path = Path(req["audio_path"])
        if not audio_path.exists() or audio_path.stat().st_size < 2048:
            return {"type": "no_speech", "id": req_id}

        text = transcribe_with_model(
            audio_path,
            model,
            verbose=False,
            show_banner=False,
            vad_filter=vad_filter,
        )
        if text:
            return {"type": "result", "id": req_id, "text": text}
        return {"type": "no_speech", "id": req_id}
    except AppError as exc:
        return {"type": "error", "id": req_id, "error": str(exc)}
    except Exception as exc:
        return {"type": "error", "id": req_id, "error": f"{exc.__class__.__name__}: {exc}"}


def socket_is_live(socket_path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            sock.connect(str(socket_path))
        return True
    except OSError:
        return False


class DaemonServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, model, vad_filter):
        self.model = model
        self.vad_filter = vad_filter
        super().__init__(server_address, handler_class)


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw_line = self.rfile.readline()
            if not raw_line:
                return

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            req_id = None
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
                req_id = req.get("id")
                payload = handle_request(req, self.server.model, self.server.vad_filter)
            except Exception as exc:
                payload = {"type": "error", "id": req_id, "error": f"{exc.__class__.__name__}: {exc}"}

            emit(self.wfile, payload)


def main() -> int:
    args = build_parser().parse_args()
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    if socket_path.exists():
        if socket_is_live(socket_path):
            return 0
        socket_path.unlink(missing_ok=True)

    try:
        model = load_model(args.model, args.compute_type, args.device, verbose=False)
    except AppError:
        return 1
    except Exception:
        return 1

    server = DaemonServer(str(socket_path), RequestHandler, model, args.vad_filter)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        socket_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
