#!/usr/bin/env python3
"""Slow test daemon that delays socket creation before replying once."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model")
    parser.add_argument("--compute-type")
    parser.add_argument("--device")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--no-vad-filter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)

    time.sleep(8.0)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen()
        while True:
            conn, _ = server.accept()
            with conn:
                raw_line = b""
                while not raw_line.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    raw_line += chunk
                if not raw_line.strip():
                    continue

                req = json.loads(raw_line.decode("utf-8"))
                if req.get("type") == "ping":
                    payload = {"type": "ready", "id": req.get("id")}
                else:
                    payload = {"type": "result", "id": req.get("id"), "text": "slow daemon ok"}
                conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
