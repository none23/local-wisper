#!/usr/bin/env python3
"""Record microphone audio locally and transcribe it with a local Whisper model."""

from __future__ import annotations

import argparse
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import termios
import tty
from pathlib import Path


class AppError(Exception):
    """Raised for user-facing runtime errors."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record from microphone and print a local Whisper transcription."
    )
    parser.add_argument(
        "--model",
        default="small.en",
        help="Whisper model name/path for faster-whisper (default: small.en).",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (default: int8).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Capture sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep recorded WAV file instead of deleting it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timing/debug logs to stderr.",
    )
    return parser


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _ensure_any_command(commands: list[str]) -> None:
    if any(shutil.which(cmd) for cmd in commands):
        return
    joined = ", ".join(commands)
    raise AppError(
        f"Missing audio capture tool. Install one of: {joined}. "
        "On Manjaro, install PipeWire tools or ffmpeg."
    )


def _start_pw_record(output_path: Path, sample_rate: int) -> subprocess.Popen[str]:
    cmd = [
        "pw-record",
        "--rate",
        str(sample_rate),
        "--channels",
        "1",
        "--format",
        "s16",
        str(output_path),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_ffmpeg_pulse(output_path: Path, sample_rate: int) -> subprocess.Popen[str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "pulse",
        "-i",
        "default",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-y",
        str(output_path),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def start_recording(output_path: Path, sample_rate: int, verbose: bool) -> tuple[subprocess.Popen[str], str]:
    _ensure_any_command(["pw-record", "ffmpeg"])
    attempts: list[tuple[str, subprocess.Popen[str]]] = []

    if shutil.which("pw-record"):
        proc = _start_pw_record(output_path, sample_rate)
        time.sleep(0.25)
        if proc.poll() is None:
            _log(verbose, "Using capture backend: pw-record\n")
            return proc, "pw-record"
        attempts.append(("pw-record", proc))

    if shutil.which("ffmpeg"):
        proc = _start_ffmpeg_pulse(output_path, sample_rate)
        time.sleep(0.4)
        if proc.poll() is None:
            _log(verbose, "Using capture backend: ffmpeg pulse\n")
            return proc, "ffmpeg"
        attempts.append(("ffmpeg", proc))

    errors = []
    for backend, proc in attempts:
        stderr = proc.stderr.read().strip() if proc.stderr else ""
        proc.stderr.close() if proc.stderr else None
        errors.append(f"{backend}: {stderr or 'failed to start'}")
    raise AppError("Could not start audio capture.\n" + "\n".join(errors))


def stop_recording(proc: subprocess.Popen[str], backend: str, verbose: bool) -> None:
    if proc.poll() is not None:
        return

    if backend == "ffmpeg":
        # ffmpeg finalizes WAV on SIGINT.
        proc.send_signal(signal.SIGINT)
    else:
        proc.terminate()

    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)

    if proc.stderr:
        stderr = proc.stderr.read().strip()
        proc.stderr.close()
        if stderr and verbose:
            print(f"Capture stderr: {stderr}", file=sys.stderr)


def _require_faster_whisper() -> None:
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        raise AppError(
            "Missing dependency 'faster-whisper'. Install dependencies with "
            "`python -m pip install -r requirements.txt`. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc


def transcribe_file(audio_path: Path, model_name: str, compute_type: str, verbose: bool) -> str:
    _require_faster_whisper()
    from faster_whisper import WhisperModel

    print(
        "Loading local Whisper model (first run may download weights)...",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    _log(verbose, f"Model load/init took {time.perf_counter() - t0:.2f}s")

    print("Transcribing audio...", file=sys.stderr)
    t1 = time.perf_counter()
    segments, _info = model.transcribe(str(audio_path), vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    _log(verbose, f"Transcription took {time.perf_counter() - t1:.2f}s")
    return text


def _create_audio_path() -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    tmpdir = tempfile.TemporaryDirectory(prefix="wisper_")
    path = Path(tmpdir.name) / "recording.wav"
    return path, tmpdir


def wait_for_stop_key() -> None:
    """Wait for Enter key using /dev/tty so terminal wrappers don't break stdin handling."""
    try:
        with open("/dev/tty", "rb", buffering=0) as tty_file:
            fd = tty_file.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    ready, _, _ = select.select([fd], [], [])
                    if not ready:
                        continue
                    ch = os.read(fd, 1)
                    if ch in (b"\n", b"\r"):
                        print("\nStop key received. Stopping recording...\n", file=sys.stderr)
                        return
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        # Fallback for environments where /dev/tty is unavailable.
        input()


def main() -> int:
    args = build_parser().parse_args()

    audio_path, tmpdir = _create_audio_path()
    print("Recording... Press Enter to stop.\n", file=sys.stderr)

    proc: subprocess.Popen[str] | None = None
    backend = ""
    try:
        proc, backend = start_recording(audio_path, args.sample_rate, args.verbose)
        wait_for_stop_key()
    except KeyboardInterrupt:
        print("\nStop requested (Ctrl+C). Finalizing...", file=sys.stderr)
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        tmpdir.cleanup()
        return 1
    finally:
        if proc is not None:
            stop_recording(proc, backend, args.verbose)
            print("Recording stopped.\n", file=sys.stderr)

    try:
        if not audio_path.exists() or audio_path.stat().st_size < 2048:
            print(
                "Error: Recording is empty or too short to transcribe.",
                file=sys.stderr,
            )
            return 2

        text = transcribe_file(audio_path, args.model, args.compute_type, args.verbose)
        if text:
            print(text)
        else:
            print("No speech detected.")
        return 0
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep_audio:
            keep_path = Path.cwd() / f"wisper_recording_{int(time.time())}.wav"
            audio_path.replace(keep_path)
            print(f"Saved audio to {keep_path}", file=sys.stderr)
            tmpdir.cleanup()
        else:
            tmpdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
