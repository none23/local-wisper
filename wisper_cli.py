#!/usr/bin/env python3
"""Record microphone audio locally and transcribe it with a local Whisper model."""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import select
import site
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import termios
import tty
from pathlib import Path


class AppError(Exception):
    """Raised for user-facing runtime errors."""


_CUDA_RUNTIME_READY = False


def _candidate_cuda_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_dirs.append(user_site)
    except Exception:
        pass

    for root in site_dirs:
        for rel in ("nvidia/cublas/lib", "nvidia/cudnn/lib"):
            path = Path(root) / rel
            key = str(path)
            if key not in seen and path.is_dir():
                seen.add(key)
                dirs.append(path)

    for path in (Path("/opt/cuda/lib64"), Path("/opt/cuda/targets/x86_64-linux/lib"), Path("/usr/local/cuda/lib64")):
        key = str(path)
        if key not in seen and path.is_dir():
            seen.add(key)
            dirs.append(path)

    return dirs


def _prepend_ld_library_path(paths: list[Path]) -> None:
    if not paths:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    for path in reversed([str(p) for p in paths]):
        if path not in parts:
            parts.insert(0, path)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _first_matching_lib(lib_dirs: list[Path], patterns: tuple[str, ...]) -> Path | None:
    for lib_dir in lib_dirs:
        for pattern in patterns:
            matches = sorted(glob.glob(str(lib_dir / pattern)))
            if matches:
                return Path(matches[0])
    return None


def _prepare_cuda_runtime(verbose: bool) -> None:
    global _CUDA_RUNTIME_READY
    if _CUDA_RUNTIME_READY:
        return

    lib_dirs = _candidate_cuda_lib_dirs()
    _prepend_ld_library_path(lib_dirs)

    libs_to_preload = [
        ("libcublas.so.12", ("libcublas.so.12", "libcublas.so.*")),
        ("libcublasLt.so.12", ("libcublasLt.so.12", "libcublasLt.so.*")),
        ("libcudnn.so.9", ("libcudnn.so.9", "libcudnn.so.*")),
    ]
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)

    for display_name, patterns in libs_to_preload:
        lib_path = _first_matching_lib(lib_dirs, patterns)
        if lib_path is None:
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=rtld_global)
            _log(verbose, f"Preloaded CUDA runtime: {display_name} from {lib_path}")
        except OSError as exc:
            raise AppError(f"Failed to load CUDA runtime library {display_name}: {exc}") from exc

    _CUDA_RUNTIME_READY = True


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
        "--device",
        default="cpu",
        help="faster-whisper device (default: cpu).",
    )
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable voice activity detection filtering (default: true).",
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
    parser.add_argument(
        "--live",
        action="store_true",
        help="Stream partial transcription while recording.",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=1.0,
        help="Seconds between live transcription refreshes (default: 1.0).",
    )
    return parser


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _status(text: str) -> None:
    print(f"\r\033[2K{text}", end="", flush=True)


def _status_done() -> None:
    print()


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


def load_model(model_name: str, compute_type: str, device: str, verbose: bool):
    if device.startswith("cuda"):
        _prepare_cuda_runtime(verbose)
    _require_faster_whisper()
    from faster_whisper import WhisperModel

    if verbose:
        print(
            "Loading local Whisper model (first run may download weights)...",
            file=sys.stderr,
        )
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _log(verbose, f"Model load/init took {time.perf_counter() - t0:.2f}s")
    return model


def transcribe_with_model(
    audio_path: Path, model, verbose: bool, show_banner: bool, vad_filter: bool
) -> str:
    if show_banner and verbose:
        print("Transcribing audio...", file=sys.stderr)
    t1 = time.perf_counter()
    segments, _info = model.transcribe(str(audio_path), vad_filter=vad_filter)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    _log(verbose, f"Transcription took {time.perf_counter() - t1:.2f}s")
    return text


def transcribe_file(
    audio_path: Path,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
    verbose: bool,
) -> str:
    model = load_model(model_name, compute_type, device, verbose)
    return transcribe_with_model(
        audio_path, model, verbose, show_banner=True, vad_filter=vad_filter
    )


def _create_audio_path() -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    tmpdir = tempfile.TemporaryDirectory(prefix="wisper_")
    path = Path(tmpdir.name) / "recording.wav"
    return path, tmpdir


def wait_for_enter() -> None:
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
                        return
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        # Fallback for environments where /dev/tty is unavailable.
        input()


def _wait_for_stop_key_event(stop_event: threading.Event) -> None:
    wait_for_enter()
    stop_event.set()


def _text_delta(previous: str, current: str) -> str:
    prev = previous.strip()
    cur = current.strip()
    if not prev:
        return cur
    if cur.startswith(prev):
        return cur[len(prev) :].lstrip()
    return cur


def copy_to_clipboard(text: str) -> bool:
    if not text:
        return False

    commands = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ]
    for cmd in commands:
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0:
                return True
        except Exception:
            continue
    return False


def main() -> int:
    args = build_parser().parse_args()
    if args.live_interval <= 0:
        print("Error: --live-interval must be > 0.", file=sys.stderr)
        return 1

    audio_path, tmpdir = _create_audio_path()
    model = None
    try:
        while True:
            if args.live:
                print("Recording... Press Enter to stop.\n", file=sys.stderr)
            else:
                _status("Recording... Press Enter to stop.")
            proc: subprocess.Popen[str] | None = None
            backend = ""
            stop_event: threading.Event | None = None
            stop_thread: threading.Thread | None = None
            last_live_text = ""
            try:
                if audio_path.exists():
                    audio_path.unlink()
                proc, backend = start_recording(audio_path, args.sample_rate, args.verbose)
                if args.live:
                    print("Live mode enabled. Partial transcription will stream below.\n", file=sys.stderr)
                    stop_event = threading.Event()
                    stop_thread = threading.Thread(
                        target=_wait_for_stop_key_event, args=(stop_event,), daemon=True
                    )
                    stop_thread.start()

                    if model is None:
                        model = load_model(args.model, args.compute_type, args.device, args.verbose)
                    while not stop_event.is_set():
                        stop_event.wait(timeout=args.live_interval)
                        if stop_event.is_set():
                            break
                        if not audio_path.exists() or audio_path.stat().st_size < 2048:
                            continue
                        live_text = transcribe_with_model(
                            audio_path,
                            model,
                            args.verbose,
                            show_banner=False,
                            vad_filter=args.vad_filter,
                        )
                        if not live_text or live_text == last_live_text:
                            continue
                        delta = _text_delta(last_live_text, live_text)
                        if delta:
                            print(delta, flush=True)
                        last_live_text = live_text
                else:
                    wait_for_enter()
            except KeyboardInterrupt:
                print("\nExiting on Ctrl+C.", file=sys.stderr)
                if stop_event is not None:
                    stop_event.set()
                return 0
            except AppError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            finally:
                if proc is not None:
                    stop_recording(proc, backend, args.verbose)
                    if args.live:
                        print("Recording stopped.\n", file=sys.stderr)
                    else:
                        _status("Recording stopped.")
                if stop_thread is not None:
                    stop_thread.join(timeout=0.1)

            if not audio_path.exists() or audio_path.stat().st_size < 2048:
                print(
                    "Error: Recording is empty or too short to transcribe.\n",
                    file=sys.stderr,
                )
            else:
                try:
                    if model is None:
                        text = transcribe_file(
                            audio_path,
                            args.model,
                            args.compute_type,
                            args.device,
                            args.vad_filter,
                            args.verbose,
                        )
                    else:
                        text = transcribe_with_model(
                            audio_path,
                            model,
                            args.verbose,
                            show_banner=True,
                            vad_filter=args.vad_filter,
                        )
                    if text:
                        if args.live:
                            print("\nFinal transcript:")
                            print(text)
                        else:
                            _status(text)
                            _status_done()
                        if copy_to_clipboard(text):
                            pass
                        else:
                            print(
                                "Warning: Could not copy to clipboard (need wl-copy, xclip, or xsel).",
                                file=sys.stderr,
                            )
                    else:
                        if args.live:
                            print("No speech detected.")
                        else:
                            _status("No speech detected.")
                            _status_done()
                except AppError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1

            if args.keep_audio and audio_path.exists():
                keep_path = Path.cwd() / f"wisper_recording_{int(time.time())}.wav"
                shutil.copy2(audio_path, keep_path)
                print(f"Saved audio to {keep_path}", file=sys.stderr)

            _status("Press Enter to start recording again. Press Ctrl+C to exit.")
            try:
                wait_for_enter()
            except KeyboardInterrupt:
                _status("Exiting on Ctrl+C.")
                _status_done()
                return 0
    finally:
        tmpdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
