#!/usr/bin/env python3
"""Record microphone audio locally and transcribe it with a local model."""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import re
import select
import shutil
import signal
import site
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import wave
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent
_CUDA_RUNTIME_READY = False
DEFAULT_BACKEND = "parakeet"
DEFAULT_POST_PROCESS_PROMPT = (
    "You are cleaning up a speech-to-text transcript for direct insertion into an editor. "
    "The transcript most likely refers to full-stack web development, including TypeScript, "
    "JavaScript, React, Next.js, Node.js, APIs, databases, CSS, command-line tools, file names, "
    "errors, and code. Preserve the user's meaning. Fix punctuation, capitalization, spacing, "
    "and obvious speech-recognition mistakes, especially web development terms. "
    "If English words are accidentally written in the wrong alphabet, especially Cyrillic phonetic "
    "spellings of English words, transliterate and normalize them back to intended English text. "
    "If a correction glossary is provided, use it for likely intended terms and recurring misheard phrases. "
    "Convert spoken decimal numbers like 'zero point one' to '0.1'. "
    "Convert explicit phrases like 'numeric one' or 'numeric zero' to literal digits. "
    "Do not add new facts. If a phrase is ambiguous, leave it unchanged. Return only the cleaned text."
)
DEFAULT_MODELS = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper": "small",
}
DEFAULT_COMPUTE_TYPES = {
    "parakeet": "float32",
    "whisper": "int8",
}
_DIGIT_WORDS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEEN_WORDS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_DIGIT_WORD_PATTERN = "|".join(sorted(map(re.escape, _DIGIT_WORDS), key=len, reverse=True))
_TEEN_WORD_PATTERN = "|".join(sorted(map(re.escape, _TEEN_WORDS), key=len, reverse=True))
_TENS_WORD_PATTERN = "|".join(sorted(map(re.escape, _TENS_WORDS), key=len, reverse=True))
_BASE_NUMBER_PATTERN = (
    rf"(?:{_TENS_WORD_PATTERN})(?:[\s-]+(?:{_DIGIT_WORD_PATTERN}))?"
    rf"|(?:{_TEEN_WORD_PATTERN})"
    rf"|(?:{_DIGIT_WORD_PATTERN})"
)
_NUMBER_PHRASE_PATTERN = (
    rf"(?:{_DIGIT_WORD_PATTERN})[\s-]+hundred"
    rf"(?:[\s-]+and)?(?:[\s-]+(?:{_BASE_NUMBER_PATTERN}))?"
    rf"|(?:{_BASE_NUMBER_PATTERN})"
)
_NUMERIC_PREFIX_RE = re.compile(
    rf"\bnumeric[\s-]+(?P<number>(?:{_NUMBER_PHRASE_PATTERN}))\b",
    re.IGNORECASE,
)
_SPOKEN_DECIMAL_RE = re.compile(
    rf"\b(?P<integer>(?:{_NUMBER_PHRASE_PATTERN}))"
    rf"[\s-]+point[\s-]+(?P<fraction>(?:{_DIGIT_WORD_PATTERN})(?:[\s-]+(?:{_DIGIT_WORD_PATTERN}))*)\b",
    re.IGNORECASE,
)


class AppError(Exception):
    """Raised for user-facing runtime errors."""


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
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                key = str(lib_dir)
                if key not in seen and lib_dir.is_dir():
                    seen.add(key)
                    dirs.append(lib_dir)

    for path in (
        Path("/opt/cuda/lib64"),
        Path("/opt/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/lib64"),
    ):
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
        ("libcublas", ("libcublas.so", "libcublas.so.*")),
        ("libcublasLt", ("libcublasLt.so", "libcublasLt.so.*")),
        ("libcudnn", ("libcudnn.so", "libcudnn.so.*")),
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
        description=(
            "Record from microphone and print a local transcription, or drive "
            "the persistent daemon used for low-latency editor and Sway integrations."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="record",
        choices=[
            "record",
            "preload",
            "sway-start",
            "sway-stop",
            "sway-cancel",
            "sway-toggle",
        ],
        help="Action to run (default: record).",
    )
    parser.add_argument(
        "--backend",
        choices=["parakeet", "whisper"],
        default=DEFAULT_BACKEND,
        help=f"Transcription backend to use (default: {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--model",
        help="Model name/path for the selected backend.",
    )
    parser.add_argument(
        "--compute-type",
        help="Compute type for the selected backend.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (default: cpu).",
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
    parser.add_argument(
        "--socket-path",
        help="Custom unix socket path for the persistent transcription daemon.",
    )
    parser.add_argument(
        "--daemon-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the daemon to become ready (default: 300).",
    )
    parser.add_argument(
        "--transcribe-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for a daemon transcription response (default: 120).",
    )
    parser.add_argument(
        "--state-path",
        help="Custom state file path for Sway recording commands.",
    )
    parser.add_argument(
        "--type-output",
        action="store_true",
        help="Type the final transcript into the focused window with wtype instead of copying it.",
    )
    parser.add_argument(
        "--post-process-model",
        help="OpenAI text model used to clean up the final transcript before delivery.",
    )
    parser.add_argument(
        "--post-process-prompt",
        default=DEFAULT_POST_PROCESS_PROMPT,
        help="Instruction prompt for transcript post-processing.",
    )
    parser.add_argument(
        "--post-process-glossary-file",
        help="Path to an extra correction glossary file appended to the post-processing prompt.",
    )
    parser.add_argument(
        "--post-process-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for transcript post-processing (default: 20).",
    )
    return parser


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _status(text: str) -> None:
    print(f"\r\033[2K{text}", end="", flush=True)


def _status_done() -> None:
    print()


def _default_model_name(backend: str) -> str:
    return DEFAULT_MODELS[backend]


def _default_compute_type(backend: str) -> str:
    return DEFAULT_COMPUTE_TYPES[backend]


def _resolve_backend_options(
    backend: str,
    model_name: str | None,
    compute_type: str | None,
) -> tuple[str, str]:
    resolved_model = model_name or _default_model_name(backend)
    resolved_compute_type = compute_type or _default_compute_type(backend)
    return resolved_model, resolved_compute_type


def _config_key(
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
) -> str:
    return "|".join((backend, model_name, compute_type, device, "true" if vad_filter else "false"))


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base).expanduser() / "lw.nvim"
    return Path.home() / ".cache" / "lw.nvim"


def daemon_socket_path(
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
    *,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    base = _cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"daemon-{_short_hash(_config_key(backend, model_name, compute_type, device, vad_filter))}.sock"


def sway_state_path(
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
    *,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    base = _cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"sway-{_short_hash(_config_key(backend, model_name, compute_type, device, vad_filter))}.json"


def _daemon_script_path() -> Path:
    return REPO_ROOT / "scripts" / "transcribe_daemon.py"


def _ensure_any_command(commands: list[str]) -> None:
    if any(shutil.which(cmd) for cmd in commands):
        return
    joined = ", ".join(commands)
    raise AppError(
        f"Missing audio capture tool. Install one of: {joined}. "
        "On Manjaro, install PipeWire tools or ffmpeg."
    )


def _pw_record_cmd(output_path: Path, sample_rate: int) -> list[str]:
    return [
        "pw-record",
        "--rate",
        str(sample_rate),
        "--channels",
        "1",
        "--format",
        "s16",
        str(output_path),
    ]


def _ffmpeg_pulse_cmd(output_path: Path, sample_rate: int) -> list[str]:
    return [
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


def _start_pw_record(output_path: Path, sample_rate: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _pw_record_cmd(output_path, sample_rate),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_ffmpeg_pulse(output_path: Path, sample_rate: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _ffmpeg_pulse_cmd(output_path, sample_rate),
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
            _log(verbose, "Using capture backend: pw-record")
            return proc, "pw-record"
        attempts.append(("pw-record", proc))

    if shutil.which("ffmpeg"):
        proc = _start_ffmpeg_pulse(output_path, sample_rate)
        time.sleep(0.4)
        if proc.poll() is None:
            _log(verbose, "Using capture backend: ffmpeg pulse")
            return proc, "ffmpeg"
        attempts.append(("ffmpeg", proc))

    errors = []
    for backend, proc in attempts:
        stderr = proc.stderr.read().strip() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        errors.append(f"{backend}: {stderr or 'failed to start'}")
    raise AppError("Could not start audio capture.\n" + "\n".join(errors))


def start_background_recording(
    output_path: Path, sample_rate: int, verbose: bool, stderr_log_path: Path
) -> tuple[subprocess.Popen[str], str]:
    _ensure_any_command(["pw-record", "ffmpeg"])
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[str, int, str]] = []

    def launch(cmd: list[str], backend: str, startup_delay: float) -> subprocess.Popen[str]:
        with stderr_log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                text=True,
                start_new_session=True,
            )
        time.sleep(startup_delay)
        if proc.poll() is None:
            _log(verbose, f"Using capture backend: {backend}")
            return proc
        detail = stderr_log_path.read_text(encoding="utf-8", errors="replace").strip()
        attempts.append((backend, proc.returncode or 1, detail))
        return proc

    if shutil.which("pw-record"):
        proc = launch(_pw_record_cmd(output_path, sample_rate), "pw-record", 0.25)
        if proc.poll() is None:
            return proc, "pw-record"

    if shutil.which("ffmpeg"):
        proc = launch(_ffmpeg_pulse_cmd(output_path, sample_rate), "ffmpeg", 0.4)
        if proc.poll() is None:
            return proc, "ffmpeg"

    errors = [f"{backend}: {detail or f'failed to start (exit {code})'}" for backend, code, detail in attempts]
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


def stop_recording_pid(pid: int, backend: str) -> None:
    if not _process_exists(pid):
        return

    stop_signal = signal.SIGINT if backend == "ffmpeg" else signal.SIGTERM
    try:
        os.kill(pid, stop_signal)
    except ProcessLookupError:
        return

    if _wait_for_process_exit(pid, timeout=4.0):
        return

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    if not _wait_for_process_exit(pid, timeout=2.0):
        raise AppError("Timed out waiting for recorder to exit.")


def _require_parakeet_runtime() -> tuple[object, object]:
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except Exception as exc:  # pragma: no cover - import failure path
        raise AppError(
            "Missing dependency 'nemo_toolkit[asr]'. Install dependencies with "
            "`python -m pip install -r requirements.txt`. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc
    return torch, nemo_asr


def _require_faster_whisper() -> None:
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        raise AppError(
            "Missing dependency 'faster-whisper'. Install dependencies with "
            "`python -m pip install -r requirements.txt`. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc


def _normalize_device(device: str, torch) -> object:
    if device == "cpu":
        return torch.device("cpu")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise AppError("CUDA device requested, but torch.cuda.is_available() is false.")
        return torch.device(device)
    raise AppError(f"Unsupported device '{device}'. Use 'cpu' or 'cuda[:index]'.")


def _resolve_torch_dtype(compute_type: str, device: object, torch, verbose: bool):
    normalized = compute_type.strip().lower()
    if normalized in {"default", "float", "float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        if device.type != "cuda":
            raise AppError(f"Compute type '{compute_type}' requires a CUDA device.")
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized.startswith("int8"):
        _log(verbose, f"Compute type '{compute_type}' is not supported by Parakeet; using float32.")
        return torch.float32
    raise AppError(
        "Unsupported compute type for Parakeet. "
        "Use one of: float32, float16, bfloat16."
    )


def _extract_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "pred_text", "transcript"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.strip()
    pred_text = getattr(value, "pred_text", None)
    if isinstance(pred_text, str):
        return pred_text.strip()
    if isinstance(value, (list, tuple)):
        parts = [_extract_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return ""


def _write_filtered_wav(audio_path: Path, verbose: bool) -> tuple[Path | None, tempfile.TemporaryDirectory[str] | None]:
    try:
        import numpy as np
    except Exception as exc:
        raise AppError(f"Failed to import numpy for VAD preprocessing: {exc}") from exc

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            nframes = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(nframes)
    except (wave.Error, OSError) as exc:
        _log(verbose, f"Skipping VAD preprocessing for {audio_path}: {exc}")
        return audio_path, None

    if sample_width != 2 or channels < 1 or nframes <= 0:
        return audio_path, None

    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    if samples.size == 0:
        return None, None
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    frame_samples = max(1, int(sample_rate * 0.03))
    total_frames = samples.shape[0] // frame_samples
    if total_frames == 0:
        return audio_path, None

    trimmed = samples[: total_frames * frame_samples].astype(np.float32)
    framed = trimmed.reshape(total_frames, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(framed), axis=1))
    peak_rms = float(frame_rms.max(initial=0.0))
    if peak_rms < 80.0:
        return None, None

    active = frame_rms >= max(120.0, peak_rms * 0.08)
    if not active.any():
        return None, None

    padding_frames = max(1, int(round(0.15 / 0.03)))
    expanded = active.copy()
    for index, is_active in enumerate(active):
        if not is_active:
            continue
        start = max(0, index - padding_frames)
        stop = min(active.shape[0], index + padding_frames + 1)
        expanded[start:stop] = True

    kept_chunks: list[np.ndarray] = []
    for index, keep in enumerate(expanded):
        if keep:
            start = index * frame_samples
            stop = start + frame_samples
            kept_chunks.append(samples[start:stop])

    remainder = samples[total_frames * frame_samples :]
    if remainder.size and expanded[-1]:
        kept_chunks.append(remainder)

    if not kept_chunks:
        return None, None

    filtered = np.concatenate(kept_chunks).astype(np.int16, copy=False)
    if filtered.size == 0:
        return None, None

    tempdir = tempfile.TemporaryDirectory(prefix="wisper_vad_")
    filtered_path = Path(tempdir.name) / audio_path.name
    with wave.open(str(filtered_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(filtered.tobytes())
    return filtered_path, tempdir


def _load_parakeet_model(model_name: str, compute_type: str, device: str, verbose: bool):
    if device.startswith("cuda"):
        _prepare_cuda_runtime(verbose)
    torch, nemo_asr = _require_parakeet_runtime()
    target_device = _normalize_device(device, torch)
    dtype = _resolve_torch_dtype(compute_type, target_device, torch, verbose)

    if verbose:
        print(
            "Loading local Parakeet model (first run may download weights)...",
            file=sys.stderr,
        )
    t0 = time.perf_counter()
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name,
            map_location=target_device,
        )
    except TypeError:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model = model.to(target_device)

    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    model = model.eval()
    _log(verbose, f"Model load/init took {time.perf_counter() - t0:.2f}s")
    return {"backend": "parakeet", "model": model, "torch": torch}


def _load_whisper_model(model_name: str, compute_type: str, device: str, verbose: bool):
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
    return {"backend": "whisper", "model": model}


def load_model(
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    verbose: bool,
):
    if backend == "parakeet":
        return _load_parakeet_model(model_name, compute_type, device, verbose)
    if backend == "whisper":
        return _load_whisper_model(model_name, compute_type, device, verbose)
    raise AppError(f"Unsupported backend '{backend}'.")


def transcribe_with_model(
    audio_path: Path, model, verbose: bool, show_banner: bool, vad_filter: bool
) -> str:
    if show_banner and verbose:
        print("Transcribing audio...", file=sys.stderr)
    t1 = time.perf_counter()
    if model["backend"] == "parakeet":
        prepared_path = audio_path
        tempdir: tempfile.TemporaryDirectory[str] | None = None
        if vad_filter:
            prepared_path, tempdir = _write_filtered_wav(audio_path, verbose)
            if prepared_path is None:
                return ""

        try:
            with model["torch"].inference_mode():
                output = model["model"].transcribe(
                    [str(prepared_path)],
                    batch_size=1,
                    verbose=False,
                )
        finally:
            if tempdir is not None:
                tempdir.cleanup()

        text = _extract_text(output)
    elif model["backend"] == "whisper":
        segments, _info = model["model"].transcribe(str(audio_path), vad_filter=vad_filter)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    else:
        raise AppError(f"Unsupported backend '{model['backend']}'.")
    _log(verbose, f"Transcription took {time.perf_counter() - t1:.2f}s")
    return text


def transcribe_file(
    audio_path: Path,
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
    verbose: bool,
) -> str:
    model = load_model(backend, model_name, compute_type, device, verbose)
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


def type_into_focused_window(text: str) -> bool:
    if not text or not shutil.which("wtype"):
        return False

    try:
        proc = subprocess.run(
            ["wtype", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def deliver_text(text: str, *, type_output: bool) -> bool:
    if type_output:
        return type_into_focused_window(text)
    return copy_to_clipboard(text)


def _openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AppError("OPENAI_API_KEY is required for transcript post-processing.")
    return api_key


def _parse_spoken_number(text: str) -> int | None:
    words = re.split(r"[\s-]+", text.lower().strip())
    current = 0
    saw_number = False
    previous_kind: str | None = None

    for index, word in enumerate(words):
        if word == "and":
            if previous_kind != "hundred" or index == len(words) - 1:
                return None
            previous_kind = "and"
            continue
        if word in _DIGIT_WORDS:
            if previous_kind in {"digit", "teen"}:
                return None
            current += _DIGIT_WORDS[word]
            saw_number = True
            previous_kind = "digit"
        elif word in _TEEN_WORDS:
            if previous_kind in {"digit", "teen", "tens"}:
                return None
            current += _TEEN_WORDS[word]
            saw_number = True
            previous_kind = "teen"
        elif word in _TENS_WORDS:
            if previous_kind in {"digit", "teen", "tens"}:
                return None
            current += _TENS_WORDS[word]
            saw_number = True
            previous_kind = "tens"
        elif word == "hundred" and saw_number:
            if previous_kind != "digit":
                return None
            current *= 100
            previous_kind = "hundred"
        else:
            return None

    if not saw_number:
        return None
    return current


def normalize_spoken_numerics(text: str) -> str:
    def replace_decimal(match: re.Match[str]) -> str:
        integer = _parse_spoken_number(match.group("integer"))
        if integer is None:
            return match.group(0)

        fraction_digits = []
        for word in re.split(r"[\s-]+", match.group("fraction").lower().strip()):
            digit = _DIGIT_WORDS.get(word)
            if digit is None:
                return match.group(0)
            fraction_digits.append(str(digit))

        return f"{integer}.{''.join(fraction_digits)}"

    def replace_numeric_prefix(match: re.Match[str]) -> str:
        number = _parse_spoken_number(match.group("number"))
        if number is None:
            return match.group(0)
        return str(number)

    text = _SPOKEN_DECIMAL_RE.sub(replace_decimal, text)
    return _NUMERIC_PREFIX_RE.sub(replace_numeric_prefix, text)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _extract_response_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _response_incomplete_reason(payload: dict) -> str | None:
    if payload.get("status") != "incomplete":
        return None
    details = payload.get("incomplete_details")
    if isinstance(details, dict):
        reason = details.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return "unknown"


def post_process_text(
    text: str,
    *,
    model_name: str | None,
    prompt: str,
    glossary_file: str | None,
    timeout: float,
    verbose: bool,
) -> str:
    if not text:
        return text

    raw_word_count = _word_count(text)
    text = normalize_spoken_numerics(text)
    if not model_name or raw_word_count < 6:
        return text

    api_key = _openai_api_key()
    full_prompt = prompt
    if glossary_file:
        try:
            extra_glossary = Path(glossary_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise AppError(f"Could not read post-processing glossary file: {exc}") from exc
        if extra_glossary:
            full_prompt += "\n\nAdditional user correction glossary:\n" + extra_glossary

    payload = {
        "model": model_name,
        "instructions": full_prompt,
        "input": text,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    organization = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
    project = os.environ.get("OPENAI_PROJECT_ID")
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project

    t0 = time.perf_counter()
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=max(timeout, 1.0),
        )
    except requests.RequestException as exc:
        raise AppError(f"Transcript post-processing request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise AppError(f"Transcript post-processing failed ({response.status_code}): {detail}")

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise AppError(f"Transcript post-processing returned invalid JSON: {exc}") from exc

    incomplete_reason = _response_incomplete_reason(response_payload)
    if incomplete_reason is not None:
        raise AppError(f"Transcript post-processing returned incomplete output ({incomplete_reason})")

    processed = _extract_response_text(response_payload)
    if not processed:
        raise AppError("Transcript post-processing returned empty text.")
    _log(verbose, f"Transcript post-processing took {time.perf_counter() - t0:.2f}s")
    return normalize_spoken_numerics(processed)


def maybe_post_process_text(text: str, args: argparse.Namespace) -> str:
    if not getattr(args, "post_process_model", None):
        return normalize_spoken_numerics(text)
    try:
        return post_process_text(
            text,
            model_name=args.post_process_model,
            prompt=args.post_process_prompt,
            glossary_file=args.post_process_glossary_file,
            timeout=args.post_process_timeout,
            verbose=args.verbose,
        )
    except AppError as exc:
        print(f"Warning: {exc}; using raw transcript.", file=sys.stderr)
        return text


def _socket_is_live(socket_path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            sock.connect(str(socket_path))
        return True
    except OSError:
        return False


def ensure_daemon(
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
    *,
    verbose: bool,
    socket_path: Path,
    timeout: float,
    wait: bool,
) -> Path:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if _socket_is_live(socket_path):
        return socket_path

    script_path = _daemon_script_path()
    if not script_path.is_file():
        raise AppError(f"Missing daemon script: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--backend",
        backend,
        "--model",
        model_name,
        "--compute-type",
        compute_type,
        "--device",
        device,
        "--socket",
        str(socket_path),
    ]
    cmd.append("--vad-filter" if vad_filter else "--no-vad-filter")

    _log(verbose, f"Starting daemon on socket {socket_path}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(REPO_ROOT),
    )

    if not wait:
        return socket_path

    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline:
        if _socket_is_live(socket_path):
            return socket_path
        if proc.poll() is not None:
            raise AppError("Transcription daemon exited before becoming ready.")
        time.sleep(0.15)

    raise AppError("Transcription daemon did not become ready.")


def _daemon_request(socket_path: Path, payload: dict, timeout: float) -> dict:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
            conn = sock.makefile("rwb")
            conn.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
            conn.flush()
            raw_line = conn.readline()
    except OSError as exc:
        raise AppError(f"Failed to talk to transcription daemon: {exc}") from exc

    if not raw_line:
        raise AppError("Transcription daemon closed the connection without replying.")

    try:
        message = json.loads(raw_line.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AppError(f"Transcription daemon returned invalid JSON: {exc}") from exc

    if not isinstance(message, dict):
        raise AppError("Transcription daemon returned an invalid response.")
    return message


def transcribe_file_via_daemon(
    audio_path: Path,
    backend: str,
    model_name: str,
    compute_type: str,
    device: str,
    vad_filter: bool,
    *,
    verbose: bool,
    socket_path: Path,
    daemon_timeout: float,
    request_timeout: float,
) -> str:
    if not audio_path.exists() or audio_path.stat().st_size < 2048:
        return ""

    ensure_daemon(
        backend,
        model_name,
        compute_type,
        device,
        vad_filter,
        verbose=verbose,
        socket_path=socket_path,
        timeout=daemon_timeout,
        wait=True,
    )
    payload = {
        "type": "transcribe",
        "id": time.time_ns(),
        "audio_path": str(audio_path),
    }
    message = _daemon_request(socket_path, payload, timeout=request_timeout)

    msg_type = message.get("type")
    if msg_type == "result":
        text = message.get("text")
        if isinstance(text, str):
            return text
        raise AppError("Transcription daemon returned a malformed transcript.")
    if msg_type == "no_speech":
        return ""
    if msg_type == "error":
        detail = message.get("error") or "unknown error"
        raise AppError(f"Transcription failed: {detail}")
    raise AppError(f"Unexpected daemon response: {message!r}")


def _state_is_active(state: dict) -> bool:
    try:
        pid = int(state["pid"])
        backend = str(state["backend"])
    except (KeyError, TypeError, ValueError):
        return False
    return _process_matches_backend(pid, backend)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _process_matches_backend(pid: int, backend: str) -> bool:
    if not _process_exists(pid):
        return False
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        cmdline = cmdline_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return backend in cmdline


def _wait_for_process_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.1)
    return not _process_exists(pid)


def _read_json_file(path: Path) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AppError(f"Could not read state file {path}: {exc}") from exc

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AppError(f"State file {path} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AppError(f"State file {path} does not contain an object.")
    return value


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _save_audio_copy(audio_path: Path) -> Path | None:
    if not audio_path.exists():
        return None
    keep_path = Path.cwd() / f"wisper_recording_{int(time.time())}.wav"
    shutil.copy2(audio_path, keep_path)
    return keep_path


def _cleanup_recording_audio(audio_path: Path, *, keep_audio: bool) -> Path | None:
    kept_audio: Path | None = None
    if keep_audio and audio_path.exists():
        kept_audio = _save_audio_copy(audio_path)

    if (not keep_audio or kept_audio is not None) and audio_path.exists():
        audio_path.unlink()

    return kept_audio


def _cleanup_sway_state(state_path: Path, state: dict | None, *, keep_audio: bool) -> Path | None:
    kept_audio: Path | None = None
    audio_path = None
    tempdir = None
    if isinstance(state, dict):
        audio_raw = state.get("audio_path")
        tempdir_raw = state.get("tempdir")
        if isinstance(audio_raw, str):
            audio_path = Path(audio_raw)
        if isinstance(tempdir_raw, str):
            tempdir = Path(tempdir_raw)

    if keep_audio and audio_path is not None:
        try:
            kept_audio = _save_audio_copy(audio_path)
        except OSError:
            kept_audio = None

    try:
        state_path.unlink()
    except FileNotFoundError:
        pass

    if tempdir is not None:
        shutil.rmtree(tempdir, ignore_errors=True)
    elif audio_path is not None:
        try:
            audio_path.unlink()
        except FileNotFoundError:
            pass

    return kept_audio


def _require_sway_state(state_path: Path) -> dict:
    state = _read_json_file(state_path)
    if state is None:
        raise AppError("No active Sway recording.")
    return state


def cmd_preload(args: argparse.Namespace) -> int:
    model_name, compute_type = _resolve_backend_options(args.backend, args.model, args.compute_type)
    socket_path = daemon_socket_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.socket_path,
    )
    ensure_daemon(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        verbose=args.verbose,
        socket_path=socket_path,
        timeout=args.daemon_timeout,
        wait=True,
    )
    return 0


def cmd_sway_start(args: argparse.Namespace) -> int:
    model_name, compute_type = _resolve_backend_options(args.backend, args.model, args.compute_type)
    state_path = sway_state_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.state_path,
    )
    state = _read_json_file(state_path)
    if state is not None:
        if _state_is_active(state):
            raise AppError("Sway recording is already active.")
        _cleanup_sway_state(state_path, state, keep_audio=False)

    tempdir = Path(tempfile.mkdtemp(prefix="wisper_sway_"))
    audio_path = tempdir / "recording.wav"
    stderr_log_path = tempdir / "recording.stderr.log"
    try:
        proc, backend = start_background_recording(
            audio_path, args.sample_rate, args.verbose, stderr_log_path
        )
    except Exception:
        shutil.rmtree(tempdir, ignore_errors=True)
        raise

    _write_json_file(
        state_path,
        {
            "pid": proc.pid,
            "backend": backend,
            "audio_path": str(audio_path),
            "stderr_log_path": str(stderr_log_path),
            "tempdir": str(tempdir),
            "started_at": time.time(),
        },
    )

    socket_path = daemon_socket_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.socket_path,
    )
    try:
        ensure_daemon(
            args.backend,
            model_name,
            compute_type,
            args.device,
            args.vad_filter,
            verbose=args.verbose,
            socket_path=socket_path,
            timeout=args.daemon_timeout,
            wait=False,
        )
    except AppError as exc:
        _log(args.verbose, f"Daemon preload failed during recording start: {exc}")

    return 0


def cmd_sway_stop(args: argparse.Namespace) -> int:
    model_name, compute_type = _resolve_backend_options(args.backend, args.model, args.compute_type)
    state_path = sway_state_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.state_path,
    )
    state = _require_sway_state(state_path)
    if not _state_is_active(state):
        kept_audio = _cleanup_sway_state(state_path, state, keep_audio=args.keep_audio)
        if kept_audio is not None:
            print(f"Saved audio to {kept_audio}", file=sys.stderr)
        raise AppError("Sway recording process is not running anymore.")

    pid = int(state["pid"])
    backend = str(state["backend"])
    audio_path = Path(str(state["audio_path"]))

    try:
        stop_recording_pid(pid, backend)
        text = transcribe_file_via_daemon(
            audio_path,
            args.backend,
            model_name,
            compute_type,
            args.device,
            args.vad_filter,
            verbose=args.verbose,
            socket_path=daemon_socket_path(
                args.backend,
                model_name,
                compute_type,
                args.device,
                args.vad_filter,
                explicit_path=args.socket_path,
            ),
            daemon_timeout=args.daemon_timeout,
            request_timeout=args.transcribe_timeout,
        )
    finally:
        kept_audio = _cleanup_sway_state(state_path, state, keep_audio=args.keep_audio)

    if kept_audio is not None:
        print(f"Saved audio to {kept_audio}", file=sys.stderr)

    if not text:
        print("No speech detected.", file=sys.stderr)
        return 0

    text = maybe_post_process_text(text, args)
    print(text)
    if not deliver_text(text, type_output=args.type_output):
        if args.type_output:
            print(
                "Warning: Could not type transcript into the focused window (need wtype).",
                file=sys.stderr,
            )
        else:
            print(
                "Warning: Could not copy to clipboard (need wl-copy, xclip, or xsel).",
                file=sys.stderr,
            )
    return 0


def cmd_sway_cancel(args: argparse.Namespace) -> int:
    model_name, compute_type = _resolve_backend_options(args.backend, args.model, args.compute_type)
    state_path = sway_state_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.state_path,
    )
    state = _read_json_file(state_path)
    if state is None:
        return 0

    if _state_is_active(state):
        stop_recording_pid(int(state["pid"]), str(state["backend"]))

    kept_audio = _cleanup_sway_state(state_path, state, keep_audio=args.keep_audio)
    if kept_audio is not None:
        print(f"Saved audio to {kept_audio}", file=sys.stderr)
    return 0


def cmd_sway_toggle(args: argparse.Namespace) -> int:
    model_name, compute_type = _resolve_backend_options(args.backend, args.model, args.compute_type)
    state_path = sway_state_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.state_path,
    )
    state = _read_json_file(state_path)
    if state is None:
        return cmd_sway_start(args)
    return cmd_sway_stop(args)


def cmd_record(args: argparse.Namespace) -> int:
    if args.live_interval <= 0:
        raise AppError("--live-interval must be > 0.")

    model_name, compute_type = _resolve_backend_options(args.backend, args.model, args.compute_type)
    socket_path = daemon_socket_path(
        args.backend,
        model_name,
        compute_type,
        args.device,
        args.vad_filter,
        explicit_path=args.socket_path,
    )
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
                        model = load_model(args.backend, model_name, compute_type, args.device, args.verbose)
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
                    try:
                        ensure_daemon(
                            args.backend,
                            model_name,
                            compute_type,
                            args.device,
                            args.vad_filter,
                            verbose=args.verbose,
                            socket_path=socket_path,
                            timeout=args.daemon_timeout,
                            wait=False,
                        )
                    except AppError as exc:
                        _log(args.verbose, f"Daemon preload failed: {exc}")
                    wait_for_enter()
            except KeyboardInterrupt:
                print("\nExiting on Ctrl+C.", file=sys.stderr)
                if stop_event is not None:
                    stop_event.set()
                return 0
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
                print("Error: Recording is empty or too short to transcribe.\n", file=sys.stderr)
            else:
                if model is None:
                    text = transcribe_file_via_daemon(
                        audio_path,
                        args.backend,
                        model_name,
                        compute_type,
                        args.device,
                        args.vad_filter,
                        verbose=args.verbose,
                        socket_path=socket_path,
                        daemon_timeout=args.daemon_timeout,
                        request_timeout=args.transcribe_timeout,
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
                    text = maybe_post_process_text(text, args)
                    if args.live:
                        print("\nFinal transcript:")
                        print(text)
                    else:
                        _status(text)
                        _status_done()
                    if not deliver_text(text, type_output=args.type_output):
                        if args.type_output:
                            print(
                                "Warning: Could not type transcript into the focused window (need wtype).",
                                file=sys.stderr,
                            )
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

            keep_path = _cleanup_recording_audio(audio_path, keep_audio=args.keep_audio)
            if keep_path is not None:
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


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "preload":
            return cmd_preload(args)
        if args.command == "sway-start":
            return cmd_sway_start(args)
        if args.command == "sway-stop":
            return cmd_sway_stop(args)
        if args.command == "sway-cancel":
            return cmd_sway_cancel(args)
        if args.command == "sway-toggle":
            return cmd_sway_toggle(args)
        return cmd_record(args)
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
