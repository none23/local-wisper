# wisper CLI

Local microphone-to-text CLI for my Linux machine (Arch/PipeWire). Vibe-coded in a few minutes.

## What it does
- Starts recording immediately when run.
- Stops recording when you press `Enter`.
- Runs local Whisper transcription and prints final text to stdout.
- Copies each final transcript to clipboard.
- Stays running and waits for `Enter` to start the next recording (`Ctrl+C` exits).

## Requirements
- Python 3.13+
- `pw-record` (preferred) or `ffmpeg` with Pulse input support

## Install
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run
```bash
python wisper_cli.py
```

Speak, then press `Enter` in the terminal to stop and print transcript.
After each transcript, press `Enter` again to start a new recording.

## Useful flags
```bash
python wisper_cli.py --model small.en --compute-type int8 --verbose
python wisper_cli.py --keep-audio
python wisper_cli.py --model base.en
python wisper_cli.py --live --live-interval 0.8
```

## Notes
- First run may download model weights, then runs local from cache.
- Default model is `small.en` for strong CPU quality/speed balance.
- `--live` streams partial text while you speak and prints a final transcript at the end.
