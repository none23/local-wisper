# Local Wisper, BAML experiment

This branch replaces the Python application with a compiled `lw` executable.
BAML defines the command workflow, cleanup policy, and optional OpenAI cleanup.
A small Rust host records audio, owns the resident Parakeet model, and selects
the fastest supported local runtime.

The application uses one fixed setup:

- `nvidia/parakeet-tdt-0.6b-v3`
- automatic FP16 CUDA or INT8 CPU inference
- 16 kHz mono recording
- no VAD

There is no Faster Whisper backend. Users do not choose a model, weight format,
or device. `lw` tries CUDA when an NVIDIA device is present and falls back to
CPU if CUDA cannot initialize the model.

## Install

The installer targets x86_64 Linux and needs `cargo` and `curl`. On a Manjaro
CUDA system without cuDNN 9, it also uses `bsdtar`, `pacman`, and `pacman-key`
to install a verified local copy. A CPU-only system skips every CUDA setup
step. Audio capture needs `pw-record` or `ffmpeg`; Sway typing needs `wtype`.

```bash
./install.sh
lw preload
```

`install.sh` builds and copies `~/.local/bin/lw`. If an NVIDIA GPU is present
and the system does not already provide cuDNN 9, it downloads the signed
Manjaro package, verifies it with the pacman keyring, and extracts its libraries under
`~/.local/lib/local-wisper`. It also downloads and verifies BAML's 0.16 runtime
library during installation. When upgrading from the Python version, it stops
the old resident model. It also stops an installed native daemon during an
upgrade. Normal use needs neither Python nor the BAML toolchain.

The first `lw preload` selects FP16 for CUDA or INT8 for CPU, downloads three
pinned Parakeet files, verifies their sizes and SHA-256 hashes, and leaves one
daemon running for the user. Later commands reuse the same model and cache.
The exclusive user lock covers detection, download, and model loading, so an
automatic fallback cannot overlap two model instances.

## Commands

```text
lw
lw preload
lw sway-start
lw sway-stop
lw sway-cancel
```

Bare `lw` records until Enter, prints the transcript, and copies it to the
clipboard. The Sway commands keep the existing wrapper contract. The supplied
wrapper can remain at `~/.config/sway/scripts/local-wisper.sh` with no changes.

## Transcript cleanup

Local cleanup always handles spoken decimals, explicit phrases such as
`numeric three`, statement style, and `[always]` glossary rules. Six-word or
longer transcripts use the BAML `gpt-5.6-luna` function when
`LW_POST_PROCESS_MODEL` and `OPENAI_API_KEY` are present. A model error or the
configured 20-second deadline returns the local result.

The glossary format is:

```text
[always]
engine x -> nginx

[likely]
cloud code -> Claude Code

[contextual]
codecs -> Codex

[terms]
TypeScript
```

The Sway wrapper reads `~/.config/local-wisper/env`. Existing configuration is
left untouched by the installer.

## Development

When a `.baml` file changes:

```bash
baml check
baml test
baml generate
cargo test
```

Build the executable with `cargo build --release`. The checked-in generated
Rust SDK embeds the BAML bytecode, so release builds do not invoke BAML.

The process split is intentionally small. BAML returns an exhaustive action
plan for each command. Rust executes those actions and holds an exclusive
per-user lock before loading Parakeet. That lock is what prevents two model
copies from entering memory at once.

The checked fixture on the development machine took 0.36 seconds with FP16
CUDA and 0.80 seconds with INT8 CPU for 11.04 seconds of audio. The CPU daemon
used about 1 GB of resident memory. CPU results will depend on the machine.
