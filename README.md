# Local Wisper, BAML experiment

This branch replaces the Python application with a compiled `lw` executable.
BAML defines the command workflow, cleanup policy, and optional OpenAI cleanup.
A small Rust host records audio, owns the resident Parakeet model, and runs CUDA
inference.

The application uses one fixed setup:

- `nvidia/parakeet-tdt-0.6b-v3`
- FP16 CUDA inference
- 16 kHz mono recording
- no VAD

There is no CPU fallback and no Faster Whisper backend.

## Install

The current installer targets this Manjaro x86_64 system. It needs `cargo`,
`curl`, `bsdtar`, `pacman`, and `pacman-key`. Audio capture needs `pw-record` or
`ffmpeg`; Sway typing needs `wtype`.

```bash
./install.sh
lw preload
```

`install.sh` builds and copies `~/.local/bin/lw`. If the system does not already
provide cuDNN 9, it downloads the signed Manjaro package, verifies it with the
pacman keyring, and extracts its shared libraries under
`~/.local/lib/local-wisper`. It also downloads and verifies BAML's 0.16 runtime
library during installation. Normal use needs neither Python nor the BAML
toolchain.

The first `lw preload` downloads three pinned Parakeet files, verifies their
sizes and SHA-256 hashes, loads the model on CUDA, and leaves one daemon running
for the user. Later commands reuse the same model and cache.

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
