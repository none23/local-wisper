# Local Wisper

Local speech-to-text for x86_64 Linux. Record from the command line or a Sway
keybinding, transcribe locally with NVIDIA Parakeet, and send the result to the
clipboard or focused window.

Local Wisper keeps one model warm for fast repeated transcription. It selects
FP16 CUDA inference when CUDA works and falls back to INT8 CPU inference. Users
do not need to choose a model, device, or weight format.

This release is experimental. It installs a native `lw` executable and does
not need Python during normal use. Faster Whisper is no longer supported.

## Requirements

- x86_64 Linux
- `pw-record` or `ffmpeg` for audio capture
- `wl-copy`, `xclip`, or `xsel` for clipboard output
- `wtype` for typing into the focused Sway window
- BAML 0.16, `cargo`, `curl`, and `sha256sum` to build and install

An NVIDIA GPU is optional. Systems without working CUDA use the CPU
automatically.

## Install

```bash
git clone https://github.com/none23/local-wisper.git
cd local-wisper
./install.sh
lw preload
```

The installer creates:

- `~/.local/bin/lw`
- `~/.config/local-wisper/env`
- `~/.config/local-wisper/glossary.txt`

It leaves existing configuration files untouched. Make sure `~/.local/bin` is
in `PATH`.

The first `lw preload` downloads and verifies the Parakeet files for the
selected device. Later commands reuse the same cached files and warm model.
Local Wisper allows only one model process per user to avoid duplicate RAM or
VRAM use.

On a Manjaro system with an NVIDIA GPU, the installer can place a private copy
of cuDNN 9 under `~/.local/lib/local-wisper` when the system does not provide
it. CPU-only systems skip CUDA setup. Normal use needs neither Python, Cargo,
nor the BAML CLI.

## Usage

- `lw`: record until Enter, transcribe, print the result, and copy it
- `lw preload`: load the model before the first recording
- `lw sway-start`: begin a detached recording
- `lw sway-stop`: stop, transcribe, and deliver the recording
- `lw sway-cancel`: discard the active Sway recording

Run `lw --help` to print the command summary.

## Sway integration

Install the supplied wrapper:

```bash
install -Dm755 integrations/sway/local-wisper.sh \
  ~/.config/sway/scripts/local-wisper.sh
```

A minimal Sway configuration looks like this:

```text
set $local_wisper $HOME/.config/sway/scripts/local-wisper.sh
set $mode_local_wisper local-wisper

exec_always $local_wisper preload

mode "$mode_local_wisper" {
    bindsym $mod+grave mode "default", exec $local_wisper sway-stop
    bindsym Return mode "default", exec $local_wisper sway-stop
    bindsym Escape mode "default", exec $local_wisper sway-cancel
}

bindsym $mod+grave exec $local_wisper sway-start, mode "$mode_local_wisper"
```

The wrapper reads `~/.config/local-wisper/env` and types completed transcripts
into the focused window by default. Existing Sway wrappers remain compatible.

Speech-to-text works better as a system-level feature than an editor feature,
so this version removes the Neovim plugin. Sway is the only bundled
integration.

## Transcript cleanup

Local cleanup handles spoken decimals, phrases such as `numeric three`,
statement formatting, and deterministic glossary replacements.

Optional OpenAI cleanup improves punctuation and recurring technical terms.
When enabled, transcripts with six or more words are sent to `gpt-5.6-luna`.
Add the following values to `~/.config/local-wisper/env`:

```bash
export OPENAI_API_KEY='...'
export LW_POST_PROCESS_MODEL='gpt-5.6-luna'
export LW_POST_PROCESS_TIMEOUT='20'
export LW_POST_PROCESS_GLOSSARY_FILE="$HOME/.config/local-wisper/glossary.txt"
```

If model cleanup fails or reaches the deadline, Local Wisper returns the local
result. The glossary supports four sections:

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

- `[always]` applies a local replacement every time.
- `[likely]` asks model cleanup to prefer the replacement.
- `[contextual]` applies when the surrounding text supports it.
- `[terms]` supplies preferred spelling and capitalization.

## Performance

On the development machine, an 11.04-second recording took 0.36 seconds with
FP16 CUDA and 0.80 seconds with INT8 CPU. The CPU model process used about 1 GB
of memory. Results depend on the machine.

## Technical notes

The `lw` executable runs the application in BAML and uses a small Rust host for
native Parakeet ONNX inference and Linux process operations. A per-user model
service keeps Parakeet warm. Its authenticated loopback endpoint is available
only through a private user runtime directory.

The model files are pinned and verified before use. A per-user lock covers
model selection, download, and loading so concurrent commands cannot create a
second model process.

## Development

Install `pre-commit` and `shellcheck`, then enable the fast local checks:

```bash
pre-commit install
pre-commit run --all-files
```

After changing a `.baml` file, run:

```bash
baml fmt
baml check
baml test
baml generate
cargo test
```

The generated Rust SDK is build output and is not committed. Run
`baml generate` before a direct Cargo build. The installer performs this step
automatically.
