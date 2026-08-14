# BAML rewrite scratchpad

This file records the agreed constraints and implementation findings for the
experimental rewrite. Keep it current while work is in progress.

## Product boundary

- Replace the Python application with an application written primarily in BAML.
- The final installed application must not require Python, a virtual environment,
  or the BAML toolchain at runtime.
- Produce one `lw` executable. Model assets remain external to the executable.
- A small Rust layer may implement Parakeet inference and native operations that
  BAML cannot express. All application behavior should remain in BAML where the
  language permits it.
- This worktree is experimental. Compatibility with the primary checkout is not
  required beyond the explicitly preserved user-facing workflow.

## Required behavior

- Use `nvidia/parakeet-tdt-0.6b-v3` only.
- CUDA inference is mandatory. CPU inference is not a useful fallback.
- Use the current machine configuration: CUDA, float16, 16 kHz mono audio, VAD
  disabled.
- Never load more than one model copy for the user. Concurrent commands must
  reuse or wait for the single resident model owner.
- Cache one verified copy of model assets per user. Download and preparation
  must be locked and atomic.
- Preserve these commands:
  - bare `lw` for manual recording
  - `lw preload`
  - `lw sway-start`
  - `lw sway-stop`
  - `lw sway-cancel`
- Accept the exact flags currently emitted by the installed Sway wrapper, but do
  not build a general configuration system.
- Preserve typed output through `wtype` for the Sway flow.
- Preserve deterministic glossary/number cleanup and optional OpenAI cleanup
  using `gpt-5.6-luna` with the current 20-second timeout.
- If OpenAI cleanup fails, warn on stderr and deliver the raw local transcript.
- Ordinary transcription must remain local. Remote tracing is not required.
- Use BAML's built-in local structured tracing if it works out of the box. Do
  not build another tracing system for this experiment.

## Explicit non-goals

- No Faster Whisper backend.
- No Neovim integration.
- No compatibility with unused Python CLI flags.
- No preservation of the old newline-delimited JSON socket protocol.
- No CPU-only success path.
- No elaborate crash recovery for the resident process.

## Toolchain strategy

- Installed BAML wrapper: 0.2.4.
- Installed BAML toolchain at project start: 0.16.0 canary.
- `baml pack` is available.
- `baml bridge` is not exposed by the installed CLI even though the public
  quickstart advertises that command. In 0.16, the working Rust path is
  `baml generate add rust`; the generated SDK embeds BAML bytecode and exposes
  typed host callables.
- The Rust SDK loads the BAML engine from a versioned native shared library. It
  can download that library into the user cache on first use. Treat it like the
  ONNX/CUDA shared libraries allowed by the packaging decision, and make the
  installer acquire it so normal runtime does not depend on a network request.
- Python and NeMo are allowed for one-time model conversion or preparation.
  They must not be required to build, install, or run the final application.

## Hard feasibility gate

Before expanding the rewrite, prove that the exact Parakeet v3 model can perform
correct native CUDA inference on this machine. The initial `nvidia-smi` probe
reported an NVML driver/library mismatch. The running kernel module is
`610.43.03`, while installed NVIDIA userspace is `610.57.04`; a reboot is likely
needed before the CUDA gate can pass. Do not substitute CPU inference and
continue as though the gate passed.

The native inference candidate is `parakeet-rs` 0.3.7 with ONNX Runtime's CUDA
execution provider. A canonical FP16 export of the exact v3 model is available
from `ysdede/parakeet-tdt-0.6b-v3-onnx` with the encoder, decoder/joint graph,
vocabulary, and preprocessing graph expected by the Rust decoder.

## Current system integration

- Sway invokes `preload`, `sway-start`, `sway-stop`, and `sway-cancel`.
- Current environment values:
  - backend: `parakeet`
  - compute type: `float16`
  - device: `cuda`
  - VAD: `false`
  - output mode: `type`
  - post-process model: `gpt-5.6-luna`
  - post-process timeout: `20`
  - glossary: `~/.config/local-wisper/glossary.txt`

## Working rules

- Make atomic commits at meaningful milestones.
- Keep this file updated when a decision or feasibility finding changes the
  implementation.
- Do not switch the system installation to this worktree until CUDA inference
  and the required Sway workflow work end to end.
