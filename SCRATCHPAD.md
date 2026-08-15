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
- BAML owns CLI parsing, complete command execution, the decision to use remote
  cleanup, and the OpenAI cleanup prompt. Rust injects typed native callbacks
  for capabilities that have not yet moved into BAML.
- This worktree is experimental. Compatibility with the primary checkout is not
  required beyond the explicitly preserved user-facing workflow.

## Required behavior

- Use `nvidia/parakeet-tdt-0.6b-v3` only.
- Device and model format are automatic implementation details. Prefer CUDA
  with the FP16 export when CUDA can initialize the model; otherwise use the
  CPU provider with the pinned INT8 export. Users should not need to choose a
  model, quantization, or device.
- Keep `--device`, `--model`, and `--compute-type` only where the existing Sway
  wrapper needs compatibility. The normal and documented mode is `auto`.
- Use 16 kHz mono audio with VAD disabled.
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
- Ordinary model failures cross the generated bridge as a typed `CleanResult`.
  The Rust host also handles bridge errors, applies the 20-second deadline, and
  falls back to local cleanup.

## Explicit non-goals

- No Faster Whisper backend.
- No Neovim integration.
- No compatibility with unused Python CLI flags.
- No preservation of the old newline-delimited JSON socket protocol.
- No user-facing model or quantization selection system.
- No elaborate crash recovery for the resident process.

The legacy Python application, Faster Whisper dependency list, Neovim plugin,
Python launchers, and their old tests were removed after the native workflows
passed. The Sway wrapper remains and now exposes only the retained commands.
The installer stops a matching legacy Python transcription daemon before the
first native preload, so migration cannot leave both model implementations in
GPU memory.

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
- The generated bridge was exercised from the native `lw` process. Its first
  invocation downloaded and verified BAML 0.16.0's
  `libbaml_cffi-x86_64-unknown-linux-gnu.so`; later calls reused the cache.
  BAML emits local structured runtime logs without extra application code.
- BAML also writes ignored structured profiles under `.baml/profiles/` during
  ordinary SDK calls. This is the advertised local tracing behavior, so no
  separate trace code was added.
- Python and NeMo are allowed for one-time model conversion or preparation.
  They must not be required to build, install, or run the final application.

## Hard feasibility gate

Before expanding the rewrite, prove that the exact Parakeet v3 model can perform
correct native CUDA inference on this machine. The initial `nvidia-smi` probe
reported an NVML driver/library mismatch. The running kernel module is
`610.43.03`, while installed NVIDIA userspace is `610.57.04`; a reboot is likely
needed before the CUDA gate can pass. Do not substitute CPU inference and
continue as though the gate passed.

The strict probe loaded the verified FP16 model files far enough to initialize
the CUDA provider, then failed at `cudaSetDevice` with CUDA error 803:
"system has unsupported display driver / cuda driver combination." This
confirms the version mismatch is the current hard blocker. Reboot before the
next probe so the running kernel module matches installed userspace.

After reboot, both the kernel module and userspace reported `610.57.04`. The
strict CUDA probe then passed with the canonical FP16 export: model load took
1.38 seconds and an 11.04-second fixture transcribed in 645 ms with the expected
sentence. Native CUDA inference is feasible on this machine.

ONNX Runtime requires cuDNN 9. The current Python environment contains the
native cuDNN libraries, and a cache-local `libcudnn.so` alias proved they work.
The installer now resolves the current signed Manjaro `cudnn` package through
pacman, verifies its detached signature with the system keyring, and extracts
the libraries under `~/.local/lib/local-wisper` when cuDNN 9 is not installed
system-wide. The executable preloads that directory before initializing CUDA.

ONNX Runtime's static build also resolved provider libraries beside the T3 Code
AppImage during the probe. Temporary symlinks proved provider discovery, then
were removed. The final package needs an explicit provider-library location
rather than relying on that environment-specific lookup.

The native inference candidate is `parakeet-rs` 0.3.7 with ONNX Runtime's CUDA
execution provider. A canonical FP16 export of the exact v3 model is available
from `ysdede/parakeet-tdt-0.6b-v3-onnx` with the encoder, decoder/joint graph,
vocabulary, and preprocessing graph expected by the Rust decoder.

The native daemon now holds an exclusive per-user file lock before touching the
model or binding its socket. This makes the one-model rule structural: racing
clients can start processes, but only the lock owner can load CUDA state. The
daemon handles requests serially and keeps that one model warm.

Automatic selection uses a real system check rather than a user-facing model
choice. An NVIDIA device selects the pinned FP16 export. With no NVIDIA device,
or when CUDA model initialization fails, the same daemon process loads the
pinned INT8 export on ONNX Runtime's CPU provider. The legacy `--device cuda`
input follows this automatic behavior so an unchanged Sway wrapper also works
on a CPU-only machine. `--device cpu` remains as a hidden test and compatibility
override.

The CPU feasibility test used the repository's INT8 encoder and decoder at the
same pinned revision as FP16. Checksums matched. The model loaded in 1.75
seconds, used about 1 GB resident memory, and transcribed the 11.04-second
fixture in 803 ms. Its raw result added a few filler tokens compared with FP16,
but preserved the sentence. With CUDA visible, automatic mode selected FP16,
loaded in 1.36 seconds, and transcribed the fixture in 363 ms. With
`nvidia-smi` hidden, automatic mode selected INT8 CPU and made no GPU
allocation.

The first implementation made Rust interpret a static action list returned by
BAML. That was the wrong boundary for this experiment: it made Rust own the
state machine and turned BAML into workflow metadata. The application now calls
one BAML `run_app` entrypoint. BAML parses the unchanged Sway invocation, owns
the command state and ordering, and invokes typed native closures supplied by
the Rust bootstrap. The generated Rust bridge supports this direction directly.
Five concurrent `preload` calls were previously tested against one resident
Rust process and one CUDA allocation.

The remaining goal is to keep moving implementations behind those callbacks
into BAML. Rust should finish as a small owner of the live `ParakeetTDT` object,
CUDA/ONNX setup, a per-user OS lock, and any Linux process operations that BAML
cannot express safely. Line count is not the goal; application ownership is.

BAML now also owns recorder selection, session paths, persisted Sway state,
interactive recording, start/stop/cancel behavior, audio validation, and
cleanup. The native recorder callback is limited to spawning and signalling a
Linux process. Persisted process identities contain both PID and `/proc` start
time, preventing stale state from signalling an unrelated process after PID
reuse.

The deterministic cleanup is implemented in Rust because BAML has no regular
expression support suitable for the established boundary-aware rules. BAML
owns the six-word decision and the complete `gpt-5.6-luna` prompt. Native tests
cover spoken decimals, `numeric` phrases, non-cascading `[always]` rules,
glossary validation, statement style, identifiers, questions, and non-Latin
text. A one-second `sway-start`/`sway-stop` capture also completed through the
earlier workflow; the empty recording correctly reported no speech.

Model assets are pinned to Hugging Face revision
`f88260fa0777fe0868dda6df85d1a98f012a4a7a`. The cache records exact sizes and
SHA-256 digests for the encoder, decoder/joint graph, and vocabulary. Downloads
land in `.part` files and are renamed only after verification. A completion
marker lets later daemon starts avoid hashing the 1.2 GB encoder again.

The optimized `lw` binary is 35 MB and links only the ordinary glibc, libstdc++,
libgcc, and libm runtime libraries at startup. A release build loaded cuDNN from
an explicit native-library directory with `LD_LIBRARY_PATH` removed, then
loaded Parakeet on CUDA in 1.33 seconds. It transcribed the 11.04-second fixture
correctly in 249 ms. The installer was run against the live user prefix after
explicit approval. It stopped the legacy Python daemon and replaced
`~/.local/bin/lw`. The first installed preload exposed that ONNX Runtime
resolves its CUDA provider shared objects beside the executable. The build now
emits those exact locked-version objects, and the installer stores them under
`~/.local/lib/local-wisper` with links beside `lw`.

## Current system integration

- Sway invokes `preload`, `sway-start`, `sway-stop`, and `sway-cancel`.
- Current environment values:
  - backend: `parakeet`
  - model format: selected automatically
  - device: automatic; the retained `cuda` compatibility value also falls back
    to CPU
  - VAD: `false`
  - output mode: `type`
  - post-process model: `gpt-5.6-luna`
  - post-process timeout: `20`
  - glossary: `~/.config/local-wisper/glossary.txt`

## Working rules

- Make atomic commits at meaningful milestones.
- Keep this file updated when a decision or feasibility finding changes the
  implementation.
- The system installation now points at this worktree's native build. A fresh
  install from main restores the Python launcher; stop this daemon first so the
  old implementation can claim the model memory.
