# lw.nvim

Local speech-to-text Neovim plugin.

In normal mode, run `:LW` (or map it) to:
- start recording (`recording (press Enter to stop)`)
- stop on `Enter`
- transcribe locally with Whisper
- insert transcript below the cursor

## Requirements
- Linux with `pw-record` (PipeWire)
- `python3` available on PATH (only needed for first-time dependency bootstrap)
- CUDA mode (`device = "cuda"`) needs CUDA 12 runtime libraries (`libcublas.so.12`/cuDNN). The plugin installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` in its venv.

## Install

### lazy.nvim
```lua
{
  "none23/local-wisper",
  config = function()
    require("lw").setup({
      model = "tiny.en",
      compute_type = "int8",
      device = "cpu",
      vad_filter = false,
      sample_rate = 16000,
    })

    vim.keymap.set("n", "<leader>lw", "<cmd>LW<CR>", { desc = "Local Whisper" })
  end,
}
```

## First run behavior
- If dependencies are missing, `:LW` automatically creates a plugin venv in:
  - `stdpath("data") .. "/lw.nvim/.venv"`
- Then it installs `requirements.txt` into that venv.
- After install completes, run `:LW` again.

You can also trigger install manually:
- `:LWInstallDeps`

## Setup options
- `python_bin` (string|nil): explicit Python executable for transcription. If set, auto-bootstrap is skipped.
- `venv_dir` (string|nil): target venv dir for auto-bootstrap. Default: `stdpath("data") .. "/lw.nvim/.venv"`.
- `auto_install_deps` (boolean): auto-install missing dependencies on `:LW`. Default: `true`.
- `model` (string): Whisper model name/path. Default: `small.en`.
- `compute_type` (string): faster-whisper compute type. Default: `int8`.
- `device` (string): faster-whisper device. Default: `cpu`.
- `vad_filter` (boolean): enable VAD filtering. Default: `true`.
- `sample_rate` (number): recording sample rate. Default: `16000`.
- `recorder_cmd` (string[]|nil): custom recording command prefix. Plugin appends output wav path.
- `preload_on_setup` (boolean): start daemon/model warmup on `setup()`. Default: `true`.

## Performance notes
- The plugin keeps a detached Python daemon with a preloaded model, so repeated `:LW` calls and new Neovim sessions avoid model reload overhead.
- For lowest latency, prefer `model = "tiny.en"` (or `tiny` for multilingual) and set `vad_filter = false`.
- If your machine supports it, `device = "cuda"` can be much faster than CPU.
- CUDA runtime libraries from the plugin venv are auto-discovered/preloaded, so users should not need manual `ldconfig` or `LD_LIBRARY_PATH` setup.

## Usage
- `:LW`: toggle recording/transcription flow.
- while recording, press `Enter` to stop and insert transcript.
- `lw preload`: start the persistent daemon and preload the model for non-Neovim integrations.
- `lw sway-start` / `lw sway-stop`: start or stop a detached recording session intended for Sway keybindings.

## Sway
- Install `lw` first:
  - `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
  - `./install.sh`
- The Sway-specific wrapper now lives in your Sway config repo:
  - `swaywm-config/sway/scripts/local-wisper.sh`
- The sample Sway config in this workspace preloads the daemon on startup and binds `Mod+\`` to:
  - start recording and enter a Sway mode shown by Waybar's `sway/mode`
  - stop on the second `Mod+\`` or `Enter`
  - cancel on `Escape`
- The stop action transcribes through the persistent daemon and types the final text into the focused window with `wtype`.
- Set `LW_OUTPUT_MODE=clipboard` in the wrapper environment if you want the old clipboard behavior back.

## Troubleshooting
- `failed to start recorder`:
  - install `pw-record` or set `recorder_cmd`
- dependency install fails:
  - make sure `python3` is installed
  - check `:messages` and rerun `:LWInstallDeps`

## Development
This repo includes a standalone CLI (`wisper_cli.py`) and helper script (`scripts/transcribe_file.py`) used by the plugin.
