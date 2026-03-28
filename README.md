# lw.nvim

Local speech-to-text Neovim plugin.

In normal mode, run `:LW` (or map it) to:
- start recording (`recording (press Enter to stop)`)
- stop on `Enter`
- transcribe locally with NVIDIA Parakeet or Whisper
- insert transcript below the cursor

## Requirements
- Linux with `pw-record` (PipeWire)
- `python3` available on PATH (only needed for first-time dependency bootstrap)
- CUDA mode (`device = "cuda"`) uses the CUDA runtime bundled with the installed PyTorch wheel. The plugin auto-discovers and preloads the relevant NVIDIA user-space libraries from the venv when needed.

## Install

### lazy.nvim
```lua
{
  "none23/local-wisper",
  config = function()
    require("lw").setup({
      backend = "parakeet",
      device = "cpu",
      vad_filter = false,
      sample_rate = 16000,
    })

    vim.keymap.set("n", "<leader>lw", "<cmd>LW<CR>", { desc = "Local Speech" })
  end,
}
```

Whisper example:
```lua
require("lw").setup({
  backend = "whisper",
  model = "small",
  compute_type = "int8",
  device = "cpu",
})
```

## First run behavior
- If dependencies are missing, `:LW` automatically creates a plugin venv in:
  - `stdpath("data") .. "/lw.nvim/.venv"`
- Then it installs `requirements.txt` into that venv.
- The first Parakeet preload can take a few minutes while Python wheels and model weights are initialized.
- After install completes, run `:LW` again.

You can also trigger install manually:
- `:LWInstallDeps`

## Setup options
- `python_bin` (string|nil): explicit Python executable for transcription. If set, auto-bootstrap is skipped.
- `venv_dir` (string|nil): target venv dir for auto-bootstrap. Default: `stdpath("data") .. "/lw.nvim/.venv"`.
- `auto_install_deps` (boolean): auto-install missing dependencies on `:LW`. Default: `true`.
- `backend` (string): `parakeet` or `whisper`. Default: `parakeet`.
- `model` (string|nil): model name/path. Defaults to `nvidia/parakeet-tdt-0.6b-v3` for `parakeet`, `small` for `whisper`.
- `compute_type` (string|nil): compute type. Defaults to `float32` for `parakeet`, `int8` for `whisper`.
- `device` (string): inference device. Default: `cpu`.
- `vad_filter` (boolean): enable VAD filtering. Default: `true`.
- `sample_rate` (number): recording sample rate. Default: `16000`.
- `recorder_cmd` (string[]|nil): custom recording command prefix. Plugin appends output wav path.
- `preload_on_setup` (boolean): start daemon/model warmup on `setup()`. Default: `true`.

## Performance notes
- The plugin keeps a detached Python daemon with a preloaded model, so repeated `:LW` calls and new Neovim sessions avoid model reload overhead.
- For lowest latency, keep `vad_filter = false`. For Parakeet, `device = "cuda"` is usually the fastest option if your machine supports it.
- If your machine supports it, `device = "cuda"` can be much faster than CPU.
- CUDA runtime libraries from the PyTorch/NVIDIA wheel stack are auto-discovered/preloaded, so users should not need manual `ldconfig` or `LD_LIBRARY_PATH` setup.
- In the shared repo venv, Parakeet CUDA is the primary GPU path. Whisper works on CPU out of the box; Whisper CUDA may require a separate CTranslate2-compatible CUDA runtime setup.

## Usage
- `:LW`: toggle recording/transcription flow.
- while recording, press `Enter` to stop and insert transcript.
- `lw preload`: start the persistent daemon and preload the model for non-Neovim integrations.
- `lw sway-start` / `lw sway-stop`: start or stop a detached recording session intended for Sway keybindings.

CLI examples:
```bash
python wisper_cli.py --backend parakeet --device cuda
python wisper_cli.py --backend whisper --model small --compute-type int8 --device cpu
```

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
