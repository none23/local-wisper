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

## Performance notes
- The plugin keeps a persistent Python worker with a preloaded model, so repeated `:LW` calls avoid model reload overhead.
- For lowest latency, prefer `model = "tiny.en"` (or `tiny` for multilingual) and set `vad_filter = false`.
- If your machine supports it, `device = "cuda"` can be much faster than CPU.

## Usage
- `:LW`: toggle recording/transcription flow.
- while recording, press `Enter` to stop and insert transcript.

## Troubleshooting
- `failed to start recorder`:
  - install `pw-record` or set `recorder_cmd`
- dependency install fails:
  - make sure `python3` is installed
  - check `:messages` and rerun `:LWInstallDeps`

## Development
This repo includes a standalone CLI (`wisper_cli.py`) and helper script (`scripts/transcribe_file.py`) used by the plugin.
