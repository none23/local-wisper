# lw.nvim

Local speech-to-text Neovim plugin.

In normal mode, run `:LW` (or map it) to:
- start recording (`recording (press Enter to stop)`)
- stop on `Enter`
- transcribe locally with Whisper
- insert transcript below the cursor

## Requirements
- Linux with `pw-record` (PipeWire)
- Python with this repo dependencies installed (`faster-whisper`)

## Install

### lazy.nvim
```lua
{
  "none23/local-wisper",
  config = function()
    require("lw").setup({
      python_bin = "/home/n/none23/local-wisper/.venv/bin/python",
      model = "small.en",
      compute_type = "int8",
      sample_rate = 16000,
      -- Optional: full custom recorder command (audio path is appended)
      -- recorder_cmd = { "pw-record", "--rate", "16000", "--channels", "1", "--format", "s16" },
    })

    vim.keymap.set("n", "<leader>lw", "<cmd>LW<CR>", { desc = "Local Whisper" })
  end,
}
```

## Setup options
- `python_bin` (string|nil): Python executable for transcription. If unset, plugin tries repo `.venv/bin/python` then `python3`.
- `model` (string): Whisper model name/path. Default: `small.en`.
- `compute_type` (string): faster-whisper compute type. Default: `int8`.
- `sample_rate` (number): recording sample rate. Default: `16000`.
- `recorder_cmd` (string[]|nil): custom recording command prefix. Plugin appends output wav path.

## Usage
- `:LW`: toggle recording/transcription flow.
- while recording, press `Enter` to stop and insert transcript.

## Troubleshooting
- `Missing dependency 'faster-whisper'`:
  - install dependencies into the Python used by `python_bin`
- `failed to start recorder`:
  - install `pw-record` or set `recorder_cmd`

## Development
This repo still includes a standalone CLI (`wisper_cli.py`) and helper script (`scripts/transcribe_file.py`) used by the plugin.
