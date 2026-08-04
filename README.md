# Local Wisper

Local speech-to-text for Linux. Record from the command line or a desktop keybinding, transcribe locally with NVIDIA Parakeet or Whisper, and deliver the result to the clipboard or the focused window.

Local Wisper keeps the transcription model warm in a background daemon, which makes repeated recordings and integrations such as Sway and Neovim much faster.

## Requirements

- Linux
- Python 3 with virtual environment support
- `pw-record` (PipeWire) or `ffmpeg` with PulseAudio input support
- `wl-copy`, `xclip`, or `xsel` for clipboard output
- `wtype` when typing directly into a Wayland window
- NVIDIA GPU support is optional; CPU transcription works out of the box

## Install

```bash
git clone https://github.com/none23/local-wisper.git
cd local-wisper
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./install.sh
```

The installer creates:

- `~/.local/bin/lw`, pointing at this checkout and its virtual environment
- `~/.config/local-wisper/env`, containing integration defaults
- `~/.config/local-wisper/glossary.txt`, containing reusable transcript corrections

Existing configuration files are left untouched. Make sure `~/.local/bin` is in `PATH`, then run:

```bash
lw --help
```

## Usage

Start an interactive recording session:

```bash
lw --backend parakeet --device cuda --compute-type float16 --no-vad-filter
```

Press Enter to finish recording. The transcript is copied to the clipboard, and the model remains available through the background daemon for the next recording.

Whisper on CPU:

```bash
lw --backend whisper --model small --compute-type int8 --device cpu
```

Useful commands:

- `lw`: record interactively and transcribe
- `lw preload`: start the daemon and load the model ahead of time
- `lw sway-start`: begin a detached recording
- `lw sway-stop`: stop a detached recording and deliver its transcript
- `lw sway-cancel`: discard a detached recording
- `lw sway-toggle`: start or stop a detached recording

Run `lw --help` for recording, daemon, output, and post-processing options.

## Sway integration

The supplied wrapper reads `~/.config/local-wisper/env`, forwards its settings to `lw`, and uses `wtype` to type completed transcripts into the focused window.

For a new Sway setup, copy it into your configuration:

```bash
install -Dm755 integrations/sway/local-wisper.sh ~/.config/sway/scripts/local-wisper.sh
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

The generated environment file defaults to Parakeet on CUDA with direct typing. Common overrides are:

```bash
export LW_BACKEND='parakeet'
export LW_COMPUTE_TYPE='float16'
export LW_DEVICE='cuda'
export LW_VAD_FILTER='false'
export LW_OUTPUT_MODE='type' # use "clipboard" to disable wtype output
```

Sway users upgrading an existing checkout do not need to copy the new wrapper or run the installer again. The existing wrapper continues to call the unchanged `lw` command and `LW_*` interface.

## Transcript cleanup

Local Wisper always performs conservative local cleanup. Optional OpenAI post-processing can improve punctuation and recurring technical terms. Luna post-processing runs with reasoning disabled:

```bash
export OPENAI_API_KEY='...'
export LW_POST_PROCESS_MODEL='gpt-5.6-luna'
export LW_POST_PROCESS_TIMEOUT='20'
export LW_POST_PROCESS_GLOSSARY_FILE="$HOME/.config/local-wisper/glossary.txt"
```

The glossary supports four sections:

```text
[always]
engine x -> nginx

[likely]
cloud code -> Claude Code

[contextual]
codecs -> Codex

[terms]
TypeScript
TanStack Query
```

- `[always]` applies deterministic, case-insensitive local replacements.
- `[likely]` asks model post-processing to prefer the replacement unless context contradicts it.
- `[contextual]` applies only when the surrounding text supports the replacement.
- `[terms]` supplies preferred spelling and capitalization without inserting absent terms.

Mappings use `recognized phrase -> intended output`. Blank lines and lines beginning with `#` are ignored. Existing unsectioned glossary files remain supported as legacy prompt text.

## Neovim integration

Neovim support remains available as an optional integration. With lazy.nvim:

```lua
{
  "none23/local-wisper",
  config = function()
    require("lw").setup({
      backend = "parakeet",
      device = "cpu",
      vad_filter = false,
      sample_rate = 16000,
      post_process_model = "gpt-5.6-luna",
      post_process_glossary_file = "~/.config/local-wisper/glossary.txt",
    })

    vim.keymap.set("n", "<leader>lw", "<cmd>LW<CR>", { desc = "Local Speech" })
  end,
}
```

Use `:LW` to start recording, then press Enter to stop and insert the transcript below the cursor. Use `:LWInstallDeps` to install dependencies manually.

If a Python environment is not configured, the plugin creates one at `stdpath("data") .. "/lw.nvim/.venv"` on first use. The first dependency installation and model preload can take several minutes.

Setup options:

- `python_bin`: explicit Python executable; disables automatic dependency bootstrap
- `venv_dir`: custom plugin virtual environment directory
- `auto_install_deps`: install missing dependencies automatically; default `true`
- `backend`: `parakeet` or `whisper`; default `parakeet`
- `model`: model name or path
- `compute_type`: backend compute type
- `device`: inference device; default `cpu`
- `vad_filter`: enable voice activity detection; default `true`
- `sample_rate`: recording sample rate; default `16000`
- `recorder_cmd`: custom recording command prefix
- `preload_on_setup`: warm the daemon during `setup()`; default `true`
- `post_process_model`: optional OpenAI text model
- `post_process_prompt`: custom cleanup prompt
- `post_process_glossary_file`: correction glossary path
- `post_process_timeout`: cleanup timeout in seconds; default `20`

## Upgrading from the Neovim-first layout

No system changes are required after merging or pulling this restructure:

- Existing `~/.local/bin/lw` launchers still execute the root `wisper_cli.py` compatibility entry point.
- Existing root `.venv` environments remain in the same location.
- Existing Sway scripts continue using the same commands, environment variables, configuration, state, and cache paths.
- Neovim plugin managers still discover `plugin/lw.lua` and `lua/lw/init.lua` at the repository root.
- `require("lw")`, `:LW`, `:LWInstallDeps`, and all setup options are unchanged.

Update the checkout with `git pull`, or update the plugin through the normal Neovim plugin-manager command. You do not need to rerun `install.sh`, reinstall Python dependencies, or modify Sway or Neovim configuration.

Rerun `install.sh` only if the checkout itself is moved to another directory, because the installed `lw` launcher intentionally stores absolute paths to the checkout and its virtual environment.

## Performance notes

- Parakeet with `device = "cuda"`, `compute_type = "float16"`, and VAD disabled is generally the lowest-latency configuration on a supported NVIDIA GPU.
- The installed PyTorch wheel supplies the CUDA runtime used by Parakeet; Local Wisper discovers and preloads its NVIDIA libraries automatically.
- Whisper works on CPU out of the box. Whisper CUDA may require a separate CTranslate2-compatible CUDA runtime.
- The daemon socket and Sway recording state remain under `~/.cache/lw.nvim`, or `$XDG_CACHE_HOME/lw.nvim` when set.

## Troubleshooting

- Recording fails: install `pw-record`, or install `ffmpeg` with PulseAudio support.
- Clipboard delivery fails: install `wl-clipboard`, `xclip`, or `xsel`.
- Sway typing fails: install `wtype` and keep `LW_OUTPUT_MODE=type`.
- Neovim dependency installation fails: check `:messages`, ensure `python3` is available, and rerun `:LWInstallDeps`.
- A moved checkout makes `lw` fail: run `./install.sh` again from the new checkout location.

## Development

The primary Python application lives in `local_wisper/`. Stable launchers remain at `wisper_cli.py` and `scripts/` for existing installations. Optional integrations live in `integrations/`, with the small root `lua/` and `plugin/` adapters required by Neovim's runtime discovery.

Run the Python tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
