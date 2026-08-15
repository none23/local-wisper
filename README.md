<p align="center">
  <img src="assets/logo.svg" alt="Local Wisper logo" width="160">
</p>

# Local Wisper

Local speech-to-text for Linux, built around Sway. Press a shortcut, speak, and
the transcript appears in the focused window without sending your audio to the
cloud.

## Requirements

- x86_64 Linux
- `pw-record` or `ffmpeg` for audio capture
- `wl-copy`, `xclip`, or `xsel` for clipboard output
- `wtype` for typing into the focused Sway window
- BAML 0.16, `cargo`, `curl`, and `sha256sum` to build and install

## Install

```bash
git clone https://github.com/none23/local-wisper.git
cd local-wisper
./install.sh
```

The installer adds `lw` to `~/.local/bin`. Make sure that directory is in your
`PATH`.

## Usage

Sway is the primary integration. The `lw` CLI is available for debugging; run
`lw --help` when you need it.

Install the supplied Sway wrapper:

```bash
install -Dm755 integrations/sway/local-wisper.sh \
  ~/.config/sway/scripts/local-wisper.sh
```

Add the following to your Sway configuration:

```sway
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

Press <kbd>Mod</kbd>+<kbd>`</kbd> to start recording. Press it again, or press
<kbd>Enter</kbd>, to transcribe and type into the focused window. Press
<kbd>Escape</kbd> to cancel.

## Configuration

Local Wisper works without an API key. OpenAI can optionally post-process
transcripts. Enable it in `~/.config/local-wisper/env`:

```bash
export OPENAI_API_KEY='...'
export LW_POST_PROCESS_MODEL='gpt-5.6-luna'
```
