use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use baml_sdk::{Command as BamlCommand, NativeAction};
use clap::{Parser, ValueEnum};

mod cleanup;
mod daemon;
mod delivery;
mod model;
mod paths;
mod recording;
mod runtime;

const MODEL_ID: &str = "nvidia/parakeet-tdt-0.6b-v3";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliCommand {
    Record,
    Preload,
    SwayStart,
    SwayStop,
    SwayCancel,
}

#[derive(Debug, Parser)]
#[command(
    name = "lw",
    about = "Record speech and transcribe it locally with Parakeet"
)]
struct Args {
    #[arg(value_enum, default_value = "record")]
    command: CliCommand,

    #[arg(long, default_value = "parakeet", hide = true)]
    backend: String,

    #[arg(long, hide = true)]
    model: Option<String>,

    #[arg(long, hide = true)]
    compute_type: Option<String>,

    #[arg(long, value_enum, default_value_t, hide = true)]
    device: model::DevicePreference,

    #[arg(long, default_value_t = 16_000, hide = true)]
    sample_rate: u32,

    #[arg(long, hide = true)]
    vad_filter: bool,

    #[arg(long, hide = true)]
    no_vad_filter: bool,

    #[arg(long, hide = true)]
    type_output: bool,

    #[arg(long, hide = true)]
    post_process_model: Option<String>,

    #[arg(long, default_value_t = 20.0, hide = true)]
    post_process_timeout: f64,

    #[arg(long, hide = true)]
    post_process_glossary_file: Option<PathBuf>,
}

struct RunState {
    audio: Option<recording::RecordedAudio>,
    transcript: Option<String>,
    delivered: bool,
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("__daemon") {
        let preference = std::env::args()
            .nth(2)
            .as_deref()
            .map(model::DevicePreference::parse)
            .transpose()?
            .unwrap_or_default();
        return daemon::serve(preference);
    }

    let args = Args::parse();
    validate_options(&args)?;
    let command = baml_command(args.command);
    let plan = baml_sdk::plan_command(command).context("BAML could not plan the command")?;
    let mut state = RunState {
        audio: None,
        transcript: None,
        delivered: false,
    };

    for action in plan.actions {
        execute(action, &args, &mut state)?;
    }

    if let Some(text) = state.transcript.as_deref() {
        println!("{text}");
        if matches!(args.command, CliCommand::Record)
            && !state.delivered
            && !delivery::copy_text(text)
        {
            eprintln!("Warning: Could not copy transcript to the clipboard.");
        }
    }
    Ok(())
}

fn execute(action: NativeAction, args: &Args, state: &mut RunState) -> Result<()> {
    match action {
        NativeAction::EnsureModel => daemon::ensure_ready(args.device),
        NativeAction::StartModel => daemon::start(args.device),
        NativeAction::RecordInteractively => {
            state.audio = Some(recording::record_interactively()?);
            Ok(())
        }
        NativeAction::StartRecording => recording::sway_start(),
        NativeAction::StopRecording => {
            state.audio = Some(recording::sway_stop()?);
            Ok(())
        }
        NativeAction::CancelRecording => recording::sway_cancel(),
        NativeAction::Transcribe => {
            let audio = state
                .audio
                .as_ref()
                .context("BAML requested transcription before recording audio")?;
            let text = daemon::transcribe(audio.path(), args.device)?;
            if text.is_empty() {
                eprintln!("No speech detected.");
            } else {
                state.transcript = Some(text);
            }
            Ok(())
        }
        NativeAction::CleanTranscript => {
            let Some(text) = state.transcript.as_deref() else {
                return Ok(());
            };
            state.transcript = Some(cleanup::process(
                text,
                &cleanup::Options {
                    model_enabled: args.post_process_model.is_some(),
                    timeout: std::time::Duration::from_secs_f64(args.post_process_timeout),
                    glossary_file: args.post_process_glossary_file.clone(),
                },
            ));
            Ok(())
        }
        NativeAction::TypeOutput => {
            let Some(text) = state.transcript.as_deref() else {
                return Ok(());
            };
            let delivered = if args.type_output {
                delivery::type_text(text)
            } else {
                delivery::copy_text(text)
            };
            if !delivered {
                eprintln!("Warning: Could not deliver transcript to the focused application.");
            }
            state.delivered = delivered;
            Ok(())
        }
    }
}

fn baml_command(command: CliCommand) -> BamlCommand {
    match command {
        CliCommand::Record => BamlCommand::Record,
        CliCommand::Preload => BamlCommand::Preload,
        CliCommand::SwayStart => BamlCommand::SwayStart,
        CliCommand::SwayStop => BamlCommand::SwayStop,
        CliCommand::SwayCancel => BamlCommand::SwayCancel,
    }
}

fn validate_options(args: &Args) -> Result<()> {
    if args.backend != "parakeet" {
        bail!("only --backend parakeet is supported")
    }
    if args.model.as_deref().is_some_and(|model| model != MODEL_ID) {
        bail!("only --model {MODEL_ID} is supported")
    }
    // Kept as a no-op because older Sway wrappers pass the former precision.
    let _ = &args.compute_type;
    if args.sample_rate != 16_000 {
        bail!("only --sample-rate 16000 is supported")
    }
    if args.vad_filter && !args.no_vad_filter {
        bail!("VAD is not supported; use --no-vad-filter")
    }
    if args
        .post_process_model
        .as_deref()
        .is_some_and(|model| model != "gpt-5.6-luna")
    {
        bail!("only --post-process-model gpt-5.6-luna is supported")
    }
    if !args.post_process_timeout.is_finite() || args.post_process_timeout <= 0.0 {
        bail!("--post-process-timeout must be greater than zero")
    }
    Ok(())
}
