use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};

mod cleanup;
mod daemon;
mod delivery;
mod model;
mod paths;
mod recording;
mod runtime;

#[derive(Debug)]
struct NativeError(String);

impl std::fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NativeError {}

fn native<T>(result: Result<T>) -> std::result::Result<T, NativeError> {
    result.map_err(|error| NativeError(format!("{error:#}")))
}

fn device_preference(device: baml_sdk::DevicePreference) -> model::DevicePreference {
    match device {
        baml_sdk::DevicePreference::Auto => model::DevicePreference::Auto,
        baml_sdk::DevicePreference::Cuda => model::DevicePreference::Cuda,
        baml_sdk::DevicePreference::Cpu => model::DevicePreference::Cpu,
    }
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

    // BAML owns the application. This vector only keeps recordings alive until
    // the BAML workflow has finished transcribing them.
    let recordings = Arc::new(Mutex::new(Vec::<recording::RecordedAudio>::new()));
    let interactive_recordings = Arc::clone(&recordings);
    let sway_recordings = Arc::clone(&recordings);

    let exit_code = baml_sdk::run_app(
        std::env::args().skip(1).collect(),
        |device| native(daemon::ensure_ready(device_preference(device))),
        |device| native(daemon::start(device_preference(device))),
        move || -> std::result::Result<String, NativeError> {
            let audio = native(recording::record_interactively())?;
            let path = audio.path().to_string_lossy().into_owned();
            interactive_recordings
                .lock()
                .map_err(|_| NativeError("recording owner lock was poisoned".to_owned()))?
                .push(audio);
            Ok(path)
        },
        || native(recording::sway_start()),
        move || -> std::result::Result<String, NativeError> {
            let audio = native(recording::sway_stop())?;
            let path = audio.path().to_string_lossy().into_owned();
            sway_recordings
                .lock()
                .map_err(|_| NativeError("recording owner lock was poisoned".to_owned()))?
                .push(audio);
            Ok(path)
        },
        || native(recording::sway_cancel()),
        |audio_path: String, device| {
            native(daemon::transcribe(
                PathBuf::from(audio_path).as_path(),
                device_preference(device),
            ))
        },
        |transcript: String,
         model_enabled: bool,
         timeout_seconds: f64,
         glossary_file: Option<String>| {
            Ok::<_, NativeError>(cleanup::process(
                &transcript,
                &cleanup::Options {
                    model_enabled,
                    timeout: Duration::from_secs_f64(timeout_seconds),
                    glossary_file: glossary_file.map(PathBuf::from),
                },
            ))
        },
        |transcript: String, mode| {
            Ok::<_, NativeError>(match mode {
                baml_sdk::DeliveryMode::Copy => delivery::copy_text(&transcript),
                baml_sdk::DeliveryMode::Type => delivery::type_text(&transcript),
            })
        },
    )
    .context("BAML application failed")?;

    if exit_code != 0 {
        std::process::exit(exit_code as i32);
    }
    Ok(())
}
