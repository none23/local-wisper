use std::sync::Arc;

use anyhow::{Context, Result};

mod daemon;
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

fn parse_device_preference(value: &str) -> Result<baml_sdk::DevicePreference> {
    match value {
        "auto" => Ok(baml_sdk::DevicePreference::Auto),
        "cuda" => Ok(baml_sdk::DevicePreference::Cuda),
        "cpu" => Ok(baml_sdk::DevicePreference::Cpu),
        _ => anyhow::bail!("invalid daemon device {value}"),
    }
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("__daemon") {
        let preference = std::env::args()
            .nth(2)
            .as_deref()
            .map(parse_device_preference)
            .transpose()?
            .unwrap_or(baml_sdk::DevicePreference::Auto);
        let runtime_dir = paths::runtime_dir()?.to_string_lossy().into_owned();
        let models = Arc::new(model::ModelHost::default());
        let locking_model = Arc::clone(&models);
        let loading_model = Arc::clone(&models);
        let transcribing_model = Arc::clone(&models);
        let exit_code = baml_sdk::run_daemon(
            runtime_dir,
            preference,
            move || native(locking_model.acquire_lock()),
            move |model_dir, variant| native(loading_model.load(model_dir, variant)),
            move |audio_path| native(transcribing_model.transcribe(audio_path)),
        )
        .context("BAML daemon failed")?;
        if exit_code != 0 {
            std::process::exit(exit_code as i32);
        }
        return Ok(());
    }

    let recorders = Arc::new(recording::RecorderHost::default());
    let spawn_recorders = Arc::clone(&recorders);
    let observed_recorders = Arc::clone(&recorders);
    let stopped_recorders = Arc::clone(&recorders);

    let exit_code = baml_sdk::run_app(
        std::env::args().skip(1).collect(),
        |device, log_path| native(daemon::spawn(device, log_path)),
        || native(paths::runtime_dir().map(|path| path.to_string_lossy().into_owned())),
        move |backend, audio_path, log_path| {
            native(spawn_recorders.spawn(backend, audio_path, log_path))
        },
        move |process| native(observed_recorders.exists(process)),
        move |process, backend| native(stopped_recorders.stop(process, backend)),
    )
    .context("BAML application failed")?;

    if exit_code != 0 {
        std::process::exit(exit_code as i32);
    }
    Ok(())
}
