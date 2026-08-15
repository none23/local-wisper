use std::path::PathBuf;
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

    let recorders = Arc::new(recording::RecorderHost::default());
    let spawn_recorders = Arc::clone(&recorders);
    let observed_recorders = Arc::clone(&recorders);
    let stopped_recorders = Arc::clone(&recorders);

    let exit_code = baml_sdk::run_app(
        std::env::args().skip(1).collect(),
        |device| native(daemon::ensure_ready(device_preference(device))),
        |device| native(daemon::start(device_preference(device))),
        || native(paths::runtime_dir().map(|path| path.to_string_lossy().into_owned())),
        move |backend, audio_path, log_path| {
            native(spawn_recorders.spawn(backend, audio_path, log_path))
        },
        move |process| native(observed_recorders.exists(process)),
        move |process, backend| native(stopped_recorders.stop(process, backend)),
        |audio_path: String, device| {
            native(daemon::transcribe(
                PathBuf::from(audio_path).as_path(),
                device_preference(device),
            ))
        },
    )
    .context("BAML application failed")?;

    if exit_code != 0 {
        std::process::exit(exit_code as i32);
    }
    Ok(())
}
