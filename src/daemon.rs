use std::fs::{self, OpenOptions};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

pub fn spawn(preference: baml_sdk::DevicePreference, log_path: String) -> Result<()> {
    let executable = std::env::current_exe().context("failed to locate the lw executable")?;
    let log_path = std::path::Path::new(&log_path);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to create daemon log {}", log_path.display()))?;
    let error_log = log.try_clone()?;

    let mut command = Command::new(executable);
    command
        .arg("__daemon")
        .arg(preference_name(preference))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(error_log));
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command
        .spawn()
        .context("failed to start transcription daemon")?;
    Ok(())
}

fn preference_name(preference: baml_sdk::DevicePreference) -> &'static str {
    match preference {
        baml_sdk::DevicePreference::Auto => "auto",
        baml_sdk::DevicePreference::Cuda => "cuda",
        baml_sdk::DevicePreference::Cpu => "cpu",
    }
}
