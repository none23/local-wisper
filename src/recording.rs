use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};

const SAMPLE_RATE: &str = "16000";

#[derive(Clone, Copy)]
enum Backend {
    PwRecord,
    Ffmpeg,
}

#[derive(Default)]
pub struct RecorderHost {
    children: Mutex<HashMap<u32, Child>>,
}

impl RecorderHost {
    pub fn spawn(
        &self,
        backend: baml_sdk::RecorderBackend,
        audio_path: String,
        log_path: String,
    ) -> Result<baml_sdk::NativeRecorder> {
        let backend = native_backend(backend);
        let child = launch(backend, Path::new(&audio_path), Path::new(&log_path))?;
        let pid = child.id();
        let started_at = process_start_time(pid)
            .with_context(|| format!("failed to identify recorder process {pid}"))?;
        self.children
            .lock()
            .map_err(|_| anyhow::anyhow!("recorder child lock was poisoned"))?
            .insert(pid, child);
        Ok(baml_sdk::NativeRecorder {
            pid: i64::from(pid),
            started_at: i64::try_from(started_at).context("recorder start time overflowed")?,
        })
    }

    pub fn exists(&self, process: baml_sdk::NativeRecorder) -> Result<bool> {
        let pid = process_pid(&process)?;
        let mut children = self
            .children
            .lock()
            .map_err(|_| anyhow::anyhow!("recorder child lock was poisoned"))?;
        if let Some(child) = children.get_mut(&pid)
            && child.try_wait()?.is_some()
        {
            children.remove(&pid);
            return Ok(false);
        }
        Ok(process_matches(&process))
    }

    pub fn stop(
        &self,
        process: baml_sdk::NativeRecorder,
        backend: baml_sdk::RecorderBackend,
    ) -> Result<()> {
        if !process_matches(&process) {
            bail!(
                "recorder process identity no longer matches PID {}",
                process.pid
            )
        }
        let pid = process_pid(&process)?;
        let backend = native_backend(backend);
        let child = self
            .children
            .lock()
            .map_err(|_| anyhow::anyhow!("recorder child lock was poisoned"))?
            .remove(&pid);
        match child {
            Some(mut child) => stop_child(&mut child, backend),
            None => stop_pid(&process, backend),
        }
    }
}

fn native_backend(backend: baml_sdk::RecorderBackend) -> Backend {
    match backend {
        baml_sdk::RecorderBackend::PwRecord => Backend::PwRecord,
        baml_sdk::RecorderBackend::Ffmpeg => Backend::Ffmpeg,
    }
}

fn launch(backend: Backend, audio: &Path, log: &Path) -> io::Result<Child> {
    let stderr = File::create(log)?;
    let mut command = match backend {
        Backend::PwRecord => {
            let mut command = Command::new("pw-record");
            command.args(["--rate", SAMPLE_RATE, "--channels", "1", "--format", "s16"]);
            command.arg(audio);
            command
        }
        Backend::Ffmpeg => {
            let mut command = Command::new("ffmpeg");
            command.args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "pulse",
                "-i",
                "default",
                "-ac",
                "1",
                "-ar",
                SAMPLE_RATE,
                "-y",
            ]);
            command.arg(audio);
            command
        }
    };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .process_group(0)
        .spawn()
}

fn stop_child(child: &mut Child, backend: Backend) -> Result<()> {
    signal(child.id(), stop_signal(backend))?;
    for _ in 0..40 {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    signal(child.id(), libc::SIGKILL)?;
    child.wait()?;
    Ok(())
}

fn stop_pid(process: &baml_sdk::NativeRecorder, backend: Backend) -> Result<()> {
    let pid = process_pid(process)?;
    signal(pid, stop_signal(backend))?;
    for _ in 0..40 {
        if !process_matches(process) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    if process_matches(process) {
        signal(pid, libc::SIGKILL)?;
    }
    for _ in 0..20 {
        if !process_matches(process) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!("Timed out waiting for recorder {pid} to exit")
}

fn stop_signal(backend: Backend) -> i32 {
    match backend {
        Backend::PwRecord => libc::SIGTERM,
        Backend::Ffmpeg => libc::SIGINT,
    }
}

fn signal(pid: u32, signal: i32) -> Result<()> {
    let result = unsafe { libc::kill(pid as i32, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(error).with_context(|| format!("failed to signal recorder {pid}"))
}

fn process_pid(process: &baml_sdk::NativeRecorder) -> Result<u32> {
    u32::try_from(process.pid).context("invalid recorder PID")
}

fn process_matches(process: &baml_sdk::NativeRecorder) -> bool {
    let Ok(pid) = process_pid(process) else {
        return false;
    };
    process_start_time(pid).is_ok_and(|started_at| {
        i64::try_from(started_at).is_ok_and(|started_at| started_at == process.started_at)
    })
}

fn process_start_time(pid: u32) -> Result<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let mut fields = stat
        .rsplit_once(')')
        .context("invalid process stat record")?
        .1
        .split_whitespace();
    fields
        .nth(19)
        .context("process stat has no start time")?
        .parse()
        .context("invalid process start time")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_process_identity_matches() {
        let pid = std::process::id();
        let process = baml_sdk::NativeRecorder {
            pid: i64::from(pid),
            started_at: i64::try_from(process_start_time(pid).unwrap()).unwrap(),
        };
        assert!(process_matches(&process));
    }

    #[test]
    fn changed_start_time_does_not_match() {
        let pid = std::process::id();
        let process = baml_sdk::NativeRecorder {
            pid: i64::from(pid),
            started_at: 0,
        };
        assert!(!process_matches(&process));
    }
}
