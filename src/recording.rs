use std::fs::{self, File};
use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::paths;

const SAMPLE_RATE: &str = "16000";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Backend {
    PwRecord,
    Ffmpeg,
}

#[derive(Deserialize, Serialize)]
struct State {
    pid: u32,
    backend: Backend,
    audio: PathBuf,
    session_dir: PathBuf,
}

pub struct RecordedAudio {
    path: PathBuf,
    session_dir: PathBuf,
}

impl RecordedAudio {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for RecordedAudio {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.session_dir);
    }
}

pub fn record_interactively() -> Result<RecordedAudio> {
    let session_dir = create_session_dir()?;
    let audio = session_dir.join("recording.wav");
    let log = session_dir.join("recording.stderr.log");
    let (mut child, backend) = start_recorder(&audio, &log, false)?;

    eprintln!("Recording... Press Enter to stop.");
    let mut input = String::new();
    let _ = io::stdin().read_line(&mut input);
    stop_child(&mut child, backend)?;
    validate_audio(&audio)?;

    Ok(RecordedAudio {
        path: audio,
        session_dir,
    })
}

pub fn sway_start() -> Result<()> {
    let state_path = state_path()?;
    if let Some(state) = read_state(&state_path)? {
        if process_exists(state.pid) {
            bail!("Sway recording is already active")
        }
        cleanup_state(&state_path, &state);
    }

    let session_dir = create_session_dir()?;
    let audio = session_dir.join("recording.wav");
    let log = session_dir.join("recording.stderr.log");
    let (child, backend) = match start_recorder(&audio, &log, true) {
        Ok(recorder) => recorder,
        Err(error) => {
            let _ = fs::remove_dir_all(&session_dir);
            return Err(error);
        }
    };
    let state = State {
        pid: child.id(),
        backend,
        audio,
        session_dir,
    };
    write_state(&state_path, &state)
}

pub fn sway_stop() -> Result<RecordedAudio> {
    let state_path = state_path()?;
    let state = read_state(&state_path)?.context("No active Sway recording")?;
    if !process_exists(state.pid) {
        cleanup_state(&state_path, &state);
        bail!("Sway recording process is not running anymore")
    }

    stop_pid(state.pid, state.backend)?;
    let _ = fs::remove_file(&state_path);
    validate_audio(&state.audio)?;
    Ok(RecordedAudio {
        path: state.audio,
        session_dir: state.session_dir,
    })
}

pub fn sway_cancel() -> Result<()> {
    let state_path = state_path()?;
    let Some(state) = read_state(&state_path)? else {
        return Ok(());
    };
    if process_exists(state.pid) {
        stop_pid(state.pid, state.backend)?;
    }
    cleanup_state(&state_path, &state);
    Ok(())
}

fn start_recorder(audio: &Path, log: &Path, detached: bool) -> Result<(Child, Backend)> {
    let mut failures = Vec::new();
    for backend in [Backend::PwRecord, Backend::Ffmpeg] {
        match launch(backend, audio, log, detached) {
            Ok(mut child) => {
                thread::sleep(match backend {
                    Backend::PwRecord => Duration::from_millis(250),
                    Backend::Ffmpeg => Duration::from_millis(400),
                });
                if child.try_wait()?.is_none() {
                    return Ok((child, backend));
                }
                failures.push(format!("{backend:?} exited during startup"));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                failures.push(format!("{backend:?} is not installed"));
            }
            Err(error) => failures.push(format!("{backend:?}: {error}")),
        }
    }
    bail!("Could not start audio capture: {}", failures.join("; "))
}

fn launch(backend: Backend, audio: &Path, log: &Path, detached: bool) -> io::Result<Child> {
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
        .stderr(Stdio::from(stderr));
    if detached {
        command.process_group(0);
    }
    command.spawn()
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

fn stop_pid(pid: u32, backend: Backend) -> Result<()> {
    signal(pid, stop_signal(backend))?;
    for _ in 0..40 {
        if !process_exists(pid) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    signal(pid, libc::SIGKILL)?;
    for _ in 0..20 {
        if !process_exists(pid) {
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

fn process_exists(pid: u32) -> bool {
    let result = unsafe { libc::kill(pid as i32, 0) };
    result == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn validate_audio(audio: &Path) -> Result<()> {
    let size = fs::metadata(audio)
        .with_context(|| format!("recording was not created at {}", audio.display()))?
        .len();
    if size < 2048 {
        bail!("Recording is empty or too short to transcribe")
    }
    Ok(())
}

fn create_session_dir() -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let path = paths::runtime_dir()?.join(format!("recording-{}-{stamp}", std::process::id()));
    fs::create_dir(&path)
        .with_context(|| format!("failed to create recording directory {}", path.display()))?;
    Ok(path)
}

fn state_path() -> Result<PathBuf> {
    Ok(paths::runtime_dir()?.join("recording.json"))
}

fn read_state(path: &Path) -> Result<Option<State>> {
    let raw = match fs::read(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    serde_json::from_slice(&raw)
        .map(Some)
        .with_context(|| format!("invalid recording state {}", path.display()))
}

fn write_state(path: &Path, state: &State) -> Result<()> {
    let part = path.with_extension("json.part");
    fs::write(&part, serde_json::to_vec(state)?)?;
    fs::rename(&part, path)?;
    Ok(())
}

fn cleanup_state(path: &Path, state: &State) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_dir_all(&state.session_dir);
}
