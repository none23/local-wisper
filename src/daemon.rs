use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize};

use crate::{model, paths};

const READY_TIMEOUT: Duration = Duration::from_secs(300);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Request {
    Ping,
    Transcribe { audio: PathBuf },
}

#[derive(Serialize, Deserialize)]
struct Response {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub fn serve(preference: model::DevicePreference) -> Result<()> {
    let lock_path = paths::daemon_lock_path()?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("failed to open daemon lock {}", lock_path.display()))?;
    if lock.try_lock_exclusive().is_err() {
        return Ok(());
    }

    let error_path = paths::daemon_error_path()?;
    let _ = fs::remove_file(&error_path);
    let result = serve_locked(preference);
    if let Err(error) = &result {
        let _ = fs::write(&error_path, format!("{error:#}\n"));
    }
    result
}

fn serve_locked(preference: model::DevicePreference) -> Result<()> {
    let mut model = model::Model::load(preference)?;

    let socket_path = paths::socket_path()?;
    let _ = fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind daemon socket {}", socket_path.display()))?;
    eprintln!("ready on {}", socket_path.display());

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_connection(stream, &mut model),
            Err(error) => eprintln!("daemon connection failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: UnixStream, model: &mut model::Model) {
    let response = read_request(&stream).and_then(|request| match request {
        Request::Ping => Ok(String::new()),
        Request::Transcribe { audio } => model.transcribe(&audio),
    });
    let response = match response {
        Ok(text) => Response {
            ok: true,
            text: (!text.is_empty()).then_some(text),
            error: None,
        },
        Err(error) => Response {
            ok: false,
            text: None,
            error: Some(format!("{error:#}")),
        },
    };
    if serde_json::to_writer(&mut stream, &response).is_ok() {
        let _ = stream.write_all(b"\n");
        let _ = stream.flush();
    }
}

fn read_request(stream: &UnixStream) -> Result<Request> {
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    if line.is_empty() {
        bail!("daemon client closed the connection without a request")
    }
    serde_json::from_str(&line).context("invalid daemon request")
}

pub fn ensure_ready(preference: model::DevicePreference) -> Result<()> {
    if ping().is_ok() {
        return Ok(());
    }

    let error_path = paths::daemon_error_path()?;
    let _ = fs::remove_file(&error_path);
    spawn(preference)?;
    let started = Instant::now();
    let mut last_spawn = Instant::now();
    while started.elapsed() < READY_TIMEOUT {
        if ping().is_ok() {
            return Ok(());
        }
        if let Ok(error) = fs::read_to_string(&error_path) {
            bail!("transcription daemon failed to start: {}", error.trim())
        }
        if last_spawn.elapsed() >= Duration::from_secs(3) {
            spawn(preference)?;
            last_spawn = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    bail!("transcription daemon did not become ready within 300 seconds")
}

pub fn start(preference: model::DevicePreference) -> Result<()> {
    if ping().is_ok() {
        return Ok(());
    }
    spawn(preference)
}

pub fn transcribe(audio: &Path, preference: model::DevicePreference) -> Result<String> {
    ensure_ready(preference)?;
    let response = request(&Request::Transcribe {
        audio: audio.to_path_buf(),
    })?;
    if response.ok {
        Ok(response.text.unwrap_or_default())
    } else {
        bail!(
            "transcription failed: {}",
            response.error.unwrap_or_else(|| "unknown error".to_owned())
        )
    }
}

fn ping() -> Result<()> {
    let response = request_with_timeout(&Request::Ping, Duration::from_millis(250))?;
    if response.ok {
        Ok(())
    } else {
        bail!("daemon ping failed")
    }
}

fn request(request: &Request) -> Result<Response> {
    request_with_timeout(request, REQUEST_TIMEOUT)
}

fn request_with_timeout(request: &Request, timeout: Duration) -> Result<Response> {
    let socket_path = paths::socket_path()?;
    let mut stream = UnixStream::connect(&socket_path)
        .with_context(|| format!("failed to connect to {}", socket_path.display()))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    serde_json::to_writer(&mut stream, request)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    stream.shutdown(Shutdown::Write)?;

    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    if line.is_empty() {
        bail!("transcription daemon closed the connection without replying")
    }
    serde_json::from_str(&line).context("invalid daemon response")
}

fn spawn(preference: model::DevicePreference) -> Result<()> {
    let executable = std::env::current_exe().context("failed to locate the lw executable")?;
    let log_path = paths::daemon_log_path()?;
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = File::create(&log_path)
        .with_context(|| format!("failed to create daemon log {}", log_path.display()))?;
    let error_log = log.try_clone()?;

    let mut command = Command::new(executable);
    command
        .arg("__daemon")
        .arg(preference.as_str())
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
